# SI-Loop

Guarded recursive self-improvement (RSI) for code agents, in two studies.

| | Study 1 | Study 2 (SI-Loop v2) |
|---|---|---|
| **Status** | Frozen (tag `study1-frozen`) | Active |
| **Setting** | Toy few-shot regression, CPU | Code agent, sealed verifier, git worktrees |
| **Files** | `si_loop_toy.py`, `run_parallel.py`, `analyze.py`, `paper.tex`, `results/` | `kernel/`, `study2/`, `run_study2.py`, `run_ablations.py`, `paper_study2.tex` |
| **Docs** | [`docs/README_study1.md`](docs/README_study1.md) | this file |

> Study 1 files are never edited. `si_loop_toy.py` is pinned by sha256 in `results/config_hashes.txt`.

---

## Quick Start for Study 2

Requirements: Python 3.10+, git 2.30+, Linux or macOS. No GPU. `torch` is needed only for Study 1.

```bash
git clone https://github.com/abdelkabirouadoukou/si-loop && cd si-loop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Check that everything works (about 40 s)

```bash
python -m pytest -q tests          # 80 tests
python run_study2.py check         # verifier self-test: reference solutions, seed baseline, tripwire
```

### 2. Run a full guarded loop with no API key (about 1 min)

```bash
python run_study2.py demo
```

This uses a **scripted proposer** (`OfflineDemoClient`), not a model. It proves that the whole pipeline works:
sealed hidden-pytest verifier, two-stage gate, audits, hash-chained ledger. The demo uses a fixed public secret, so the result is identical on every run: the pass rate on the
selection families goes from 0.28 to 0.78, the held-out family stays flat at 0.38, and the cross-family alarm fires. This is the expected, correct behaviour.

### 3. Run it with a real LLM

```bash
# Local server (Ollama, vLLM, llama.cpp, LM Studio): any OpenAI-compatible endpoint
python run_study2.py run --client openai --base-url http://localhost:11434/v1 --model llama3.1 --cycles 40

# OpenAI
export OPENAI_API_KEY=...
python run_study2.py run --client openai --model <model-id> --cycles 40

# Anthropic
export ANTHROPIC_API_KEY=...
python run_study2.py run --client anthropic --model <model-id> --cycles 40
```

Results go to `runs/study2/`: `summary.json`, `final_eval.json` (fresh batches, champion vs frozen cycle-0 anchor,
selection families and held-out family) and the tamper-evident `runs/ledger.jsonl`.
Use `python run_study2.py report --out runs/study2` to print them again.

**Before a real study:** calibrate the gate for your batch size (section [Calibration](#calibration)),
pre-register your ablation plan (section [Experiments](#experiments)), and use container mode
(`--container-image`) for untrusted code.

---

## Architecture and trust boundaries

```
 +--------------------------------------------------------------------------------------+
 |  IMMUTABLE KERNEL            kernel/*.py   (sha256-checked every cycle; :ro in docker)  |
 |                                                                                        |
 |   Orchestrator --> TwoStageJudge --> Gate (paired non-inferiority, e-values, FDR)      |
 |        |               |   screen on batch A -> confirm on independent batch B         |
 |        |               +-- margin budget, anchor / peak / held-out audits              |
 |        +-- git worktrees, atomic rollback, hash-chained ledger, circuit breakers       |
 +-----------+-----------------------------------------------+---------------------------+
             | PatchProposal (JSON, strict parse)            | EvalResult only
             |  AST scan + path policy + test ratchet        |  (booleans, aggregates, flags)
             v                                               ^
 +-----------------------------+            +---------------------------------------------+
 |  AGENT WORKTREE             |            |  SEALED VERIFIER      study2/verifier.py    |
 |  agent/**  (only writable   |   copy of  |                                             |
 |  subtree)                   |  agent/    |  secret (HMAC problem ids)  oracles          |
 |                             |----------->|  hidden pytest file         expected values  |
 |  lib.py, tests/, meta/      |            |  held-out family (never in selection)       |
 |                             |            |       |                                     |
 |  MetaAgent reads it,        |            |       v   inputs only                       |
 |  never writes it            |            |  +----------------------------+             |
 +--------------^--------------+            |  | SANDBOX PROCESS            |             |
                |                           |  | candidate code, rlimits,   |             |
   coarse feedback only:                    |  | no network, tripwires,     |             |
   accept | stepping_stone | reject         |  | per-call timeout           |             |
   + brief error type                       |  +----------------------------+             |
                                            +---------------------------------------------+
```

| Zone | Can write | Can read | Notes |
|---|---|---|---|
| **Kernel** | nobody at run time | everything it needs | `KernelGuard` aborts the run if a kernel file changes |
| **Agent worktree** | the meta-agent, only through validated proposals | its own code | a dirty worktree after `propose()` is rejected |
| **Sealed verifier** | the experimenter | oracles, secret, hidden tests | the candidate never shares a process with pytest or the oracles |
| **Held-out family** | nobody during selection | the auditor only | results go to the ledger, never to the agent |

What the agent never sees: hidden tests, expected values, problem content, per-family numbers, pass counts, audit
results, held-out results.

---

## What is built in

| Problem | Defence | Code |
|---|---|---|
| Creeping degradation | Frozen **cycle-0 anchor**, finite **margin budget** (refilled only by new proven gain), **anchor and peak audits** with rollback | `kernel/anchor.py` |
| Adaptive overfitting | **Two-stage judge** (screen on A, confirm on B), SPRT and e-value early stopping, e-LOND FDR, coarse-only feedback, **held-out family**, noisy audit | `kernel/pipeline.py`, `stats.py` |
| Is the improver improving? | **Yield test** vs frozen reference M0, **clade** parent selection, **operator bandit** | `kernel/yield_test.py`, `clade.py`, `bandit.py` |
| Weak verification | Sealed families, **3 tiers**, **honeypots**, **mutation kill rate** | `kernel/verifier.py`, `mutation.py` |
| Long-term autonomy | **Hash-chained ledger**, **circuit breakers**, worktree GC, model fingerprint, watchdog | `kernel/ledger.py`, `breakers.py`, `hygiene.py` |
| Measuring real RSI | Pre-registered **A0-A3** ablations, paired by seed, Holm-corrected | `run_ablations.py` |
| Defence in depth per patch | Seven layers: strict parse, path policy, AST scan, test ratchet, atomic apply, diff audit, sandbox | `kernel/patching.py` |

### The gate in one minute

Parent and candidate run on the **same fresh problems**. Let `b` = problems the patch fixed, `c` = problems it broke, `n` = problems.

```
delta    = (b - c) / n
se       = sqrt(b + c - (b - c)^2 / n) / n
lower    = delta - 1.645 * se
p_better = P(Binomial(b + c + 1, 1/2) <= b)
```

| Verdict | Condition |
|---|---|
| **accept** | `lower >= -eps`, `delta > 0` and `p_better >= p_improve_min` (or: non-inferior and clearly smaller / cheaper) |
| **stepping_stone** | non-inferior and behaviourally novel (archived, not promoted) |
| **reject** | otherwise |

Two stages: a lenient screen on batch A (with sequential early stopping), then the strict gate on an independent batch B.
Smaller or cheaper code with no proven gain spends from a finite **margin budget**; only a new high-water mark of
proven gain over the anchor refills it. Novelty can never rescue a regression.

---

## Layout

```
kernel/            IMMUTABLE to the agent (digest-checked)
  orchestrator.py  loop, git worktrees, sandbox, halting, calibration CLI
  pipeline.py      two-stage judge        stats.py       exact tests, e-values, e-LOND
  anchor.py        anchor, margin budget, audits
  patching.py      proposal parser, static validator, test ratchet
  ledger.py breakers.py hygiene.py        tamper-evident log, circuit breakers, GC, watchdog
  yield_test.py clade.py bandit.py        meta-level evaluation
  families.py verifier.py mutation.py     synthetic sealed world + verifier quality
  improver.py seed_agent.py world.py      synthetic improvers (no LLM), used by ablations and tests
study2/            REFERENCE implementation (hidden pytest, local problem suite, LLM meta-agent)
  tasks.py         15 tasks in 3 families (string, algorithmic, held-out parsing)
  verifier.py      PytestVerifier       llm_meta.py    LLMMetaAgent + clients
  seed_repo.py     cycle-0 agent        _worker.py     sandboxed runner
run_study2.py      check | demo | run | report
run_ablations.py   prereg | run | analyze | all    (A0-A3)
paper_study2.tex   Study 2 paper          results/study2/   calibration and verifier-quality data
tests/             80 tests
```

---

## Experiments

### Calibration

Measure the false-accept rate of null patches for your own batch size and baseline pass rate:

```bash
python -m kernel.orchestrator calibrate --n 200 --p 0.5 --noise 0.1 --target 0.05
python -m kernel.orchestrator calibrate --n 200 --p 0.5 --noise 0.1 --target 0.05 --two-stage --n-confirm 200 --screen-p 0.80
python results/study2/make_calibration.py      # curves used in the paper (false accept and power)
```

### Verifier quality

```bash
python -m kernel.mutation --per-kind 30 --instances 2 --out runs/verifier_quality.json   # exit 1 if unhealthy
```

### Pre-registered ablation arms

| Arm | Meaning |
|---|---|
| **A0** | frozen improver, solver patches only |
| **A1** | full recursive loop (solver and improver, improver judged by yield tests) |
| **A2** | random-patch control |
| **A3** | compute-matched best-of-N (every proposal made from the anchor) |

```bash
python run_ablations.py prereg --out runs/abl --arms A0 A1 A2 A3 --seeds 10 --cycles 60 --profile full
git add runs/abl/prereg.json runs/abl/prereg.sha256 && git commit -m "preregister ablations"    # BEFORE running
python run_ablations.py run     --out runs/abl --jobs 2     # resumable
python run_ablations.py analyze --out runs/abl              # paired sign-flip tests, Holm correction
```

The runner refuses to start if `prereg.json` was edited or the kernel changed. The seed is the unit of analysis.
Anything short of H1 (A1 > A0, Holm-adjusted) is reported as iterative improvement only.

### Yield test of one improver policy

```bash
python -m kernel.yield_test --policy '{"w_adopt": 8.0}' --profile smoke --starts 2 --seeds 3 --k 5
```

### Operations

```bash
python -m kernel.hygiene verify-ledger --run-dir runs/study2/runs
python -m kernel.hygiene watchdog      --run-dir runs/study2/runs --max-age 900     # run from cron
python -m kernel.hygiene gc --repo runs/study2/agent_repo --worktrees runs/study2/runs/worktrees
touch runs/study2/runs/STOP        # stop cleanly after the current batch
rm    runs/study2/runs/HALT        # clear a HALT latch after you inspected the cause
```

---

## Writing your own verifier or meta-agent

Both are small interfaces in `kernel/orchestrator.py`.

```python
class Verifier(ABC):
    def draw_batch(self, cycle, n, split="selection") -> list: ...
    def evaluate(self, worktree, batch, wall_s) -> EvalResult: ...      # passes, metrics, family_profile, flags

class MetaAgent(ABC):
    def propose(self, worktree, parent_sha, broadcast, budget) -> PatchProposal: ...   # use parse_proposal()
```

`study2/verifier.py` and `study2/llm_meta.py` are complete examples. A verifier must keep oracles and tests out of the
candidate's process, derive problem ids from a secret, and return nothing but `EvalResult`.

---

## Reproducibility

- Freeze the kernel before a report run: `shasum -a 256 kernel/*.py > runs/study2/config_hashes.txt`, and commit it.
- Keep `prereg.json`, the calibration output and the ledger with the results.
- The fingerprint (model, provider, kernel digest) is logged; a silent change halts the run.

## Known limitations

- **No claim about LLM self-improvement is made by the synthetic world** (`kernel/world.py`); its "meta-agent" is a mutation engine. The `study2/` reference runs a real LLM, but its tasks are small.
- Container mode is untested here. The local sandbox is **not** a security boundary. Honeypots are tripwires.
- Utility-path creep is bounded by the margin budget and audits, not eliminated.
- Yield tests are expensive. Costs are counted in LOC and tokens, not instruction counts.
- The static scan is strict and can reject benign code. `RLIMIT_AS` is unreliable on macOS.
- LLM decoding is stochastic: run several seeds and report intervals.

## Licence

See [`LICENSE`](LICENSE).
