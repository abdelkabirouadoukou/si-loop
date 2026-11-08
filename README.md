# SI-Loop

> **Two studies live in this repo.** Study 1 (toy few-shot regression; frozen, tag `study1-frozen`) is documented in [`docs/README_study1.md`](docs/README_study1.md) and uses `si_loop_toy.py`. Study 2 (SI-Loop v2, below) lives in `kernel/`, `tests/` and `example_patch.py`.

## SI-Loop v2 (Study 2)

A guarded self-modifying agent loop. A meta-agent proposes patches to its own code; each patch is applied in an
isolated git worktree, statically validated, judged by a sealed verifier and a statistical gate, and then **promoted
or atomically rolled back**. This version adds the defences against the six hard problems of guarded recursive
self-improvement (RSI):

| Problem | What is built in | Where |
|---|---|---|
| Creeping degradation (many "non-inferior" accepts stack up) | Cycle-0 **anchor**, finite **margin budget** for utility-path accepts (refilled only by *new proven gain*), periodic **anchor/peak audits** with rollback | `kernel/anchor.py` |
| Adaptive overfitting | **Two-stage** judge (screen on batch A, strict gate on independent batch B), SPRT / e-value early stopping, e-LOND online FDR, **coarse-only** agent feedback, **held-out family** (interpreter) never used in selection, Thresholdout-style noisy audit | `kernel/pipeline.py`, `kernel/stats.py`, `kernel/anchor.py` |
| Is the *improver* really improving? | `yield_test`: improver patches judged by **yield** vs the parent policy and vs frozen reference **M0** from identical starts; **clade** parent selection; **operator bandit** with per-tag EMA | `kernel/yield_test.py`, `clade.py`, `bandit.py` |
| Weak verification | Sealed families, **3 tiers** (small-scope exhaustive, fuzz, metamorphic), **honeypots** (impossible tasks, decoy oracle files, canary, audit hook), **mutation kill-rate** harness | `kernel/families.py`, `verifier.py`, `_harness.py`, `mutation.py` |
| Long-term autonomy | **Hash-chained ledger**, **circuit breakers** (accept-rate anomaly, LOC explosion, cost spike, sandbox-escape attempts, audit regressions), worktree **disk GC**, **model/kernel fingerprint** halt, heartbeat watchdog, HALT/STOP latch files | `kernel/ledger.py`, `breakers.py`, `hygiene.py` |
| Measuring true RSI | Pre-registered ablations **A0-A3**, paired by seed, Holm-corrected | `run_ablations.py` |

> **Honest scope.** `kernel/world.py` is a *synthetic* world: the "meta-agent" is a mutation engine (`MutationMetaAgent`),
> not an LLM. It tests the machinery (gate, audits, verifier, ledger, ablations), and **makes no claim about LLM
> self-improvement**. To use a real LLM, implement the `MetaAgent` interface (`propose(worktree, parent_sha, broadcast, budget)`).
> The sandbox in local mode is **not a security boundary**; use container mode for untrusted code (container mode is
> untested here). Honeypots are tripwires, not proofs of safety.

## Layout

```
kernel/                  IMMUTABLE to the agent (digest-checked every cycle; mount :ro in containers)
  patching.py            proposal parser, static validator, test ratchet, AST symbol replacer
  orchestrator.py        loop, git worktrees, sandbox, calibration CLI, audits, halting
  pipeline.py            TwoStageJudge: screen(A) -> confirm(B), early stopping, margin budget, FDR
  stats.py               exact tests, e-values, e-LOND, sequential monitor, bootstrap, sign-flip
  anchor.py              anchor, margin budget, anchor/peak audit, held-out audit
  families.py verifier.py _harness.py   sealed problem families + 3-tier verifier + honeypots
  mutation.py            verifier quality (mutation kill rate)
  improver.py seed_agent.py             synthetic improver (policy, templates) and seed agent
  yield_test.py clade.py bandit.py      meta-level evaluation and parent/operator selection
  ledger.py breakers.py hygiene.py      tamper-evident log, circuit breakers, GC, fingerprint, watchdog
  world.py               wires a synthetic world (profiles tiny / smoke / full)
run_ablations.py         pre-registered A0-A3 experiment runner
example_patch.py         3-scenario kernel demo (good / evil / degrading)
tests/                   68 tests
```

## Trust boundaries

| Zone | May write | Notes |
|---|---|---|
| Kernel | nobody at run time | `KernelGuard` hashes `kernel/*.py`; any change aborts the run |
| `agent/**` | the meta-agent, via validated proposals only | a dirty worktree after `propose()` is rejected |
| Sealed verifier | the experimenter | problem ids are HMACs of a secret; oracle and compare run in the judge, never in the agent's process |
| Held-out family | nobody during selection | never drawn by selection / confirm / credit / anchor / yield splits; results go to the ledger only |
| Agent feedback | kernel | only `{accept, stepping_stone, reject}` + a brief error type; no counts, no per-family numbers |

Patch defence in depth (parse, path policy, AST scan, test ratchet, atomic apply, diff audit, sandbox) is unchanged
from v1 and is still a first filter, not a security boundary. Improver patches (all hunks under `agent/meta/`) go to
the yield gate; patches mixing meta and solver code are rejected.

## The non-inferiority gate

Parent and candidate are evaluated on the **same freshly drawn sealed problems**. Let `b` be the
problems the patch fixed and `c` the problems it broke (discordant pairs), `n` the number of problems:

```
delta = (b - c) / n
se    = sqrt(b + c - (b - c)^2 / n) / n          # paired-proportion standard error
lower = delta - z * se                             # z = 1.645 (one-sided 95%)
p_better = P(Binomial(b + c + 1, 1/2) <= b)        # exact P(Beta(1+b, 1+c) > 0.5)
```

| Verdict | Condition |
|---|---|
| **accept** | `lower >= -eps` **and** `delta > 0` **and** `p_better >= p_improve_min` (or non-inferior, `delta >= 0`, and >= 5% smaller / >= 10% fewer tokens) |
| **stepping_stone** | non-inferior and behaviourally novel (archived, not promoted) |
| **reject** | otherwise |

Heuristic utility (size, tokens) can only break ties among non-inferior candidates; novelty can never
rescue a regression. **A/A calibration** measures the false-accept rate of null patches directly:

```bash
python -m kernel.orchestrator calibrate --n 200 --p 0.5 --noise 0.1 --target 0.05
```

At `n=200`, `p=0.5`, `noise=0.1` the default `p_improve_min=0.95` accepts about 3-4% of null patches. Pre-register
the target and rerun this for your real `n` and baseline pass rate before Study 2. The simulation's
noise model is an assumption; estimate the real run-to-run noise from repeated evaluations of an
unchanged agent.

### v2 additions to the gate

- **Two stages.** Stage 1 (lenient, `screen_p_improve=0.80`) runs in chunks with an anytime-valid e-value *harm* stop
  and an SPRT *futility* stop. Stage 2 applies the strict gate on batch B **alone** (B is independent, so the winner's
  curse is removed). False-accept rates multiply: calibrated at about 1.5% (n=200/200, p_improve 0.95) vs 3.6% single stage.
- **Utility path** (smaller / cheaper code with no proven gain) spends from a finite **margin budget**. Each charge is
  `max(0, -lower)` with a floor (`margin_min_charge=0.005`). Only a *new* high-water mark of proven gain over the anchor
  refills it. This bounds creep; it does not remove it, so the anchor audit is the backstop.
- **Anchor audit** every `anchor_every` cycles on fresh batches: champion vs anchor and vs its own peak; rollback on regression.
- **Held-out audit**: cross-family gap = selection delta - held-out delta, with noisy decisions; rollback or pause on a large gap.

## Quickstart and exact CLI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch is needed only for Study 1
python -m pytest -q tests                  # full test suite (about 25 s)
python example_patch.py                    # kernel demo; exit code 0 iff all checks pass
```

**Calibration** (do this for your real n and baseline pass rate before any real run):

```bash
python -m kernel.orchestrator calibrate --n 200 --p 0.5 --noise 0.1 --target 0.05
python -m kernel.orchestrator calibrate --n 200 --p 0.5 --noise 0.1 --target 0.05 \
       --two-stage --n-confirm 200 --screen-p 0.80
```

**Verifier quality** (mutation kill rate, honeypot self-test; exit code 1 if unhealthy):

```bash
python -m kernel.mutation --per-kind 30 --instances 2 --out runs/verifier_quality.json
```

**Yield test** of one improver policy against a baseline (expensive; start with `--profile tiny`):

```bash
python -m kernel.yield_test --policy '{"w_adopt": 8.0}' --profile smoke --starts 2 --seeds 3 --k 5
```

**Ablation arms** (A0 frozen improver, A1 full recursive, A2 random-patch control, A3 compute-matched best-of-N):

```bash
python run_ablations.py prereg --out runs/abl --arms A0 A1 A2 A3 --seeds 10 --cycles 60 --profile full
git add runs/abl/prereg.json runs/abl/prereg.sha256 && git commit -m "preregister ablations"   # BEFORE running
python run_ablations.py run     --out runs/abl          # resumable: finished runs are skipped
python run_ablations.py analyze --out runs/abl          # writes runs/abl/analysis.txt
python run_ablations.py all     --out runs/abl          # run + analyze
```

`run_ablations.py` refuses to run if `prereg.json` was edited or the kernel digest changed. The run (seed) is the unit
of analysis. Tests: H1 = A1-A0, H2 = A0-A2, H3 = A0-A3 (one-sided sign-flip, Holm-corrected), H4 = held-out
non-regression. Anything short of H1 is reported as *iterative* improvement only. Use `--profile tiny` for a smoke check
only; its numbers mean nothing.

**Operations:**

```bash
python -m kernel.hygiene verify-ledger --run-dir runs/abl/A1_s0/runs      # recompute the hash chain
python -m kernel.hygiene watchdog --run-dir runs/abl/A1_s0/runs --max-age 900  # latch HALT if the heartbeat is stale (run from cron)
python -m kernel.hygiene gc --repo runs/abl/A1_s0/repo --worktrees runs/abl/A1_s0/runs/worktrees  # remove orphan worktrees and stale branches
touch runs/abl/A1_s0/runs/STOP                                  # ask a run to stop cleanly after the current batch
rm runs/abl/A1_s0/runs/HALT                                     # clear a HALT latch after you inspected the cause
```

## Wiring a real LLM

Implement `MetaAgent.propose(...)` (read-only access to the worktree, output through `parse_proposal`) and reuse
`World.build(...)` or construct `Orchestrator` yourself with `SealedVerifier` (use a secret kept outside the agent's
reach). Log the model id through the fingerprint config so a silent provider-side model change halts the run.

## Reproducibility

Freeze the kernel before a report run: `shasum -a 256 kernel/*.py > runs/study2/config_hashes.txt`, commit it, and keep
the calibration and `prereg.json` outputs with it. Study 1 files are frozen (`si_loop_toy.py` sha256 is pinned in
`results/config_hashes.txt`).

## Known limitations

- The synthetic world has no LLM. Results from it say nothing about LLM self-improvement.
- Container mode is not tested; local mode is not a security boundary. Honeypots are tripwires.
- Utility-path creep is bounded by the margin budget and anchor audits, not eliminated.
- Yield tests are expensive (several inner loops per improver patch).
- Cost accounting uses LOC and token counters, not deterministic instruction counts.
- The static scan is strict and can reject benign code; `RLIMIT_AS` is unreliable on macOS.
- Pass@1 under stochastic decoding is noisy; use fixed seeds or several samples per problem.
