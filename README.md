# SI-Loop

> **Two studies live in this repo.** Study 1 (toy few-shot regression; frozen, tag `study1-frozen`) is documented in [`docs/README_study1.md`](docs/README_study1.md) and uses `si_loop_toy.py`. Study 2 (SI-Loop v2, below) lives in `kernel/`, `tests/` and `example_patch.py`.

## SI-Loop v2 (Study 2)

A self-modifying agent loop with a hardened, sealed judge. An autonomous meta-agent proposes
structural patches to its own Python codebase, prompts and strategies. Every patch is applied in an
isolated git worktree, statically validated, tested, and either **promoted** or **atomically
rolled back** based on a paired statistical gate over hidden pytest outcomes.

v2 moves beyond Study 1 (synthetic few-shot regression, see `paper.pdf` in the Study 1 repo).
Study 1's judge was over-conservative (about 73% of rounds rejected; `no_judge` won), so the
v2 gate is a calibrated paired non-inferiority test with an explicit A/A false-accept target.

> **Status:** the kernel (patching, git, sandbox, gate, calibration, archive, Mom Center,
> Shapley credit, ledger) is implemented and tested. The two experiment-specific components,
> the **sealed `Verifier`** (problem families + hidden pytest) and the **LLM `MetaAgent`**, are
> abstract interfaces you must implement for Study 2. `example_patch.py` uses a simulated
> verifier and scripted proposals, so it demonstrates the kernel, **not** any claim about
> self-improvement.

## Layout

```
kernel/                  IMMUTABLE to the agent (digest-checked every cycle; mount :ro in containers)
  patching.py            proposal parser, static validator, test ratchet, AST symbol replacer
  orchestrator.py        loop, git worktrees, sandbox, gate, A/A calibration, archive,
                         Mom Center, Shapley attribution, ledger, crash recovery, CLI
example_patch.py         3-scenario end-to-end demo (good / evil / degrading)
tests/test_kernel.py     kernel test-suite
requirements.txt
```

At run time you add (outside this repo or on separate volumes):

```
agent_repo/agent/        the ONLY subtree the meta-agent may change (its own git repo, branch `main`)
  tests/                 the agent's self-tests: non-regression only, protected by a ratchet
sealed/                  problem generators, oracles, hidden pytest, secret seed (never mounted
                         into agent processes)
runs/<exp>/              ledger.jsonl, archive.json, mom.db, worktrees/
```

## Trust boundaries

| Zone | May write | Notes |
|---|---|---|
| Kernel | nobody at run time | `KernelGuard` hashes `kernel/*.py`; any change aborts the run |
| `agent/**` | the meta-agent, **via validated proposals only** | no direct writes: a dirty worktree after `propose()` is rejected |
| Sealed | the experimenter | candidate code never reads it; the agent sees only aggregate EvalResult-derived signals |

Defence in depth for each patch:

1. **Strict parse** (`parse_proposal`): schema, size, tag and key whitelists.
2. **Path policy**: only `agent/**`, suffixes `.py .md .txt .yaml`, no traversal, no `conftest.py`, no dotfiles.
3. **AST scan** of every changed `.py`: no `socket`, `subprocess`, `ctypes`, `multiprocessing`, `shutil`,
   `importlib`, `pickle`, `sys`, `kernel`/`sealed` imports; no `eval`, `exec`, `compile`, `open`, `__import__`,
   `globals`, `setattr`, `delattr`; no dangerous `os.*` (including aliases and `from os import ...`);
   no `getattr` with a computed name; no dunder escape routes; no relative imports.
4. **Test ratchet**: the set of agent self-test ids may only grow; `skip`, `xfail`, `importorskip`,
   `pytest_plugins`, `pytestmark` are banned in tests.
5. **Exact, atomic application**: `replace_symbol` replaces exactly one same-named definition by AST span;
   all edits are staged in memory and written only if every op succeeds.
6. **Diff audit**: after commit, the changed files must equal the files named in the proposal.
7. **Sandbox**: self-tests run in a subprocess with rlimits (address space, CPU, file size), a scrubbed
   environment, its own session, and process-group SIGKILL on timeout. Container mode adds
   `--network none`, read-only root, memory/cpu/pids caps.

The AST scan is a first filter, **not a security boundary**. For real runs use container mode
(`Config.container_image`) and give the sealed side no network or filesystem path from the agent.

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

## Other components

- **Archive** (quality-diversity): stores stepping stones with a behaviour descriptor (per-family pass profile,
  agent size, tokens); parents are the champion with probability `p_exploit`, otherwise sampled by
  performance x novelty / visits.
- **Mom Center**: SQLite entries (patch, failure, credit) with outcome-based Beta trust, TTL, and a capacity-limited
  broadcast using salience plus MMR diversity. All payloads are sanitised and marked `untrusted`.
- **Shapley credit**: exact over hunks up to `credit_max_exact` (default 6), else seeded Monte Carlo, on a separate
  sealed *credit* split; non-applicable coalitions score 0.
- **Ledger**: fsynced write-ahead log; `begin` without `end` triggers cleanup on the next start (`recover()`).
- **Circuit breaker**: after `max_consecutive_failures` failed cycles the loop cools off.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # kernel test-suite
python example_patch.py            # 3-scenario demo; exit code 0 iff all checks pass
python -m kernel.orchestrator calibrate --n 200 --p 0.5
```

Requirements: Python >= 3.10, git >= 2.30, POSIX OS (Linux/macOS) for the local sandbox.
`git` is driven through its CLI, so no git Python library is needed.

## Wiring up Study 2

```python
from kernel.orchestrator import *

class MyVerifier(Verifier):      # sealed families + hidden pytest behind an RPC solve() proxy
    def draw_batch(self, cycle, n, split="selection"): ...
    def evaluate(self, worktree, batch, wall_s): ...      # -> EvalResult(passes, metrics, family_profile)

class MyMeta(MetaAgent):         # LLM tool-use; read-only access to the worktree
    def propose(self, worktree, parent_sha, broadcast, budget): ...   # -> parse_proposal(model_output)

cfg   = Config(repo_dir="agent_repo", runs_dir="runs/study2", families=("algorithmic", "ds", "compiler"),
               container_image="si-sandbox:latest")
repo  = GitRepo(cfg.repo_dir, cfg.runs_dir / "worktrees")
orch  = Orchestrator(cfg, repo, Sandbox(cfg), MyVerifier(), MyMeta(),
                     Archive(cfg.runs_dir / "archive.json", cfg.seed), ShapleyAttributor(cfg, repo, MyVerifier()),
                     MomCenter(cfg.runs_dir / "mom.db"), KernelGuard("kernel"))
orch.run(max_cycles=1000, workers=2)
```

Your verifier must run candidate code in a separate sandboxed process, keep the oracle and tests out of
its reach, and return nothing but `EvalResult`.

## Reproducibility

Freeze the kernel before the report run, as in Study 1: `shasum -a 256 kernel/*.py > runs/study2/config_hashes.txt`,
commit it, and record the calibration output with it.

## Known limitations

- The verifier and meta-agent are interfaces; no sealed problem families or LLM client are included.
- Adaptive overfitting to the sealed distribution remains possible over many cycles; track the gap between
  the selection split and a rarely-read audit split.
- The static scan has known blind spots (it is intentionally strict, so it can also reject benign code).
- `RLIMIT_AS` is unreliable on macOS and the local sandbox is POSIX only; use container mode for untrusted runs.
- Pass@1 under stochastic decoding is noisy; use temperature 0 with a fixed seed or several samples per problem.
- A learned or LLM-based judge (the main reward-hacking risk) is out of scope.
