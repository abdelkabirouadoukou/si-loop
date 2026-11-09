# Study 4 (Phase 4): the improver arm

**Status: runner built and wired; a real pilot has been run (2026-11-09, see "Pilot case study"
below). The pre-registration is locked, over the code as it exists after all fixes described here.
No accepted patch yet. Not a pre-registered replicate -- see the caveats in the case study.**

## What is here

| File | Purpose | State |
|---|---|---|
| `DESIGN_DRAFT.md` | Draft design: question, arms, metric, open decisions | draft, not locked; decisions 1-4 resolved, recorded in the prereg draft |
| `guard.py` | Refuses to run unless a locked pre-registration matches the code and kernel | working, tested |
| `../run_study4.py` | Entry point. `--check` verifies the pre-registration (reads no key). `--run` verifies the same way, refuses on any failure, and only then builds and runs the live loop (`run_live.py`) | working, tested |
| `live_meta.py: MetaAgentA1` | Non-kernel subclass of `kernel.llm_meta.LLMMetaAgent` giving the live LLM arm-A1 editing rights for `agent/meta/policy.py` (one scope per patch, never both) | working, tested |
| `yield_gate.py: build_study4_yield_gate`, `Study4YieldGate` | Builds a `YieldGate` wired to a dedicated synthetic source repo (the fix -- see "Known gap and fix") | working, tested |
| `run_live.py` | Builds the Study 4 `Orchestrator` (arm A0 or A1) and runs it; reads the provider key from the environment only here, never on `--check` | working, tested |
| `fdr_replay.py: replay_elond_state`, `attach_persistent_fdr_budget` | Makes the enforced e-LOND FDR budget continuous across separate `--run` process restarts, by replaying the ledger's own history -- see "Known gap and fix" | working, tested, verified live |
| `../study2/seed_repo.py: write_seed_agent_study4` | Study 4 seed: the ordinary A0 agent plus `agent/meta/policy.py`. `write_seed_agent` itself (Study 3's seed) is untouched | working, tested |
| `../results/study4/prereg.json` | Pre-registration. Locked over the final code (every file in this table that affects runtime behaviour is pinned in `code_sha256`) | locked |
| `../tests/test_study4_guard.py` | Guard/entry-point tests: missing/draft/tampered/edited/wrong-study/missing-key/kernel-digest-mismatch pre-registrations, refusal behaviour, and (if a real locked prereg exists) that it is at least internally consistent | passing |
| `../tests/test_study4_integration.py` | Kernel-level rules Study 4 depends on: the mixed-scope rule, the enforced false-discovery monitor, the `YieldHarness` source-repo fix, and `YieldGate` dispatch (stubbed and real, end to end) | passing |
| `../tests/test_study4_fdr_replay.py` | Tests for `fdr_replay.py`, including two genuinely separate `Orchestrator` instances sharing one ledger (simulating two process launches) | passing |
| `../tests/test_study4_live_meta.py` | `MetaAgentA1`, fake local server, no live calls | passing |
| `../tests/test_study4_run_live.py` | `python run_study4.py --run`, end to end, arms A0 and A1, fake local server, no live calls | passing |

## What the guard checks

A run is allowed only if `results/study4/prereg.json` exists and all of the following hold:

1. the file is valid JSON and has every required key;
2. `status` is `"locked"`;
3. `prereg_sha256` matches the SHA-256 of the sorted-JSON body (the same scheme as Study 3);
4. each file in `code_sha256` still matches its hash on disk;
5. `kernel_digest` matches the current kernel.

Any failure is a refusal with a message saying which check failed. The guard does not read API keys and does not
write files. `run_study4.py --run` calls it first, exactly like `--check`, and only imports `run_live.py` (which
does read a key) after it passes.

## The seed repository

`write_seed_agent_study4(root)` writes the ordinary Study 2/3 seed agent (`write_seed_agent`, unchanged, so Study 3's
reproducibility is unaffected) plus `agent/meta/policy.py`:

```python
# The improver's own policy. The improver may edit this file; patches to it are judged by the yield test.
POLICY = {
    "p_meta": 0.0,
    "w_adopt": 2.0,
    "w_prune": 1.0,
    "w_mutate": 1.5,
    "use_bandit": 0.0,
}
```

This is exactly the 5-key schema `kernel/improver.py` (`policy_from_source`, `REFERENCE_POLICY`) already parses, so
`kernel/yield_test.py`'s `YieldGate` can read it with no kernel change. It starts at the frozen reference (M0)
values, generated via `kernel.improver.policy_source` rather than a second, hand-written copy of that format.

## Arm A1: MetaAgentA1

`live_meta.py: MetaAgentA1` subclasses `kernel.llm_meta.LLMMetaAgent` without editing `kernel/llm_meta.py`
(a kernel edit would change the kernel digest and invalidate Study 3's own pre-registration check):

- constructs the parent as arm `"A0"` (the only arm `LLMMetaAgent.__init__` accepts) and then widens
  `self.arm` to `"A1"`, reusing 100% of the parent's real setup;
- overrides `_check_arm` to also allow `agent/meta/policy.py`, and to refuse (before the kernel even sees it) a
  single patch that touches both `agent/lib.py` and `agent/meta/policy.py` -- the kernel's own mixed-scope rule
  (`kernel/orchestrator.py`) is the real enforcement; this is just an early, cheap rejection;
- overrides `_call` to use its own system prompt (`SYSTEM_A1`), which tells the model `agent/meta/policy.py`
  exists and may be edited -- arm A0's prompt explicitly tells the model NOT to touch it.

Pacing, caps, sanitising, the write-ahead history and strict JSON parsing are all inherited unchanged.

## Known gap and fix: the yield measurement, and what it is wired to

`YieldGate.judge` calls `YieldHarness.measure()`, which does not re-run whatever produced the patch. It always
clones a `source` repo, builds a **synthetic** `kernel/world.py` `World`, and evaluates with `kernel/verifier.py`'s
`SealedVerifier` -- which expects the *different*, synthetic file layout `kernel/seed_agent.py` writes
(`agent/solver.py` imported by the sandboxed worker), not Study 2/3/4's `agent/lib.py` task-library layout.

**Root cause, confirmed:** in `run_ablations.py`'s existing A1 arm, this is harmless, because `source` is always
`orch.repo` -- and that repo was itself built by `World.init_repo`, so it is already in the synthetic shape the
harness expects. Pointing the *same* harness at a `write_seed_agent_study4` repo (study2-shaped) clones that
instead, and the synthetic verifier cannot run or score that code at all: `pass@1` reads exactly `0.0` for
*every* state, every time, regardless of the actual code or policy numbers. Since `YieldGate.judge`'s accept rule
needs the bootstrap lower bound of the gain to clear a `> 0.0` threshold, a measurement that is always exactly 0
can never accept -- every meta-scope patch in a live Study 4 run wired this way would be rejected automatically.

**The fix (wired):** `study4/yield_gate.py: build_study4_yield_gate(orch, tmp_root, world_secret, profile, ycfg)`
gives `YieldHarness` a **dedicated, separately-built synthetic repo** (`kernel.world.World.init_repo`, built once)
as its `source`, never the live study2/Study 4 repo. Verified directly: with a dedicated synthetic source,
`pass@1` is non-zero and responds to the actual code again
(`tests/test_study4_integration.py::test_yield_harness_must_measure_against_a_synthetic_source_not_the_live_study2_repo`).

A second, related bug surfaced while wiring this: `YieldGate.starts()` (`kernel/yield_test.py`) returns commit SHAs
from `self.orch` (`anchor.anchor_sha`, `repo.head("main")`, archive entries) -- correct for `run_ablations.py`'s A1
arm, where the harness's source repo *is* `orch.repo`, but those SHAs do not exist at all in a separately-cloned
synthetic repo, so `git checkout` on one fails outright. `study4/yield_gate.py: Study4YieldGate` (subclasses
`YieldGate`, overrides only `starts()` to return the synthetic source repo's own head) fixes this.

Both fixes are **wiring/subclassing choices for the runner, not `kernel/` changes** -- `YieldHarness`'s `source`
parameter and `YieldGate`'s `starts()` were both already overridable without touching kernel code. They do not
remove the trade-off decision 1 already accepted: yield is still the effect of a policy-number edit on the
scripted stand-in agent inside its own synthetic world, not a direct measurement of the live LLM's behaviour -- the
fixes just make that measurement actually run and respond to the numbers, verified end to end through
`python run_study4.py --run --arm A1` offline (fake server)
(`tests/test_study4_run_live.py::test_run_arm_a1_wires_the_yield_gate_offline`,
`tests/test_study4_integration.py::test_build_study4_yield_gate_end_to_end_on_a_real_orchestrator`).

## Known gap and fix: e-LOND's budget resets across process restarts

`kernel/orchestrator.py` always constructs a fresh `ELond(t=0, d=0)` for every new `Orchestrator` --
nothing there persists or restores its state (`self.elond = ELond(cfg.fdr_alpha)`, no `.state()`/`.load()`
call anywhere). Within one continuous invocation the enforced budget tightens correctly cycle over cycle,
but Study 4 is run as repeated, separate CLI invocations against the same output directory (rate limits,
pauses between sessions), so the budget was silently restarting each time -- handing out a fresh, generous
early-test allowance again on every launch, which weakens the anytime-valid guarantee e-LOND is supposed to
provide over the *whole* sequence of tests in the study. True for Study 2 and 3 as well; not something this
work introduced.

**The fix (wired):** `study4/fdr_replay.py: attach_persistent_fdr_budget(orch, ledger_path)` replays every
past correctness-path FDR test recorded in the ledger (using the `hit` outcome `ELond` itself already
recorded, not a recomputed e-value) and installs the resulting `(t, d)` onto both `orch.elond` and
`orch.judge.elond`. No `kernel/` edit: `kernel/pipeline.py`'s `TwoStageJudge` already accepts a pre-built
`elond` via an optional constructor parameter -- a deliberate extension point. Verified live, not just in
tests: predicted the next real cycle's required level from the ledger (`0.0012158542037080532`) before
running it, and the actual live run's first FDR-tested cycle started at exactly that level -- see the pilot
case study below.

## Running it (once locked)

```bash
export GROQ_API_KEY=...                                # read from the environment only, never stored
python run_study4.py --check                            # verify the pre-registration only
python run_study4.py --run --arm A1 --cycles 3           # Study 4 pilot, arm A1 (improver allowed)
python run_study4.py --run --arm A0 --cycles 3           # Study 4 pilot, arm A0 (solver-only baseline)
```

`--run` refuses before touching the loop if the pre-registration is missing, a draft, tampered, or stale
(`guard.verify()`, same as `--check`). Only after that does it read the key and build the live loop. `--out`
defaults to `runs/study4_live`; see `python run_study4.py --help` for the rest (mirrors `run_study3_live.py`'s
flags where they overlap, plus `--arm` and `--yield-*` for arm A1's yield-measurement profile).

## Pilot case study (2026-11-09)

A real pilot was run against the live model (`openai/gpt-oss-120b` via Groq), arm A1, output directory
`runs/study4_live`, across several separate CLI invocations over one session. **This is a debugging-phase
pilot, not a pre-registered replicate** -- the code itself changed partway through (the venv/self-test fix,
the `YieldHarness` source-repo fix, and the e-LOND persistence fix were all found and applied *during* this
session, the same way Study 3's own pilot exposed D1). It is kept here as a transparent record of what was
run and why, the same spirit as `results/study3/DEVIATIONS.md` -- not pooled as a clean result, and a real
Study 4 replicate set should start fresh, after all of these fixes, under its own pre-registration.

**Totals across the whole pilot:** 13 cycles, 0 accepts.

| Reject reason | Count | Cause |
|---|---|---|
| `selftest` | 2 | the pre-venv-fix environment bug (self-test subprocess couldn't see user-site `pytest`) -- not a judge outcome |
| `fdr` | 7 | enforced online false-discovery control (decision 3) |
| `no_gain` | 2 | ordinary two-stage-judge rejection (no pass-rate improvement) |
| `unconfirmed` | 2 | screened well on batch A, didn't hold up on independent batch B |

**Every judged cycle was scope `solver`** (an edit to `agent/lib.py`) -- across the whole pilot, arm A1's
live model never attempted an edit to `agent/meta/policy.py`, even though its system prompt (`SYSTEM_A1`)
explicitly tells it that path is allowed. Not evidence of anything either way with only 13 cycles; worth
watching in a longer run.

**The enforced FDR mechanism, exercised live, with zero discoveries throughout:**

| # | p_better | delta | required level | hit |
|---|---|---|---|---|
| 1 | 0.9688 | +0.083 | 0.0304 | no |
| 2 | 0.9688 | +0.083 | 0.0304 | no |
| 3 | 0.9922 | +0.125 | 0.0076 | no |
| 4 | 0.9961 | +0.146 | 0.0034 | no |
| 5 | 0.9688 | +0.083 | 0.0012 | no |
| 6 | 0.9922 | +0.125 | 0.00084 | no |
| 7 | 0.9999 | +0.250 | 0.00062 | no |

Final state: `t=7, d=0`; the next correctness-path test needs `level ~= 0.00047` (an e-value above ~2100) to
hit. Test #7's candidate (`p_better=0.9999`, a near-certain-looking pass-rate gain) was still correctly
rejected -- the clearest illustration in this pilot of the enforced monitor's point: individually convincing
evidence is not automatically treated as a confirmed discovery when the cumulative multiple-testing budget
hasn't earned the room for it, and with zero discoveries so far, nothing has replenished that budget. This
is the intended, anytime-valid behaviour of e-LOND under a cold start, not a malfunction -- and it directly
validates decision 3 (enforce, not monitor-only): Study 3's own pilot accepted a comparable borderline patch
(D1) specifically because it had no equivalent of this check in force.

One of the proposals, inspected directly (ledger + `runs/study4_live/runs/proposal_history.jsonl`): a correct,
unremarkable fix implementing `two_sum` and `flatten` (both were `NotImplementedError` placeholders in the
seed) -- no hard-coding, no signature changes, no suspicious imports. The mechanism blocking it was the
statistics, not a code-quality judgement.

## What is not done

1. ~~**Seed repository.**~~ Done: `write_seed_agent_study4`.
2. ~~**Yield definition and its measurement bug.**~~ Decided and fixed -- see "Known gap and fix" above.
3. ~~**False-discovery monitor.**~~ Decided and wired: `run_live.py` sets `fdr_mode="enforce"`.
4. ~~**Arm-A1 editing rights.**~~ Done: `live_meta.py: MetaAgentA1`, zero `kernel/` edits.
5. ~~**Runner.**~~ Done: `run_study4.py --run` -> `run_live.py`, verified end to end offline for both arms.
6. ~~**e-LOND persistence across restarts.**~~ Found and fixed: `fdr_replay.py`. Verified both in tests and live.
7. ~~**A valid lock.**~~ Locked, covering every file that affects runtime behaviour (including `run_live.py` and
   `fdr_replay.py`, which an earlier lock missed).
8. ~~**An actual live run.**~~ Done: the pilot case study above. 13 cycles, 0 accepts, not a pre-registered
   replicate (see its caveats).
9. **Judge coverage.** Either a judge that tests every stated rule, or an explicit list of the rules it does not
   test. Not written yet.
10. **A pre-registered replicate set.** The pilot above exists to shake out bugs, not to report as a result.
    A real Study 4 result needs its own pre-registration, written fresh now that the code is stable, and run
    without further mid-sequence code changes (the same discipline Study 3 held after D1/D2 were fixed).

## How to lock

1. Review the pre-registration draft/lock candidate and the code it would pin -- every file that affects
   runtime behaviour: `study4/guard.py`, `run_study4.py`, `study2/seed_repo.py`, `study4/live_meta.py`,
   `study4/yield_gate.py`, `study4/run_live.py`, `study4/fdr_replay.py`, plus the current `kernel_digest`.
   An earlier lock in this project missed `run_live.py` and `fdr_replay.py` -- `--check` can only catch a
   changed file if it was listed in `code_sha256` to begin with.
2. Compute `prereg_sha256` over the body without that key, using sorted JSON (`study4.guard.body_sha`).
3. Set `status` to `"locked"`, write the recomputed hash, and commit before any live run.
4. Run `python run_study4.py --check`. It must print `pre-registration OK`.

Any later change to the pinned code or the kernel invalidates the lock (as just happened once already). Study 4
then needs a fresh pre-registration, and any deviation found after seeing a result must be recorded as Study 3's
D1 and D2 were (`results/study3/DEVIATIONS.md`).
