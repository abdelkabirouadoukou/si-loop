# Study 4 (Phase 4): the improver arm

**Status: not implemented. No Study 4 run has been made, and no Study 4 pre-registration is locked.**

This folder holds the foundation for Study 4 only. It does not contain the improver loop, the seed repository
with `agent/meta/`, or any result.

## What is here

| File | Purpose | State |
|---|---|---|
| `DESIGN_DRAFT.md` | Draft design: question, arms, metric, open decisions | draft, not locked |
| `guard.py` | Refuses to run unless a locked pre-registration matches the code and kernel | working, tested |
| `../run_study4.py` | Entry point. `--check` verifies the pre-registration. `--run` is refused until the runner exists | working, tested |
| `../tests/test_study4_guard.py` | Tests for the guard and entry point, using temporary pre-registrations | passing |

## What the guard checks

A run is allowed only if `results/study4/prereg.json` exists and all of the following hold:

1. the file is valid JSON and has every required key;
2. `status` is `"locked"`;
3. `prereg_sha256` matches the SHA-256 of the sorted-JSON body (the same scheme as Study 3);
4. each file in `code_sha256` still matches its hash on disk;
5. `kernel_digest` matches the current kernel.

Any failure is a refusal with a message saying which check failed. The guard does not read API keys and does not
write files.

## What is not done (required before a Study 4 run)

1. **Seed repository.** A small `agent/meta/` package the improver can edit. Study 3's seed has none.
2. **Yield definition.** A written definition of what counts as an improver that raises yield, and a check that
   `kernel/yield_test.py` (`YieldGate`) implements it.
3. **False-discovery monitor.** Decide whether it is enforced or stays in monitor mode.
4. **Judge coverage.** Either a judge that tests every stated rule, or an explicit list of the rules it does not test.
5. **Locked pre-registration.** Written, reviewed, and locked by the researcher. The file is not created by this code.
6. **Runner.** The improver loop itself, wired to the guard. `run_study4.py --run` must not be changed to bypass it.

## How to lock (for later)

1. Write `results/study4/prereg.json` with `"status": "draft"` and every required key.
2. Review it. Fill in `code_sha256` and `kernel_digest` from the code that will run.
3. Compute `prereg_sha256` over the body without that key, using sorted JSON.
4. Change `status` to `"locked"`, recompute the hash, and commit before any live run.
5. Run `python run_study4.py --check`. It must print `pre-registration OK`.

Any later change to the code or kernel invalidates the lock. Study 4 then needs a new pre-registration, and any
deviation must be recorded as Study 3's D1 and D2 were.
