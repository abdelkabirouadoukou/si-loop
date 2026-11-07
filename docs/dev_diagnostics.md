# Dev-seed diagnostics (run before the report run)

Disclosure addendum to `docs/hypotheses.md`. **Commit this file before `results/ablation.json`.**
Dev seeds only (100-104). No report seed (1-10) had been run when this was written.

## What was run

- `python run_parallel.py --seeds 100 101 102 103 104 --out results/dev.json` (all 4 arms, defaults 12 rounds / 40 steps / meta-batch 8).
- A gap-monitor sweep on the `full` arm only (dev seeds), with the selection rule **declared before running**:
  "choose the smallest gap-monitor relative threshold whose mean dev accept rate is >= 0.4; do not use `final_*` metrics for selection."

## Observations (dev seeds, qualitative; not evidence for any hypothesis)

| Quantity | `full` | `no_ws` | `random_ws` | `no_judge` |
|---|---|---|---|---|
| mean accept rate | 0.17 | 0.17 | 0.15 | 1.00 (forced) |
| rounds with `lcb <= 0` | 0.50 | 0.47 | 0.48 | 0.52 |
| rounds flagged `gap_bad` | 0.75 | 0.73 | 0.75 | 0.82 (logged only) |
| rounds flagged `canary_bad` | 0.28 | 0.30 | 0.33 | 0.22 (logged only) |
| mean `final_all` | 1.082 | 1.075 | 1.096 | 0.892 |
| mean `final_hard` | 1.213 | 1.206 | 1.210 | 1.195 |

- Predicting 0 for every input gives MSE about 1.29 on this task family (E[a^2]/2 with a ~ U(0.5, 2.5)). All arms are close to that on the hard set, so the toy barely learns hard tasks at this budget.
- The curriculum (C) concentrates on easy bins (`hard_share` about 0.02 or lower by the last round).
- On dev seeds 102-104 the judge accepted only round 0, so `full`, `no_ws` and `random_ws` produced identical results there: the workspace broadcast only reaches the models in rounds after an acceptance, and none was accepted.
- The gap monitor is anchored to the round-0 generalisation gap (updated only on accepted rounds), so it flags learning progress as "overfitting" in most rounds.

## Gap-monitor sweep (`full` arm, dev seeds 100-104)

| relative threshold | mean accept rate |
|---|---|
| 0.25 (current default) | 0.17 |
| 0.5 | 0.17 |
| 1.0 | 0.20 |
| 2.0 | 0.23 |
| monitor off | 0.33 |

## Decision

No gap-monitor setting reached the pre-declared target (accept rate >= 0.4), even with the monitor disabled. The rule therefore selected nothing, and **the report run uses the unchanged pre-registered defaults**. No thresholds were tuned. The strictness of the judge (gap monitor, lower-confidence-bound gate and canary together) is reported as a finding and a limitation, and a recalibrated judge is left to a separate, newly pre-registered experiment.

## Integrity record

SHA-256 of the exact files used for the report run is in `results/config_hashes.txt`. Compare with `shasum -a 256 si_loop_toy.py` on your own checkout: it must match before you tag `frozen-config`.
