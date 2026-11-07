# SI-Loop: Pre-registered hypotheses and analysis plan

> **Status:** FROZEN PRE-REGISTRATION, EXECUTED (Study 1, report run 2026-11-07). Sections 1-5 (setup, metrics, hypotheses, credences, analysis plan) are **unchanged** from the version committed before the report run (SHA-256 `e2c1d356272b04d3dfb9b39f7545526e8192511315160f87d67a7042665e9764`; recover with `git show <hypotheses-commit>:docs/hypotheses.md`). Only this status line, section 0 and section 6 were edited afterwards, to disclose what actually happened. Outcomes are reported in `paper.pdf` and `results/analysis.json`.
> This pre-registration is informal (no third-party registry); its integrity rests on git commit order and the hashes in `results/config_hashes.txt`.

## 0. What had been observed before the report run (disclosure)

- `--arm full --seed 0` was run for smoke tests and debugging. Seed 0 is **not** a report seed. Report seeds are **1-10**.
- Debugging on seed 0 led to stability guards (tanh-bounded adapter, inner-loop gradient clipping, loss cap, NaN guard) and to a fix of a workspace bug (a rejected round now retries under the last accepted broadcast; before the fix `full`, `no_ws` and `random_ws` were identical). The code was frozen after these changes.
- Observed in that single run, qualitative only: the curriculum (C) drove `hard_share` to ~0 by round 8; the judge accepted rounds where harness `eval_all` got slightly worse (r03, r10); interaction was ~0 after round 0.
- Dev seeds 100-104 were run for all four arms and a gap-monitor sweep was run on `full` before the report run, by an AI assistant. Outcomes were examined. The sweep selection rule was declared first and selected nothing, so **no hyperparameter was changed**. Details: `docs/dev_diagnostics.md`.

## 1. Setup frozen for the report run

| Item | Value |
|---|---|
| Arms | `full`, `no_ws`, `random_ws`, `no_judge` |
| Rounds / steps / meta-batch | 12 / 40 / 8 (defaults) |
| Report seeds | 1-10 (paired: same seed across arms) |
| Dev seeds (tuning only) | 100-104 |
| Tunable on dev seeds | judge thresholds (gap monitor, canary margin), curriculum entropy floor (optional), `K`, `theta`, `ttl` |
| Not tunable after freeze | everything. Tag `frozen-config` before the report run |
| Harness eval sets | `final_*`: seed 90000+s. Curves: seed 91000+s. Neither is seen by any component |

If any hyperparameter changes after the report run is seen, the report seeds are discarded and replaced with fresh ones (11-20), and the change is logged in section 6.

## 2. Metrics

**Primary:** `final_hard` (query MSE on the hard harness set; lower is better).

**Secondary:**
- `final_all` (query MSE, uniform difficulty).
- `auc_eval_all`: mean of `eval_all` across the 12 rounds (sample-efficiency proxy; lower is better).
- `accept_rate`.
- `n_diverged` summed over rounds (stability).

**Effect convention:** `delta(X) = metric(X) - metric(full)`, per seed. `delta > 0` means `full` is better.

**Smallest effect of interest (SESOI):** 0.05 MSE on `final_hard`. (Revisable only before freeze, using seed-to-seed std from dev seeds. Record any change in section 6.)

## 3. Hypotheses and predictions

Probabilities are my credence that the stated outcome will be observed, written before seeing ablation data. **Replace with your own.**

| # | Hypothesis | Comparison | Predicted outcome | Credence | What would refute it |
|---|---|---|---|---|---|
| H1 | The workspace helps | `no_ws` vs `full` | No meaningful difference | 70% that CI includes 0 | 95% CI for `delta(no_ws)` > 0 and mean >= SESOI |
| H2 | Competition helps beyond shared memory | `random_ws` vs `full` | No meaningful difference | 75% that CI includes 0 | 95% CI for `delta(random_ws)` > 0 and mean >= SESOI |
| H3 | The judge helps generalization | `no_judge` vs `full` | `no_judge` equal or slightly worse | 55% `full` better by < SESOI or CI includes 0; 25% `full` clearly better; 20% `no_judge` clearly better | `delta(no_judge)` < 0 with CI excluding 0 means the judge **hurts** |
| H4 | A/B interaction is larger on rounds the judge rejects | `shapley_interaction` | No difference (accepted vs rejected) | 65% | Mean abs interaction on rejected rounds larger than on accepted, CI excluding 0 |

**Rationale for the priors (for transparency):** the workspace broadcast is a 16-dim random projection of scalar training statistics. It carries little task-relevant information for A, so H1 and H2 should be null in this toy. The judge removes noisy-LCB regressions but also rejects roughly 40% of rounds, so its net effect is ambiguous. Interaction was ~0 after round 0 in the single observed run.

## 4. Analysis plan

1. **Per-arm table:** mean +/- std over seeds for `final_hard`, `final_all`, `auc_eval_all`, `accept_rate`.
2. **Paired bootstrap CI** (10,000 resamples over seeds, percentile method) for `delta(arm)` on `final_hard`, for each non-`full` arm. Report both the unadjusted 95% CI and a Bonferroni-adjusted 98.33% CI (three primary comparisons).
3. **Decision rule:** a hypothesis of "X helps" is *supported* only if the adjusted CI excludes 0 in the favorable direction **and** the mean effect is >= SESOI. A CI that includes 0 is reported as "no detectable difference", not "no effect". With 10 seeds, say what effect size the study could detect (use dev-seed std).
4. **H4 procedure:** exclude round 0. For each seed, compute mean `|shapley_interaction|` over accepted rounds and over rejected rounds (seeds with no rejected rounds are dropped and counted). Paired bootstrap on the per-seed difference. Also report the signed version. Pool across arms **only** for the `full` arm; report others separately.
5. **Learning curves:** `eval_all` and `eval_hard` per round, mean +/- SEM over seeds, for all arms.
6. **Exploratory (labelled as such):** correlation of `shapley_phi_A` with `lcb`; frequency of accepted rounds where `eval_all` worsened; curriculum `hard_share` trajectories per arm.
7. **Exclusions:** none. Runs with `n_diverged > 0` are kept and counted.

## 5. How results will be reported

- Every number in the paper comes from `results/ablation.json`.
- Null and negative results are reported in the same detail as positive ones.
- Claims are limited to this synthetic sinusoid-regression toy.

## 6. Deviations log

| Date | Change | Reason | Made before/after seeing report results? |
|---|---|---|---|
| 2026-11-07 | Minimum detectable effect computed from the SD of paired differences of the report seeds (normal approximation, 80% power, Bonferroni), not from dev-seed std as written in section 4.3 | Simpler and uses the data actually analysed; the paper states the method | After |
| 2026-11-07 | Paired bootstrap CIs (95% and Bonferroni 98.33%) also reported for `final_all` and `auc_eval_all`, which section 4.2 specified only for `final_hard` | Secondary metrics; decision rule applies to `final_hard` only. The `random_ws` advantage on `auc_eval_all` is therefore a secondary, uncorrected-across-metrics finding | After |
| 2026-11-07 | None of the hyperparameters in section 1 were changed after dev diagnostics | Pre-declared sweep rule selected nothing | Before |
