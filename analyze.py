"""Analysis per docs/hypotheses.md (section 4). Every number comes from the results JSON.

    python analyze.py results/ablation.json --out results/
"""
import argparse
import json
import os

import numpy as np

SESOI = 0.05          # smallest effect of interest on final_hard (from hypotheses.md)
N_BOOT = 10_000
N_COMPARISONS = 3     # three primary comparisons -> Bonferroni


def load(path):
    return json.load(open(path))


def per_seed(res, arm, fn):
    return np.array([fn(run) for run in res[arm]])


def auc_eval_all(run):
    return float(np.mean([h["eval_all"] for h in run["history"]]))


def n_diverged(run):
    return float(np.sum([h["n_diverged"] for h in run["history"]]))


def paired_boot(d, rng, alpha):
    """Percentile bootstrap CI for the mean of paired differences d."""
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="results/")
    a = ap.parse_args()
    res = load(a.path)
    arms = list(res)
    seeds = [r["seed"] for r in res[arms[0]]]
    for arm in arms:   # paired design requires identical seed lists
        assert [r["seed"] for r in res[arm]] == seeds, "seed lists differ across arms"
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    out = {"seeds": seeds, "n_seeds": len(seeds)}

    # 1) per-arm table
    metrics = {
        "final_hard": lambda r: r["final_hard"],
        "final_all": lambda r: r["final_all"],
        "auc_eval_all": auc_eval_all,
        "accept_rate": lambda r: r["accept_rate"],
        "n_diverged": n_diverged,
    }
    table = {}
    print(f"{'arm':<10}" + "".join(f"{m:>22}" for m in metrics))
    for arm in arms:
        row = {}
        line = f"{arm:<10}"
        for m, fn in metrics.items():
            v = per_seed(res, arm, fn)
            row[m] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), values=v.tolist())
            line += f"{v.mean():>12.3f} ±{v.std(ddof=1):>7.3f}"
        table[arm] = row
        print(line)
    out["table"] = table

    # 2) paired bootstrap deltas vs full, on each metric of interest
    deltas = {}
    print("\ndelta(arm) = metric(arm) - metric(full); > 0 means full is better (lower-is-better metrics)")
    for m in ["final_hard", "final_all", "auc_eval_all"]:
        base = per_seed(res, "full", metrics[m])
        for arm in arms:
            if arm == "full":
                continue
            d = per_seed(res, arm, metrics[m]) - base
            lo, hi = paired_boot(d, rng, 0.05)
            lo_b, hi_b = paired_boot(d, rng, 0.05 / N_COMPARISONS)
            deltas[f"{m}:{arm}"] = dict(mean=float(d.mean()), ci95=[float(lo), float(hi)],
                                        ci_bonf=[float(lo_b), float(hi_b)], per_seed=d.tolist())
            print(f"{m:<13} {arm:<10} mean={d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"Bonf 98.33% CI [{lo_b:+.3f}, {hi_b:+.3f}]")
    out["deltas"] = deltas

    # 3) decision rule on the primary metric (final_hard), per hypothesis
    print("\nDecision rule on final_hard (supported only if Bonferroni CI excludes 0 favorably AND mean >= SESOI):")
    verdicts = {}
    for h, arm in [("H1", "no_ws"), ("H2", "random_ws"), ("H3", "no_judge")]:
        dd = deltas[f"final_hard:{arm}"]
        lo, hi = dd["ci_bonf"]
        if lo > 0 and dd["mean"] >= SESOI:
            v = "full clearly better (supported: 'helps')"
        elif hi < 0 and -dd["mean"] >= SESOI:
            v = "full clearly WORSE (the component hurts)"
        else:
            v = "no detectable difference (NOT 'no effect')"
        verdicts[h] = v
        print(f"  {h} full vs {arm}: {v}")
    out["verdicts"] = verdicts

    # detectable effect with these seeds (std of paired diffs, normal approx, 80% power, alpha 0.05/3)
    from math import sqrt
    sd_d = {k: float(np.std(v["per_seed"], ddof=1)) for k, v in deltas.items() if k.startswith("final_hard")}
    z = 2.394 + 0.842          # z_{1-0.0167/2}=2.394 (two-sided Bonferroni 0.05/3), z_power(80%)=0.842
    out["mde_final_hard"] = {k: z * s / sqrt(len(seeds)) for k, s in sd_d.items()}
    print("\nApprox. minimum detectable effect on final_hard (80% power, Bonferroni):",
          {k.split(':')[1]: round(v, 3) for k, v in out["mde_final_hard"].items()})

    # 4) H4: |interaction| on accepted vs rejected rounds, excluding round 0, full arm
    acc_m, rej_m, acc_s, rej_s, dropped = [], [], [], [], 0
    for run in res["full"]:
        hs = [h for h in run["history"] if h["round"] > 0]
        a_i = [h["shapley_interaction"] for h in hs if h["accept"]]
        r_i = [h["shapley_interaction"] for h in hs if not h["accept"]]
        if not a_i or not r_i:
            dropped += 1
            continue
        acc_m.append(np.mean(np.abs(a_i))); rej_m.append(np.mean(np.abs(r_i)))
        acc_s.append(np.mean(a_i)); rej_s.append(np.mean(r_i))
    h4 = dict(n_used=len(acc_m), n_dropped=dropped)
    if len(acc_m) >= 2:
        d_abs = np.array(rej_m) - np.array(acc_m)       # > 0 => rejected rounds have larger |interaction|
        d_sig = np.array(rej_s) - np.array(acc_s)
        h4.update(abs_mean_diff=float(d_abs.mean()), abs_ci95=[float(x) for x in paired_boot(d_abs, rng, 0.05)],
                  signed_mean_diff=float(d_sig.mean()), signed_ci95=[float(x) for x in paired_boot(d_sig, rng, 0.05)],
                  mean_abs_accepted=float(np.mean(acc_m)), mean_abs_rejected=float(np.mean(rej_m)))
        print(f"\nH4 (full arm, round 0 excluded): seeds used={len(acc_m)}, dropped={dropped}")
        print(f"  mean |interaction| accepted={np.mean(acc_m):.4f}  rejected={np.mean(rej_m):.4f}  "
              f"diff(rej-acc)={d_abs.mean():+.4f}  95% CI [{h4['abs_ci95'][0]:+.4f}, {h4['abs_ci95'][1]:+.4f}]")
    out["H4"] = h4

    # 5) exploratory
    phiA, lcb, worse, acc_total = [], [], 0, 0
    for run in res["full"]:
        prev = None
        for h in run["history"]:
            phiA.append(h["shapley_phi_A"]); lcb.append(h["lcb"])
            if prev is not None and h["accept"]:
                acc_total += 1
                worse += h["eval_all"] > prev + 1e-12
            prev = h["eval_all"]
    out["exploratory"] = dict(corr_phiA_lcb=float(np.corrcoef(phiA, lcb)[0, 1]),
                              accepted_rounds_eval_all_worse=int(worse), accepted_rounds_total=int(acc_total))
    print("\nExploratory:", out["exploratory"])

    json.dump(out, open(os.path.join(a.out, "analysis.json"), "w"), indent=2)

    # 6) figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    means = [table[x]["final_hard"]["mean"] for x in arms]
    sems = [table[x]["final_hard"]["std"] / np.sqrt(len(seeds)) for x in arms]
    ax.bar(arms, means, yerr=sems, capsize=4, color="#4c78a8")
    for i, x in enumerate(arms):
        ax.scatter([i] * len(seeds), table[x]["final_hard"]["values"], s=10, color="k", alpha=0.5, zorder=3)
    ax.set_ylabel("final_hard (query MSE, lower is better)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "final_hard.png"), dpi=200); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for ax, key, title in [(axs[0], "eval_all", "eval_all (uniform difficulty)"), (axs[1], "eval_hard", "eval_hard")]:
        for arm in arms:
            c = np.array([[h[key] for h in run["history"]] for run in res[arm]])
            x = np.arange(1, c.shape[1] + 1); m = c.mean(0); se = c.std(0, ddof=1) / np.sqrt(len(c))
            ax.plot(x, m, label=arm); ax.fill_between(x, m - se, m + se, alpha=0.2)
        ax.set_xlabel("round"); ax.set_ylabel("query MSE"); ax.set_title(title)
    axs[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "curves.png"), dpi=200); plt.close()
    print("\nwrote", os.path.join(a.out, "analysis.json"), "final_hard.png curves.png")


if __name__ == "__main__":
    main()
