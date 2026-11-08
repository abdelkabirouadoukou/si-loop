"""Builds every number, table and figure used in paper_study2.tex from raw result files.
    python results/study2/make_paper_numbers.py --abl results/study2/ablation --demo results/study2/demo
Inputs : ablation/A*_s*/result.json + prereg.json, calibration.json, verifier_quality.json, demo/final_eval.json + summary.json
Outputs: numbers.tex, tab_ablation.tex, tab_calibration.tex, fig_calibration.pdf, ablation_summary.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from kernel.stats import bootstrap_ci, sign_flip_p                      # noqa: E402
from run_ablations import HYPOTHESES, holm                              # noqa: E402


WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def w(text):
    """LaTeX macro names cannot contain digits: A0 -> AZero, H1 -> HOne."""
    return "".join(WORD.get(ch, ch) for ch in text)


def f(x, d=3):
    return f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", default=str(HERE / "ablation"))
    ap.add_argument("--demo", default=str(HERE / "demo"))
    a = ap.parse_args()
    abl, demo = Path(a.abl), Path(a.demo)
    M = {}                                                              # macro name -> text
    runs = {}
    for p in sorted(abl.glob("A*_s*/result.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    pre = json.loads((abl / "prereg.json").read_text())
    M["PreregHash"] = pre["prereg_sha256"][:12]
    M["NSeeds"] = str(len(pre["seeds"]))
    M["NCycles"] = str(pre["cycles"])
    M["AblProfile"] = pre["profile"]

    # ---- ablation table
    rows, summ = [], {"arms": {}, "tests": {}}
    for arm in ("A0", "A1", "A2", "A3"):
        rs = list(runs.get(arm, {}).values())
        if not rs:
            continue
        g = [r["gain_selection"] for r in rs]
        h = [r["gain_heldout"] for r in rs]
        acc = [r["accepts"] for r in rs]
        tok = [r["tokens"] + r["yield_test_tokens"] for r in rs]
        sd = lambda x: st.stdev(x) if len(x) > 1 else 0.0
        summ["arms"][arm] = {"n": len(rs), "gain_sel_mean": st.mean(g), "gain_sel_sd": sd(g), "gain_held_mean": st.mean(h),
                             "accepts": st.mean(acc), "tokens": st.mean(tok), "halted": sum(bool(r["halted"]) for r in rs),
                             "ledger_ok": all(r["ledger_ok"] for r in rs), "flags": sum(r["security_flags"] for r in rs)}
        rows.append(f"{arm} & {len(rs)} & ${st.mean(g):+.3f}\\pm{sd(g):.3f}$ & ${st.mean(h):+.3f}$ & {st.mean(acc):.1f} & {st.mean(tok):,.0f} \\\\")
    Path(HERE / "tab_ablation.tex").write_text(
        "\\begin{tabular}{lcccrr}\\toprule\nArm & seeds & gain, selection families & gain, held-out & accepts & tokens \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

    def paired(x, y, key):
        sx, sy = runs.get(x, {}), runs.get(y, {})
        return [sx[s][key] - sy[s][key] for s in sorted(set(sx) & set(sy))]
    ps, tests = {}, {}
    for h, (x, y, key) in {"H1": ("A1", "A0", "gain_selection"), "H2": ("A0", "A2", "gain_selection"),
                           "H3": ("A0", "A3", "gain_selection")}.items():
        d = paired(x, y, key)
        if d:
            ps[h], tests[h] = sign_flip_p(d), d
    adj = holm(ps) if ps else {}
    trows = []
    for h, d in tests.items():
        lo, hi = bootstrap_ci(d)
        x, y, _ = {"H1": ("A1", "A0", ""), "H2": ("A0", "A2", ""), "H3": ("A0", "A3", "")}[h]
        summ["tests"][h] = {"mean": st.mean(d), "ci": [lo, hi], "p": ps[h], "holm": adj[h], "n": len(d)}
        trows.append(f"{h} & {x}$-${y} & {len(d)} & ${st.mean(d):+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & {ps[h]:.4f} & {adj[h]:.4f} \\\\")
        for k, v in (("Mean", f(st.mean(d))), ("P", f"{ps[h]:.4f}"), ("Holm", f"{adj[h]:.4f}"), ("Lo", f(lo)), ("Hi", f(hi))):
            M[w(f"{h}{k}")] = v
    d4 = paired("A1", "A0", "gain_heldout")
    if d4:
        M["HFourMean"] = f(st.mean(d4))
        summ["tests"]["H4"] = {"mean": st.mean(d4), "n": len(d4)}
    Path(HERE / "tab_tests.tex").write_text(
        "\\begin{tabular}{llcccrr}\\toprule\nHyp. & contrast & pairs & mean diff. & 95\\% bootstrap CI & p (one-sided) & Holm p \\\\\\midrule\n"
        + "\n".join(trows) + "\n\\bottomrule\\end{tabular}")
    for arm, s in summ["arms"].items():
        M[w(f"Gain{arm}")] = f"{s['gain_sel_mean']:+.3f}"
        M[w(f"Held{arm}")] = f"{s['gain_held_mean']:+.3f}"
    M["MinP"] = f"{1 / 2 ** len(pre['seeds']):.4f}"
    M["BadRuns"] = str(sum(s["halted"] + s["flags"] + (not s["ledger_ok"]) for s in summ["arms"].values()))

    # ---- calibration
    cal = json.loads((HERE / "calibration.json").read_text())
    rows = [f"{r['p_improve_min']:.3f} & {100 * r['single']:.1f}\\% & {100 * r['two_stage']:.1f}\\% \\\\" for r in cal["false_accept"]]
    Path(HERE / "tab_calibration.tex").write_text(
        "\\begin{tabular}{lcc}\\toprule\n$p_{\\mathrm{improve\\_min}}$ & single stage & two stage \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    fa95 = next(r for r in cal["false_accept"] if abs(r["p_improve_min"] - 0.95) < 1e-9)
    M["FASingle"], M["FATwo"] = f"{100 * fa95['single']:.1f}", f"{100 * fa95['two_stage']:.1f}"
    for r in cal["power"]:
        k = w(f"{int(round(r['lift'] * 100)):02d}")
        M[f"PowSingle{k}"], M[f"PowTwo{k}"] = f"{100 * r['single']:.0f}", f"{100 * r['two_stage']:.0f}"
    M["CalN"], M["CalTrials"] = str(cal["n"]), str(cal["trials"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    g = [r["p_improve_min"] for r in cal["false_accept"]]
    ax[0].plot(g, [100 * r["single"] for r in cal["false_accept"]], "o-", label="single stage", color="#c0392b")
    ax[0].plot(g, [100 * r["two_stage"] for r in cal["false_accept"]], "s-", label="two stage", color="#1f4e79")
    ax[0].axhline(5, ls=":", color="gray")
    ax[0].set(xlabel=r"$p_{\mathrm{improve\_min}}$", ylabel="false-accept rate of null patches (%)")
    ax[0].legend(frameon=False)
    l = [100 * r["lift"] for r in cal["power"]]
    ax[1].plot(l, [100 * r["single"] for r in cal["power"]], "o-", color="#c0392b")
    ax[1].plot(l, [100 * r["two_stage"] for r in cal["power"]], "s-", color="#1f4e79")
    ax[1].set(xlabel="true pass-rate gain of the patch (points)", ylabel="accept rate (%)")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_calibration.pdf")

    # ---- verifier quality
    vq = HERE / "verifier_quality.json"
    if vq.exists():
        q = json.loads(vq.read_text())
        M["MutN"], M["MutKilled"] = str(q["mutants"]), str(q["killed"])
        M["MutRate"], M["MutAdj"] = f"{100 * q['kill_rate']:.1f}", f"{100 * q['kill_rate_adjusted']:.1f}"
        M["MutEquiv"], M["RefFalseFail"] = str(q["likely_equivalent"]), str(q["reference_false_fail"])
        kt = q["kills_by_tier"]
        M["KillSmall"], M["KillFuzz"], M["KillMeta"] = (str(kt.get(t, 0)) for t in ("small", "fuzz", "meta"))
        ot = q["kills_only_this_tier"]
        M["OnlySmall"], M["OnlyFuzz"], M["OnlyMeta"] = (str(ot.get(t, 0)) for t in ("small", "fuzz", "meta"))
        M["HoneypotOK"] = "yes" if q.get("honeypot_detects") else "no"

    # ---- reference (study2) demo run
    fe = demo / "final_eval.json"
    if fe.exists():
        e = json.loads(fe.read_text())
        s = json.loads((demo / "summary.json").read_text())
        M["DemoAnchor"], M["DemoChamp"] = f(e["selection_families"]["anchor"], 2), f(e["selection_families"]["champion"], 2)
        M["DemoHeldA"], M["DemoHeldC"] = f(e["heldout_family"]["anchor"], 2), f(e["heldout_family"]["champion"], 2)
        M["DemoGap"] = f(e["cross_family_gap"], 2)
        M["DemoFixed"], M["DemoBroken"] = str(e["selection_families"]["fixed"]), str(e["selection_families"]["broken"])
        M["DemoAcc"] = str(s["verdicts"].get("accept", 0))
        M["DemoRej"] = str(s["verdicts"].get("reject", 0))
        M["DemoStep"] = str(s["verdicts"].get("stepping_stone", 0))
        M["DemoCycles"] = str(s["cycles"])
        M["DemoLedger"] = "verified" if s["ledger_ok"] else "BROKEN"
    out = ["% generated by results/study2/make_paper_numbers.py: do not edit"]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    Path(HERE / "numbers.tex").write_text("\n".join(out) + "\n")
    Path(HERE / "ablation_summary.json").write_text(json.dumps(summ, indent=1))
    print("wrote", len(M), "macros")


if __name__ == "__main__":
    main()
