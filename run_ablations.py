#!/usr/bin/env python3
"""run_ablations.py: PRE-REGISTERED control arms for Study 2.

    A0  frozen improver        solver evolves, improver is M0 (never edits itself)       -> ITERATIVE improvement
    A1  full recursive loop    solver AND improver evolve; improver patches go through the YIELD TEST
    A2  random-patch control   knowledge-free random edits through the SAME gate           -> the gate's own selection effect
    A3  compute-matched best-of-N   N = #cycles proposals from M0, each against the SEED agent only
                               (no accumulation); the best single patch wins              -> is iterating better than sampling?

All arms: same seed agent, same sealed world, same two-stage judge, same cycle budget, and the SAME fresh
`final` evaluation batches (paired across arms). The unit of analysis is the RUN (seed), never the cycle.

    python run_ablations.py prereg  --out runs/ablation --seeds 6 --cycles 40 --profile smoke
    python run_ablations.py run     --out runs/ablation [--jobs 2] [--only A1]
    python run_ablations.py analyze --out runs/ablation
    python run_ablations.py all     --out runs/ablation --seeds 6 --cycles 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kernel.orchestrator import KernelGuard, Orchestrator
from kernel.stats import bootstrap_ci, sign_flip_p
from kernel.world import KERNEL_DIR, World

ARMS = {"A0": "frozen improver (solver only)", "A1": "full recursive loop (solver + improver, yield-tested)",
        "A2": "random-patch control (same gate)", "A3": "compute-matched best-of-N from the seed agent"}

HYPOTHESES = {
    "H1": "A1 - A0 > 0 on final selection gain (evolving the improver adds improvement beyond a frozen improver)",
    "H2": "A0 - A2 > 0 (informed search + gate beats random patches + gate: the gate alone is not the effect)",
    "H3": "A0 - A3 > 0 (iterating and accumulating beats compute-matched sampling of single patches)",
    "H4": "A1 - A0 on the HELD-OUT family gain >= -0.02 (no cross-family overfitting cost)",
}
PLAN = ("Unit = run (seed). Paired by seed. Report mean difference, 95% bootstrap CI and one-sided sign-flip p; "
        "Holm correction over H1-H3. A recursive-improvement claim requires H1 (Holm-adjusted p < 0.05) AND H4 AND "
        "no halted/tampered run. Anything else is reported as iterative improvement only.")


class BestOfNOrchestrator(Orchestrator):
    """A3: every proposal is made against the SEED agent; the champion is the best single patch."""

    def _select_parent(self, champion: str) -> str:
        return self.anchor.anchor_sha


def prereg_hash(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "prereg_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def secret_for(label: str) -> bytes:
    import os
    env = os.environ.get("SI_LOOP_SECRET")
    return env.encode() if env else hashlib.sha256(label.encode()).digest()


# =============================================================== prereg
def cmd_prereg(a) -> Path:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = {"arms": a.arms, "seeds": list(range(a.seeds)), "cycles": a.cycles, "profile": a.profile,
         "world_secret_label": "study2-synthetic-world-v1", "parent_policy": "clade",
         "anchor_every": max(5, a.cycles // 4), "heldout_every": max(10, a.cycles // 2),
         "yield": {"n_starts": a.yield_starts, "seeds": a.yield_seeds, "k_cycles": a.yield_k, "n_eval": 32},
         "final_eval": {"n": 96, "cycle_id": 7_000_000}, "hypotheses": HYPOTHESES, "analysis_plan": PLAN,
         "kernel_digest": KernelGuard(KERNEL_DIR).digest(), "arms_description": {k: ARMS[k] for k in a.arms}}
    d["prereg_sha256"] = prereg_hash(d)
    path = out / "prereg.json"
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists (pre-registrations are immutable; use --force to overwrite)")
    path.write_text(json.dumps(d, indent=1, sort_keys=True))
    (out / "prereg.sha256").write_text(d["prereg_sha256"] + "\n")
    print(f"pre-registered {len(a.arms)} arms x {a.seeds} seeds x {a.cycles} cycles -> {path}\nsha256 {d['prereg_sha256']}\n"
          "COMMIT this file (and the hash) BEFORE running the arms.")
    return path


def load_prereg(out: Path, force: bool = False) -> dict:
    d = json.loads((Path(out) / "prereg.json").read_text())
    if prereg_hash(d) != d["prereg_sha256"] or (Path(out) / "prereg.sha256").read_text().strip() != d["prereg_sha256"]:
        raise SystemExit("prereg.json was modified after pre-registration: refusing to run")
    if KernelGuard(KERNEL_DIR).digest() != d["kernel_digest"] and not force:
        raise SystemExit("kernel changed since pre-registration: refusing to run (use --force to override and re-register)")
    return d


# =============================================================== one run
def run_one(out: str, arm: str, seed: int) -> dict:
    out = Path(out)
    d = json.loads((out / "prereg.json").read_text())
    res_path = out / f"{arm}_s{seed}" / "result.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    work = out / f"{arm}_s{seed}"
    shutil.rmtree(work / "repo", ignore_errors=True)
    shutil.rmtree(work / "runs", ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    world = World(secret_for(d["world_secret_label"]), d["profile"])
    world.init_repo(work / "repo")
    common = dict(seed=seed, anchor_every=d["anchor_every"], heldout_every=d["heldout_every"], gc_every=5,
                  parent_policy=d["parent_policy"])
    kw, yh = {}, None
    if arm == "A0":
        meta = World.make_meta("m0", seed)
    elif arm == "A1":
        meta = World.make_meta("repo", seed, allow_meta=True)

        def factory(orch):
            from kernel.yield_test import YieldConfig, YieldGate, YieldHarness
            y = d["yield"]
            h = YieldHarness(world, orch.repo, YieldConfig(n_starts=y["n_starts"], seeds=tuple(range(y["seeds"])),
                                                           k_cycles=y["k_cycles"], n_eval=y["n_eval"]), work / "yield")
            return YieldGate(h, orch)
        kw["yield_gate_factory"] = factory
    elif arm == "A2":
        meta = World.make_meta("random", seed)
    elif arm == "A3":
        meta = World.make_meta("m0", seed)
        kw["orch_cls"] = BestOfNOrchestrator
        common["parent_policy"] = "archive"
    else:
        raise SystemExit(f"unknown arm {arm}")
    t0 = time.time()
    orch = world.build(work / "repo", work / "runs", meta, **kw, **common)
    summary = orch.run(d["cycles"])
    # ---- final paired evaluation on FRESH batches shared by every arm and seed
    fe = d["final_eval"]
    sel = orch.verifier.draw_batch(fe["cycle_id"], fe["n"], "final")
    held = orch.verifier.draw_batch(fe["cycle_id"], fe["n"], "heldout")
    champ, anchor = orch.repo.head("main"), orch.anchor.anchor_sha
    ev = lambda sha, b: orch.judge.eval_sha(sha, b)
    traj = []
    for rec in orch.ledger.records():
        if rec.get("ev") == "end" and rec.get("verdict") == "accept":
            pass
    for tag in sorted(t for t in orch.repo._git("tag", "--list", "champ/*").split()):
        traj.append({"cycle": int(tag.split("/")[1]), "pass": ev(orch.repo.head(tag), sel).pass_at_1})
    ends = [r for r in orch.ledger.records() if r.get("ev") == "end"]
    tokens = sum(r.get("cost_tokens", 0) for r in ends)
    yield_tokens = 0
    for r in ends:
        y = (r.get("detail") or {}).get("yield")
        if y:
            yield_tokens += y.get("tokens_cand", 0) + y.get("tokens_m0", 0)
    half = len(ends) // 2
    rate = lambda rs: sum(r["verdict"] == "accept" for r in rs) / max(1, len(rs))
    res = {"arm": arm, "seed": seed, "cycles": summary["cycles"], "accepts": summary["accepts"], "halted": summary["halted"],
           "anchor_pass": ev(anchor, sel).pass_at_1, "final_pass": ev(champ, sel).pass_at_1,
           "gain_selection": ev(champ, sel).pass_at_1 - ev(anchor, sel).pass_at_1,
           "anchor_heldout": ev(anchor, held).pass_at_1, "final_heldout": ev(champ, held).pass_at_1,
           "gain_heldout": ev(champ, held).pass_at_1 - ev(anchor, held).pass_at_1,
           "accept_rate_first_half": rate(ends[:half]), "accept_rate_second_half": rate(ends[half:]),
           "meta_patches": sum((r.get("scope") == "meta") for r in ends),
           "meta_accepts": sum((r.get("scope") == "meta" and r["verdict"] == "accept") for r in ends),
           "tokens": tokens, "yield_test_tokens": yield_tokens, "margin_spent": orch.budget.spent_total,
           "rollbacks": sum(r.get("ev") == "rollback" for r in orch.ledger.records()),
           "security_flags": sum(r.get("security_flags", 0) for r in ends),
           "ledger_ok": orch.ledger.verify()[0], "trajectory": traj, "wall_s": round(time.time() - t0, 1)}
    res_path.write_text(json.dumps(res, indent=1))
    shutil.rmtree(work / "repo", ignore_errors=True)               # keep results + ledger, drop checkouts
    return res


def cmd_run(a) -> None:
    d = load_prereg(Path(a.out), a.force)
    jobs = [(arm, s) for s in d["seeds"] for arm in d["arms"] if not a.only or arm in a.only]
    print(f"{len(jobs)} runs, {a.jobs} parallel")
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(run_one, str(a.out), arm, s) for arm, s in jobs]
            for (arm, s), f in zip(jobs, futs):
                r = f.result()
                print(f"  {arm} seed {s}: gain {r['gain_selection']:+.3f} heldout {r['gain_heldout']:+.3f} accepts {r['accepts']} ({r['wall_s']}s)")
    else:
        for arm, s in jobs:
            r = run_one(str(a.out), arm, s)
            print(f"  {arm} seed {s}: gain {r['gain_selection']:+.3f} heldout {r['gain_heldout']:+.3f} accepts {r['accepts']} ({r['wall_s']}s)")


# =============================================================== analysis
def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, run_max = {}, 0.0
    for i, k in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run_max
    return out


def analyze(out: Path) -> str:
    d = json.loads((Path(out) / "prereg.json").read_text())
    runs = {}
    for f in sorted(Path(out).glob("A*_s*/result.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    L = [f"PRE-REGISTERED ANALYSIS  (prereg sha256 {d['prereg_sha256'][:16]}...)", ""]
    L.append(f"{'arm':4s} {'n':>2s} {'gain_sel':>9s} {'gain_held':>10s} {'accepts':>8s} {'tokens':>9s} {'meta_acc':>8s} {'acc1/acc2':>10s}")
    for arm in sorted(runs):
        rs = list(runs[arm].values())
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        L.append(f"{arm:4s} {len(rs):2d} {m('gain_selection'):+9.3f} {m('gain_heldout'):+10.3f} {m('accepts'):8.1f} "
                 f"{m('tokens') + m('yield_test_tokens'):9.0f} {m('meta_accepts'):8.1f} "
                 f"{m('accept_rate_first_half'):.2f}/{m('accept_rate_second_half'):.2f}")
    L.append("")

    def paired(a, b, key):
        sa, sb = runs.get(a, {}), runs.get(b, {})
        seeds = sorted(set(sa) & set(sb))
        diffs = [sa[s][key] - sb[s][key] for s in seeds]
        return diffs

    tests, ps = {}, {}
    for h, (a, b, key) in {"H1": ("A1", "A0", "gain_selection"), "H2": ("A0", "A2", "gain_selection"),
                           "H3": ("A0", "A3", "gain_selection")}.items():
        df = paired(a, b, key)
        if df:
            ps[h] = sign_flip_p(df)
            tests[h] = (a, b, key, df)
    adj = holm(ps)
    for h, (a, b, key, df) in tests.items():
        lo, hi = bootstrap_ci(df)
        L.append(f"{h}: {a}-{b} on {key}: mean {sum(df) / len(df):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                 f"p(one-sided)={ps[h]:.4f}  Holm p={adj[h]:.4f}  n={len(df)}   <- {HYPOTHESES[h]}")
    df4 = paired("A1", "A0", "gain_heldout")
    if df4:
        lo, hi = bootstrap_ci(df4)
        ok4 = sum(df4) / len(df4) >= -0.02
        L.append(f"H4: A1-A0 on gain_heldout: mean {sum(df4) / len(df4):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                 f"(non-regression margin -0.02: {'OK' if ok4 else 'VIOLATED'})")
    bad = [f"{r['arm']}_s{r['seed']}" for rs in runs.values() for r in rs.values()
           if r["halted"] or not r["ledger_ok"] or r["security_flags"]]
    L += ["", f"halted / tampered / flagged runs: {bad or 'none'}", "", "VERDICT"]
    h1 = adj.get("H1")
    if h1 is not None and h1 < 0.05 and not bad and (not df4 or sum(df4) / len(df4) >= -0.02):
        L.append("  H1 supported (Holm p < 0.05), H4 holds, no halted runs: evidence of SCAFFOLD-LEVEL recursive improvement "
                 "(the improver policy, not model weights).")
    else:
        L.append("  H1 NOT supported at the pre-registered level: report ITERATIVE improvement only (see A0 vs A2/A3).")
    L.append("  (Interpretation rules are fixed in prereg.json; do not change them after seeing results.)")
    text = "\n".join(L)
    (Path(out) / "analysis.txt").write_text(text + "\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python run_ablations.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prereg", "run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="runs/ablation")
        p.add_argument("--force", action="store_true")
        if name in ("prereg", "all"):
            p.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"], choices=list(ARMS))
            p.add_argument("--seeds", type=int, default=6)
            p.add_argument("--cycles", type=int, default=40)
            p.add_argument("--profile", default="smoke", choices=["tiny", "smoke", "full"])
            p.add_argument("--yield-starts", type=int, default=2)
            p.add_argument("--yield-seeds", type=int, default=2)
            p.add_argument("--yield-k", type=int, default=3)
        if name in ("run", "all"):
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd in ("prereg", "all"):
        cmd_prereg(a)
    if a.cmd in ("run", "all"):
        cmd_run(a)
    if a.cmd in ("analyze", "all"):
        print(analyze(Path(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
