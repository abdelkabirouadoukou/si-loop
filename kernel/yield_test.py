"""kernel/yield_test.py: structured YIELD TESTING of the improver (meta-level self-improvement).

A solver patch is judged by whether the solver got better. An IMPROVER patch (one that edits
agent/meta/**) cannot be judged that way: it changes how future patches are made, not the solver.
Its value is its YIELD: how much solver improvement it produces per unit of work.

    yield(policy, start, seed) = pass@1(champion after K cycles) - pass@1(start)
                                 on one fresh `yield` batch shared by start and final (paired)

Protocol (all arms run from IDENTICAL starting states and seeds, in isolated git clones):
    candidate improver   policy parsed from the candidate worktree (parent + patch)
    parent improver      policy of the parent commit          (is the patch an improvement?)
    M0                   frozen reference improver            (is it still at least as good as the baseline?)
ACCEPT iff  bootstrap lower bound of mean(yield_cand - yield_parent) > min_gain
       AND  mean(yield_cand - yield_M0) >= -tol_vs_m0   AND  #pairs >= min_pairs.
Inner loops use the SAME two-stage judge, but never edit agent/meta (allow_meta=False), so the
measurement is about the improver's effect on the solver, not about recursion inside the test.

CLI:  python -m kernel.yield_test --policy '{"w_adopt": 8.0}' --starts 2 --seeds 3 --k 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kernel.improver import POLICY_PATH, REFERENCE_POLICY, policy_from_source, policy_hash
from kernel.orchestrator import GitRepo
from kernel.pipeline import Judgement
from kernel.stats import bootstrap_lower, sign_flip_p
from kernel.world import World


@dataclass
class YieldConfig:
    n_starts: int = 2
    seeds: tuple = (0, 1)
    k_cycles: int = 4
    n_eval: int = 32
    min_gain: float = 0.0
    tol_vs_m0: float = 0.02
    alpha: float = 0.05
    min_pairs: int = 4


class YieldHarness:
    def __init__(self, world: World, source: GitRepo, ycfg: YieldConfig, tmp_root: Optional[Path] = None):
        self.world, self.source, self.y = world, source, ycfg
        self.tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="sil_yield_"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}

    # ------------------------------------------------------------------ one measurement
    def measure(self, policy: Optional[dict], start_sha: str, seed: int) -> dict:
        key = (policy_hash(policy or REFERENCE_POLICY), start_sha, seed)
        if key in self._cache:
            return self._cache[key]
        work = Path(tempfile.mkdtemp(prefix="run_", dir=self.tmp_root))
        try:
            clone = work / "repo"
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(self.source.root), str(clone)], check=True,
                           capture_output=True)
            subprocess.run(["git", "checkout", "-q", "-B", "main", start_sha], cwd=clone, check=True, capture_output=True)
            meta = World.make_meta("m0", seed, policy_override=policy)
            orch = self.world.build(clone, work / "runs", meta, anchor_every=0, heldout_every=0, gc_every=0,
                                    fdr_mode="off", bandit_enabled=False, credit_enabled=False,
                                    parent_policy="archive", p_exploit=1.0, seed=seed)
            summary = orch.run(self.y.k_cycles, workers=1, start_cycle=1_000_000 + 1000 * seed)
            batch = orch.verifier.draw_batch(5_000_000 + seed, self.y.n_eval, "yield")
            before = orch.judge.eval_sha(start_sha, batch)
            after = orch.judge.eval_sha(orch.repo.head("main"), batch)
            tokens = sum(r.get("cost_tokens", 0) for r in orch.ledger.records() if r.get("ev") == "end")
            res = {"gain": after.pass_at_1 - before.pass_at_1, "start_pass": before.pass_at_1,
                   "final_pass": after.pass_at_1, "accepts": summary["accepts"], "tokens": tokens,
                   "halted": summary["halted"]}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self._cache[key] = res
        return res

    def sample(self, policy: Optional[dict], starts: list, seeds=None) -> list:
        return [self.measure(policy, s, sd) for s in starts for sd in (seeds or self.y.seeds)]

    # ------------------------------------------------------------------ paired comparison
    def compare(self, cand: dict, parent: dict, starts: list, seeds=None) -> dict:
        c = self.sample(cand, starts, seeds)
        p = self.sample(parent, starts, seeds)
        m = self.sample(None, starts, seeds)
        d_par = [a["gain"] - b["gain"] for a, b in zip(c, p)]
        d_m0 = [a["gain"] - b["gain"] for a, b in zip(c, m)]
        n = len(d_par)
        lower = bootstrap_lower(d_par, self.y.alpha)
        rep = {"pairs": n, "mean_yield_cand": sum(x["gain"] for x in c) / n, "mean_yield_parent": sum(x["gain"] for x in p) / n,
               "mean_yield_m0": sum(x["gain"] for x in m) / n, "mean_diff_vs_parent": sum(d_par) / n,
               "lower_vs_parent": lower, "p_vs_parent": sign_flip_p(d_par), "mean_diff_vs_m0": sum(d_m0) / n,
               "tokens_cand": sum(x["tokens"] for x in c), "tokens_m0": sum(x["tokens"] for x in m)}
        rep["accept"] = bool(n >= self.y.min_pairs and lower > self.y.min_gain and rep["mean_diff_vs_m0"] >= -self.y.tol_vs_m0)
        return rep


class YieldGate:
    """Plugs into Orchestrator.yield_gate: judges patches whose hunks all live under agent/meta/."""

    def __init__(self, harness: YieldHarness, orch):
        self.h, self.orch = harness, orch

    def starts(self) -> list:
        o = self.orch
        cand = [o.anchor.anchor_sha, o.repo.head("main")] + [e["sha"] for e in sorted(o.archive.entries, key=lambda e: e["cycle"])]
        out = []
        for s in cand:
            if s not in out and o.repo.exists(s) and s not in o.banned:
                out.append(s)
        return out[: self.h.y.n_starts]

    def judge(self, wt, sha: str, parent_sha: str, cycle: int, proposal) -> Judgement:
        o = self.orch
        cand = policy_from_source((Path(wt) / POLICY_PATH).read_text())
        try:
            par = policy_from_source(o.repo._git("show", f"{parent_sha}:{POLICY_PATH}"))
        except RuntimeError:
            par = dict(REFERENCE_POLICY)
        rep = self.h.compare(cand, par, self.starts())
        why = [f"yield pairs={rep['pairs']} cand={rep['mean_yield_cand']:+.3f} parent={rep['mean_yield_parent']:+.3f} "
               f"m0={rep['mean_yield_m0']:+.3f} lower_vs_parent={rep['lower_vs_parent']:+.3f}"]
        return Judgement("accept" if rep["accept"] else "reject", why, "yield",
                         "none" if rep["accept"] else "yield", [], [], 0.0, rep["mean_yield_cand"],
                         rep["mean_yield_parent"], rep["mean_diff_vs_parent"], 0.0,
                         {"yield": rep, "tokens": rep["tokens_cand"] / max(1, rep["pairs"])})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernel.yield_test")
    ap.add_argument("--policy", default="{}", help="JSON dict of policy values to test (rest = seed policy)")
    ap.add_argument("--baseline", default="", help="JSON dict for the baseline improver (default: M0)")
    ap.add_argument("--profile", default="smoke", choices=["tiny", "smoke", "full"])
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="sil_ytest_"))
    world = World(b"yield-test-secret-000000000000", a.profile)
    world.init_repo(tmp / "repo")
    repo = GitRepo(tmp / "repo", tmp / "wt")
    ycfg = YieldConfig(n_starts=a.starts, seeds=tuple(range(a.seeds)), k_cycles=a.k)
    h = YieldHarness(world, repo, ycfg, tmp / "y")
    cand = {**REFERENCE_POLICY, **json.loads(a.policy)}
    base = {**REFERENCE_POLICY, **json.loads(a.baseline)} if a.baseline else dict(REFERENCE_POLICY)
    rep = h.compare(cand, base, [repo.head("main")])
    print(json.dumps(rep, indent=1))
    print("VERDICT:", "improver patch ACCEPTED" if rep["accept"] else "improver patch rejected")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
