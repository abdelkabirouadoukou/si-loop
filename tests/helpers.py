"""Shared fakes for the kernel tests (a controllable verifier over real git worktrees)."""
import random
from pathlib import Path

from kernel.anchor import AnchorStore, MarginBudget
from kernel.orchestrator import Config, EvalResult, GitRepo, Verifier


class FakeVerifier(Verifier):
    """Pass probability comes from agent/profile.txt and the SPLIT of each problem id.
    table: profile -> {split: p}; '*' is the default. Shared latent difficulty per problem id,
    independent per-profile noise (the paired design of the real verifier)."""

    def __init__(self, table, flag_profiles=(), noise=0.08):
        self.table, self.flag_profiles, self.noise, self.calls = table, set(flag_profiles), noise, 0

    def draw_batch(self, cycle, n, split="selection"):
        return [f"{split}-{cycle}-{i}" for i in range(n)]

    def evaluate(self, wt, batch, wall_s):
        self.calls += 1
        prof = (Path(wt) / "agent/profile.txt").read_text().strip()
        loc_f = Path(wt) / "agent/loc.txt"
        loc = int(loc_f.read_text()) if loc_f.exists() else 100
        t = self.table[prof]
        passes = {}
        for pid in batch:
            split = pid.split("-")[0]
            p = t.get(split, t["*"])
            passes[pid] = random.Random(pid).random() + random.Random(f"{pid}|{prof}").gauss(0, self.noise) < p
        fl = ["honeypot_passed@x"] if prof in self.flag_profiles else []
        return EvalResult(passes, {"agent_loc": loc, "tokens": 1000}, {"algorithmic": sum(passes.values()) / max(1, len(passes))}, fl)


def make_repo(tmp: Path, profile="base", loc=100):
    root = tmp / "repo"
    (root / "agent").mkdir(parents=True)
    (root / "agent/profile.txt").write_text(profile)
    (root / "agent/loc.txt").write_text(str(loc))
    repo = GitRepo(root, tmp / "wt")
    sha = repo.init("seed")
    return repo, sha


def candidate(repo: GitRepo, parent_sha: str, profile: str, loc=100, name="cand/000001"):
    wt = repo.add_worktree(name, parent_sha)
    (wt / "agent/profile.txt").write_text(profile)
    (wt / "agent/loc.txt").write_text(str(loc))
    sha = repo.commit_all(wt, "candidate")
    return wt, sha


def cfg_for(tmp: Path, **kw) -> Config:
    base = dict(n_selection=100, n_confirm=100, min_pairs=20, screen_chunks=4, families=("algorithmic",),
                anchor_every=0, heldout_every=0, gc_every=0, cooloff_s=0)
    return Config(repo_dir=tmp / "repo", runs_dir=tmp / "runs", **{**base, **kw})


def frozen_anchor(tmp: Path, sha: str, loc=100) -> AnchorStore:
    a = AnchorStore(tmp / "anchor.json")
    a.freeze(sha, loc)
    return a


def budget(tmp: Path, cfg: Config) -> MarginBudget:
    return MarginBudget(tmp / "margin.json", cfg.margin_budget, cfg.margin_min_charge)
