"""kernel/world.py: assembles the synthetic Study-2 world (seed agent + sealed verifier + improvers).

Used by run_ablations.py, yield_test.py and the tests. A real study swaps `make_meta` for an LLM
MetaAgent and the families for richer ones; the kernel does not change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from kernel.improver import MutationMetaAgent, RandomPatchMeta
from kernel.orchestrator import (Archive, Config, GitRepo, KernelGuard, MomCenter, Orchestrator, Sandbox,
                                 ShapleyAttributor)
from kernel.seed_agent import write_seed_agent
from kernel.verifier import SealedVerifier

KERNEL_DIR = Path(__file__).resolve().parent

PROFILES = {
    # name: (n_selection, n_confirm, n_credit, n_anchor, n_heldout)
    "tiny": (16, 16, 8, 24, 24),
    "smoke": (32, 32, 16, 48, 48),
    "full": (64, 64, 32, 96, 96),
}


class World:
    def __init__(self, secret: bytes, profile: str = "smoke", honeypot_rate: float = 0.12, **cfg_over):
        self.secret, self.profile, self.honeypot_rate = secret, profile, honeypot_rate
        ns, nc, ncr, na, nh = PROFILES[profile]
        self.cfg_kw = dict(n_selection=ns, n_confirm=nc, n_credit=ncr, n_anchor=na, n_heldout=nh,
                           min_pairs=min(12, ns // 2), screen_chunks=3, families=("algorithmic", "datastructure"),
                           eval_wall_s=300, cycle_wall_s=900, p_exploit=0.5, cooloff_s=0,
                           expected_accept_rate=0.5, accept_anomaly_p=1e-6, accept_halt_p=1e-12,
                           max_consecutive_failures=50, escape_attempt_limit=10_000, **cfg_over)

    def config(self, repo_dir: Path, runs_dir: Path, **over) -> Config:
        return Config(repo_dir=repo_dir, runs_dir=runs_dir, **{**self.cfg_kw, **over})

    def verifier(self, cfg: Config) -> SealedVerifier:
        return SealedVerifier(self.secret, Sandbox(cfg), honeypot_rate=self.honeypot_rate)

    def init_repo(self, repo_dir: Path, policy: Optional[dict] = None) -> None:
        repo_dir = Path(repo_dir)
        repo_dir.mkdir(parents=True, exist_ok=True)
        write_seed_agent(repo_dir, policy)
        GitRepo(repo_dir, repo_dir.parent / "_init_wt").init("seed agent")

    @staticmethod
    def make_meta(kind: str, seed: int, policy_override: Optional[dict] = None, allow_meta: bool = False):
        if kind == "m0":                                         # frozen reference improver
            return MutationMetaAgent(seed, "frozen", False, policy_override)
        if kind == "repo":                                       # improver reads its policy from the evolving repo
            return MutationMetaAgent(seed, "repo", allow_meta, policy_override)
        if kind == "random":
            return RandomPatchMeta(seed)
        raise ValueError(kind)

    def build(self, repo_dir: Path, runs_dir: Path, meta, yield_gate_factory=None, orch_cls=Orchestrator,
              **cfg_over) -> Orchestrator:
        cfg = self.config(repo_dir, runs_dir, **cfg_over)
        repo = GitRepo(cfg.repo_dir, cfg.runs_dir / "worktrees")
        ver = self.verifier(cfg)
        orch = orch_cls(cfg, repo, Sandbox(cfg), ver, meta,
                            Archive(cfg.runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                            ShapleyAttributor(cfg, repo, ver), MomCenter(cfg.runs_dir / "mom.db"),
                            KernelGuard(KERNEL_DIR))
        if yield_gate_factory:
            orch.yield_gate = yield_gate_factory(orch)
        return orch
