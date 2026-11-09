"""study4/yield_gate.py: builds a kernel/yield_test.py YieldGate for Study 4, wired correctly.

See study4/README.md "Known gap and fix". kernel/yield_test.py's YieldHarness.measure() always builds
a SYNTHETIC kernel/world.py World and evaluates with its synthetic SealedVerifier, no matter what
`source` repo it is given. run_ablations.py's A1 arm works today only because its source repo IS a
World-built repo already. Pointing the same harness at Study 4's live study2-shaped repo makes every
measurement read exactly pass@1 = 0.0, unconditionally -- never a real signal.

The fix is a wiring choice, not a kernel/ change: give YieldHarness a DEDICATED synthetic repo, built
once via kernel.world.World.init_repo, as its `source` -- never the live Study 4 repo. This keeps the
already-disclosed trade-off from decision 1 (yield measures the scripted stand-in agent in its own
synthetic world, not the live LLM's own behaviour) but makes the measurement respond to the policy
numbers instead of silently always reading zero.

    gate = build_study4_yield_gate(orch, tmp_root=cfg.runs_dir / "yield", world_secret=b"...")
    orch.yield_gate = gate
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from kernel.orchestrator import GitRepo
from kernel.world import World
from kernel.yield_test import YieldConfig, YieldGate, YieldHarness


class Study4YieldGate(YieldGate):
    """YieldGate whose starts() are commits of its OWN dedicated synthetic source repo, not of the
    live orchestrator's repo. The base class's starts() (kernel/yield_test.py) returns SHAs from
    `self.orch` (anchor_sha, repo.head('main'), archive entries) -- correct for run_ablations.py's
    A1 arm, where the harness's source repo IS orch.repo, but those SHAs do not exist at all in a
    separate synthetic clone, and `git checkout` on one fails outright. See build_study4_yield_gate."""

    def starts(self) -> list:
        return [self.h.source.head("main")][: self.h.y.n_starts]


def build_study4_yield_gate(orch, tmp_root: Path, world_secret: bytes, profile: str = "smoke",
                            ycfg: Optional[YieldConfig] = None) -> YieldGate:
    """Builds a YieldGate backed by a dedicated synthetic source repo (never orch.repo)."""
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    world = World(world_secret, profile)
    synthetic_dir = tmp_root / "synthetic_source"
    if not (synthetic_dir / ".git").exists():
        world.init_repo(synthetic_dir)
    source = GitRepo(synthetic_dir, tmp_root / "synthetic_wt")
    harness = YieldHarness(world, source, ycfg or YieldConfig(), tmp_root / "work")
    return Study4YieldGate(harness, orch)
