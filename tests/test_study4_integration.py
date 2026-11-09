"""Tests for the kernel-level rules Study 4 (the improver arm) depends on:

  * the mixed-scope rule (kernel/orchestrator.py): a patch may touch agent/meta/** or the solver,
    never both -- exercised here against the real Study 4 seed (write_seed_agent_study4).
  * the enforced false-discovery monitor (decision 3): fdr_mode="enforce" blocks a weak accept,
    exercised at the full Orchestrator.run_cycle level, not just the TwoStageJudge unit.
  * YieldGate dispatch: a meta-scope patch is routed to YieldGate.judge, not the normal two-stage
    judge. The harness is STUBBED here -- see study4/README.md "Known gap": the real
    YieldHarness.measure() does not work on a study2-shaped repo (it always reports pass@1 = 0
    because kernel/verifier.py's synthetic SealedVerifier expects a different file layout). This
    test proves the orchestrator's scope-detection and dispatch are correct; it does NOT claim the
    real measurement works.

No live API calls. No kernel/ file is modified by any of this.
"""
from pathlib import Path

import pytest

from helpers import FakeVerifier, cfg_for
from kernel.orchestrator import (Archive, Config, GitRepo, KernelGuard, MetaAgent, MomCenter, Orchestrator,
                                 Sandbox, ShapleyAttributor)
from kernel.patching import EditOp, PatchProposal
from kernel.pipeline import Judgement
from kernel.world import World
from kernel.yield_test import YieldConfig, YieldHarness
from study2.seed_repo import write_seed_agent_study4
from study2.verifier import PytestVerifier
from study4.yield_gate import build_study4_yield_gate

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "kernel"
SECRET = b"study4-test-secret-0123456789ab"


class FakeMetaAgent(MetaAgent):
    """Returns one pre-built PatchProposal per call (or repeats the last one). No LLM, no network."""

    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = 0

    def propose(self, worktree, parent_sha, broadcast, budget):
        p = self.proposals[min(self.calls, len(self.proposals) - 1)]
        self.calls += 1
        p.parent_sha = parent_sha
        return p


class StubYieldGate:
    """Stands in for kernel/yield_test.py's YieldGate: same interface (judge(...) -> Judgement),
    canned result instead of a real (and, for a study2-shaped repo, currently meaningless -- see
    module docstring) measurement."""

    def __init__(self, verdict="accept"):
        self.verdict = verdict
        self.judged = []

    def judge(self, wt, sha, parent_sha, cycle, proposal):
        self.judged.append({"sha": sha, "parent_sha": parent_sha, "cycle": cycle})
        return Judgement(self.verdict, ["stubbed yield decision"], "yield",
                         "none" if self.verdict == "accept" else "yield")


def build_study4_orch(tmp_path, meta, verifier=None, **cfg_over) -> Orchestrator:
    """A Study-4-shaped Orchestrator: the real study2 seed + agent/meta/policy.py
    (write_seed_agent_study4), the real study2 PytestVerifier unless overridden."""
    repo_dir, runs_dir = tmp_path / "agent_repo", tmp_path / "runs"
    write_seed_agent_study4(repo_dir)
    GitRepo(repo_dir, tmp_path / "_init_wt").init("seed agent")
    repo = GitRepo(repo_dir, runs_dir / "worktrees")
    ver = verifier or PytestVerifier(SECRET, Sandbox(Config(repo_dir=repo_dir, runs_dir=runs_dir)),
                                     cases_per_problem=4, suite="v2")
    cfg = Config(repo_dir=repo_dir, runs_dir=runs_dir, seed=0, families=("string", "algorithmic"),
                n_selection=8, n_confirm=8, n_credit=4, min_pairs=4, screen_chunks=1,
                eval_wall_s=60, cycle_wall_s=120, anchor_every=0, heldout_every=0, gc_every=0,
                # Sandbox._build() scrubs $HOME to the ephemeral worktree for isolation; on a machine
                # where pytest is a user-site package (resolved via $HOME), that hides it from the
                # self-test subprocess. Not a kernel bug and not what these tests are about -- skip
                # the real pytest self-test step here (mixed-scope rejection happens before it anyway;
                # the other tests cover scope dispatch, not the self-test mechanism itself).
                selftest_cmd=("python", "-c", "pass"),
                **cfg_over)
    orch = Orchestrator(cfg, repo, Sandbox(cfg), ver, meta,
                        Archive(runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                        ShapleyAttributor(cfg, repo, ver), MomCenter(runs_dir / "mom.db"), KernelGuard(KERNEL_DIR))
    return orch


# =============================================================== mixed-scope rule
def test_mixed_scope_patch_is_rejected(tmp_path):
    proposal = PatchProposal("p1", "", "Mixed-scope patch: touches both the solver and the policy.", [
        EditOp("agent/lib.py", "replace_symbol", symbol="reverse_words",
              new_source='def reverse_words(s):\n    return " ".join(reversed(s.split()))\n', tag="fix.string"),
        EditOp("agent/meta/policy.py", "replace_text", old_text='    "w_adopt": 2.0,',
              new_text='    "w_adopt": 3.0,', tag="meta.tune"),
    ])
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]))
    out = orch.run_cycle(0)
    assert out["verdict"] == "reject"
    assert out["error_type"] == "mixed_scope"


def test_solver_only_patch_is_not_mixed_scope(tmp_path):
    """Sanity check: a patch touching ONLY agent/lib.py is scope='solver', not rejected as mixed."""
    proposal = PatchProposal("p1", "", "Fix reverse_words whitespace handling.", [
        EditOp("agent/lib.py", "replace_symbol", symbol="reverse_words",
              new_source='def reverse_words(s):\n    return " ".join(reversed(s.split()))\n', tag="fix.string"),
    ])
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]))
    out = orch.run_cycle(0)
    assert out["error_type"] != "mixed_scope"
    assert out["scope"] == "solver"


def test_meta_only_patch_is_not_mixed_scope(tmp_path):
    """Sanity check: a patch touching ONLY agent/meta/policy.py is scope='meta', not rejected as mixed."""
    proposal = PatchProposal("p1", "", "Tune w_adopt.", [
        EditOp("agent/meta/policy.py", "replace_text", old_text='    "w_adopt": 2.0,',
              new_text='    "w_adopt": 3.0,', tag="meta.tune"),
    ])
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]))
    orch.yield_gate = StubYieldGate("accept")
    out = orch.run_cycle(0)
    assert out.get("error_type") != "mixed_scope"
    assert out["scope"] == "meta"


# =============================================================== YieldGate dispatch (stubbed measurement)
def test_meta_scope_patch_is_routed_to_yield_gate_not_the_two_stage_judge(tmp_path):
    proposal = PatchProposal("p1", "", "Tune w_adopt.", [
        EditOp("agent/meta/policy.py", "replace_text", old_text='    "w_adopt": 2.0,',
              new_text='    "w_adopt": 3.0,', tag="meta.tune"),
    ])
    fake_ver = FakeVerifier({"base": {"*": 0.5}})
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]), verifier=fake_ver)
    stub = StubYieldGate("accept")
    orch.yield_gate = stub
    out = orch.run_cycle(0)
    assert out["scope"] == "meta"
    assert out["verdict"] == "accept"
    assert len(stub.judged) == 1, "YieldGate.judge should be called exactly once for a meta-scope patch"
    assert fake_ver.calls == 0, "a meta-scope patch must NOT go through the normal two-stage (solver) judge"


def test_meta_scope_patch_rejected_by_yield_gate_is_not_promoted(tmp_path):
    proposal = PatchProposal("p1", "", "Tune w_adopt (bad idea).", [
        EditOp("agent/meta/policy.py", "replace_text", old_text='    "w_adopt": 2.0,',
              new_text='    "w_adopt": 0.1,', tag="meta.tune"),
    ])
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]), verifier=FakeVerifier({"base": {"*": 0.5}}))
    stub = StubYieldGate("reject")
    orch.yield_gate = stub
    before = orch.repo.head("main")
    out = orch.run_cycle(0)
    assert out["scope"] == "meta"
    assert out["verdict"] == "reject"
    assert len(stub.judged) == 1
    assert orch.repo.head("main") == before, "a rejected meta-scope patch must not move the champion"


# =============================================================== YieldHarness source-repo fix
def test_yield_harness_must_measure_against_a_synthetic_source_not_the_live_study2_repo(tmp_path):
    """Root-cause regression test for the "Known gap" in study4/README.md: YieldHarness.measure()
    builds kernel/world.py's SYNTHETIC World + SealedVerifier internally, no matter what `source`
    repo it clones. If `source` is a study2-shaped repo (write_seed_agent_study4), the synthetic
    verifier cannot read its code at all and silently reports pass@1 = 0.0 for every state, every
    time -- so no policy edit could ever be measured as a real gain. If `source` is instead a
    DEDICATED synthetic repo (built once via kernel.world.World.init_repo, as run_ablations.py's
    A1 arm already does), the measurement responds to the actual code again. The fix for Study 4's
    runner wiring (step 5d) is therefore to give YieldHarness a dedicated synthetic source repo,
    never orch.repo -- a wiring choice, not a kernel/ change."""
    ycfg = YieldConfig(n_starts=1, seeds=(0,), k_cycles=2, n_eval=16, min_pairs=1)

    # (a) study2-shaped source: pass@1 is always exactly 0, confirming the bug is real.
    study2_src = tmp_path / "study2_source"
    write_seed_agent_study4(study2_src)
    GitRepo(study2_src, tmp_path / "study2_wt").init("seed agent")
    study2_repo = GitRepo(study2_src, tmp_path / "study2_wt2")
    broken_world = World(b"regression-test-secret-aaaaaaaaaa", "tiny")
    broken = YieldHarness(broken_world, study2_repo, ycfg, tmp_path / "y_broken")
    broken_result = broken.measure(None, study2_repo.head("main"), 0)
    assert broken_result["start_pass"] == 0.0, "study2-shaped source must still read as unmeasurable (the known bug)"

    # (b) dedicated synthetic source (the fix): pass@1 reflects the repo's actual code.
    synth_world = World(b"regression-test-secret-bbbbbbbbbb", "tiny")
    synth_world.init_repo(tmp_path / "synthetic_source")
    synth_repo = GitRepo(tmp_path / "synthetic_source", tmp_path / "synth_wt")
    fixed = YieldHarness(synth_world, synth_repo, ycfg, tmp_path / "y_fixed")
    fixed_result = fixed.measure(None, synth_repo.head("main"), 0)
    assert fixed_result["start_pass"] > 0.0, "a dedicated synthetic source must be measurable (the fix)"


def test_build_study4_yield_gate_end_to_end_on_a_real_orchestrator(tmp_path):
    """The real (non-stubbed) wiring: build_study4_yield_gate's dedicated synthetic source repo,
    attached to a real Study-4 Orchestrator (write_seed_agent_study4), judging a real meta-scope
    patch end to end through Orchestrator.run_cycle. Small profile/cycle counts for test speed --
    this proves the wiring runs and produces a real (non-degenerate) measurement, not that a
    specific patch gets accepted (that depends on luck with so few cycles, same as any gate test)."""
    proposal = PatchProposal("p1", "", "Tune w_adopt.", [
        EditOp("agent/meta/policy.py", "replace_text", old_text='    "w_adopt": 2.0,',
              new_text='    "w_adopt": 9.0,', tag="meta.tune"),
    ])
    orch = build_study4_orch(tmp_path, FakeMetaAgent([proposal]))
    orch.bootstrap()                                   # YieldGate.starts() needs the anchor frozen
    orch.yield_gate = build_study4_yield_gate(
        orch, tmp_path / "yield", b"study4-yield-gate-test-secret-0000", profile="tiny",
        ycfg=YieldConfig(n_starts=1, seeds=(0,), k_cycles=2, n_eval=16, min_pairs=1))
    out = orch.run_cycle(0)
    assert out["scope"] == "meta"
    assert out["verdict"] in ("accept", "reject")
    assert "yield" in out["detail"]
    rep = out["detail"]["yield"]
    assert rep["pairs"] >= 1
    assert isinstance(rep["mean_yield_cand"], float) and isinstance(rep["mean_yield_parent"], float)


# =============================================================== enforced false-discovery monitor
def _minimal_repo_orch(tmp_path, meta, fdr_mode, fdr_alpha=1e-6, **cfg_over):
    """A minimal (non-study2) repo for the generic, verifier-agnostic FDR mechanism -- same table-driven
    FakeVerifier tests/test_pipeline.py uses for this exact check, but run through the FULL
    Orchestrator.run_cycle (not just TwoStageJudge), to confirm Study 4's config choice (decision 3:
    enforce, not monitor) is actually honoured end to end by the orchestrator it will reuse unmodified."""
    root = tmp_path / "repo"
    (root / "agent").mkdir(parents=True)
    (root / "agent/profile.txt").write_text("base")
    (root / "agent/loc.txt").write_text("100")
    repo = GitRepo(root, tmp_path / "wt")
    repo.init("seed")
    runs_dir = tmp_path / "runs"
    ver = FakeVerifier({"base": {"*": 0.5}, "cand": {"*": 0.6}})
    cfg = Config(repo_dir=root, runs_dir=runs_dir, n_selection=100, n_confirm=100, n_credit=8, min_pairs=20,
                screen_chunks=4, families=("algorithmic",), anchor_every=0, heldout_every=0, gc_every=0,
                cooloff_s=0, fdr_mode=fdr_mode, fdr_alpha=fdr_alpha,
                selftest_cmd=("python", "-c", "pass"),            # this minimal repo has no agent/tests
                **cfg_over)
    orch = Orchestrator(cfg, repo, Sandbox(cfg), ver, meta,
                        Archive(runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                        ShapleyAttributor(cfg, repo, ver), MomCenter(runs_dir / "mom.db"), KernelGuard(KERNEL_DIR))
    return orch


def _profile_patch(profile: str) -> PatchProposal:
    return PatchProposal("p1", "", f"Switch profile to {profile}.",
                         [EditOp("agent/profile.txt", "replace_text", old_text="base", new_text=profile, tag="x")])


def test_enforced_monitor_blocks_a_weak_discovery_that_monitor_mode_would_accept(tmp_path):
    enforced = _minimal_repo_orch(tmp_path / "enforce", FakeMetaAgent([_profile_patch("cand")]), fdr_mode="enforce")
    out_enforce = enforced.run_cycle(0)
    assert out_enforce["verdict"] == "reject"
    assert out_enforce["error_type"] in ("fdr", "unconfirmed", "no_gain")

    monitored = _minimal_repo_orch(tmp_path / "monitor", FakeMetaAgent([_profile_patch("cand")]), fdr_mode="monitor")
    out_monitor = monitored.run_cycle(0)
    assert out_monitor["verdict"] == "accept"
    assert "fdr" in out_monitor.get("detail", {}), "monitor mode must still RECORD the fdr check, just not block on it"
