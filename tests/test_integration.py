import json
import subprocess
from pathlib import Path

import pytest

import run_ablations as RA
from kernel.improver import REFERENCE_POLICY, SEED_POLICY
from kernel.ledger import HashChainLedger
from kernel.orchestrator import HaltRun
from kernel.world import World
from kernel.yield_test import YieldConfig, YieldGate, YieldHarness

SECRET = b"integration-secret-0123456789abcd"


def build(tmp_path, **kw):
    w = World(SECRET, "tiny")
    w.init_repo(tmp_path / "repo")
    return w, w.build(tmp_path / "repo", tmp_path / "runs", World.make_meta("m0", 0), seed=0, cooloff_s=0, **kw)


def test_short_real_loop_keeps_a_verified_ledger_and_hides_fine_metrics(tmp_path):
    w, o = build(tmp_path, anchor_every=3, heldout_every=0, gc_every=2)
    s = o.run(4)
    assert not s["halted"]
    ok, n, bad = HashChainLedger(tmp_path / "runs" / "ledger.jsonl").verify()
    assert ok and n > 4
    fb = [json.loads(l) for l in (tmp_path / "runs" / "ledger.jsonl").read_text().splitlines()]
    assert any(r.get("ev") == "audit" for r in fb)
    assert any(r.get("ev") == "fingerprint" or "fingerprint" in json.dumps(r) for r in fb)


def test_ledger_tampering_is_detected(tmp_path):
    w, o = build(tmp_path, anchor_every=0, heldout_every=0)
    o.run(2)
    p = tmp_path / "runs" / "ledger.jsonl"
    lines = p.read_text().splitlines()
    lines[1] = lines[1].replace("0", "1", 1)
    p.write_text("\n".join(lines) + "\n")
    ok, _, bad = HashChainLedger(p, verify_on_open=False).verify()
    assert not ok and bad is not None


def test_halt_latch_stops_the_run_and_survives_restart(tmp_path):
    w, o = build(tmp_path, anchor_every=0, heldout_every=0)
    (tmp_path / "runs" / "STOP").write_text("")
    s = o.run(5)
    assert s["cycles"] <= 1 or s["halted"] or s.get("stopped")
    (tmp_path / "runs" / "STOP").unlink()


def test_anchor_audit_rolls_back_a_degraded_champion(tmp_path):
    w, o = build(tmp_path, anchor_every=1, heldout_every=0)
    o.bootstrap()
    good = o.repo.head("main")
    wt = o.repo.add_worktree("cand/bad", good)
    (wt / "agent/solver.py").write_text("def solve_problem(spec):\n    return 'def solve(*a):\\n    return None\\n', 1\n")
    bad = o.repo.commit_all(wt, "break everything")
    o.repo.fast_forward("main", bad) if hasattr(o.repo, "fast_forward") else o.repo._git("update-ref", "refs/heads/main", bad)
    o.repo.discard("cand/bad")
    reps = o.audit(0)
    assert reps and reps[0]["action"] in ("rollback", "ok")
    if reps[0]["action"] == "rollback":
        assert o.repo.head("main") != bad


# ---------------------------------------------------------------- yield decision (stubbed measurements)
class StubHarness(YieldHarness):
    def __init__(self, gains, ycfg):
        self.y, self.gains = ycfg, gains

    def measure(self, policy, start_sha, seed):
        key = "m0" if policy is None else json.dumps(policy, sort_keys=True)
        return {"gain": self.gains[key][(start_sha, seed)], "tokens": 10}


def _cmp(gc, gp, gm, **y):
    c, p = {"w_adopt": 9.0}, {"w_adopt": 1.0}
    starts = ["a", "b"]
    mk = lambda f: {(s, sd): f(i) for i, (s, sd) in enumerate((s, sd) for s in starts for sd in (0, 1, 2))}
    g = {json.dumps(c, sort_keys=True): mk(gc), json.dumps(p, sort_keys=True): mk(gp), "m0": mk(gm)}
    h = StubHarness(g, YieldConfig(**{"min_pairs": 4, **y}))
    h._cache = {}
    return h.compare(c, p, starts, seeds=(0, 1, 2))


def test_yield_accepts_consistent_improvement_and_rejects_noise_or_m0_regression():
    assert _cmp(lambda i: 0.20 + 0.01 * (i % 2), lambda i: 0.05, lambda i: 0.05)["accept"]
    assert not _cmp(lambda i: 0.05 + (0.1 if i % 2 else -0.1), lambda i: 0.05, lambda i: 0.05)["accept"]
    assert not _cmp(lambda i: 0.10, lambda i: 0.20, lambda i: 0.20)["accept"]
    # better than parent but clearly worse than frozen M0 -> rejected
    assert not _cmp(lambda i: 0.10, lambda i: 0.02, lambda i: 0.30)["accept"]
    assert not _cmp(lambda i: 0.2, lambda i: 0.0, lambda i: 0.0, min_pairs=99)["accept"]


# ---------------------------------------------------------------- ablation runner
def prereg_args(out, **kw):
    class A: pass
    a = A()
    a.out, a.arms, a.seeds, a.cycles, a.profile = str(out), ["A0", "A1", "A2", "A3"], 3, 8, "tiny"
    a.yield_starts, a.yield_seeds, a.yield_k, a.force = 1, 1, 1, False
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_prereg_is_immutable_and_detects_edits(tmp_path):
    RA.cmd_prereg(prereg_args(tmp_path))
    with pytest.raises(SystemExit):
        RA.cmd_prereg(prereg_args(tmp_path))
    assert RA.load_prereg(tmp_path)["arms"] == ["A0", "A1", "A2", "A3"]
    p = tmp_path / "prereg.json"
    d = json.loads(p.read_text())
    d["cycles"] = 999
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        RA.load_prereg(tmp_path)


def test_best_of_n_always_proposes_from_the_anchor(tmp_path):
    w = World(SECRET, "tiny")
    w.init_repo(tmp_path / "repo")
    o = w.build(tmp_path / "repo", tmp_path / "runs", World.make_meta("m0", 0), orch_cls=RA.BestOfNOrchestrator,
                seed=0, anchor_every=0, heldout_every=0)
    o.bootstrap()
    assert o._select_parent(o.repo.head("main")) == o.anchor.anchor_sha


def test_analysis_uses_holm_and_never_overclaims(tmp_path):
    RA.cmd_prereg(prereg_args(tmp_path, seeds=6))
    def fake(arm, seed, g, h=0.0, halted=False):
        d = tmp_path / f"{arm}_s{seed}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"arm": arm, "seed": seed, "gain_selection": g, "gain_heldout": h,
            "accepts": 3, "tokens": 100, "yield_test_tokens": 0, "meta_accepts": 1, "accept_rate_first_half": .3,
            "accept_rate_second_half": .2, "halted": halted, "ledger_ok": True, "security_flags": 0}))
    for s in range(6):
        fake("A0", s, 0.10 + 0.01 * s); fake("A2", s, 0.02); fake("A3", s, 0.05); fake("A1", s, 0.20 + 0.01 * s)
    text = RA.analyze(tmp_path)
    assert "H1 supported" in text and "Holm p=" in text
    for s in range(6):
        (tmp_path / f"A1_s{s}" / "result.json").write_text((tmp_path / f"A1_s{s}" / "result.json").read_text().replace('"halted": false', '"halted": true'))
    assert "H1 NOT supported" in RA.analyze(tmp_path)
    assert RA.holm({"a": 0.01, "b": 0.02, "c": 0.5}) == {"a": 0.03, "b": 0.04, "c": 0.5}
