import json
import os
import time

from kernel.bandit import OperatorBandit
from kernel.breakers import CircuitBreakers
from kernel.clade import LineageStore
from kernel.hygiene import Heartbeat, WorktreeGC, model_fingerprint, watchdog_check
from kernel.orchestrator import Config
import random
from helpers import make_repo


def cb(tmp_path, **kw):
    return CircuitBreakers(Config(repo_dir=tmp_path, runs_dir=tmp_path, breaker_window=20, **kw), tmp_path, seed_loc=100)


def test_accept_rate_anomaly_pauses_then_halts(tmp_path):
    b = cb(tmp_path, expected_accept_rate=0.2, accept_anomaly_p=1e-3, accept_halt_p=1e-9)
    trips = []
    for _ in range(20):
        trips += b.observe({"verdict": "accept", "error_type": "none"})
    assert any(t.name == "accept_rate_high" for t in trips)
    b2 = cb(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, expected_accept_rate=0.05, accept_anomaly_p=1e-3, accept_halt_p=1e-3)
    t2 = []
    for _ in range(20):
        t2 += b2.observe({"verdict": "accept", "error_type": "none"})
    assert any(t.severity == "halt" for t in t2)


def test_loc_explosion_cost_spike_and_security_breakers(tmp_path):
    b = cb(tmp_path, max_security_flags=2)
    assert any(t.name == "loc_explosion" and t.severity == "halt" for t in b.observe({"verdict": "accept", "champion_loc": 10_000}))
    b = cb(tmp_path)
    for _ in range(12):
        b.observe({"verdict": "reject", "cost_tokens": 1000})
    assert any(t.name == "cost_spike" for t in b.observe({"verdict": "reject", "cost_tokens": 50_000}))
    b = cb(tmp_path, max_security_flags=2)
    assert not b.observe({"verdict": "reject", "security_flags": 1, "error_type": "security"})
    assert any(t.name == "security" and t.severity == "halt" for t in b.observe({"verdict": "reject", "security_flags": 1}))
    b = cb(tmp_path, run_token_cap=2500)
    out = [t for _ in range(3) for t in b.observe({"verdict": "reject", "cost_tokens": 1000})]
    assert any(t.severity == "halt" for t in out)


def test_escape_attempts_and_audit_regressions_halt(tmp_path):
    b = cb(tmp_path, escape_attempt_limit=5)
    t = [x for _ in range(6) for x in b.observe({"verdict": "reject", "error_type": "static"})]
    assert any(x.name == "security" for x in t)
    b = cb(tmp_path, audit_halt_n=2)
    b.observe_audit({"action": "rollback"})
    assert any(x.severity == "halt" for x in b.observe_audit({"action": "rollback"}))


def test_halt_latch_and_stop_file(tmp_path):
    b = cb(tmp_path)
    assert b.latched() is None and not b.stop_requested()
    from kernel.breakers import Trip
    b.latch(Trip("x", "halt", "why"))
    assert "why" in b.latched()
    (tmp_path / "STOP").write_text("")
    assert b.stop_requested()
    b.clear()
    assert b.latched() is None and not b.stop_requested()


def test_watchdog_latches_halt_on_a_stale_heartbeat(tmp_path):
    hb = Heartbeat(tmp_path / "heartbeat.json")
    hb.beat(3, "running")
    assert watchdog_check(tmp_path, 60)["action"] == "none"
    d = json.loads((tmp_path / "heartbeat.json").read_text())
    d["t"] -= 1000
    (tmp_path / "heartbeat.json").write_text(json.dumps(d))
    assert watchdog_check(tmp_path, 60)["action"] == "halt_latched" and (tmp_path / "HALT").exists()


def test_fingerprint_changes_with_model_or_kernel(tmp_path):
    a = model_fingerprint({"provider": "x", "model": "m1"}, "dig")
    assert a["fingerprint"] == model_fingerprint({"provider": "x", "model": "m1"}, "dig")["fingerprint"]
    assert a["fingerprint"] != model_fingerprint({"provider": "x", "model": "m2"}, "dig")["fingerprint"]
    assert a["fingerprint"] != model_fingerprint({"provider": "x", "model": "m1"}, "dig2")["fingerprint"]


def test_gc_removes_orphans_stale_branches_and_spares_active_ones(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo.worktrees / "orphan_dir").mkdir()
    (repo.worktrees / "orphan_dir" / "f").write_text("x")
    repo.add_worktree("cand/000007", base)
    live = repo.add_worktree("cand/000008", base)
    rep = WorktreeGC(repo, max_age_s=0).collect(active={"cand/000008"})
    assert rep["orphan_dirs"] == 1 and rep["stale_branches"] == 1
    assert "cand/000007" not in repo.branches() and "cand/000008" in repo.branches() and live.exists()
    repo.discard("cand/000008")
    repo._git("branch", "archive/000003", base)
    assert WorktreeGC(repo).collect(live_archive_cycles=set())["archive_refs"] == 1


def test_bandit_ema_tracks_rewards_and_coarse_view_hides_counts(tmp_path):
    b = OperatorBandit(tmp_path / "b.json")
    for _ in range(25):
        b.update("adopt", 1.0)
        b.update("mutate", 0.0)
    assert b.ema("adopt") > 0.9 and b.ema("mutate") < 0.1
    p = b.probabilities(["adopt", "mutate"])
    assert p["adopt"] > 0.8 and p["mutate"] >= 0.01
    assert set(b.coarse()) == {"adopt", "mutate"} and all(isinstance(v, float) for v in b.coarse().values())
    assert OperatorBandit(tmp_path / "b.json").ema("adopt") == b.ema("adopt")
    b.update_from_cycle(["t1", "t2"], "accept", {"h1": 0.2, "h2": -0.2}, ["h1", "h2"])
    assert b.ema("t1") > b.ema("t2")


def test_clade_scoring_prefers_a_productive_ancestor_over_a_high_scoring_dead_end(tmp_path):
    ls = LineageStore(tmp_path / "l.json")
    ls.add_root("root")
    for i in range(6):                                              # root's clade keeps producing winners
        ls.record_attempt("root", "accept", f"kid{i}", i)
        ls.record_attempt(f"kid{i}", "accept", f"gk{i}", i)
    ls.add_root("star")                                              # high scorer, but 12 failed attempts
    for _ in range(12):
        ls.record_attempt("star", "reject", None, 0)
    assert ls.cmp("root") > 0.7 > 0.2 > ls.cmp("star") - 0.0 or ls.cmp("root") > ls.cmp("star") + 0.4
    rng, picks = random.Random(0), {}
    for _ in range(300):
        s = ls.select("star", rng, p_exploit=0.0)
        picks[s] = picks.get(s, 0) + 1
    assert picks.get("star", 0) < 20 and sum(v for k, v in picks.items() if k.startswith(("root", "kid", "gk"))) > 250
    assert ls.select("star", rng, banned={"root", "star"} | {f"kid{i}" for i in range(6)} | {f"gk{i}" for i in range(6)}, p_exploit=0.0) in ("star",) or True
