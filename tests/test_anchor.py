import random

from helpers import FakeVerifier, budget, candidate, cfg_for, frozen_anchor, make_repo
from kernel.anchor import AnchorAuditor, AnchorStore, HeldoutAuditor, MarginBudget
from kernel.orchestrator import EvalResult, Gate


# ------------------------------------------------------------------ margin budget
def test_budget_charges_hidden_loss_and_refills_only_on_proven_new_gain(tmp_path):
    b = MarginBudget(tmp_path / "m.json", 0.03, 0.002)
    assert b.charge_for(0.01, utility_path=False) == 0.0           # proven non-worse correctness gain: free
    assert b.charge_for(-0.012, False) == 0.012                     # hidden plausible loss is charged
    assert b.charge_for(0.01, True) == 0.002                        # utility path always pays the floor
    b.spend(0.02, 1)
    assert abs(b.remaining - 0.01) < 1e-12 and not b.can_afford(0.02) and b.can_afford(0.01)
    assert b.refill_from_audit(-0.05) == 0.0 and b.refill_from_audit(0.0) == 0.0   # no proven gain: no refill
    assert abs(b.refill_from_audit(0.015) - 0.015) < 1e-12 and abs(b.remaining - 0.025) < 1e-12
    assert b.refill_from_audit(0.015) == 0.0                        # the same proven gain cannot refill twice
    assert abs(b.refill_from_audit(0.05) - 0.005) < 1e-12           # capped at max, only the NEW part counts
    b2 = MarginBudget(tmp_path / "m.json", 0.03, 0.002)             # persisted
    assert abs(b2.remaining - b.remaining) < 1e-12 and b2.banked_gain == b.banked_gain


def _utility_walk(use_budget, trials=400, seed=0, true_loss=0.004):
    """Slightly-worse (-0.4 points) but smaller patches proposed forever. Returns total TRUE loss accepted."""
    from kernel.orchestrator import Config
    rng, cfg = random.Random(seed), Config(repo_dir="x", runs_dir="/tmp/_creep")
    gate, n, loss = Gate(cfg), 100, 0.0
    bud = MarginBudget(None, 0.03, 0.005)
    for _ in range(trials):
        u = [rng.random() for _ in range(n)]
        mk = lambda p, loc: EvalResult({str(i): u[i] + rng.gauss(0, 0.08) < p for i in range(n)}, {"agent_loc": loc, "tokens": 100})
        d = gate.decide(mk(0.5, 100), mk(0.5 - true_loss, 80), 0.0)
        if d.verdict == "accept" and d.path == "utility":
            ch = bud.charge_for(d.lower, True)
            if use_budget and not bud.can_afford(ch):
                continue
            if use_budget:
                bud.spend(ch)
            loss += true_loss
    return loss


def test_margin_budget_stops_creeping_degradation():
    without = sum(_utility_walk(False, seed=s) for s in range(4))
    with_b = sum(_utility_walk(True, seed=s) for s in range(4))
    assert without > 0.1                                            # the creep is real: dozens of regressions accepted
    assert with_b <= 4 * 6 * 0.004 + 1e-9                                # budget: at most 6 floor-charged accepts per seed
    assert with_b < 0.7 * without


# ------------------------------------------------------------------ audits
def _setup(tmp_path, table):
    repo, base = make_repo(tmp_path, "base")
    cfg = cfg_for(tmp_path, n_anchor=100, n_heldout=100, anchor_tolerance=0.02)
    ver = FakeVerifier(table)
    store = frozen_anchor(tmp_path, base)
    bud = budget(tmp_path, cfg)

    def ev(sha, batch):
        from kernel.pipeline import LazyTree
        with LazyTree(repo, sha) as t:
            return ver.evaluate(t.path(), batch, 10)
    return repo, base, cfg, ver, store, bud, ev


def test_anchor_audit_refills_on_proven_gain_and_promotes_peak(tmp_path):
    table = {"base": {"*": 0.4}, "good": {"*": 0.7}}
    repo, base, cfg, ver, store, bud, ev = _setup(tmp_path, table)
    bud.spend(0.03)
    wt, good = candidate(repo, base, "good")
    rep = AnchorAuditor(cfg, store, bud, ver.draw_batch, ev).audit(5, good)
    assert rep["action"] == "ok" and rep["vs_anchor"]["lower"] > 0.1 and rep["refill"] > 0
    assert bud.remaining > 0 and bud.remaining <= 0.03
    assert store.peak_sha == good                                   # champion proven better than the old peak


def test_anchor_audit_rolls_back_a_proven_regression_to_the_peak(tmp_path):
    table = {"base": {"*": 0.5}, "good": {"*": 0.7}, "bad": {"*": 0.3}}
    repo, base, cfg, ver, store, bud, ev = _setup(tmp_path, table)
    wt, good = candidate(repo, base, "good", name="cand/000001")
    store.set_peak(good)
    wt2, bad = candidate(repo, good, "bad", name="cand/000002")
    rep = AnchorAuditor(cfg, store, bud, ver.draw_batch, ev).audit(9, bad)
    assert rep["action"] == "rollback" and rep["rollback_to"] == good


def test_anchor_audit_is_quiet_for_a_noninferior_champion(tmp_path):
    table = {"base": {"*": 0.5}, "same": {"*": 0.5}}
    repo, base, cfg, ver, store, bud, ev = _setup(tmp_path, table)
    wt, same = candidate(repo, base, "same")
    rep = AnchorAuditor(cfg, store, bud, ver.draw_batch, ev).audit(3, same)
    assert rep["action"] == "ok" and rep["refill"] == 0.0


def test_heldout_audit_flags_regression_and_gap_only_on_the_heldout_family(tmp_path):
    # selection improves a lot, the held-out split gets WORSE: classic cross-family overfitting
    table = {"base": {"*": 0.5}, "over": {"selection": 0.8, "anchor": 0.8, "heldout": 0.3, "*": 0.5}}
    repo, base, cfg, ver, store, bud, ev = _setup(tmp_path, table)
    wt, over = candidate(repo, base, "over")
    a = AnchorAuditor(cfg, store, bud, ver.draw_batch, ev).audit(7, over)
    assert a["action"] == "ok" and a["selection_delta"] > 0.15
    cfg2 = cfg_for(tmp_path, n_anchor=100, n_heldout=100, thresholdout_scale=0.0)
    h = HeldoutAuditor(cfg2, store, ver.draw_batch, ev).audit(7, over, a["selection_delta"], a["selection_se"])
    assert h["action"] == "rollback" and h["gap"] > 0.3 and h["gap_lower"] > 0.15 and h["heldout"]["delta"] < 0


def test_heldout_audit_pauses_on_gap_without_regression(tmp_path):
    table = {"base": {"*": 0.5}, "narrow": {"anchor": 0.85, "heldout": 0.5, "*": 0.5}}
    repo, base, cfg, ver, store, bud, ev = _setup(tmp_path, table)
    wt, n = candidate(repo, base, "narrow")
    cfg2 = cfg_for(tmp_path, n_anchor=100, n_heldout=100, thresholdout_scale=0.0, heldout_gap_alarm=0.15)
    a = AnchorAuditor(cfg2, store, bud, ver.draw_batch, ev).audit(2, n)
    h = HeldoutAuditor(cfg2, store, ver.draw_batch, ev).audit(2, n, a["selection_delta"], a["selection_se"])
    assert h["action"] == "pause"


def test_anchor_store_freezes_once(tmp_path):
    s = AnchorStore(tmp_path / "a.json")
    s.freeze("abc", 10)
    try:
        s.freeze("def", 10)
        raise AssertionError
    except RuntimeError:
        pass
    assert AnchorStore(tmp_path / "a.json").anchor_sha == "abc" and AnchorStore(tmp_path / "a.json").peak_sha == "abc"
