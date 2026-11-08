from helpers import FakeVerifier, budget, candidate, cfg_for, frozen_anchor, make_repo
from kernel.pipeline import TwoStageJudge


def run(tmp_path, table, cand_profile="cand", cand_loc=100, flag=(), bud_max=None, noise=0.08, **cfgkw):
    repo, base = make_repo(tmp_path, "base")
    cfg = cfg_for(tmp_path, **cfgkw)
    ver = FakeVerifier(table, flag_profiles=flag, noise=noise)
    bud = budget(tmp_path, cfg)
    if bud_max is not None:
        bud.remaining = bud_max
    judge = TwoStageJudge(cfg, repo, ver, bud, frozen_anchor(tmp_path, base))
    wt, sha = candidate(repo, base, cand_profile, cand_loc)
    j = judge.judge(wt, sha, base, 1, lambda: 60, lambda ev: [0.0], lambda d, s=None: 0.0)
    return j, ver, bud


def test_real_improvement_is_accepted_after_confirmation(tmp_path):
    j, _, _ = run(tmp_path, {"base": {"*": 0.5}, "cand": {"*": 0.68}})
    assert j.verdict == "accept" and j.stage == "confirm" and j.detail["confirm"]["b"] > j.detail["confirm"]["c"]
    assert j.coarse() == {"verdict": "accept", "error_type": "none"}


def test_null_patch_is_rejected(tmp_path):
    j, _, _ = run(tmp_path, {"base": {"*": 0.5}, "cand": {"*": 0.5}})
    assert j.verdict == "reject" and j.error_type in ("no_gain", "regression")


def test_winners_curse_gain_on_batch_A_only_dies_at_confirmation(tmp_path):
    table = {"base": {"*": 0.5}, "cand": {"selection": 0.72, "confirm": 0.5, "*": 0.5}}
    j, _, _ = run(tmp_path, table)
    assert j.verdict == "reject" and j.stage == "confirm" and j.error_type in ("unconfirmed", "regression")
    assert j.detail["screen"]["b"] > j.detail["screen"]["c"]            # it did look good on A


def test_sequential_harm_stop_saves_evaluations(tmp_path):
    j, ver, _ = run(tmp_path, {"base": {"*": 0.6}, "cand": {"*": 0.2}})
    assert j.verdict == "reject" and j.stage == "screen_harm" and j.error_type == "regression"
    assert ver.calls < 8                                                # 4 chunks x (cand+parent) would be 8


def test_margin_budget_closes_the_utility_path(tmp_path):
    table = {"base": {"*": 0.5}, "cand": {"*": 0.5}}
    ok, _, bud = run(tmp_path, table, cand_loc=70, noise=0.0)
    assert ok.verdict == "accept" and ok.charge >= 0.005
    (tmp_path / "b").mkdir()
    j, _, _ = run(tmp_path / "b", table, cand_loc=70, bud_max=0.0, noise=0.0)
    assert j.verdict == "reject" and j.error_type == "margin"


def test_security_flag_is_an_automatic_reject(tmp_path):
    j, _, _ = run(tmp_path, {"base": {"*": 0.5}, "cand": {"*": 0.9}}, flag=("cand",))
    assert j.verdict == "reject" and j.stage == "honeypot" and j.error_type == "security" and j.flags


def test_loc_cap_rejects_bloat(tmp_path):
    j, _, _ = run(tmp_path, {"base": {"*": 0.5}, "cand": {"*": 0.9}}, cand_loc=5000)
    assert j.verdict == "reject" and j.error_type == "loc_cap"


def test_fdr_enforce_blocks_weak_discoveries(tmp_path):
    table = {"base": {"*": 0.5}, "cand": {"*": 0.6}}
    j, _, _ = run(tmp_path, table, fdr_mode="enforce", fdr_alpha=1e-6)
    assert j.verdict == "reject" and j.error_type in ("fdr", "unconfirmed", "no_gain")
    j2, _, _ = run(tmp_path / "x", {"base": {"*": 0.5}, "cand": {"*": 0.9}}, fdr_mode="monitor") if (tmp_path / "x").mkdir() is None else (0, 0, 0)
    assert j2.verdict == "accept" and "fdr" in j2.detail
