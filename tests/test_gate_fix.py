"""Regression tests for the utility-path bug found after the Study 3 pilot.

Bug: the 'cheaper' test also accepted a token metric. The reference verifier reports tokens = 0 for every
candidate, so `0 <= 0.90 * 0` was always true and any non-harmful patch passed the utility path.
Fix: 'cheaper' means at least 5% fewer lines of code only.
"""
from kernel.orchestrator import Config, EvalResult, Gate


def _cfg(tmp_path):
    return Config(repo_dir=tmp_path, runs_dir=tmp_path)


def _ev(passes, loc, tokens=0):
    return EvalResult(passes=passes, metrics={"agent_loc": loc, "tokens": tokens}, family_profile={}, flags=[])


def _pair(b, c, n=48):
    """Parent fails the first b problems and passes c; candidate fixes b and breaks c."""
    ids = [f"p{i:02d}" for i in range(n)]
    parent = {i: False for i in ids}
    cand = {i: False for i in ids}
    for i in ids[:c]:
        parent[i] = True                      # broken by candidate
    for i in ids[c:c + b]:
        cand[i] = True                        # fixed by candidate
    for i in ids[c + b:]:
        parent[i] = cand[i] = True            # passed by both
    return parent, cand


def test_no_size_drop_and_no_token_data_is_rejected_not_accepted(tmp_path):
    """The exact cycle-0 situation: b=3, c=0, p_better 0.9375 (< 0.95), code grew, tokens are 0."""
    parent, cand = _pair(b=3, c=0)
    gate = Gate(_cfg(tmp_path))
    d = gate.decide(_ev(parent, loc=88), _ev(cand, loc=89), novelty=0.0)
    assert d.verdict == "reject"
    assert d.path == ""


def test_real_size_reduction_still_accepts_on_utility_path(tmp_path):
    parent, cand = _pair(b=3, c=0)
    gate = Gate(_cfg(tmp_path))
    d = gate.decide(_ev(parent, loc=88), _ev(cand, loc=80), novelty=0.0)   # 80 <= 0.95 * 88
    assert d.verdict == "accept" and d.path == "utility"


def test_small_size_reduction_below_5_percent_is_rejected(tmp_path):
    parent, cand = _pair(b=3, c=0)
    gate = Gate(_cfg(tmp_path))
    d = gate.decide(_ev(parent, loc=88), _ev(cand, loc=85), novelty=0.0)   # 85 > 0.95 * 88 = 83.6
    assert d.verdict == "reject"


def test_correctness_path_is_unchanged(tmp_path):
    parent, cand = _pair(b=7, c=0)                                        # p_better 0.996
    gate = Gate(_cfg(tmp_path))
    d = gate.decide(_ev(parent, loc=88), _ev(cand, loc=89), novelty=0.0)
    assert d.verdict == "accept" and d.path == "correctness"
