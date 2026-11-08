"""Suite v2 (Study 3 replicates). v1, used by Study 2, must not change."""
import pytest

from study2.tasks import TASKS, make_cases, make_cases_v2
from study2.verifier import PytestVerifier


def _two_sum_candidate_cycle2(nums, target):
    """The cycle-2 patch from the seed-3 run: returns the first pair found, not the smallest first index."""
    seen = {}
    for i, num in enumerate(nums):
        c = target - num
        if c in seen:
            return [seen[c], i]
        if num not in seen:
            seen[num] = i
    return []


def test_v1_is_unchanged_and_is_the_default(tmp_path):
    from kernel.orchestrator import Sandbox, Config
    cfg = Config(repo_dir=tmp_path, runs_dir=tmp_path)
    v = PytestVerifier(b"x" * 16, Sandbox(cfg))
    assert v.suite == "v1"
    pid = v.draw_batch(0, 1, "selection")[0]
    task, seed = v.problem(pid)
    assert v.cases_for(pid)["args"] == make_cases(task, seed, v.k)


def test_v2_adds_targeted_cases_for_two_sum():
    t = TASKS["two_sum"]
    cases = make_cases_v2(t, 7, 12)
    assert len(cases) == 12 + 12
    # every targeted case has repeated values, which is what makes the tie-break matter
    tail = cases[12:]
    assert all(len(set(xs)) < len(xs) for xs, _ in tail)


def test_v2_catches_the_cycle2_tie_break_bug_more_often_than_v1():
    t = TASKS["two_sum"]
    N = 300
    v1 = sum(any(_two_sum_candidate_cycle2(*a) != t.oracle(*a) for a in make_cases(t, p, 6)) for p in range(N))
    v2 = sum(any(_two_sum_candidate_cycle2(*a) != t.oracle(*a) for a in make_cases_v2(t, p, 12)) for p in range(N))
    assert v2 > v1
    assert v2 / N > 0.5


def test_other_tasks_are_unchanged_in_v2():
    for name in ("reverse_words", "roman_to_int", "flatten"):
        t = TASKS[name]
        assert make_cases_v2(t, 3, 12)[:12] == make_cases(t, 3, 12)
        assert len(make_cases_v2(t, 3, 12)) == 12


def test_unknown_suite_is_refused(tmp_path):
    from kernel.orchestrator import Sandbox, Config
    cfg = Config(repo_dir=tmp_path, runs_dir=tmp_path)
    with pytest.raises(ValueError):
        PytestVerifier(b"x" * 16, Sandbox(cfg), suite="v3")
