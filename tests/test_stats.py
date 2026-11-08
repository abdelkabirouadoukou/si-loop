import math
import random

from kernel.stats import (ELond, SequentialMonitor, bootstrap_ci, bootstrap_lower, e_value, gate_stats, laplace,
                          log_e_down, log_e_up, noisy_exceeds, p_better, paired_counts, sign_flip_p, sign_test_p,
                          sprt_llr)


def test_p_better_and_sign_test_known_values():
    assert abs(p_better(0, 0) - 0.5) < 1e-12 and abs(p_better(1, 0) - 0.75) < 1e-12
    assert abs(sign_test_p(3, 0) - 0.125) < 1e-12 and sign_test_p(0, 0) == 1.0
    assert sign_test_p(10, 0) < 0.001 < sign_test_p(5, 5)


def test_e_value_closed_form_matches_numeric_integral():
    for b, c in [(0, 0), (1, 0), (3, 2), (7, 1), (2, 9)]:
        # E = 2 * integral_{1/2}^{1} q^b (1-q)^c / (1/2)^(b+c) dq
        n, tot = 20000, 0.0
        for i in range(n):
            q = 0.5 + 0.5 * (i + 0.5) / n
            tot += q ** b * (1 - q) ** c / 0.5 ** (b + c)
        num = 2 * tot * 0.5 / n
        assert abs(e_value(log_e_up(b, c)) - num) < 1e-3 * max(1, num), (b, c)


def test_e_values_are_valid_under_null_even_with_optional_stopping():
    """Ville: P(sup_t E_t >= 1/alpha) <= alpha, however we stop."""
    rng, hits, trials = random.Random(1), 0, 600
    for _ in range(trials):
        b = c = 0
        for _ in range(60):
            if rng.random() < 0.5:
                b += 1
            else:
                c += 1
            if log_e_up(b, c) >= math.log(20):
                hits += 1
                break
    assert hits / trials <= 0.05 + 0.03


def test_e_value_has_power_against_real_improvement():
    assert e_value(log_e_up(25, 3)) > 1e3 and e_value(log_e_down(3, 25)) > 1e3
    assert e_value(log_e_up(3, 25)) < 1.0


def test_sequential_monitor_stops_for_harm_and_futility_but_not_for_noise():
    m = SequentialMonitor()
    m.update(0, 9, 20)
    assert m.harmful
    m = SequentialMonitor()
    m.update(1, 9, 30)
    m2 = SequentialMonitor()
    m2.update(6, 6, 50)
    assert m2.futile and not m2.harmful
    m3 = SequentialMonitor()
    m3.update(8, 2, 40)
    assert not m3.futile and not m3.harmful
    assert sprt_llr(8, 0, 0.75) > 0 > sprt_llr(0, 8, 0.75)


def test_elond_level_grows_with_discoveries_and_is_bounded():
    e = ELond(0.05)
    lv0 = e.level()
    hit, _ = e.test(math.log(1e9))
    assert hit and e.level() < lv0 * 0.3 + 1 and e.d == 1
    e2 = ELond(0.05, "uniform", horizon=100)
    assert abs(e2.level() - 0.05 / 100) < 1e-12
    assert all(ELond(0.05).level() <= 0.05 for _ in range(3))


def test_thresholdout_noise_is_symmetric_and_scaled():
    rng = random.Random(0)
    xs = [laplace(rng, 0.1) for _ in range(4000)]
    assert abs(sum(xs) / len(xs)) < 0.02 and 0.07 < sum(abs(x) for x in xs) / len(xs) < 0.13
    assert laplace(rng, 0.0) == 0.0
    assert noisy_exceeds(1.0, 0.0, 0.01, rng) and not noisy_exceeds(-1.0, 0.0, 0.01, rng)


def test_gate_stats_and_paired_counts():
    p, c = {"a": 1, "b": 0, "c": 1, "d": 0}, {"a": 1, "b": 1, "c": 0, "d": 0}
    assert paired_counts(p, c) == (1, 1, 4)
    s = gate_stats(10, 2, 100)
    assert abs(s["delta"] - 0.08) < 1e-12 and s["lower"] < s["delta"] < s["upper"]


def test_run_level_inference_helpers():
    assert sign_flip_p([0.1, 0.2, 0.15, 0.3, 0.12]) <= 1 / 16 + 1e-9
    assert sign_flip_p([]) == 1.0 and sign_flip_p([0.0, 0.0]) == 1.0
    lo, hi = bootstrap_ci([1.0, 1.1, 0.9, 1.05])
    assert 0.8 < lo < 1.0 < hi < 1.2
    assert bootstrap_lower([0.0, 0.0, 0.0]) == 0.0 and bootstrap_lower([]) == -math.inf
