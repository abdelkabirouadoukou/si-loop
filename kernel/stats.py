"""kernel/stats.py: pure statistics used by the judge. No kernel imports (no cycles).

Everything here works on DISCORDANT PAIRS of a paired design: on the same problems,
b = problems the candidate fixed, c = problems it broke, n = problems.

  * p_better / sign_test_p        exact tail probabilities
  * log_e_up / log_e_down         anytime-valid e-values (mixture likelihood ratios), so the
                                  sequential screen may stop at ANY time without inflating errors
  * sprt_llr / SequentialMonitor  early stopping on discordant pairs (harm / futility)
  * ELond                         online FDR control over thousands of tests (e-LOND, Xu & Ramdas)
  * thresholdout helpers          noisy decisions for reused audit data (Dwork et al. 2015)
  * bootstrap_lower, sign_flip_p  run-level inference for yield tests and ablation arms
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence


# --------------------------------------------------------------------------- exact tails

def _lower_upper(b: int, c: int) -> tuple[int, int]:
    """Integer tails of Binomial(b+c+1, 1/2): lower = sum_{k<=b} C(n,k), upper = 2^n - lower."""
    n = b + c + 1
    lower = sum(math.comb(n, k) for k in range(b + 1))
    return lower, (1 << n) - lower


def p_better(b: int, c: int) -> float:
    """Exact P(Beta(1+b, 1+c) > 0.5) = P(Binomial(b+c+1, 1/2) <= b). Flat prior, discordant pairs."""
    lower, _ = _lower_upper(b, c)
    return lower / (1 << (b + c + 1))


def sign_test_p(b: int, c: int) -> float:
    """One-sided exact sign-test p-value for 'candidate is better': P(Binomial(b+c, 1/2) >= b)."""
    m = b + c
    if m == 0:
        return 1.0
    return sum(math.comb(m, k) for k in range(b, m + 1)) / (1 << m)


def _log_beta(a: int, b: int) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_e_up(b: int, c: int) -> float:
    """log of the e-value for H0: 'not better' (P(fix | discordant) <= 1/2).

    Mixture over q ~ Uniform(1/2, 1) of the likelihood ratio q^b (1-q)^c / (1/2)^(b+c). Each
    component has expectation <= 1 under every p <= 1/2, so the mixture is an e-value, and it
    is a martingale over the sequence of discordant pairs (Ville: valid under optional stopping).
    Closed form: E = B(b+1, c+1) * sum_{k<=b} C(b+c+1, k)."""
    lower, _ = _lower_upper(b, c)
    return _log_beta(b + 1, c + 1) + math.log(lower)


def log_e_down(b: int, c: int) -> float:
    """log e-value for H0: 'not worse' (P(fix | discordant) >= 1/2), mixture over q < 1/2."""
    _, upper = _lower_upper(b, c)
    return _log_beta(b + 1, c + 1) + math.log(upper)


def e_value(log_e: float) -> float:
    return math.exp(log_e) if log_e < 700 else math.inf


# --------------------------------------------------------------------------- paired summary

def paired_counts(parent: dict, cand: dict, ids: Optional[Sequence] = None) -> tuple[int, int, int]:
    ids = list(ids) if ids is not None else sorted(parent)
    b = sum(1 for i in ids if cand[i] and not parent[i])
    c = sum(1 for i in ids if parent[i] and not cand[i])
    return b, c, len(ids)


def gate_stats(b: int, c: int, n: int, z: float = 1.645) -> dict:
    """delta, se, lower and upper bound of the paired pass-rate difference (normal approx)."""
    delta = (b - c) / max(1, n)
    se = math.sqrt(max(b + c - (b - c) ** 2 / max(1, n), 0.0)) / max(1, n)
    return {"delta": delta, "se": se, "lower": delta - z * se, "upper": delta + z * se}


# --------------------------------------------------------------------------- sequential monitor

def sprt_llr(b: int, c: int, p1: float) -> float:
    """Log likelihood ratio of H1: P(fix | discordant) = p1 against H0: 1/2."""
    return b * math.log(p1 / 0.5) + c * math.log((1 - p1) / 0.5)


@dataclass
class SequentialMonitor:
    """Early stopping on discordant pairs. A stop for HARM or FUTILITY only ever costs power
    (a candidate is dropped early); it can never create a false accept, because acceptance
    happens on the independent confirmation batch.

      harm    : e-value against 'not worse' >= 1/harm_alpha          (anytime-valid)
      futility: SPRT log-LR (H1: p1 = 0.5 + theta) <= log(beta / (1 - fut_alpha))
    """
    harm_alpha: float = 0.01
    theta: float = 0.25
    fut_beta: float = 0.20
    fut_alpha: float = 0.05
    min_discordant: int = 6
    b: int = 0
    c: int = 0
    n: int = 0

    def update(self, b: int, c: int, n: int) -> None:
        self.b, self.c, self.n = self.b + b, self.c + c, self.n + n

    @property
    def harmful(self) -> bool:
        return self.c > self.b and log_e_down(self.b, self.c) >= math.log(1.0 / self.harm_alpha)

    @property
    def futile(self) -> bool:
        if self.b + self.c < self.min_discordant:
            return False
        return sprt_llr(self.b, self.c, 0.5 + self.theta) <= math.log(self.fut_beta / (1 - self.fut_alpha))


# --------------------------------------------------------------------------- online FDR

class ELond:
    """e-LOND online FDR control (Xu & Ramdas 2024), valid under ARBITRARY dependence, which is
    what an adaptive loop produces. Test t is rejected (a 'discovery') iff e_t >= 1/alpha_t with
        alpha_t = alpha * gamma_t * (D_{t-1} + 1),   D = discoveries so far,  sum_t gamma_t <= 1.
    gamma: 'inv_sq' (6/(pi^2 t^2), anytime) or 'uniform' with a known horizon T."""

    def __init__(self, alpha: float = 0.05, gamma: str = "inv_sq", horizon: int = 1000):
        self.alpha, self.kind, self.horizon = alpha, gamma, horizon
        self.t, self.d = 0, 0

    def _gamma(self, t: int) -> float:
        return 6.0 / (math.pi ** 2 * t * t) if self.kind == "inv_sq" else 1.0 / self.horizon

    def level(self) -> float:
        """Level the NEXT test would be run at."""
        return min(1.0, self.alpha * self._gamma(self.t + 1) * (self.d + 1))

    def test(self, log_e: float) -> tuple[bool, float]:
        lvl = self.level()
        self.t += 1
        hit = log_e >= math.log(1.0 / lvl)
        self.d += int(hit)
        return hit, lvl

    def state(self) -> dict:
        return {"t": self.t, "d": self.d}

    def load(self, st: dict) -> None:
        self.t, self.d = int(st.get("t", 0)), int(st.get("d", 0))


# --------------------------------------------------------------------------- thresholdout-style noise

def laplace(rng: random.Random, scale: float) -> float:
    if scale <= 0:
        return 0.0
    u = rng.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(max(1e-12, 1 - 2 * abs(u)))


def noisy_exceeds(stat: float, threshold: float, scale: float, rng: random.Random) -> bool:
    """Reusable-holdout answer: 1 bit, with Laplace noise on both the statistic and the threshold
    (sparse-vector style). Limits what repeated audits leak about the audit data."""
    return stat + laplace(rng, scale) > threshold + laplace(rng, scale)


# --------------------------------------------------------------------------- run-level inference

def bootstrap_lower(values: Sequence[float], alpha: float = 0.05, reps: int = 4000,
                    seed: int = 0) -> float:
    """One-sided percentile-bootstrap lower bound on the mean (run is the unit of analysis)."""
    v = list(values)
    if not v:
        return -math.inf
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(reps))
    return means[int(alpha * reps)]


def bootstrap_ci(values: Sequence[float], level: float = 0.95, reps: int = 4000, seed: int = 0) -> tuple:
    v = list(values)
    if not v:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(reps))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return means[int(lo * reps)], means[min(reps - 1, int(hi * reps))]


def sign_flip_p(diffs: Sequence[float], reps: int = 20000, seed: int = 0) -> float:
    """One-sided paired permutation (sign-flip) p-value for mean(diffs) > 0. Exact when 2^n is small."""
    d = [x for x in diffs]
    n = len(d)
    if n == 0:
        return 1.0
    obs = sum(d)
    if n <= 16:
        cnt = 0
        for mask in range(1 << n):
            s = sum(x if (mask >> i) & 1 else -x for i, x in enumerate(d))
            cnt += s >= obs - 1e-12
        return cnt / (1 << n)
    rng = random.Random(seed)
    cnt = sum(sum(x if rng.random() < 0.5 else -x for x in d) >= obs - 1e-12 for _ in range(reps))
    return (cnt + 1) / (reps + 1)


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))
