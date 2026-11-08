"""kernel/bandit.py: operator bandit over hunk edit TAGS with per-tag EMA reward tracking.

Reward per hunk: accept=1.0, stepping_stone=0.3, reject=0.0; when Shapley credit exists the
accepted reward is split by the hunk's sign (helped 1.0, neutral 0.5, hurt 0.0). The agent only
ever sees the rounded EMA per tag (coarse), never raw outcomes of sealed problems.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
from pathlib import Path
from typing import Optional, Sequence


class OperatorBandit:
    def __init__(self, path: Optional[Path] = None, alpha: float = 0.2, prior: float = 0.3,
                 ucb_c: float = 0.15, temperature: float = 0.2, floor: float = 0.03):
        self.path = Path(path) if path else None
        self.alpha, self.prior, self.ucb_c, self.temperature, self.floor = alpha, prior, ucb_c, temperature, floor
        self.arms: dict = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self.arms = json.loads(self.path.read_text())

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.arms, sort_keys=True))
        os.replace(tmp, self.path)

    def update(self, tag: str, reward: float) -> float:
        """EMA with a 1/(n+1) warm-up so the first observations are not dominated by the prior."""
        with self._lock:
            a = self.arms.setdefault(tag, {"ema": self.prior, "n": 0})
            step = max(self.alpha, 1.0 / (a["n"] + 1))
            a["ema"] += step * (float(reward) - a["ema"])
            a["n"] += 1
            self._save()
            return a["ema"]

    def ema(self, tag: str) -> float:
        return self.arms.get(tag, {"ema": self.prior})["ema"]

    def score(self, tag: str) -> float:
        total = sum(a["n"] for a in self.arms.values()) + 1
        n = self.arms.get(tag, {"n": 0})["n"]
        return self.ema(tag) + self.ucb_c * math.sqrt(math.log(total + 1) / (n + 1))

    def probabilities(self, tags: Sequence[str]) -> dict:
        """Softmax over EMA+UCB with a probability floor, so no operator is ever starved."""
        if not tags:
            return {}
        sc = [self.score(t) / self.temperature for t in tags]
        m = max(sc)
        ex = [math.exp(s - m) for s in sc]
        z = sum(ex)
        p = [max(self.floor / len(tags), e / z) for e in ex]
        z2 = sum(p)
        return {t: x / z2 for t, x in zip(tags, p)}

    def choose(self, tags: Sequence[str], rng: random.Random) -> str:
        probs = self.probabilities(tags)
        return rng.choices(list(probs), weights=list(probs.values()))[0]

    def coarse(self, step: float = 0.1) -> dict:
        """What the agent may see: EMA rounded to `step` (never counts or raw outcomes)."""
        return {t: round(round(a["ema"] / step) * step, 2) for t, a in sorted(self.arms.items())}

    def update_from_cycle(self, tags: Sequence[str], verdict: str, credit: Optional[dict] = None,
                          hunk_ids: Optional[Sequence[str]] = None, credit_thr: float = 0.01) -> None:
        base = {"accept": 1.0, "stepping_stone": 0.3}.get(verdict, 0.0)
        for i, tag in enumerate(tags):
            r = base
            if credit and hunk_ids and verdict == "accept":
                phi = credit.get(hunk_ids[i], 0.0)
                r = 1.0 if phi > credit_thr else (0.0 if phi < -credit_thr else 0.5)
            self.update(tag, r)
