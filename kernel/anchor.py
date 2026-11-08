"""kernel/anchor.py: anchor, margin budget and audits. Eliminates CREEPING DEGRADATION.

Problem. A gate that compares a candidate only to its parent tolerates a small loss (eps) per
accept. Over thousands of cycles many "non-inferior but slightly worse" patches (typically the
smaller-code utility path) can stack up and nobody ever sees the total.

Mechanisms
  1. ANCHOR      the cycle-0 agent is frozen (tag `anchor/cycle0`, sha stored in anchor.json).
  2. MARGIN BUDGET  every accept spends the plausible loss it was allowed to hide:
                 charge = max(0, -lower)  (+ a floor `min_charge` on the utility path).
                 When the budget is empty the utility path is closed and every accept must show
                 lower >= 0. Only a PROVEN gain over the anchor (lower bound > 0 on a FRESH batch)
                 refills it, and only for the part of the gain not already banked.
  3. ANCHOR AUDIT   every `anchor_every` cycles, champion vs anchor vs peak on fresh problems.
                 Champion proven worse than the best audited champion -> roll back to it.
  4. HELD-OUT AUDIT champion vs anchor on a family that selection never touches, plus the
                 cross-family generalisation gap. Decisions use Thresholdout-style noise. Results
                 go to the ledger only: the agent never sees them.
"""
from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Callable, Optional

from kernel.stats import gate_stats, noisy_exceeds, paired_counts


class MarginBudget:
    def __init__(self, path: Optional[Path], max_budget: float = 0.03, min_charge: float = 0.002):
        self.path, self.max, self.min_charge = (Path(path) if path else None), max_budget, min_charge
        self.remaining, self.banked_gain = max_budget, 0.0
        self.spent_total = self.refilled_total = 0.0
        self.events: list = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.remaining, self.banked_gain = d["remaining"], d["banked_gain"]
            self.spent_total, self.refilled_total, self.events = d["spent_total"], d["refilled_total"], d["events"]

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"remaining": self.remaining, "banked_gain": self.banked_gain,
                                   "spent_total": self.spent_total, "refilled_total": self.refilled_total,
                                   "max": self.max, "events": self.events[-100:]}))
        os.replace(tmp, self.path)

    def charge_for(self, lower: float, utility_path: bool) -> float:
        """Plausible loss hidden by this accept: the amount its lower bound dips below zero."""
        hidden = max(0.0, -lower)
        return max(self.min_charge, hidden) if utility_path else hidden

    def can_afford(self, charge: float) -> bool:
        return charge <= self.remaining + 1e-12

    def spend(self, charge: float, cycle: int = -1) -> None:
        with self._lock:
            self.remaining = max(0.0, self.remaining - charge)
            self.spent_total += charge
            self.events.append({"cycle": cycle, "spend": round(charge, 5), "remaining": round(self.remaining, 5)})
            self._save()

    def refill_from_audit(self, lower_vs_anchor: float, cycle: int = -1) -> float:
        """Refill only by the NEW proven gain over the anchor (high-water mark), capped at max."""
        with self._lock:
            new_gain = max(0.0, lower_vs_anchor - self.banked_gain)
            amount = min(new_gain, self.max - self.remaining)
            if lower_vs_anchor > self.banked_gain:
                self.banked_gain = lower_vs_anchor
            self.remaining += amount
            self.refilled_total += amount
            if amount > 0:
                self.events.append({"cycle": cycle, "refill": round(amount, 5), "remaining": round(self.remaining, 5)})
            self._save()
            return amount


class AnchorStore:
    """Frozen anchor + 'peak' (best champion proven on audits). Persisted; tag-verified on load."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.d: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    @property
    def frozen(self) -> bool:
        return "anchor_sha" in self.d

    @property
    def anchor_sha(self) -> str:
        return self.d["anchor_sha"]

    @property
    def peak_sha(self) -> str:
        return self.d.get("peak_sha", self.d["anchor_sha"])

    @property
    def seed_loc(self) -> float:
        return float(self.d.get("seed_loc", 0.0))

    def freeze(self, sha: str, loc: float) -> None:
        if self.frozen:
            raise RuntimeError("anchor already frozen")
        self.d = {"anchor_sha": sha, "peak_sha": sha, "seed_loc": loc, "audits": 0}
        self.save()

    def set_peak(self, sha: str) -> None:
        self.d["peak_sha"] = sha
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d))
        os.replace(tmp, self.path)


EvalFn = Callable[[str, list], object]          # (sha, batch) -> EvalResult-like with .passes


class AnchorAuditor:
    """Periodic audit on FRESH problems (new ids every audit)."""

    def __init__(self, cfg, store: AnchorStore, budget: MarginBudget, draw: Callable, evaluate: EvalFn):
        self.cfg, self.store, self.budget, self.draw, self.evaluate = cfg, store, budget, draw, evaluate

    def audit(self, cycle: int, champion: str) -> dict:
        z = self.cfg.z_one_sided
        batch = self.draw(cycle, self.cfg.n_anchor, "anchor")
        ch, an = self.evaluate(champion, batch), self.evaluate(self.store.anchor_sha, batch)
        ids = sorted(k for k in ch.passes if k in an.passes)
        b, c, n = paired_counts(an.passes, ch.passes, ids)
        vs_anchor = gate_stats(b, c, n, z)
        rep = {"kind": "anchor", "cycle": cycle, "n": n, "champion": champion[:10],
               "vs_anchor": {**vs_anchor, "b": b, "c": c}, "champion_pass": ch.pass_at_1,
               "anchor_pass": an.pass_at_1, "action": "ok"}
        rep["refill"] = self.budget.refill_from_audit(vs_anchor["lower"], cycle)
        peak = self.store.peak_sha
        if peak != champion:
            pk = self.evaluate(peak, batch)
            bp, cp, _ = paired_counts(pk.passes, ch.passes, ids)
            vs_peak = gate_stats(bp, cp, n, z)
            rep["vs_peak"] = {**vs_peak, "b": bp, "c": cp}
            if vs_peak["upper"] < -self.cfg.anchor_tolerance:
                rep.update(action="rollback", rollback_to=peak, reason="champion proven worse than best audited champion")
            elif vs_peak["lower"] > 0:
                self.store.set_peak(champion)
                rep["new_peak"] = True
        elif vs_anchor["lower"] > 0:
            rep["new_peak"] = False
        if vs_anchor["upper"] < 0 and rep["action"] == "ok":
            # champion proven worse than the CYCLE-0 agent: the only safe state is the anchor itself
            rep.update(action="rollback", rollback_to=self.store.anchor_sha, reason="champion proven worse than the anchor")
        self.store.d["audits"] = self.store.d.get("audits", 0) + 1
        self.store.save()
        rep["selection_delta"] = vs_anchor["delta"]
        rep["selection_se"] = vs_anchor["se"]
        return rep


class HeldoutAuditor:
    """Champion vs anchor on the HELD-OUT family (never drawn by selection/confirm/credit/anchor
    splits) and the cross-family generalisation gap = selection_delta - heldout_delta."""

    def __init__(self, cfg, store: AnchorStore, draw: Callable, evaluate: EvalFn, seed: int = 0):
        self.cfg, self.store, self.draw, self.evaluate = cfg, store, draw, evaluate
        self.rng = random.Random(seed ^ 0xA0D17)

    def audit(self, cycle: int, champion: str, selection_delta: float = 0.0, selection_se: float = 0.0) -> dict:
        z = self.cfg.z_one_sided
        batch = self.draw(cycle, self.cfg.n_heldout, "heldout")
        ch, an = self.evaluate(champion, batch), self.evaluate(self.store.anchor_sha, batch)
        ids = sorted(k for k in ch.passes if k in an.passes)
        b, c, n = paired_counts(an.passes, ch.passes, ids)
        h = gate_stats(b, c, n, z)
        gap = selection_delta - h["delta"]
        gap_se = (selection_se ** 2 + h["se"] ** 2) ** 0.5
        rep = {"kind": "heldout", "cycle": cycle, "n": n, "heldout": {**h, "b": b, "c": c},
               "heldout_pass": ch.pass_at_1, "anchor_heldout_pass": an.pass_at_1,
               "gap": gap, "gap_lower": gap - z * gap_se, "action": "ok"}
        # regression on the unseen family: noisy 1-bit decision (limits leakage through repeated audits)
        if noisy_exceeds(-h["upper"], self.cfg.anchor_tolerance, self.cfg.thresholdout_scale, self.rng):
            rep.update(action="rollback", rollback_to=self.store.peak_sha,
                       reason="champion regressed on the held-out family")
        elif rep["gap_lower"] > self.cfg.heldout_gap_alarm:
            rep.update(action="pause", reason="cross-family generalisation gap above alarm threshold")
        return rep
