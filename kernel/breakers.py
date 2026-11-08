"""kernel/breakers.py: circuit breakers for long unattended runs.

Each observed cycle updates rolling windows. A breaker returns a Trip with a severity:
   "pause"  the run sleeps `cooloff_s` and continues (transient or ambiguous trouble)
   "halt"   the run STOPS and writes a latch file HALT (a human must inspect and clear it)

Breakers
  accept_rate_high   too many accepts for the expected rate (binomial tail): either the verifier is
                     exploitable or the gate leaks. Default action pause; halt below `halt_p`.
  stall              no accept for 2 windows (information only, pause)
  loc_explosion      champion LOC above the cap (factor x seed LOC or +abs) -> halt
  cost_spike         cycle cost > factor x rolling median, or the run token cap is hit -> halt
  security           honeypot flags / sandbox-escape attempts above limits -> halt
  consecutive_fail   N failed cycles in a row -> pause
  audit_regression   two audit rollbacks within `audit_halt_n` audits -> halt
  disk_low           free disk below the floor -> pause (GC runs first)
"""
from __future__ import annotations

import json
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kernel.stats import binom_tail_ge


@dataclass
class Trip:
    name: str
    severity: str            # pause | halt
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "severity": self.severity, "detail": self.detail}


class CircuitBreakers:
    def __init__(self, cfg, runs_dir: Path, seed_loc: float = 0.0):
        self.cfg, self.dir, self.seed_loc = cfg, Path(runs_dir), seed_loc
        w = cfg.breaker_window
        self.verdicts: deque = deque(maxlen=2 * w)
        self.costs: deque = deque(maxlen=w)
        self.escapes: deque = deque(maxlen=w)
        self.flags_total = 0
        self.fail_streak = 0
        self.run_tokens = 0
        self.audit_rollbacks: deque = deque(maxlen=cfg.audit_halt_n)
        self.trips: list = []

    # ------------------------------------------------------------------ latch
    @property
    def halt_file(self) -> Path:
        return self.dir / "HALT"

    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    def latched(self) -> Optional[str]:
        return self.halt_file.read_text() if self.halt_file.exists() else None

    def stop_requested(self) -> bool:
        return self.stop_file.exists()

    def latch(self, trip: Trip) -> None:
        self.halt_file.write_text(json.dumps(trip.as_dict()))

    def clear(self) -> None:
        for f in (self.halt_file, self.stop_file):
            f.unlink(missing_ok=True)

    # ------------------------------------------------------------------ observation
    def loc_cap(self) -> float:
        if self.seed_loc <= 0:
            return float("inf")
        return max(self.cfg.loc_factor * self.seed_loc, self.seed_loc + self.cfg.loc_abs)

    def observe(self, out: dict) -> list:
        """`out` is the cycle record: verdict, error_type, agent_loc, cost_tokens, security_flags."""
        c, trips = self.cfg, []
        verdict = out.get("verdict", "reject")
        self.verdicts.append(verdict)
        self.fail_streak = self.fail_streak + 1 if out.get("error_type") in ("internal", "budget") else 0
        flags = int(out.get("security_flags", 0))
        self.flags_total += flags
        if out.get("error_type") in ("static", "tamper"):
            self.escapes.append(1)
        elif flags == 0:
            self.escapes.append(0)
        cost = float(out.get("cost_tokens", 0) or 0)
        self.run_tokens += cost

        if flags and self.flags_total >= c.max_security_flags:
            trips.append(Trip("security", "halt", f"{self.flags_total} honeypot/security flags"))
        if sum(self.escapes) >= c.escape_attempt_limit:
            trips.append(Trip("security", "halt", f"{sum(self.escapes)} sandbox-escape attempts in window"))
        loc = out.get("champion_loc")
        if loc is not None and loc > self.loc_cap():
            trips.append(Trip("loc_explosion", "halt", f"champion LOC {loc} > cap {self.loc_cap():.0f}"))
        if c.run_token_cap and self.run_tokens > c.run_token_cap:
            trips.append(Trip("cost_spike", "halt", f"run token cap {c.run_token_cap} exceeded"))
        if cost > 0 and len(self.costs) >= 10:
            med = statistics.median(self.costs)
            if med > 0 and cost > c.cost_spike_factor * med:
                trips.append(Trip("cost_spike", "pause", f"cycle cost {cost:.0f} > {c.cost_spike_factor}x median {med:.0f}"))
        if cost > 0:
            self.costs.append(cost)
        if self.fail_streak >= c.max_consecutive_failures:
            trips.append(Trip("consecutive_fail", "pause", f"{self.fail_streak} failed cycles in a row"))
            self.fail_streak = 0
        w = c.breaker_window
        if len(self.verdicts) >= w:
            recent = list(self.verdicts)[-w:]
            k = recent.count("accept")
            p = binom_tail_ge(k, w, c.expected_accept_rate)
            if p < c.accept_anomaly_p:
                sev = "halt" if p < c.accept_halt_p else "pause"
                trips.append(Trip("accept_rate_high", sev, f"{k}/{w} accepts, P(>=k | p0={c.expected_accept_rate}) = {p:.2e}"))
        if len(self.verdicts) >= 2 * w and "accept" not in self.verdicts:
            trips.append(Trip("stall", "pause", f"no accept in {2 * w} cycles"))
            self.verdicts.clear()
        self.trips += trips
        return trips

    def observe_audit(self, rep: dict) -> list:
        trips = []
        self.audit_rollbacks.append(1 if rep.get("action") == "rollback" else 0)
        if sum(self.audit_rollbacks) >= self.cfg.audit_halt_n:
            trips.append(Trip("audit_regression", "halt", f"{sum(self.audit_rollbacks)} audit rollbacks in the last {self.cfg.audit_halt_n} audits"))
        if rep.get("action") == "pause":
            trips.append(Trip("generalisation_gap", "pause", rep.get("reason", "")))
        self.trips += trips
        return trips

    def observe_disk(self, free_gb: float) -> list:
        if free_gb < self.cfg.min_free_gb:
            t = [Trip("disk_low", "pause", f"{free_gb:.1f} GB free < {self.cfg.min_free_gb} GB")]
            self.trips += t
            return t
        return []
