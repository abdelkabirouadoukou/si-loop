"""study4/fdr_replay.py: makes the enforced online false-discovery control (kernel/stats.py: ELond)
continuous across separate `python run_study4.py --run` process restarts.

kernel/orchestrator.py always constructs a FRESH ELond(t=0, d=0) for every new Orchestrator; nothing
there persists or restores its state. Within one continuous invocation this is fine, but Study 4 is
run as repeated, separate CLI invocations against the same output directory (rate limits, pauses
between sessions, etc.) -- so the enforced budget was silently restarting each time, undermining the
anytime-valid guarantee e-LOND is supposed to provide over the WHOLE sequence of correctness-path
tests in the study, not just the tests in one process's lifetime.

The fix needs no kernel/ edit: kernel/pipeline.py's TwoStageJudge already accepts a pre-built `elond`
via an optional constructor parameter. replay_elond_state() rebuilds (t, d) by replaying every past
correctness-path FDR test recorded in the ledger, in order, using the "hit" outcome ELond ITSELF
already recorded for that test (ELond.test()'s own update rule: t += 1, d += int(hit)) -- not by
recomputing the underlying e-value, so this is exactly consistent with what actually happened, not
an approximation of it.
"""
from __future__ import annotations

import json
from pathlib import Path

from kernel.stats import ELond


def replay_elond_state(ledger_path: Path, fdr_alpha: float) -> ELond:
    """A fresh ELond(fdr_alpha) advanced through every PAST correctness-path FDR test in the ledger
    (in file order, i.e. chronological), or an untouched ELond(t=0, d=0) if the ledger has none yet."""
    e = ELond(fdr_alpha)
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return e
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                                    # a torn last line after a crash is skipped
        if rec.get("ev") != "end":
            continue
        fdr = (rec.get("detail") or {}).get("fdr")
        if not fdr:
            continue                                     # no correctness-path test this cycle
        e.t += 1
        e.d += int(bool(fdr.get("hit")))
    return e


def attach_persistent_fdr_budget(orch, ledger_path: Path) -> ELond:
    """Replays the ledger and installs the resulting ELond on BOTH orch.elond and orch.judge.elond
    (TwoStageJudge captured orch.elond's object identity at construction time -- replacing only one
    of the two attributes would leave them judging against different budgets)."""
    restored = replay_elond_state(ledger_path, orch.cfg.fdr_alpha)
    orch.elond = restored
    orch.judge.elond = restored
    return restored
