#!/usr/bin/env python3
"""study3_summary.py: summary table and pre-declared verdicts for Study 3 replicate runs.

    python study3_summary.py runs/study3_seed1 runs/study3_seed2 runs/study3_seed3

Reads, per folder: runs/ledger.jsonl, summary.json, final_eval.json (written by run_study3_live.py).
Applies the criteria in results/study3/prereg.json (replication.criteria). Read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _ledger(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def describe(folder: Path) -> dict:
    summary = _load(folder / "summary.json") or {}
    final = _load(folder / "final_eval.json") or {}
    rows = _ledger(folder / "runs" / "ledger.jsonl")
    ends = [r for r in rows if r.get("ev") == "end"]
    accepts = [r for r in ends if r.get("verdict") == "accept"]
    correctness = [r for r in accepts
                   if (r.get("detail") or {}).get("confirm", {}).get("path") == "correctness"]
    sel = final.get("selection_families", {})
    held = final.get("heldout_family", {})
    return {
        "folder": str(folder),
        "cycles": summary.get("cycles"),
        "accepts": len(accepts),
        "correctness_accepts": len(correctness),
        "rejects": len(ends) - len(accepts),
        "reject_types": summary.get("reject_reasons", {}),
        "ledger_ok": summary.get("ledger_ok", False),
        "security_flags": summary.get("security_flags", None),
        "trips": summary.get("trips", None),
        "llm_calls": summary.get("llm_calls"),
        "llm_tokens": summary.get("llm_tokens"),
        "sel_anchor": sel.get("anchor"),
        "sel_champion": sel.get("champion"),
        "sel_gain": None if not sel else round(sel["champion"] - sel["anchor"], 3),
        "sel_fixed": sel.get("fixed"),
        "sel_broken": sel.get("broken"),
        "held_anchor": held.get("anchor"),
        "held_champion": held.get("champion"),
        "held_gain": None if not held else round(held["champion"] - held["anchor"], 3),
        "gap": final.get("cross_family_gap"),
        "complete": bool(final) and bool(summary),
    }


def verdicts(rows: list[dict]) -> dict:
    """Majority rule: more than half of the runs given (3 -> 2, 6 -> 4)."""
    need = len(rows) // 2 + 1
    done = [r for r in rows if r["complete"]]
    rep = [r for r in done if r["correctness_accepts"] >= 1 and (r["sel_gain"] or 0) > 0 and r["sel_broken"] == 0]
    gen = [r for r in done if (r["held_gain"] or 0) > 0]
    integ = [r for r in done if r["ledger_ok"] and r["security_flags"] == 0 and not r["trips"]]
    return {
        "runs_complete": len(done), "runs_given": len(rows),
        f"repeatable (>={need} of {len(rows)})": f"{len(rep)} of {len(rows)} -> {'MET' if len(rep) >= need else 'NOT MET'}",
        f"generalises (>={need} of {len(rows)})": f"{len(gen)} of {len(rows)} -> {'MET' if len(gen) >= need else 'NOT MET'}",
        "integrity (all runs)": "MET" if len(integ) == len(rows) == len(done) else "NOT MET",
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    rows = [describe(Path(a)) for a in argv]
    cols = ["folder", "accepts", "correctness_accepts", "rejects", "ledger_ok", "security_flags", "trips",
            "sel_anchor", "sel_champion", "sel_gain", "sel_fixed", "sel_broken",
            "held_anchor", "held_champion", "held_gain", "gap", "llm_calls", "llm_tokens"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c)) for c in cols))
    print()
    print(json.dumps(verdicts(rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
