"""study3_summary: reads a replicate folder and applies the pre-declared criteria."""
import json

import study3_summary as S


def _folder(root, name, *, sel=(0.34, 0.53, 22, 0), held=(0.35, 0.35), accept_path="correctness", ok=True):
    d = root / name
    (d / "runs").mkdir(parents=True)
    rows = [{"ev": "end", "cycle": 0, "verdict": "accept", "detail": {"confirm": {"path": accept_path}}},
            {"ev": "end", "cycle": 1, "verdict": "reject", "error_type": "unconfirmed", "detail": {}}]
    (d / "runs" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "summary.json").write_text(json.dumps({"cycles": 2, "ledger_ok": ok, "security_flags": 0,
                                                "trips": [], "reject_reasons": {"unconfirmed": 1},
                                                "llm_calls": 3, "llm_tokens": 9000}))
    (d / "final_eval.json").write_text(json.dumps({
        "selection_families": {"anchor": sel[0], "champion": sel[1], "fixed": sel[2], "broken": sel[3], "n": 120},
        "heldout_family": {"anchor": held[0], "champion": held[1], "fixed": 0, "broken": 0, "n": 120},
        "cross_family_gap": round((sel[1] - sel[0]) - (held[1] - held[0]), 3)}))
    return d


def test_describe_reads_all_fields(tmp_path):
    r = S.describe(_folder(tmp_path, "a"))
    assert r["accepts"] == 1 and r["correctness_accepts"] == 1 and r["rejects"] == 1
    assert r["sel_gain"] == 0.19 and r["held_gain"] == 0.0 and r["gap"] == 0.19
    assert r["complete"] is True


def test_verdicts_follow_the_declared_rules(tmp_path):
    rows = [S.describe(_folder(tmp_path, f"s{i}")) for i in range(3)]
    v = S.verdicts(rows)
    assert v["repeatable (>=2 of 3)"].endswith("MET")
    assert v["generalises (>=2 of 3)"].endswith("NOT MET")          # held-out gain is 0 everywhere
    assert v["integrity (all runs)"] == "MET"


def test_utility_accept_does_not_count_as_repeatable(tmp_path):
    rows = [S.describe(_folder(tmp_path, f"u{i}", accept_path="utility")) for i in range(3)]
    assert S.verdicts(rows)["repeatable (>=2 of 3)"].endswith("NOT MET")


def test_integrity_fails_on_broken_ledger(tmp_path):
    rows = [S.describe(_folder(tmp_path, "bad", ok=False))]
    assert S.verdicts(rows)["integrity (all runs)"] == "NOT MET"


def test_missing_final_eval_is_incomplete(tmp_path):
    d = _folder(tmp_path, "partial")
    (d / "final_eval.json").unlink()
    r = S.describe(d)
    assert r["complete"] is False
    assert S.verdicts([r])["runs_complete"] == 0


def test_majority_rule_for_six_runs(tmp_path):
    good = lambda name: S.describe(_folder(tmp_path, name, held=(0.3, 0.6)))     # held-out gain > 0
    flat = lambda name: S.describe(_folder(tmp_path, name))                      # held-out gain 0
    four = [good(f"g{i}") for i in range(4)] + [flat("f0"), flat("f1")]
    three = [good(f"h{i}") for i in range(3)] + [flat(f"k{i}") for i in range(3)]
    assert S.verdicts(four)["generalises (>=4 of 6)"] == "4 of 6 -> MET"
    assert S.verdicts(three)["generalises (>=4 of 6)"] == "3 of 6 -> NOT MET"
