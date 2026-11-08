import json

import pytest

from kernel.ledger import GENESIS, HashChainLedger, LedgerTampered


def fill(path, n=6):
    led = HashChainLedger(path)
    for i in range(n):
        led.write(ev="begin" if i % 2 == 0 else "end", cycle=i // 2, x=i)
    return led


def test_chain_links_and_verifies(tmp_path):
    led = fill(tmp_path / "l.jsonl")
    recs = led.records()
    assert recs[0]["prev"] == GENESIS and all(recs[i]["prev"] == recs[i - 1]["hash"] for i in range(1, len(recs)))
    assert led.verify() == (True, 6, -1) and led.head()["seq"] == 5
    again = HashChainLedger(tmp_path / "l.jsonl")                    # reopen continues the chain
    again.write(ev="x")
    assert again.verify()[0] and len(again.records()) == 7


def test_editing_any_record_is_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    fill(p)
    lines = p.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["x"] = 999
    lines[2] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerTampered):
        HashChainLedger(p)
    ok, n, bad = HashChainLedger(p, verify_on_open=False).verify()
    assert not ok and bad == 2


def test_deleting_middle_or_tail_is_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    fill(p)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:2] + lines[3:]) + "\n")             # delete record 2
    assert not HashChainLedger(p, verify_on_open=False).verify()[0]
    p2 = tmp_path / "m.jsonl"
    fill(p2)
    lines = p2.read_text().splitlines()
    p2.write_text("\n".join(lines[:-2]) + "\n")                       # truncate the tail (head file now disagrees)
    assert not HashChainLedger(p2, verify_on_open=False).verify()[0]


def test_partial_trailing_write_is_repaired_not_flagged(tmp_path):
    p = tmp_path / "l.jsonl"
    fill(p, 4)
    with open(p, "a") as f:
        f.write('{"seq": 4, "t": 1, "prev": "abc')                    # crash mid-write
    led = HashChainLedger(p)
    assert led.verify()[0] and len(led.records()) == 4
    led.write(ev="after_crash")
    assert led.verify()[0]


def test_incomplete_cycles_found(tmp_path):
    led = HashChainLedger(tmp_path / "l.jsonl")
    led.write(ev="begin", cycle=1, branch="cand/000001")
    led.write(ev="begin", cycle=2, branch="cand/000002")
    led.write(ev="end", cycle=1)
    assert [r["cycle"] for r in led.incomplete_cycles()] == [2]
