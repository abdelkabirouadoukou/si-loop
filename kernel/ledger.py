"""kernel/ledger.py: append-only, fsynced, HASH-CHAINED ledger (tamper-evident history).

Each line is  {"seq", "t", "prev", ...fields..., "hash"}  with
    hash = sha256(prev_hash || canonical_json(record without "hash")).
Editing, deleting or re-ordering any line breaks every later hash. A small `.head` file holds
(seq, hash) of the last record so that deleting the TAIL is also detected. An attacker who can
rewrite both files can still forge history: pin `head()` somewhere the loop cannot write
(another machine, a git tag, a printed line). `begin` without `end` marks a crashed cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


class LedgerTampered(Exception):
    """The hash chain does not verify."""


def _canon(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prev: str, rec: dict) -> str:
    return hashlib.sha256((prev + _canon(rec)).encode()).hexdigest()


class HashChainLedger:
    def __init__(self, path: Path, verify_on_open: bool = True):
        self.path = Path(path)
        self.head_path = self.path.with_suffix(self.path.suffix + ".head")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.seq, self.last = 0, GENESIS
        if self.path.exists():
            self._repair_partial_tail()
            if verify_on_open:
                ok, n, bad = self.verify()
                if not ok:
                    raise LedgerTampered(f"ledger chain broken at record {bad} ({self.path})")
            recs = self._raw()
            if recs:
                self.seq, self.last = recs[-1]["seq"] + 1, recs[-1]["hash"]

    # ------------------------------------------------------------------ io
    def _raw(self) -> list:
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_corrupt": line})
        return out

    def _repair_partial_tail(self) -> None:
        """A crash mid-write leaves a final line without a newline. Drop ONLY that partial line."""
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            self.path.write_bytes(data[:cut])

    def write(self, **fields) -> dict:
        with self._lock:
            rec = {"seq": self.seq, "t": round(time.time(), 3), "prev": self.last, **fields}
            h = _digest(self.last, rec)
            line = _canon({**rec, "hash": h})
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.seq, self.last = self.seq + 1, h
            tmp = self.head_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"seq": rec["seq"], "hash": h}))
            os.replace(tmp, self.head_path)
            return {**rec, "hash": h}

    # ------------------------------------------------------------------ reading
    def records(self) -> list:
        if not self.path.exists():
            return []
        return [r for r in self._raw() if "_corrupt" not in r]

    def incomplete_cycles(self) -> list:
        begun, ended = {}, set()
        for r in self.records():
            if r.get("ev") == "begin":
                begun[r["cycle"]] = r
            elif r.get("ev") == "end":
                ended.add(r["cycle"])
        return [begun[c] for c in begun if c not in ended]

    def head(self) -> dict:
        return {"seq": self.seq - 1, "hash": self.last}

    # ------------------------------------------------------------------ verification
    def verify(self) -> tuple:
        """(ok, n_records, first_bad_index or -1)."""
        prev, n = GENESIS, 0
        if not self.path.exists():
            return True, 0, -1
        for i, rec in enumerate(self._raw()):
            if "_corrupt" in rec or rec.get("prev") != prev or rec.get("seq") != i:
                return False, n, i
            body = {k: v for k, v in rec.items() if k != "hash"}
            if _digest(prev, body) != rec.get("hash"):
                return False, n, i
            prev, n = rec["hash"], n + 1
        if self.head_path.exists():
            try:
                head = json.loads(self.head_path.read_text())
                if head["seq"] != n - 1 or head["hash"] != prev:
                    return False, n, n                       # tail deleted or head forged
            except (json.JSONDecodeError, KeyError):
                return False, n, n
        return True, n, -1
