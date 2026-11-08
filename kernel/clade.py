"""kernel/clade.py: clade-based parent selection (Huxley-Goedel-Machine idea, simplified).

A node's own pass rate says little about how PRODUCTIVE it is as a parent: a mediocre agent can
be the root of a lineage that keeps improving. We therefore score a parent by what its whole
CLADE (itself and all descendants) produced, counting every proposal made from each node:

    S_v = sum_{u in clade(v)} gamma^depth(u,v) * successes_u
    F_v = sum_{u in clade(v)} gamma^depth(u,v) * failures_u
    score_v ~ Beta(1 + S_v, 1 + F_v)                 (Thompson sampling picks the parent)

success weights: accept 1.0, stepping_stone 0.3 (counted as 0.7 failure), reject 0.
Rejected children are rolled back and never become nodes, but they DO count as failures of the
parent, which is exactly the information a parent-selection rule needs.
"""
from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Optional

SUCC = {"accept": 1.0, "stepping_stone": 0.3}


class LineageStore:
    def __init__(self, path: Optional[Path] = None, gamma: float = 0.8):
        self.path, self.gamma = (Path(path) if path else None), gamma
        self.nodes: dict = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            self.nodes = json.loads(self.path.read_text())

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.nodes))
        os.replace(tmp, self.path)

    def add_root(self, sha: str, cycle: int = -1) -> None:
        with self._lock:
            self.nodes.setdefault(sha, {"parent": None, "cycle": cycle, "succ": 0.0, "fail": 0.0, "att": 0})
            self._save()

    def record_attempt(self, parent: str, verdict: str, child: Optional[str] = None, cycle: int = 0) -> None:
        with self._lock:
            p = self.nodes.setdefault(parent, {"parent": None, "cycle": -1, "succ": 0.0, "fail": 0.0, "att": 0})
            w = SUCC.get(verdict, 0.0)
            p["att"] += 1
            p["succ"] += w
            p["fail"] += 1.0 - w
            if child and verdict in SUCC and child not in self.nodes:
                self.nodes[child] = {"parent": parent, "cycle": cycle, "succ": 0.0, "fail": 0.0, "att": 0}
            self._save()

    def _children(self) -> dict:
        ch: dict = {}
        for sha, n in self.nodes.items():
            if n["parent"]:
                ch.setdefault(n["parent"], []).append(sha)
        return ch

    def clade_counts(self, sha: str, children: Optional[dict] = None) -> tuple:
        children = children if children is not None else self._children()
        s = f = 0.0
        stack, seen = [(sha, 0)], set()
        while stack:
            v, d = stack.pop()
            if v in seen or v not in self.nodes:
                continue
            seen.add(v)
            w = self.gamma ** d
            s += w * self.nodes[v]["succ"]
            f += w * self.nodes[v]["fail"]
            stack.extend((c, d + 1) for c in children.get(v, []))
        return s, f

    def cmp(self, sha: str) -> float:
        """Posterior mean productivity of a clade."""
        s, f = self.clade_counts(sha)
        return (1 + s) / (2 + s + f)

    def select(self, champion: str, rng: random.Random, banned=(), p_exploit: float = 0.3) -> str:
        with self._lock:
            if champion not in self.nodes:
                self.nodes[champion] = {"parent": None, "cycle": -1, "succ": 0.0, "fail": 0.0, "att": 0}
            if rng.random() < p_exploit:
                return champion
            children = self._children()
            best, best_draw = champion, -1.0
            for sha in self.nodes:
                if sha in banned:
                    continue
                s, f = self.clade_counts(sha, children)
                draw = rng.betavariate(1 + s, 1 + f)
                if draw > best_draw:
                    best, best_draw = sha, draw
            return best
