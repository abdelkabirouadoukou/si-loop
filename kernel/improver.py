"""kernel/improver.py: the synthetic Study-2 world's improvers (no LLM needed) + frozen reference M0.

WHAT THIS IS. A deterministic, seeded MetaAgent that edits the agent repo through ordinary
validated patch proposals. It proposes four kinds of hunks:
    adopt.<kind>   register a library strategy for a task kind (good or buggy: it cannot tell)
    prune          delete a shadowed (dead) strategy        -> efficiency path of the gate
    mutate         swap one operator inside the newest strategy of a kind (random bug fix / bug)
    meta.tune      edit its OWN policy file agent/meta/policy.py  -> judged by the YIELD TEST
It exists so the kernel, the gate and the ablation arms can be run and tested end to end without an
API key. It makes NO claim about LLM self-improvement; plug a real model in via the MetaAgent
interface (`--meta pkg.module:Class` in run_ablations.py).

M0 = REFERENCE_POLICY frozen in this (digest-protected) file: the control improver for arms and
yield tests. In arm A1 the improver reads its policy from the evolving repo; in A0 it never does.
"""
from __future__ import annotations

import ast
import hashlib
import random
import threading
from pathlib import Path
from typing import Optional

from kernel.orchestrator import Budget, MetaAgent
from kernel.patching import EditOp, PatchProposal

END_MARK = "# -- end of strategies --"
POLICY_PATH = "agent/meta/policy.py"
STRATEGIES_PATH = "agent/strategies.py"
SELECTION_KINDS = ("dedupe", "rotate", "prefix_mod", "kth_smallest", "bracket_balance", "min_stack", "lru_cache")
ALL_KINDS = SELECTION_KINDS + ("arith_eval", "rpn_eval")

REFERENCE_POLICY = {"p_meta": 0.0, "w_adopt": 2.0, "w_prune": 1.0, "w_mutate": 1.5, "use_bandit": 0.0}
TUNABLE = {"p_meta": (0.0, 0.5), "w_adopt": (0.05, 20.0), "w_prune": (0.05, 20.0),
           "w_mutate": (0.05, 20.0), "use_bandit": (0.0, 1.0)}
# A1 starts from EXACTLY M0 except that it spends 15% of its proposals on editing its own policy (the cost of recursion).
SEED_POLICY = {**REFERENCE_POLICY, "p_meta": 0.15}

# =============================================================== strategy library (templates)
# A template is the SOURCE the agent emits for the candidate; @@name@@ is replaced by repr(param).

LIB: dict = {
    "dedupe": [
        ("good", '''def solve(xs):
    keep = @@keep@@
    seq = xs if keep == "first" else xs[::-1]
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out if keep == "first" else out[::-1]
'''),
        ("bad", '''def solve(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
'''),
    ],
    "rotate": [
        ("good", '''def solve(xs, k):
    n = len(xs)
    if n == 0:
        return []
    k = k % n
    if @@dir@@ == "left":
        return xs[k:] + xs[:k]
    return xs[n - k:] + xs[:n - k] if k else list(xs)
'''),
        ("bad", '''def solve(xs, k):
    if not xs:
        return []
    return xs[k:] + xs[:k]
'''),
    ],
    "prefix_mod": [
        ("good", '''def solve(xs):
    out, s = [], 0
    for x in xs:
        s = (s + x) % @@m@@
        out.append(s)
    return out
'''),
        ("bad", '''def solve(xs):
    out, s = [], 0
    for x in xs:
        s = (s + x) % 10
        out.append(s)
    return out
'''),
    ],
    "kth_smallest": [
        ("good", '''def solve(xs, k):
    ys = sorted(set(xs)) if @@distinct@@ else sorted(xs)
    if k < 1 or k > len(ys):
        return None
    return ys[k - 1]
'''),
        ("bad", '''def solve(xs, k):
    ys = sorted(xs)
    if k < 0 or k >= len(ys):
        return None
    return ys[k]
'''),
    ],
    "bracket_balance": [
        ("good", '''def solve(s):
    alpha = @@alpha@@
    close = {alpha[i + 1]: alpha[i] for i in range(0, len(alpha), 2)}
    st = []
    for ch in s:
        if ch in close:
            if not st or st.pop() != close[ch]:
                return False
        else:
            st.append(ch)
    return not st
'''),
        ("bad", '''def solve(s):
    depth = 0
    for i, ch in enumerate(s):
        depth += 1 if i % 2 == 0 and ch in "([{" else -1
        if depth < 0:
            return False
    return depth == 0
'''),
    ],
    "min_stack": [
        ("good", '''class Solution:
    def __init__(self):
        self.d = []
        self.m = []

    def push(self, x):
        cap = @@cap@@
        if cap is not None and len(self.d) >= cap:
            raise OverflowError("full")
        self.d.append(x)
        self.m.append(x if not self.m else min(x, self.m[-1]))

    def pop(self):
        if not self.d:
            raise IndexError("empty")
        self.m.pop()
        return self.d.pop()

    def top(self):
        if not self.d:
            raise IndexError("empty")
        return self.d[-1]

    def get_min(self):
        if not self.d:
            raise IndexError("empty")
        return self.m[-1]

    def size(self):
        return len(self.d)
'''),
        ("bad", '''class Solution:
    def __init__(self):
        self.d = []
        self.lo = None

    def push(self, x):
        cap = @@cap@@
        if cap is not None and len(self.d) >= cap:
            raise OverflowError("full")
        self.d.append(x)
        self.lo = x if self.lo is None else min(self.lo, x)

    def pop(self):
        if not self.d:
            raise IndexError("empty")
        return self.d.pop()

    def top(self):
        if not self.d:
            raise IndexError("empty")
        return self.d[-1]

    def get_min(self):
        if not self.d:
            raise IndexError("empty")
        return self.lo

    def size(self):
        return len(self.d)
'''),
    ],
    "lru_cache": [
        ("good", '''from collections import OrderedDict


class Solution:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()

    def get(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        return @@miss@@

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
'''),
        ("bad", '''from collections import OrderedDict


class Solution:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()

    def get(self, key):
        return self.d.get(key, @@miss@@)

    def put(self, key, value):
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
'''),
    ],
    "arith_eval": [
        ("good", '''def solve(expr):
    div = @@div@@
    toks, i = [], 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            toks.append(int(expr[i:j]))
            i = j
        elif ch in "+-*/()":
            toks.append(ch)
            i += 1
        else:
            raise ValueError("bad char")
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat():
        pos[0] += 1
        return toks[pos[0] - 1]

    def idiv(a, b):
        if b == 0:
            raise ZeroDivisionError("div")
        if div == "floor":
            return a // b
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b > 0) else -q

    def factor():
        t = peek()
        if isinstance(t, int):
            return eat()
        if t == "(":
            eat()
            v = expr_()
            if peek() != ")":
                raise ValueError("expected )")
            eat()
            return v
        raise ValueError("unexpected")

    def term():
        v = factor()
        while peek() in ("*", "/"):
            op = eat()
            w = factor()
            v = v * w if op == "*" else idiv(v, w)
        return v

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            op = eat()
            w = term()
            v = v + w if op == "+" else v - w
        return v

    v = expr_()
    if pos[0] != len(toks):
        raise ValueError("trailing")
    return v
'''),
        ("bad", '''def solve(expr):
    for op in "+-*/":
        expr = expr.replace(op, " " + op + " ")
    t = expr.split()
    v = int(t[0])
    for i in range(1, len(t) - 1, 2):
        op, w = t[i], int(t[i + 1])
        v = v + w if op == "+" else v - w if op == "-" else v * w if op == "*" else v // w
    return v
'''),
    ],
    "rpn_eval": [
        ("good", '''def solve(tokens):
    div = @@div@@
    st = []
    for t in tokens.split():
        if t in ("+", "-", "*", "/"):
            if len(st) < 2:
                raise ValueError("underflow")
            b = st.pop()
            a = st.pop()
            if t == "+":
                st.append(a + b)
            elif t == "-":
                st.append(a - b)
            elif t == "*":
                st.append(a * b)
            else:
                if b == 0:
                    raise ZeroDivisionError("div")
                if div == "floor":
                    st.append(a // b)
                else:
                    q = abs(a) // abs(b)
                    st.append(q if (a >= 0) == (b > 0) else -q)
        else:
            st.append(int(t))
    if len(st) != 1:
        raise ValueError("leftover")
    return st[0]
'''),
        ("bad", '''def solve(tokens):
    st = []
    for t in tokens.split(" "):
        if t in ("+", "-", "*", "/"):
            if len(st) < 2:
                raise ValueError("underflow")
            b = st.pop()
            a = st.pop()
            if t == "+":
                st.append(a + b)
            elif t == "-":
                st.append(a - b)
            elif t == "*":
                st.append(a * b)
            else:
                if b == 0:
                    raise ZeroDivisionError("div")
                st.append(a // b)
        else:
            st.append(int(t))
    if len(st) != 1:
        raise ValueError("leftover")
    return st[0]
'''),
    ],
}


def strategy_source(kind: str, quality: str, uid: str) -> str:
    """Python source of one registered strategy function (appended to agent/strategies.py)."""
    tpl = dict(LIB[kind])[quality]
    assert "'''" not in tpl
    return (f'@register("{kind}")\n'
            f"def s_{kind}_{quality}_{uid}(spec):\n"
            f"    return fill('''{tpl}''', spec[\"params\"])\n\n\n")


def reference_solution(kind: str, params: dict) -> str:
    """The GOOD template filled for `params` (used as the mutation-testing source)."""
    tpl = dict(LIB[kind])["good"]
    for k in sorted(params):
        tpl = tpl.replace(f"@@{k}@@", repr(params[k]))
    return tpl


# =============================================================== policy file

def policy_source(policy: dict) -> str:
    body = "".join(f'    "{k}": {v!r},\n' for k, v in policy.items())
    return f"# The improver's own policy. The improver may edit this file; patches to it are judged by the yield test.\nPOLICY = {{\n{body}}}\n"


def policy_from_source(text: str) -> dict:
    """Parse POLICY from policy.py source with literal_eval only (never exec). Unknown/bad values fall back to M0."""
    out = dict(REFERENCE_POLICY)
    try:
        for n in ast.parse(text).body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "POLICY":
                raw = ast.literal_eval(n.value)
                for k, (lo, hi) in TUNABLE.items():
                    if isinstance(raw.get(k), (int, float)):
                        out[k] = min(hi, max(lo, float(raw[k])))
    except (SyntaxError, ValueError, AttributeError):
        pass
    return out


def read_policy(worktree: Path) -> dict:
    try:
        return policy_from_source((Path(worktree) / POLICY_PATH).read_text())
    except OSError:
        return dict(REFERENCE_POLICY)


def policy_hash(policy: dict) -> str:
    return hashlib.sha256(repr(sorted(policy.items())).encode()).hexdigest()[:12]


# =============================================================== operators

SWAPS = [("<=", "<"), ("<", "<="), (">=", ">"), (">", ">="), (" + 1", " + 0"), (" - 1", " - 0"),
         ("== ", "!= "), ("min(", "max("), ("[::-1]", ""), (" % ", " // "), ("and", "or"), ("not st or", "st and")]


def _registered(src: str) -> list:
    """[(kind, func_name, first_line, last_line)] for every @register(...) function."""
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "register" and d.args:
                    out.append((d.args[0].value, n.name, min(x.lineno for x in n.decorator_list), n.end_lineno))
    return out


def _segment(src: str, a: int, b: int) -> str:
    lines = src.splitlines(keepends=True)
    return "".join(lines[a - 1:b])


class MutationMetaAgent(MetaAgent):
    """Seeded improver. policy_mode: 'frozen' (M0, never reads the repo policy) | 'repo' (reads
    agent/meta/policy.py from the PARENT worktree: the recursive arm). allow_meta=False forbids
    meta.tune proposals (inner yield-test loops and arm A0)."""

    def __init__(self, seed: int = 0, policy_mode: str = "frozen", allow_meta: bool = False,
                 policy_override: Optional[dict] = None):
        self.seed, self.mode, self.allow_meta, self.override = seed, policy_mode, allow_meta, policy_override
        self._n, self._lock = 0, threading.Lock()
        self.tokens_per_proposal = 800

    def policy(self, worktree: Path) -> dict:
        if self.override is not None:
            return {**REFERENCE_POLICY, **self.override}
        return read_policy(worktree) if self.mode == "repo" else dict(REFERENCE_POLICY)

    def _rng(self, parent_sha: str) -> random.Random:
        with self._lock:
            self._n += 1
            n = self._n
        return random.Random(int(hashlib.sha256(f"{self.seed}|{parent_sha}|{n}".encode()).hexdigest()[:12], 16))

    def propose(self, worktree: Path, parent_sha: str, broadcast: list, budget: Budget) -> PatchProposal:
        budget.charge(self.tokens_per_proposal)
        rng, pol = self._rng(parent_sha), self.policy(worktree)
        ema = next((e["payload"] for e in broadcast if e.get("kind") == "operator_stats"), {})
        pid = f"m{rng.getrandbits(24):06x}"
        if self.allow_meta and rng.random() < pol["p_meta"]:
            prop = self._meta(worktree, pol, rng, pid)
            if prop:
                return prop
        ops = {"adopt": pol["w_adopt"], "prune": pol["w_prune"], "mutate": pol["w_mutate"]}
        if pol["use_bandit"] >= 0.5:
            ops = {k: w * (0.25 + float(ema.get(k, 0.3))) for k, w in ops.items()}
        order = []
        pool = dict(ops)
        while pool:
            k = rng.choices(list(pool), weights=list(pool.values()))[0]
            order.append(k)
            pool.pop(k)
        for name in order:
            prop = getattr(self, "_" + name)(worktree, rng, pid, ema)
            if prop:
                return prop
        return self._adopt(worktree, rng, pid, ema, force=True)

    # ------------------------------------------------------------------ operators
    def _adopt(self, wt, rng, pid, ema, force=False):
        kind = rng.choice(SELECTION_KINDS)
        quality = rng.choice([q for q, _ in LIB[kind]])
        uid = f"{rng.getrandbits(20):05x}"
        new = strategy_source(kind, quality, uid)
        tag = f"adopt.{kind}"
        return PatchProposal(pid, "", f"Register a library strategy for {kind}.",
                             [EditOp(STRATEGIES_PATH, "replace_text", old_text=END_MARK, new_text=new + END_MARK, tag=tag)],
                             {"kind": kind})

    def _prune(self, wt, rng, pid, ema):
        src = (Path(wt) / STRATEGIES_PATH).read_text()
        regs = _registered(src)
        dead = [r for i, r in enumerate(regs) if any(o[0] == r[0] for o in regs[i + 1:])]
        if not dead:
            return None
        kind, name, a, b = rng.choice(dead)
        seg = _segment(src, a, b)
        return PatchProposal(pid, "", f"Remove the shadowed strategy {name} (dead code).",
                             [EditOp(STRATEGIES_PATH, "replace_text", old_text=seg + "\n\n", new_text="", tag="prune")])

    def _mutate(self, wt, rng, pid, ema):
        src = (Path(wt) / STRATEGIES_PATH).read_text()
        regs = _registered(src)
        if not regs:
            return None
        kinds = sorted({r[0] for r in regs})
        kind = rng.choice(kinds)
        _, name, a, b = [r for r in regs if r[0] == kind][-1]
        seg = _segment(src, a, b)
        cands = [(o, n) for o, n in SWAPS if seg.count(o) >= 1 and src.count(o) == 1]
        if not cands:
            cands = [(o, n) for o, n in SWAPS if seg.count(o) == 1 and src.count(o) >= 1]
        # exact-once anchor: widen the swap with its line
        rng.shuffle(cands)
        for o, n in cands:
            for line in seg.splitlines():
                if o in line and src.count(line) == 1:
                    return PatchProposal(pid, "", f"Perturb an operator in {name}.",
                                         [EditOp(STRATEGIES_PATH, "replace_text", old_text=line, new_text=line.replace(o, n, 1), tag="mutate")])
        return None

    def _meta(self, wt, pol, rng, pid):
        src = (Path(wt) / POLICY_PATH).read_text() if (Path(wt) / POLICY_PATH).exists() else ""
        key = rng.choice(sorted(TUNABLE))
        lo, hi = TUNABLE[key]
        cur = pol[key]
        new = (1.0 - cur) if key == "use_bandit" else min(hi, max(lo, cur * rng.choice([0.5, 0.7, 1.4, 2.0])))
        new = round(new, 3)
        old_line = next((ln for ln in src.splitlines() if ln.strip().startswith(f'"{key}":')), None)
        if old_line is None or new == cur:
            return None
        return PatchProposal(pid, "", f"Tune improver policy {key}: {cur} -> {new}.",
                             [EditOp(POLICY_PATH, "replace_text", old_text=old_line,
                                     new_text=f'    "{key}": {new!r},', tag="meta.tune")])


class RandomPatchMeta(MetaAgent):
    """A2 control: random, knowledge-free edits of agent code (no library, no policy, no bandit).
    Same validator, same gate. Measures what 'random search + the gate' achieves on its own."""

    def __init__(self, seed: int = 0):
        self.seed, self._n, self._lock = seed, 0, threading.Lock()

    def propose(self, worktree: Path, parent_sha: str, broadcast: list, budget: Budget) -> PatchProposal:
        budget.charge(800)
        with self._lock:
            self._n += 1
            n = self._n
        rng = random.Random(int(hashlib.sha256(f"R|{self.seed}|{parent_sha}|{n}".encode()).hexdigest()[:12], 16))
        pid = f"r{rng.getrandbits(24):06x}"
        src = (Path(worktree) / STRATEGIES_PATH).read_text()
        regs = _registered(src)
        choice = rng.random()
        if choice < 0.35 and regs:                                   # delete a random strategy
            _, name, a, b = rng.choice(regs)
            return PatchProposal(pid, "", f"Delete {name}.",
                                 [EditOp(STRATEGIES_PATH, "replace_text", old_text=_segment(src, a, b) + "\n\n", new_text="", tag="rand.delete")])
        if choice < 0.8:                                             # random operator swap on a random line
            lines = [ln for ln in src.splitlines() if any(o in ln for o, _ in SWAPS) and src.count(ln) == 1]
            if lines:
                ln = rng.choice(lines)
                o, n_ = rng.choice([(o, n) for o, n in SWAPS if o in ln])
                return PatchProposal(pid, "", "Random operator swap.",
                                     [EditOp(STRATEGIES_PATH, "replace_text", old_text=ln, new_text=ln.replace(o, n_, 1), tag="rand.swap")])
        return PatchProposal(pid, "", "Add a comment.",
                             [EditOp(STRATEGIES_PATH, "replace_text", old_text=END_MARK,
                                     new_text=f"# note {rng.getrandbits(20):05x}\n{END_MARK}", tag="rand.comment")])
