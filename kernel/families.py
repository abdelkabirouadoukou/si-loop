"""kernel/families.py: SEALED problem families (judge side only; never importable by the agent).

Each task KIND provides, for one parameter draw:
  reference    executable oracle (the ONLY source of truth for expected outputs)
  small        TIER 1  small-scope exhaustive (or shortlex-capped) cases: complete inside the scope
  fuzz         TIER 2  seeded random large-scope cases
  meta         TIER 3  metamorphic pairs (case_a, case_b, relation): checks the CANDIDATE's own
               consistency (permutation invariance, shifting by the modulus, wrapping in
               parentheses ...). Needs no oracle, so hard-coded answer tables break on it.

Families:  algorithmic, datastructure  -> used by selection / confirm / credit / anchor splits
           interpreter                 -> HELD-OUT audit family, never drawn by those splits

Result encoding (both oracle and executor): {"v": value} | {"e": "ExceptionName"}; a session case
(stateful API) is a list of such results, one per operation. Only JSON values are used.
"""
from __future__ import annotations

import itertools
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

SELECTION_FAMILIES = ("algorithmic", "datastructure")
HELDOUT_FAMILY = "interpreter"


def norm(x: Any) -> Any:
    return json.loads(json.dumps(x))


def enc_call(fn: Callable, args: list) -> dict:
    try:
        return {"v": norm(fn(*args))}
    except Exception as e:                                   # noqa: BLE001 (oracle: any exception is an outcome)
        return {"e": type(e).__name__}


def shortlex(alphabet: list, maxlen: int, cap: int, rng: random.Random) -> list:
    """All sequences by increasing length while they fit in `cap`; the next length is randomly
    subsampled to fill the cap. Deterministic given rng."""
    out: list = [[]]
    for L in range(1, maxlen + 1):
        total = len(alphabet) ** L
        if len(out) + total <= cap:
            out += [list(t) for t in itertools.product(alphabet, repeat=L)]
        else:
            for idx in rng.sample(range(total), max(0, cap - len(out))):
                seq, v = [], idx
                for _ in range(L):
                    v, r = divmod(v, len(alphabet))
                    seq.append(alphabet[r])
                out.append(seq)
            break
    return out


@dataclass
class Kind:
    name: str
    family: str
    entry: str                     # function name ("solve") or class name ("Solution")
    mode: str                      # "call" | "session"
    signature: str
    make_params: Callable
    statement: Callable
    reference: Callable            # call: (params, case) -> encoded ; session: (params, case) -> [encoded]
    small: Callable
    fuzz: Callable
    meta: Callable
    relate: Callable               # (params, rel, case_a, case_b, out_a, out_b) -> bool


KINDS: dict = {}


def register(k: Kind) -> Kind:
    KINDS[k.name] = k
    return k


def _eq_relate(params, rel, ca, cb, oa, ob):
    return oa == ob


def _call_ref(fn):
    return lambda params, case: enc_call(lambda *a: fn(params, *a), case)


# =========================================================================== algorithmic

def _dedupe(p, xs):
    seen, out = set(), []
    for x in (xs if p["keep"] == "first" else reversed(xs)):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out if p["keep"] == "first" else out[::-1]


register(Kind(
    "dedupe", "algorithmic", "solve", "call", "solve(xs: list[int]) -> list[int]",
    lambda r: {"keep": r.choice(["first", "last"])},
    lambda p: f"Return xs with duplicate values removed, keeping the {p['keep']} occurrence of every value; "
              "kept items stay in their original relative order.",
    _call_ref(_dedupe),
    lambda p, r: [[xs] for xs in shortlex([0, 1, 2, 3], 4, 400, r)],
    lambda p, r, n: [[[r.randint(0, 8) for _ in range(r.randint(0, 40))]] for _ in range(n)]
    + [[[r.randint(0, 300) for _ in range(1500)]]],
    lambda p, r: [([xs], [xs + xs], "dup") for xs in ([r.randint(0, 5) for _ in range(r.randint(0, 12))] for _ in range(8))],
    _eq_relate))


def _rotate(p, xs, k):
    if not xs:
        return []
    kk = k % len(xs)
    if p["dir"] == "left":
        return xs[kk:] + xs[:kk]
    return xs[len(xs) - kk:] + xs[:len(xs) - kk] if kk else xs[:]


def _rot_meta(p, r):
    out = []
    for _ in range(8):
        xs = [r.randint(0, 9) for _ in range(r.randint(1, 9))]
        k = r.randint(-12, 12)
        out.append(([xs, k], [xs, k + len(xs) * r.choice([-2, -1, 1, 2])], "period"))
    return out


register(Kind(
    "rotate", "algorithmic", "solve", "call", "solve(xs: list[int], k: int) -> list[int]",
    lambda r: {"dir": r.choice(["left", "right"])},
    lambda p: f"Rotate xs by k positions to the {p['dir']} (k may be negative or larger than len(xs)); "
              "an empty list stays empty. Return a new list.",
    _call_ref(_rotate),
    lambda p, r: [[xs, k] for xs in shortlex([0, 1, 2], 3, 40, r) for k in range(-4, 6)],
    lambda p, r, n: [[[r.randint(0, 50) for _ in range(r.randint(0, 30))], r.randint(-100, 100)] for _ in range(n)],
    _rot_meta, _eq_relate))


def _prefix_mod(p, xs):
    out, s = [], 0
    for x in xs:
        s = (s + x) % p["m"]
        out.append(s)
    return out


register(Kind(
    "prefix_mod", "algorithmic", "solve", "call", "solve(xs: list[int]) -> list[int]",
    lambda r: {"m": r.randint(2, 11)},
    lambda p: f"Return the running prefix sums of xs, each reduced modulo {p['m']} (result in 0..{p['m'] - 1}, "
              "also for negative inputs).",
    _call_ref(_prefix_mod),
    lambda p, r: [[xs] for xs in shortlex([-2, -1, 0, 1, 2, 3], 3, 300, r)],
    lambda p, r, n: [[[r.randint(-10 ** 6, 10 ** 6) for _ in range(r.randint(0, 60))]] for _ in range(n)],
    lambda p, r: [([xs], [[x + p["m"] * r.randint(-3, 3) for x in xs]], "shift")
                  for xs in ([r.randint(-9, 9) for _ in range(r.randint(0, 10))] for _ in range(8))],
    _eq_relate))


def _kth(p, xs, k):
    ys = sorted(set(xs)) if p["distinct"] else sorted(xs)
    return ys[k - 1] if 1 <= k <= len(ys) else None


def _kth_meta(p, r):
    out = []
    for _ in range(8):
        xs = [r.randint(0, 6) for _ in range(r.randint(1, 10))]
        ys = xs[:]
        r.shuffle(ys)
        k = r.randint(1, len(xs))
        out.append(([xs, k], [ys, k], "perm"))
    return out


register(Kind(
    "kth_smallest", "algorithmic", "solve", "call", "solve(xs: list[int], k: int) -> int | None",
    lambda r: {"distinct": r.choice([True, False])},
    lambda p: "Return the k-th smallest (1-indexed) of "
              + ("the DISTINCT values of xs" if p["distinct"] else "xs, counting repeated values separately")
              + "; return None if k is out of range.",
    _call_ref(_kth),
    lambda p, r: [[xs, k] for xs in shortlex([0, 1, 2], 3, 40, r) for k in range(0, 5)],
    lambda p, r, n: [[[r.randint(0, 30) for _ in range(r.randint(0, 40))], r.randint(0, 45)] for _ in range(n)],
    _kth_meta, _eq_relate))


def _balanced(p, s):
    pairs = {p["alpha"][i + 1]: p["alpha"][i] for i in range(0, len(p["alpha"]), 2)}
    st = []
    for ch in s:
        if ch in pairs.values():
            st.append(ch)
        elif not st or st.pop() != pairs[ch]:
            return False
    return not st


def _rand_balanced(alpha, r, pairs):
    s, st = [], []
    for _ in range(pairs * 2):
        if st and (r.random() < 0.5 or len(st) >= pairs):
            s.append(st.pop())
        else:
            i = r.randrange(0, len(alpha), 2)
            s.append(alpha[i])
            st.append(alpha[i + 1])
    s += st[::-1]
    return "".join(s)


register(Kind(
    "bracket_balance", "algorithmic", "solve", "call", "solve(s: str) -> bool",
    lambda r: {"alpha": r.choice(["()", "()[]", "()[]{}"])},
    lambda p: f"s only contains the characters {p['alpha']!r} (pairs: "
              + ", ".join(p["alpha"][i:i + 2] for i in range(0, len(p["alpha"]), 2))
              + "). Return True iff the brackets are properly nested and every one is closed.",
    _call_ref(_balanced),
    lambda p, r: [["".join(s)] for s in shortlex(list(p["alpha"]), 4, 500, r)],
    lambda p, r, n: [[_rand_balanced(p["alpha"], r, r.randint(0, 40))] if i % 2 == 0 else
                     [_mutate_str(_rand_balanced(p["alpha"], r, r.randint(1, 40)), p["alpha"], r)] for i in range(n)],
    lambda p, r: [([s], [s + p["alpha"][0] + p["alpha"][1]], "append")
                  for s in (_rand_balanced(p["alpha"], r, r.randint(0, 6)) if i % 2 else
                            "".join(r.choice(p["alpha"]) for _ in range(r.randint(0, 6))) for i in range(10))],
    _eq_relate))


def _mutate_str(s: str, alpha: str, r: random.Random) -> str:
    if not s:
        return alpha[0]
    i = r.randrange(len(s))
    return s[:i] + r.choice(alpha) + s[i + 1:]


# =========================================================================== datastructure (sessions)

class _RefMinStack:
    def __init__(self, p):
        self.p, self.d = p, []

    def push(self, x):
        if self.p["cap"] is not None and len(self.d) >= self.p["cap"]:
            raise OverflowError
        self.d.append(x)

    def pop(self):
        if not self.d:
            raise IndexError
        return self.d.pop()

    def top(self):
        if not self.d:
            raise IndexError
        return self.d[-1]

    def get_min(self):
        if not self.d:
            raise IndexError
        return min(self.d)

    def size(self):
        return len(self.d)


class _RefLRU:
    def __init__(self, p, cap):
        self.p, self.cap, self.d = p, cap, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return self.p["miss"]

    def put(self, k, v):
        if k in self.d:
            self.d.move_to_end(k)
        self.d[k] = v
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


def _session_ref(factory):
    def run(p, case):
        obj = factory(p, case["ctor"])
        out = []
        for name, args in case["ops"]:
            try:
                out.append({"v": norm(getattr(obj, name)(*args))})
            except Exception as e:                           # noqa: BLE001
                out.append({"e": type(e).__name__})
        return out
    return run


def _ops_meta(query_ops, ctor_fn):
    """Insert one pure query op at a random position: outputs of every other op must not change."""
    def meta(p, r, gen):
        out = []
        for _ in range(8):
            ops = gen(p, r)
            i = r.randint(0, len(ops))
            ca = {"ctor": ctor_fn(p, r), "ops": ops}
            cb = {"ctor": ca["ctor"], "ops": ops[:i] + [r.choice(query_ops)] + ops[i:], "ins": i}
            out.append((ca, cb, "insert"))
        return out
    return meta


def _insert_relate(p, rel, ca, cb, oa, ob):
    return ob[:cb["ins"]] + ob[cb["ins"] + 1:] == oa


_MS_OPS = [["push", [0]], ["push", [1]], ["push", [2]], ["pop", []], ["top", []], ["get_min", []], ["size", []]]


def _ms_gen(p, r):
    return [r.choice(_MS_OPS[:3] * 2 + _MS_OPS[3:]) for _ in range(r.randint(3, 25))]


register(Kind(
    "min_stack", "datastructure", "Solution", "session",
    "class Solution: push(x), pop(), top(), get_min(), size()",
    lambda r: {"cap": r.choice([None, 4, 6])},
    lambda p: "Implement class Solution (no constructor arguments): a stack of ints with push(x), pop() -> top "
              "value, top() -> top value, get_min() -> smallest value currently stored, size() -> count. "
              "pop/top/get_min on an empty stack raise IndexError."
              + ("" if p["cap"] is None else f" push raises OverflowError when the stack already holds {p['cap']} items."),
    _session_ref(lambda p, ctor: _RefMinStack(p)),
    lambda p, r: [{"ctor": [], "ops": ops} for ops in shortlex(_MS_OPS[:6], 4, 600, r)],
    lambda p, r, n: [{"ctor": [], "ops": [r.choice(_MS_OPS) for _ in range(r.randint(10, 120))]} for _ in range(n)],
    lambda p, r: _ops_meta([_MS_OPS[4], _MS_OPS[5], _MS_OPS[6]], lambda *_: [])(p, r, _ms_gen),
    _insert_relate))


def _lru_gen(p, r):
    return [r.choice([["get", [r.randint(0, 5)]], ["put", [r.randint(0, 5), r.randint(0, 9)]]]) for _ in range(r.randint(3, 25))]


_LRU_OPS = [["get", [k]] for k in range(3)] + [["put", [k, v]] for k in range(3) for v in range(2)]

register(Kind(
    "lru_cache", "datastructure", "Solution", "session",
    "class Solution: __init__(capacity), get(key), put(key, value)",
    lambda r: {"miss": r.choice([-1, None])},
    lambda p: "Implement class Solution(capacity): an LRU cache. get(key) returns the value (and marks the key "
              f"most recently used) or {p['miss']!r} if absent; put(key, value) inserts or updates (marking the key "
              "most recently used) and evicts the least recently used key when over capacity; put returns None.",
    _session_ref(lambda p, ctor: _RefLRU(p, ctor[0])),
    lambda p, r: [{"ctor": [cap], "ops": ops} for cap in (1, 2) for ops in shortlex(_LRU_OPS, 3, 300, r)],
    lambda p, r, n: [{"ctor": [r.randint(1, 4)], "ops": [r.choice([["get", [r.randint(0, 9)]], ["put", [r.randint(0, 9), r.randint(0, 99)]]])
                                                         for _ in range(r.randint(10, 150))]} for _ in range(n)],
    lambda p, r: _ops_meta([["get", [99]]], lambda pp, rr: [rr.randint(1, 3)])(p, r, _lru_gen),
    _insert_relate))


# =========================================================================== interpreter (HELD-OUT)

def _idiv(p, a, b):
    if b == 0:
        raise ZeroDivisionError
    if p["div"] == "floor":
        return a // b
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _arith(p, expr):
    toks, pos = [], 0
    while pos < len(expr):
        ch = expr[pos]
        if ch.isspace():
            pos += 1
        elif ch.isdigit():
            j = pos
            while j < len(expr) and expr[j].isdigit():
                j += 1
            toks.append(int(expr[pos:j]))
            pos = j
        elif ch in "+-*/()":
            toks.append(ch)
            pos += 1
        else:
            raise ValueError("bad char")
    i = [0]

    def peek():
        return toks[i[0]] if i[0] < len(toks) else None

    def eat():
        i[0] += 1
        return toks[i[0] - 1]

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
        raise ValueError("unexpected token")

    def term():
        v = factor()
        while peek() in ("*", "/"):
            op = eat()
            w = factor()
            v = v * w if op == "*" else _idiv(p, v, w)
        return v

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            op = eat()
            w = term()
            v = v + w if op == "+" else v - w
        return v

    v = expr_()
    if i[0] != len(toks):
        raise ValueError("trailing tokens")
    return v


def _rand_expr(r, depth):
    if depth == 0 or r.random() < 0.25:
        return str(r.randint(0, 20))
    a, b, op = _rand_expr(r, depth - 1), _rand_expr(r, depth - 1), r.choice("+-*/")
    s = f"{a} {op} {b}"
    return f"({s})" if r.random() < 0.5 else s


def _arith_small(p, r):
    ds = "0123"
    flat = [f"{a}{o1}{b}{o2}{c}" for a in ds for b in ds for c in ds for o1 in "+-*/" for o2 in "+-*/"]
    par = [f"({a}{o1}{b}){o2}{c}" for a in ds for b in ds for c in ds for o1 in "+-*/" for o2 in "+-*/"]
    par2 = [f"{a}{o1}({b}{o2}{c})" for a in ds for b in ds for c in ds for o1 in "+-*/" for o2 in "+-*/"]
    pool = flat + par + par2
    return [[s] for s in r.sample(pool, 450)]


register(Kind(
    "arith_eval", "interpreter", "solve", "call", "solve(expr: str) -> int",
    lambda r: {"div": r.choice(["floor", "trunc"])},
    lambda p: "Evaluate an integer arithmetic expression with + - * / and parentheses, usual precedence, left "
              f"associativity, non-negative integer literals, optional spaces. '/' is integer division that "
              f"{'rounds toward negative infinity' if p['div'] == 'floor' else 'truncates toward zero'}. "
              "Division by zero raises ZeroDivisionError; malformed input raises ValueError.",
    _call_ref(_arith),
    _arith_small,
    lambda p, r, n: [[_rand_expr(r, r.randint(1, 4))] for _ in range(n)],
    lambda p, r: [([e], [f"({e})"], "wrap") for e in (_rand_expr(r, 3) for _ in range(5))]
    + [([e], [re.sub(r"([-+*/()])", r" \1 ", e)], "spaced") for e in ((_rand_expr(r, 2)).replace(" ", "") for _ in range(5))],
    _eq_relate))


def _rpn(p, s):
    st = []
    for t in s.split():
        if t in "+-*/" and len(t) == 1:
            if len(st) < 2:
                raise ValueError("underflow")
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b if t == "*" else _idiv(p, a, b))
        else:
            st.append(int(t))
    if len(st) != 1:
        raise ValueError("leftover")
    return st[0]


def _rand_rpn(r, depth):
    if depth == 0 or r.random() < 0.3:
        return [str(r.randint(0, 20))]
    return _rand_rpn(r, depth - 1) + _rand_rpn(r, depth - 1) + [r.choice("+-*/")]


register(Kind(
    "rpn_eval", "interpreter", "solve", "call", "solve(tokens: str) -> int",
    lambda r: {"div": r.choice(["floor", "trunc"])},
    lambda p: "Evaluate a reverse-polish expression: whitespace-separated tokens, non-negative integers and + - * /. "
              f"'/' {'rounds toward negative infinity' if p['div'] == 'floor' else 'truncates toward zero'}. "
              "Too few operands or leftover operands raise ValueError; division by zero raises ZeroDivisionError.",
    _call_ref(_rpn),
    lambda p, r: [[" ".join(t)] for t in shortlex(["1", "2", "3", "+", "-", "*", "/"], 5, 500, r)],
    lambda p, r, n: [[" ".join(_rand_rpn(r, r.randint(1, 4)))] if i % 4 else
                     [" ".join(r.choice(["1", "7", "+", "*", "-", "/"]) for _ in range(r.randint(1, 9)))] for i in range(n)],
    lambda p, r: [([s], [s.replace(" ", "   ")], "spaced") for s in (" ".join(_rand_rpn(r, 3)) for _ in range(8))],
    _eq_relate))


FAMILIES: dict = {}
for _k in KINDS.values():
    FAMILIES.setdefault(_k.family, []).append(_k.name)
