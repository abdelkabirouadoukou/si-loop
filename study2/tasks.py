"""study2/tasks.py: the local problem suite. Oracles live ONLY on the judge side.

Families
    string       selection / confirm / credit / anchor / yield / final splits
    algorithmic  selection / confirm / credit / anchor / yield / final splits
    parsing      HELD-OUT family: drawn only by the `heldout` split, never by selection

Every task has: a spec (public docstring, shown to the agent inside agent/lib.py), an oracle, and a
case generator that mixes ordinary and edge-case inputs. All arguments and results are JSON values.
"""
from __future__ import annotations

import bisect
import random
import string
from dataclasses import dataclass
from typing import Callable

SELECTION_FAMILIES = ("string", "algorithmic")
HELDOUT_FAMILY = "parsing"


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    oracle: Callable
    gen: Callable            # (rng, edge: bool) -> list of args


def _word(r, lo=1, hi=7, alpha=string.ascii_letters):
    return "".join(r.choice(alpha) for _ in range(r.randint(lo, hi)))


def _ws(r):
    return r.choice([" ", " ", " ", "  ", "   ", "\t", " \n"])


def _sentence(r, edge):
    words = [_word(r) for _ in range(r.randint(0 if edge else 1, 6))]
    s = "".join(w + _ws(r) if edge else w + " " for w in words)
    return (r.choice(["", " ", "  "]) + s.rstrip(" ") + r.choice(["", "  "])) if edge else s.strip()


# ---------------------------------------------------------------- string
def reverse_words(s):
    return " ".join(reversed(s.split()))


def run_length_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]


def longest_common_prefix(strs):
    if not strs:
        return ""
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p):
            p = p[:-1]
    return p


def count_vowels(s):
    return sum(c in "aeiouAEIOU" for c in s)


def title_case(s):
    return " ".join(w.capitalize() for w in s.split(" "))


def _g_rle(r, edge):
    runs = [(r.choice("abcxyz"), r.randint(1, 5)) for _ in range(r.randint(0 if edge else 1, 6))]
    return ["".join(c * n for c, n in runs)]


def _g_pal(r, edge):
    half = _word(r, 0 if edge else 1, 5)
    mid = r.choice(["", "x", " ", ","])
    s = half + mid + (half[::-1] if r.random() < .6 else _word(r, 0, 5))
    return [r.choice([s, s.upper(), s.replace("a", " a,")])]


def _g_lcp(r, edge):
    base = _word(r, 0, 5, "ab")
    n = r.randint(0 if edge else 1, 5)
    return [[base + _word(r, 0, 3, "abc") for _ in range(n)]]


def _g_title(r, edge):
    ws = [r.choice([_word(r, 1, 6, string.ascii_lowercase), _word(r, 1, 5).upper(), "don't", "o'neil", "x"])
          for _ in range(r.randint(0 if edge else 1, 5))]
    return [(r.choice(["  ", " "]) if edge and r.random() < .4 else " ").join(ws)]


# ---------------------------------------------------------------- algorithmic
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def max_subarray(nums):
    if not nums:
        return 0
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def insert_position(xs, x):
    return bisect.bisect_left(xs, x)


def flatten(x):
    if not isinstance(x, list):
        return [x]
    out = []
    for y in x:
        out += flatten(y)
    return out


def rotate_list(xs, k):
    if not xs:
        return []
    k %= len(xs)
    return xs[-k:] + xs[:-k] if k else list(xs)


def _g_two(r, edge):
    xs = [r.randint(-9, 9) for _ in range(r.randint(0 if edge else 2, 8))]
    t = (xs[r.randrange(len(xs))] + xs[r.randrange(len(xs))]) if xs and r.random() < .75 else r.randint(-30, 30)
    return [xs, t]


def _g_iv(r, edge):
    iv = []
    for _ in range(r.randint(0 if edge else 1, 6)):
        a = r.randint(0, 20)
        iv.append([a, a + r.randint(0, 6)])
    r.shuffle(iv)
    return [iv]


def _g_sub(r, edge):
    neg = edge and r.random() < .4
    return [[(-r.randint(1, 9) if neg else r.randint(-9, 9)) for _ in range(r.randint(0 if edge else 1, 9))]]


def _g_ins(r, edge):
    xs = sorted(r.randint(0, 12) for _ in range(r.randint(0, 8)))
    return [xs, r.randint(-1, 13)]


def _g_flat(r, edge, depth=0):
    def mk(d):
        return [(mk(d + 1) if d < 3 and r.random() < .35 else r.randint(0, 9)) for _ in range(r.randint(0, 4))]
    return [mk(0)]


def _g_rot(r, edge):
    xs = [r.randint(0, 9) for _ in range(r.randint(0 if edge else 1, 7))]
    return [xs, r.choice([0, 1, 2, 3, 7, 10, -1, -3, 25])]


# ---------------------------------------------------------------- parsing (held-out)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s):
    t = 0
    for i, c in enumerate(s):
        v = _ROMAN[c]
        t += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return t


def _to_roman(n):
    out = ""
    for v, sym in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                   (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out, n = out + sym, n - v
    return out


def parse_kv(s):
    d = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        d[k.strip()] = v.strip()
    return d


def caesar_decode(s, k):
    out = []
    for c in s:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base - k) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


def _g_rom(r, edge):
    return [_to_roman(r.choice([4, 9, 14, 40, 90, 400, 1994, 3999]) if r.random() < .5 else r.randint(1, 3999))]


def _g_kv(r, edge):
    parts = [f"{_word(r, 1, 4, string.ascii_lowercase)}{r.choice(['=', ' = '])}{_word(r, 0, 4)}"
             for _ in range(r.randint(0 if edge else 1, 4))]
    return [r.choice([";", "; ", " ;"]).join(parts) + r.choice(["", ";"])]


def _g_cae(r, edge):
    s = " ".join(_word(r, 1, 6) for _ in range(r.randint(0 if edge else 1, 3))) + r.choice(["", "!", ", 1"])
    return [s, r.choice([0, 1, 3, 13, 25, 27, -2])]


def _simple(f):
    return lambda r, edge: [f(r, 1, 6) if False else _sentence(r, edge)]


TASKS: dict = {t.name: t for t in [
    Task("reverse_words", "string", reverse_words, lambda r, e: [_sentence(r, e)]),
    Task("run_length_encode", "string", run_length_encode, _g_rle),
    Task("is_palindrome", "string", is_palindrome, _g_pal),
    Task("longest_common_prefix", "string", longest_common_prefix, _g_lcp),
    Task("count_vowels", "string", count_vowels, lambda r, e: [_sentence(r, e) + r.choice(["", "AEIOU"])]),
    Task("title_case", "string", title_case, _g_title),
    Task("two_sum", "algorithmic", two_sum, _g_two),
    Task("merge_intervals", "algorithmic", merge_intervals, _g_iv),
    Task("max_subarray", "algorithmic", max_subarray, _g_sub),
    Task("insert_position", "algorithmic", insert_position, _g_ins),
    Task("flatten", "algorithmic", flatten, _g_flat),
    Task("rotate_list", "algorithmic", rotate_list, _g_rot),
    Task("roman_to_int", "parsing", roman_to_int, _g_rom),
    Task("parse_kv", "parsing", parse_kv, _g_kv),
    Task("caesar_decode", "parsing", caesar_decode, _g_cae),
]}


def families_for(split: str) -> tuple:
    return (HELDOUT_FAMILY,) if split == "heldout" else SELECTION_FAMILIES


def tasks_for(split: str) -> list:
    fams = families_for(split)
    return [t for t in TASKS.values() if t.family in fams]


def make_cases(task: Task, seed: int, n: int = 6) -> list:
    """n argument lists; the first half are edge-biased. Deterministic in (task, seed)."""
    r = random.Random(seed)
    return [task.gen(r, i < n // 2) for i in range(n)]


# ---------------------------------------------------------------- suite v2 (Study 3 replicates only)
# v1 (make_cases above) is what Study 2 used and is NOT changed. v2 adds targeted cases where the
# docstring's tie-break rules matter. Found by auditing the Study 3 seed-3 patches (see DEVIATIONS.md, D2).
def _g_two_ties(r, edge):
    """Small values with forced duplicates, so several pairs can match and the tie-break rule decides."""
    xs = [r.randint(0, 3) for _ in range(r.randint(4, 8))]
    a, b = sorted(r.sample(range(len(xs)), 2))
    return [xs, xs[a] + xs[b]]


EXTRA_V2 = {"two_sum": _g_two_ties}
EXTRA_V2_PER_TASK = 12   # catches the two_sum tie-break bug on ~71% of problems (v1: ~14%)


def make_cases_v2(task: Task, seed: int, n: int = 12) -> list:
    """Like make_cases, but with n cases total, and EXTRA_V2_PER_TASK targeted cases appended when the
    task has one. Deterministic in (task, seed). Study 2 never calls this."""
    base = make_cases(task, seed, n)
    gen = EXTRA_V2.get(task.name)
    if gen is None:
        return base
    r = random.Random(f"v2|{seed}")
    return base + [gen(r, False) for _ in range(EXTRA_V2_PER_TASK)]
