"""study2/seed_repo.py: writes the cycle-0 agent. The agent is a small library of task functions.

About half of them are buggy or missing, so there is real headroom. The docstrings are the PUBLIC
specification (the meta-agent can read them). The hidden tests, oracles and inputs are never here.
"""
from __future__ import annotations

from pathlib import Path

from kernel.improver import REFERENCE_POLICY, policy_source

LIB = '''"""agent/lib.py: task library. TASKS maps task name -> function. Specs are in the docstrings."""


def reverse_words(s):
    """Reverse the ORDER of the whitespace-separated words; join with single spaces (no leading/trailing space)."""
    return " ".join(reversed(s.split(" ")))


def run_length_encode(s):
    """'aaabcc' -> 'a3b1c2'. Every run gets its count, including runs of length 1. '' -> ''."""
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + (str(j - i) if j - i > 1 else ""))
        i = j
    return "".join(out)


def is_palindrome(s):
    """True iff s reads the same backwards, looking only at letters/digits and ignoring case. '' is a palindrome."""
    return s == s[::-1]


def longest_common_prefix(strs):
    """Longest common prefix of a list of strings. [] -> ''."""
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p):
            p = p[:-1]
    return p


def count_vowels(s):
    """Number of vowels (a, e, i, o, u), case-insensitive."""
    return sum(c in "aeiouAEIOU" for c in s)


def title_case(s):
    """Capitalise the first letter of each space-separated word and lowercase the rest ("don't" -> "Don't")."""
    return s.title()


def two_sum(nums, target):
    """Indices [i, j] with i < j and nums[i] + nums[j] == target; smallest i first, then smallest j. [] if none."""
    raise NotImplementedError


def merge_intervals(iv):
    """Merge overlapping closed intervals [a, b] (touching ones, e.g. [1,2] and [2,3], merge). Return them sorted by start."""
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def max_subarray(nums):
    """Largest sum of a non-empty contiguous subarray. [] -> 0. For all-negative input it is the largest element."""
    best = cur = 0
    for x in nums:
        cur = max(0, cur + x)
        best = max(best, cur)
    return best


def insert_position(xs, x):
    """Index where x would be inserted into the sorted list xs, before any equal elements."""
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def flatten(x):
    """Flatten arbitrarily nested lists of ints into one flat list, in order."""
    raise NotImplementedError


def rotate_list(xs, k):
    """Rotate the list RIGHT by k places (k may be negative or larger than len(xs)). [] -> []."""
    if not xs:
        return []
    return xs[-k:] + xs[:-k]


def roman_to_int(s):
    """Convert a Roman numeral (I V X L C D M, with subtractive pairs such as IV, IX, XL) to an int."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(vals[c] for c in s)


def parse_kv(s):
    """Parse 'a=1; b = 2;' into {'a': '1', 'b': '2'}. Strip spaces around keys and values; ignore empty parts."""
    return dict(p.split("=") for p in s.split(";") if p)


def caesar_decode(s, k):
    """Undo a Caesar shift of k: move every letter k places BACK in the alphabet (wrapping, any integer k), keep case and non-letters."""
    out = []
    for c in s:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base - k) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


TASKS = {f.__name__: f for f in [
    reverse_words, run_length_encode, is_palindrome, longest_common_prefix, count_vowels, title_case,
    two_sum, merge_intervals, max_subarray, insert_position, flatten, rotate_list,
    roman_to_int, parse_kv, caesar_decode]}
'''

TESTS = '''from agent.lib import TASKS


def test_count_vowels_public_example():
    assert TASKS["count_vowels"]("Hello World") == 3


def test_insert_position_public_example():
    assert TASKS["insert_position"]([1, 3, 5], 4) == 2


def test_caesar_decode_public_example():
    assert TASKS["caesar_decode"]("Khoor", 3) == "Hello"


def test_every_task_is_registered():
    assert len(TASKS) >= 15
'''


def write_seed_agent(root: Path) -> None:
    root = Path(root)
    files = {"agent/__init__.py": "", "agent/lib.py": LIB, "agent/tests/__init__.py": "",
             "agent/tests/test_public.py": TESTS, ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n"}
    for rel, txt in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)


def write_seed_agent_study4(root: Path) -> None:
    """Study 4 seed: the ordinary A0 seed agent plus agent/meta/policy.py, which the improver
    may edit under arm A1. Content and format come from kernel/improver.py (policy_source,
    REFERENCE_POLICY) -- the exact schema kernel/yield_test.py's YieldGate already parses.
    Starts at the frozen reference M0."""
    write_seed_agent(root)
    root = Path(root)
    (root / "agent" / "meta").mkdir(parents=True, exist_ok=True)
    (root / "agent" / "meta" / "__init__.py").write_text("")
    (root / "agent" / "meta" / "policy.py").write_text(policy_source(REFERENCE_POLICY))
