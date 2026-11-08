#!/usr/bin/env python3
"""results/study3/suite_detection.py: how often does each judge suite catch the two_sum tie-break defect (D2)?

The defect: a one-pass hash-map solution that returns the first pair whose second index is smallest,
not the pair with the smallest first index (the docstring's rule). Example from seed 3: [1, 3, 3, 5], target 6
returns [1, 2]; the correct answer is [0, 3].

For N random problems (task seed drawn from a fixed RNG), a problem counts as detected when the buggy
solution disagrees with the oracle on at least one case of that problem's suite. Suite v1 uses 6 cases per
problem (the Study 2 suite). Suite v2 (Study 3) uses 12 generated cases, and for two_sum it adds 12 targeted
tie-break cases, so a two_sum problem has 24 cases. The verifier builds the same cases (cases_per_problem=12).

    python results/study3/suite_detection.py            # writes results/study3/suite_detection.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from study2.tasks import TASKS, make_cases, make_cases_v2  # noqa: E402

OUT = Path(__file__).resolve().parent / "suite_detection.json"
N_PROBLEMS = 2000
SEED = 20261108


def loop_by_second(nums, target):
    """Nested loop ordered by the SECOND index: returns the pair with the smallest second index.
    On [1, 3, 3, 5] with target 6 it returns [1, 2]; the docstring requires [0, 3]. Matches the seed 3 failure."""
    for j in range(len(nums)):
        for i in range(j):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def hash_map_one_pass(nums, target):
    """One-pass hash map: returns (index of earlier complement, current index). Also wrong on [1, 3, 3, 5]."""
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], j]
        seen[x] = j
    return []


VARIANTS = {"loop_by_second": loop_by_second, "hash_map_one_pass": hash_map_one_pass}


def detection_rate(variant: str, suite: str, k: int, n: int = N_PROBLEMS) -> dict:
    task = TASKS["two_sum"]
    bug = VARIANTS[variant]
    rng = random.Random(SEED)
    caught = 0
    for _ in range(n):
        seed = rng.getrandbits(48)
        cases = make_cases_v2(task, seed, k) if suite == "v2" else make_cases(task, seed, k)
        if any(bug(*a) != task.oracle(*a) for a in cases):
            caught += 1
    return {"variant": variant, "suite": suite, "cases_per_problem": k, "problems": n, "caught": caught,
            "rate": round(caught / n, 4)}


def main() -> int:
    rows = []
    for variant in VARIANTS:
        for suite, k in (("v1", 6), ("v2", 12)):
            r = detection_rate(variant, suite, k)
            rows.append(r)
            print(f"{variant:18s} suite {suite} ({k:2d} cases): caught {r['caught']:4d} of {r['problems']} "
                  f"= {100 * r['rate']:.1f}%")
    OUT.write_text(json.dumps({"defect": "two_sum tie-break (D2)", "seed": SEED, "rows": rows}, indent=1) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
