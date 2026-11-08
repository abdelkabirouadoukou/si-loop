# Study 3: deviations from the pre-registration

## D1: utility-path gate fix (found after the pilot)

**Pilot:** `runs/study3_live` (3 cycles, `openai/gpt-oss-120b`, arm A0).

**What went wrong.** The gate accepts a non-harmful patch on its utility path if it is "cheaper".
"Cheaper" was defined as at least 5% fewer lines of code OR at least 10% fewer tokens. The reference
verifier reports `tokens = 0` for every candidate, so the token test `0 <= 0.90 * 0` was always true.
Cycle 0 (p_better 0.9375, below the 0.95 correctness threshold, code grew from 88 to 89 lines) was
accepted with the label "efficiency gain" that no measurement supported.

**Fix.** "Cheaper" now means only at least 5% fewer lines of code. The token clause is removed.
Regression tests: `tests/test_gate_fix.py`.

**Status of the pilot.** The pilot is kept unchanged as evidence of the bug. Its cycle-0 accept is
not counted as a result. Its cycle-1 accept (correctness path, p_better 0.996) is not affected.

**When this was decided.** After seeing the pilot result. This is a post-hoc change and is reported as such.

**Scope.** The same verifier is used by Study 2. Study 2's results were produced with the old gate.
Its paper text describes the utility path as "smaller or cheaper" and must be corrected. Study 2 is not
re-run by this deviation.

## D2: judge suite v2 for the replicates (found after seed 3)

**What happened.** Seed 3's cycle-2 patch changed `two_sum`. The docstring says to return the pair with the
smallest first index. The patch returns the first pair found while scanning, so `[1, 3, 3, 5]` with target 6
gives `[1, 2]` instead of `[0, 3]`. The judge's suite v1 caught this on about 14% of `two_sum` problems.

**Fix.** Suite v2 (`study2/tasks.py`, `make_cases_v2`) uses 12 cases per problem, plus 12 targeted cases for
`two_sum` with repeated values. Measured detection: about 71% of `two_sum` problems (v1: about 14%).
Suite v1, which Study 2 used, is unchanged. Tests: `tests/test_suite_v2.py`.

**Status.** This change was made after seeing seed 3, so it is a post-hoc change to the judge. Seeds 1 to 3
were judged by v1 and are reported separately. The replicates in the new pre-registration (seeds 4 to 9)
use v2.

**Limits.** v2 still misses about 29% of `two_sum` problems. Other docstring rules were not audited.
