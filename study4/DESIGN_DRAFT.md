# Study 4 design draft (not locked)

**This is a draft for discussion. It is not a pre-registration, and nothing here is binding until a locked
`results/study4/prereg.json` exists (see `README.md`).**

## 1. Question

Can the loop improve the code that proposes and judges its own changes, in a way that holds up on problems the
improver never saw?

Primary question: does the live loop produce at least one accepted improver patch (scope `meta`) that passes the
yield test on the parent, without reward hacking?

## 2. Why this is harder than Study 3

A patch to `agent/meta/**` changes how later patches are proposed and judged. A bad improver patch can therefore
corrupt every later cycle, and it can make the judge look better than it is. The kernel already has two guards:

- the yield test (`kernel/yield_test.py`, `YieldGate.judge`), applied to any patch with scope `meta`;
- the mixed-scope rule (`kernel/orchestrator.py`), under which one patch may touch the solver or the meta code,
  but not both.

Neither guard has been exercised on a live model.

## 3. Arms (proposed)

| Arm | Description | Role |
|---|---|---|
| A1 | Self-improvement: the proposer may edit `agent/meta/**` and `agent/lib.py`, subject to the yield test and mixed-scope rule | treatment |
| A3 | Compute-matched best-of-N from the anchor, no improver changes | control for compute |
| A0 | Solver-only, as in Study 3, rerun on the Study 4 judge | baseline |

Whether A1 and A3 are both run is an open decision (section 6).

## 4. Metric (proposed, to be fixed before locking)

- Primary: count of accepted `meta`-scope patches that pass `YieldGate` and do not reduce held-out performance.
- Secondary: held-out gain on the parsing family, as in Study 3, reported with exact intervals.
- Integrity: ledger verifies, no security flags, no tripped breakers, and the monitor's flags reported for every run.

## 5. Preconditions

1. A seed repository with `agent/meta/` (none exists yet).
2. A written yield definition that `YieldGate` is checked against.
3. The judge-coverage statement from Study 3 (section 5.4), updated.
4. A locked pre-registration committed before any live run.

## 6. Open decisions for the researcher

1. **Seed repository.** What the `agent/meta/` package contains at the start. A minimal package keeps the
   improver's surface small, but a trivial one may make improvement easy to find by accident.
2. **Yield definition.** The exact measurement. Study 3 ran on pass rates, so the proposed default is the same
   paired gate applied to yield, but this has not been decided.
3. **Monitor enforcement.** Whether the false-discovery monitor blocks in Study 4. Study 3 ran it in monitor mode.
4. **Arms.** Run A1 and A3 together, or A1 alone first as a pilot. A1 alone cannot separate improver effects from
   added compute.
5. **Stopping rules.** The number of cycles and runs, fixed in advance.
6. **Target metric threshold.** What counts as "at least one" for the primary question, and whether a single
   accepted improver patch is enough to call the question answered.

## 7. Known risks

- A yield gate that is too lenient lets a harmful improver change through. One that is too strict blocks every change.
- The improver may learn to edit the yield test itself. The kernel digest and the mixed-scope rule are meant to stop
  this, but neither has been tested against a live model.
- Judge defects found in Study 3 (D1, D2) must not recur in the improver judge. Each judge rule needs a test.
