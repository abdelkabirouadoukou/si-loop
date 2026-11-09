"""Tests for study4/fdr_replay.py: making the enforced online false-discovery control (ELond)
continuous across separate run_study4.py --run process restarts, by replaying the ledger's own
history -- not a new persistence file, and no kernel/ edit (TwoStageJudge already accepts a
pre-built `elond`).
"""
import json

from kernel.stats import ELond
from study4.fdr_replay import attach_persistent_fdr_budget, replay_elond_state

from kernel.orchestrator import Archive, Config, GitRepo, KernelGuard, MomCenter, Orchestrator, Sandbox, ShapleyAttributor
from kernel.patching import EditOp, PatchProposal
from study2.verifier import PytestVerifier

from test_study4_integration import KERNEL_DIR, SECRET, FakeMetaAgent, build_study4_orch


def _build_orch_reusing_existing_repo(tmp_path, meta, **cfg_over) -> Orchestrator:
    """Mirrors build_study4_orch, but -- like the real study4/run_live.py: build() -- skips
    re-initialising agent_repo/.git if it already exists, as it will on a second 'process launch'
    against the same output directory."""
    repo_dir, runs_dir = tmp_path / "agent_repo", tmp_path / "runs"
    repo = GitRepo(repo_dir, runs_dir / "worktrees")
    ver = PytestVerifier(SECRET, Sandbox(Config(repo_dir=repo_dir, runs_dir=runs_dir)), cases_per_problem=4, suite="v2")
    cfg = Config(repo_dir=repo_dir, runs_dir=runs_dir, seed=0, families=("string", "algorithmic"),
                n_selection=8, n_confirm=8, n_credit=4, min_pairs=4, screen_chunks=1,
                eval_wall_s=60, cycle_wall_s=120, anchor_every=0, heldout_every=0, gc_every=0,
                selftest_cmd=("python", "-c", "pass"), **cfg_over)
    return Orchestrator(cfg, repo, Sandbox(cfg), ver, meta,
                        Archive(runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                        ShapleyAttributor(cfg, repo, ver), MomCenter(runs_dir / "mom.db"), KernelGuard(KERNEL_DIR))


def _write_ledger(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_empty_or_missing_ledger_gives_a_fresh_budget(tmp_path):
    e = replay_elond_state(tmp_path / "absent.jsonl", 0.05)
    assert (e.t, e.d) == (0, 0)

    p = tmp_path / "ledger.jsonl"
    _write_ledger(p, [{"ev": "begin", "cycle": 0}])
    e2 = replay_elond_state(p, 0.05)
    assert (e2.t, e2.d) == (0, 0)


def test_replay_matches_hand_rolled_elond_sequence(tmp_path):
    """Replaying a ledger of N correctness-path tests must land on exactly the same (t, d) --
    and therefore the same NEXT level() -- as running those N tests through a live ELond directly."""
    live = ELond(0.05)
    hits = [False, False, True, False]                 # fabricated outcomes, not recomputed e-values
    for h in hits:
        live.t += 1
        live.d += int(h)                                # exactly ELond.test()'s own update rule

    records = [{"ev": "begin", "cycle": i} for i in range(len(hits))]
    for i, h in enumerate(hits):
        records.append({"ev": "end", "cycle": i, "scope": "solver",
                        "detail": {"confirm": {"b": 4, "c": 0}, "fdr": {"hit": h, "level": 0.01, "mode": "enforce"}}})
    p = tmp_path / "ledger.jsonl"
    _write_ledger(p, records)

    replayed = replay_elond_state(p, 0.05)
    assert (replayed.t, replayed.d) == (live.t, live.d) == (4, 1)
    assert replayed.level() == live.level()


def test_non_fdr_cycles_and_torn_lines_are_skipped(tmp_path):
    records = [
        {"ev": "end", "cycle": 0, "scope": "solver", "detail": {}},                      # no fdr key: skip
        {"ev": "end", "cycle": 1, "scope": "meta", "detail": {"yield": {}}},              # yield, not fdr: skip
        {"ev": "end", "cycle": 2, "scope": "solver", "error_type": "selftest"},           # no detail at all: skip
        {"ev": "end", "cycle": 3, "scope": "solver", "detail": {"fdr": {"hit": True}}},   # counts
    ]
    p = tmp_path / "ledger.jsonl"
    _write_ledger(p, records)
    (p.open("a")).write('{"ev": "end", "broken json...')                                 # torn last line

    e = replay_elond_state(p, 0.05)
    assert (e.t, e.d) == (1, 1)


def test_attach_persistent_fdr_budget_survives_a_fresh_orchestrator(tmp_path):
    """The real scenario: build an Orchestrator, run one FDR-tested cycle (consuming budget),
    discard it (simulating the process exiting), then build a SECOND, completely fresh Orchestrator
    against the SAME runs dir -- attach_persistent_fdr_budget must make its ELond pick up where the
    first one left off, not restart at (t=0, d=0)."""
    proposal = PatchProposal("p1", "", "Fix reverse_words whitespace handling.", [
        EditOp("agent/lib.py", "replace_symbol", symbol="reverse_words",
              new_source='def reverse_words(s):\n    return " ".join(reversed(s.split()))\n', tag="fix.string"),
    ])
    orch1 = build_study4_orch(tmp_path, FakeMetaAgent([proposal]), fdr_mode="enforce")
    ledger_path = orch1.cfg.runs_dir / "ledger.jsonl"
    attach_persistent_fdr_budget(orch1, ledger_path)      # ledger empty: no-op, t=d=0
    assert (orch1.elond.t, orch1.elond.d) == (0, 0)
    orch1.run_cycle(0)                                     # may accept or reject; either way the
                                                            # ledger now has a cycle (fdr key present
                                                            # only if the confirm stage reached "correctness")
    all_recs = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    ended = [r for r in all_recs if r.get("ev") == "end"]
    assert ended, f"expected at least one end record, got events: {[r.get('ev') for r in all_recs]}"

    # A second, independent Orchestrator -- as if run_study4.py --run were launched again as a new process.
    orch2 = _build_orch_reusing_existing_repo(tmp_path, FakeMetaAgent([proposal]), fdr_mode="enforce")
    assert (orch2.elond.t, orch2.elond.d) == (0, 0), "a fresh Orchestrator always starts at (0, 0) on its own"
    restored = attach_persistent_fdr_budget(orch2, ledger_path)
    expected = replay_elond_state(ledger_path, orch2.cfg.fdr_alpha)
    assert (orch2.elond.t, orch2.elond.d) == (expected.t, expected.d)
    assert orch2.judge.elond is orch2.elond is restored, "orch.elond and orch.judge.elond must be the SAME object"
