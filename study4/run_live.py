"""study4/run_live.py: Study 4 live loop (pilot: arm A0 or A1). Imported and called by
run_study4.py ONLY after study4/guard.py has verified a LOCKED, matching pre-registration --
never on --check, and never if verification fails.

Mirrors run_study3_live.py closely and reuses its client/key handling (make_client) unchanged:
the key is read from the environment, never written to a file, log, or error message. Arm A1
additionally attaches a YieldGate (study4/yield_gate.py) for any patch scoped to agent/meta/**,
with zero kernel/ changes -- see study4/README.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from kernel.llm_meta import CallLimits, LLMMetaAgent, ProposalHistory
from kernel.orchestrator import Archive, GitRepo, KernelGuard, MomCenter, Orchestrator, Sandbox, ShapleyAttributor
from kernel.yield_test import YieldConfig
from run_study2 import KERNEL_DIR, final_eval, load_secret, make_config, summarize
from run_study3_live import make_client
from study2.seed_repo import write_seed_agent_study4
from study2.verifier import PytestVerifier
from study4.fdr_replay import attach_persistent_fdr_budget
from study4.live_meta import MetaAgentA1
from study4.yield_gate import build_study4_yield_gate

log = logging.getLogger("study4")

# Not a credential: seeds the deterministic SYNTHETIC yield-measurement world only (kernel/world.py),
# never the live study2 problem secret (that one comes from load_secret(), per-run, random).
YIELD_WORLD_SECRET = b"study4-yield-synthetic-world-not-a-secret-00000"


class LiveOrchestrator(Orchestrator):
    """Records one coarse outcome line per cycle in the proposal history (WAL), same as Study 3."""
    history: ProposalHistory | None = None

    def run_cycle(self, cycle: int) -> dict:
        out = super().run_cycle(cycle)
        if self.history is not None:
            self.history.outcome(cycle, out.get("verdict", "reject"), out.get("error_type", "none"))
        return out


def build(out: Path, a):
    client = make_client(a)
    history = ProposalHistory(out / "runs" / "proposal_history.jsonl")
    limits = CallLimits(max_calls_per_cycle=a.max_calls_per_cycle, min_interval_s=a.min_interval,
                        max_run_tokens=a.run_token_cap)
    meta_cls = MetaAgentA1 if a.arm == "A1" else LLMMetaAgent
    meta = meta_cls(client, history=history, limits=limits, temperature=a.temperature,
                    max_context_chars=a.max_context_chars, max_tokens=a.max_tokens)
    cfg = make_config(out, a, meta.describe())
    cfg.fdr_mode = "enforce"                        # Study 4 decision 3: enforced, not monitor-only
    if not (cfg.repo_dir / ".git").exists():
        write_seed_agent_study4(cfg.repo_dir)       # the ordinary seed plus agent/meta/policy.py
        GitRepo(cfg.repo_dir, out / "_init_wt").init("seed agent")
    repo = GitRepo(cfg.repo_dir, cfg.runs_dir / "worktrees")
    ver = PytestVerifier(load_secret(out), Sandbox(cfg), cases_per_problem=12, suite="v2")
    orch = LiveOrchestrator(cfg, repo, Sandbox(cfg), ver, meta,
                            Archive(cfg.runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                            ShapleyAttributor(cfg, repo, ver), MomCenter(cfg.runs_dir / "mom.db"), KernelGuard(KERNEL_DIR))
    orch.history = history
    # Enforced FDR (decision 3) is only genuinely "online" if the budget survives separate CLI
    # invocations against the same output dir -- kernel/orchestrator.py always builds a fresh
    # ELond(t=0, d=0) otherwise. Replays the ledger's own past correctness-path tests to restore it.
    attach_persistent_fdr_budget(orch, cfg.runs_dir / "ledger.jsonl")
    if a.arm == "A1":
        orch.bootstrap()                            # YieldGate.starts() needs the anchor frozen
        orch.yield_gate = build_study4_yield_gate(
            orch, out / "yield", YIELD_WORLD_SECRET, profile=a.yield_profile,
            ycfg=YieldConfig(n_starts=a.yield_starts, seeds=tuple(range(a.yield_seeds)), k_cycles=a.yield_k))
    return orch, meta


def run(a) -> int:
    out = Path(a.out)
    orch, meta = build(out, a)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"argv_flags": vars(a), "meta": meta.describe()}, indent=1, default=str))
    print(f"arm: {a.arm}  |  model: {json.dumps(meta.describe())}  |  out: {out}  |  cycles: {a.cycles}")
    summary = orch.run(a.cycles, workers=1)
    print("run:", summary)
    if not summary.get("halted"):
        fe = final_eval(orch)
        (out / "final_eval.json").write_text(json.dumps(fe, indent=1))
        print("final evaluation (fresh batches):", json.dumps(fe, indent=1))
    rep = summarize(out)
    rep["llm_calls"], rep["llm_tokens"] = meta.total_calls, meta.total_tokens
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    print("summary:", json.dumps(rep, indent=1))
    return 0 if not summary.get("halted") and rep["ledger_ok"] else 2
