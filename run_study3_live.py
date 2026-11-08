#!/usr/bin/env python3
"""run_study3_live.py: Study 3, live LLM meta-agent (arm A0: solver patches only).

    # 1. put the key in your environment (never in a file):
    export GROQ_API_KEY=...                       # OPENAI_API_KEY also works
    # 2. one LLM call only, to check the model replies with a valid proposal (cheapest test):
    python run_study3_live.py --probe
    # 3. a short live run (3 cycles), output folder runs/study3_live:
    python run_study3_live.py --cycles 3 --out runs/study3_live

Wiring: LLMMetaAgent (kernel/llm_meta.py) -> Orchestrator (kernel/orchestrator.py) with the
two-stage judge, the hash-chained ledger and the circuit breakers, which the orchestrator builds
itself. ProposalHistory is a write-ahead log, and LiveOrchestrator writes one outcome line per cycle.

Exit codes: 0 ok, 2 halted or ledger failed verification, 3 provider or key error.
The API key is read from the environment and is never written to any file or log line.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kernel.llm_meta import (GROQ_BASE_URL, GROQ_MODEL, KEY_ENVS, CallLimits, LLMError, LLMMetaAgent,  # noqa: E402
                             OpenAICompatibleClient, ProposalHistory)
from kernel.orchestrator import (Archive, GitRepo, KernelGuard, MomCenter, Orchestrator, Budget,  # noqa: E402
                                 Sandbox, ShapleyAttributor)
from run_study2 import KERNEL_DIR, final_eval, load_secret, make_config, summarize  # noqa: E402
from kernel.patching import PatchError  # noqa: E402
from study2.seed_repo import write_seed_agent  # noqa: E402
from study2.verifier import PytestVerifier  # noqa: E402

log = logging.getLogger("study3")


class LiveOrchestrator(Orchestrator):
    """Orchestrator that records one coarse outcome line per cycle in the proposal history (WAL)."""
    history: ProposalHistory | None = None

    def run_cycle(self, cycle: int) -> dict:
        out = super().run_cycle(cycle)
        if self.history is not None:
            self.history.outcome(cycle, out.get("verdict", "reject"), out.get("error_type", "none"))
        return out


def make_client(a) -> OpenAICompatibleClient:
    """The key comes from the environment only. Errors name the variable, never the value."""
    if a.api_key_env:
        key = os.environ.get(a.api_key_env, "").strip()
        if not key:
            raise LLMError(f"environment variable {a.api_key_env} is empty or not set")
        return OpenAICompatibleClient(model=a.model, base_url=a.base_url, api_key=key)
    names = [n for n in KEY_ENVS if os.environ.get(n, "").strip()]
    if not names:
        raise LLMError(f"no key found: set {' or '.join(KEY_ENVS)} in the environment")
    log.info("API key taken from $%s (value not shown)", names[0])
    return OpenAICompatibleClient(model=a.model, base_url=a.base_url, env=os.environ)


def build(out: Path, a):
    client = make_client(a)
    history = ProposalHistory(out / "runs" / "proposal_history.jsonl")
    limits = CallLimits(max_calls_per_cycle=a.max_calls_per_cycle, min_interval_s=a.min_interval,
                        max_run_tokens=a.run_token_cap)
    meta = LLMMetaAgent(client, history=history, limits=limits, arm=a.arm, temperature=a.temperature,
                        max_context_chars=a.max_context_chars, max_tokens=a.max_tokens)
    cfg = make_config(out, a, meta.describe())
    if not (cfg.repo_dir / ".git").exists():
        write_seed_agent(cfg.repo_dir)
        GitRepo(cfg.repo_dir, out / "_init_wt").init("seed agent")
    repo = GitRepo(cfg.repo_dir, cfg.runs_dir / "worktrees")
    ver = PytestVerifier(load_secret(out), Sandbox(cfg), cases_per_problem=12, suite="v2")
    orch = LiveOrchestrator(cfg, repo, Sandbox(cfg), ver, meta,
                            Archive(cfg.runs_dir / "archive.json", cfg.seed, p_exploit=cfg.p_exploit),
                            ShapleyAttributor(cfg, repo, ver), MomCenter(cfg.runs_dir / "mom.db"), KernelGuard(KERNEL_DIR))
    orch.history = history
    return orch, meta


def probe(a) -> int:
    """One LLM call on a fresh seed agent. Nothing is applied, judged or committed."""
    out = Path(a.out)
    client = make_client(a)
    meta = LLMMetaAgent(client, history=None, limits=CallLimits(max_calls_per_cycle=a.max_calls_per_cycle,
                                                                 min_interval_s=0.0), arm=a.arm,
                        max_context_chars=a.max_context_chars, max_tokens=a.max_tokens)
    wt = out / "_probe_wt"
    write_seed_agent(wt)
    print("provider:", json.dumps(client.describe()))
    try:
        p = meta.propose(wt, "0" * 40, [], Budget(120, a.token_budget))
    except PatchError:
        print("the model's raw reply (first 800 chars, for diagnosis):")
        print(meta.last_reply_text[:800])
        raise
    print("hypothesis:", p.hypothesis)
    for o in p.ops:
        print(f"  op: {o.kind} {o.path} symbol={o.symbol} tag={o.tag} chars={len(o.new_source or '')}")
    print(f"tokens used: {meta.total_tokens} in {meta.total_calls} call(s)")
    return 0


def run(a) -> int:
    out = Path(a.out)
    orch, meta = build(out, a)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"argv_flags": vars(a), "meta": meta.describe()}, indent=1, default=str))
    print(f"model: {json.dumps(meta.describe())}  |  out: {out}  |  cycles: {a.cycles}")
    summary = orch.run(a.cycles, workers=1, accept_fingerprint_change=a.accept_fingerprint_change)
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


PREREG_PATH = ROOT / "results" / "study3" / "prereg.json"


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prereg(path: Path = PREREG_PATH) -> dict:
    """Pre-registration, written and committed BEFORE any live run. The sha256 of the body is stored
    beside it, and the code hashes make any later change to the runner or the meta-agent detectable."""
    from kernel.orchestrator import KernelGuard
    body = {
        "study": "Study 3: live LLM meta-agent (SI-Loop)",
        "question": "Does a live LLM produce at least one accepted solver patch from the frozen seed agent, "
                    "without p-hacking or reward hacking?",
        "primary_metric": "count of cycles with verdict accept or stepping_stone, plus the fresh-batch final_eval "
                          "(champion vs anchor, selection and held-out families)",
        "arms_run_now": ["A0"],
        "arms_declared_phase4": {"A1": "full self-improvement with yield tests (agent/meta/**)",
                                 "A3": "compute-matched best-of-N from the anchor"},
        "provider": {"base_url": GROQ_BASE_URL, "model": GROQ_MODEL, "key_source": list(KEY_ENVS)},
        "config": {"n": 48, "seed": 0, "cycles": 3, "token_budget": 20_000, "run_token_cap": 200_000,
                   "max_calls_per_cycle": 3, "min_interval_s": 3.0, "max_context_chars": 12_000,
                   "max_tokens": 1500, "temperature": 0.7, "workers": 1, "max_hunks": 8},
        "yield_gate": "not wired in A0 (no agent/meta in the seed repo); A1 is Phase 4",
        "stop_rules": ["circuit breakers (kernel/breakers.py) halt the run", "run_token_cap", "per-cycle caps",
                       "kernel digest tamper aborts the run"],
        "analysis_rules": ["all runs are reported, including failed and halted ones",
                           "no change to hypotheses, metric or stop rules after the first live result",
                           "no re-running a configuration to find a better outcome",
                           "the LLM sees only coarse outcomes, never counts or per-family numbers"],
        "code_sha256": {"run_study3_live.py": _sha(ROOT / "run_study3_live.py"),
                        "kernel/llm_meta.py": _sha(ROOT / "kernel" / "llm_meta.py"),
                        "study3_summary.py": _sha(ROOT / "study3_summary.py")},
        "kernel_digest": KernelGuard(ROOT / "kernel").digest(),
        "replication": {
            "design": "solver arm A0, six independent replicates (seeds 4-9), one new folder and secret each, judged by suite v2",
            "runs": [{"folder": f"runs/study3_seed{k}", "seed": k} for k in range(4, 10)],
            "earlier_runs_reported_separately": "runs/study3_seed1..3 (judged by suite v1, before this design) "
                                                "and runs/study3_run2, runs/study3_live (pilot); not pooled",
            "independence": "each folder gets its own random secret (secret.key), which seeds the sealed hidden "
                            "problems; --seed only changes archive randomness; LLM sampling is not seeded",
            "must_not_set": "SI_LOOP_SECRET",
            "verifier_suite": "v2 (12 cases per problem, plus 12 targeted cases for two_sum tie-breaks)",
            "command": "python run_study3_live.py --cycles 3 --seed K --out runs/study3_seedK",
            "criteria": {
                "repeatable": "a strict majority (at least 4 of 6) of runs have a correctness-path accept AND selection gain > 0 AND broken = 0",
                "generalises": "a strict majority (at least 4 of 6) of runs have held-out family gain > 0",
                "integrity": "every run has ledger_ok true, security_flags 0, and no circuit-breaker trips"
            }
        },
        "deviations": [
            {"id": "D1", "what": "Gate utility path: 'cheaper' now means >= 5% fewer lines of code only. "
                          "The token clause was removed.",
             "why": "Found after the first pilot (runs/study3_live). The verifier reports tokens = 0 for every "
                    "candidate, so the token clause was always true; cycle 0 was accepted through it with no "
                    "measured gain. This was found AFTER seeing the pilot result.",
             "pilot_kept_as_evidence": "runs/study3_live is kept unchanged and is reported as the pilot that "
                                       "exposed the bug. Its cycle-0 accept is not counted as a result."},
            {"id": "D2", "what": "Judge suite v2 for the replicates: 12 cases per problem plus 12 targeted cases "
                          "for two_sum tie-breaks. Suite v1 (Study 2) is unchanged.",
             "why": "Found after seed 3: its cycle-2 two_sum returns the first pair found, not the smallest first index "
                    "(docstring rule). Suite v1 caught it on about 14% of two_sum problems; v2 on about 71%. "
                    "Decided after seeing seed 3, so it is a post-hoc change to the judge.",
             "residual_gap": "v2 still misses about 29% of two_sum problems; other docstring rules were not audited."},
        ],
        "note_study2_kernel_digest": "results/study2/ablation/prereg.json records the kernel digest before "
                                     "kernel/llm_meta.py was added; the Study 2 results are unchanged.",
    }
    import hashlib
    body["prereg_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    (path.parent / "prereg.sha256").write_text(body["prereg_sha256"] + "\n")
    return body


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/study3_live")
    p.add_argument("--probe", action="store_true", help="one LLM call only; nothing is applied")
    p.add_argument("--prereg", action="store_true", help="write results/study3/prereg.json and exit")
    p.add_argument("--backend", choices=["groq", "openai-compatible"], default="groq")
    p.add_argument("--model", default=GROQ_MODEL)
    p.add_argument("--base-url", default="", help="default: Groq's OpenAI-compatible endpoint")
    p.add_argument("--api-key-env", default="", help="name of the env var holding the key (default: OPENAI_API_KEY, then GROQ_API_KEY)")
    p.add_argument("--arm", default="A0", choices=["A0"], help="A1 and A3 are Phase 4")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--n", type=int, default=48, help="problems per selection / confirm batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor-every", type=int, default=5)
    p.add_argument("--heldout-every", type=int, default=10)
    p.add_argument("--token-budget", type=int, default=20_000, help="LLM tokens per cycle")
    p.add_argument("--run-token-cap", type=int, default=200_000, help="LLM tokens for the whole run (0 = no cap)")
    p.add_argument("--max-calls-per-cycle", type=int, default=3)
    p.add_argument("--min-interval", type=float, default=3.0, help="seconds between LLM calls (rate limiting)")
    p.add_argument("--max-context-chars", type=int, default=12_000, help="repo text sent per call")
    p.add_argument("--max-tokens", type=int, default=1500, help="reply length cap per call")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--container-image", default="")
    p.add_argument("--accept-fingerprint-change", action="store_true")
    a = p.parse_args(argv)
    if a.backend == "openai-compatible" and not a.base_url:
        p.error("--backend openai-compatible needs --base-url")
    if not a.base_url:
        a.base_url = GROQ_BASE_URL
    if a.backend == "groq" and a.model == "":
        a.model = GROQ_MODEL
    return a


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    if a.prereg:
        body = write_prereg()
        print("pre-registered Study 3 ->", PREREG_PATH, "\nsha256", body["prereg_sha256"])
        return 0
    try:
        return probe(a) if a.probe else run(a)
    except KeyboardInterrupt:
        print(f"\nstopped (Ctrl+C). Partial results are in {a.out}; the ledger keeps every finished cycle.",
              file=sys.stderr)
        return 130
    except LLMError as e:
        print(f"provider error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
