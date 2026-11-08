"""kernel/pipeline.py: the two-stage judge. Eliminates WINNER'S CURSE and limits LEAKAGE.

    Stage A  SCREEN   fresh batch A, evaluated in chunks. After each chunk a sequential monitor looks
                      at the discordant pairs only: anytime-valid HARM stop (e-value) and SPRT FUTILITY
                      stop. Early stops can only lose power, never create a false accept.
                      Lenient gate (screen_p_improve) decides whether the patch is promising.
    Stage B  CONFIRM  a second, INDEPENDENT batch B. The strict gate (p_improve_min) is applied to B
                      ALONE. The patch was selected on A, so A's estimate is biased upward; B is not.
                      With independent batches the false-accept rates multiply (see calibrate --two-stage).
    Budget            utility-path accepts spend the margin budget (anchor.py); correctness accepts that
                      hide a possible loss spend it too. An empty budget closes the utility path.
    FDR               e-LOND over all confirm tests (monitor or enforce).
    Honeypots         any security flag from the verifier = immediate reject (stage "honeypot").

Everything numeric stays in `Judgement.detail` (kernel-private ledger). What may reach the agent is
only `coarse()`: {verdict, error_type}.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from kernel.anchor import AnchorStore, MarginBudget
from kernel.orchestrator import Config, EvalResult, Gate, GitRepo, Verifier
from kernel.stats import ELond, SequentialMonitor, log_e_up, paired_counts

ERROR_TYPES = ("none", "regression", "no_gain", "unconfirmed", "margin", "loc_cap", "security", "fdr", "static",
               "selftest", "parse", "tamper", "budget", "internal", "yield", "mixed_scope")


@dataclass
class Judgement:
    verdict: str                              # accept | stepping_stone | reject
    reasons: list
    stage: str = ""
    error_type: str = "none"
    desc: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    charge: float = 0.0
    cand_pass: float = 0.0
    parent_pass: float = 0.0
    delta: float = 0.0
    p_better: float = 0.0
    detail: dict = field(default_factory=dict)
    cand_confirm: Optional[EvalResult] = None
    confirm_batch: Optional[list] = None

    def coarse(self) -> dict:
        """The ONLY outcome information the agent may ever see."""
        return {"verdict": self.verdict, "error_type": self.error_type}


def merge(results: list) -> EvalResult:
    passes, flags, fam, tok, loc, fail = {}, [], {}, 0.0, 0.0, {}
    for r in results:
        passes.update(r.passes)
        flags += r.flags
        tok += r.metrics.get("tokens", 0.0)
        loc = max(loc, r.metrics.get("agent_loc", 0.0))
        for k, v in r.family_profile.items():
            fam.setdefault(k, []).append((v, len(r.passes)))
        for k, v in r.metrics.get("fail", {}).items():
            fail[k] = fail.get(k, 0) + v
    prof = {k: sum(v * w for v, w in xs) / max(1, sum(w for _, w in xs)) for k, xs in fam.items()}
    return EvalResult(passes, {"tokens": tok, "agent_loc": loc, "fail": fail}, prof, sorted(set(flags)))


def chunked(batch: list, k: int) -> list:
    k = max(1, min(k, len(batch)))
    size = -(-len(batch) // k)
    return [batch[i:i + size] for i in range(0, len(batch), size)]


class LazyTree:
    """A worktree created only if an evaluation is actually needed (cache misses)."""

    def __init__(self, repo: GitRepo, sha: str, prefix: str = "eval"):
        self.repo, self.sha, self.prefix, self.branch, self.wt = repo, sha, prefix, None, None

    def path(self):
        if self.wt is None:
            self.branch = f"{self.prefix}/{self.sha[:8]}-{time.time_ns()}"
            self.wt = self.repo.add_worktree(self.branch, self.sha)
        return self.wt

    def close(self):
        if self.branch:
            self.repo.discard(self.branch)
            self.branch = self.wt = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class TwoStageJudge:
    def __init__(self, cfg: Config, repo: GitRepo, verifier: Verifier, budget: MarginBudget,
                 anchor: AnchorStore, elond: Optional[ELond] = None):
        self.cfg, self.repo, self.verifier, self.budget, self.anchor = cfg, repo, verifier, budget, anchor
        self.gate = Gate(cfg)
        self.screen_gate = Gate(Config(**{**cfg.__dict__, "p_improve_min": cfg.screen_p_improve}))
        self.elond = elond or ELond(cfg.fdr_alpha)
        self._cache: dict = {}

    # ------------------------------------------------------------------ evaluation helpers
    def loc_cap(self) -> float:
        s = self.anchor.seed_loc if self.anchor.frozen else 0.0
        return float("inf") if s <= 0 else max(self.cfg.loc_factor * s, s + self.cfg.loc_abs)

    def eval_sha(self, sha: str, batch: list, tree: Optional[LazyTree] = None, wall: Optional[float] = None) -> EvalResult:
        key = (sha, tuple(batch))
        if key not in self._cache:
            own = tree is None
            tree = tree or LazyTree(self.repo, sha)
            try:
                if len(self._cache) >= 512:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = self.verifier.evaluate(tree.path(), batch, wall or self.cfg.eval_wall_s)
            finally:
                if own:
                    tree.close()
        return self._cache[key]

    def _stage(self, wt, parent_sha: str, ptree: LazyTree, batch: list, wall: Callable, monitor: Optional[SequentialMonitor],
               futile_ok: Optional[Callable] = None):
        """Evaluate candidate and parent chunk by chunk; returns (cand, parent, stop_reason, flags)."""
        cres, pres, stop = [], [], None
        for ch in chunked(batch, self.cfg.screen_chunks):
            c = self.verifier.evaluate(wt, ch, wall())
            if c.flags:
                return merge(cres + [c]), merge(pres) if pres else c, "security", c.flags
            p = self.eval_sha(parent_sha, ch, ptree, wall())
            cres.append(c)
            pres.append(p)
            if monitor is not None and self.cfg.sequential_stop:
                b, cc, n = paired_counts(p.passes, c.passes, sorted(c.passes))
                monitor.update(b, cc, n)
                if monitor.harmful:
                    stop = "harm"
                    break
                if monitor.futile and futile_ok and futile_ok(merge(cres)):
                    stop = "futile"
                    break
        return merge(cres), merge(pres), stop, []

    # ------------------------------------------------------------------ the judge
    def judge(self, wt, sha: str, parent_sha: str, cycle: int, wall: Callable[[], float],
              descriptor_fn: Callable, novelty_fn: Callable) -> Judgement:
        cfg, det = self.cfg, {}
        mon = SequentialMonitor(cfg.harm_alpha, cfg.sprt_theta, cfg.sprt_beta, 0.05, cfg.sprt_min_discordant)
        with LazyTree(self.repo, parent_sha) as ptree:
            # ---------------- stage A: screen
            batch_a = self.verifier.draw_batch(cycle, cfg.n_selection, "selection")

            def futile_ok(partial: EvalResult) -> bool:
                # keep behaviourally novel or cheaper candidates alive: they may still earn a slot
                pm = partial.metrics
                return novelty_fn(descriptor_fn(partial), sha) < cfg.novelty_min and pm.get("agent_loc", 1e9) > 0.95 * self._parent_loc(parent_sha)
            cand, par, stop, flags = self._stage(wt, parent_sha, ptree, batch_a, wall, mon, futile_ok)
            if flags:
                return Judgement("reject", ["honeypot / security flag: " + ", ".join(flags[:3])], "honeypot", "security",
                                 flags=flags, detail={"flags": flags})
            desc = descriptor_fn(cand)
            det["loc"], det["tokens"] = cand.metrics.get("agent_loc", 0), cand.metrics.get("tokens", 0)
            if cand.metrics.get("agent_loc", 0) > self.loc_cap():
                return Judgement("reject", [f"agent_loc {cand.metrics['agent_loc']:.0f} > cap {self.loc_cap():.0f}"],
                                 "screen", "loc_cap", desc)
            b, c, n = paired_counts(par.passes, cand.passes, sorted(cand.passes))
            det["screen"] = {"b": b, "c": c, "n": n, "stop": stop, "harm_e": mon.harmful, "futile": mon.futile}
            base = dict(desc=desc, cand_pass=cand.pass_at_1, parent_pass=par.pass_at_1,
                        delta=(b - c) / max(1, n), detail=det)
            if stop == "harm":
                return Judgement("reject", [f"b={b} c={c} n={n}", "not non-inferior (sequential harm stop)"], "screen_harm", "regression", **base)
            if stop == "futile":
                return Judgement("reject", [f"b={b} c={c} n={n}", "no demonstrated gain (sequential futility stop)"], "screen_futility", "no_gain", **base)
            d1 = self.screen_gate.decide(par, cand, novelty_fn(desc, sha))
            det["screen"].update(delta=d1.delta, lower=d1.lower, p_better=d1.p_improve, verdict=d1.verdict)
            if d1.verdict == "reject":
                et = "regression" if "not non-inferior" in d1.reasons[-1] else "no_gain"
                return Judgement("reject", d1.reasons, "screen", et, p_better=d1.p_improve, **base)
            if d1.verdict == "stepping_stone":
                return Judgement("stepping_stone", d1.reasons, "screen", "none", p_better=d1.p_improve, **base)

            # ---------------- stage B: independent confirmation (decision on B ALONE)
            batch_b = self.verifier.draw_batch(cycle, cfg.n_confirm, "confirm")
            cand2, par2, stop2, flags2 = self._stage(wt, parent_sha, ptree, batch_b, wall, SequentialMonitor(cfg.harm_alpha))
            if flags2:
                return Judgement("reject", ["honeypot / security flag: " + ", ".join(flags2[:3])], "honeypot", "security",
                                 flags=flags2, detail={**det, "flags": flags2})
            b2, c2, n2 = paired_counts(par2.passes, cand2.passes, sorted(cand2.passes))
            det["confirm"] = {"b": b2, "c": c2, "n": n2, "stop": stop2}
            base.update(cand_pass=cand2.pass_at_1, parent_pass=par2.pass_at_1, delta=(b2 - c2) / max(1, n2))
            if stop2 == "harm":
                return Judgement("reject", [f"confirm b={b2} c={c2} n={n2}", "not non-inferior (confirm harm stop)"],
                                 "confirm_harm", "regression", **base)
            d2 = self.gate.decide(par2, cand2, novelty_fn(desc, sha))
            det["confirm"].update(delta=d2.delta, lower=d2.lower, p_better=d2.p_improve, verdict=d2.verdict, path=d2.path)
            pooled_b, pooled_c = b + b2, c + c2
            det["pooled_for_report_only"] = {"b": pooled_b, "c": pooled_c, "n": n + n2}
            base.update(p_better=d2.p_improve)
            if d2.verdict == "stepping_stone":
                return Judgement("stepping_stone", ["confirm: " + d2.reasons[-1]], "confirm", "none", **base)
            if d2.verdict == "reject":
                et = "regression" if "not non-inferior" in d2.reasons[-1] else "unconfirmed"
                return Judgement("reject", ["screen passed but confirmation failed"] + d2.reasons, "confirm", et, **base)

            # ---------------- margin budget + FDR (accepts only)
            charge = self.budget.charge_for(d2.lower, d2.path == "utility")
            det["margin"] = {"charge": charge, "remaining": self.budget.remaining, "path": d2.path}
            if not self.budget.can_afford(charge):
                return Judgement("reject", d2.reasons + [f"margin budget exhausted (need {charge:.4f}, have {self.budget.remaining:.4f})"],
                                 "margin", "margin", **base)
            if d2.path == "correctness" and cfg.fdr_mode != "off":
                hit, lvl = self.elond.test(log_e_up(b2, c2))
                det["fdr"] = {"hit": hit, "level": lvl, "mode": cfg.fdr_mode}
                if cfg.fdr_mode == "enforce" and not hit:
                    return Judgement("reject", d2.reasons + [f"online FDR: e-value below 1/{lvl:.2e}"], "fdr", "fdr", **base)
            return Judgement("accept", d2.reasons, "confirm", "none", charge=charge,
                             cand_confirm=cand2, confirm_batch=batch_b, **base)

    def _parent_loc(self, parent_sha: str) -> float:
        for (s, _), r in self._cache.items():
            if s == parent_sha:
                return r.metrics.get("agent_loc", 0.0)
        return 0.0

    def confirm_vs(self, champion_sha: str, j: Judgement):
        """Lineage switch: the candidate must also beat the CURRENT champion on the confirmation batch."""
        champ = self.eval_sha(champion_sha, j.confirm_batch)
        return self.gate.decide(champ, j.cand_confirm, 0.0)
