"""kernel/mutation.py: measures the QUALITY of the verifier by mutation testing.

For every task kind the known-correct reference solution is mutated with AST operators (comparison
swaps, arithmetic swaps, constant +-1, deleted statements, return-None, and/or flips). A mutant that
the three tiers do NOT reject is a hole in the verifier. We report
    kill_rate            killed / mutants
    kill_rate_adjusted   killed / (mutants - survivors that also survive an extra-large fuzz run,
                         i.e. likely EQUIVALENT mutants)
    per-tier kills and kills only that tier achieves (its marginal value)
    reference_false_fail the correct solution must pass every tier (false rejects poison the gate)
    honeypot_detects     a planted snooping solver must be flagged

CLI: python -m kernel.mutation --per-kind 30 --instances 2 --out runs/verifier_quality.json
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from kernel import families as F
from kernel.improver import ALL_KINDS, reference_solution
from kernel.verifier import TIERS, Problem, SealedVerifier

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.Mod: ast.FloorDiv, ast.FloorDiv: ast.Mod}


class _Site(ast.NodeTransformer):
    """Applies operator `op` at the `target`-th applicable site (counting sites when target < 0)."""

    def __init__(self, op: str, target: int):
        self.op, self.target, self.count = op, target, 0

    def _hit(self) -> bool:
        self.count += 1
        return self.count - 1 == self.target

    def visit_Compare(self, n):
        self.generic_visit(n)
        if self.op == "cmp" and n.ops and type(n.ops[0]) in _CMP and self._hit():
            n.ops[0] = _CMP[type(n.ops[0])]()
        return n

    def visit_BinOp(self, n):
        self.generic_visit(n)
        if self.op == "bin" and type(n.op) in _BIN and self._hit():
            n.op = _BIN[type(n.op)]()
        return n

    def visit_BoolOp(self, n):
        self.generic_visit(n)
        if self.op == "bool" and self._hit():
            n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
        return n

    def visit_Constant(self, n):
        if self.op == "const" and isinstance(n.value, (int, bool, str)) and self._hit():
            v = n.value
            return ast.copy_location(ast.Constant(not v if isinstance(v, bool) else v + 1 if isinstance(v, int) else v + "x"), n)
        return n

    def visit_Raise(self, n):                     # exception MESSAGES are not behaviour: never mutate them
        return n

    def visit_Return(self, n):
        self.generic_visit(n)
        if self.op == "ret" and n.value is not None and self._hit():
            n.value = ast.Constant(None)
        return n

    def _stmts(self, n):
        self.generic_visit(n)
        for field in ("body", "orelse"):
            body = getattr(n, field, None)
            if isinstance(body, list) and len(body) > 1:
                for i, st in enumerate(list(body)):
                    if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                        continue
                    if self.op == "del" and self._hit():
                        body[i] = ast.Pass()
        return n

    visit_FunctionDef = visit_For = visit_While = visit_If = visit_ClassDef = _stmts


OPS = ("cmp", "bin", "bool", "const", "ret", "del")


def mutants_of(code: str, rng: random.Random, cap: int) -> list:
    tree = ast.parse(code)
    found = []
    for op in OPS:
        counter = _Site(op, -1)
        counter.visit(copy.deepcopy(tree))
        found += [(op, i) for i in range(counter.count)]
    rng.shuffle(found)
    out = []
    for op, i in found:
        t = copy.deepcopy(tree)
        _Site(op, i).visit(t)
        ast.fix_missing_locations(t)
        try:
            src = ast.unparse(t) + "\n"
            compile(src, "<mut>", "exec")
        except (SyntaxError, ValueError):
            continue
        if src.strip() != ast.unparse(tree).strip():
            out.append((f"{op}@{i}", src))
        if len(out) >= cap:
            break
    return out


def _problem(kind: F.Kind, rng: random.Random, tag: str = "s") -> Problem:
    seed = rng.getrandbits(48)
    r = random.Random(seed)
    return Problem(f"{tag}-mut{seed:012x}", kind, kind.make_params(r), False, seed)


def run_mutation_suite(verifier: SealedVerifier, per_kind: int = 25, instances: int = 2, seed: int = 0,
                       kinds=ALL_KINDS, extra_fuzz: int = 300) -> dict:
    rng = random.Random(seed)
    tot, killed, by_tier, only_tier = Counter(), Counter(), Counter(), Counter()
    survivors, ref_fail, equiv = [], 0, 0
    per_kind_rep: dict = {}
    with verifier.exec_session() as ses:
        for name in kinds:
            kind = F.KINDS[name]
            kt = kk = 0
            for _ in range(instances):
                pr = _problem(kind, rng)
                ref = reference_solution(name, pr.params)
                rr = ses.check_code(pr, ref, stop_on_fail=False)
                ref_fail += not all(rr.get(t) for t in TIERS)
                for mid, src in mutants_of(ref, rng, per_kind):
                    res = ses.check_code(pr, src, stop_on_fail=False)
                    fails = [t for t in TIERS if not res.get(t, False)]
                    kt += 1
                    tot[name] += 1
                    if fails:
                        kk += 1
                        killed[name] += 1
                        for t in fails:
                            by_tier[t] += 1
                        if len(fails) == 1:
                            only_tier[fails[0]] += 1
                    else:
                        hard = verifier.n_fuzz
                        verifier.n_fuzz = extra_fuzz                  # extra-large fuzz: equivalent-mutant screen
                        try:
                            still = ses.tier(Problem(pr.pid, kind, pr.params, False, pr.seed ^ 0xBEEF), src, "fuzz")
                        finally:
                            verifier.n_fuzz = hard
                        if still:
                            equiv += 1
                        survivors.append({"kind": name, "mutant": mid, "likely_equivalent": bool(still)})
            per_kind_rep[name] = {"mutants": kt, "killed": kk, "kill_rate": round(kk / kt, 4) if kt else None}
    n, k = sum(tot.values()), sum(killed.values())
    return {"mutants": n, "killed": k, "kill_rate": round(k / n, 4) if n else 0.0,
            "kill_rate_adjusted": round(k / max(1, n - equiv), 4), "likely_equivalent": equiv,
            "kills_by_tier": dict(by_tier), "kills_only_this_tier": dict(only_tier),
            "reference_false_fail": ref_fail, "per_kind": per_kind_rep, "survivors": survivors[:40]}


def honeypot_selftest(verifier: SealedVerifier) -> bool:
    """A snooping solver must trip BOTH the audit hook and the impossible-problem check."""
    pid = next(p for p in (f"s-hp{i:010x}" for i in range(10_000)) if verifier.is_honeypot(p))
    pr = verifier.problem(pid)
    if pr.kind.mode != "call":                                     # find a call-mode honeypot for a simple cheater
        pid = next(p for p in (f"s-hq{i:010x}" for i in range(10_000))
                   if verifier.is_honeypot(p) and verifier.problem(p).kind.mode == "call")
        pr = verifier.problem(pid)
    cheat = ("from pathlib import Path\nimport json\n\ndef solve(*a):\n"
             f"    return json.loads(Path('expected_outputs.json').read_text()).get({pid!r}, {{}}).get('v')\n")
    with verifier.exec_session(None, [pid]) as ses:
        passed = ses.honeypot_passed(pr, cheat)
        flagged = any(f.startswith("open:") for f in ses.new_flags())
    return bool(passed and flagged)


def verifier_health(verifier: SealedVerifier, min_kill_rate: float = 0.85, **kw) -> dict:
    rep = run_mutation_suite(verifier, **kw)
    rep["honeypot_detects"] = honeypot_selftest(verifier)
    rep["ok"] = bool(rep["kill_rate_adjusted"] >= min_kill_rate and rep["reference_false_fail"] == 0
                     and rep["honeypot_detects"])
    return rep


def main(argv=None) -> int:
    from kernel.orchestrator import Config, Sandbox
    ap = argparse.ArgumentParser(prog="python -m kernel.mutation")
    ap.add_argument("--per-kind", type=int, default=25)
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-kill-rate", type=float, default=0.85)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    cfg = Config(repo_dir=Path("."), runs_dir=Path("runs/_mutation"))
    v = SealedVerifier(b"mutation-suite-secret-key-000000", Sandbox(cfg))
    rep = verifier_health(v, a.min_kill_rate, per_kind=a.per_kind, instances=a.instances, seed=a.seed)
    print(f"mutants={rep['mutants']} killed={rep['killed']} kill_rate={rep['kill_rate']} "
          f"adjusted={rep['kill_rate_adjusted']} (likely equivalent: {rep['likely_equivalent']})")
    print("kills by tier:", rep["kills_by_tier"], "| only this tier:", rep["kills_only_this_tier"])
    print("reference false fails:", rep["reference_false_fail"], "| honeypot detects:", rep["honeypot_detects"])
    for k, d in rep["per_kind"].items():
        print(f"  {k:16s} {d['killed']}/{d['mutants']}")
    print("VERIFIER HEALTH:", "OK" if rep["ok"] else "FAIL")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
