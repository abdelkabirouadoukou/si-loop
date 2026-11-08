"""kernel/seed_agent.py: writes the SEED agent codebase (the cycle-0 agent) into a directory.

The seed solver is a registry of strategy functions: strategy(spec) -> candidate SOURCE for the
task. It starts with a few correct and a few buggy strategies, so there is real headroom. The
meta-agent may change anything under agent/ (including agent/meta/policy.py, its own policy).
"""
from __future__ import annotations

from pathlib import Path

from kernel.improver import END_MARK, SEED_POLICY, policy_source, strategy_source

REGISTRY = '''STRATEGIES = {}


def register(kind):
    def deco(fn):
        STRATEGIES[kind] = fn
        return fn
    return deco


def fill(tpl, params):
    for k in sorted(params):
        tpl = tpl.replace("@@%s@@" % k, repr(params[k]))
    return tpl
'''

SOLVER = '''from agent import strategies  # noqa: F401  (registers every strategy)
from agent.registry import STRATEGIES

FALLBACK_CALL = "def solve(*args):\\n    return None\\n"
FALLBACK_CLASS = "class Solution:\\n    pass\\n"


def solve_problem(spec):
    fn = STRATEGIES.get(spec["kind"])
    fallback = FALLBACK_CLASS if spec["signature"].startswith("class") else FALLBACK_CALL
    if fn is None:
        return fallback, 0
    try:
        return fn(spec), 10
    except Exception:
        return fallback, 10
'''

TEST = '''import ast

from agent.registry import STRATEGIES
from agent.solver import solve_problem

SAMPLE = {
    "dedupe": {"keep": "first"}, "rotate": {"dir": "left"}, "prefix_mod": {"m": 7},
    "kth_smallest": {"distinct": True}, "bracket_balance": {"alpha": "()[]"},
    "min_stack": {"cap": None}, "lru_cache": {"miss": -1},
    "arith_eval": {"div": "floor"}, "rpn_eval": {"div": "floor"},
}


def test_every_strategy_emits_parseable_code():
    for kind, fn in STRATEGIES.items():
        ast.parse(fn({"kind": kind, "params": SAMPLE[kind]}))


def test_unknown_kind_falls_back():
    code, _ = solve_problem({"kind": "nope", "signature": "solve(x)", "params": {}})
    assert "def solve" in code
'''

SEED_STRATEGIES = [("bracket_balance", "good"), ("dedupe", "bad"), ("rotate", "bad"), ("prefix_mod", "bad"),
                   ("min_stack", "bad"), ("arith_eval", "bad"), ("rpn_eval", "bad")]


def write_seed_agent(root: Path, policy: dict = None) -> None:
    root = Path(root)
    body = "from agent.registry import register, fill\n\n\n"
    for i, (kind, q) in enumerate(SEED_STRATEGIES):
        body += strategy_source(kind, q, f"seed{i}")
    body += END_MARK + "\n"
    files = {"agent/__init__.py": "", "agent/registry.py": REGISTRY, "agent/solver.py": SOLVER,
             "agent/strategies.py": body, "agent/meta/__init__.py": "",
             "agent/meta/policy.py": policy_source(policy or SEED_POLICY),
             "agent/tests/__init__.py": "", "agent/tests/test_registry.py": TEST,
             ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n"}
    for rel, txt in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
