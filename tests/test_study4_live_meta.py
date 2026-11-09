"""Tests for study4/live_meta.py (MetaAgentA1) and study4/yield_gate.py (build_study4_yield_gate).
No real API is called: a fake OpenAI-compatible server runs on localhost, the same pattern
tests/test_study3.py uses for kernel/llm_meta.py's LLMMetaAgent (arm A0).
"""
import json

import pytest

import kernel.llm_meta as L
from kernel.orchestrator import Budget, BudgetExceeded
from kernel.patching import PatchError
from study2.seed_repo import write_seed_agent_study4
from study4.live_meta import A1_META_PATH, MetaAgentA1, SYSTEM_A1

from test_study3 import KEY, FakeServer, _chat  # reuse the exact fake-server plumbing, not a copy of it

pytest_plugins = ()


@pytest.fixture
def make_server(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    servers = []

    def factory(replies):
        s = FakeServer(replies)
        servers.append(s)
        return s

    yield factory
    for s in servers:
        s.close()


def _policy_proposal(old="2.0", new="3.0", n=1):
    ops = [{"path": A1_META_PATH, "kind": "replace_text", "old_text": f'    "w_adopt": {old},',
            "new_text": f'    "w_adopt": {new},', "tag": "meta.tune"} for _ in range(n)]
    return json.dumps({"hypothesis": "tune w_adopt", "ops": ops, "expected_effect": {}})


def _mixed_proposal():
    return json.dumps({"hypothesis": "mixed", "ops": [
        {"path": "agent/lib.py", "kind": "replace_symbol", "symbol": "reverse_words", "tag": "fix.string",
         "new_source": "def reverse_words(s):\n    return s\n"},
        {"path": A1_META_PATH, "kind": "replace_text", "old_text": '    "w_adopt": 2.0,',
         "new_text": '    "w_adopt": 3.0,', "tag": "meta.tune"},
    ], "expected_effect": {}})


def _client(srv):
    return L.OpenAICompatibleClient(model="openai/gpt-oss-120b", base_url=srv.url, api_key=KEY,
                                    sleep=lambda s: None)


def _agent(srv, tmp_path, **kw):
    kw.setdefault("limits", L.CallLimits(min_interval_s=0.0))
    return MetaAgentA1(_client(srv), history=None, **kw)


def test_arm_a1_accepts_a_policy_only_patch(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    srv = make_server([(200, _chat(_policy_proposal()), {})])
    agent = _agent(srv, tmp_path)
    p = agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))
    assert p.ops[0].path == A1_META_PATH
    assert p.ops[0].old_text == '    "w_adopt": 2.0,'
    req = srv.requests[0]
    assert req["body"]["messages"][0]["content"] == SYSTEM_A1, "arm A1 must use its own system prompt"


def test_arm_a1_still_accepts_a_solver_only_patch(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    ops = [{"path": "agent/lib.py", "kind": "replace_symbol", "symbol": "reverse_words", "tag": "fix.string",
            "new_source": "def reverse_words(s):\n    return s\n"}]
    body = json.dumps({"hypothesis": "fix reverse", "ops": ops, "expected_effect": {}})
    srv = make_server([(200, _chat(body), {})])
    agent = _agent(srv, tmp_path)
    p = agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))
    assert p.ops[0].path == "agent/lib.py"


def test_arm_a1_refuses_a_mixed_scope_patch_before_the_kernel_ever_sees_it(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    srv = make_server([(200, _chat(_mixed_proposal()), {})] * 1)
    agent = _agent(srv, tmp_path, max_retries=0)
    with pytest.raises(PatchError, match="not both"):
        agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))


def test_arm_a1_refuses_a_path_outside_lib_or_policy(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    ops = [{"path": "agent/tests/test_public.py", "kind": "replace_symbol", "symbol": "x",
            "new_source": "def x():\n    pass\n", "tag": "x"}]
    body = json.dumps({"hypothesis": "sneaky", "ops": ops, "expected_effect": {}})
    srv = make_server([(200, _chat(body), {})])
    agent = _agent(srv, tmp_path, max_retries=0)
    with pytest.raises(PatchError, match="not allowed"):
        agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))


def test_arm_a1_too_many_hunks_refused(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    srv = make_server([(200, _chat(_policy_proposal(n=9)), {})])
    agent = _agent(srv, tmp_path, max_retries=0)
    with pytest.raises(PatchError, match="too many hunks"):
        agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))


def test_arm_a1_cycle_token_budget_still_enforced(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    srv = make_server([(200, _chat(_policy_proposal(), 500), {})])
    with pytest.raises(BudgetExceeded):
        _agent(srv, tmp_path).propose(tmp_path, "b" * 40, [], Budget(60, 100))


def test_arm_a1_never_leaks_the_key(tmp_path, make_server):
    write_seed_agent_study4(tmp_path)
    srv = make_server([(200, _chat(_policy_proposal()), {})])
    agent = _agent(srv, tmp_path)
    agent.propose(tmp_path, "b" * 40, [], Budget(60, 10_000))
    assert KEY not in json.dumps(agent.describe())
    assert agent.describe()["arm"] == "A1"


def test_arm_a1_workspace_reader_sees_policy_py(tmp_path, make_server):
    """Sanity check: the live model's read-only context already includes agent/meta/policy.py
    (WorkspaceReader scans agent/**/*.py), so arm A1 does not need its own workspace-reading code."""
    write_seed_agent_study4(tmp_path)
    ctx = L.WorkspaceReader(tmp_path).context()
    assert "agent/meta/policy.py" in ctx
    assert "w_adopt" in ctx
