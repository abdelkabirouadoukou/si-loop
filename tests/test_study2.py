import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from kernel.orchestrator import Budget, BudgetExceeded, Config, Sandbox
from kernel.patching import PatchError
from study2.llm_meta import LLMClient, LLMMetaAgent, OfflineDemoClient, OpenAICompatClient, Reply, make_client
from study2.seed_repo import write_seed_agent
from study2.tasks import HELDOUT_FAMILY, TASKS, make_cases
from study2.verifier import PytestVerifier

SECRET = b"study2-test-secret-0123456789abc"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    td = tmp_path_factory.mktemp("s2")
    write_seed_agent(td / "seed")
    return PytestVerifier(SECRET, Sandbox(Config(repo_dir=td, runs_dir=td / "r"))), td


def test_oracles_are_deterministic_and_json_safe():
    for t in TASKS.values():
        a, b = make_cases(t, 7), make_cases(t, 7)
        assert a == b
        for args in a:
            json.dumps(t.oracle(*args))


def test_heldout_family_is_isolated_and_ids_depend_on_the_secret(env):
    v, _ = env
    fam = lambda split: {v.cases_for(p)["family"] for p in v.draw_batch(1, 80, split)}
    assert fam("heldout") == {HELDOUT_FAMILY}
    assert all(HELDOUT_FAMILY not in fam(s) for s in ("selection", "confirm", "credit", "anchor", "yield", "final"))
    other = PytestVerifier(b"another-secret-0123456789abcdef", v.sandbox)
    assert set(v.draw_batch(1, 20)).isdisjoint(other.draw_batch(1, 20))
    assert set(v.draw_batch(1, 20, "selection")).isdisjoint(v.draw_batch(1, 20, "confirm"))


def test_seed_agent_has_headroom_and_large_batches_are_not_truncated(env):
    v, td = env
    r = v.evaluate(td / "seed", v.draw_batch(9, 150, "selection"), 120)      # large output once broke the sandbox pipe
    assert 0.1 < r.pass_at_1 < 0.7 and not r.metrics["fail"] and r.flags == []


def test_a_correct_library_passes_everything_and_a_bad_one_does_not(env):
    v, td = env
    import inspect
    import study2.tasks as T
    lib = "import bisect\n_ROMAN = " + repr(T._ROMAN) + "\n" + "\n\n".join(inspect.getsource(t.oracle) for t in TASKS.values())
    lib += "\n\nTASKS = {" + ", ".join(f"'{n}': {n}" for n in TASKS) + "}\n"
    (td / "ref/agent").mkdir(parents=True, exist_ok=True)
    (td / "ref/agent/__init__.py").write_text("")
    (td / "ref/agent/lib.py").write_text(lib)
    assert v.evaluate(td / "ref", v.draw_batch(2, 40, "selection") + v.draw_batch(2, 20, "heldout"), 120).pass_at_1 == 1.0
    write_seed_agent(td / "broken")
    (td / "broken/agent/lib.py").write_text("raise RuntimeError('boom')\n")
    r = v.evaluate(td / "broken", v.draw_batch(2, 10), 60)
    assert r.pass_at_1 == 0.0


def test_snooping_and_hanging_agents(env):
    v, td = env
    write_seed_agent(td / "snoop")
    (td / "snoop/agent/lib.py").write_text(
        "from pathlib import Path\nTASKS = {}\nx = Path('hidden_tests/expected_outputs.json')\ny = x.read_text() if x.exists() else ''\n")
    assert any(f.startswith("open:") for f in v.evaluate(td / "snoop", v.draw_batch(3, 4), 60).flags)
    write_seed_agent(td / "hang")
    (td / "hang/agent/lib.py").write_text("def reverse_words(s):\n    while True:\n        pass\nTASKS = {'reverse_words': reverse_words}\n")
    fast = PytestVerifier(SECRET, v.sandbox, case_timeout=0.3)
    r = fast.evaluate(td / "hang", fast.draw_batch(4, 12), 60)
    assert r.pass_at_1 == 0.0


def test_a_forged_result_without_the_nonce_is_ignored(env):
    v, td = env
    write_seed_agent(td / "forge")
    (td / "forge/agent/lib.py").write_text(
        "import json\nprint('RESULT:00:' + json.dumps({'out': {}, 'flags': []}))\nTASKS = {}\n")
    assert v.evaluate(td / "forge", v.draw_batch(5, 6), 60).pass_at_1 == 0.0


# ------------------------------------------------------------------ LLM meta-agent
class Scripted(LLMClient):
    provider, model = "test", "t"

    def __init__(self, replies):
        self.replies, self.calls, self.prompts = list(replies), 0, []

    def complete(self, system, user, max_tokens=2000, temperature=0.7):
        self.prompts.append(user)
        self.calls += 1
        return Reply(self.replies.pop(0), 100)


GOOD = json.dumps({"hypothesis": "h", "ops": [{"path": "agent/lib.py", "kind": "replace_symbol", "symbol": "count_vowels",
                                                  "new_source": "def count_vowels(s):\n    return 1\n", "tag": "fix.x"}]})


def test_meta_retries_malformed_replies_then_parses(env):
    _, td = env
    c = Scripted(["not json at all", '{"hypothesis": "h", "ops": []}', GOOD])
    p = LLMMetaAgent(c).propose(td / "seed", "abc", [], Budget(60, 10_000))
    assert c.calls == 3 and p.ops[0].symbol == "count_vowels" and "rejected" in c.prompts[1]
    assert "def reverse_words" in c.prompts[0]                      # the agent saw its own code


def test_meta_gives_up_after_retries_and_charges_the_budget(env):
    _, td = env
    with pytest.raises(PatchError):
        LLMMetaAgent(Scripted(["x"] * 5), max_retries=1).propose(td / "seed", "a", [], Budget(60, 10_000))
    with pytest.raises(BudgetExceeded):
        LLMMetaAgent(Scripted([GOOD])).propose(td / "seed", "a", [], Budget(60, 50))


def test_meta_prompt_never_contains_tests_oracles_or_secret(env):
    v, td = env
    c = Scripted([GOOD])
    LLMMetaAgent(c).propose(td / "seed", "a", [{"kind": "lesson", "payload": "careful"}], Budget(60, 10_000))
    p = c.prompts[0]
    assert "oracle" not in p.lower() and SECRET.decode() not in p and "make_cases" not in p and "careful" in p


def test_openai_compatible_client_against_a_local_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"choices": [{"message": {"content": GOOD}}], "usage": {"total_tokens": 77}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cl = OpenAICompatClient("m", f"http://127.0.0.1:{srv.server_port}/v1")
        r = cl.complete("sys", "user")
    finally:
        srv.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["body"]["model"] == "m" and r.tokens == 77 and "count_vowels" in r.text


def test_offline_demo_replies_all_parse_or_are_deliberately_malformed():
    from kernel.patching import parse_proposal
    bad = 0
    from study2.llm_meta import DEMO_REPLIES
    for text in DEMO_REPLIES:
        try:
            parse_proposal(text)
        except PatchError:
            bad += 1
    assert bad == 1
    with pytest.raises(ValueError):
        make_client("nope")


def test_end_to_end_offline_cycle(tmp_path):
    import argparse

    import run_study2 as R
    a = argparse.Namespace(seed=0, n=16, token_budget=60_000, run_token_cap=0, anchor_every=0, heldout_every=0,
                           container_image="", temperature=0.7, workers=1, accept_fingerprint_change=False)
    orch = R.build(tmp_path, a, make_client("offline"))
    s = orch.run(3)
    assert not s["halted"] and s["cycles"] == 3
    rep = R.summarize(tmp_path)
    assert rep["ledger_ok"] and rep["cycles"] == 3
    fe = R.final_eval(orch, n=30)
    assert fe["heldout_family"]["fixed"] == 0 and fe["selection_families"]["champion"] >= fe["selection_families"]["anchor"]
