"""End-to-end test of `python run_study4.py --run` (study4/run_live.py), offline: a fake local
OpenAI-compatible server stands in for the provider, the same way tests/test_study3.py tests
run_study3_live.py. No real API is called.
"""
import hashlib
import json

import pytest

import run_study4
import study4.run_live as RL
from study4.guard import ROOT
from test_study3 import KEY, _chat, _valid  # reuse the exact fake-server plumbing, not a copy of it
from test_study4_guard import _make
from test_study4_live_meta import make_server  # noqa: F401 (pytest fixture)


def _locked_prereg_for(tmp_path):
    return _make(tmp_path, code_sha256={"run_study4.py": hashlib.sha256(
        (ROOT / "run_study4.py").read_bytes()).hexdigest()})


@pytest.fixture(autouse=True)
def _skip_real_pytest_selftest(monkeypatch):
    # Sandbox._build() (kernel/orchestrator.py) scrubs $HOME to the ephemeral worktree for isolation;
    # on a machine where pytest is a user-site package (resolved via $HOME), that hides it from the
    # self-test subprocess -- see study4/README.md / tests/test_study4_integration.py for the same
    # issue. Not a kernel bug and not what these CLI-level tests are about: they exercise run_study4.py
    # --run's wiring (guard -> client -> orchestrator -> judge/yield gate), not the self-test mechanism.
    # Config's dataclass default is baked into __init__'s own defaults at class-definition time, so
    # patching the class attribute would not reach it -- override the instance make_config() returns.
    orig_make_config = RL.make_config

    def patched(out, a, model_desc):
        cfg = orig_make_config(out, a, model_desc)
        cfg.selftest_cmd = ("python", "-c", "pass")
        return cfg

    monkeypatch.setattr(RL, "make_config", patched)


def test_run_arm_a0_completes_one_cycle_offline(tmp_path, monkeypatch, make_server, capsys):
    p = _locked_prereg_for(tmp_path)
    srv = make_server([(200, _chat(_valid(), 100), {})] * 6)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "out"
    rc = run_study4.main(["--run", "--prereg", str(p), "--out", str(out), "--arm", "A0",
                          "--cycles", "1", "--n", "8", "--anchor-every", "0", "--heldout-every", "0",
                          "--base-url", srv.url, "--min-interval", "0"])
    assert rc in (0, 2), capsys.readouterr().err
    ledger = [json.loads(l) for l in (out / "runs" / "ledger.jsonl").read_text().splitlines()]
    ends = [r for r in ledger if r.get("ev") == "end"]
    assert ends and ends[0]["error_type"] != "selftest", ends[0]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cycles"] == 1
    assert KEY not in (out / "run_config.json").read_text()


def test_run_arm_a1_wires_the_yield_gate_offline(tmp_path, monkeypatch, make_server, capsys):
    p = _locked_prereg_for(tmp_path)
    policy_edit = json.dumps({"hypothesis": "tune w_adopt", "ops": [
        {"path": "agent/meta/policy.py", "kind": "replace_text", "old_text": '    "w_adopt": 2.0,',
         "new_text": '    "w_adopt": 9.0,', "tag": "meta.tune"}], "expected_effect": {}})
    srv = make_server([(200, _chat(policy_edit, 100), {})] * 6)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "out"
    rc = run_study4.main(["--run", "--prereg", str(p), "--out", str(out), "--arm", "A1",
                          "--cycles", "1", "--n", "8", "--anchor-every", "0", "--heldout-every", "0",
                          "--base-url", srv.url, "--min-interval", "0",
                          "--yield-profile", "tiny", "--yield-starts", "1", "--yield-seeds", "1", "--yield-k", "2"])
    assert rc in (0, 2), capsys.readouterr().err
    ledger = [json.loads(l) for l in (out / "runs" / "ledger.jsonl").read_text().splitlines()]
    ends = [r for r in ledger if r.get("ev") == "end"]
    assert ends and ends[0]["scope"] == "meta"
    assert "yield" in ends[0].get("detail", {}), "arm A1's meta-scope patch must be judged by YieldGate"


def test_run_refused_before_touching_the_loop_when_guard_fails(tmp_path, capsys):
    rc = run_study4.main(["--run", "--prereg", str(tmp_path / "absent.json"), "--out", str(tmp_path / "out")])
    assert rc == 3
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "out").exists(), "the runner must not touch the output dir before the guard passes"
