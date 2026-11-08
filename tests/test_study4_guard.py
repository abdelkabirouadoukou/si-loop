"""Tests for the Study 4 guard. They use temporary pre-registrations only; no real Study 4 prereg is created."""
import json
from pathlib import Path

import pytest

import run_study4
from study4.guard import PREREG_PATH, PreregError, ROOT, body_sha, verify
from kernel.orchestrator import KernelGuard


def _make(tmp_path: Path, **overrides) -> Path:
    code = tmp_path / "code.py"
    code.write_text("x = 1\n")
    body = {
        "study": "study4",
        "status": "locked",
        "question": "does an accepted improver patch pass the yield test?",
        "primary_metric": "accepted improver patches passing yield",
        "arms_run": ["A1", "A3", "A0"],
        "code_sha256": {"code.py": __import__("hashlib").sha256(code.read_bytes()).hexdigest()},
        "kernel_digest": KernelGuard(ROOT / "kernel").digest(),
        "stop_rules": ["stop at 3 cycles"],
        "analysis_rules": ["report every run"],
    }
    body.update(overrides)
    body = {k: v for k, v in body.items() if v is not None}
    body["prereg_sha256"] = body_sha(body)
    p = tmp_path / "prereg.json"
    p.write_text(json.dumps(body, indent=1, sort_keys=True))
    return p


def test_missing_prereg_is_refused(tmp_path):
    with pytest.raises(PreregError, match="no pre-registration"):
        verify(tmp_path / "absent.json")


def test_valid_locked_prereg_passes(tmp_path):
    # root is the real repo so the kernel digest is checked for real; code.py is then looked up under root,
    # so point the code hash at a file that exists in the repo instead.
    p = _make(tmp_path, code_sha256={"run_study4.py": __import__("hashlib").sha256(
        (ROOT / "run_study4.py").read_bytes()).hexdigest()})
    assert verify(p)["status"] == "locked"


def test_draft_status_is_refused(tmp_path):
    p = _make(tmp_path, status="draft")
    with pytest.raises(PreregError, match="only 'locked' may run"):
        verify(p)


def test_edited_body_is_refused(tmp_path):
    p = _make(tmp_path)
    d = json.loads(p.read_text())
    d["primary_metric"] = "something easier"          # edit after locking, hash left as it was
    p.write_text(json.dumps(d))
    with pytest.raises(PreregError, match="edited after locking"):
        verify(p)


def test_changed_code_is_refused(tmp_path):
    p = _make(tmp_path, code_sha256={"run_study4.py": "0" * 64})
    with pytest.raises(PreregError, match="changed since locking"):
        verify(p)


def test_wrong_study_is_refused(tmp_path):
    p = _make(tmp_path, study="study3")
    with pytest.raises(PreregError, match="not 'study4'"):
        verify(p)


def test_missing_required_key_is_refused(tmp_path):
    p = _make(tmp_path)
    d = json.loads(p.read_text())
    del d["stop_rules"]
    d["prereg_sha256"] = body_sha(d)
    p.write_text(json.dumps(d))
    with pytest.raises(PreregError, match="required keys"):
        verify(p)


def test_runner_refuses_without_prereg(tmp_path, capsys):
    rc = run_study4.main(["--run", "--prereg", str(tmp_path / "none.json")])
    assert rc == 3
    assert "REFUSED" in capsys.readouterr().err


def test_runner_check_with_valid_prereg_exits_zero(tmp_path, capsys):
    p = _make(tmp_path, code_sha256={"run_study4.py": __import__("hashlib").sha256(
        (ROOT / "run_study4.py").read_bytes()).hexdigest()})
    assert run_study4.main(["--check", "--prereg", str(p)]) == 0


def test_runner_run_is_never_implemented(tmp_path, capsys):
    # Even with a valid locked prereg, the runner must not pretend to run Study 4.
    p = _make(tmp_path, code_sha256={"run_study4.py": __import__("hashlib").sha256(
        (ROOT / "run_study4.py").read_bytes()).hexdigest()})
    assert run_study4.main(["--run", "--prereg", str(p)]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_no_study4_prereg_is_committed():
    # Locking is a human decision. The repo must not contain a locked Study 4 prereg by accident.
    assert not PREREG_PATH.exists()
