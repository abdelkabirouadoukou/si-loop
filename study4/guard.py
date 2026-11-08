"""study4/guard.py: refuses to run Study 4 unless a LOCKED pre-registration exists and still matches the code.

A pre-registration is accepted only if all of these hold:
  1. results/study4/prereg.json exists and is valid JSON;
  2. its "status" is "locked" (drafts are refused);
  3. its "prereg_sha256" equals the SHA-256 of the sorted-JSON body (same scheme as Study 3);
  4. every file hash in "code_sha256" matches the file on disk now;
  5. its "kernel_digest" matches the current kernel (KernelGuard).

Nothing here reads API keys, and nothing here writes files. Locking a pre-registration is a human decision
and is not done by this code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = ROOT / "results" / "study4" / "prereg.json"
REQUIRED_KEYS = ("study", "status", "question", "primary_metric", "arms_run", "code_sha256", "kernel_digest",
                 "stop_rules", "analysis_rules", "prereg_sha256")


class PreregError(RuntimeError):
    """The Study 4 pre-registration is missing, unlocked, tampered with, or out of date."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def body_sha(prereg: dict) -> str:
    body = {k: v for k, v in prereg.items() if k != "prereg_sha256"}
    return _sha_bytes(json.dumps(body, sort_keys=True).encode())


def verify(path: Path = PREREG_PATH, root: Path = ROOT) -> dict:
    """Return the verified pre-registration, or raise PreregError saying exactly what failed."""
    if not path.exists():
        raise PreregError(f"no pre-registration at {path}: Study 4 cannot run before one is locked")
    try:
        prereg = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PreregError(f"pre-registration is not valid JSON: {e}") from e
    missing = [k for k in REQUIRED_KEYS if k not in prereg]
    if missing:
        raise PreregError(f"pre-registration lacks required keys: {missing}")
    if prereg["study"] != "study4":
        raise PreregError(f"pre-registration is for {prereg['study']!r}, not 'study4'")
    if prereg["status"] != "locked":
        raise PreregError(f"pre-registration status is {prereg['status']!r}; only 'locked' may run")
    if body_sha(prereg) != prereg["prereg_sha256"]:
        raise PreregError("pre-registration body does not match its prereg_sha256: the file was edited after locking")
    for rel, expected in prereg["code_sha256"].items():
        f = root / rel
        if not f.exists():
            raise PreregError(f"code file named in pre-registration is missing: {rel}")
        if _sha_bytes(f.read_bytes()) != expected:
            raise PreregError(f"code file changed since locking: {rel}")
    from kernel.orchestrator import KernelGuard
    current = KernelGuard(root / "kernel").digest()
    if current != prereg["kernel_digest"]:
        raise PreregError("kernel digest differs from the locked pre-registration")
    return prereg
