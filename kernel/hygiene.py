"""kernel/hygiene.py: heartbeat, dead-man-switch watchdog, worktree garbage collection, and the
model/environment fingerprint that is logged with every run.

CLI (external watchdog, run it from cron/launchd on ANOTHER process or machine):
    python -m kernel.hygiene watchdog --run-dir runs/study2 --max-age 900 [--kill]
    python -m kernel.hygiene verify-ledger --run-dir runs/study2
    python -m kernel.hygiene gc --repo agent_repo --worktrees runs/study2/worktrees
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GC_PREFIXES = ("cand/", "eval/", "credit/", "yield/")


# =============================================================== heartbeat / watchdog

class Heartbeat:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, cycle: int, status: str = "running") -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"t": time.time(), "pid": os.getpid(), "cycle": cycle, "status": status}))
        os.replace(tmp, self.path)

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def age(self) -> float:
        d = self.read()
        return time.time() - d["t"] if d else float("inf")


def watchdog_check(run_dir: Path, max_age_s: float, kill: bool = False) -> dict:
    """Dead-man switch. Stale heartbeat while status is 'running' -> latch HALT (and optionally
    SIGTERM the pid). Meant to run OUTSIDE the loop's process tree and trust domain."""
    run_dir = Path(run_dir)
    hb = Heartbeat(run_dir / "heartbeat.json")
    d = hb.read()
    res = {"age_s": hb.age(), "status": d.get("status"), "action": "none"}
    if d and d.get("status") == "running" and hb.age() > max_age_s:
        (run_dir / "HALT").write_text(json.dumps({"name": "watchdog", "severity": "halt",
                                                  "detail": f"heartbeat stale for {hb.age():.0f}s"}))
        res["action"] = "halt_latched"
        if kill and d.get("pid"):
            try:
                os.kill(int(d["pid"]), signal.SIGTERM)
                res["action"] = "halt_latched+sigterm"
            except (ProcessLookupError, PermissionError):
                pass
    return res


# =============================================================== fingerprint

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint(model: dict, kernel_digest: str = "") -> dict:
    """Everything that could silently change results. Logged at start and compared on resume: a
    changed model or kernel invalidates the A/A calibration, so a mismatch halts the run."""
    try:
        import pytest
        pt = pytest.__version__
    except Exception:
        pt = "none"
    env = {"python": platform.python_version(), "platform": platform.platform(), "pytest": pt,
           "git": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()}
    core = {"model": dict(sorted(model.items())), "kernel_digest": kernel_digest}
    return {**core, "env": env, "fingerprint": hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]}


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def dir_size(path: Path) -> int:
    tot = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


# =============================================================== worktree GC

class WorktreeGC:
    """Removes what a crashed or interrupted cycle left behind:
       orphan directories, stale worktrees/branches (cand/ eval/ credit/ yield/), archive refs whose
       archive entry was evicted, and (oldest first) anything over the disk cap."""

    def __init__(self, repo, max_age_s: float = 3600.0, max_gb: float = 5.0):
        self.repo, self.max_age_s, self.max_bytes = repo, max_age_s, int(max_gb * 2 ** 30)

    def collect(self, active=(), live_archive_cycles=None, now=None) -> dict:
        now, active = now or time.time(), set(active)
        rep = {"orphan_dirs": 0, "stale_worktrees": 0, "stale_branches": 0, "archive_refs": 0, "over_cap": 0, "freed_mb": 0.0}
        before = dir_size(self.repo.worktrees)
        registered = {p.resolve() for p in self.repo.worktree_paths()}
        for d in list(self.repo.worktrees.iterdir()) if self.repo.worktrees.exists() else []:
            if d.is_dir() and d.resolve() not in registered:
                shutil.rmtree(d, ignore_errors=True)
                rep["orphan_dirs"] += 1
        for br in self.repo.branches():
            if br.startswith(GC_PREFIXES) and br not in active:
                p = self.repo.wt_path(br)
                age = now - p.stat().st_mtime if p.exists() else float("inf")
                if age >= self.max_age_s:
                    rep["stale_worktrees"] += int(p.exists())
                    self.repo.discard(br)
                    rep["stale_branches"] += 1
        self.repo._git("worktree", "prune", check=False)
        if live_archive_cycles is not None:
            for br in self.repo.branches():
                if br.startswith("archive/"):
                    try:
                        cyc = int(br.split("/", 1)[1])
                    except ValueError:
                        continue
                    if cyc not in live_archive_cycles:
                        self.repo._git("branch", "-D", br, check=False)
                        rep["archive_refs"] += 1
        if dir_size(self.repo.worktrees) > self.max_bytes:                 # disk cap: oldest first
            entries = sorted((p for p in self.repo.worktrees.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            for p in entries:
                if dir_size(self.repo.worktrees) <= self.max_bytes:
                    break
                br = next((b for b in self.repo.branches() if self.repo.wt_path(b) == p), None)
                if br in active:
                    continue
                if br:
                    self.repo.discard(br)
                else:
                    shutil.rmtree(p, ignore_errors=True)
                rep["over_cap"] += 1
        self.repo._git("gc", "--auto", "-q", check=False)
        rep["freed_mb"] = round((before - dir_size(self.repo.worktrees)) / 2 ** 20, 2)
        return rep


# =============================================================== CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernel.hygiene")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watchdog")
    w.add_argument("--run-dir", required=True)
    w.add_argument("--max-age", type=float, default=900)
    w.add_argument("--kill", action="store_true")
    v = sub.add_parser("verify-ledger")
    v.add_argument("--run-dir", required=True)
    g = sub.add_parser("gc")
    g.add_argument("--repo", required=True)
    g.add_argument("--worktrees", required=True)
    g.add_argument("--max-age", type=float, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "watchdog":
        print(json.dumps(watchdog_check(Path(a.run_dir), a.max_age, a.kill)))
        return 0
    if a.cmd == "verify-ledger":
        from kernel.ledger import GENESIS, HashChainLedger, LedgerTampered
        try:
            led = HashChainLedger(Path(a.run_dir) / "ledger.jsonl", verify_on_open=False)
            ok, n, bad = led.verify()
        except LedgerTampered as e:                                       # pragma: no cover
            print(e)
            return 2
        print(json.dumps({"ok": ok, "records": n, "first_bad": bad, "head": led.head() if n else {"hash": GENESIS}}))
        return 0 if ok else 2
    from kernel.orchestrator import GitRepo
    repo = GitRepo(Path(a.repo), Path(a.worktrees))
    print(json.dumps(WorktreeGC(repo, max_age_s=a.max_age).collect()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
