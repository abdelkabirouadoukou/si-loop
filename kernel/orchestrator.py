"""kernel/orchestrator.py: the SI-Loop v2 kernel.

Trust boundary
    KERNEL (this package, the sealed generators, the verifier): immutable to the agent.
    AGENT  (agent_repo/agent/**): the only subtree the meta-agent may change.

Atomicity: every candidate lives on its own branch in its own `git worktree`. Rollback is
"delete worktree + branch"; the champion branch (`main`) is only ever moved by a
fast-forward (or a champion-vs-candidate re-gated reset) after the gate says "accept".

Two interfaces must be supplied by the experiment (they hold the sealed problem families
and the LLM): `Verifier` and `MetaAgent`. Everything else is implemented here.

CLI:  python -m kernel.orchestrator calibrate --n 200 --p 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kernel.ledger import HashChainLedger
from kernel.patching import PatchProposal, PatchValidator, apply_proposal, collect_tests
from kernel.stats import p_better  # noqa: F401  (re-exported: the exact tail lives in stats.py)

log = logging.getLogger("si_loop")


# =============================================================== config / errors

@dataclass
class Config:
    repo_dir: Path                           # agent repo; champion lives on `main`
    runs_dir: Path
    seed: int = 0
    families: tuple = ()                     # problem families -> behaviour descriptor axes
    n_selection: int = 64                    # sealed selection problems per cycle
    n_credit: int = 32                       # sealed credit-split problems per attribution
    min_pairs: int = 20                      # refuse to gate on fewer paired problems
    eps_noninferior: float = 0.02            # tolerated pass@1 loss (non-inferiority margin)
    z_one_sided: float = 1.645               # one-sided 95% bound
    p_improve_min: float = 0.95              # P(better | discordant pairs); A/A-calibrated
    novelty_min: float = 0.35                # stepping-stone admission threshold
    p_exploit: float = 0.5                   # P(parent = champion) vs archive sampling
    cycle_wall_s: int = 1800
    eval_wall_s: int = 900
    selftest_wall_s: int = 300
    mem_mb: int = 4096
    token_budget: int = 400_000
    max_consecutive_failures: int = 8
    cooloff_s: int = 600
    credit_enabled: bool = True
    credit_max_exact: int = 6                # exact Shapley up to this many hunks, else Monte Carlo
    credit_mc_perms: int = 24
    container_image: Optional[str] = None    # None -> local rlimit sandbox (POSIX only)
    selftest_cmd: tuple = ("python", "-m", "pytest", "-q", "-x", "agent/tests", "-p", "no:cacheprovider")

    # ---- two-stage judge (screen on batch A, confirm on independent batch B)
    n_confirm: int = 64                      # confirmation batch size (independent of the screen batch)
    screen_chunks: int = 4                   # screen is evaluated in chunks so SPRT can stop early
    screen_p_improve: float = 0.80           # lenient screen threshold; the STRICT test is on batch B
    harm_alpha: float = 0.01                 # sequential harm stop: e-value >= 1/harm_alpha
    sprt_theta: float = 0.25                 # futility SPRT alternative: P(fix | discordant) = 0.5 + theta
    sprt_beta: float = 0.20                  # futility SPRT type-II error
    sprt_min_discordant: int = 6
    sequential_stop: bool = True
    utility_eps: float = 0.005               # non-inferiority margin on the UTILITY path (tighter than eps)
    loc_factor: float = 2.0                  # per-patch / champion LOC cap = max(factor*seed, seed+loc_abs)
    loc_abs: int = 400
    fdr_mode: str = "monitor"                # off | monitor | enforce   (e-LOND online FDR over confirms)
    fdr_alpha: float = 0.05
    min_kill_rate: float = 0.85              # verifier health gate (mutation kill rate) before a run

    # ---- anchor, margin budget, audits
    margin_budget: float = 0.03              # total cumulative loss the utility path may spend
    margin_min_charge: float = 0.005           # = utility_eps: a utility accept always pays what it may hide
    anchor_every: int = 20                   # cycles between anchor audits (0 = off)
    n_anchor: int = 96
    anchor_tolerance: float = 0.02
    heldout_every: int = 40                  # cycles between held-out-family audits (0 = off)
    n_heldout: int = 96
    heldout_gap_alarm: float = 0.15
    thresholdout_scale: float = 0.01

    # ---- parent selection / operator bandit
    parent_policy: str = "clade"             # clade | archive
    clade_gamma: float = 0.8
    bandit_enabled: bool = True

    # ---- circuit breakers / autonomy
    breaker_window: int = 40
    expected_accept_rate: float = 0.25
    accept_anomaly_p: float = 1e-3
    accept_halt_p: float = 1e-8
    cost_spike_factor: float = 4.0
    run_token_cap: int = 0                   # 0 = unlimited
    max_security_flags: int = 2
    escape_attempt_limit: int = 10
    audit_halt_n: int = 3
    min_free_gb: float = 1.0
    gc_every: int = 10
    gc_max_age_s: float = 3600.0
    gc_max_gb: float = 5.0
    halt_on_model_change: bool = True
    model: dict = field(default_factory=lambda: {"provider": "none", "model": "none"})

    def __post_init__(self):
        self.repo_dir, self.runs_dir = Path(self.repo_dir).resolve(), Path(self.runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)


class GuardViolation(Exception):
    """A safety rule was broken (static check, tampering, protocol violation)."""


class BudgetExceeded(Exception):
    """Wall-clock or token budget exhausted."""


# =============================================================== budgets & sandbox

class Budget:
    """Per-cycle wall-clock + token budget. Checked cooperatively; the sandbox additionally
    kills runaway subprocesses, so a stuck agent cannot outlive its cycle."""

    def __init__(self, wall_s: float, tokens: int):
        self.deadline, self.tokens_left = time.monotonic() + wall_s, tokens

    def charge(self, tokens: int = 0):
        self.tokens_left -= tokens
        if self.tokens_left < 0:
            raise BudgetExceeded("token budget exhausted")
        self.check()

    def check(self):
        if time.monotonic() > self.deadline:
            raise BudgetExceeded("cycle wall-clock budget exhausted")

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float


# Sets rlimits then replaces itself with the target command (no preexec_fn: thread-safe).
_BOOT = (
    "import os, sys, resource\n"
    "mem, cpu, fsz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])\n"
    "for r, v in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu), (resource.RLIMIT_FSIZE, fsz)):\n"
    "    try:\n"
    "        resource.setrlimit(r, (v, v))\n"
    "    except (ValueError, OSError):\n"
    "        pass\n"
    "os.execvp(sys.argv[4], sys.argv[4:])\n"
)


class Sandbox:
    """Local mode: own session, rlimits (address space, CPU seconds, file size), scrubbed
    environment, whole process group SIGKILLed on timeout. RLIMIT_AS is unreliable on macOS.
    Container mode: docker/podman with --network none, read-only root, tmpfs /tmp, memory,
    cpu and pids caps, and sealed/kernel paths mounted read-only.

    `run`   : one-shot command, captured output.
    `popen` : long-lived child with raw pipes (the sealed verifier's RPC workers)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _build(self, cmd: list, cwd: Path, wall_s: float, ro_mounts: tuple, env_extra: Optional[dict]):
        cwd = Path(cwd).resolve()
        extra = dict(env_extra or {})
        pypath = extra.pop("PYTHONPATH", str(cwd))
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(cwd), "PYTHONPATH": pypath,
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8", **extra}
        if self.cfg.container_image:
            argv = ["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m",
                    "--memory", f"{self.cfg.mem_mb}m", "--cpus", "2", "--pids-limit", "256",
                    "-v", f"{cwd}:{cwd}:rw", "-w", str(cwd),
                    *itertools.chain.from_iterable(["-e", f"{k}={v}"] for k, v in env.items() if k != "PATH"),
                    *itertools.chain.from_iterable(["-v", f"{m}:{m}:ro"] for m in ro_mounts),
                    self.cfg.container_image, *cmd]
        else:
            real = [sys.executable if (i == 0 and c == "python") else str(c) for i, c in enumerate(cmd)]
            argv = [sys.executable, "-c", _BOOT, str(self.cfg.mem_mb * 2**20), str(int(wall_s) + 5),
                    str(256 * 2**20), *real]
        return argv, env, cwd

    @staticmethod
    def kill(p: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                p.kill()
            except ProcessLookupError:
                pass

    def run(self, cmd: list, cwd: Path, wall_s: float, ro_mounts: tuple = (),
            env_extra: Optional[dict] = None) -> RunResult:
        wall_s = max(1.0, float(wall_s))
        argv, env, cwd = self._build(cmd, cwd, wall_s, ro_mounts, env_extra)
        t0 = time.monotonic()
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, errors="replace", start_new_session=True)
        try:
            out, err = p.communicate(timeout=wall_s)
            return RunResult(p.returncode, out[-20000:], err[-20000:], False, time.monotonic() - t0)
        except subprocess.TimeoutExpired:
            self.kill(p)
            out, err = p.communicate()
            return RunResult(-9, (out or "")[-20000:], (err or "")[-20000:], True, time.monotonic() - t0)

    def popen(self, cmd: list, cwd: Path, wall_s: float, ro_mounts: tuple = (),
              env_extra: Optional[dict] = None, stderr=None) -> subprocess.Popen:
        """Long-lived sandboxed child. stdin/stdout are RAW binary pipes (bufsize=0)."""
        argv, env, cwd = self._build(cmd, cwd, max(1.0, float(wall_s)), ro_mounts, env_extra)
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr if stderr is not None else subprocess.DEVNULL,
                                bufsize=0, start_new_session=True)


# =============================================================== git (atomic branches)

class GitRepo:
    """Thin, thread-safe wrapper over the git CLI. The root checkout stays on `main` and is
    never edited by anyone but the kernel."""

    IDENT = ("-c", "user.name=si-loop", "-c", "user.email=si-loop@localhost")

    def __init__(self, root: Path, worktrees: Path):
        self.root, self.worktrees = Path(root).resolve(), Path(worktrees).resolve()
        self.worktrees.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.active: set = set()                 # branches with a live worktree (GC must not touch them)

    def worktree_paths(self) -> list:
        out = self._git("worktree", "list", "--porcelain", check=False)
        return [Path(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("worktree ")]

    def branches(self) -> list:
        return [b for b in self._git("for-each-ref", "--format=%(refname:short)", "refs/heads/").splitlines() if b]

    def exists(self, sha: str) -> bool:
        with self._lock:
            return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=self.root,
                                  capture_output=True).returncode == 0

    def _git(self, *args, cwd: Optional[Path] = None, check: bool = True) -> str:
        with self._lock:
            r = subprocess.run(["git", *args], cwd=cwd or self.root, capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    def init(self, message: str = "seed") -> str:
        """Initialise `root` as a repo on `main` with everything committed (first-time setup)."""
        self._git("init", "-q", "-b", "main")
        self._git("add", "-A")
        self._git(*self.IDENT, "commit", "-q", "-m", message)
        return self.head()

    def head(self, ref: str = "main") -> str:
        return self._git("rev-parse", ref)

    def wt_path(self, branch: str) -> Path:
        return self.worktrees / branch.replace("/", "_")

    def add_worktree(self, branch: str, base_sha: str) -> Path:
        path = self.wt_path(branch)
        self._git("worktree", "add", "-q", "-b", branch, str(path), base_sha)
        self.active.add(branch)
        return path

    def is_clean(self, wt: Path) -> bool:
        return self._git("status", "--porcelain", cwd=wt) == ""

    def commit_all(self, wt: Path, msg: str) -> str:
        self._git("add", "-A", cwd=wt)
        self._git(*self.IDENT, "commit", "-q", "-m", msg, cwd=wt)
        return self._git("rev-parse", "HEAD", cwd=wt)

    def changed_files(self, base: str, head: str) -> set:
        out = self._git("diff", "--name-only", base, head)
        return {line for line in out.splitlines() if line}

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        with self._lock:
            return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant],
                                  cwd=self.root, capture_output=True).returncode == 0

    def discard(self, branch: str, keep_branch: bool = False):
        """Hard rollback. Idempotent: safe from crash recovery and from `finally` blocks."""
        path = self.wt_path(branch)
        self.active.discard(branch)
        self._git("worktree", "remove", "--force", str(path), check=False)
        self._git("worktree", "prune", check=False)
        if not keep_branch:
            self._git("branch", "-D", branch, check=False)

    def ff_merge(self, branch: str):
        self._git("merge", "--ff-only", "-q", branch)

    def move_main(self, sha: str):
        """Switch the champion lineage (candidate does not descend from current main)."""
        self._git("reset", "--hard", "-q", sha)

    def tag(self, name: str, sha: Optional[str] = None):
        self._git("tag", "-f", name, *([sha] if sha else []))

    def archive_ref(self, branch: str, name: str):
        self._git("branch", "-f", f"archive/{name}", branch)


class KernelGuard:
    """Detects tampering with the immutable kernel while a run is in progress."""

    def __init__(self, kernel_dir: Path):
        self.dir = Path(kernel_dir)
        self.manifest = self.digest()

    def digest(self) -> str:
        h = hashlib.sha256()
        for f in sorted(self.dir.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(self.dir).as_posix().encode() + b"\0" + f.read_bytes())
        return h.hexdigest()

    def assert_intact(self):
        if self.digest() != self.manifest:
            raise GuardViolation("kernel digest changed during run: aborting")


Ledger = HashChainLedger          # append-only, fsynced, hash-chained (see kernel/ledger.py)


# =============================================================== judge: results, gate, calibration

@dataclass
class EvalResult:
    passes: dict                              # problem_id -> hidden-pytest pass (HARD signal)
    metrics: dict = field(default_factory=dict)          # tokens, wall_s, agent_loc, ...
    family_profile: dict = field(default_factory=dict)   # family -> pass rate (descriptor input)
    flags: list = field(default_factory=list)            # SECURITY flags (honeypot touched, tamper...): kernel-only

    @property
    def pass_at_1(self) -> float:
        return sum(self.passes.values()) / max(1, len(self.passes))


@dataclass
class GateDecision:
    verdict: str                              # accept | stepping_stone | reject
    delta: float
    lower: float
    p_improve: float
    b: int
    c: int
    reasons: list
    path: str = ""                            # correctness | utility | "" (set on accept)


class Gate:
    """Paired non-inferiority + improvement test on binary hidden-test outcomes.

    On the SAME sealed problems, b = problems the patch fixed, c = problems it broke.
      delta = (b - c) / n,  se = sqrt(b + c - (b - c)^2 / n) / n,  lower = delta - z * se
    accept          : lower >= -eps  AND delta > 0 AND p_better(b, c) >= p_improve_min
                      (or non-inferior, delta >= 0 and measurably cheaper/smaller)
    stepping_stone  : non-inferior and behaviourally novel (kept in the archive, not promoted)
    reject          : otherwise
    Utility (size/tokens) never outvotes correctness: non-inferiority is a hard prerequisite."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def decide(self, parent: EvalResult, cand: EvalResult, novelty: float) -> GateDecision:
        ids = sorted(parent.passes)
        if ids != sorted(cand.passes):
            raise GuardViolation("parent/candidate evaluated on different problems")
        n = len(ids)
        if n < self.cfg.min_pairs:
            raise GuardViolation(f"only {n} paired problems (< min_pairs={self.cfg.min_pairs})")
        b = sum(1 for i in ids if cand.passes[i] and not parent.passes[i])
        c = sum(1 for i in ids if parent.passes[i] and not cand.passes[i])
        delta = (b - c) / n
        se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
        lower = delta - self.cfg.z_one_sided * se
        p_imp = p_better(b, c)
        pm, cm = parent.metrics, cand.metrics
        cheaper = (cm.get("agent_loc", math.inf) <= 0.95 * pm.get("agent_loc", 0.0)
                   or cm.get("tokens", math.inf) <= 0.90 * pm.get("tokens", 0.0))
        why = [f"b={b} c={c} n={n}", f"delta={delta:+.3f} lower={lower:+.3f} p_better={p_imp:.3f}"]

        def dec(v, extra, path=""):
            return GateDecision(v, delta, lower, p_imp, b, c, why + [extra], path)

        if lower < -self.cfg.eps_noninferior:
            return dec("reject", "not non-inferior")
        if delta > 0 and p_imp >= self.cfg.p_improve_min:
            return dec("accept", "correctness gain", "correctness")
        if delta >= 0 and cheaper and lower >= -self.cfg.utility_eps:      # tighter margin on the utility path
            return dec("accept", "efficiency gain at non-inferior correctness", "utility")
        if novelty >= self.cfg.novelty_min:
            return dec("stepping_stone", f"novelty={novelty:.2f}")
        return dec("reject", "no demonstrated gain")


def simulate_false_accept(cfg: Config, n: int, p: float, noise: float = 0.1,
                          trials: int = 2000, seed: int = 0) -> float:
    """A/A calibration: fraction of NULL patches (identical true pass rate, independent
    run-to-run noise) the gate would accept. Pre-register the target (e.g. <= 5%)."""
    rng, gate, acc = random.Random(seed), Gate(cfg), 0
    for _ in range(trials):
        u = [rng.random() for _ in range(n)]

        def run():
            return EvalResult({str(i): (u[i] + rng.gauss(0, noise)) < p for i in range(n)},
                              {"agent_loc": 100, "tokens": 100})
        acc += gate.decide(run(), run(), novelty=0.0).verdict == "accept"
    return acc / trials


def calibrate_gate(cfg: Config, n: int, p: float, noise: float = 0.1, target: float = 0.05,
                   grid=(0.80, 0.90, 0.95, 0.975, 0.99, 0.995), trials: int = 1000):
    """Smallest p_improve_min whose A/A false-accept rate is <= target. Returns (value, table)."""
    table, chosen = [], grid[-1]
    for g in grid:
        trial_cfg = Config(**{**cfg.__dict__, "p_improve_min": g})
        fa = simulate_false_accept(trial_cfg, n, p, noise, trials)
        table.append((g, fa))
    for g, fa in table:
        if fa <= target:
            chosen = g
            break
    return chosen, table


def simulate_false_accept_two_stage(cfg: Config, n_a: int, n_b: int, p: float, noise: float = 0.1,
                                    trials: int = 2000, seed: int = 0) -> float:
    """A/A calibration of the TWO-STAGE judge: a null patch must pass the lenient screen on batch A AND the
    strict gate on the independent batch B. (Sequential early stops only remove accepts, so ignoring them
    is conservative.)"""
    rng = random.Random(seed)
    screen = Gate(Config(**{**cfg.__dict__, "p_improve_min": cfg.screen_p_improve}))
    strict, acc = Gate(cfg), 0

    def stage(gate, n):
        u = [rng.random() for _ in range(n)]

        def run():
            return EvalResult({str(i): (u[i] + rng.gauss(0, noise)) < p for i in range(n)},
                              {"agent_loc": 100, "tokens": 100})
        return gate.decide(run(), run(), novelty=0.0).verdict == "accept"
    for _ in range(trials):
        acc += stage(screen, n_a) and stage(strict, n_b)
    return acc / trials


def calibrate_two_stage(cfg: Config, n_a: int, n_b: int, p: float, noise: float = 0.1, target: float = 0.05,
                        grid=(0.80, 0.90, 0.95, 0.975, 0.99), trials: int = 1000):
    table = [(g, simulate_false_accept_two_stage(Config(**{**cfg.__dict__, "p_improve_min": g}), n_a, n_b, p, noise, trials))
             for g in grid]
    chosen = next((g for g, fa in table if fa <= target), grid[-1])
    return chosen, table


# =============================================================== interfaces supplied by the experiment

class Verifier(ABC):
    """Sealed judge. Implementations MUST: run hidden pytest in the judge process, run candidate
    code in a separate sandboxed process behind an RPC `solve()` proxy (so it cannot patch pytest,
    write the results file, exit(0) or read the oracle), derive per-cycle problem ids from a secret
    seed by HMAC, and expose nothing beyond EvalResult (never problem content)."""

    @abstractmethod
    def draw_batch(self, cycle: int, n: int, split: str = "selection") -> list: ...

    @abstractmethod
    def evaluate(self, worktree: Path, batch: list, wall_s: float) -> EvalResult: ...


class MetaAgent(ABC):
    """Proposer. May READ the worktree (and run public dev tests in its own sandbox) but must not
    modify it: the kernel rejects a dirty worktree. Returns a PatchProposal (see parse_proposal)."""

    @abstractmethod
    def propose(self, worktree: Path, parent_sha: str, broadcast: list, budget: Budget) -> PatchProposal: ...


# =============================================================== exploration archive

class Archive:
    """Quality-diversity archive of agent versions (stepping stones). Persisted as JSON.
    Parent selection: champion with prob p_exploit, else sample proportional to
    (0.05 + score) * (0.5 + novelty) / (1 + visits)."""

    def __init__(self, path: Path, seed: int = 0, k: int = 3, p_exploit: float = 0.5, max_size: int = 200):
        self.path, self.k, self.p_exploit, self.max_size = Path(path), k, p_exploit, max_size
        self.rng, self._lock = random.Random(seed), threading.Lock()
        self.entries: list = json.loads(self.path.read_text()) if self.path.exists() else []

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries))
        os.replace(tmp, self.path)

    @staticmethod
    def _dist(a: list, b: list) -> float:
        m = min(len(a), len(b))
        return sum(abs(x - y) for x, y in zip(a, b)) / m if m else 0.0

    def novelty(self, d: list, exclude: Optional[str] = None) -> float:
        with self._lock:
            ds = sorted(self._dist(d, e["descriptor"]) for e in self.entries if e["sha"] != exclude)
        return 1.0 if not ds else sum(ds[:self.k]) / min(self.k, len(ds))

    def select_parent(self, champion_sha: str) -> str:
        with self._lock:
            if not self.entries or self.rng.random() < self.p_exploit:
                return champion_sha
            ws = []
            for e in self.entries:
                others = sorted(self._dist(e["descriptor"], o["descriptor"]) for o in self.entries if o is not e)
                nov = sum(others[:self.k]) / max(1, min(self.k, len(others))) if others else 1.0
                ws.append((0.05 + e["score"]) * (0.5 + nov) / (1 + e["visits"]))
            e = self.rng.choices(self.entries, weights=ws)[0]
            e["visits"] += 1
            self._save()
            return e["sha"]

    def admit(self, sha: str, descriptor: list, score: float, verdict: str, cycle: int):
        with self._lock:
            if any(e["sha"] == sha for e in self.entries):
                return
            self.entries.append({"sha": sha, "descriptor": list(descriptor), "score": float(score),
                                 "verdict": verdict, "cycle": cycle, "visits": 0})
            if len(self.entries) > self.max_size:
                self.entries.remove(min(self.entries, key=lambda e: e["score"]))
            self._save()


# =============================================================== Mom Center (workspace memory)

_CTRL = re.compile(r"[^\x20-\x7e\n]")
_DANGEROUS = re.compile(r"[`<>{}]|ignore (all|previous)|system prompt|tool_call", re.I)


def sanitize(value, limit: int = 300):
    """Everything agent- or sandbox-derived is untrusted data: strip control/markup characters
    and obvious injection phrases, cap length. Numbers pass through unchanged."""
    if isinstance(value, str):
        return _DANGEROUS.sub("?", _CTRL.sub("", value))[:limit]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [sanitize(v, limit) for v in value[:20]]
    if isinstance(value, dict):
        return {sanitize(str(k), 40): sanitize(v, limit) for k, v in list(value.items())[:20]}
    return sanitize(str(value), limit)


class MomCenter:
    """Shared workspace: scored, expiring, outcome-trusted entries; capacity-limited broadcast.
    salience = 0.35*trust + 0.30*utility + 0.20*recency + 0.15*verified, then greedy MMR
    (tag overlap penalty 0.3) so one pattern cannot fill every slot. Trust is a Beta posterior
    over 'did a cycle that received this entry improve?'."""

    def __init__(self, db_path: Path, ttl: int = 50):
        self.ttl = ttl
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, payload TEXT, tags TEXT,
            verified INTEGER, score REAL, a REAL DEFAULT 1, b REAL DEFAULT 1, cycle INTEGER)""")
        self.db.commit()

    def record(self, kind: str, payload: dict, verified: bool, score: float, cycle: int, tags=()) -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO entries(kind,payload,tags,verified,score,cycle) VALUES (?,?,?,?,?,?)",
                (kind, json.dumps(sanitize(payload)), json.dumps(sorted(set(map(str, tags)))),
                 int(verified), float(score), cycle))
            self.db.commit()
            return cur.lastrowid

    def broadcast(self, cycle: int, k: int = 5) -> list:
        with self._lock:
            rows = self.db.execute("SELECT id,kind,payload,tags,verified,score,a,b,cycle FROM entries "
                                   "WHERE ? - cycle <= ?", (cycle, self.ttl)).fetchall()
        cands = []
        for id_, kind, payload, tags, ver, score, a, b, cyc in rows:
            util = 0.5 + 0.5 * math.tanh(5 * score)
            rec = max(0.0, 1 - (cycle - cyc) / self.ttl)
            sal = 0.35 * a / (a + b) + 0.30 * util + 0.20 * rec + 0.15 * ver
            cands.append((sal, id_, kind, json.loads(payload), set(json.loads(tags))))
        chosen: list = []
        while cands and len(chosen) < k:
            def mmr(c):
                overlap = max((len(c[4] & s[4]) / max(1, len(c[4] | s[4])) for s in chosen), default=0.0)
                return c[0] - 0.3 * overlap
            best = max(cands, key=mmr)
            cands.remove(best)
            chosen.append(best)
        return [{"id": c[1], "kind": c[2], "payload": c[3], "tags": sorted(c[4]),
                 "salience": round(c[0], 3), "untrusted": True} for c in chosen]

    def feedback(self, entry_ids: list, helped: bool):
        if not entry_ids:
            return
        col = "a" if helped else "b"
        with self._lock:
            self.db.executemany(f"UPDATE entries SET {col} = {col} + 1 WHERE id = ?", [(i,) for i in entry_ids])
            self.db.commit()


# =============================================================== credit assignment (Shapley over hunks)

class ShapleyAttributor:
    """Shapley values over the hunks of a patch. v(S) = pass@1 on the sealed CREDIT split of the
    parent with only the ops in S applied; a coalition whose ops cannot be applied scores 0.
    Exact for <= credit_max_exact hunks (efficiency: sum(phi) = v(N) - v(empty)); otherwise
    seeded Monte Carlo over permutations. The agent only ever sees hunk-level phi values."""

    def __init__(self, cfg: Config, repo: GitRepo, verifier: Verifier):
        self.cfg, self.repo, self.verifier = cfg, repo, verifier

    def _value(self, parent_sha: str, proposal: PatchProposal, batch: list, coalition: frozenset,
               cache: dict) -> float:
        if coalition in cache:
            return cache[coalition]
        branch = f"credit/{proposal.id}-{'-'.join(map(str, sorted(coalition))) or 'none'}-{time.time_ns()}"
        wt = None
        try:
            wt = self.repo.add_worktree(branch, parent_sha)
            apply_proposal(wt, proposal, only=set(coalition))
            val = self.verifier.evaluate(wt, batch, self.cfg.eval_wall_s).pass_at_1
        except Exception as e:                      # inapplicable coalition => 0 (documented)
            log.debug("coalition %s failed: %s", sorted(coalition), e)
            val = 0.0
        finally:
            self.repo.discard(branch)
        cache[coalition] = val
        return val

    def attribute(self, parent_sha: str, proposal: PatchProposal, batch: list) -> dict:
        n, cache = len(proposal.ops), {}
        ids = [f"{proposal.id}:{i}" for i in range(n)]
        val = lambda s: self._value(parent_sha, proposal, batch, frozenset(s), cache)
        phi = [0.0] * n
        if n <= self.cfg.credit_max_exact:
            fact = math.factorial
            for i in range(n):
                rest = [j for j in range(n) if j != i]
                for r in range(n):
                    w = fact(r) * fact(n - r - 1) / fact(n)
                    for s in itertools.combinations(rest, r):
                        phi[i] += w * (val(set(s) | {i}) - val(set(s)))
        else:
            rng = random.Random(self.cfg.seed)
            for _ in range(self.cfg.credit_mc_perms):
                order, cur, prev = list(range(n)), set(), val(set())
                rng.shuffle(order)
                for i in order:
                    cur.add(i)
                    now = val(cur)
                    phi[i] += (now - prev) / self.cfg.credit_mc_perms
                    prev = now
        return {h: round(v, 5) for h, v in zip(ids, phi)}


# =============================================================== the loop

class HaltRun(Exception):
    """A halt-severity circuit breaker tripped (or a HALT latch exists): the run stops cleanly."""


def classify(e: Exception) -> str:
    """Coarse, agent-safe error type for a failed cycle (never carries sealed information)."""
    from kernel.patching import PatchError
    if isinstance(e, PatchError):
        return "parse"
    if isinstance(e, BudgetExceeded):
        return "budget"
    if isinstance(e, GuardViolation):
        m = str(e)
        if "mixed" in m:
            return "mixed_scope"
        if "self-tests failed" in m:
            return "selftest"
        if "modified the worktree" in m or "outside the proposal" in m or "kernel digest" in m:
            return "tamper"
        return "static"
    return "internal"


COARSE_SCORE = {"accept": 0.2, "stepping_stone": 0.05, "reject": -0.05}
AGENT_VISIBLE_MSG = ("static", "parse", "selftest", "mixed_scope", "tamper")   # concern the agent's OWN code only


class Orchestrator:
    def __init__(self, cfg: Config, repo: GitRepo, sandbox: Sandbox, verifier: Verifier, meta: MetaAgent,
                 archive: Archive, attributor: ShapleyAttributor, mom: MomCenter, guard: KernelGuard,
                 lineage=None, bandit=None, yield_gate=None, breakers=None):
        from kernel.anchor import AnchorAuditor, AnchorStore, HeldoutAuditor, MarginBudget
        from kernel.bandit import OperatorBandit
        from kernel.breakers import CircuitBreakers
        from kernel.clade import LineageStore
        from kernel.hygiene import Heartbeat, WorktreeGC
        from kernel.pipeline import TwoStageJudge
        from kernel.stats import ELond
        self.cfg, self.repo, self.sb, self.verifier, self.meta = cfg, repo, sandbox, verifier, meta
        self.archive, self.attr, self.mom, self.guard = archive, attributor, mom, guard
        self.validator = PatchValidator()
        runs = cfg.runs_dir
        self.ledger = Ledger(runs / "ledger.jsonl")
        self.anchor = AnchorStore(runs / "anchor.json")
        self.budget = MarginBudget(runs / "margin.json", cfg.margin_budget, cfg.margin_min_charge)
        self.elond = ELond(cfg.fdr_alpha)
        self.judge = TwoStageJudge(cfg, repo, verifier, self.budget, self.anchor, self.elond)
        self.gate = self.judge.gate
        self.lineage = lineage or LineageStore(runs / "lineage.json", cfg.clade_gamma)
        self.bandit = bandit if bandit is not None else (OperatorBandit(runs / "bandit.json") if cfg.bandit_enabled else None)
        self.yield_gate = yield_gate
        self.breakers = breakers or CircuitBreakers(cfg, runs, self.anchor.seed_loc if self.anchor.frozen else 0.0)
        self.heartbeat, self.gc = Heartbeat(runs / "heartbeat.json"), WorktreeGC(repo, cfg.gc_max_age_s, cfg.gc_max_gb)
        self.anchor_auditor = AnchorAuditor(cfg, self.anchor, self.budget, verifier.draw_batch, self.judge.eval_sha)
        self.heldout_auditor = HeldoutAuditor(cfg, self.anchor, verifier.draw_batch, self.judge.eval_sha, cfg.seed)
        self._banned_path = runs / "banned.json"
        self.banned = set(json.loads(self._banned_path.read_text())) if self._banned_path.exists() else set()
        self._promote_lock = threading.Lock()
        self.rng = random.Random(cfg.seed)
        self.last_selection_stats = (0.0, 0.0)

    # ---- helpers
    def descriptor(self, ev: EvalResult) -> list:
        return [ev.family_profile.get(f, 0.0) for f in self.cfg.families] + \
               [ev.metrics.get("agent_loc", 0.0) / 1000, ev.metrics.get("tokens", 0.0) / 1e5]

    def novelty(self, desc: list, sha: str = None) -> float:
        return self.archive.novelty(desc, exclude=sha)

    def _eval_cached(self, sha: str, batch: list) -> EvalResult:
        return self.judge.eval_sha(sha, batch)

    def recover(self):
        """Crash recovery: discard worktrees/branches of cycles that began but never ended."""
        for rec in self.ledger.incomplete_cycles():
            self.repo.discard(rec["branch"])
            self.ledger.write(ev="end", cycle=rec["cycle"], verdict="reject", error_type="internal",
                              why=["recovered after crash"])
            log.warning("recovered crashed cycle %s", rec["cycle"])

    def bootstrap(self):
        """Freeze the cycle-0 ANCHOR (tag anchor/cycle0 + anchor.json) and seed the archive. Idempotent."""
        sha = self.repo.head("main")
        if self.anchor.frozen:
            if self.repo.head("anchor/cycle0") != self.anchor.anchor_sha:
                raise GuardViolation("anchor tag moved: refusing to run")
        else:
            ev = self._eval_cached(sha, self.verifier.draw_batch(-1, self.cfg.n_selection))
            self.anchor.freeze(sha, ev.metrics.get("agent_loc", 0.0))
            self.repo.tag("anchor/cycle0", sha)
            self.breakers.seed_loc = self.anchor.seed_loc
            self.ledger.write(ev="anchor_frozen", sha=sha, loc=self.anchor.seed_loc)
        self.lineage.add_root(sha)
        if not self.archive.entries:
            ev = self._eval_cached(sha, self.verifier.draw_batch(-1, self.cfg.n_selection))
            self.archive.admit(sha, self.descriptor(ev), ev.pass_at_1, "seed", -1)

    def _fingerprint(self, accept_change: bool):
        from kernel.hygiene import model_fingerprint
        fp = model_fingerprint(self.cfg.model, self.guard.manifest)
        prev = [r for r in self.ledger.records() if r.get("ev") == "fingerprint"]
        if prev and prev[-1]["fingerprint"] != fp["fingerprint"]:
            self.ledger.write(ev="fingerprint_changed", old=prev[-1]["fingerprint"], new=fp["fingerprint"])
            if self.cfg.halt_on_model_change and not accept_change:
                raise HaltRun("model/kernel fingerprint changed since the last run: A/A calibration is no longer valid "
                              "(rerun calibration, or pass accept_fingerprint_change=True)")
        self.ledger.write(ev="fingerprint", **fp)

    def _select_parent(self, champion: str) -> str:
        if self.cfg.parent_policy == "clade":
            sha = self.lineage.select(champion, self.rng, self.banned, self.cfg.p_exploit)
        else:
            sha = self.archive.select_parent(champion)
        return champion if (sha in self.banned or not self.repo.exists(sha)) else sha

    def _quarantine(self, sha: str, flags: list):
        self.banned.add(sha)
        self._banned_path.write_text(json.dumps(sorted(self.banned)))
        self.ledger.write(ev="security_flag", sha=sha, flags=flags[:20])

    # ---- one cycle
    def run_cycle(self, cycle: int) -> dict:
        self.guard.assert_intact()                      # tampering aborts the whole run (propagates)
        champion = self.repo.head("main")
        parent_sha = self._select_parent(champion)
        branch = f"cand/{cycle:06d}"
        budget = Budget(self.cfg.cycle_wall_s, self.cfg.token_budget)
        broadcast = self.mom.broadcast(cycle)
        if self.bandit:
            broadcast = broadcast + [{"id": -1, "kind": "operator_stats", "payload": self.bandit.coarse(), "tags": [],
                                      "salience": 1.0, "untrusted": True}]
        self.ledger.write(ev="begin", cycle=cycle, parent=parent_sha, branch=branch)
        out = {"cycle": cycle, "parent": parent_sha[:8], "verdict": "reject", "error_type": "none", "why": [],
               "security_flags": 0}
        keep, proposal, hunks, sha, j, scope = False, None, [], None, None, "solver"
        try:
            wt = self.repo.add_worktree(branch, parent_sha)
            baseline = collect_tests(wt)

            # 1. the agent proposes a structured patch (read-only access to the worktree)
            proposal = self.meta.propose(wt, parent_sha, broadcast, budget)
            proposal.parent_sha = parent_sha
            if not self.repo.is_clean(wt):
                raise GuardViolation("meta-agent modified the worktree directly (only proposals are allowed)")
            if (errs := self.validator.validate_proposal(proposal)):
                raise GuardViolation("; ".join(errs))
            meta_ops = [o.path.startswith("agent/meta/") for o in proposal.ops]
            if any(meta_ops) and not all(meta_ops):
                raise GuardViolation("mixed scope: a patch must touch either agent/meta/** or the solver, not both")
            scope = "meta" if all(meta_ops) else "solver"

            # 2. apply atomically, static safety scan, commit, verify the diff is exactly the proposal
            hunks = apply_proposal(wt, proposal)
            if (errs := self.validator.validate_tree(wt, proposal, baseline)):
                raise GuardViolation("; ".join(errs))
            sha = self.repo.commit_all(wt, f"[{proposal.id}] {proposal.hypothesis[:72]}")
            stray = self.repo.changed_files(parent_sha, sha) - {o.path for o in proposal.ops}
            if stray:
                raise GuardViolation(f"commit touches files outside the proposal: {sorted(stray)[:5]}")
            budget.check()

            # 3. L1: the agent's own tests (non-regression only; can block, never promote)
            r = self.sb.run(list(self.cfg.selftest_cmd), wt, min(self.cfg.selftest_wall_s, budget.remaining_s()))
            if r.returncode:
                raise GuardViolation(f"agent self-tests failed (timeout={r.timed_out}): "
                                     f"{(r.stdout + r.stderr)[-300:]}")

            # 4. L2 + L3: two-stage sealed judgement (or the yield test for improver patches)
            wall = lambda: max(1.0, min(self.cfg.eval_wall_s, budget.remaining_s()))
            if scope == "meta" and self.yield_gate is not None:
                j = self.yield_gate.judge(wt, sha, parent_sha, cycle, proposal)
            else:
                j = self.judge.judge(wt, sha, parent_sha, cycle, wall, self.descriptor, self.novelty)
            out.update(verdict=j.verdict, error_type=j.error_type, why=j.reasons, stage=j.stage, delta=j.delta,
                       p_better=round(j.p_better, 4), parent_pass=j.parent_pass, cand_pass=j.cand_pass,
                       detail=j.detail, scope=scope, charge=j.charge)
            if j.flags:
                out["security_flags"] = len(j.flags)
                self._quarantine(sha, j.flags)

            # 5. atomic commit-or-rollback
            desc = j.desc
            if j.verdict == "accept":
                with self._promote_lock:
                    champ = self.repo.head("main")
                    promoted = False
                    if not self.budget.can_afford(j.charge):
                        out["why"].append("margin budget exhausted by a concurrent accept")
                        out["error_type"] = "margin"
                    elif self.repo.is_ancestor(champ, sha):
                        self.repo.ff_merge(branch)
                        promoted = True
                    elif j.cand_confirm is not None:        # lineage switch: must also beat the champion
                        d2 = self.judge.confirm_vs(champ, j)
                        out["why"].append(f"vs champion: {d2.verdict}")
                        if d2.verdict == "accept":
                            self.repo.move_main(sha)
                            promoted = True
                    if promoted:
                        self.repo.tag(f"champ/{cycle:06d}")
                        self.budget.spend(j.charge, cycle)
                        self.archive.admit(sha, desc, j.cand_pass, "accept", cycle)
                        out["champion_loc"] = j.detail.get("loc", None)
                    elif out["error_type"] == "margin":
                        out["verdict"] = "reject"
                    else:
                        out["verdict"] = "stepping_stone"
                        self.repo.archive_ref(branch, f"{cycle:06d}")
                        self.archive.admit(sha, desc, j.cand_pass, "stepping_stone", cycle)
                        keep = True
            elif j.verdict == "stepping_stone":
                self.repo.archive_ref(branch, f"{cycle:06d}")
                self.archive.admit(sha, desc, j.cand_pass, "stepping_stone", cycle)
                keep = True

            # 6. credit assignment for kept solver patches (failure never undoes a promotion). The agent
            #    only ever sees the SIGN of a hunk's credit; exact values stay in the ledger.
            coarse_credit = None
            if out["verdict"] != "reject" and self.cfg.credit_enabled and scope == "solver":
                try:
                    credit = self.attr.attribute(parent_sha, proposal,
                                                 self.verifier.draw_batch(cycle, self.cfg.n_credit, "credit"))
                    out["credit"] = credit
                    coarse_credit = credit
                    for h in hunks:
                        sign = "+" if credit[h.id] > 0.01 else ("-" if credit[h.id] < -0.01 else "0")
                        self.mom.record("credit", {"tag": h.tag, "path": h.path, "effect": sign}, True,
                                        {"+": 0.1, "0": 0.0, "-": -0.1}[sign], cycle, tags=[h.tag])
                except Exception as e:
                    out["credit_error"] = f"{type(e).__name__}: {e}"
            self._record_outcome(cycle, out, proposal, hunks, parent_sha, sha, broadcast, coarse_credit)
        except (KeyboardInterrupt, SystemExit, HaltRun):
            raise
        except Exception as e:
            et = classify(e)
            out.update(verdict="reject", error_type=et, why=out["why"] + [f"{type(e).__name__}: {e}"])
            log.debug("cycle %s failed", cycle, exc_info=True)
            msg = str(e)[:120] if et in AGENT_VISIBLE_MSG else ""
            self.mom.record("failure", {"error_type": et, "message": msg}, True, COARSE_SCORE["reject"], cycle, tags=[et])
            self.mom.feedback([x["id"] for x in broadcast if x["id"] >= 0], helped=False)
            if self.bandit and hunks:
                self.bandit.update_from_cycle([h.tag for h in hunks], "reject")
            self.lineage.record_attempt(parent_sha, "reject", None, cycle)
        finally:
            out["cost_tokens"] = (self.cfg.token_budget - budget.tokens_left) + (j.detail.get("tokens", 0) if j else 0)
            self.repo.discard(branch, keep_branch=keep)     # hard rollback unless promoted/archived
            self.ledger.write(ev="end", **out)
        return out

    def _record_outcome(self, cycle, out, proposal, hunks, parent_sha, sha, broadcast, credit):
        v = out["verdict"]
        et = out["error_type"]
        self.mom.record("patch", {"hypothesis": proposal.hypothesis, "verdict": v, "error_type": et}, True,
                        COARSE_SCORE.get(v, -0.05), cycle, tags=[h.tag for h in hunks])
        self.mom.feedback([e["id"] for e in broadcast if e["id"] >= 0], helped=v == "accept")
        if self.bandit:
            self.bandit.update_from_cycle([h.tag for h in hunks], v, credit, [h.id for h in hunks])
        self.lineage.record_attempt(parent_sha, v, sha if v in ("accept", "stepping_stone") else None, cycle)

    # ---- audits, rollback, hygiene (serial: run between batches of cycles)
    def rollback(self, target: str, reason: str, cycle: int):
        with self._promote_lock:
            self.repo.move_main(target)
            self.repo.tag(f"rollback/{cycle:06d}", target)
        self.ledger.write(ev="rollback", cycle=cycle, target=target, reason=reason)

    def audit(self, cycle: int, force_heldout: bool = False) -> list:
        """Anchor audit (+ held-out-family audit when due). Results go to the ledger only."""
        cfg, reps = self.cfg, []
        due_a = cfg.anchor_every and (cycle + 1) % cfg.anchor_every == 0
        due_h = force_heldout or (cfg.heldout_every and (cycle + 1) % cfg.heldout_every == 0)
        if not (due_a or due_h):
            return reps
        champion = self.repo.head("main")
        rep = self.anchor_auditor.audit(cycle, champion)
        self.last_selection_stats = (rep["selection_delta"], rep["selection_se"])
        reps.append(rep)
        self.ledger.write(ev="audit", **rep)
        if rep["action"] == "rollback":
            self.rollback(rep["rollback_to"], rep["reason"], cycle)
        elif due_h:
            rep2 = self.heldout_auditor.audit(cycle, self.repo.head("main"), *self.last_selection_stats)
            reps.append(rep2)
            self.ledger.write(ev="audit", **rep2)
            if rep2["action"] == "rollback":
                self.rollback(rep2["rollback_to"], rep2["reason"], cycle)
        return reps

    def _handle_trips(self, trips: list):
        for t in trips:
            self.ledger.write(ev="trip", **t.as_dict())
            log.error("circuit breaker %s: %s (%s)", t.name, t.severity, t.detail)
        halts = [t for t in trips if t.severity == "halt"]
        if halts:
            self.breakers.latch(halts[0])
            raise HaltRun(f"{halts[0].name}: {halts[0].detail}")
        if trips:
            time.sleep(self.cfg.cooloff_s)

    def _after_batch(self, results: list):
        from kernel.hygiene import free_gb
        last = results[-1]["cycle"]
        self.heartbeat.beat(last, "running")
        for res in results:
            self._handle_trips(self.breakers.observe(res))
        for rep in self.audit(last):
            self._handle_trips(self.breakers.observe_audit(rep))
        if self.cfg.gc_every and (last + 1) % self.cfg.gc_every == 0:
            rep = self.gc.collect(active=self.repo.active, live_archive_cycles={e["cycle"] for e in self.archive.entries})
            if any(v for k, v in rep.items() if k != "freed_mb"):
                self.ledger.write(ev="gc", **rep)
        self._handle_trips(self.breakers.observe_disk(free_gb(self.cfg.runs_dir)))

    def run(self, max_cycles: int, workers: int = 1, start_cycle: int = 0, accept_fingerprint_change: bool = False) -> dict:
        """`workers > 1` evaluates several candidates concurrently, each in its own worktree; promotion
        is serialised by a lock plus the fast-forward / champion re-gate check. Audits, breakers and
        GC run between batches, when no cycle is in flight."""
        summary = {"cycles": 0, "accepts": 0, "halted": None}
        if (latched := self.breakers.latched()):
            summary["halted"] = f"HALT latch present: {latched}"
            return summary
        try:
            self.recover()
            self.bootstrap()
            self._fingerprint(accept_fingerprint_change)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for s in range(start_cycle, start_cycle + max_cycles, workers):
                    if self.breakers.stop_requested():
                        self.ledger.write(ev="stop_requested")
                        break
                    results = list(pool.map(self.run_cycle, range(s, min(s + workers, start_cycle + max_cycles))))
                    for res in results:
                        summary["cycles"] += 1
                        summary["accepts"] += res["verdict"] == "accept"
                        log.info("cycle %s -> %s | %s", res["cycle"], res["verdict"], res["why"][-1] if res["why"] else "")
                    self._after_batch(results)
        except HaltRun as e:
            summary["halted"] = str(e)
            self.ledger.write(ev="halt", reason=str(e))
        self.heartbeat.beat(summary["cycles"], "halted" if summary["halted"] else "stopped")
        self.ledger.write(ev="run_end", **summary)
        return summary


# =============================================================== CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kernel.orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="A/A calibration of the gate's p_improve_min")
    c.add_argument("--n", type=int, default=200, help="paired problems per cycle")
    c.add_argument("--p", type=float, default=0.5, help="baseline pass rate")
    c.add_argument("--noise", type=float, default=0.1, help="run-to-run noise sd on the latent scale")
    c.add_argument("--target", type=float, default=0.05, help="max tolerated false-accept rate")
    c.add_argument("--trials", type=int, default=1000)
    c.add_argument("--two-stage", action="store_true", help="calibrate screen(A) + confirm(B) instead of one stage")
    c.add_argument("--n-confirm", type=int, default=64, help="confirmation batch size (two-stage)")
    c.add_argument("--screen-p", type=float, default=0.80, help="screen p_improve (two-stage)")
    a = ap.parse_args(argv)
    cfg = Config(repo_dir=Path("."), runs_dir=Path("runs/_calibration"), screen_p_improve=a.screen_p)
    if a.two_stage:
        best, table = calibrate_two_stage(cfg, a.n, a.n_confirm, a.p, a.noise, a.target, trials=a.trials)
        print(f"TWO-STAGE A/A false-accept rate by confirm p_improve_min (screen n={a.n} p>={a.screen_p}, "
              f"confirm n={a.n_confirm}, base p={a.p}, noise={a.noise}):")
    else:
        best, table = calibrate_gate(cfg, a.n, a.p, a.noise, a.target, trials=a.trials)
        print(f"A/A false-accept rate by p_improve_min (n={a.n}, p={a.p}, noise={a.noise}):")
    for g, fa in table:
        print(f"  p_improve_min={g:<6} false_accept={fa:.3f}{'  <= target' if fa <= a.target else ''}")
    print(f"chosen p_improve_min = {best}")


if __name__ == "__main__":
    main()
