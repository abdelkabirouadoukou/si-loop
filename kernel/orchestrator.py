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

from kernel.patching import PatchProposal, PatchValidator, apply_proposal, collect_tests

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
    cpu and pids caps, and sealed/kernel paths mounted read-only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, cmd: list, cwd: Path, wall_s: float, ro_mounts: tuple = ()) -> RunResult:
        cwd = Path(cwd).resolve()
        wall_s = max(1.0, float(wall_s))
        if self.cfg.container_image:
            argv = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:size=256m",
                    "--memory", f"{self.cfg.mem_mb}m", "--cpus", "2", "--pids-limit", "256",
                    "-v", f"{cwd}:/work:rw", "-w", "/work", "-e", "PYTHONPATH=/work",
                    *itertools.chain.from_iterable(["-v", f"{m}:{m}:ro"] for m in ro_mounts),
                    self.cfg.container_image, *cmd]
        else:
            real = [sys.executable if (i == 0 and c == "python") else str(c) for i, c in enumerate(cmd)]
            argv = [sys.executable, "-c", _BOOT, str(self.cfg.mem_mb * 2**20), str(int(wall_s) + 5),
                    str(256 * 2**20), *real]
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(cwd), "PYTHONPATH": str(cwd),
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        t0 = time.monotonic()
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, errors="replace", start_new_session=True)
        try:
            out, err = p.communicate(timeout=wall_s)
            return RunResult(p.returncode, out[-20000:], err[-20000:], False, time.monotonic() - t0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()
            out, err = p.communicate()
            return RunResult(-9, (out or "")[-20000:], (err or "")[-20000:], True, time.monotonic() - t0)


# =============================================================== git (atomic branches)

class GitRepo:
    """Thin, thread-safe wrapper over the git CLI. The root checkout stays on `main` and is
    never edited by anyone but the kernel."""

    IDENT = ("-c", "user.name=si-loop", "-c", "user.email=si-loop@localhost")

    def __init__(self, root: Path, worktrees: Path):
        self.root, self.worktrees = Path(root).resolve(), Path(worktrees).resolve()
        self.worktrees.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

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
        self._git("worktree", "remove", "--force", str(path), check=False)
        self._git("worktree", "prune", check=False)
        if not keep_branch:
            self._git("branch", "-D", branch, check=False)

    def ff_merge(self, branch: str):
        self._git("merge", "--ff-only", "-q", branch)

    def move_main(self, sha: str):
        """Switch the champion lineage (candidate does not descend from current main)."""
        self._git("reset", "--hard", "-q", sha)

    def tag(self, name: str):
        self._git("tag", "-f", name)

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


class Ledger:
    """Append-only, fsynced write-ahead log. `begin` without `end` marks a crashed cycle."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **rec):
        line = json.dumps({"t": round(time.time(), 3), **rec}, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def incomplete_cycles(self) -> list:
        begun, ended = {}, set()
        for r in self.records():
            if r.get("ev") == "begin":
                begun[r["cycle"]] = r
            elif r.get("ev") == "end":
                ended.add(r["cycle"])
        return [begun[c] for c in begun if c not in ended]


# =============================================================== judge: results, gate, calibration

@dataclass
class EvalResult:
    passes: dict                              # problem_id -> hidden-pytest pass (HARD signal)
    metrics: dict = field(default_factory=dict)          # tokens, wall_s, agent_loc, ...
    family_profile: dict = field(default_factory=dict)   # family -> pass rate (descriptor input)

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


def p_better(b: int, c: int) -> float:
    """Exact P(Beta(1+b, 1+c) > 0.5) = P(Binomial(b+c+1, 1/2) <= b): posterior probability that
    the candidate fixes problems more often than it breaks them, flat prior, discordant pairs only."""
    n = b + c + 1
    return sum(math.comb(n, k) for k in range(b + 1)) / (1 << n)


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

        def dec(v, extra):
            return GateDecision(v, delta, lower, p_imp, b, c, why + [extra])

        if lower < -self.cfg.eps_noninferior:
            return dec("reject", "not non-inferior")
        if delta > 0 and p_imp >= self.cfg.p_improve_min:
            return dec("accept", "correctness gain")
        if delta >= 0 and cheaper:
            return dec("accept", "efficiency gain at non-inferior correctness")
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

class Orchestrator:
    def __init__(self, cfg: Config, repo: GitRepo, sandbox: Sandbox, verifier: Verifier, meta: MetaAgent,
                 archive: Archive, attributor: ShapleyAttributor, mom: MomCenter, guard: KernelGuard):
        self.cfg, self.repo, self.sb, self.verifier, self.meta = cfg, repo, sandbox, verifier, meta
        self.archive, self.attr, self.mom, self.guard = archive, attributor, mom, guard
        self.gate, self.validator = Gate(cfg), PatchValidator()
        self.ledger = Ledger(cfg.runs_dir / "ledger.jsonl")
        self._cache: dict = {}
        self._promote_lock = threading.Lock()
        self.fail_streak = 0

    # ---- helpers
    def descriptor(self, ev: EvalResult) -> list:
        return [ev.family_profile.get(f, 0.0) for f in self.cfg.families] + \
               [ev.metrics.get("agent_loc", 0.0) / 1000, ev.metrics.get("tokens", 0.0) / 1e5]

    def _eval_cached(self, sha: str, batch: list) -> EvalResult:
        key = (sha, tuple(batch))
        if key not in self._cache:
            branch = f"eval/{sha[:8]}-{time.time_ns()}"
            wt = self.repo.add_worktree(branch, sha)
            try:
                if len(self._cache) >= 256:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = self.verifier.evaluate(wt, batch, self.cfg.eval_wall_s)
            finally:
                self.repo.discard(branch)
        return self._cache[key]

    def recover(self):
        """Crash recovery: discard worktrees/branches of cycles that began but never ended."""
        for rec in self.ledger.incomplete_cycles():
            self.repo.discard(rec["branch"])
            self.ledger.write(ev="end", cycle=rec["cycle"], verdict="reject", why=["recovered after crash"])
            log.warning("recovered crashed cycle %s", rec["cycle"])

    def bootstrap(self):
        """Seed the archive with the current champion (idempotent)."""
        if self.archive.entries:
            return
        sha = self.repo.head("main")
        ev = self._eval_cached(sha, self.verifier.draw_batch(-1, self.cfg.n_selection))
        self.archive.admit(sha, self.descriptor(ev), ev.pass_at_1, "seed", -1)

    # ---- one cycle
    def run_cycle(self, cycle: int) -> dict:
        self.guard.assert_intact()                      # tampering aborts the whole run (propagates)
        champion = self.repo.head("main")
        parent_sha = self.archive.select_parent(champion)
        branch = f"cand/{cycle:06d}"
        budget = Budget(self.cfg.cycle_wall_s, self.cfg.token_budget)
        broadcast = self.mom.broadcast(cycle)
        self.ledger.write(ev="begin", cycle=cycle, parent=parent_sha, branch=branch)
        out = {"cycle": cycle, "parent": parent_sha[:8], "verdict": "reject", "why": []}
        keep, proposal = False, None
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

            # 4. L2: sealed hidden pytest on the SAME fresh batch for parent and candidate
            batch = self.verifier.draw_batch(cycle, self.cfg.n_selection)
            cand = self.verifier.evaluate(wt, batch, min(self.cfg.eval_wall_s, budget.remaining_s()))
            parent = self._eval_cached(parent_sha, batch)

            # 5. L3 + gate
            desc = self.descriptor(cand)
            decision = self.gate.decide(parent, cand, self.archive.novelty(desc, exclude=sha))
            out.update(verdict=decision.verdict, why=decision.reasons, delta=decision.delta,
                       p_better=round(decision.p_improve, 4), parent_pass=parent.pass_at_1,
                       cand_pass=cand.pass_at_1, agent_loc=cand.metrics.get("agent_loc"))

            # 6. atomic commit-or-rollback
            if decision.verdict == "accept":
                with self._promote_lock:
                    champ = self.repo.head("main")
                    promoted = False
                    if self.repo.is_ancestor(champ, sha):
                        self.repo.ff_merge(branch)
                        promoted = True
                    else:                               # lineage switch: must also beat the champion
                        d2 = self.gate.decide(self._eval_cached(champ, batch), cand, 0.0)
                        if d2.verdict == "accept":
                            self.repo.move_main(sha)
                            promoted = True
                        out["why"].append(f"vs champion: {d2.verdict} ({d2.reasons[1]})")
                    if promoted:
                        self.repo.tag(f"champ/{cycle:06d}")
                        self.archive.admit(sha, desc, cand.pass_at_1, "accept", cycle)
                    else:
                        out["verdict"] = "stepping_stone"
                        self.repo.archive_ref(branch, f"{cycle:06d}")
                        self.archive.admit(sha, desc, cand.pass_at_1, "stepping_stone", cycle)
                        keep = True
            elif decision.verdict == "stepping_stone":
                self.repo.archive_ref(branch, f"{cycle:06d}")
                self.archive.admit(sha, desc, cand.pass_at_1, "stepping_stone", cycle)
                keep = True

            # 7. credit assignment for kept patches (failure here never undoes a promotion)
            if out["verdict"] != "reject" and self.cfg.credit_enabled:
                try:
                    credit = self.attr.attribute(parent_sha, proposal,
                                                 self.verifier.draw_batch(cycle, self.cfg.n_credit, "credit"))
                    out["credit"] = credit
                    for h in hunks:
                        self.mom.record("credit", {"tag": h.tag, "path": h.path, "phi": credit[h.id]},
                                        True, credit[h.id], cycle, tags=[h.tag])
                except Exception as e:
                    out["credit_error"] = f"{type(e).__name__}: {e}"
            self.mom.record("patch", {"hypothesis": proposal.hypothesis, "verdict": out["verdict"],
                                      "delta": decision.delta}, True, decision.delta, cycle,
                            tags=[h.tag for h in hunks])
            self.mom.feedback([e["id"] for e in broadcast], helped=out["verdict"] == "accept")
            self.fail_streak = 0
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            out.update(verdict="reject", why=out["why"] + [f"{type(e).__name__}: {e}"])
            log.debug("cycle %s failed", cycle, exc_info=True)
            self.mom.record("failure", {"error": f"{type(e).__name__}: {e}"}, True, -1.0, cycle,
                            tags=[type(e).__name__])
            self.mom.feedback([x["id"] for x in broadcast], helped=False)
            self.fail_streak += 1
        finally:
            self.repo.discard(branch, keep_branch=keep)     # hard rollback unless promoted/archived
            self.ledger.write(ev="end", **out)
        return out

    def run(self, max_cycles: int, workers: int = 1, start_cycle: int = 0):
        """`workers > 1` evaluates several candidates concurrently, each in its own worktree;
        promotion is serialised by a lock plus the fast-forward / champion re-gate check."""
        self.recover()
        self.bootstrap()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s in range(start_cycle, start_cycle + max_cycles, workers):
                if self.fail_streak >= self.cfg.max_consecutive_failures:
                    log.error("circuit breaker open: cooling off %ss", self.cfg.cooloff_s)
                    time.sleep(self.cfg.cooloff_s)
                    self.fail_streak = 0
                for res in pool.map(self.run_cycle, range(s, min(s + workers, start_cycle + max_cycles))):
                    log.info("cycle %s -> %s | %s", res["cycle"], res["verdict"], res["why"][-1] if res["why"] else "")


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
    a = ap.parse_args(argv)
    cfg = Config(repo_dir=Path("."), runs_dir=Path("runs/_calibration"))
    best, table = calibrate_gate(cfg, a.n, a.p, a.noise, a.target, trials=a.trials)
    print(f"A/A false-accept rate by p_improve_min (n={a.n}, p={a.p}, noise={a.noise}):")
    for g, fa in table:
        print(f"  p_improve_min={g:<6} false_accept={fa:.3f}{'  <= target' if fa <= a.target else ''}")
    print(f"chosen p_improve_min = {best}")


if __name__ == "__main__":
    main()
