"""kernel/patching.py: structured patch proposals.

The meta-agent never writes files. It emits a JSON PatchProposal which the kernel
  1. parses strictly            -> parse_proposal()
  2. validates statically       -> PatchValidator.validate_proposal() / validate_tree()
  3. applies atomically via AST -> apply_proposal()
Each EditOp is one atomic "hunk": the unit of credit assignment (Shapley coalitions
are built with apply_proposal(only=...)).

Static checks are the FIRST line of defence, not a security boundary. Real isolation
comes from the sandbox (no network, read-only kernel/sealed mounts, rlimits).
"""
from __future__ import annotations

import ast
import json
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional

MUTABLE_ROOT = "agent"                       # the ONLY writable subtree
ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".yaml"}
MAX_OPS, MAX_CHARS, MAX_HYPOTHESIS = 8, 20_000, 500
KINDS = ("replace_symbol", "create_file", "delete_file", "replace_text")
TAG_RE = re.compile(r"^[a-z0-9_.\-]{1,40}$")

FORBIDDEN_MODULES = {
    "socket", "subprocess", "ctypes", "multiprocessing", "shutil", "importlib", "pickle",
    "marshal", "urllib", "http", "requests", "ftplib", "signal", "resource", "sys",
    "builtins", "kernel", "sealed", "pty", "telnetlib", "smtplib", "sqlite3", "gc",
}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "open", "breakpoint",
                   "globals", "setattr", "delattr", "__builtins__"}
FORBIDDEN_OS_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "removedirs", "rename",
                      "kill", "killpg", "fork", "forkpty", "execv", "execve", "execvp",
                      "spawnv", "spawnl", "chmod", "chown", "environ", "putenv", "chdir",
                      "symlink", "link", "truncate"}
FORBIDDEN_DUNDERS = {"__subclasses__", "__globals__", "__builtins__", "__code__",
                     "__class__", "__bases__", "__mro__", "__dict__"}
TEST_ESCAPES = {"skip", "skipif", "xfail", "importorskip"}


class PatchError(ValueError):
    """Malformed or inapplicable patch (never a kernel bug)."""


# ------------------------------------------------------------------ data model

@dataclass
class EditOp:
    path: str
    kind: str
    symbol: Optional[str] = None
    new_source: Optional[str] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    tag: str = "untagged"


@dataclass
class PatchProposal:
    id: str
    parent_sha: str
    hypothesis: str
    ops: list[EditOp]
    expected_effect: dict = field(default_factory=dict)


@dataclass
class Hunk:
    id: str
    op_index: int
    path: str
    kind: str
    symbol: Optional[str]
    tag: str


# ------------------------------------------------------------------ parsing LLM output

_OP_KEYS = set(EditOp.__dataclass_fields__)


def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError as e:
            raise PatchError(f"proposal is not valid JSON: {e}") from e
    raise PatchError("no JSON object found in proposal")


def parse_proposal(raw: "str | dict", parent_sha: str = "") -> PatchProposal:
    """Strict schema parse of a model-emitted proposal (JSON string, fenced block, or dict)."""
    data = _extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise PatchError("proposal must be a JSON object")
    unknown = set(data) - {"id", "hypothesis", "ops", "expected_effect"}
    if unknown:
        raise PatchError(f"unknown proposal keys: {sorted(unknown)}")
    hyp = data.get("hypothesis")
    if not isinstance(hyp, str) or not hyp.strip() or len(hyp) > MAX_HYPOTHESIS:
        raise PatchError(f"hypothesis must be a non-empty string <= {MAX_HYPOTHESIS} chars")
    ops_raw = data.get("ops")
    if not isinstance(ops_raw, list) or not ops_raw:
        raise PatchError("ops must be a non-empty list")
    ops: list[EditOp] = []
    for i, o in enumerate(ops_raw):
        if not isinstance(o, dict):
            raise PatchError(f"op {i} must be an object")
        bad = set(o) - _OP_KEYS
        if bad:
            raise PatchError(f"op {i}: unknown keys {sorted(bad)}")
        if o.get("kind") not in KINDS:
            raise PatchError(f"op {i}: kind must be one of {KINDS}")
        for k, v in o.items():
            if v is not None and not isinstance(v, str):
                raise PatchError(f"op {i}: '{k}' must be a string")
        if not isinstance(o.get("path"), str):
            raise PatchError(f"op {i}: path required")
        if not TAG_RE.match(o.get("tag", "untagged")):
            raise PatchError(f"op {i}: bad tag")
        ops.append(EditOp(**o))
    eff = data.get("expected_effect", {})
    if not isinstance(eff, dict):
        raise PatchError("expected_effect must be an object")
    return PatchProposal(str(data.get("id") or uuid.uuid4().hex[:8]), parent_sha, hyp.strip(), ops, eff)


# ------------------------------------------------------------------ AST helpers

def _safe_rel(path: str) -> Optional[str]:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in path or "\x00" in path:
        return f"path escapes repo: {path!r}"
    if p.parts[0] != MUTABLE_ROOT or len(p.parts) < 2:
        return f"path outside mutable root '{MUTABLE_ROOT}/': {path}"
    if p.suffix not in ALLOWED_SUFFIXES:
        return f"suffix not allowed: {path}"
    if p.name == "conftest.py" or p.name.startswith("."):
        return f"forbidden file name: {path}"
    return None


def _find_symbol(tree: ast.Module, dotted: str):
    scope, node = tree.body, None
    for part in dotted.split("."):
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            return None
        scope = getattr(node, "body", [])
    return node


def _span(node) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start - 1, node.end_lineno                      # [start, end) 0-based line indices


def replace_symbol_source(src: str, dotted: str, new_source: str) -> str:
    """Exact symbol replacement. `dotted` is 'func' or 'Class.method'. The replacement must be
    exactly one top-level def/class with the same name; indentation follows the original.
    (Multi-line string literals inside a nested replacement are re-indented.)"""
    try:
        node = _find_symbol(ast.parse(src), dotted)
    except SyntaxError as e:
        raise PatchError(f"target file does not parse: {e}") from e
    if node is None:
        raise PatchError(f"symbol {dotted!r} not found")
    new = textwrap.dedent(new_source).strip("\n")
    try:
        nt = ast.parse(new)
    except SyntaxError as e:
        raise PatchError(f"replacement for {dotted!r} has a syntax error: {e}") from e
    defs = [n for n in nt.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    want = dotted.split(".")[-1]
    if len(nt.body) != 1 or len(defs) != 1 or defs[0].name != want:
        raise PatchError(f"replacement must be exactly one definition named {want!r}")
    s, e = _span(node)
    lines = src.splitlines(keepends=True)
    body = textwrap.indent(new + "\n", " " * node.col_offset)
    out = "".join(lines[:s]) + body + "".join(lines[e:])
    try:
        ast.parse(out)
    except SyntaxError as ex:
        raise PatchError(f"result of replacing {dotted!r} does not parse: {ex}") from ex
    return out


def scan_python(source: str, filename: str, is_test: bool) -> list[str]:
    """Static scan: forbidden imports/builtins, os escapes, dunder tricks, test escape hatches."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [f"{filename}: syntax error: {e}"]
    errs: list[str] = []
    os_names = {"os"}
    for n in ast.walk(tree):                                # pass 1: aliases of `os`
        if isinstance(n, ast.Import):
            os_names |= {a.asname or "os" for a in n.names if a.name == "os"}
    for n in ast.walk(tree):                                # pass 2: checks
        ln = getattr(n, "lineno", 0)
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    errs.append(f"{filename}:{ln} forbidden import {a.name}")
        elif isinstance(n, ast.ImportFrom):
            top = (n.module or "").split(".")[0]
            if n.level:
                errs.append(f"{filename}:{ln} relative imports are not allowed")
            if top in FORBIDDEN_MODULES:
                errs.append(f"{filename}:{ln} forbidden import from {n.module}")
            if top == "os":
                for a in n.names:
                    if a.name in FORBIDDEN_OS_ATTRS or a.name == "*":
                        errs.append(f"{filename}:{ln} forbidden 'from os import {a.name}'")
        elif isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load) and n.id in FORBIDDEN_NAMES:
                errs.append(f"{filename}:{ln} forbidden name {n.id}")
        elif isinstance(n, ast.Attribute):
            if n.attr in FORBIDDEN_DUNDERS:
                errs.append(f"{filename}:{ln} forbidden attribute {n.attr}")
            if isinstance(n.value, ast.Name) and n.value.id in os_names and n.attr in FORBIDDEN_OS_ATTRS:
                errs.append(f"{filename}:{ln} forbidden os.{n.attr}")
            if is_test and n.attr in TEST_ESCAPES:
                errs.append(f"{filename}:{ln} test escape hatch .{n.attr}")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr":
            lit = n.args[1] if len(n.args) >= 2 else None
            if not (isinstance(lit, ast.Constant) and isinstance(lit.value, str)):
                errs.append(f"{filename}:{ln} getattr with non-literal name")
            elif lit.value in FORBIDDEN_OS_ATTRS | FORBIDDEN_DUNDERS | FORBIDDEN_NAMES:
                errs.append(f"{filename}:{ln} getattr of forbidden attribute {lit.value}")
        elif isinstance(n, ast.Assign) and is_test:
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in {"pytest_plugins", "pytestmark"}:
                    errs.append(f"{filename}:{ln} forbidden test module global {t.id}")
    return errs


def collect_tests(root: Path) -> set[str]:
    """Stable ids ('relpath::Class.test_x') of every agent self-test; the ratchet's baseline."""
    ids: set[str] = set()
    base = root / MUTABLE_ROOT / "tests"
    for f in (sorted(base.rglob("test_*.py")) if base.exists() else []):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        rel = f.relative_to(root).as_posix()
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_"):
                ids.add(f"{rel}::{n.name}")
            elif isinstance(n, ast.ClassDef) and n.name.startswith("Test"):
                ids |= {f"{rel}::{n.name}.{m.name}" for m in n.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test_")}
    return ids


# ------------------------------------------------------------------ validation

class PatchValidator:
    def validate_proposal(self, p: PatchProposal) -> list[str]:
        """Pre-apply structural checks (no filesystem access)."""
        errs: list[str] = []
        if not p.ops:
            errs.append("empty patch")
        if len(p.ops) > MAX_OPS:
            errs.append(f"too many ops ({len(p.ops)} > {MAX_OPS})")
        size = sum(len(o.new_source or "") + len(o.new_text or "") + len(o.old_text or "") for o in p.ops)
        if size > MAX_CHARS:
            errs.append("patch too large")
        for o in p.ops:
            if o.kind not in KINDS:
                errs.append(f"unknown op kind {o.kind!r}")
                continue
            if (e := _safe_rel(o.path)):
                errs.append(e)
            if o.kind == "replace_symbol" and not (o.symbol and o.new_source and o.path.endswith(".py")):
                errs.append(f"replace_symbol needs symbol+new_source on a .py file: {o.path}")
            if o.kind == "replace_text" and not (o.old_text and o.new_text is not None):
                errs.append(f"replace_text needs old_text/new_text: {o.path}")
            if o.kind == "create_file" and o.new_source is None:
                errs.append(f"create_file needs new_source: {o.path}")
        return errs

    def validate_tree(self, root: Path, p: PatchProposal, baseline_tests: set[str]) -> list[str]:
        """Post-apply checks on the patched worktree: forbidden constructs + test ratchet."""
        errs: list[str] = []
        for o in p.ops:
            f = root / o.path
            if o.kind != "delete_file" and f.suffix == ".py" and f.exists():
                is_test = o.path.startswith(f"{MUTABLE_ROOT}/tests/")
                errs += scan_python(f.read_text(), o.path, is_test=is_test)
        lost = baseline_tests - collect_tests(root)
        if lost:
            errs.append(f"test ratchet: self-tests removed or renamed: {sorted(lost)[:5]}")
        return errs


# ------------------------------------------------------------------ application

def apply_proposal(root: Path, p: PatchProposal, only: Optional[set[int]] = None) -> list[Hunk]:
    """Apply ops to `root` ATOMICALLY: every edit is staged in memory and files are written
    only if all ops succeed, so a failure leaves the tree untouched. `only` selects a subset
    of op indices (Shapley coalitions)."""
    staged: dict[str, Optional[str]] = {}                  # path -> new content (None = delete)

    def read(path: str) -> str:
        if path in staged:
            if staged[path] is None:
                raise PatchError(f"{path} was deleted earlier in this patch")
            return staged[path]                              # type: ignore[return-value]
        f = root / path
        if not f.is_file():
            raise PatchError(f"file not found: {path}")
        return f.read_text()

    def exists(path: str) -> bool:
        return staged[path] is not None if path in staged else (root / path).exists()

    hunks: list[Hunk] = []
    for i, o in enumerate(p.ops):
        if only is not None and i not in only:
            continue
        if (e := _safe_rel(o.path)):
            raise PatchError(e)
        if o.kind == "create_file":
            if exists(o.path):
                raise PatchError(f"create_file target exists: {o.path}")
            staged[o.path] = o.new_source
        elif o.kind == "delete_file":
            read(o.path)
            staged[o.path] = None
        elif o.kind == "replace_text":
            src = read(o.path)
            n = src.count(o.old_text or "")
            if n != 1:
                raise PatchError(f"old_text must occur exactly once in {o.path} (found {n})")
            staged[o.path] = src.replace(o.old_text, o.new_text, 1)
        elif o.kind == "replace_symbol":
            staged[o.path] = replace_symbol_source(read(o.path), o.symbol or "", o.new_source or "")
        else:
            raise PatchError(f"unknown op kind {o.kind!r}")
        hunks.append(Hunk(f"{p.id}:{i}", i, o.path, o.kind, o.symbol, o.tag))
    for path, content in staged.items():
        f = root / path
        if content is None:
            f.unlink()
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
    return hunks
