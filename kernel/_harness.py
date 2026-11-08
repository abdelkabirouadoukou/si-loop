"""kernel/_harness.py: RPC worker that runs INSIDE the sandbox. Stdlib only, never imported by the kernel.

    python harness.py solver  <worktree>     # imports agent.solver, answers {"kind":"solve", spec}
    python harness.py executor               # executes candidate source on hidden CASES, no oracle

Protocol: one JSON object per line on stdin -> one JSON object per line on stdout.
  * every job runs in a FORKED CHILD: a crash, hang or memory bomb kills only that job, and the
    candidate cannot keep state from one problem to the next;
  * the child's stdin/stdout are /dev/null, so candidate code cannot read later requests or
    corrupt the protocol; results come back over a private pipe;
  * an audit hook flags any touch of the honeypot decoys (open/import/list of the jail) and
    writes it synchronously to a flag file the judge reads, so a kill mid-job loses nothing.
The executor never receives expected outputs; the judge compares outside the sandbox.
"""
import json
import os
import select
import signal
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
JAIL = os.path.realpath(os.environ.get("SIL_JAIL", "/nonexistent-jail"))
FLAG_FILE = os.environ.get("SIL_FLAGS", "")
DECOY_MODS = set(filter(None, os.environ.get("SIL_DECOY_MODS", "").split(",")))
_in_hook = [False]


def _flag(what):
    if not FLAG_FILE:
        return
    try:
        fd = os.open(FLAG_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.write(fd, (json.dumps({"event": what, "pid": os.getpid(), "t": time.time()}) + "\n").encode())
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def _hook(event, args):
    if _in_hook[0]:
        return
    _in_hook[0] = True
    try:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.path.realpath(os.fsdecode(args[0]))
            if p == JAIL or p.startswith(JAIL + os.sep):
                _flag("open:" + os.path.basename(p))
        elif event == "import" and args and str(args[0]).split(".")[0] in DECOY_MODS:
            _flag("import:" + str(args[0]))
        elif event in ("os.listdir", "os.scandir"):
            a = args[0] if args else None
            p = os.path.realpath(os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else os.getcwd())
            if p == JAIL:
                _flag("list:jail")
    except Exception:                                      # a hook must never break the harness
        pass
    finally:
        _in_hook[0] = False


class _Alarm(BaseException):                               # BaseException: `except Exception` cannot swallow it
    pass


def _on_alarm(signum, frame):
    raise _Alarm()


def _enc_value(v):
    try:
        return {"v": json.loads(json.dumps(v))}
    except (TypeError, ValueError, OverflowError):
        return {"e": "Unserializable"}


def _timed(fn, limit):
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        return _enc_value(fn())
    except _Alarm:
        return {"e": "Timeout"}
    except Exception as e:                                 # candidate exceptions are outcomes
        return {"e": type(e).__name__}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _job_run(job):
    ns = {"__name__": "candidate"}
    try:
        exec(compile(job["code"], "<candidate>", "exec"), ns)
    except BaseException as e:                              # includes SystemExit from candidate import-time code
        return {"ok": False, "error": "load:" + type(e).__name__}
    entry, limit, out = job["entry"], float(job.get("case_timeout", 2.0)), []
    if entry not in ns:
        return {"ok": False, "error": "missing_entry"}
    for case in job["cases"]:
        if job["mode"] == "call":
            out.append(_timed(lambda: ns[entry](*case), limit))
        else:
            holder = {}

            def build():
                holder["o"] = ns[entry](*case["ctor"])
            r = _timed(build, limit)
            if "e" in r:
                out.append([{"e": "CtorError"}] * len(case["ops"]))
                continue
            out.append([_timed(lambda n=n, a=a: getattr(holder["o"], n)(*a), limit) for n, a in case["ops"]])
    return {"ok": True, "results": out}


def _job_solve(req, solver):
    r = solver(req["spec"])
    code, tokens = (r if isinstance(r, tuple) else (r, 0))
    if not isinstance(code, str):
        return {"ok": False, "error": "solver_returned_non_string"}
    return {"ok": True, "code": code, "tokens": int(tokens)}


def _child(req, w, solver):
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    try:
        import resource
        cpu = int(req.get("cpu", 10)) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass
    signal.signal(signal.SIGALRM, _on_alarm)
    try:
        res = _job_solve(req, solver) if req["kind"] == "solve" else _job_run(req["job"])
    except BaseException as e:
        res = {"ok": False, "error": "exc:" + type(e).__name__}
    data = json.dumps(res).encode()
    os.write(w, data) if len(data) < 60000 else _write_all(w, data)
    os._exit(0)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _serve(solver):
    out = sys.stdout.buffer
    out.write(b'{"ready": true}\n')
    out.flush()
    for line in sys.stdin.buffer:
        req = json.loads(line)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            _child(req, w, solver)
        os.close(w)
        buf, deadline = b"", time.monotonic() + float(req.get("timeout", 10))
        timed_out = False
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                timed_out = True
                break
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 8_000_000:
                break
        if timed_out or len(buf) > 8_000_000:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status = os.waitpid(pid, 0)
        os.close(r)
        if timed_out:
            res = {"ok": False, "error": "timeout"}
        elif not buf:
            res = {"ok": False, "error": "crashed:%d" % (status & 0x7F)}
        else:
            try:
                res = json.loads(buf)
            except ValueError:
                res = {"ok": False, "error": "bad_result"}
        out.write((json.dumps({"id": req.get("id"), **res}) + "\n").encode())
        out.flush()


def main():
    sys.addaudithook(_hook)
    solver = None
    if MODE == "solver":
        sys.path.insert(0, sys.argv[2])
        try:
            from agent.solver import solve_problem as solver
        except BaseException as e:
            sys.stdout.buffer.write((json.dumps({"ready": False, "error": "import:" + type(e).__name__}) + "\n").encode())
            sys.stdout.buffer.flush()
            sys.exit(0)
    _serve(solver)


if __name__ == "__main__":
    main()
