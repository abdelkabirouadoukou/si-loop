"""Reproduces the false-accept calibration curves and the power curve used in paper_study2.tex.
    python results/study2/make_calibration.py            # writes results/study2/calibration.json
Single-stage vs two-stage (screen A + confirm B) A/A false-accept rate over p_improve_min, and the accept
rate (power) of a patch that truly improves pass rate by `lift`."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kernel.orchestrator import Config, EvalResult, Gate, simulate_false_accept, simulate_false_accept_two_stage  # noqa: E402

N, P, NOISE, TRIALS = 200, 0.5, 0.1, 2000
GRID = (0.80, 0.90, 0.95, 0.975, 0.99)


def power(cfg, n_a, n_b, p, lift, noise, trials, two_stage, seed=1):
    rng = random.Random(seed)
    screen = Gate(Config(**{**cfg.__dict__, "p_improve_min": cfg.screen_p_improve}))
    strict = Gate(cfg)

    def stage(gate, n):
        u = [rng.random() for _ in range(n)]
        base = EvalResult({str(i): (u[i] + rng.gauss(0, noise)) < p for i in range(n)}, {"agent_loc": 100, "tokens": 100})
        cand = EvalResult({str(i): (u[i] + rng.gauss(0, noise)) < p + lift for i in range(n)}, {"agent_loc": 100, "tokens": 100})
        return gate.decide(base, cand, novelty=0.0).verdict == "accept"
    return sum((stage(screen, n_a) and stage(strict, n_b)) if two_stage else stage(strict, n_a) for _ in range(trials)) / trials


def main():
    base = Config(repo_dir=Path("."), runs_dir=Path("."))
    out = {"n": N, "p": P, "noise": NOISE, "trials": TRIALS, "false_accept": [], "power": []}
    for g in GRID:
        cfg = Config(**{**base.__dict__, "p_improve_min": g})
        out["false_accept"].append({"p_improve_min": g, "single": simulate_false_accept(cfg, N, P, NOISE, TRIALS),
                                    "two_stage": simulate_false_accept_two_stage(cfg, N, N, P, NOISE, TRIALS)})
    cfg = Config(**{**base.__dict__, "p_improve_min": 0.95})
    for lift in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        out["power"].append({"lift": lift, "single": power(cfg, N, N, P, lift, NOISE, 1000, False),
                             "two_stage": power(cfg, N, N, P, lift, NOISE, 1000, True)})
    Path(__file__).with_name("calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
