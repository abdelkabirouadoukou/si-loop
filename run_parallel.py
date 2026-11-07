"""Parallel driver for the SI-Loop ablation.

Imports si_loop_toy unchanged and runs (arm, seed) pairs in separate processes.
Each run reseeds itself, so results do not depend on scheduling order.
Output JSON has the same structure as `si_loop_toy.py --ablation`: {arm: [run dict per seed]}.

    python run_parallel.py --seeds 100 101 102 103 104 --out results/dev.json
    python run_parallel.py --seeds 1 2 3 4 5 6 7 8 9 10 --out results/ablation.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import torch

import si_loop_toy as s


def _job(args):
    arm, seed, rounds, steps, meta_bs = args
    torch.set_num_threads(1)
    r = s.run(arm, seed, rounds, steps, meta_bs, verbose=False)
    print(f"done {arm} seed={seed} final_hard={r['final_hard']:.3f} accept={r['accept_rate']:.2f}", flush=True)
    return arm, seed, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--meta-bs", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = [(arm, sd, a.rounds, a.steps, a.meta_bs) for sd in a.seeds for arm in s.ARMS]
    with Pool(a.workers) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    results = {arm: [] for arm in s.ARMS}
    for arm, sd, r in sorted(out, key=lambda z: (list(s.ARMS).index(z[0]), z[1])):
        results[arm].append(r)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
