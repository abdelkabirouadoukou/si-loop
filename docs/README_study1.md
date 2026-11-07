# SI-Loop

**Super Interintelligent Loop with a Global Workspace ("Mom Center")**: a research scaffold for studying a closed loop of weight generation, task execution, curriculum synthesis and guarded oversight, coordinated through a limited-capacity workspace.

> **Status:** research scaffold with one completed, pre-specified ablation (Study 1; see [Results](#results-study-1) and `paper.pdf`). The code runs on a synthetic few-shot regression task and makes **no claim** about scaling to language models or about recursive self-improvement. Treat every design choice below as a hypothesis, not an established finding.

## Motivation

Recent work shows pieces of self-improving systems in isolation:

- weight generation from task descriptions or prompts (hypernetworks),
- agents that rewrite their own code or harness and validate changes empirically,
- proposer/solver self-play with verifiable rewards,
- judge exploitation under self-play, and
- workspace-style coordination for multi-agent LLM systems.

SI-Loop asks a narrower, testable question: **what happens when these pieces are wired into one loop with an adversarially hardened judge and a workspace that must *select* what is shared?** The two open issues it targets are (1) evaluator co-adaptation (Goodhart pressure on the judge) and (2) whether competitive selection in a shared workspace beats passive or random sharing.

## Architecture

| Component | Role | Implementation in this repo |
|---|---|---|
| **Model A: Hypernetwork** | Generates the initial adapter for B from a support set and the workspace broadcast | MLP that outputs a low-rank adapter (`U`, `V`) and biases |
| **Model B: Worker** | Executes the task; its query loss after inner adaptation drives A and B updates | Small MLP with an adapter hook, adapted by a few SGD steps |
| **Model C: Curriculum Engine** | Chooses what the loop trains on | REINFORCE over difficulty bins with a learnability reward (peaks at 50% success) |
| **Model D: Guarded Judge** | Decides whether a candidate round is accepted | Sealed RNG, fresh selection tasks each round, lower-confidence-bound gate, hidden noisy canary with a query budget, generalisation-gap monitor, rollback |
| **Mom Center: Workspace** | Global Workspace-style broadcast | Limited capacity `K`, salience competition, ignition threshold, TTL, typed numeric payloads, per-source rate limit, outcome-based trust |

### Loop

```
for each round:
    C proposes tasks; A, B train by first-order meta-updates (workspace broadcast conditions A)
    B and C post summaries to W; W runs the competition and broadcasts the winners
    D compares candidate vs last accepted on fresh tasks + hidden canary
        accept  -> snapshot, Shapley attribution (A vs B), update trust/utility
        reject  -> roll back A, B, C and the optimizers
```

### Judge controls (Model D)

- **Sealed randomness:** A, B and C never see D's generator.
- **Fresh selection tasks** every round, so improvement cannot be adaptively overfit to a fixed set.
- **Conservative gate:** accept only if `mean(d) - std(d)/sqrt(n) > 0`, where `d` is the paired per-task improvement.
- **Hidden canary:** released as one Laplace-noised bit, with a finite query budget.
- **Gap monitor:** rejects rounds where the query-minus-support loss jumps above its running average.

### Credit assignment

Two-player Shapley over model versions on a fixed diagnostic set. With old and new versions of A and B,
`phi_A + phi_B` equals the total improvement exactly, and the interaction term measures A/B co-adaptation.
C is not included in the Shapley value because its effect flows through the data distribution; the learnability reward is only a proxy.

### Workspace (Mom Center)

Candidates compete on

```
salience = 0.5*relevance + 0.3*novelty + 0.3*trust*verified + 0.2*utility - 0.3*age/TTL
```

The top `K` above threshold `theta` are broadcast (mean embedding). Entries expire after `ttl` rounds. The toy embeds scalar training statistics with a fixed random projection, which stands in for a semantic embedding.

## Installation

```bash
git clone https://github.com/abdelkabirouadoukou/si-loop.git si-loop && cd si-loop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ and a CPU are sufficient.

## Usage

Single run of one arm:

```bash
python si_loop_toy.py --arm full --seed 0
```

### Ablation study

Four arms, compared on harness-side evaluation sets that no component sees during training or judging:

| Arm | Workspace | Judge |
|---|---|---|
| `full` | competition for broadcast | on |
| `no_ws` | broadcast replaced by zeros | on |
| `random_ws` | same capacity and TTL, random winners | on |
| `no_judge` | competition for broadcast | off (every round accepted) |

```bash
python si_loop_toy.py --ablation --seeds 0 1 2 3 4 --rounds 12 --steps 40 --out results/ablation.json
```

It prints mean ± std over seeds of final query MSE on a uniform-difficulty set (`final_all`) and a hard set (`final_hard`), plus the accept rate, and writes per-round histories (including Shapley terms) to JSON.

Suggested questions:

1. Does `full` beat `no_ws`? (does a workspace help at all)
2. Does `full` beat `random_ws`? (does *competition* help beyond shared memory)
3. Does `full` beat `no_judge` on `final_hard`? (do the judge controls help)
4. Is the Shapley interaction term large in rounds that the judge later rejects?

Report negative results too. With few seeds, differences will often be within noise; use more seeds before drawing conclusions.

The Study 1 report run used the parallel driver and the analysis script (hypotheses and plan in `docs/hypotheses.md`, frozen; code hashes in `results/config_hashes.txt`):

```bash
python run_parallel.py --seeds 1 2 3 4 5 6 7 8 9 10 --out results/ablation.json
python -I analyze.py results/ablation.json --out results/
```

## Results (Study 1)

Ten paired seeds, 12 rounds, query MSE (lower is better); full details, CIs and caveats in `paper.pdf`.

| Arm | `final_hard` | `final_all` | accept rate |
|---|---|---|---|
| `full` | 1.048 ± 0.190 | 0.872 ± 0.148 | 0.267 |
| `no_ws` | 1.038 ± 0.189 | 0.872 ± 0.166 | 0.275 |
| `random_ws` | 1.022 ± 0.162 | 0.848 ± 0.154 | 0.275 |
| `no_judge` | 0.962 ± 0.146 | 0.755 ± 0.092 | 1.000 |

- **H1, H2:** no detectable difference between the competitive workspace and no workspace or random winners (low power; this is not evidence of no effect).
- **H3:** removing the judge *improved* `final_hard` by 0.086 (Bonferroni 98.3% CI of the difference: [-0.168, -0.014]). The judge rejected about 73% of rounds, so guarded arms got far fewer updates. The toy has no adversarial pressure, so this does not show that judge guards are harmful in general.
- **H4:** the interaction term was slightly larger on rejected rounds (small, confounded).
- All arms stay near the zero-predictor level (about 1.29) on hard tasks, and the curriculum collapses to easy bins.

## Limitations

- The judge is over-conservative at the default settings (see paper); a recalibrated judge needs a new pre-registered study.
- Synthetic regression only; no language models, no real data engine, no open-ended tasks.
- First-order meta-gradients; no second-order MAML terms.
- Model D here is a fixed, non-learned judge with exact ground truth. A *learned* judge, which is where reward hacking is most likely, is not implemented.
- The workspace carries numeric summaries, not free text, so prompt-injection risks of a real LLM workspace are not exercised.
- Shapley attribution covers A and B only.

## Related work

- Darwin Gödel Machine (Zhang et al., 2025, arXiv:2505.22954) and Hyperagents (Zhang et al., 2026, arXiv:2603.19461): self-modifying agents with empirical validation.
- SIA (Hebbar et al., 2026, arXiv:2605.27276): joint harness and weight updates in a self-improving loop.
- SEAL (Zweiger et al., NeurIPS 2025, arXiv:2506.10943): self-generated finetuning data with RL on downstream reward.
- Absolute Zero (Zhao et al., 2025, arXiv:2505.03335): proposer/solver self-play with a code executor.
- Text-to-LoRA (Charakorn et al., ICML 2025) and Drag-and-Drop LLMs (Liang et al., NeurIPS 2025, arXiv:2506.16406): hypernetwork weight generation.
- "Theater of Mind" for LLMs, a Global Workspace Theory architecture (Shang, 2026, arXiv:2604.08206), BIGMAS (arXiv:2603.15371), blackboard LLM multi-agent systems (arXiv:2507.01701): workspace-style coordination.
- Judge exploitation: Gao et al. (arXiv:2210.10760), Coste et al. (arXiv:2310.02743), and "More Convincing, Not More Correct" (arXiv:2607.05904).
- Recursive self-improvement survey (Chen et al., 2026, arXiv:2607.07663).
- Foundations: Baars (1988) on Global Workspace Theory; Finn et al. (2017, arXiv:1703.03400) on MAML.

Individual components here have prior art; the intended contribution is the combination and its ablations.

## AI assistance disclosure

This code and documentation were developed with AI assistance (Anthropic's Claude), with the author directing the design and responsible for the repository.

## Citation

Study 1 technical report: `paper.pdf` / `paper.tex`. Software citation:

```bibtex
@software{ouadoukou2026siloop,
  author = {Ouadoukou, Abdelkabir},
  title  = {SI-Loop: a research scaffold for guarded self-improvement with a competitive global workspace},
  year   = {2026},
  note   = {Research scaffold, version 0.1}
}
```

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Abdelkabir Ouadoukou.
