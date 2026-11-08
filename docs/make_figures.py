"""Regenerates every image in docs/img/ from the raw results. Run:  python docs/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, GREY, INK, GREEN = "#1f4e79", "#c0392b", "#7f8c8d", "#1b2631", "#1e8449"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=12, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, label="", color=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label, ha="center", fontsize=8.5, color=color, style="italic")


# 1. logo ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal")
ax.axis("off")
ax.add_patch(plt.Circle((0, 0), 1, fc=BLUE))
th = [i / 100 * 1.75 * 3.14159 for i in range(100)]
ax.plot([0.45 * __import__("math").cos(t) for t in th], [0.45 * __import__("math").sin(t) for t in th], color="white", lw=5)
ax.annotate("", xy=(0.45, 0.02), xytext=(0.42, 0.18), arrowprops=dict(arrowstyle="-|>", color="white", lw=5))
ax.text(0, -0.02, "SI", ha="center", va="center", color="white", fontsize=34, weight="bold")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
fig.savefig(OUT / "logo.png", dpi=200, bbox_inches="tight", transparent=True)
plt.close(fig)

# 2. architecture / trust boundaries -----------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
box(ax, 0.5, 4.5, 9, 1.6, "IMMUTABLE KERNEL  (kernel/, digest-checked every cycle)",
    "Orchestrator  ->  two-stage judge  ->  gate  ->  git promote / rollback\nledger  |  circuit breakers  |  anchor & held-out audits", "#d6eaf8")
box(ax, 0.5, 1.9, 4.2, 2.0, "AGENT WORKTREE  (agent/**)",
    "the only writable subtree\nmeta-agent READS it and returns\none validated JSON proposal", "#fdebd0")
box(ax, 5.3, 1.9, 4.2, 2.0, "SEALED VERIFIER  (study2/)",
    "secret (HMAC problem ids), oracles,\nhidden pytest, held-out family\ncandidate runs in a sandbox process", "#d5f5e3")
arrow(ax, 2.6, 4.5, 2.6, 3.92, "", RED)
arrow(ax, 2.7, 3.92, 2.7, 4.5, "", INK)
ax.text(2.85, 4.2, "proposal (strict JSON, AST scan)", fontsize=8.5, color=RED, style="italic", va="center")
arrow(ax, 7.4, 3.92, 7.4, 4.5, "", GREEN)
ax.text(7.55, 4.2, "EvalResult only (booleans, flags)", fontsize=8.5, color=GREEN, style="italic", va="center")
arrow(ax, 5.3, 2.9, 4.7, 2.9, "", GREY)
ax.text(5.0, 3.05, "inputs only", fontsize=8.5, color=GREY, style="italic", ha="center")
ax.text(5.0, 1.0, "What the agent never sees:  hidden tests, expected values, pass counts,\n"
        "per-family numbers, audit results, held-out results.", ha="center", fontsize=10, color=INK)
ax.text(5.0, 0.25, "Feedback to the agent is coarse: accept | stepping_stone | reject + error type",
        ha="center", fontsize=10, color=BLUE, weight="bold")
fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 3. the two-stage gate -------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 3.8))
ax.axis("off")
ax.set_xlim(0, 13)
ax.set_ylim(0, 3.8)
steps = [(0.2, "Proposal", "candidate patch\nin its own worktree", "#fdebd0"),
         (3.45, "Stage 1: screen", "batch A, p_screen = 0.80\nchunked, e-value harm stop\nSPRT futility stop", "#d6eaf8"),
         (6.7, "Stage 2: confirm", "independent batch B\nstrict gate, p_min = 0.95\n(no winner's curse)", "#d6eaf8"),
         (9.95, "Decision", "accept -> promote\nstepping stone -> archive\nreject -> roll back", "#d5f5e3")]
for x, t, b, c in steps:
    box(ax, x, 0.6, 2.85, 2.5, t, b, c)
for x1, x2 in [(3.05, 3.45), (6.3, 6.7), (9.55, 9.95)]:
    arrow(ax, x1, 1.85, x2, 1.85)
ax.text(6.5, 0.2, "A candidate that fails the screen is dropped early. Only acceptance on batch B can promote it.",
        ha="center", fontsize=10, color=INK)
fig.savefig(OUT / "gate.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 4. calibration (from calibration.json) -------------------------------------
cal = json.loads((ROOT / "results/study2/calibration.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
g = [r["p_improve_min"] for r in cal["false_accept"]]
ax[0].plot(g, [100 * r["single"] for r in cal["false_accept"]], "o-", color=RED, label="single stage")
ax[0].plot(g, [100 * r["two_stage"] for r in cal["false_accept"]], "s-", color=BLUE, label="two stage (default)")
ax[0].axhline(5, ls=":", color=GREY)
ax[0].set(xlabel="p_improve_min", ylabel="false-accept rate of null patches (%)", title="A/A calibration (n=200)")
ax[0].legend(frameon=False)
l = [100 * r["lift"] for r in cal["power"]]
ax[1].plot(l, [100 * r["single"] for r in cal["power"]], "o-", color=RED, label="single stage")
ax[1].plot(l, [100 * r["two_stage"] for r in cal["power"]], "s-", color=BLUE, label="two stage")
ax[1].set(xlabel="true pass-rate gain of the patch (points)", ylabel="accept rate (%)", title="Power")
ax[1].legend(frameon=False)
for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=180)
plt.close(fig)

# 5. ablation arms (every seed shown) ----------------------------------------
arms = {}
for p in sorted((ROOT / "results/study2/ablation").glob("A*_s*/result.json")):
    r = json.loads(p.read_text())
    arms.setdefault(r["arm"], []).append(r["gain_selection"])
names = {"A0": "A0\nfrozen improver", "A1": "A1\nfull recursion", "A2": "A2\nrandom patches", "A3": "A3\nbest-of-N"}
fig, ax = plt.subplots(figsize=(8, 4.2))
import statistics as st
for i, a in enumerate(["A0", "A1", "A2", "A3"]):
    v = arms.get(a, [])
    if not v:
        continue
    ax.bar(i, st.mean(v), color=[GREY, BLUE, RED, GREEN][i], alpha=0.85, width=0.6)
    ax.errorbar(i, st.mean(v), yerr=st.stdev(v), color=INK, capsize=6, lw=1.4)
    ax.scatter([i + (k - len(v) / 2) * 0.03 for k in range(len(v))], v, color=INK, s=12, zorder=3)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(4), [names[a] for a in ["A0", "A1", "A2", "A3"]])
ax.set_ylabel("final gain over cycle-0 anchor (selection families)")
ax.set_title("Pre-registered ablation pilot: mean +- SD, each dot is one seed", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ablation.png", dpi=180)
plt.close(fig)

# 6. verifier quality ---------------------------------------------------------
vq = json.loads((ROOT / "results/study2/verifier_quality.json").read_text())
tiers = ["small", "fuzz", "meta"]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
kills = [vq["kills_by_tier"].get(t, 0) for t in tiers]
only = [vq["kills_only_this_tier"].get(t, 0) for t in tiers]
x = range(3)
ax.bar([i - 0.2 for i in x], kills, width=0.4, color=BLUE, label="killed by this tier (any)")
ax.bar([i + 0.2 for i in x], only, width=0.4, color=GREEN, label="killed ONLY by this tier")
ax.set_xticks(list(x), ["small-scope\nexhaustive", "random fuzz", "metamorphic"])
ax.set_ylabel("mutants")
ax.set_title(f"Verifier kill rate {100 * vq['kill_rate']:.1f}% of {vq['mutants']} injected faults", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "verifier.png", dpi=180)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
