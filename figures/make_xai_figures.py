"""XAI chapter figures, from results/xai_treeshap/xai_treeshap.json (matplotlib only).

Group colours are fixed across every figure: SVC blue, TIME orange, SIZE aqua, FLAG yellow
(validated categorical slots 1-4; aqua/yellow sit below 3:1 on white, so every segment that
carries a value is direct-labelled).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
R = json.loads((ROOT / "results" / "xai_treeshap" / "xai_treeshap.json").read_text(encoding="utf-8"))

GROUPS = ["SVC", "TIME", "SIZE", "FLAG"]
GLABEL = {"SVC": "Service identity (SVC)", "TIME": "Timing (TIME)",
          "SIZE": "Size / length (SIZE)", "FLAG": "TCP flags (FLAG)"}
GC = {"SVC": "#2a78d6", "TIME": "#eb6834", "SIZE": "#1baf7a", "FLAG": "#eda100"}
SURF, INK, INK2, GRID, NEUTRAL = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def legend_groups(fig, title):
    from matplotlib.patches import Patch
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(handles=[Patch(color=GC[g], label=GLABEL[g]) for g in GROUPS], frameon=False,
               ncol=4, fontsize=9, labelcolor=INK2, handlelength=1.2,
               loc="upper left", bbox_to_anchor=(0.0, 0.93))


def stacked_rows(ax, labels, rows, min_label=0.045):
    """100% stacked horizontal bars with a 2px surface gap and direct percentage labels."""
    y = np.arange(len(labels))[::-1]
    for yi, row in zip(y, rows):
        left = 0.0
        for g, v in zip(GROUPS, row):
            ax.barh(yi, v, left=left, height=0.62, color=GC[g], edgecolor=SURF, linewidth=2)
            if v >= min_label:
                ax.text(left + v / 2, yi, "%d%%" % round(100 * v), ha="center", va="center",
                        fontsize=8.5, color=INK if g in ("SIZE", "FLAG") else "white")
            left += v
    ax.set_yticks(y); ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------- Fig 1: SHAP vs gain shares
S = R["shares"]
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = ["All columns (reference)",
          "Gate - SHAP", "Gate - gain importance",
          "Stage B ensemble - SHAP", "Stage B (XGB a=0.25) - gain importance"]
rows = [S["columns"], S["gate_shap"], S["gate_gain"], S["stageB_ensemble_shap"], S["xgb_a025_gain"]]
stacked_rows(ax, labels, rows)
for yy in (3.5, 1.5):
    ax.axhline(yy, color=GRID, lw=1)
ax.set_xlabel("Share of attribution by feature group")
legend_groups(fig, "Gain importance overstates the leading group at both stages")
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT / "xai_fig1_shap_vs_gain.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: per-family reliance
PC = R["per_class_share"]
order = sorted(PC, key=lambda k: -PC[k][0])
nice = {"Benign": "Benign (gate leaks)", "Reconnaissance": "Reconnaissance"}
fig, ax = plt.subplots(figsize=(9, 4.8))
stacked_rows(ax, [nice.get(k, k) for k in order], [PC[k] for k in order])
ax.set_xlabel("Share of mean |SHAP| on the true-class ensemble score")
legend_groups(fig, "Each family leans on different evidence")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "xai_fig2_per_family.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: overlap pairs
short = lambda s: s.replace("Reconnaissance", "Recon")
P = R["pairs"]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True, sharey=True)
xmax = max(max(abs(p["mean_D"][g]) + p["ci95"][g] for g in GROUPS) for p in P) * 1.15
for ax, p in zip(axes.flat, P):
    y = np.arange(len(GROUPS))[::-1]
    for yi, g in zip(y, GROUPS):
        v, e = p["mean_D"][g], p["ci95"][g]
        ax.barh(yi, v, height=0.6, color=GC[g], edgecolor=SURF, linewidth=2)
        ax.errorbar(v, yi, xerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(GROUPS, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-xmax * 0.35, xmax)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("%s  (n=%d)\ndriver: %s" % (short(p["pair"]).replace("->", "→"), p["n"], p["driver"]),
                 loc="left", fontsize=9.5)
for ax in axes[-1]:
    ax.set_xlabel("Push toward wrong class (mean D)")
fig.suptitle("What pushes a confused row toward the wrong family", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.text(0.01, 0.925, "D = SHAP(wrong-class score) − SHAP(true-class score); "
         "bars right of 0 push toward the error; whiskers = 95% CI",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "xai_fig3_overlap_pairs.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: faithfulness
F = R["faithfulness"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for ax, (nm, title) in zip(axes, [("gate", "Gate (benign vs attack)"),
                                  ("XGB a=0.25", "Stage B member XGB a=0.25")]):
    ks = sorted(int(k) for k in F[nm])
    top = [F[nm][str(k)]["top"] for k in ks]; rnd = [F[nm][str(k)]["random"] for k in ks]
    ax.plot(ks, top, color=GC["SVC"], lw=2, marker="o", ms=6)
    ax.plot(ks, rnd, color=NEUTRAL, lw=2, marker="o", ms=6)
    ax.text(ks[-1] + 0.3, top[-1], "top-k SHAP", va="center", fontsize=9, color=INK)
    ax.text(ks[-1] + 0.3, rnd[-1], "random k", va="center", fontsize=9, color=INK2)
    k5 = ks.index(5)
    ax.annotate("%.1f× at k=5" % F[nm]["5"]["ratio"], (5, top[k5]), xytext=(5, top[k5] - 0.13 * max(top)),
                ha="center", fontsize=9, color=INK2)
    ax.set_xticks(ks); ax.set_xlim(0.5, ks[-1] + 3.2); ax.set_ylim(0, max(top) * 1.12)
    ax.set_xlabel("Columns deleted (set to train median)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("Drop in explained margin (log-odds)")
fig.suptitle("Explanations are faithful: deleting top-SHAP columns hurts 11-12× more than random",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "xai_fig4_faithfulness.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: stability under noise
ST = R["stability"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, (key, title, thr) in zip(axes, [
        ("median_spearman", "Full ranking (median row Spearman)", 0.90),
        ("median_jaccard5", "Top-5 columns (median Jaccard)", 0.60)]):
    xs = np.arange(2); w = 0.32
    for j, (nm, lab) in enumerate([("gate", "Gate"), ("XGB a=0.25", "XGB a=0.25")]):
        for b_i, b in enumerate(["0.002", "0.02"]):
            d = ST[nm][b]
            for kind, col, off in [("stable_rows", GC["SVC"], -w / 2), ("flipped_rows", NEUTRAL, w / 2)]:
                v = d["%s_%s" % (kind, key)]
                x = j * 2.6 + b_i + off
                ax.bar(x, v, width=w, color=col, edgecolor=SURF, linewidth=2)
                ax.text(x, v - 0.02, "%.2f" % v, ha="center", va="top", fontsize=7.5,
                        color="white" if kind == "stable_rows" else INK)
        ax.text(j * 2.6 + 0.5, -0.17, lab, ha="center", fontsize=9.5, color=INK,
                transform=ax.get_xaxis_transform())
    ax.axhline(thr, color=INK2, lw=1, ls="--")
    ax.set_xticks([0, 1, 2.6, 3.6]); ax.set_xticklabels(["0.002", "0.020"] * 2)
    ax.set_xlim(-0.6, 4.3); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Noise budget (IQR)", labelpad=18)
    ax.set_title("%s\ndashed line = pre-registered threshold %.2f" % (title, thr), loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=GC["SVC"], label="decision unchanged"),
                        Patch(color=NEUTRAL, label="decision flipped")],
               frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
fig.suptitle("The top of an explanation is mostly stable under noise; the full ranking is not",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "xai_fig5_stability.png", dpi=200)
plt.close(fig)
print("wrote 5 figures to", OUT)
