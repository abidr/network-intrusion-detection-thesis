"""v7.1b full-architecture diagram.

Follows the reference layout (Part 3 + Part 4 + Part 5) with one change: the combination
rule is not fixed at HYBRID. Static / Dynamic / Hybrid are three candidates in one nested
family, and the winner is SELECTED PER DATASET on the SELV split.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------- palette
C = dict(
    flow_f="#dae8fc", flow_e="#4a7ebb",
    gate_f="#fff2cc", gate_e="#d6b656",
    good_f="#d5e8d4", good_e="#5a9e56",
    panelB_f="#eef5fd", panelB_e="#1f4e9c",
    model_f="#ffffff", model_e="#4a86c8",
    prob_f="#e9dcf3", prob_e="#7d5ba6",
    stat_f="#d9ead3", stat_e="#5a9e56",
    dyn_f="#ffe6cc", dyn_e="#d79b00",
    sel_f="#fde7ef", sel_e="#c2185b",
    out_f="#d5e8d4", out_e="#3f7a3b",
    note_f="#f7f7f7", note_e="#999999",
)

fig, ax = plt.subplots(figsize=(16.5, 15.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, fc, ec, lw=1.8, r=0.9, ls="-", z=2):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))


def txt(x, y, s, size=11, weight="normal", color="#111111", ha="center", va="center",
        style="normal", z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            style=style, zorder=z)


def arrow(x1, y1, x2, y2, color="#222222", lw=2.0, ls="-", z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
        linewidth=lw, color=color, linestyle=ls, zorder=z,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def elbow(x1, y1, x2, y2, color="#222222", lw=2.0, z=3):
    """Down from (x1,y1), across, then into (x2,y2) pointing down."""
    ymid = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, ymid], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    ax.plot([x1, x2], [ymid, ymid], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    arrow(x2, ymid, x2, y2, color=color, lw=lw, z=z)


# ================================================================ title
txt(50, 98.3, "v7.1b Full Architecture — Two-Stage Detector with Reject Class",
    size=21, weight="bold", color="#1a1a1a")
txt(50, 96.1, "Stage A gate  →  alpha-diverse ensemble  →  combination rule selected per dataset",
    size=12.5, color="#555555", style="italic")

# ================================================================ 1) flow
box(50, 92.4, 24, 3.4, C["flow_f"], C["flow_e"])
txt(50, 92.4, "1)  Network flow", size=13.5, weight="bold", color="#123c72")
arrow(50, 90.7, 50, 88.6)

# ================================================================ 2) gate
box(50, 86.0, 38, 5.0, C["gate_f"], C["gate_e"])
txt(50, 87.2, "2)  Stage A: Gate", size=14, weight="bold", color="#7a5c00")
txt(50, 85.0, "Binary classifier  (Attack vs Benign)  —  outputs probability p",
    size=11, color="#333333")

# benign exit
box(13.5, 86.0, 19, 5.0, C["good_f"], C["good_e"])
txt(13.5, 87.0, "Final output:", size=11.5, color="#26541f")
txt(13.5, 85.0, "BENIGN", size=13, weight="bold", color="#26541f")
arrow(31.0, 86.0, 23.2, 86.0, color="#3f7a3b")
txt(27.0, 87.4, "p ≤ t", size=11, weight="bold", color="#3f7a3b")

arrow(50, 83.5, 50, 81.4, color="#b3261e")
txt(52.0, 82.5, "if p > t   (flagged as Attack)", size=11, weight="bold",
    color="#b3261e", ha="left")

# ================================================================ 3) Stage B pool
box(50, 68.6, 80, 24.2, C["panelB_f"], C["panelB_e"], lw=2.4, r=1.2, z=1)
txt(50, 79.3, "3)  Stage B — alpha-diverse ensemble  (7 members)",
    size=15, weight="bold", color="#1f4e9c")
txt(50, 77.4, "All 7 members see the same rows and the same 99 features. They differ only in "
              "how they weight class imbalance.", size=10.5, color="#333333")

models = [
    ("Member 1:  XGBoost (400 trees, depth 8)", "alpha = 0.0"),
    ("Member 2:  XGBoost (400 trees, depth 8)", "alpha = 0.25"),
    ("Member 3:  XGBoost (400 trees, depth 8)", "alpha = 0.5"),
    ("Member 4:  XGBoost (400 trees, depth 8)", "alpha = 1.0"),
    ("Member 5:  XGBoost + SMOTE", "capped at 2,000 / class"),
    ("Member 6:  Random Forest (300 trees)", "alpha = 0.25"),
    ("Member 7:  Extra Trees (300 trees)", "alpha = 0.25"),
]
y0, dy = 75.0, 2.55
for i, (name, param) in enumerate(models):
    yy = y0 - i * dy
    hl = (i == 3)
    box(37.5, yy, 44, 2.15, "#fff8e1" if hl else C["model_f"],
        "#d79b00" if hl else C["model_e"], lw=1.9 if hl else 1.2, r=0.5, z=3)
    txt(17.0, yy, name, size=10.3, weight="bold", ha="left", color="#123c72")
    txt(58.0, yy, param, size=10.0, ha="right", color="#444444")

# side note
box(78.5, 69.0, 18.5, 12.5, "#ffffff", C["panelB_e"], lw=1.3, ls="--", r=0.6, z=3)
txt(78.5, 74.2, "Role of alpha", size=11, weight="bold", color="#1f4e9c")
txt(78.5, 70.3, "The alpha dial sets how\nmuch a rare-class mistake\ncosts. Different"
                " values give different\nbehaviours, from cautious\nto aggressive.",
    size=9.2, color="#333333")
txt(78.5, 65.2, "Member 4 is weakest individually\nbut is retained: it disagrees\nmost with the others.",
    size=9.0, color="#b3261e", weight="bold")

arrow(50, 56.5, 50, 54.4)

# ================================================================ 4) probability bar
box(50, 51.8, 50, 4.4, C["prob_f"], C["prob_e"], lw=1.8)
txt(50, 52.9, "Each member outputs 10 probabilities", size=12.5, weight="bold", color="#5b3a86")
txt(50, 50.9, "9 attack families  +  REJECT  (a gate false positive)", size=10.5, color="#333333")

# split to the two weight sources
elbow(50, 49.6, 26, 45.6)
elbow(50, 49.6, 74, 45.6)

# ================================================================ 5) static weights
box(26, 38.0, 44, 15.0, C["stat_f"], C["stat_e"], lw=2.0, r=1.0, z=1)
txt(26, 44.0, "4)  Static term — weight table W", size=13, weight="bold", color="#2f6b2b")
txt(26, 41.9, "7 members × 10 classes = 70 learned weights", size=10.3, color="#333333")
box(26, 38.6, 40, 3.0, "#ffffff", C["stat_e"], lw=1.2, r=0.5, z=3)
txt(26, 38.6, "W[m, c]   ×   P$_m$(c | x)", size=12, weight="bold", color="#1a1a1a")
txt(26, 36.2, "General reliability of member m on class c", size=10.2,
    style="italic", color="#333333")
txt(26, 33.5, "Fitted on COMB by coordinate ascent,\nunder the END-TO-END objective.",
    size=9.8, color="#2f6b2b", weight="bold")

# ================================================================ 6) dynamic weights
box(74, 38.0, 44, 15.0, C["dyn_f"], C["dyn_e"], lw=2.0, r=1.0, z=1)
txt(74, 44.0, "5)  Dynamic term — local competence", size=13, weight="bold", color="#a06800")
steps = [
    "build a 70-number prediction fingerprint for this flow",
    "find its 30 nearest neighbours in DECISION space",
    "check how often each member was right on those",
    "sharpen:  w̃ = w$^p$ / Σ w$^p$      (p = 3)",
]
sy = 42.0
for i, s in enumerate(steps):
    box(74, sy - i * 2.35, 40, 1.70, "#ffffff", C["dyn_e"], lw=1.1, r=0.45, z=3)
    txt(74, sy - i * 2.35, s, size=9.9, color="#1a1a1a")
    if i < len(steps) - 1:
        arrow(74, sy - i * 2.35 - 0.90, 74, sy - (i + 1) * 2.35 + 0.88, lw=1.3)
txt(74, 32.6, "Reliability of member m in this flow's local region", size=10.2,
    style="italic", color="#333333")

# both feed the selector
elbow(26, 30.5, 40, 27.3)
elbow(74, 30.5, 60, 27.3)

# ================================================================ 7) THE CHANGE — selection
box(50, 20.2, 86, 14.4, C["sel_f"], C["sel_e"], lw=2.6, r=1.2, z=1)
txt(50, 25.6, "6)  Combination rule — SELECTED PER DATASET on the SELV split",
    size=14.5, weight="bold", color="#a3124f")
txt(50, 23.6, "The three rules are one nested family, so this is a parameter sweep, not "
              "a comparison of unrelated models:  Wclass=None → pure dynamic,  p=0 → pure static.",
    size=10.0, color="#333333")

rules = [
    (20.0, "STATIC", "score(c) = Σ  W[m,c] · P$_m$(c)",
     "class weights only", "#2f6b2b", C["stat_f"], C["stat_e"]),
    (50.0, "DYNAMIC", "score(c) = Σ  w̃$_m$(x) · P$_m$(c)",
     "per-flow weights only", "#a06800", C["dyn_f"], C["dyn_e"]),
    (80.0, "HYBRID", "score(c) = Σ  w̃$_m$(x) · W[m,c] · P$_m$(c)",
     "both terms", "#5b3a86", C["prob_f"], C["prob_e"]),
]
for cx, name, formula, sub, col, fc, ec in rules:
    box(cx, 18.9, 27, 6.4, fc, ec, lw=1.8, r=0.7, z=3)
    txt(cx, 20.9, name, size=12.5, weight="bold", color=col)
    txt(cx, 18.9, formula, size=9.9, color="#1a1a1a")
    txt(cx, 17.1, sub, size=9.3, style="italic", color="#444444")

txt(50, 14.6, "SELV ranks all 21 candidates end-to-end and the winner is refit before test is "
              "opened  →  the SAME protocol picks a different rule on a different dataset.",
    size=10.0, weight="bold", color="#a3124f")

# per-dataset outcome strip
box(50, 10.3, 86, 5.2, "#ffffff", C["sel_e"], lw=1.4, ls="--", r=0.6, z=3)
picks = [
    (17.5, "CIC-UNSW full release", "HYBRID  W × OLA-cw k=30 p=3", "#5b3a86"),
    (41.0, "CIC-UNSW curated (v6)", "DYNAMIC  OLA-cw k=30 p=8", "#a06800"),
    (64.0, "UNSW-NB15 extractor", "STATIC  per-class W", "#2f6b2b"),
    (86.0, "CIC-IDS2017", "any — all tie within 0.0007", "#666666"),
]
for cx, ds, pick, col in picks:
    txt(cx, 11.5, ds, size=9.6, weight="bold", color="#333333")
    txt(cx, 9.5, pick, size=9.4, color=col, weight="bold")
for xsep in (29.2, 52.5, 75.0):
    ax.plot([xsep, xsep], [8.4, 12.3], color="#cccccc", lw=1.0, zorder=4)

arrow(50, 7.6, 50, 6.2, color="#a3124f", lw=2.2)

# ================================================================ 8) argmax + prediction
box(50, 4.9, 42, 2.2, "#f3ecfa", C["prob_e"], lw=1.4, r=0.5)
txt(50, 4.9, "7)  take the class with the highest score  (argmax)", size=10.6,
    weight="bold", color="#5b3a86")
arrow(50, 3.7, 50, 2.6, lw=1.8)

box(50, 1.3, 52, 2.6, C["out_f"], C["out_e"], lw=2.0)
txt(50, 1.3, "8)  Final output:  BENIGN   or   one of 9 attack families",
    size=12, weight="bold", color="#26541f")

# reject side note
box(88.0, 3.4, 20, 4.0, C["note_f"], C["note_e"], lw=1.4, ls="--", r=0.6)
txt(88.0, 4.3, "REJECT is reported", size=9.5, weight="bold", color="#444444")
txt(88.0, 2.6, "as BENIGN by the\nfull system", size=9.2, color="#444444")
ax.add_patch(FancyArrowPatch((71.2, 4.9), (78.0, 3.9), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.4, color="#666666", linestyle="--", zorder=3))

plt.tight_layout(pad=0.4)
out = Path(__file__).resolve().parent
out.mkdir(exist_ok=True)
p = out / "v71b_architecture.png"
fig.savefig(p, dpi=170, facecolor="white", bbox_inches="tight")
print("wrote", p)
