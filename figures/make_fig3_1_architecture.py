"""Thesis Figure 3.1 -- architecture of the deployed two-stage system.

Clone of the original Figure 3.1 with two corrections taken from
final_model_v7_1_reject_stages.ipynb (43 + 31 + 25 = 99 features; gate threshold t = 0.7
selected on SELV) and the combiner expanded into its static and dynamic terms, as in
v71b_architecture.png. Only the deployed rule (HYBRID W x OLA-cw, k=30, p=3) is drawn.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

INK = "#232b36"
C = dict(
    io="#dfe6f2", feat="#e8edf5", gate="#e3f0e5", grey="#eeeeee",
    poolf="#faf8f1", poole="#b08d3a", member="#fdf1d6",
    prob="#ebe4f1", stat="#e2efdc", state="#5a8f4e", dyn="#fde9d3", dyne="#c98a1c",
    fuse="#fdf1d6", fam="#f6e2e2",
)
W_, H_ = 100, 150

fig, ax = plt.subplots(figsize=(10, 15))
ax.set_xlim(0, W_)
ax.set_ylim(H_, 0)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, fc, ec=INK, lw=2.0, ls="-", r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                                zorder=z))


def txt(x, y, s, size=12, weight="normal", color=INK, ha="center", style="normal", z=6):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center",
            style=style, zorder=z, linespacing=1.15)


def arrow(x1, y1, x2, y2, lw=2.0, color=INK, ms=17, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z))


def line(xs, ys, lw=2.0, color=INK, z=4):
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt")


def titled(cx, cy, w, h, fc, title, sub, ec=INK, tsize=15, ssize=12.5):
    box(cx, cy, w, h, fc, ec)
    txt(cx, cy - h * 0.19, title, size=tsize, weight="bold")
    txt(cx, cy + h * 0.22, sub, size=ssize)


# ---------------------------------------------------------------- input + features
titled(50, 6, 56, 7.5, C["io"], "Network flow record",
       "76 CICFlowMeter statistics + destination port, protocol")
arrow(50, 9.75, 50, 14.2)
titled(50, 19, 74, 9, C["feat"], "Feature construction layer",
       "43 pruned base features + 31 behavioural ratios + 25 service-identity features = 99")
arrow(50, 23.5, 50, 28.2)

# ---------------------------------------------------------------- Stage A
box(50, 33.5, 40, 10.5, C["gate"])
txt(50, 30.6, "Stage A: detection gate", size=15, weight="bold")
txt(50, 34.0, "binary XGBoost, balanced weights", size=12.5)
txt(50, 36.9, "threshold t = 0.7 (selected on held-out data)", size=11, style="italic")

# benign exit
line([30, 12], [33.5, 33.5])
arrow(12, 33.5, 12, 44.6)
txt(21, 31.4, "$P$(attack) $\\leq$ $t$", size=12.5, style="italic")
titled(12, 49, 19, 8.8, C["grey"], "Benign", "no alert")

# into Stage B
arrow(50, 38.75, 50, 44.5)
txt(51.5, 41.4, "$P$(attack) > $t$", size=12.5, style="italic", ha="left")

# ---------------------------------------------------------------- Stage B container
X0, X1, Y0, Y1 = 26.5, 97.5, 44.5, 120.0
box((X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0, C["poolf"], C["poole"], lw=2.2, ls="--",
    r=1.2, z=1)
txt(X1 - 1.5, 47.6, "Stage B: attack-family ensemble", size=14.5, weight="bold",
    color="#8a6a1f", ha="right")
txt(X1 - 1.5, 50.3, "trained on attack flows + benign flows that fooled the gate",
    size=10.5, style="italic", color="#6b5520", ha="right")

members = [("XGB", "α = 0"), ("XGB", "α = 0.25"), ("XGB", "α = 0.5"), ("XGB", "α = 1"),
           ("XGB", "SMOTE"), ("RF", "α = 0.25"), ("ET", "α = 0.25")]
mx = [31.0 + i * 10.4 for i in range(7)]
ybar, ymem = 53.5, 60.5
line([mx[0], mx[-1]], [ybar, ybar])
line([50, 50], [44.5, ybar])
for x, (a, b) in zip(mx, members):
    arrow(x, ybar, x, ymem - 4.4)
    box(x, ymem, 9.0, 8.8, C["member"])
    txt(x, ymem - 1.6, a, size=12.5, weight="bold")
    txt(x, ymem + 1.8, b, size=12.5, weight="bold")
    arrow(x, ymem + 4.4, x, 67.6)

PCX = (X0 + X1) / 2
box(PCX, 70, 66, 4.8, C["prob"], lw=1.6)
txt(PCX, 70, "each member: 10 probabilities  $P_m(c\\,|\\,x)$  =  9 families + reject",
    size=12)

# ---------------------------------------------------------------- expanded combiner
box(PCX, 95.5, 67, 45, "#fffdf6", INK, lw=2.0, r=1.0, z=1)
txt(PCX - 31.5, 76.6, "Combiner (hybrid rule)", size=15, weight="bold", ha="left")

SX, DX, TY, TH = 45.0, 79.5, 90.5, 20.5
# static term
line([PCX, PCX], [72.4, 78.6]); line([SX, DX], [78.6, 78.6])
arrow(SX, 78.6, SX, TY - TH / 2); arrow(DX, 78.6, DX, TY - TH / 2)
box(SX, TY, 31, TH, C["stat"], C["state"], lw=1.8)
txt(SX, 82.6, "Static:  per-class weights  $W$", size=13.5, weight="bold", color="#2f5f27")
# descriptor tags on the bottom edge, left of the outgoing lines
for x, tag, ec, col in ((SX, "same for every flow", C["state"], "#2f5f27"),
                        (DX, "recomputed per flow", C["dyne"], "#8a5a00")):
    box(x - 7.6, TY + TH / 2, 14, 2.5, "white", ec, lw=1.4, r=1.0, z=5)
    txt(x - 7.6, TY + TH / 2, tag, size=9.5, weight="bold", color=col, z=7)
txt(SX, 85.9, "7 members × 10 classes = 70 weights", size=11)
box(SX, 90.4, 25, 3.8, "white", C["state"], lw=1.1, r=0.5, z=3)
txt(SX, 90.4, "$W[m,c]$", size=13)
txt(SX, 94.6, "how reliable member $m$ is\non class $c$ in general", size=11, style="italic")
txt(SX, 98.6, "fitted on end-to-end macro-F1", size=10.5, weight="bold", color="#2f5f27")

# dynamic term
box(DX, TY, 31, TH, C["dyn"], C["dyne"], lw=1.8)
txt(DX, 82.6, "Dynamic:  local competence  $\\tilde{w}_m(x)$", size=13.5, weight="bold",
    color="#8a5a00")
steps = ["70-value probability profile of the flow",
         "30 nearest neighbours in that space",
         "how often each member was right there",
         "sharpen:  $\\tilde{w} = w^p / \\Sigma\\, w^p$,  $p = 3$"]
for i, s in enumerate(steps):
    yy = 86.2 + i * 3.6
    box(DX, yy, 28, 2.7, "white", C["dyne"], lw=1.0, r=0.45, z=3)
    txt(DX, yy, s, size=10.3)
    if i < 3:
        arrow(DX, yy + 1.35, DX, yy + 2.25, lw=1.2, ms=10)

# fusion
FY = 111.0
line([SX, SX], [TY + TH / 2, 103.6]); line([DX, DX], [TY + TH / 2, 103.6])
line([SX, DX], [103.6, 103.6]); arrow(PCX, 103.6, PCX, FY - 3.4)
box(PCX, FY, 58, 6.8, C["fuse"])
txt(PCX, FY - 1.2, "score$(c) \\;=\\; \\Sigma_m \\;\\tilde{w}_m(x)\\,\\cdot\\,W[m,c]\\,\\cdot\\,P_m(c\\,|\\,x)$",
    size=14)
txt(PCX, FY + 2.2, "hybrid: class-level weight × flow-level weight", size=10.5, style="italic")

arrow(PCX, FY + 3.4, PCX, Y1 + 4.6)

# ---------------------------------------------------------------- decision
DY = Y1 + 9
titled(PCX, DY, 38, 8.8, C["prob"], "Decision", "argmax over 9 families + reject")
arrow(PCX - 19, DY, 40.6, DY)
titled(32, DY, 17, 8.8, C["grey"], "Reject", "→ Benign")
arrow(PCX + 19, DY, 83.4, DY)
titled(90, DY, 13, 8.8, C["fam"], "Family", "1 of 9")

# ---------------------------------------------------------------- output
OY = 145.0
titled(50, OY, 94, 9, C["io"], "System output",
       "Benign (no alert)   or   ALERT + attack family: Analysis, Backdoor, DoS, Exploits, "
       "Fuzzers,\nGeneric, Reconnaissance, Shellcode, Worms", tsize=15, ssize=12)
arrow(12, 53.4, 12, OY - 4.5)
arrow(32, DY + 4.4, 32, OY - 4.5)
arrow(90, DY + 4.4, 90, OY - 4.5)

ax.set_ylim(OY + 5, -0.5)
plt.tight_layout(pad=0.3)
out = Path(__file__).resolve().parent / "fig3_1_architecture.png"
fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight")
print("wrote", out)
