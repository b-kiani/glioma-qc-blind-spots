"""Fig. 1 for the ISBI revision, generated from the per-case evaluation file.

(A, B) failures captured vs review count for three parameter-free rankings.
(C) uncertainty rank of every failure, by failure type: shows both blind spots.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(sys.argv[1])
out = sys.argv[2]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#1f1f1e", "#6b6b66"
NS = np.array([14, 27, 41, 54, 81])
N = len(d)

def prep(m):
    dice = d[f"mean_dice_legacy_{m}"].values
    y = (dice < 0.5).astype(int)
    u = d[f"mean_fgvar_fg_{m}"].values
    lv = d[f"log_pred_fg_volume_{m}"].values
    empty = lv == 0
    wt = d[f"dice_WT_{m}"].values
    return dice, y, u, lv, empty, wt

def order(s):
    return np.lexsort((np.arange(N), -s))

# drawn at its printed width (3.35 in, one ISBI column) so that every label prints at >= 8 pt
fig = plt.figure(figsize=(3.35, 4.55), dpi=300)
gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.135, right=0.985, top=0.95, bottom=0.615)
gsc = fig.add_gridspec(1, 1, left=0.375, right=0.955, top=0.405, bottom=0.095)

# ---------------- A, B: capture curves ----------------
for k, (m, name) in enumerate((("base", "Baseline"), ("topo", "TopoGraph"))):
    ax = fig.add_subplot(gs[0, k])
    dice, y, u, lv, empty, wt = prep(m)
    F = y.sum()
    rules = [
        ("Empty first, then U", np.where(empty, np.inf, u), AQUA, "^", "-"),
        ("Uncertainty (U)", u, BLUE, "o", "-"),
        ("Volume (V)", -lv, ORANGE, "s", "--"),
    ]
    for lab, s, c, mk, ls in rules:
        o = order(s)
        cap = np.array([y[o[:n]].sum() for n in NS]) / F * 100
        ax.plot(NS, cap, color=c, marker=mk, ms=3.2, lw=1.2, ls=ls, mec="white", mew=0.5,
                label=lab, zorder=3)
    ax.plot(NS, NS / N * 100, color=GREY, lw=0.9, ls=":", zorder=2)
    ax.axvline(54, color="#c9c8c2", lw=0.6, zorder=1)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_ylim(0, 100)
    ax.set_xlim(8, 86)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e6e5e0", lw=0.5, zorder=0)
    ax.set_title(f"{'AB'[k]}  {name}", loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("Timepoints reviewed")
    if k == 0:
        ax.set_ylabel("Failures captured (%)")
    # values at n = 54 in a block (avoids label collisions between close lines)
    caps = {lab: y[order(s)[:54]].sum() for lab, s, *_ in rules}
    ax.text(0.03, 0.98, "At n = 54\n"
            f"E {caps['Empty first, then U']}/{F}\n"
            f"U {caps['Uncertainty (U)']}/{F}\n"
            f"V {caps['Volume (V)']}/{F}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK,
            linespacing=1.1)
    ax.text(84, 81 / N * 100 + 3, "random", fontsize=8, color=MUTED, ha="right", va="bottom")
    if k == 1:
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, ["Empty first, then U (E)", "Uncertainty (U)", "Volume (V)"], loc="upper center", bbox_to_anchor=(0.55, 0.522), ncol=2,
                   frameon=False, handlelength=2.0, columnspacing=1.0, handletextpad=0.4, labelspacing=0.2)

# ---------------- C: uncertainty rank of each failure, by type ----------------
ax = fig.add_subplot(gsc[0, 0])
cats = [
    ("Empty prediction", lambda e, w: e),
    ("Whole tumour missed\n(WT Dice < 0.5)", lambda e, w: ~e & (w < 0.5)),
    ("Sub-region failure\n(0.5 ≤ WT < 0.8)", lambda e, w: ~e & (w >= 0.5) & (w < 0.8)),
    ("Sub-region failure\n(WT ≥ 0.8)", lambda e, w: ~e & (w >= 0.8)),
]
rng = np.random.default_rng(3)
for j, (m, name, mk, fc) in enumerate((("base", "Baseline", "o", BLUE), ("topo", "TopoGraph", "D", "white"))):
    dice, y, u, lv, empty, wt = prep(m)
    rank = np.empty(N, int)
    rank[order(u)] = np.arange(1, N + 1)
    for i, (lab, f) in enumerate(cats):
        sel = (y == 1) & f(empty, wt)
        yy = len(cats) - 1 - i + (0.13 if j == 0 else -0.13)
        r = rank[sel]
        ax.scatter(r, np.full(len(r), yy) + rng.uniform(-0.05, 0.05, len(r)), s=12, marker=mk,
                   facecolor=fc, edgecolor=BLUE, linewidth=0.6, zorder=3,
                   label=name if i == 0 else None)
ax.axvspan(0.5, 54.5, color="#eef4fc", zorder=0)
ax.axvline(54.5, color=BLUE, lw=0.6, ls="--", zorder=1)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([c[0] for c in cats][::-1], fontsize=8)
ax.set_ylim(-0.5, len(cats) - 0.15)
ax.set_xlim(0, 280)
ax.set_xticks([1, 54, 150, 271])
ax.set_xlabel("Rank by uncertainty (1 = most uncertain)")
ax.set_title("C  Uncertainty rank of every failure", loc="left", color=INK,
             fontweight="bold", x=-0.60)
hc, lc = ax.get_legend_handles_labels()
fig.legend(hc, lc, loc="lower left", bbox_to_anchor=(0.02, 0.0), frameon=False, handletextpad=0.3,
           labelspacing=0.25, markerscale=1.1)
ax.grid(axis="x", color="#e6e5e0", lw=0.5, zorder=0)
ax.tick_params(axis="y", length=0)
for t in ax.get_yticklabels():
    t.set_linespacing(0.95)

fig.savefig(out, dpi=300)
print("saved", out)
