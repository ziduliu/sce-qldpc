#!/usr/bin/env python3
"""Circuit-level sweep in two layouts (A: one axis; B: one panel per rate cell) from
figures/data/pl_sweep_20261030_onecircuit_fig.json; also provides the helpers used by pl_sweep_61_ab_v.py.
Env: PL_CELLS (default 0-5), PL_SKIP ("plc:0,plc:4"). Writes pl_sweep_61_cleanA/B.{pdf,png}."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8})
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "..", "data", "pl_sweep_20261030_onecircuit_fig.json")))
CELLS = [int(x) for x in os.environ.get("PL_CELLS", "0,1,2,3,4,5").split(",") if x]
SKIP = {(a, int(c)) for a, c in (x.split(":") for x in os.environ.get("PL_SKIP", "").split(",") if x)}
RATE = {0: "<0.125", 1: "0.15", 2: "0.20", 3: "0.25", 4: "0.30", 5: "0.35", 6: "0.40", 7: "0.45", 8: "0.50", 9: ">0.525"}
# ordered palette (dark to light along the rate) and one marker per rate
PAL = ["#3b1f5e", "#5b3a8e", "#2f6fa6", "#1f9e89", "#7bbf4a", "#e0a32a", "#d1603d", "#a13d63", "#6b6b6b", "#999999"]
MARK = {0: "o", 1: "s", 2: "^", 3: "v", 4: "p", 5: "h", 6: "8", 7: "<", 8: ">", 9: "*"}
REF = "#9a9a9a"


# points drawn although they have fewer than ten failures
EXTRA = {("dist2", 0, 0.002)}


def shown(r):
    return r.get("drawn") or (r["arm"], r["cell"], round(r["p"], 6)) in EXTRA


def pts(arm, c):
    if (arm, c) in SKIP:
        return []
    out = []
    for r in sorted((r for r in rows if r["arm"] == arm and (arm == "bb144" or r["cell"] == c)), key=lambda r: r["p"]):
        if not shown(r):
            continue
        wb = r.get("worse_basis") or "Z"
        x, n = int(r[wb + "_shot_failures"]), int(r[wb + "_shots"])
        conv = lambda P: (1.0 - (1.0 - P) ** (1.0 / float(r["rounds"]))) / float(r["k"])
        lo = conv(beta_dist.ppf(0.025, x, n - x + 1)) if x > 0 else 0.0
        hi = conv(beta_dist.ppf(0.975, x + 1, n - x))
        out.append((r["p"], r[wb + "_pl_round_per_logical"], lo, hi))
    return out


def draw(ax, data, color, marker, filled, ls, size=46, lw=1.4, z=3):
    if not data:
        return
    x = [d[0] for d in data]; y = [d[1] for d in data]
    ax.plot(x, y, ls=ls, color=color, lw=lw, zorder=z - 1)
    ax.errorbar(x, y, yerr=[[d[1] - d[2] for d in data], [d[3] - d[1] for d in data]], fmt="none",
                ecolor=color, elinewidth=0.9, capsize=2, zorder=z - 0.5)
    ax.scatter(x, y, s=size, marker=marker, facecolors=color if filled else "white", edgecolors=color,
               linewidths=1.2, zorder=z)


def reference(ax, size=34):
    ref = pts("bb144", None)
    draw(ax, ref, REF, "D", False, ":", size=size, lw=1.1, z=2)
    # power-law fit of the reference through its drawn points, continued to 0.001
    p = np.log([d[0] for d in ref]); y = np.log([d[1] for d in ref])
    b, a = np.polyfit(p, y, 1)
    pp = np.logspace(np.log10(0.9e-3), np.log10(min(d[0] for d in ref)), 40)
    ax.plot(pp, np.exp(a) * pp ** b, ls=":", color=REF, lw=0.9, zorder=1)


def style(ax, compact=False):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.88e-3, 5.6e-3); ax.set_ylim(1e-10, 1e-2)
    if compact:
        ax.set_xticks([0.001, 0.002, 0.003, 0.005]); ax.set_xticklabels(["1", "2", "3", "5"])
    else:
        ax.set_xticks([0.001, 0.002, 0.003, 0.004, 0.005]); ax.set_xticklabels(["0.001", "0.002", "0.003", "0.004", "0.005"])
    ax.minorticks_off()
    ax.set_yticks([1e-9, 1e-7, 1e-5, 1e-3])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6, zorder=0)


# ---------------- layout A: one axis ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.6))
reference(ax)
for c in CELLS:
    col, mk = PAL[c], MARK[c]
    draw(ax, pts("dist", c), col, mk, False, (0, (4, 2)), lw=1.2)
    if c == 0:
        draw(ax, pts("dist2", 0), col, mk, False, (0, (5, 1.5, 1, 1.5)), lw=1.0)
    draw(ax, pts("plc", c), col, mk, True, "-", lw=1.6)
style(ax)
ax.set_xlabel("physical error rate $p$")
ax.set_ylabel("failure rate per round and logical qubit")
h_rate = [Line2D([], [], color=PAL[c], marker=MARK[c], ls="none", ms=6.5, label="$k/n$ " + RATE[c]) for c in CELLS]
h_style = [Line2D([], [], color="#333333", ls=(0, (4, 2)), marker="o", mfc="white", ms=5.5, lw=1.2, label="distance champion"),
           Line2D([], [], color="#333333", ls="-", marker="o", ms=5.5, lw=1.6, label="circuit-level champion"),
           Line2D([], [], color="#333333", ls=(0, (5, 1.5, 1, 1.5)), marker="o", mfc="white", ms=5.5, lw=1.0, label="second two-block code ($k/n$<0.125)"),
           Line2D([], [], color=REF, ls=":", marker="D", mfc="white", ms=5, lw=1.1, label="[[144,12,12]]")]
l1 = ax.legend(handles=h_rate, loc="lower right", frameon=False, fontsize=8, ncol=2, columnspacing=0.8, handletextpad=0.3)
ax.add_artist(l1)
ax.legend(handles=h_style, loc="upper left", frameon=False, fontsize=8, handlelength=2.6)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "pl_sweep_61_cleanA." + ext), dpi=220, bbox_inches="tight", pad_inches=0.03)

# ---------------- layout B: one panel per rate cell ----------------
ncol = 3
nrow = int(np.ceil(len(CELLS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.35 * nrow + 0.3), sharex=True, sharey=True)
axes = np.atleast_2d(axes)
for i, c in enumerate(CELLS):
    ax = axes[i // ncol][i % ncol]
    reference(ax, size=22)
    col = PAL[c]
    draw(ax, pts("dist", c), col, MARK[c], False, (0, (4, 2)), size=38, lw=1.2)
    if c == 0:
        draw(ax, pts("dist2", 0), col, MARK[c], False, (0, (5, 1.5, 1, 1.5)), size=38, lw=1.0)
    draw(ax, pts("plc", c), col, MARK[c], True, "-", size=38, lw=1.6)
    style(ax, compact=True)
    ax.text(0.04, 0.95, "$k/n$ " + RATE[c], transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color=col)
    if ("plc", c) in SKIP:
        ax.text(0.96, 0.05, "circuit-level champion\nre-measured", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#888888")
for j in range(len(CELLS), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
for a in axes[-1]:
    a.set_xlabel("$p$ ($10^{-3}$)")
for a in axes[:, 0]:
    a.set_ylabel("failure rate / round / logical")
h = [Line2D([], [], color="#333333", ls=(0, (4, 2)), marker="o", mfc="white", ms=5, lw=1.2, label="distance champion"),
     Line2D([], [], color="#333333", ls="-", marker="o", ms=5, lw=1.6, label="circuit-level champion"),
     Line2D([], [], color="#333333", ls=(0, (5, 1.5, 1, 1.5)), marker="o", mfc="white", ms=5, lw=1.0, label="second two-block code"),
     Line2D([], [], color=REF, ls=":", marker="D", mfc="white", ms=4.5, lw=1.1, label="[[144,12,12]] reference")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "pl_sweep_61_cleanB." + ext), dpi=220, bbox_inches="tight", pad_inches=0.03)
print("written pl_sweep_61_cleanA/B")
