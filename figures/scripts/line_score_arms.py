"""Figure 3: run score against steps for every run of Fig. 3 (one model call or one random proposal per step).
Data: figures/data/line_score_arms_iter.json (per line, the run score after steps 0..500; the seeds are step 0).
Panel (a): FOM objective; panel (b): CLER objective. Language-model runs are drawn as lines with markers every
STEP steps; the five random-mutation repeats and the five hill-climbing repeats as bands (minimum to maximum
across the repeats) with their mean.
Usage: python line_score_arms.py [PAL] [STEP] [LINESTYLE] [YLABEL]; writes line_score_arms.{pdf,png}.
Fira Sans is used when the font is installed, the default font otherwise."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager as fm
_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
if os.path.isdir(_FONTS):
    for _f in os.listdir(_FONTS):
        if _f.endswith(".ttf"):
            fm.fontManager.addfont(os.path.join(_FONTS, _f))
if any(f.name == "Fira Sans" for f in fm.fontManager.ttflist):
    plt.rcParams.update({"font.family": "Fira Sans", "mathtext.fontset": "custom", "mathtext.rm": "Fira Sans",
                         "mathtext.it": "Fira Sans:italic", "mathtext.bf": "Fira Sans:semibold"})
plt.rcParams["pdf.fonttype"] = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", "line_score_arms_iter.json")
D = json.load(open(os.path.join(HERE, "..", "data", DATA)))
TAG = ""
N = 500
import sys
PALETTES = {
    # A: hue = model family (luna blue, sol orange-red), lightness = effort / version; random neutral grey
    "A": dict(v5="#6fbf73", local="#ef7f7f", luna_low="#9ecae1", luna_high="#4292c6", luna_xhigh="#08519c",
              sol6="#fd8d3c", sol61="#b30000"),
    # B: Okabe-Ito colour-blind-safe hues, one per line
    "B": dict(v5="#8c8c8c", local="#3b9c94", luna_low="#56B4E9", luna_high="#009E73", luna_xhigh="#0072B2",
              sol6="#E69F00", sol61="#D55E00"),
    # C: viridis-like ramp from weak to strong (luna low -> 6.1), random grey
    "C": dict(v5="#8c8c8c", local="#3b9c94", luna_low="#7ad151", luna_high="#22a884", luna_xhigh="#2a788e",
              sol6="#414487", sol61="#440154"),
}
PAL = sys.argv[1] if len(sys.argv) > 1 else "A"
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LS = sys.argv[3] if len(sys.argv) > 3 else "--"   # line style between the markers
YL = sys.argv[4] if len(sys.argv) > 4 else "score"  # y-axis label: full (with the formula), score, runscore
YLABELS = {"full": ("run score $\\sum_{\\mathrm{cells}}\\max\\,\\mathcal{S}_{\\mathrm{FOM}}$", "run score $\\sum_{\\mathrm{cells}}\\max\\,\\mathcal{S}_{\\mathrm{CLER}}$"),
           "score": ("score", "score"), "runscore": ("run score", "run score")}
XS = np.unique(np.append(np.arange(0, N + 1, STEP), N))   # markers every STEP iterations plus the final iteration N
MK = dict(luna_low="o", luna_high="s", luna_xhigh="D", gpt6_sol_high="^", gpt61_sol_high="v")
COL = PALETTES[PAL]


def curve(block, key):
    return np.array(D[block][key]["total"][:N + 1], dtype=float)


BAND = os.environ.get("BAND", "minmax")   # minmax (default) or sigma
SIGMA_K = float(os.environ.get("SIGMA_K", "1"))   # band = mean +- SIGMA_K standard deviations
LOCAL = os.environ.get("LOCAL", "1") == "1"       # 1: also draw the local-mutation band
BANDFULL = os.environ.get("BANDFULL", "1") == "1"   # 1: band and mean at every iteration, markers every STEP


def band(ax, curves, col, label, z):
    c = np.array(curves)
    if BAND == "sigma":   # mean +- SIGMA_K standard deviations across the repeats (ddof=1)
        lo, hi = c.mean(0) - SIGMA_K * c.std(0, ddof=1), c.mean(0) + SIGMA_K * c.std(0, ddof=1)
    else:                 # minimum to maximum across the repeats
        lo, hi = c.min(0), c.max(0)
    if BANDFULL:
        x = np.arange(c.shape[1])
        ax.fill_between(x, lo, hi, color=col, alpha=0.20, lw=0, zorder=z)
        ax.plot(x, c.mean(0), color=col, lw=0.9, ls=LS, marker="x", ms=3.0, mew=0.8, markevery=list(XS), zorder=z + 0.2, label=label)
        return
    c, lo, hi = c[:, XS], lo[XS], hi[XS]
    ax.fill_between(XS, lo, hi, color=col, alpha=0.20, lw=0, zorder=z)
    ax.plot(XS, c.mean(0), color=col, lw=0.9, ls=LS, marker="x", ms=3.0, mew=0.8, zorder=z + 0.2, label=label)


def panel(ax, block, rnd_keys, loc_keys, local_label, ylab, legend):
    if not LOCAL:
        loc_keys = []
    x = np.arange(0, N + 1)
    band(ax, [curve(block, k) for k in rnd_keys], COL["v5"], "random", 1.5)
    if loc_keys:
        band(ax, [curve(block, k) for k in loc_keys], COL["local"], local_label, 2)
    for key, ck, lab, z in (("luna_low", "luna_low", "gpt-6-luna, low effort", 3), ("luna_high", "luna_high", "gpt-6-luna, high effort", 3),
                            ("luna_xhigh", "luna_xhigh", "gpt-6-luna, extra-high effort", 3), ("gpt6_sol_high", "sol6", "gpt-6-sol, high effort", 4),
                            ("gpt61_sol_high", "sol61", "gpt-6.1-sol, high effort", 4)):
        ax.plot(XS, curve(block, key)[XS], color=COL[ck], lw=0.9, ls=LS, marker=MK[key], ms=3.4, mew=0.7, mfc="white", zorder=z, label=lab)
    ax.set_xlim(0, N)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlabel(os.environ.get("XLABEL", "steps"), fontsize=8.5, labelpad=2)
    ax.set_ylabel(ylab, fontsize=8 if YL == "full" else 8.5, labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    if legend:
        ax.legend(loc="lower right", fontsize=5.6, frameon=False, handlelength=2.6, labelspacing=0.22, borderaxespad=0.2)


fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.41, 5.3), gridspec_kw=dict(hspace=0.42))
panel(ax, "distance", ["random_v5_rep%d" % i for i in range(1, 6)], ["local_v5_rep%d" % i for i in range(1, 6)],
      "hill-climbing", YLABELS[YL][0], False)
ax.set_ylim(0, 500)
panel(bx, "circuit", ["random_v5_rep%d" % i for i in range(1, 6)], [os.environ.get("CIRC_LOCAL", "local_v5") + "_rep%d" % i for i in range(1, 6)],
      "hill-climbing", YLABELS[YL][1], True)
bx.set_ylim(20, 60)
for a_, lab in ((ax, "a"), (bx, "b")):
    fig.text(0.0, a_.get_position().y1 + 0.008, lab, fontsize=10, fontweight="bold", va="bottom", ha="left")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "line_score_arms." + ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
print("written points variant", PAL, "step", STEP, "linestyle", LS, "ylabel", YL)
