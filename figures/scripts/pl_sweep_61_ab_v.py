"""Figure 2: logical error rate per round and per logical qubit of the FOM and CLER elite codes.
Data: figures/data/pl_sweep_20261009_61_fig_bbibm.json (rows per code, basis and physical error rate; the worse basis
is drawn, with 95% Clopper-Pearson intervals), read through pl_sweep_61_clean.py. (a) against the physical error
rate p, the six rate cells below k/n = 0.4: CLER elite codes as solid lines with filled markers, FOM elite codes as
dashed lines with open markers, the [[144,12,12]] reference as grey diamonds; (b) the same codes at p = 0.002 against
their encoding rate. Writes pl_sweep_61_ab_v.{pdf,png}. Fira Sans is used when the font is installed."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PL_CELLS", "0,1,2,3,4,5")
os.environ.setdefault("PL_SKIP", "")
src = open(os.path.join(HERE, "pl_sweep_61_clean.py")).read().split("# ---------------- layout A")[0]
src = src.replace("pl_sweep_20261007_61_fig.json", "pl_sweep_20261009_61_fig_bbibm.json")
C = {"__file__": os.path.join(HERE, "pl_sweep_61_clean.py")}
exec(compile(src, "pl_sweep_61_clean.py", "exec"), C)
rows, CELLS, MARK, RATE, pts = C["rows"], C["CELLS"], C["MARK"], C["RATE"], C["pts"]

_FONTS = os.path.join(HERE, "fonts")
if os.path.isdir(_FONTS):
    for f in os.listdir(_FONTS):
        if f.endswith(".ttf"):
            fm.fontManager.addfont(os.path.join(_FONTS, f))
if any(f.name == "Fira Sans" for f in fm.fontManager.ttflist):
    plt.rcParams.update({"font.family": "Fira Sans", "mathtext.fontset": "custom", "mathtext.rm": "Fira Sans",
                         "mathtext.it": "Fira Sans:italic", "mathtext.bf": "Fira Sans:semibold"})
else:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})
PAL = {0: "#283f8f", 1: "#2a86c4", 2: "#1fa59a", 3: "#79b83a", 4: "#f0a330", 5: "#d8483b"}
REF, INK, GRID = "#9a9a9a", "#2a2a2a", "#e9e9e9"
P_OBS = 0.002


def tint(col, f):
    r, g, b = mc.to_rgb(col)
    return (1 - (1 - r) * f, 1 - (1 - g) * f, 1 - (1 - b) * f)


def series(ax, data, col, mk, cler, z):
    if not data:
        return
    x = [d[0] for d in data]; y = [d[1] for d in data]
    lo = [d[1] - d[2] for d in data]; hi = [d[3] - d[1] for d in data]
    if cler:
        ax.plot(x, y, "-", color=col, lw=1.25, zorder=z, solid_capstyle="round")
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=0.8, capsize=1.6, zorder=z + 0.1)
        ax.scatter(x, y, s=24, marker=mk, facecolors=col, edgecolors="white", linewidths=0.6, zorder=z + 0.2)
    else:
        c2 = tint(col, 0.6)
        ax.plot(x, y, ls=(0, (3, 1.6)), color=c2, lw=0.85, zorder=z - 1)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=c2, elinewidth=0.6, capsize=1.2, zorder=z - 0.9)
        ax.scatter(x, y, s=15, marker=mk, facecolors="white", edgecolors=c2, linewidths=0.9, zorder=z - 0.8)


def frame(ax):
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e-2)
    ax.set_yticks([1e-9, 1e-7, 1e-5, 1e-3])
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.6, pad=2)


fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.41, 5.0), gridspec_kw=dict(hspace=0.36))

# (a) p sweep: reference, FOM elites (light, dashed), CLER elites (solid) on top
ref = pts("bb144", None)
ax.errorbar([d[0] for d in ref], [d[1] for d in ref], yerr=[[d[1] - d[2] for d in ref], [d[3] - d[1] for d in ref]],
            fmt="none", ecolor=REF, elinewidth=0.6, capsize=1.2, zorder=1.4)
ax.scatter([d[0] for d in ref], [d[1] for d in ref], s=15, marker="D", facecolors="white", edgecolors=REF, linewidths=0.9,
           zorder=1.5)
for c in CELLS:
    series(ax, pts("dist", c), PAL[c], MARK[c], False, 2)
    if c == 0:
        series(ax, pts("dist2", 0), PAL[c], "X", False, 2)
for c in CELLS:
    series(ax, pts("plc", c), PAL[c], MARK[c], True, 4)
ax.set_xscale("log")
ax.set_xlim(0.88e-3, 5.6e-3)
ax.set_xticks([0.001, 0.002, 0.003, 0.004, 0.005])
ax.set_xticklabels(["0.001", "0.002", "0.003", "0.004", "0.005"])
frame(ax)
ax.set_xlabel("physical error rate $p$", labelpad=2)
ax.set_ylabel("$p_{\\mathrm{L}}^{\\mathrm{CLER}}$", labelpad=2)


# (b) p = 0.002 against k/n, FOM and CLER elite of each cell joined
def at_p(arm, c):
    for d in pts(arm, c):
        if abs(d[0] - P_OBS) < 1e-12:
            r = next(r for r in rows if r["arm"] == arm and (arm == "bb144" or r["cell"] == c))
            return (r["k"] / r["n"],) + tuple(d[1:])
    return None


for c in CELLS:
    f, g = at_p("dist", c), at_p("plc", c)
    for v, cler, mk in ((f, False, MARK[c]), (g, True, MARK[c])) + (((at_p("dist2", 0), False, "X"),) if c == 0 else ()):
        if not v:
            continue
        col = PAL[c] if cler else tint(PAL[c], 0.6)
        bx.errorbar([v[0]], [v[1]], yerr=[[v[1] - v[2]], [v[3] - v[1]]], fmt="none", ecolor=col,
                    elinewidth=0.8 if cler else 0.6, capsize=1.6 if cler else 1.2, zorder=2)
        bx.scatter([v[0]], [v[1]], s=26 if cler else 17, marker=mk, facecolors=col if cler else "white",
                   edgecolors="white" if cler else col, linewidths=0.6 if cler else 0.9, zorder=3)
r = at_p("bb144", None)
if r:
    bx.errorbar([r[0]], [r[1]], yerr=[[r[1] - r[2]], [r[3] - r[1]]], fmt="none", ecolor=REF, elinewidth=0.6, capsize=1.2)
    bx.scatter([r[0]], [r[1]], s=14, marker="D", facecolors="white", edgecolors=REF, linewidths=0.8, zorder=3)
bx.set_xlim(0.03, 0.4)
frame(bx)
bx.set_xlabel("encoding rate $k/n$", labelpad=2)
bx.set_ylabel("$p_{\\mathrm{L}}^{\\mathrm{CLER}}(p=0.002)$", labelpad=2)
NKD = {("dist", 0): (510, 40, "\\leq42"), ("dist2", 0): (578, 32, "\\leq46"), ("dist", 1): (592, 83, "18"),
       ("dist", 2): (585, 121, "24"), ("dist", 3): (560, 130, "14"), ("dist", 4): (588, 176, "12"),
       ("dist", 5): (544, 200, "8"), ("plc", 0): (572, 67, "22"), ("plc", 1): (576, 82, "15"),
       ("plc", 2): (580, 116, "12"), ("plc", 3): (580, 132, "14"), ("plc", 4): (588, 176, "10"),
       ("plc", 5): (598, 195, "7")}
for (arm, c), (n, k, _) in NKD.items():
    rr = [r for r in rows if r["arm"] == arm and r["cell"] == c]
    assert rr and all((r["n"], r["k"]) == (n, k) for r in rr), (arm, c)


def entry(arm, c):
    n, k, d = NKD[(arm, c)]
    if arm == "plc":
        return Line2D([], [], color=PAL[c], marker=MARK[c], mfc=PAL[c], mec="white", mew=0.5, ls="-", lw=1.25, ms=4,
                      label="$[[%d,%d,%s]]$" % (n, k, d))
    col = tint(PAL[c], 0.6)
    return Line2D([], [], color=col, marker="X" if arm == "dist2" else MARK[c], mfc="white", mec=col, mew=0.9,
                  ls=(0, (3, 1.6)), lw=0.85, ms=3.6, label="$[[%d,%d,%s]]$" % (n, k, d))


h_codes = ([entry("dist", 0), entry("dist2", 0)] + [entry("dist", c) for c in CELLS if c > 0]
           + [entry("plc", c) for c in CELLS]
           + [Line2D([], [], color=REF, marker="D", mfc="white", mec=REF, mew=0.9, ls="none", ms=3.4, label="$[[144,12,12]]$")])
bx.legend(handles=h_codes, loc="upper left", frameon=False, fontsize=5.6, ncol=2, columnspacing=0.8, handletextpad=0.3,
          handlelength=1.9, labelspacing=0.12, borderaxespad=0.1)

for a_, lab in ((ax, "a"), (bx, "b")):
    fig.text(0.0, a_.get_position().y1 + 0.006, lab, fontsize=10, fontweight="bold", va="bottom", ha="left")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "pl_sweep_61_ab_v." + ext), dpi=400, bbox_inches="tight", pad_inches=0.02)
print("written pl_sweep_61_ab_v.{pdf,png}")
