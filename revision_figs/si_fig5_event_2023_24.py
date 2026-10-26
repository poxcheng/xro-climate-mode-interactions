# SI figure (3x2): the 2023-24 El Nino against the other eleven ONI events,
# tropical-mean LTT.
# (a)   spanning the top row, OBS tropical-mean LTT bars for the composite
#       with (n=12) and without (n=11) the 2023-24 event plus the event
#       itself, with the mature-phase Dec(0)-Jun(+1) sensitivity annotated.
# (b,c) monthly pathway contributions BASE-(C2=0) and BASE-(C1=0), composite
#       (n=11) vs the 2023-24 event.
# (d)   as (b,c) but for the residual OBS-BASE (warmth not captured by the
#       linear 10-mode replay).
# (e)   per-event scatter: coupling-pathway contribution, BASE-(C1&C2=0),
#       vs residual warming, both averaged over Dec(0)-Jun(+1); colored by
#       the event's peak Nino3.4.
# In (a)-(d) dashed curves show Nino3.4 (n=11 composite and 2023-24) and the
# gray band marks one month on either side of the Nino3.4 peak (Dec(0) for
# both).
import os, json
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
FIGS = os.environ.get("XRO_FIG_DIR", HERE)
os.environ.setdefault("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
os.environ.setdefault("XRO_CACHE_DIR", os.path.normpath(os.path.join(HERE, "..", "outputs_cache")))

import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

FS = {"base": 15.4, "panel_title": 16.1, "axis_label": 16.1, "tick": 14.7,
      "legend": 12.6, "annot": 12.6, "suptitle": 17.5}
plt.rcParams.update({"font.size": FS["base"], "axes.labelsize": FS["axis_label"],
                     "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"]})

nbj = json.load(open(os.path.join(PKG, "Full_code_reproduce_v1.ipynb")))
_code = [c for c in nbj["cells"] if c["cell_type"] == "code"]
_stop = next(i for i, c in enumerate(_code) if "CELL TT" in "".join(c["source"]))
for _c in _code[:_stop + 1]:
    _src = "".join(_c["source"])
    if _src.lstrip().startswith("!"):
        continue
    exec(_src, globals())

def ser1(da):
    return pd.Series(np.asarray(da.values, float),
                     index=pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp())

obs = ser1(hov_raw["OBS"].mean("lon"))
base = ser1(hov_raw["BASE"].mean("lon"))
c2c = ser1((hov_raw["BASE"] - hov_raw["C2"]).mean("lon"))   # C2 contribution
c1c = ser1((hov_raw["BASE"] - hov_raw["C1"]).mean("lon"))   # C1 contribution
res = obs - base                                            # residual
c12 = ser1(hov_raw["C12"].mean("lon"))
n34 = ser1(get_nino34_series(ds_idx))

def win24(s, y0):
    return s.reindex(pd.date_range(pd.Timestamp(y0, 1, 1), periods=24, freq="MS")).values

YEARS12 = [1982, 1987, 1991, 1994, 1997, 2002, 2004, 2006, 2009, 2015, 2019, 2023]
years11 = [y for y in YEARS12 if y != 2023]
def comp(s, years):
    return np.nanmean(np.stack([win24(s, y) for y in years]), axis=0)

mlab = [pd.Timestamp(2001, (i % 12) + 1, 1).strftime("%b") + ("(0)" if i < 12 else "(+1)")
        for i in range(24)]
xm = np.arange(24)
VLIM = 2.0
MAT = slice(11, 18)   # Dec(0) - Jun(+1)
CBLK, CRED, CBLU = "k", "#cb181d", "#2171b5"

n34c, n34e = comp(n34, years11), win24(n34, 2023)
knc, kne = int(np.nanargmax(n34c)), int(np.nanargmax(n34e))
assert knc == kne  # both Nino3.4 peaks fall on Dec(0); one shading serves both

fig = plt.figure(figsize=(13.0, 13.6))
gs = fig.add_gridspec(3, 2, left=0.065, right=0.985, top=0.86, bottom=0.055,
                      hspace=0.34, wspace=0.28, height_ratios=[1.08, 1.0, 1.0])
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1], sharey=axb)
axd = fig.add_subplot(gs[2, 0], sharey=axb)
axe = fig.add_subplot(gs[2, 1])

fig.suptitle("The 2023-24 El Nino against the other eleven events, tropical-mean LTT\n"
             "composite sensitivity, pathway timing, and residual warming",
             fontsize=FS["suptitle"], fontweight="bold", y=0.985)

def nino_overlay(ax, show_right):
    ax2 = ax.twinx()
    ax2.plot(xm, n34c, "--", color="0.45", lw=1.2)
    ax2.plot(xm, n34e, "--", color=CRED, lw=1.2, alpha=0.75)
    ax2.set_ylim(-3.2, 3.2)
    if show_right:
        ax2.set_ylabel("Nino3.4 (K, dashed)", color="0.45")
    else:
        ax2.set_yticklabels([])
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

# ---- (a) OBS bars, composite n=12 / n=11 / event, spanning the top row ----
c12m, c11m = comp(obs, YEARS12), comp(obs, years11)
ye_obs = win24(obs, 2023)
bw3 = 0.26
axa.axvspan(knc - 1, knc + 1, color="0.82", alpha=0.6, zorder=0)
axa.axhline(0, color="0.6", lw=0.8)
axa.bar(xm - bw3, c12m, width=bw3, color=CBLU, edgecolor="none")
axa.bar(xm, c11m, width=bw3, color=CBLK, edgecolor="none")
axa.bar(xm + bw3, ye_obs, width=bw3, color=CRED, edgecolor="none")
nino_overlay(axa, show_right=True)
a12, a11 = np.nanmean(c12m[MAT]), np.nanmean(c11m[MAT])
axa.annotate("mature phase Dec(0)-Jun(+1):\n"
             f"n=12 mean {a12:+.3f} K, n=11 mean {a11:+.3f} K ({(a12-a11)/a12*100:.0f}% lower)",
             xy=(0.02, 0.96), xycoords="axes fraction", va="top", fontsize=FS["annot"],
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
axa.set_ylim(-0.35, 0.72)
axa.set_xticks(xm[::3])
axa.set_xticklabels([mlab[i] for i in range(0, 24, 3)], fontsize=FS["tick"] - 1)
axa.set_ylabel("LTT anomaly (K)")
axa.grid(alpha=0.2, lw=0.5)
axa.set_title("(a) OBS tropical-mean LTT, composite with and without the 2023-24 event",
              loc="left", fontsize=FS["panel_title"])

# ---- (b,c,d) pathway timing bars, composite (n=11) vs 2023-24 ----
PANELS = [(axb, r"(b) C$_2$ contribution, BASE$-$(C$_2$=0)", c2c, False),
          (axc, r"(c) C$_1$ contribution, BASE$-$(C$_1$=0)", c1c, True),
          (axd, r"(d) residual, OBS$-$BASE", res, False)]
bw = 0.38
for ax, tag, s, show_right in PANELS:
    yc, ye = comp(s, years11), win24(s, 2023)
    ax.axvspan(knc - 1, knc + 1, color="0.82", alpha=0.6, zorder=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.bar(xm - bw / 2, yc, width=bw, color=CBLK, edgecolor="none")
    ax.bar(xm + bw / 2, ye, width=bw, color=CRED, edgecolor="none")
    nino_overlay(ax, show_right=show_right)
    ax.set_xticks(xm[::6])
    ax.set_xticklabels([mlab[i] for i in range(0, 24, 6)], fontsize=12.6)
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title(tag, loc="left", fontsize=FS["panel_title"])
axb.set_ylabel("LTT contribution (K)")
axd.set_ylabel("LTT contribution (K)")
plt.setp(axc.get_yticklabels(), visible=False)

# ---- (e) per-event coupling contribution vs residual warming ----
cx, cy, cc = [], [], []
for y0 in YEARS12:
    coupling = np.nanmean((win24(base, y0) - win24(c12, y0))[MAT])
    residual = np.nanmean((win24(obs, y0) - win24(base, y0))[MAT])
    cx.append(coupling); cy.append(residual)
    cc.append(np.nanmax(win24(n34, y0)))
    print(f"  {y0}: coupling {coupling:+.3f} K, residual {residual:+.3f} K, N34 peak {cc[-1]:+.2f}")
axe.axhline(0, color="0.6", lw=0.8)
axe.axvline(0, color="0.6", lw=0.8)
scd = axe.scatter(cx, cy, c=cc, s=120, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM,
                  edgecolors="k", linewidths=0.9, zorder=5)
LAB_OFF = {2004: (6, -12), 2006: (6, 4), 2009: (6, -12), 2015: (-26, 6), 2023: (-14, -16)}
for x0, y0v, yr in zip(cx, cy, YEARS12):
    axe.annotate(str(yr), (x0, y0v), xytext=LAB_OFF.get(yr, (6, 6)),
                 textcoords="offset points",
                 fontsize=FS["annot"], fontweight="bold" if yr == 2023 else "normal")
r_all = np.corrcoef(cx, cy)[0, 1]
axe.annotate(f"r = {r_all:.2f} (n=12)", xy=(0.03, 0.03), xycoords="axes fraction",
             fontsize=FS["annot"])
axe.set_xlabel(r"coupling contribution, BASE - ($C_1$&$C_2=0$) (K)")
axe.set_ylabel("residual warming, OBS - BASE (K)")
axe.grid(alpha=0.2, lw=0.5)
axe.set_title("(e) coupling vs residual warming",
              loc="left", fontsize=FS["panel_title"])
cb = fig.colorbar(scd, ax=axe, pad=0.02, fraction=0.05)
cb.set_label("event peak Nino3.4 (K)", fontsize=FS["axis_label"] - 1)
cb.ax.tick_params(labelsize=FS["tick"] - 1)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(color=CBLU), Patch(color=CBLK), Patch(color=CRED),
     Line2D([], [], ls="--", color="0.45"), Line2D([], [], ls="--", color=CRED, alpha=0.75),
     Patch(color="0.82", alpha=0.6)]
l = ["composite, n=12 (with 2023-24)", "composite, n=11 (without)", "2023-24 event",
     "Nino3.4, composite (n=11)", "Nino3.4, 2023-24",
     "Nino3.4 peak +/-1 mo (both Dec(0))"]
fig.legend(h, l, fontsize=FS["legend"], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.942))

fp = os.path.join(FIGS, "FigS5_event_2023-24.png")
fig.savefig(fp, dpi=300, bbox_inches="tight")
print("saved", fp)
