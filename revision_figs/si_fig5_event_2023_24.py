# SI figure (2x2): the 2023-24 El Nino against the other eleven ONI events.
# (a,b) pathway phase space of tropical-mean LTT, x = (C1=0)-BASE,
#       y = (C2=0)-BASE, monthly values over Jan(0)-Dec(+1), colored by
#       Nino3.4; (a) composite n=11, (b) 2023-24.
# (c)   composite OBS tropical-mean LTT with (n=12) and without (n=11) the
#       2023-24 event, plus the event itself.
# (d)   per-event scatter: coupling-pathway contribution, BASE-(C1&C2=0),
#       vs residual warming, OBS-BASE (warmth not captured by the linear
#       10-mode replay), both averaged over the mature phase Dec(0)-Jun(+1);
#       colored by the event's peak Nino3.4.
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

FS = {"base": 11, "panel_title": 11.5, "axis_label": 11.5, "tick": 10.5,
      "legend": 9.0, "annot": 9.0, "suptitle": 12.5}
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
d1 = ser1((hov_raw["C1"] - hov_raw["BASE"]).mean("lon"))
d2 = ser1((hov_raw["C2"] - hov_raw["BASE"]).mean("lon"))
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
MARKED = (0, 6, 11, 17, 23)
VLIM, LIM = 2.0, 0.16
MAT = slice(11, 18)   # Dec(0) - Jun(+1)

fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.6))
fig.subplots_adjust(left=0.075, right=0.88, top=0.90, bottom=0.07, wspace=0.26, hspace=0.30)
fig.suptitle("The 2023-24 El Nino against the other eleven events, tropical-mean LTT\n"
             "pathway phase space, composite sensitivity, and residual warming",
             fontsize=FS["suptitle"], fontweight="bold")

# ---- (a,b) phase space ----
def phase_panel(ax, x, y, col, mk, ls, lc, ms, bold):
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.plot([-LIM, LIM], [-LIM, LIM], color="0.85", lw=0.8, ls=":", zorder=0)
    ax.plot(x, y, ls=ls, color=lc, lw=1.5, zorder=4)
    sc = ax.scatter(x, y, c=col, s=ms, marker=mk, linewidths=0.8,
                    cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, zorder=5, edgecolors="k")
    placed = []
    for k in MARKED:
        nb = [j for j in range(max(0, k - 3), min(24, k + 4)) if j != k]
        vx = x[k] - np.nanmean(x[nb]); vy = y[k] - np.nanmean(y[nb])
        nn = np.hypot(vx, vy)
        a0 = np.arctan2(vy, vx) if (np.isfinite(nn) and nn > 1e-9) else 0.6
        best = None
        for rr in (0.17 * LIM, 0.26 * LIM, 0.36 * LIM):
            for da in (0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, np.pi):
                lx, ly = x[k] + rr * np.cos(a0 + da), y[k] + rr * np.sin(a0 + da)
                if abs(lx) > 0.97 * LIM or abs(ly) > 0.97 * LIM:
                    continue
                dmin_lab = min([np.hypot(lx - px, ly - py) for px, py in placed], default=9e9)
                pts = np.hypot(np.asarray(x) - lx, np.asarray(y) - ly)
                pts[k] = 9e9
                dmin_pt = np.nanmin(pts)
                score = min(dmin_lab, 0.8 * dmin_pt) - 0.10 * rr
                if best is None or score > best[0]:
                    best = (score, lx, ly)
        _, lx, ly = best
        placed.append((lx, ly))
        ax.annotate(mlab[k], (x[k], y[k]), xytext=(lx, ly), textcoords="data",
                    ha="center", va="center", fontsize=FS["annot"], color="0.1",
                    fontweight="bold" if bold else "normal",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, shrinkA=0, shrinkB=2))
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_xlabel(r"($C_1=0$) - BASE (K)")
    ax.set_ylabel(r"($C_2=0$) - BASE (K)")
    return sc

sc = phase_panel(axes[0, 0], comp(d1, years11), comp(d2, years11), comp(n34, years11),
                 "s", "--", "0.55", 52, False)
axes[0, 0].set_title(f"(a) composite (n={len(years11)}): pathway phase space",
                     loc="left", fontsize=FS["panel_title"])
phase_panel(axes[0, 1], win24(d1, 2023), win24(d2, 2023), win24(n34, 2023),
            "o", "-", "0.15", 68, True)
axes[0, 1].set_title("(b) 2023-24 event: pathway phase space",
                     loc="left", fontsize=FS["panel_title"])

# ---- (c) composite n=12 vs n=11 vs the event ----
axc = axes[1, 0]
xm = np.arange(24)
c12m, c11m = comp(obs, YEARS12), comp(obs, years11)
axc.axhline(0, color="0.6", lw=0.8)
axc.axvspan(11, 17, color="0.92", zorder=0)
axc.plot(xm, win24(obs, 2023), "-o", color="#cb181d", lw=1.8, ms=3.6, label="2023-24 event")
axc.plot(xm, c12m, "-o", color="k", lw=1.8, ms=3.6, label="composite, n=12 (with 2023-24)")
axc.plot(xm, c11m, "--s", color="#2171b5", lw=1.8, ms=3.6, label="composite, n=11 (without)")
a12, a11 = np.nanmean(c12m[MAT]), np.nanmean(c11m[MAT])
axc.annotate("mature phase Dec(0)-Jun(+1):\n"
             f"n=12 mean {a12:+.3f} K, n=11 mean {a11:+.3f} K ({(a12-a11)/a12*100:.0f}% lower)",
             xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=FS["annot"],
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
axc.set_ylim(top=axc.get_ylim()[1] + 0.16)
axc.set_xticks(xm[::3]); axc.set_xticklabels([mlab[i] for i in range(0, 24, 3)], fontsize=FS["tick"] - 1)
axc.set_ylabel("tropical-mean LTT anomaly (K)")
axc.legend(fontsize=FS["legend"], loc="lower center")
axc.grid(alpha=0.2, lw=0.5)
axc.set_title("(c) composite with and without the 2023-24 event",
              loc="left", fontsize=FS["panel_title"])

# ---- (d) per-event coupling contribution vs residual warming ----
axd = axes[1, 1]
cx, cy, cc = [], [], []
for y0 in YEARS12:
    coupling = np.nanmean((win24(base, y0) - win24(c12, y0))[MAT])
    residual = np.nanmean((win24(obs, y0) - win24(base, y0))[MAT])
    cx.append(coupling); cy.append(residual)
    cc.append(np.nanmax(win24(n34, y0)))
    print(f"  {y0}: coupling {coupling:+.3f} K, residual {residual:+.3f} K, N34 peak {cc[-1]:+.2f}")
axd.axhline(0, color="0.6", lw=0.8)
axd.axvline(0, color="0.6", lw=0.8)
scd = axd.scatter(cx, cy, c=cc, s=120, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM,
                  edgecolors="k", linewidths=0.9, zorder=5)
for x0, y0v, yr in zip(cx, cy, YEARS12):
    axd.annotate(str(yr), (x0, y0v), xytext=(6, 6), textcoords="offset points",
                 fontsize=FS["annot"], fontweight="bold" if yr == 2023 else "normal")
r_all = np.corrcoef(cx, cy)[0, 1]
axd.annotate(f"r = {r_all:.2f} (n=12)", xy=(0.03, 0.03), xycoords="axes fraction",
             fontsize=FS["annot"])
axd.set_xlabel(r"coupling contribution, BASE - ($C_1$&$C_2=0$) (K)")
axd.set_ylabel("residual warming, OBS - BASE (K)")
axd.grid(alpha=0.2, lw=0.5)
axd.set_title("(d) coupling pathways vs residual warming, Dec(0)-Jun(+1)",
              loc="left", fontsize=FS["panel_title"])

cax = fig.add_axes([0.905, 0.25, 0.016, 0.5])
cb = fig.colorbar(sc, cax=cax)
cb.set_label(u"Nino3.4 SST anomaly (K); in (d) the event peak", fontsize=FS["axis_label"])
cb.ax.tick_params(labelsize=FS["tick"])

fp = os.path.join(FIGS, "FigS5_event_2023-24.png")
fig.savefig(fp, dpi=300, bbox_inches="tight")
print("saved", fp)
