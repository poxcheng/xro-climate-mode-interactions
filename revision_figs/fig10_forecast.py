import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.rcParams["font.size"] = 12
HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "forecast_engine.py")
g = {"__file__": ENG}
exec(open(ENG).read(), g)

# ---- rebuild: simplified main Fig10 (CC column) + SI variants fig (BD column) ----
import numpy as np, pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ftime_d = g["ftime"].astype("datetime64[D]")
otime, obs_recon = g["otime"], g["obs_recon"]
band, nino_band = g["band"], g["nino_band"]
nino_fc, nino_obs = g["nino_fc"], g["nino_obs"]
mean_cc, mean_bd = g["mean_cc"], g["mean_bd"]
CURVE_CC, CURVE_BD = g["CURVE_CC"], g["CURVE_BD"]
VAR_NAME = {"SST": "Tropical-mean SST", "LTT": "Tropical-mean LTT"}
t0 = ftime_d[0]

def panel(ax, v, CURVES, MEANS, show_nino_label):
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(t0, color="0.4", lw=1.0, ls="--")
    lo90, hi90 = band[v]["1090"]; lo75, hi75 = band[v]["2575"]
    ax.fill_between(ftime_d, lo90, hi90, color="#cb181d", alpha=0.10, lw=0, zorder=1, label="XRO 10-90% (red noise)")
    ax.fill_between(ftime_d, lo75, hi75, color="#cb181d", alpha=0.20, lw=0, zorder=1, label="XRO 25-75%")
    ax.plot(otime, obs_recon[v], "-o", color="k", ms=3.5, lw=1.4, label="OBS (recon)", zorder=6)
    for lab, fd, col, mk, ls in CURVES:
        ax.plot(ftime_d, MEANS[lab][v], marker=mk, ls=ls, color=col, ms=3.8, lw=1.6, label=lab, zorder=5)
    ax2 = ax.twinx()
    nlo90, nhi90 = nino_band["1090"]
    ax2.fill_between(ftime_d, nlo90, nhi90, color="0.45", alpha=0.16, lw=0, zorder=0, label=u"Nino3.4 10-90%")
    ax2.plot(otime, nino_obs, color="0.40", ls="-", lw=1.2, label=u"Nino3.4 (obs)")
    ax2.plot(ftime_d, nino_fc, color="0.55", ls="--", lw=1.2, label=u"Nino3.4 (forecast)")
    ax2.set_ylim(-1.5, 4.0); ax2.tick_params(axis="y", colors="0.4", labelsize=8.4)
    if show_nino_label: ax2.set_ylabel(u"Nino3.4 SST anomaly", color="0.4", fontsize=10.8)
    ax.set_xlim(otime[0], ftime_d[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_ylabel("Temperature anomaly (K)")
    for yr in (2026, 2027):
        jan = np.datetime64("%d-01-01" % yr)
        if otime[0] <= jan <= ftime_d[-1]:
            ax.annotate(str(yr), xy=(jan, 0), xytext=(0, -26), textcoords="offset points",
                        xycoords=("data", "axes fraction"), ha="center", fontsize=9.6, annotation_clip=False)
    return ax2

# main simplified
fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), sharex=True)
for i, v in enumerate(("SST", "LTT")):
    ax2 = panel(axes[i], v, CURVE_CC, mean_cc, True)
    axes[i].set_title(f"({'ab'[i]}) {VAR_NAME[v]}", loc="left", fontsize=12.0)
    if i == 0:
        h1,l1 = axes[i].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, ncol=2, loc="upper left", frameon=True, fontsize=8.2)
fig.suptitle("Forecast of tropical-mean SST and LTT warming with mode contributions\nXRO initialised 2026-04", fontsize=13.2, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("Fig10_simplified.png", dpi=300, bbox_inches="tight")

# SI variants
fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.6), sharex=True)
for i, v in enumerate(("SST", "LTT")):
    ax2 = panel(axes[i], v, CURVE_BD, mean_bd, True)
    axes[i].set_title(f"({'ab'[i]}) {VAR_NAME[v]}, basin x feedback-direction variants", loc="left", fontsize=12.0)
    if i == 0:
        h1,l1 = axes[i].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, ncol=2, loc="upper left", frameon=True, fontsize=7.7)
fig.suptitle("Pathway-variant forecasts by basin and feedback direction\nXRO initialised 2026-04", fontsize=13.2, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("FigS_forecast_variants.png", dpi=300, bbox_inches="tight")
print("done both figures")

# ---- Fig10 v2: a,b + attribution bar panel c ----
import matplotlib.gridspec as gridspec
labels_cc = [lab for lab,*_ in CURVE_CC[1:]]
labels_bd = [lab for lab,*_ in CURVE_BD[1:]]
contrib = {}
for v in ("SST","LTT"):
    xro = mean_cc["XRO"][v]; kpk = int(np.nanargmax(xro))
    d = {}
    for lab in labels_cc: d[lab] = xro[kpk] - mean_cc[lab][v][kpk]
    for lab in labels_bd: d[lab] = xro[kpk] - mean_bd[lab][v][kpk]
    contrib[v] = d
order = labels_cc + labels_bd
fig = plt.figure(figsize=(7.6, 10.6))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.62], hspace=0.34)
axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)]
for i, v in enumerate(("SST", "LTT")):
    ax2 = panel(axs[i], v, CURVE_CC, mean_cc, True)
    axs[i].set_title(f"({'ab'[i]}) {VAR_NAME[v]}", loc="left", fontsize=12.0)
    if i == 0:
        h1,l1 = axs[i].get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, ncol=2, loc="upper left", frameon=True, fontsize=7.9)
axc = fig.add_subplot(gs[2])
x = np.arange(len(order)); w = 0.38
axc.bar(x - w/2, [contrib["SST"][k] for k in order], w, color="#444444", label="SST")
axc.bar(x + w/2, [contrib["LTT"][k] for k in order], w, color="#b8b8b8", edgecolor="0.4", lw=0.5, label="LTT")
axc.set_xticks(x); axc.set_xticklabels(order, fontsize=9.6, rotation=20)
axc.axvline(2.5, color="0.6", lw=0.8, ls=":")
axc.axvline(5.5, color="0.6", lw=0.8, ls=":")
axc.set_ylabel("peak warming loss (K)")
axc.set_title("(c) Attribution of the peak warming (XRO minus variant)", loc="left", fontsize=12.0)
axc.legend(fontsize=9.6, loc="upper right")
axc.grid(alpha=0.2, lw=0.5, axis="y")
fig.suptitle("Forecast of tropical-mean SST and LTT warming with mode contributions\nXRO initialised 2026-04", fontsize=13.2, fontweight="bold")
fig.savefig("Fig10_v2_3panel.png", dpi=300, bbox_inches="tight")
print("saved Fig10_v2_3panel.png")
