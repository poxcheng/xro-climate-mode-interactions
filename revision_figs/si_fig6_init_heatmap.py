# SI figure: forecast evolution across initialization months (heat map).
# Rows of each panel = initialization month (Jul 2025 - Aug 2026), columns =
# target month, color = forecast tropical-mean anomaly (lead 0-18 months).
# Panels (a,b): full XRO forecast of SST / LTT.
# Panels (c-h): per-basin contribution, XRO minus the forecast with that
# basin's C1 and C2 blocks both zeroed (ExPO, IO, AO).
# Engine, training (1982-2022) and beta mapping identical to Fig. 10, using
# the real-time pipeline extended to 2026-09.
import os, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["XRO_INIT"] = "2026-08"
os.environ["XRO_OISST_FILE"] = os.path.join(HERE, "data_r2", "realtime_oisst_sst_indices_2026-09.nc")
os.environ["XRO_WWV_FILE"] = os.path.join(HERE, "data_r2", "realtime_tao_pmel_wwv_2026-09.nc")

ENG = os.path.join(HERE, "engine_core.py")
FIGS = os.environ.get("XRO_FIG_DIR", HERE)

import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

FS = {"base": 11, "panel_title": 11.5, "axis_label": 11.5, "tick": 10.5,
      "legend": 8.2, "annot": 9.5, "suptitle": 12.5}
plt.rcParams.update({"font.size": FS["base"], "axes.labelsize": FS["axis_label"],
                     "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"]})

g = {"__file__": ENG}
exec(open(ENG).read(), g)

_model, _fit, _patch = g["_model"], g["_fit"], g["_patch"]
ds_fc, MODES9, COL, BASIN = g["ds_fc"], g["MODES9"], g["COL"], g["BASIN"]
ENSO_ROWS, NLEAD = g["ENSO_ROWS"], g["NLEAD"]
_to_trop = g["_to_trop"]

def decouple_basin(b):
    cols = [COL[m] for m in BASIN[b]]
    f = _fit.copy(deep=True); L = f["Lac"].values.copy()
    for r in ENSO_ROWS:
        for c in cols: L[r, c, :] = 0.0        # C1: basin modes -> ENSO
    for r in cols:
        for c in ENSO_ROWS: L[r, c, :] = 0.0   # C2: ENSO -> basin modes
    f["Lac"] = (f["Lac"].dims, L); return f

FITS = {"XRO": _fit, "ExPO": decouple_basin("ExPO"),
        "IO": decouple_basin("IO"), "AO": decouple_basin("AO")}

INITS = pd.date_range("2026-01-01", "2026-08-01", freq="MS")
init_strs = [t.strftime("%Y-%m") for t in INITS]
TARGETS = pd.date_range("2026-01-01", "2028-02-01", freq="MS")
tgt_idx = {t.strftime("%Y-%m"): k for k, t in enumerate(TARGETS)}

print("reforecasting", len(INITS), "inits x", len(FITS), "fits ...")
HM = {}   # HM[fitname][var] = (ninit, ntarget) grid
for name, fd in FITS.items():
    fc = _model.reforecast(fit_ds=fd, init_ds=ds_fc, n_month=NLEAD,
                           ncopy=1, noise_type="zero", seed=0)
    grids = {v: np.full((len(INITS), len(TARGETS)), np.nan) for v in ("SST", "LTT")}
    for i, ini in enumerate(init_strs):
        arrs = [fc[m].sel(init=ini).squeeze().values for m in MODES9]
        st = np.stack(arrs, axis=0)                      # (mode, lead+1)
        ft = np.array([np.datetime64(ini, "M") + np.timedelta64(L, "M") for L in range(NLEAD + 1)])
        fmon = pd.DatetimeIndex(ft.astype("datetime64[D]")).month.values
        for v in ("SST", "LTT"):
            tr = _to_trop(st, fmon, v)
            for L in range(NLEAD + 1):
                key = str(ft[L])
                if key in tgt_idx:
                    grids[v][i, tgt_idx[key]] = tr[L]
    HM[name] = grids
    print("  done", name)

# observed-mode reconstruction along the top for reference
ds_obs = ds_fc.sel(time=slice("2025-07", "2026-08"))
omon = pd.DatetimeIndex(ds_obs.time.values).month.values
ost = ds_obs[MODES9].to_array("mode").transpose("mode", "time").values
obs_recon = {v: _to_trop(ost, omon, v) for v in ("SST", "LTT")}

# ---- plot ----
VAR_NAME = {"SST": "Tropical-mean SST", "LTT": "Tropical-mean LTT"}
VLIM_ALL = 0.9
ROWS = [("XRO", "Full XRO forecast", VLIM_ALL),
        ("ExPO", "ExPO contribution (XRO - ExPO decoupled)", VLIM_ALL),
        ("IO", "IO contribution (XRO - IO decoupled)", VLIM_ALL),
        ("AO", "AO contribution (XRO - AO decoupled)", VLIM_ALL)]
PL = "abcdefgh"

fig, axes = plt.subplots(4, 2, figsize=(13.2, 13.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.06, wspace=0.10, hspace=0.30)
fig.suptitle("Forecast evolution with initialization month\n"
             "XRO trained 1982-2022, real-time pipeline to 2026-09",
             fontsize=FS["suptitle"], fontweight="bold")

xi = np.arange(len(TARGETS)); yi = np.arange(len(INITS))
xticks = xi[::3]
xtlab = [TARGETS[k].strftime("%b\n%Y") if TARGETS[k].month == 1 else TARGETS[k].strftime("%b")
         for k in xticks]
ytlab = [t.strftime("%b %Y") for t in INITS]

ims = {}
for r, (name, rtitle, vlim) in enumerate(ROWS):
    for c, v in enumerate(("SST", "LTT")):
        ax = axes[r, c]
        Z = HM[name][v] if name == "XRO" else HM["XRO"][v] - HM[name][v]
        im = ax.pcolormesh(xi, yi, Z, cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
        if r == 0: ims["top"] = im
        else: ims["sub"] = im
        ax.set_title(f"({PL[r*2+c]}) {VAR_NAME[v]}: {rtitle}", loc="left", fontsize=FS["panel_title"])
        ax.set_xticks(xticks); ax.set_xticklabels(xtlab)
        ax.set_yticks(yi); ax.set_yticklabels(ytlab)
        ax.invert_yaxis()
for ax in axes[-1]:
    ax.set_xlabel("Target month")
for r in range(4):
    axes[r, 0].set_ylabel("Initialization")

cax1 = fig.add_axes([0.905, 0.25, 0.013, 0.5])
cb1 = fig.colorbar(ims["top"], cax=cax1)
cb1.set_label("forecast anomaly / basin contribution (K)", fontsize=FS["axis_label"])
cb1.ax.tick_params(labelsize=FS["tick"])

fp = os.path.join(FIGS, "FigS6_init_heatmap.png")
fig.savefig(fp, dpi=300, bbox_inches="tight")
print("saved", fp)
for v in ("SST", "LTT"):
    row = HM["XRO"][v][-1]
    k = np.nanargmax(row)
    print(f"last init (2026-08) {v}: peak {row[k]:.3f} K @ {TARGETS[k].strftime('%Y-%m')}")
