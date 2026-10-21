import os
DATA_DIR = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
# ============================================================
# CELL FC -- Real-time forecast of tropical-mean SST & LTT warming
#            with mode contributions (init 2026-04)
#
# Self-contained: does NOT depend on CELL A / CELL B globals.
#   - Initial conditions: real-time OISST (SST indices) + PMEL/TAO (WWV),
#     anomaly + linear-detrended, assembled 1982-01 -> 2026-04.
#   - Engine: official operational config XRO(ncycle=12, ac_order=2) with
#     maskb=['IOD'], maskNT=['T2','TH'] (nonlinear recharge oscillator),
#     trained 1982-2022, reforecast 18 months from 2026-04. Same engine as
#     XRO_official_reproduce_2026.ipynb, so the Nino3.4 forecast matches the
#     operational plume. Contribution blocks are zeroed in the linear Lac
#     operator; the nonlinear T^2 / T*H ENSO terms remain in all variants.
#   - Tropical-mean (30S-30N) mapping: monthly beta regression of the 9-mode
#     state (Nino34 + 8 non-ENSO modes, z-scored) onto ERA5 tropical-mean
#     SST / LTT, fitted in-sample (recon-vs-OBS r ~ 0.86 SST, 0.84 LTT).
#   - Contributions diagnosed two ways:
#       (left)  per-basin counterfactual forecasts -- zero each basin's
#               coupling into the Nino3.4 tendency row only (operational
#               XRO "quantify" convention).
#       (right) feedback-direction blocks over all 8 extra-basin modes:
#               C1=0 (modes->ENSO), C2=0 (ENSO->modes), C1&C2=0.
#   - 1000-member red-noise spread (10-90%) on the full XRO forecast.
#
# Required files in DATA_DIR:
#   realtime_oisst_sst_indices.nc, realtime_tao_pmel_wwv.nc,
#   ERA5_tropical_mean_sst_anom_detr_1950-2025.nc,
#   ERA5_LTT_1000-850_detr_tropmean_1979-2025.nc
# ============================================================
import numpy as np, xarray as xr, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from XRO import XRO

INIT, NLEAD, NMEM = "2026-04", 18, 1000
TRAIN = slice("1982-01", "2022-12")
STATE10 = ["Nino34","WWV","NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]
MODES9  = ["Nino34","NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]   # no WWV
COL   = {v: i for i, v in enumerate(STATE10)}
BASIN = {"ExPO": ["NPMM","SPMM"], "IO": ["IOB","IOD","SIOD"], "AO": ["TNA","ATL3","SASD"]}
ALL8  = BASIN["ExPO"] + BASIN["IO"] + BASIN["AO"]
ENSO_ROWS = [COL["Nino34"], COL["WWV"]]
SUB = [COL[m] for m in ALL8]

def _mz(da):
    da = da.assign_coords(time=da.time.values.astype("datetime64[M]").astype("datetime64[ns]"))
    if "month" in da.coords:
        da = da.drop_vars("month")
    return da

# ---- real-time initial conditions (OISST + PMEL/TAO) ----
_o = _mz(xr.open_dataset(DATA_DIR + "/realtime_oisst_sst_indices.nc"))
_w = _mz(xr.open_dataset(DATA_DIR + "/realtime_tao_pmel_wwv.nc")["WWV"])
ds_fc = xr.Dataset({"Nino34": _o["Nino34"], "WWV": _w})
for v in STATE10[2:]:
    ds_fc[v] = _o[v]
ds_fc = ds_fc[STATE10].sel(time=slice("1982-01", INIT)).dropna("time")
print("real-time IC:", str(ds_fc.time.values[0])[:7], "->", str(ds_fc.time.values[-1])[:7],
      "n=", ds_fc.sizes["time"])

# ---- tropical-mean OBS targets ----
_ys = xr.open_dataset(DATA_DIR + "/ERA5_tropical_mean_sst_anom_detr_1950-2025.nc")
y_sst = _mz(_ys[list(_ys.data_vars)[0]].rename("SST"))
_yl = xr.open_dataset(DATA_DIR + "/ERA5_LTT_1000-850_detr_tropmean_1979-2025.nc")
y_ltt = _mz(_yl[list(_yl.data_vars)[0]].rename("LTT"))

# ---- monthly beta mapping (z-scored 9-mode state -> tropical mean) ----
_Z  = ds_fc[MODES9].to_array("mode").transpose("time", "mode")
_MU  = _Z.groupby("time.month").mean("time")
_SIG = _Z.groupby("time.month").std("time")

def _zscore(vals, months):
    mu  = _MU.sel(month=xr.DataArray(months, dims="t")).values.T
    sig = _SIG.sel(month=xr.DataArray(months, dims="t")).values.T
    if vals.ndim == 3:
        out = (vals - mu[:, :, None]) / sig[:, :, None]
    else:
        out = (vals - mu) / sig
    return np.where(np.isfinite(out), out, 0.0)

def _fit_beta(y):
    Zz = ((_Z - _MU.sel(month=_Z["time.month"]).transpose("time","mode")) /
          _SIG.sel(month=_Z["time.month"]).transpose("time","mode")).where(np.isfinite(_Z), 0.0)
    Zz, yy = xr.align(Zz, y, join="inner")
    ti = pd.DatetimeIndex(Zz.time.values)
    beta = np.full((len(MODES9), 12), np.nan)
    for mm in range(1, 13):
        sel = ti.month == mm
        Zm, ym = Zz.values[sel], yy.values[sel].astype(float)
        ok = np.isfinite(ym) & np.all(np.isfinite(Zm), axis=1)
        if ok.sum() >= 12:
            beta[:, mm-1], *_ = np.linalg.lstsq(Zm[ok], ym[ok], rcond=None)
    return beta

BETA = {"SST": _fit_beta(y_sst), "LTT": _fit_beta(y_ltt)}

def _to_trop(state, months, var):
    z = _zscore(state, months)
    b = BETA[var][:, months-1]
    return np.einsum("ml,mlk->lk", b, z) if z.ndim == 3 else np.einsum("ml,ml->l", b, z)

# ---- XRO fit + counterfactual fits (official operational config) ----
_model = XRO(ncycle=12, ac_order=2)
_fit = _model.fit_matrix(ds_fc.sel(time=TRAIN), var_names=STATE10,
                         maskb=["IOD"], maskNT=["T2", "TH"])

def _patch(rows, cols):
    f = _fit.copy(deep=True); L = f["Lac"].values.copy()
    for r in rows:
        for c in cols:
            L[r, c, :] = 0.0
    f["Lac"] = (f["Lac"].dims, L); return f

# feedback-direction blocks over all 8 extra-basin modes
_fitC1  = _patch(ENSO_ROWS, SUB)                 # C1: modes -> ENSO
_fitC2  = _patch(SUB, ENSO_ROWS)                 # C2: ENSO -> modes
_fitC12 = _fit.copy(deep=True); _L = _fitC12["Lac"].values.copy()
for r in ENSO_ROWS:
    for c in SUB: _L[r, c, :] = 0.0
for r in SUB:
    for c in ENSO_ROWS: _L[r, c, :] = 0.0
_fitC12["Lac"] = (_fitC12["Lac"].dims, _L)

# basin-split feedback-direction blocks (only this basin's modes coupled)
def _c1_basin(b): return _patch(ENSO_ROWS, [COL[m] for m in BASIN[b]])   # basin modes -> ENSO
def _c2_basin(b): return _patch([COL[m] for m in BASIN[b]], ENSO_ROWS)   # ENSO -> basin modes

# ---- forecast leads + calendar months ----
ftime  = np.array([np.datetime64(INIT, "M") + np.timedelta64(L, "M") for L in range(NLEAD+1)])
fmonth = pd.DatetimeIndex(ftime.astype("datetime64[D]")).month.values

def _fc_state(fit_ds, ncopy=1, noise="zero", seed=0):
    fc = _model.reforecast(fit_ds=fit_ds, init_ds=ds_fc, n_month=NLEAD,
                           ncopy=ncopy, noise_type=noise, seed=seed)
    arrs = []
    for m in MODES9:
        a = fc[m].sel(init=INIT).squeeze()
        order = ["member","lead"] if "member" in a.dims else ["lead"]
        arrs.append(a.transpose(*order).values)
    st = np.stack(arrs, axis=0)
    if st.ndim == 3:
        st = np.transpose(st, (0, 2, 1))   # (mode,lead,member)
    return st

def _curves(fit_ds):
    st = _fc_state(fit_ds)
    return {v: _to_trop(st, fmonth, v) for v in ("SST","LTT")}

# (label, fit, color, marker, linestyle)
BCOL = {"ExPO": "#2b8cbe", "IO": "#1fb6d6", "AO": "#f5a300"}
CURVE_CC = [("XRO", _fit, "#cb181d", "o", "-"),
            (r"$C_1=0$",       _fitC1,  "#e15759", "s", "-"),
            (r"$C_2=0$",       _fitC2,  "#76b7b2", "s", "-"),
            (r"$C_1$&$C_2=0$", _fitC12, "#f28e2b", "s", "-")]
CURVE_BD = [("XRO", _fit, "#cb181d", "o", "-")]
for b in ("ExPO", "IO", "AO"):
    CURVE_BD.append((rf"$C1_{{{b}}}=0$", _c1_basin(b), BCOL[b], "o", "-"))
for b in ("ExPO", "IO", "AO"):
    CURVE_BD.append((rf"$C2_{{{b}}}=0$", _c2_basin(b), BCOL[b], "^", "--"))
mean_cc = {lab: _curves(fd) for lab, fd, *_ in CURVE_CC}
mean_bd = {lab: _curves(fd) for lab, fd, *_ in CURVE_BD}

# 1000-member red-noise spread on the full XRO forecast (10-90 and 25-75)
_st_ens = _fc_state(_fit, ncopy=NMEM, noise="red", seed=42)
band = {}
for v in ("SST","LTT"):
    ens = _to_trop(_st_ens, fmonth, v)
    band[v] = {"1090": (np.percentile(ens, 10, axis=1), np.percentile(ens, 90, axis=1)),
               "2575": (np.percentile(ens, 25, axis=1), np.percentile(ens, 75, axis=1))}
# Nino3.4 ensemble spread (row 0 of the mode state, physical units)
_nino_ens = _st_ens[0]                                  # (lead, member)
nino_band = {"1090": (np.percentile(_nino_ens, 10, axis=1), np.percentile(_nino_ens, 90, axis=1)),
             "2575": (np.percentile(_nino_ens, 25, axis=1), np.percentile(_nino_ens, 75, axis=1))}

# observed tail (reconstructed from real-time modes via the same beta)
_dst = ds_fc.sel(time=slice("2025-01", INIT))
otime = _dst.time.values.astype("datetime64[D]")
_tmon = pd.DatetimeIndex(_dst.time.values).month.values
_stt  = _dst[MODES9].to_array("mode").transpose("mode","time").values
obs_recon = {v: _to_trop(_stt, _tmon, v) for v in ("SST","LTT")}

# Nino3.4: ENSO driver shown as grey dashed reference (full XRO forecast + obs tail)
nino_fc  = _fc_state(_fit)[0]                       # row 0 of MODES9 = Nino34, (lead+1,)
nino_obs = _dst["Nino34"].values

for v in ("SST","LTT"):
    xro = mean_cc["XRO"][v]; kpk = int(np.nanargmax(xro))
    print(f"-- {v}: XRO peak {xro[kpk]:.3f} K @ lead {kpk}; contribution (XRO-variant) --")
    for lab, *_ in CURVE_CC[1:]:
        print(f"   {lab:14s} {xro[kpk]-mean_cc[lab][v][kpk]:+.4f}")
    for lab, *_ in CURVE_BD[1:]:
        print(f"   {lab:14s} {xro[kpk]-mean_bd[lab][v][kpk]:+.4f}")

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), sharex=True, sharey=True)
ftime_d = ftime.astype("datetime64[D]")
VAR_NAME = {"SST": "Tropical-mean SST", "LTT": "Tropical LTT"}
EXP_NAME = {0: r"feedback-direction ($C_1$/$C_2$) experiment",
            1: r"basin $\times$ feedback-direction experiment"}
PANEL_LBL = [["a", "b"], ["c", "d"]]
ROWS   = [("SST", None), ("LTT", None)]
PANELS = [(CURVE_CC, mean_cc), (CURVE_BD, mean_bd)]
t0 = ftime_d[0]
NINO_YL = (-1.5, 4.0)
for i, (v, _) in enumerate(ROWS):
    for j, (CURVES, MEANS) in enumerate(PANELS):
        ax = axes[i, j]
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(t0, color="0.4", lw=1.0, ls="--")
        lo90, hi90 = band[v]["1090"]; lo75, hi75 = band[v]["2575"]
        ax.fill_between(ftime_d, lo90, hi90, color="#cb181d",
                        alpha=0.10, lw=0, zorder=1, label="XRO 10-90% (red noise)")
        ax.fill_between(ftime_d, lo75, hi75, color="#cb181d",
                        alpha=0.20, lw=0, zorder=1, label="XRO 25-75%")
        ax.plot(otime, obs_recon[v], "-o", color="k", ms=3.5, lw=1.4,
                label="OBS (recon)", zorder=6)
        for lab, fd, col, mk, ls in CURVES:
            ax.plot(ftime_d, MEANS[lab][v], marker=mk, ls=ls,
                    color=col, ms=3.8, lw=1.6, label=lab, zorder=5)
        # Nino3.4 ENSO driver on secondary right axis (fixed scale)
        #   observed (OISST real-time) = solid; XRO forecast = dashed + grey spread
        ax2 = ax.twinx()
        nlo90, nhi90 = nino_band["1090"]; nlo75, nhi75 = nino_band["2575"]
        ax2.fill_between(ftime_d, nlo90, nhi90, color="0.45",
                         alpha=0.16, lw=0, zorder=0, label=u"Niño3.4 10-90%")
        ax2.fill_between(ftime_d, nlo75, nhi75, color="0.45",
                         alpha=0.28, lw=0, zorder=0, label=u"Niño3.4 25-75%")
        ax2.plot(otime, nino_obs, color="0.40", ls="-",  lw=1.3, label=u"Niño3.4 (obs)")
        ax2.plot(ftime_d, nino_fc, color="0.55", ls="--", lw=1.3, label=u"Niño3.4 (forecast)")
        ax2.set_ylim(NINO_YL)
        ax2.tick_params(axis="y", colors="0.4", labelsize=7)
        if j == 1:
            ax2.set_ylabel(u"Niño3.4 SST anomaly", color="0.4", fontsize=9)
        else:
            ax2.set_yticklabels([])
        ax.set_title(f"({PANEL_LBL[i][j]}) {VAR_NAME[v]}: {EXP_NAME[j]}",
                     loc="left", fontsize=9.5)
        if j == 0: ax.set_ylabel(u"Temperature anomaly (K)")
        ax.set_xlim(otime[0], ftime_d[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(alpha=0.2, lw=0.5)
        _h1, _l1 = ax.get_legend_handles_labels()
        _h2, _l2 = ax2.get_legend_handles_labels()
        ax2.legend(_h1 + _h2, _l1 + _l2, ncol=2, loc="upper left",
                   frameon=True, fontsize=6.8)
        for yr in (2026, 2027):
            jan = np.datetime64("%d-01-01" % yr)
            if otime[0] <= jan <= ftime[-1]:
                ax.annotate(str(yr), xy=(jan, 0), xytext=(0, -26), textcoords="offset points",
                            xycoords=("data", "axes fraction"), ha="center", fontsize=8,
                            annotation_clip=False)
fig.suptitle("Forecast of tropical-mean SST and LTT warming with mode contributions\n"
             "XRO initialised from real-time OISST+PMEL pipeline (init 2026-04)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
FIG_DIR = os.environ.get("XRO_FIG_DIR", ".")
_fp = FIG_DIR + "/Fig_tropmean_forecast_SST_LTT_contributions"
fig.savefig(_fp + ".png", dpi=300, bbox_inches="tight")
fig.savefig(_fp + ".pdf", bbox_inches="tight")
print("saved:", _fp + ".png / .pdf")

