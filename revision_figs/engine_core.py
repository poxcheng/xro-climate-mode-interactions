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

INIT  = os.environ.get("XRO_INIT", "2026-04")
NLEAD = int(os.environ.get("XRO_NLEAD", "18"))
NMEM  = 1000
OISST_FILE = os.environ.get("XRO_OISST_FILE", "realtime_oisst_sst_indices.nc")
WWV_FILE   = os.environ.get("XRO_WWV_FILE", "realtime_tao_pmel_wwv.nc")
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
_o = _mz(xr.open_dataset(os.path.join(DATA_DIR, OISST_FILE)))
_w = _mz(xr.open_dataset(os.path.join(DATA_DIR, WWV_FILE))["WWV"])
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
_tail0 = str((np.datetime64(INIT, "M") - np.timedelta64(15, "M")))
_dst = ds_fc.sel(time=slice(_tail0, INIT))
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
