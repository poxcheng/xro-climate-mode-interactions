import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.normpath(os.path.join(HERE, "..", "intermediates"))
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import numpy as np, xarray as xr, pandas as pd
from scipy.signal import detrend

F = (DATA + "/ERA5_T_plevs_1deg_30S30N_1979-2025.nc")
ds = xr.open_dataset(F)
t = ds["t"].rename({"valid_time":"time","pressure_level":"lev","latitude":"lat","longitude":"lon"})

def dp_weights(plevs):
    p = np.sort(np.asarray(plevs, float))[::-1]
    dp = np.zeros_like(p)
    dp[0] = (p[0]-p[1])/2; dp[-1] = (p[-2]-p[-1])/2
    dp[1:-1] = (p[:-2]-p[2:])/2
    return p, dp/dp.sum()

def layer_mean(t, plevs):
    p, w = dp_weights(plevs)
    sub = t.sel(lev=p)
    wda = xr.DataArray(w, dims=["lev"], coords={"lev": p})
    return (sub*wda).sum("lev")

# MSU/AMSU TLT-like static weighting (approx of RSS TLT wt fn, peak ~700-850 hPa,
# tails to 300 hPa; normalized). Values interpolated from published TLT weighting curves.
TLT_P = np.array([1000., 925., 850., 700., 600., 500., 400., 300.])
TLT_W = np.array([0.09, 0.11, 0.13, 0.145, 0.125, 0.10, 0.07, 0.04])
TLT_W = TLT_W/TLT_W.sum()

def tlt_proxy(t):
    sub = t.sel(lev=TLT_P)
    wda = xr.DataArray(TLT_W, dims=["lev"], coords={"lev": TLT_P})
    return (sub*wda).sum("lev")

wlat = np.cos(np.deg2rad(t.lat))
def tropmean(da): return da.weighted(wlat).mean(["lat","lon"])

def anom_detrend(s):
    s = s.groupby("time.month") - s.groupby("time.month").mean("time")
    return s.copy(data=detrend(s.values, type="linear"))

series = {}
series["L3 (1000-850)"]  = anom_detrend(tropmean(layer_mean(t, [1000,925,850])))
series["L6 (1000-500)"]  = anom_detrend(tropmean(layer_mean(t, [1000,925,850,700,600,500])))
series["L8 (1000-300)"]  = anom_detrend(tropmean(layer_mean(t, [1000,925,850,700,600,500,400,300])))
series["TLT proxy"]      = anom_detrend(tropmean(tlt_proxy(t)))

ref = series["L3 (1000-850)"]
print(f"{'series':16s} {'r vs L3':>8s} {'std (K)':>8s} {'std ratio':>9s} {'regr slope':>10s}")
for k, v in series.items():
    r = float(np.corrcoef(ref, v)[0,1])
    sr = float(v.std()/ref.std())
    sl = float(np.polyfit(ref, v, 1)[0])
    print(f"{k:16s} {r:8.4f} {float(v.std()):8.4f} {sr:9.3f} {sl:10.3f}")

# lag check: cross-corr of each with L3 at +-3 months
print("\nmax-correlation lag vs L3 (months, positive = series lags L3):")
for k, v in series.items():
    best = max(range(-3,4), key=lambda L: np.corrcoef(ref[max(0,-L):len(ref)-max(0,L)], v[max(0,L):len(v)-max(0,-L)])[0,1])
    print(f"  {k:16s} lag {best:+d}")

import pickle
pickle.dump({k: v.values for k,v in series.items()} | {"time": ref.time.values, "TLT_P":TLT_P, "TLT_W":TLT_W},
            open("ltt_depth_check.pkl","wb"))
print("\nsaved ltt_depth_check.pkl")
