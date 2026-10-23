# Rebuild the real-time XRO input files, extended to the latest available month.
#   realtime_oisst_sst_indices.nc  (9 SST-mode indices from NOAA OISST v2.1, PSL OPeNDAP)
#   realtime_tao_pmel_wwv.nc       (WWV from the PMEL warm water volume text file)
# Same recipe as XRO_official_reproduce_2026.ipynb cells 3 and 5:
#   1982-2020 monthly climatology, linear detrend over the full record,
#   5-year chunked OPeNDAP reads (the PSL server truncates large transfers).
import os, time as _t
import numpy as np, pandas as pd, xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_r2")
CLIMBASE = slice("1982-01", "2020-12")
END = "2026-09"   # last complete month with both OISST and PMEL WWV

OISST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"

boxes = {
 "Nino34": (-5,5,190,240),  "NPMM": (10,25,200,240), "SPMM": (-25,-15,250,270),
 "IOB":    (-20,20,40,100), "TNA":  (5,25,305,345),  "ATL3": (-3,3,340,360),
 "IOD_W":  (-10,10,50,70),  "IOD_E":(-10,0,90,110),
 "SIOD_W": (-37,-27,55,65), "SIOD_E":(-28,-18,90,100),
 "SASD_NE":(-15,0,345,360), "SASD_NE2":(-15,0,0.125,5), "SASD_SW":(-30,-15,320,345),
}

def anom_detrend(da, base=CLIMBASE):
    clim = da.sel(time=base).groupby("time.month").mean("time")
    a = da.groupby("time.month") - clim
    t = np.arange(a.sizes["time"]); y = np.asarray(a.values, float); m = np.isfinite(y)
    coef = np.polyfit(t[m], y[m], 1)
    return a.copy(data=y - np.polyval(coef, t))

# ---- OISST indices ----
out_oisst = os.path.join(DATA, "realtime_oisst_sst_indices_2026-09.nc")
if not os.path.exists(out_oisst):
    ds = xr.open_dataset(OISST_URL); sst = ds["sst"]
    if sst.lat.values[0] > sst.lat.values[-1]: sst = sst.sortby("lat")
    YRS = list(range(1982, 2027, 5)) + [2027]
    def box(s, n, w, e):
        sub = sst.sel(lat=slice(s, n), lon=slice(w, e)); wt = np.cos(np.deg2rad(sub.lat))
        parts = []
        for a, b in zip(YRS[:-1], YRS[1:]):
            for _ in range(5):
                c = sub.sel(time=slice(f"{a}-01", f"{b-1}-12")).load()
                if float(np.nanstd(c.values)) > 0.05: break
                _t.sleep(3)
            parts.append(c)
        bm = xr.concat(parts, dim="time").weighted(wt).mean(("lat","lon"), skipna=True)
        bm = bm.copy(data=np.asarray(bm.values, float)); bm.encoding = {}; return bm
    R = {}
    for k, v in boxes.items():
        R[k] = box(*v)
        print("box", k, "->", str(R[k].time.values[-1])[:7], flush=True)
    A = {v: anom_detrend(R[v]) for v in ["Nino34","NPMM","SPMM","IOB","TNA","ATL3"]}
    A["IOD"]  = anom_detrend(R["IOD_W"])  - anom_detrend(R["IOD_E"])
    A["SIOD"] = anom_detrend(R["SIOD_W"]) - anom_detrend(R["SIOD_E"])
    A["SASD"] = anom_detrend((R["SASD_NE"]+R["SASD_NE2"])/2.0) - anom_detrend(R["SASD_SW"])
    oo = xr.Dataset(A).sel(time=slice("1982-01", END))
    for v in oo.data_vars: oo[v].encoding = {}
    oo["time"].encoding = {}
    oo.to_netcdf(out_oisst, encoding={v: {"dtype":"float64","_FillValue":None} for v in oo.data_vars})
    print("built", out_oisst, flush=True)

oo = xr.open_dataset(out_oisst)
print("OISST indices:", str(oo.time.values[0])[:7], "->", str(oo.time.values[-1])[:7])

# ---- WWV ----
out_wwv = os.path.join(DATA, "realtime_tao_pmel_wwv_2026-09.nc")
WWV_DAT = os.path.join(DATA, "wwv.dat")
if not os.path.exists(WWV_DAT):
    import urllib.request
    urllib.request.urlretrieve("https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat", WWV_DAT)
    print("downloaded wwv.dat")
rows = []
for ln in open(WWV_DAT):
    p = ln.split()
    if len(p) == 3 and p[0].isdigit() and len(p[0]) == 6:
        rows.append((p[0], float(p[1])))
df = pd.DataFrame(rows, columns=["yyyymm","vol"])
tt = pd.to_datetime(df["yyyymm"] + "01", format="%Y%m%d")
vol = xr.DataArray(df["vol"].values, coords={"time": tt.values}, dims="time")
R_E = 111320.7
area = (10*R_E) * (160*R_E*np.cos(np.deg2rad(np.linspace(-5, 5, 21))).mean())
clim = vol.sel(time=CLIMBASE).groupby("time.month").mean("time")
a = (vol.groupby("time.month") - clim) / area
t = np.arange(a.sizes["time"]); y = a.values; m = np.isfinite(y)
coef = np.polyfit(t[m], y[m], 1)
wwv = a.copy(data=y - np.polyval(coef, t)); wwv.name = "WWV"
wwv = wwv.sel(time=slice(None, END))
wwv.encoding = {}; wwv["time"].encoding = {}
wwv.to_netcdf(out_wwv, encoding={"WWV": {"dtype":"float64","_FillValue":None}})
print("WWV:", str(wwv.time.values[0])[:7], "->", str(wwv.time.values[-1])[:7],
      "| std=%.2f m" % float(wwv.std()))

# ---- consistency check vs the files used for the submitted Fig 10 ----
MAIN = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
old_o = xr.open_dataset(os.path.join(MAIN, "realtime_oisst_sst_indices.nc"))
old_w = xr.open_dataset(os.path.join(MAIN, "realtime_tao_pmel_wwv.nc"))["WWV"]
for v in ["Nino34","NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]:
    a1, a2 = xr.align(old_o[v], oo[v], join="inner")
    r = float(np.corrcoef(a1.values, a2.values)[0, 1])
    d = float(np.nanmax(np.abs(a1.values - a2.values)))
    print(f"overlap check {v:7s} r={r:.5f} max|diff|={d:.4f}")
w1, w2 = xr.align(old_w, wwv, join="inner")
print("overlap check WWV     r=%.5f max|diff|=%.4f" %
      (float(np.corrcoef(w1.values, w2.values)[0,1]), float(np.nanmax(np.abs(w1.values - w2.values)))))
