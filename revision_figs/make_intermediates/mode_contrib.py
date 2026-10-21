import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.normpath(os.path.join(HERE, "..", "intermediates"))
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import numpy as np, pandas as pd, xarray as xr, pickle
from scipy.signal import detrend

D = DATA + "/"
def to_per(ix): return pd.DatetimeIndex(ix).to_per if False else pd.DatetimeIndex(ix).to_period("M")
idx = xr.open_dataset(D + "XRO_indices_oras5_extended_2025.nc")
ti = to_per(idx.time.values)
MODES = ["Nino34","NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]
X = np.column_stack([np.asarray(idx[m].values, float) for m in MODES])
sstds = xr.open_dataset(D + "ERA5_tropical_mean_sst_anom_detr_1950-2025.nc")
ysst = pd.Series(np.asarray(sstds[list(sstds.data_vars)[0]].values, float), index=to_per(sstds.time.values))
d = pickle.load(open(os.path.join(INT, "ltt_depth_check.pkl"), "rb"))
yltt = pd.Series(d["L3 (1000-850)"], index=to_per(d["time"]))

out = {}
for nm, yy in (("SST", ysst), ("LTT", yltt)):
    com = ti.intersection(yy.index)
    sel = np.isin(ti, com)
    Xv = X[sel]; months = com.month.values - 1
    yv = yy.loc[com].values
    # month-wise z-score + month-wise OLS (paper recipe)
    Z = np.zeros_like(Xv); B = np.zeros((12, 9))
    for m in range(12):
        s = months == m
        mu, sd = Xv[s].mean(0), Xv[s].std(0)
        Z[s] = (Xv[s]-mu)/sd
        B[m] = np.linalg.lstsq(Z[s], yv[s], rcond=None)[0]
    contrib = Z * B[months]           # (time, mode) contributions
    recon = contrib.sum(1)
    print(f"{nm}: recon r = {np.corrcoef(recon, yv)[0,1]:.3f}")
    # mature-phase composite per mode over 12 events
    EV=[1982,1987,1991,1994,1997,2002,2004,2006,2009,2015,2019,2023]
    yrs, mons = com.year.values, com.month.values
    rows = []
    for ev in EV:
        mask = ((yrs==ev)&(mons>=11)) | ((yrs==ev+1)&(mons<=2))
        rows.append(contrib[mask].mean(0))
    C = pd.DataFrame(rows, index=EV, columns=MODES)
    out[nm] = C
    mean = C.mean(0)
    tot = mean.sum()
    print(f"  composite mature mode-explained total = {tot:+.3f} K")
    for m in mean.sort_values(ascending=False).index:
        print(f"    {m:7s} {mean[m]:+.4f} K  ({100*mean[m]/tot:+.0f}%)")
pickle.dump(out, open("mode_contrib.pkl","wb"))
print("saved mode_contrib.pkl")
