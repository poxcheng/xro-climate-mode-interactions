import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.join(HERE, "intermediates")
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import numpy as np, pandas as pd, xarray as xr, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.rcParams["font.size"] = 12
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

D = DATA + "/"
def to_per(ix): return pd.DatetimeIndex(ix).to_period("M")
idx = xr.open_dataset(D + "XRO_indices_oras5_extended_2025.nc")
n34 = pd.Series(np.asarray(idx["Nino34"].values, float), index=to_per(idx.time.values))
n34s = n34/n34.std()
sstds = xr.open_dataset(D + "ERA5_tropical_mean_sst_anom_detr_1950-2025.nc")
ysst = pd.Series(np.asarray(sstds[list(sstds.data_vars)[0]].values, float), index=to_per(sstds.time.values))
d = pickle.load(open(os.path.join(INT, "ltt_depth_check.pkl"),"rb"))
yltt = pd.Series(d["L3 (1000-850)"], index=to_per(d["time"]))
com = yltt.index.intersection(ysst.index).intersection(n34s.index)
sm = lambda s: s.loc[com].rolling(3, center=True, min_periods=1).mean()
t = com.to_timestamp()
tnum = matplotlib.dates.date2num(t.to_pydatetime())

fig, ax = plt.subplots(figsize=(11.2, 3.8))
# background shading: vertical bands colored by standardized Nino3.4, saturate +-2.5
edges = np.concatenate([tnum - 15, [tnum[-1] + 15]])
pm = ax.pcolormesh(edges, np.array([-0.75, 0.75]), n34s.loc[com].values[None, :],
              cmap="RdBu_r", vmin=-3.5, vmax=3.5, shading="flat", rasterized=True)
ax.plot(t, sm(yltt).values, color="0.35", lw=1.7, ls="--", path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()], label="tropical-mean LTT")
ax.plot(t, sm(ysst).values, color="k", lw=2.2, path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()], label="tropical-mean SST")
ax.axhline(0, color="gray", lw=0.6)
ax.set_ylabel("SST / LTT anomaly (K)")
ax.set_ylim(-0.75, 0.75)
ax.set_xlim(t[0], t[-1])
ax.legend(loc="upper left", fontsize=10.8, framealpha=0.95)
ax.set_title("Tropical-mean (30S-30N) SST and LTT anomalies with standardized Nino3.4",
             fontsize=12.6, loc="left")
cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.035)
cb.set_label("standardized Nino3.4 (s.d.)", fontsize=10.8)
fig.tight_layout()
fig.savefig("Fig1_new_v4.png", dpi=200)
print("saved")
