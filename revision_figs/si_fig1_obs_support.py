import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.join(HERE, "intermediates")
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import numpy as np, pandas as pd, xarray as xr, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = DATA + "/"
def to_per(ix): return pd.DatetimeIndex(ix).to_period("M")
idx = xr.open_dataset(D + "XRO_indices_oras5_extended_2025.nc")
ti = to_per(idx.time.values)
def S(v):
    s = pd.Series(np.asarray(idx[v].values, float), index=ti); return s/s.std()
sstds = xr.open_dataset(D + "ERA5_tropical_mean_sst_anom_detr_1950-2025.nc")
ysst = pd.Series(np.asarray(sstds[list(sstds.data_vars)[0]].values, float), index=to_per(sstds.time.values))
dpk = pickle.load(open(os.path.join(INT, "ltt_depth_check.pkl"),"rb"))
yltt = pd.Series(dpk["L3 (1000-850)"], index=to_per(dpk["time"]))
common = ti.intersection(ysst.index)
MODES8 = ["NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]
rem8 = (sum(S(m) for m in MODES8)/8).loc[common]
IOB  = S("IOB"); n34 = S("Nino34")
y    = ysst.loc[common]
df = pd.read_pickle(os.path.join(INT, "obs_support_events.pkl"))
REM8ev = df[MODES8].mean(axis=1)
EV = list(df.index)

cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")

fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0]))
fig.subplots_adjust(left=0.05, right=0.965, top=0.80, bottom=0.15, wspace=0.42)
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")

# (a) heatmap all-8
ax = axes[0]
xb = np.arange(-1.3, 1.35, 0.2); yb = np.arange(-2.7, 3.3, 0.3)
H,_,_ = np.histogram2d(rem8.values, n34.loc[common].values, bins=[xb, yb])
W,_,_ = np.histogram2d(rem8.values, n34.loc[common].values, bins=[xb, yb], weights=y.values)
M = np.where(H>=2, W/np.maximum(H,1), np.nan)
pm = ax.pcolormesh(xb, yb, np.ma.masked_invalid(M).T, cmap=cmap, vmin=-0.45, vmax=0.45)
# ax.scatter(rem8.values, n34.loc[common].values, s=2.5, color="gray", alpha=0.3, zorder=2)
ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("remote-mode index, all-8 mean (s.d.)"); ax.set_ylabel("Nino3.4 (s.d.)")
ax.set_title("(a) Tropical-mean SST on the (modes, ENSO)\nplane, monthly binned mean 1979-2025", loc="left", fontsize=9.5)
cb = fig.colorbar(pm, ax=ax, pad=0.02, fraction=0.05); cb.set_label("tropical-mean SST (K)", fontsize=8)

# (b) event scatter all-8
ax = axes[1]
sc = ax.scatter(REM8ev, df.sst, c=df.n34, cmap="Reds", vmin=0, vmax=2.7, s=140, edgecolor="k", zorder=3)
OFF = {2023:(-44,2), 2015:(-44,2), 1997:(-44,-4), 2019:(-14,10), 1982:(10,-4), 2002:(-42,-6),
       1987:(10,2), 2009:(-44,-4), 2004:(8,-16), 2006:(-40,-16), 1994:(9,6), 1991:(10,-4)}
for ev in EV:
    dx,dy = OFF[ev]
    ax.annotate(str(ev), (REM8ev[ev], df.sst[ev]), textcoords="offset points", xytext=(dx,dy),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                            shrinkA=0, shrinkB=4) if abs(dx)>20 or abs(dy)>12 else None)
b = np.polyfit(REM8ev, df.sst, 1); xx = np.linspace(REM8ev.min(), REM8ev.max(), 10)
ax.plot(xx, np.polyval(b, xx), "k--", lw=1)
ax.axvline(0, color="gray", lw=0.6); ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("remote-mode index, all-8 mean (s.d.)"); ax.set_ylabel("Tropical-mean SST (K)")
ax.set_title(f"(b) Event-mean SST vs remote modes\nr = {np.corrcoef(REM8ev,df.sst)[0,1]:+.2f}  (Nino3.4 alone r = {np.corrcoef(df.n34,df.sst)[0,1]:+.2f})", loc="left", fontsize=9.5)
cbb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046); cbb.set_label("Nino3.4, Nov(0)-Feb(+1) (K)", fontsize=8)

# (c) IOB at the Nino3.4 peak month vs 2 months later (C2-removal peak, Figs 4-5)
ax = axes[2]
X = 2
iob_pk, iob_pk2 = {}, {}
for ev in EV:
    w = pd.period_range(f"{ev}-01", periods=24, freq="M")
    nv = n34.reindex(w)
    mpk = nv.index[int(np.nanargmax(nv.values))]
    iob_pk[ev]  = IOB.loc[mpk]
    iob_pk2[ev] = IOB.loc[mpk + X]
iob_pk = pd.Series(iob_pk); iob_pk2 = pd.Series(iob_pk2)
r0 = np.corrcoef(iob_pk, df.sst)[0,1]; r2 = np.corrcoef(iob_pk2, df.sst)[0,1]
ax.scatter(iob_pk2, df.sst, s=120, color="tab:red", edgecolor="k", lw=0.8, zorder=4)
ax.scatter(iob_pk, df.sst, s=110, marker="^", facecolor="none", edgecolor="tab:blue", lw=1.6, zorder=3)
for ev in EV:
    ax.plot([iob_pk[ev], iob_pk2[ev]], [df.sst[ev]]*2, color="gray", lw=0.6, zorder=2)
b1 = np.polyfit(iob_pk2, df.sst, 1); b2 = np.polyfit(iob_pk, df.sst, 1)
xx = np.linspace(min(iob_pk.min(), iob_pk2.min()), max(iob_pk.max(), iob_pk2.max()), 10)
ax.plot(xx, np.polyval(b1, xx), color="tab:red", lw=1.7)
ax.plot(xx, np.polyval(b2, xx), color="tab:blue", ls="--", lw=1.4)
ax.axvline(0, color="gray", lw=0.6); ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("IOB index (s.d.)"); ax.set_ylabel("Tropical-mean SST (K)")
ax.set_title("(c) As (b), IOB at single months tied to the Nino3.4 peak\nat the peak vs 2 months later (C2-removal peak in Figs. 4-5)", loc="left", fontsize=9.5)
handles=[Line2D([],[],marker="^",mec="tab:blue",mfc="none",color="tab:blue",ls="--",label=f"IOB at the Nino3.4 peak month, r={r0:+.2f}"),
         Line2D([],[],marker="o",color="tab:red",mec="k",ls="-",label=f"IOB 2 months after the peak, r={r2:+.2f}")]
ax.legend(handles=handles, fontsize=8, loc="upper left")
print(f"IOB@N34pk r={r0:+.2f}; IOB@N34pk+2 r={r2:+.2f}")

fig.suptitle("Observational support (ERA5 / ORAS5, no XRO): remote climate modes and El Nino-driven tropical-mean SST",
             fontsize=11.5, x=0.5, y=0.97)
fig.savefig("obs_support_v25.png", dpi=160)
print("saved")
