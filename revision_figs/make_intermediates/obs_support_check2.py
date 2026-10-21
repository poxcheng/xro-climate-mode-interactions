import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.normpath(os.path.join(HERE, "..", "intermediates"))
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import numpy as np, pandas as pd, xarray as xr, pickle

D = DATA + "/"
def to_per(ix): return pd.DatetimeIndex(ix).to_period("M")
idx = xr.open_dataset(D + "XRO_indices_oras5_extended_2025.nc")
ti = to_per(idx.time.values)
def Sraw(v): return pd.Series(np.asarray(idx[v].values, float), index=ti)
def S(v):
    s = Sraw(v); return s/s.std()
n34 = Sraw("Nino34")
sst = xr.open_dataset(D + "ERA5_tropical_mean_sst_anom_detr_1950-2025.nc")
ysst = pd.Series(np.asarray(sst[list(sst.data_vars)[0]].values, float), index=to_per(sst.time.values))
d = pickle.load(open(os.path.join(INT, "ltt_depth_check.pkl"), "rb"))
yltt = pd.Series(d["L3 (1000-850)"], index=to_per(d["time"]))

MODES = ["NPMM","SPMM","IOB","IOD","SIOD","TNA","ATL3","SASD"]
EV=[1982,1987,1991,1994,1997,2002,2004,2006,2009,2015,2019,2023]
rows=[]
for ev in EV:
    mat = pd.period_range(f"{ev}-11", periods=4, freq="M")
    pre = pd.period_range(f"{ev}-07", periods=4, freq="M")  # JASO(0): ENSO developing
    row = dict(event=ev, n34=n34.loc[mat].mean(), n34pre=n34.loc[pre].mean(),
               sst=ysst.loc[mat].mean(), ltt=yltt.loc[mat].mean())
    for m in MODES: row[m] = S(m).loc[mat].mean()
    rows.append(row)
df = pd.DataFrame(rows).set_index("event")

def partial(y, x, z):
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return np.corrcoef(ry, rx)[0,1]

print("Q1: which modes carry the event-level tropical-mean signal (n=12)")
print(f"{'index':16s} {'r(ltt)':>7s} {'r(sst)':>7s} {'r(ltt|n34)':>10s} {'r(sst|n34)':>10s}")
cands = {m: df[m] for m in MODES}
cands["(IOB+TNA)/2"]   = (df.IOB+df.TNA)/2
cands["warm3 IOB+TNA+NPMM"] = (df.IOB+df.TNA+df.NPMM)/3
cands["all-8 mean"]    = df[MODES].mean(axis=1)
for k,v in cands.items():
    print(f"{k:16s} {np.corrcoef(v,df.ltt)[0,1]:+7.2f} {np.corrcoef(v,df.sst)[0,1]:+7.2f} "
          f"{partial(df.ltt,v,df.n34):+10.2f} {partial(df.sst,v,df.n34):+10.2f}")

REM = (df.IOB+df.TNA)/2
print("\nQ2: SST version headline numbers")
print(f"  r(n34, SST)={np.corrcoef(df.n34,df.sst)[0,1]:+.2f}  r(REM, SST)={np.corrcoef(REM,df.sst)[0,1]:+.2f}")
print(f"  r(n34, LTT)={np.corrcoef(df.n34,df.ltt)[0,1]:+.2f}  r(REM, LTT)={np.corrcoef(REM,df.ltt)[0,1]:+.2f}")

print("\nQ3: ENSO->modes required (mediation logic)")
print(f"  (i) ENSO forces modes: r(n34pre JASO(0), REM mature)={np.corrcoef(df.n34pre,REM)[0,1]:+.2f}; r(n34 mature, REM)={np.corrcoef(df.n34,REM)[0,1]:+.2f}")
b = np.polyfit(df.n34, REM, 1)
REM_enso = np.polyval(b, df.n34); REM_res = REM - REM_enso
print(f"  (ii) split REM = ENSO-congruent + residual:")
for y,nm in [(df.ltt,"LTT"),(df.sst,"SST")]:
    print(f"      {nm}: r(y, REM_enso)={np.corrcoef(y,REM_enso)[0,1]:+.2f}, r(y, REM_res)={np.corrcoef(y,REM_res)[0,1]:+.2f}")
print(f"  (iii) partials: r(LTT, n34 | REM)={partial(df.ltt,df.n34,REM.values):+.2f} vs r(LTT, REM | n34)={partial(df.ltt,REM.values,df.n34):+.2f}")
print(f"          r(SST, n34 | REM)={partial(df.sst,df.n34,REM.values):+.2f} vs r(SST, REM | n34)={partial(df.sst,REM.values,df.n34):+.2f}")
df.to_pickle("obs_support_events.pkl")
