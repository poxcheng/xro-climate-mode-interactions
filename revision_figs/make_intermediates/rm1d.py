import os
HERE = os.path.dirname(os.path.abspath(__file__))
INT = os.path.normpath(os.path.join(HERE, "..", "intermediates"))
DATA = os.environ.get("XRO_DATA_DIR", os.path.normpath(os.path.join(HERE, "..", "data")))
import json, re, time, numpy as np, pandas as pd, xarray as xr, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NBPATH = os.path.normpath(os.path.join(HERE, "..", "..", "Full_code_reproduce_v1.ipynb"))
nbj = json.load(open(NBPATH))
_code = [c for c in nbj["cells"] if c["cell_type"] == "code"]
_k = next(i for i, c in enumerate(_code) if "_build_stepper_B" in "".join(c["source"]))
for _c in _code[:_k]:
    _src = "".join(_c["source"])
    if _src.lstrip().startswith("!"):
        continue  # Colab-only pip cells
    exec(_src, globals())
s = "".join(_code[_k]["source"])
s=s.replace("'per_model_series': per_model_series,",
    "'per_model_series': per_model_series,\n"
    "        '_ctx': dict(Lac_obs_B=Lac_obs_B, DELTA_B=DELTA_B, BEST_SHIFT_B=BEST_SHIFT_B,\n"
    "                     obs_time=obs_time, X_np=X_np, pairs=pairs, idx_map_B=idx_map_B,\n"
    "                     MODES9_SST_B=MODES9_SST_B, MODES9_LTT_B=MODES9_LTT_B,\n"
    "                     stepper=_build_stepper_B, subset_fit=_subset_reorder_fit,\n"
    "                     intersect=_intersect_time, mz=_month_z_with_B, recon=_recon_from_beta_B,\n"
    "                     mu_ltt=mu_ltt_B, sig_ltt=sig_ltt_B, beta_ltt=beta_ltt_B,\n"
    "                     mu_sst=mu_sst_B, sig_sst=sig_sst_B, beta_sst=beta_sst_B,\n"
    "                     STATE10_B=STATE10_B),")
s = re.sub(r'CASES_TO_RUN = \[[^\]]*\]', 'CASES_TO_RUN = ["BASE"]', s, count=1)
exec(s, globals())
C=multi['_ctx']; NSTEP_B=10; EXPLODE=1e6
ST=[v.upper() for v in C['STATE10_B']]
ENSO=[ST.index("NINO34"), ST.index("WWV")]
SUB=[i for i in range(10) if i not in ENSO]
EVENTS=[1982,1987,1991,1994,1997,2002,2004,2006,2009]

def ser(YY, common_t, var):
    if var=='LTT':
        Z=xr.concat([xr.DataArray(YY[C['idx_map_B']["NINO34"],:],dims=["time"],coords={"time":common_t}).rename("NINO34")]+
            [xr.DataArray(YY[C['idx_map_B'][m.upper()],:],dims=["time"],coords={"time":common_t}).rename(m) for m in C['MODES9_LTT_B'][1:]],
            dim="mode").assign_coords(mode=("mode",C['MODES9_LTT_B'])).transpose("time","mode")
        return C['recon'](C['beta_ltt'], C['mz'](C['mu_ltt'],C['sig_ltt'],Z))
    Z=xr.DataArray(np.stack([YY[C['idx_map_B'][m.upper()],:] for m in C['MODES9_SST_B']],axis=1),
                   dims=["time","mode"],coords={"time":common_t,"mode":C['MODES9_SST_B']})
    return C['recon'](C['beta_sst'], C['mz'](C['mu_sst'],C['sig_sst'],Z))

def composite24(y):
    t=pd.DatetimeIndex(y.time.values)
    s2=pd.Series(y.values,index=t.to_period('M').to_timestamp())
    acc=[]
    for ey in EVENTS:
        tgt=pd.date_range(pd.Timestamp(ey,1,1),periods=24,freq='MS')
        w=s2.reindex(tgt)
        if w.notna().mean()>0.95: acc.append(w.values)
    return np.nanmean(np.stack(acc),axis=0)

def replay(Lac,args):
    common_t,mon_c,dlt,X0=args
    step=C['stepper'](Lac,nstep=NSTEP_B)
    YY=np.zeros((10,len(common_t))); YY[:,0]=X0
    for i in range(len(common_t)-1):
        mon=int((mon_c[i]+C['BEST_SHIFT_B'])%12)
        YY[:,i+1]=(step(YY[:,i].reshape(-1,1),mon)+dlt[:,i+1].reshape(-1,1)).squeeze()
        if not np.all(np.isfinite(YY[:,i+1])) or np.nanmax(np.abs(YY[:,i+1]))>EXPLODE: return None
    return YY

def rm(L, which):
    out=L.copy()
    if which=='C1':
        for r in ENSO:
            for c in SUB: out[r,c,:]=0.0
    elif which=='C2':
        for r in SUB:
            for c in ENSO: out[r,c,:]=0.0
    else:
        for r in SUB:
            for c in SUB:
                if r!=c: out[r,c,:]=0.0
    return out
def fix(L, which, Lobs):
    out=L.copy()
    if which=='C1':
        for r in ENSO:
            for c in SUB: out[r,c,:]=Lobs[r,c,:]
    elif which=='C2':
        for r in SUB:
            for c in ENSO: out[r,c,:]=Lobs[r,c,:]
    elif which=='L8':
        for r in SUB:
            for c in SUB: out[r,c,:]=Lobs[r,c,:]
    else:  # LENSO
        for r in ENSO:
            for c in ENSO: out[r,c,:]=Lobs[r,c,:]
    return out

BLOCKS=['C1','C2','L8off']
OTHERS={'C1':['C2','L8','LENSO'],'C2':['C1','L8','LENSO'],'L8off':['C1','C2','LENSO']}
Lobs=C['Lac_obs_B']
# ---- OBS reference curves ----
obs_args=(C['obs_time'],pd.DatetimeIndex(C['obs_time']).month.values-1,C['DELTA_B'],np.asarray(C['X_np'],float)[:,0])
YY0=replay(Lobs,obs_args)
obs_curve={}
for v in ('SST','LTT'):
    b=composite24(ser(YY0,C['obs_time'],v))
    for B in BLOCKS:
        Yr=replay(rm(Lobs,B),obs_args)
        obs_curve[(v,B)]=composite24(ser(Yr,C['obs_time'],v))-b

# ---- CMIP ----
cut=pd.Timestamp("2006-01-01")
raw={ (v,B):[] for v in ('SST','LTT') for B in BLOCKS }
corr={ (v,B,O):[] for v in ('SST','LTT') for B in BLOCKS for O in OTHERS[B] }
t0=time.time(); nM=0
for key,fp,tp in C['pairs']:
    try:
        ts=xr.open_dataset(tp)
        if pd.DatetimeIndex(ts.time.values)[-1]<cut: continue
        fit=C['subset_fit'](xr.open_dataset(fp),C['STATE10_B']); Lm=fit["Lac"].values.copy()
        common_t,obs_idx,_=C['intersect'](C['obs_time'],ts.time.values)
        mon_c=pd.DatetimeIndex(common_t).month.values-1
        args=(common_t,mon_c,C['DELTA_B'][:,obs_idx],np.asarray(C['X_np'],float)[:,obs_idx[0]])
        YYb=replay(Lm,args)
        if YYb is None: continue
        base={v:composite24(ser(YYb,common_t,v)) for v in ('SST','LTT')}
        okm=True; tmpR={}; tmpC={}
        for B in BLOCKS:
            Yr=replay(rm(Lm,B),args)
            if Yr is None: okm=False; break
            for v in ('SST','LTT'):
                tmpR[(v,B)]=composite24(ser(Yr,common_t,v))-base[v]
            for O in OTHERS[B]:
                Lc=fix(Lm,O,Lobs)
                Ycb=replay(Lc,args); Ycr=replay(rm(Lc,B),args)
                if Ycb is None or Ycr is None: okm=False; break
                for v in ('SST','LTT'):
                    tmpC[(v,B,O)]=composite24(ser(Ycr,common_t,v))-composite24(ser(Ycb,common_t,v))
            if not okm: break
        if not okm: continue
        for k,x in tmpR.items(): raw[k].append(x)
        for k,x in tmpC.items(): corr[k].append(x)
        nM+=1
        if nM%20==0: print(f"  {nM} models ({time.time()-t0:.0f}s)",flush=True)
    except Exception as e:
        print("  skip",key,type(e).__name__,flush=True)
print(f"models: {nM}")
pickle.dump(dict(obs=obs_curve,raw=raw,corr=corr,n=nM),open('rm1d.pkl','wb'))

# ---- figure 2x3 ----
rel=np.arange(24)
xt=[0,5,10,13,17,23]; xl=['Jan(0)','Jun(0)','Nov(0)','Feb(+1)','Jun(+1)','Dec(+1)']
CC={'C1':'tab:green','C2':'tab:blue','L8':'tab:purple','LENSO':'tab:orange'}
fig,axes=plt.subplots(2,3,figsize=(14.5,7.6),sharex=True)
for ri,v in enumerate(('SST','LTT')):
    for ci,B in enumerate(BLOCKS):
        ax=axes[ri,ci]
        R=np.stack(raw[(v,B)])
        ax.fill_between(rel,np.percentile(R,10,axis=0),np.percentile(R,90,axis=0),
                        color='tab:red',alpha=0.15,lw=0,label='CMIP 10-90%')
        ax.plot(rel,R.mean(axis=0),color='tab:red',lw=2,label='CMIP mean (raw)')
        for O in OTHERS[B]:
            Cv=np.stack(corr[(v,B,O)])
            ax.plot(rel,Cv.mean(axis=0),color=CC[O],lw=1.4,ls='--',label=f'+{O}'+r'$\to$obs')
        ax.plot(rel,obs_curve[(v,B)],color='k',lw=2.2,label='OBS operator')
        ax.axhline(0,color='0.6',lw=0.7)
        lab={'C1':r'$C_1$=0','C2':r'$C_2$=0','L8off':r'$L_8^{off}$=0'}[B]
        ax.set_title(f"({'abcdef'[ri*3+ci]}) {v}: {lab} $-$ BASE",loc='left',fontsize=10.5)
        ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=8)
        if ci==0: ax.set_ylabel(f'tropical-mean {v} diff (K)')
        if ri==0 and ci==0: ax.legend(fontsize=7.5,ncol=1,loc='lower left')
fig.suptitle(f"Removal experiments as tropical-mean composites, raw CMIP vs block-corrected operators (n={nM}; 9 ONI events 1979-2010)",fontsize=11.5)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('rm1d.png',dpi=160,bbox_inches='tight')
print("saved rm1d.png")
