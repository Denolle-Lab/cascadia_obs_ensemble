"""Phase 17: compare Route A (response-removed) ML with the counts-based _kpos ML.

Joins cascadia_catalog_ML_routeA.csv and cascadia_catalog_ML_kpos.csv on event_id and
prints the offset/slope, the difference per ML bin, onshore vs offshore, residuals
against ComCat ML for each catalog, and Mc/b. Panels e-f test the magnitudes against
ComCat moment-tensor Mw (data/focal/comcat_mt_matched.csv, from phase18 -- run it first).
Writes routeA_vs_kpos_comparison.png.

Usage (amplitude env, run from 4_relocation/magnitude):
    python phase17_routeA_vs_kpos.py [KPOS_DIR]   # default ../../data/magnitude
"""
import pandas as pd, numpy as np, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import use, FULL, panel, save, OURS, REF, INK, MUTED, CLASS_COLORS
from phase18_moment_tensor_match import hutton_boore, RMAX
use()
D="../../data/magnitude"; O=sys.argv[1] if len(sys.argv)>1 else D
A=pd.read_csv(f"{D}/cascadia_catalog_ML_routeA.csv"); K=pd.read_csv(f"{O}/cascadia_catalog_ML_kpos.csv")
m=A.merge(K,on="event_id",suffixes=("_A","_K")); d=m.ML_A-m.ML_K
print(f"events: routeA {len(A):,}  kpos {len(K):,}  both {len(m):,}")
p=np.polyfit(m.ML_K,m.ML_A,1)
print(f"ML_A - ML_K: mean {d.mean():+.3f} median {d.median():+.3f} std {d.std():.3f}; fit ML_A = {p[0]:.3f}*ML_K {p[1]:+.3f}; corr {np.corrcoef(m.ML_A,m.ML_K)[0,1]:.3f}")
print("median sta-mag scatter: A %.3f  K %.3f"%(A.M_sta_std.median(),K.M_sta_std.median()))
bins=np.arange(-1,5,0.5); m["bin"]=pd.cut(m.ML_K,bins)
print("\nby kpos ML bin: n, median(A-K), IQR"); 
for b,g in m.groupby("bin",observed=True):
    dd=g.ML_A-g.ML_K; print(f"  {str(b):12s} {len(g):6d} {dd.median():+.2f} {dd.quantile(.75)-dd.quantile(.25):.2f}")
m["off"]=m.evlo_A< -124.6
for o,g in m.groupby("off"): print(("offshore" if o else "onshore ")+f" n={len(g):,} median(A-K) {(g.ML_A-g.ML_K).median():+.3f}")
# vs ComCat anchors: residual (ours - comcat) by comcat ML
for tag,cat,anc in [("A",A,f"{D}/route_b_ml_anchors_routeA.csv"),("K",K,f"{O}/route_b_ml_anchors_kpos.csv")]:
    a=pd.read_csv(anc)[["event_id","ml"]].merge(cat[["event_id","ML"]],on="event_id"); r=a.ML-a.ml
    s=np.polyfit(a.ml,a.ML,1)
    print(f"\n{tag} vs ComCat ML: n={len(a)} resid std {r.std():.3f}; slope ours~comcat {s[0]:.3f}")
    for lo in [1.5,2,2.5,3,3.5]:
        g=r[(a.ml>=lo)&(a.ml<lo+.5)]; print(f"   comcat {lo:.1f}-{lo+.5:.1f}: n={len(g):4d} median resid {g.median():+.2f}")
def mc_b(x):
    x=np.round(x,1); h=x.value_counts().sort_index(); mc=h.idxmax()+0.2; y=x[x>=mc]
    return mc, np.log10(np.e)/(y.mean()-(mc-0.05)), len(y)
for tag,c in [("A",A),("K",K)]:
    mc,b,n=mc_b(c.ML); print(f"{tag}: Mc(maxc+0.2)={mc:.1f}  b={b:.2f}  n>=Mc={n:,}  min {c.ML.min():.2f} p1 {c.ML.quantile(.01):.2f} median {c.ML.median():.2f}")
RA,RK="response removed","raw counts"; COLS={RA:OURS,RK:REF}
fig,ax=plt.subplots(3,2,figsize=(FULL,7.8)); ax=ax.ravel()
hb=ax[0].hexbin(m.ML_K,m.ML_A,gridsize=90,bins="log",cmap="cividis",mincnt=1,linewidths=0,rasterized=True); l=[-1.5,4.7]; ax[0].plot(l,l,"--",c=INK,lw=0.6)
ax[0].set(xlabel="$M_L$, raw counts",ylabel="$M_L$, response removed"); ax[0].set_aspect("equal")
cbar=fig.colorbar(hb,ax=ax[0],pad=0.02,fraction=0.05,aspect=18); cbar.set_label("Events per bin"); cbar.outline.set_linewidth(0.5)
m["dA"]=m.ML_A-m.ML_K; gb=m.groupby("bin",observed=True)["dA"]; g={q:gb.quantile(q) for q in (.25,.5,.75)}; c=[i.mid for i in g[.5].index]; n=gb.size().values; k=n>=100; c=np.array(c)[k]; g={q:v.values[k] for q,v in g.items()}
ax[1].fill_between(c,g[.25],g[.75],color=REF,alpha=.25,lw=0,label="interquartile range"); ax[1].plot(c,g[.5],"o-",color=REF,ms=3,label="median"); ax[1].axhline(0,c=MUTED,lw=.6)
ax[1].set(xlabel="$M_L$, raw counts",ylabel="$M_L$ difference, response removed $-$ raw"); ax[1].legend(loc="lower right")
b=np.arange(-1.5,4.8,0.1)
for tag,cc in [(RA,A),(RK,K)]:
    h,_=np.histogram(cc.ML,b); ax[2].semilogy(b[:-1]+.05,np.cumsum(h[::-1])[::-1],color=COLS[tag],label=tag)
ax[2].set(xlabel="Local magnitude $M_L$",ylabel="Number of events $\\geq M_L$"); ax[2].legend(loc="upper right")
cb=np.arange(1.5,4.01,0.5)
for (tag,cat,anc),off in zip([(RA,A,f"{D}/route_b_ml_anchors_routeA.csv"),(RK,K,f"{O}/route_b_ml_anchors_kpos.csv")],(-.04,.04)):
    a_=pd.read_csv(anc)[["event_id","ml"]].merge(cat[["event_id","ML"]],on="event_id"); r=a_.ML-a_.ml
    gb=r.groupby(pd.cut(a_.ml,cb,right=False),observed=True); x=np.array([i.mid for i in gb.median().index])+off
    ax[3].errorbar(x,gb.median(),yerr=[gb.median()-gb.quantile(.25),gb.quantile(.75)-gb.median()],fmt="o-",ms=3,lw=1,elinewidth=0.8,capsize=2,color=COLS[tag],label=f"{tag}, n={len(a_):,}")
ax[3].axhline(0,c=MUTED,lw=.6); ax[3].set(xlabel="ComCat $M_L$",ylabel="$M_L$ residual, this study $-$ ComCat"); ax[3].legend(loc="lower left")
# (e-f) against ComCat moment-tensor Mw
T=pd.read_csv("../../data/focal/comcat_mt_matched.csv")
ds=pd.read_csv(f"{D}/amp_distance_dataset_routeA.csv",usecols=["event_id","phase","dist_hypo_km","log10A"])
ds["m"]=hutton_boore(ds.log10A,ds.dist_hypo_km)
an=pd.read_csv(f"{D}/route_b_ml_anchors_routeA.csv")[["event_id","ml"]]
nr=ds[ds.dist_hypo_km<=RMAX].groupby("event_id").m.agg(["median","size"]); nr=nr[nr["size"]>=3]["median"]
aa=an.join(nr.rename("M"),on="event_id",how="inner"); T["test"]=T.event_id.map(nr)+(aa.ml-aa.M).median()
TEST=CLASS_COLORS["oceanic"]
for col,lab,c,mk in [("ML_routeA","catalog $M_L$",OURS,"o"),("MW_routeA","catalog $M_W$ (calibrated)",MUTED,"s"),("test",f"test $M_L$ (Hutton–Boore, $r\\leq${RMAX:.0f} km)",TEST,"^")]:
    ok=T[col].notna(); ax[4].scatter(T.Mw_mt[ok],T[col][ok]-T.Mw_mt[ok],s=9,marker=mk,facecolor="none",edgecolor=c,lw=0.7,label=f"{lab}, n={ok.sum()}")
ax[4].axhline(0,c=MUTED,lw=.6); ax[4].set(ylim=(-2.5,2.6),xlabel="ComCat moment-tensor $M_w$",ylabel="Magnitude $-$ $M_w$"); ax[4].legend(loc="upper right",frameon=True,facecolor="white",edgecolor="none",framealpha=0.9)
db=np.array([0,30,60,100,150,200,300,400,500,700,1000])
for name,ref,c in [("ComCat $M_L\\geq$2.5 anchors",an[an.ml>=2.5].rename(columns={"ml":"ref"}),INK),
                   ("tensor $M_w<$4.5",T[T.Mw_mt<4.5][["event_id","Mw_mt"]].rename(columns={"Mw_mt":"ref"}),REF),
                   ("tensor $M_w\\geq$4.5",T[T.Mw_mt>=4.5][["event_id","Mw_mt"]].rename(columns={"Mw_mt":"ref"}),OURS)]:
    x=ds[ds.phase=="S"].merge(ref,on="event_id"); r=x.m-x.ref; gb=r.groupby(pd.cut(x.dist_hypo_km,db),observed=True)
    k=gb.size()>=15; xc=np.array([np.sqrt(max(i.left,10)*i.right) for i in gb.median().index])[k.values]   # geometric bin centre
    q=[gb.quantile(v)[k] for v in (.25,.5,.75)]
    ax[5].fill_between(xc,q[0],q[2],color=c,alpha=.18,lw=0); ax[5].plot(xc,q[1],"o-",color=c,ms=3,lw=1,label=f"{name}, {x.event_id.nunique()} events")
ax[5].axhline(0,c=MUTED,lw=.6); ax[5].axvline(RMAX,c=MUTED,lw=.6,ls=":"); ax[5].set_xscale("log")
ax[5].set(xlabel="Hypocentral distance (km)",ylabel="S station $M_L$ $-$ reference magnitude"); ax[5].legend(loc="lower left",frameon=True,facecolor="white",edgecolor="none",framealpha=0.9)
for i,x in enumerate(ax): panel(x,"abcdef"[i],x=(-0.24,-0.13,-0.11,-0.13,-0.11,-0.13)[i])
fig.tight_layout(h_pad=1.0,w_pad=1.5); save(fig,f"{D}/routeA_vs_kpos_comparison.png"); print("wrote fig")
