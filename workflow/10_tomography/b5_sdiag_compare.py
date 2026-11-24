"""Compare tomoDD and B4-grid calculated differential times on the same dt rows at B4 and B5 positions."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/mdenolle/wt_b5_full/workflow/10_tomography")
from b2_fwd_common import eft_depth, lambert_km
import b5_relocate as BR

cid = sys.argv[1]
W = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/s_mismatch") / cid
SRC = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/clusters") / cid
P = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/prep")
st = pd.read_csv(P / "stations.csv").set_index("code")
ev = pd.read_parquet(P / "events.parquet").set_index("ID")
rel = pd.read_csv(SRC / "reloc.csv").set_index("ID")
G = BR.Grids("/wd1/mdenolle_data/reloc_v5/time_b4_v0.3.1")

def read_res(tag):
    r = pd.read_csv(W / tag / "tomoDD-SE.res.ini", sep=r"\s+", skiprows=1, header=None,
                    names=["STA", "DT", "C1", "C2", "IDX", "QUAL", "RES", "WT", "OFFS"])
    r = r[r.C1 != r.C2].copy()
    r["ph"] = np.where(r.IDX == 3, "P", np.where(r.IDX == 4, "S", "?"))
    r["calc_t"] = r.DT - r.RES / 1000.0
    return r

def nll_times(ids, sta, ph, pos):
    x, y, zf = pos
    out = np.full(len(ids), np.nan)
    for (s, p), g in pd.DataFrame({"s": sta, "p": ph}).groupby(["s", "p"]).groups.items():
        g = np.asarray(list(g))
        ii = ids[g]
        out[g] = G.time(p, st.loc[s, "station"], x.loc[ii].values, y.loc[ii].values, zf.loc[ii].values)
    return out

x4, y4, z4 = ev.x_km, ev.y_km, ev.zf_km
x5, y5 = lambert_km(rel.lon.values, rel.lat.values)
x5, y5, z5 = pd.Series(x5, rel.index), pd.Series(y5, rel.index), pd.Series(eft_depth(rel.depth.values), rel.index)
rows = []
for tag, pos in (("b4", (x4, y4, z4)), ("b5", (x5, y5, z5))):
    r = read_res(tag)
    r = r[r.C1.isin(pos[0].index) & r.C2.isin(pos[0].index)]
    t1 = nll_times(r.C1.values, r.STA.values, r.ph.values, pos)
    t2 = nll_times(r.C2.values, r.STA.values, r.ph.values, pos)
    r["calc_n"] = t1 - t2
    r["d"] = r.calc_t - r.calc_n                     # tomoDD minus NLL calculated dt
    for res_col, calc in (("res_t", "calc_t"), ("res_n", "calc_n")):
        rr = r.DT - r[calc]
        r[res_col] = rr - rr.groupby([r.C1, r.C2]).transform("median")
    r["tag"] = tag
    rows.append(r)
R = pd.concat(rows)
R.to_parquet(W / "compare.parquet")
rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
mad = lambda v: float(np.nanmedian(np.abs(v - np.nanmedian(v))))
print(f"== {cid}: residual rms (s), per-pair median removed; tomoDD forward vs B4 grids")
for (tag, ph), g in R.groupby(["tag", "ph"]):
    print(f"  {tag} {ph}: n {len(g):6d}  tomoDD {rms(g.res_t):.3f}  grids {rms(g.res_n):.3f}   "
          f"calc tomoDD-grids: median {g.d.median()*1e3:+6.1f} ms, MAD {mad(g.d)*1e3:5.1f} ms, rms {rms(g.d)*1e3:6.1f} ms; "
          f"slope grids~tomoDD {np.polyfit(g.calc_t, g.calc_n, 1)[0]:.4f}")
