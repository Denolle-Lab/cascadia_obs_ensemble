import sys
from pathlib import Path
import numpy as np, pandas as pd
cid = sys.argv[1]
B = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/s_mismatch") / cid
def res(path, keep=None):
    r = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=["STA", "DT", "C1", "C2", "IDX", "QUAL", "RES", "WT", "OFFS"])
    r = r[(r.C1 != r.C2) & r.IDX.isin([3, 4])]
    r["k"] = r.STA + ":" + r.C1.astype(str) + ":" + r.C2.astype(str) + ":" + r.IDX.astype(str)
    if keep is not None:
        r = r[r.k.isin(keep)]
    rr = r.RES / 1000.0
    r["r"] = rr - rr.groupby([r.C1, r.C2]).transform("median")
    return r
base = res(B / "b4" / "tomoDD-SE.res.ini")
rows = [("B4 start", base)]
for n in ("prod", "s_eq", "norew", "s_eq_norew", "p_only", "s_only"):
    f = B / "var" / n / "eval" / "tomoDD-SE.res.ini"
    if f.exists():
        rows.append((n, res(f)))
common = set.intersection(*[set(r.k) for _, r in rows])
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
print(f"== {cid}: dt residual rms (s) on {len(common):,} common rows, per-pair median removed (tomoDD forward)")
for n, r in rows:
    r = r[r.k.isin(common)]
    rr = r.RES / 1000.0
    r = r.assign(r=rr - rr.groupby([r.C1, r.C2]).transform("median"))
    p, s = r[r.IDX == 3].r, r[r.IDX == 4].r
    print(f"  {n:12s} P {rms(p):.3f}  S {rms(s):.3f}   P MAD {np.median(np.abs(p)):.3f}  S MAD {np.median(np.abs(s)):.3f}")
