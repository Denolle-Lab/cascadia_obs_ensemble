"""D-B028 P-6: OBS S statics recomputed with the P-1 reference times (D-B020 rule unchanged).

Static = median no-term S residual (observed iter1 travel time minus P-1 reference time) of the v4-NLL3D iter1
S picks (weight > 0) within 60 km (n >= 10); else within 100 km (n >= 10); else none. Observed travel time =
pick time - iter1 origin time, at the iter1 location (the old statics used the same picks with the production
NLL prediction: res + tcorr of the iter1 phase file).

Input:  <runs>/<region>/p1_reference/obs_S_iter1_p1.parquet (b2_fwd_p1.py), old statics
        data/tomo_v1/obs_check/obs_near_station_S_terms.csv
Output: the region's `forward.statics_file` (default data/tomo_v1/obs_check/obs_near_station_S_terms_p1.csv), with
        the old values side by side; only the region's OBS stations (other OBS: rerun per region).

  python b2_fwd_statics.py mendocino_gorda_v2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manifest import REPO, load_params, write_manifest

P = load_params()
T = P["tomodd"]
OLD = REPO / "data/tomo_v1/obs_check/obs_near_station_S_terms.csv"


def main():
    region = sys.argv[1]
    F = T["regions"][region]["forward"]
    src = Path(T["runs"]) / F["p1_reference"] / "obs_S_iter1_p1.parquet"
    ph = pd.read_parquet(src)
    ph["res0_old"] = ph.res + ph.tcorr
    ph["res0_new"] = ph.obs_tt - ph.t_p1
    chk = (ph.obs_tt - ph.ttpred - ph.res0_old)
    old = pd.read_csv(OLD).set_index("station")
    stations = sorted(ph.sta.unique())
    o = pd.DataFrame(index=pd.Index(stations, name="station"))
    for D in (60, 100):
        q = ph[ph.dist <= D]
        g = q.groupby("sta").res0_new.agg(["median", "size"])
        o[f"near{D}"] = g["median"].where(g["size"] >= 10)
        o[f"n{D}"] = g["size"].reindex(o.index).fillna(0).astype(int)
        go = q.groupby("sta").res0_old.median()
        o[f"near{D}_oldpred"] = go.where(g["size"] >= 10)
    o["S_static_proposed"] = o.near60.fillna(o.near100)
    o["static_source"] = np.where(o.near60.notna(), "near60", np.where(o.near100.notna(), "near100", "none"))
    o["S_static_old"] = old.S_static_proposed.reindex(o.index)
    o["static_source_old"] = old.static_source.reindex(o.index)
    o["S_static_old_recomputed"] = o.near60_oldpred.fillna(o.near100_oldpred)
    o["change"] = o.S_static_proposed - o.S_static_old
    # P-1 minus production prediction over the picks used for each static
    ph["p1_minus_prod"] = ph.t_p1 - ph.ttpred
    use = np.where(o.static_source == "near60", 60, 100)
    o["med_p1_minus_prod"] = [ph[(ph.sta == s) & (ph.dist <= d)].p1_minus_prod.median() for s, d in zip(o.index, use)]
    out = REPO / F["statics_file"]
    o.round(4).reset_index().to_csv(out, index=False)
    has = o.S_static_proposed.notna() & o.S_static_old.notna()
    counts = dict(stations=len(o), with_static=int(o.S_static_proposed.notna().sum()),
                  with_static_old=int(o.S_static_old.notna().sum()),
                  check_obs_minus_ttpred_minus_res0_maxabs=round(float(chk.abs().max()), 4),
                  old_recomputed_vs_file_maxabs=round(float((o.S_static_old_recomputed - o.S_static_old).abs().max()), 4),
                  change_median=round(float(o.change[has].median()), 3),
                  change_p5_p95=[round(float(x), 3) for x in np.percentile(o.change[has], [5, 95])],
                  below_minus_0p5_new=int((o.S_static_proposed < -0.5).sum()),
                  below_minus_0p5_old=int((o.S_static_old < -0.5).sum()))
    print(counts)
    print(o.groupby("static_source").change.describe().round(3))
    write_manifest(f"b2_fwd_statics_{region}", out.parent, inputs=[src, OLD], outputs=[out], params={"forward": F},
                   counts=counts)


if __name__ == "__main__":
    main()
