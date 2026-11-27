#!/usr/bin/env python3
"""Changed metrics from the D-B113 axes: the v6.1 tables built with PB2002 only (<PB_OUT>/../tables_pb2002_axes) against
the current ones (<PB_OUT>), for the segments that gained or lost events. Writes <PB_OUT>/axes_changed_metrics.csv
(segment x metric: before, after) and <PB_OUT>/axes_changed_contrasts.csv (contrasts whose verdict or value changed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

OLD = P.OUT.parent / "tables_pb2002_axes"
MET = ["n_events", "n_depth_good", "Mc", "b", "b_lo", "b_hi", "pct_bg", "pct_MA", "pct_swarm", "pct_multiplet",
       "bsf_med", "bsf_p90", "depth_class", "frac_below_z75", "n_surfaces", "dip_wmed", "burst_all", "Mmax", "n_M4",
       "n_swarm_fam_migrating", "n_dbscan_migrating"]
FEA = ["nP_10", "nPS_10", "span_P10", "span_PS10", "sd_Vp_10", "sd_VpVs_10", "class"]


def main():
    mv = pd.read_csv(P.OUT / "axes_moved_by_segment.csv")
    segs = mv[(mv.moved_in > 0) | (mv.moved_out > 0)].seg.tolist()
    so, sn = (pd.read_csv(d / "seismicity_metrics.csv").set_index("seg") for d in (OLD, P.OUT))
    fo, fn = (pd.read_csv(d / "structure_feasibility.csv").set_index("seg") for d in (OLD, P.OUT))
    rows = []
    for s in segs:
        for k in MET:
            rows.append(dict(seg=s, metric=k, before=so.at[s, k] if k in so else None, after=sn.at[s, k]))
        for k in FEA:
            rows.append(dict(seg=s, metric=k, before=fo.at[s, k], after=fn.at[s, k]))
    t = pd.DataFrame(rows)
    t.to_csv(P.OUT / "axes_changed_metrics.csv", index=False)
    co, cn = (pd.read_csv(d / "circularity_contrasts.csv") for d in (OLD, P.OUT))
    k = ["seg", "var", "contrast"]
    m = co[k + ["D_real", "sigma_ens", "S95", "survives"]].merge(cn[k + ["D_real", "sigma_ens", "S95", "survives"]],
                                                                 on=k, suffixes=("_before", "_after"))
    ch = m[(m.survives_before != m.survives_after) | ((m.D_real_before - m.D_real_after).abs() > 1e-6)]
    ch.to_csv(P.OUT / "axes_changed_contrasts.csv", index=False, float_format="%.4g")
    P.log(t.pivot(index="metric", columns="seg", values="after").loc[MET + FEA].to_string())
    P.log(ch.to_string())


if __name__ == "__main__":
    main()
