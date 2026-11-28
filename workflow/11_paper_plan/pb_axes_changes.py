#!/usr/bin/env python3
"""Changed metrics from an axes change: the tables built with the previous axes (<PB_OUT>/../tables_<PB_AXES_OLD>_axes,
default tables_gmrt_axes = D-B113 axes v1) against the current ones (<PB_OUT>, default axes v2, D-B117), for the
segments that gained or lost events. Writes <PB_OUT>/axes_changed_metrics.csv (segment x metric: before, after),
<PB_OUT>/axes_changed_contrasts.csv (contrasts whose verdict or value changed) and <PB_OUT>/axes_headline_changes.csv
(every segment: b, Mc, swarm share and the 90th percentile of the depth below the surface before and after, and
whether the new value lies outside the previous 95 % bootstrap interval)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

import os  # noqa: E402

OLD = Path(os.environ.get("PB_OLD_TABLES", P.OUT.parent / f"tables_{os.environ.get('PB_AXES_OLD', 'gmrt')}_axes"))
HEAD = [("b", "b_lo", "b_hi"), ("Mc", "Mc_lo", "Mc_hi"), ("pct_swarm", "pct_swarm_lo", "pct_swarm_hi"),
        ("bsf_p90", "bsf_p90_lo", "bsf_p90_hi")]
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
    hr = []
    for s in sn.index:
        for k, lo, hi in HEAD:
            b0, b1 = so.at[s, k] if s in so.index else float("nan"), sn.at[s, k]
            l0, h0 = (so.at[s, lo], so.at[s, hi]) if s in so.index else (float("nan"), float("nan"))
            l1, h1 = sn.at[s, lo], sn.at[s, hi]
            out = bool(pd.notna(b1) and pd.notna(l0) and pd.notna(h0) and not (l0 <= b1 <= h0))
            hr.append(dict(seg=s, metric=k, before=b0, before_lo=l0, before_hi=h0, after=b1, after_lo=l1, after_hi=h1,
                           n_before=so.at[s, "n_events"] if s in so.index else None, n_after=sn.at[s, "n_events"],
                           outside_previous_ci=out))
    h = pd.DataFrame(hr)
    h.to_csv(P.OUT / "axes_headline_changes.csv", index=False, float_format="%.4g")
    P.log(t.pivot(index="metric", columns="seg", values="after").loc[MET + FEA].to_string())
    P.log(ch.to_string())
    P.log(h[h.outside_previous_ci].to_string())


if __name__ == "__main__":
    main()
