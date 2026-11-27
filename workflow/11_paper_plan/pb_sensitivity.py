#!/usr/bin/env python3
"""Sensitivity of the segment event counts (tier A-C earthquakes, review P1 left out) to the thresholds of the segment
rule: R_MTJ_KM 30/40/50, D_AXIS_KM 10/15/20, LAT_GS 41.0/41.5/42.0, SLAB_IF_KM 3/5/8.
Writes data/paper_plan/plate_boundaries/segment_rule_sensitivity.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402


def main():
    base = P.load_catalog()
    sel = base.is_eq & base.abc & ~base.review_P1
    runs = [("default", {})] + [(f"{k}={v}", {k: v}) for k, vals in
                                (("R_MTJ_KM", (30.0, 50.0)), ("D_AXIS_KM", (10.0, 20.0)), ("LAT_GS", (41.0, 42.0)),
                                 ("SLAB_IF_KM", (3.0, 8.0))) for v in vals]
    out = {}
    keep = {k: getattr(P, k) for k in ("R_MTJ_KM", "D_AXIS_KM", "LAT_GS", "SLAB_IF_KM")}
    for name, ch in runs:
        for k, v in keep.items():
            setattr(P, k, v)
        for k, v in ch.items():
            setattr(P, k, v)
        c = P.assign_segments(base)
        out[name] = c[sel.values].seg.value_counts()
        P.log(name)
    for k, v in keep.items():
        setattr(P, k, v)
    t = pd.DataFrame(out).reindex(P.SEG_ORDER).fillna(0).astype(int)
    t.index.name = "seg"
    t.to_csv(P.OUT / "segment_rule_sensitivity.csv")
    P.log(t.to_string())


if __name__ == "__main__":
    main()
