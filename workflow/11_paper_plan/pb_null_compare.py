#!/usr/bin/env python3
"""Claimed contrasts of the null-truth control (D-B112) on the current axes against an earlier scoring
(<PB_OUT>/null_truth_contrasts_<PB_AXES_OLD>.csv, default gmrt = D-B113 axes v1). Writes
<PB_OUT>/null_claimed_compare.csv (every contrast: claimed before and after, values) and prints the gained and lost
claims. Numbers come from the two tables only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

OLD_AX = os.environ.get("PB_AXES_OLD", "gmrt")


def main():
    k = ["seg", "var", "contrast"]
    old = pd.read_csv(P.OUT / f"null_truth_contrasts_{OLD_AX}.csv")
    new = pd.read_csv(P.OUT / "null_truth_contrasts.csv")
    cols = ["D_real", "sigma_ens", "S95", "N95", "survives_DB111", "pass_null", "claimed", "n_null"]
    m = old[k + cols].merge(new[k + cols], on=k, how="outer", suffixes=(f"_{OLD_AX}", f"_{P.AXES}"))
    co, cn = m[f"claimed_{OLD_AX}"].fillna(False).astype(bool), m[f"claimed_{P.AXES}"].fillna(False).astype(bool)
    m["change"] = ["gained" if (not a and b) else "lost" if (a and not b) else "claimed both" if a else ""
                   for a, b in zip(co, cn)]
    m.to_csv(P.OUT / "null_claimed_compare.csv", index=False, float_format="%.4g")
    P.log(f"claimed: {OLD_AX} {int(co.sum())}, {P.AXES} {int(cn.sum())}; gained {int((m.change == 'gained').sum())}, "
          f"lost {int((m.change == 'lost').sum())}")
    P.log(m[m.change.isin(["gained", "lost"])][k + [f"D_real_{OLD_AX}", f"D_real_{P.AXES}", "change"]].to_string())


if __name__ == "__main__":
    main()
