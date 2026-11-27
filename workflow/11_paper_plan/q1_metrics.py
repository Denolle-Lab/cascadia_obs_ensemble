#!/usr/bin/env python3
"""Q1 (D-B115): seismicity-style metrics per unit for the main run and every pre-registered control run.

Units: seg (plate-boundary segments, D-B113 axes, >= 100 events), cell (0.2 degree, >= 30 events kept here; the
tests use >= 50 and the N >= 30 / N >= 100 controls), cell04 (0.4 degree, >= 50).
Runs (RESUME.md, controls):
  main          v6.1, main clustering                                          (full metrics, bootstraps)
  mcu_count     uniform Mc 2.15: only event-set events with M >= 2.15, main families
  mcu_recluster clustering redone with Mc = 2.15 in every region (q1_recluster.py output)
  mc_local      only events at or above the mc_best of their 0.1 degree Mc map cell
  ci_window     only events from 2011-10-01 to 2015-09-30
  var_<v>       the 12 clustering variants of v6.1
  v60           catalog v6.0 with its clustering and the same segment rule
Writes BIG/tables/metrics_<run>.csv (all units, column 'unit' = seg | cell | cell04), the main table also to OUT.
Run: python q1_metrics.py [--runs main,mcu_count,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1_common as Q  # noqa: E402

TAB = Q.BIG / "tables"
UNITS = [("seg", Q.NMIN_SEG), ("cell", 30), ("cell04", Q.NMIN_CELL)]


def run_units(e, fam, mcreg, full, boot, mc_count=None):
    out = []
    for unit, nmin in UNITS:
        m = Q.unit_metrics(e, fam, unit, mcreg, nmin, boot=boot, full=full, mc_count=mc_count,
                           rng=np.random.default_rng(Q.SEED))
        m = m.rename(columns={unit: "unit_id"})
        m.insert(0, "unit", unit)
        out.append(m)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="main,mcu_count,mc_local,ci_window,variants,v60,mcu_recluster")
    a = ap.parse_args()
    runs = a.runs.split(",")
    if "variants" in runs:
        runs = [r for r in runs if r != "variants"] + [f"var_{v}" for v in Q.VARIANTS]
    TAB.mkdir(parents=True, exist_ok=True)
    Q.OUT.mkdir(parents=True, exist_ok=True)
    cat = Q.load_catalog("v61")
    cat["mc_map"] = Q.mc_map_values(cat.lon.values, cat.lat.values, "v61")
    mcreg = Q.regional_mc("v61")
    zmain = Q.load_zbz("v61")
    info = {}
    for run in runs:
        t0 = time.time()
        if run == "main":
            e, f = Q.event_table("v61", cat=cat, zbz=zmain)
            m = run_units(e, f, mcreg, full=True, boot=True)
            seg_counts = e[e.event_set].seg.value_counts().to_dict()
            info["main"] = dict(n_event_set=int(e.event_set.sum()), seg_counts=seg_counts,
                                segments_axes=Q.P.AXES, axes_file=str(Q.P.AXES_FILE),
                                axes_file_md5=Q.md5(Q.P.AXES_FILE) if Q.P.AXES_FILE.exists() else None,
                                catalog_md5=Q.md5(Q.CATS["v61"]["csv"]))
            e[["ID", "seg", "seg_type", "cell", "cell04", "event_set", "cls", "family_id", "offshore", "lon", "lat",
               "M", "mc_map", "t"]].to_parquet(TAB / "events_q1_v61.parquet", index=False)
        elif run == "mcu_count":
            e, f = Q.event_table("v61", cat=cat, zbz=zmain)
            m = run_units(e, f, mcreg, full=False, boot=False, mc_count=Q.MC_U)
        elif run == "mc_local":
            e, f = Q.event_table("v61", cat=cat, zbz=zmain)
            e["event_set"] &= e.M >= e.mc_map - 1e-9
            m = run_units(e, f, mcreg, full=False, boot=False)
        elif run == "ci_window":
            e, f = Q.event_table("v61", cat=cat, zbz=zmain)
            t0w, t1w = (pd.Timestamp(x, tz="UTC") for x in Q.CI_WINDOW)
            e["event_set"] &= (e.t >= t0w) & (e.t <= t1w + pd.Timedelta(days=1))
            m = run_units(e, f, mcreg, full=False, boot=False)
        elif run.startswith("var_"):
            v = run[4:]
            e, f = Q.event_table("v61", variant=v, cat=cat)
            m = run_units(e, f, mcreg, full=False, boot=False)
        elif run == "v60":
            c0 = Q.load_catalog("v60")
            c0["mc_map"] = Q.mc_map_values(c0.lon.values, c0.lat.values, "v60")
            e, f = Q.event_table("v60", cat=c0)
            m = run_units(e, f, Q.regional_mc("v60"), full=True, boot=False)
            e[["ID", "seg", "cell", "event_set", "cls"]].to_parquet(TAB / "events_q1_v60.parquet", index=False)
        elif run == "mcu_recluster":
            p = Q.BIG / "nulls" / "recluster_mcu"
            if not (p / "events.csv.gz").exists():
                Q.log("mcu_recluster: run q1_recluster.py --steps mcu first; skipped")
                continue
            z = Q.normalise_zbz(pd.read_csv(p / "events.csv.gz", low_memory=False))
            f = pd.read_csv(p / "families.csv", low_memory=False)
            e, f = Q.event_table("v61", cat=cat, zbz=(z, f))
            m = run_units(e, f, mcreg, full=False, boot=False)
        else:
            raise SystemExit(f"unknown run {run}")
        m.insert(0, "run", run)
        m.to_csv(TAB / f"metrics_{run}.csv", index=False, float_format="%.5g")
        if run == "main":
            m.to_csv(Q.OUT / "metrics_main.csv", index=False, float_format="%.5g")
        s = m[m.unit == "seg"].set_index("unit_id")
        Q.log(f"{run}: {time.time() - t0:.0f} s; cells(>=50) {int(((m.unit == 'cell') & (m.n >= 50)).sum())}; "
              f"seg swarm share " + ", ".join(f"{k} {v:.1f}" for k, v in s.swarm_share.items()))
    if info:
        json.dump(info, open(TAB / "metrics_info.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
