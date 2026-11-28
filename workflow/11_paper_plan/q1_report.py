#!/usr/bin/env python3
"""Q1 (D-B115) report: /wd1/mdenolle_data/paper_plan/q1_slip_style/report_q1.html (self-contained).

Reads the tables of q1_tests.py, q1_metrics.py, q1_vpvs.py and the figures of q1_figures.py; the interpretation text
lives in q1_report_text.py (written after the results, against the pre-registered rules).
Run: python q1_report.py "time label from date"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q1_common as Q  # noqa: E402
import q1_report_text as TXT  # noqa: E402
from q1_report_lib import esc, figure, fmt, page, table  # noqa: E402

TAB = Q.BIG / "tables"
FIG = Q.BIG / "figures"


def load():
    d = {}
    d["dec"] = json.load(open(TAB / "decision.json"))
    d["dec_int"] = json.load(open(TAB / "decision_interim.json")) if (TAB / "decision_interim.json").exists() else None
    d["T"] = pd.read_csv(TAB / "tests_external.csv")
    d["Ti"] = pd.read_csv(TAB / "tests_interim.csv") if (TAB / "tests_interim.csv").exists() else None
    d["C"] = pd.read_csv(TAB / "controls_external.csv")
    d["N"] = pd.read_csv(TAB / "nulls_units.csv")
    d["S"] = pd.read_csv(TAB / "segments_main_external.csv")
    d["cells"] = pd.read_csv(TAB / "cells_main_external.csv")
    d["V"] = pd.read_csv(Q.BIG / "vpvs" / "vpvs_families.csv")
    d["info"] = json.load(open(TAB / "metrics_info.json"))
    d["m"] = {r: pd.read_csv(TAB / f"metrics_{r}.csv") for r in ["main", "mcu_count", "mcu_recluster", "v60"]
              if (TAB / f"metrics_{r}.csv").exists()}
    d["ev"] = pd.read_parquet(TAB / "events_q1_v61.parquet")
    d["cat"] = pd.read_csv(Q.CATS["v61"]["csv"], usecols=["ID", "tier", "depth"])
    d["regions"] = pd.read_csv(Q.CATS["v61"]["clu"] / "regions_zbz.csv")
    prev = Q.BIG / "runs" / "db113_axes_v1" / "tables"       # the first run (D-B113 axes v1), for the v2 note
    if prev.exists():
        d["prev"] = dict(T=pd.read_csv(prev / "tests_external.csv"), dec=json.load(open(prev / "decision_external.json")),
                         m=pd.read_csv(prev / "metrics_main.csv"), ev=pd.read_parquet(prev / "events_q1_v61.parquet"),
                         cells=pd.read_csv(prev / "cells_main_external.csv"))
    return d


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    d = load()
    body = TXT.render(d, label, dict(esc=esc, figure=figure, fmt=fmt, table=table, FIG=FIG, Q=Q))
    out = Q.BIG / "report_q1.html"
    out.write_text(page("Q1 slip style", body, "What sets swarm versus mainshock-aftershock release on the offshore "
                        "Cascadia plate boundaries: pre-registered tests on catalog v6.1 (D-B115)."))
    Q.log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
