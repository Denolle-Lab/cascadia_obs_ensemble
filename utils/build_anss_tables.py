#!/usr/bin/env python3
"""Build the three ANSS-style catalog tables (origins, arrivals, picks), version v3.

    origins_v3.csv   one row per relocated event (63,887). The first 22 columns are
                     the ANSS ComCat CSV format, in its order; project columns follow.
    arrivals_v3.csv  one row per event-pick association (1,004,335): the phase the
                     associator assigned, residual, distance, azimuth, and the Route A
                     amplitude measured on that pick. Links origins (origin_id) to
                     picks (pick_id).
    picks_v3.csv.gz  every ELEP pick (39,597,551), with the phase the picker assigned.

The split follows QuakeML (Pick / Arrival / Origin), so each table maps onto one
QuakeML element; column definitions are in data/DATA_DICTIONARY.md. GENIE relabels
the phase of about 3% of the associated picks, so an arrival's `phase` can differ
from its pick's `phase_hint`.

    python utils/build_anss_tables.py                 # all three -> data/catalog_v3/
    python utils/build_anss_tables.py --skip-picks    # origins + arrivals only

Environment: pixi `amplitude` (pandas + numpy). The picks table takes ~15 min and
~40 GB of memory.
"""
from __future__ import annotations

import argparse
import datetime
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "datasets_all_regions"
MAG = ROOT / "data" / "magnitude"
OUT = ROOT / "data" / "catalog_v3"

ORIGINS = SRC / "origin_2010_2015_reloc_cog_ver3_cc.csv"
ARRIVALS = SRC / "arrival_2010_2015_reloc_cog_ver3_cc.csv"
ASSOC = SRC / "assoc_2010_2015_reloc_cog_ver3_cc.csv"
QC = SRC / "origin_2010_2015_reloc_cog_ver3_cc_p_4_s_4_rms_2_5.csv"
PICKS = SRC / "all_picks_all_regions_2010_2015_ver3.csv"
CLASSIFIED = MAG / "cascadia_catalog_classified.csv"          # phase10: M, qc_pass, event_class
MAGNITUDES = MAG / "cascadia_catalog_M_routeA.csv"            # phase19: n_ML, Mw_mt, comcat_id
AMPLITUDES = ROOT / "4_relocation" / "magnitude" / "rerun_v2" / "raw_wa_amplitudes_v2.csv"
ML_DATASET = MAG / "amp_distance_dataset_routeA.csv"         # picks that entered the ML inversion

# Catalog code used for `net`, `id`, `locationSource`, `magSource`. It is a project
# code, not an ANSS-registered network code.
CATALOG = "cobs"
VERSION = "v3"
KM_PER_DEG = 111.19
LINK_TOL_S = 0.005        # arrival time vs ELEP pick time

COMCAT_COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "nst", "gap",
                  "dmin", "rms", "net", "id", "updated", "place", "type", "horizontalError",
                  "depthError", "magError", "magNst", "status", "locationSource", "magSource"]


def iso(epoch_s, decimals=3):
    """Epoch seconds -> ISO 8601 UTC strings ending in Z."""
    ticks = np.round(np.asarray(epoch_s, dtype=float) * 10**decimals).astype("int64")
    t = pd.to_datetime(ticks * 10**(9 - decimals), unit="ns")
    s = t.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return s.str[: 20 + decimals] + "Z"


def event_ids(orid):
    return CATALOG + pd.Series(orid).astype(int).map("{:06d}".format).values


def build_origins(assoc: pd.DataFrame, updated: str) -> pd.DataFrame:
    o = pd.read_csv(ORIGINS, index_col=0)
    c = pd.read_csv(CLASSIFIED)
    m = pd.read_csv(MAGNITUDES, usecols=["event_id", "n_ML", "Mw_mt", "comcat_id"])
    qc = set(pd.read_csv(QC, index_col=0).orid)
    o = (o.merge(c[["orid", "ML", "ML_unc", "M", "M_unc", "M_type", "ML_method", "h_unc_km",
                    "qc_pass", "event_class"]], on="orid", how="left", validate="1:1")
          .merge(m.rename(columns={"event_id": "orid"}), on="orid", how="left", validate="1:1"))
    if set(o.orid[o.qc_pass.astype(bool)]) != qc:
        raise SystemExit("qc_pass in the classified catalog does not match the QC file")

    per_event = assoc.groupby("orid").agg(nst=("sta", "nunique"), dmin_km=("delta", "min"))
    o = o.join(per_event, on="orid")

    mag_type = o.M_type.map({"ML": "ml", "Mw": "mw", "Mw_cal": "mw"})
    df = pd.DataFrame({
        "time": iso(o.time),
        "latitude": o.lat.round(5),
        "longitude": o.lon.round(5),
        "depth": o.depth.round(3),
        "mag": o.M.round(2),
        "magType": mag_type,
        "nst": o.nst.astype("Int64"),
        "gap": o.gap.round(1),
        "dmin": (o.dmin_km / KM_PER_DEG).round(4),
        "rms": o.rms.round(3),
        "net": CATALOG,
        "id": event_ids(o.orid),
        "updated": updated,
        "place": "",
        "type": "earthquake",
        "horizontalError": o.h_unc_km.round(2),
        "depthError": np.nan,
        "magError": o.M_unc.round(2),
        "magNst": np.where(o.M_type == "Mw", np.nan, o.n_ML).astype(float),
        "status": "automatic",
        "locationSource": CATALOG,
        "magSource": np.where(o.M_type == "Mw", "comcat", np.where(o.M.notna(), CATALOG, "")),
        # project columns
        "orid": o.orid,
        "qc_pass": o.qc_pass.astype(bool),
        "M_type": o.M_type,
        "ML": o.ML.round(3),
        "ML_unc": o.ML_unc.round(3),
        "ML_method": o.ML_method,
        "Mw_mt": o.Mw_mt,
        "comcat_id": o.comcat_id,
        "n_arrivals": o.nass,
        "n_p": o.p_picks,
        "n_s": o.s_picks,
        "event_class": o.event_class,
    })
    return df.sort_values("time").reset_index(drop=True)


def link_picks(arr: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Nearest ELEP pick of the same station within LINK_TOL_S; same phase preferred."""
    arr = arr.sort_values("time")
    picks = picks.sort_values("ts")
    cols = ["ts", "pick_id", "location", "band_inst", "label"]
    same = pd.merge_asof(arr, picks[cols + ["ns"]], left_on="time", right_on="ts",
                         left_by=["sta", "iphase"], right_by=["ns", "label"],
                         direction="nearest", tolerance=LINK_TOL_S)
    miss = same.pick_id.isna()
    anyp = pd.merge_asof(arr[miss.values], picks[cols + ["ns"]], left_on="time", right_on="ts",
                         left_by="sta", right_by="ns", direction="nearest", tolerance=LINK_TOL_S)
    same = same.set_index("arid")
    same.loc[anyp.arid.values, cols] = anyp[cols].values
    return same.reset_index()


def build_arrivals(picks: pd.DataFrame) -> pd.DataFrame:
    arr = pd.read_csv(ARRIVALS, index_col=0)
    assoc = pd.read_csv(ASSOC, index_col=0)
    a = arr.merge(assoc.drop(columns=["sta", "prob"]), on="arid", how="left", validate="1:1")
    a = link_picks(a, picks)

    amp = pd.read_csv(AMPLITUDES, usecols=["arid", "wa_amp_mm", "disp_amp_um", "snr", "n_comp",
                                           "epoch", "sensor", "reason"])
    used = set(pd.read_csv(ML_DATASET, usecols=["arid"]).arid)
    a = a.merge(amp, on="arid", how="left", validate="1:1")

    net_sta = a.sta.str.split(".", n=1, expand=True)
    df = pd.DataFrame({
        "origin_id": event_ids(a.orid),
        "arid": a.arid,
        "pick_id": pd.to_numeric(a.pick_id).astype("Int64"),
        "network": net_sta[0],
        "station": net_sta[1],
        "location": a.location.fillna(""),
        "channel": a.band_inst,
        "phase": a.iphase,
        "phase_hint": a.label,
        "time": iso(a.time, 4),
        "time_residual": a.timeres,
        "distance": (a.delta / KM_PER_DEG).round(4),
        "distance_km": a.delta.round(3),
        "azimuth": a.esaz.round(2),
        "backazimuth": a.seaz.round(2),
        "station_latitude": a.slatitude,
        "station_longitude": a.slongitude,
        "station_elevation": a.selevation,
        "assoc_prob": a.prob,
        "wa_amp_mm": a.wa_amp_mm,
        "disp_amp_um": a.disp_amp_um,
        "amp_snr": a.snr.round(2),
        "amp_n_comp": a.n_comp.astype("Int64"),
        "amp_station_epoch": a.epoch,
        "amp_sensor": a.sensor,
        "amp_status": a.reason,
        "amp_in_ml": a.arid.isin(used),
        "orid": a.orid,
    })
    return df.sort_values(["origin_id", "time"]).reset_index(drop=True), assoc


def seed_location(loc: pd.Series) -> pd.Series:
    """Location codes read as numbers ('0.0', '2.0') back to SEED form ('00', '02')."""
    loc = loc.fillna("")
    num = loc.str.fullmatch(r"\d+\.0")
    return loc.where(~num, loc[num].str[:-2].str.zfill(2))


def read_picks() -> pd.DataFrame:
    p = pd.read_csv(PICKS, usecols=["network", "station", "location", "band_inst", "label",
                                    "trigger_onset", "pick_time", "trigger_offset", "max_prob",
                                    "thresh_prob", "pick_id"],
                    dtype={"network": str, "station": str, "location": str, "band_inst": str})
    p["location"] = seed_location(p.location)
    p["ns"] = p.network + "." + p.station
    p["ts"] = pd.to_datetime(p.pick_time.str.rstrip("Z"), format="ISO8601").astype("int64") / 1e9
    return p


def write_picks(p: pd.DataFrame, path: Path) -> None:
    df = pd.DataFrame({
        "pick_id": p.pick_id,
        "network": p.network,
        "station": p.station,
        "location": p.location,
        "channel": p.band_inst,
        "phase_hint": p.label,
        "time": p.pick_time,
        "onset_time": p.trigger_onset.str.strip(),     # v1 rows hold a single space
        "offset_time": p.trigger_offset.str.strip(),
        "probability": p.max_prob,
        "threshold": p.thresh_prob,
        "method": "ELEP",
        # v1: first run (2011-2015, HH/BH at the native rate, no channel or probability kept);
        # v2: the region runs (100 Hz, band code and probability kept)
        "picker_run": np.where(p.band_inst.isna(), "v1", "v2"),
        "evaluation_mode": "automatic",
    }).sort_values("pick_id")
    # compress while writing (pigz if present): the plain CSV is ~6 GB
    gz = shutil.which("pigz") or shutil.which("gzip")
    with open(path, "wb") as f, subprocess.Popen([gz, "-c"], stdin=subprocess.PIPE, stdout=f) as proc:
        df.to_csv(io.TextIOWrapper(proc.stdin, encoding="utf-8", newline=""), index=False)
    if proc.returncode:
        raise SystemExit(f"{gz} failed writing {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-picks", action="store_true", help="do not write picks_v3.csv.gz")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    updated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    print("reading ELEP picks ...", flush=True)
    picks = read_picks()
    print(f"  {len(picks):,} picks")

    arrivals, assoc = build_arrivals(picks)
    n_link = arrivals.pick_id.notna().sum()
    n_relabel = (arrivals.pick_id.notna() & (arrivals.phase != arrivals.phase_hint)).sum()
    arrivals.drop(columns="orid").to_csv(args.out / f"arrivals_{VERSION}.csv", index=False)
    print(f"arrivals: {len(arrivals):,}; linked to a pick {n_link:,}; "
          f"phase relabeled by the associator {n_relabel:,}")

    origins = build_origins(assoc, updated)
    origins.to_csv(args.out / f"origins_{VERSION}.csv", index=False)
    print(f"origins: {len(origins):,}; qc_pass {origins.qc_pass.sum():,}; "
          f"with magnitude {origins.mag.notna().sum():,}")

    if not args.skip_picks:
        write_picks(picks, args.out / f"picks_{VERSION}.csv.gz")
        print(f"picks: {len(picks):,}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
