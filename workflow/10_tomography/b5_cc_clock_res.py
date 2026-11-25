"""Station clock check, part 1: NLLoc phase residuals of the v5.1 relocation (run b4_v51_iter1, read only).

Parses every per-event hyp file (loc.*.grid0.loc.hyp) of the tiles whose station list holds a given station
(default MH07.XD or MG07.XD: the tiles around Mount St Helens) and writes one row per phase: pick time (UTC),
station, phase, TTpred, Res (s, after the NLLoc station correction Tcorr), Weight, SDist, Tcorr.
A station clock offset c shifts its P and S picks alike, so Res steps by ~c (a little is absorbed by the location).

  python b5_cc_clock_res.py [--stations MH07.XD,MG07.XD] [--workers 12]
  -> /wd1/mdenolle_data/tomo_v1/cc_dt/clock/nll_residuals.parquet
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402

RUN = Path("/wd1/mdenolle_data/reloc_v5/runs/b4_v51_iter1")
OUTD = C.OUT / "clock"


def parse_tile(tile):
    rows = []
    for f in sorted(glob.glob(str(RUN / tile / "loc.*.grid0.loc.hyp"))):
        ot = None
        with open(f) as fh:
            inphase = False
            for line in fh:
                if line.startswith("GEOGRAPHIC"):
                    t = line.split()
                    ot = f"{t[2]}-{int(t[3]):02d}-{int(t[4]):02d}T{int(t[5]):02d}:{int(t[6]):02d}:{float(t[7]):09.6f}"
                elif line.startswith("PHASE "):
                    inphase = True
                elif line.startswith("END_PHASE"):
                    inphase = False
                elif inphase and ">" in line:
                    a, b = line.split(">", 1)
                    ta, tb = a.split(), b.split()
                    if len(ta) < 9 or len(tb) < 12:
                        continue
                    d, hm, sec = ta[6], ta[7], float(ta[8])
                    pick = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{hm[:2]}:{hm[2:4]}:00"
                    rows.append((tile, os.path.basename(f), ot, ta[0], ta[4], pick, sec, float(tb[0]), float(tb[1]),
                                 float(tb[2]), float(tb[6]), float(tb[11]) if len(tb) > 11 else np.nan))
    return pd.DataFrame(rows, columns=["tile", "hyp", "origin", "station", "phase", "pick_min", "pick_sec", "ttpred",
                                       "res", "weight", "sdist", "tcorr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="MH07.XD,MG07.XD")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    want = a.stations.split(",")
    tiles = []
    for f in sorted(RUN.glob("*/last.stations")):
        txt = f.read_text()
        if any(f"\n{s} " in txt for s in want):
            tiles.append(f.parent.name)
    print(f"{len(tiles)} tiles", flush=True)
    with Pool(a.workers) as p:
        parts = p.map(parse_tile, tiles)
    df = pd.concat(parts, ignore_index=True)
    df["t_pick"] = (pd.to_datetime(df.pick_min).astype("int64") / 1e9 + df.pick_sec)
    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTD / "nll_residuals.parquet", index=False)
    print(f"{len(df)} phase rows, {df.station.nunique()} stations, {df.hyp.nunique()} events", flush=True)


if __name__ == "__main__":
    main()
