"""Read the PHASE lines of all per-event NLLoc hyp files of a v4-NLL3D run into one table (cached as parquet).

Per-event files: <reloc_v5>/runs/<run>/<tile>[_cNN]/loc.2*.grid0.loc.hyp (NLL_FORMAT_VER_2). PHASE line tokens:
0 station, 4 phase, 6 date, 7 hhmm, 8 sec (observed pick time), 10 pick error (s), 16 TTpred, 17 residual,
18 EDT weight, 22 epicentral distance (km), 27 Tcorr (station delay applied), 28 TTerr (LOCGAU2).
NLLoc residual = (pick - OT) - Tcorr - TTpred. Only LOCATED events are kept.

Usage (also imported by b1_select_events.py): python b1_hyp_phases.py RUN  -> <base>/b1_dataset/cache/phases_RUN.parquet
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from manifest import load_params  # noqa: E402

COLS = ["ID", "dir", "sta", "phase", "pick_time", "err", "ttpred", "res", "w", "dist", "tcorr"]


def read_file(path):
    out, eid, keep, inph = [], None, False, False
    d = path.parent.name
    for line in open(path):
        if line.startswith("NLLOC "):
            keep = '"LOCATED"' in line
        elif line.startswith("PUBLIC_ID"):
            eid = int(line.split()[1].split("_")[1])
        elif line.startswith("PHASE ID"):
            inph = True
        elif line.startswith("END_PHASE"):
            inph = False
        elif inph and keep:
            t = line.split()
            if len(t) >= 29 and t[4] in ("P", "S"):
                pt = f"{t[6][:4]}-{t[6][4:6]}-{t[6][6:]}T{t[7][:2]}:{t[7][2:]}:00"
                out.append((eid, d, t[0], t[4], pt + f"+{float(t[8]):.4f}", float(t[10]), float(t[16]),
                            float(t[17]), float(t[18]), float(t[22]), float(t[27])))
    return out


def load_phases(run, refresh=False):
    P = load_params()
    cache = Path(P["paths"]["base"]) / "b1_dataset/cache" / f"phases_{run}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rd = Path(P["paths"]["reloc_v5"]) / "runs" / run
    files = sorted(rd.glob("*/loc.2*.grid0.loc.hyp"))
    with ProcessPoolExecutor(64) as ex:
        rows = [r for chunk in ex.map(read_file, files, chunksize=200) for r in chunk]
    ph = pd.DataFrame(rows, columns=COLS)
    # pick time: minute start + seconds (seconds may be written >= 60 by NLLoc)
    base, sec = ph.pick_time.str.split("+", n=1, expand=True).T.values
    ph["pick_time"] = pd.to_datetime(base) + pd.to_timedelta(sec.astype(float), unit="s")
    for c in ("sta", "phase", "dir"):
        ph[c] = ph[c].astype("category")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ph.to_parquet(cache)
    print(f"{run}: {len(files)} files, {len(ph)} phases, {ph.ID.nunique()} events -> {cache}")
    return ph


if __name__ == "__main__":
    load_phases(sys.argv[1], refresh=True)
