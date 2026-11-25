"""Static station terms (P and S) from an NLLoc run, as NLL LOCDELAY lines for the next iteration.

delay_new = Tcorr (delay already applied) + median residual over phases used (weight > 0) of well-located
events: LOCATED, >= 8 phases, azimuthal gap <= 270 deg, RMS <= 0.6 s. Station-phases with fewer than
MIN_N residuals get no term. Delays are subtracted from observed times by NLLoc.

Usage: python a2_station_terms.py RUN_NAME  ->  runs/RUN_NAME/station_terms.{csv,in}
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/wd1/mdenolle_data/reloc_v5")
MIN_N = 10


def read_phases(path):
    out, eid, keep, inph = [], None, False, False
    for line in open(path):
        if line.startswith("PUBLIC_ID"):
            eid = int(line.split()[1].split("_")[1])
        elif line.startswith("NLLOC "):
            keep = '"LOCATED"' in line
        elif line.startswith("PHASE ID"):
            inph = True
        elif line.startswith("END_PHASE"):
            inph = False
        elif inph and keep:
            t = line.split()
            if len(t) >= 29 and t[4] in ("P", "S"):
                out.append((eid, t[0], t[4], float(t[17]), float(t[18]), float(t[22]), float(t[27])))
    return out


def main(run):
    rd = B / "runs" / run
    loc = pd.read_csv(rd / "locations.csv")
    good = set(loc.query("status == 'LOCATED' and nphs >= 8 and gap <= 270 and rms <= 0.6").ID)
    files = sorted(rd.glob("*/loc.2*.grid0.loc.hyp"))
    with ProcessPoolExecutor(64) as ex:
        rows = [r for chunk in ex.map(read_phases, files, chunksize=200) for r in chunk]
    ph = pd.DataFrame(rows, columns=["ID", "sta", "phase", "res", "w", "dist", "tcorr"])
    ph = ph[(ph.w > 0) & ph.ID.isin(good)]
    st = ph.groupby(["sta", "phase"]).agg(n=("res", "size"), med_res=("res", "median"),
                                          mad=("res", lambda s: np.median(np.abs(s - s.median()))),
                                          tcorr_old=("tcorr", "median"), dist_med=("dist", "median")).reset_index()
    st = st[st.n >= MIN_N].copy()
    st["delay"] = (st.tcorr_old + st.med_res).round(4)
    st.to_csv(rd / "station_terms.csv", index=False)
    (rd / "station_terms.in").write_text(
        "\n".join(f"LOCDELAY {r.sta} {r.phase} {r.n} {r.delay:.4f}" for r in st.itertuples()) + "\n")
    print(f"{len(good)} good events, {len(ph)} residuals, {len(st)} station-phase terms")
    print(st.groupby("phase")[["delay", "med_res", "mad"]].describe(percentiles=[.05, .5, .95]).round(3).T)


if __name__ == "__main__":
    main(sys.argv[1])
