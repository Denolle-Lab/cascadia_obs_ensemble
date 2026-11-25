"""Static station terms (P and S) from an NLLoc run, as NLL LOCDELAY lines for the next iteration.

delay_new = Tcorr (delay already applied) + median residual over phases used (weight > 0) of well-located
events: LOCATED, >= 8 phases, azimuthal gap <= 270 deg, RMS <= 0.6 s. Station-phases with fewer than
MIN_N residuals get no term. Delays are subtracted from observed times by NLLoc.

Only arrivals served by the fine grids enter (v5.1): with a2_run_nlloc.py --time-wide, arrivals beyond the fine grids'
reach are served by the 2 km wide grids, whose times differ from the fine grids by a per station-phase offset
(round-2 verifier: -0.36 to +0.62 s at the hypocentres, 91 of 222 station-phases beyond +-0.1 s), which a term from
mixed residuals would carry into the fine-grid arrivals. Which grid served an arrival follows the runner's header rule
(a2_run_nlloc.serving_root: the first LOCFILES root of the workdir's nlloc.in whose grid contains its LOCGRID box;
the runner's arrival accounting checks this rule against NLLoc's counts for every event). For runs without wide grids
every used arrival is a fine-grid arrival and the terms are unchanged. Deep-pass runs (no fine-grid arrival) are
refused.

Usage: python a2_station_terms.py RUN [--out DIR] [--nproc 64]
  RUN: run name under reloc_v5/runs or an absolute run folder (needs locations.csv of a2_parse_hyp.py)
  -> <out, default the run folder>/station_terms.{csv,in}
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_run_nlloc import read_control, run_dir, serving_root  # noqa: E402

MIN_N = 10


def read_phases(path):
    """phase rows of LOCATED events of one hyp file: (ID, station, phase, residual, weight, distance, Tcorr, root),
    root = index of the LOCFILES root that served the arrival (0 fine, 1 wide; header rule)."""
    box, roots = read_control((path.parent / "nlloc.in").read_text())
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
                out.append((eid, t[0], t[4], float(t[17]), float(t[18]), float(t[22]), float(t[27]),
                            serving_root(box, (t[4], t[0]), roots)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out", help="output folder (default: the run folder)")
    ap.add_argument("--nproc", type=int, default=64)
    a = ap.parse_args()
    rd = run_dir(a.run)
    cfg = rd / "run_config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("pass") == "deep":
        sys.exit(f"REFUSED: {rd} is a deep-pass run (no fine-grid arrivals); station terms come from the main pass")
    out = Path(a.out) if a.out else rd
    out.mkdir(parents=True, exist_ok=True)
    loc = pd.read_csv(rd / "locations.csv")
    good = set(loc.query("status == 'LOCATED' and nphs >= 8 and gap <= 270 and rms <= 0.6").ID)
    files = sorted(rd.glob("*/loc.2*.grid0.loc.hyp"))
    with ProcessPoolExecutor(a.nproc) as ex:
        rows = [r for chunk in ex.map(read_phases, files, chunksize=200) for r in chunk]
    ph = pd.DataFrame(rows, columns=["ID", "sta", "phase", "res", "w", "dist", "tcorr", "root"])
    ph = ph[(ph.w > 0) & ph.ID.isin(good)]
    if ph.root.isna().any():
        sys.exit(f"ERROR: {int(ph.root.isna().sum())} weighted arrivals have no grid containing the box by the header "
                 "rule (the runner's arrival accounting would have failed)")
    n_wide = int((ph.root != 0).sum())
    ph = ph[ph.root == 0]
    st = ph.groupby(["sta", "phase"]).agg(n=("res", "size"), med_res=("res", "median"),
                                          mad=("res", lambda s: np.median(np.abs(s - s.median()))),
                                          tcorr_old=("tcorr", "median"), dist_med=("dist", "median")).reset_index()
    st = st[st.n >= MIN_N].copy()
    st["delay"] = (st.tcorr_old + st.med_res).round(4)
    st.to_csv(out / "station_terms.csv", index=False)
    (out / "station_terms.in").write_text(
        "\n".join(f"LOCDELAY {r.sta} {r.phase} {r.n} {r.delay:.4f}" for r in st.itertuples()) + "\n")
    print(f"{len(good)} good events, {len(ph)} fine-grid residuals ({n_wide} wide-grid residuals left out), "
          f"{len(st)} station-phase terms")
    print(st.groupby("phase")[["delay", "med_res", "mad"]].describe(percentiles=[.05, .5, .95]).round(3).T)


if __name__ == "__main__":
    main()
