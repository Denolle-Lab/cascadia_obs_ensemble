"""Step 5b (post-hoc diagnostic, NOT part of the pre-registered rule; changes no flag).

Waveform-only sweep: events in scope with no pick in the pulse windows, 3 or more picks on signal channels and a
P-pick span on those channels of 1 s or less (common-mode look), up to 4 picks each (P first, one per station):
the same per-pick edge test as the rule. An event is listed as "waveform_only" when 2 or more picks at 2 or more
stations match and they are half or more of the picks read. These are artificial pulses outside the scheduled and
scanned windows (irregular calibrations); they are listed for the PI's review (v6.2), not flagged in v6.1.
Output: WORK/sweep_rows.csv (cache), OUT/msh_analog_waveform_only.csv
"""
import csv
import multiprocessing as mp
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import pandas as pd  # noqa: E402

import maf_common as C  # noqa: E402
from maf_score import WF_COLS, wf_one  # noqa: E402

SPAN_MAX = 1.0
N_PER = 4


def main():
    import json
    ev = pd.read_csv(C.WORK / "events_scored.csv", low_memory=False)
    S = set(json.load(open(C.WORK / "signal_stations.json"))["signal_stations"])
    cand = ev[(ev.n_in == 0) & (ev.n_S >= 3) & (ev.S_P_span_s <= SPAN_MAX)]
    p = C.picks()
    p = p[p.ID.isin(cand.ID) & p.station.isin(S)].copy()
    p["tp"] = p.t.astype("int64").values / 1e9
    p["po"] = (p.phase != "P").astype(int)
    p = p.sort_values(["ID", "po", "tp"]).drop_duplicates(["ID", "station"]).groupby("ID").head(N_PER)
    jobs = [(int(r.ID), "sweep", r.station, r.phase, float(r.tp)) for r in p.itertuples()]
    f = C.WORK / "sweep_rows.csv"
    done = set()
    if f.exists():
        old = pd.read_csv(f)
        done = set(zip(old.ID, old.station, old.phase, old.t.round(3)))
    todo = [j for j in jobs if (j[0], j[2], j[3], round(j[4], 3)) not in done]
    print(f"sweep: {len(cand)} events, {len(jobs)} picks, {len(todo)} to do", flush=True)
    if todo:
        new = not f.exists()
        with open(f, "a", newline="") as fh, mp.get_context("spawn").Pool(C.NPROC) as pool:
            w = csv.DictWriter(fh, fieldnames=WF_COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            for i, r in enumerate(pool.imap_unordered(wf_one, todo, chunksize=8)):
                w.writerow(r)
                if i % 1000 == 0:
                    fh.flush()
                    print(f"sweep {i + 1}/{len(todo)}", flush=True)
    s = pd.read_csv(f)
    s["has"] = s.status == "ok"
    g = s.groupby("ID").agg(n_wf=("has", "sum"), n_match=("matched", "sum"),
                            n_sta_match=("station", lambda x: x[s.loc[x.index, "matched"].astype(bool)].nunique()))
    g["WF"] = (g.n_match >= 2) & (g.n_sta_match >= 2) & (g.n_match >= 0.5 * g.n_wf)
    ev = ev.drop(columns=["n_wf", "n_match", "WF"])
    out = ev.set_index("ID").join(g[g.WF], how="inner")
    out["local"] = pd.to_datetime(out.t_origin, utc=True, format="ISO8601").map(
        lambda t: C.local(t).strftime("%Y-%m-%d %H:%M:%S %Z %a") if pd.notna(t) else "")
    cols = ["local", "lat", "lon", "depth", "tier", "M", "in_box", "n_tot", "n_S", "n_nonS", "S_P_span_s", "n_wf",
            "n_match", "n_sta_match", "comcat_id"]
    out[cols].sort_values("local").to_csv(C.OUT / "msh_analog_waveform_only.csv")
    print(f"waveform_only: {len(out)} of {len(g)} swept events; nonS>=4: {(out.n_nonS >= 4).sum()}", flush=True)


if __name__ == "__main__":
    main()
