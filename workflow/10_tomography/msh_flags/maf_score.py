"""Step 5: score the events with the pre-registered rule (RULE.md, registered 2026-11-26 16:27:27 PST).

  python maf_score.py [--nproc 6]

Outputs:
  OUT/event_flags_v61.csv          ID,event_flag: every non-empty v6.0 event_flag (unchanged) + "analog_test_signal"
                                   for the certain class (joined with ';' after an existing flag)
  OUT/msh_analog_candidates.csv    certain, probable and mixed events with their numbers
  WORK/events_scored.csv           every event in scope; WORK/wf_rows.csv per-pick waveform checks (cache, resumable)
  WORK/summary.json                counts and checks
"""
import argparse
import csv
import json
import multiprocessing as mp
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import maf_common as C  # noqa: E402
import maf_calendar as K  # noqa: E402

FLAG = "analog_test_signal"
N_IN_MIN, SHARE_MIN = 3, 0.5
N_WF_MAX, N_MATCH_MIN, MATCH_SHARE = 8, 2, 0.5
NONS_MAX = 3
N_CTRL = 400
WF_COLS = ["ID", "set", "station", "phase", "t", "status", "matched", "edge", "cc", "snr", "dt", "amp", "edge_far",
           "cc_far", "dt_far"]


def in_windows(t, W):
    """t POSIX array -> index of the window holding t (or -1)."""
    lo, hi = W.t_lo.values, W.t_hi.values
    i = np.searchsorted(lo, t, side="right") - 1
    ok = (i >= 0) & (t <= hi[np.clip(i, 0, None)])
    return np.where(ok, i, -1)


def wf_one(a):
    import maf_wave as WV
    eid, setname, sta, ph, t = a
    try:
        r = WV.check_pick(sta, t)
    except Exception as e:
        r = dict(status=f"error {type(e).__name__}", matched=False)
    return dict(ID=eid, set=setname, station=sta, phase=ph, t=t, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=C.NPROC)
    a = ap.parse_args()
    cat = C.catalog()
    p = C.picks()
    S = set(json.load(open(C.WORK / "signal_stations.json"))["signal_stations"])
    W = pd.read_csv(C.WORK / "windows.csv").sort_values("t_lo").reset_index(drop=True)
    assert (W.t_lo.values[1:] > W.t_hi.values[:-1]).all(), "overlapping windows"

    p["tp"] = p.t.astype("int64").values / 1e9
    p["onS"] = p.station.isin(S)
    wi = in_windows(p.tp.values, W)
    p["win"] = wi
    p["inW"] = p.onS & (wi >= 0)
    lat = cat.lat.where(cat.lat.notna(), cat.v4_lat)
    lon = cat.lon.where(cat.lon.notna(), cat.v4_lon)
    cat["in_box"] = lat.between(*C.MSH_BOX["lat"]) & lon.between(*C.MSH_BOX["lon"])
    ids_S = set(p[p.onS].ID)
    scope = cat[cat.in_box | cat.ID.isin(ids_S)].copy()

    g = p[p.ID.isin(scope.ID)].groupby("ID")
    ev = pd.DataFrame({"n_tot": g.size(), "n_S": g.onS.sum(), "n_in": g.inW.sum(),
                       "n_in_P": g.apply(lambda x: int((x.inW & (x.phase == "P")).sum()))})
    ev["n_nonS"] = ev.n_tot - ev.n_S
    ev["share"] = ev.n_in / ev.n_tot
    pin = p[p.inW & p.ID.isin(scope.ID)]
    kind = pin.groupby("ID").win.agg(lambda w: ",".join(sorted(set(W.kind.values[w] + ":" + W.source.values[w]))))
    ev["window"] = kind
    sp = p[p.onS & (p.phase == "P") & p.ID.isin(scope.ID)].groupby("ID").tp
    ev["S_P_span_s"] = sp.max() - sp.min()
    ev = scope.set_index("ID").join(ev, how="left")
    for c in ("n_tot", "n_S", "n_in", "n_in_P", "n_nonS"):
        ev[c] = ev[c].fillna(0).astype(int)
    ev["share"] = ev.share.fillna(0.0)
    ev["T"] = (ev.n_in >= N_IN_MIN) & (ev.share >= SHARE_MIN)
    ev["G"] = ev.n_nonS <= NONS_MAX
    t0 = pd.to_datetime(ev.time.where(ev.time.notna(), ev.v4_time), utc=True, format="ISO8601")
    ev["t_origin"] = t0
    dv = t0.map(lambda x: K.device_time(x) if pd.notna(x) else pd.NaT)
    ev["device_dow"] = dv.map(lambda x: x.dayofweek if pd.notna(x) else -1)
    ev["device_min"] = dv.map(lambda x: x.hour * 60 + x.minute + x.second / 60 if pd.notna(x) else np.nan)

    # ---- waveform jobs
    jobs = []
    tested = p[p.inW & p.ID.isin(ev.index[ev.n_in >= 1])].copy()
    tested["po"] = (tested.phase != "P").astype(int)
    tested = tested.sort_values(["ID", "po", "tp"]).groupby("ID").head(N_WF_MAX)
    jobs += [(int(r.ID), "in_window", r.station, r.phase, float(r.tp)) for r in tested.itertuples()]
    rng = np.random.default_rng(0)
    pool_a = ev[(ev.n_in == 0) & (ev.n_S >= 3)].index.values
    ctrl_a = set(rng.choice(pool_a, size=min(N_CTRL, len(pool_a)), replace=False).tolist())
    ctrl_b = set(ev[(ev.n_in == 0) & ev.in_box & (ev.device_dow == 2) & ev.device_min.between(660, 810)].index)
    for name, ids in (("control_random", ctrl_a), ("control_wed_noon_box", ctrl_b - ctrl_a)):
        x = p[p.onS & p.ID.isin(ids)].copy()
        x["po"] = (x.phase != "P").astype(int)
        x = x.sort_values(["ID", "po", "tp"]).groupby("ID").head(N_WF_MAX)
        jobs += [(int(r.ID), name, r.station, r.phase, float(r.tp)) for r in x.itertuples()]
    f = C.WORK / "wf_rows.csv"
    done = set()
    if f.exists():
        old = pd.read_csv(f)
        done = set(zip(old.ID, old.station, old.phase, old.t.round(3)))
    todo = [j for j in jobs if (j[0], j[2], j[3], round(j[4], 3)) not in done]
    print(f"waveform checks: {len(jobs)} picks, {len(todo)} to do", flush=True)
    if todo:
        new = not f.exists()
        with open(f, "a", newline="") as fh, mp.get_context("spawn").Pool(a.nproc) as pool:
            w = csv.DictWriter(fh, fieldnames=WF_COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            for i, r in enumerate(pool.imap_unordered(wf_one, todo, chunksize=4)):
                w.writerow(r)
                if i % 200 == 0:
                    fh.flush()
                    print(f"wf {i + 1}/{len(todo)}", flush=True)
    wf = pd.read_csv(f)
    wf = wf.drop_duplicates(["ID", "set", "station", "phase", "t"])
    wf["has"] = wf.status == "ok"
    wg = wf.groupby(["ID", "set"]).agg(n_wf=("has", "sum"), n_match=("matched", "sum"),
                                         n_read=("status", "size")).reset_index()
    wg["WF"] = (wg.n_match >= N_MATCH_MIN) & (wg.n_match >= MATCH_SHARE * wg.n_wf)
    inw = wg[wg.set == "in_window"].set_index("ID")
    ev = ev.join(inw[["n_wf", "n_match", "WF"]], how="left")
    ev["n_wf"] = ev.n_wf.fillna(0).astype(int)
    ev["n_match"] = ev.n_match.fillna(0).astype(int)
    ev["WF"] = ev.WF.fillna(False).astype(bool)
    ev["cls"] = np.where(ev["T"] & ev.WF & ev.G, "certain",
                         np.where(ev["T"] & ev.G, "probable", np.where(ev["T"], "mixed", "none")))
    ev["year"] = ev.t_origin.dt.year
    ev.reset_index().to_csv(C.WORK / "events_scored.csv", index=False)

    # ---- flag file
    fl = cat[["ID", "event_flag"]].copy()
    cert = set(ev.index[ev.cls == "certain"])
    fl["event_flag"] = [(f + ";" + FLAG if f else FLAG) if i in cert else f for i, f in zip(fl.ID, fl.event_flag)]
    out = fl[fl.event_flag != ""].sort_values("ID")
    out.to_csv(C.OUT / "event_flags_v61.csv", index=False)
    cols = ["cls", "t_origin", "lat", "lon", "depth", "tier", "M", "comcat_id", "in_box", "n_tot", "n_S", "n_in",
            "n_in_P", "n_nonS", "share", "n_wf", "n_match", "window", "S_P_span_s", "event_flag",
            "review_auto_indicators"]
    cand = ev[ev.cls != "none"][cols].reset_index().sort_values(["cls", "t_origin"])
    cand.to_csv(C.OUT / "msh_analog_candidates.csv", index=False)

    # ---- summary
    ctrl = wg[wg.set != "in_window"]
    summ = dict(
        n_scope=int(len(ev)), n_box=int(ev.in_box.sum()), n_with_S_picks=int((ev.n_S > 0).sum()),
        n_with_pick_in_W=int((ev.n_in > 0).sum()),
        classes=ev.cls.value_counts().to_dict(),
        certain_by_year=ev[ev.cls == "certain"].year.value_counts().sort_index().to_dict(),
        probable_by_year=ev[ev.cls == "probable"].year.value_counts().sort_index().to_dict(),
        mixed_by_year=ev[ev.cls == "mixed"].year.value_counts().sort_index().to_dict(),
        certain_in_box=int(((ev.cls == "certain") & ev.in_box).sum()),
        certain_by_window=ev[ev.cls == "certain"].window.value_counts().to_dict(),
        certain_with_nonS=ev[ev.cls == "certain"].n_nonS.value_counts().sort_index().to_dict(),
        control=dict(sets=ctrl.groupby("set").agg(n=("ID", "size"), wf_true=("WF", "sum")).reset_index()
                     .to_dict("records"),
                     picks_matched=wf[wf.set != "in_window"].groupby("set").matched.sum().to_dict(),
                     picks_with_data=wf[wf.set != "in_window"].groupby("set").has.sum().to_dict()),
        in_window_picks=dict(n=int((wf.set == "in_window").sum()), with_data=int(wf[wf.set == "in_window"].has.sum()),
                             matched=int(wf[wf.set == "in_window"].matched.sum())),
        flag_file=dict(rows=int(len(out)), analog=int(out.event_flag.str.contains(FLAG).sum()),
                       carried_v60=int((cat.event_flag != "").sum()),
                       analog_and_v60_flag=int(len(cert & set(cat.ID[cat.event_flag != ""])))),
        rule=dict(N_IN_MIN=N_IN_MIN, SHARE_MIN=SHARE_MIN, N_WF_MAX=N_WF_MAX, N_MATCH_MIN=N_MATCH_MIN,
                  MATCH_SHARE=MATCH_SHARE, NONS_MAX=NONS_MAX),
    )
    json.dump(summ, open(C.WORK / "summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
