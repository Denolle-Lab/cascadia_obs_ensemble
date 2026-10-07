"""Score the v5 base-model dry run (dryrun_models.py output) by group and option.

References, all on the same station-days:
  v3 arrivals   picks associated and relocated in the v3 catalog (data/catalog_v3/arrivals_v3.csv).
                They came from the v3 model set, so this measures what (b) keeps of (a).
  ANSS events   ComCat events within 150 km of the station; P and S times predicted with
                straight rays at 6.5 and 3.75 km/s, matched within 1.5 s + 5% (P) and
                3 s + 8% (S) of the travel time. Independent of the picker.
  ver4 picks    the raw ver4 picks on the station-day: how close option (a) with the v5
                channel rule comes to the picks we have.

    python 1_picking/dryrun_score.py --run DIR
Writes DIR/score_by_group.csv and DIR/score_by_station_day.csv, prints the summary.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = __file__.rsplit("/1_picking/", 1)[0]
ARRIVALS = f"{ROOT}/data/catalog_v3/arrivals_v3.csv"
ANSS = f"{ROOT}/data/datasets_anss/anss_2010-15.csv"
STATIONS = f"{ROOT}/data/station_coverage_review_2026-09-30.csv"
VER4 = f"{ROOT}/data/picks_v4/all_picks_all_regions_2010_2015_ver4.csv.gz"
TOL = {"P": 0.5, "S": 1.0}          # s, match against v3 arrivals and ver4 picks
VP, VS, RMAX = 6.5, 3.75, 150.0


def to_s(t):
    return pd.to_datetime(t, utc=True, format="mixed").astype("int64").to_numpy() / 1e9


def matched(ref, picks, tol):
    """Boolean per reference time: a pick within tol (tol scalar or per-reference)."""
    if len(ref) == 0:
        return np.zeros(0, bool)
    if len(picks) == 0:
        return np.zeros(len(ref), bool)
    p = np.sort(picks)
    i = np.clip(np.searchsorted(p, ref), 1, len(p) - 1)
    d = np.minimum(np.abs(p[i] - ref), np.abs(p[i - 1] - ref))
    return d <= tol


def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_ver4(keys):
    want = pd.DataFrame(list(keys), columns=["network", "station", "day"])
    out = []
    for ch in pd.read_csv(VER4, usecols=["network", "station", "label", "pick_time"],
                          dtype=str, chunksize=5_000_000):
        ch["day"] = ch.pick_time.str[:10]
        out.append(ch.merge(want, on=["network", "station", "day"]))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-ver4", action="store_true", help="skip the (slow) ver4 comparison")
    a = ap.parse_args()

    tasks = pd.read_csv(os.path.join(a.run, "tasks.csv"), dtype={"day": str})
    log = pd.read_csv(os.path.join(a.run, "log.csv"), dtype=str)
    ok = log[log.status == "ok"][["network", "station", "day", "band", "z_copied"]]
    tasks = tasks.merge(ok, on=["network", "station", "day"])
    keys = set(zip(tasks.network, tasks.station, tasks.day))

    picks = pd.concat([pd.read_csv(f, usecols=["network", "station", "label", "pick_time", "config"], dtype=str)
                       .assign(day=os.path.basename(f).split(".")[2])
                       for f in glob.glob(os.path.join(a.run, "picks", "*.csv"))])
    picks["t"] = to_s(picks.pick_time)

    arr = pd.read_csv(ARRIVALS, usecols=["network", "station", "phase", "time"], dtype=str)
    arr["day"] = arr.time.str[:10]
    arr = arr[[k in keys for k in zip(arr.network, arr.station, arr.day)]].copy()
    arr["t"] = to_s(arr.time)

    sta = pd.read_csv(STATIONS).groupby(["network", "station"])[["lat", "lon"]].first()
    ev = pd.read_csv(ANSS, usecols=["time", "latitude", "longitude", "depth", "mag"])
    ev["t"] = to_s(ev.time)

    v4 = None
    if not a.no_ver4:
        v4 = load_ver4(keys)
        v4["t"] = to_s(v4.pick_time)

    rows = []
    for r in tasks.itertuples():
        t0 = to_s(pd.Series([r.day]))[0]
        la, lo = sta.loc[(r.network, r.station)]
        e = ev[(ev.t > t0 - 120) & (ev.t < t0 + 86400)]
        dist = haversine(la, lo, e.latitude.to_numpy(), e.longitude.to_numpy())
        hyp = np.hypot(dist, e.depth.to_numpy())
        e = e[dist <= RMAX]
        hyp = hyp[dist <= RMAX]
        pred = {"P": e.t.to_numpy() + hyp / VP, "S": e.t.to_numpy() + hyp / VS}
        tol = {"P": 1.5 + 0.05 * hyp / VP, "S": 3.0 + 0.08 * hyp / VS}
        inday = {ph: (pred[ph] >= t0 + 5) & (pred[ph] < t0 + 86395) for ph in pred}
        p_sd = picks[(picks.network == r.network) & (picks.station == r.station) & (picks.day == r.day)]
        a_sd = arr[(arr.network == r.network) & (arr.station == r.station) & (arr.day == r.day)]
        for cfg in ("a", "b", "o"):
            for ph in ("P", "S"):
                pt = p_sd[(p_sd.config == cfg) & (p_sd.label == ph)].t.to_numpy()
                rt = a_sd[a_sd.phase == ph].t.to_numpy()
                m_anss = matched(pred[ph][inday[ph]], pt, tol[ph][inday[ph]])
                rec = dict(network=r.network, station=r.station, day=r.day, group=r.group,
                           config=cfg, phase=ph, n_picks=len(pt),
                           n_v3=len(rt), hit_v3=int(matched(rt, pt, TOL[ph]).sum()),
                           n_anss=int(inday[ph].sum()), hit_anss=int(m_anss.sum()),
                           n_anss_m2=int((e.mag.to_numpy()[inday[ph]] >= 2).sum()),
                           hit_anss_m2=int(m_anss[e.mag.to_numpy()[inday[ph]] >= 2].sum()))
                if cfg == "b":
                    pa = p_sd[(p_sd.config == "a") & (p_sd.label == ph)].t.to_numpy()
                    rec["b_new_vs_a"] = int((~matched(pt, pa, 0.2)).sum())
                if v4 is not None:
                    q = v4[(v4.network == r.network) & (v4.station == r.station) & (v4.day == r.day)
                           & (v4.label == ph)].t.to_numpy()
                    rec["n_ver4"] = len(q)
                    rec["ver4_found"] = int(matched(q, pt, 0.1).sum())
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run, "score_by_station_day.csv"), index=False)

    agg = {c: "sum" for c in df.columns if c.startswith(("n_", "hit_", "ver4_", "b_new"))}
    g = df.groupby(["group", "phase", "config"]).agg({**agg, "day": "count"}).rename(columns={"day": "station_days"})
    g["picks_per_day"] = g.n_picks / g.station_days
    g["recall_v3"] = g.hit_v3 / g.n_v3
    g["recall_anss"] = g.hit_anss / g.n_anss
    g["recall_anss_m2"] = g.hit_anss_m2 / g.n_anss_m2
    if "n_ver4" in g:
        g["ver4_reproduced"] = g.ver4_found / g.n_ver4
    g.to_csv(os.path.join(a.run, "score_by_group.csv"))
    cols = ["station_days", "picks_per_day", "recall_v3", "recall_anss", "recall_anss_m2", "n_anss"]
    cols += ["ver4_reproduced"] if "ver4_reproduced" in g else []
    cols += ["b_new_vs_a"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(g[cols].to_string())


if __name__ == "__main__":
    main()
