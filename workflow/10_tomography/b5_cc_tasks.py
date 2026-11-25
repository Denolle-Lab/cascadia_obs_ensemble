"""Step 1 of the CC differential times (D-B040, D-B078 test sets): picks, pick timing status, pairs.

For each set (axial, blanco, msh, jdf) of /wd1/mdenolle_data/tomo_v1/b5/cc_test (built by -18, read only):
* picks.parquet: every pick of the set's events in Cascadia_picks_catalog_v4.1t.csv (md5 checked), with the
  v5.1 origin time (events_<set>.csv), travel time tt = pick - T0(v5.1), and the D-B072 timing status from
  corrections.csv (retimed / checked / unretimable / not_checked; flag, applied correction);
* obs.parquet: one row per pair and common station-phase (both events picked the same station and phase), for
  the given pairs (pairs_<set>.csv, all kept, source=given) and the added pairs (source=added): for each event its
  K nearest neighbours within R km (3-D, v5.1 hypocentres) with >= NMIN common station-phases that are not a
  given pair. Catalog dt = tt1 - tt2 (v5.1 origin times, OTC = 0).

  python b5_cc_tasks.py [--sets axial,blanco,msh,jdf]   -> /wd1/mdenolle_data/tomo_v1/cc_dt/<set>/{picks,obs,pairs}.parquet
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b5_cc_common as C  # noqa: E402
from manifest import write_manifest  # noqa: E402

ADD_K, ADD_R_KM, ADD_NMIN = 20, 3.0, 4


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def xyz(ev):
    la0, lo0 = ev.lat.mean(), ev.lon.mean()
    x = (ev.lon - lo0) * 111.195 * np.cos(np.radians(la0))
    y = (ev.lat - la0) * 111.195
    return np.c_[x, y, ev.depth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(C.SETS))
    a = ap.parse_args()
    m = md5(C.PICKS)
    assert m == C.PICKS_MD5, f"pick file md5 {m} != {C.PICKS_MD5}"
    pk = pd.read_csv(C.PICKS, skipinitialspace=True)
    pk["pick_row"] = np.arange(len(pk))
    co = pd.read_csv(C.CORR, usecols=["pick_row", "channel", "flag", "applied", "timing_correction_s",
                                      "measured_correction_s", "epoch_class"])
    counts = {}
    for name in a.sets.split(","):
        ev = C.load_events(name)
        out = C.OUT / name
        out.mkdir(parents=True, exist_ok=True)
        p = pk[pk.event_ID.isin(ev.ID)].rename(columns={"event_ID": "ID"}).copy()
        p = p.merge(co.rename(columns={"channel": "retime_channel", "timing_correction_s": "corr_applied_s"}),
                    on="pick_row", how="left")
        p["timing"] = [C.timing_status(f, ap_) for f, ap_ in zip(p.flag, p.applied)]
        p["t_pick"] = [C.UTCDateTime(t.strip()).timestamp for t in p.pick_datetime]
        p = p.merge(ev[["ID", "T0"]], on="ID")
        p["tt"] = p.t_pick - p.T0
        p["net"] = p.station.str.split(".").str[1]
        # duplicates (same event, station, phase) would make the dt ambiguous: keep the highest probability
        nd = len(p)
        p = p.sort_values("pick_prob", ascending=False).drop_duplicates(["ID", "station", "phase"])
        ndup = nd - len(p)
        # S-P at the station: from the picks, else from the P travel time with Vp/Vs 1.73
        sp = p.pivot_table(index=["ID", "station"], columns="phase", values="tt", aggfunc="first")
        sp["sp"] = sp.get("S") - sp.get("P")
        sp["sp_est"] = np.where(sp.sp.notna(), sp.sp, np.where(sp.get("P").notna(), 0.73 * sp.get("P"),
                                                               sp.get("S") * 0.73 / 1.73))
        p = p.merge(sp[["sp", "sp_est"]].reset_index(), on=["ID", "station"], how="left")
        p = p.sort_values(["ID", "station", "phase"]).reset_index(drop=True)
        cols = ["ID", "station", "net", "phase", "pick_datetime", "t_pick", "T0", "tt", "pick_prob", "pick_row",
                "timing", "flag", "applied", "corr_applied_s", "measured_correction_s", "epoch_class",
                "retime_channel", "timing_corrected", "timing_correction_s", "sp", "sp_est"]
        p[cols].to_parquet(out / "picks.parquet", index=False)

        # pairs
        gp = C.load_given_pairs(name)
        gp["source"] = "given"
        X = xyz(ev)
        tree = cKDTree(X)
        idpos = {i: k for k, i in enumerate(ev.ID)}
        keyset = {frozenset((a_, b_)) for a_, b_ in zip(gp.ID1, gp.ID2)}
        sp_sets = p.groupby("ID").apply(lambda g: set(zip(g.station, g.phase))).to_dict()
        added = []
        for k, i in enumerate(ev.ID):
            d, j = tree.query(X[k], k=min(len(ev), 4 * ADD_K + 1), distance_upper_bound=ADD_R_KM)
            n = 0
            for dd, jj in zip(d, j):
                if not np.isfinite(dd) or jj == k:
                    continue
                o = int(ev.ID.iloc[jj])
                key = frozenset((i, o))
                ncom = len(sp_sets.get(i, set()) & sp_sets.get(o, set()))
                if ncom < ADD_NMIN:
                    continue
                n += 1
                if key not in keyset:
                    keyset.add(key)
                    added.append((min(i, o), max(i, o), ncom, dd))
                if n >= ADD_K:
                    break
        ap_ = pd.DataFrame(added, columns=["ID1", "ID2", "n_dt", "sep_km"])
        ap_["source"] = "added"
        ap_["n_dt"] = -1  # n_dt (catalog dt rows of B5 prep) is defined only for the given pairs
        pairs = pd.concat([gp, ap_], ignore_index=True)
        pairs["sep_km"] = np.linalg.norm(X[[idpos[i] for i in pairs.ID1]] - X[[idpos[i] for i in pairs.ID2]], axis=1)
        pairs["pair"] = np.arange(len(pairs))
        # common station-phases
        pi = p.set_index(["ID", "station", "phase"])
        q = p[["ID", "station", "phase"]]
        o1 = pairs[["pair", "ID1", "ID2", "source", "sep_km"]].merge(q.rename(columns={"ID": "ID1"}), on="ID1")
        o1 = o1.merge(q.rename(columns={"ID": "ID2"}), on=["ID2", "station", "phase"])
        keep = ["t_pick", "tt", "pick_prob", "pick_row", "timing", "flag", "sp_est"]
        a1 = pi.loc[list(zip(o1.ID1, o1.station, o1.phase)), keep].reset_index(drop=True).add_suffix("1")
        a2 = pi.loc[list(zip(o1.ID2, o1.station, o1.phase)), keep].reset_index(drop=True).add_suffix("2")
        obs = pd.concat([o1.reset_index(drop=True), a1, a2], axis=1)
        obs["dt_cat"] = obs.tt1 - obs.tt2
        ncom = obs.groupby("pair").size()
        pairs["n_common"] = pairs.pair.map(ncom).fillna(0).astype(int)
        pairs.to_parquet(out / "pairs.parquet", index=False)
        obs.to_parquet(out / "obs.parquet", index=False)
        c = dict(events=len(ev), picks=len(p), duplicate_picks_dropped=ndup,
                 timing=p.timing.value_counts().to_dict(), pairs_given=int((pairs.source == "given").sum()),
                 pairs_added=int((pairs.source == "added").sum()), obs=len(obs),
                 obs_given=int((obs.source == "given").sum()),
                 given_pairs_ncommon_lt_ndt=int(((pairs.source == "given") & (pairs.n_common < pairs.n_dt)).sum()),
                 given_pairs_ncommon_eq_ndt=int(((pairs.source == "given") & (pairs.n_common == pairs.n_dt)).sum()))
        counts[name] = c
        print(name, c, flush=True)
    write_manifest("b5_cc_tasks", C.OUT, inputs=[C.PICKS, C.CORR] + [C.SETDIR / f"{k}_{s}.csv" for s in counts
                                                                       for k in ("events", "pairs")],
                   outputs=[C.OUT / s / f for s in counts for f in ("picks.parquet", "pairs.parquet", "obs.parquet")],
                   params=dict(add_k=ADD_K, add_r_km=ADD_R_KM, add_nmin=ADD_NMIN), counts=counts)


if __name__ == "__main__":
    main()
