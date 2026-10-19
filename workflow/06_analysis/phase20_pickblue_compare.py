#!/usr/bin/env python3
"""Compare the ELEP (ver5) and PickBlue (QuakeScope 2026) picks at the shared OBS stations.

Four comparisons, from the most direct to the most picker-agnostic:

  1. Pick-to-pick matching: same station, same phase, mutual nearest neighbour within
     TOL s. Done at each picker's own threshold, and at EQUAL COUNT (the top-N picks of
     each picker by confidence, N = the smaller of the two counts), which removes the
     difference in how the two pickers' confidences are calibrated.
  2. Timing: PickBlue minus ELEP time for matched pairs.
  3. Catalog-anchored recall: the share of the ELEP arrivals of the relocated
     catalog (qc_pass events) that PickBlue also picked, by distance and magnitude, and
     the travel-time residual of the PickBlue pick at those arrivals.
  4. Network coincidence (no association, no catalog): the share of picks that have
     a same-phase pick within +-WIN s at >= 2 stations within RAD km, minus the share
     expected by chance (neighbour times shifted by SHIFT s).

Only station-days where PickBlue ran with data (QuakeScope avail status 'loaded') are
used; ELEP picks outside them are dropped. Input is the comparison table written by
compare_elep_pickblue.py (one row per pick: both / elep_only / pickblue_only).

    python phase20_pickblue_compare.py            # compute (~15 min) + plot
    python phase20_pickblue_compare.py --plot     # plot from the cached stats
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from paper_style import FULL, INK, MUTED, OURS, REF, panel, save, use  # noqa: E402

QS = "/wd1/mdenolle_data/quakescope_compare"
CMP = f"{QS}/comparative_catalog_obs_2010_2016.parquet"
ARR = "../../data/catalog_v3/arrivals_v3.csv"
ORI = "../../data/catalog_v3/origins_v3.csv"
OUTD = "../../data/picks_compare"
STATS = f"{OUTD}/pickblue_compare_stats{os.environ.get('PH20_NET', '')}.json"
OUT = f"{OUTD}/pickblue_compare.png"

TOL = 500          # ms, pick matching tolerance
RAD = 60.0         # km, neighbour radius for the coincidence test
WIN = 10_000       # ms, coincidence half-window
SHIFT = 600_000    # ms, neighbour time shift for the chance level
E_THR = (0.05, 0.1, 0.2, 0.3, 0.5)   # threshold sweep, ELEP semblance
B_THR = (0.2, 0.3, 0.5, 0.7)         # threshold sweep, PickBlue probability
DIST_BINS = [0, 25, 50, 100, 150, 250, 400]
ML_BINS = [-1, 0.5, 1.0, 1.5, 2.0, 2.5, 6]


def mutual_nearest(ka, kb, tol):
    """Indices of mutual-nearest pairs between two int64 key arrays (group*1e12 + ms)."""
    ia, ib = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    sa, sb = ka[ia], kb[ib]

    def nearest(src, dst):
        pos = np.searchsorted(dst, src)
        lo, hi = np.clip(pos - 1, 0, len(dst) - 1), np.clip(pos, 0, len(dst) - 1)
        return np.where(np.abs(src - dst[lo]) <= np.abs(dst[hi] - src), lo, hi)

    j, i_of_b = nearest(sa, sb), nearest(sb, sa)
    dt = sb[j] - sa
    ok = (i_of_b[j] == np.arange(len(sa))) & (np.abs(dt) <= tol)
    return ia[ok], ib[j[ok]], dt[ok]


def load():
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(CMP, columns=["source", "tid", "pha", "elep_time", "pickblue_time",
                                    "elep_prob", "pickblue_conf", "in_window"])
    t = t.filter(t["in_window"])
    if os.environ.get("PH20_NET"):  # quick test on one network
        t = t.filter(pc.starts_with(t["tid"], os.environ["PH20_NET"] + "."))

    def ms(col):
        x = pc.cast(t[col], pa.timestamp("ms")).cast(pa.int64())
        return x.to_numpy(zero_copy_only=False)

    tid = t["tid"].to_numpy(zero_copy_only=False)
    pha = t["pha"].to_numpy(zero_copy_only=False)
    te = ms("elep_time")          # NaT -> null -> float NaN after to_numpy
    tb = ms("pickblue_time")
    pe = t["elep_prob"].to_numpy(zero_copy_only=False)
    cb = t["pickblue_conf"].to_numpy(zero_copy_only=False)
    # ELEP picks of the first run (v1) carry no probability: select on the time, not the prob
    he, hb = ~np.isnan(te), ~np.isnan(cb)
    E = dict(tid=tid[he], pha=pha[he], t=te[he].astype(np.int64), c=pe[he].astype(np.float32))
    B = dict(tid=tid[hb], pha=pha[hb], t=tb[hb].astype(np.int64), c=cb[hb].astype(np.float32))
    # station-days are either all v1 (no probability) or all scored, never mixed
    import pandas as pd
    E["v1"] = np.isnan(E["c"])
    dE = pd.Series(E["v1"]).groupby([E["tid"], E["t"] // 86_400_000]).first()
    dB = pd.MultiIndex.from_arrays([B["tid"], B["t"] // 86_400_000])
    cls = dE.reindex(dB).to_numpy()
    B["dayclass"] = np.where(pd.isna(cls), "none", np.where(cls == True, "v1", "scored"))  # noqa: E712
    E["dayclass"] = np.where(E["v1"], "v1", "scored")
    return E, B


def group_codes(E, B):
    import pandas as pd
    g = pd.factorize(np.concatenate([E["tid"], B["tid"]]).astype(object)
                     + "|" + np.concatenate([E["pha"], B["pha"]]).astype(object))[0]
    E["g"], B["g"] = g[:len(E["t"])].astype(np.int64), g[len(E["t"]):].astype(np.int64)
    t0 = min(E["t"].min(), B["t"].min())
    E["k"] = E["g"] * 10**12 + (E["t"] - t0)
    B["k"] = B["g"] * 10**12 + (B["t"] - t0)


def match(E, B, me, mb):
    a, b, dt = mutual_nearest(E["k"][me], B["k"][mb], TOL)
    return np.where(me)[0][a], np.where(mb)[0][b], dt


def top_n(c, mask, n):
    """Mask of the n highest-confidence picks among mask."""
    idx = np.where(mask)[0]
    keep = idx[np.argsort(-c[idx], kind="stable")[:n]]
    m = np.zeros(len(c), bool)
    m[keep] = True
    return m


def counts(E, B, me, mb, a, b, pha):
    ne, nb = int((me & (E["pha"] == pha)).sum()), int((mb & (B["pha"] == pha)).sum())
    nm = int((E["pha"][a] == pha).sum())
    return dict(elep=ne, pickblue=nb, both=nm, elep_only=ne - nm, pickblue_only=nb - nm,
                frac_elep=nm / max(ne, 1), frac_pickblue=nm / max(nb, 1),
                jaccard=nm / max(ne + nb - nm, 1))


def dt_stats(dt):
    q = np.percentile(dt / 1000, [5, 25, 50, 75, 95])
    return dict(n=int(len(dt)), p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4],
                mad=float(np.median(np.abs(dt / 1000 - q[2]))))


def coincidence(S, mask, xy, nbrs):
    """Per pick (in mask): number of neighbour stations with a same-phase pick within
    +-WIN, observed and with neighbour times shifted by SHIFT. -1 where the station has
    fewer than 2 neighbours."""
    obs = np.full(len(S["t"]), -1, np.int16)
    chance = np.full(len(S["t"]), -1, np.int16)
    for pha in ("P", "S"):
        sel = mask & (S["pha"] == pha)
        idx = np.where(sel)[0]
        tids = S["tid"][idx]
        order = np.argsort(tids, kind="stable")
        idx, tids = idx[order], tids[order]
        u, start = np.unique(tids, return_index=True)
        stop = np.r_[start[1:], len(idx)]
        times = {}
        for s, a, z in zip(u, start, stop):
            ii = idx[a:z]
            times[s] = (ii, np.sort(S["t"][ii]), S["t"][ii])
        for s, (ii, _, ti) in times.items():
            nb = [n for n in nbrs.get(s, []) if n in times]
            if len(nb) < 2:
                continue
            co, ch = np.zeros(len(ii), np.int16), np.zeros(len(ii), np.int16)
            for n in nb:
                tn = times[n][1]
                co += (np.searchsorted(tn, ti + WIN, "right") - np.searchsorted(tn, ti - WIN, "left")) > 0
                ts = ti + SHIFT
                ch += (np.searchsorted(tn, ts + WIN, "right") - np.searchsorted(tn, ts - WIN, "left")) > 0
            obs[ii], chance[ii] = co, ch
    return obs, chance


def neighbours(tids):
    import pyarrow.parquet as pq
    st = pq.read_table(f"{QS}/stations.parquet", columns=["id", "latitude", "longitude"]).to_pandas()
    st = st.drop_duplicates("id").set_index("id").reindex(tids).dropna()
    lat, lon = np.radians(st.latitude.values), np.radians(st.longitude.values)
    d = 6371 * np.arccos(np.clip(np.sin(lat[:, None]) * np.sin(lat) + np.cos(lat[:, None])
                                 * np.cos(lat) * np.cos(lon[:, None] - lon), -1, 1))
    ids = st.index.values
    return {ids[i]: [ids[j] for j in np.where((d[i] <= RAD) & (d[i] > 0))[0]] for i in range(len(ids))}


def catalog_recall(E, B):
    import pandas as pd
    a = pd.read_csv(ARR, usecols=["origin_id", "network", "station", "location", "phase", "time",
                                  "time_residual", "distance_km"], dtype={"location": str})
    o = pd.read_csv(ORI, usecols=["id", "qc_pass", "ML"])
    a = a.merge(o.rename(columns={"id": "origin_id"}), on="origin_id")
    a = a[a.qc_pass]
    a["tid"] = a.network + "." + a.station + "." + a.location.fillna("")
    common = set(np.unique(B["tid"]))
    a = a[a.tid.isin(common) & a.phase.isin(["P", "S"])].copy()
    a["t"] = pd.to_datetime(a.time.str.rstrip("Z")).values.astype("datetime64[ms]").astype(np.int64)
    # only station-days PickBlue ran on: a day with any PickBlue pick at that station
    bday = set(zip(B["tid"], B["t"] // 86_400_000))
    a = a[[k in bday for k in zip(a.tid, a.t // 86_400_000)]]
    out = {}
    for name, S in (("pickblue", B), ("elep", E)):
        gk = pd.factorize(np.concatenate([a.tid.values.astype(object) + "|" + a.phase.values.astype(object),
                                          S["tid"].astype(object) + "|" + S["pha"].astype(object)]))[0]
        ga, gs = gk[:len(a)].astype(np.int64), gk[len(a):].astype(np.int64)
        t0 = min(a.t.min(), S["t"].min())
        ka, ks = ga * 10**12 + (a.t.values - t0), gs * 10**12 + (S["t"] - t0)
        ia, ib, dt = mutual_nearest(ka, ks, TOL)
        hit = np.zeros(len(a), bool)
        hit[ia] = True
        dts = np.full(len(a), np.nan)
        dts[ia] = dt / 1000
        a[f"hit_{name}"], a[f"dt_{name}"] = hit, dts
        a[f"conf_{name}"] = np.nan
        a.loc[a.index[ia], f"conf_{name}"] = S["c"][ib]
    a["res_pickblue"] = a.time_residual + a.dt_pickblue
    v1days = set(zip(E["tid"][E["v1"]], E["t"][E["v1"]] // 86_400_000))
    a["v1day"] = [k in v1days for k in zip(a.tid, a.t // 86_400_000)]
    for pha in ("P", "S"):
        x = a[a.phase == pha]
        dcut = pd.cut(x.distance_km, DIST_BINS)
        mcut = pd.cut(x.ML, ML_BINS)
        both = x[x.hit_pickblue]
        out[pha] = dict(
            n=int(len(x)), recall_pickblue=float(x.hit_pickblue.mean()), recall_elep=float(x.hit_elep.mean()),
            recall_pickblue_v1days=float(x[x.v1day].hit_pickblue.mean()),
            recall_pickblue_scored=float(x[~x.v1day].hit_pickblue.mean()),
            n_v1days=int(x.v1day.sum()),
            by_dist=x.groupby(dcut, observed=False).hit_pickblue.agg(["mean", "size"]).reset_index()
                     .assign(bin=lambda d: d.distance_km.astype(str)).drop(columns="distance_km").to_dict("list"),
            by_ml=x.groupby(mcut, observed=False).hit_pickblue.agg(["mean", "size"]).reset_index()
                   .assign(bin=lambda d: d.ML.astype(str)).drop(columns="ML").to_dict("list"),
            res_elep_mad=float(np.median(np.abs(both.time_residual - both.time_residual.median()))),
            res_pickblue_mad=float(np.median(np.abs(both.res_pickblue - both.res_pickblue.median()))),
            res_elep_median=float(both.time_residual.median()),
            res_pickblue_median=float(both.res_pickblue.median()),
            dt_median=float(both.dt_pickblue.median()),
            hist_res_elep=np.histogram(both.time_residual, np.arange(-2, 2.01, 0.05))[0].tolist(),
            hist_res_pickblue=np.histogram(both.res_pickblue, np.arange(-2, 2.01, 0.05))[0].tolist(),
        )
    # S-P times at stations where both pickers picked both phases of an event
    ok = a[a.hit_pickblue]
    sp = ok.pivot_table(index=["origin_id", "tid"], columns="phase",
                        values=["dt_pickblue"], aggfunc="first").dropna()
    if len(sp):
        d = (sp[("dt_pickblue", "S")] - sp[("dt_pickblue", "P")]).values
        out["sp_diff"] = dict(n=int(len(d)), median=float(np.median(d)),
                              mad=float(np.median(np.abs(d - np.median(d)))))
    return out, a


def compute():
    os.makedirs(OUTD, exist_ok=True)
    E, B = load()
    print(f"ELEP {len(E['t']):,}  PickBlue {len(B['t']):,}", flush=True)
    group_codes(E, B)
    res = dict(config=dict(tol_s=TOL / 1000, rad_km=RAD, win_s=WIN / 1000, shift_s=SHIFT / 1000))

    # 1. native thresholds
    me, mb = np.ones(len(E["t"]), bool), np.ones(len(B["t"]), bool)
    a, b, dt = match(E, B, me, mb)
    res["native"] = {p: counts(E, B, me, mb, a, b, p) for p in ("P", "S")}
    res["native_dt"] = {p: dt_stats(dt[E["pha"][a] == p]) for p in ("P", "S")}
    res["hist_dt"] = {p: np.histogram(dt[E["pha"][a] == p] / 1000, np.arange(-0.5, 0.501, 0.02))[0].tolist()
                      for p in ("P", "S")}
    nets = np.array([s.split(".")[0] for s in E["tid"][a]])
    res["dt_by_network"] = {n: {p: float(np.median(dt[(nets == n) & (E["pha"][a] == p)]) / 1000)
                                for p in ("P", "S")} for n in np.unique(nets)}
    res["native_by_dayclass"] = {}
    for dc in ("v1", "scored"):
        m1, m2 = E["dayclass"] == dc, B["dayclass"] == dc
        a1, b1, d1 = match(E, B, m1, m2)
        res["native_by_dayclass"][dc] = {p: counts(E, B, m1, m2, a1, b1, p) for p in ("P", "S")}
        res["native_by_dayclass"][dc]["dt"] = {p: dt_stats(d1[E["pha"][a1] == p]) for p in ("P", "S")}
        res["native_by_dayclass"][dc]["station_days"] = int(len(set(zip(E["tid"][m1], E["t"][m1] // 86_400_000))))
    print("native", res["native"], res["native_by_dayclass"], flush=True)

    # threshold-controlled comparisons: station-days whose ELEP picks have a probability
    sE, sB = E["dayclass"] == "scored", B["dayclass"] == "scored"

    # 1b. equal count, per phase
    meq, mbq = np.zeros_like(sE), np.zeros_like(sB)
    res["equal_count_thresholds"] = {}
    for p in ("P", "S"):
        pe, pb = sE & (E["pha"] == p), sB & (B["pha"] == p)
        n = int(min(pe.sum(), pb.sum()))
        meq |= top_n(E["c"], pe, n)
        mbq |= top_n(B["c"], pb, n)
        res["equal_count_thresholds"][p] = dict(
            n=n, elep=float(E["c"][meq & pe].min()), pickblue=float(B["c"][mbq & pb].min()))
    a2, b2, dt2 = match(E, B, meq, mbq)
    res["equal_count"] = {p: counts(E, B, meq, mbq, a2, b2, p) for p in ("P", "S")}
    res["equal_count_dt"] = {p: dt_stats(dt2[E["pha"][a2] == p]) for p in ("P", "S")}
    print("equal count", res["equal_count"], flush=True)

    # 1c. threshold sweep (rematched each time)
    sweep = []
    for te in E_THR:
        for tb in B_THR:
            m1, m2 = sE & (E["c"] >= te), sB & (B["c"] >= tb)
            a3, b3, _ = match(E, B, m1, m2)
            for p in ("P", "S"):
                sweep.append(dict(elep_thr=te, pickblue_thr=tb, phase=p, **counts(E, B, m1, m2, a3, b3, p)))
    res["sweep"] = sweep
    print("sweep done", flush=True)

    # 3. catalog-anchored recall
    res["catalog"], arr = catalog_recall(E, B)
    # yield: share of picks (equal-count sets, 2010-2015) within TOL of a catalog arrival
    import pandas as pd
    yr = np.datetime64("2016-01-01", "ms").astype(np.int64)
    res["yield"] = {}
    for name, S, m in (("elep", E, meq), ("pickblue", B, mbq)):
        sel = np.where(m & (S["t"] < yr))[0]
        gk = pd.factorize(np.concatenate([arr.tid.values.astype(object) + "|" + arr.phase.values.astype(object),
                                          S["tid"][sel].astype(object) + "|" + S["pha"][sel].astype(object)]))[0]
        ka = np.sort(gk[:len(arr)].astype(np.int64) * 10**12 + arr.t.values)
        ks = gk[len(arr):].astype(np.int64) * 10**12 + S["t"][sel]
        pos = np.searchsorted(ka, ks)
        lo, hi = ka[np.clip(pos - 1, 0, len(ka) - 1)], ka[np.clip(pos, 0, len(ka) - 1)]
        on = (np.abs(ks - lo) <= TOL) | (np.abs(hi - ks) <= TOL)
        tp = S["pha"][sel]
        res["yield"][name] = {p: float(on[tp == p].mean()) for p in ("P", "S")}
    print("yield", res["yield"], flush=True)

    # 4. network coincidence, equal-count sets
    nbrs = neighbours(np.unique(np.concatenate([E["tid"], B["tid"]])))
    res["neighbours_per_station"] = float(np.median([len(v) for v in nbrs.values()]))
    res["coincidence"] = {}
    qs = np.linspace(0, 1, 11)
    for name, S, m in (("elep", E, meq), ("pickblue", B, mbq),
                       ("elep_scored_native", E, sE), ("pickblue_scored_native", B, sB),
                       ("elep_v1", E, E["dayclass"] == "v1"), ("pickblue_v1days", B, B["dayclass"] == "v1")):
        obs, ch = coincidence(S, m, None, nbrs)
        d = {}
        for p in ("P", "S"):
            v = m & (S["pha"] == p) & (obs >= 0)
            c = S["c"][v]
            if np.isnan(c).all():
                d[p] = dict(n=int(v.sum()), coinc=float((obs[v] >= 2).mean()), chance=float((ch[v] >= 2).mean()))
                continue
            edges = np.quantile(c, qs)
            k = np.clip(np.searchsorted(edges, c, "right") - 1, 0, 9)
            d[p] = dict(n=int(v.sum()), covered=float(v.sum() / max((m & (S["pha"] == p)).sum(), 1)),
                        coinc=float((obs[v] >= 2).mean()), chance=float((ch[v] >= 2).mean()),
                        conf_mid=[float(np.median(c[k == i])) for i in range(10)],
                        coinc_by_conf=[float((obs[v][k == i] >= 2).mean()) for i in range(10)],
                        chance_by_conf=[float((ch[v][k == i] >= 2).mean()) for i in range(10)])
        res["coincidence"][name] = d
        print("coincidence", name, {p: (d[p]["coinc"], d[p]["chance"]) for p in d}, flush=True)

    # 5. daily pick rates: per-station rank correlation of daily counts
    from scipy.stats import spearmanr
    rho = []
    de = pd.DataFrame({"tid": E["tid"], "d": E["t"] // 86_400_000}).value_counts().rename("e")
    db = pd.DataFrame({"tid": B["tid"], "d": B["t"] // 86_400_000}).value_counts().rename("b")
    j = pd.concat([de, db], axis=1).fillna(0).reset_index()
    for s, g in j.groupby("tid"):
        if len(g) >= 30:
            rho.append(spearmanr(g.e, g.b).statistic)
    res["daily_rate_spearman"] = dict(n_stations=len(rho), median=float(np.nanmedian(rho)),
                                      p25=float(np.nanpercentile(rho, 25)), p75=float(np.nanpercentile(rho, 75)))

    with open(STATS, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {STATS}")


def plot():
    import matplotlib.pyplot as plt
    use()
    r = json.load(open(STATS))
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 4.6))
    ax = axs.ravel()
    LS = {"P": "-", "S": "--"}

    # a: overlap at each picker's own threshold (all days), and at equal count (scored days)
    a = ax[0]
    rows = [("P", "native"), ("S", "native"), ("P", "equal_count"), ("S", "equal_count")]
    labels = ["P, own thr.", "S, own thr.", "P, equal N", "S, equal N"]
    for i, (p, k) in enumerate(rows):
        c = r[k][p]
        x0 = 0
        for val, col, alpha in ((c["elep_only"], OURS, 0.5), (c["both"], INK, 0.85), (c["pickblue_only"], REF, 0.5)):
            a.barh(i, val / 1e6, left=x0, color=col, alpha=alpha, height=0.62, edgecolor="white", linewidth=0.8)
            x0 += val / 1e6
        a.text(x0 + 0.4, i, f"{100 * c['jaccard']:.0f}%", va="center", fontsize=6.5, color=MUTED)
    a.set_yticks(range(4), labels)
    a.invert_yaxis()
    a.set_xlabel("picks (millions)")
    a.set_xlim(0, 33)
    a.text(0.98, 0.04, "ELEP only | both | PickBlue only\n% = shared / union", transform=a.transAxes,
           ha="right", va="bottom", fontsize=6, color=MUTED)
    panel(a, "a")

    # b: share of one picker's picks above a threshold that the other picker also has
    a = ax[1]
    sw = r["sweep"]
    for p in ("P", "S"):
        e = sorted((s["elep_thr"], s["frac_elep"]) for s in sw if s["phase"] == p and s["pickblue_thr"] == 0.2)
        b = sorted((s["pickblue_thr"], s["frac_pickblue"]) for s in sw if s["phase"] == p and s["elep_thr"] == 0.05)
        a.plot(*zip(*e), LS[p], marker="o", ms=3, color=OURS, lw=1.2, label=f"ELEP {p} found by PickBlue")
        a.plot(*zip(*b), LS[p], marker="s", ms=3, color=REF, lw=1.2, label=f"PickBlue {p} found by ELEP")
    a.set_ylim(0, 1)
    a.set_xlabel("threshold of the first-named picker")
    a.set_ylabel("share also picked by the other\n(at its own threshold)")
    a.legend(frameon=False, fontsize=5.5, loc="lower right")
    panel(a, "b")

    # c: timing of matched pairs
    a = ax[2]
    edges = np.arange(-0.5, 0.501, 0.02)
    cen = edges[:-1] + 0.01
    for p in ("P", "S"):
        h = np.array(r["hist_dt"][p], float)
        a.plot(cen, h / h.sum(), LS[p], color=INK, lw=1.2, label=f"{p}: median {r['native_dt'][p]['median']:+.2f} s")
    a.axvline(0, color=MUTED, lw=0.6)
    a.set_xlabel("PickBlue − ELEP pick time (s)")
    a.set_ylabel("fraction of matched pairs")
    a.legend(frameon=False, fontsize=6, loc="upper right")
    panel(a, "c")

    # d, e: catalog arrivals that PickBlue also picked, by distance and by magnitude
    for a, key, xl in ((ax[3], "by_dist", "epicentral distance (km)"), (ax[4], "by_ml", "event $M_L$")):
        for p in ("P", "S"):
            d = r["catalog"][p][key]
            lo_hi = [[float(v) for v in b.strip("(]").split(",")] for b in d["bin"]]
            if key == "by_ml":
                lo_hi = [[max(lo, 0.0), min(hi, 3.0)] for lo, hi in lo_hi]
            x = [np.mean(v) for v in lo_hi]
            ok = np.array(d["size"]) >= 200
            a.plot(np.array(x)[ok], np.array(d["mean"])[ok], LS[p], marker="o", ms=3.5, color=REF, lw=1.2,
                   label=f"{p} ({100 * r['catalog'][p]['recall_pickblue']:.0f}% of {r['catalog'][p]['n']:,})")
        a.set_ylim(0, 1)
        a.set_xlabel(xl)
        a.set_ylabel("catalog arrivals also\npicked by PickBlue")
        a.legend(frameon=False, fontsize=6, loc="lower left" if key == "by_dist" else "lower right")
    panel(ax[3], "d")
    panel(ax[4], "e")

    # f: network coincidence above chance, by confidence decile (equal-count sets)
    a = ax[5]
    for name, col, lab in (("elep", OURS, "ELEP"), ("pickblue", REF, "PickBlue")):
        for p in ("P", "S"):
            d = r["coincidence"][name][p]
            y = np.array(d["coinc_by_conf"]) - np.array(d["chance_by_conf"])
            a.plot(np.arange(1, 11), 100 * y, LS[p], color=col, lw=1.2, label=f"{lab} {p}")
    a.set_xlabel("confidence decile (own scale)")
    a.set_ylabel("picks on ≥2 neighbours (%),\nminus chance")
    a.set_xticks([1, 5, 10])
    a.set_ylim(bottom=0)
    a.legend(frameon=False, fontsize=6, loc="upper left")
    panel(a, "f")

    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    save(fig, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if "--plot" not in sys.argv:
        compute()
    plot()
