#!/usr/bin/env python3
"""Seismicity-style metrics per plate-boundary segment on catalog v6.0 (D-B107 feasibility study).

Event set: located earthquakes (event_flag empty), tiers A-C, waveform-review P1 (Mount St Helens weekly signal)
left out; segment rule pb_common.SEGMENT_RULE. Input events with segment, 8 km/s depth and Slab2 columns come from
pb_structure.py (/wd1/mdenolle_data/paper_plan/events_segments_v60.parquet).

Metrics (95 % bootstrap intervals, 500 draws, where cheap):
* counts, Mmax, number of M >= 4;
* Mc (maximum curvature + 0.2) and b (Aki-Utsu, continuous magnitudes) of the segment's tier A-C events; both
  re-estimated in each event bootstrap draw; the event-weighted median Mc of the v6.0 Mc map cells (mc_best, flagged
  events excluded) as a cross-check;
* clustering (v6.0 nearest-neighbour clustering, events above the regional Mc of that run, tiers A-C): background
  share, shares of mainshock-aftershock, swarm and multiplet events; unit bootstrap (a background event or a whole
  family is one unit); families by type (segment of a family = most of its members); migrating swarms (zbz swarm
  families with >= 10 events flagged migrating by either migration test) and DBSCAN swarm-like clusters flagged
  migrating;
* depth below the solid surface (bsf) of depth-good events: p10, median, p90 (event bootstrap); relative to the
  8 km/s level of the model column where resolved (share below it, median offset); relative to Slab2;
* fault surfaces v1 of the segment: number, stable, member-weighted dip, width, seismogenic depth range (2-98 % of
  depth-ok members);
* temporal: events >= own Mc per year, coefficient of variation of inter-event times (all and background),
  burstiness (sigma - mu) / (sigma + mu), share of events on the busiest 1 % of days, share of the largest family.

Writes data/paper_plan/plate_boundaries/seismicity_metrics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402

NB = 500
RNG = np.random.default_rng(P.SEED)


def mc_maxc(m, corr=0.2):
    m = m[np.isfinite(m)]
    if len(m) < 50:
        return np.nan
    h, e = np.histogram(m, bins=np.arange(-2, 8, 0.1))
    return float(e[np.argmax(h)] + 0.05 + corr)


def b_aki(m, mc):
    m = m[np.isfinite(m) & (m >= mc)]
    if len(m) < 30:
        return np.nan, len(m)
    return float(1.0 / (np.log(10) * (m.mean() - mc))), len(m)


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan, np.nan
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def interevent_stats(t_days):
    t = np.sort(np.asarray(t_days, float))
    if len(t) < 20:
        return np.nan, np.nan
    dt = np.diff(t)
    mu, sd = dt.mean(), dt.std()
    return float(sd / mu), float((sd - mu) / (sd + mu))


def main():
    c = pd.read_parquet(P.BIG / f"events_segments_{P.TAG}.parquet")
    E = c[c.is_eq & c.abc & ~c.review_P1].copy()
    z = pd.read_csv(P.CLU / f"events_zbz_{P.TAG}.csv.gz", usecols=["event_id", "in_analysis", "family_id", "family_type",
                                                                "cls", "role", "background", "region"])
    z = z.rename(columns={"event_id": "ID", "region": "zbz_region"})
    fam = pd.read_csv(P.CLU / f"families_zbz_{P.TAG}.csv", low_memory=False)
    dbc = pd.read_csv(P.CLU / f"dbscan_clusters_{P.TAG}.csv")
    dbm = pd.read_csv(P.CLU / f"dbscan_members_{P.TAG}.csv")
    seg_of = c.set_index("ID").seg
    E = E.merge(z, on="ID", how="left")
    # family -> segment (all members, any tier)
    zz = z[z.family_id >= 0].copy()
    zz["seg"] = zz.ID.map(seg_of)
    fseg = zz.dropna(subset=["seg"]).groupby("family_id").seg.agg(lambda s: s.value_counts().index[0])
    fam["seg"] = fam.family_id.map(fseg)
    dbm["seg"] = dbm.event_id.map(seg_of)
    dseg = dbm.dropna(subset=["seg"]).groupby("cluster").seg.agg(lambda s: s.value_counts().index[0])
    dbc["seg"] = dbc.cluster.map(dseg)
    # Mc map cells (0.1 deg), mc_best
    mg = pd.read_csv(P.MCMAP / "mc_grid.csv")
    mg = mg[np.isfinite(mg.mc_best)]
    key = lambda lo, la: (np.round(lo * 10).astype(int) * 10000 + np.round(la * 10).astype(int))  # noqa: E731
    mcmap = pd.Series(mg.mc_best.values, index=key(mg.lon.values, mg.lat.values))
    mcmap = mcmap[~mcmap.index.duplicated()]
    E["mc_cell"] = key(E.lon.values, E.lat.values)
    E["mc_map"] = E.mc_cell.map(mcmap)
    summ, _, _ = P.fault_tables()
    sseg = pd.read_csv(P.OUT / "surface_segments.csv")
    zl = np.load(P.BIG / "level_depth_members.npz")
    lon_g = np.arange(-131.75, -120.124, 0.125)
    lat_g = np.round(np.arange(38.9, 50.61, 0.1), 2)
    E["iy"] = np.clip(np.round((E.lat.values - lat_g[0]) / 0.1).astype(int), 0, len(lat_g) - 1)
    E["ix"] = np.clip(np.round((E.lon.values - lon_g[0]) / 0.125).astype(int), 0, len(lon_g) - 1)
    t0 = E.t.min()
    E["td"] = (E.t - t0).dt.total_seconds() / 86400.0
    rows = []
    for seg in P.SEG_ORDER:
        e = E[E.seg == seg]
        if len(e) == 0:
            continue
        r = dict(seg=seg, name=P.SEGMENTS[seg][0], type=P.SEGMENTS[seg][1], n_events=len(e),
                 n_with_M=int(e.M.notna().sum()), n_depth_good=int(e.depth_good.sum()),
                 Mmax=float(e.M.max()), n_M4=int((e.M >= 4).sum()),
                 first=str(e.t.min().date()), last=str(e.t.max().date()))
        m = e.M.dropna().values
        mc = mc_maxc(m)
        b, nb = b_aki(m, mc) if np.isfinite(mc) else (np.nan, 0)
        r.update(Mc=mc, b=b, n_above_Mc=nb)
        if len(m) >= 100:
            bs, ms = [], []
            for _ in range(NB):
                mm = RNG.choice(m, len(m))
                q = mc_maxc(mm)
                ms.append(q)
                bs.append(b_aki(mm, q)[0])
            r["Mc_lo"], r["Mc_hi"] = ci(ms)
            r["b_lo"], r["b_hi"] = ci(bs)
        r["Mc_map_median"] = float(e.mc_map.median()) if e.mc_map.notna().any() else np.nan
        r["Mc_map_p90"] = float(e.mc_map.quantile(0.9)) if e.mc_map.notna().any() else np.nan
        # clustering (events in the zbz analysis set)
        a = e[e.in_analysis.eq(True)]
        r["n_zbz"] = len(a)
        if len(a) >= 30:
            cls = a.cls.fillna("background")
            for k, lab in (("background", "bg"), ("mainshock-aftershock", "MA"), ("swarm", "swarm"),
                           ("multiplet", "multiplet")):
                r[f"pct_{lab}"] = 100.0 * float((cls == k).mean())
            # unit bootstrap: background events are single units, families whole units
            unit = np.where(a.family_id.fillna(-1).values >= 0, a.family_id.fillna(-1).values.astype(int),
                            -1 - np.arange(len(a)))
            uu, inv = np.unique(unit, return_inverse=True)
            cl = cls.values
            isbg = np.bincount(inv, weights=(cl == "background").astype(float))
            issw = np.bincount(inv, weights=(cl == "swarm").astype(float))
            isma = np.bincount(inv, weights=(cl == "mainshock-aftershock").astype(float))
            nn = np.bincount(inv).astype(float)
            bb, ss, mm_ = [], [], []
            for _ in range(NB):
                k = RNG.integers(0, len(uu), len(uu))
                bb.append(100 * isbg[k].sum() / nn[k].sum())
                ss.append(100 * issw[k].sum() / nn[k].sum())
                mm_.append(100 * isma[k].sum() / nn[k].sum())
            r["pct_bg_lo"], r["pct_bg_hi"] = ci(bb)
            r["pct_swarm_lo"], r["pct_swarm_hi"] = ci(ss)
            r["pct_MA_lo"], r["pct_MA_hi"] = ci(mm_)
            clu = a[cls != "background"]
            r["swarm_share_of_clustered"] = 100.0 * float((clu.cls == "swarm").mean()) if len(clu) else np.nan
        f = fam[fam.seg == seg]
        for t in ("mainshock-aftershock", "swarm", "multiplet"):
            r[f"n_fam_{t}"] = int((f.type == t).sum())
        sw = f[(f.type == "swarm") & (f.n >= 10)]
        mig = sw.migrating.astype(str).eq("True") | sw.mig_migrating.astype(str).eq("True")
        r["n_swarm_fam_ge10"] = len(sw)
        r["n_swarm_fam_migrating"] = int(mig.sum())
        d = dbc[dbc.seg == seg]
        r["n_dbscan_swarmlike"] = int(d.swarm_like.astype(str).eq("True").sum())
        r["n_dbscan_migrating"] = int((d.swarm_like.astype(str).eq("True")
                                       & d.mig_migrating.astype(str).eq("True")).sum())
        r["largest_family_share"] = (float(f.n_all.max()) / len(e)) if len(f) else 0.0
        # depth
        g = e[e.depth_good]
        sz = float(np.median(g.sig_z)) if len(g) else np.nan
        r["depth_class"] = ("none" if len(g) < 50 else "good" if sz <= 3.0 else "fair" if sz <= 6.0 else "poor")
        if len(g) >= 20:
            v = g.bsf.values
            r["bsf_p10"], r["bsf_med"], r["bsf_p90"] = (float(np.percentile(v, q)) for q in (10, 50, 90))
            bm = [np.median(RNG.choice(v, len(v))) for _ in range(NB)]
            b9 = [np.percentile(RNG.choice(v, len(v)), 90) for _ in range(NB)]
            r["bsf_med_lo"], r["bsf_med_hi"] = ci(bm)
            r["bsf_p90_lo"], r["bsf_p90_hi"] = ci(b9)
            r["depth_med"], r["depth_p90"] = float(np.median(g.depth)), float(np.percentile(g.depth, 90))
            r["sig_z_med_good"] = float(np.median(g.sig_z))
            for tag in ("z8", "z75"):
                h = g[np.isfinite(g[tag])]
                r[f"n_{tag}"] = len(h)
                r[f"frac_with_{tag}"] = len(h) / len(g)
                if len(h) >= 20:
                    r[f"{tag}_bsf_med"] = float(np.median(h[f"{tag}_bsf"]))
                    r[f"rel_{tag}_med"] = float(np.median(h[f"rel_{tag}"]))
                    r[f"frac_below_{tag}"] = float((h[f"rel_{tag}"] > 0).mean())
                    fb = [np.mean(RNG.choice(h[f"rel_{tag}"].values, len(h)) > 0) for _ in range(NB)]
                    r[f"frac_below_{tag}_lo"], r[f"frac_below_{tag}_hi"] = ci(fb)
                    # structure uncertainty: the same share with each member's level depth (raw member spread)
                    zm = zl[tag][:, g.iy.values, g.ix.values]                # [member, event]
                    dv = g.depth.values
                    fm = np.array([np.mean(dv[np.isfinite(z)] > z[np.isfinite(z)]) if np.isfinite(z).sum() >= 20
                                   else np.nan for z in zm])
                    r[f"frac_below_{tag}_mem_p05"] = float(np.nanpercentile(fm, 5))
                    r[f"frac_below_{tag}_mem_p95"] = float(np.nanpercentile(fm, 95))
                    r[f"{tag}_member_sd_med_km"] = float(np.nanmedian(np.nanstd(zm, axis=0)))
            s = g[np.isfinite(g.rel_slab)]
            if len(s) >= 20:
                r["rel_slab_med"] = float(np.median(s.rel_slab))
                r["frac_within5_slab"] = float((s.rel_slab.abs() < 5).mean())
        else:
            # unconstrained depths: report the raw numbers, marked
            r["bsf_med_raw"] = float(np.median(e.bsf))
        r["sig_z_med"] = float(np.median(e.sig_z))
        # fault surfaces
        s = sseg[sseg.seg == seg]
        r["n_surfaces"] = len(s)
        if len(s):
            w = s.n_events.values
            r["n_surfaces_stable"] = int(s.stable.astype(str).eq("True").sum())
            r["dip_wmed"] = float(np.average(s.dip.values, weights=w))
            r["dip_min"], r["dip_max"] = float(s.dip.min()), float(s.dip.max())
            r["width_med_km"] = float(np.median(s.width_km))
            sw_ = (s.events_depth_ok_98pct_km - s.events_depth_ok_2pct_km)
            r["seis_width_med_km"] = float(np.nanmedian(sw_)) if sw_.notna().any() else np.nan
            r["seis_bottom_med_km"] = float(np.nanmedian(s.events_depth_ok_98pct_km))
            r["length_sum_km"] = float(s.length_km.sum())
            r["frac_events_on_surfaces"] = float(e.ID.isin(
                pd.read_csv(P.FLT / "fault_surfaces_v1_members.csv").query("surface_id in @s.surface_id").event_id
            ).mean())
        # temporal
        if np.isfinite(mc):
            ea = e[e.M >= mc]
            yrs = max((ea.t.max() - ea.t.min()).days / 365.25, 0.1) if len(ea) > 1 else np.nan
            r["rate_per_yr_geMc"] = len(ea) / yrs if len(ea) > 1 else np.nan
            r["cov_all"], r["burst_all"] = interevent_stats(ea.td)
            bg = ea[ea.cls.fillna("background").eq("background")]
            r["cov_bg"], r["burst_bg"] = interevent_stats(bg.td)
            day = np.floor(ea.td).astype(int)
            cnt = day.value_counts().sort_values(ascending=False)
            ndays = int(np.ceil(ea.td.max() - ea.td.min())) + 1 if len(ea) else 0
            top = max(1, int(round(0.01 * ndays)))
            r["share_busiest_1pct_days"] = float(cnt.iloc[:top].sum() / len(ea)) if len(ea) else np.nan
            r["years_with_events"] = int(ea.t.dt.year.nunique())
        rows.append(r)
        P.log(seg, len(e), round(mc, 2) if np.isfinite(mc) else mc, round(b, 2) if np.isfinite(b) else b)
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT / "seismicity_metrics.csv", index=False, float_format="%.4g")
    P.log(out[["seg", "n_events", "Mc", "b", "b_lo", "b_hi", "pct_bg", "pct_MA", "pct_swarm", "bsf_med", "bsf_p90",
               "frac_below_z8", "dip_wmed", "cov_all"]].to_string())


if __name__ == "__main__":
    main()
