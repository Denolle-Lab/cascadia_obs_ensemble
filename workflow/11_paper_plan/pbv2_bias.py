#!/usr/bin/env python3
"""Plate boundaries v2 (D-B116), step 1: offsets of the seismicity axis from PB2002 and from the bathymetric axis per
section, and the location-bias tests of RULES.md section 7 (pre-registered).

Offsets: per 5 km bin, the S0 pick minus the PB2002 trace (the PB2002 point of the bin nearest to the pick) and minus
the bathymetric candidate; section and segment offset O = median over bins; 95 % interval from the 300 day-block
bootstrap picks of pbv2_fit.py.
Bias tests (paired by bin: a subset's pick minus the all-event pick in the same bin, median over common bins):
* versions: the same S0 events located in v4.1, v5.1, v6.0 (compare_catalogs.py tables);
* geometry: gap <= 180 deg and dist_min <= 15 km vs the rest; Cascadia Initiative years;
* common events (|dt| <= 3 s, <= 50 km): across-strike shift ours minus theirs, against ComCat/ANSS, Lange et al.
  2026 (absolute, relative), Zhou et al. 2024, Niksejel & Zhang 2026; median, day-block 95 % interval;
* independent catalogs' own seismicity axis (their picks, >= 10 events per bin) minus PB2002 and minus ours; also
  Liu et al. 2026 (2016 and 2024 Mendocino sequences, other years) and Gong & Lee 2025 (triple-junction area).
Verdict (RULES 7): B = max(B_pair, B_ver, B_geo); "not location bias" if |O| >= 2B and an independent catalog's own
offset has the same sign; "partly" if |O| >= B; else "bias cannot be excluded".

Outputs (OUT): offsets.csv, bias_tests.csv, verdict.csv, residual_geometry.csv. Usage: python -I pbv2_bias.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbv2_common as C  # noqa: E402
from pbv2_common import SECTIONS, log  # noqa: E402

OBS_BASED = ("Lange 2026 abs", "Lange 2026 rel", "Zhou 2024", "Niksejel 2026")


def ref_per_bin(sec, pk, rs, ru):
    """Per bin, the reference point (s, u) of the bin nearest in u to the pick (NaN where none)."""
    e = sec.edges
    out = np.full(len(pk), np.nan)
    ib = np.searchsorted(e, rs, side="right") - 1
    for i in range(len(pk)):
        m = ib == i
        if m.any() and np.isfinite(pk[i]):
            out[i] = ru[m][np.argmin(np.abs(ru[m] - pk[i]))]
    return out


def med_ci(x, boot=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if boot is None:
        return float(np.median(x)), np.nan, np.nan, len(x)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi), len(x)


# ------------------------------------------------------------------ external catalogs
def _tparse(x: pd.Series) -> pd.Series:
    """'YYYY-MM-DDTHH:MM:SS.ff' with seconds possibly '60.00' (Lange et al. pair tables)."""
    x = x.astype(str)
    base = pd.to_datetime(x.str.slice(0, 16), utc=True, format="%Y-%m-%dT%H:%M")
    return base + pd.to_timedelta(x.str.slice(17).astype(float), unit="s")


def load_external():
    E = C.EXT
    out = {}
    a = pd.read_csv(C.ANSS, usecols=["time", "latitude", "longitude", "horizontalError", "net"])
    out["ComCat/ANSS"] = pd.DataFrame(dict(t=pd.to_datetime(a.time, utc=True), lat=a.latitude, lon=a.longitude,
                                           herr=a.horizontalError))
    z = pd.read_csv(E / "pbv2_zhou2024_blanco/Blanco_EQ_2012to2013.csv")
    out["Zhou 2024"] = pd.DataFrame(dict(t=pd.to_datetime(z.Date + "T" + z.time, utc=True), lat=z.latitude,
                                         lon=z.longitude))
    n = pd.read_csv(E / "pbv2_niksejel2026_offcasc/cascadia_catalog_Niksejel_Zhang_2026.csv")
    out["Niksejel 2026"] = pd.DataFrame(dict(t=pd.to_datetime(n.iloc[:, 0].str.replace("/", "-") + "T" + n.iloc[:, 1],
                                                              utc=True), lat=n.iloc[:, 2], lon=n.iloc[:, 3]))
    g = pd.read_csv(E / "pbv2_gong2025_mtj/growclust_catalog.txt")
    out["Gong 2025 (land, MTJ)"] = pd.DataFrame(dict(t=pd.to_datetime(g.Time, utc=True, format="ISO8601"), lat=g.Lat, lon=g.Lon))
    for k, f in (("Liu 2026 (2024 seq.)", "Dataset_S1.dat"), ("Liu 2026 (2016 seq.)", "Dataset_S2.dat")):
        d = pd.read_csv(E / "pbv2_liu2026_mendocino_otf" / f, sep=r"\s+", skiprows=1, header=None,
                        names=["t", "lat", "lon", "dep", "mag"])
        out[k] = pd.DataFrame(dict(t=pd.to_datetime(d.t.str.replace("/", "-"), utc=True, format="ISO8601"), lat=d.lat, lon=d.lon))
    for k in ("abs", "rel"):
        p = pd.read_csv(C.LANGE / f"blanco_lange2026_{k}_pairs.csv")
        out[f"Lange 2026 {k}"] = pd.DataFrame(dict(ID=p.ID, t=_tparse(p.ref_time), lat=p.ref_lat,
                                                   lon=p.ref_lon))
    return out


def match(ours, ext, dt=3.0, dmax=50.0):
    """Index into ext of the event nearest in time (|dt| <= 3 s) and <= 50 km, per row of ours (or -1)."""
    te = ext.t.astype("int64").to_numpy() / 1e9
    o = np.argsort(te)
    te = te[o]
    to = ours.t.astype("int64").to_numpy() / 1e9
    i0 = np.searchsorted(te, to - dt)
    i1 = np.searchsorted(te, to + dt)
    best = np.full(len(to), -1)
    bdt = np.full(len(to), np.inf)
    for k in range(int((i1 - i0).max()) if len(to) else 0):
        j = np.minimum(i0 + k, len(te) - 1)
        ok = (i0 + k) < i1
        jj = o[j]
        d = C.P.hav(ours.lat.values, ours.lon.values, ext.lat.values[jj], ext.lon.values[jj])
        ddt = np.abs(to - te[j])
        better = ok & (d <= dmax) & (ddt < bdt)
        best[better], bdt[better] = jj[better], ddt[better]
    return best


# ------------------------------------------------------------------ main
def main():
    with open(C.BIG / "work" / "fit.pkl", "rb") as fh:
        F = pickle.load(fh)
    res = F["res"]
    cat = C.load_v61().set_index("ID", drop=False)
    rng = np.random.default_rng(C.SEED + 1)
    ext = load_external()
    rows_off, rows_t, rows_v, rows_r = [], [], [], []
    for code, sec in SECTIONS.items():
        r = res[code]
        pk, B = r["pk"], r["B"]
        ev = cat.loc[r["ids"]]
        ps_, pu_ = r["ps"], r["pu"]
        pb = ref_per_bin(sec, pk, ps_, pu_)
        ba = ref_per_bin(sec, pk, r["bs"], r["bu"])
        L = r["lines"]
        sc = sec.centres
        line = np.full(len(sc), np.nan)
        for ln in L:
            m = (sc >= ln["s0"]) & (sc <= ln["s1"])
            line[m] = ln["a"] + ln["b"] * (sc[m] - ln["sm"])
        use = r["keep"]
        npk = int(np.isfinite(np.where(use, pk, np.nan)).sum())

        def offs(ref, mask):
            d = np.where(use & mask, pk - ref, np.nan)
            bt = np.array([np.nanmedian(np.where(use & mask, B[b] - ref_per_bin(sec, B[b], *refsrc[1]), np.nan))
                           if refsrc[0] else np.nanmedian(np.where(use & mask, B[b] - ref, np.nan))
                           for b in range(0, len(B), 3)]) if np.isfinite(d).sum() else None
            return med_ci(d, bt)
        spans = [("section", np.ones(len(sc), bool))] + [(f"segment {ln['k']}", (sc >= ln["s0"]) & (sc <= ln["s1"])) for ln in L]
        for lab, msk in spans:
            row = dict(section=code, name=sec.name, span=lab, n_bins=int((use & msk & np.isfinite(pk)).sum()),
                       mode=r["mode"])
            for nm, ref, src in (("pb2002", pb, (True, (ps_, pu_))), ("bathy", ba, (True, (r["bs"], r["bu"]))),
                                 ("proposal", line, (False, None))):
                refsrc = src
                o, lo, hi, n = offs(ref, msk)
                row.update({f"O_{nm}": o, f"O_{nm}_lo": lo, f"O_{nm}_hi": hi})
            # proposal lines vs PB2002 (all spans, also bathymetry-only sections)
            m2 = (ps_ >= sc[msk].min() - C.BIN / 2) & (ps_ <= sc[msk].max() + C.BIN / 2) if msk.any() else ps_ < -1e9
            lp = np.full(m2.sum(), np.nan)
            for ln in L:
                mm = (ps_[m2] >= ln["s0"]) & (ps_[m2] <= ln["s1"])
                lp[mm] = ln["a"] + ln["b"] * (ps_[m2][mm] - ln["sm"])
            dd = lp - pu_[m2]
            row.update(proposal_minus_pb2002_median=float(np.nanmedian(dd)) if np.isfinite(dd).any() else np.nan,
                       proposal_minus_pb2002_p10=float(np.nanpercentile(dd, 10)) if np.isfinite(dd).any() else np.nan,
                       proposal_minus_pb2002_p90=float(np.nanpercentile(dd, 90)) if np.isfinite(dd).any() else np.nan)
            # S2 variant offset from PB2002 (sparse sections)
            pb2 = ref_per_bin(sec, r["pk2"], ps_, pu_)
            d2 = np.where(msk, r["pk2"] - pb2, np.nan)
            row.update(O_pb2002_S2=float(np.nanmedian(d2)) if np.isfinite(d2).any() else np.nan,
                       n_bins_S2=int(np.isfinite(d2).sum()))
            rows_off.append(row)
        if npk < 5:
            log(f"{code}: {npk} S0 picks, bias tests skipped")
            continue
        s_all, u_all = r["s"], r["u"]

        def sub_picks(mask):
            p, _ = C.bin_picks(sec, s_all[mask], u_all[mask])
            return p

        def paired(p):
            d = np.where(use, p - pk, np.nan)
            return float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan, int(np.isfinite(d).sum())
        # versions (same events)
        vp = C.version_positions(r["ids"])
        for v, t in vp.items():
            ok = t.located.fillna(False).values.astype(bool) & np.isfinite(t.lat.values)
            sv, uv = sec.su(np.where(ok, t.lon.values, 0), np.where(ok, t.lat.values, 0))
            m = ok & (sv >= 0) & (sv < sec.length) & (uv >= sec.uwin[0]) & (uv <= sec.uwin[1])
            p, _ = C.bin_picks(sec, sv[m], uv[m])
            d, nb = paired(p)
            o = np.where(use, p - ref_per_bin(sec, p, ps_, pu_), np.nan)
            rows_t.append(dict(section=code, test="version", label=v, D_km=d, n_bins=nb,
                               O_pb2002=float(np.nanmedian(o)) if np.isfinite(o).any() else np.nan))
        # geometry
        good = (ev.gap.values <= 180) & (ev.dist_min.values <= 15)
        for lab, m in (("gap<=180 & dist_min<=15 km", good), ("gap>180 or dist_min>15 km", ~good)):
            d, nb = paired(sub_picks(m))
            rows_t.append(dict(section=code, test="geometry", label=lab, D_km=d, n_bins=nb, n_events=int(m.sum())))
        m = (ev.h_max.values <= 1.5) & (ev.ell_len3.values <= 6.0)
        d, nb = paired(sub_picks(m))
        rows_t.append(dict(section=code, test="selection", label="S1 strict (h_max <= 1.5, ell_len3 <= 6 km)", D_km=d,
                           n_bins=nb, n_events=int(m.sum())))
        for y in C.CI_YEARS:
            m = ev.ci_year.values == y
            d, nb = paired(sub_picks(m))
            rows_t.append(dict(section=code, test="CI year", label=y, D_km=d, n_bins=nb, n_events=int(m.sum())))
        # residuals from the proposal line vs geometry
        lu = np.full(len(s_all), np.nan)
        for ln in L:
            mm = (s_all >= ln["s0"]) & (s_all <= ln["s1"])
            lu[mm] = ln["a"] + ln["b"] * (s_all[mm] - ln["sm"])
        res_u = u_all - lu
        near = np.abs(res_u) <= 10
        for var, edges in (("gap", [0, 90, 135, 180, 225, 360]), ("dist_min", [0, 5, 10, 20, 40, 200])):
            x = ev[var].values
            for a, b in zip(edges[:-1], edges[1:]):
                m = near & (x >= a) & (x < b)
                if m.sum() >= 20:
                    rows_r.append(dict(section=code, var=var, lo=a, hi=b, n=int(m.sum()),
                                       median_residual_km=float(np.median(res_u[m])),
                                       mad_km=float(np.median(np.abs(res_u[m] - np.median(res_u[m]))))))
        # common events and own axes of the independent catalogs
        ours = pd.DataFrame(dict(t=ev.t.values, lat=ev.lat.values, lon=ev.lon.values, day=ev.day.values,
                                 ID=ev.ID.values))
        ours["t"] = pd.to_datetime(ours.t, utc=True)
        for name, x in ext.items():
            se, ue = sec.su(x.lon.values, x.lat.values)
            inwin = (se >= 0) & (se < sec.length) & (ue >= sec.uwin[0]) & (ue <= sec.uwin[1])
            row = dict(section=code, test="independent catalog", label=name, n_events_window=int(inwin.sum()))
            if name.startswith("Lange"):
                j = pd.Series(np.arange(len(x)), index=x.ID.values)
                k = ours.ID.map(j).fillna(-1).astype(int).values
            elif name.startswith(("Liu", "Gong")):
                k = np.full(len(ours), -1)
            else:
                k = match(ours, x)
            ok = k >= 0
            if ok.sum() >= 10:
                so, uo = sec.su(ours.lon.values[ok], ours.lat.values[ok])
                sx, ux = sec.su(x.lon.values[k[ok]], x.lat.values[k[ok]])
                du, ds = uo - ux, so - sx
                codes, uniq = pd.factorize(ours.day.values[ok])
                bt = []
                for _ in range(500):
                    w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
                    bt.append(np.median(np.repeat(du, w)) if w.sum() else np.nan)
                lo, hi = np.nanpercentile(bt, [2.5, 97.5])
                row.update(n_pairs=int(ok.sum()), du_median_km=float(np.median(du)), du_lo=float(lo), du_hi=float(hi),
                           ds_median_km=float(np.median(ds)), dh_median_km=float(np.median(np.hypot(du, ds))))
                if "herr" in x:
                    row["their_herr_median_km"] = float(np.nanmedian(x.herr.values[k[ok]]))
            if inwin.sum() >= 10:
                p, _ = C.bin_picks(sec, se[inwin], ue[inwin])
                o = p - ref_per_bin(sec, p, ps_, pu_)
                d_ours = p - pk
                d_prop = p - line
                row.update(own_n_bins=int(np.isfinite(p).sum()),
                           own_O_pb2002=float(np.nanmedian(o)) if np.isfinite(o).any() else np.nan,
                           own_minus_ours=float(np.nanmedian(d_ours)) if np.isfinite(d_ours).any() else np.nan,
                           own_minus_proposal=float(np.nanmedian(d_prop)) if np.isfinite(d_prop).any() else np.nan)
            rows_t.append(row)
    off = pd.DataFrame(rows_off)
    tests = pd.DataFrame(rows_t)
    off.to_csv(C.OUT / "offsets.csv", index=False, float_format="%.3f")
    tests.to_csv(C.OUT / "bias_tests.csv", index=False, float_format="%.3f")
    pd.DataFrame(rows_r).to_csv(C.OUT / "residual_geometry.csv", index=False, float_format="%.3f")
    verdict(off, tests).to_csv(C.OUT / "verdict.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    log(off.to_string())
    log(tests.to_string())
    log(pd.read_csv(C.OUT / "verdict.csv").to_string())


def verdict(off, tests):
    rows = []
    for code in off.section.unique():
        o = off[(off.section == code) & (off.span == "section")].iloc[0]
        t = tests[tests.section == code]
        if t.empty or not np.isfinite(o.O_pb2002):
            rows.append(dict(section=code, O_pb2002=o.O_pb2002, verdict="not tested (no seismicity axis)"))
            continue
        ind = t[(t.test == "independent catalog") & t.label.isin(OBS_BASED) & (t.get("n_pairs", 0).fillna(0) >= 30)]
        b_pair = float(ind.du_median_km.abs().max()) if len(ind) else np.nan
        v = t[t.test == "version"]
        b_ver = float(v.D_km.max() - v.D_km.min()) if v.D_km.notna().any() else np.nan
        g = t[(t.test == "geometry") & t.label.str.startswith("gap<=")]
        y = t[(t.test == "CI year") & (t.n_bins >= 5)]
        b_geo = float(np.nanmax(np.r_[g.D_km.abs().values, y.D_km.abs().values])) if (len(g) or len(y)) else np.nan
        B = float(np.nanmax([b_pair, b_ver, b_geo]))
        own = t[(t.test == "independent catalog") & (t.own_n_bins.fillna(0) >= 5)] if "own_n_bins" in t else t.iloc[:0]
        own_ind = own[~own.label.str.startswith("ComCat")]
        same_sign = bool((np.sign(own_ind.own_O_pb2002) == np.sign(o.O_pb2002)).any()) if len(own_ind) else False
        if abs(o.O_pb2002) >= 2 * B and same_sign:
            vd = "offset is not location bias"
        elif abs(o.O_pb2002) >= B:
            vd = "partly"
        else:
            vd = "bias cannot be excluded"
        rows.append(dict(section=code, O_pb2002=o.O_pb2002, O_lo=o.O_pb2002_lo, O_hi=o.O_pb2002_hi,
                         O_bathy=o.O_bathy, B_pair=b_pair, B_ver=b_ver, B_geo=b_geo, B=B,
                         independent_same_sign=same_sign,
                         independent_own_offsets="; ".join(f"{a}: {b:+.1f} km ({int(n)} bins)" for a, b, n in
                                                           zip(own_ind.label, own_ind.own_O_pb2002, own_ind.own_n_bins)),
                         verdict=vd))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
