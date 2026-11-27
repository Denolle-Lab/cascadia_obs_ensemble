#!/usr/bin/env python3
"""Q1 (D-B115): in-situ Vp/Vs of earthquake families from P and S differential times (Lin & Shearer 2007 style).

Pre-registered (RESUME.md):
* Data: waveform cross-correlation differential times of every v5.1 family (/wd1/mdenolle_data/tomo_v1/cc_dt_all,
  cc_<block>.parquet; accepted rows `above`: cc >= 0.7, cc - cc2 >= 0.1, peak not at the lag edge; dt_cc is clock
  corrected); pairs inside one v5.1 family (same_family) whose events are <= 5 km apart (v5.1 hypocentres, sep_km).
* Per pair: stations with both an accepted P and an accepted S row, >= 3 of them; P and S differential times demeaned
  over those stations (removes the origin-time difference) -> points (dtP', dtS').
* Per family: >= 5 events in contributing pairs and >= 30 points; Vp/Vs = slope through the origin of dtS' on dtP' by
  least absolute deviations (L1; = the |dtP'|-weighted median of dtS'/dtP'); 95 % interval from 200 bootstraps over
  pairs; valid when the half-width is <= 0.10 and the value is within 1.4-2.5 (values outside are kept and flagged).
* Variants: orthogonal (total least squares) slope through the origin on the points within 5 MAD of the L1 line;
  catalog picks (dt_cat of the same parquet rows, any CC status) with the same pair and station rules.
* Family type: the v6.1 family holding most of the v5.1 family's members (carried over when it holds >= 50 %); the
  v5.1 type as a variant. Segment of a family: the segment (D-B113 rule, v6.1 positions) of most of its members.
Writes BIG/vpvs/vpvs_families.csv and OUT/vpvs_families.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q1_common as Q  # noqa: E402

CC = Q.WD1 / "tomo_v1/cc_dt_all"
BLOCKS = ["axial", "blanco", "cascades", "endeavour", "explorer", "forearc", "gorda", "jdf", "mendocino", "msh",
          "onshore", "puget"]
SEP_KM = 5.0
NSTA = 3
NEV, NPTS = 5, 30
NBOOT = 200
HALFW = 0.10
VALID = (1.4, 2.5)


def wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def l1_slope(x, y, w=None):
    ok = np.abs(x) > 0
    ww = np.abs(x[ok]) * (1.0 if w is None else w[ok])
    if ww.sum() <= 0:
        return np.nan
    return wmedian(y[ok] / x[ok], ww)


def tls_slope(x, y):
    S = np.array([[np.sum(x * x), np.sum(x * y)], [np.sum(x * y), np.sum(y * y)]])
    w, v = np.linalg.eigh(S)
    u = v[:, np.argmax(w)]
    return float(u[1] / u[0]) if u[0] != 0 else np.nan


def points(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pair-station points with P and S, >= NSTA stations per pair, demeaned per pair."""
    p = d[d.phase == "P"].groupby(["pair", "sta"])[col].first()
    s = d[d.phase == "S"].groupby(["pair", "sta"])[col].first()
    j = pd.concat([p.rename("dP"), s.rename("dS")], axis=1, join="inner").reset_index()
    n = j.groupby("pair").sta.transform("size")
    j = j[n >= NSTA].copy()
    j["dP"] -= j.groupby("pair").dP.transform("mean")
    j["dS"] -= j.groupby("pair").dS.transform("mean")
    return j


def family_fit(j: pd.DataFrame, pairs: pd.DataFrame, rng) -> dict:
    """j: points of one family (pair, dP, dS); pairs: pair -> ID1, ID2."""
    pr = j.pair.unique()
    ev = np.unique(pairs.loc[pr, ["ID1", "ID2"]].values)
    r = dict(n_pairs=len(pr), n_events=len(ev), n_points=len(j))
    if len(ev) < NEV or len(j) < NPTS:
        return r
    x, y = j.dP.values, j.dS.values
    k = l1_slope(x, y)
    r["vpvs"] = k
    res = y - k * x
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    keep = np.abs(res) <= 5 * mad if mad > 0 else np.ones(len(x), bool)
    r["vpvs_tls"] = tls_slope(x[keep], y[keep])
    r["resid_mad_s"] = float(mad)
    r["dP_iqr_s"] = float(np.subtract(*np.percentile(x, [75, 25])))
    _, inv = np.unique(j.pair.values, return_inverse=True)
    bs = []
    for _ in range(NBOOT):
        cnt = np.bincount(rng.integers(0, len(pr), len(pr)), minlength=len(pr)).astype(float)
        w = cnt[inv]
        if w.sum() == 0:
            continue
        bs.append(l1_slope(x, y, w))
    r["vpvs_lo"], r["vpvs_hi"] = Q.ci(bs)
    return r


def main():
    t00 = time.time()
    rng = np.random.default_rng(Q.SEED)
    z51 = pd.read_csv(Q.WD1 / "catalog_v5/v5.1_analysis/clusters/events_zbz_v51.csv.gz",
                      usecols=["event_id", "family_id", "family_type"])
    z51 = z51[z51.family_id > 0]
    z61, f61 = Q.load_zbz("v61")
    cat = Q.load_catalog("v61")
    rows_cc, rows_cat = [], []
    cols = ["pair", "ID1", "ID2", "family_id", "station", "net", "phase", "above", "dt_cc", "dt_cat"]
    for b in BLOCKS:
        t0 = time.time()
        dset = ds.dataset(CC / f"cc_{b}.parquet")
        tb = dset.to_table(columns=cols, filter=(ds.field("same_family") == True) & (ds.field("sep_km") <= SEP_KM))  # noqa: E712
        d = tb.to_pandas()
        for c_ in ("station", "net", "phase"):
            d[c_] = d[c_].astype(str)
        d["sta"] = d.station + "." + d.net
        pairs = d.groupby("pair")[["ID1", "ID2", "family_id"]].first()
        for col, store, sub in (("dt_cc", rows_cc, d[d.above.astype(bool)]), ("dt_cat", rows_cat, d)):
            j = points(sub[np.isfinite(sub[col])], col)
            j["family_id"] = j.pair.map(pairs.family_id)
            for fid, g in j.groupby("family_id"):
                r = family_fit(g, pairs, rng)
                r.update(fam51=int(fid), block=b)
                store.append(r)
        Q.log(f"{b}: {len(d)} rows, {len(pairs)} pairs, {time.time() - t0:.0f} s")
    V = pd.DataFrame(rows_cc)
    C = pd.DataFrame(rows_cat).add_suffix("_cat").rename(columns={"fam51_cat": "fam51", "block_cat": "block"})
    V = V.merge(C.drop(columns=["block"]), on="fam51", how="outer")
    # validity
    V["halfw"] = (V.vpvs_hi - V.vpvs_lo) / 2
    V["in_range"] = V.vpvs.between(*VALID)
    V["valid"] = V.vpvs.notna() & (V.halfw <= HALFW) & V.in_range
    V["valid_cat"] = V.vpvs_cat.notna() & ((V.vpvs_hi_cat - V.vpvs_lo_cat) / 2 <= HALFW) & V.vpvs_cat.between(*VALID)
    # v5.1 -> v6.1 family match
    m = z51.rename(columns={"event_id": "ID", "family_id": "fam51", "family_type": "type51"}).merge(
        z61[["ID", "family_id"]].rename(columns={"family_id": "fam61"}), on="ID", how="left")
    n51 = m.groupby("fam51").size()
    top = m[m.fam61 > 0].groupby("fam51").fam61.agg(lambda s: s.value_counts().index[0])
    topn = m[m.fam61 > 0].groupby("fam51").fam61.agg(lambda s: s.value_counts().iloc[0])
    t51 = m.groupby("fam51").type51.first()
    F = pd.DataFrame(dict(n51=n51, type51=t51, fam61=top, share61=topn / n51)).reset_index()
    F["carried"] = F.share61 >= 0.5
    ty61 = f61.set_index("family_id").type
    F["type61"] = np.where(F.carried, F.fam61.map(ty61), "")
    seg = cat.set_index("ID").seg
    stype = cat.set_index("ID").seg_type
    m["seg"] = m.ID.map(seg)
    m["seg_type"] = m.ID.map(stype)
    m["lon"], m["lat"] = m.ID.map(cat.set_index("ID").lon), m.ID.map(cat.set_index("ID").lat)
    g = m.dropna(subset=["seg"]).groupby("fam51")
    F["seg"] = F.fam51.map(g.seg.agg(lambda s: s.value_counts().index[0]))
    F["seg_type"] = F.fam51.map(g.seg_type.agg(lambda s: s.value_counts().index[0]))
    F["lon"], F["lat"] = F.fam51.map(g.lon.median()), F.fam51.map(g.lat.median())
    V = F.merge(V, on="fam51", how="left")
    for c_ in ("valid", "valid_cat", "in_range"):
        V[c_] = V[c_].fillna(False).astype(bool)
    V["snr"] = V.dP_iqr_s / V.resid_mad_s
    V = Q.add_cells(V)
    out = Q.BIG / "vpvs"
    out.mkdir(parents=True, exist_ok=True)
    V.to_csv(out / "vpvs_families.csv", index=False, float_format="%.5g")
    V.to_csv(Q.OUT / "vpvs_families.csv", index=False, float_format="%.5g")
    Q.log(f"families {len(V)}; carried to v6.1 {int(V.carried.sum())}; with a CC estimate {int(V.vpvs.notna().sum())}; "
          f"valid {int(V.valid.sum())}; valid catalog-pick {int(V.valid_cat.sum())}; {time.time() - t00:.0f} s")
    Q.log(V[V.valid].groupby(["seg_type", "type61"]).vpvs.describe().round(3).to_string())
    e = V[V.vpvs.notna()].copy()
    e["snr_q"] = pd.qcut(e.snr, 5)
    Q.log("estimate against signal-to-noise (IQR of dtP' / MAD of the L1 residuals):\n" + e.groupby(
        "snr_q", observed=True).agg(n=("vpvs", "size"), l1=("vpvs", "median"), tls=("vpvs_tls", "median"),
                                    halfw=("halfw", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
