"""A3a: frame transfer. Keep the v4 (GraphDD) relative geometry, take the absolute frame from Track A (NLL 3D).

For every event, the shift field is the median NLL-minus-v4 shift (east, north, depth) of the NLL-located events
around it (in v4 coordinates): up to K nearest within a radius R, at least NMIN of them. The transferred location is
v4 + field. Events without enough neighbours keep their NLL 3D location (or the v4 location for tier U, which NLL did
not locate). Tier U events with neighbours get the field too.

The idea: the smooth part of the shift is the velocity-model (1D -> 3D) correction; the event-by-event remainder is
mostly NLL single-event error, and dropping it restores the double-difference tightness of v4.

Radius test: the script loops over radii and depth columns (NLL maximum-likelihood `depth` and expectation
`exp_depth`) and writes, for each, the catalog (to /wd1) and QC tables (to data/reloc_v5/a3_frame_transfer):
  metrics.csv      local thickness, length, depth IQR in 1 km columns, NN distance, above-surface share, median depth
                   for v4, NLL and the transfer, by region and ALL (NLL-located events only, so all three compare)
  consistency.csv  share of transferred events inside the NLL uncertainty (|dz| <= sig_z, dh <= h_max), by tier
  field.csv        variance of the shift explained by the field, effective radius, coverage
  pinning.csv      share of events in the most common 0.01 km depth (grid-node pile-up check), by region
Results are appended after each radius, so a partial run still leaves usable tables.

Run (detached):  see the A3a section of README.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_EARTH = 6371.0
KM_DEG = 111.19
REPO = Path(__file__).resolve().parents[2]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ecef(lat, lon, z):
    """Cartesian km; z is km below sea level."""
    la, lo = np.radians(lat), np.radians(lon)
    r = R_EARTH - np.asarray(z, float)
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)]


def thickness(x, k=20):
    """sqrt of smallest and largest covariance eigenvalue of each event's k nearest neighbours (km)."""
    _, ix = cKDTree(x).query(x, k=k)
    th, ln = np.empty(len(x)), np.empty(len(x))
    for s in range(0, len(x), 50000):
        p = x[ix[s:s + 50000]]
        p = p - p.mean(axis=1, keepdims=True)
        ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", p, p) / (k - 1))
        th[s:s + 50000] = np.sqrt(np.clip(ev[:, 0], 0, None))
        ln[s:s + 50000] = np.sqrt(ev[:, 2])
    return th, ln


def zspread(lat, lon, z, r=1.0, step=10):
    """median depth IQR of events within r km horizontally (columns with >= 10 events, every step-th event)."""
    h = ecef(lat, lon, np.zeros(len(lat)))
    t = cKDTree(h)
    out = []
    for i in range(0, len(h), step):
        ix = t.query_ball_point(h[i], r=r)
        if len(ix) >= 10:
            q = np.quantile(z[ix], [0.25, 0.75])
            out.append(q[1] - q[0])
    return float(np.median(out)) if out else np.nan


def geom_metrics(lat, lon, z, surface_km):
    x = ecef(lat, lon, z)
    nn = cKDTree(x).query(x, k=2)[0][:, 1]
    th, ln = thickness(x)
    return dict(n=len(z), thick=np.median(th), length=np.median(ln), zIQR_1km=zspread(lat, lon, z),
                nn=np.median(nn), above_surface=np.mean(z < -surface_km), depth_med=np.median(z))


def pinning(z, region):
    rows = []
    for reg, g in pd.Series(np.round(z, 2)).groupby(region.values):
        vc = g.value_counts()
        rows.append(dict(region=reg, n=len(g), top1_share=vc.iloc[0] / len(g), top1_depth=vc.index[0],
                         top3_share=vc.iloc[:3].sum() / len(g)))
    return pd.DataFrame(rows)


def field(df, zcol, R, K, nmin, field_mask):
    """median shift of up to K field events within R km of each event (v4 coordinates)."""
    shift = np.c_[(df.lon - df.v4_lon) * KM_DEG * np.cos(np.radians(df.v4_lat)),
                  (df.lat - df.v4_lat) * KM_DEG,
                  df[zcol] - df.v4_depth][field_mask]
    xv = ecef(df.v4_lat.values, df.v4_lon.values, df.v4_depth.values)
    tree = cKDTree(xv[field_mask])
    nf = field_mask.sum()
    out = np.full((len(df), 3), np.nan)
    cnt = np.zeros(len(df), int)
    reff = np.full(len(df), np.nan)
    for s in range(0, len(df), 20000):
        d, ix = tree.query(xv[s:s + 20000], k=min(K, nf), distance_upper_bound=R)
        ok = np.isfinite(d)
        c = ok.sum(1)
        sh = shift[np.where(ok, ix, 0)]
        sh[~ok] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(sh, axis=1)
        use = c >= nmin
        out[s:s + 20000][use] = med[use]
        cnt[s:s + 20000] = c
        reff[s:s + 20000] = np.where(c > 0, np.nanmax(np.where(ok, d, np.nan), axis=1), np.nan)
    return out, cnt, reff, shift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="/wd1/mdenolle_data/reloc_v5/catalog/catalog_v5_trackA_iter1_vpvs176.csv")
    ap.add_argument("--radii", type=float, nargs="+", default=[1, 2, 3, 5, 7.5, 10, 15])
    ap.add_argument("--zcols", nargs="+", default=["depth", "exp_depth"])
    ap.add_argument("--k", type=int, default=300, help="max neighbours per field estimate")
    ap.add_argument("--nmin", type=int, default=10)
    ap.add_argument("--field-tiers", default="ABCD", help="tiers of NLL events used to build the field")
    ap.add_argument("--outdir", default="/wd1/mdenolle_data/reloc_v5/frame_transfer")
    ap.add_argument("--qcdir", default=str(REPO / "data/reloc_v5/a3_frame_transfer"))
    a = ap.parse_args()

    out, qc = Path(a.outdir), Path(a.qcdir)
    out.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.csv", "consistency.csv", "field.csv", "pinning.csv"):
        (qc / f).unlink(missing_ok=True)
    (qc / "run_meta.json").write_text(json.dumps(vars(a) | {"started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    df = pd.read_csv(a.catalog)
    log(f"catalog {len(df)} events; loc_source {df.loc_source.value_counts().to_dict()}")
    nll = (df.loc_source == "nll3d").values
    fmask = nll & df.tier.isin(list(a.field_tiers)).values
    sub = df[nll]
    surf = sub.surface_km.values

    def append(name, frame):
        p = qc / name
        frame.to_csv(p, mode="a", header=not p.exists(), index=False)

    # pinning of the NLL depths themselves and of v4
    for zc in ["v4_depth"] + a.zcols:
        append("pinning.csv", pinning(sub[zc].values, sub.region).assign(depth=zc))

    def region_metrics(lat, lon, z, tag, zc, R):
        rows = [dict(region="ALL", **geom_metrics(lat, lon, z, surf))]
        for reg, ix in sub.groupby("region").indices.items():
            if len(ix) >= 2000:
                rows.append(dict(region=reg, **geom_metrics(lat[ix], lon[ix], z[ix], surf[ix])))
        return pd.DataFrame(rows).assign(variant=tag, zcol=zc, R_km=R)

    append("metrics.csv", region_metrics(sub.v4_lat.values, sub.v4_lon.values, sub.v4_depth.values, "v4", "-", np.nan))
    log("v4 metrics done")
    for zc in a.zcols:
        append("metrics.csv", region_metrics(sub.lat.values, sub.lon.values, sub[zc].values, "nll3d", zc, np.nan))
        log(f"nll3d metrics done ({zc})")
        for R in a.radii:
            fld, cnt, reff, shift = field(df, zc, R, a.k, a.nmin, fmask)
            has = ~np.isnan(fld[:, 0])
            lat = np.where(has, df.v4_lat + fld[:, 1] / KM_DEG, np.where(nll, df.lat, df.v4_lat))
            lon = np.where(has, df.v4_lon + fld[:, 0] / (KM_DEG * np.cos(np.radians(df.v4_lat))),
                           np.where(nll, df.lon, df.v4_lon))
            z = np.where(has, df.v4_depth + fld[:, 2], np.where(nll, df[zc], df.v4_depth))
            src = np.where(has, "xfer", np.where(nll, "nll3d", "v4"))
            cat = pd.DataFrame(dict(ID=df.ID, lat=lat.round(5), lon=lon.round(5), depth=z.round(3), src=src,
                                    n_field=cnt, r_eff_km=np.round(reff, 2), fE=fld[:, 0].round(3),
                                    fN=fld[:, 1].round(3), fZ=fld[:, 2].round(3), tier=df.tier, region=df.region))
            cat.to_csv(out / f"catalog_xfer_{zc}_R{R:g}.csv.gz", index=False)

            # metrics on NLL-located events
            append("metrics.csv", region_metrics(lat[nll], lon[nll], z[nll], "xfer", zc, R))
            # field statistics
            fs = fld[fmask]
            okf = ~np.isnan(fs[:, 0])
            fr = dict(zcol=zc, R_km=R, coverage_nll=has[nll].mean(), coverage_U=has[~nll].mean() if (~nll).any() else np.nan,
                      n_U_transferred=int((has & ~nll).sum()), r_eff_med=np.nanmedian(reff[has]))
            for j, c in enumerate("ENZ"):
                t, s = shift[okf, j], fs[okf, j]
                fr[f"var_expl_{c}"] = 1 - np.var(t - s) / np.var(t)
                fr[f"abs_shift_med_{c}"] = np.median(np.abs(t))
                fr[f"abs_resid_med_{c}"] = np.median(np.abs(t - s))
            append("field.csv", pd.DataFrame([fr]))
            # consistency with NLL uncertainty
            g = df[nll & has].assign(dz=(z - df[zc])[nll & has],
                                     dh=np.hypot((lon - df.lon) * KM_DEG * np.cos(np.radians(df.lat)),
                                                 (lat - df.lat) * KM_DEG)[nll & has])
            cons = g.groupby("tier").apply(lambda t: pd.Series(dict(
                n=len(t), dz_med=np.median(np.abs(t.dz)), dz_over_sigz_med=np.median(np.abs(t.dz) / t.sig_z),
                in_1sig_z=np.mean(np.abs(t.dz) <= t.sig_z), in_2sig_z=np.mean(np.abs(t.dz) <= 2 * t.sig_z),
                dh_med=np.median(t.dh), in_hmax=np.mean(t.dh <= t.h_max))), include_groups=False).reset_index()
            append("consistency.csv", cons.assign(zcol=zc, R_km=R))
            append("pinning.csv", pinning(z[nll], sub.region).assign(depth=f"xfer_{zc}_R{R:g}"))
            log(f"{zc} R={R:g} km: coverage {has.mean():.1%}, U transferred {fr['n_U_transferred']}, "
                f"var expl Z {fr['var_expl_Z']:.0%}, r_eff {fr['r_eff_med']:.2f} km")
    (qc / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("DONE")


if __name__ == "__main__":
    sys.exit(main())
