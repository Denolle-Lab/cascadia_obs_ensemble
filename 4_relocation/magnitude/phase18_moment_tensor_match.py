"""Phase 18: match ComCat moment tensors to our catalog and test our magnitudes against Mw.

Each ComCat moment tensor (data/focal/comcat_mt.csv, utils/fetch_focal_mechanisms.py)
inside the catalog window is matched to the event of cascadia_catalog_ML_routeA.csv
that minimises |dt|/5 s + distance/20 km among events within 60 s; a match needs
|dt| < 5 s and < 50 km. Writes data/focal/comcat_mt_matched.csv (one row per matched
tensor: ComCat id, tensor Mw and nodal plane, our event_id, offsets, our ML and MW),
which map_context.load_mt reads and phase17 plots (panels e-f).

It then prints the diagnostics behind the high-end magnitude shortfall:
  * our ML and calibrated MW minus tensor Mw, by Mw class and by region;
  * per-pick S station magnitudes (Hutton & Boore 1987 distance correction, no
    station terms) minus the reference magnitude, by hypocentral distance, for the
    tensor events and for the ComCat ML anchors;
  * a test catalog: Hutton-Boore station magnitudes, P and S picks within RMAX km,
    event median, one offset to the ComCat anchors (slope fixed at 1).

Usage (default env, run from 4_relocation/magnitude):
    python phase18_moment_tensor_match.py
    # a variant run (e.g. the 0.5 Hz high-pass test), without touching the paper's files:
    python phase18_moment_tensor_match.py --data ../../data/magnitude_hp05 --suffix _routeA_hp05 \
        --matched ../../data/magnitude_hp05/comcat_mt_matched_routeA_hp05.csv
"""
import argparse
import os
import numpy as np, pandas as pd

D = "../../data/magnitude"
SUFFIX = "_routeA"
FOCAL = "../../data/focal"
MATCHED = f"{FOCAL}/comcat_mt_matched.csv"
TOL_S, TOL_KM, SEARCH_S = 5.0, 50.0, 60.0
RMAX = 150.0                                 # test catalog: picks within this distance


def tensor_mw(mt):
    comp = mt[["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]].to_numpy()
    m0 = np.sqrt((comp[:, :3] ** 2).sum(1) / 2 + (comp[:, 3:] ** 2).sum(1))
    return 2 / 3 * (np.log10(m0) - 9.1)


def hutton_boore(log10a, r):
    """Station ML from Wood-Anderson amplitude (mm) at hypocentral distance r (km)."""
    return log10a + 1.110 * np.log10(r / 100) + 0.00189 * (r - 100) + 3.0


def match(mt, cat):
    t = cat.otime.dt.tz_convert(None).to_numpy()
    la, lo = cat.evla.to_numpy(), cat.evlo.to_numpy()
    rows = []
    for _, r in mt.iterrows():
        dt = (t - r.time.tz_convert(None).to_datetime64()) / np.timedelta64(1, "s")
        near = np.flatnonzero(np.abs(dt) < SEARCH_S)
        if not len(near):
            continue
        km = np.hypot((la[near] - r.lat) * 111.2,
                      (lo[near] - r.lon) * 111.2 * np.cos(np.radians(r.lat)))
        i = np.argmin(np.abs(dt[near]) / TOL_S + km / 20.0)
        if abs(dt[near][i]) < TOL_S and km[i] < TOL_KM:
            j = near[i]
            rows.append(dict(comcat_id=r.id, event_id=int(cat.event_id.iat[j]),
                             dt_s=round(dt[j], 2), dist_km=round(km[i], 1),
                             ddepth_km=round(cat.evdp.iat[j] - r.depth, 1)))
    return pd.DataFrame(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="match ComCat moment tensors; magnitude diagnostics")
    ap.add_argument("--data", default=D, help="magnitude directory (default %(default)s)")
    ap.add_argument("--suffix", default=SUFFIX, help="file suffix (default %(default)s)")
    ap.add_argument("--matched", default=MATCHED, help="output: matched tensors (default %(default)s)")
    return ap.parse_args()


def main():
    global D, SUFFIX, MATCHED
    a = parse_args()
    D, SUFFIX, MATCHED = a.data, a.suffix, a.matched
    cat = pd.read_csv(f"{D}/cascadia_catalog_ML{SUFFIX}.csv")
    cat["otime"] = pd.to_datetime(cat.otime, utc=True, format="ISO8601")
    mw = pd.read_csv(f"{D}/cascadia_catalog{SUFFIX}_calibrated_mw.csv", usecols=["event_id", "MW"])
    mt = pd.read_csv(f"{FOCAL}/comcat_mt.csv")
    mt["time"] = pd.to_datetime(mt.time, utc=True, format="ISO8601")
    mt["Mw_mt"] = tensor_mw(mt).round(2)
    mt["mt_type"] = mt.mt_type.str.split("/").str[-1]
    win = mt[mt.time.between(cat.otime.min(), cat.otime.max())]
    box = win[win.lon.between(cat.evlo.min(), cat.evlo.max())
              & win.lat.between(cat.evla.min(), cat.evla.max())]
    m = match(box, cat)
    out = (box.rename(columns={"id": "comcat_id"})
              [["comcat_id", "time", "lat", "lon", "depth", "mag", "magtype", "mt_type",
                "Mw_mt", "strike1", "dip1", "rake1"]]
              .merge(m, on="comcat_id")
              .merge(cat[["event_id", "ML"]].rename(columns={"ML": "ML_routeA"}), on="event_id")
              .merge(mw.rename(columns={"MW": "MW_routeA"}), on="event_id", how="left"))
    out = out.round({"ML_routeA": 2, "MW_routeA": 2})
    out.to_csv(MATCHED, index=False)
    print(f"tensors in window {len(win)}, in catalog box {len(box)}, matched {len(out)} "
          f"(|dt|<{TOL_S:.0f} s, <{TOL_KM:.0f} km) -> {MATCHED}")
    print(f"  dt median {out.dt_s.median():+.2f} s; distance median {out.dist_km.median():.1f} km, "
          f"90% {out.dist_km.quantile(.9):.1f} km; depth diff IQR "
          f"{out.ddepth_km.quantile(.25):+.1f}..{out.ddepth_km.quantile(.75):+.1f} km")
    miss = box[~box.id.isin(out.comcat_id)]
    print(f"  unmatched in box: {len(miss)}, e.g.", "; ".join(miss.place.str.replace(r"^\d+ km \w+ of ", "", regex=True)
                                                        .value_counts().head(5).index))

    classes = [(3, 4.5), (4.5, 5.5), (5.5, 7.5)]
    print("\nour magnitude - tensor Mw (median), by tensor Mw")
    for lo, hi in classes:
        s = out[out.Mw_mt.between(lo, hi, inclusive="left")]
        print(f"  Mw {lo}-{hi}: n={len(s):3d}  ML {np.median(s.ML_routeA - s.Mw_mt):+.2f}  "
              f"MW {np.nanmedian(s.MW_routeA - s.Mw_mt):+.2f}")

    ds = pd.read_csv(f"{D}/amp_distance_dataset{SUFFIX}.csv",
                     usecols=["event_id", "phase", "dist_hypo_km", "log10A"])
    ds["m"] = hutton_boore(ds.log10A, ds.dist_hypo_km)
    an = pd.read_csv(f"{D}/route_b_ml_anchors{SUFFIX}.csv")[["event_id", "ml"]]
    bins = [0, 100, 200, 300, 500, 1000]
    refs = [("ComCat ML>=2.5", an[an.ml >= 2.5].rename(columns={"ml": "ref"})),
            ("tensor Mw<4.5", out[out.Mw_mt < 4.5][["event_id", "Mw_mt"]].rename(columns={"Mw_mt": "ref"})),
            ("tensor Mw>=4.5", out[out.Mw_mt >= 4.5][["event_id", "Mw_mt"]].rename(columns={"Mw_mt": "ref"}))]
    tab = {}
    for name, ref in refs:
        x = ds[ds.phase == "S"].merge(ref, on="event_id")
        tab[name] = (x.m - x.ref).groupby(pd.cut(x.dist_hypo_km, bins), observed=False).median().round(2)
    print("\nS station magnitude (Hutton-Boore, no station terms) - reference, by distance")
    print(pd.DataFrame(tab).to_string())

    near = ds[ds.dist_hypo_km <= RMAX].groupby("event_id").m.agg(["median", "size"])
    near = near[near["size"] >= 3]["median"]
    a = an.join(near.rename("M"), on="event_id", how="inner")
    off = (a.ml - a.M).median()
    t = out.join(near.rename("M"), on="event_id", how="inner")
    res = t.M + off - t.Mw_mt
    print(f"\ntest catalog (HB, P+S within {RMAX:.0f} km, offset {off:+.2f} to ComCat): "
          f"{len(near):,} events ({len(near) / len(cat):.0%}); anchor residual std "
          f"{(a.M + off - a.ml).std():.2f}")
    for lo, hi in classes:
        s = res[t.Mw_mt.between(lo, hi, inclusive="left")]
        print(f"  Mw {lo}-{hi}: n={len(s):3d}  test ML - Mw {s.median():+.2f}")


if __name__ == "__main__":
    main()
