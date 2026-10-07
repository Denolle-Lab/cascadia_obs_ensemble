#!/usr/bin/env python3
"""
Phase 2 (anchor): calibrate Route-B RELATIVE magnitudes to absolute ML using
USGS/ComCat events that carry an ML magnitude.

Steps:
  1. Fetch ComCat events (USGS FDSN) for the catalog's box + period; keep only
     events that have an **ML**-type magnitude. Cached to CSV so we fetch once.
  2. Match ComCat-ML events to our relocated events (origin time + epicentral
     distance).
  3. Robust calibration  ML = a * M_rel + b  (Theil-Sen) on the matched anchors.
  4. Apply to ALL events -> absolute ML; write the final magnitude catalog.
  5. Validation: 1:1 and residual diagnostics.

Usage:
    python phase2_anchor_comcat_ml.py \
        --events   ../../data/magnitude/route_b_event_relative_mag.csv \
        --catalog  ../../data/Cascadia_relocated_catalog_ver_3.csv \
        --outdir   ../../data/magnitude
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fetch_comcat_ml(box, t0, t1, minmag, cache):
    """Return DataFrame(time, lat, lon, depth_km, ml, evid) of ComCat events that
    have an ML magnitude. Cached to `cache`."""
    if os.path.exists(cache):
        print(f"[comcat] using cache {cache}")
        return pd.read_csv(cache, parse_dates=["time"])

    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    client = Client("USGS")
    minlat, maxlat, minlon, maxlon = box
    rows = []
    # fetch year-by-year to stay under the 20k/query cap, clamped to [t0, t1] so the
    # first/last (partial) years do not pull extra events outside the catalog window
    t0u, t1u = UTCDateTime(t0.isoformat()), UTCDateTime(t1.isoformat())
    for yr in range(t0.year, t1.year + 1):
        s = max(UTCDateTime(f"{yr}-01-01"), t0u)
        e = min(UTCDateTime(f"{yr+1}-01-01"), t1u)
        if e <= s:
            continue
        try:
            cat = client.get_events(starttime=s, endtime=e, minmagnitude=minmag,
                                    minlatitude=minlat, maxlatitude=maxlat,
                                    minlongitude=minlon, maxlongitude=maxlon)
        except Exception as ex:
            print(f"[comcat] {yr}: no events / error: {ex}")
            continue
        for ev in cat:
            o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            if o is None:
                continue
            ml = None
            for mg in ev.magnitudes:
                # ML only (accept ML / MLv variants); explicitly ignore Md, Mw, mb, ...
                if str(mg.magnitude_type or "").lower() in ("ml", "mlv") and mg.mag is not None:
                    ml = mg.mag
                    break
            if ml is None:
                continue
            rows.append(dict(time=o.time.datetime, lat=o.latitude, lon=o.longitude,
                             depth_km=(o.depth or 0) / 1000.0, ml=ml,
                             evid=str(ev.resource_id).split("/")[-1]))
        print(f"[comcat] {yr}: {sum(1 for r in rows if r['time'].year == yr)} ML events")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(cache, index=False)
        print(f"[comcat] cached {len(df)} ML events -> {cache}")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", default="../../data/magnitude/route_b_event_relative_mag.csv")
    p.add_argument("--catalog", default="../../data/Cascadia_relocated_catalog_ver_3.csv")
    p.add_argument("--outdir", default="../../data/magnitude")
    p.add_argument("--minmag", type=float, default=1.5, help="ComCat minmagnitude to fetch")
    p.add_argument("--dt-s", type=float, default=15.0, help="match: max origin-time diff (s)")
    p.add_argument("--dist-km", type=float, default=50.0, help="match: max epicentral dist (km)")
    p.add_argument("--suffix", default="", help="suffix for output filenames, e.g. _kpos")
    args = p.parse_args(argv)

    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    ev = pd.read_csv(os.path.expanduser(args.events))
    cat = pd.read_csv(os.path.expanduser(args.catalog))
    cat.columns = [c.strip() for c in cat.columns]
    otime = pd.to_datetime(cat.set_index("Event ID")["Origin Time (UTC)"])
    ev = ev.merge(otime.rename("otime"), left_on="event_id", right_index=True, how="left")
    ev["otime"] = pd.to_datetime(ev["otime"], utc=True)

    box = (ev.evla.min() - 0.2, ev.evla.max() + 0.2, ev.evlo.min() - 0.2, ev.evlo.max() + 0.2)
    t0, t1 = ev.otime.min(), ev.otime.max() + pd.Timedelta(days=1)
    print(f"box lat[{box[0]:.2f},{box[1]:.2f}] lon[{box[2]:.2f},{box[3]:.2f}]  "
          f"period {t0.date()}..{t1.date()}")

    from obspy import UTCDateTime  # noqa: for year attrs below
    cc = fetch_comcat_ml(box, t0, t1, args.minmag,
                         os.path.join(outdir, "comcat_ml_events.csv"))
    if not len(cc):
        raise SystemExit("No ComCat ML events fetched (no network?). Run on a host "
                         "with internet; the cache will then be reused.")
    cc["time"] = pd.to_datetime(cc["time"], utc=True)

    # ---- match each ComCat-ML event to the nearest-in-time relocated event ----
    EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
    ev_sorted = ev.dropna(subset=["otime"]).sort_values("otime").reset_index(drop=True)
    ev_epoch = ((ev_sorted["otime"] - EPOCH) / pd.Timedelta(seconds=1)).to_numpy()
    cc_epoch = ((cc["time"] - EPOCH) / pd.Timedelta(seconds=1)).to_numpy()
    anchors = []
    for i in range(len(cc)):
        te = cc_epoch[i]
        lo = np.searchsorted(ev_epoch, te - args.dt_s)
        hi = np.searchsorted(ev_epoch, te + args.dt_s)
        if hi <= lo:
            continue
        cand = ev_sorted.iloc[lo:hi]
        d = haversine_km(cc.lat.iloc[i], cc.lon.iloc[i], cand.evla.values, cand.evlo.values)
        j = int(np.argmin(d))
        if d[j] <= args.dist_km:
            row = cand.iloc[j]
            anchors.append(dict(event_id=row.event_id, M_rel=row.M_rel, ml=float(cc.ml.iloc[i]),
                                dt_s=abs(ev_epoch[lo + j] - te), dist_km=float(d[j])))
    anchors = pd.DataFrame(anchors).drop_duplicates("event_id")
    print(f"ComCat ML events: {len(cc):,}   matched anchors: {len(anchors):,}")
    if len(anchors) < 8:
        raise SystemExit(f"Too few anchors ({len(anchors)}) to calibrate; widen "
                         "--dt-s/--dist-km or lower --minmag.")

    # ---- robust calibration ML = a*M_rel + b (Theil-Sen) ----
    a, b, alo, ahi = stats.theilslopes(anchors.ml, anchors.M_rel)
    pred = a * anchors.M_rel + b
    res = anchors.ml - pred
    rstd = np.std(res); rmad = 1.4826 * np.median(np.abs(res - np.median(res)))
    corr = np.corrcoef(anchors.M_rel, anchors.ml)[0, 1]
    print(f"calibration  ML = {a:.3f} * M_rel + {b:.3f}   (slope 95% CI {alo:.3f}..{ahi:.3f})")
    print(f"anchors: n={len(anchors)}  corr={corr:.3f}  resid std={rstd:.2f}  MAD={rmad:.2f}  "
          f"ML range {anchors.ml.min():.1f}..{anchors.ml.max():.1f}")

    # ---- apply to ALL events ----
    ev["ML"] = a * ev["M_rel"] + b
    sem = ev["M_rel_sem"] if "M_rel_sem" in ev.columns else pd.Series(0.0, index=ev.index)
    ev["ML_unc"] = np.hypot(a * sem.fillna(0.0), rmad)
    out_cols = ["event_id", "otime", "evla", "evlo", "evdp", "ML", "ML_unc",
                "M_rel", "n_obs", "n_P", "n_S", "M_sta_std"]
    out_cols = [c for c in out_cols if c in ev.columns]
    final = ev[out_cols].sort_values("otime")
    fout = os.path.join(outdir, f"cascadia_catalog_ML{args.suffix}.csv")
    final.to_csv(fout, index=False)
    anchors.to_csv(os.path.join(outdir, f"route_b_ml_anchors{args.suffix}.csv"), index=False)
    print(f"final ML: n={len(final):,}  range {final.ML.min():.2f}..{final.ML.max():.2f}  "
          f"median {final.ML.median():.2f}")
    print(f"wrote {fout}")

    # ---- validation plots ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        lim = [anchors.ml.min() - .5, anchors.ml.max() + .5]
        ax[0].plot(pred, anchors.ml, ".", alpha=.4); ax[0].plot(lim, lim, "k--")
        ax[0].set_xlabel("calibrated ML (from M_rel)"); ax[0].set_ylabel("ComCat ML")
        ax[0].set_title(f"anchors 1:1 (n={len(anchors)}, r={corr:.2f})"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
        ax[1].scatter(anchors.M_rel, anchors.ml, s=8, alpha=.4)
        xs = np.linspace(anchors.M_rel.min(), anchors.M_rel.max(), 50)
        ax[1].plot(xs, a * xs + b, "r-"); ax[1].set_xlabel("M_rel"); ax[1].set_ylabel("ComCat ML")
        ax[1].set_title(f"ML = {a:.2f}*M_rel + {b:.2f}")
        ax[2].hist(final.ML, bins=80); ax[2].set_title("final ML (all events)"); ax[2].set_xlabel("ML")
        fig.suptitle("Route B -> absolute ML calibration"); fig.tight_layout()
        png = os.path.join(outdir, f"route_b_ml_calibration{args.suffix}.png"); fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
