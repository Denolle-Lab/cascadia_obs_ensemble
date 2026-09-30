"""Tectonic context layers for the PyGMT catalog maps (phase5_pygmt_map.py).

  * plate boundaries: Bird (2003) PB2002 steps, split into subduction (teeth),
    spreading ridges and transforms.
  * plate motions: NNR-MORVEL56 (Argus et al., 2011) relative velocities.
  * moment tensors: USGS ComCat preferred moment tensors
    (utils/fetch_focal_mechanisms.py -> data/focal/comcat_mt.csv), split into those
    matched to our events (phase18) and the others (other dates, or not in our catalog).
  * interseismic slip-deficit rate from a published Cascadia model, as distributed by
    the Coupling Cloud (Oryan et al., 2026, https://couplingcloud.ucsd.edu).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DATA = "../../data"
PB2002 = f"{DATA}/tectonics/PB2002_steps.json"
FOCAL = f"{DATA}/focal/comcat_mt.csv"
FOCAL_MATCHED = f"{DATA}/focal/comcat_mt_matched.csv"     # from phase18_moment_tensor_match.py

# Coupling Cloud models: (file, variable, citation)
# (Lindsey.nc's "slip_def" grows with depth as its coupling falls -- it is the creep rate,
# not the deficit -- so that model is only offered as coupling x its local plate rate.)
COUPLING = {
    "pollitz":  ("coupling/Cascadia/Pollitz/2025/Pollitz_2025.nc", "slip_def", "Pollitz (2025)"),
    "sherrill": ("coupling/Cascadia/Sherrill/Sherrill.nc", "slip_def", "Sherrill et al."),
    "michel":   ("coupling/Cascadia/Michel/Michel.nc", "slip_def_locked", "Michel et al. (2019)"),
}

# NNR-MORVEL56 angular velocities (Argus et al., 2011, Table 1): lat, lon, deg/Myr
MORVEL = {"PA": (-63.58, 114.70, 0.651), "NA": (-4.85, -80.64, 0.209),
          "JF": (-38.31, 60.04, 0.951)}

# Default catalog window; load_mt narrows it to the catalog's actual span
WINDOW = (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2016-01-01", tz="UTC"))


# ---------------------------------------------------------------- plate boundaries
def plate_boundaries(region):
    """PB2002 steps inside `region`, grouped by class -> list of (lon, lat) segments.
    Consecutive steps are joined into polylines so the subduction teeth are continuous."""
    xmin, xmax, ymin, ymax = region
    with open(os.path.expanduser(PB2002)) as f:
        feats = json.load(f)["features"]
    groups = {"SUB": [], "OSR": [], "TF": []}
    for ft in feats:
        c = np.asarray(ft["geometry"]["coordinates"], float)
        if not ((c[:, 0] > xmin - 1).all() and (c[:, 0] < xmax + 1).all()
                and (c[:, 1] > ymin - 1).all() and (c[:, 1] < ymax + 1).all()):
            continue
        cls = ft["properties"]["STEPCLASS"]
        key = "SUB" if cls == "SUB" else "OSR" if cls == "OSR" else "TF"
        groups[key].append(c)
    return {k: _join(v) for k, v in groups.items()}


def _join(steps, tol=1e-3):
    """Chain 2-point steps whose ends touch into polylines."""
    lines = []
    for s in steps:
        for L in lines:
            if np.allclose(L[-1], s[0], atol=tol):
                L.extend(s[1:].tolist()); break
            if np.allclose(L[0], s[-1], atol=tol):
                L[:0] = s[:-1].tolist(); break
        else:
            lines.append(s.tolist())
    return [np.asarray(L) for L in lines]


def draw_boundaries(fig, region, pen_scale=1.0):
    pb = plate_boundaries(region)
    for L in pb["OSR"]:
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{1.1 * pen_scale:.2f}p,gray15")
    for L in pb["TF"]:
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{0.6 * pen_scale:.2f}p,gray15")
    for L in pb["SUB"]:
        L = L[np.argsort(-L[:, 1])]          # north -> south: overriding plate on the left
        fig.plot(x=L[:, 0], y=L[:, 1], pen=f"{0.6 * pen_scale:.2f}p,gray15",
                 style=f"f{0.30 * pen_scale:.2f}c/{0.07 * pen_scale:.2f}c+l+t", fill="gray15")


# ---------------------------------------------------------------- plate motions
def _omega(p):
    la, lo, r = np.radians(p[0]), np.radians(p[1]), np.radians(p[2]) / 1e6
    return r * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def plate_velocity(moving, fixed, lat, lon):
    """Velocity of plate `moving` relative to `fixed` at (lat, lon): (mm/yr, azimuth deg)."""
    om = _omega(MORVEL[moving]) - _omega(MORVEL[fixed])
    la, lo = np.radians(lat), np.radians(lon)
    r = 6371e3 * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    v = np.cross(om, r) * 1e3
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return float(np.hypot(v @ e, v @ n)), float(np.degrees(np.arctan2(v @ e, v @ n)) % 360)


def draw_plate_motion(fig, points, km_per_mm=3.0, font="5.5p,Helvetica-Bold,black"):
    """Geovector arrows (moving plate relative to fixed plate) starting at each (lat, lon),
    length = rate * km_per_mm; the rate is written beside the arrow's midpoint, on its
    north side."""
    for moving, fixed, lat, lon in points:
        rate, az = plate_velocity(moving, fixed, lat, lon)
        L = rate * km_per_mm
        vec = np.array([[lon, lat, az, L]])                    # geovector: azimuth, km
        fig.plot(data=vec, style="=0.26c+e+h0.5", pen="2.2p,white", fill="white")
        fig.plot(data=vec, style="=0.24c+e+h0.5", pen="1.0p,black", fill="black")
        a = np.radians(az)
        mla = lat + 0.5 * L * np.cos(a) / 111.2
        mlo = lon + 0.5 * L * np.sin(a) / (111.2 * np.cos(np.radians(lat)))
        # label on the arrow's inboard side (SE of a NE arrow, NE of a NW arrow), which
        # keeps it off the map's west edge
        just, off = ("LT", "0.08c/-0.06c") if az < 180 else ("LB", "0.08c/0.06c")
        fig.text(x=mlo, y=mla, text=f"{rate:.0f} mm/yr", justify=just, offset=off,
                 font=font, fill="white@30", clearance="0.03c/0.02c")


# ---------------------------------------------------------------- moment tensors
def load_mt(region, catalog=None):
    """ComCat moment tensors in `region`, with Mw from the tensor, a flag for the
    catalog window, and `matched`/`ML_ours` from phase18's match to our events
    (comcat_mt_matched.csv; |dt|<5 s, <50 km). The window is the catalog's own span
    when `catalog` (our events, with otime) is given."""
    p = os.path.expanduser(FOCAL)
    if not os.path.exists(p):
        return None
    mt = pd.read_csv(p)
    xmin, xmax, ymin, ymax = region
    mt = mt[mt.lon.between(xmin, xmax) & mt.lat.between(ymin, ymax)].copy()
    mt["time"] = pd.to_datetime(mt["time"], utc=True, format="ISO8601")
    comp = mt[["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]].to_numpy()
    m0 = np.sqrt((comp[:, :3] ** 2).sum(1) / 2 + (comp[:, 3:] ** 2).sum(1))
    mt["Mw"] = 2 / 3 * (np.log10(m0) - 9.1)
    win = WINDOW
    if catalog is not None:                  # the catalog's own span (it ends 2015-06-23)
        tc = pd.to_datetime(catalog["otime"], utc=True, format="ISO8601")
        win = (tc.min().floor("D"), tc.max().ceil("D"))
    mt["in_window"] = mt.time.between(*win)
    pm = os.path.expanduser(FOCAL_MATCHED)
    if os.path.exists(pm):
        mm = pd.read_csv(pm, usecols=["comcat_id", "event_id", "ML_routeA"])
        mt = mt.merge(mm.rename(columns={"comcat_id": "id", "ML_routeA": "ML_ours"}),
                      on="id", how="left")
    else:
        print(f"  (no {pm}; run phase18_moment_tensor_match.py -- no tensor is matched)")
        mt["event_id"], mt["ML_ours"] = np.nan, np.nan
    mt["matched"] = mt.in_window & mt.event_id.notna()
    return mt.sort_values("Mw", ascending=False)


def mt_spec(mt):
    """PyGMT meca 'mt' spec (dyne-cm mantissas + exponent)."""
    comp = mt[["mrr", "mtt", "mpp", "mrt", "mrp", "mtp"]].to_numpy() * 1e7   # N m -> dyn cm
    ex = np.floor(np.log10(np.abs(comp).max(1)))
    man = comp / 10 ** ex[:, None]
    return dict(mrr=man[:, 0], mtt=man[:, 1], mff=man[:, 2], mrt=man[:, 3],
                mrf=man[:, 4], mtf=man[:, 5], exponent=ex)


def draw_mechanisms(fig, mt, scale_cm, fill, pen="0.25p,gray10"):
    """Beachballs at the ComCat epicenters, largest first (small ones stay visible).
    GMT scales the ball linearly with magnitude; `scale_cm` is the size at M5."""
    if mt is None or mt.empty:
        return
    mt = mt.sort_values("Mw", ascending=False)
    fig.meca(spec=mt_spec(mt), convention="mt", longitude=mt.lon.to_numpy(),
             latitude=mt.lat.to_numpy(), depth=mt.depth.to_numpy(),
             scale=f"{scale_cm:.3f}c", compressionfill=fill, extensionfill="white", pen=pen)


# ---------------------------------------------------------------- slip deficit
def load_slip_deficit(model, region):
    import xarray as xr
    fn, var, cite = COUPLING[model]
    d = xr.open_dataset(os.path.expanduser(f"{DATA}/{fn}"))[var]
    xmin, xmax, ymin, ymax = region
    d = d.sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
    return d.where(d > 2.0), cite            # leave the creeping (~0) part clear
