"""Network-relative residuals: a station's residual minus the median residual of nearby stations of the event.

A station clock error c(t) adds c to that station's P and S picks only. An origin-time or location error moves
the residuals of all stations of the event together. Removing, per event, the median residual of the other
stations within ``neighbor_km`` of the station (same phase by default) keeps c and removes most of the event
term. With fewer than ``min_neighbors`` neighbours the reference is the median of all other stations of the
event (``fallback: event``) or the row is dropped (``fallback: none``). The location absorbs part of a clock
error (the residuals understate it; fit the size from CC differential times or re-located residuals when you can).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd


def station_xy(res: pd.DataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-row station coordinates in km (x_km, y_km): from the residual table if present, else from a station
    table with x_km/y_km or lat/lon (local equirectangular projection about the network centre)."""
    if {"x_km", "y_km"} <= set(res.columns) and res.x_km.notna().all():
        return res
    if stations is None:
        raise ValueError("station coordinates needed: x_km/y_km in the residual table or a stations table")
    st = stations.copy()
    if not {"x_km", "y_km"} <= set(st.columns):
        lat0, lon0 = np.nanmedian(st.lat), np.nanmedian(st.lon)
        st["x_km"] = (st.lon - lon0) * 111.195 * np.cos(np.radians(lat0))
        st["y_km"] = (st.lat - lat0) * 111.195
    return res.drop(columns=[c for c in ("x_km", "y_km") if c in res.columns]).merge(
        st[["station", "x_km", "y_km"]], on="station", how="left")


def event_xyz(res: pd.DataFrame, origins: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-row event coordinates in km (ev_x_km, ev_y_km, ev_z_km) for the co-located-event control."""
    if {"ev_x_km", "ev_y_km", "ev_z_km"} <= set(res.columns):
        return res
    src = res if {"ev_lat", "ev_lon"} <= set(res.columns) else None
    if src is None and origins is not None and {"lat", "lon"} <= set(origins.columns):
        o = origins[["event_id", "lat", "lon"] + (["depth"] if "depth" in origins.columns else [])].rename(
            columns={"lat": "ev_lat", "lon": "ev_lon", "depth": "ev_depth_km"})
        res = res.merge(o, on="event_id", how="left")
        src = res
    if src is None:
        return res
    lat0, lon0 = np.nanmedian(res.ev_lat), np.nanmedian(res.ev_lon)
    res["ev_x_km"] = (res.ev_lon - lon0) * 111.195 * np.cos(np.radians(lat0))
    res["ev_y_km"] = (res.ev_lat - lat0) * 111.195
    res["ev_z_km"] = res.get("ev_depth_km", pd.Series(0.0, index=res.index)).fillna(0.0)
    return res


def _chunk(args):
    ev_bounds, x, y, r, ph, D, kmin, fallback, same = args
    n = len(r)
    rd = np.full(n, np.nan)
    nref = np.zeros(n, int)
    kind = np.zeros(n, np.int8)  # 0 none, 1 neighbours, 2 event
    for a, b in ev_bounds:
        if b - a < 2:
            continue
        xx, yy, rr, pp = x[a:b], y[a:b], r[a:b], ph[a:b]
        d = np.hypot(xx[:, None] - xx[None, :], yy[:, None] - yy[None, :])
        other = ~np.eye(b - a, dtype=bool)
        if same:
            other &= pp[:, None] == pp[None, :]
        near = other & (d <= D)
        cnt = near.sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med_n = np.nanmedian(np.where(near, rr[None, :], np.nan), axis=1)
                med_e = np.nanmedian(np.where(other, rr[None, :], np.nan), axis=1)
        use_n = cnt >= kmin
        use_e = (~use_n) & (other.sum(1) >= kmin) & (fallback == "event")
        rd[a:b] = np.where(use_n, rr - med_n, np.where(use_e, rr - med_e, np.nan))
        nref[a:b] = np.where(use_n, cnt, np.where(use_e, other.sum(1), 0))
        kind[a:b] = np.where(use_n, 1, np.where(use_e, 2, 0))
    return rd, nref, kind


def network_relative(res: pd.DataFrame, p: dict, workers: int = 1) -> pd.DataFrame:
    """Add rd (network-relative residual, s), n_ref and ref_kind to a residual table (canonical columns)."""
    keep = np.isfinite(res.residual.values) & (np.abs(res.residual.values) <= p["max_abs_residual_s"])
    if "weight" in res.columns:
        keep &= res.weight.values > p["min_weight"]
    d = res[keep].sort_values(["event_id", "station", "phase"], kind="stable").reset_index(drop=True)
    ev = d.event_id.values
    starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1]
    ends = np.r_[starts[1:], len(d)]
    bounds = list(zip(starts, ends))
    x, y = d.x_km.values.astype(float), d.y_km.values.astype(float)
    r = d.residual.values.astype(float)
    ph = d.phase.astype(str).str.upper().str[0].values
    nch = max(1, min(64, workers * 4))
    parts = np.array_split(np.arange(len(bounds)), nch)
    jobs = []
    for idx in parts:
        if len(idx) == 0:
            continue
        a0, b1 = bounds[idx[0]][0], bounds[idx[-1]][1]
        jobs.append(([(a - a0, b - a0) for a, b in bounds[idx[0]:idx[-1] + 1]], x[a0:b1], y[a0:b1], r[a0:b1],
                     ph[a0:b1], p["neighbor_km"], p["min_neighbors"], p["fallback"], p["same_phase"]))
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            out = pool.map(_chunk, jobs)
    else:
        out = [_chunk(j) for j in jobs]
    d["rd"] = np.concatenate([o[0] for o in out])
    d["n_ref"] = np.concatenate([o[1] for o in out])
    d["ref_kind"] = pd.Categorical.from_codes(np.concatenate([o[2] for o in out]), ["none", "neighbors", "event"])
    d["phase1"] = ph
    return d
