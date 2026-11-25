"""model-clock: confirmed clock epochs -> a clock-error model c(t) per station, with uncertainty.

Confirmed-epoch CSV (written by ``ingest-review``, or by hand), one row per epoch:
  epoch_id            unique id, e.g. STA.NET#1
  station             station id as in the pick file
  model               drift_reset | drift_open | offset_epoch | step
  t_start_utc         start of the epoch (drift start / offset start), ISO UTC
  t_start_sd_days     1-sigma uncertainty of the start (days)
  t_end_utc           end (reset: the header restart that re-synchronised the clock); empty for open models
  size                rate (s/day) for drift models, offset (s) for offset models
  size_sd             1-sigma uncertainty of size
  size_source         where size comes from (cc | residuals | manual)
  confirmed_by        who confirmed the epoch (a person)
  decision_ref        decision / ticket reference
  evidence            short text
Sign: c(t) = header time - true time; positive = the station's picks are late. The pick correction is -c(t).
Outside every epoch c = 0 (the station's reference clock).

c(t) for t in [t_start, t_end):  drift: size * (t - t_start) [days];  offset: size.
sd(t): drift: sqrt((t - t_start)^2 size_sd^2 + size^2 t_start_sd^2); offset: size_sd.
A pick within ``near_start_sigmas`` * t_start_sd_days of an epoch start is flagged near_start (its correction is
small and uncertain; it is applied as modelled).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import SIGN_CONVENTION, __version__
from .tables import code_version, md5, to_posix, write_json

DAY = 86400.0
DRIFT = {"drift_reset", "drift_open"}
OFFSET = {"offset_epoch", "step"}
REQUIRED = ["epoch_id", "station", "model", "t_start_utc", "size"]


def read_epochs(path) -> pd.DataFrame:
    e = pd.read_csv(path, dtype={"t_end_utc": str, "t_start_utc": str})
    miss = [c for c in REQUIRED if c not in e.columns]
    if miss:
        raise KeyError(f"{path}: missing columns {miss}")
    bad = e[~e.model.isin(DRIFT | OFFSET)]
    if len(bad):
        raise ValueError(f"unknown model(s) {sorted(bad.model.unique())}")
    if e.epoch_id.duplicated().any():
        raise ValueError("duplicate epoch_id")
    for c in ("t_start_sd_days", "size_sd"):
        if c not in e.columns:
            e[c] = 0.0
        e[c] = e[c].fillna(0.0)
    e["t_end_utc"] = e.get("t_end_utc", pd.Series("", index=e.index)).fillna("")
    closed = e.model.isin({"drift_reset", "offset_epoch"})
    if (closed & (e.t_end_utc == "")).any():
        raise ValueError("drift_reset / offset_epoch need t_end_utc")
    return e


def build_model(epochs: pd.DataFrame, source: str | None = None, near_start_sigmas: float = 2.0) -> dict:
    st = {}
    for r in epochs.itertuples():
        t0 = float(to_posix([r.t_start_utc])[0])
        t1 = float(to_posix([r.t_end_utc])[0]) if r.t_end_utc else None
        if t1 is not None and t1 <= t0:
            raise ValueError(f"{r.epoch_id}: end before start")
        st.setdefault(r.station, []).append(dict(
            epoch_id=r.epoch_id, model=r.model, t_start_s=t0, t_start_utc=r.t_start_utc,
            t_start_sd_days=float(r.t_start_sd_days), t_end_s=t1, t_end_utc=r.t_end_utc or None,
            size=float(r.size), size_sd=float(r.size_sd),
            size_units="s/day" if r.model in DRIFT else "s",
            **{k: getattr(r, k) for k in ("size_source", "confirmed_by", "decision_ref", "evidence")
               if hasattr(r, k) and isinstance(getattr(r, k), str)}))
    for s, eps in st.items():  # no overlaps within a station
        eps.sort(key=lambda e: e["t_start_s"])
        for a, b in zip(eps[:-1], eps[1:]):
            if a["t_end_s"] is None or a["t_end_s"] > b["t_start_s"]:
                raise ValueError(f"{s}: epochs {a['epoch_id']} and {b['epoch_id']} overlap")
    return dict(format="pick_timing.clock_model", format_version=1, package_version=__version__,
                created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                sign_convention=SIGN_CONVENTION, near_start_sigmas=near_start_sigmas,
                source={"epochs": source, "md5": md5(source) if source and Path(source).exists() else None},
                code=code_version(), stations=st)


def load_model(path) -> dict:
    import json
    m = json.loads(Path(path).read_text())
    if m.get("format") != "pick_timing.clock_model":
        raise ValueError(f"{path} is not a pick_timing clock model")
    return m


def evaluate(model: dict, stations, t_s) -> pd.DataFrame:
    """c(t), sd, epoch id and flag for arrays of station ids and POSIX times."""
    stations = np.asarray(stations).astype(str)
    t_s = np.asarray(t_s, float)
    n = len(t_s)
    c = np.zeros(n)
    sd = np.zeros(n)
    ep = np.full(n, "", dtype=object)
    flag = np.full(n, "none", dtype=object)
    k = model.get("near_start_sigmas", 2.0)
    for s, eps in model["stations"].items():
        m = stations == s
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        t = t_s[idx]
        for e in eps:
            t0 = e["t_start_s"]
            t1 = e["t_end_s"] if e["t_end_s"] is not None else np.inf
            inn = (t >= t0) & (t < t1)
            d = (t - t0) / DAY
            if e["model"] in DRIFT:
                ci = e["size"] * d
                si = np.sqrt((d * e["size_sd"]) ** 2 + (e["size"] * e["t_start_sd_days"]) ** 2)
                kind = "drift"
            else:
                ci = np.full(len(t), e["size"])
                si = np.full(len(t), e["size_sd"])
                kind = "offset"
            ii = idx[inn]
            c[ii] = ci[inn]
            sd[ii] = si[inn]
            ep[ii] = e["epoch_id"]
            flag[ii] = kind
            near = np.abs(d) <= k * e["t_start_sd_days"]
            flag[idx[near & inn]] = "near_start"
            flag[idx[near & ~inn]] = "near_start_outside"
            ep[idx[near & ~inn]] = e["epoch_id"]
    return pd.DataFrame({"clock_c_s": c, "clock_sd_s": sd, "clock_epoch": ep, "clock_flag": flag})


def cli_model(cfg, a):
    e = read_epochs(a.epochs)
    m = build_model(e, str(Path(a.epochs).resolve()), cfg["model"]["near_start_sigmas"])
    out = a.output or str(Path(cfg["output_dir"]) / "clock_model.json")
    write_json(m, out)
    for s, eps in m["stations"].items():
        for x in eps:
            print(f"{x['epoch_id']}: {x['model']} {x['t_start_utc']} -> {x['t_end_utc']}, size {x['size']:+.6g} "
                  f"+- {x['size_sd']:.2g} {x['size_units']}")
    print(f"clock model: {sum(len(v) for v in m['stations'].values())} epochs at {len(m['stations'])} stations -> {out}")
    return 0
