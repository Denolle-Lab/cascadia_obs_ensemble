"""Synthetic data: clean and clock-affected stations (residual tables), and day-merged MiniSEED with a drifting
sample count. Used by the tests, the false-alarm calibration and ``make-example``."""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400.0
T0 = pd.Timestamp("2020-01-01").timestamp()


def clock_term(model, t_days, t1=None, t2=np.inf, a=0.0):
    t = np.asarray(t_days, float)
    if model == "none":
        return np.zeros_like(t)
    if model == "offset_epoch":
        return np.where((t >= t1) & (t < t2), a, 0.0)
    if model == "step":
        return np.where(t >= t1, a, 0.0)
    if model == "drift_reset":
        return np.where((t >= t1) & (t < t2), a * (t - t1), 0.0)
    if model == "drift_open":
        return np.where(t >= t1, a * (t - t1), 0.0)
    raise ValueError(model)


def network(n_sta=12, n_ev=1500, days=400, seed=0, clocks=None, noise=0.08, s_noise=0.15, origin_sd=0.3,
            spacing_km=15.0):
    """Synthetic residual table for a small network.

    Events at random times and places; residual(station, event) = origin-time error of the event (common to all
    stations) + path noise (Student-t, P sd `noise`, S sd `s_noise`) + c_station(t) for the stations in `clocks`:
    {station: dict(model=..., t1=day, t2=day, a=...)} with t in days since 2020-01-01.
    Returns (residuals, stations, origins)."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_sta)))
    st = pd.DataFrame({"station": [f"S{k:02d}.XX" for k in range(n_sta)],
                       "x_km": [(k % side) * spacing_km for k in range(n_sta)],
                       "y_km": [(k // side) * spacing_km for k in range(n_sta)]})
    st["lat"] = 45 + st.y_km / 111.195
    st["lon"] = -122 + st.x_km / (111.195 * np.cos(np.radians(45)))
    t_ev = np.sort(rng.uniform(0, days, n_ev))
    # events in a few clusters so the co-located control has pairs
    cx = rng.uniform(0, side * spacing_km, 6)
    cy = rng.uniform(0, side * spacing_km, 6)
    which = rng.integers(0, 6, n_ev)
    ev = pd.DataFrame({"event_id": np.arange(n_ev), "t_days": t_ev,
                       "ev_x_km": cx[which] + rng.normal(0, 1.5, n_ev), "ev_y_km": cy[which] + rng.normal(0, 1.5, n_ev),
                       "ev_z_km": rng.uniform(2, 15, n_ev), "ot_err": rng.normal(0, origin_sd, n_ev)})
    ev["time_s"] = T0 + ev.t_days * DAY
    rows = []
    for e in ev.itertuples():
        d = np.hypot(st.x_km - e.ev_x_km, st.y_km - e.ev_y_km)
        seen = d < 80
        for s, dd in zip(st.station[seen], d[seen]):
            for ph, sd, v in (("P", noise, 6.0), ("S", s_noise, 3.5)):
                if ph == "S" and rng.random() < 0.3:
                    continue
                tt = np.hypot(dd, e.ev_z_km) / v
                c = 0.0
                if clocks and s in clocks:
                    c = float(clock_term(t_days=e.t_days, **clocks[s]))
                r = e.ot_err + sd * rng.standard_t(4) / np.sqrt(2) + c
                rows.append((e.event_id, s, ph, e.time_s + tt + r, r, 1.0, tt))
    res = pd.DataFrame(rows, columns=["event_id", "station", "phase", "time_s", "residual", "weight", "ttpred"])
    res = res.merge(st[["station", "x_km", "y_km"]], on="station").merge(
        ev[["event_id", "ev_x_km", "ev_y_km", "ev_z_km"]], on="event_id")
    from .tables import posix_to_iso
    res["time"] = [posix_to_iso(t, 4) for t in res.time_s]
    origins = ev.assign(time=[posix_to_iso(t, 6) for t in ev.time_s])
    return res.sort_values(["event_id", "station", "phase"]).reset_index(drop=True), st, origins


def picks_from_residuals(res: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    """A pick file in this project's style (', ' separated) from a synthetic residual table."""
    from .tables import posix_to_iso
    o = origins.set_index("event_id").time_s
    p = pd.DataFrame({"event_ID": res.event_id, "pick_datetime": [posix_to_iso(t, 6) for t in res.time_s],
                      "travel_time(s)": (res.time_s - res.event_id.map(o)).round(4), "station": res.station,
                      "phase": res.phase, "pick_prob": 0.9})
    return p


# --------------------------------------------------------------------------------------------- MiniSEED
def drifting_day_mseed(path_dir, net="XX", sta="DRFT", cha="EHZ", day="2020-01-01", seconds=600,
                       true_rate=99.8, nominal=100.0, rec_samples=100, gap_at=None, seed=0):
    """Write an SDS day file whose records carry correct header times while each record holds `rec_samples`
    samples taken at `true_rate`: headers advance by rec_samples / true_rate per record, so a reader that merges
    the records (gaps below half a sample) and counts at `nominal` drifts by (nominal/true_rate - 1) per sample.
    gap_at: record index where the data restart (a 2.5 s gap) happens. Returns (file path, record start times)."""
    from obspy import Stream, Trace, UTCDateTime
    rng = np.random.default_rng(seed)
    t0 = UTCDateTime(day)
    nrec = int(seconds * true_rate / rec_samples)
    starts, buf = [], io.BytesIO()
    tt = float(t0) + 0.0
    k = 0
    for r in range(nrec):
        if gap_at is not None and r == gap_at:
            tt += 2.5
        data = (np.arange(rec_samples) + k).astype(np.int32) * 7 % 1000 + rng.integers(-3, 4, rec_samples).astype(np.int32)
        tr = Trace(data=data.astype(np.int32), header=dict(network=net, station=sta, location="", channel=cha,
                                                           sampling_rate=nominal, starttime=UTCDateTime(tt)))
        b = io.BytesIO()
        Stream([tr]).write(b, format="MSEED", reclen=512, encoding="INT32")
        buf.write(b.getvalue())
        starts.append(tt)
        tt += rec_samples / true_rate
        k += rec_samples
    d = pd.Timestamp(day)
    sub = Path(path_dir) / f"{d.year}" / net / sta / f"{cha}.D"
    sub.mkdir(parents=True, exist_ok=True)
    fn = sub / f"{net}.{sta}..{cha}.D.{d.year}.{d.dayofyear:03d}"
    fn.write_bytes(buf.getvalue())
    return str(fn), np.array(starts)


def cli_example(cfg, a):
    """Write the example dataset: a 12-station synthetic network with one drifting clock (S03.XX) and one offset
    epoch (S07.XX), its pick file, origins, stations, and a drifting day-merged MiniSEED day (SDS, station DRFT.XX,
    five picks of a day-merging picker in picks.csv; true re-timing corrections in truth_retime.csv)."""
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    clocks = {"S03.XX": dict(model="drift_reset", t1=150.0, t2=240.0, a=0.006),
              "S07.XX": dict(model="offset_epoch", t1=300.0, t2=340.0, a=-0.25)}
    res, st, ori = network(seed=a.seed, clocks=clocks)
    res.to_csv(out / "residuals.csv", index=False)
    st.to_csv(out / "stations.csv", index=False)
    ori[["event_id", "time", "ev_x_km", "ev_y_km", "ev_z_km"]].to_csv(out / "origins.csv", index=False)
    picks = picks_from_residuals(res, ori)
    # a day-merging picker on a drifting channel (DRFT.XX, 2020-01-01): picks on the merged 100 Hz grid
    fn, starts = drifting_day_mseed(out / "sds", seconds=600)
    import obspy
    st = obspy.read(fn).merge(method=0, fill_value=0)
    t_m = float(st[0].stats.starttime)
    idx = np.array([1500, 12000, 30000, 45123, 59000])
    from .tables import posix_to_iso
    dr = pd.DataFrame({"event_ID": 99001 + np.arange(len(idx)), "pick_datetime": [posix_to_iso(t_m + i / 100, 6) for i in idx],
                       "travel_time(s)": 5.0, "station": "DRFT.XX", "phase": "P", "pick_prob": 0.9})
    picks = pd.concat([picks, dr], ignore_index=True)
    truth_rt = pd.DataFrame({"event_ID": dr.event_ID, "true_correction_s": starts[idx // 100] + (idx % 100) / 100 - (t_m + idx / 100)})
    truth_rt.to_csv(out / "truth_retime.csv", index=False)
    with open(out / "picks.csv", "w") as f:
        f.write(", ".join(picks.columns) + "\n")
        for r in picks.itertuples(index=False):
            f.write(f"{r[0]}, {r[1]}, {r[2]:.4f}, {r[3]}, {r[4]}, {r[5]:.4f}\n")
    pd.DataFrame([dict(station=s, **c) for s, c in clocks.items()]).to_csv(out / "truth_clocks.csv", index=False)
    print(f"example written to {out}: {len(res)} residual rows, {len(picks)} picks, truth in truth_clocks.csv, "
          f"drifting MiniSEED {fn}")
    return 0
