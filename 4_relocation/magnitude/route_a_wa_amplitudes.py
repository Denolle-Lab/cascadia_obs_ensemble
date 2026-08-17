#!/usr/bin/env python3
"""
Route A, step 1: Wood-Anderson amplitudes for each P and S pick.

For every pick this script fetches the waveform (NC/BK -> NCEDC via
utils/data_client.py), removes the instrument response, and measures TWO amplitudes
per pick in a *distance-scaled* window, taken as the max over all components
(vertical fallback):
  - wa_amp_mm : peak Wood-Anderson displacement (near 1 Hz) -> local magnitude ML,
    comparable to other ML catalogs.
  - disp_amp_um : peak broadband displacement in a low band (--disp-lo/--disp-hi,
    default 0.05-2 Hz) -> a moment-scale amplitude for Mw. Wood-Anderson deliberately
    narrowbands and saturates for large events, so the low-frequency displacement is
    the amplitude to calibrate toward Mw (up to the magnitude where the corner drops
    below the usable OBS band). A proper Mw still needs a spectral Omega-0 fit; this
    displacement amplitude is the first step and shares the same response removal.
It also records a pre-signal signal-to-noise ratio and the response *epoch* the
pick falls in (so redeployed OBS get an epoch-specific station term downstream).

This addresses the main Method-B seismological caveats:
  * counts -> physical Wood-Anderson displacement (instrument response removed);
  * distance-scaled measurement window (captures the delayed S/Lg peak);
  * SNR for quality gating; per-epoch tagging for OBS redeployments;
  * NC/BK filled via NCEDC (missing from the old IRIS-only Wood-Anderson run).

MUST run on a host with pnwstore + FDSN/NCEDC access
(`pixi install --environment internal`). It is slow (one request per pick); it
appends to --out on the fly and supports --start-index for resume / sharding.
VALIDATE on a small --limit first and sanity-check wa_amp_mm before the full run.

Usage:
  python route_a_wa_amplitudes.py \
      --picks   /path/Cascadia_updated_catalog_picks_assignment_ver_3.csv \
      --inventory station_inventory.xml \
      --out raw_wa_amplitudes.csv --source pnwstore
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_inventory

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from data_client import get_waveforms  # noqa: E402  (routes NC/BK -> NCEDC)

R_EARTH_KM = 6371.0
VP, VS = 6.0, 3.5   # km/s, for S-P timing and window scaling

# IASPEI-standard Wood-Anderson (T0 = 0.8 s, damping h = 0.7, static gain 2080),
# applied to GROUND VELOCITY (one zero at the origin). The absolute gain is later
# absorbed by the ComCat-ML calibration, so 2080 vs 2800 does not affect final ML.
PAZ_WA = {"poles": [-5.49779 - 5.60886j, -5.49779 + 5.60886j],
          "zeros": [0j], "gain": 1.0, "sensitivity": 2080.0}

OUT_COLS = ["arid", "event_id", "network", "station", "phase", "evla", "evlo", "evdp",
            "stla", "stlo", "stel_m", "dist_hypo_km", "wa_amp_mm", "disp_amp_um",
            "snr", "n_comp", "epoch", "reason"]


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def phase_window(phase, r_km):
    """(pre, post) seconds around the pick. Post-window grows with distance to
    capture the delayed peak; the P window is capped short of the S arrival."""
    tsp = r_km * (1.0 / VS - 1.0 / VP)               # S-P time (s)
    if phase == "P":
        post = min(1.0 + 0.03 * r_km, 15.0, 0.8 * max(tsp, 1.25))
        return 0.3, max(post, 1.0)
    post = min(2.0 + 0.06 * r_km, 60.0)              # S/Lg coda grows with distance
    return 0.3, post


def epoch_id(inv, net, sta, t):
    """Index of the response epoch containing time t for net.sta (redeployment tag)."""
    try:
        chans = inv.select(network=net, station=sta, time=t).get_contents()["channels"]
        starts = sorted({inv.get_channel_metadata(cid, t).get("starttime")
                         for cid in chans} - {None})
        return f"{starts[0].date}" if starts else ""
    except Exception:
        return ""


def _phase(v):
    s = str(v).strip().upper()
    return "P" if s in ("P", "0") else "S" if s in ("S", "1") else s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="raw_wa_amplitudes.csv")
    ap.add_argument("--source", default="pnwstore")
    ap.add_argument("--sample-rate", type=int, default=100)
    ap.add_argument("--highpass", type=float, default=1.0, help="post-WA high-pass (Hz)")
    ap.add_argument("--disp-lo", type=float, default=0.05,
                    help="low corner (Hz) for the displacement (Mw) band")
    ap.add_argument("--disp-hi", type=float, default=2.0,
                    help="high corner (Hz) for the displacement (Mw) band")
    ap.add_argument("--noise-win", type=float, default=10.0, help="pre-signal noise window (s)")
    ap.add_argument("--start-index", type=int, default=0, help="resume / shard start row")
    ap.add_argument("--limit", type=int, default=None, help="process at most N picks (testing)")
    ap.add_argument("--chunk-rows", type=int, default=0,
                    help="rotate output into <out>_partNNN.csv every N rows (0 = single "
                         "file). Use for the full ~40M-pick set; parts are numbered by "
                         "absolute row index so shards/resumes don't collide.")
    args = ap.parse_args(argv)

    inv = read_inventory(args.inventory)
    picks = pd.read_csv(args.picks)
    picks.columns = [c.strip() for c in picks.columns]
    ev_col = "idx" if "idx" in picks.columns else "event_id"

    sl = slice(args.start_index, args.start_index + args.limit if args.limit else None)

    # Rotating writer: with --chunk-rows>0, output goes to <out>_partNNN.csv, the part
    # index derived from the ABSOLUTE row index so resumes/shards land in stable parts.
    stem, ext = os.path.splitext(args.out)

    def part_path(abs_idx):
        if not args.chunk_rows:
            return args.out
        return f"{stem}_part{abs_idx // args.chunk_rows:04d}{ext}"

    state = {"path": None, "fh": None, "w": None}

    def writer_for(abs_idx):
        p = part_path(abs_idx)
        if p != state["path"]:
            if state["fh"]:
                state["fh"].close()
            new = not os.path.exists(p)          # header only when the part is created
            state["fh"] = open(p, "a", newline="")
            state["w"] = csv.DictWriter(state["fh"], fieldnames=OUT_COLS)
            if new:
                state["w"].writeheader()
            state["path"] = p
        return state["w"], state["fh"]

    for local_i, (_, row) in enumerate(picks.iloc[sl].iterrows()):
        abs_idx = args.start_index + local_i
        w, fh = writer_for(abs_idx)
        net, sta = str(row["station"]).split(".")[0].strip(), str(row["station"]).split(".")[1].strip()
        phase = _phase(row["phase"])
        rec = {k: "" for k in OUT_COLS}
        rec.update(arid=row.get("arid"), event_id=row.get(ev_col), network=net,
                   station=f"{net}.{sta}", phase=phase, n_comp=0)
        try:
            tp = UTCDateTime(str(row["time_pick"]))
            evla, evlo, evdp = float(row["latitude"]), float(row["longitude"]), float(row["depth"])
            stla, stlo, stel = float(row["slatitude"]), float(row["slongitude"]), float(row["selevation"])
        except Exception as e:
            rec["reason"] = f"row:{e}"; w.writerow(rec); continue
        r = float(np.hypot(haversine_km(evla, evlo, stla, stlo), evdp + stel / 1000.0))
        pre, post = phase_window(phase, r)
        rec.update(evla=evla, evlo=evlo, evdp=evdp, stla=stla, stlo=stlo, stel_m=stel,
                   dist_hypo_km=round(r, 3))

        try:
            st = get_waveforms(net, sta, "*H*", tp - (args.noise_win + 5), tp + post + 5,
                               source=args.source)
        except Exception as e:
            rec["reason"] = f"fetch:{str(e)[:80]}"; w.writerow(rec); continue
        if len(st) == 0:
            rec["reason"] = "no_data"; w.writerow(rec); continue

        try:
            st.merge(method=1, fill_value="interpolate")
            st.resample(args.sample_rate)
            st.detrend("demean"); st.taper(0.05)
            st_disp = st.copy()                       # broadband displacement for Mw
            st.remove_response(inventory=inv, output="VEL", water_level=60)
            st.simulate(paz_simulate=PAZ_WA)          # ground velocity -> WA displacement (m)
            st.filter("highpass", freq=args.highpass)
            # displacement path: response -> DISP, low band for the moment-scale amplitude
            st_disp.remove_response(inventory=inv, output="DISP", water_level=60)
            st_disp.filter("bandpass", freqmin=args.disp_lo, freqmax=args.disp_hi,
                           zerophase=True)
        except Exception as e:
            rec["reason"] = f"resp:{str(e)[:80]}"; w.writerow(rec); continue

        rec["epoch"] = epoch_id(inv, net, sta, tp)
        sig_lo, sig_hi = tp - pre, tp + post
        noi_lo, noi_hi = tp - args.noise_win - 1, tp - 1
        amp, noise, ncomp = 0.0, 0.0, 0
        for tr in st:
            d = tr.slice(sig_lo, sig_hi).data
            n = tr.slice(noi_lo, noi_hi).data
            if len(d):
                amp = max(amp, float(np.max(np.abs(d)))); ncomp += 1
            if len(n):
                noise = max(noise, float(np.sqrt(np.mean(n.astype(float) ** 2))))
        if ncomp == 0:
            rec["reason"] = "no_window_data"; w.writerow(rec); continue

        disp = 0.0                                    # peak displacement (m) in the low band
        for tr in st_disp:
            dd = tr.slice(sig_lo, sig_hi).data
            if len(dd):
                disp = max(disp, float(np.max(np.abs(dd))))

        rec.update(wa_amp_mm=amp * 1000.0,                     # m -> mm  (for ML)
                   disp_amp_um=disp * 1e6,                     # m -> um  (for Mw)
                   snr=(amp / noise) if noise > 0 else np.nan,
                   n_comp=ncomp, reason="ok")
        w.writerow(rec); fh.flush()
        time.sleep(0.05)

    if state["fh"]:
        state["fh"].close()
    where = f"{stem}_part*.csv" if args.chunk_rows else args.out
    print(f"done -> {where}")


if __name__ == "__main__":
    main()
