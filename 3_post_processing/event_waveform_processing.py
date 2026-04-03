#!/usr/bin/env python3
"""
Event Waveform Processing — CLI script (mirrors the notebook logic).

For each event:
  1. Downloads waveforms via utils.data_client.get_waveforms (pnwstore or FDSN)
  2. Resamples and highpass-filters traces
  3. Measures peak amplitude around each station's P pick
  4. Writes amplitudes back into the picks CSV

Checkpoint / resume:
  If the output CSV already exists, events that have at least one non-NaN
  amplitude are skipped on restart, so the job picks up where it left off.

Usage examples:
  # Full run with pnwstore (UW internal)
  python event_waveform_processing.py \
      --events ../data/Cascadia_relocated_catalog_ver_3.csv \
      --picks  ../data/Cascadia_relocated_catalog_picks_ver_3.csv \
      --source pnwstore --save-every 50 --verbose

  # Resume after crash (same command — skips already-done events)
  python event_waveform_processing.py \
      --events ../data/Cascadia_relocated_catalog_ver_3.csv \
      --picks  ../data/Cascadia_relocated_catalog_picks_ver_3.csv \
      --source pnwstore --save-every 50 --verbose

  # Public FDSN, single event
  python event_waveform_processing.py \
      --events ../data/Cascadia_relocated_catalog_ver_3.csv \
      --picks  ../data/Cascadia_relocated_catalog_picks_ver_3.csv \
      --source fdsn --event-id 42

Author: Marine Denolle (mdenolle@uw.edu)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

# Allow imports from the repo root regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.data_client import get_waveforms, SOURCES  # noqa: E402

Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core processing (matches the notebook's process_event exactly)
# ---------------------------------------------------------------------------

def process_event(
    event_id,
    events_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    source: str = "fdsn",
    sample_rate: int = 100,
    highpass_freq: float = 2.0,
    window_before: int = 30,
    window_after: int = 540,
    P_before: float = 1.0,
    P_after: float = 5.0,
):
    """Process a single event — download waveforms, filter, measure amplitudes.

    Returns
    -------
    st : obspy.Stream
        All processed traces for this event.
    station_amplitudes : dict
        {network.station: mean_peak_amplitude} measured around each P pick.
    origin_time : UTCDateTime
    """
    # Column name detection (robust to leading/trailing spaces)
    event_cols = [c for c in events_df.columns if "event" in c.lower()][0]
    origin_cols = [c for c in events_df.columns if "origin" in c.lower()][0]
    station_cols = [c for c in picks_df.columns if "station" in c.lower()][0]
    pick_time_col = [c for c in picks_df.columns if "pick" in c.lower()][0]

    event = events_df[events_df[event_cols] == event_id].iloc[0]
    origin_time = UTCDateTime(event[origin_cols])

    event_picks = picks_df[picks_df[event_cols] == event_id]

    st = obspy.Stream()
    station_amplitudes: dict = {}

    for _, pick in event_picks.iterrows():
        station = network = "?"
        try:
            # Picks stored as "station.network"
            station, network = pick[station_cols].split(".")
            station = station.strip()
            network = network.strip()

            # Per-station P-pick time
            pick_time = UTCDateTime(pick[pick_time_col])

            st_temp = get_waveforms(
                network=network,
                station=station,
                channel="*H*",
                starttime=origin_time - window_before,
                endtime=origin_time + window_after,
                source=source,
            )

            # Resample + highpass
            for tr in st_temp:
                tr.resample(sample_rate)
                tr.filter("highpass", freq=highpass_freq)
                st += tr

            # Amplitude around the P pick (mean of per-component peak |amp|)
            station_amps: list[float] = []
            for comp in ("Z", "N", "E"):
                comp_tr = st_temp.select(component=comp).copy()
                comp_tr.trim(
                    starttime=pick_time - P_before,
                    endtime=pick_time + P_after,
                )
                if len(comp_tr) > 0:
                    station_amps.append(float(np.max(np.abs(comp_tr[0].data))))

            if station_amps:
                station_amplitudes[f"{network}.{station}"] = float(np.mean(station_amps))

        except Exception as exc:
            Logger.warning("Error processing %s.%s: %s", network, station, exc)
            continue

    return st, station_amplitudes, origin_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Process event waveforms and compute P-wave amplitudes."
    )
    parser.add_argument("--events", required=True, help="Path to events CSV")
    parser.add_argument("--picks", required=True, help="Path to picks CSV")
    parser.add_argument(
        "--out",
        help="Output picks CSV (default: <picks>_with_amplitudes.csv next to --picks)",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="pnwstore",
        help="Waveform backend (default: pnwstore)",
    )
    parser.add_argument("--sample-rate", type=int, default=100)
    parser.add_argument("--highpass", type=float, default=2.0, help="Highpass freq (Hz)")
    parser.add_argument("--window-before", type=int, default=30, help="Seconds before origin")
    parser.add_argument("--window-after", type=int, default=540, help="Seconds after origin")
    parser.add_argument("--P-before", type=float, default=1.0, help="Seconds before P pick for amplitude")
    parser.add_argument("--P-after", type=float, default=5.0, help="Seconds after P pick for amplitude")
    parser.add_argument("--save-every", type=int, default=50, help="Flush CSV every N events")
    parser.add_argument("--event-id", help="Process only this single event ID (exact match)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    events_path = os.path.expanduser(args.events)
    picks_path = os.path.expanduser(args.picks)

    if args.out:
        out_path = os.path.expanduser(args.out)
    else:
        base, ext = os.path.splitext(picks_path)
        out_path = f"{base}_with_amplitudes{ext}"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    Logger.info("Reading events from %s", events_path)
    events_df = pd.read_csv(events_path)

    Logger.info("Reading picks from %s", picks_path)
    picks_df = pd.read_csv(picks_path)

    # Detect column names
    event_col = [c for c in events_df.columns if "event" in c.lower()][0]
    pick_event_col = [c for c in picks_df.columns if "event" in c.lower()][0]
    station_col = [c for c in picks_df.columns if "station" in c.lower()][0]

    # ------------------------------------------------------------------
    # Checkpoint / resume — load existing output if present
    # ------------------------------------------------------------------
    amp_col = None
    for c in picks_df.columns:
        if "amplitude" in c.strip().lower():
            amp_col = c
            break
    if amp_col is None:
        amp_col = " Amplitude "
        picks_df[amp_col] = np.nan

    done_set: set = set()
    if os.path.isfile(out_path):
        Logger.info("Found existing output %s — loading for resume", out_path)
        prev_df = pd.read_csv(out_path)
        if amp_col in prev_df.columns:
            picks_df[amp_col] = prev_df[amp_col]
        has_amp = picks_df[picks_df[amp_col].notna()]
        done_set = set(has_amp[pick_event_col].unique())
        Logger.info("Resuming: %d events already processed, will skip them", len(done_set))

    # ------------------------------------------------------------------
    # Build event list
    # ------------------------------------------------------------------
    if args.event_id is not None:
        try:
            eid = int(args.event_id)
        except ValueError:
            eid = args.event_id
        event_ids = [eid]
    else:
        event_ids = events_df[event_col].dropna().unique().tolist()

    total = len(event_ids)
    to_process = [eid for eid in event_ids if eid not in done_set]
    Logger.info(
        "%d total events, %d already done, %d to process",
        total,
        total - len(to_process),
        len(to_process),
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    processed = 0
    for i, eid in enumerate(to_process, start=1):
        Logger.info("[%d/%d] Processing event %s", i, len(to_process), eid)

        _, amplitudes, origin_time = process_event(
            eid,
            events_df,
            picks_df,
            source=args.source,
            sample_rate=args.sample_rate,
            highpass_freq=args.highpass,
            window_before=args.window_before,
            window_after=args.window_after,
            P_before=getattr(args, "P_before"),
            P_after=getattr(args, "P_after"),
        )

        # Store amplitudes back into picks_df (same logic as notebook)
        for station_key, amp in amplitudes.items():
            network, sta = station_key.split(".")
            sta_stripped = picks_df[station_col].str.strip()
            mask = (picks_df[pick_event_col] == eid) & (
                sta_stripped == f"{sta}.{network}"
            )
            picks_df.loc[mask, amp_col] = amp

        processed += 1

        # Periodic flush
        if processed % args.save_every == 0:
            Logger.info("Checkpoint: saving after %d events to %s", processed, out_path)
            picks_df.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Final write
    # ------------------------------------------------------------------
    picks_df.to_csv(out_path, index=False)
    Logger.info("Done — wrote %s (%d events processed this run)", out_path, processed)


if __name__ == "__main__":
    main()
