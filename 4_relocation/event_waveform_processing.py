#!/usr/bin/env python3
"""
Event Waveform Processing (script version of the notebook)

Usage:
    python event_waveform_processing.py --events ../data/Cascadia_relocated_catalog_ver_3.csv \
        --picks ../data/Cascadia_relocated_catalog_picks_ver_3.csv \
        --out ../data/Cascadia_relocated_catalog_picks_with_amplitudes_ver_3.csv

Designed to run in a detached terminal (screen/tmux).

This script:
 - reads event and pick CSV files
 - downloads waveforms from IRIS FDSN for each pick's station
 - resamples and highpass filters traces
 - computes peak amplitudes (averaged over available components)
 - stores amplitudes back into the picks CSV and writes output

Author: converted from notebook by automation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


def find_column(df: pd.DataFrame, keywords: Tuple[str, ...]) -> Optional[str]:
    """Find a column name in df whose lowered name contains any of the keywords.

    Returns the first matching column name or None.
    """
    for col in df.columns:
        name = col.strip().lower()
        for kw in keywords:
            if kw in name:
                return col
    return None


def process_event(
    event_id: str,
    events_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    client: Client,
    sample_rate: int = 100,
    highpass_freq: float = 4.0,
    window_before: int = 300,
    window_after: int = 3000,
) -> Tuple[Optional[object], pd.DataFrame, Optional[UTCDateTime]]:
    """Process event and return waveform measurements per station/component.
    
    Returns:
        Tuple containing:
        - st: Stream object (None in current implementation)
        - measurements_df: DataFrame with columns:
            - station: station identifier (NET.STA format)
            - component: Z, N, or E
            - max_amp: maximum amplitude in mm
            - min_amp: minimum amplitude in mm
            - duration: time between min and max in seconds
        - origin_time: event origin time
    """
    """Process a single event and return stream metadata and amplitudes.

    Returns (st, station_amplitudes, origin_time). st may be None if no data.
    """
    # Detect column names robustly
    event_col = find_column(events_df, ("event id", "event_id", "event"))
    origin_col = find_column(events_df, ("origin", "time", "origin time", "datetime"))
    if event_col is None or origin_col is None:
        logging.error("Could not find event id or origin time columns in events CSV")
        return None, {}, None

    pick_event_col = find_column(picks_df, ("event id", "event_id", "event"))
    station_col = find_column(picks_df, ("station name", "station", "sta"))
    if pick_event_col is None or station_col is None:
        logging.error("Could not find event id or station columns in picks CSV")
        return None, {}, None

    # locate event row
    event_rows = events_df[events_df[event_col] == event_id]
    if len(event_rows) == 0:
        logging.warning("Event id %s not found in events DataFrame", event_id)
        return None, {}, None
    event = event_rows.iloc[0]
    origin_time = UTCDateTime(event[origin_col])

    # picks associated
    event_picks = picks_df[picks_df[pick_event_col] == event_id]

    # Dictionary to store measurements per station and component
    measurements = {}
    
    for _, pick in event_picks.iterrows():
        raw_station = str(pick[station_col]).strip()
        if raw_station in ("nan", "None", ""):
            continue
        # Expect format STA.NET or NET.STA; try to parse STA.NET -> station,network
        parts = raw_station.split(".")
        if len(parts) != 2:
            logging.warning("Unexpected station name format '%s' for event %s", raw_station, event_id)
            continue
        station, network = parts[0].strip(), parts[1].strip()

        try:
            st_temp = client.get_waveforms(
                network=network,
                station=station,
                location="*",
                channel="*H*",
                starttime=origin_time - window_before,
                endtime=origin_time + window_after,
                attach_response=False,
            )
        except Exception as e:
            logging.warning("Failed to download waveforms for %s.%s: %s", network, station, e)
            continue

        # Process traces and store component-specific measurements
        for tr in st_temp:
            try:
                # Make a copy to avoid modifying original
                tr_wa = tr.copy()
                
                # Resample if needed (Wood-Anderson operates better on cleaned data)
                tr_wa.resample(sample_rate)
                
                # Remove instrument response to displacement if response available
                try:
                    tr_wa.remove_response(output="DISP")
                except Exception as e:
                    logging.debug("Could not remove response for %s.%s: %s", network, station, e)
                    # If no response, at least detrend and taper
                    tr_wa.detrend('demean')
                    tr_wa.taper(0.05)
                
                # Simulate Wood-Anderson seismometer
                # Parameters from Uhrhammer & Collins, 1990
                # Natural period = 0.8s, damping = 0.7, gain = 2800
                # -6.2832-4.7124j -6.2832+4.7124j
                paz_wa = {'poles': [-6.2832 + 4.7124j, -6.2832 - 4.7124j],
                         'zeros': [0j],
                         'gain': 1.0,
                         'sensitivity': 2800}
                tr_wa.simulate(paz_simulate=paz_wa)
                
                # Apply highpass to remove long-period noise
                tr_wa.filter('highpass', freq=highpass_freq)

                # trim to window around origin
                tr_wa.trim(starttime=origin_time, endtime=origin_time + 120)
                
                        
                # Initialize measurements for this station
                station_key = f"{network}.{station}"
                if station_key not in measurements:
                    measurements[station_key] = {
                        'Z': {'max_amp': np.nan, 'min_amp': np.nan, 'duration': np.nan},
                        'N': {'max_amp': np.nan, 'min_amp': np.nan, 'duration': np.nan},
                        'E': {'max_amp': np.nan, 'min_amp': np.nan, 'duration': np.nan}
                    }
                
                # Get component and check if it's one we want
                comp = tr_wa.stats.channel[-1]  # last character should be Z/N/E
                if comp not in ['Z', 'N', 'E']:
                    logging.debug("Skipping channel %s - not Z/N/E component", tr_wa.stats.channel)
                    continue
                
                # Convert to millimeters and compute measurements
                data_mm = tr_wa.data * 1000
                max_amp = float(np.max(data_mm))
                min_amp = float(np.min(data_mm))
                
                # Find times of extrema and duration
                max_idx = np.argmax(data_mm)
                min_idx = np.argmin(data_mm)
                max_time = tr_wa.times()[max_idx]
                min_time = tr_wa.times()[min_idx]
                duration = abs(max_time - min_time)
                
                # Store measurements for this component
                measurements[station_key][comp] = {
                    'max_amp': max_amp,
                    'min_amp': min_amp,
                    'duration': duration
                }

                # print if there is any no Nans
                if not np.isnan(max_amp) and not np.isnan(min_amp):
                    logging.info(
                        "Event %s Station %s.%s Component %s: max_amp=%.2f mm, min_amp=%.2f mm, duration=%.2f s",
                        event_id, network, station, comp, max_amp, min_amp, duration
                    )

            except Exception as e:
                logging.debug("Trace processing failed for %s.%s: %s", network, station, e)

    return None, measurements, origin_time


def main(argv=None):
    parser = argparse.ArgumentParser(description="Process event waveforms and compute amplitudes")
    parser.add_argument("--events", required=True, help="Path to events CSV")
    parser.add_argument("--picks", required=True, help="Path to picks CSV")
    parser.add_argument("--out", required=False, help="Output picks CSV path (default: write next to events) ")
    parser.add_argument("--sample-rate", type=int, default=100, help="Resample rate (Hz)")
    parser.add_argument("--highpass", type=float, default=4.0, help="Highpass frequency (Hz)")
    parser.add_argument("--window-before", type=int, default=30, help="Seconds before origin to request")
    parser.add_argument("--window-after", type=int, default=120, help="Seconds after origin to request")
    parser.add_argument("--fdsn", default="IRIS", help="FDSN client name (default IRIS)")
    parser.add_argument("--event-id", help="Optional: only process this event id (exact match)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    events_path = os.path.expanduser(args.events)
    picks_path = os.path.expanduser(args.picks)
    out_path = os.path.expanduser(args.out) if args.out else None

    logging.info("Reading events from %s", events_path)
    events_df = pd.read_csv(events_path)
    logging.info("Reading picks from %s", picks_path)
    picks_df = pd.read_csv(picks_path)

    # Prepare output path
    if out_path is None:
        out_dir = os.path.dirname(events_path)
        out_path = os.path.join(out_dir, os.path.basename(picks_path).replace('.csv', '_with_amplitudes.csv'))

    client = Client(args.fdsn)

    # Determine list of event ids to process
    event_col = find_column(events_df, ("event id", "event_id", "event"))
    if event_col is None:
        logging.error("Unable to find event id column in events CSV; aborting")
        sys.exit(1)
    if args.event_id:
        event_ids = [args.event_id]
    else:
        event_ids = events_df[event_col].dropna().unique().tolist()

    logging.info("Will process %d event(s)", len(event_ids))

    # Ensure amplitude column exists (preserve spaces if present in original picks)
    amplitude_col = None
    # try to find an existing amplitude column
    for c in picks_df.columns:
        if "amplitude" in c.strip().lower():
            amplitude_col = c
            break
    if amplitude_col is None:
        amplitude_col = " Amplitude "  # match notebook naming convention with spaces
        picks_df[amplitude_col] = np.nan

    for i, eid in enumerate(event_ids, start=1):
        logging.info("[%d/%d] Processing event %s", i, len(event_ids), eid)
        _, measurements, origin = process_event(
            eid,
            events_df,
            picks_df,
            client,
            sample_rate=args.sample_rate,
            highpass_freq=args.highpass,
            window_before=args.window_before,
            window_after=args.window_after,
        )

        # Store measurements back into picks_df if we got any
        if measurements:
            # Create component-specific columns if they don't exist
            for comp in ['Z', 'N', 'E']:
                for mtype in ['max_amp', 'min_amp', 'duration']:
                    col_name = f" {comp}_{mtype} "  # keeping space convention
                    if col_name not in picks_df.columns:
                        picks_df[col_name] = np.nan

            # Get station column name
            pick_station_col = find_column(picks_df, ("station name", "station", "sta"))
            pick_event_col = find_column(picks_df, ("event id", "event_id", "event"))
            
            # Update each pick's measurements
            for station_key, station_data in measurements.items():
                # Try both NET.STA and STA.NET formats
                network, sta = station_key.split('.')
                candidates = [f"{sta}.{network}", f"{network}.{sta}"]
                
                # Find matching rows in picks_df
                mask = pd.Series(False, index=picks_df.index)
                if pick_event_col is None:
                    mask = picks_df[pick_station_col].astype(str).str.strip().isin(candidates)
                else:
                    mask = (picks_df[pick_event_col] == eid) & (picks_df[pick_station_col].astype(str).str.strip().isin(candidates))
                
                # Update measurements for each component
                max_all_comps = float('-inf')
                for comp in ['Z', 'N', 'E']:
                    comp_data = station_data[comp]
                    for mtype in ['max_amp', 'min_amp', 'duration']:
                        col_name = f" {comp}_{mtype} "
                        picks_df.loc[mask, col_name] = comp_data[mtype]
                    
                    # Track maximum amplitude across components
                    if not np.isnan(comp_data['max_amp']):
                        max_all_comps = max(max_all_comps, comp_data['max_amp'])
                
                # Update the original amplitude column with maximum amplitude across components
                if max_all_comps != float('-inf'):
                    picks_df.loc[mask, amplitude_col] = max_all_comps
            
            # Print a sample of updated measurements for verification
            sample_mask = picks_df[amplitude_col].notna()
            if sample_mask.any():
                logging.info("Sample of updated measurements for event %s:", eid)
                cols_to_show = [pick_station_col, amplitude_col]
                cols_to_show.extend([f" {c}_{m} " for c in ['Z', 'N', 'E'] 
                                  for m in ['max_amp', 'min_amp', 'duration']])
                logging.info("\n%s", picks_df.loc[sample_mask, cols_to_show].head())
        # print the last few rows for verification
        print(picks_df.loc[mask].tail())
        # optional: flush to disk every N events to avoid data loss (here N=10)
        if i % 10 == 0:
            logging.info("Flushing intermediate output to %s", out_path)
            picks_df.to_csv(out_path, index=False)

    # final write
    logging.info("Writing output picks with amplitudes to %s", out_path)
    picks_df.to_csv(out_path, index=False)
    logging.info("Done")


if __name__ == "__main__":
    main()
