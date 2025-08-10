import os
import logging
import numpy as np
import pandas as pd
import h5py
import obspy
from obspy import Stream
from obspy.clients.fdsn import Client
from pnwstore.mseed import WaveformClient
from datetime import timedelta
from tqdm import tqdm
import csv
import random

# === Setup ===
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = WaveformClient()

# Constants
sampling_rate = 100  # Hz
pre_arrival_time = 50
window_length = 150

# Input/output paths
assoc_df = pd.read_csv('/wd1/hbito_data/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
output_waveform_file = "/wd1/hbito_data/data/datasets_all_regions/waveforms_HH_BH_on_the_fly.h5"
output_metadata_file = "/wd1/hbito_data/data/datasets_all_regions/metadata_HH_BH_on_the_fly.csv"
error_log_file = "/wd1/hbito_data/data/datasets_all_regions/save_errors.csv"

# Logger
Logger = logging.getLogger(__name__)

# Preprocess dataframe
assoc_df[['network', 'station']] = assoc_df['sta'].str.split('.', expand=True)
assoc_df['event_id'] = 'ev' + assoc_df['otime'].astype(str).str.replace('.', '_')
group_iter = assoc_df.groupby(['event_id', 'network', 'station'])

# Resume support
processed_keys = set()
if os.path.exists(output_metadata_file):
    processed_df = pd.read_csv(output_metadata_file)
    processed_keys = set(zip(processed_df['event_id'], processed_df['station_network_code'], processed_df['station_code']))
    print(f"Loaded {len(processed_keys)} processed entries.")

# Open output files
h5f = h5py.File(output_waveform_file, "a")
meta_out = open(output_metadata_file, "a")
write_header = os.stat(output_metadata_file).st_size == 0 if os.path.exists(output_metadata_file) else True

fieldnames = ['event_id', 'source_origin_time', 'source_latitude_deg', 'source_longitude_deg',
              'source_type', 'source_depth_km', 'preferred_source_magnitude', 'preferred_source_magnitude_type',
              'preferred_source_magnitude_uncertainty', 'source_depth_uncertainty_km', 'source_horizontal_uncertainty_km',
              'station_network_code', 'station_channel_code', 'station_code', 'station_location_code',
              'station_latitude_deg', 'station_longitude_deg', 'station_elevation_m', 'trace_name',
              'trace_sampling_rate_hz', 'trace_start_time', 'trace_S_arrival_sample', 'trace_P_arrival_sample',
              'trace_S_arrival_uncertainty_s', 'trace_P_arrival_uncertainty_s', 'trace_P_polarity',
              'trace_S_onset', 'trace_P_onset', 'trace_snr_db', 'source_type_pnsn_label',
              'source_local_magnitude', 'source_local_magnitude_uncertainty', 'source_duration_magnitude',
              'source_duration_magnitude_uncertainty', 'source_hand_magnitude', 'trace_missing_channel', 'trace_has_offset']

meta_writer = csv.DictWriter(meta_out, fieldnames=fieldnames)
if write_header:
    meta_writer.writeheader()

# Error log
save_errors = []

# === Helper ===
def get_waveform_across_midnight(client, network, station, location, channel, starttime, endtime, event_id=None):
    stream = Stream()
    current = starttime
    while current < endtime:
        next_day = current.date + timedelta(days=1)
        chunk_end = min(obspy.UTCDateTime(next_day), endtime)
        st_chunk = client.get_waveforms(
            network=network, station=station, location=location, channel=channel,
            starttime=current, endtime=chunk_end
        )
        stream += st_chunk
        current = chunk_end
    return stream

count_EH_pairs = 0 

# === Main Loop ===
for (event_id, network, station), group in tqdm(group_iter):
    key = (event_id, network, station)
    if key in processed_keys:
        continue

    p_arrival = group[group['iphase'] == 'P']
    s_arrival = group[group['iphase'] == 'S']
    if s_arrival.empty:
        continue

    first_arrival = group['otime'].min()
    trace_start = obspy.UTCDateTime(first_arrival - pre_arrival_time)
    trace_end = trace_start + window_length

    # Get station metadata
    try:
        sta = client_iris.get_stations(network=network, station=station, location="*", channel="*", starttime=trace_start, endtime=trace_end)
    except Exception as e:
        save_errors.append({'event_id': event_id, 'station': station, 'stage': 'station_metadata', 'error': str(e)})
        continue

    # Get waveform
    try:
        client = client_ncedc if network in ['NC', 'BK'] else client_waveform
        _waveform = get_waveform_across_midnight(client, network, station, "*", "?H?", trace_start, trace_end)
        _waveform.merge(method=1, fill_value='interpolate')
        for tr in _waveform:
            tr.data = tr.data.astype(np.float64)
        _waveform.trim(trace_start, trace_end, pad=True, fill_value=0.0)
        _waveform.detrend()
        _waveform.resample(sampling_rate)
    except Exception as e:
        print(f"Error fetching waveform for {event_id} / {station}: {e}")
        save_errors.append({'event_id': event_id, 'station': station, 'stage': 'waveform_fetch', 'error': str(e)})
        continue

    has_Z = bool(_waveform.select(channel='??Z'))
    has_HH = bool(_waveform.select(channel='HH?'))
    has_BH = bool(_waveform.select(channel='BH?'))
    if not has_Z or not (has_HH or has_BH):
        count_EH_pairs += 1
        print("count_EH_pairs", count_EH_pairs)
        continue

    waveform = _waveform.select(channel='HH?' if has_HH else 'BH?')
    waveform = sorted(waveform, key=lambda tr: tr.stats.channel)

    try:
        data = np.stack([tr.data[:window_length * sampling_rate - 2] for tr in waveform], axis=0)
    except Exception as e:
        save_errors.append({'event_id': event_id, 'station': station, 'stage': 'stack_waveform', 'error': str(e)})
        continue

    bucket = str(random.randint(0, 10))
    trace_name = f"{bucket}${0},:{data.shape[0]},:{data.shape[1]}"

    try:
        dset_path = f"/data/{bucket}"
        if dset_path not in h5f:
            h5f.create_dataset(dset_path, data=np.expand_dims(data, axis=0), maxshape=(None, *data.shape), chunks=True, dtype='float32')
        else:
            dset = h5f[dset_path]
            dset.resize((dset.shape[0] + 1), axis=0)
            dset[-1] = data
    except Exception as e:
        print(f"Error writing to HDF5 for bucket {bucket}: {e}")
        save_errors.append({'event_id': event_id, 'station': station, 'stage': 'hdf5_write', 'error': str(e)})
        continue

    try:
        row = {
            'event_id': event_id,
            'source_origin_time': obspy.UTCDateTime(first_arrival),
            'source_latitude_deg': group['lat'].iloc[0],
            'source_longitude_deg': group['lon'].iloc[0],
            'source_type': "earthquake",
            'source_depth_km': group['depth'].iloc[0],
            'preferred_source_magnitude': None,
            'preferred_source_magnitude_type': None,
            'preferred_source_magnitude_uncertainty': None,
            'source_depth_uncertainty_km': None,
            'source_horizontal_uncertainty_km': None,
            'station_network_code': network,
            'station_channel_code': waveform[0].stats.channel[:-1],
            'station_code': station,
            'station_location_code': "",
            'station_latitude_deg': sta[0][0].latitude,
            'station_longitude_deg': sta[0][0].longitude,
            'station_elevation_m': sta[0][0].elevation,
            'trace_name': trace_name,
            'trace_sampling_rate_hz': sampling_rate,
            'trace_start_time': trace_start,
            'trace_S_arrival_sample': int((s_arrival['pick_time'].iloc[0] - (first_arrival - pre_arrival_time)) * sampling_rate),
            'trace_P_arrival_sample': int((p_arrival['pick_time'].iloc[0] - (first_arrival - pre_arrival_time)) * sampling_rate) if not p_arrival.empty else None,
            'trace_S_arrival_uncertainty_s': None,
            'trace_P_arrival_uncertainty_s': None,
            'trace_P_polarity': None,
            'trace_S_onset': "impulsive",
            'trace_P_onset': "impulsive" if not p_arrival.empty else None,
            'trace_snr_db': None,
            'source_type_pnsn_label': None,
            'source_local_magnitude': None,
            'source_local_magnitude_uncertainty': None,
            'source_duration_magnitude': None,
            'source_duration_magnitude_uncertainty': None,
            'source_hand_magnitude': None,
            'trace_missing_channel': "",
            'trace_has_offset': None
        }
        meta_writer.writerow(row)
        meta_out.flush()
    except Exception as e:
        print(f"Error writing metadata for {event_id} / {station}: {e}")
        save_errors.append({'event_id': event_id, 'station': station, 'stage': 'csv_write', 'error': str(e)})
        continue

# === Cleanup ===
h5f.close()
meta_out.close()

if save_errors:
    with open(error_log_file, "w", newline="") as errfile:
        writer = csv.DictWriter(errfile, fieldnames=['event_id', 'station', 'stage', 'error'])
        writer.writeheader()
        writer.writerows(save_errors)
    print(f"Logged {len(save_errors)} errors to {error_log_file}")

print("Done with incremental save.")
