# create_waveform_dataset.py (save individual traces grouped by stream within buckets)

import os
import logging
from obspy.clients.fdsn import Client
import numpy as np
import pickle
import obspy
from obspy import Stream
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pandas as pd
import h5py
from tqdm import tqdm
import random

client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = client_iris

assoc_df = pd.read_csv('/wd1/hbito_data/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)

Logger = logging.getLogger(__name__)

output_waveform_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/waveforms_HH_BH_save_each_trace_test.h5"
output_metadata_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/metadata_HH_BH_save_each_trace_test.csv"
output_waveform_file_EH = "/wd1/hbito_data/data/datasets_all_regions/waveforms_EH_save_each_trace_test.h5"
output_metadata_file_EH = "/wd1/hbito_data/data/datasets_all_regions/metadata_EH_save_each_trace_test.csv"

sampling_rate = 100
pre_arrival_time = 50
window_length = 150


df = assoc_df.copy()
df[['network', 'station']] = df['sta'].str.split('.', expand=True)
df['event_id'] = 'ev' + df['otime'].astype(str).str.replace('.', '_')

waveform_buckets_HH_BH = {str(i): [] for i in range(11)}
waveform_buckets_EH = {str(i): [] for i in range(11)}
rows_HH_BH, rows_EH = [], []

def get_waveform_across_midnight(client, network, station, location, channel, starttime, endtime):
    stream = Stream()
    current = starttime
    while current < endtime:
        next_day = current.date + timedelta(days=1)
        chunk_end = min(obspy.UTCDateTime(next_day), endtime)
        try:
            st_chunk = client.get_waveforms(
                network=network, station=station,
                location=location, channel=channel,
                starttime=current, endtime=chunk_end
            )
            stream += st_chunk
        except Exception as e:
            print(f"Failed to get data from {current} to {chunk_end}: {e}")
        current = chunk_end
    return stream
groups = list(df.groupby(['event_id', 'network', 'station']))[:5]

for (event_id, network, station), group in tqdm(groups, total=len(groups), desc="Processing events"):
    p_arrival = group[group['iphase'] == 'P']
    s_arrival = group[group['iphase'] == 'S']
    if s_arrival.empty:
        continue

    first_arrival = group['otime'].min()
    trace_start = first_arrival - pre_arrival_time
    trace_end = trace_start + window_length

    otime = obspy.UTCDateTime(first_arrival)
    trace_start1 = obspy.UTCDateTime(trace_start)
    trace_end1 = obspy.UTCDateTime(trace_end)

    try:
        sta = client_iris.get_stations(network=network, station=station, location="*", channel="*", starttime=trace_start1, endtime=trace_end1)
    except Exception:
        continue

    try:
        if network in ['NC', 'BK']:
            _waveform = get_waveform_across_midnight(client_ncedc, network, station, "*", "*", trace_start1, trace_end1)
        else:
            _waveform = get_waveform_across_midnight(client_waveform, network, station, "*", "?H?", trace_start1, trace_end1)

        _waveform.merge(method=1, fill_value='interpolate')
        for tr in _waveform:
            tr.data = tr.data.astype(np.float64)
        _waveform.trim(trace_start1, trace_end1, pad=True, fill_value=0.0)
        _waveform.detrend()
        _waveform.resample(sampling_rate)
    except Exception:
        continue

    olat = group['lat'].iloc[0]
    olon = group['lon'].iloc[0]
    odepth = group['depth'].iloc[0] * 1000
    slat = sta[0][0].latitude
    slon = sta[0][0].longitude
    selev = sta[0][0].elevation

    waveform = Stream()
    has_Z = bool(_waveform.select(channel='??Z'))
    has_HH = bool(_waveform.select(channel="HH?"))
    has_BH = bool(_waveform.select(channel="BH?"))
    has_EH = bool(_waveform.select(channel="EH?"))

    if not has_Z:
        continue

    if has_HH:
        waveform += _waveform.select(channel="HH?")
    elif has_BH:
        waveform += _waveform.select(channel="BH?")
    elif has_EH:
        waveform += _waveform.select(channel="EH?")
    else:
        continue

    waveform = sorted(waveform, key=lambda tr: tr.stats.channel)

    expected_len = window_length * sampling_rate
    clean_traces = []
    for tr in waveform:
        trace_data = tr.data[:expected_len]
        if len(trace_data) < expected_len:
            trace_data = np.pad(trace_data, (0, expected_len - len(trace_data)), mode="constant")
        elif len(trace_data) > expected_len:
            trace_data = trace_data[:expected_len]
        clean_traces.append(trace_data)

    p_sample = int((p_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not p_arrival.empty else None
    s_sample = int((s_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not s_arrival.empty else None

    row = {
        'event_id': event_id,
        'source_origin_time': otime,
        'source_latitude_deg': olat,
        'source_longitude_deg': olon,
        'source_type': "earthquake",
        'source_depth_km': odepth,
        'preferred_source_magnitude': None,
        'preferred_source_magnitude_type': None,
        'preferred_source_magnitude_uncertainty': None,
        'source_depth_uncertainty_km': None,
        'source_horizontal_uncertainty_km': None,
        'station_network_code': network,
        'station_channel_code': waveform[0].stats.channel[:-1],
        'station_code': station,
        'station_location_code': "",
        'station_latitude_deg': slat,
        'station_longitude_deg': slon,
        'station_elevation_m': selev,
        'trace_name': "",  # Will be updated below
        'trace_sampling_rate_hz': sampling_rate,
        'trace_start_time': trace_start1,
        'trace_S_arrival_sample': s_sample,
        'trace_P_arrival_sample': p_sample,
        'trace_S_arrival_uncertainty_s': None,
        'trace_P_arrival_uncertainty_s': None,
        'trace_P_polarity': None,
        'trace_S_onset': "impulsive" if s_sample is not None else None,
        'trace_P_onset': "impulsive" if p_sample is not None else None,
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

    if has_HH or has_BH:
        bucket = str(random.randint(0, 10))
        index = len(waveform_buckets_HH_BH[bucket])
        trace_name = f"HH_BH/data/{bucket}/stream_{index}"
        row['trace_name'] = trace_name
        waveform_buckets_HH_BH[bucket].append(clean_traces)
        rows_HH_BH.append(row)
    elif has_EH:
        bucket = str(random.randint(0, 10))
        index = len(waveform_buckets_EH[bucket])
        trace_name = f"EH/data/{bucket}/stream_{index}"
        row['trace_name'] = trace_name
        waveform_buckets_EH[bucket].append(clean_traces)
        rows_EH.append(row)

# with open('/wd1/hbito_data/data/datasets_all_regions/waveform_buckets_HH_BH_save_each_trace.pkl', 'wb') as f:
#     pickle.dump(waveform_buckets_HH_BH, f)

# with open('/wd1/hbito_data/data/datasets_all_regions/waveform_buckets_EH.pkl', 'wb') as f:
#     pickle.dump(waveform_buckets_EH, f)

pd.DataFrame(rows_HH_BH).to_csv(output_metadata_file_HH_BH, index=False)
pd.DataFrame(rows_EH).to_csv(output_metadata_file_EH, index=False)

with h5py.File(output_waveform_file_HH_BH, "w") as f:
    for bucket, streams in waveform_buckets_HH_BH.items():
        grp = f.create_group(f"/data/{bucket}")
        for i, stream in enumerate(streams):
            stream_grp = grp.create_group(f"stream_{i}")
            for j, tr in enumerate(stream):
                stream_grp.create_dataset(f"trace_{j}", data=tr, dtype="float32")

with h5py.File(output_waveform_file_EH, "w") as f:
    for bucket, streams in waveform_buckets_EH.items():
        grp = f.create_group(f"/data/{bucket}")
        for i, stream in enumerate(streams):
            stream_grp = grp.create_group(f"stream_{i}")
            for j, tr in enumerate(stream):
                stream_grp.create_dataset(f"trace_{j}", data=tr, dtype="float32")

print("Done.")
