import os
import logging
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from obspy.clients.fdsn import Client
from obspy import Stream, UTCDateTime
from datetime import timedelta
import pickle
import random

# Set up logging to capture only error messages
logger = logging.getLogger('waveform_logger')
logger.setLevel(logging.DEBUG)

error_handler = logging.FileHandler('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveform_errors.log')
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)

logger.addHandler(error_handler)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder for client instances
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = client_iris

# Load the assignment data frames of picks
assoc_df = pd.read_csv('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
print("assoc_df.head()", assoc_df.head())

# Define the output file names
output_waveform_file_HH_BH = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveforms_HH_BH.h5"
output_metadata_file_HH_BH = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/metadata_HH_BH.csv"

output_waveform_file_EH = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveforms_EH.h5"
output_metadata_file_EH = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/metadata_EH.csv"

sampling_rate = 100
pre_arrival_time = 50
window_length = 150

df = assoc_df.copy()
df[['network', 'station']] = df['sta'].str.split('.', expand=True)
df['event_id'] = 'ev' + df['otime'].astype(str).str.replace('.', '_')

waveform_buckets_HH_BH = {str(i): [] for i in range(11)}
waveform_buckets_EH = {str(i): [] for i in range(11)}
rows_HH_BH, rows_EH = [], []

def process_trace_tensor(trace_data_np):
    trace_tensor = torch.tensor(trace_data_np, dtype=torch.float32, device=device)
    trace_tensor = (trace_tensor - trace_tensor.mean()) / (trace_tensor.std() + 1e-8)
    return trace_tensor

def get_waveform_across_midnight(client, network, station, location, channel, starttime, endtime):
    stream = Stream()
    current = starttime
    while current < endtime:
        next_day = current.date + timedelta(days=1)
        chunk_end = min(UTCDateTime(next_day), endtime)
        try:
            st_chunk = client.get_waveforms(
                network=network, station=station,
                location=location, channel=channel,
                starttime=current, endtime=chunk_end
            )
            stream += st_chunk
        except Exception as e:
            logger.error(f"Failed to get data from {current} to {chunk_end}: {e}")
        current = chunk_end
    return stream

group_iter = df.groupby(['event_id', 'network', 'station'])
for (event_id, network, station), group in tqdm(group_iter, total=len(group_iter), desc="Processing events"):
    print("-" * 50)
    print("network:", network)
    print("station:", station)

    p_arrival = group[group['iphase'] == 'P']
    s_arrival = group[group['iphase'] == 'S']

    first_arrival = group['otime'].min()
    trace_start = first_arrival - pre_arrival_time
    trace_end = trace_start + window_length

    trace_start1 = UTCDateTime(trace_start)
    trace_end1 = UTCDateTime(trace_end)

    try:
        sta = client_iris.get_stations(network=network, station=station, location="*", channel="*", starttime=trace_start1, endtime=trace_end1)
    except Exception as e:
        logger.error("Error during download or processing station info: %s", e)
        continue

    try:
        if network in ['NC', 'BK']:
            _waveform = get_waveform_across_midnight(client_ncedc, network, station, "*", "*", trace_start1, trace_end1)
        else:
            _waveform = get_waveform_across_midnight(client_waveform, network, station, "*", "?H?", trace_start1, trace_end1)

        _waveform.merge(method=1, fill_value='interpolate')
        _waveform.trim(trace_start1, trace_end1, pad=True, fill_value=0.0)
        _waveform.detrend()
        _waveform.resample(sampling_rate)

    except Exception as e:
        logger.error("Error during waveform processing: %s", e)
        continue

    olat, olon, odepth = group['lat'].iloc[0], group['lon'].iloc[0], group['depth'].iloc[0] * 1000
    slat, slon, selev = sta[0][0].latitude, sta[0][0].longitude, sta[0][0].elevation

    waveform = Stream()
    if _waveform.select(channel='??Z'):
        if _waveform.select(channel="HH?"):
            waveform += _waveform.select(channel="HH?")
        elif _waveform.select(channel="BH?"):
            waveform += _waveform.select(channel="BH?")
        elif _waveform.select(channel="EH?"):
            waveform += _waveform.select(channel="EHZ")
        else:
            continue

        waveform = sorted(waveform, key=lambda tr: tr.stats.channel)

        data_np = np.stack([tr.data[:window_length * sampling_rate - 2].astype(np.float32) for tr in waveform], axis=0)
        data_tensor = process_trace_tensor(data_np)
        data = data_tensor.cpu().numpy()

        p_sample = int((p_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not p_arrival.empty else None
        s_sample = int((s_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not s_arrival.empty else None

        bucket = str(random.randint(0, 10))
        trace_name = f"{bucket}${len(waveform_buckets_HH_BH[bucket] if waveform[0].stats.channel.startswith(('HH', 'BH')) else waveform_buckets_EH[bucket])},:{data.shape[0]},:{data.shape[1]}"

        if waveform[0].stats.channel.startswith("HH") or waveform[0].stats.channel.startswith("BH"):
            waveform_buckets_HH_BH[bucket].append(data)
            rows_HH_BH.append({
                'event_id': event_id,
                'source_origin_time': UTCDateTime(first_arrival),
                'source_latitude_deg': olat,
                'source_longitude_deg': olon,
                'source_type': "earthquake",
                'source_depth_km': odepth,
                'station_network_code': network,
                'station_channel_code': waveform[0].stats.channel[:-1],
                'station_code': station,
                'station_latitude_deg': slat,
                'station_longitude_deg': slon,
                'station_elevation_m': selev,
                'trace_name': trace_name,
                'trace_sampling_rate_hz': sampling_rate,
                'trace_start_time': trace_start1,
                'trace_S_arrival_sample': s_sample,
                'trace_P_arrival_sample': p_sample,
            })
        else:
            waveform_buckets_EH[bucket].append(data)
            rows_EH.append({
                'event_id': event_id,
                'source_origin_time': UTCDateTime(first_arrival),
                'source_latitude_deg': olat,
                'source_longitude_deg': olon,
                'source_type': "earthquake",
                'source_depth_km': odepth,
                'station_network_code': network,
                'station_channel_code': waveform[0].stats.channel[:-1],
                'station_code': station,
                'station_latitude_deg': slat,
                'station_longitude_deg': slon,
                'station_elevation_m': selev,
                'trace_name': trace_name,
                'trace_sampling_rate_hz': sampling_rate,
                'trace_start_time': trace_start1,
                'trace_S_arrival_sample': s_sample,
                'trace_P_arrival_sample': p_sample,
            })

# Save waveform data and metadata

pd.DataFrame(rows_HH_BH).to_csv(output_metadata_file_HH_BH, index=False)
pd.DataFrame(rows_EH).to_csv(output_metadata_file_EH, index=False)

with open('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveform_buckets_HH_BH.pkl', 'wb') as f:
    pickle.dump(waveform_buckets_HH_BH, f)

with open('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveform_buckets_EH.pkl', 'wb') as f:
    pickle.dump(waveform_buckets_EH, f)

with h5py.File(output_waveform_file_HH_BH, "w") as f:
    for bucket, traces in waveform_buckets_HH_BH.items():
        if traces:
            arr = np.stack(traces, axis=0)
            f.create_dataset(f"/data/{bucket}", data=arr, dtype="float32")

with h5py.File(output_waveform_file_EH, "w") as f:
    for bucket, traces in waveform_buckets_EH.items():
        if traces:
            arr = np.stack(traces, axis=0)
            f.create_dataset(f"/data/{bucket}", data=arr, dtype="float32")

print("Done.")
