import os
import logging
from obspy.clients.fdsn import Client
import numpy as np
import obspy
from obspy import Stream
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pandas as pd
import h5py
from tqdm import tqdm
from itertools import islice
import random


# Placeholder for client instances
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = client_iris  # replace with actual waveform client if different

# Load the assignment data frames of picks
assoc_df = pd.read_csv('/wd1/hbito_data/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
print("assoc_df.head()", assoc_df.head())

# Start making the datasets
# Initiate module logger
Logger = logging.getLogger(__name__)

# Define the output file names for the waveform data and metadata
output_waveform_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/waveforms_HH_BH_test1.h5"
output_metadata_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/metadata_HH_BH_test1.csv"

output_waveform_file_EH = "/wd1/hbito_data/data/datasets_all_regions/waveforms_EH_test1.h5"
output_metadata_file_EH = "/wd1/hbito_data/data/datasets_all_regions/metadata_EH_test1.csv"

# Constants
sampling_rate = 100  # Hz
pre_arrival_time = 50
window_length = 150


# Replace with your actual DataFrame
# df = pd.read_csv(...) or load your 'close_to_midnight'
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
        next_day = current.date + timedelta(days=1)  # Correct way to get the next day
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


group_iter = df.groupby(['event_id', 'network', 'station'])
for (event_id, network, station), group in islice(group_iter, 200):
    print("-" * 50)
    print("network:", network)
    print("station:", station)

    p_arrival = group[group['iphase'] == 'P']
    s_arrival = group[group['iphase'] == 'S']
    if s_arrival.empty:
        print(f"No S arrival for event {event_id} at station {station}")

    first_arrival = group['otime'].min()
    trace_start = first_arrival - pre_arrival_time
    trace_end = trace_start + window_length

    otime = obspy.UTCDateTime(first_arrival)
    trace_start1 = obspy.UTCDateTime(trace_start)
    trace_end1 = obspy.UTCDateTime(trace_end)

    print(f"Trace start: {trace_start1}, Trace end: {trace_end1}")

    try:
        sta = client_iris.get_stations(network=network, station=station, location="*", channel="*",
                                       starttime=trace_start1, endtime=trace_end1)
    except Exception as e:
        print("Error during download or processing station info:", e)
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

    except Exception as e:
        print("Error during waveform processing:", e)
        continue
    
    # print(_waveform)
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
        Logger.warning('No Vertical Component Data Present. Skipping')
        continue

    if has_HH:
        waveform += _waveform.select(channel="HH?")
    elif has_BH:
        waveform += _waveform.select(channel="BH?")
    elif has_EH:
        waveform += _waveform.select(channel="EHZ")
    else:
        continue

    waveform = sorted(waveform, key=lambda tr: tr.stats.channel)

    for i, tr in enumerate(waveform):
        print(f"Trace {i}: id={tr.id}, channel={tr.stats.channel}, shape={tr.data.shape}")

    station_channel_code = waveform[0].stats.channel[:-1]
    data = np.stack([tr.data[:window_length * sampling_rate - 2] for tr in waveform], axis=0)
    num_trs, num_tps = data.shape

    p_sample = int((p_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not p_arrival.empty else None
    s_sample = int((s_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not s_arrival.empty else None

    # Create row
    if has_HH or has_BH: # If the waveforms are from HH or BH channels
        bucket_HH_BH = str(random.randint(0, 10))
        index_HH_BH = len(waveform_buckets_HH_BH[bucket_HH_BH])
        trace_name = f"{bucket_HH_BH}${index_HH_BH},:{num_trs},:{num_tps}"
        waveform_buckets_HH_BH[bucket_HH_BH].append(data)

    elif has_EH: # IF the waveforms are from EH channels
        bucket_EH = str(random.randint(0, 10))
        print(f"EH bucket: {bucket_EH}")
        index_EH = len(waveform_buckets_EH[bucket_EH])
        trace_name = f"{bucket_EH}${index_EH},:{num_trs},:{num_tps}"
        waveform_buckets_EH[bucket_EH].append(data)

    else:
        print(f"No valid channel data for event {event_id} at station {station}")
        continue

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
        'station_channel_code': station_channel_code,
        'station_code': station,
        'station_location_code': "",
        'station_latitude_deg': slat,
        'station_longitude_deg': slon,
        'station_elevation_m': selev,
        'trace_name': trace_name,
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
        ##################################################


    # Create row
    if has_HH or has_BH: # If the waveforms are from HH or BH channels
        rows_HH_BH.append(row)

    elif has_EH: # IF the waveforms are from EH channels
        rows_EH.append(row)

    else:
        print(f"No valid channel data for event {event_id} at station {station}")
        continue
        
    ##################################################

# Create the seisbench metadata dataframe
seisbench_df_HH_BH = pd.DataFrame(rows_HH_BH)
seisbench_df_EH = pd.DataFrame(rows_EH)

# Display the first few rows
print(f"Created {len(seisbench_df_HH_BH)} metadata entries for HH/BH channels")
print(f"Created {len(seisbench_df_EH)} metadata entries for EH channels")

# === WRITE TO HDF5 ===
print(f"Saving waveform data to {output_waveform_file_HH_BH}...")
with h5py.File(output_waveform_file_HH_BH, "w") as f:
    for bucket, traces in waveform_buckets_HH_BH.items():
        print(f"HH/BH bucket: {bucket}, number of traces: {len(traces)}")
        if not traces:
            continue
        for tr in traces:
            print("HH/BH trace shape:", tr.shape)
            print("HH/BH trace:", tr)
        arr = np.stack(traces, axis=0)  # (N, 3, T)
        print("HH/BH arr shape:", arr.shape)
        f.create_dataset(f"/data/{bucket}", data=arr, dtype="float32")

print(f"Saving waveform data to {output_waveform_file_EH}...")
with h5py.File(output_waveform_file_EH, "w") as f:
    for bucket, traces in waveform_buckets_EH.items():
        print(f"EH bucket: {bucket}, number of traces: {len(traces)}")
        if not traces:
            continue
        for tr in traces:
            print("EH trace shape:", tr.shape)
            print("EH trace:", tr)

        arr = np.stack(traces, axis=0)  # (N, 3, T)
        print("EH arr shape:", arr.shape)
        f.create_dataset(f"/data/{bucket}", data=arr, dtype="float32")

# === WRITE METADATA ===
print(f"Saving metadata to {output_metadata_file_HH_BH}...")
seisbench_df_HH_BH.to_csv(output_metadata_file_HH_BH, index=False)   

print(f"Saving metadata to {output_metadata_file_EH}...")
seisbench_df_EH.to_csv(output_metadata_file_EH, index=False)   

print("Done.")
