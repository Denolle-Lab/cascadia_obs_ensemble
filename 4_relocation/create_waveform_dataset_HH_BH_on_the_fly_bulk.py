import os
import logging
import numpy as np
import pandas as pd
import h5py
import obspy
from obspy import Stream
from obspy.clients.fdsn import Client
from pnwstore.mseed import WaveformClient
from obspy import Stream
from datetime import timedelta
from tqdm import tqdm
import csv
import random
from itertools import islice

# === Setup ===
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = WaveformClient()

# Constants
sampling_rate = 100  # Hz
pre_arrival_time = 50
window_length = 150

# Input/output paths
assoc_df = pd.read_csv('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
output_waveform_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveforms_HH_BH_on_the_fly_bulk.h5"
output_metadata_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/metadata_HH_BH_on_the_fly_bulk.csv"
error_log_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/save_errors_on_the_fly_bulk.csv"

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
from datetime import timedelta


def get_waveform_across_midnight(client, network, station, location, channel, starttime, endtime, event_id=None):
    """
    Fetch waveform in 1-day chunks to avoid 'multi-day streaming not implemented' errors.
    """
    st_total = Stream()

    current = starttime
    while current < endtime:
        # Define the chunk end as the end of the same day (23:59:59.999999)
        day_end = current.date + timedelta(days=1)
        chunk_end = obspy.UTCDateTime(day_end) - 1e-6  # 23:59:59.999999

        # Clip chunk_end to not go beyond requested endtime
        chunk_end = min(chunk_end, endtime)

        try:
            st_chunk = client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=current,
                endtime=chunk_end
            )
            st_total += st_chunk
        except Exception as e:
            print(f"Error fetching waveform for {event_id} {network}.{station} {current} - {chunk_end}: {e}")

        if chunk_end <= current:
            print(f"Breaking to avoid infinite loop at {current}")
            break

        # Advance to the next microsecond to avoid overlap
        current = chunk_end + 1e-6

    return st_total

# do batched requests
# form the bulk request
# use id wird 
# time.sleep(0.01)


def pad_waveform_stream(stream: Stream, expected_len: int) -> np.ndarray:
    """
    Converts an ObsPy stream into a (3, expected_len) numpy array, 
    consistently ordered as [Z, E, N], padding missing components with zeros.

    Parameters:
    - stream: ObsPy Stream containing cleaned traces (padded to expected_len)
    - expected_len: Target length of each waveform trace

    Returns:
    - data_array: np.ndarray of shape (3, expected_len)
    """
    # Fixed component order: Z → 0, E → 1, N → 2
    comp_to_index = {"Z": 0, "E": 1, "N": 2}
    data_list = [np.zeros(expected_len) for _ in range(3)]  # Default to zeros

    for tr in stream:
        chan_suffix = tr.stats.channel[-1]
#         print('chan_suffix',chan_suffix)
        if chan_suffix in comp_to_index:
            idx = comp_to_index[chan_suffix]
#             print('idx',idx)
            data_list[idx] = tr.data[:expected_len]  # Assumes already padded/truncated

    return np.vstack(data_list)  # Shape: (3, expected_len)

count_EH_pairs = 0 
count_any_pair = 0
i_iter = 0

# Pull unique network.station codes present in associated picks
unique_ns = assoc_df.sta.unique()

# compose 
bulk =[]
for u_ns in unique_ns:
    n,s = u_ns.split('.')

    for bi in ['EH', 'BH','HH?']:
        line = (n, s, '*', bi)
        bulk.append(line)

inv = client_iris.get_stations_bulk(bulk, level='level')
    


# === Main Loop ===
for (event_id, network, station), group in tqdm(group_iter):
    i_iter += 1
    # print("i_iter", i_iter)
    
    key = (event_id, network, station)
    if key in processed_keys:
        continue

    p_arrival = group[group['iphase'] == 'P']
    s_arrival = group[group['iphase'] == 'S']
#     if s_arrival.empty:
        
#         continue

    first_arrival = group['otime'].min()
    trace_start = obspy.UTCDateTime(first_arrival - pre_arrival_time)
    trace_end = trace_start + window_length

    # 

    # Get station metadata
    sta = inv.select(network=network, station=station,starttime=trace_start,endtime=trace_end)

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
        save_errors.append({'event_id': event_id, 'i_iter': i_iter, 'station': station, 'starttime': trace_start, 'endtime': trace_end, 'stage': 'waveform_fetch', 'error': str(e)})
        continue
        
    count_any_pair += 1
    # print("count_any_pair", count_any_pair)

    has_Z = bool(_waveform.select(channel='??Z'))
    has_HH = bool(_waveform.select(channel='HH?'))
    has_BH = bool(_waveform.select(channel='BH?'))
    if not has_Z or not (has_HH or has_BH):
        count_EH_pairs += 1
        # print("count_EH_pairs", count_EH_pairs)
        continue

    waveform = _waveform.select(channel='HH?' if has_HH else 'BH?')
    waveform = sorted(waveform, key=lambda tr: tr.stats.channel)
    
    # Define expected length (samples)
    expected_len = int(window_length * sampling_rate)

    # Clean and pad each trace to expected_len
    cleaned_stream = Stream()
    for tr in _waveform:
        trace_data = tr.data[:expected_len]
        if len(trace_data) < expected_len:
            trace_data = np.pad(trace_data, (0, expected_len - len(trace_data)), mode="constant")
        tr.data = trace_data
        cleaned_stream.append(tr)
#     print('_waveform 1',_waveform)
    # Now pad to shape (3, expected_len) with consistent channel order
    _waveform = pad_waveform_stream(cleaned_stream, expected_len)

#     print('_waveform 2',_waveform)

#     print(waveform)
    try:
        data = np.stack(_waveform, axis=0)
#         data = np.stack([tr.data[:window_length * sampling_rate - 2] for tr in waveform], axis=0)
    except Exception as e:
        save_errors.append({'event_id': event_id, 'i_iter': i_iter, 'station': station, 'starttime': trace_start, 'endtime': trace_end, 'stage': 'stack_waveform', 'error': str(e)})
        continue

    bucket = str(random.randint(0, 10))
#     trace_name = f"{bucket}${0},:{data.shape[0]},:{data.shape[1]}"
    

    try:
        dset_path = f"/data/{bucket}"
        if dset_path not in h5f:
            h5f.create_dataset(dset_path, data=np.expand_dims(data, axis=0), maxshape=(None, *data.shape), chunks=True, dtype='float32')
            dataset_index = 0
        else:
            dset = h5f[dset_path]
            dataset_index = dset.shape[0]
            dset.resize((dataset_index + 1), axis=0)
            dset[dataset_index] = data
    except Exception as e:
        print(f"Error writing to HDF5 for bucket {bucket}: {e}")
        save_errors.append({'event_id': event_id, 'i_iter': i_iter, 'station': station, 'starttime': trace_start, 'endtime': trace_end, 'stage': 'hdf5_write', 'error': str(e)})
        continue
        
    trace_name = f"{bucket}${dataset_index},:{data.shape[0]},:{data.shape[1]}"


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
            'trace_S_arrival_sample': int((s_arrival['pick_time'].iloc[0] - (first_arrival - pre_arrival_time)) * sampling_rate)if not s_arrival.empty else None,
            'trace_P_arrival_sample': int((p_arrival['pick_time'].iloc[0] - (first_arrival - pre_arrival_time)) * sampling_rate) if not p_arrival.empty else None,
            'trace_S_arrival_uncertainty_s': None,
            'trace_P_arrival_uncertainty_s': None,
            'trace_P_polarity': None,
            'trace_S_onset': "impulsive"if not s_arrival.empty else None,
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
        save_errors.append({'event_id': event_id, 'i_iter': i_iter, 'station': station, 'starttime': trace_start, 'endtime': trace_end, 'stage': 'csv_write', 'error': str(e)})
        continue
        
    if save_errors:
        file_exists = os.path.exists(error_log_file)
        with open(error_log_file, "a", newline="") as errfile:
            writer = csv.DictWriter(errfile, fieldnames=['event_id', 'num_sta', 'i_iter','station', 'starttime', 'endtime', 'stage', 'error'])
            if not file_exists:
                writer.writeheader()
            writer.writerows(save_errors)
        print(f"Appended {len(save_errors)} errors to {error_log_file}")

# === Cleanup ===
h5f.close()
meta_out.close()


print("Done with incremental save.")
