import os
import numpy as np
import matplotlib.pyplot as plt
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
import time 

#--------------Initiate clients and constants----------------#
# Define clients
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")
client_waveform = WaveformClient()

# Define constants
sampling_rate = 100  # Hz
pre_arrival_time = 50
window_length = 300

# Load the arrival table and define the output file names
assoc_df = pd.read_csv('/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
output_waveform_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveforms_EH_on_the_fly_bulk.h5"
output_metadata_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/metadata_EH_on_the_fly_bulk.csv"
error_log_file = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/save_errors_EH_on_the_fly_bulk.csv"

# Preprocess dataframe
assoc_df[['network', 'station']] = assoc_df['sta'].str.split('.', expand=True)
assoc_df['event_id'] = 'ev' + assoc_df['otime'].astype(str).str.replace('.', '_')

# Define the function to reorder the traces in a stream
def order_traces(stream: Stream, expected_len: int) -> np.ndarray:
    """
    Converts an ObsPy stream into a (3, expected_len) numpy array, 
    consistently ordered as [Z, E, N].

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
        if chan_suffix in comp_to_index:
            idx = comp_to_index[chan_suffix]
            data_list[idx] = tr.data  

    return np.vstack(data_list)  # Shape: (3, expected_len)

#--------------Gather Station Information----------------#
# Obtain unique network-station combinations
unique_ns = assoc_df.sta.unique()

# Define the start and end times for requesting station information
starttime_bulk = obspy.UTCDateTime("2010-01-01T00:00:00")
endtime_bulk = obspy.UTCDateTime("2015-12-31T23:59:59")

# Make a list of stations for bulk request 
bulk =[]
for u_ns in unique_ns:
    n,s = u_ns.split('.')

    for bi in ['EH?', 'BH?', 'HH?']:
        line = (n, s, '*', bi, starttime_bulk, endtime_bulk)
        bulk.append(line)

# Make a bulk request 
inv = client_iris.get_stations_bulk(bulk, level='channel')
time.sleep(0.2)

#--------------Gather Waveform Information----------------#
# Obtain uniquee otime-network-station combinations
unique_n_s_otime = assoc_df.drop_duplicates(['event_id', 'network', 'station'],keep='first').reset_index(drop=True)
unique_n_s_otime

# Define functions to append entries for the bulk request
def append_bulk_lists_chunks(bulk_waveforms, n, s, bi, trace_start, trace_end, day_end, next_day_start):
    """
    Append waveform requests to the bulk list based on the availability of HH? and BH? channels. If the stream runs over the midnight, split the request into two.
    """
    if day_end > trace_end:
        # If the trace end is within the same day, we can use HH?
        bulk_waveforms.append((n, s, '*', bi, trace_start, trace_end))
    else:
        # If the trace end goes beyond the day, we need to adjust
        bulk_waveforms.append((n, s, '*', bi, trace_start, day_end))
        bulk_waveforms.append((n, s, '*', bi, next_day_start, trace_end))
    return bulk_waveforms

def append_bulk_lists(bulk_waveforms, n, s, bi, trace_start, trace_end):
    """
    Append waveform requests to the bulk list based on the availability of HH? and BH? channels.
    """
    bulk_waveforms.append((n, s, '*', bi, trace_start, trace_end))

    return bulk_waveforms

# compose 
batches_bulk_waveforms_chunks =[]
batches_bulk_waveforms_chunks_ncedc =[]

batches_bulk_waveforms = []
num_batches = 100
len_batches = len(unique_n_s_otime) // num_batches

count_EH_pairs = 0

for i in tqdm(range(0, num_batches+1)):
    bulk_waveforms_chunks = []
    bulk_waveforms_chunks_ncedc = []
    bulk_waveforms = []
    time.sleep(0.2)

    for index, u_ns in islice(unique_n_s_otime.iterrows(), i*len_batches, (i + 1) * len_batches):
        n,s = u_ns['network'], u_ns['station']

        otime = u_ns['otime']
        pick_time = u_ns['pick_time']
        trace_start = obspy.UTCDateTime(otime - pre_arrival_time)
        trace_end = trace_start + window_length

        day_end = obspy.UTCDateTime(trace_start.date + timedelta(days=1))-1e-6
        next_day_start = obspy.UTCDateTime(trace_start.date + timedelta(days=1))

        # print(trace_start, trace_end)

        sta = inv.select(network=n, station=s, time=pick_time)

        has_Z = bool(sta.select(channel='??Z'))
        # has_HH = bool(sta.select(channel='HH?'))
        # has_BH = bool(sta.select(channel='BH?'))
        has_EH = bool(sta.select(channel='EH?'))    

        if not has_Z or not has_EH:
            count_HH_BH_pairs += 1
            # print("count_EH_pairs", count_EH_pairs)
            continue

        if has_HH:
            if n in ['NC', 'BK']:
                bulk_waveforms_chunks_ncedc = append_bulk_lists_chunks(bulk_waveforms_chunks_ncedc, n, s, 'EH?', trace_start, trace_end, day_end, next_day_start)
            else:
                bulk_waveforms_chunks = append_bulk_lists_chunks(bulk_waveforms_chunks, n, s, 'EH?', trace_start, trace_end, day_end, next_day_start)
            
            bulk_waveforms = append_bulk_lists(bulk_waveforms, n, s, 'EH?', trace_start, trace_end)

        else:
            if n in ['NC', 'BK']:
                bulk_waveforms_chunks_ncedc = append_bulk_lists_chunks(bulk_waveforms_chunks_ncedc, n, s, 'EH?', trace_start, trace_end, day_end, next_day_start)
            else:
                bulk_waveforms_chunks = append_bulk_lists_chunks(bulk_waveforms_chunks, n, s, 'EH?', trace_start, trace_end, day_end, next_day_start)
            
            bulk_waveforms = append_bulk_lists(bulk_waveforms, n, s, 'EH?', trace_start, trace_end)

    batches_bulk_waveforms_chunks.append(bulk_waveforms_chunks)
    batches_bulk_waveforms_chunks_ncedc.append(bulk_waveforms_chunks_ncedc)
    batches_bulk_waveforms.append(bulk_waveforms)

#--------------Create Waveform Datasets in batches----------------#
# Find entries that have already been processed
processed_keys = set()
if os.path.exists(output_metadata_file):
    processed_df = pd.read_csv(output_metadata_file)
    processed_keys = set(zip(processed_df['trace_start_time'], processed_df['station_network_code'], processed_df['station_code']))
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

# Define the expected length of the samples
expected_len = int(sampling_rate * window_length)

i_iter = 0

for i in range(len(batches_bulk_waveforms)):
# for i in range(1):
    print("Batch",i)
    batch_chunk = batches_bulk_waveforms_chunks[i]
    batch_chunk_ncedc = batches_bulk_waveforms_chunks_ncedc[i]
    batch = batches_bulk_waveforms[i]

    save_errors = []


    st = Stream()

    for j in range(len(batch_chunk)):
        n, s, loc, bi, trace_start_time, trace_end_time = batch_chunk[j]
        try: 
            st1 = client_waveform.get_waveforms(network=n, station=s, location=loc, channel=bi,
                                                starttime=trace_start_time, endtime=trace_end_time)
            st.extend(st1)
        except Exception as e:
            print(f"Error fetching waveforms for {n}.{s} {bi} from {trace_start_time} to {trace_end_time}: {e}")
            # Write error immediately
            continue
    print('Finished downloading from WaveformClient')

    for j in range(len(batch_chunk_ncedc)):
        n, s, loc, bi, trace_start_time, trace_end_time = batch_chunk_ncedc[j]
        try: 
            st2 = client_ncedc.get_waveforms(network=n, station=s, location=loc, channel=bi,
                                                 starttime=trace_start_time, endtime=trace_end_time)
            time.sleep(0.2)
            st.extend(st2)
        except Exception as e:
            print(f"Error fetching waveforms for {n}.{s} {bi} from {trace_start_time} to {trace_end_time}: {e}")
            # Write error immediately
            continue
    print('Finished downloading NCEDC')    

    

    # print("Requesting waveforms.")
    # if len(batch_chunk) != 0:
    #     st1 = client_waveform.get_waveforms_bulk(batch_chunk)
    #     time.sleep(0.2) # Stop the execution to avoid making too many requests to the server
    # if len(batch_chunk_ncedc) != 0:
    #     st2 = client_ncedc.get_waveforms_bulk(batch_chunk_ncedc)
    #     time.sleep(0.2) # Stop the execution to avoid making too many requests to the server
    # if len(st1) == 0 and len(st2) == 0:
    #     print(f"Batch {i+1} has no waveform requests.")
    #     continue

    time.sleep(0.2) # Stop the execution to avoid making too many requests to the server

    # st = st1.extend(st2) if len(st2) != 0 else st1

    for n_s_time in tqdm(batch):
        i_iter += 1
        network, station, location, channel, trace_start_time, trace_end_time = n_s_time

        rows_sta  = assoc_df.loc[(assoc_df['sta'] == f"{network}.{station}") & (abs(assoc_df['otime'] - float(trace_start_time + timedelta(seconds=pre_arrival_time))) < 1)]
        

        p_arrival = rows_sta[rows_sta['iphase'] == 'P']
        s_arrival = rows_sta[rows_sta['iphase'] == 'S']

        key = (str(trace_start_time), network, station)
        if key in processed_keys:
            print(f"Skipping already processed entry: {key}")
            # time.sleep(0.2)
            continue

        # inv_n_s_time = inv.select(network=network, station=station, location=location, channel='*',
        #                            starttime=trace_start_time, endtime=trace_end_time)

        # inv_n_s_time = inv.select(network=network, station=station, location=location, channel='*')
        # print('inv_n_s_time', inv_n_s_time)
        st_n_s = st.select(id=f"{network}.{station}.*.{channel}",)
        # print('st_n_s', st_n_s)

        st_n_s_time = Stream([tr for tr in st_n_s if tr.stats.starttime > (trace_start_time-1) and tr.stats.endtime < (trace_end_time+1)]) # Tolerate the error of 1 second when selecting the traces in the stream for the specific time window
        st_n_s_time.merge(method=0, fill_value='interpolate')
        st_n_s_time.detrend()
        st_n_s_time.resample(sampling_rate)

        cleaned_stream = Stream()
        # print('st_n_s_time', st_n_s_time)
        for tr in st_n_s_time:
            trace_data = tr.data[:expected_len]
            if len(trace_data) < expected_len:
                trace_data = np.pad(trace_data, (0, expected_len - len(trace_data)), mode="constant") # Pads zeros at the end
            tr.data = trace_data
            cleaned_stream.append(tr)

        # print('cleaned_stream', cleaned_stream)

        _cleaned_stream = order_traces(cleaned_stream, expected_len)

        try:
            data = np.stack(_cleaned_stream, axis=0)
    #         data = np.stack([tr.data[:window_length * sampling_rate - 2] for tr in waveform], axis=0)
        except Exception as e:
            # Write error immediately
            file_exists = os.path.exists(error_log_file)
            with open(error_log_file, "a", newline="") as errfile:
                writer = csv.DictWriter(errfile, fieldnames=['i_iter', 'network', 'station', 'starttime', 'endtime', 'stage', 'error'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'i_iter': i_iter, 'network': network, 'station': station, 'starttime': trace_start_time, 'endtime': trace_end_time, 'stage': 'metadata_write', 'error': str(e)})
            continue

        bucket = str(random.randint(0, 10))
        
        
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
            # Write error immediately
            file_exists = os.path.exists(error_log_file)
            with open(error_log_file, "a", newline="") as errfile:
                writer = csv.DictWriter(errfile, fieldnames=['i_iter', 'network', 'station', 'starttime', 'endtime', 'stage', 'error'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'i_iter': i_iter, 'network': network, 'station': station, 'starttime': trace_start_time, 'endtime': trace_end_time, 'stage': 'metadata_write', 'error': str(e)})
            continue

        trace_name = f"{bucket}${dataset_index},:{data.shape[0]},:{data.shape[1]}"

#         print(network, station, location, channel, trace_start_time, trace_end_time)
        # print(rows_sta)
        # print(rows_sta['lat'].iloc[0])
        # print(rows_sta['lat'].iloc[0])
        # print(rows_sta['lon'].iloc[0])
        # print(rows_sta['depth'].iloc[0])
        # print(s_arrival['pick_time'].iloc[0] if not s_arrival.empty else None)
        # print(inv_n_s_time[0][0].latitude)
#         print(cleaned_stream[0].stats.channel[:-1])


        try:
            row = {
                'event_id': rows_sta['event_id'].iloc[0],
                'source_origin_time': rows_sta['otime'].iloc[0],
                'source_latitude_deg': rows_sta['lat'].iloc[0],
                'source_longitude_deg': rows_sta['lon'].iloc[0],
                'source_type': "earthquake",
                'source_depth_km': rows_sta['depth'].iloc[0],
                'preferred_source_magnitude': None,
                'preferred_source_magnitude_type': None,
                'preferred_source_magnitude_uncertainty': None,
                'source_depth_uncertainty_km': None,
                'source_horizontal_uncertainty_km': None,
                'station_network_code': network,
                'station_channel_code': cleaned_stream[0].stats.channel[:-1],
                'station_code': station,
                'station_location_code': "",
                'station_latitude_deg': None,
                'station_longitude_deg': None,
                'station_elevation_m': None,
                'trace_name': trace_name,
                'trace_sampling_rate_hz': sampling_rate,
                'trace_start_time': trace_start_time,
                'trace_S_arrival_sample': int((s_arrival['pick_time'].iloc[0] - (rows_sta['otime'].iloc[0] - pre_arrival_time)) * sampling_rate)if not s_arrival.empty else None,
                'trace_P_arrival_sample': int((p_arrival['pick_time'].iloc[0] - (rows_sta['otime'].iloc[0] - pre_arrival_time)) * sampling_rate) if not p_arrival.empty else None,
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
            print(f"Error writing metadata for {station}/{trace_start_time}: {e}")
            # Write error immediately
            file_exists = os.path.exists(error_log_file)
            with open(error_log_file, "a", newline="") as errfile:
                writer = csv.DictWriter(errfile, fieldnames=['i_iter', 'network', 'station', 'starttime', 'endtime', 'stage', 'error'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'i_iter': i_iter, 'network': network, 'station': station, 'starttime': trace_start_time, 'endtime': trace_end_time, 'stage': 'metadata_write', 'error': str(e)})
            continue
            

h5f.close()
meta_out.close()