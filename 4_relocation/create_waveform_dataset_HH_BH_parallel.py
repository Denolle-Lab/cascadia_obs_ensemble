import os
import logging
import numpy as np
import pickle
import obspy
import pandas as pd
import h5py
import random
from tqdm import tqdm
from datetime import timedelta
from itertools import islice
from multiprocessing import Pool, cpu_count

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from pnwstore.mseed import WaveformClient

# Clients
client_waveform = WaveformClient()

# Constants
sampling_rate = 100  # Hz
pre_arrival_time = 50
window_length = 150

# Inputs
assoc_df = pd.read_csv('/wd1/hbito_data/data/datasets_all_regions/arrival_assoc_origin_2010_2015_reloc_cog_ver3.csv', index_col=0)
assoc_df[['network', 'station']] = assoc_df['sta'].str.split('.', expand=True)
assoc_df['event_id'] = 'ev' + assoc_df['otime'].astype(str).str.replace('.', '_')

# Outputs
output_waveform_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/waveforms_HH_BH_parallel.h5"
output_metadata_file_HH_BH = "/wd1/hbito_data/data/datasets_all_regions/metadata_HH_BH_parallel.csv"
output_pickle = "/home/hbito/cascadia_obs_ensemble_backup/data/datasets_all_regions/waveforms_HH_BH_parallel.pkl"
failed_requests_path = "/wd1/hbito_data/data/datasets_all_regions/failed_waveform_requests_parallel.csv"
error_log_path = output_waveform_file_HH_BH.replace(".h5", "_save_errors.csv")

# Logging setup
Logger = logging.getLogger(__name__)

# Global waveform clients (must be recreated per process if needed)
client_iris = Client("IRIS")
client_ncedc = Client("NCEDC")


def get_waveform_across_midnight(client, network, station, location, channel, starttime, endtime, event_id=None, failed_requests=None):
    stream = Stream()
    current = starttime
    while current < endtime:
        next_day = current.date + timedelta(days=1)
        chunk_end = min(UTCDateTime(next_day), endtime)
        try:
            st_chunk = client.get_waveforms(network=network, station=station,
                                            location=location, channel=channel,
                                            starttime=current, endtime=chunk_end)
            stream += st_chunk
        except Exception as e:
            failed_requests.append({
                'event_id': event_id,
                'network': network,
                'station': station,
                'start_time': str(current),
                'end_time': str(chunk_end),
                'error': str(e)
            })
        current = chunk_end
    return stream


def process_group(args):
    (event_id, network, station), group = args
    result = {
        "bucket": None,
        "trace_data": None,
        "metadata": None,
        "error": None,
        "failed_waveform_requests": []
    }

    try:
        p_arrival = group[group['iphase'] == 'P']
        s_arrival = group[group['iphase'] == 'S']
        if s_arrival.empty:
            return result  # skip if no S arrival

        first_arrival = group['otime'].min()
        trace_start = first_arrival - pre_arrival_time
        trace_end = trace_start + window_length

        otime = UTCDateTime(first_arrival)
        trace_start1 = UTCDateTime(trace_start)
        trace_end1 = UTCDateTime(trace_end)

        try:
            sta = client_iris.get_stations(network=network, station=station, location="*", channel="*",
                                           starttime=trace_start1, endtime=trace_end1)
        except Exception as e:
            result['error'] = f"[station] {event_id} {station} {e}"
            return result

        # Get waveform
        try:
            if network in ['NC', 'BK']:
                _waveform = get_waveform_across_midnight(client_ncedc, network, station, "*", "*", trace_start1, trace_end1, event_id, result['failed_waveform_requests'])
            else:
                _waveform = get_waveform_across_midnight(client_waveform, network, station, "*", "?H?", trace_start1, trace_end1, event_id, result['failed_waveform_requests'])

            _waveform.merge(method=1, fill_value='interpolate')
            for tr in _waveform:
                tr.data = tr.data.astype(np.float64)
            _waveform.trim(trace_start1, trace_end1, pad=True, fill_value=0.0)
            _waveform.detrend()
            _waveform.resample(sampling_rate)
        except Exception as e:
            result['error'] = f"[waveform] {event_id} {station} {e}"
            return result

        # Metadata
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

        if not has_Z or (not has_HH and not has_BH):
            return result

        if has_HH:
            waveform += _waveform.select(channel="HH?")
        elif has_BH:
            waveform += _waveform.select(channel="BH?")
        waveform = sorted(waveform, key=lambda tr: tr.stats.channel)

        station_channel_code = waveform[0].stats.channel[:-1]
        data = np.stack([tr.data[:window_length * sampling_rate - 2] for tr in waveform], axis=0)

        p_sample = int((p_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not p_arrival.empty else None
        s_sample = int((s_arrival['pick_time'].iloc[0] - trace_start) * sampling_rate) if not s_arrival.empty else None

        bucket = str(random.randint(0, 10))
        result["bucket"] = bucket
        result["trace_data"] = data

        trace_name = f"{bucket}${random.randint(0, 999999)},:{data.shape[0]},:{data.shape[1]}"
        result["metadata"] = {
            'event_id': event_id,
            'source_origin_time': otime,
            'source_latitude_deg': olat,
            'source_longitude_deg': olon,
            'source_type': "earthquake",
            'source_depth_km': odepth,
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

    except Exception as e:
        result["error"] = f"[unexpected] {event_id} {station}: {str(e)}"

    return result


if __name__ == "__main__":
    waveform_buckets_HH_BH = {str(i): [] for i in range(11)}
    rows_HH_BH = []
    failed_waveform_requests = []
    save_errors = []

    group_iter = list(assoc_df.groupby(['event_id', 'network', 'station']))
    num_workers = min(cpu_count(), 1)

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_group, group_iter), total=len(group_iter)):
            if result["error"]:
                print(f"{result['error']}")
            if result["trace_data"] is not None:
                waveform_buckets_HH_BH[result["bucket"]].append(result["trace_data"])
            if result["metadata"] is not None:
                rows_HH_BH.append(result["metadata"])
            if result["failed_waveform_requests"]:
                failed_waveform_requests.extend(result["failed_waveform_requests"])

    # Save failed downloads
    if failed_waveform_requests:
        pd.DataFrame(failed_waveform_requests).to_csv(failed_requests_path, index=False)

    # Save metadata
    seisbench_df_HH_BH = pd.DataFrame(rows_HH_BH)
    seisbench_df_HH_BH.to_csv(output_metadata_file_HH_BH, index=False)

    # Save waveform pickle
    with open(output_pickle, 'wb') as f:
        pickle.dump(waveform_buckets_HH_BH, f)

    # Save waveform HDF5
    with h5py.File(output_waveform_file_HH_BH, "w") as f:
        for bucket, traces in waveform_buckets_HH_BH.items():
            if not traces:
                continue
            try:
                arr = np.stack(traces, axis=0)
                f.create_dataset(f"/data/{bucket}", data=arr, dtype="float32")
            except Exception as e:
                save_errors.append({
                    "bucket": bucket,
                    "num_traces": len(traces),
                    "error": str(e)
                })

    if save_errors:
        pd.DataFrame(save_errors).to_csv(error_log_path, index=False)
        print(f"Save errors occurred. Logged at: {error_log_path}")

    print("Done.")
