"""
This is a script to calculate the amplitude of the waveforms around picks 
Agentic AI was used to implement some code in this script.

Auth: Hiroto Bito
Date: 5/31/2026
"""

# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.stream import Stream

parent_dir = '/home/hbito/cascadia_obs_ensemble/utils'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_client import get_waveforms

# Read the data frame
datasets_dir =  '/wd1/hbito_data/data/datasets_all_regions'
path_assigned_picks_df = f'{datasets_dir}/Cascadia_updated_catalog_picks_assignment_ver_3.csv'

# Prepare output CSV path 
output_csv_path = f'{datasets_dir}/Cascadia_updated_catalog_picks_assignment_ver_3_w_amp.csv'

# File to save skipped picks
skipped_csv_path = f'{datasets_dir}/calculate_amplitudes_skipped_picks.csv'

assigned_picks_df = pd.read_csv(path_assigned_picks_df, index_col=False).copy()


# Define the arguments
window_before = 0.5 # in sec
window_after = 2 # in sec
source = 'pnwstore'

freq_highpass = 2 # in Hz
new_sampling_rate = 100 # in Hz

# Run the loop
amplitudes = []

for idx, row in tqdm(assigned_picks_df.iterrows(), total=len(assigned_picks_df)):

    # Define the arguments 
    date, _time = row['time'].split(' ')
    datetime_str = date+'T'+_time
    origin_time = UTCDateTime(datetime_str)  # Accept ISO string directly

    network = row['station'].split('.')[0].strip()
    station = row['station'].split('.')[1].strip()
    channel = '*H*'
    starttime = origin_time - window_before 
    endtime = origin_time + window_after

    time_pick = row['time_pick']    

    # Print the number of items in amplitudes
    print('len(amplitudes)',len(amplitudes))    

    # Request a waveform
    time.sleep(0.1)

    try:
        st = get_waveforms(network=network, station=station, channel=channel,
                            starttime=starttime, endtime=endtime,
                            source=source)
    except Exception as e:
        print(f"Request failed: {e}")

        # Save amplitude to the output DataFrame and CSV on the fly
        amp = np.nan
        amplitudes.append(amp)

        # Save skipped info to CSV
        skipped_info = {
            'network': network,
            'station': station,
            'channel': channel,
            'origin_time': origin_time,
            'time_pick': time_pick,
            'starttime': starttime,
            'endtime': endtime,
            'reason': f'Request failed: {e}'
        }
        df_skipped = pd.DataFrame([skipped_info])
        if not os.path.isfile(skipped_csv_path):
            df_skipped.to_csv(skipped_csv_path, mode='w', header=True, index=False)
        else:
            df_skipped.to_csv(skipped_csv_path, mode='a', header=False, index=False)

        continue
        

    # time.sleep(0.1)


    # Create a new stream
    sdata = Stream()
    
    # Check if loaded data have a vertical component (minimum requirement)
    has_Z = bool(st.select(id=f'{network}.{station}..??Z'))
    # Check for HH and BH channels presence
    has_HH = bool(st.select(id=f'{network}.{station}..HH?'))
    has_BH = bool(st.select(id=f'{network}.{station}..BH?'))
    has_EH = bool(st.select(id=f'{network}.{station}..EH?'))

    if not has_Z:
        e = f'No Vertical Component Data Present at {network}.{station} with HHZ, BHZ or EHZ channels at {time_pick}. Skipping'
        print(e)

        # Save amplitude to the output DataFrame and CSV on the fly
        amp = np.nan
        amplitudes.append(amp)

        # Save skipped info to CSV
        skipped_info = {
            'network': network,
            'station': station,
            'channel': channel,
            'origin_time': origin_time,
            'time_pick': time_pick,
            'starttime': starttime,
            'endtime': endtime,
            'reason': f'Request failed: {e}'
        }
        df_skipped = pd.DataFrame([skipped_info])
        if not os.path.isfile(skipped_csv_path):
            df_skipped.to_csv(skipped_csv_path, mode='w', header=True, index=False)
        else:
            df_skipped.to_csv(skipped_csv_path, mode='a', header=False, index=False)

        continue

    # Apply selection logic based on channel presence
    if has_HH:
        # If all HH, BH, and EH, channels are present, select only HH
        sdata += st.select(id=f'{network}.{station}..HH*')
    elif has_BH:
        # If BH and EH channels are present, select only BH
        sdata += st.select(id=f'{network}.{station}..BH*')
    elif has_EH:
        # If only EH channels are present, select only EH
        # NTS: This may result in getting only vertical component data - EH? is used for PNSN analog stations
        # NTS: This may also be tricky for pulling full day-volumes because the sampling rate shifts for
        #      analog stations due to the remote digitization scheme used with analog stations
        sdata += st.select(id=f'{network}.{station}..EH*')
    else:
        e = f'No data available at {network}.{station} with HHZ, BHZ or EHZ channels at {time_pick}. Skipping.'
        print(e)

        # Save amplitude to the output DataFrame and CSV on the fly
        amp = np.nan
        amplitudes.append(amp)

        # Save skipped info to CSV
        skipped_info = {
            'network': network,
            'station': station,
            'channel': channel,
            'origin_time': origin_time,
            'time_pick': time_pick,
            'starttime': starttime,
            'endtime': endtime,
            'reason': f'Request failed: {e}'
        }
        df_skipped = pd.DataFrame([skipped_info])
        if not os.path.isfile(skipped_csv_path):
            df_skipped.to_csv(skipped_csv_path, mode='w', header=True, index=False)
        else:
            df_skipped.to_csv(skipped_csv_path, mode='a', header=False, index=False)
        continue

    # Resample
    sdata.resample(new_sampling_rate)
        
    # Apply highpass filter
    sdata.detrend(type='demean')
    sdata.taper(max_percentage=0.05)
    sdata.filter(type='highpass', freq=freq_highpass)

    max_amp = 0
    for tr in sdata:
        max_amp = max(max_amp, abs(tr.data).max())

    amplitudes.append(max_amp)

# Append to DataFrame
assigned_picks_df.loc[:,"Amplitude"] = amplitudes

assigned_picks_df.to_csv(output_csv_path, index=False)