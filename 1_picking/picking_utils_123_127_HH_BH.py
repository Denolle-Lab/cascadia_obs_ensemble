import logging
import os

from obspy.clients.fdsn import Client
import numpy as np
import obspy
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pandas as pd
import dask
from dask.diagnostics import ProgressBar

from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy import Stream

from pnwstore.mseed import WaveformClient
import torch
import numpy as np
from tqdm import tqdm
import time 
import pandas as pd
import gc
import seisbench.models as sbm
from ELEP.elep.ensemble_statistics import ensemble_statistics
from ELEP.elep.ensemble_coherence import ensemble_semblance 
from ELEP.elep.trigger_func import picks_summary_simple

""" 
module containing functions needed in parallel_pick_201?.py
""" 

device = torch.device("cpu")

# Define clients
client_inventory = Client('IRIS')
client_waveform = WaveformClient()
client_ncedc = Client('NCEDC')

twin = 6000     # length of time window
step = 3000     # step length
l_blnd, r_blnd = 500, 500

def stacking(data, npts, l_blnd, r_blnd, nseg):
    _data = data.copy()
    stack = np.full(npts, np.nan, dtype = np.float32)
    _data[:, :l_blnd] = np.nan; _data[:, -r_blnd:] = np.nan
    stack[:twin] = _data[0, :]
    for iseg in range(nseg-1):
        idx = step*(iseg+1)
        stack[idx:idx + twin] = \
                np.nanmax([stack[idx:idx + twin], _data[iseg+1, :]], axis = 0)
    return stack

def run_detection(network,station,t1,t2,filepath,twin,step,l_blnd,r_blnd,lat,lon,elev):
    # Define tstring
    tstring = t1.strftime('%Y%m%d')

    if os.path.exists(filepath+station+'_'+tstring+'.csv'):
        print('File '+filepath+station+'_'+tstring+'.csv already exists')
        return
    # print the file path 
    print('test1')
    print(filepath+station+'_'+tstring+'.csv')
    
	# Load data
	# Reshape data
	# Predict on base models
	# Stack
	# Create and write csv file. Define file name using the station code and the input filepath
    network = network
    channels = '?H?'
    
   
    
    # Get waveforms and filter
    try:
        if network in ['NC', 'BK']:
            # Query waveforms
            _sdata = client_ncedc.get_waveforms(network=network, station=station, location="*", channel=channels,
                                               starttime=UTCDateTime(t1), endtime=UTCDateTime(t1 + timedelta(days=1)))
        else: 
            _sdata = client_waveform.get_waveforms(network=network, station=station, channel=channels, 
                                              year=t1.strftime('%Y'), month=t1.strftime('%m'), 
                                              day=t1.strftime('%d'))
    except obspy.clients.fdsn.header.FDSNNoDataException:
        print(f"WARNING: No data for {network}.{station}.{channels} on {t1}.")
        return
    
# Create a new stream
    sdata = Stream()
# Check for HH and BH channels presence
    has_HH = bool(_sdata.select(channel="HH?"))
    has_BH = bool(_sdata.select(channel="BH?"))

    # Apply selection logic based on channel presence
    if has_HH and has_BH:
        # If both HH and BH channels are present, select only HH
        sdata += _sdata.select(channel="HH?")
    elif has_HH:
        # If only HH channels are present
        sdata += _sdata.select(channel="HH?")
    elif has_BH:
        # If only BH channels are present
        sdata += _sdata.select(channel="BH?")

    ###############################
    # If no data returned, skipping
    if len(sdata) == 0:
        logging.warning("No stream returned. Skipping.")
        return
    ###############################
    
    sdata.filter(type='bandpass',freqmin=4,freqmax=15)
    
    ###############################
    sdata.merge(fill_value='interpolate') # fill gaps if there are any.
    ###############################

    # Get the necassary information about the station
    delta = sdata[0].stats.delta
    starttime = sdata[0].stats.starttime
    fs = sdata[0].stats.sampling_rate
    dt = 1/fs
    

    # Make all the traces in the stream have the same lengths
    max_starttime = max([tr.stats.starttime for tr in sdata])
    min_endtime = min([tr.stats.endtime for tr in sdata])
    
    for tr in sdata:
        tr.trim(starttime=max_starttime,endtime=min_endtime, nearest_sample=True)    
        
    # Reshaping data
    arr_sdata = np.array(sdata)
    npts = arr_sdata.shape[1]
    ############################### avoiding errors at the end of a stream
   #nseg = int(np.ceil((npts - twin) / step)) + 1
    nseg = int(np.floor((npts - twin) / step)) + 1
    ###############################
    windows = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
    tap = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 6)))
    
    # Define the parameters for semblance
    paras_semblance = {'dt':dt, 'semblance_order':2, 'window_flag':True, 
                   'semblance_win':0.5, 'weight_flag':'max'}
    p_thrd, s_thrd = 0.05, 0.05

    windows_std = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
    windows_max = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
    windows = np.zeros(shape=(nseg, 3, twin), dtype= np.float32)
    windows_idx = np.zeros(nseg, dtype=np.int32)

    for iseg in range(nseg):
        idx = iseg * step
        windows[iseg, :] = arr_sdata[:, idx:idx + twin]
        windows[iseg, :] -= np.mean(windows[iseg, :], axis=-1, keepdims=True)
        # original use std norm
        windows_std[iseg, :] = windows[iseg, :] / np.std(windows[iseg, :]) + 1e-10
        # others use max norm
        windows_max[iseg, :] = windows[iseg, :] / (np.max(np.abs(windows[iseg, :]), axis=-1, keepdims=True))
        windows_idx[iseg] = idx

    # taper
    windows_std[:, :, :6] *= tap; windows_std[:, :, -6:] *= tap[::-1]; 
    windows_max[:, :, :6] *= tap; windows_max[:, :, -6:] *= tap[::-1];
    del windows

    print(f"Window data shape: {windows_std.shape}")
    
    # Predict on base models
    pretrain_list = ['original', 'ethz', 'instance', 'scedc', 'stead']

    # dim 0: 0 = P, 1 = S
    batch_pred = np.zeros([2, len(pretrain_list), nseg, twin], dtype = np.float32) 
    for ipre, pretrain in enumerate(pretrain_list):
        print('test10')
        t0 = time.time()
        eqt = sbm.EQTransformer.from_pretrained(pretrain)
        eqt.to(device);
        eqt._annotate_args['overlap'] = ('Overlap between prediction windows in samples \
                                        (only for window prediction models)', step)
        eqt._annotate_args['blinding'] = ('Number of prediction samples to discard on \
                                         each side of each window prediction', (l_blnd, r_blnd))
        eqt.eval();
        if pretrain == 'original':
            # batch prediction through torch model
            windows_std_tt = torch.Tensor(windows_std)
            _torch_pred = eqt(windows_std_tt.to(device))
        else:
            windows_max_tt = torch.Tensor(windows_max)
            _torch_pred = eqt(windows_max_tt.to(device))
        batch_pred[0, ipre, :] = _torch_pred[1].detach().cpu().numpy()
        batch_pred[1, ipre, :] = _torch_pred[2].detach().cpu().numpy()

    # clean up memory
    del _torch_pred, windows_max_tt, windows_std_tt
    del windows_std, windows_max
    gc.collect()
    torch.cuda.empty_cache()

    print(f"All prediction shape: {batch_pred.shape}")
    
    smb_pred = np.zeros([2, nseg, twin], dtype = np.float32)
    # calculate the semblance
    ## the semblance may takes a while bit to calculate
    
    for iseg in range(nseg):
    #############################
        # 0 for P-wave
        smb_pred[0, iseg, :] = ensemble_semblance(batch_pred[0, :, iseg, :], paras_semblance)

        # 1 for P-wave
        smb_pred[1, iseg, :] = ensemble_semblance(batch_pred[1, :, iseg, :], paras_semblance)

    ## ... and stack
    # 0 for P-wave
    smb_p = stacking(smb_pred[0, :], npts, l_blnd, r_blnd, nseg)

    # 1 for P-wave
    smb_s = stacking(smb_pred[1, :], npts, l_blnd, r_blnd, nseg)
  
    del smb_pred, batch_pred

    p_index = picks_summary_simple(smb_p, p_thrd)
    s_index = picks_summary_simple(smb_s, s_thrd)
    print(f"{len(p_index)} P picks\n{len(s_index)} S picks")
    
    print('test2')
    # Create lists and a data frame
    event_id = []
    source_type = []
    station_network_code = []
    station_channel_code = []
    station_code = []
    station_location_code = []
    station_latitude_deg= []
    station_longitude_deg = []
    station_elevation_m = []
    trace_name = []
    trace_sampling_rate_hz = []
    trace_start_time = []
    trace_S_arrival_sample = []
    trace_P_arrival_sample = []
    trace_S_onset = []
    trace_P_onset = []
    trace_snr_db = []
    trace_p_arrival = []
    trace_s_arrival = []
    
    print("This is the cwd:"+str(os.getcwd()))
    print('test3',len(p_index),len(s_index))
    for i, idx in enumerate(p_index):
        event_id.append(' ')
        source_type.append(' ')
        station_network_code.append(network)   # Change to otehr networks
        station_channel_code.append(' ')
        station_code.append(station)
        station_location_code.append(sdata[0].stats.location) 
        print('test3-1')
        station_latitude_deg.append(lat)
        print('test3-2')
        station_longitude_deg.append(lon) 
        print('test3-3')
        station_elevation_m.append(elev)
        print('test3-4')
        trace_name.append(' ')
        trace_sampling_rate_hz.append(sdata[0].stats.sampling_rate)
        print('test3-5')
        trace_start_time.append(sdata[0].stats.starttime)
        trace_S_arrival_sample.append(' ')
        trace_P_arrival_sample.append(' ')
        trace_S_onset.append(' ')
        trace_P_onset.append(' ')
        trace_snr_db.append(' ')
        trace_s_arrival.append(np.nan)
        print('test3-6')
        trace_p_arrival.append(str(starttime  + idx * delta))

    print('test4')
    for i, idx in enumerate(s_index):
        event_id.append(' ')
        source_type.append(' ')
        station_network_code.append(network) # Change to otehr networks
        station_channel_code.append(' ')
        station_code.append(station)
        station_location_code.append(sdata[0].stats.location)   
        print('test3-7')
        station_latitude_deg.append(lat)
        print('test3-8')
        station_longitude_deg.append(lon)
        print('test3-9')
        station_elevation_m.append(elev)
        print('test3-10')
        trace_name.append(' ')
        trace_sampling_rate_hz.append(sdata[0].stats.sampling_rate)
        trace_start_time.append(sdata[0].stats.starttime)
        trace_S_arrival_sample.append(' ')
        trace_P_arrival_sample.append(' ')
        trace_S_onset.append(' ')
        trace_P_onset.append(' ')
        trace_snr_db.append(' ')
        trace_s_arrival.append(str(starttime  + idx * delta))
        print('test3-11')
        trace_p_arrival.append(np.nan)
    print('test5')
    # dictionary of lists
    dict = {'event_id':event_id,'source_type':source_type,'station_network_code':station_network_code,\
            'station_channel_code':station_channel_code,'station_code':station_code,'station_location_code':station_location_code,\
            'station_latitude_deg':station_latitude_deg,'station_longitude_deg':station_longitude_deg, \
            'station_elevation_m':station_elevation_m,'trace_name':trace_name,'trace_sampling_rate_hz':trace_sampling_rate_hz,\
            'trace_start_time':trace_start_time,'trace_S_arrival_sample':trace_S_arrival_sample,\
            'trace_P_arrival_sample':trace_P_arrival_sample, 'trace_S_onset':trace_S_onset,'trace_P_onset':trace_P_onset,\
            'trace_snr_db':trace_snr_db, 'trace_s_arrival':trace_s_arrival, 'trace_p_arrival':trace_p_arrival}

    df = pd.DataFrame(dict)
    
    print('test6')
    # Make the specific day into a string:
    tstring = t1.strftime('%Y%m%d')
    # Build the full file name:
    print("test7")
    print("This is the cwd:"+str(os.getcwd()))
    print('This is the filepath:'+str(filepath))
    file_name = filepath+station+'_'+tstring+'.csv'
    ##################################################
    # Write to file using that name
    print(file_name,'this is before test9')
    print(df)
    print(f"P and S summary:\n{len(p_index)} P picks\n{len(s_index)} S picks")
    df.to_csv(file_name)
    print('test9')