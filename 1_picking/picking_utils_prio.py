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
from obspy import Stream, Trace
from obspy.signal.trigger import trigger_onset

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
# Initialize module logger 
Logger = logging.getLogger(__name__)

device = torch.device("cpu")

# Define clients
client_inventory = Client('IRIS')
client_waveform = WaveformClient()
client_ncedc = Client('NCEDC')

twin = 6000     # length of time window
step = 3000     # step length
l_blnd, r_blnd = 500, 500

def pred_trigger_pick(pred, source_trace, label, thrd=0.1, **kwargs):
    """Use a simple, single-threshold trigger to detect local maxima
    in a positive valued prediction time-series output by EQTransformer

    :param pred: model prediction output time-series 
    :type pred: numpy.ndarray
    :param source_trace: source obspy Trace-like object from which **pred**
    :type source_trace: obspy.core.trace.Trace
    :param label: prediction label
    :type label: str, optional
    :param thrd: Detection threshold for this label, defaults to 0.1
    :type thrd: float, optional
    :param kwargs: key-word argument collector passed to :meth:`~obspy.signal.trigger.trigger_onset`

    :raises ValueError: _description_
    :raises TypeError: _description_
    :return: picks
    :rtype: pandas.core.dataframe.DataFrame
    """    
    if len(pred) < 300:
        raise ValueError('insufficient samples in pred')
    if not isinstance(source_trace, Trace):
        raise TypeError('source_trace must be type obspy.Trace')
    pred[:300] = 0.
    triggers = trigger_onset(pred, thrd, thrd, **kwargs)
    t0 = source_trace.stats.starttime
    sr = source_trace.stats.sampling_rate
    id = source_trace.id
    picks = []
    
    for s0, s1 in triggers:
        to = t0 + s0/sr
        tp = t0 + (s0 + np.argmax(pred[s0:s1+1]))/sr
        tf = t0 + s1/sr
        pv = np.max(pred[s0:s1+1])
        line = id[:-1].split('.') + [label, t0, to, tp, tf, pv, thrd]
        # pick = Pick(trace_id = id, start_time=to, end_time=tf, peak_time=tp, peak_value=pv, phase=phz_name)
        picks.append(line)
    
    picks = pd.DataFrame(data=picks,columns=['network','station','location','band_inst','label','trace_starttime',
                                             'trigger_onset','pick_time','trigger_offset','max_prob','thresh_prob'])
    return picks

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

def run_detection(network,station,t1,t2,filepath,twin,step,l_blnd,r_blnd):
    """Run an ensemble machine learning model semblance detection workflow on a
    specified job

    :param network: network code for the station being analyzed
    :type network: str
    :param station: station code for the station being analyzed
    :type station: str
    :param t1: start time for waveform records to analyze
    :type t1: pandas.core.timestamp.Timestamp
    :param t2: end time for waveform records to analyze
    :type t2: pandas.core.timestamp.Timestamp
    :param filepath: file name and path to save results to (do not include *.csv extension)
    :type filepath: str
    :param twin: scale of the input layer for the specified ML model architecture
    :type twin: int
    :param step: samples to advance subsequent windows
    :type step: int
    :param l_blnd: number of samples to ignore ("blind") on the left side of each
        model input
    :type l_blnd: int
    :param r_blnd: number of samples to blind on the right side of each model input
    :type r_blnd: int
    :param lat: station location latitude (NTS: Obsolite this...)
    :type lat: float
    :param lon: station location longitude (NTS: Obsolite this...)
    :type lon: float
    :param elev: station elevation (NTS: Obsolite this...)
    :type elev: float
    """    
    # TODO: Should install compatability checks for inputs


    # Columns for the end-product
    columns = ['network','station','band','instrument','label','t0','sr','p_trig','i_on','i_max','i_off','pmax']
    
    # Define tstring
    tstring = t1.strftime('%Y%m%d')
    tstring2 = t2.strftime('%Y%m%d')
    save_file_name = filepath+network+'_'+station+'_'+tstring+'_'+tstring2+'.csv'
    
    check_filepath = filepath.replace("_122-123_46-50/", "_122-129/")
    #################################
    check_file_name = check_filepath+network+'_'+station+'_'+tstring+'_'+tstring2+'.csv'
    if os.path.exists(check_file_name):
        print(f'File {check_file_name} already exists')
        return
    #################################
    # Safety catch against overwriting previous analyses
    if os.path.exists(save_file_name):
        print(f'File {save_file_name} already exists')
        return
    # print the file path 
    print('test1')
    print(save_file_name)
    
	# Load data
	# Reshape data
	# Predict on base models
	# Stack
	# Create and write csv file. Define file name using the station code and the input filepath
    network = network
    channels = '*'
    
   
    
    # Get waveforms and filter
    # NTS: This sampling scheme leans heavily on prior data QC going into PNW store
    try:
        if network in ['NC', 'BK']:
            # Query waveforms
            _sdata = client_ncedc.get_waveforms(network=network, station=station, location="*", channel=channels,
                                               starttime=UTCDateTime(t1), endtime=UTCDateTime(t2))
        else:
            # Shouldn't this have an explicit starttime + endtime inputs?
            _sdata = client_waveform.get_waveforms(network=network, station=station, channel=channels, 
                                              year=t1.strftime('%Y'), month=t1.strftime('%m'), 
                                              day=t1.strftime('%d'))
    except obspy.clients.fdsn.header.FDSNNoDataException:
        Logger.warning(f"WARNING: No data for {network}.{station}.{channels} on {t1} - {t2}.")
        return
    
    # Create a new stream
    sdata = Stream()
    # Check if loaded data have a vertical component (minimum requirement)
    has_Z = bool(_sdata.select(channel='??Z'))
    # Check for HH and BH channels presence
    has_HH = bool(_sdata.select(channel="HH?"))
    has_BH = bool(_sdata.select(channel="BH?"))
    has_EH = bool(_sdata.select(channel="EH?"))
#     has_EN = bool(_sdata.select(channel="EN?"))


#     # Apply selection logic based on channel presence
#     if has_HH and has_BH and has_EH and has_EN:
#         # If all HH, BH, EH, and EN channels are present, select only HH
#         sdata += _sdata.select(channel="HH?")
#     elif has_BH and has_EH and has_EN:
#         # If BH, EH, and EN channels are present, select only BH
#         sdata += _sdata.select(channel="BH?")
#     elif has_EH and has_EN:
#         # If only EH and EN channels are present, select only EH
#         sdata += _sdata.select(channel="EH?")
#     elif has_EN:
#         # If only EN channels are present
#         sdata += _sdata.select(channel="EN?")
    if not has_Z:
        Logger.warning('No Vertical Component Data Present. Skipping')
        return
   # Apply selection logic based on channel presence
    if has_HH:
        # If all HH, BH, EH, and EN channels are present, select only HH
        sdata += _sdata.select(channel="HH?")
    elif has_BH:
        # If BH, EH, and EN channels are present, select only BH
        sdata += _sdata.select(channel="BH?")
    elif has_EH:
        # If only EH and EN channels are present, select only EH
        # NTS: This may result in getting only vertical component data - EH? is used for PNSN analog stations
        # NTS: This may also be tricky for pulling full day-volumes because the sampling rate shifts for
        #      analog stations due to the remote digitization scheme used with analog stations
        sdata += _sdata.select(channel="EH?")
    else:
        return

    ###############################
    # If no data returned, skipping
    if len(sdata) == 0:
        Logger.warning("No stream returned. Skipping.")
        return
    if np.abs(np.mean(sdata[0].data[1:] - sdata[0].data[0:-1])) <= 1e-8:
        Logger.warning("constant/no data in the stream. Skipping.")
        return
    ###############################
    # NTS: Filter and then resample
    # Filter
    sdata.filter(type='bandpass',freqmin=4,freqmax=15)
    # Resample
    sdata.resample(100)
    
    ###############################
    # NTS: This may produce unintended swathes of filled gaps - advise revising this
    sdata.merge(fill_value='interpolate') # fill gaps if there are any.
    ###############################

    # Get the necassary information about the station
    delta = sdata[0].stats.delta
    starttime = sdata[0].stats.starttime
    fs = sdata[0].stats.sampling_rate
    dt = 1/fs
    

    # Make all the traces in the stream have the same lengths
    # This is risky as it may result in gappy data
    max_starttime = max([tr.stats.starttime for tr in sdata])
    min_endtime = min([tr.stats.endtime for tr in sdata])
    
    for tr in sdata:
        tr.trim(starttime=max_starttime,endtime=min_endtime, nearest_sample=True)    

    # NTS: Make sure traces are in Z[1E][2N] order
    _s2d = Stream()
    _s2x = Stream()
    for _c in ['[3Z]','[1E]','[2N]']:
        _s = sdata.select(channel=f'??{_c}')
        # Prioritize only the first if more than one is present
        for _e, _tr in enumerate(_s):
            if _e == 0:
                _s2d += _tr
            else:
                _s2x += _tr
    # Tack the extra traces back onto the leading 3 ordered traces
    _s2d += _s2x
    # Overwrite sdata with re-ordered traces
    sdata = _s2d

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
    # NTS: Carefully note that batch_pred and _torch_pred are offset by 1 in their
    #      0-axis. The native 0-axis of EQTransformer predictions are the "detection"
    #      predictor.
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
    # python parameters
    del _torch_pred, windows_max_tt, windows_std_tt
    del windows_std, windows_max
    # gARBAGE cOLLECTOR cleaup
    gc.collect()
    # Torch GPU cleanup (if used?)
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

    # NTS: Went scorched-earth past this point - largely empty format
    idf_p = pred_trigger_pick(smb_p, sdata[0],'P', thrd=p_thrd)
    idf_s = pred_trigger_pick(smb_s, sdata[0], 'S', thrd=s_thrd)

    df = pd.concat([idf_p, idf_s], axis=0, ignore_index=True)
    df.to_csv(save_file_name)

    # p_index = picks_summary_simple(smb_p, p_thrd)
    # s_index = picks_summary_simple(smb_s, s_thrd)
    # print(f"{len(p_index)} P picks\n{len(s_index)} S picks")
    
    # print('test2')
    # # Create lists and a data frame
    # event_id = []
    # source_type = []
    # station_network_code = []
    # station_channel_code = []
    # station_code = []
    # station_location_code = []
    # station_latitude_deg= []
    # station_longitude_deg = []
    # station_elevation_m = []
    # trace_name = []
    # trace_sampling_rate_hz = []
    # trace_start_time = []
    # trace_S_arrival_sample = []
    # trace_P_arrival_sample = []
    # trace_S_onset = []
    # trace_P_onset = []
    # trace_snr_db = []
    # trace_p_arrival = []
    # trace_s_arrival = []
    
    # print("This is the cwd:"+str(os.getcwd()))
    # print('test3',len(p_index),len(s_index))
    # for i, idx in enumerate(p_index):
    #     event_id.append(' ')
    #     source_type.append(' ')
    #     station_network_code.append(network)   # Change to otehr networks
    #     station_channel_code.append(' ')
    #     station_code.append(station)
    #     station_location_code.append(sdata[0].stats.location) 
    #     print('test3-1')
    #     station_latitude_deg.append(lat)
    #     print('test3-2')
    #     station_longitude_deg.append(lon) 
    #     print('test3-3')
    #     station_elevation_m.append(elev)
    #     print('test3-4')
    #     trace_name.append(' ')
    #     trace_sampling_rate_hz.append(sdata[0].stats.sampling_rate)
    #     print('test3-5')
    #     trace_start_time.append(sdata[0].stats.starttime)
    #     trace_S_arrival_sample.append(' ')
    #     trace_P_arrival_sample.append(' ')
    #     trace_S_onset.append(' ')
    #     trace_P_onset.append(' ')
    #     trace_snr_db.append(' ')
    #     trace_s_arrival.append(np.nan)
    #     print('test3-6')
    #     trace_p_arrival.append(str(starttime  + idx * delta))

    # print('test4')
    # for i, idx in enumerate(s_index):
    #     event_id.append(' ')
    #     source_type.append(' ')
    #     station_network_code.append(network) # Change to otehr networks
    #     station_channel_code.append(' ')
    #     station_code.append(station)
    #     station_location_code.append(sdata[0].stats.location)   
    #     print('test3-7')
    #     station_latitude_deg.append(lat)
    #     print('test3-8')
    #     station_longitude_deg.append(lon)
    #     print('test3-9')
    #     station_elevation_m.append(elev)
    #     print('test3-10')
    #     trace_name.append(' ')
    #     trace_sampling_rate_hz.append(sdata[0].stats.sampling_rate)
    #     trace_start_time.append(sdata[0].stats.starttime)
    #     trace_S_arrival_sample.append(' ')
    #     trace_P_arrival_sample.append(' ')
    #     trace_S_onset.append(' ')
    #     trace_P_onset.append(' ')
    #     trace_snr_db.append(' ')
    #     trace_s_arrival.append(str(starttime  + idx * delta))
    #     print('test3-11')
    #     trace_p_arrival.append(np.nan)
    # print('test5')
    # # dictionary of lists
    # dict = {'event_id':event_id,'source_type':source_type,'station_network_code':station_network_code,\
    #         'station_channel_code':station_channel_code,'station_code':station_code,'station_location_code':station_location_code,\
    #         'station_latitude_deg':station_latitude_deg,'station_longitude_deg':station_longitude_deg, \
    #         'station_elevation_m':station_elevation_m,'trace_name':trace_name,'trace_sampling_rate_hz':trace_sampling_rate_hz,\
    #         'trace_start_time':trace_start_time,'trace_S_arrival_sample':trace_S_arrival_sample,\
    #         'trace_P_arrival_sample':trace_P_arrival_sample, 'trace_S_onset':trace_S_onset,'trace_P_onset':trace_P_onset,\
    #         'trace_snr_db':trace_snr_db, 'trace_s_arrival':trace_s_arrival, 'trace_p_arrival':trace_p_arrival}

    # df = pd.DataFrame(dict)
    
    # print('test6')
    # # Make the specific day into a string:
    # tstring = t1.strftime('%Y%m%d')
    # # Build the full file name:
    # print("test7")
    # print("This is the cwd:"+str(os.getcwd()))
    # print('This is the filepath:'+str(filepath))
    # file_name = filepath+station+'_'+tstring+'.csv'
    # ##################################################
    # # Write to file using that name
    # print(file_name,'this is before test9')
    # print(df)
    # print(f"P and S summary:\n{len(p_index)} P picks\n{len(s_index)} S picks")
    # df.to_csv(file_name)
    # print('test9')
    
