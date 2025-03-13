import os
from obspy.clients.fdsn import Client
import numpy as np
import obspy
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, datetime
import pandas as pd
import dask
from dask.diagnostics import ProgressBar

from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy.geodetics import locations2degrees, degrees2kilometers
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_pdf import PdfPages


from obspy import Stream

from pnwstore.mseed import WaveformClient
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import gc
import seisbench.models as sbm
from ELEP.elep.ensemble_statistics import ensemble_statistics
from ELEP.elep.ensemble_coherence import ensemble_semblance 
from ELEP.elep.trigger_func import picks_summary_simple

# pip install basemap


def plot_waveforms(idx,mycatalog,mycatalog_picks,network,channel,idx_sta,title,fig_title,ylim,xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    network: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., (0,400))
    xlim: xlim range (e.g., (20,150))
    """
    
    # Define the clients 
    client = WaveformClient()
    client2 = Client("IRIS")

    # Plot the earthquake moveout for one of the unmatched events for all stations 
    # event = new_events_deg.iloc[idx]
    event=mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])

    # otime = UTCDateTime(event['datetime'])
    otime =UTCDateTime(event["datetime"].iloc[idx])
    distances = []

    # Assuming networks_stas is a list of tuples with network and station identifiers
    for station in pick_sta:
        try:
            sta_inv = client2.get_stations(network=network,
                                           station=station, channel="?H?", 
                                           starttime=otime - 1e4, endtime=otime + 1e4)[0][0]
        except Exception as e:
            print(f"Failed to fetch for {network} {station} {otime}: {e}")
            continue

        slat = sta_inv.latitude
        slon = sta_inv.longitude
        olat = event['latitude']
        olon = event['longitude']

        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
        distances.append([None,station,dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[0:idx_sta+1]
    
    # Create a figure
    plt.figure()
    #  Plot the waveforms in a loop   
    for i, ii in enumerate(distances):
        # Obtain the waveforms and filter
        st = client.get_waveforms(network="*",
                                  station=ii[1], channel=channel, starttime=otime-30, endtime=otime+120)
        st = obspy.Stream(filter(lambda st:st.stats.sampling_rate>10, st))
        st.filter(type='bandpass',freqmin=4,freqmax=15)

        trim_st = st.copy()
        
        if len(trim_st)>0:
            # Normalize the waveform
            trim_st = trim_st.normalize()
            
            offsets1  = ii[2]
            
            wave=trim_st[0].data
            wave=wave/np.nanmax(wave,axis=-1,keepdims=True)
            plt.plot(trim_st[0].times(),wave *30+offsets1, 
                     color = 'black', alpha=0.7, lw=0.5)    
    
            plt.text(trim_st[0].times()[0]-5, trim_st[0].data[0] * 10 + offsets1-2, 
                         [ii[1]], fontsize=8, verticalalignment='bottom')
        
            sta_picks = picks_idx[picks_idx['station'] == ii[1]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
       

            if len(p_picks)>0:
                plt.vlines(UTCDateTime(p_picks.iloc[0]['time_pick'])-otime+30, offsets1-(1/3)*30, 
                             offsets1+(1/3)*30, color='r')

            if len(s_picks)>0:
                plt.vlines(UTCDateTime(s_picks.iloc[0]['time_pick'])-otime+30, offsets1-(1/3)*30, 
                             offsets1+(1/3)*30, color='b')
        else:                 
            pass 
        
    plt.title(f"{title}: Origin Time={otime}, \n Latitude={round(event['latitude'].iloc[0],2)}, Longtitude={round(event['longitude'].iloc[0],2)}")
    plt.xlabel('Time [sec]')
    plt.ylabel('Distance [km]')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid(alpha=0.5)

    plt.savefig(fig_title)
    plt.show()
    
def subplots_3_channels(sta_names,mycatalog_picks, networks, channel, num_events,title, file_title):
    """
    sta_names: names of stations (e.g., ['FN14A','FN07A'])
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the type of the channel (i.e., "?HZ", "?HE" or "?HN")
    num_events: choose how many plots to plot for each station (e.g., 5)
    title: title in a string
    fig_title: figure title in a string
    """
        
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')
    
    picks = mycatalog_picks
    picks['datetime'] = pd.to_datetime(picks['time'], utc = True)
    picks['pick_datetime'] = picks["time_pick"].apply(datetime.utcfromtimestamp)
    picks_sta = picks[picks['station'].isin(sta_names)]
    
    ind = picks_sta.drop_duplicates(subset='event_idx')['event_idx']
    print(f'ind[0:num_plots]:{ind[0:num_events]}')
    print(f'len(ind[0:num_plots]:{len(ind[0:num_events])}')

    p = PdfPages(file_title) 
    
    for idx in tqdm(ind[0:num_events],total=len(ind[0:num_events])):
        
        picks_sta_idx = picks_sta[picks_sta['event_idx']==idx]
        
        otime = UTCDateTime(str(picks_sta_idx["datetime"].values[0]))   

        for sta in picks_sta_idx['station'].drop_duplicates():
            
            sta_inv = client2.get_stations(network=networks,
                                   station=sta, channel=channel, 
                                   starttime=otime - 1e8, endtime=otime + 1e8,level="response")
            if len(sta_inv) == 0:
                print(f"Failed to fetch for {networks} {station} {otime}")
                continue

            network = sta_inv[0].code
    
            picks_sta_idx_sta = picks_sta_idx[picks_sta_idx['station']==sta]
            p_picks = picks_sta_idx_sta[picks_sta_idx_sta['phase']=='P']
            s_picks = picks_sta_idx_sta[picks_sta_idx_sta['phase']=='S']

            if len(p_picks)>0 and len(s_picks)>0:
                print(p_picks['pick_datetime'].values,s_picks['pick_datetime'].values)
                p_pick_time = UTCDateTime(str(p_picks['pick_datetime'].values[0]))
                s_pick_time = UTCDateTime(str(s_picks['pick_datetime'].values[0]))
                
                starttime_offset = 20
                endtime_offset = 20

                starttime_st = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=starttime_offset)
                endtime_st = UTCDateTime(s_pick_time)+datetime1.timedelta(seconds=endtime_offset)

                time_trunc1 = UTCDateTime(p_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)

                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1


                if dt_start<0 and dt_end>0:
                    print('test1')
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {network}.{sta}.{channel} on {otime}.")   
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")   
                        continue

                    st = st1+st2

                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    print('test2')
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    st = st1+st2

                else: 
                    if network in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                ###############################


                print(f'First st print:{st}')
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HH?")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HH?")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BH?")

                # Skip empty streams
                if len(_st) == 0:
                    continue

                _st.merge(fill_value='interpolate') # fill gaps if there are any.

            if len(p_picks)>0 and len(s_picks)==0:

                print(p_picks['pick_datetime'].values,s_picks['pick_datetime'].values)
                p_pick_time = UTCDateTime(str(p_picks['pick_datetime'].values[0]))
                starttime_offset = 20
                endtime_offset = 40

                starttime_st = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=starttime_offset)
                endtime_st = UTCDateTime(p_pick_time)+datetime1.timedelta(seconds=endtime_offset)

                time_trunc1 = UTCDateTime(p_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)

                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1

                if dt_start<0 and dt_end>0:
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")   
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")   
                        continue

                    st = st1+st2

                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    st = st1+st2

                else: 
                    if network in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}")    
                        continue
                        
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HH?")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HH?")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BH?")


                _st.merge(fill_value='interpolate') # fill gaps if there are any.

            if len(p_picks)==0 and len(s_picks)>0:
                starttime_offset = 30
                endtime_offset = 30

                s_pick_time = UTCDateTime(str(s_picks['pick_datetime'].values[0]))

                starttime_st = UTCDateTime(s_pick_time)-datetime1.timedelta(seconds=starttime_offset)
                endtime_st = UTCDateTime(s_pick_time)+datetime1.timedelta(seconds=endtime_offset)

                time_trunc1 = UTCDateTime(s_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)

                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1


                if dt_start<0 and dt_end>0:
                    print('test5')
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")   
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")   
                        continue

                    st = st1+st2

                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    print('test6')
                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif network in networks: 
                        st1 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    if network in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st2 = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue

                    st = st1+st2

                else: 
                    ('test7')
                    if network in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=network, station=sta,
                                                        location="*", channel=channel,starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif network in networks: 
                        st = client_waveform.get_waveforms(network=network, station=sta,
                                                           channel=channel,starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {sta}.{channel} on {otime}.")    
                        continue
                        
                print(f'First st print:{st}')
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HH?")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HH?")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BH?")

                _st.merge(fill_value='interpolate') # fill gaps if there are any.

            st = _st
            st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
            st.taper(max_percentage=0.05)
            st.filter(type='bandpass', freqmin=2, freqmax=25)
            st.merge(fill_value='interpolate') # fill gaps if there are any.

            unique_channels = set(tr.stats.channel for tr in st)
            selected_traces = []

            for ch in unique_channels:
                selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
            st = Stream(selected_traces)

            trim_st = st.copy()

            fig, axs = plt.subplots(len(trim_st), 1, figsize=(12, 6), sharex=True)
            plt.subplots_adjust(hspace=0)
            
            for iax in range(len(trim_st)):
                print(iax)
                sampling_rate = trim_st[iax].stats.sampling_rate
                trim_st = trim_st.normalize()

                if len(p_picks)>0:
                    tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime_st
                    i1 = int((tp-5)*sampling_rate)
                    i2 = int((tp+15)*sampling_rate)
                elif len(s_picks)>0:
                    ts = UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime_st
                    i1 = int((ts-10)*sampling_rate)
                    i2 = int((ts+10)*sampling_rate)
                else:
                    print(f"WARNING: No pick time for {sta}.{channel} on {otime}.")

                try: 
                    scaling_factor = 8
                    wave = trim_st[iax].data
                    wave = wave *(scaling_factor)/ (np.nanmax(wave[i1:i2], axis=-1)*10)
                except:
                    continue 
                    
                min_y=min(wave)
                max_y=max(wave)
                min_x=starttime_st-starttime_st 
                max_x=endtime_st-starttime_st
                
                axs[iax].plot(trim_st[iax].times(), wave, 
                              color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
        
                if len(p_picks) > 0:
                    axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime_st, min_y-0.5, 
                                    max_y+0.5, color='r')
                if len(s_picks) > 0:
                    axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime_st, min_y-0.5, 
                                    max_y+0.5, color='b')

                axs[iax].set_ylim([min_y-0.5,max_y+0.5])
                axs[iax].set_xlim([min_x,max_x])
                axs[iax].grid(alpha=0.5)
                fig.supxlabel('Time [sec]', y=0.03)
                fig.supylabel(f'Amplitude*{scaling_factor} []',x=0.06)
                fig.suptitle(f"{title}{sta}: Origin Time={otime}, \n"
                             +f"Latitude={round(picks_sta_idx_sta['latitude'].values[0], 2)}, "
                             +f"Longtitude={round(picks_sta_idx_sta['longitude'].values[0],2)}, "
                             +f"Depth={round(picks_sta_idx_sta['depth'].values[0], 2)}")
                axs[iax].legend(loc='upper right')
                
        fig.savefig(p, format='pdf')  

    p.close() 
    
def subplots_waveforms(idx, mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title, ylim, xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., [0,400])
    xlim: xlim range (e.g., [20,150])
    """
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')


    # Plot the earthquake moveout for one of the unmatched events for all stations 
    event = mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])
    otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
    distances = []

    for station in pick_sta:
        
        sta_inv = client2.get_stations(network=networks,
                                       station=station, channel="?H?", 
                                       starttime=otime - 1e8, endtime=otime + 1e8,level="response")
        if len(sta_inv) == 0:
            print(f"Failed to fetch for {networks} {station} {otime}")
            continue
            
        _network = sta_inv[0].code
        
        slat = sta_inv[0][0].latitude
        slon = sta_inv[0][0].longitude
        olat = event['latitude'].iloc[idx]
        olon = event['longitude'].iloc[idx]
#         print(slat,slon,olat,olon)
        
        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
        distances.append([None, _network, station, dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[:idx_sta+1]
    
    # Index for those stations that have traces
    idx_trace_exists=0

    # Create a figure
    fig = plt.figure(figsize=(6,12))
    gs = fig.add_gridspec(3, hspace=0, figure=fig)
    axs = gs.subplots(sharex=True, sharey=True)

    # Plot the waveforms in a loop   
    for i, ii in enumerate(distances):
#         print(ii[1],ii[2],channel)
#         st = client.get_waveforms(network="*",
#                                   station=ii[2], channel=channel, starttime=otime-30, endtime=otime+120)
        # Get waveforms and filter
        try:
            if ii[1] in ['NC', 'BK']:
                # Query waveforms
                st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=otime-30, endtime=otime+120)
                
            else: 
                st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=otime-30, endtime=otime+120)
                
        except obspy.clients.fdsn.header.FDSNNoDataException:
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
        
        
        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.filter(type='bandpass', freqmin=4, freqmax=15)
#         print(st)
        # Select only one trace per channel
        unique_channels = set(tr.stats.channel for tr in st)
        selected_traces = []
        
        for ch in unique_channels:
            selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
        st = Stream(selected_traces)
        
        # Skip empty traces
        if len(st) == 0:
                continue
                
        # Increase the number of indices if there are actually traces in that specific station   
        idx_trace_exists+=1
            
        trim_st = st.copy()
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
        print(p_picks,s_picks)
        print('This is after print(p_picks,s_picks)')
#         for iax in range(len(trim_st)):
#             trim_st = trim_st.normalize()
#             offsets1 = ii[3]
#             wave = trim_st[iax].data
#             wave = wave / np.nanmax(wave, axis=-1, keepdims=True)
            
#             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, 
#                           color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#             axs[iax].text(xlim[-1] + 2, trim_st[iax].data[0] * 10 + offsets1 - 2, 
#                               [ii[2]], fontsize=8, verticalalignment='bottom')
                
#             if idx_trace_exists == 1:
#                 axs[iax].legend([trim_st[iax].stats.channel],loc='upper right', handlelength=0)
#                 axs[iax].set_ylim(ylim)
#                 axs[iax].set_xlim(xlim)
#                 axs[iax].grid(alpha=0.5)
  
#             if len(p_picks) > 0:
#                 axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 30, 
#                                 offsets1 + (1/3) * 30, color='r')
#             if len(s_picks) > 0:
#                 axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 30, 
#                                 offsets1 + (1/3) * 30, color='b')
    
#     fig.supxlabel('Time [sec]', y=0.07)
#     fig.supylabel('Distance [km]')
#     fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event[event['idx']==1]['latitude'].values[0], 2)}, Longtitude={round(event[event['idx']==1]['longitude'].values[0], 2)}", y=0.92)
#     plt.savefig(fig_title)
#     plt.show()
    
    
def subplots_cluster0(idx, mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title, ylim, xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., [0,400])
    xlim: xlim range (e.g., [20,150])
    """
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')


    # Plot the earthquake moveout for one of the unmatched events for all stations 
    event = mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])
    otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
    distances = []
    max_dist = 10
    min_dist = 0
    for station in pick_sta:
        
        
        sta_inv = client2.get_stations(network=networks,
                                       station=station, channel="?H?", 
                                       starttime=otime - 1e8, endtime=otime + 1e8,level="response")
        if len(sta_inv) == 0:
            print(f"Failed to fetch for {networks} {station} {otime}")
            continue
            
        _network = sta_inv[0].code
        slat = sta_inv[0][0].latitude
        slon = sta_inv[0][0].longitude
        olat = event.loc[event['idx']==idx, 'latitude'].values[0]
        olon = event.loc[event['idx']==idx, 'longitude'].values[0]
        
        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
#         if max_dist < dist:
#             max_dist = dist
            
#         if min_dist > dist:
#             min_dist = dist
            
        distances.append([None, _network, station, dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[:idx_sta+1]
    
    max_y = 0
    min_y = 0
    
    max_x = 0
    min_x = 0
    min_count = 0
    
    # Create a figure
    fig,axs = plt.subplots(1,3,figsize=(14,6))
    gs = fig.add_gridspec(3, hspace=0, figure=fig)
#     axs = gs.subplots(sharex=True, sharey=True)
    starttime = otime -30
    endtime = otime + 120
    # Plot the waveforms in a loop   
    for i, ii in enumerate(distances):

        if ii[1] in ['NC', 'BK']:
            # Query waveforms
                st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
        print(f"len(st):{len(st)}")
        print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
                
        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.taper(max_percentage=0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=25)
        st.merge(fill_value='interpolate') # fill gaps if there are any.

#         print(st)
        # Select only one trace per channel
        unique_channels = set(tr.stats.channel for tr in st)
        selected_traces = []
        
        for ch in unique_channels:
            selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
        st = Stream(selected_traces)
                   
        trim_st = st.copy()
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
        print(len(p_picks),len(s_picks))
        
        # Define the xlim values
        # Define the maximum x value
        if len(s_picks) > 0:
            if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - otime:
                max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - otime
        elif len(p_picks) > 0:
            if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - otime: 
                max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - otime
        else:
            print('No picks for this station. Skippin.')
            continue 
            
        # Define the minimum x value
        if len(p_picks) > 0:
            if min_count == 0:
                if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - otime :
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - otime
                    min_count += 1           
            else:
                if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - otime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - otime            
        elif len(s_picks) > 0:
            if min_count == 0:
                if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- otime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- otime 
                    min_count += 1                
            else:
                if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- otime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - otime
        else:
            print('No picks for this station. Skippin.')
            continue
                  
                
            
        if len(p_picks) == 0:
            continue 
        print('This is after the p_pick continue statement')
        for iax in range(len(trim_st)):
            sampling_rate = trim_st[iax].stats.sampling_rate
            trim_st = trim_st.normalize()
            
            tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
            i1 = int((tp-5)*sampling_rate)
            i2 = int((tp+15)*sampling_rate)

            offsets1 = ii[3]
            try: 
                wave = trim_st[iax].data
                wave = wave / (np.nanmax(wave[i1:i2], axis=-1)*10)
            except:
                continue 
            
#             print(trim_st[iax].stats.sampling_rate)
            axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, 
                          color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)

            axs[iax].text(xlim[-1] + 2, trim_st[iax].data[0] * 10 + offsets1 - 2, 
                              [ii[2]], fontsize=8, verticalalignment='bottom')

            
        
                
            print(xlim)

            if len(p_picks) > 0:
                axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 5, 
                                offsets1 + (1/3) * 5, color='r')
            if len(s_picks) > 0:
                axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 5, 
                                offsets1 + (1/3) * 5, color='b')
        max_dist = ii[3] 
        print(max_dist)
    
    chs = ['2','1','Z']
    for iax in range(3):
        
        axs[iax].legend(chs[iax],loc='upper right', handlelength=0)
        axs[iax].set_ylim([0 ,max_dist+5])
        axs[iax].set_xlim([min_x,max_x])
        axs[iax].grid(alpha=0.5)
    fig.supxlabel('Time [sec]', y=0.07)
    fig.supylabel('Distance [km]')
    fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event[event['idx']==idx]['latitude'].values[0], 2)}, Longtitude={round(event[event['idx']==idx]['longitude'].values[0], 2)}", y=0.92)
    plt.savefig(fig_title)
    plt.show()

def subplots_cluster_no_scale(idx, mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title, ylim, xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., [0,400])
    xlim: xlim range (e.g., [20,150])
    """
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')


    # Plot the earthquake moveout for one of the unmatched events for all stations 
    event = mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])
    otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
    distances = []
    max_dist = 10
    min_dist = 0
    for station in pick_sta:
        
        
        sta_inv = client2.get_stations(network=networks,
                                       station=station, channel="?H?", 
                                       starttime=otime - 1e8, endtime=otime + 1e8,level="response")
        if len(sta_inv) == 0:
            print(f"Failed to fetch for {networks} {station} {otime}")
            continue
            
        _network = sta_inv[0].code
        slat = sta_inv[0][0].latitude
        slon = sta_inv[0][0].longitude
        olat = event.loc[event['idx']==idx, 'latitude'].values[0]
        olon = event.loc[event['idx']==idx, 'longitude'].values[0]
        
        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
#         if max_dist < dist:
#             max_dist = dist
            
#         if min_dist > dist:
#             min_dist = dist
            
        distances.append([None, _network, station, dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[:idx_sta+1]
    
    # Set up to define the xlim and ylim
    max_y = 0
    min_y = 0
    # This count is for the if statements. Only used to ensure that min_y_count 
    #is changed from 0 to either the first positive value of the distance of one of the stations from the event
    min_y_count = 0 
    
    max_x = 0
    min_x = 0
    
    # This count is for the if statements. Only used to ensure that min_x_count 
    #is changed from 0 to either the first positive value of P pick time or the first positive value of S pick time
    min_x_count= 0
    # Create a figure
    fig,axs = plt.subplots(1,3,figsize=(14,6))
    gs = fig.add_gridspec(3, hspace=0, figure=fig)
#     axs = gs.subplots(sharex=True, sharey=True)
    starttime = otime -30
    endtime = otime + 120
    
    # Define texts
    texts = []
    
   
    for i, ii in enumerate(distances):
            
        if ii[1] in ['NC', 'BK']:
            # Query waveforms
                st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
        print(f"len(st):{len(st)}")
        print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
                
        # Define ylim values
        if min_y_count == 0:
            if min_y < ii[3]:
                min_y = ii[3] - 5
                min_y_count += 1           
        else:
            if min_y >= ii[3]:
                min_y = ii[3] - 5       
        
        
        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.taper(max_percentage=0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=25)
        st.merge(fill_value='interpolate') # fill gaps if there are any.

#         print(st)
        # Select only one trace per channel
        unique_channels = set(tr.stats.channel for tr in st)
        selected_traces = []
        
        for ch in unique_channels:
            selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
        st = Stream(selected_traces)
                   
        trim_st = st.copy()
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
        print(len(p_picks),len(s_picks))
        
        # Define the xlim values
        # Define the maximum x value
        if len(s_picks) > 0:
            if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
        elif len(p_picks) > 0:
            if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue 
            
        # Define the minimum x value
        if len(p_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                    min_x_count += 1           
            else:
                if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
        elif len(s_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                    min_x_count += 1                
            else:
                if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue
                  
                
            
        if len(p_picks) == 0:
            continue 
        print('This is after the p_pick continue statement')
        for iax in range(len(trim_st)):
            sampling_rate = trim_st[iax].stats.sampling_rate
            trim_st = trim_st.normalize()
            
            tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
            i1 = int((tp-5)*sampling_rate)
            i2 = int((tp+15)*sampling_rate)

            offsets1 = ii[3]
            try: 
                wave = trim_st[iax].data
                wave = wave / (np.nanmax(wave[i1:i2], axis=-1)*10)
            except:
                continue 
            
#             print(trim_st[iax].stats.sampling_rate)
            axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, 
                          color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)

#             axs[iax].text(xlim[-1] + 2,   offsets1, 
#                               [ii[2]], fontsize=8, verticalalignment='bottom')

            if len(p_picks) > 0:
                axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 5, 
                                offsets1 + (1/3) * 5, color='r')
            if len(s_picks) > 0:
                axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/3) * 5, 
                                offsets1 + (1/3) * 5, color='b')
        texts.append([ii[2],ii[3]])

        max_y = ii[3] + 5
        print(max_dist)
    print(max_y,min_y)
    chs = ['2','1','Z']
    for iax in range(3):
        for i, ii in enumerate(texts):
            offsets1 = ii[1]
            axs[iax].text(max_x + 0.5, offsets1, 
                                  [ii[0]], fontsize=8, verticalalignment='bottom')
        axs[iax].legend(chs[iax],loc='upper right', handlelength=0)
        axs[iax].set_ylim([min_y,max_y])
        axs[iax].set_xlim([min_x,max_x])
        axs[iax].grid(alpha=0.5)
    fig.supxlabel('Time [sec]', y=0.07)
    fig.supylabel('Distance [km]')
    fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event[event['idx']==idx]['latitude'].values[0], 2)}, Longtitude={round(event[event['idx']==idx]['longitude'].values[0], 2)}", y=1)
    plt.savefig(fig_title)
    plt.show()
    
# def subplots_cluster_scale(mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title):
#     """
#     idx: event_idx
#     mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
#     mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
#     networks: string of networks (e.g., "NV,OO,7A")
#     channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
#     idx_sta: choose the station to which you want to show the waveforms
#     title: title in a string
#     fig_title: figure title in as string
#     """
        
#     # Define the clients 
#     client_waveform = WaveformClient()
#     client2 = Client("IRIS")
#     client_ncedc = Client('NCEDC')


#     # Plot the earthquake moveout for one of the unmatched events for all stations 
#     events = mycatalog
#     picks = mycatalog_picks
#     events['datetime'] = pd.to_datetime(events['datetime'], utc = True)
#     p = PdfPages(fig_title) 
    
#     for idx in tqdm(events['idx'],total=len(events['idx'])):
        
#         picks_idx = picks.loc[picks['idx']==idx]
#         pick_sta = np.unique(picks_idx['station'])

#         otime = UTCDateTime(str(events[events['idx'] == idx]["datetime"].values[0]))
#         distances = []
#         max_dist = 10
#         min_dist = 0

#         print(events[events['idx'] == idx]['picks'].values[0])
#         for station in pick_sta:


#             sta_inv = client2.get_stations(network=networks,
#                                            station=station, channel="?H?", 
#                                            starttime=otime - 1e8, endtime=otime + 1e8,level="response")
#             if len(sta_inv) == 0:
#     #             print(f"Failed to fetch for {networks} {station} {otime}")
#                 continue

#             _network = sta_inv[0].code
#             slat = sta_inv[0][0].latitude
#             slon = sta_inv[0][0].longitude
#             olat = events.loc[events['idx']==idx, 'latitude'].values[0]
#             olon = events.loc[events['idx']==idx, 'longitude'].values[0]

#             dis1 = locations2degrees(olat, olon, slat, slon)
#             dist = degrees2kilometers(dis1)
#     #         if max_dist < dist:
#     #             max_dist = dist

#     #         if min_dist > dist:
#     #             min_dist = dist

#             distances.append([None, _network, station, dist])

#         # Sort distances
#         distances = sorted(distances, key=lambda item: item[-1])
#         distances = distances[:idx_sta+1]

#         # Set up to define the xlim and ylim
#         max_y = 0
#         min_y = 0
#         # This count is for the if statements. Only used to ensure that min_y_count 
#         #is changed from 0 to either the first positive value of the distance of one of the stations from the event
#         min_y_count = 0 

#         max_x = 0
#         min_x = 0

#         # This count is for the if statements. Only used to ensure that min_x_count 
#         #is changed from 0 to either the first positive value of P pick time or the first positive value of S pick time
#         min_x_count= 0
#         # Create a figure
#         fig,axs = plt.subplots(1,4,figsize=(18,6))
#         gs = fig.add_gridspec(3, hspace=0, figure=fig)
#     #     axs = gs.subplots(sharex=True, sharey=True)
#         starttime = otime -30
#         endtime = otime + 120

#         # Define texts
#         texts = []

#         for i, ii in enumerate(distances):

#             if ii[1] in ['NC', 'BK']:
#                 # Query waveforms
#                 st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

#             elif ii[1] in networks: 
#                 st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)

#             else: 
#                 st =  Stream()
#                 print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
#                 continue

#     #         print(f"len(st):{len(st)}")
#     #         print(st)

#             # Skip empty traces
#             if len(st) == 0:
#                     continue

#             sta_picks = picks_idx[picks_idx['station'] == ii[2]]
#             p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
#             s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
#     #         print(len(p_picks),len(s_picks))

#             # Define the xlim values
#             # Define the maximum x value
#             if len(s_picks) > 0:
#                 if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
#                     max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
#             elif len(p_picks) > 0:
#                 if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
#                     max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
#             else:
#                 print('No picks for this station. Skipping.')
#                 continue 

#             # Define the minimum x value
#             if len(p_picks) > 0:
#                 if min_x_count == 0:
#                     if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
#                         min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
#                         min_x_count += 1           
#                 else:
#                     if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
#                         min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
#             elif len(s_picks) > 0:
#                 if min_x_count == 0:
#                     if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
#                         min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
#                         min_x_count += 1                
#                 else:
#                     if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
#                         min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
#             else:
#                 print('No picks for this station. Skipping.')
#                 continue    

#     #         if len(p_picks) == 0:
#     #             continue

#     #         print('This is after the p_pick continue statement')

#             # Define ylim values
#             if min_y_count == 0:
#                 if min_y < ii[3]:
#                     min_y = ii[3] - 5
#                     min_y_count += 1           
#             else:
#                 if min_y >= ii[3]:
#                     min_y = ii[3] - 5 

#             max_y = ii[3] + 5

#         scaling_factor = (1/2)*(max_y-min_y)   

#         for i, ii in enumerate(distances):

#             if ii[1] in ['NC', 'BK']:
#                 # Query waveforms
#                 st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

#             elif ii[1] in networks: 
#                 st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)

#             else: 
#                 st =  Stream()
#                 print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
#                 continue

#     #         print(f"len(st):{len(st)}")
#     #         print(st)

#             # Skip empty traces
#             if len(st) == 0:
#                     continue
#             _st = Stream()
#             # Check for HH and BH channels presence
#             has_HH = bool(st.select(channel="HH?"))
#             has_BH = bool(st.select(channel="BH?"))

#             # Apply selection logic based on channel presence
#             if has_HH and has_BH:
#                 # If both HH and BH channels are present, select only HH
#                 _st += st.select(channel="HH?")
#             elif has_HH:
#                 # If only HH channels are present
#                 _st += st.select(channel="HH?")
#             elif has_BH:
#                 # If only BH channels are present
#                 _st += st.select(channel="BH?")

#             st = _st

#             print(f'Second st print:{_st}')

#             st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
#             st.taper(max_percentage=0.05)
#             st.filter(type='bandpass', freqmin=2, freqmax=25)
#             st.merge(fill_value='interpolate') # fill gaps if there are any.

#     #         print(st)
#             # Select only one trace per channel
#             unique_channels = set(tr.stats.channel for tr in st)
#             selected_traces = []

#             for ch in unique_channels:
#                 selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
#             st = Stream(selected_traces)

#             trim_st = st.copy()
#             sta_picks = picks_idx[picks_idx['station'] == ii[2]]
#             p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
#             s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
#     #         print(len(p_picks),len(s_picks))




#     #         if len(p_picks) == 0:
#     #             continue 
#     #         print('This is after the p_pick continue statement')
#             print(trim_st)

#             for iax in range(len(trim_st)):
#                 print(iax)
#                 sampling_rate = trim_st[iax].stats.sampling_rate
#                 trim_st = trim_st.normalize()

#                 if len(p_picks)>0:
#                     tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
#                     i1 = int((tp-5)*sampling_rate)
#                     i2 = int((tp+15)*sampling_rate)
#                 elif len(s_picks)>0:
#                     ts = UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30
#                     i1 = int((ts-10)*sampling_rate)
#                     i2 = int((ts+10)*sampling_rate)
#                 else:
#                     print(f"WARNING: No pick time for {ii[1]}.{ii[2]}.{channel} on {otime}.")

#                 offsets1 = ii[3]
#                 try: 
#                     wave = trim_st[iax].data
#                     wave = wave / (np.nanmax(wave[i1:i2], axis=-1)*10)
#                 except:
#                     continue 

#     #             print(trim_st[iax].stats.sampling_rate)
#                 axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
#                               color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#     #             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)

#     #             axs[iax].text(xlim[-1] + 2,   offsets1, 
#     #                               [ii[2]], fontsize=8, verticalalignment='bottom')

#                 if len(p_picks) > 0:
#                     axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
#                                     offsets1 + (1/35) * scaling_factor, color='r')
#                 if len(s_picks) > 0:
#                     axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
#                                     offsets1 + (1/35) * scaling_factor, color='b')
#             texts.append([ii[2],ii[3]])


#     #     print(max_y,min_y)
#         chs = ['2','1','Z']
#         for iax in range(3):
#             for i, ii in enumerate(texts):
#                 offsets1 = ii[1]
#                 axs[iax].text(max_x + 0.5, offsets1, 
#                                       [ii[0]], fontsize=8, verticalalignment='bottom')
#             axs[iax].legend(chs[iax],loc='upper right', handlelength=0)
#             axs[iax].set_ylim([min_y,max_y])
#             axs[iax].set_xlim([min_x,max_x])
#             axs[iax].grid(alpha=0.5)
#         fig.supxlabel('Time [sec]', y=0.07)
#         fig.supylabel('Distance [km]')
#         fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(events[events['idx']==idx]['latitude'].values[0], 2)}, Longtitude={round(events[events['idx']==idx]['longitude'].values[0], 2)}, Depth={round(events[events['idx']==idx]['depth'].values[0], 2)}", y=1)

#         m = Basemap(projection='merc', llcrnrlat=40, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-120, resolution='i', ax=axs[3])
#         m.drawcoastlines()
#         m.drawcountries()
#         m.drawstates()
#         m.drawmapboundary()
#         m.drawparallels(np.arange(40, 51, 1), labels=[1,0,0,0])
#         m.drawmeridians(np.arange(-130, -119, 1), labels=[0,0,0,1])
#         x, y = m(events[events['idx']==idx]['longitude'].values[0], events[events['idx']==idx]['latitude'].values[0])
#         m.plot(x, y, 'ro', markersize=9)
#         axs[3].set_title('Event Location')
        
#         fig.savefig(p, format='pdf')  

#     p.close() 
def subplots_cluster_scale_p(idx, mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    """
    
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')


    # Plot the earthquake moveout for one of the unmatched events for all stations 
    event = mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])
    
    otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
    distances = []
    max_dist = 10
    min_dist = 0
    
    print(event[event['idx'] == idx]['picks'].values[0])
    for station in pick_sta:
        
        
        sta_inv = client2.get_stations(network=networks,
                                       station=station, channel="?H?", 
                                       starttime=otime - 1e8, endtime=otime + 1e8,level="response")
        if len(sta_inv) == 0:
#             print(f"Failed to fetch for {networks} {station} {otime}")
            continue
            
        _network = sta_inv[0].code
        slat = sta_inv[0][0].latitude
        slon = sta_inv[0][0].longitude
        olat = event.loc[event['idx']==idx, 'latitude'].values[0]
        olon = event.loc[event['idx']==idx, 'longitude'].values[0]
        
        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
#         if max_dist < dist:
#             max_dist = dist
            
#         if min_dist > dist:
#             min_dist = dist
            
        distances.append([None, _network, station, dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[:idx_sta+1]
    
    # Set up to define the xlim and ylim
    max_y = 0
    min_y = 0
    # This count is for the if statements. Only used to ensure that min_y_count 
    #is changed from 0 to either the first positive value of the distance of one of the stations from the event
    min_y_count = 0 
    
    max_x = 0
    min_x = 0
    
    # This count is for the if statements. Only used to ensure that min_x_count 
    #is changed from 0 to either the first positive value of P pick time or the first positive value of S pick time
    min_x_count= 0
    # Create a figure
    fig,axs = plt.subplots(1,4,figsize=(18,6))
    gs = fig.add_gridspec(3, hspace=0, figure=fig)
#     axs = gs.subplots(sharex=True, sharey=True)
    starttime = otime -30
    endtime = otime + 120
    
    # Define texts
    texts = []
    
    for i, ii in enumerate(distances):
            
        if ii[1] in ['NC', 'BK']:
            # Query waveforms
            st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
#         print(f"len(st):{len(st)}")
#         print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
                
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
#         print(len(p_picks),len(s_picks))
        
        # Define the xlim values
        # Define the maximum x value
        if len(s_picks) > 0:
            if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
        elif len(p_picks) > 0:
            if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue 
            
        # Define the minimum x value
        if len(p_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                    min_x_count += 1           
            else:
                if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
        elif len(s_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                    min_x_count += 1                
            else:
                if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue    
            
        if len(p_picks) == 0:
            continue
            
#         print('This is after the p_pick continue statement')
    
        # Define ylim values
        if min_y_count == 0:
            if min_y < ii[3]:
                min_y = ii[3] - 5
                min_y_count += 1           
        else:
            if min_y >= ii[3]:
                min_y = ii[3] - 5 
                
        max_y = ii[3] + 5
        
    scaling_factor = (1/2)*(max_y-min_y)   
        
    for i, ii in enumerate(distances):
            
        if ii[1] in ['NC', 'BK']:
            # Query waveforms
            st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
#         print(f"len(st):{len(st)}")
#         print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
              
        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.taper(max_percentage=0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=25)
        st.merge(fill_value='interpolate') # fill gaps if there are any.

#         print(st)
        # Select only one trace per channel
        unique_channels = set(tr.stats.channel for tr in st)
        selected_traces = []
        
        for ch in unique_channels:
            selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
        st = Stream(selected_traces)
                   
        trim_st = st.copy()
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
#         print(len(p_picks),len(s_picks))
        
        
                
            
        if len(p_picks) == 0:
            continue 
#         print('This is after the p_pick continue statement')
        for iax in range(len(trim_st)):
            sampling_rate = trim_st[iax].stats.sampling_rate
            trim_st = trim_st.normalize()
            
            tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
            i1 = int((tp-5)*sampling_rate)
            i2 = int((tp+15)*sampling_rate)

            offsets1 = ii[3]
            try: 
                wave = trim_st[iax].data
                wave = wave / (np.nanmax(wave[i1:i2], axis=-1)*10)
            except:
                continue 
            
#             print(trim_st[iax].stats.sampling_rate)
            axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
                          color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)

#             axs[iax].text(xlim[-1] + 2,   offsets1, 
#                               [ii[2]], fontsize=8, verticalalignment='bottom')

            if len(p_picks) > 0:
                axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                offsets1 + (1/35) * scaling_factor, color='r')
            if len(s_picks) > 0:
                axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                offsets1 + (1/35) * scaling_factor, color='b')
        texts.append([ii[2],ii[3]])

    
#     print(max_y,min_y)
    chs = ['2','1','Z']
    for iax in range(3):
        for i, ii in enumerate(texts):
            offsets1 = ii[1]
            axs[iax].text(max_x + 0.5, offsets1, 
                                  [ii[0]], fontsize=8, verticalalignment='bottom')
        axs[iax].legend(chs[iax],loc='upper right', handlelength=0)
        axs[iax].set_ylim([min_y,max_y])
        axs[iax].set_xlim([min_x,max_x])
        axs[iax].grid(alpha=0.5)
    fig.supxlabel('Time [sec]', y=0.07)
    fig.supylabel('Distance [km]')
    fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event[event['idx']==idx]['latitude'].values[0], 2)}, Longtitude={round(event[event['idx']==idx]['longitude'].values[0], 2)}, Depth={round(event[event['idx']==idx]['depth'].values[0], 2)}", y=1)
    
    m = Basemap(projection='merc', llcrnrlat=40, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-120, resolution='i', ax=axs[3])
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary()
    m.drawparallels(np.arange(40, 51, 1), labels=[1,0,0,0])
    m.drawmeridians(np.arange(-130, -119, 1), labels=[0,0,0,1])
    x, y = m(event[event['idx']==idx]['longitude'].values[0], event[event['idx']==idx]['latitude'].values[0])
    m.plot(x, y, 'ro', markersize=9)
    axs[3].set_title('Event Location')
    plt.show()

# haven't started below
def subplots_cluster_scale_s(idx, mycatalog, mycatalog_picks, networks, channel, idx_sta, title, fig_title, ylim, xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., [0,400])
    xlim: xlim range (e.g., [20,150])
    """
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')


    # Plot the earthquake moveout for one of the unmatched events for all stations 
    event = mycatalog
    picks = mycatalog_picks
    picks_idx = picks.loc[picks['idx']==idx]
    pick_sta = np.unique(picks_idx['station'])
    otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
    distances = []
    max_dist = 10
    min_dist = 0
    for station in pick_sta:
        
        
        sta_inv = client2.get_stations(network=networks,
                                       station=station, channel="?H?", 
                                       starttime=otime - 1e8, endtime=otime + 1e8,level="response")
        if len(sta_inv) == 0:
            print(f"Failed to fetch for {networks} {station} {otime}")
            continue
            
        _network = sta_inv[0].code
        slat = sta_inv[0][0].latitude
        slon = sta_inv[0][0].longitude
        olat = event.loc[event['idx']==idx, 'latitude'].values[0]
        olon = event.loc[event['idx']==idx, 'longitude'].values[0]
        
        dis1 = locations2degrees(olat, olon, slat, slon)
        dist = degrees2kilometers(dis1)
#         if max_dist < dist:
#             max_dist = dist
            
#         if min_dist > dist:
#             min_dist = dist
            
        distances.append([None, _network, station, dist])

    # Sort distances
    distances = sorted(distances, key=lambda item: item[-1])
    distances = distances[:idx_sta+1]
    
    # Set up to define the xlim and ylim
    max_y = 0
    min_y = 0
    # This count is for the if statements. Only used to ensure that min_y_count 
    #is changed from 0 to either the first positive value of the distance of one of the stations from the event
    min_y_count = 0 
    
    max_x = 0
    min_x = 0
    
    # This count is for the if statements. Only used to ensure that min_x_count 
    #is changed from 0 to either the first positive value of P pick time or the first positive value of S pick time
    min_x_count= 0
    # Create a figure
    fig,axs = plt.subplots(1,3,figsize=(14,6))
    gs = fig.add_gridspec(3, hspace=0, figure=fig)
#     axs = gs.subplots(sharex=True, sharey=True)
    starttime = otime -30
    endtime = otime + 120
    
    # Define texts
    texts = []
    
    for i, ii in enumerate(distances):
            
        if ii[1] in ['NC', 'BK']:
            # Query waveforms
            st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
        print(f"len(st):{len(st)}")
        print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
                
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
        print(len(p_picks),len(s_picks))
        
        # Define the xlim values
        # Define the maximum x value
        if len(s_picks) > 0:
            if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
        elif len(p_picks) > 0:
            if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue 
            
        # Define the minimum x value
        if len(p_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                    min_x_count += 1           
            else:
                if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                    min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
        elif len(s_picks) > 0:
            if min_x_count == 0:
                if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                    min_x_count += 1                
            else:
                if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                    min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
        else:
            print('No picks for this station. Skipping.')
            continue    
            
        if len(p_picks) == 0:
            continue
            
        print('This is after the p_pick continue statement')
    
        # Define ylim values
        if min_y_count == 0:
            if min_y < ii[3]:
                min_y = ii[3] - 5
                min_y_count += 1           
        else:
            if min_y >= ii[3]:
                min_y = ii[3] - 5 
                
        max_y = ii[3] + 5
        
    scaling_factor = (1/2)*(max_y-min_y)   
        
    for i, ii in enumerate(distances):
            
        if ii[1] in ['NC', 'BK']:
            # Query waveforms
            st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel,starttime=starttime, endtime=endtime)

        elif ii[1] in networks: 
            st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel,starttime=starttime, endtime=endtime)
  
        else: 
            st =  Stream()
            print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
            continue
            
        print(f"len(st):{len(st)}")
        print(st)
    
        # Skip empty traces
        if len(st) == 0:
                continue
              
        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.taper(max_percentage=0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=25)
        st.merge(fill_value='interpolate') # fill gaps if there are any.

#         print(st)
        # Select only one trace per channel
        unique_channels = set(tr.stats.channel for tr in st)
        selected_traces = []
        
        for ch in unique_channels:
            selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
        st = Stream(selected_traces)
                   
        trim_st = st.copy()
        sta_picks = picks_idx[picks_idx['station'] == ii[2]]
        p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
        s_picks = sta_picks.loc[sta_picks['phase'] == 'S']
        print(len(p_picks),len(s_picks))
        
        
                
            
        if len(p_picks) == 0:
            continue 
        print('This is after the p_pick continue statement')
        for iax in range(len(trim_st)):
            sampling_rate = trim_st[iax].stats.sampling_rate
            trim_st = trim_st.normalize()
            
            tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
            i1 = int((tp-5)*sampling_rate)
            i2 = int((tp+15)*sampling_rate)

            offsets1 = ii[3]
            try: 
                wave = trim_st[iax].data
                wave = wave / (np.nanmax(wave[i1:i2], axis=-1)*10)
            except:
                continue 
            
#             print(trim_st[iax].stats.sampling_rate)
            axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
                          color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)
#             axs[iax].plot(trim_st[iax].times(), wave * 30 + offsets1, color='black', label=f"{trim_st[iax].stats.channel}", alpha=0.7, lw=0.5)

#             axs[iax].text(xlim[-1] + 2,   offsets1, 
#                               [ii[2]], fontsize=8, verticalalignment='bottom')

            if len(p_picks) > 0:
                axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                offsets1 + (1/35) * scaling_factor, color='r')
            if len(s_picks) > 0:
                axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                offsets1 + (1/35) * scaling_factor, color='b')
        texts.append([ii[2],ii[3]])

    
    print(max_y,min_y)
    chs = ['2','1','Z']
    for iax in range(3):
        for i, ii in enumerate(texts):
            offsets1 = ii[1]
            axs[iax].text(max_x + 0.5, offsets1, 
                                  [ii[0]], fontsize=8, verticalalignment='bottom')
        axs[iax].legend(chs[iax],loc='upper right', handlelength=0)
        axs[iax].set_ylim([min_y,max_y])
        axs[iax].set_xlim([min_x,max_x])
        axs[iax].grid(alpha=0.5)
    fig.supxlabel('Time [sec]', y=0.07)
    fig.supylabel('Distance [km]')
    fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event[event['idx']==idx]['latitude'].values[0], 2)}, Longtitude={round(event[event['idx']==idx]['longitude'].values[0], 2)}", y=1)
    plt.savefig(fig_title)
    plt.show()

def subplots_cluster_scale_rand(mycatalog, mycatalog_picks, networks, channel, fig_title, path):
    """
    mycatalog: dataframe that contains only the unique events (e.g., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: csv file listing at least the networks of stations that picked the events in mycatalog (e.g., pd.read_csv('../data/networks/networks.csv))
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    fig_title: title in a string
    path: folder path in a string to which the figures will be saved
    """
    os.makedirs(path,exist_ok=True)
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')

    # Plot the earthquake moveout for one of the unmatched events for all stations 
    events = mycatalog
    picks = mycatalog_picks
    events['datetime'] = pd.to_datetime(events['time'], utc=True)
    picks['datetime'] = pd.to_datetime(picks['time'], utc=True)
    networks = ','.join(list(networks['networks']))

    random_events = events.sample(n=5)

    for idx, time in tqdm(zip(random_events['idx'],random_events['time']), total=len(random_events['time'])):
        
        condition = (random_events['idx'] == idx) & (random_events['datetime'] == time)
        picks_idx = picks.loc[picks['time'] == time]

        pick_sta = np.unique(picks_idx['station'])
        otime = UTCDateTime(str(random_events[condition]["datetime"].values[0]))
        distances = []
        

        for station in pick_sta:
            sta_inv = client2.get_stations(network=networks,
                                           station=station, channel="?H?", 
                                           starttime=otime - 1e8, endtime=otime + 1e8, level="response")
            if len(sta_inv) == 0:
                continue
            
            _network = sta_inv[0].code
            slat = sta_inv[0][0].latitude
            slon = sta_inv[0][0].longitude
            olat = random_events.loc[condition, 'latitude'].values[0]
            olon = random_events.loc[condition, 'longitude'].values[0]
            dis1 = locations2degrees(olat, olon, slat, slon)
            dist = degrees2kilometers(dis1)

            distances.append([None, _network, station, dist])

        # Sort distances
        distances = sorted(distances, key=lambda item: item[-1])
        
        # Set up to define the xlim and ylim
        max_y = 0
        min_y = 0
        min_y_count = 0 

        max_x = 0
        min_x = 0
        min_x_count= 0

        # Create a figure
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        gs = fig.add_gridspec(3, hspace=0, figure=fig)
        starttime = otime - 30
        endtime = otime + 120
        # Define texts
        texts = []
        # print('starttime:', starttime)
        # print('endtime:', endtime)
        print(distances)
        for i, ii in enumerate(distances):
            # print('Network:', ii[1])
            # print('Station:', ii[2])
            
            ####################
            time_trunc1 = UTCDateTime(starttime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
            time_trunc2 = UTCDateTime(endtime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
            # time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
            time_trunc_prev_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
            
            # dt_start = starttime_st - time_trunc1
            dt_end = endtime - time_trunc1
                
            if dt_end > pd.Timedelta(days=1).total_seconds():
                # If start and end times are on different days
                print('test4')
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st1 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=time_trunc_prev_day)

                elif ii[1] in networks: 
                    st1 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=time_trunc_prev_day)

                else: 
                    st1 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st2 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=time_trunc2,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st2 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=time_trunc2, endtime=endtime)

                else: 
                    st2 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                    
                st = st1+st2

            else: 
                print('i:', i)
                if ii[1] in ['NC', 'BK']:
            # Query waveforms[
                    st = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=endtime)

                else: 
                    st =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
            #####################
            
          

            if len(st) == 0:
                continue
            sta_picks = picks_idx[picks_idx['station'] == ii[2]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            if len(s_picks) > 0:
                if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                    max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
            elif len(p_picks) > 0:
                if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                    max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue 

            if len(p_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                        min_x_count += 1           
                else:
                    if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
            elif len(s_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                        min_x_count += 1                
                else:
                    if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue    

            if min_y_count == 0:
                if min_y < ii[3]:
                    min_y = ii[3] - 5
                    min_y_count += 1           
            else:
                if min_y >= ii[3]:
                    min_y = ii[3] - 5 

            max_y = ii[3] + 5

        scaling_factor = (1/2) * (max_y - min_y)

        chs = []  # Initialize chs here
        for i, ii in enumerate(distances):

            time_trunc1 = UTCDateTime(starttime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
            time_trunc2 = UTCDateTime(endtime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
            # time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
            time_trunc_prev_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
            
            # dt_start = starttime_st - time_trunc1
            dt_end = endtime - time_trunc1
                
            if dt_end > pd.Timedelta(days=1).total_seconds():
                # If start and end times are on different days
                print('test4')
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st1 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=time_trunc_prev_day)

                elif ii[1] in networks: 
                    st1 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=time_trunc_prev_day)

                else: 
                    st1 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st2 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=time_trunc2,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st2 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=time_trunc2, endtime=endtime)

                else: 
                    st2 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                    
                st = st1+st2

            else: 
                if ii[1] in ['NC', 'BK']:
            # Query waveforms
                    st = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=endtime)

                else: 
                    st =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue

            if len(st) == 0:
                continue

            _st = Stream()
            has_HH = bool(st.select(channel="HH?"))
            has_BH = bool(st.select(channel="BH?"))

            if has_HH and has_BH:
                _st += st.select(channel="HH?")
            elif has_HH:
                _st += st.select(channel="HH?")
            elif has_BH:
                _st += st.select(channel="BH?")

            st = _st

            st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
            st.taper(max_percentage=0.05)
            st.filter(type='bandpass', freqmin=2, freqmax=25)
            st.merge(fill_value='interpolate')

            unique_channels = set(tr.stats.channel for tr in st)
            selected_traces = []

            for ch in unique_channels:
                selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
            st = Stream(selected_traces)

            trim_st = st.copy()
            sta_picks = picks_idx[picks_idx['station'] == ii[2]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            # Define the desired order of channels
            desired_order = {
                'Z': ['HHZ', 'BHZ'],
                'N': ['HHN', 'HH1', 'BHN', 'BH1'],
                'E': ['HHE', 'HH2', 'BHE', 'BH2']
            }

            # Function to map channels to their desired order
            def get_channel_priority(channel):
                for priority, (key, values) in enumerate(desired_order.items()):
                    if channel in values:
                        return priority
                return float('inf')  # Return a high value for channels not in the desired order

            # Sort the traces in trim_st based on the desired order
            trim_st = sorted(trim_st, key=lambda trace: get_channel_priority(trace.stats.channel))
            
            trim_st = Stream(trim_st)
            
            # plt.figure()
            # trim_st.plot()
            # plt.show()
            for iax in range(len(trim_st)):
                sampling_rate = trim_st[iax].stats.sampling_rate
                trim_st = trim_st.normalize()
                if i == 0:
                    chs.append(str(trim_st[iax].stats.channel))

                if len(p_picks) > 0:
                    tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((tp-5) * sampling_rate)
                    i2 = int((tp+15) * sampling_rate)
                elif len(s_picks) > 0:
                    ts = UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((ts-10) * sampling_rate)
                    i2 = int((ts+10) * sampling_rate)
                else:
                    print(f"WARNING: No pick time for {ii[1]}.{ii[2]}.{channel} on {otime}.")

            

                offsets1 = ii[3]
                try: 
                    wave = trim_st[iax].data
                    wave = wave / (np.nanmax(wave[i1:i2], axis=-1) * 10)
                except:
                    continue 

                # Plot the waveform
                axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
                              color='black', alpha=0.7, lw=0.5)

                # Add the label only once per channel
                if i == 0:
                    axs[iax].plot([], [], color='black', label=f"{trim_st[iax].stats.channel}")

                if len(p_picks) > 0:
                    axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='r')
                if len(s_picks) > 0:
                    axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='b')

            texts.append([ii[2], ii[3]])

        for iax in range(3):
            for i, ii in enumerate(texts):
                offsets1 = ii[1]
                axs[iax].text(max_x + 0.5, offsets1, 
                              [ii[0]], fontsize=8, verticalalignment='bottom')
            if chs:  # Only set ncol if chs is not empty
                axs[iax].legend(loc='upper right', ncol=len(chs), handlelength=0,handletextpad=0, columnspacing=0.5)  # Adjust handletextpad and columnspacing
            axs[iax].set_ylim([min_y, max_y])
            axs[iax].set_xlim([min_x, max_x])
            axs[iax].grid(alpha=0.5)

        fig.supxlabel('Time [sec]', y=0.07)
        fig.supylabel('Distance [km]')
        fig.suptitle(f"{fig_title}: Origin Time={otime}, \n Latitude={round(random_events[condition]['latitude'].values[0], 2)}, Longtitude={round(random_events[condition]['longitude'].values[0], 2)}, Depth={round(random_events[condition]['depth'].values[0], 2)}", y=1)

        m = Basemap(projection='merc', llcrnrlat=38, urcrnrlat=51, llcrnrlon=-132, urcrnrlon=-119, resolution='i', ax=axs[3])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary()
        m.drawparallels(np.arange(38, 51, 1), labels=[0, 1, 0, 0])  # Labels on the right side
        m.drawmeridians(np.arange(-132, -119, 1), labels=[0, 0, 0, 1])

        # Rotate the tick labels for the x axis by 45 degrees
        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
        
        x, y = m(random_events[condition]['longitude'].values[0], random_events[condition]['latitude'].values[0])
        m.plot(x, y, 'ro', markersize=9)
        axs[3].set_title('Event Location')


        dt = datetime.fromisoformat(time)
        compact_time = dt.strftime("%Y%m%dT%H%M%SZ") 
        filepath = path + compact_time + '.png'

        fig.savefig(filepath, format='png')

        # fig.savefig(p, format='pdf')  

    # p.close() 

def subplots_cluster_scale_backup(mycatalog, mycatalog_picks, networks, channel, fig_title, path):
    """
    mycatalog: dataframe that contains only the unique events (e.g., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: csv file listing at least the networks of stations that picked the events in mycatalog (e.g., pd.read_csv('../data/networks/networks.csv))
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    fig_title: title in a string
    path: folder path in a string to which the figures will be saved
    """
    os.makedirs(path,exist_ok=True)
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')

    # Plot the earthquake moveout for one of the unmatched events for all stations 
    events = mycatalog
    picks = mycatalog_picks
    events['datetime'] = pd.to_datetime(events['time'], utc=True)
    picks['datetime'] = pd.to_datetime(picks['time'], utc=True)
    networks = ','.join(list(networks['networks']))


    for idx, time in tqdm(zip(events['idx'],events['time']), total=len(events['time'])):
        
        condition = (events['idx'] == idx) & (events['datetime'] == time)
        picks_idx = picks.loc[picks['time'] == time]

        pick_sta = np.unique(picks_idx['station'])
        otime = UTCDateTime(str(events[condition]["datetime"].values[0]))
        distances = []
        

        for station in pick_sta:
            sta_inv = client2.get_stations(network=networks,
                                           station=station, channel="?H?", 
                                           starttime=otime - 1e8, endtime=otime + 1e8, level="response")
            if len(sta_inv) == 0:
                continue
            
            _network = sta_inv[0].code
            slat = sta_inv[0][0].latitude
            slon = sta_inv[0][0].longitude
            olat = events.loc[condition, 'latitude'].values[0]
            olon = events.loc[condition, 'longitude'].values[0]
            dis1 = locations2degrees(olat, olon, slat, slon)
            dist = degrees2kilometers(dis1)

            distances.append([None, _network, station, dist])

        # Sort distances
        distances = sorted(distances, key=lambda item: item[-1])
        
        # Set up to define the xlim and ylim
        max_y = 0
        min_y = 0
        min_y_count = 0 

        max_x = 0
        min_x = 0
        min_x_count= 0

        # Create a figure
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        gs = fig.add_gridspec(3, hspace=0, figure=fig)
        starttime = otime - 30
        endtime = otime + 120
        # Define texts
        texts = []
        # print('starttime:', starttime)
        # print('endtime:', endtime)
        print(distances)
        for i, ii in enumerate(distances):
            # print('Network:', ii[1])
            # print('Station:', ii[2])
            
            ####################
            time_trunc1 = UTCDateTime(starttime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
            time_trunc2 = UTCDateTime(endtime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
            # time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
            time_trunc_prev_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
            
            # dt_start = starttime_st - time_trunc1
            dt_end = endtime - time_trunc1
                
            if dt_end > pd.Timedelta(days=1).total_seconds():
                # If start and end times are on different days
                print('test4')
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st1 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=time_trunc_prev_day)

                elif ii[1] in networks: 
                    st1 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=time_trunc_prev_day)

                else: 
                    st1 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st2 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=time_trunc2,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st2 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=time_trunc2, endtime=endtime)

                else: 
                    st2 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                    
                st = st1+st2

            else: 
                print('i:', i)
                if ii[1] in ['NC', 'BK']:
            # Query waveforms[
                    st = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=endtime)

                else: 
                    st =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
            #####################
            
          

            if len(st) == 0:
                continue
            sta_picks = picks_idx[picks_idx['station'] == ii[2]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            if len(s_picks) > 0:
                if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                    max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
            elif len(p_picks) > 0:
                if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                    max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue 

            if len(p_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                        min_x_count += 1           
                else:
                    if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
            elif len(s_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                        min_x_count += 1                
                else:
                    if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue    

            if min_y_count == 0:
                if min_y < ii[3]:
                    min_y = ii[3] - 5
                    min_y_count += 1           
            else:
                if min_y >= ii[3]:
                    min_y = ii[3] - 5 

            max_y = ii[3] + 5

        scaling_factor = (1/2) * (max_y - min_y)

        chs = []  # Initialize chs here
        for i, ii in enumerate(distances):

            time_trunc1 = UTCDateTime(starttime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
            time_trunc2 = UTCDateTime(endtime.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
            # time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
            time_trunc_prev_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
            
            # dt_start = starttime_st - time_trunc1
            dt_end = endtime - time_trunc1
                
            if dt_end > pd.Timedelta(days=1).total_seconds():
                # If start and end times are on different days
                print('test4')
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st1 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=time_trunc_prev_day)

                elif ii[1] in networks: 
                    st1 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=time_trunc_prev_day)

                else: 
                    st1 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                
                if ii[1] in ['NC', 'BK']:
                # Query waveforms
                    st2 = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=time_trunc2,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st2 = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=time_trunc2, endtime=endtime)

                else: 
                    st2 =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue
                    
                st = st1+st2

            else: 
                if ii[1] in ['NC', 'BK']:
            # Query waveforms
                    st = client_ncedc.get_waveforms(network=ii[1], station=ii[2],
                                                    location="*", channel="?H?",starttime=starttime,
                                                    endtime=endtime)

                elif ii[1] in networks: 
                    st = client_waveform.get_waveforms(network=ii[1], station=ii[2],
                                                        channel='?H?',starttime=starttime, endtime=endtime)

                else: 
                    st =  Stream()
                    print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                    continue

            if len(st) == 0:
                continue

            _st = Stream()
            has_HH = bool(st.select(channel="HH?"))
            has_BH = bool(st.select(channel="BH?"))

            if has_HH and has_BH:
                _st += st.select(channel="HH?")
            elif has_HH:
                _st += st.select(channel="HH?")
            elif has_BH:
                _st += st.select(channel="BH?")

            st = _st

            st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
            st.taper(max_percentage=0.05)
            st.filter(type='bandpass', freqmin=2, freqmax=25)
            st.merge(fill_value='interpolate')

            unique_channels = set(tr.stats.channel for tr in st)
            selected_traces = []

            for ch in unique_channels:
                selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
            st = Stream(selected_traces)

            trim_st = st.copy()
            sta_picks = picks_idx[picks_idx['station'] == ii[2]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            # Define the desired order of channels
            desired_order = {
                'Z': ['HHZ', 'BHZ'],
                'N': ['HHN', 'HH1', 'BHN', 'BH1'],
                'E': ['HHE', 'HH2', 'BHE', 'BH2']
            }

            # Function to map channels to their desired order
            def get_channel_priority(channel):
                for priority, (key, values) in enumerate(desired_order.items()):
                    if channel in values:
                        return priority
                return float('inf')  # Return a high value for channels not in the desired order

            # Sort the traces in trim_st based on the desired order
            trim_st = sorted(trim_st, key=lambda trace: get_channel_priority(trace.stats.channel))
            
            trim_st = Stream(trim_st)
            
            # plt.figure()
            # trim_st.plot()
            # plt.show()
            for iax in range(len(trim_st)):
                sampling_rate = trim_st[iax].stats.sampling_rate
                trim_st = trim_st.normalize()
                if i == 0:
                    chs.append(str(trim_st[iax].stats.channel))

                if len(p_picks) > 0:
                    tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((tp-5) * sampling_rate)
                    i2 = int((tp+15) * sampling_rate)
                elif len(s_picks) > 0:
                    ts = UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((ts-10) * sampling_rate)
                    i2 = int((ts+10) * sampling_rate)
                else:
                    print(f"WARNING: No pick time for {ii[1]}.{ii[2]}.{channel} on {otime}.")

            

                offsets1 = ii[3]
                try: 
                    wave = trim_st[iax].data
                    wave = wave / (np.nanmax(wave[i1:i2], axis=-1) * 10)
                except:
                    continue 

                # Plot the waveform
                axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
                              color='black', alpha=0.7, lw=0.5)

                # Add the label only once per channel
                if i == 0:
                    axs[iax].plot([], [], color='black', label=f"{trim_st[iax].stats.channel}")

                if len(p_picks) > 0:
                    axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='r')
                if len(s_picks) > 0:
                    axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='b')

            texts.append([ii[2], ii[3]])

        for iax in range(3):
            for i, ii in enumerate(texts):
                offsets1 = ii[1]
                axs[iax].text(max_x + 0.5, offsets1, 
                              [ii[0]], fontsize=8, verticalalignment='bottom')
            if chs:  # Only set ncol if chs is not empty
                axs[iax].legend(loc='upper right', ncol=len(chs), handlelength=0,handletextpad=0, columnspacing=0.5)  # Adjust handletextpad and columnspacing
            axs[iax].set_ylim([min_y, max_y])
            axs[iax].set_xlim([min_x, max_x])
            axs[iax].grid(alpha=0.5)

        fig.supxlabel('Time [sec]', y=0.07)
        fig.supylabel('Distance [km]')
        fig.suptitle(f"{fig_title}: Origin Time={otime}, \n Latitude={round(events[condition]['latitude'].values[0], 2)}, Longtitude={round(events[condition]['longitude'].values[0], 2)}, Depth={round(events[condition]['depth'].values[0], 2)}", y=1)

        m = Basemap(projection='merc', llcrnrlat=38, urcrnrlat=51, llcrnrlon=-132, urcrnrlon=-119, resolution='i', ax=axs[3])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary()
        m.drawparallels(np.arange(38, 51, 1), labels=[0, 1, 0, 0])  # Labels on the right side
        m.drawmeridians(np.arange(-132, -119, 1), labels=[0, 0, 0, 1])

        # Rotate the tick labels for the x-axis by 45 degrees
        plt.setp(axs[3].xaxis.get_majorticklabels(), rotation=45, ha="right")

        x, y = m(events[condition]['longitude'].values[0], events[condition]['latitude'].values[0])
        m.plot(x, y, 'ro', markersize=9)
        axs[3].set_title('Event Location')


        dt = datetime.fromisoformat(time)
        compact_time = dt.strftime("%Y%m%dT%H%M%SZ") 
        filepath = path + compact_time + '.png'

        fig.savefig(filepath, format='png')

        # fig.savefig(p, format='pdf')  


def subplots_cluster_scale(mycatalog, mycatalog_picks, networks, channel, fig_title, path):
    """
    mycatalog: dataframe that contains only the unique events (e.g., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    networks: csv file listing at least the networks of stations that picked the events in mycatalog (e.g., pd.read_csv('../data/networks/networks.csv))
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    fig_title: title in a string
    filepath: file path in a string
    """
    os.makedirs(path,exist_ok=True)

    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')

    # Plot the earthquake moveout for one of the unmatched events for all stations 
    events = mycatalog
    picks = mycatalog_picks
    events['datetime'] = pd.to_datetime(events['time'], utc=True)
    picks['datetime'] = pd.to_datetime(picks['time'], utc=True)
    networks = ','.join(list(networks['networks']))
    # Randomly select 5 events
    random_events = events
    # random_events = events.sample(n=3,random_state=44)

    for idx, time in tqdm(zip(random_events['idx'],random_events['time']), total=len(random_events['time'])):
        
        condition = (random_events['idx'] == idx) & (random_events['datetime'] == time)
        picks_idx = picks.loc[picks['time'] == time]

        pick_sta = np.unique(picks_idx['station'])
        otime = UTCDateTime(str(random_events[condition]["datetime"].values[0]))
        distances = []
        

        for sta in pick_sta:
            station = sta.split('.')[1]
            sta_inv = client2.get_stations(network=networks,
                                           station=station, channel="?H?", 
                                           starttime=otime - 1e8, endtime=otime + 1e8, level="response")
            if len(sta_inv) == 0:
                continue
            
            _network = sta_inv[0].code
            slat = sta_inv[0][0].latitude
            slon = sta_inv[0][0].longitude
            olat = random_events.loc[condition, 'latitude'].values[0]
            olon = random_events.loc[condition, 'longitude'].values[0]
            dis1 = locations2degrees(olat, olon, slat, slon)
            dist = degrees2kilometers(dis1)

            distances.append([None, _network, station, dist,sta])

        # Sort distances
        distances = sorted(distances, key=lambda item: item[-1])
        print(distances)
        
        # Set up to define the xlim and ylim
        max_y = 0
        min_y = 0
        min_y_count = 0 

        max_x = 0
        min_x = 0
        min_x_count= 0
        print('test1')
        # Create a figure
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        gs = fig.add_gridspec(3, hspace=0, figure=fig)
        starttime = otime - 30
        endtime = otime + 120
        # Define texts
        texts = []
        # print('starttime:', starttime)
        # print('endtime:', endtime)
        print('test2')
        for i, ii in enumerate(distances):
            print('Network:', ii[1])
            print('Station:', ii[2])
            print('test3')
            if ii[1] in ['NC', 'BK']:
                st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel, starttime=starttime, endtime=endtime)
            elif ii[1] in networks: 
                st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel, starttime=starttime, endtime=endtime)
            else: 
                st = Stream()
                print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                continue

            if len(st) == 0:
                continue
                
            print('test',st)
            sta_picks = picks_idx[picks_idx['station'] == ii[4]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            if len(s_picks) > 0:
                if max_x < UTCDateTime(s_picks.iloc[0]['time_pick']) - starttime:
                    max_x = UTCDateTime(s_picks.iloc[0]['time_pick']+5) - starttime
            elif len(p_picks) > 0:
                if max_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime: 
                    max_x = UTCDateTime(p_picks.iloc[0]['time_pick']+5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue 

            if len(p_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime
                        min_x_count += 1           
                else:
                    if min_x >= UTCDateTime(p_picks.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_picks.iloc[0]['time_pick']-5) - starttime            
            elif len(s_picks) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5)- starttime
                        min_x_count += 1                
                else:
                    if min_x >= UTCDateTime(s_picks.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_picks.iloc[0]['time_pick']-5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue    

            if min_y_count == 0:
                if min_y < ii[3]:
                    min_y = ii[3] - 5
                    min_y_count += 1           
            else:
                if min_y >= ii[3]:
                    min_y = ii[3] - 5 

            max_y = ii[3] + 5

        scaling_factor = (1/2) * (max_y - min_y)

        chs = []  # Initialize chs here
        for i, ii in enumerate(distances):

            if ii[1] in ['NC', 'BK']:
                st = client_ncedc.get_waveforms(network=ii[1], station=ii[2], location="*", channel=channel, starttime=starttime, endtime=endtime)
            elif ii[1] in networks: 
                st = client_waveform.get_waveforms(network=ii[1], station=ii[2], channel=channel, starttime=starttime, endtime=endtime)
            else: 
                st = Stream()
                print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                continue

            if len(st) == 0:
                continue

            _st = Stream()
            has_HH = bool(st.select(channel="HH?"))
            has_BH = bool(st.select(channel="BH?"))

            if has_HH and has_BH:
                _st += st.select(channel="HH?")
            elif has_HH:
                _st += st.select(channel="HH?")
            elif has_BH:
                _st += st.select(channel="BH?")

            st = _st

            st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
            st.taper(max_percentage=0.05)
            st.filter(type='bandpass', freqmin=2, freqmax=25)
            st.merge(fill_value='interpolate')

            unique_channels = set(tr.stats.channel for tr in st)
            selected_traces = []

            for ch in unique_channels:
                selected_traces.append(next(tr for tr in st if tr.stats.channel == ch))
            st = Stream(selected_traces)

            trim_st = st.copy()
            sta_picks = picks_idx[picks_idx['station'] == ii[4]]
            p_picks = sta_picks.loc[sta_picks['phase'] == 'P']
            s_picks = sta_picks.loc[sta_picks['phase'] == 'S']

            # Define the desired order of channels
            desired_order = {
                'Z': ['HHZ', 'BHZ'],
                'N': ['HHN', 'HH1', 'BHN', 'BH1'],
                'E': ['HHE', 'HH2', 'BHE', 'BH2']
            }

            # Function to map channels to their desired order
            def get_channel_priority(channel):
                for priority, (key, values) in enumerate(desired_order.items()):
                    if channel in values:
                        return priority
                return float('inf')  # Return a high value for channels not in the desired order

            # Sort the traces in trim_st based on the desired order
            trim_st = sorted(trim_st, key=lambda trace: get_channel_priority(trace.stats.channel))
            
            trim_st = Stream(trim_st)
            
            # plt.figure()
            # trim_st.plot()
            # plt.show()
            for iax in range(len(trim_st)):
                sampling_rate = trim_st[iax].stats.sampling_rate
                trim_st = trim_st.normalize()
                if i == 0:
                    chs.append(str(trim_st[iax].stats.channel))

                if len(p_picks) > 0:
                    tp = UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((tp-5) * sampling_rate)
                    i2 = int((tp+15) * sampling_rate)
                elif len(s_picks) > 0:
                    ts = UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30
                    i1 = int((ts-10) * sampling_rate)
                    i2 = int((ts+10) * sampling_rate)
                else:
                    print(f"WARNING: No pick time for {ii[1]}.{ii[2]}.{channel} on {otime}.")

            

                offsets1 = ii[3]
                try: 
                    wave = trim_st[iax].data
                    wave = wave / (np.nanmax(wave[i1:i2], axis=-1) * 10)
                except:
                    continue 

                # Plot the waveform
                axs[iax].plot(trim_st[iax].times(), wave * scaling_factor + offsets1, 
                              color='black', alpha=0.7, lw=0.5)

                # Add the label only once per channel
                if i == 0:
                    axs[iax].plot([], [], color='black', label=f"{trim_st[iax].stats.channel}")

                if len(p_picks) > 0:
                    axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='r')
                if len(s_picks) > 0:
                    axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/35) * scaling_factor, 
                                    offsets1 + (1/35) * scaling_factor, color='b')

            texts.append([ii[2], ii[3]])

        for iax in range(3):
            for i, ii in enumerate(texts):
                offsets1 = ii[1]
                axs[iax].text(max_x + 0.5, offsets1, 
                              [ii[0]], fontsize=8, verticalalignment='bottom')
            if chs:  # Only set ncol if chs is not empty
                axs[iax].legend(loc='upper right', ncol=len(chs), handlelength=0,handletextpad=0, columnspacing=0.5)  # Adjust handletextpad and columnspacing
            axs[iax].set_ylim([min_y, max_y])
            axs[iax].set_xlim([min_x, max_x])
            axs[iax].grid(alpha=0.5)

        fig.supxlabel('Time [sec]', y=0.04)
        fig.supylabel('Distance [km]',x=0.09)
        fig.suptitle(f"{fig_title}: Origin Time={otime}, \n Latitude={round(random_events[condition]['latitude'].values[0], 2)}, Longtitude={round(random_events[condition]['longitude'].values[0], 2)}, Depth={round(random_events[condition]['depth'].values[0], 2)}", y=1)

        m = Basemap(projection='merc', llcrnrlat=38, urcrnrlat=51, llcrnrlon=-132, urcrnrlon=-119, resolution='i', ax=axs[3])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary()
        m.drawparallels(np.arange(38, 51, 2), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-132, -119, 2), labels=[0, 0, 0, 1],rotation=45)
        

        x, y = m(random_events[condition]['longitude'].values[0], random_events[condition]['latitude'].values[0])
        m.plot(x, y, 'ro', markersize=9)
        axs[3].set_title('Event Location')

        dt = datetime.fromisoformat(time)
        compact_time = dt.strftime("%Y%m%dT%H%M%SZ") 
        filepath = path + compact_time + '.png'

        fig.savefig(filepath, format='png')
    # p.close()