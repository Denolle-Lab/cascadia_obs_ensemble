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
from obspy.geodetics import locations2degrees, degrees2kilometers
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
    pick_sta = np.unique(picks['station'])

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
        
            sta_picks = picks.loc[picks['station']==ii[1]]

            p_picks = sta_picks.loc[sta_picks['phase']=='P']
            s_picks = sta_picks.loc[sta_picks['phase']=='S']

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
    
def plot_waveforms_3_channels(idx,mycatalog,mycatalog_picks,network,channel,idx_sta,title,fig_title,ylim,xlim):
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
    pick_sta = np.unique(picks['station'])

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
    plt.figure(figsize=(6,12))
    #  Plot the waveforms in a loop   
    for i, ii in enumerate(distances):
        st = client.get_waveforms(network="*",
                                  station=ii[1], channel=channel, starttime=otime-30, endtime=otime+120)
        st = obspy.Stream(filter(lambda st:st.stats.sampling_rate>10, st))
        st.filter(type='bandpass',freqmin=4,freqmax=15)

        trim_st = st.copy()
        if len(trim_st)>0:
            trim_st = trim_st.normalize()
            offsets1  = ii[2]
            offsets2 = np.arange(0,len(pick_sta)* (len(trim_st)),len(trim_st))
            offsets3 =2
            if i==0:
                print(offsets2)
            for iii in range(len(trim_st)):
                wave=trim_st[iii].data
                wave=wave/np.nanmax(wave,axis=-1,keepdims=True)
                plt.plot(trim_st[iii].times(),wave *30+offsets1+offsets2[iii], 
                         color = 'black', alpha=0.5, lw=0.5)    
        #         time_pick = [[x['time_pick'], x['phase']] for _, x in mycatalog[mycatalog['idx'] == idx].iterrows() 
        #                      if x['station'] == sta]
        #         if len(time_pick) > 0:
        #             for p in time_pick:
        #                 if p[1] == 'P':
                plt.text(trim_st[iii].times()[0]-5, trim_st[iii].data[0] * 10 + offsets1-2+offsets2[iii],[st[iii].stats.channel], fontsize=8, verticalalignment='bottom')
                if iii== offsets3:
                    plt.text(trim_st[iii].times()[0]-20, trim_st[iii].data[0] * 10 + offsets1-2+offsets2[iii],[ii[1]], fontsize=8, verticalalignment='bottom')

        #         plt.vlines(ii[2]/5, offsets1-5, 
        #                          offsets1+5, color='r')
                sta_picks = picks.loc[picks['station']==ii[1]]

                p_picks = sta_picks.loc[sta_picks['phase']=='P']
                s_picks = sta_picks.loc[sta_picks['phase']=='S']


                if len(p_picks)>0:
                    plt.vlines(UTCDateTime(p_picks.iloc[0]['time_pick'])-otime+30, offsets1-5+offsets2[iii], offsets1+5+offsets2[iii], color='r')

                if len(s_picks)>0:
                    plt.vlines(UTCDateTime(s_picks.iloc[0]['time_pick'])-otime+30, offsets1-5+offsets2[iii],offsets1+5+offsets2[iii], color='b')

        #                 else:
        #                     plt.vlines(p[0], offsets1[ii]*0.5+offsets2[i]-1, 
        #                                      offsets1[ii]*0.5+offsets2[i]+1, color='b')
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
    
def subplots_waveforms(idx,mycatalog,mycatalog_picks,network,channel,idx_sta,title,fig_title,ylim,xlim):
    """
    idx: event_idx
    mycatalog: dataframe that contains only the unique picks (i.e., mycatalog_picks.drop_duplicates(subset=['idx']).copy())
    mycatalog_picks: all pick assignments csv file (e.g., pd.read_csv('../data/datasets_OR/all_pick_assignments_OR.csv'))
    network: string of networks (e.g., "NV,OO,7A")
    channel: specify the direction of the channel (i.e., "?HZ", "?HE" or "?HN")
    idx_sta: choose the station to which you want to show the waveforms
    title: title in a string
    fig_title: figure title in as string
    ylim: ylim range (e.g., [0,400])
    xlim: xlim range (e.g., [20,150])
    """
    
    # Define the clients 
    client = WaveformClient()
    client2 = Client("IRIS")

    # Plot the earthquake moveout for one of the unmatched events for all stations 
    # event = new_events_deg.iloc[idx]
    event=mycatalog
    picks = mycatalog_picks
    pick_sta = np.unique(picks['station'])

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
    fig=plt.figure(figsize=(6,12))
    gs = fig.add_gridspec(3, hspace=0,figure=fig)
    axs = gs.subplots(sharex=True, sharey=True)
    #  Plot the waveforms in a loop   
    for i, ii in enumerate(distances):
        # Obtain the waveforms and filter
        st = client.get_waveforms(network="*",
                                  station=ii[1], channel=channel, starttime=otime-30, endtime=otime+120)
        st = obspy.Stream(filter(lambda st:st.stats.sampling_rate>10, st))
        st.filter(type='bandpass',freqmin=4,freqmax=15)

        trim_st = st.copy()

        sta_picks = picks.loc[picks['station']==ii[1]]
        p_picks = sta_picks.loc[sta_picks['phase']=='P']
        s_picks = sta_picks.loc[sta_picks['phase']=='S']
        
        for iax in range(len(trim_st)):
            if len(trim_st)>0:
                # Normalize the waveform
                trim_st = trim_st.normalize()

                offsets1  = ii[2]
                wave=trim_st[iax].data
                wave=wave/np.nanmax(wave,axis=-1,keepdims=True)
            if i==0:
                axs[iax].plot(trim_st[iax].times(),wave *30+offsets1, 
                         color = 'black', label=f"{trim_st[iax].stats.channel}",alpha=0.7, lw=0.5)
                axs[iax].text(trim_st[iax].times()[-1]+2, trim_st[iax].data[0] * 10 + offsets1-2, 
                         [ii[1]], fontsize=8, verticalalignment='bottom')
                axs[iax].legend(loc='upper right',handlelength=0)
                axs[iax].set_ylim(ylim)
                axs[iax].set_xlim(xlim)
                axs[iax].grid(alpha=0.5)
                
            else:    
                axs[iax].plot(trim_st[iax].times(),wave *30+offsets1, 
                             color = 'black', alpha=0.7, lw=0.5)    
                axs[iax].text(trim_st[iax].times()[-1]+2, trim_st[iax].data[0] * 10 + offsets1-2, 
                             [ii[1]], fontsize=8, verticalalignment='bottom')
                axs[iax].legend(loc='upper right',handlelength=0)
                axs[iax].set_ylim(ylim)
                axs[iax].set_xlim(xlim)
                axs[iax].grid(alpha=0.5)

            if len(p_picks)>0:
                axs[iax].vlines(UTCDateTime(p_picks.iloc[0]['time_pick'])-otime+30, offsets1-(1/3)*30, 
                             offsets1+(1/3)*30, color='r')
            if len(s_picks)>0:
                axs[iax].vlines(UTCDateTime(s_picks.iloc[0]['time_pick'])-otime+30, offsets1-(1/3)*30, 
                             offsets1+(1/3)*30, color='b')
    fig.supxlabel('Time [sec]',y=0.07)
    fig.supylabel('Distance [km]')
#     plt.legend(loc='upper left')
    fig.suptitle(f"{title}: Origin Time={otime}, \n Latitude={round(event['latitude'].iloc[0],2)}, Longtitude={round(event['longitude'].iloc[0],2)}",y=0.92)
    plt.savefig(fig_title)
    plt.show()
    