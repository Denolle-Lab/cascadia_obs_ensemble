import os
import pandas as pd
from obspy import UTCDateTime,Stream
from obspy.clients.fdsn import Client
from pnwstore.mseed import WaveformClient
from obspy.geodetics import locations2degrees, degrees2kilometers
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

def subplot_waveforms(picks,dir,fig_title):
    os.makedirs(dir, exist_ok=True)
    client_iris = Client("IRIS")
    client_pnw = WaveformClient()
    client_ncedc = Client('NCEDC')

    for event_idx in picks.idx.unique():
        event_picks = picks[picks.idx==event_idx]
        event_stations = event_picks.station.unique()
        # print(event_picks)

        otime = UTCDateTime(pd.to_datetime(event_picks.iloc[0].time))
        olat = event_picks.iloc[0].latitude
        olon = event_picks.iloc[0].longitude
        odepth = event_picks.iloc[0].depth

        bulk_sta = []
        for sta in event_stations:
            network = sta.split('.')[0]
            station = sta.split('.')[1]
            loc = '*'
            ch = '?H?' 
            t1 = otime- pd.Timedelta(1,'days')
            t2 = otime + pd.Timedelta(1,'days')

            bulk_sta.append([network,station,loc,ch,t1,t2])
            
        inv = client_iris.get_stations_bulk(bulk_sta)

        # print(inv)
        distances = []
        for network in inv:
            network_code = network.code
            for sta in network:
                station_code = sta.code
                slat = sta.latitude
                slon = sta.longitude
                selev = sta.elevation
                
                dis1 = locations2degrees(olat, olon, slat, slon)
                dist = degrees2kilometers(dis1)

                distances.append([network_code,station_code,olat,olon,odepth,slat,slon,selev,dist])
                
        # Sort distances
        distances = sorted(distances, key=lambda item: item[-1])

        st = Stream()

        starttime = otime - 30
        endtime = otime + 120
        ch = '?H?'
        loc = '*'

        # Set up to define the xlim and ylim
        max_y = 0
        min_y = 0
        min_y_count = 0 

        max_x = 0
        min_x = 0
        min_x_count= 0

        bulk_ncedc = []
        bulk_pnw = []

        for item in distances:
            network_code, station_code, olat, olon, odepth, slat, slon, selev, dist = item

            # Make a bulk request for the waveforms
            if network_code in ['NC','BK']:
                bulk_ncedc.append([network_code,station_code,loc,ch,starttime,endtime])
            else:
                bulk_pnw.append([network_code,station_code,loc,ch,starttime,endtime])

            # Adjust the time window and scaling of the data
            station = network_code+'.'+station_code+'.'
            p_pick = event_picks.loc[(event_picks.station==station)&(event_picks.phase=='P')]
            s_pick = event_picks.loc[(event_picks.station==station)&(event_picks.phase=='S')]

            # Append p_pick and s_pick to distances
            item.extend([p_pick,s_pick])
            if len(p_pick)==0 and len(s_pick)==0:
                print('No picks for this station. Skipping.')
                continue

            if len(s_pick) > 0:
                if max_x < UTCDateTime(s_pick.iloc[0]['time_pick']) - starttime:
                    max_x = UTCDateTime(s_pick.iloc[0]['time_pick']+5) - starttime
            elif len(p_pick) > 0:
                if max_x < UTCDateTime(p_pick.iloc[0]['time_pick']) - starttime: 
                    max_x = UTCDateTime(p_pick.iloc[0]['time_pick']+5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue 

            if len(p_pick) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(p_pick.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_pick.iloc[0]['time_pick']-5) - starttime
                        min_x_count += 1           
                else:
                    if min_x >= UTCDateTime(p_pick.iloc[0]['time_pick']) - starttime:
                        min_x = UTCDateTime(p_pick.iloc[0]['time_pick']-5) - starttime            
            elif len(s_pick) > 0:
                if min_x_count == 0:
                    if min_x < UTCDateTime(s_pick.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_pick.iloc[0]['time_pick']-5)- starttime
                        min_x_count += 1                
                else:
                    if min_x >= UTCDateTime(s_pick.iloc[0]['time_pick'])- starttime:
                        min_x = UTCDateTime(s_pick.iloc[0]['time_pick']-5) - starttime
            else:
                print('No picks for this station. Skipping.')
                continue    

            if min_y_count == 0:
                if min_y < dist:
                    min_y = dist - 5
                    min_y_count += 1           
            else:
                if min_y >= dist:
                    min_y = dist - 5 

            max_y = dist + 5

            distances

        scaling_factor = (1/2) * (max_y - min_y)
            
        # Download the waveforms
        st_ncedc = Stream()
        st_pnw = Stream()
        if len(bulk_ncedc) > 0:
            st_ncedc += client_ncedc.get_waveforms_bulk(bulk_ncedc)
        if len(bulk_pnw) > 0:
            st_pnw += client_pnw.get_waveforms_bulk(bulk_pnw)

        st = st_ncedc + st_pnw   

        st = Stream(filter(lambda st: st.stats.sampling_rate > 10, st))
        st.taper(max_percentage=0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=25)
        st.merge(fill_value='interpolate')

        # Plot the waveforms
        # print('test1',st)
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        # Plot the event and station locations
        m = Basemap(projection='merc', llcrnrlat=38, urcrnrlat=51, llcrnrlon=-132, urcrnrlon=-119, resolution='i', ax=axs[3])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawmapboundary()
        m.drawparallels(np.arange(38, 51, 2), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-132, -119, 2), labels=[0, 0, 0, 1],rotation=45)
        axs[3].set_title('Event Location')

        for i,item in enumerate(distances):
            network_code, station_code, olat, olon, odepth, slat, slon, selev, dist,p_pick,s_pick = item
            st_sta = st.select(network=network_code,station=station_code)

            # Select only HH or BH channels
            _st = Stream()
            has_HH = bool(st_sta.select(channel="HH?"))
            has_BH = bool(st_sta.select(channel="BH?"))

            if has_HH and has_BH:
                _st += st_sta.select(channel="HH?")
            elif has_HH:
                _st += st_sta.select(channel="HH?")
            elif has_BH:
                _st += st_sta.select(channel="BH?")
            
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
            _st = sorted(_st, key=lambda trace: get_channel_priority(trace.stats.channel))

            _st = Stream(_st)
            # print(_st)
            for ax in range(len(_st)):
                tr = _st[ax]
                sampling_rate = tr.stats.sampling_rate
                channel = tr.stats.channel
                
                tr = tr.normalize()
                
                if len(p_pick) > 0:
                    tp = UTCDateTime(p_pick.iloc[0]['time_pick']) - otime + 30
                    i1 = int((tp-5) * sampling_rate)
                    i2 = int((tp+15) * sampling_rate)
                elif len(s_pick) > 0:
                    ts = UTCDateTime(s_pick.iloc[0]['time_pick']) - otime + 30
                    i1 = int((ts-10) * sampling_rate)
                    i2 = int((ts+10) * sampling_rate)
                else:
                    print(f"WARNING: No pick time for {network}.{station}.{channel} on {otime}.")

            

                offsets1 = dist
                # print(offsets1)
                try: 
                    wave = tr.data
                    wave = wave / (np.nanmax(wave[i1:i2], axis=-1) * 10)
                except:
                    continue 

                # Plot the waveform
                axs[ax].plot(tr.times(), wave * scaling_factor + offsets1, 
                                color='black', alpha=0.7, lw=0.5)
                # print(tr.stats.channel)
                # Add the label only once per channel
                # print(i)
                if i == 0:
                    axs[ax].plot([], [], color='black', label=f"{tr.stats.channel[2]}")

                if len(p_pick) > 0:
                    axs[ax].vlines(UTCDateTime(p_pick.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/12) * scaling_factor, 
                                    offsets1 + (1/12) * scaling_factor, color='r')
                if len(s_pick) > 0:
                    axs[ax].vlines(UTCDateTime(s_pick.iloc[0]['time_pick']) - otime + 30, offsets1 - (1/12) * scaling_factor, 
                                    offsets1 + (1/12) * scaling_factor, color='b')
            
            for ax in range(len(_st)):
                for i,ii in enumerate(distances):
                    station = ii[0]+'.'+ii[1]+'.'
                    offsets1 = ii[-3]
                    axs[ax].text(min_x + 0.5, offsets1+1.2, 
                                    station, fontsize=8, color='red', verticalalignment='bottom')
                axs[ax].legend(loc='upper right',handlelength=0, handletextpad=0) 
                axs[ax].set_ylim([min_y-(1/8) * scaling_factor, max_y+(1/8) * scaling_factor])
                axs[ax].set_xlim([min_x, max_x])
                axs[ax].grid(alpha=0.5)
                axs[ax].set_xlabel('Offset from the Origin Time [s]')

            x_sta,y_sta = m(slon,slat)
            m.plot(x_sta, y_sta, '^', color='blue', markersize=4)
        
        # Plot the event location
        x_event,y_event = m(olon,olat)
        m.plot(x_event, y_event, 'yo', markersize=4.5)

        # fig.supxlabel('Time [sec]', y=0.02)
        fig.supylabel('Epicentral Distance [km]',x=0.09)
        fig.suptitle(f"{fig_title}: Origin Time={otime}, \n Latitude={round(olat, 2)}, Longtitude={round(olon, 2)}, Depth={round(odepth, 2)}", y=0.96)

        tstring = otime.strftime('%Y%m%dT%H%M%SZ')
        path = dir + f"{tstring}.png"
        fig.savefig(path,format='png')  