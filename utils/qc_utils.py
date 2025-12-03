import os
import sys
import pandas as pd
import numpy as np
import obspy
from tqdm import tqdm
from obspy.core.stream import Stream

from obspy.clients.fdsn import Client
from obspy.clients.fdsn import Client as FDSNClient

from mpl_toolkits.basemap import Basemap


from pnwstore.mseed import WaveformClient
import datetime as datetime1
from datetime import datetime

from matplotlib import pyplot as plt
from obspy import UTCDateTime

from tqdm import tqdm
from obspy.geodetics import locations2degrees, degrees2kilometers

def match_events(event, catalog, time_threshold, dist_threshold):
        matched = catalog[(abs(catalog['datetime'] - event['datetime']) <= pd.Timedelta(seconds=time_threshold)) &
                          (degrees2kilometers(locations2degrees(event['latitude'], event['longitude'], catalog['latitude'], catalog['longitude'])) <= dist_threshold)]
        if not matched.empty:
            # print('Matched: ',matched)
            matched_drop_duplicates = matched.drop_duplicates(subset=['idx'], keep='first')

            diffs = abs(matched_drop_duplicates['datetime'] - event['datetime'])
            closest_index_matched_drop_duplicates = diffs.idxmin()
            closest_index = matched_drop_duplicates.loc[closest_index_matched_drop_duplicates]['idx']
            # print('Closest index: ',closest_index)
            # print('Closest: ',matched.loc[matched['idx']==closest_index])
            # Add a new column to indicate the matching event_id
            matched.loc[matched['idx'] == closest_index, 'matched_event_id'] = event['event_id']
            return matched.loc[matched['idx']==closest_index]
        return pd.DataFrame()

def filter_and_match_events(events_morton, events_anss, our_catalog, time_threshold=120, dist_threshold=1000):
    
  # Convert datetime columns to UTCDateTime for comparison
    events_morton['datetime'] = pd.to_datetime(events_morton['datetime'], utc=True)
    events_anss['datetime'] = pd.to_datetime(events_anss['datetime'], utc=True)
    our_catalog['datetime'] = pd.to_datetime(our_catalog['datetime'], utc=True)

    # Define the time and region coverage for Morton's catalog
    t1_morton = events_morton['datetime'].min()
    t2_morton = events_morton['datetime'].max()
    lat_min_morton = events_morton['latitude'].min()
    lat_max_morton = events_morton['latitude'].max()
    lon_min_morton = events_morton['longitude'].min()
    lon_max_morton = events_morton['longitude'].max()

    # Filter our catalog to match Morton's time and region coverage
    our_catalog_morton_time_region = our_catalog[(our_catalog['datetime'] >= t1_morton) & (our_catalog['datetime'] <= t2_morton) &
                                                    (our_catalog['latitude'] >= lat_min_morton) & (our_catalog['latitude'] <= lat_max_morton) &
                                                    (our_catalog['longitude'] >= lon_min_morton) & (our_catalog['longitude'] <= lon_max_morton)]

    # # Filter ANSS catalog to match Morton's time and region coverage
    # events_anss_morton_time_region = events_anss[(events_anss['datetime'] >= t1_morton) & (events_anss['datetime'] <= t2_morton) &
    #                                              (events_anss['latitude'] >= lat_min_morton) & (events_anss['latitude'] <= lat_max_morton) &
    #                                              (events_anss['longitude'] >= lon_min_morton) & (events_anss['longitude'] <= lon_max_morton)]

    # Filter our catalog to match ANSS time and region coverage
    our_catalog_anss_time_region = our_catalog.copy()

    # Initialize lists to store matched and unmatched events
    our_events_morton_region_time_matched_morton = []
    our_events_morton_region_time_matched_anss = []
    our_events_our_region_time_matched_anss = []
    our_events_morton_region_time_unmatched_morton = []
    our_events_our_region_time_unmatched_morton = []
    morton_events_unmatched_our = []
    anss_events_unmatched_our_our_region_time = []
    events_in_all = []

    # Function to match events based on time and distance thresholds
    for _, event in events_morton.iterrows():
        matched = match_events(event, our_catalog_morton_time_region, time_threshold, dist_threshold)
        if not matched.empty:
            our_events_morton_region_time_matched_morton.append(matched)
        else:
            morton_events_unmatched_our.append(event)

    for _, event in events_anss.iterrows():
        matched = match_events(event, our_catalog, time_threshold, dist_threshold)
        if not matched.empty:
            our_events_our_region_time_matched_anss.append(matched)
        else:
            anss_events_unmatched_our_our_region_time.append(event)

    for _, event in events_anss.iterrows():
        matched = match_events(event, our_catalog_morton_time_region, time_threshold, dist_threshold)
        if not matched.empty:
            our_events_morton_region_time_matched_anss.append(matched)
        # else:
        #     anss_events_unmatched_our_morton_region_time.append(event)


    # Compare with the Morton-2023 catalog
    # Find unmatched events in our catalog with region and time coverage of the Morton-2023 catalog 
    our_events_morton_region_time_unmatched_morton = our_catalog_morton_time_region[~our_catalog_morton_time_region['idx'].isin(pd.concat(our_events_morton_region_time_matched_morton)['idx'])]

    # Find unmatched events in our catalog with our region and time coverage 
    our_events_our_region_time_unmatched_morton = our_catalog[~our_catalog['idx'].isin(pd.concat(our_events_morton_region_time_matched_morton)['idx'])]

    our_events_morton_region_time_unmatched_anss = our_catalog_morton_time_region[~our_catalog_morton_time_region['idx'].isin(pd.concat(our_events_morton_region_time_matched_anss)['idx'])]

    # Find unmatched events in our catalog with our region and time coverage 
    our_events_our_region_time_unmatched_anss = our_catalog[~our_catalog['idx'].isin(pd.concat(our_events_our_region_time_matched_anss)['idx'])]




    # Create dataframes
    df_our_events_morton_region_time_matched_morton  = pd.concat(our_events_morton_region_time_matched_morton).reset_index(drop=True)
    df_our_events_morton_region_time_matched_anss = pd.concat(our_events_morton_region_time_matched_anss).reset_index(drop=True)
    df_our_events_our_region_time_matched_anss = pd.concat(our_events_our_region_time_matched_anss).reset_index(drop=True)
    df_our_events_morton_region_time_unmatched_morton = pd.DataFrame(our_events_morton_region_time_unmatched_morton).reset_index(drop=True)
    df_our_events_our_region_time_unmatched_morton = pd.DataFrame(our_events_our_region_time_unmatched_morton).reset_index(drop=True)
    df_morton_events_unmatched_our = pd.DataFrame(morton_events_unmatched_our).reset_index(drop=True)
    df_anss_events_unmatched_our_our_region_time = pd.DataFrame(anss_events_unmatched_our_our_region_time).reset_index(drop=True)
    df_events_all = df_our_events_morton_region_time_matched_morton[df_our_events_morton_region_time_matched_morton['idx'].isin(df_our_events_morton_region_time_matched_anss['idx'])]

    df_new_events_morton_region_time = df_our_events_morton_region_time_unmatched_morton[df_our_events_morton_region_time_unmatched_morton['idx'].isin(our_events_morton_region_time_unmatched_anss['idx'])]

    df_new_events_our_region_time = df_our_events_our_region_time_unmatched_morton[df_our_events_our_region_time_unmatched_morton['idx'].isin(our_events_our_region_time_unmatched_anss['idx'])]



    df_our_events_morton_region_time_matched_anss_not_morton = df_our_events_morton_region_time_matched_anss[~df_our_events_morton_region_time_matched_anss['idx'].isin(df_events_all['idx'])]

    df_new_events_morton_region_time1 = df_our_events_morton_region_time_unmatched_morton[~df_our_events_morton_region_time_unmatched_morton['idx'].isin(df_our_events_morton_region_time_matched_anss_not_morton['idx'])]

    df_our_events_our_region_time_matched_anss_not_morton = df_our_events_our_region_time_matched_anss[~df_our_events_our_region_time_matched_anss['idx'].isin(df_events_all['idx'])]

    df_new_events_our_region_time1 = df_our_events_our_region_time_unmatched_morton[~df_our_events_our_region_time_unmatched_morton['idx'].isin(df_our_events_our_region_time_matched_anss_not_morton['idx'])]



    # Create the required dataframes
    df1 = df_our_events_morton_region_time_matched_morton
    df2 = df_our_events_morton_region_time_matched_anss
    df3 = df_our_events_our_region_time_matched_anss
    df4 = df_our_events_morton_region_time_unmatched_morton
    df5 = df_our_events_our_region_time_unmatched_morton
    df6 = df_morton_events_unmatched_our
    df7 = df_anss_events_unmatched_our_our_region_time
    df8 = df_events_all
    df9 = df_new_events_morton_region_time
    df10 = df_new_events_our_region_time
    df11 = df_new_events_morton_region_time1
    df12 = df_new_events_our_region_time1


    return df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12



def merge_events(event_files):
    """
    Merge events from multiple files into a single DataFrame.

    Inputs:
    event_files: list of strings, paths to event files. Each file should be a CSV file with at least columns 'datetime', 'latitude', 'longitude'.

    Outputs:
    A merged DataFrame with columns at least'datetime', 'latitude', 'longitude'.
    
    """
    
    # Read all event files into DataFrames
    dfs = [pd.read_csv(file, index_col=0) for file in event_files]

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs)
    
    # Sort the DataFrame by 'datetime'
    merged_df['datetime'] = pd.to_datetime(merged_df['time'])
    merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)

    # Define thresholds
    time_threshold = 10
    dist_threshold = 25  # in kilometers

    # Initialize a list to collect indices of rows to be dropped
    rows_to_drop = []

    for i in tqdm(range(len(merged_df)), total=len(merged_df)):
        if i in rows_to_drop:
            continue

        event = merged_df.loc[i]
        t1 = event['datetime']
        olat = event['latitude']
        olon = event['longitude']
        condition = (merged_df['datetime'] >= t1 - pd.Timedelta(seconds=time_threshold)) & \
                    (merged_df['datetime'] <= t1 + pd.Timedelta(seconds=time_threshold)) & \
                    (degrees2kilometers(locations2degrees(olat, olon, merged_df['latitude'], merged_df['longitude'])) <= dist_threshold) & \
                    (merged_df.index != i)   
        rows_to_drop = rows_to_drop + merged_df.loc[condition].index.tolist()

    # Make rows_to_drop unique
    rows_to_drop = list(set(rows_to_drop))

    # Drop the collected rows
    merged_df.drop(rows_to_drop, inplace=True)

    # Reset the index to ensure it is sequential
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def calc_snr(all_picks,all_pick_assignments):
    """ 
    This function calculates the SNRs for each station in each event.
    
    Inputs:
    1. The all_pick.csv file from the 2_format_pick2associate file.
    2. The all_pick_assignments.csv file from the 3_association file.
    
    Oututs:
    The new all_pick_assignments_snr file with the additional SNR column.
    """
    
    # Create the list of networks
    all_picks_networks = all_picks['station_network_code'].drop_duplicates()
    list_networks = list(all_picks_networks)
    networks = ','.join(all_picks_networks)
    
    
    
    # Define the clients 
    client_waveform = WaveformClient()
    client2 = Client("IRIS")
    client_ncedc = Client('NCEDC')
    
    all_pick_assignments['datetime'] = pd.to_datetime(all_pick_assignments['time'], utc = True)
    all_pick_assignments['pick_datetime'] = all_pick_assignments["time_pick"].apply(datetime.utcfromtimestamp)

    
    # Define parameters
    percentile=98
    
    # Create empty lists
    snr_list = []
#     sta_list = []
        
    # Make sure if a station for an event has more than 1 P or S pick
    for idx in tqdm(all_pick_assignments['idx'].drop_duplicates()):
        # Define parameters
        otime = UTCDateTime(str(all_pick_assignments[all_pick_assignments['idx'] == idx]["datetime"].values[0]))


        # Create empty lists
        networks_stas = []

        # Plot the earthquake moveout for one of the unmatched events for all stations 
    #     event = mycatalog
        pick_idx = all_pick_assignments.loc[all_pick_assignments['idx']==idx]
        pick_sta = np.unique(pick_idx['station'])
        
    #     distances = []
    #     max_dist = 10
    #     min_dist = 0
        for station in pick_sta:

            sta_inv = client2.get_stations(network=networks,
                                           station=station, channel="?H?", 
                                           starttime=otime - 1e8, endtime=otime + 1e8,level="response")
            if len(sta_inv) == 0:
                print(f'No inventory for station {station}. Skipping.')
                continue

            network = sta_inv[0].code
    #         slat = sta_inv[0][0].latitude
    #         slon = sta_inv[0][0].longitude
    #         olat = event.loc[event['idx']==idx, 'latitude'].values[0]
    #         olon = event.loc[event['idx']==idx, 'longitude'].values[0]

    #         dis1 = locations2degrees(olat, olon, slat, slon)
    #         dist = degrees2kilometers(dis1)
    # #         if max_dist < dist:
    # #             max_dist = dist

    # #         if min_dist > dist:
    # #             min_dist = dist

            networks_stas.append([network,station])
        
   
        events = all_pick_assignments[all_pick_assignments['idx']==idx]
        for i in networks_stas:
            events1 = events[events['station']==i[1]]
            p_picks = events1[events1['phase']=='P']
            s_picks = events1[events1['phase']=='S']
            

            if len(p_picks)>0 and len(s_picks)>0:
                print(p_picks['pick_datetime'].values,s_picks['pick_datetime'].values)
                p_pick_time = UTCDateTime(str(p_picks['pick_datetime'].values[0]))
                s_pick_time = UTCDateTime(str(s_picks['pick_datetime'].values[0]))
                
                starttime_st = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=120)
                endtime_st = UTCDateTime(p_pick_time)+datetime1.timedelta(seconds=120)
                
                starttime_noise = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=8)
                endtime_noise = UTCDateTime(p_pick_time)
                
                starttime_signal = UTCDateTime(s_pick_time)-datetime1.timedelta(seconds=1)
                endtime_signal = UTCDateTime(s_pick_time)+datetime1.timedelta(seconds=2)
                
                #####################################

                time_trunc1 = UTCDateTime(p_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
                
                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1
                
                print(f"P and S pick times:{p_pick_time} and {s_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")
                print(f'time_trunc1:{time_trunc1}')
                print(f'time_trunc_prev_day:{time_trunc_prev_day}')
                print(f'time_trunc2:{time_trunc2}')
                print(f'time_trunc_current_day:{time_trunc_current_day}')
                print(f'dt_start:{dt_start}')
                print(f'dt_end:{dt_end}')
                

                if dt_start<0 and dt_end>0:
                    print('test1')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                        
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                    
                    st = st1+st2
                    
                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    print('test2')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                        
                    st = st1+st2

                else: 
                    if i[0] in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                #####################################
                
                print(starttime_noise,endtime_noise)
            
            
#                 if i[0] in ['NC', 'BK']:
#                 # Query waveforms
#                     st = client_ncedc.get_waveforms(network=i[0], station=i[1],
#                                                     location="*", channel="?HZ",starttime=starttime_st,
#                                                     endtime=endtime_st)

#                 elif i[0] in networks: 
#                     st = client_waveform.get_waveforms(network=i[0], station=i[1],
#                                                        channel='?HZ',starttime=starttime_st, endtime=endtime_st)

#                 else: 
#                     st =  Stream()
#                     print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
#                     continue

        #         print(f"len(st):{len(st)}")
        #         print(st)

                    
                print(f'First st print:{st}')
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HHZ")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HHZ")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BHZ")
                    
                # Skip empty traces
                if len(_st) == 0:
                    snr = np.nan
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    continue

                _st.merge(fill_value='interpolate') # fill gaps if there are any.

                print(f'Second st print:{_st}')
                
                if _st[0].stats.starttime>starttime_noise or _st[0].stats.endtime<endtime_signal:
                    snr = np.nan 
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    
                    continue
            
                noise = _st.copy().trim(starttime=starttime_noise,endtime=endtime_noise)    
                signal = _st.copy().trim(starttime=starttime_signal,endtime=endtime_signal) 
                
                print(f"P and S pick times:{p_pick_time} and {s_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")
                
                print(f'Noise:{noise}')
                print(f'Signal:{signal}')

                noise_abs = np.percentile(abs(noise[0].data),percentile)
                signal_abs = np.percentile(abs(signal[0].data),percentile)

                snr = 20 * np.log10((signal_abs/noise_abs))

                snr_list.append(snr)
#                 sta_list.append(i[1])
                
                snr_list.append(snr)
#                 sta_list.append(i[1])
                
            if len(p_picks)>0 and len(s_picks)==0:
                
                print(p_picks['pick_datetime'].values,s_picks['pick_datetime'].values)
                p_pick_time = UTCDateTime(str(p_picks['pick_datetime'].values[0]))
#                 s_pick_time = UTCDateTime(str(s_picks['datetime'].values[0]))

                starttime_st = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=120)
                endtime_st = UTCDateTime(p_pick_time)+datetime1.timedelta(seconds=120)
                
                starttime_noise = UTCDateTime(p_pick_time)-datetime1.timedelta(seconds=8)
                endtime_noise = UTCDateTime(p_pick_time)
                
                starttime_signal = UTCDateTime(p_pick_time)+datetime1.timedelta(seconds=1)
                endtime_signal = UTCDateTime(p_pick_time)+datetime1.timedelta(seconds=4)
                
                #####################################

                time_trunc1 = UTCDateTime(p_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
                
                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1
                
                print(f"P pick time:{p_pick_time} and {s_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")
                print(f'time_trunc1:{time_trunc1}')
                print(f'time_trunc_prev_day:{time_trunc_prev_day}')
                print(f'time_trunc2:{time_trunc2}')
                print(f'time_trunc_current_day:{time_trunc_current_day}')
                print(f'dt_start:{dt_start}')
                print(f'dt_end:{dt_end}')
                

                if dt_start<0 and dt_end>0:
                    print('test3')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                        
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                    
                    st = st1+st2
                    
                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    # If start and end times are on different days
                    print('test4')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                        
                    st = st1+st2

                else: 
                    if i[0] in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                #####################################
                
                print(f'First st print:{st}')
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HHZ")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HHZ")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BHZ")
                    
                # Skip empty traces
                if len(_st) == 0:
                    snr = np.nan
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    
                    continue

                _st.merge(fill_value='interpolate') # fill gaps if there are any.

                print(f'Second st print:{_st}')
                
                if _st[0].stats.starttime>starttime_noise or _st[0].stats.endtime<endtime_signal:
                    snr = np.nan 
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    
                    continue 

                noise = _st.copy().trim(starttime=starttime_noise,endtime=endtime_noise)    
                signal = _st.copy().trim(starttime=starttime_signal,endtime=endtime_signal) 
                
                print(f"P pick time:{p_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")

                print(f'Noise:{noise}')
                print(f'Signal:{signal}')
                

                noise_abs = np.percentile(abs(noise[0].data),percentile)
                signal_abs = np.percentile(abs(signal[0].data),percentile)

                snr = 20 * np.log10((signal_abs/noise_abs))

                snr_list.append(snr)
#                 sta_list.append(i[1])
                    
            if len(p_picks)==0 and len(s_picks)>0:
                print(p_picks['pick_datetime'].values,s_picks['pick_datetime'].values)
#                 p_pick_time = UTCDateTime(str(p_picks['datetime'].values))
                s_pick_time = UTCDateTime(str(s_picks['pick_datetime'].values[0]))
    
                starttime_st = UTCDateTime(s_pick_time)-datetime1.timedelta(seconds=120)
                endtime_st = UTCDateTime(s_pick_time)+datetime1.timedelta(seconds=120)
                
                starttime_noise = UTCDateTime(s_pick_time)-datetime1.timedelta(seconds=8)
                endtime_noise = UTCDateTime(s_pick_time)
                
                starttime_signal = UTCDateTime(s_pick_time)
                endtime_signal = UTCDateTime(s_pick_time)+datetime1.timedelta(seconds=3)
                
                print(starttime_noise,endtime_noise)
                
                #####################################

                time_trunc1 = UTCDateTime(s_pick_time.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
                time_trunc2 = UTCDateTime(endtime_st.datetime.replace(hour=0, minute=0, second=0, microsecond=0))                
                time_trunc_prev_day = time_trunc1 - pd.Timedelta(microseconds=1)
                time_trunc_current_day = time_trunc2 - pd.Timedelta(microseconds=1)


#                 dt1 = starttime_st - time_trunc1
#                 dt2 = starttime_st - time_trunc2
                
                dt_start = starttime_st - time_trunc1
                dt_end = endtime_st - time_trunc1
                
                print(f"S pick time: {s_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")
                print(f'time_trunc1:{time_trunc1}')
                print(f'time_trunc_prev_day:{time_trunc_prev_day}')
                print(f'time_trunc2:{time_trunc2}')
                print(f'time_trunc_current_day:{time_trunc_current_day}')
                print(f'dt_start:{dt_start}')
                print(f'dt_end:{dt_end}')
                

                if dt_start<0 and dt_end>0:
                    print('test5')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_prev_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_prev_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                        
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")   
                        continue
                    
                    st = st1+st2
                    
                elif dt_start <= pd.Timedelta(days=1).total_seconds() and dt_end > pd.Timedelta(days=1).total_seconds():
                    print('test6')
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st1 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=time_trunc_current_day)

                    elif i[0] in networks: 
                        st1 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=time_trunc_current_day)

                    else: 
                        st1 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                    if i[0] in ['NC', 'BK']:
                    # Query waveforms
                        st2 = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=time_trunc2,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st2 = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=time_trunc2, endtime=endtime_st)

                    else: 
                        st2 =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                        
                    st = st1+st2

                else: 
                    if i[0] in ['NC', 'BK']:
                # Query waveforms
                        st = client_ncedc.get_waveforms(network=i[0], station=i[1],
                                                        location="*", channel="?H?",starttime=starttime_st,
                                                        endtime=endtime_st)

                    elif i[0] in networks: 
                        st = client_waveform.get_waveforms(network=i[0], station=i[1],
                                                           channel='?H?',starttime=starttime_st, endtime=endtime_st)

                    else: 
                        st =  Stream()
                        print(f"WARNING: No data for {ii[1]}.{ii[2]}.{channel} on {otime}.")    
                        continue
                    
                #####################################
        #         print(f"len(st):{len(st)}")
        #         print(st)

                    
                print(f'First st print:{st}')
                # Create a new stream
                _st = Stream()
                # Check for HH and BH channels presence
                has_HH = bool(st.select(channel="HH?"))
                has_BH = bool(st.select(channel="BH?"))

                # Apply selection logic based on channel presence
                if has_HH and has_BH:
                    # If both HH and BH channels are present, select only HH
                    _st += st.select(channel="HHZ")
                elif has_HH:
                    # If only HH channels are present
                    _st += st.select(channel="HHZ")
                elif has_BH:
                    # If only BH channels are present
                    _st += st.select(channel="BHZ")
                
                 # Skip empty traces
                if len(_st) == 0:
                    snr = np.nan
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    continue

                _st.merge(fill_value='interpolate') # fill gaps if there are any.

                print(f'Second st print:{_st}')

                if _st[0].stats.starttime>starttime_noise or _st[0].stats.endtime<endtime_signal:
                    snr = np.nan 
                    snr_list.append(snr)
#                     sta_list.append(i[1])
                    continue 
                    
                noise = _st.copy().trim(starttime=starttime_noise,endtime=endtime_noise)    
                signal = _st.copy().trim(starttime=starttime_signal,endtime=endtime_signal) 
                
                print(f"S pick time:{s_pick_time}")
                print(f"Stream start and end times:{starttime_st} and {endtime_st}")
                print(f"Noise start and end times:{starttime_noise} and {endtime_noise}")
                print(f"Signal start and end times:{starttime_signal} and {endtime_signal}")
                
                print(f'Noise:{noise}')
                print(f'Signal:{signal}')

                noise_abs = np.percentile(abs(noise[0].data),percentile)
                signal_abs = np.percentile(abs(signal[0].data),percentile)

                snr = 20 * np.log10((signal_abs/noise_abs))

                snr_list.append(snr)
#                 sta_list.append(i[1])
            
    all_pick_assignments['snr']=snr_list
#     all_pick_assignments['sta']=sta_list
     
    return all_pick_assignments
    
def filter_sta(events, all_pick_assignments):
    """
    Inputs:
    1. Either of the following files can be the input:
        1. matched_events_with_morton_mycatalog.csv from the 4_quality_control file
        2. matched_events_with_anss_mycatalog.csv from the 4_quality_control file
        3. new_events.csv from the 4_quality_control file
    2. The all_pick_assignments CSV file from the 3_associate file: e.g., all_pick_assignments = pd.read_csv('../data/datasets_2012/all_pick_assignments_2012.csv')

    
    Outputs:
    1. A new dataframe that only has events that fall into the following categories:
        1. For an event, at least two stations have to be less than 50 km from the event and no station can be 100 km apart from each other.
    2. An event has to have more than or equal to 6 picks
    """
    
    # Parameters
    client2 = Client("IRIS")

    mycatalog = all_pick_assignments.drop_duplicates(subset=['idx'])
    mycatalog['datetime'] = pd.to_datetime(mycatalog['time'], utc = True)

    for i, idx in tqdm(enumerate(events['idx']),total=len(events['idx'])):
        event = mycatalog
        picks = all_pick_assignments
        picks_idx = picks.loc[picks['idx']==idx]
        pick_sta = np.unique(picks_idx['station'])
        otime = UTCDateTime(str(event[event['idx'] == idx]["datetime"].values[0]))
        distances = []
        max_dist = 10
        min_dist = 0
        for station in pick_sta:


            sta_inv = client2.get_stations(network='*',
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

            distances.append([None, _network, station, dist])

        # Sort distances
        distances = sorted(distances, key=lambda item: item[-1])

        # This is for the first criterion in the markdown above
        # Determine if any two of the numbers in the distances list are less than or equal to 50
        found = False
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                if distances[i][3] <= 50 and distances[j][3] <= 50:
                    found = True
                    break
            if found:
                break

        # Make a list that includes the differences between the consecutive numbers in the distances list.
        differences = [distances[i+1][3] - distances[i][3] for i in range(len(distances) - 1)]

        if found == False: # If there were not at least two distances between the station and the event less than or equal to 50 km
            print(distances)
            index = events[events['idx'] == idx].index
            events = events.drop(index=index)

        elif any(differences > 100 for differences in differences): # If any of the distances between two stations were greater than 100 km
            print(distances)
            index = events[events['idx'] == idx].index
            events = events.drop(index=index)

        else: 
            continue


    return events 

def calc_active_days_of_stas(inventory, start_time, end_time):
    """
    auth: Hiroto Bito
    org: Department of Earth and Space Sciences, University of Washington
    email: hbito@uw.edu
    purpose: This function calculates the active days of the stations in an inventory given the start and end times of the time period of interest.

    Inputs:
    inventory: obspy.Inventory object
    start_time: UTCDateTime object
    end_time: UTCDateTime object

    Outputs:
    stations: a DataFrame containing the active days of the stations in the inventory as well as their network codes, station codes, latitudes, and longitudes
    """
    stations = []
    for network in inventory:
        for station in network:
            latitude = station.latitude
            longitude = station.longitude
            network_code = network.code
            station_code = station.code
            
            # Ensure station.start_date and station.end_date are handled
            station_start = max(start_time, station.start_date) if station.start_date else start_time
            station_end = min(end_time, station.end_date) if station.end_date else end_time
            
            # Calculate active days safely (convert seconds to days)
            active_days = int((station_end - station_start) / 86400) if station_end > station_start else 0
            
            stations.append({
                'station': network_code + '.' + station_code,
                "latitude": latitude,
                "longitude": longitude,
                "active_days": active_days,
            })
    return stations

# def pull_station_info(inventory, start_time, end_time):