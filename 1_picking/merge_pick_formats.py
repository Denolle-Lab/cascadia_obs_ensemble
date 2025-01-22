"""
program: cascadia_obs_ensemble/1_picking/merge_pick_formats.py
auth: Nathan T. Stevens
org: Pacific Northwest Seismic Network
email: ntsteven@uw.edu
license: MIT
purpose: This program provides a command-line interface tool to merge
    pick file formats originally provided by Congcong Yuan (Cornell U.)
    in the ELEP GitHub repository and the updated pick file format provided
    by N. Stevens in Dec. 2024.

    This program includes some logging-related classes/methods originally developed
    for the `PULSE` project that were ported over on 14. JAN 2025. (https://github.com/pnsn/PULSE)
"""


import argparse, glob, os, logging, sys

import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

class CriticalExitHandler(logging.Handler):
    """A custom :class:`~logging.Handler` sub-class that emits a sys.exit
    if a logging instance emits a logging.CRITICAL level message

    Constructed through a prompt to ChatGPT and independently
    tested by the author.

    Originally from the `PULSE` project
    """
    def __init__(self, exit_code=1, **kwargs):
        super().__init__(**kwargs)
        self.exit_code = exit_code
        
    def emit(self, record):
        if record.levelno == logging.CRITICAL:
            sys.exit(self.exit_code)

def rich_error_message(e):
    """Given the raw output of an "except Exception as e"
    formatted clause, return a string that emulates
    the error message print-out from python

    e.g., 
    'TypeError: input "input" is not type int'

    :param e: Error object
    :type e: _type_
    :return: error message
    :rtype: str

    Oritinally from the `PULSE` project
    """    
    etype = type(e).__name__
    emsg = str(e)
    return f'{etype}: {emsg}'

### TEST SET FILES ###
def get_test_filenames():
    fnames = glob.glob('../data/nate_tmp/*.csv')
    return fnames

### PARSING METHODS ###
def fmtfromname(file):
    path, name = os.path.split(file)
    name, ext = os.path.splitext(name)
    if ext != '.csv':
        raise SyntaxError('Input file must have a `.csv` extension')
    parts = name.split('_')
    if len(parts) == 4:
        fmt = 'stevens'
    elif len(parts) == 2:
        fmt = 'yuan'
    else:
        raise SyntaxError(f'File name {file} does not look like Stevens or Yuan formatted file.')
    return fmt


def format_location(locs):
    """helper routine to format pandas.read_csv read-in location codes

    :param locs: iterable collection of locations
    :type locs: array-like
    :return: list of formatted location code strings
    :rtype: list of str
    """    
    formatted_locs = []
    for _loc in locs:
        if np.isfinite(_loc):
            formatted_locs.append(f'{int(_loc):02d}')
        else:
            formatted_locs.append('')
    return formatted_locs

def read_stevens_pick_file(file):
    # Format check
    if fmtfromname(file) != 'stevens':
        raise SyntaxError(f'file {file} does not look like Stevens format')

    # Read the CSV to dataframe
    df = pd.read_csv(file, index_col=[0], parse_dates=['trace_starttime','trigger_onset', 'pick_time','trigger_offset'])
    # Run custom formatting script on location codes
    df.location = format_location(df.location)
    # Return
    return df

def read_yuan_pick_file(file):
    # Format check
    if fmtfromname(file) != 'yuan':
        raise SyntaxError(f'file {file} does not look like Yuan format')

    # Read from CSV and parse dates
    df = pd.read_csv(file, index_col=[0], parse_dates=['trace_start_time','trace_p_arrival','trace_s_arrival'])
    # Run custom formatting script on location codes
    df.station_location_code = format_location(df.station_location_code)
    # Return
    return df

### FORMAT CONVERSION METHODS ###
def yuan2stevens(df):
    """
    Convert a dataframe in Yuan format into a Stevens format

    :param df: Yuan formatted pick dataframe parsed by :meth:`~.read_yuan_pick_file`
    :type df: pandas.DataFrame

    :returns: **df_out** (*pandas.DataFrame*) -- Stevens pick formatted dataframe
    """
    stevens_columns=['network', 'station', 'location', 'band_inst', 'label',
                    'trace_starttime', 'trigger_onset', 'pick_time', 'trigger_offset',
                    'max_prob', 'thresh_prob']
    output = []
    for i_, row in df.iterrows():
        if isinstance(row.trace_p_arrival, pd.Timestamp):
            line = [row.station_network_code, row.station_code, row.station_location_code, row.station_channel_code[:-1],
                    'P', row.trace_start_time, row.trace_P_onset, row.trace_p_arrival, pd.NaT, np.nan, np.nan]
            output.append(line)
        if isinstance(row.trace_s_arrival, pd.Timestamp):
            line = [row.station_network_code, row.station_code, row.station_location_code, row.station_channel_code[:-1],
                    'S', row.trace_start_time, row.trace_S_onset, row.trace_s_arrival, pd.NaT, np.nan, np.nan]
    df_out = pd.DataFrame(data=output, columns=stevens_columns)
    return df_out

def stevens2yuan(df, client):
    """
    Convert a dataframe in Stevens format into a Yuan format using a
    :class:`~.Client`-like metadata server connection to retrieve station
    location metadata to fill out missing fields

    :param df: Stevens formatted pick dataframe parsed by :meth:`~.read_yuan_pick_file`
    :type df: pandas.DataFrame

    :returns: **df_out** (*pandas.DataFrame*) -- Yuan pick formatted dataframe
    """
    yuan_columns=['event_id','source_type',
                  'station_network_code','station_channel_code',
                  'station_code','station_location_code',
                  'station_latitude_deg','station_longitude_deg',
                  'station_elevation_m','trace_name',
                  'trace_sampling_rate_hz','trace_start_time',
                  'trace_S_arrival_sample','trace_P_arrival_sample',
                  'trace_S_onset','trace_P_onset','trace_snr_db',
                  'trace_s_arrival','trace_p_arrival']


    # Get unique channel codes
    chan_codes = df.groupby(['network','station','location','band_inst']).size().index.values
    # Get starttime and endtime approximation for query
    tmin = df.trace_starttime.min()
    tmax = df.trigger_offset.max()

    # Compose bulk request for channel metadata
    ibulk = []
    for _n, _s, _l, _c in chan_codes:
        line = (_n, _s, _l, _c + '?', UTCDateTime(tmin), UTCDateTime(tmax))
        ibulk.append(line)

    # Get channel inventory for station metadata
    inv = client.get_stations_bulk(ibulk, level='channel')

    output = []
    for _, row in df.iterrows():
        iinv = inv.select(network=row.network, station=row.station, channel=row.band_inst+'?')
        # Safety check for station metadata
        if len(iinv) != 1:
            raise AttributeError(f'Query for {row.network}.{row.station}.*.{row.band_inst}? returned {len(iinv)} networks')
        elif len(iinv.networks[0]) != 1:
            raise AttributeError(f'Query for {row.network}.{row.station}.*.{row.band_inst}? returned {len(iinv.networks[0])} stations')
        elif len(iinv.networks[0].stations[0]) == 0:
            raise AttributeError(f'Query for {row.network}.{row.station}.*.{row.band_inst}? returned 0 channels')
        else:
            sta = iinv.networks[0].stations[0]
        # Shared formatting for all P and S picks
        line = ['','',
                row.network, row.band_inst+'?', row.station, row.location,
                sta.latitude, sta.longitude, sta.elevation,'',
                sta[0].sample_rate, row.trace_starttime]
        dt_max_sec = (row.pick_time - row.trace_starttime).total_seconds()
        dt_on_sec = (row.trigger_onset - row.trace_starttime).total_seconds()
        ## PHASE SPECIFIC FORMATTING
        # trace_S_arrival_sample, trace_P_arrival_sample,
        # trace_S_onset, trace_P_onset,
        # trace_snr_db,
        # trace_s_arrival, trace_p_arrival

        # Specific formatting for P-picks
        if row.label == 'P':
            line += [np.nan, dt_max_sec*sta[0].sample_rate,
                     np.nan, dt_on_sec*sta[0].sample_rate,
                     np.nan,
                     pd.NaT, row.pick_time]
        # Specific formatting for S-picks
        elif row.label == 'S':
            line += [dt_max_sec*sta[0].sample_rate, np.nan,
                     dt_on_sec*sta[0].sample_rate, np.nan,
                     np.nan,
                     row.pick_time, pd.NaT]
        else:
            raise ValueError(f'label "{row.label}" not supported - must be "P" or "S".')
        # Append line to output holder
        output.append(line)  
    df_out = pd.DataFrame(output, columns=yuan_columns)

    return df_out

### WRITING METHODS ###
def write_to_stevens_format(df, savename, **kwargs):
    # Safety catch on doubled call of index & header in to_csv
    if 'index' in kwargs.keys():
        _ = kwargs.pop('index')
    if 'header' in kwargs.keys():
        _ = kwargs.pop('header')

    if 'band_inst' in df.columns:
        pass
    elif 'station_channel_code' in df.columns:
        df = yuan2stevens(df)
    else:
        raise AttributeError('df does not appear to be in Stevens or Yuan format')

    df.index = list(range(len(df)))
    df.to_csv(savename, index=True, header=True, **kwargs)

    return df

def write_to_yuan_format(df, savename, **kwargs):
    # Safety catch on client being passed to stevens2yuan
    if 'client' in kwargs.keys():
        if hasattr(kwargs['client'], 'get_stations_bulk'):
            client = kwargs.pop('client')
        else:
            raise AttributeError('passed client does not have method get_stations_bulk')
    else:
        client = Client('IRIS')

    # Safety catch on doubled call of index & header in to_csv
    if 'index' in kwargs.keys():
        _ = kwargs.pop('index')
    if 'header' in kwargs.keys():
        _ = kwargs.pop('header')

    if 'station_channel_code' in df.columns:
        pass
    elif 'band_inst' in df.columns:
        df = stevens2yuan(df,client)
    else:
        raise AttributeError('df does not appear to be in Stevens or Yuan format')
    
    df.index = list(range(len(df)))
    df.to_csv(savename, index=True, header=True, **kwargs)

    return df


### MAIN PROCESS ###
def main(args):
    # Set Up Logging
    Logger = logging.getLogger('merge_pick_versions.py')
    if args.v:
        Logger.setLevel(logging.INFO)
    elif args.vv:
        Logger.setLevel(logging.DEBUG)
    else:
        Logger.setLevel(logging.WARNING)
    
    # Ensure no duplicate handler types
    shc = None
    chc = None
    # Get handlers
    hs = Logger.handlers
    for h in hs:
        if isinstance(h, logging.StreamHandler):
            shc = h
        if isinstance(h, CriticalExitHandler):
            chc = h
    # StreamHandler Setup
    if shc is None:
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        # Add formatting & handler
        Logger.addHandler(ch)

    # CriticalExitHandler Setup
    if chc is None:
        Logger.addHandler(CriticalExitHandler(exit_code=1))

    # Get client
    client = Client(args.client)
    Logger.debug(f'Loaded ObsPy client for {args.client}')

    # Get file list & check that it has reasonable values
    flist = []
    if len(args.input) > 0:
        for input in args.input:
            if not os.path.exists(input):
                Logger.warning(f'input file {input} does not exist - skipping')
            else:
                flist.append(input)
    else:
        Logger.critical('no input files provided')
    
    if len(flist) == 0:
        Logger.critical('no valid input files provided')
    
    # Check if programmatic file naming
    if '{' in args.output and '}' in args.output:
        fstr_out = True
    else:
        fstr_out = False
        if not os.path.exists(args.output):
            Logger.warning(f'Creating new output directory {args.output}')
            os.makedirs(args.output)

    supported_fields = ['{year}','{month}','{day}']
    if fstr_out:
        used_fields = []
        for sf in supported_fields:
            if sf in args.output:
                used_fields.append(sf)

    df_full = pd.DataFrame()
    for _f in flist:
        Logger.info(f'processing {_f}')
        # Determine format from naming convention
        fmt = fmtfromname(_f)
        # Load dataframes using relevant parser
        if fmt == 'stevens':
            df = read_stevens_pick_file(_f)
        elif fmt == 'yuan':
            df = read_yuan_pick_file(_f)
        else:
            Logger.warning(f'file {_f} did not appear to have Stevens or Yuan naming conventions')
            continue
        # Convert dataframe format if needed to meet output format specifications
        if fmt not in args.format:
            if fmt == 'stevens':
                df = stevens2yuan(df, client)
            else:
                df = yuan2stevens(df)
        # Append to full dataframe
        try:

            df_full = pd.concat([df_full, df], axis=0, ignore_index=True)
        except:
            breakpoint()
    # Get file naming formats and sort chronologically
    if 'stevens' in args.format:
        fname = '{network}_{station}_{startdate}_{enddate}.csv'
        df_full = df_full.sort_values(['trace_starttime','pick_time'])
    elif 'yuan' in args.format:
        fname = '{station}_{startdate}.csv'
        df_full = df_full.sort_values(['trace_start_time','trace_p_arrival','trace_s_arrival'])

    
    df = df_full
    # If not using a programmatically structured output path
    if not fstr_out:
        df_out_tuples = [(df_full, args.output)]

    else:
        if any(_e in used_fields for _e in ['{year}','{month}','{day}']):
            if 'yuan' in args.format:
                times = df.trace_start_time
            elif 'stevens' in args.format:
                times = df.trace_starttime
        else:
            Logger.critical('format string entries besides {year} {month} and {date} are not supported for --output')
        
        df_out_tuples = []

        # Get initial time to start time-chunking      
        t0 = pd.Timestamp(f'{times.min().year}-{times.min().month:02d}-{times.min().day:02d}', tz='UTC')

        if '{year}' in used_fields:
            iterlevel = 'year'
            t1 = pd.Timestamp(f'{times.min().year + 1}-01-01', tz='UTC')
            if '{month}' in used_fields:
                if '{year}' not in used_fields:
                    Logger.critical('If using {month}, must also use {year} in --output')
                iterlevel = 'month'
                if times.min().month < 12:
                    t1 = pd.Timestamp(f'{times.min().year:04d}-{times.min().month + 1:02d}-01', tz='UTC')
                else:
                    t1 = pd.Timestamp(f'{times.min().year + 1:04d}-01-01')
                if '{day}' in used_fields:
                    if '{year}' not in used_fields or '{month}' not in used_fields:
                        Logger.critical('If using {day}, must also use {year} and {month} in --output')
                    t1 = t0 + pd.Timedelta(1, unit='day', tz='UTC')
                    iterlevel = 'day'

        # Get last timestamp in this file
        if 'stevens' in args.format:
            tf = df.trigger_offset.max()
        else:
            tfp = df[df.trace_p_arrival.notna()].trace_p_arrival.max()
            tfs = df[df.trace_s_arrival.notna()].trace_s_arrival.max()
            tf = max([tfp, tfs])

        # Iterate across time span of ingested files
        while t0 < tf:
            # Subset dataframe into time-chunk
            if 'yuan' in args.format:
                odf = df[(df.trace_start_time >= t0) & (df.trace_start_time < t1)]
            else:
                odf = df[(df.trace_starttime >= t0) & (df.trace_starttime < t1)]

            # If there are picks
            if len(odf) > 0:
                ofmt = args.output.format(year=f'{t0.year:4d}', month=f'{t0.month:02d}', day=f'{t0.day:02d}')
                # Make directories if they don't exist
                if not os.path.exists(ofmt):
                    Logger.warning(f'Making new output directory {ofmt}')
                    os.makedirs(ofmt)
                # Ensure file formatting is as specified
                    
                

                Logger.info(f'Slice of inputs for {t0} -- {t1} has {len(odf)} entries')

                # If running Yuan formatting, split by station code only
                if 'yuan' in args.format:
                    # Iterate cross stations in _odf
                    for sta in odf.station_code:
                        _odf = odf[odf.station_code == sta]
                        if len(_odf) > 0:
                            starttime = _odf.trace_start_time.min()
                            startdate = f'{starttime.year:04d}{starttime.month:02d}{starttime.day:02d}'
                            savename = os.path.join(ofmt, fname.format(station=sta, startdate=startdate))
                            write_to_yuan_format(_odf, savename, client=client)
                        else:
                            Logger.warning(f'Encountered empty pick subset for {sta} using Yuan output format - skipping')
                            continue
                
                elif 'stevens' in args.format:
                    for net, sta in odf.groupby(['network','station']).size().index.values:
                        _odf = odf[(odf.network==net) & odf.station==sta]
                        if len(_odf) > 0:
                            starttime = _odf.trace_starttime.min()
                            startdate = f'{starttime.year:04d}{starttime.month:02d}{starttime.day:02d}'
                            endtime = _odf.pick_time.max()
                            enddate = f'{endtime.year:04d}{endtime.month:02d}{endtime.day:02d}'
                            savename = os.path.join(ofmt, fname.format(network=net, station=sta, startdate=startdate, enddate=enddate))
                            write_to_stevens_format(_odf, savename)
                        else:
                            Logger.warning(f'Encountered empty pick subset for {net}.{sta} using Stevens output format - skipping')

            # If there are no timestamps in this window
            else:
                Logger.warning(f'Slice of inputs for {t0} -- {t1} was empty -- skipping')


            # Increment up
            t0 = t1
            if iterlevel == 'year':
                t1 = pd.Timestamp(f'{t0.year + 1:04d}-01-01', tz='UTC')
            elif iterlevel == 'month':
                if t0.month < 12:
                    t1 = pd.Timestamp(f'{t0.year:04d}-{t0.month + 1:02d}-01', tz='UTC')
                else:
                    t1 = pd.Timestamp(f'{t0.year + 1:04d}-01-01', tz='UTC')
            elif iterlevel == 'day':
                t1 = t0 + pd.Timedelta(1, unit='day')
    


### COMMAND LINE INTERFACE
if __name__ == '__main__':

    # Initialize ArgParse
    parser = argparse.ArgumentParser(
        prog='merge_pick_formats.py',
        description='command-line interface tool for merging different pick file formats from Congcong Yuan and Nate Stevens',
    )
    # Add Input Arg
    parser.add_argument(
        '-i', '--input', dest='input', nargs='*', action='store', type=str, default='.',
        help='Input file name(s) to parse, accepts unix wild-cards (e.g., `path/to/my/UW/STA/picks*`). Must be a single string'
    )
    # Add Output Arg
    parser.add_argument(
        '-o', '--output', dest='output', action='store', type=str, default='./{year}/{month}/{day}',
        help='output path for joined data. Accepts use of the following format strings: {year}, {month}, {day}'+\
        'specific file naming conventions are tied to the "format" argument'
    )
    # Add Client Arg
    parser.add_argument(
        '-c','--client', dest='client', action='store', type=str, default='IRIS',
        help='Specifies which obspy.clients.fdsn.Client webservice to use for station metadata cross-checks'
    )
    # Add Output Format Arg
    parser.add_argument(
        '-f','--output_format', dest='format', type=str, default='stevens',
        help='specify the output file format convention you would like to use. This also dictates the file naming convention. Also see --output',
        choices=['stevens','yuan']
    )
    # Add INFO Level Logging Switch
    parser.add_argument(
        '-v', dest='v', action='store_true',
        help='Some verbosity, sets command-line logging at the INFO level'
    )
    # Add DEBUG Level Logging Switch
    parser.add_argument(
        '-V', dest='vv', action='store_true',
        help='High verbosity, sets command-line logging at the DEBUG level'
    )

    ### PARSE ARGS
    args = parser.parse_args()
    
    ### RUN MAIN
    main(args)