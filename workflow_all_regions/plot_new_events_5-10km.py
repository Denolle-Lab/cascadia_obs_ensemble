import os
import sys
import pandas as pd
import numpy as np
import obspy
from tqdm import tqdm

from obspy.clients.fdsn import Client
from obspy.clients.fdsn import Client as FDSNClient

from mpl_toolkits.basemap import Basemap


from pnwstore.mseed import WaveformClient
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from obspy import UTCDateTime

from tqdm import tqdm
from obspy.geodetics import locations2degrees, degrees2kilometers

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, '../'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from plot_utils import *

# Load all the pick assignments 
year = 'all_regions'

# Load all event data into a list of DataFrames
new_events = pd.read_csv(f'../data/datasets_{year}/new_events.csv')
new_events = new_events.sort_values(by='time').reset_index(drop=True)


# Pick only the events less than 5 km deep
new_events = new_events.loc[(new_events['depth'] >= 5)&(new_events['depth'] < 10)]

# Load all pick_assignments data into a list of DataFrames
pick_assignment_files = [
    '../data/datasets_nwa_shelf_trench/all_pick_assignments_nwa_shelf_trench.csv',
    '../data/datasets_nwa_shore/all_pick_assignments_nwa_shore.csv',
    '../data/datasets_or_shelf_trench/all_pick_assignments_or_shelf_trench.csv',
    '../data/datasets_or_shore/all_pick_assignments_or_shore.csv',
    '../data/datasets_pnsn_jdf/all_pick_assignments_pnsn_jdf.csv',
    '../data/datasets_pnsn_nor/all_pick_assignments_pnsn_nor.csv',
    '../data/datasets_pnsn_sor/all_pick_assignments_pnsn_sor.csv',
    '../data/datasets_pnsn_wa/all_pick_assignments_pnsn_wa.csv',
    '../data/datasets_swa_shelf_trench/all_pick_assignments_swa_shelf_trench.csv',
    '../data/datasets_swa_shore/all_pick_assignments_swa_shore.csv'
]

# Read all event files into DataFrames
dfs = [pd.read_csv(file, index_col=0) for file in pick_assignment_files]

# Concatenate all DataFrames into a single DataFrame
mycatalog_picks_merged = pd.concat(dfs)

# Load networks
networks = pd.read_csv('../data/networks.csv')

# Define the channels to show in the plot
channel = '?H?'

fig_title = 'New Event (5-10 km deep) from any region'

path = f'/home/hbito/cascadia_obs_ensemble/data/datasets_{year}/plots_new_events_5-10km/'

subplots_cluster_scale(new_events, mycatalog_picks_merged, networks, channel, fig_title,path)