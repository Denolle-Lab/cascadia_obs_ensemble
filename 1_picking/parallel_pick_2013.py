import logging
import os
import sys

from obspy.clients.fdsn import Client
import numpy as np
import obspy
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pandas as pd
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar

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
from picking_utils_2013 import *

device = torch.device("cpu")
print('test')

# Define clients
client_inventory = Client('IRIS')
client_waveform = WaveformClient()
client_ncedc = Client('NCEDC')

# Parameters
year= 2013
filepath = f"/home/hbito/cascadia_obs_ensemble_backup/data/picks_{year}_122-129/"
os.makedirs(filepath,exist_ok=True)

twin = 6000     # length of time window
step = 3000     # step length
l_blnd, r_blnd = 500, 500

# Now create your list of days to loop over!
time1 = datetime.datetime(year=year,month=1,day=1)
time2 = datetime.datetime(year=year+1,month=1,day=1)
time_bins = pd.to_datetime(np.arange(time1,time2,pd.Timedelta(1,'days')))

inventory = client_inventory.get_stations(network="C8,7D,7A,CN,NV,UW,UO,NC,BK,TA,OO,PB,X6,Z5,X9", station="*", minlatitude=40,minlongitude=-129,maxlatitude=50,maxlongitude=-122, starttime=time1.strftime('%Y%m%d'),endtime=time2.strftime('%Y%m%d'))



# Make a list of networks and stations
networks_stas = []
lat =[]
lon =[]
elev =[]

for i in range(len(inventory)):
    network = inventory[i].code
    
    for j in range(len(inventory[i])):
        networks_stas.append([network,inventory[i].stations[j].code,
                              inventory[i].stations[j].latitude,
                              inventory[i].stations[j].longitude,inventory[i].stations[j].elevation])
    

networks_stas =np.array(networks_stas)
    
# download models
pretrain_list = ["pnw","ethz","instance","scedc","stead","geofon"]
pn_pnw_model = sbm.EQTransformer.from_pretrained('pnw')
pn_ethz_model = sbm.EQTransformer.from_pretrained("ethz")
pn_instance_model = sbm.EQTransformer.from_pretrained("instance")
pn_scedc_model = sbm.EQTransformer.from_pretrained("scedc")
pn_stead_model = sbm.EQTransformer.from_pretrained("stead")
pn_geofon_model = sbm.EQTransformer.from_pretrained("geofon")

# Combine that list of days with the list of stations and networks
# We are essentially creating a list of the number of tasks we have to do with the information that is unique to each task; we will do them in parallel
task_list = []
for i in range(len(networks_stas)):
	for t in time_bins:
		task_list.append([networks_stas[i][0], networks_stas[i][1],networks_stas[i][2],networks_stas[i][3],networks_stas[i][4],t])
# Now we start setting up a parallel operation using a package called Dask.

@dask.delayed
def loop_days(task, filepath, twin, step, l_blnd, r_blnd):
    # Define the parameters that are specific to each task
    t1 = obspy.UTCDateTime(task[5])
    t2 = obspy.UTCDateTime(t1 + pd.Timedelta(1,'days'))
    network = task[0]
    station = task[1]
    lat = task[2]
    lon = task[3]
    elev = task[4]

    # Print network and station
    print([network, station, t1])
    # Call to the function that will perform the operation and write the results to file
    try:
        run_detection(network, station, t1, t2, filepath, twin, step, l_blnd, r_blnd, lat, lon, elev)
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    # Define your task_list, filepath, twin, step, l_blnd, r_blnd here or import them from another module
    task_list = task_list # Replace with your actual task list
    filepath = filepath  # Replace with your actual file path
    twin = twin
    step = step
    l_blnd = l_blnd
    r_blnd = r_blnd

    # Wrap loop_days with dask.delayed
    lazy_results = [dask.delayed(loop_days)(task, filepath, twin, step, l_blnd, r_blnd) for task in task_list]

    # Use ProgressBar to track the progress
    with ProgressBar():
        # Using the processes scheduler with num_workers specified
        compute(lazy_results, scheduler='processes', num_workers=2)
    

