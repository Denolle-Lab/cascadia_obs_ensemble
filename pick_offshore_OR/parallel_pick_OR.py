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
from picking_utils_OR import *

device = torch.device("cpu")

# Define clients
client_inventory = Client('IRIS')
client_waveform = WaveformClient()
client_ncedc = Client('NCEDC')

# Parameters
year = 2016
month=6
day=4
filepath = "/home/hbito/cascadia_obs_ensemble/data/picks_OR/"
os.makedirs(filepath,exist_ok=True)

twin = 6000     # length of time window
step = 3000     # step length
l_blnd, r_blnd = 500, 500

# Now create your list of days to loop over!
time1 = datetime.datetime(year=year,month=month,day=day,hour=0,minute=15,second=0, microsecond=0)
time2 = datetime.datetime(year=year,month=month,day=day,hour=0,minute=45,second=0, microsecond=0)

inventory = client_inventory.get_stations(network="PB,NV,UW,UO,OO", station="*", minlatitude=40,minlongitude=-127,maxlatitude=50,maxlongitude=-123, starttime=time1.strftime('%Y%m%d'),endtime=time2.strftime('%Y%m%d'))



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
	task_list.append([networks_stas[i][0], networks_stas[i][1],networks_stas[i][2],networks_stas[i][3],networks_stas[i][4]])
        
# Now we start setting up a parallel operation using a package called Dask.

@dask.delayed
def loop_days(task,filepath,twin,step,l_blnd,r_blnd):

    # Define the parameters that are specific to each task
    t1 = obspy.UTCDateTime(time1)
    t2 = obspy.UTCDateTime(time2)
    network = task[0]
    station = task[1]
    lat =task[2]
    lon= task[3]
    elev=task[4]                    

    #print network and station
    print([network,station,t1])
    # Call to the function that will perform the operation and write the results to file
    try: 
        run_detection(network,station,t1,t2,filepath,twin,step,l_blnd,r_blnd,lat,lon,elev)
    except:
        return


# Now we set up the parallel operation
# The below builds a framework for the computer to run in parallel. This doesn't actually execute anything.
lazy_results = [loop_days(task,filepath,twin,step,l_blnd,r_blnd) for task in task_list]
    

# The below actually executes the parallel operation!
# It's nice to do it with the ProgressBar so you can see how long things are taking.
# Each operation should also write a file so that is another way to check on progress.
with ProgressBar():
    #################################
    # Add scheduler = 'single-threaded'
	dask.compute(lazy_results, scheduler='single-threaded') 
    

