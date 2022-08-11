
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr
import os
from pathlib import Path
import csv

#%% extract tracking IDs of PRIMAVERA

base_path = Path('/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables')

for mypath in sorted(base_path.glob('**')): # get paths of all directories we want to scan
    # e.g. mypath = /home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables/clt/Amon/HadGEM/present
    ncfiles = sorted(mypath.glob('*.nc'))

    if len(ncfiles) > 0: # otherwise we got a higher-up folder

        var = mypath.parents[2].stem # e.g. 'clt'
        modelname = mypath.parents[0].stem # e.g. 'CNRM'
        simu = mypath.stem # e.g. 'future'

        # define csv filename, prepare csv file
        csv_dir = '/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/tracking_ids_PRIMAVERA/'
        csv_name = var + '_' + modelname + '_' + simu + '_tracking_IDs.csv'

        # initialize list for tracking IDs
        myids = []

        for myfile in ncfiles:
            # load each file, extract attributes and save as csv
            mync = xr.open_dataset(myfile)
            # extract tracking id
            myids.append(mync.attrs['tracking_id'])

        # now write to csv
        df = pd.DataFrame(myids)
        df.to_csv(csv_dir + csv_name, index=False, header=False)


#%% tests
# # load one .nc file and try to extract the tracking ID
# file_name = 'ta_Amon_CMCC-CM2-VHR4_highres-future_r1i1p1f1_gn_201501-201501.nc'
# mync = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables/ta/Amon/CMCC/future/' + file_name)
# # extract tracking id
# mytrack = mync.attrs['tracking_id']
# # write tracking_id to csv (append)
# csv_name = file_name[:-20] # e.g. 
# with open(csv_name, )


# %%

