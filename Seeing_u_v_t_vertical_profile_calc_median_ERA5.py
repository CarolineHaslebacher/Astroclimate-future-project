# this script plots the vertical profile of the astronomical seeing


#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
 
import netCDF4
import xarray as xr

import xarray.plot as xplt 
#import sunpy.timeseries as ts 

########## for cycle ##############
from matplotlib import dates as d
import datetime as dt
import time

from itertools import cycle
from functools import reduce
from scipy import stats
import csv
 
from matplotlib.lines import Line2D

import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa
import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


#%% RELOAD CLIMXA
#%%
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file

import climxa

import importlib
importlib.reload(climxa)

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')


#%% load in site specific data

# [0]: Mauna Kea
# [1]: Cerro Paranal
# [2]: La Silla
# [3]: Cerro Tololo
# [4]: La Palma
# [5]: Siding Spring
# [6]: Sutherland
# [7]: SPM

# d_site_lonlat_data = pd.read_csv('/home/haslebacher/chaldene/Astroclimate_Project/Sites_lon_lat_ele_data.csv')
d_site_lonlat_data = pickle.load( open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))



#%% temperature

# variable = 'T'
# list_of_clim_vars = ['t']
# list_of_model_clim_params = ['ta']
# my_xlabel = r'Temperature [$^{\circ}$ C]'

#%%
# read in seeing vars (t, u, v)
# only ERA5!
def get_seeing_variables(idx, d_site_lonlat_data):

    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    chile_grid = ['Tololo', 'Pachon', 'Silla']
    if any(x in site_name_folder for x in chile_grid):
        site_ERA = 'Paranal'
    else:
        site_ERA = site_name_folder

    if lon > 180:
        my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
        print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
    else:
        my_ERA5_lon = lon

    # use function which loads in all specific humidity datasets 
    # and integrates them to specific humidity

    if site_ERA == 'Paranal': # Paranal
        seeing_data_path =  './sites/' + site_ERA + '/Era5_data/seeing/'
    else:
        seeing_data_path =  './sites/' + site_ERA + '/Data/Era_5/seeing/'

    ds_seeing_vars = climxa.read_ERA5_seeing_data(seeing_data_path, my_ERA5_lon, lat)
    ds_seeing_vars = ds_seeing_vars.load() # load here to prevent program from running 

    return ds_seeing_vars



#%% 

for idx in range(0,8):

    if idx == 7:
        # already saved
        continue

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    # read data (this is hourly data!)
    ds_u_v_t = get_seeing_variables(idx, d_site_lonlat_data)
    # or resample monthly first? --> yes, this  makes the PIs a bit narrower and comparable to PRIMAVERA
    ds_u_v_t_resampled = ds_u_v_t.resample(time = '1m').mean()

    # calculate time median and store! (load next time)
    # ds_median = ds_u_v_t.median(dim='time')
    # ds_25PI = ds_u_v_t.quantile(q=0.25, dim='time')
    # ds_75PI = ds_u_v_t.quantile(q=0.75, dim='time')

    # take monthly resampled data
    ds_median = ds_u_v_t_resampled.median(dim='time')
    ds_25PI = ds_u_v_t_resampled.quantile(q=0.25, dim='time')
    ds_75PI = ds_u_v_t_resampled.quantile(q=0.75, dim='time')

    # test with:
    #   plt.plot(ds_median.u)
    #   plt.plot(ds_25PI.u)
    #   plt.plot(ds_75PI.u)

    # combine to one xarray if quickly done
    ds_iqr = xr.concat([ds_25PI, ds_75PI], dim='quantile')

    # save to netcdf
    median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'
    iqr_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_IQR_ERA5_u_v_t_z.nc'

    ds_median.to_netcdf(median_path)
    ds_iqr.to_netcdf(iqr_path)


#%%


# # load
# median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'

# ds_median_load = xr.open_dataset(median_path).load()

