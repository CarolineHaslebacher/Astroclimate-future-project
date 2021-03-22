# 2020-08-10

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr

import xarray.plot as xplt

from matplotlib import dates as d
import datetime as dt
import time

from itertools import cycle
from functools import reduce
from scipy import stats

import csv

import seaborn as sns
sns.set()
from matplotlib import cycler

import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

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

#%% global parameters

list_of_single_level_vars = ['tcw']
variable = 'TCW'
list_of_clim_vars = ['q_integrated']
list_of_model_clim_params = ['q_integrated']
list_of_single_level_model_clim_params = ['prw']

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


#%% integrate SH to precipitable water vapor
# do this only once, save it as netcdf file and load it next time

# ERA5, PWV

importlib.reload(climxa)

for idx in range(0, 8):

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
    print(surface_pressure_observation)
    # check available pressure for ERA5 
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    pr_levels_ERA5 = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa

    given_value = surface_pressure_observation
    closest_value = min(pr_levels_ERA5, key=absolute_difference_function)

    SH_integral_pressure = closest_value # find nearest pressure

    print('closest match: {}'.format(SH_integral_pressure))

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

    ds_q_integrated, ds_tcw_profile = climxa.SH_integral_to_TCW(SH_integral_pressure, site_ERA, my_ERA5_lon, lat)

    # add attributes
    ds_q_integrated['q_integrated'].attrs['long_name'] = 'Precipitable Water Vapor'
    ds_q_integrated['q_integrated'].attrs['units'] = 'mmH2O' # = 1kg/m^2

    # add coordinate 'level'
    ds_q_integrated['q_integrated'] = ds_q_integrated['q_integrated'].assign_coords(level=SH_integral_pressure)
    # add dimension 'level'
    ds_q_integrated['q_integrated'] = ds_q_integrated['q_integrated'].expand_dims(dim='level')
    # now, the level can be selected with e.g. ds.sel(level=775)

    # define path
    path = './sites/'+ site_name_folder + '/Data/Era_5/q_integrated/ds_ERA5_q_integrated_hourly_nearest_' + str(SH_integral_pressure) + 'hPa.nc' # where to save the files
    
    # make directory if not available
    os.makedirs(os.path.dirname(path), exist_ok=True) 

    # save array to netcdf file, store it 
    ds_q_integrated.to_netcdf(path)

# %%
