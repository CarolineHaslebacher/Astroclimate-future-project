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
d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

#%%


def read_ERA5_seeing_data(pressure_levels_list, base_path):

  # in hPa
  #pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
  pr_levels = pressure_levels_list

  for i in range(0,len(pr_levels)):
    #'./sites/Paranal/Era5_data/seeing/'
    path = str(base_path) + str(pr_levels[i]) + 'hPa/*.nc'
    ds = xr.open_mfdataset(path, combine = 'by_coords')
    ds_sel = ds.sel(expver = 5, longitude= -70.25,latitude= -24.75,method='nearest') # green point
    
    # select only UTC 00:00 to 09:00 (nighttime hours)
    #ds_sel_nighttime = ds_sel.where(ds_sel['time'].dt.hour <= 9) 
    
    # create new coordinate pressure "level"
    ds_P = ds_sel.assign_coords({"level": pr_levels[i]})

    # concatenate DataArrays along new dimension 'level'
    if i == 0:
      ds_full = ds_P
    else:
      ds_full = xr.concat([ds_full, ds_P], 'level')

    # ds_full is a dataArray that can be accessed via ds_full.variable[pressure level index][time index]
    # for example: ds_full.u[10][0]

    ## monthly means
    #ds_sel = ds_sel.resample(time = 'M').mean()
    
    # # append xarray's to list
    #Seeing_pressure_levels.append(ds_sel_nighttime)

  #df_full = ds_full.to_dask_dataframe()

  return ds_full

#%%

# read in ERA5 u and v on pressure levels and u_surf and v_surf
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
ds_paranal = read_ERA5_seeing_data(pr_levels, './sites/Paranal/Era5_data/seeing/')

# calculate seeing and CN2 profile datasets
ds_seeing, ds_Cn2_profile = climxa.ERA5_seeing_calc(ds_paranal, pr_levels[:-1])
# calculate median of Cn2 profile
Cn2_median = ds_Cn2_profile.Cn2.median(dim='time').load() # next time: read in dataset that is already saved!
# quickly save to csv, so that it does not have to load every time
Cn2_median.to_dataframe().to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/Cn2_profile_median_test.csv')


# select lon and lat
lon = -70.5
lat = -24.75

# calculate also for La Silla and Cerro Tololo
for idx in range(2, 4):
    # or define index for one iteration only
    # idx = 3

    # if idx == 6:
    #     continue

    file_name = ''

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    site = 'Paranal'
    variable = '10m_u_component_of_wind'
    single_lev_var = 'u10'

    ds_u10 = climxa.read_ERA5_sg_level_data(site, variable , single_lev_var, ERA5_path=None)

    variable = '10m_v_component_of_wind'
    single_lev_var = 'v10'

    ds_v10 = climxa.read_ERA5_sg_level_data(site, variable , single_lev_var, ERA5_path=None)


    ds_u10 = ds_u10.sel(longitude= lon, latitude= lat, method='nearest')
    ds_v10 = ds_v10.sel(longitude= lon, latitude= lat, method='nearest')

    # calculate surface seeing
    g = 9.80665 #m/s^2, from Era5 documentation 
    A = 5 * 10**(-16) # calibration factor, we just take something to get values in the correct order
    B = 5 * 10**(-14)
    i = 5

    J_surf = B * (ds_u10.u10**2 + ds_v10.v10**2)
    J_upper = A * (ds_paranal.u[i]**2 + ds_paranal.v[i]**2)

    # J_surf[0].values
    # J_upper[0].values
    # COMMENT: If I take the same calibratin factor A for both J_surf and J_upper, then J_upper dominates
    # play around with A and B

    # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
    ds_seeing_surf = xr.Dataset({"surface_seeing": climxa.epsilon(J_surf + J_upper)})


    # ds_seeing_surf = xr.Dataset({"surface_seeing": climxa.epsilon(J_surf)})
    # load
    ds_seeing_surf_loaded =  ds_seeing_surf.load()

    # change coords
    ds_seeing_surf_loaded = ds_seeing_surf_loaded.reset_coords(drop=True)
    # and save to netcdf
    ds_seeing_surf_loaded.to_netcdf('/home/haslebacher/chaldene/Astroclimate_Project/sites/' + site_name_folder + '/Data/Era_5/seeing_nc/ds_surface_seeing.nc')



# resample yearly
# ds_yearly = ds_seeing_surf_loaded.resample(time = 'y').mean()
ds_yearly = ds_seeing_surf_loaded.groupby('time.year').mean(dim='time')

# plot ds_seeing_surf 
plt.plot(ds_yearly.year, ds_yearly['surface_seeing'])
# just run xarray_all_sites_seeing.py once for Paranal and take data from there
# in situ
plt.plot(d_obs['ds_mean_year']['seeing'].year, d_obs['ds_mean_year']['seeing'])
# plt.plot(d_obs['ds_mean_year']['wind speed seeing'].year, d_obs['ds_mean_year']['wind speed seeing'])
plt.plot(d_obs['ds_mean_year']['Seeing Paranal'].year, d_obs['ds_mean_year']['Seeing Paranal'])
# plot vertical profile, add u_surf and v_surf --> takes too much computing time!


# calculate new seeing

# plot
plt.plot(Cn2_median, ds_Cn2_profile.level)
plt.gca().invert_yaxis()
plt.xscale('log')


#%%