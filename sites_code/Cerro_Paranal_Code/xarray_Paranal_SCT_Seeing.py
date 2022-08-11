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

list_of_single_level_vars = ['wind speed seeing']
list_of_clim_vars = ['seeing']
variable = 'seeing_nc'
list_of_clim_vars = ['seeing']
list_of_model_clim_params = ['seeing']
# list_of_single_level_model_clim_params = ['seeing']

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

#%% seeing classic

# df_Seeing_integral = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2.csv')
# list_of_single_level_vars = ['seeing']

# df_comp_seeing = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2000to2020_dropnan.csv', index_col='time', parse_dates=True)
# ds_seeing = df_comp_seeing.to_xarray()
# ds_seeing = ds_seeing.resample(time='1m').mean()
# ds_seeing = ds_seeing.assign_coords({'longitude': -70.40, 'latitude': -24.63}) # not coords, but dims!
# ds_seeing = ds_seeing.expand_dims(dim = ['longitude', 'latitude'])
# ds_seeing.seeing.attrs['long_name'] = 'Seeing'
# ds_seeing.seeing.attrs['units'] = 'arcsec'
# # save netcdf file where my function is looking for it
# ds_seeing.to_netcdf('./sites/Paranal/Data/Era_5/seeing_nc/1979to2020_monthly_seeing_Paranal.nc')

#%% free atmosphere

# list_of_single_level_vars = ['free_seeing']

# df_comp_seeing = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_free_Seeing_integral_2000to2020_dropnan.csv', index_col='time', parse_dates=True)
# ds_seeing = df_comp_seeing.to_xarray()
# ds_seeing = ds_seeing.resample(time='1m').mean()
# ds_seeing = ds_seeing.assign_coords({'longitude': -70.40, 'latitude': -24.63}) # not coords, but dims!
# ds_seeing = ds_seeing.expand_dims(dim = ['longitude', 'latitude'])
# ds_seeing.free_seeing.attrs['long_name'] = 'Free Atmosphere Seeing'
# ds_seeing.free_seeing.attrs['units'] = 'arcsec'
# # save netcdf file where my function is looking for it
# ds_seeing.to_netcdf('./sites/Paranal/Data/Era_5/seeing_nc/1979to2020_monthly_seeing_Paranal.nc')


#%%


file_name = '_ERA5_HadGEM_to_dimm-mass-seeing_'


idx = 1

print(d_site_lonlat_data['site_name'][idx])
# lon_obs and lat_obs are in 0-360 format!!
lon = d_site_lonlat_data['lon_obs'][idx]
lat = d_site_lonlat_data['lat_obs'][idx]

site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
# time_slice_var = slice('2000-01-01', '2016-01-01')
time_slice_var = slice('2016-01-01', '2019-01-01')

path = './sites/'+ site_name_folder + '/Output/Plots/seeing/' # where to save the files

# read in ds_hourly (in-situ data)
ds_hourly_1 = climxa.df_to_xarray('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
ds_hourly_2 = climxa.df_to_xarray('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv')

ds_hourly = xr.merge([ds_hourly_1, ds_hourly_2])

# define observational dict
# 'Seeing Paranal', 
d_obs = {"ds": ds_hourly, "insitu_var": ['MASS_DIMM_Seeing', 'free_atmosphere_seeing'] , "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars} # 

# d_obs = {"single_lev": list_of_single_level_vars} # only ERA5
# d_obs_2 = {'ds': ds_hourly, "insitu_var":  ['Seeing Paranal'], "single_lev": list_of_single_level_vars}

# d_model = None


# # define climate model levels
ls_pr_levels_clim_model = [850]
d_model = {"HadGEM": {"folders": ['present'],"taylor_folder": ['present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"}}

importlib.reload(climxa)

import time
start_time = time.time() # measure elapsed time
d_obs_ss, d_model_ss, fig_ss, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                time_slice_var, d_obs, lon, lat, path, diurnal=False,
                                    d_model = d_model, SH_integral=False)
print("--- %s seconds ---" % (time.time() - start_time))
# save as png with high resolution. Only save as pdf in the 'last round'
# fig_ss.savefig('./Model_evaluation/' + variable + '/' + folder_for_path + '/' + site_name_folder + '_' + variable + '_DSC_T.png', dpi=400)
fig_ss.savefig('./Model_evaluation/' + variable + '/' + site_name_folder + '_' + variable + file_name + '.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)

# fig_ss.savefig('./Model_evaluation/' + variable + '/' + site_name_folder + '_' + variable + '_DSC_T.png', dpi=400)

# save dict to csv
path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/csv_info/'
os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
(pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
    .to_csv(path_skill_folder + site_name_folder + file_name + '_sorted_skill_dict.csv', header=False))



# %%
# seeing to 200hPa wind speed
