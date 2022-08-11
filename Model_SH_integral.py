# 2020-08-29
# this function integrates the specific humidity
# from the model data sets
# and saves them into a separate netcdf file

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
list_of_model_clim_params = None
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

#%%

# decide if we should save it:
save = True

# climate models: SH integral

importlib.reload(climxa)


pr_levels_model = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# CNRM:     100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# CMCC:     100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# EC-Earth: 100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# ECMWF:    100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# HadGEM:   100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# MPI:      100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# --> all the same!


list_of_model_clim_params = ['hus']
list_of_single_level_model_clim_params = None
ls_pr_levels_clim_model = pr_levels_model # load in all pressure levels
# the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
            "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
            "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
            "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
            "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
            "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

# EC-Earth, 'present' --> but only leads to errors (I think data download was issue) --> run again without 'present'
d_model = {'EC-Earth': {"folders": ['hist', 'future'], "taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"}}

for idx in range(0, 8):

    # idx = 5

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]
    site = d_site_lonlat_data['site_name_folder'][idx]

    surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
    print(surface_pressure_observation)
    # check available pressure for climate models
    absolute_difference_function = lambda list_value : abs(list_value - given_value)

    given_value = surface_pressure_observation
    closest_value = min(pr_levels_model, key=absolute_difference_function)

    SH_integral_pressure = closest_value # find nearest pressure

    print('closest match: {}'.format(SH_integral_pressure))


    for clim_key in d_model.keys():

        # use function which loads in all specific humidity datasets
        d_model[clim_key]['ds_hus'] = climxa.get_PRIMAVERA(d_model, clim_key, site, pressure_level=True)

        # select lon/lat
        d_model[clim_key]['ds_sel'] = climxa.xr_sel(d_model[clim_key], 'ds_hus', lon, lat)

        # find maximal index
        pr_max_idx = pr_levels_model.index(SH_integral_pressure)

        # initialize dataset
        ds_q_integrated = xr.Dataset()

        # integrate
        for forc_idx, forcing in enumerate(d_model[clim_key]['folders']):
            summe = climxa.Euler_centred_PWV(d_model[clim_key]['ds_sel'], pr_max_idx, 'hus ' + forcing)

            # print(summe)
            # write integral to dataset
            ds_q_integrated["q_integrated " + forcing] = summe

            # add attributes
            ds_q_integrated["q_integrated " + forcing].attrs['long_name'] = 'Precipitable Water Vapor'
            ds_q_integrated["q_integrated " + forcing].attrs['units'] = 'mmH2O' # = 1kg/m^2

            # add coordinate 'level'
            if forc_idx == 0 and 'level' in ds_q_integrated.coords:
                # drop level, otherwise it is not overwritten properly sometimes
                ds_q_integrated = ds_q_integrated.drop('level')

            ds_q_integrated["q_integrated " + forcing] = ds_q_integrated["q_integrated " + forcing].assign_coords(level=SH_integral_pressure)
            # add dimension 'level'
            ds_q_integrated["q_integrated " + forcing] = ds_q_integrated["q_integrated " + forcing].expand_dims(dim='level')
            # now, the level can be selected with e.g. ds.sel(level=775)

        if save:
            # define path
            path = './sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(SH_integral_pressure) + 'hPa.nc' # where to save the files

            # make directory if not available
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # save array to netcdf file, store it
            ds_q_integrated.to_netcdf(path)

#%%