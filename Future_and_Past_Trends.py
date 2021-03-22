# Caroline Haslebacher, 2020
# this script must be run before the Bayesian Analysis
# It extracts and stores the monthly timeseries in a csv file which serves
# as the input to the Bayesian analysis
# as an aside, it also calculates a trendline with classical linear regression

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
import time

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






#%% TEMPERATURE
# write function for d_Ensemble trends

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# input: units
# units = '[$^{\circ} C$/decade]'
# unit = '[$^{\circ} C$]'
units = '[deg C/decade]'
unit = '[deg C]'

variable = 'T'
significant_digits = 3
ylabel = r'Temperature [$^{\circ} C$]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):
    # if idx == 0 or idx == 1 or idx == 3 or idx == 6: # Mauna Kea, Paranal, Tololo, Sutherland
    #     list_of_model_clim_params = None
    #     ls_pr_levels_clim_model = None
    #     list_of_single_level_model_clim_params = ['tas']

    # elif idx == 7: # SPM
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [700]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 2 or idx == 4: # La Silla, La Palma
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [850]
    #     list_of_single_level_model_clim_params = None


    # elif idx == 5: # siding spring
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [925]
    #     list_of_single_level_model_clim_params = None

    # new best matches after comparing PRIMAVERA to ERA5 instead of to in-situ
    # if idx == 5 or idx == 6: # siding spring, Sutherland
    #     list_of_model_clim_params = None
    #     ls_pr_levels_clim_model = None
    #     list_of_single_level_model_clim_params = ['tas']

    # elif idx == 0: # SPM, La Palma, Paranal
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [600]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 7 or idx == 4 or idx == 1: # SPM, La Palma, Paranal
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [700]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 2 or idx == 3: # La Silla, Tololo
    #     list_of_model_clim_params = ['ta']
    #     ls_pr_levels_clim_model = [850]
    #     list_of_single_level_model_clim_params = None

    list_of_model_clim_params = ['ta']
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    list_of_single_level_model_clim_params = None


    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)

climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)



#%% RELATIVE HUMIDITY
# Future trendlines

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# input: units (for LATEX table)
units = '[%/decade]'
unit = '[%]'

variable = 'RH'
significant_digits = 3
ylabel = r'Relative Humidity [%]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):
    # if idx == 5: # siding Spring
    #     list_of_model_clim_params = ['hur']
    #     ls_pr_levels_clim_model = [925] # instead of 1000hPa (too many nan...)
    #     list_of_single_level_model_clim_params = None

    # elif idx == 7: # SPM
    #     list_of_model_clim_params = ['hur']
    #     ls_pr_levels_clim_model = [700]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 2 or idx == 3 or idx == 1 or  idx == 4 or idx == 6: # La Silla, Tololo, Paranal, La Palma, sutherland
    #     list_of_model_clim_params = ['hur']
    #     ls_pr_levels_clim_model = [850]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 0: # Mauna Kea
    #     list_of_model_clim_params = ['hur']
    #     ls_pr_levels_clim_model = [600]
    #     list_of_single_level_model_clim_params = None

    list_of_model_clim_params = ['hur']
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    list_of_single_level_model_clim_params = None

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# test
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)

#%% SPECIFIC HUMIDITY
# future trendlines

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# input: units
units = '[$kg/kg$/decade]'
unit = '[$kg/kg$]'

variable = 'SH'
significant_digits = 3
ylabel = r'Specific Humidity [$kg/kg$]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):
    # if idx == 5: # siding Spring
    #     list_of_model_clim_params = ['hus']
    #     ls_pr_levels_clim_model = [925]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 1 or idx == 2 or idx == 3 or idx == 4 or idx == 6: # Paranal, La Silla, Tololo, La Palma, sutherland
    #     list_of_model_clim_params = ['hus']
    #     ls_pr_levels_clim_model = [850]
    #     list_of_single_level_model_clim_params = None


    # elif idx == 7: # SPM
    #     list_of_model_clim_params = ['hus']
    #     ls_pr_levels_clim_model = [700]
    #     list_of_single_level_model_clim_params = None

    # elif idx == 0: # Mauna KEa
    #     list_of_model_clim_params = ['hus']
    #     ls_pr_levels_clim_model = [600]
    #     list_of_single_level_model_clim_params = None

    list_of_model_clim_params = ['hus']
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    list_of_single_level_model_clim_params = None
        

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# test
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)

#%% PWV 
# future trendlines

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# input: units (for LATEX table)
units = '[$mmH_2O$/decade]'
unit = '[$mmH_2O$]'

variable = 'TCW'
significant_digits = 3
ylabel = r'Precipitable Water \nVapor [mmH_2O]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):
    if idx == 5: # siding Spring
        list_of_model_clim_params = ['q_integrated']
        ls_pr_levels_clim_model = [925]
        list_of_single_level_model_clim_params = None

    elif idx == 1 or idx == 2 or idx == 4 or idx == 7: # SPM, Paranal, La Silla, La Palma
        list_of_model_clim_params = ['q_integrated']
        ls_pr_levels_clim_model = [700]
        list_of_single_level_model_clim_params = None

    elif idx == 3 or idx == 6: # Tololo, sutherland
        list_of_model_clim_params = ['q_integrated']
        ls_pr_levels_clim_model = [850]
        list_of_single_level_model_clim_params = None

    elif idx == 0: # Mauna Kea
        list_of_model_clim_params = ['q_integrated']
        ls_pr_levels_clim_model = [600]
        list_of_single_level_model_clim_params = None


    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# test
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)

#%% clouds
# future trendlines

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# input: units (for LATEX table)
units = '[fraction/decade]'
unit = '[fractional]'

variable = 'total_cloud_cover'
significant_digits = 3
ylabel = r'total cloud cover [fractional]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):

    list_of_single_level_model_clim_params = ['clt']
    ls_pr_levels_clim_model = None
    list_of_model_clim_params = None

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# future trendlines (plots and saves data to csv)
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)

#%% seeing
# ATTENTION: we do not save the csv files for the two different seeing approaches separately!

# take ds_ensemble_mean_year (d_Ensemble['ds_ensemble_mean_year'])
importlib.reload(climxa)
# calculate linear regression stats

# osborn seeing

# input: units (for LATEX table)
units = '[arcsec/decade]'
unit = '[arcsec]'

variable = 'seeing_nc'
significant_digits = 3
ylabel = r'astronomical seeing (osborn) [arcsec]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):

    list_of_single_level_model_clim_params = None
    list_of_model_clim_params = ['seeing']
    ls_pr_levels_clim_model = [5] # not 200 anymore


    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# future trendlines (plots and saves data to csv)
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)

#%% wind speed seeing
# saved as 'single_level'

variable = 'seeing_nc'
significant_digits = 3
ylabel = r'astronomical seeing (wind speed seeing) [arcsec]'

d_future_trends = {}

# write to dict (for every site) the data to plot
for idx in range(0, 8):

    list_of_single_level_model_clim_params = ['wind_speed_seeing']
    list_of_model_clim_params = None
    ls_pr_levels_clim_model = None


    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    d_future_trends = climxa.fill_d_future_trends(d_future_trends, 
                                        list_of_model_clim_params, 
                                        ls_pr_levels_clim_model, 
                                        list_of_single_level_model_clim_params,
                                            site_name_folder)
# future trendlines (plots and saves data to csv)
climxa.future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel)
# saves monthly_timeseries to csv (for analysing past and future trends)
climxa.ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable)

climxa.ds_monthly_ERA5_to_csv(variable)


# %%
# also calculate past trends --> rename script, rewrite function, save future and historical simulations

# in the end, combine past and future trends in one plot!

