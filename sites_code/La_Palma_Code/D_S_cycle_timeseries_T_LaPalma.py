# this code reads in Era-5 and in-situ measurement data and plots the diurnal and the seasonal cycle and a long timeseries

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

#%%
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file
import Astroclimate_function_pool
import importlib
importlib.reload(Astroclimate_function_pool)

from Astroclimate_function_pool import netcdf_to_df
from Astroclimate_function_pool import  mes_prep
from Astroclimate_function_pool import  merge_df 
from Astroclimate_function_pool import  merge_df_long
from Astroclimate_function_pool import  df_prep #(df, parameter, colname)
from Astroclimate_function_pool import  plot_cycle #(cycle_name, cycle_string,  CFHT_parameter, filename, *args)
from Astroclimate_function_pool import  plot_timeseries_merged
from Astroclimate_function_pool import plot_timeseries_long
from Astroclimate_function_pool import plot_timeseries_movav
from Astroclimate_function_pool import correlation_plot
from Astroclimate_function_pool import corr_plots_hourly_monthly_yearly

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

# open NETCDF files
ds_T_800 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/T/pressure_levels/800hPa/*.nc', combine = 'by_coords')
ds_T_750 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/T/pressure_levels/750hPa/*.nc', combine = 'by_coords')
ds_T_775 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/T/pressure_levels/775hPa/*.nc', combine = 'by_coords')

# 2m temperature
ds_T_2m = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/T/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
# T_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/ESO/hourly_meteo/hourly_La_Palma_RH_T_P.csv')
# 1997 to 2020
T_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_RH_T_P.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# change 't', 'r', 'q',..

# nearest
df_T_800 = netcdf_to_df(ds_T_800,-17.88, 28.76)
df_prep_T_800 = df_prep(df_T_800, 't', '800hPa')-273.15 # convert Kelvin to Celsius

df_T_750 = netcdf_to_df(ds_T_750,-17.88, 28.76)
df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')-273.15 # convert Kelvin to Celsius

df_T_775 = netcdf_to_df(ds_T_775,-17.88, 28.76)
df_prep_T_775 = df_prep(df_T_775, 't', '775hPa')-273.15 # convert Kelvin to Celsius

# ds_T_2m
df_T_2m = netcdf_to_df(ds_T_2m,-17.88, 28.76)
df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

print('netcdf to df done')

#%% prepare La_Palma data
# do not shift (timezone = None)
T_hourly_preped = mes_prep(T_hourly) # attention, data is not filtered!

#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(T_hourly_preped,
df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_T_era5, monthly_grouped_T_era5, yearly_grouped_T_era5  = merge_df_long(df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle La_Palma', 'La_Palma Temperature', 
           './sites/La_Palma/Output/Plots/T/diurnal_cycle_UTC_T_La_Palma_1997to2020.pdf', 
           '800hPa', '750hPa', '775hPa', 't2m')
           
# %% seasonal cycle
plot_cycle(seasonal_T,'seasonal cycle La_Palma', 'La_Palma Temperature', 
           './sites/La_Palma/Output/Plots/T/seasonal_cycle_UTC_T_La_Palma_1997to2020.pdf', 
           '800hPa', '750hPa', '775hPa','t2m')

# %%
#plot_timeseries_merged('./sites/La_Palma/Output/Plots/T/Timeseries_UTC_T_all_ 1997to2020.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 
#'relative_humidity', '800hPa', '750hPa', '775hPa')

# %%
# plot timeseries, moving average

# plot only full years
yearly_grouped_T = yearly_grouped_T.loc['1997-12-31':'2019-12-31']

# Temperature
plot_timeseries_movav('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_T_all_1997to2020_movav.pdf', yearly_grouped_T.loc[:2019.0],
number_of_insitu_params=1,
 In_situ_2m = 'La_Palma Temperature',
   Era5_750hPa = '750hPa', Era5_775hPa = '775hPa',Era5_800hPa = '800hPa', Era5_T2m = 't2m')

# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_T_1997to2020_long_movav.pdf', yearly_grouped_T.loc[:2019.0],
['La_Palma Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, 
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa', Era5_T2m = 't2m')

plot_timeseries_long('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_T_1997to2020_long.pdf', yearly_grouped_T.loc[:2019.0],
['La_Palma Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = False,
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa', Era5_T2m = 't2m')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '750hPa'
insitu_param = 'La_Palma Temperature'
path = './sites/La_Palma/Output/Plots/T/correlation_T_La_Palma_' + parameter +'_1997to2020'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)

#%%


# %%
#################################################### middle of island ##################################################
#%%
# not nearest, but higher elevated (middle of island)
df_T_800 = netcdf_to_df(ds_T_800,-17.75, 28.76)
df_prep_T_800 = df_prep(df_T_800, 't', '800hPa')-273.15 # convert Kelvin to Celsius

df_T_750 = netcdf_to_df(ds_T_750,-17.75, 28.76)
df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')-273.15 # convert Kelvin to Celsius

df_T_775 = netcdf_to_df(ds_T_775,-17.75, 28.76)
df_prep_T_775 = df_prep(df_T_775, 't', '775hPa')-273.15 # convert Kelvin to Celsius

# ds_T_2m
df_T_2m = netcdf_to_df(ds_T_2m,-17.75, 28.76)
df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

print('netcdf to df done')

#%%

T_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_RH_T_P.csv')

#%% prepare La_Palma data
# do not shift (timezone = None)
T_hourly_preped = mes_prep(T_hourly) # attention, data is not filtered!

#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(T_hourly_preped,
df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_T_era5, monthly_grouped_T_era5, yearly_grouped_T_era5  = merge_df_long(df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle La_Palma', 'La_Palma Temperature', 
           './sites/La_Palma/Output/Plots/T/diurnal_cycle_UTC_T_La_Palma_middle_1997to2020.pdf', 
           '800hPa', '750hPa', '775hPa', 't2m')
           
# %% seasonal cycle
plot_cycle(seasonal_T,'seasonal cycle La_Palma', 'La_Palma Temperature', 
           './sites/La_Palma/Output/Plots/T/seasonal_cycle_UTC_T_La_Palma_middle_1997to2020.pdf', 
           '800hPa', '750hPa', '775hPa','t2m')

# %%
#plot_timeseries_merged('./sites/La_Palma/Output/Plots/T/Timeseries_UTC_T_all_ 1997to2020.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 
#'relative_humidity', '800hPa', '750hPa', '775hPa')

# %%
# plot timeseries, moving average

# plot only full years
yearly_grouped_T = yearly_grouped_T.loc['1997-12-31':'2019-12-31']

# Temperature
plot_timeseries_movav('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_middle_T_all_1997to2020_movav.pdf', yearly_grouped_T.loc[:2019.0],
number_of_insitu_params=1,
 In_situ_2m = 'La_Palma Temperature',
   Era5_750hPa = '750hPa', Era5_775hPa = '775hPa',Era5_800hPa = '800hPa', Era5_T2m = 't2m')

# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_middle_T_1997to2020_long_movav.pdf', yearly_grouped_T.loc[:2019.0],
['La_Palma Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, 
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa', Era5_T2m = 't2m')

plot_timeseries_long('./sites/La_Palma/Output/Plots/T/timeseries_La_Palma_middle_T_1997to2020_long.pdf', yearly_grouped_T.loc[:2019.0],
['La_Palma Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = False,
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa', Era5_T2m = 't2m')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '750hPa'
insitu_param = 'La_Palma Temperature'
path = './sites/La_Palma/Output/Plots/T/correlation_T_La_Palma_middle_' + parameter +'_1997to2020'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)

# %%
