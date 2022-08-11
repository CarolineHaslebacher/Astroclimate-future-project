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
#sys.path.append('/chaldene/Astroclimate_Project/sites/')

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

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
os.getcwd()
# open NETCDF files on 700hPa to 775hPa
#ds_T_650 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/650hPa/*.nc', combine = 'by_coords')
ds_T_700 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/700hPa/*.nc', combine = 'by_coords')
ds_T_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/750hPa/*.nc', combine = 'by_coords')
ds_T_775 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/775hPa/*.nc', combine = 'by_coords')

# 2m temperature
ds_T_2m = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/single_levels/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
T_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_RH_T_P.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# change 't', 'r', 'q',..

# not nearest,  but 'green' point, higher elevated
df_T_700 = netcdf_to_df(ds_T_700, -70.25, -24.75)
df_prep_T_700 = df_prep(df_T_700, 't', '700hPa')-273.15 # convert Kelvin to Celsius

df_T_750 = netcdf_to_df(ds_T_750, -70.25, -24.75)
df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')-273.15 # convert Kelvin to Celsius

df_T_775 = netcdf_to_df(ds_T_775, -70.25, -24.75)
df_prep_T_775 = df_prep(df_T_775, 't', '775hPa')-273.15 # convert Kelvin to Celsius

# ds_T_2m
df_T_2m = netcdf_to_df(ds_T_2m, -70.25, -24.75)
df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

print('netcdf to df done')

#%% nearest

# not nearest,  but 'green' point, higher elevated
# df_T_700 = netcdf_to_df(ds_T_700, -70.5, -24.75)
# df_prep_T_700 = df_prep(df_T_700, 't', '700hPa')-273.15 # convert Kelvin to Celsius

# df_T_750 = netcdf_to_df(ds_T_750, -70.5, -24.75)
# df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')-273.15 # convert Kelvin to Celsius

# df_T_775 = netcdf_to_df(ds_T_775, -70.5, -24.75)
# df_prep_T_775 = df_prep(df_T_775, 't', '775hPa')-273.15 # convert Kelvin to Celsius

# # ds_T_2m
# df_T_2m = netcdf_to_df(ds_T_2m, -70.5, -24.75)
# df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

# print('netcdf to df done')

#%% prepare Paranal data
# do not shift (timezone = None)
T_hourly_preped = mes_prep(T_hourly) # attention, data is not filtered!

#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(T_hourly_preped,
df_prep_T_700, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_T_era5, monthly_grouped_T_era5, yearly_grouped_T_era5  = merge_df_long(df_prep_T_700, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle Cerro Paranal', 'Paranal T 2m', 
           './sites/Paranal/Output/Plots/T/diurnal_cycle_UTC_T_all_Paranal_2000to2019.pdf', 
           '700hPa', '750hPa', '775hPa', 't2m', insitu_parameter_2 = 'Paranal T -20m')
           

# %% seasonal cycle
plot_cycle(seasonal_T,'seasonal cycle Cerro Paranal', 'Paranal T 2m', 
           './sites/Paranal/Output/Plots/T/seasonal_cycle_UTC_T_all_Paranal_2000to2019.pdf', 
           '700hPa', '750hPa', '775hPa', insitu_parameter_2 = 'Paranal T -20m')


# %%
#plot_timeseries_merged('./sites/Paranal/Output/Plots/T/Timeseries_UTC_T_all_2000to2019.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 
#'relative_humidity', '700hPa', '750hPa', '775hPa')

# %%
# plot timeseries, moving average
# T 2m
plot_timeseries_movav('./sites/Paranal/Output/Plots/T/timeseries_Paranal_T_all_2000to2019_movav.pdf', yearly_grouped_T,
 In_situ_2m = 'Paranal T 2m', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_T2m = 't2m',
  in_situ_20m_below = 'Paranal T -20m')

# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/T/timeseries_Paranal_T_-20m_2000to2019_long.pdf', yearly_grouped_T,
'Paranal T -20m', yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa',
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_T2m = 't2m' )

plot_timeseries_long('./sites/Paranal/Output/Plots/T/timeseries_Paranal_T_-20m_2000to2019_long.pdf', yearly_grouped_T,
'Paranal T -20m', yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa',
 Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_T2m = 't2m' )

#%% timeseries
# T -20 m (starts in 2006)

plot_timeseries_movav('./sites/Paranal/Output/Plots/T/timeseries_Paranal_T_2m_2000to2019_movav.pdf', yearly_grouped_T,
 In_situ = 'Paranal T -20m', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_T2m = 't2m')

# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/T/timeseries_Paranal_T_2m_2000to2019_long.pdf', yearly_grouped_T,
'Paranal T -20m', yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_T2m = 't2m')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '750hPa'
insitu_param = 'Paranal T -20m'
path = './sites/Paranal/Output/Plots/T/correlation_T_-20m_Paranal_' + parameter +'_2006to2020'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'
#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_T[[parameter, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_hourly.pdf',
'hourly means',
merged_df_ESO[parameter], xax,
merged_df_ESO[insitu_param], yax)

# monthly
monthly_corr = monthly_grouped_T[[parameter, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_monthly.pdf',
'monthly means',
monthly_corr[parameter]['mean'], xax,
monthly_corr[insitu_param]['mean'], yax)

# yearly
correlation_plot(path + '_yearly.pdf',
'yearly means',
yearly_grouped_T[parameter]['mean'], xax,
yearly_grouped_T[insitu_param]['mean'], yax)


# %%
