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
os.getcwd()
# open NETCDF files, in folder Paranal, data is stored for all sites in Chile
ds_T_700 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/700hPa/*.nc', combine = 'by_coords')
ds_T_800 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/800hPa/*.nc', combine = 'by_coords')
ds_T_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/750hPa/*.nc', combine = 'by_coords')
ds_T_775 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/pressure_levels/775hPa/*.nc', combine = 'by_coords')

# 2m temperature
ds_T_2m = xr.open_mfdataset('./sites/Paranal/Data/Era_5/T/single_levels/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
T_hourly = pd.read_csv('./sites/cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_T_RH_P.csv')

# as a comparison, load in data from Cerro Pachon (same gridpoint)
# T 50m
T_hourly_Pachon = pd.read_csv('./sites/cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_T_50m.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# change 't', 'r', 'q',..
df_T_700 = netcdf_to_df(ds_T_700,-70.81, -30.17)
df_prep_T_700 = df_prep(df_T_700, 't', '700hPa')-273.15 # convert Kelvin to Celsius

df_T_800 = netcdf_to_df(ds_T_800,-70.81, -30.17)
df_prep_T_800 = df_prep(df_T_800, 't', '800hPa')-273.15 # convert Kelvin to Celsius

df_T_750 = netcdf_to_df(ds_T_750, -70.81, -30.17)
df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')-273.15 # convert Kelvin to Celsius

df_T_775 = netcdf_to_df(ds_T_775, -70.81, -30.17)
df_prep_T_775 = df_prep(df_T_775, 't', '775hPa')-273.15 # convert Kelvin to Celsius

# ds_T_2m
df_T_2m = netcdf_to_df(ds_T_2m, -70.81, -30.17)
df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

print('netcdf to df done')

#%% prepare Cerro_Tololo data
# do not shift (timezone = None)
T_hourly_preped = mes_prep(T_hourly) # attention, data is not filtered!
T_hourly_Pachon_preped = mes_prep(T_hourly_Pachon)

T_hourly_Pachon_preped = T_hourly_Pachon_preped.rename(columns={'Temperature': 'Cerro Pachon Temperature'})
#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(T_hourly_preped,T_hourly_Pachon_preped['Cerro Pachon Temperature'],
df_prep_T_700, df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_T_era5, monthly_grouped_T_era5, yearly_grouped_T_era5  = merge_df_long(df_prep_T_700, df_prep_T_800, df_prep_T_750, df_prep_T_775, df_prep_T_2m)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle Cerro_Tololo', 'Cerro Tololo Temperature', 
           './sites/Cerro_Tololo/Output/Plots/T/diurnal_cycle_UTC_T_Cerro_Tololo_and_Pachon_2002to2019.pdf', 
            '700hPa', '750hPa', '775hPa', '800hPa', 't2m', insitu_parameter_2 = 'Cerro Pachon Temperature')#, insitu_parameter_3 = 'Cerro_Tololo T ground')
           

# %% seasonal cycle
plot_cycle(seasonal_T,'seasonal cycle Cerro_Tololo', 'Cerro Tololo Temperature', 
           './sites/Cerro_Tololo/Output/Plots/T/seasonal_cycle_UTC_T_Cerro_Tololo_and_Pachon_2002to2019.pdf', 
           '700hPa', '750hPa', '775hPa','800hPa','t2m', insitu_parameter_2 = 'Cerro Pachon Temperature') #, insitu_parameter_2 = 'Cerro_Tololo T 30m', insitu_parameter_3 = 'Cerro_Tololo T ground')

# %%
#plot_timeseries_merged('./sites/Cerro_Tololo/Output/Plots/T/Timeseries_UTC_T_all_ 2002to2019.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 
#'relative_humidity', '800hPa', '750hPa', '775hPa')

# %%
# plot timeseries, moving average
# T 2m
plot_timeseries_movav('./sites/Cerro_Tololo/Output/Plots/T/timeseries_Cerro_Tololo_and_Pachon_T_2002to2018_movav.pdf', yearly_grouped_T.loc[:'2018-12-31'],
number_of_insitu_params = 2,
 In_situ_Tololo = 'Cerro Tololo Temperature' , In_situ_Pachon = 'Cerro Pachon Temperature', #, In_situ_ground = 'Cerro_Tololo T ground',
  Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa', Era5_T2m = 't2m')

#%%
# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Cerro_Tololo/Output/Plots/T/timeseries_Cerro_Tololo_and_Pachon_T_2002to2018_long.pdf', yearly_grouped_T.loc[:'2018-12-31'],
['Cerro Tololo Temperature', 'Cerro Pachon Temperature'], yearly_grouped_T_era5.loc[:'2018-12-31'], moving = True,
 Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa',Era5_800hPa = '800hPa', Era5_T2m = 't2m')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '800hPa'
insitu_param = 'Cerro Tololo Temperature'
path = './sites/Cerro_Tololo/Output/Plots/T/correlation_T_Cerro_Tololo_' + parameter +'_2002to2019'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)

# %%
parameter = '775hPa'
insitu_param = 'Cerro Tololo Temperature'
path = './sites/Cerro_Tololo/Output/Plots/T/correlation_T_Cerro_Tololo_' + parameter +'_2002to2019'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)


# %% Tololo vs. Pachon
param1 = 'Cerro Pachon Temperature'
param2 = 'Cerro Tololo Temperature'
path = './sites/Cerro_Tololo/Output/Plots/T/correlation_T_Cerro_Tololo_and_Pachon_2002to2019'
xax = 'temperature (°C) ' + parameter
yax = 'temperature (°C) ' + insitu_param

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, param1, param2, xax, yax)


# %%
