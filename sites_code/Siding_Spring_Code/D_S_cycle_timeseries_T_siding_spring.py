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
# open NETCDF files, in folder siding_spring, data is stored for all sites in Chile
ds_T_825 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/T/pressure_levels/825hPa/*.nc', combine = 'by_coords')
ds_T_850 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/T/pressure_levels/850hPa/*.nc', combine = 'by_coords')
ds_T_875 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/T/pressure_levels/875hPa/*.nc', combine = 'by_coords')
ds_T_900 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/T/pressure_levels/900hPa/*.nc', combine = 'by_coords')
# 2m temperature
ds_T_2m = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/T/single_levels/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
T_hourly = pd.read_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/hourly_siding_spring_T_RH_P_time.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# change 't', 'r', 'q',..

df_T_825 = netcdf_to_df(ds_T_825, 149.07, -31.28)
df_prep_T_825 = df_prep(df_T_825, 't', '825hPa')-273.15 # convert Kelvin to Celsius

df_T_850 = netcdf_to_df(ds_T_850, 149.07, -31.28)
df_prep_T_850 = df_prep(df_T_850, 't', '850hPa')-273.15 # convert Kelvin to Celsius

df_T_875 = netcdf_to_df(ds_T_875, 149.07, -31.28)
df_prep_T_875 = df_prep(df_T_875, 't', '875hPa')-273.15 # convert Kelvin to Celsius

df_T_900 = netcdf_to_df(ds_T_900, 149.07, -31.28)
df_prep_T_900 = df_prep(df_T_900, 't', '900hPa')-273.15 # convert Kelvin to Celsius


# ds_T_2m
df_T_2m = netcdf_to_df(ds_T_2m, 149.07, -31.28)
df_prep_T_2m = df_prep(df_T_2m, 't2m', 't2m')-273.15 # convert Kelvin to Celsius

print('netcdf to df done')

#%% prepare siding_spring data
# do not shift (timezone = None)
T_hourly_preped = mes_prep(T_hourly, parameter = 'siding spring Temperature')

#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(T_hourly_preped,
df_prep_T_825, df_prep_T_850, df_prep_T_875,  df_prep_T_900, df_prep_T_2m, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_T_era5, monthly_grouped_T_era5, yearly_grouped_T_era5  = merge_df_long(df_prep_T_825, df_prep_T_850, df_prep_T_875, df_prep_T_900,  df_prep_T_2m)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle siding spring', 'siding spring Temperature', 
           './sites/siding_spring/Output/Plots/T/diurnal_cycle_UTC_T_siding_spring_2003to2020.pdf', 
           '825hPa', '850hPa', '875hPa','900hPa', 't2m') #, insitu_parameter_2 = 'siding_spring T 30m', insitu_parameter_3 = 'siding_spring T ground')
           

# %% seasonal cycle
plot_cycle(seasonal_T,'seasonal cycle siding spring', 'siding spring Temperature', 
           './sites/siding_spring/Output/Plots/T/seasonal_cycle_UTC_T_siding_spring_2003to2020.pdf', 
           '825hPa', '850hPa', '875hPa','900hPa', 't2m') #, insitu_parameter_2 = 'siding_spring T 30m', insitu_parameter_3 = 'siding_spring T ground')

# %%
#plot_timeseries_merged('./sites/siding_spring/Output/Plots/T/Timeseries_UTC_T_all_ 2003to2020.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 
#'relative_humidity', '825hPa', '850hPa', '875hPa')

# %%
# plot timeseries, moving average
yearly_grouped_T = yearly_grouped_T.loc['2003-12-31':'2019-12-31']

# T 2m
plot_timeseries_movav('./sites/siding_spring/Output/Plots/T/timeseries_siding_spring_T_2003to2020_movav.pdf', yearly_grouped_T,
number_of_insitu_params=1,
 In_situ_av = 'siding spring Temperature', # , In_situ_30m = 'siding_spring T 30m', In_situ_ground = 'siding_spring T ground',
  Era5_825hPa = '825hPa', Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa', Era5_T2m = 't2m')

# use yearly_grouped df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/siding_spring/Output/Plots/T/timeseries_siding_spring_T_2003to2020_long_movav.pdf', yearly_grouped_T,
['siding spring Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = True, Era5_825hPa = '825hPa',
 Era5_850hPa = '850hPa', Era5_875hPa = '875hPa',Era5_900hPa = '900hPa',  Era5_T2m = 't2m')

plot_timeseries_long('./sites/siding_spring/Output/Plots/T/timeseries_siding_spring_T_2003to2020_long.pdf', yearly_grouped_T,
['siding spring Temperature'], yearly_grouped_T_era5.loc[:'2019-12-31'], moving = False, Era5_825hPa = '825hPa',
 Era5_850hPa = '850hPa', Era5_875hPa = '875hPa',Era5_900hPa = '900hPa',  Era5_T2m = 't2m')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '900hPa'
insitu_param = 'siding spring Temperature'
path = './sites/siding_spring/Output/Plots/T/correlation_T_siding_spring_' + parameter +'_2003to2020'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ av'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)

# %%
parameter = 't2m'
insitu_param = 'siding spring Temperature'
path = './sites/siding_spring/Output/Plots/T/correlation_T_siding_spring_' + parameter +'_2003to2020'
xax = 'temperature (°C) Era 5 ' + parameter +' pressure level'
yax = 'temperature (°C) in-situ'

corr_plots_hourly_monthly_yearly(path, merged_df_T, monthly_grouped_T, yearly_grouped_T, parameter, insitu_param, xax, yax)


# %%
