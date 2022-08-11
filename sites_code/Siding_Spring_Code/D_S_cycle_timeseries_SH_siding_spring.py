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
# os.getcwd()
# open NETCDF files
ds_SH_875 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/SH/875hPa/*.nc', combine = 'by_coords')
ds_SH_850 = xr.open_mfdataset('./sites/siding_spring/Data/Era_5/SH/850hPa/*.nc', combine = 'by_coords')
ds_SH_900 =xr.open_mfdataset('./sites/siding_spring/Data/Era_5/SH/900hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
SH_hourly = pd.read_csv('./sites/siding_spring/Data/in-situ/Specific_humidity_siding_spring_2003to2020.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_SH_875 = netcdf_to_df(ds_SH_875, -70.74, -29.26)
df_prep_SH_875 = df_prep(df_SH_875, 'q', '875hPa')

df_SH_850 = netcdf_to_df(ds_SH_850, -70.74, -29.26)
df_prep_SH_850 = df_prep(df_SH_850, 'q', '850hPa')

df_SH_900 = netcdf_to_df(ds_SH_900, -70.74, -29.26)
df_prep_SH_900 = df_prep(df_SH_900, 'q', '900hPa')

print('netcdf to df done')

#%% prepare siding_spring calculated data

# do not shift specific humidity siding_spring array by 10hours, it is already in UTC! only add month and hours
SH_hourly['time'] = pd.to_datetime(SH_hourly['time']) 

#check the format
print(SH_hourly['time'].dtype)
    
#set index 
SH_hourly.set_index('time', inplace=True)

# create a new column consisting of the cycle parameter of the correspondend entry
#for seasonal cycle (12 months), create column with "months"
SH_hourly['months'] = pd.DatetimeIndex(SH_hourly.index).month                                            

#for diurnal cycle (24 hours), create column with "hours"
SH_hourly['hours'] = pd.DatetimeIndex(SH_hourly.index).hour

# for monthly timeseries, create column with "YYYY-MM"
SH_hourly['YYYY_MM'] = SH_hourly.index.strftime('%Y-%m')

# for 1 year averages, create column with years
SH_hourly['years'] = pd.DatetimeIndex(SH_hourly.index).year

SH_hourly_preped = SH_hourly

#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(SH_hourly_preped,
df_prep_SH_875, df_prep_SH_850, df_prep_SH_900, dropnan = False)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_875, df_prep_SH_850, df_prep_SH_900)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Cerro siding_spring', 'specific_humidity', 
           './sites/siding_spring/Output/Plots/SH/diurnal_cycle_UTC_SH_siding_spring_2003to2019.pdf', 
           '850hPa', '875hPa', '900hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Cerro siding_spring', 'specific_humidity', 
           './sites/siding_spring/Output/Plots/SH/seasonal_cycle_UTC_SH_siding_spring_2003to2019.pdf',
           '850hPa', '875hPa', '900hPa')

# %%
#plot_timeseries_merged('./sites/siding_spring/Output/Plots/SH/Timeseries_UTC_SH_all_2003to2019.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 
#'specific_humidity', '875hPa', '850hPa', '900hPa')

# %%
# plot timeseries, moving average
yearly_grouped_SH = yearly_grouped_SH.loc['2003-12-31':'2019-12-31']
# # function assumes that specific humidity is in first column
# SH_hourly_preped.iloc[:,0] = SH_hourly_preped.iloc[:,6]
# SH_hourly_preped.rename(columns={'Unnamed: 0': 'specific_humidity'})

plot_timeseries_movav('./sites/siding_spring/Output/Plots/SH/timeseries_siding_spring_2003to2019_movav.pdf', yearly_grouped_SH.loc[:'2019-12-31'],
 number_of_insitu_params=1,
 In_situ = 'specific_humidity',  Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/siding_spring/Output/Plots/SH/timeseries_siding_spring_1979to2019_long.pdf', yearly_grouped_SH.loc[:'2019-12-31'],
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = True, Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa')

# %% correlation

parameter = '900hPa'
insitu_param = 'specific_humidity'
path = './sites/siding_spring/Output/Plots/SH/correlation_SH_siding_spring_' + parameter +'_2003to2019'
xax = 'specific humidity Era 5 ' + parameter +' pressure level [kg/kg]'
yax = 'specific humidity in-situ [kg/kg]'

corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)

# %%
