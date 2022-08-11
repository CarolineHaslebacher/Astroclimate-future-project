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
#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
os.getcwd()
# open NETCDF files on 600hPa to 750hPa
ds_SH_775 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/775hPa/*.nc', combine = 'by_coords')
ds_SH_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')
ds_SH_800 =xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/800hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
# T 2m
#SH_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_La_Silla_ESO_2000to2019.csv')
# # T 30m
SH_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_30m_La_Silla_ESO_2000to2019.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!


df_SH_775 = netcdf_to_df(ds_SH_775, -70.74, -29.26)
df_prep_SH_775 = df_prep(df_SH_775, 'q', '775hPa')

df_SH_750 = netcdf_to_df(ds_SH_750, -70.74, -29.26)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

df_SH_800 = netcdf_to_df(ds_SH_800, -70.74, -29.26)
df_prep_SH_800 = df_prep(df_SH_800, 'q', '800hPa')

print('netcdf to df done')

#%% prepare La_Silla calculated data

# do not shift specific humidity La_Silla array by 10hours, it is already in UTC! only add month and hours
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
df_prep_SH_775, df_prep_SH_750, df_prep_SH_800, dropnan = False)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_775, df_prep_SH_750, df_prep_SH_800)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Cerro La_Silla', 'specific_humidity', 
           './sites/La_Silla/Output/Plots/SH/diurnal_cycle_UTC_SH_La_Silla_T30m_2000to2019.pdf', 
           '750hPa', '775hPa', '800hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Cerro La_Silla', 'specific_humidity', 
           './sites/La_Silla/Output/Plots/SH/seasonal_cycle_UTC_SH_La_Silla_T30m_2000to2019.pdf',
           '750hPa', '775hPa', '800hPa')

# %%
#plot_timeseries_merged('./sites/La_Silla/Output/Plots/SH/Timeseries_UTC_SH_all_2000to2019.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 
#'specific_humidity', '775hPa', '750hPa', '800hPa')

# %%
# plot timeseries, moving average

# # function assumes that specific humidity is in first column
# SH_hourly_preped.iloc[:,0] = SH_hourly_preped.iloc[:,6]
# SH_hourly_preped.rename(columns={'Unnamed: 0': 'specific_humidity'})

plot_timeseries_movav('./sites/La_Silla/Output/Plots/SH/timeseries_La_Silla_T30m_2000to2019_movav.pdf', yearly_grouped_SH,
 In_situ = 'specific_humidity',  Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Silla/Output/Plots/SH/timeseries_La_Silla_T30m_1979to2019_long.pdf', yearly_grouped_SH,
'specific_humidity', yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = True, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %% correlation

pressure_level = '775hPa'
insitu_param = 'specific_humidity'
path = './sites/La_Silla/Output/Plots/SH/correlation_SH_La_Silla_T30m_' + pressure_level +'_2000to2019'
xax = 'specific humidity Era 5 ' + pressure_level +' pressure level [kg/kg]'
yax = 'specific humidity in-situ [kg/kg]'
#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_SH[[pressure_level, 'specific_humidity']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_hourly.pdf',
'hourly means',
merged_df_ESO[pressure_level], xax,
merged_df_ESO[insitu_param], yax)

# monthly
monthly_corr = monthly_grouped_SH[[pressure_level, 'specific_humidity']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_monthly.pdf',
'monthly means',
monthly_corr[pressure_level]['mean'], xax,
monthly_corr[insitu_param]['mean'], yax)

# yearly
correlation_plot(path + '_yearly.pdf',
'yearly means',
yearly_grouped_SH[pressure_level]['mean'], xax,
yearly_grouped_SH['specific_humidity']['mean'], yax)

# %%
