# this code reads in Era-5 and in-situ measurement data and plots the diurnal and the seasonal cycle and a long timeseries

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
# open NETCDF files
ds_RH_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
ds_RH_775 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/775hPa/*.nc', combine = 'by_coords')
ds_RH_800 =xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/800hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
#RH_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P.csv')
RH_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P_1994to2020.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_RH_750 = netcdf_to_df(ds_RH_750, -70.74, -29.26)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

df_RH_775 = netcdf_to_df(ds_RH_775, -70.74, -29.26)
df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

df_RH_800 = netcdf_to_df(ds_RH_800, -70.74, -29.26)
df_prep_RH_800 = df_prep(df_RH_800, 'r', '800hPa')

print('netcdf to df done')


#%% prepare La_Silla calculated data

RH_hourly_preped = mes_prep(RH_hourly)

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped,
df_prep_RH_750, df_prep_RH_775, df_prep_RH_800)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_750, df_prep_RH_775, df_prep_RH_800)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Cerro de la Silla', 'La_Silla RH 2m', 
           './sites/La_Silla/Output/Plots/RH/diurnal_cycle_UTC_RH_La_Silla_1994to2020.pdf', 
           '750hPa', '775hPa', '800hPa')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle Cerro de la Silla', 'La_Silla RH 2m', 
           './sites/La_Silla/Output/Plots/RH/seasonal_cycle_UTC_RH_La_Silla_1994to2020.pdf',
           '750hPa', '775hPa', '800hPa')

# %%
#plot_timeseries_merged('./sites/La_Silla/Output/Plots/RH/Timeseries_UTC_RH_all_1994to2020.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '750hPa', '775hPa', '800hPa')


# %%
# plot timeseries, moving average

# 

plot_timeseries_movav('./sites/La_Silla/Output/Plots/RH/timeseries_La_Silla_RH_2m_1994to2020_movav.pdf', yearly_grouped_RH,
 In_situ_2m = 'La_Silla RH 2m' , Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Silla/Output/Plots/RH/timeseries_La_Silla_RH_2m_1994to2020_long.pdf', yearly_grouped_RH,
'La_Silla RH 2m', yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = True, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

pressure_level = '775hPa'
insitu_param = 'La_Silla RH 2m'
path = './sites/La_Silla/Output/Plots/RH/correlation_RH_La_Silla_2m_' + pressure_level +'_1994to2020_'
xax = 'relative humidity Era 5 ' + pressure_level +' pressure level [%]'
yax = 'relative humidity in-situ 2m [%]'
#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_RH[[pressure_level, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + 'hourly.pdf',
'hourly means',
merged_df_ESO[pressure_level], xax,
merged_df_ESO[insitu_param], yax)

# monthly
monthly_corr = monthly_grouped_RH[[pressure_level, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + 'monthly.pdf',
'monthly means',
monthly_corr[pressure_level]['mean'], 'relative humidity Era 5 ' +  pressure_level +' pressure level [kg/kg]',
monthly_corr[insitu_param]['mean'], 'relative humidity in-situ [%]')

# yearly
correlation_plot(path + 'yearly.pdf',
'yearly means',
yearly_grouped_RH[pressure_level]['mean'], xax,
yearly_grouped_RH[insitu_param]['mean'], yax)


# %%
