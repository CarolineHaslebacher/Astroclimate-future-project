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
# open NETCDF file
ds_surf_Pr = xr.open_mfdataset('./sites/Paranal/Data/Era_5/surface_pressure/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
P_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_RH_T_P.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# not nearest,  but 'green' point, higher elevated
df_surf_Pr = netcdf_to_df(ds_surf_Pr, -70.25, -24.75)
df_prep_surf_Pr = df_prep(df_surf_Pr, 'sp', 'sp')/100 # convert to hPa

print('netcdf to df done')

#%%

# nearest
# df_surf_Pr = netcdf_to_df(ds_surf_Pr, -70.5, -24.75)
# df_prep_surf_Pr = df_prep(df_surf_Pr, 'sp', 'sp')/100 # convert to hPa

# print('netcdf to df done')

#%% prepare Paranal calculated data

P_hourly_preped = mes_prep(P_hourly) # not filtered!

#%%
# merge datasets
merged_df_P, seasonal_P, diurnal_P, monthly_grouped_P, yearly_grouped_P  = merge_df(P_hourly_preped, df_prep_surf_Pr)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_P_era5, monthly_grouped_P_era5, yearly_grouped_P_era5  = merge_df_long(df_prep_surf_Pr)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_P,'diurnal cycle Cerro Paranal', 'Paranal Pressure', 
           './sites/Paranal/Output/Plots/P/diurnal_cycle_UTC_P_Paranal_2000to2019.pdf', 
           'sp')
           
# ./sites/MaunaKea/Output/Plots/P/
# %%
plot_cycle(seasonal_P,'seasonal cycle Cerro Paranal', 'Paranal Pressure', 
           './sites/Paranal/Output/Plots/P/seasonal_cycle_UTC_P_Paranal_2000to2019.pdf',
           'sp')


# %%
#plot_timeseries_merged('./sites/Paranal/Output/Plots/P/Timeseries_UTC_P_all_2000to2019.pdf', merged_df_P, monthly_grouped_P, yearly_grouped_P, 
#'relative_humidity', '700hPa', '750hPa', '800hPa')

# %%
# plot timeseries, moving average

# # function assumes that relative humidity is in first column
# P_hourly_preped.iloc[:,0] = P_hourly_preped.iloc[:,6]
# P_hourly_preped.rename(columns={'Unnamed: 0': 'relative_humidity'})

plot_timeseries_movav('./sites/Paranal/Output/Plots/P/timeseries_Paranal_P_2000to2019_movav.pdf', yearly_grouped_P,
df_merged = merged_df_P,
 In_situ = 'Paranal Pressure', Era5_surface_pressure = 'sp')


# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/P/timeseries_Paranal_P_2000to2019_long.pdf', yearly_grouped_P,
'Paranal Pressure', yearly_grouped_P_era5.loc[:'2019-12-31'], moving = False, Era5_surface_pressure = 'sp')

# %%
# correlation plot: alpha = 0.05
# hourly

pressure_level = 'sp'
#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_P[[pressure_level, 'Paranal Pressure']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot('./sites/Paranal/Output/Plots/P/correlation_P_Paranal_' + pressure_level +'_2000to2020_hourly.pdf',
'hourly means',
merged_df_ESO[pressure_level], 'surface pressure Era 5 ' + pressure_level + '[hPa]',
merged_df_ESO['Paranal Pressure'], 'pressure in-situ [hPa]')

# monthly
monthly_corr = monthly_grouped_P[[pressure_level, 'Paranal Pressure']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot('./sites/Paranal/Output/Plots/P/correlation_P_Paranal_2000to2020_monthly.pdf',
'monthly means',
monthly_corr[pressure_level]['mean'], 'surface pressure Era 5 ' + pressure_level + '[hPa]',
monthly_corr['Paranal Pressure']['mean'], 'pressure in-situ [hPa]')

# yearly
correlation_plot('./sites/Paranal/Output/Plots/P/correlation_P_Paranal__2000to2020_yearly.pdf',
'yearly means',
yearly_grouped_P[pressure_level]['mean'], 'surface pressure Era 5 ' + pressure_level + '[hPa]',
yearly_grouped_P['Paranal Pressure']['mean'], 'pressure in-situ [hPa]')


# %%
