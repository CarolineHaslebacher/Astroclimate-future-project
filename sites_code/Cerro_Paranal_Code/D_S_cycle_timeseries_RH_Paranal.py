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
# open NETCDF files on 600hPa to 750hPa
ds_RH_700 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/700hPa/*.nc', combine = 'by_coords')
ds_RH_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
ds_RH_775 =xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/775hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
RH_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_RH_T_P.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# not nearest,  but 'green' point, higher elevated
df_RH_700 = netcdf_to_df(ds_RH_700, -70.25, -24.75)
df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

df_RH_750 = netcdf_to_df(ds_RH_750, -70.25, -24.75)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

df_RH_775 = netcdf_to_df(ds_RH_775, -70.25, -24.75)
df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

print('netcdf to df done')

#%%
# # nearest
# df_RH_700 = netcdf_to_df(ds_RH_700, -70.5, -24.75)
# df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

# df_RH_750 = netcdf_to_df(ds_RH_750, -70.5, -24.75)
# df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

# df_RH_775 = netcdf_to_df(ds_RH_775, -70.5, -24.75)
# df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

# print('netcdf to df done')

#%% interpolation of 4 surrounding gridpoints

# def four_points_mean(ds, parameter, pressure_level):
#     lon = -70.5
#     lat = -24.75
#     ds_SW = ds.sel(longitude=lon, latitude= lat, method='nearest')

#     lon = -70.25
#     lat = -24.75
#     ds_SE = ds.sel(longitude=lon, latitude= lat, method='nearest')

#     lon = -70.5
#     lat = -24.5
#     ds_NW = ds.sel(longitude=lon, latitude= lat, method='nearest')

#     lon = -70.25
#     lat = -24.5
#     ds_NE = ds.sel(longitude=lon, latitude= lat, method='nearest')

#     # take the mean of the 4 points
#     ds_4p = xr.Dataset({"r": 1/4 * (ds_SW.r + ds_SE.r + ds_NW.r + ds_NE.r)})

#     df_4p = ds_4p.to_dask_dataframe()

#     # # from df_prep()
#     # if 'latitude' in df.columns:
#     #     df_4p= df_4p.drop('latitude', axis=1)
#     # df_4p = df_4p.set_index('time')
#     # df_4p = df_4p.rename(columns={'r': '700hPa_4p'})#colname = 'r_700'
#     # df_comp= df_4p.compute()

#     df_prep_4p = df_prep(df_4p, parameter, pressure_level)

#     return df_prep_4p

# df_prep_RH_700 = four_points_mean(ds_RH_700, 'r', '700hPa')
# df_prep_RH_750 = four_points_mean(ds_RH_750, 'r', '750hPa')
# df_prep_RH_775 = four_points_mean(ds_RH_775, 'r', '775hPa')
#%% prepare Paranal calculated data

RH_hourly_preped = mes_prep(RH_hourly)

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped,
df_prep_RH_700, df_prep_RH_750, df_prep_RH_775)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_700, df_prep_RH_750, df_prep_RH_775)
# nearest
# df_RH_700 = netcdf_to_df(ds_RH_700, -70.5, -24.75)
# df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

# df_RH_750 = netcdf_to_df(ds_RH_750, -70.5, -24.75)
# df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

# df_RH_775 = netcdf_to_df(ds_RH_775, -70.5, -24.75)
# df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')
# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Cerro Paranal', 'Paranal RH 2m', 
           './sites/Paranal/Output/Plots/RH/diurnal_cycle_UTC_RH_greenpoint_Paranal_2000to2019.pdf', 
           '700hPa', '750hPa', '775hPa', insitu_parameter_2 = 'Paranal RH -20m')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle Cerro Paranal', 'Paranal RH 2m', 
           './sites/Paranal/Output/Plots/RH/seasonal_cycle_UTC_RH_greenpoint_Paranal_2000to2019.pdf',
           '700hPa', '750hPa', '775hPa' ,  insitu_parameter_2 = 'Paranal RH -20m')

# %%
#plot_timeseries_merged('./sites/Paranal/Output/Plots/RH/Timeseries_UTC_RH_all_2000to2019.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '700hPa', '750hPa', '775hPa')


# %%
# plot timeseries, moving average

# # function assumes that relative humidity is in first column
# RH_hourly_preped.iloc[:,0] = RH_hourly_preped.iloc[:,6]
# RH_hourly_preped.rename(columns={'Unnamed: 0': 'relative_humidity'})

plot_timeseries_movav('./sites/Paranal/Output/Plots/RH/timeseries_Paranal_RH_2m_greenpoint_2000to2019_movav.pdf', yearly_grouped_RH,
 In_situ_2m = 'Paranal RH 2m' , Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/RH/timeseries_Paranal_RH_2m_greenpoint_2000to2019_long.pdf', yearly_grouped_RH,
['Paranal RH 2m'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

pressure_level = '750hPa'
insitu_param = 'Paranal RH 2m'
path = './sites/Paranal/Output/Plots/RH/correlation_RH_Paranal_2m_greenpoint_' + pressure_level +'_2000to2020_'
xax = 'relative humidity Era 5' + pressure_level +' pressure level [%]'
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

correlation_plot(path + '_monthly.pdf',
'monthly means',
monthly_corr[pressure_level]['mean'], 'relative humidity Era 5 ' +  pressure_level +' pressure level [kg/kg]',
monthly_corr[insitu_param]['mean'], 'relative humidity in-situ [%]')

# yearly
correlation_plot(path + '_yearly.pdf',
'yearly means',
yearly_grouped_RH[pressure_level]['mean'], xax,
yearly_grouped_RH[insitu_param]['mean'], yax)


# %%
