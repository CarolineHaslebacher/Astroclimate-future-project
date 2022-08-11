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
#SH_hourly = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_Cerro_Tololo_ESO_2002to2019.csv')
# # T 30m
SH_hourly = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/Specific_humidity_Cerro_Tololo_2002to2019.csv')
#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!


df_SH_775 = netcdf_to_df(ds_SH_775,-70.81, -30.17)
df_prep_SH_775 = df_prep(df_SH_775, 'q', '775hPa')

df_SH_750 = netcdf_to_df(ds_SH_750,-70.81, -30.17)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

df_SH_800 = netcdf_to_df(ds_SH_800,-70.81, -30.17)
df_prep_SH_800 = df_prep(df_SH_800, 'q', '800hPa')

print('netcdf to df done')

#%% prepare Cerro_Tololo calculated data

SH_hourly_preped = mes_prep(SH_hourly)

#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(SH_hourly_preped,
df_prep_SH_775, df_prep_SH_750, df_prep_SH_800, dropnan = False)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_775, df_prep_SH_750, df_prep_SH_800)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Cerro Cerro_Tololo', 'specific_humidity', 
           './sites/Cerro_Tololo/Output/Plots/SH/diurnal_cycle_UTC_SH_Cerro_Tololo_2002to2019.pdf', 
           '750hPa', '775hPa', '800hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Cerro Cerro_Tololo', 'specific_humidity', 
           './sites/Cerro_Tololo/Output/Plots/SH/seasonal_cycle_UTC_SH_Cerro_Tololo_2002to2019.pdf',
           '750hPa', '775hPa', '800hPa')

# %%
# plot timeseries, moving average
plot_timeseries_movav('./sites/Cerro_Tololo/Output/Plots/SH/timeseries_Cerro_Tololo_2002to2019_movav.pdf', yearly_grouped_SH.loc[:'2018-12-31'],
number_of_insitu_params=1,
 In_situ = 'specific_humidity',  Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Cerro_Tololo/Output/Plots/SH/timeseries_Cerro_Tololo_1979to2019_long.pdf', yearly_grouped_SH.loc[:'2018-12-31'],
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2018-12-31'], moving = False, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

plot_timeseries_long('./sites/Cerro_Tololo/Output/Plots/SH/timeseries_Cerro_Tololo_1979to2019_long_movav.pdf', yearly_grouped_SH.loc[:'2018-12-31'],
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2018-12-31'], moving = True, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %% correlation

parameter = '775hPa'
insitu_param = 'specific_humidity'
path = './sites/Cerro_Tololo/Output/Plots/SH/correlation_SH_Cerro_Tololo_' + parameter +'_2002to2019'
xax = 'specific humidity Era 5 ' + parameter +' pressure level [kg/kg]'
yax = 'specific humidity in-situ [kg/kg]'

corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)

# %%
parameter = '750hPa'
insitu_param = 'specific_humidity'
path = './sites/Cerro_Tololo/Output/Plots/SH/correlation_SH_Cerro_Tololo_' + parameter +'_2002to2019'
xax = 'specific humidity Era 5 ' + parameter +' pressure level [kg/kg]'
yax = 'specific humidity in-situ [kg/kg]'

corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)


# %%
