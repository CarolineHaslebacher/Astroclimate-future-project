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
from Astroclimate_function_pool import corr_plots_hourly_monthly_yearly

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
os.getcwd()
# open NETCDF files
ds_RH_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
ds_RH_775 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/775hPa/*.nc', combine = 'by_coords')
ds_RH_700 =xr.open_mfdataset('./sites/Paranal/Data/Era_5/RH/700hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe

RH_hourly = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_T_RH_P.csv')

# as a comparison, load in data from Cerro Pachon (same gridpoint)
# read cp dataset
RH_hourly_Pachon = pd.read_csv('./sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_cp.csv')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_RH_750 = netcdf_to_df(ds_RH_750,-70.81, -30.17)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

df_RH_775 = netcdf_to_df(ds_RH_775,-70.81, -30.17)
df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

df_RH_700 = netcdf_to_df(ds_RH_700,-70.81, -30.17)
df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

df_RH_800 = netcdf_to_df(ds_RH_800,-70.81, -30.17)
df_prep_RH_800 = df_prep(df_RH_800, 'r', '800hPa')

print('netcdf to df done')


#%% prepare Cerro_Tololo calculated data

RH_hourly_preped = mes_prep(RH_hourly)
RH_hourly_Pachon_preped = mes_prep(RH_hourly_Pachon)

RH_hourly_Pachon_preped = RH_hourly_Pachon_preped.rename(columns={'RH': 'Cerro Pachon Relative Humidity'})

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped, RH_hourly_Pachon_preped['Cerro Pachon Relative Humidity'],
df_prep_RH_750, df_prep_RH_775, df_prep_RH_700, df_prep_RH_800)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_750, df_prep_RH_775, df_prep_RH_700, df_prep_RH_800)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Cerro Tololo and Pachon', 'Cerro Tololo Relative Humidity', 
           './sites/Cerro_Tololo/Output/Plots/RH/diurnal_cycle_UTC_RH_Cerro_Tololo_and_Pachon_2002to2019.pdf', 
           '700hPa', '750hPa', '775hPa', '800hPa', insitu_parameter_2 = 'Cerro Pachon Relative Humidity')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle Cerro Tololo and Pachon', 'Cerro Tololo Relative Humidity', 
           './sites/Cerro_Tololo/Output/Plots/RH/seasonal_cycle_UTC_RH_Cerro_Tololo_and_Pachon_2002to2019.pdf',
           '700hPa', '750hPa', '775hPa','800hPa', insitu_parameter_2 = 'Cerro Pachon Relative Humidity')

# %%
#plot_timeseries_merged('./sites/Cerro_Tololo/Output/Plots/RH/Timeseries_UTC_RH_all_2002to2019.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '750hPa', '775hPa', '700hPa')


# %%
# plot timeseries, moving average

plot_timeseries_movav('./sites/Cerro_Tololo/Output/Plots/RH/timeseries_Cerro_Tololo_and_Pachon_RH_2002to2019_movav.pdf', yearly_grouped_RH.loc[:'2018-12-31'],
number_of_insitu_params=2,
In_situ_Tololo = 'Cerro Tololo Relative Humidity', in_situ_Pachon = 'Cerro Pachon Relative Humidity', Era5_700hPa = '700hPa' , Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Cerro_Tololo/Output/Plots/RH/timeseries_Cerro_Tololo_and_Pachon_RH_2002to2018_long_movav.pdf', yearly_grouped_RH.loc[:'2018-12-31'], # data ends in 2019-03-26 already
['Cerro Tololo Relative Humidity', 'Cerro Pachon Relative Humidity'], yearly_grouped_RH_era5.loc[:'2018-12-31'], moving = False, 
Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '750hPa'
insitu_param = 'Cerro Tololo Relative Humidity'
path = './sites/Cerro_Tololo/Output/Plots/RH/correlation_RH_Cerro_Tololo_and_Pachon_' + parameter +'_2002to2019_'
xax = 'relative humidity Era 5 ' + parameter +' pressure level [%]'
yax = 'relative humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)


# %% Tololo vs. Pachon
parameter_1 = 'Cerro Pachon Relative Humidity'
parameter_2 = 'Cerro Tololo Relative Humidity'
path = './sites/Cerro_Tololo/Output/Plots/RH/correlation_RH_Cerro_Tololo_and_Pachon_2002to2019_'
xax = 'in-situ ' + parameter_1 + '[%]' 
yax = 'in-situ ' + parameter_2 + '[%]' 

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter_1, parameter_2, xax, yax)

# %%
