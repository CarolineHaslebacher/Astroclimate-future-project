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
# open NETCDF files
ds_SH_775 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/SH/775hPa/*.nc', combine = 'by_coords')
ds_SH_750 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')
ds_SH_800 =xr.open_mfdataset('./sites/La_Palma/Data/Era_5/SH/800hPa/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
SH_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv', parse_dates=True, index_col='time')

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_SH_775 = netcdf_to_df(ds_SH_775, -17.88, 28.76)
df_prep_SH_775 = df_prep(df_SH_775, 'q', '775hPa')

df_SH_750 = netcdf_to_df(ds_SH_750, -17.88, 28.76)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

df_SH_800 = netcdf_to_df(ds_SH_800, -17.88, 28.76)
df_prep_SH_800 = df_prep(df_SH_800, 'q', '800hPa')

print('netcdf to df done')

#%% prepare La_Palma calculated data

SH_hourly_preped = mes_prep(SH_hourly)

#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(SH_hourly_preped,
df_prep_SH_775, df_prep_SH_750, df_prep_SH_800, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_775, df_prep_SH_750, df_prep_SH_800)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Cerro La_Palma', 'specific_humidity', 
           './sites/La_Palma/Output/Plots/SH/diurnal_cycle_UTC_SH_La_Palma_1997to2020.pdf', 
           '750hPa', '775hPa', '800hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Cerro La_Palma', 'specific_humidity', 
           './sites/La_Palma/Output/Plots/SH/seasonal_cycle_UTC_SH_La_Palma_1997to2020.pdf',
           '750hPa', '775hPa', '800hPa')

# %%
#plot_timeseries_merged('./sites/La_Palma/Output/Plots/SH/Timeseries_UTC_SH_all_2000to2019.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 
#'specific_humidity', '775hPa', '750hPa', '800hPa')

# %%
# plot timeseries, moving average
yearly_grouped_SH = yearly_grouped_SH.loc['1997-12-31':'2019-12-31']

plot_timeseries_movav('./sites/La_Palma/Output/Plots/SH/timeseries_La_Palma_1997to2020_movav.pdf', yearly_grouped_SH,
 In_situ = 'specific_humidity',  Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Palma/Output/Plots/SH/timeseries_La_Palma_1997to2019_long_movav.pdf', yearly_grouped_SH,
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = True, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

plot_timeseries_long('./sites/La_Palma/Output/Plots/SH/timeseries_La_Palma_1997to2019_long.pdf', yearly_grouped_SH,
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = False, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %% correlation

parameter = '775hPa'
insitu_param = 'specific_humidity'
path = './sites/La_Palma/Output/Plots/SH/correlation_SH_La_Palma_' + parameter +'_1997to2019'
xax = 'specific humidity Era 5 ' + parameter +' pressure level [kg/kg]'
yax = 'specific humidity in-situ [kg/kg]'

corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)

# %%
