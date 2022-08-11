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
ds_RH_750 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
ds_RH_775 = xr.open_mfdataset('./sites/La_Palma/Data/Era_5/RH/775hPa/*.nc', combine = 'by_coords')
ds_RH_800 =xr.open_mfdataset('./sites/La_Palma/Data/Era_5/RH/800hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
#RH_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/ESO/hourly_meteo/hourly_La_Palma_RH_T_P.csv')
RH_hourly = pd.read_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_RH_T_P.csv')

# REMEMBER, that orography of ERA5 for la Palma shows an elevation of 169m only

#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_RH_750 = netcdf_to_df(ds_RH_750, -17.88, 28.76)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

df_RH_775 = netcdf_to_df(ds_RH_775, -17.88, 28.76)
df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

df_RH_800 = netcdf_to_df(ds_RH_800, -17.88, 28.76)
df_prep_RH_800 = df_prep(df_RH_800, 'r', '800hPa')

print('netcdf to df done')


#%% prepare La_Palma calculated data

RH_hourly_preped = mes_prep(RH_hourly)

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped,
df_prep_RH_750, df_prep_RH_775, df_prep_RH_800, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_750, df_prep_RH_775, df_prep_RH_800)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Cerro de la Palma', 'La_Palma Relative Humidity', 
           './sites/La_Palma/Output/Plots/RH/diurnal_cycle_UTC_RH_La_Palma_1997to2020.pdf', 
           '750hPa', '775hPa', '800hPa')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle Cerro de la Palma', 'La_Palma Relative Humidity', 
           './sites/La_Palma/Output/Plots/RH/seasonal_cycle_UTC_RH_La_Palma_1997to2020.pdf',
           '750hPa', '775hPa', '800hPa')

# %%
#plot_timeseries_merged('./sites/La_Palma/Output/Plots/RH/Timeseries_UTC_RH_all_1997to2020.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '750hPa', '775hPa', '800hPa')


# %%
# plot timeseries, moving average

yearly_grouped_RH = yearly_grouped_RH.loc['1997-12-31':'2019-12-31']

# # function assumes that relative humidity is in first column
# RH_hourly_preped.iloc[:,0] = RH_hourly_preped.iloc[:,6]
# RH_hourly_preped.rename(columns={'Unnamed: 0': 'relative_humidity'})

plot_timeseries_movav('./sites/La_Palma/Output/Plots/RH/timeseries_La_Palma_RH_1997to2020_movav.pdf', yearly_grouped_RH,
 In_situ = 'La_Palma Relative Humidity' , Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Palma/Output/Plots/RH/timeseries_La_Palma_RH_1997to2020_long.pdf', yearly_grouped_RH,
['La_Palma Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = True, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

plot_timeseries_long('./sites/La_Palma/Output/Plots/RH/timeseries_La_Palma_RH_1997to2020_long.pdf', yearly_grouped_RH,
['La_Palma Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = False, Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_800hPa = '800hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '800hPa'
insitu_param = 'La_Palma Relative Humidity'
path = './sites/La_Palma/Output/Plots/RH/correlation_RH_La_Palma_' + parameter +'_1997to2020_'
xax = 'relative humidity ERA 5 ' + parameter +' pressure level [%]'
yax = 'relative humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)





# %%
