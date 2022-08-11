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
ds_RH_850 = xr.open_mfdataset('./sites/SPM/Data/Era_5/RH/850hPa/*.nc', combine = 'by_coords')
ds_RH_750 = xr.open_mfdataset('./sites/SPM/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
ds_RH_775 = xr.open_mfdataset('./sites/SPM/Data/Era_5/RH/775hPa/*.nc', combine = 'by_coords')
ds_RH_700 = xr.open_mfdataset('./sites/SPM/Data/Era_5/RH/700hPa/*.nc', combine = 'by_coords')


#open in-situ measurements as pandas dataframe

RH_hourly = pd.read_csv('./sites/SPM/Data/in-situ/hourly_meteo/hourly_SPM_T_RH_time_in_what.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_RH_850 = netcdf_to_df(ds_RH_850, -115.46, 31.04)
df_prep_RH_850 = df_prep(df_RH_850, 'r', '850hPa')

df_RH_750 = netcdf_to_df(ds_RH_750, -115.46, 31.04)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

df_RH_775 = netcdf_to_df(ds_RH_775, -115.46, 31.04)
df_prep_RH_775 = df_prep(df_RH_775, 'r', '775hPa')

df_RH_700 = netcdf_to_df(ds_RH_700, -115.46, 31.04)
df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

print('netcdf to df done')


#%% prepare SPM calculated data

RH_hourly_preped = mes_prep(RH_hourly, parameter = 'SPM Relative Humidity')

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped,
df_prep_RH_850, df_prep_RH_750, df_prep_RH_775, df_prep_RH_700)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_850, df_prep_RH_750, df_prep_RH_775, df_prep_RH_700)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle SPM', 'SPM Relative Humidity', 
           './sites/SPM/Output/Plots/RH/diurnal_cycle_UTC_RH_SPM_2006to2019.pdf', 
           '700hPa','750hPa','775hPa','850hPa')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle SPM', 'SPM Relative Humidity', 
           './sites/SPM/Output/Plots/RH/seasonal_cycle_UTC_RHa_SPM_2006to2019.pdf',
           '700hPa','750hPa','775hPa','850hPa')

# %%
#plot_timeseries_merged('./sites/SPM/Output/Plots/RH/Timeseries_UTC_RH_all_2006to2019.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '750hPa', '775hPa', '700hPa')


# %%
# plot timeseries, moving average
yearly_grouped_RH = yearly_grouped_RH.loc['2006-12-31':'2019-12-31']

plot_timeseries_movav('./sites/SPM/Output/Plots/RH/timeseries_SPM_RH_2006to2019_movav.pdf', yearly_grouped_RH,
number_of_insitu_params=1,
In_situ = 'SPM Relative Humidity' , Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/SPM/Output/Plots/RH/timeseries_SPM_RH_2006to2019_long.pdf', yearly_grouped_RH,
['SPM Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = False,Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/SPM/Output/Plots/RH/timeseries_SPM_RH_2006to2019_long_movav.pdf', yearly_grouped_RH,
['SPM Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

# parameter = '700hPa'
# insitu_param = 'SPM Relative Humidity'
# path = './sites/SPM/Output/Plots/RH/correlation_RH_SPM_' + parameter +'_2006to2019_'
# xax = 'relative humidity Era 5 ' + parameter +' pressure level [%]'
# yax = 'relative humidity in-situ [%]'

# corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)



# %%
parameter = '850hPa'
insitu_param = 'SPM Relative Humidity'
path = './sites/SPM/Output/Plots/RH/correlation_RH_SPM_' + parameter +'_2006to2019_'
xax = 'relative humidity Era 5 ' + parameter +' pressure level [%]'
yax = 'relative humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)



# %%
