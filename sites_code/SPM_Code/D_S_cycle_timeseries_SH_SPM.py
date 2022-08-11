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
ds_SH_850 = xr.open_mfdataset('./sites/Baja/Data/Era_5/SH/850hPa/*.nc', combine = 'by_coords')
ds_SH_750 = xr.open_mfdataset('./sites/Baja/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')
ds_SH_775 = xr.open_mfdataset('./sites/Baja/Data/Era_5/SH/775hPa/*.nc', combine = 'by_coords')
ds_SH_700 = xr.open_mfdataset('./sites/Baja/Data/Era_5/SH/700hPa/*.nc', combine = 'by_coords')


#open in-situ measurements as pandas dataframe

SH_hourly = pd.read_csv('./sites/Baja/Data/in-situ/hourly_meteo/Specific_humidity_SPM_2006to2020.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_SH_850 = netcdf_to_df(ds_SH_850, -115.46, 31.04)
df_prep_SH_850 = df_prep(df_SH_850, 'q', '850hPa')

df_SH_750 = netcdf_to_df(ds_SH_750, -115.46, 31.04)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

df_SH_775 = netcdf_to_df(ds_SH_775, -115.46, 31.04)
df_prep_SH_775 = df_prep(df_SH_775, 'q', '775hPa')

df_SH_700 = netcdf_to_df(ds_SH_700, -115.46, 31.04)
df_prep_SH_700 = df_prep(df_SH_700, 'q', '700hPa')

print('netcdf to df done')


#%% prepare SPM calculated data

SH_hourly_preped = mes_prep(SH_hourly)

#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(SH_hourly_preped,
df_prep_SH_850, df_prep_SH_750, df_prep_SH_775, df_prep_SH_700, dropnan=True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_850, df_prep_SH_750, df_prep_SH_775, df_prep_SH_700)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle SPM', 'specific_humidity', 
           './sites/Baja/Output/Plots/SH/diurnal_cycle_UTC_SH_SPM_2006to2019.pdf', 
           '700hPa','750hPa','775hPa','850hPa')
           
# %% seasonal cycle
plot_cycle(seasonal_SH,'seasonal cycle SPM', 'specific_humidity', 
           './sites/Baja/Output/Plots/SH/seasonal_cycle_UTC_SH_SPM_2006to2019.pdf',
           '700hPa','750hPa','775hPa','850hPa')

# %%
#plot_timeseries_merged('./sites/Baja/Output/Plots/SH/Timeseries_UTC_SH_all_2006to2019.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 
#'Specific_humidity', '750hPa', '775hPa', '700hPa')


# %%
# plot timeseries, moving average
yearly_grouped_SH = yearly_grouped_SH.loc['2006-12-31':'2019-12-31']

plot_timeseries_movav('./sites/Baja/Output/Plots/SH/timeseries_SPM_SH_2006to2019_movav.pdf', yearly_grouped_SH,
number_of_insitu_params=1,
In_situ = 'specific_humidity' , Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Baja/Output/Plots/SH/timeseries_SPM_SH_2006to2019_long.pdf', yearly_grouped_SH,
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = False,Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Baja/Output/Plots/SH/timeseries_SPM_SH_2006to2019_long_movav.pdf', yearly_grouped_SH,
['specific_humidity'], yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_775hPa = '775hPa', Era5_850hPa = '850hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

# parameter = '700hPa'
# insitu_param = 'specific_humidity'
# path = './sites/Baja/Output/Plots/SH/correlation_SH_SPM_' + parameter +'_2006to2019_'
# xax = 'Specific humidity Era 5 ' + parameter +' pressure level [%]'
# yax = 'Specific humidity in-situ [%]'

# corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)



# %%
parameter = '775hPa'
insitu_param = 'specific_humidity'
path = './sites/Baja/Output/Plots/SH/correlation_SH_SPM_' + parameter +'_2006to2019_'
xax = 'Specific humidity Era 5 ' + parameter +' pressure level [%]'
yax = 'Specific humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, parameter, insitu_param, xax, yax)



# %%
