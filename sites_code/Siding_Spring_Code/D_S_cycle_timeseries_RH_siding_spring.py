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
ds_RH_925 = xr.open_mfdataset('./sites/Siding_spring/Data/Era_5/RH/925hPa/*.nc', combine = 'by_coords')
ds_RH_850 = xr.open_mfdataset('./sites/Siding_spring/Data/Era_5/RH/850hPa/*.nc', combine = 'by_coords')
ds_RH_875 = xr.open_mfdataset('./sites/Siding_spring/Data/Era_5/RH/875hPa/*.nc', combine = 'by_coords')
ds_RH_900 = xr.open_mfdataset('./sites/Siding_spring/Data/Era_5/RH/900hPa/*.nc', combine = 'by_coords')


#open in-situ measurements as pandas dataframe

RH_hourly = pd.read_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/hourly_siding_spring_T_RH_P_time.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_RH_925 = netcdf_to_df(ds_RH_925, 149.07, -31.28)
df_prep_RH_925 = df_prep(df_RH_925, 'r', '925hPa')

df_RH_850 = netcdf_to_df(ds_RH_850, 149.07, -31.28)
df_prep_RH_850 = df_prep(df_RH_850, 'r', '850hPa')

df_RH_875 = netcdf_to_df(ds_RH_875, 149.07, -31.28)
df_prep_RH_875 = df_prep(df_RH_875, 'r', '875hPa')

df_RH_900 = netcdf_to_df(ds_RH_900, 149.07, -31.28)
df_prep_RH_900 = df_prep(df_RH_900, 'r', '900hPa')

print('netcdf to df done')


#%% prepare siding_spring calculated data

RH_hourly_preped = mes_prep(RH_hourly, parameter = 'siding spring Relative Humidity')

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(RH_hourly_preped,
df_prep_RH_925, df_prep_RH_850, df_prep_RH_875, df_prep_RH_900)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_RH_era5, monthly_grouped_RH_era5, yearly_grouped_RH_era5  = merge_df_long(df_prep_RH_925, df_prep_RH_850, df_prep_RH_875, df_prep_RH_900)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Siding Spring', 'siding spring Relative Humidity', 
           './sites/siding_spring/Output/Plots/RH/diurnal_cycle_UTC_RH_siding_spring_2003to2019.pdf', 
            '850hPa', '875hPa', '900hPa', '925hPa')
           
# %%
plot_cycle(seasonal_RH,'seasonal cycle Siding Spring', 'siding spring Relative Humidity', 
           './sites/siding_spring/Output/Plots/RH/seasonal_cycle_UTC_RH_siding_spring_2003to2019.pdf',
           '850hPa', '875hPa', '900hPa', '925hPa')

# %%
#plot_timeseries_merged('./sites/siding_spring/Output/Plots/RH/Timeseries_UTC_RH_all_2003to2019.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
#'relative_humidity', '850hPa', '875hPa', '900hPa')


# %%
# plot timeseries, moving average
yearly_grouped_RH = yearly_grouped_RH.loc['2003-12-31':'2019-12-31']

plot_timeseries_movav('./sites/siding_spring/Output/Plots/RH/timeseries_siding_spring_RH_2003to2019_movav.pdf', yearly_grouped_RH,
number_of_insitu_params=1,
In_situ = 'siding spring Relative Humidity' , Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa', Era5_925hPa = '925hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/siding_spring/Output/Plots/RH/timeseries_siding_spring_RH_2003to2019_long.pdf', yearly_grouped_RH,
['siding spring Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = False, Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa', Era5_925hPa = '925hPa')

#%%
# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/siding_spring/Output/Plots/RH/timeseries_siding_spring_RH_2003to2019_long_movav.pdf', yearly_grouped_RH,
['siding spring Relative Humidity'], yearly_grouped_RH_era5.loc[:'2019-12-31'], moving = True, Era5_850hPa = '850hPa', Era5_875hPa = '875hPa', Era5_900hPa = '900hPa', Era5_925hPa = '925hPa')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = '900hPa'
insitu_param = 'siding spring Relative Humidity'
path = './sites/siding_spring/Output/Plots/RH/correlation_RH_siding_spring_' + parameter +'_2003to2019_'
xax = 'relative humidity Era 5 ' + parameter +' pressure level [%]'
yax = 'relative humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)



# %%
parameter = '925hPa'
insitu_param = 'siding spring Relative Humidity'
path = './sites/siding_spring/Output/Plots/RH/correlation_RH_siding_spring_' + parameter +'_2003to2019_'
xax = 'relative humidity Era 5 ' + parameter +' pressure level [%]'
yax = 'relative humidity in-situ [%]'

corr_plots_hourly_monthly_yearly(path, merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, parameter, insitu_param, xax, yax)



# %%
