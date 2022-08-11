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

# open NETCDF files on 600hPa to 750hPa
ds_tcw = xr.open_mfdataset('./sites/Paranal/Data/Era_5/TCW/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
PWV_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_PWV.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# not nearest,  but 'green' point, higher elevated
df_tcw = netcdf_to_df(ds_tcw, -70.25, -24.75)
df_tcw_prep = df_prep(df_tcw, 'tcw', 'tcw')

print('netcdf to df done')

#%% prepare Paranal calculated data

PWV_hourly_preped = mes_prep(PWV_hourly) # timezone = None, already in UTC

#%%
# merge datasets
merged_df_PWV, seasonal_PWV, diurnal_PWV, monthly_grouped_PWV, yearly_grouped_PWV  = merge_df(PWV_hourly_preped,
df_tcw_prep)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_PWV_era5, monthly_grouped_PWV_era5, yearly_grouped_PWV_era5  = merge_df_long(df_tcw_prep)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_PWV,'diurnal cycle Cerro Paranal', 'PWV Paranal', 
           './sites/Paranal/Output/Plots/TCW/diurnal_cycle_UTC_TCW_Paranal_2015to2019.pdf', 
           'tcw')
           
# %% seasonal cycle
plot_cycle(seasonal_PWV,'seasonal cycle Cerro Paranal', 'PWV Paranal', 
           './sites/Paranal/Output/Plots/TCW/seasonal_cycle_UTC_TCW_Paranal_2015to2019.pdf', 
           'tcw')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/Paranal/Output/Plots/TCW/timeseries_Paranal_TCW_2015to2019_movav.pdf', yearly_grouped_PWV,
# In_situ = 'PWV Paranal', Era5_TCW = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/TCW/timeseries_Paranal_TCW_2015to2019_long.pdf', yearly_grouped_PWV,
'PWV Paranal', yearly_grouped_PWV_era5.loc[:'2019-12-31'], moving = False, Era5_TCW = 'tcw')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'tcw'
path = './sites/Paranal/Output/Plots/TCW/correlation_Paranal_' + parameter +'_2015to2020_'
xax = r'Era 5 total column water [$kg/m^2$]'
yax = r'in-situ total column water [$kg/m^2$]'

#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_PWV[[parameter, 'PWV Paranal']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path +'hourly.pdf',
'hourly means',
merged_df_ESO[parameter], xax,
merged_df_ESO['PWV Paranal'], yax)

# monthly
monthly_corr = monthly_grouped_PWV[[parameter, 'PWV Paranal']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_monthly.pdf',
'monthly means',
monthly_corr[parameter]['mean'],  xax,
monthly_corr['PWV Paranal']['mean'], yax)

# yearly
correlation_plot(path + 'yearly.pdf',
'yearly means',
yearly_grouped_PWV[parameter]['mean'],  xax,
yearly_grouped_PWV['PWV Paranal']['mean'], yax)


# %%
