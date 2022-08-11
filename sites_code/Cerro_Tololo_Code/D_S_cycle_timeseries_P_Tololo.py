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

# open NETCDF file
ds_surf_Pr = xr.open_mfdataset('./sites/Paranal/Data/Era_5/surface_pressure/*.nc', combine = 'by_coords')

#open in-situ measurements as pandas dataframe
P_hourly = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_T_RH_P.csv')

 
#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

df_surf_Pr = netcdf_to_df(ds_surf_Pr,-70.81, -30.17)
df_prep_surf_Pr = df_prep(df_surf_Pr, 'sp', 'sp')/100 # convert to hPa

print('netcdf to df done')

#%% prepare Cerro_Tololo calculated data

P_hourly_preped = mes_prep(P_hourly) # already filtered!

#%%
# merge datasets
merged_df_P, seasonal_P, diurnal_P, monthly_grouped_P, yearly_grouped_P  = merge_df(P_hourly_preped, df_prep_surf_Pr)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_P_era5, monthly_grouped_P_era5, yearly_grouped_P_era5  = merge_df_long(df_prep_surf_Pr)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_P,'diurnal cycle Cerro Tololo', 'Cerro Tololo Pressure (hPa)', 
           './sites/Cerro_Tololo/Output/Plots/P/diurnal_cycle_UTC_P_Cerro_Tololo_2002to2019.pdf', 
           'sp')
           
# ./sites/MaunaKea/Output/Plots/P/
# %%
plot_cycle(seasonal_P,'seasonal cycle Cerro Tololo', 'Cerro Tololo Pressure (hPa)', 
           './sites/Cerro_Tololo/Output/Plots/P/seasonal_cycle_UTC_P_Cerro_Tololo_2002to2019.pdf',
           'sp')


# %%
#plot_timeseries_merged('./sites/Cerro_Tololo/Output/Plots/P/Timeseries_UTC_P_all_1997to2019.pdf', merged_df_P, monthly_grouped_P, yearly_grouped_P, 
#'relative_humidity', '700hPa', '750hPa', '800hPa')

# %%
# plot timeseries, moving average

yearly_grouped_P = yearly_grouped_P.loc['2002-01-01':'2019-12-31']

plot_timeseries_movav('./sites/Cerro_Tololo/Output/Plots/P/timeseries_Cerro_Tololo_P_2002to2019_movav.pdf', yearly_grouped_P,
 df_merged = merged_df_P, In_situ = 'Cerro Tololo Pressure (hPa)', Era5_surface_pressure = 'sp')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Cerro_Tololo/Output/Plots/P/timeseries_Cerro_Tololo_P_2002to2019_long.pdf', yearly_grouped_P,
['Cerro Tololo Pressure (hPa)'], yearly_grouped_P_era5.loc[:'2019-12-31'], moving = False, Era5_surface_pressure = 'sp')

# %%
# correlation plot: alpha = 0.05

parameter = 'sp'
insitu_param = 'Cerro Tololo Pressure (hPa)'
path = './sites/Cerro_Tololo/Output/Plots/P/correlation_P_Cerro_Tololo_' + parameter +'_2002to2019'
xax = 'surface pressure Era 5 [hPa]'
yax = 'pressure in-situ [hPa]'

corr_plots_hourly_monthly_yearly(path, merged_df_P, monthly_grouped_P, yearly_grouped_P, parameter, insitu_param, xax, yax)

# %%
