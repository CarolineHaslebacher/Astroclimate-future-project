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

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Code/')
from Astroclimate_function_pool import netcdf_to_df
from Astroclimate_function_pool import  mes_prep
from Astroclimate_function_pool import  merge_df 
from Astroclimate_function_pool import  df_prep #(df, parameter, colname)
from Astroclimate_function_pool import  plot_cycle #(cycle_name, cycle_string,  CFHT_parameter, filename, *args)
from Astroclimate_function_pool import  plot_timeseries_merged
from Astroclimate_function_pool import plot_timeseries_long

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
os.getcwd()
# open NETCDF files on 600hPa to 750hPa
ds_T_600 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/T/pressure_levels/600hPa/*.nc', combine = 'by_coords')
ds_T_650 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/T/pressure_levels/650hPa/*.nc', combine = 'by_coords')
ds_T_700 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/T/pressure_levels/700hPa/*.nc', combine = 'by_coords')
ds_T_750 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/T/pressure_levels/750hPa/*.nc', combine = 'by_coords')

# open 2m surface temperature
ds_T2m = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/T/single_levels/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene')
#ds_T2m_2008to2012 = xr.open_dataset('./cds_data_ERA5/Era5_single_level_2008to2012_2mTemperature.nc')

# open observational data from CFHT for Mauna Kea
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
CFHT_T_hourly = pd.read_csv('./sites/MaunaKea/Data/in-situ/T/downsampled_CFHT_T_1991to2018_hourly_means.csv')#/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/in-situ/RH/


# %%
#start only in year 2000
mask = (CFHT_T_hourly['Unnamed: 0'] >= '2000-01-01 00:00:00') 
CFHT_T_hourly_2000 = CFHT_T_hourly[mask]
CFHT_T_hourly_preped = mes_prep(CFHT_T_hourly_2000, 'temperature(C)')

# %%
df_T2m = netcdf_to_df(ds_T2m, -155.5, 19.75)
df_prep_T2m = df_prep(df_T2m, 't2m', 't2m')
df_prep_T2m_Celsius = (df_prep_T2m['t2m']-273.15)

#%%
# df_T2m_2008to2012 = netcdf_to_df(ds_T2m_2008to2012, 204.5, 19.75)
# df_prep_T2m_2008to2012 = df_prep(df_T2m_2008to2012, 't2m', 't2m 2008to2012')
# df_prep_T2m_Celsius_2008to2012 = (df_prep_T2m_2008to2012['t2m 2008to2012']-273.15)

#%%
df_T_600 = netcdf_to_df(ds_T_600, -155.5, 19.75)
df_prep_T_600 = df_prep(df_T_600, 't', '600hPa')
df_prep_T_600_Celsius = (df_prep_T_600['600hPa']-273.15)

df_T_650 = netcdf_to_df(ds_T_650, -155.5, 19.75)
df_prep_T_650 = df_prep(df_T_650, 't', '650hPa')
df_prep_T_650_Celsius = (df_prep_T_650['650hPa']-273.15)

df_T_700 = netcdf_to_df(ds_T_700,-155.5, 19.75)
df_prep_T_700 = df_prep(df_T_700, 't', '700hPa')
df_prep_T_700_Celsius = (df_prep_T_700['700hPa']-273.15)

df_T_750 = netcdf_to_df(ds_T_750, -155.5, 19.75)
df_prep_T_750 = df_prep(df_T_750, 't', '750hPa')
df_prep_T_750_Celsius = (df_prep_T_750['750hPa']-273.15)

print('netcdf to df done')
# %%
# to reproduce values from 2008 to 2012:
#mask2 = (df_prep_T2m_Celsius.index>= '2008-01-01 00:00:00') & (df_prep_T2m_Celsius.index < '2013-01-01 00:00:00')
#df_prep_T2m_Celsius_2008to2012_newdata = df_prep_T2m_Celsius[mask2]

#%%
# merge datasets
merged_df_T, seasonal_T, diurnal_T, monthly_grouped_T, yearly_grouped_T  = merge_df(CFHT_T_hourly_preped, df_prep_T2m_Celsius, df_prep_T_600_Celsius,
df_prep_T_650_Celsius, df_prep_T_700_Celsius, df_prep_T_750_Celsius)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_T,'diurnal cycle Mauna Kea', 'temperature(C)', 
           './sites/MaunaKea/Output/Plots/T/diurnal_cycle_UTC_T_all_MK_masked_2000to2020.pdf', 't2m', 
           '600hPa', '650hPa', '700hPa', '750hPa')
           
# ./sites/MaunaKea/Output/Plots/T/
# %%
plot_cycle(seasonal_T,'seasonal cycle Mauna Kea', 'temperature(C)', 
           './sites/MaunaKea/Output/Plots/T/seasonal_cycle_UTC_T_all_MK_masked_2000to2020.pdf', 't2m',
           '600hPa', '650hPa', '700hPa', '750hPa')

# %%
plot_timeseries_merged('./sites/MaunaKea/Output/Plots/T/Timeseries_UTC_T_all_2000to2020.pdf', merged_df_T, monthly_grouped_T, yearly_grouped_T, 'temperature(C)', 't2m', '600hPa', '650hPa', '700hPa', '750hPa')

# %%
# plot timeseries, Era5 data

plot_timeseries_long('./sites/MaunaKea/Output/Plots/T/timeseries_long_T_all_MK_1979to2020.pdf',CFHT_T_hourly_preped , ERA5_T2m = df_prep_T2m_Celsius,
Era5_600hPa = df_prep_T_600_Celsius, Era5_650hPa = df_prep_T_650_Celsius,
Era5_700hPa = df_prep_T_700_Celsius,Era5_750hPa = df_prep_T_750_Celsius)

