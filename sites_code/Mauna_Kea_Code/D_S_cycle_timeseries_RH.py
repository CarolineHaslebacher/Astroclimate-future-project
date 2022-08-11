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
ds_RH_600 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/RH/600hPa/*.nc', combine = 'by_coords')
ds_RH_650 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/RH/650hPa/*.nc', combine = 'by_coords')
ds_RH_700 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/RH/700hPa/*.nc', combine = 'by_coords')
ds_RH_750 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/RH/750hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
CFHT_RH_hourly = pd.read_csv('./sites/MaunaKea/Data/in-situ/RH/downsampled_masked_RH_1991to2018_hourly_means.csv')


#%%
df_RH_600 = netcdf_to_df(ds_RH_600, -155.5, 19.75)
df_prep_RH_600 = df_prep(df_RH_600, 'r', '600hPa')

df_RH_650 = netcdf_to_df(ds_RH_650, -155.5, 19.75)
df_prep_RH_650 = df_prep(df_RH_650, 'r', '650hPa')

df_RH_700 = netcdf_to_df(ds_RH_700,-155.5, 19.75)
df_prep_RH_700 = df_prep(df_RH_700, 'r', '700hPa')

df_RH_750 = netcdf_to_df(ds_RH_750, -155.5, 19.75)
df_prep_RH_750 = df_prep(df_RH_750, 'r', '750hPa')

print('netcdf to df done')


#%% prepare CFHT calculated data

CFHT_RH_hourly_preped = mes_prep(CFHT_RH_hourly)

#%%
# merge datasets
merged_df_RH, seasonal_RH, diurnal_RH, monthly_grouped_RH, yearly_grouped_RH  = merge_df(CFHT_RH_hourly_preped, df_prep_RH_600,
df_prep_RH_650, df_prep_RH_700, df_prep_RH_750)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_RH,'diurnal cycle Mauna Kea', 'relative_humidity(%)', 
           './sites/MaunaKea/Output/Plots/RH/diurnal_cycle_UTC_RH_all_MK_masked_2000to2020.pdf', 
           '600hPa', '650hPa', '700hPa', '750hPa')
           
# ./sites/MaunaKea/Output/Plots/RH/
# %%
plot_cycle(seasonal_RH,'seasonal cycle Mauna Kea', 'relative_humidity(%)', 
           './sites/MaunaKea/Output/Plots/RH/seasonal_cycle_UTC_RH_all_MK_masked_2000to2020.pdf',
           '600hPa', '650hPa', '700hPa', '750hPa')

# %%
plot_timeseries_merged('./sites/MaunaKea/Output/Plots/RH/Timeseries_UTC_RH_all_2000to2020.pdf', merged_df_RH, monthly_grouped_RH, yearly_grouped_RH, 
CFHT = 'relative_humidity(%)', Era5_600hPa = '600hPa',Era5_650hPa = '650hPa', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa')

# %%
# plot timeseries, Era5 data

#%%
plot_timeseries_long('./sites/MaunaKea/Output/Plots/RH/timeseries_long_RH_all_MK_1979to2020.pdf',CFHT_RH_hourly_preped,
Era5_600hPa = df_prep_RH_600, Era5_650hPa = df_prep_RH_650,
Era5_700hPa = df_prep_RH_700,Era5_750hPa = df_prep_RH_750)



# %%
