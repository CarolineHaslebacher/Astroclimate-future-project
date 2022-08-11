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

# open NETCDF file
ds_surf_Pr = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/surface_pressure/*.nc', combine = 'by_coords')

CFHT_P_hourly = pd.read_csv('./sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv')


#%%
df_surf_Pr = netcdf_to_df(ds_surf_Pr, -155.5, 19.75)
df_prep_surf_Pr = df_prep(df_surf_Pr, 'sp', 'sp')/100

print('netcdf to df done')


#%% prepare CFHT calculated data

CFHT_P_hourly_preped = mes_prep(CFHT_P_hourly)

#%%
# merge datasets
merged_df_P, seasonal_P, diurnal_P, monthly_grouped_P, yearly_grouped_P  = merge_df(CFHT_P_hourly_preped, df_prep_surf_Pr)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_P,'diurnal cycle Mauna Kea', 'P', 
           './sites/MaunaKea/Output/Plots/P/diurnal_cycle_UTC_P_MaunaKea_2000to2020.pdf', 
           'sp')
           
# ./sites/MaunaKea/Output/Plots/P/
# %%
plot_cycle(seasonal_P,'seasonal cycle Mauna Kea', 'P', 
           './sites/MaunaKea/Output/Plots/P/seasonal_cycle_UTC_P_MaunaKea_2000to2020.pdf',
           'sp')


# %%

yearly_grouped_P = yearly_grouped_P.loc['2000-01-01':'2019-12-31']


plot_timeseries_movav('./sites/MaunaKea/Output/Plots/P/Timeseries_UTC_P_all_2000to2020.pdf', yearly_grouped_P, 
df_merged = merged_df_P, In_situ = 'P',  Era5_surface_pressure = 'sp')


# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/MaunaKea/Output/Plots/P/timeseries_MaunaKea_P_2000to2019_long.pdf', yearly_grouped_P,
['P'], yearly_grouped_P_era5.loc[:'2019-12-31'], moving = False, Era5_surface_pressure = 'sp')


# %%
