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
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
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
ds_SH_600 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/600hPa/*.nc', combine = 'by_coords')
ds_SH_650 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/650hPa/*.nc', combine = 'by_coords')
ds_SH_700 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/700hPa/*.nc', combine = 'by_coords')
ds_SH_750 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
CFHT_SH_hourly = pd.read_csv('./sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv')


#%% convert netcdf to dataframe, label pressure levels
df_SH_600 = netcdf_to_df(ds_SH_600, -155.5, 19.75)
df_prep_SH_600 = df_prep(df_SH_600, 'q', '600hPa')

df_SH_650 = netcdf_to_df(ds_SH_650, -155.5, 19.75)
df_prep_SH_650 = df_prep(df_SH_650, 'q', '650hPa')

df_SH_700 = netcdf_to_df(ds_SH_700,-155.5, 19.75)
df_prep_SH_700 = df_prep(df_SH_700, 'q', '700hPa')

df_SH_750 = netcdf_to_df(ds_SH_750, -155.5, 19.75)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

print('netcdf to df done')


#%% prepare CFHT calculated data

# do not shift specific humidity CFHT array by 10hours, it is already done! only add month and hours
CFHT_SH_hourly['time'] = pd.to_datetime(CFHT_SH_hourly['time']) 

#check the format
print(CFHT_SH_hourly['time'].dtype)
#print(CFHT_hourly['time'][0])
    
#set index 
CFHT_SH_hourly.set_index('time', inplace=True)

# create a new column consisting of the cycle parameter of the correspondend entry
#for seasonal cycle (12 months), create column with "months"
CFHT_SH_hourly['months'] = pd.DatetimeIndex(CFHT_SH_hourly.index).month                                            

#for diurnal cycle (24 hours), create column with "hours"
CFHT_SH_hourly['hours'] = pd.DatetimeIndex(CFHT_SH_hourly.index).hour

# for monthly timeseries, create column with "YYYY-MM"
#CFHT_hourly['YYYY_MM'] = pd.DatetimeIndex()
CFHT_SH_hourly['YYYY_MM'] = CFHT_SH_hourly.index.strftime('%Y-%m')

# for 1 year averages, create column with years
CFHT_SH_hourly['years'] = pd.DatetimeIndex(CFHT_SH_hourly.index).year

CFHT_SH_hourly_preped = CFHT_SH_hourly
#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(CFHT_SH_hourly_preped, df_prep_SH_600,
df_prep_SH_650, df_prep_SH_700, df_prep_SH_750)

# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Mauna Kea', 'specific_humidity', 
           './sites/MaunaKea/Output/Plots/SH/diurnal_cycle_UTC_SH_all_MK_masked_2000to2020.pdf', 
           '600hPa', '650hPa', '700hPa', '750hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Mauna Kea', 'specific_humidity', 
           './sites/MaunaKea/Output/Plots/SH/seasonal_cycle_UTC_SH_all_MK_masked_2000to2020.pdf',
           '600hPa', '650hPa', '700hPa', '750hPa')

# %%
plot_timeseries_merged('./sites/MaunaKea/Output/Plots/SH/Timeseries_UTC_SH_all_2000to2020.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 'specific_humidity', '600hPa', '650hPa', '700hPa', '750hPa')

# %%
# plot timeseries, Era5 data

# function assumes that specific humidity is in first column
CFHT_SH_hourly_preped.iloc[:,0] = CFHT_SH_hourly_preped.iloc[:,3]
CFHT_SH_hourly_preped.rename(columns={'Temp': 'specific_humidity'})
#%%
plot_timeseries_long('./sites/MaunaKea/Output/Plots/SH/timeseries_long_SH_all_MK_1979to2020.pdf',CFHT_SH_hourly_preped,
Era5_600hPa = df_prep_SH_600, Era5_650hPa = df_prep_SH_650,
Era5_700hPa = df_prep_SH_700,Era5_750hPa = df_prep_SH_750)

plot_timeseries_movav('./sites/MaunaKea/Output/Plots/SH/timeseries_Paranal_2000to2020_movav.pdf', yearly_grouped_SH,
 In_situ = 'specific_humidity', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_800hPa = '800hPa')

plot_timeseries_movav('./sites/MaunaKea/Output/Plots/SH/timeseries_Paranal_1979to2020_long.pdf', monthly_grouped_SH,yearly_grouped_SH,
 In_situ = 'specific_humidity', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_800hPa = '800hPa')


# %%
# correlation plot: 650hPa Era5 data and in-situ measurements
# use merged_df_SH, hourly data
plt.scatter(merged_df_SH['650hPa'], merged_df_SH['specific_humidity'], s = 1)
x = np.arange(0,40)
plt.plot(x,x, color = 'k')
plt.xlim(0 , 0.01)
plt.ylim(0, 0.01)
plt.xlabel('specific humidity of Era 5 650hPa pressure levels [kg/kg]')
plt.ylabel('specific humidity of in-situ measurements [kg/kg]')
plt.title('correlation plot specific humidity, Mauna Kea')
plt.savefig('./sites/MaunaKea/Output/Plots/SH/correlation_SH_all_MK_2000to2020.pdf')

# %%
# why does this not work?
def correlation_plot(mytitle, filename, data_x = x, data_y = y):
    plt.scatter(x, y, s = 1)
    x_eq_y = np.arange(0,100)
    plt.plot(x_eq_y, x_eq_y, color = 'k')
    plt.xlim(0 , x.max())
    plt.ylim(0, x.max())
    plt.xlabel(str(data_x))
    plt.ylabel(str(data_y))
    plt.title(mytitle)
    plt.savefig(filename)
    plt.close

#%%
from scipy.stats import pearsonr
def correlation_plot(mytitle, filename, data_x, string_x, data_y ,string_y):
    plt.scatter(data_x, data_y, s = 1)
    x_eq_y = np.arange(0,100)
    plt.plot(x_eq_y, x_eq_y, color = 'k')
    plt.xlim(0 , data_x.max())
    plt.ylim(0, data_x.max())
    plt.xlabel(string_x)
    plt.ylabel(string_y)
    plt.title(mytitle)
    plt.savefig(filename)
    plt.close

    corr, _ = pearsonr(data_x, data_y)
    print('Pearsons correlation: %.3f' % corr)
# %%
correlation_plot('correlation plot specific humidity, Mauna Kea',
'./sites/MaunaKea/Output/Plots/SH/correlation_SH_all_MK_2000to2020.pdf',
merged_df_SH['650hPa'], 'specific humidity of Era 5 650hPa pressure levels [kg/kg]',
merged_df_SH['specific_humidity'], 'specific humidity of in-situ measurements [kg/kg]')

# %% correlation

from scipy.stats import pearsonr

corr, _ = pearsonr(merged_df_SH['650hPa'], merged_df_SH['specific_humidity'])
print('Pearsons correlation: %.3f' % corr)


# %%
