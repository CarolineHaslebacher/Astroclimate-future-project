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
os.getcwd()
# open NETCDF files on 600hPa to 750hPa
ds_SH_700 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/700hPa/*.nc', combine = 'by_coords')
ds_SH_750 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')
ds_SH_800 =xr.open_mfdataset('./sites/Paranal/Data/Era_5/SH/800hPa/*.nc', combine = 'by_coords')
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#open in-situ measurements as pandas dataframe
SH_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/Specific_humidity_Paranal_ESO_2000to2019.csv')


#%% convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# not nearest,  but 'green' point, higher elevated

# df_SH_700 = netcdf_to_df(ds_SH_700, -70.25, -24.75)
# df_prep_SH_700 = df_prep(df_SH_700, 'q', '700hPa')

# df_SH_750 = netcdf_to_df(ds_SH_750, -70.25, -24.75)
# df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

# df_SH_800 = netcdf_to_df(ds_SH_800, -70.25, -24.75)
# df_prep_SH_800 = df_prep(df_SH_800, 'q', '800hPa')

# print('netcdf to df done')

#%% nearest
df_SH_700 = netcdf_to_df(ds_SH_700, -70.5, -24.75)
df_prep_SH_700 = df_prep(df_SH_700, 'q', '700hPa')

df_SH_750 = netcdf_to_df(ds_SH_750, -70.5, -24.75)
df_prep_SH_750 = df_prep(df_SH_750, 'q', '750hPa')

df_SH_800 = netcdf_to_df(ds_SH_800, -70.5, -24.75)
df_prep_SH_800 = df_prep(df_SH_800, 'q', '800hPa')

print('netcdf to df done')

#%% prepare Paranal calculated data

# do not shift specific humidity Paranal array by 10hours, it is already in UTC! only add month and hours
SH_hourly['time'] = pd.to_datetime(SH_hourly['time']) 

#check the format
print(SH_hourly['time'].dtype)
    
#set index 
SH_hourly.set_index('time', inplace=True)

# create a new column consisting of the cycle parameter of the correspondend entry
#for seasonal cycle (12 months), create column with "months"
SH_hourly['months'] = pd.DatetimeIndex(SH_hourly.index).month                                            

#for diurnal cycle (24 hours), create column with "hours"
SH_hourly['hours'] = pd.DatetimeIndex(SH_hourly.index).hour

# for monthly timeseries, create column with "YYYY-MM"
SH_hourly['YYYY_MM'] = SH_hourly.index.strftime('%Y-%m')

# for 1 year averages, create column with years
SH_hourly['years'] = pd.DatetimeIndex(SH_hourly.index).year

SH_hourly_preped = SH_hourly

#%%
# merge datasets
merged_df_SH, seasonal_SH, diurnal_SH, monthly_grouped_SH, yearly_grouped_SH  = merge_df(SH_hourly_preped,
df_prep_SH_700, df_prep_SH_750, df_prep_SH_800, dropnan = False)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_SH_era5, monthly_grouped_SH_era5, yearly_grouped_SH_era5  = merge_df_long(df_prep_SH_700, df_prep_SH_750, df_prep_SH_800)


# %%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_SH,'diurnal cycle Cerro Paranal', 'specific_humidity', 
           './sites/Paranal/Output/Plots/SH/diurnal_cycle_UTC_SH_Paranal_2000to2019.pdf', 
           '700hPa', '750hPa', '800hPa')
           
# ./sites/MaunaKea/Output/Plots/SH/
# %%
plot_cycle(seasonal_SH,'seasonal cycle Cerro Paranal', 'specific_humidity', 
           './sites/Paranal/Output/Plots/SH/seasonal_cycle_UTC_SH_Paranal_2000to2019.pdf',
           '700hPa', '750hPa', '800hPa')

# %%
#plot_timeseries_merged('./sites/Paranal/Output/Plots/SH/Timeseries_UTC_SH_all_2000to2019.pdf', merged_df_SH, monthly_grouped_SH, yearly_grouped_SH, 
#'specific_humidity', '700hPa', '750hPa', '800hPa')

# %%
# plot timeseries, moving average

# # function assumes that specific humidity is in first column
# SH_hourly_preped.iloc[:,0] = SH_hourly_preped.iloc[:,6]
# SH_hourly_preped.rename(columns={'Unnamed: 0': 'specific_humidity'})

plot_timeseries_movav('./sites/Paranal/Output/Plots/SH/timeseries_Paranal_2000to2019_movav.pdf', yearly_grouped_SH,
 In_situ = 'specific_humidity', Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_800hPa = '800hPa')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/SH/timeseries_Paranal_1979to2019_long.pdf', yearly_grouped_SH,
'specific_humidity', yearly_grouped_SH_era5.loc[:'2019-12-31'], moving = True, Era5_700hPa = '700hPa', Era5_750hPa = '750hPa', Era5_800hPa = '800hPa')

# %% correlation

pressure_level = '750hPa'
#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_SH[[pressure_level, 'specific_humidity']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot('./sites/Paranal/Output/Plots/SH/correlation_SH_Paranal_' + pressure_level +'_2000to2020_hourly.pdf',
'hourly means',
merged_df_ESO[pressure_level], 'specific humidity Era 5 ' + pressure_level +' pressure level [kg/kg]',
merged_df_ESO['specific_humidity'], 'specific humidity in-situ [kg/kg]')

# monthly
monthly_corr = monthly_grouped_SH[[pressure_level, 'specific_humidity']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot('./sites/Paranal/Output/Plots/SH/correlation_SH_Paranal_' +  pressure_level +'_2000to2020_monthly.pdf',
'monthly means',
monthly_corr[pressure_level]['mean'], 'specific humidity Era 5 ' +  pressure_level +' pressure level [kg/kg]',
monthly_corr['specific_humidity']['mean'], 'specific humidity in-situ [kg/kg]')

# yearly
correlation_plot('./sites/Paranal/Output/Plots/SH/correlation_SH_Paranal_' +  pressure_level +'_2000to2020_yearly.pdf',
'yearly means',
yearly_grouped_SH[pressure_level]['mean'], 'specific humidity Era 5 ' +  pressure_level +' pressure level [kg/kg]',
yearly_grouped_SH['specific_humidity']['mean'], 'specific humidity in-situ [kg/kg]')

#%%
# test 5-year running average
# use pandas rolling mean function

#df_yearly_gr = yearly_grouped_SH.rolling(5, min_periods = 5).mean().shift(-2)

# df_res = merged_df_SH.resample('1Y').mean()
# df_roll = df_res.rolling(5, min_periods = 5).mean().shift(-2)
# df_res_r = merged_df_SH.resample('5Y').mean()
# df_from_scratch = df_prep_SH_700.loc[:'2019-12-31'].rolling(5*365).mean()

#%% plot it
# plt.plot(df_roll.index, df_roll['700hPa'], label = '700hPa rolling')
# plt.plot(df_res.index, df_res['700hPa'], label = '700hPa yearly')
# plt.plot(df_res_r.index, df_res_r['700hPa'],'-o', label = '700hPa 5-yearly')
# plt.plot(df_prep_SH_700.loc[:'2019-12-31'].resample('5Y').mean(), '-o',  label = '5-yearly df_prep', markersize = '7')

# plt.plot(pd.to_datetime(df_yearly_gr.index.astype(int), format = '%Y'), df_yearly_gr['700hPa']['mean'], '-o',  label = 'Era 5 700hPa', markersize = '3')
# plt.plot(pd.to_datetime(df_yearly_gr.index.astype(int), format = '%Y'), df_yearly_gr['specific_humidity']['mean'], '-o',  label = 'ESO Paranal', markersize = '3')
# plt.plot(pd.to_datetime(df_yearly_gr.index.astype(int), format = '%Y'), df_yearly_gr['750hPa']['mean'], '-o',  label = 'Era 5 750hPa', markersize = '3')

# plt.plot(df_roll.index, df_roll['specific_humidity'], label = 'ESO rolling')
# plt.plot(df_res.index, df_res['specific_humidity'], label = 'ESO yearly')
# plt.plot(df_res_r.index, df_res_r['specific_humidity'],'-o', label = 'ESO 5-yearly')

#%%
# check for a trend with linear regression

# for machine learning
# import seaborn as seabornInstance 
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics

# plt.figure(figsize=(15,10))
# plt.tight_layout()
# # average maximum temperature
# seabornInstance.distplot(yearly_grouped_SH['specific_humidity']['max'])
# plt.legend()

# LinearRegression(varx[mask], vary[mask])
# reg = LinearRegression().fit(varx[mask], vary[mask])
#%% standard linear regression formula

# varx = SH_hourly.index
# vary = SH_hourly['specific_humidity']
# # data['days_since'] = (data.date - pd.to_datetime('2003-02-25') ).astype('timedelta64[D]')
# from scipy import stats
# mask = ~np.isnan(varx) & ~np.isnan(vary)
# slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

# plt.plot(varx, vary[mask])
# plt.plot(varx, slope * varx[mask] + intercept)



# # linreg(SH_hourly.index, SH_hourly['specific_humidity'])
# linreg(df_yearly_gr.index ,df_yearly_gr['specific_humidity']['mean'])
# linreg(df_yearly_gr.index ,df_yearly_gr['700hPa']['mean'])
# linreg(df_yearly_gr.index ,df_yearly_gr['750hPa']['mean'])
# linreg(df_yearly_gr.index ,df_yearly_gr['800hPa']['mean'])
# %%
# why does this not work?
# def correlation_plot(mytitle, filename, x = 'string_x', y = 'string_y'):
#     plt.scatter(x, y, s = 1)
#     x_eq_y = np.arange(0,100)
#     plt.plot(x_eq_y, x_eq_y, color = 'k')
#     plt.xlim(0 , x.max())
#     plt.ylim(0, x.max())
#     plt.xlabel(string_x)
#     plt.ylabel(string_y)
#     plt.title(mytitle)
#     plt.savefig(filename)
#     plt.close

    
# %% correlation

# from scipy.stats import pearsonr
# # [['700hPa', 'specific_humidity']]
# merged_df_700_ESO =  merged_df_SH[['700hPa', 'specific_humidity']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

# # merged_df_SH_nonan = merged_df_SH.dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)
# corr, p_1 = pearsonr(merged_df_700_ESO['700hPa'], merged_df_700_ESO['specific_humidity'])
# print('Pearsons correlation: %.3f' % corr)
# r, p_2, lo, hi = pearsonr_ci(merged_df_700_ESO['700hPa'],merged_df_700_ESO['specific_humidity'],alpha=0.05)


# %%



# %%
