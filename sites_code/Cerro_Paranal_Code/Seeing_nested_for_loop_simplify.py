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
# test = ds_full.u[:20, :20]

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

# Seeing_pressure_levels = []

def read_ERA5_seeing_data(pressure_levels_list, base_path):

  # in hPa
  #pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
  pr_levels = pressure_levels_list

  for i in range(0,len(pr_levels)):
    #'./sites/Paranal/Era5_data/seeing/'
    path = str(base_path) + str(pr_levels[i]) + 'hPa/*.nc'
    ds = xr.open_mfdataset(path, combine = 'by_coords')
    ds_sel = ds.sel(expver = 5, longitude= -70.25,latitude= -24.75,method='nearest') # green point
    
    # select only UTC 00:00 to 09:00 (nighttime hours)
    #ds_sel_nighttime = ds_sel.where(ds_sel['time'].dt.hour <= 9) 
    
    # create new coordinate pressure "level"
    ds_P = ds_sel.assign_coords({"level": pr_levels[i]})

    # concatenate DataArrays along new dimension 'level'
    if i == 0:
      ds_full = ds_P
    else:
      ds_full = xr.concat([ds_full, ds_P], 'level')

    # ds_full is a dataArray that can be accessed via ds_full.variable[pressure level index][time index]
    # for example: ds_full.u[10][0]

    ## monthly means
    #ds_sel = ds_sel.resample(time = 'M').mean()
    
    # # append xarray's to list
    #Seeing_pressure_levels.append(ds_sel_nighttime)

  #df_full = ds_full.to_dask_dataframe()

  return ds_full

#%% formula implementation native
# gravitational acceleration ERA5
g = 9.80665 #m/s^2, from Era5 documentation
k_var = 6
RCp = 0.286 # R/cp
P_0 = 1000 #mbar

def theta(T, P):
  RCp = 0.286 # R/cp
  P_0 = 1000 #mbar

  return(T * (P_0 / P)**RCp)

def E(u_i0, u_i1, v_i0, v_i1, delta_z):
  return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )

def Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1): # P1 and T1 stand for the values from the next level, i+1
  g = 9.80665 #m/s^2, from Era5 documentation 
  k_var = 6 # from Osborn 2018
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  
  Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  return(Cn2_var * delta_z)

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)
  
#%%
#integrate until closest_value
# sample over pressure levels, up to end of pr_levels_list
def ERA5_seeing_calc(ds_full, pr_levels_list):
  J = 0
  # for seeing value, integrate Cn2 over all pressure levels
  #pr_levels_list = pr_levels[:-2]
  
  for i in range(0, len(pr_levels_list)):
    T = ds_full.t[i]
    P = ds_full.level[i]
    u_i0 = ds_full.u[i]
    u_i1 = ds_full.u[int(i+1)]
    v_i0 = ds_full.v[i]    
    v_i1 = ds_full.v[int(i+1)]
    T1 = ds_full.t[int(i+1)] 
    P1 = ds_full.level[int(i+1)] 
    df_z_0 = ds_full.z[i]/g
    if i == 0:
      df_z_m1 = 0
    else:
      df_z_m1 = ds_full.z[i-1]/g
    df_z_p1 = ds_full.z[i+1]/g
    # integrate (sum)
    J = J + Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)

    # for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension

  # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
  ds_seeing = xr.Dataset({"seeing": epsilon(J)})

  return ds_seeing

#%%
# use functions
# lon = 
# lat = 
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
ds_paranal = read_ERA5_seeing_data(pr_levels, './sites/Paranal/Era5_data/seeing/')
ds_seeing = ERA5_seeing_calc(ds_paranal, pr_levels[:-2])

# # save dataset
# ds_seeing.to_netcdf('...path...')

#%%
# for free seeing, integrate only up to an altitude of 3.6km

# first, find pressure level which is nearest to 3600m
for p_index in range(0, len(pr_levels)):
  if (ds_full.z[p_index]/g).mean() < 3600:
    print('the value of altitude which is smaller than 3600m is {}'.format((ds_full.z[p_index]/g).mean().values))
    print('the index is {}'.format(p_index))
    break

# max altitude of index 14 is 3818m
# max altitude of index 15 is 3200m
# mean altitude of index 14 is 3748m
# mean altitude of index 15 is 3138m
# --> sum until and with index 14, which is closer to 3600m than index 15 for most values

#%% integral of free atmosphere seeing

J_free = 0
# for seeing value, integrate Cn2 over all pressure levels
for i in range(0, p_index):
    T = ds_full.t[i]
    P = ds_full.level[i]
    u_i0 = ds_full.u[i]
    u_i1 = ds_full.u[int(i+1)]
    v_i0 = ds_full.v[i]    
    v_i1 = ds_full.v[int(i+1)]
    T1 = ds_full.t[int(i+1)] 
    P1 = ds_full.level[int(i+1)] 
    df_z_0 = ds_full.z[i]/g

    # initialize delta z if the index is 0 and we cannot take the index 0-1
    if i == 0:
      df_z_m1 = 0
    else:
      df_z_m1 = ds_full.z[i-1]/g

    df_z_p1 = ds_full.z[i+1]/g
    # integrate (sum)
    J_free = J_free + Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)

# ds_free_seeing = xr.Dataset({"free_seeing": epsilon(J_free)})

#%%
# next step: J_free to xarray with variable J, time, lon, lat
# then, calculate epsilon(J)
ds_free_seeing = xr.Dataset({"free_seeing": epsilon(J_free)})
#ds_free_seeing = ds_J.assign({"J": J,"free_seeing": epsilon(J)})
df_free_seeing = ds_free_seeing.to_dask_dataframe()

df_comp_free_seeing = df_prep(df_free_seeing, 'free_seeing', 'free_seeing') #dropnan's
df_nonan_free = df_comp_free_seeing.dropna(axis = 0, how = 'any')

# save to csv
df_comp_free_seeing.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_free_Seeing_integral_2000to2020.csv', header = True, index = True)
df_nonan_free.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_free_Seeing_integral_2000to2020_dropnan.csv', header = True, index = True)

# get value for 2017-01-10
df_comp_free_seeing.loc[df_comp_free_seeing.index == dt.datetime(2017, 1, 10)]
# seeing value is: 0.199526


#%%
#next step: J to xarray with variable J, time, lon, lat
# then, calculate epsilon(J)
#ds_seeing = xr.Dataset({"seeing": epsilon(J)})

#ds_seeing = ds_J.assign({"J": J,"seeing": epsilon(J)})
df_seeing = ds_seeing.to_dask_dataframe()

df_comp_seeing = df_prep(df_seeing, 'seeing', 'seeing') #dropnan's
df_nonan = df_comp_seeing.dropna(axis = 0, how = 'any')

# save to csv
df_comp_seeing.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2000to2020.csv', header = True, index = True)
df_nonan.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2000to2020_dropnan.csv', header = True, index = True)

# get value for 2017-01-10
df_comp_seeing.loc[df_comp_seeing.index == dt.datetime(2017, 1, 10)]
# seeing value is: 0.261417
# J is 7.346058e-14

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# open calculated seeing ERA5 data
df_comp_seeing_csv = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2000to2020.csv', index_col='time', parse_dates=True)
df_comp_free_seeing = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_free_Seeing_integral_2000to2020.csv')


#open in-situ measurements as pandas dataframe
seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv')
free_seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv')

#%% prepare Paranal calculated data

seeing_hourly_preped = mes_prep(seeing_hourly) # timezone = None, already in UTC
free_seeing_hourly_preped = mes_prep(free_seeing_hourly) # timezone = None, already in UTC

#%% seeing all
# merge datasets
merged_df_seeing, seasonal_seeing, diurnal_seeing, monthly_grouped_seeing, yearly_grouped_seeing  = merge_df(seeing_hourly_preped,
df_comp_seeing, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_seeing_era5, monthly_grouped_seeing_era5, yearly_grouped_seeing_era5  = merge_df_long(df_comp_seeing)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_seeing,'diurnal cycle Cerro Paranal', 'Seeing Paranal', 
           './sites/Paranal/Output/Plots/seeing/diurnal_cycle_UTC_seeing_Paranal_2000to2020.pdf', 
           'seeing')
           
# %% seasonal cycle
plot_cycle(seasonal_seeing,'seasonal cycle Cerro Paranal', 'Seeing Paranal', 
           './sites/Paranal/Output/Plots/seeing/seasonal_cycle_UTC_seeing_Paranal_2000to2020.pdf', 
           'seeing')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_seeing_2015to2019_movav.pdf', yearly_grouped_seeing,
# In_situ = 'seeing Paranal', Era5_seeing = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_seeing_2000to2020_long.pdf', yearly_grouped_seeing,
'Seeing Paranal', yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_seeing = 'seeing')


# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'seeing'
path = './sites/Paranal/Output/Plots/seeing/correlation_Paranal_' + parameter +'_2000to2020_'
xax = 'Era 5 seeing [arcsec]'
yax = 'in-situ seeing [arcsec]'

#hourly
# drop nans of merged df to make dataframes equal in size
merged_df_ESO =  merged_df_seeing[[parameter, 'Seeing Paranal']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path +'hourly.pdf',
'hourly means',
merged_df_ESO[parameter], xax,
merged_df_ESO['Seeing Paranal'], yax)

# monthly
monthly_corr = monthly_grouped_seeing[[parameter, 'Seeing Paranal']].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

correlation_plot(path + '_monthly.pdf',
'monthly means',
monthly_corr[parameter]['mean'],  xax,
monthly_corr['Seeing Paranal']['mean'], yax)

correlation_plot(path + 'yearly.pdf',
'yearly means',
yearly_grouped_seeing[parameter]['mean'],  xax,
yearly_grouped_seeing['Seeing Paranal']['mean'], yax)

# %% ############################# SEEING FREE ATMOSPHERE ##################################

# read calculated ERA5 data:
Era5_FA_seeing = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_free_Seeing_integral_2000to2020_dropnan.csv')

# prepare Era5 free seeing csv data
Era5_FA_seeing['time'] = pd.to_datetime(Era5_FA_seeing['time'])
Era5_FA_seeing.set_index('time', inplace=True)
Era5_FA_seeing_preped = Era5_FA_seeing

#%%
# merge datasets
merged_df_free_seeing, seasonal_free_seeing, diurnal_free_seeing, monthly_grouped_free_seeing, yearly_grouped_free_seeing  = merge_df(free_seeing_hourly_preped,
Era5_FA_seeing_preped, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979) with months, years,... rows for timeseries
merged_df_free_seeing_era5, monthly_grouped_free_seeing_era5, yearly_grouped_free_seeing_era5  = merge_df_long(Era5_FA_seeing_preped)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_free_seeing,'diurnal cycle Cerro Paranal', 'free_atmosphere_seeing', 
           './sites/Paranal/Output/Plots/seeing/diurnal_cycle_UTC_FA_seeing_instantaneous_Paranal_2000to2020.pdf', 
           'free_seeing')
           
# %% seasonal cycle
plot_cycle(seasonal_free_seeing,'seasonal cycle Cerro Paranal', 'free_atmosphere_seeing', 
           './sites/Paranal/Output/Plots/seeing/seasonal_cycle_UTC_FA_seeing_instantaneous_Paranal_2000to2020.pdf', 
           'free_seeing')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_free_seeing_2015to2019_movav.pdf', yearly_grouped_free_seeing,
# In_situ = 'seeing Paranal', Era5_FA_seeing = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_FA_seeing_instantaneous_2000to2020_long.pdf', yearly_grouped_free_seeing,
['free_atmosphere_seeing'], yearly_grouped_free_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_FA_seeing = 'free_seeing')


# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'free_seeing'
insitu_param = 'free_atmosphere_seeing'
path = './sites/Paranal/Output/Plots/seeing/correlation_Paranal_FA_' + parameter +'_2000to2020_'
xax = 'Era 5 FA seeing [arcsec]'
yax = 'in-situ FA seeing [arcsec]'

corr_plots_hourly_monthly_yearly(path, merged_df_free_seeing, monthly_grouped_free_seeing, 
yearly_grouped_free_seeing, parameter, insitu_param, xax, yax)



# %%
