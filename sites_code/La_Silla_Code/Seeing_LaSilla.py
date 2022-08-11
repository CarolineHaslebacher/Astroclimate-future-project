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

# in hPa
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825]

for i in range(0,len(pr_levels)):
  path = './sites/Paranal/Era5_data/seeing/' + str(pr_levels[i]) + 'hPa/*.nc'
  ds = xr.open_mfdataset(path, combine = 'by_coords')
  ds_sel = ds.sel(expver = 1 ,longitude= -70.74,latitude= -29.26,method='nearest')
  # select only UTC 00:00 to 09:00 (nighttime hours)
  # ds_sel_nighttime = ds_sel.where(ds_sel['time'].dt.hour <= 9) 
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

# load full dataset into memory  
# ds_fully_loaded_LaSilla = ds_full.load()

# df_full = ds_full.to_dask_dataframe()

#%% formula implementation native
# gravitational acceleration ERA5
g = 9.80665 #m/s^2, from Era5 documentation
k_var = 6
RCp = 0.286 # R/cp
P_0 = 1000 #mbar
rho = 1.225 # kg/m^3, density of air

def theta(T, P):
  return(T * (P_0 / P)**RCp)

def E(u_i0, u_i1, v_i0, v_i1, delta_z):
  return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )

def Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var): # P1 and T1 stand for the values from the next level, i+1
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  # print(f'delta_z = {delta_z.values}')
  Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  return(Cn2_var * delta_z)

def Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var): # P1 and T1 stand for the values from the next level, i+1
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  
  Cn2_var_profile = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  return(Cn2_var_profile)

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)
  
#%%
#integrate until closest_value
# sample over pressure levels, up to 800hPa for a start

J = 0

# for seeing value, integrate Cn2 over all pressure levels
for i in range(0, len(pr_levels[:-2])):

  T = ds_full.t[i]
  P = ds_full.level[i]
  u_i0 = ds_full.u[i]
  u_i1 = ds_full.u[int(i+1)]
  v_i0 = ds_full.v[i]    
  v_i1 = ds_full.v[int(i+1)]
  T1 = ds_full.t[int(i+1)] 
  P1 = ds_full.level[int(i+1)] 
  Pm1 = ds_full.level[int(i-1)] 
  df_z_0 = ds_full.z[i]/g
  if i == 0:
    df_z_m1 = 0
  else:
    df_z_m1 = ds_full.z[i-1]/g
  df_z_p1 = ds_full.z[i+1]/g
  # integrate (sum)

  # k = 6.0
  k_var = 6
  J = J + Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var)
  
  k_var = 6
  ds_Cn2 = Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var)
  ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels[i]})

  # concatenate DataArrays along new dimension 'level'
  if i == 0:
    ds_Cn2_profile = ds_Cn2
  else:
    ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')
# create xarray dataset with Cn2 variable and pressure level coordinates
ds_Cn2_profile = xr.Dataset({"Cn2": ds_Cn2_profile})

# calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
ds_seeing = xr.Dataset({"seeing": epsilon(J), "J": J})

#%%
# to Dataframe
# df_seeing = ds_seeing.to_dataframe()

df_seeing = ds_seeing.to_dask_dataframe()

# df_comp_seeing = df_prep(df_seeing, 'seeing', 'seeing') 

# make sure that J (second column) is not getting lost
df_seeing = df_seeing[['time', 'seeing', 'J']]
# df_seeing = df_seeing.set_index('time')
df_comp_seeing = df_seeing.compute()


# save to csv
df_comp_seeing.to_csv('./sites/La_Silla/Data/Era_5/seeing/hourly_Seeing_integral_Cn2_1979to2020.csv', header = True)

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#open in-situ measurements as pandas dataframe
seeing_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_Seeing.csv')

#%% prepare La_Silla calculated data

seeing_hourly_preped = mes_prep(seeing_hourly) # timezone = None, already in UTC

#%% seeing all

df_comp_seeing = df_comp_seeing.set_index('time')

#%%
# merge datasets
merged_df_seeing, seasonal_seeing, diurnal_seeing, monthly_grouped_seeing, yearly_grouped_seeing  = merge_df(seeing_hourly_preped,
df_comp_seeing[['seeing', 'J']], dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_seeing_era5, monthly_grouped_seeing_era5, yearly_grouped_seeing_era5  = merge_df_long(df_comp_seeing[['seeing', 'J']])


# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_seeing,'diurnal cycle Cerro La_Silla', 'Seeing La_Silla', 
           './sites/La_Silla/Output/Plots/seeing/diurnal_cycle_UTC_seeing_La_Silla_2000to2020.pdf', 
           'seeing')
           
# %% seasonal cycle
plot_cycle(seasonal_seeing,'seasonal cycle Cerro La_Silla', 'Seeing La_Silla', 
           './sites/La_Silla/Output/Plots/seeing/seasonal_cycle_UTC_seeing_La_Silla_2000to2020.pdf', 
           'seeing')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/La_Silla/Output/Plots/seeing/timeseries_La_Silla_seeing_2015to2019_movav.pdf', yearly_grouped_seeing,
# In_situ = 'seeing La_Silla', Era5_seeing = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/La_Silla/Output/Plots/seeing/timeseries_La_Silla_seeing_2000to2020_long.pdf', yearly_grouped_seeing, # select full years only!
['Seeing La_Silla'], yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_seeing = 'seeing')

# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'seeing'
insitu_param = 'Seeing La_Silla'
path = './sites/La_Silla/Output/Plots/seeing/correlation_La_Silla_' + parameter +'_2000to2020_'
xax = 'Era 5 seeing [arcsec]'
yax = 'in-situ seeing [arcsec]'

corr_plots_hourly_monthly_yearly(path, merged_df_seeing, monthly_grouped_seeing, yearly_grouped_seeing, parameter, insitu_param, xax, yax)


# %%
