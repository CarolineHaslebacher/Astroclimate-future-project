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
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000] # you forgot to download 400!

ds_P = []
for i in range(0,len(pr_levels)):
    path = './sites/Paranal/Era5_data/seeing/' + str(pr_levels[i]) + 'hPa/*.nc'
    ds = xr.open_mfdataset(path, combine = 'by_coords')
    ds_sel = ds.sel(expver = 1 ,longitude= -70.25,latitude= -24.75,method='nearest') # green point
    # select only UTC 00:00 to 09:00 (nighttime hours)
    #ds_sel_nighttime = ds_sel.where(ds_sel['time'].dt.hour <= 9) 
    # create new coordinate pressure "level"
    ds_P.append(ds_sel.assign_coords({"level": pr_levels[i]}))

    # concatenate DataArrays along new dimension 'level'
    # if i == 0:
    #   ds_full = ds_P
    # else:
    #   ds_full = xr.concat([ds_full, ds_P], 'level')

ds_full = xr.concat(ds_P, 'level')

    # ds_full is a dataArray that can be accessed via ds_full.variable[pressure level index][time index]
    # for example: ds_full.u[10][0]

    ## monthly means
    #ds_sel = ds_sel.resample(time = 'M').mean()
    
    # # append xarray's to list
    #Seeing_pressure_levels.append(ds_sel_nighttime)


#ds_fully_loaded = ds_full.load()

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

# # version with delta_z coming from hydrostatic equilibrium equation
# def Cn2_func_hydrostatic(T, P, u_i0, u_i1, v_i0, v_i1, T1, Pp1, Pm1, i): # P1 and T1 stand for the values from the next level, i+1
#   if int(i) == 0:
#     delta_z = (Pp1 - P)/(rho * g) # hydrostatic equilibrium, Euler forward
#   else:
#     delta_z =  0.5 * (Pp1 - Pm1)/(rho * g) # centered differences
#   # print(f'delta_z hydrostatic = {delta_z.values}')
#   Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
#   return(Cn2_var * delta_z)


def Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var): # P1 and T1 stand for the values from the next level, i+1
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  
  Cn2_var_profile = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  return(Cn2_var_profile)


# def delta_z_now(i, df_z_0, df_z_m1, df_z_p1): # df_z is the geopotential height (variable z/g in Era5) at one time, i describes the pressure level
#   if int(i) == 0:
#     delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
#   else:
#     delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
#   return(delta_z)  

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)

# #%% 

# # gravitational acceleration ERA5
# g = 9.80665 #m/s^2, from Era5 documentation
# k_var = 6
# RCp = 0.286 # R/cp
# P_0 = 1000 #mbar

# def theta(T, P):
#   return(T * (P_0 / P)**RCp)

# def E(u_i0, u_i1, v_i0, v_i1, delta_z):
#   return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )



# def delta_z_now(i, df_z_0, df_z_m1, df_z_p1): # df_z is the geopotential height (variable z/g in Era5) at one time, i describes the pressure level
#   if int(i) == 0:
#     delta_z = df_z_0 - df_z_p1#df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
#   else:
#     delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
#   return(delta_z)  

# def Cn2_func_dz(ds, pr_index, time_index): # P1 and T1 stand for the values from the next level, i+1
#   T = ds.t[pr_index][time_index]
#   P = ds.level[pr_index]
#   T1 = ds.t[int(pr_index + 1)][time_index]
#   P1 = ds.level[int(pr_index + 1)]
#   delta_z = delta_z_now(pr_index, ds.z[pr_index][time_index]/g, ds.z[int(pr_index - 1)][time_index]/g, ds.z[int(pr_index + 1)][time_index]/g)
#   E_value = E(ds.u[pr_index][time_index], ds.u[int(pr_index + 1)][time_index], ds.v[pr_index][time_index], ds.v[int(pr_index + 1)][time_index], delta_z) 
#   Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E_value / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
#   J_contribution = Cn2_var * delta_z
#   return(J_contribution)

# def epsilon(J):
#   # calculate seeing (epsilon) in arcsec
#   wv = 500 * 10**(-9) # wavelength 500nm
#   k_wave = 2 * np.pi /wv # wavenumber k
#   epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
#   return(epsilon)

#%% ####################################### integral of Cn2 ####################
# # calculate seeing for big dataset

# # variation of seeing calculation verified!

# # in hPa
# pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]

# Seeing_Era5 = []
# Seeing_time = []
# cont_J = []
# cont_delta_z = []

# for k in range(0, len(Seeing_pressure_levels[0].time)): # sample over time
#   J = 0
#   # why does this not work?
#   results = [Cn2_func_dz(ds_full, item, k) for item in range(0, len(pr_levels[:-2]))]
#   J = sum(results)
#   # store J as variable? 
  
#%%
#integrate until closest_value
# sample over pressure levels, up to 800hPa for a start

J = 0
#J_7 = 0
# J_hydrost = 0
Cn2_list = []
# for seeing value, integrate Cn2 over all pressure levels
for i in range(0, len(pr_levels[:-1])):

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
  
  # k = 7.0
  #k_var = 7
  #J_7 = J_7 + Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var)
  
  # J_hydrost = J_hydrost + Cn2_func_hydrostatic(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, Pm1, i)

  #Cn2_list.append(Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1))
  k_var = 6
  ds_Cn2 = Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var)
  ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels[i]})

  # concatenate DataArrays along new dimension 'level'
  if i == 0:
    ds_Cn2_profile = ds_Cn2
  else:
    ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')

ds_Cn2_profile = xr.Dataset({"Cn2": ds_Cn2_profile})
# calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
ds_seeing = xr.Dataset({"seeing": epsilon(J), "J": J})
 
#%%
# # save to netcdf
ds_seeing.encoding={'seeing': {'dtype': 'float32', 'scale_factor': 0.0001, '_FillValue': -9999}, 'J': {'dtype': 'float32', 'scale_factor': 10**(-20), '_FillValue': -9999}}

#ds_seeing.to_netcdf('./sites/Paranal/Data/Era_5/seeing/ERA5_Paranal_greenpoint_2_seeing_J_1979to2020.nc')
ds_seeing.to_netcdf('./sites/Paranal/Data/Era_5/seeing/ERA5_Paranal_greenpoint_to1000hPa_seeing_J_1979to2020.nc')

#ds_seeing_hydrost = xr.Dataset({"seeing_hydro": epsilon(J_hydrost)})

# for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension
# ds_Cn2 = xr.Dataset({"Cn2"})
# ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels[i]})


#%% define SCIDAR dates for ERA5 data (day plus 1 in UT)
SCIDAR_dates_ERA5 = []

for year in [2016, 2017, 2018]:
    if year == 2016:
        months_SCIDAR = [4,7,10,11,12]
    elif year == 2017:
        months_SCIDAR = [3,4,5,6,7,8,11,12]
    elif year == 2018:
        months_SCIDAR = [1]
    for month in months_SCIDAR:
        # 2016
        if (year == 2016) & (month == 4):
            days_SCIDAR = [26,27,28,29]
        elif (year == 2016) & (month == 7):
            days_SCIDAR = [22,23,24,25,26]
        elif (year == 2016) & (month == 10):
            days_SCIDAR = [30,31]
        elif (year == 2016) & (month == 11):
            days_SCIDAR = [1,2]
        elif (year == 2016) & (month == 12):
            days_SCIDAR = [10,11,12]
        #2017
        elif (year == 2017) & (month == 3):
            days_SCIDAR = [7,8,9]
        elif (year == 2017) & (month == 4):
            days_SCIDAR = [12,13,14,15,16,17,18]
        elif (year == 2017) & (month == 5):
            days_SCIDAR = [5,6,7,8,9]
        elif (year == 2017) & (month == 6):
            days_SCIDAR = [8,9,10]
        elif (year == 2017) & (month == 7):
            days_SCIDAR = [3,4,5,6,7,8,9]
        elif (year == 2017) & (month == 8):
            days_SCIDAR = [3,4,5,6,7,8]
        elif (year == 2017) & (month == 11):
            days_SCIDAR = [4,5,6,7,8,9, 18,19,20, 29,30]
        elif (year == 2017) & (month == 12):
            days_SCIDAR = [1,2, 5,6, 8,9,10,11,12,13,14,15,16,17,18]
        # 2018
        elif (year == 2018) & (month == 1):
            days_SCIDAR = [13,14,15, 18,19,20,21,22,23,24]

        for day in days_SCIDAR:
            SCIDAR_dates_ERA5.append('{}'.format(dt.date(year, month, day) + dt.timedelta(days = 1))) # day plus one for balancing 'wrong' filenames (wasn't wrong for Paranal)
        del(days_SCIDAR)

#%% extract Cn2 profile for SCIDAR dates only

datasets = []

for scid_date in SCIDAR_dates_ERA5:
  #geopot_height_sel = ds_full.z.isel(level=slice(None,-1))/g
  geopot_height_sel = ds_full.z/g
  geopot_height = geopot_height_sel.loc[:,scid_date] # take same values as in calculation of seeing and J
  Profile = ds_Cn2_profile.Cn2.loc[:,scid_date]
#Profile_2 = ds_Cn2_profile.Cn2.loc[:,'2016-07-22']

# # take mean over the night
#   geopot_height_mean = geopot_height[:,0:10].mean(dim = 'time')
#   Profile_night_mean = Profile[:,0:10].mean(dim = 'time')
  
  # write file to netcdf
  # Profile.to_netcdf('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_' + scid_date + '.nc')
  
  # or append information to xarray
  ds = xr.Dataset({"Cn2_profile": Profile, "geopotential_height": geopot_height})
  # ds = ds.assign_coords(level = geopot_height.loc[:,scid_date].mean(dim = 'time'))
  datasets.append(ds)

ds_Cn2_profile_SCIDAR = xr.concat(datasets, dim = 'time')

#ds_Cn2_profile_SCIDAR.to_netcdf('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_all_SCIDAR_dates_ERA5_24hours.nc')
ds_Cn2_profile_SCIDAR.to_netcdf('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_all_SCIDAR_dates_ERA5_24hours_to1000hPa.nc')

#%% plot profile with imshow

fig, ax1 = plt.subplots()
im = ax1.imshow(ds_Cn2_profile_SCIDAR.Cn2_profile.loc[:,'2016-04-27'])
ax1.set_ylim(0,22)
#ax1.set_xlim(0,10)
plt.gca().invert_yaxis()
 
# create new axes with same x-axis
ax2 = ax1.twinx()
# ax2.set_yticks(ticks = ds_Cn2_profile_SCIDAR.level, minor = False)

# plt.imshow(ds_Cn2_profile_SCIDAR.Cn2_profile.loc[:,'2016-04-26'], cmap = 'viridis')
#plt.xlim(0,10)

#plt.yticks(ticks = ds_Cn2_profile_SCIDAR.geopotential_height.loc[:,'2016-04-26'].mean(dim = 'time'))
fig.colorbar(im, ax=ax1)

fig.tight_layout() 
plt.savefig('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/2016-04-26_colorplot.png')
plt.show()

# or use xarray plot
# replace date by scid_date
ds_pl = ds_Cn2_profile_SCIDAR.Cn2_profile.loc[:800,'2016-04-27T00:00:00':'2016-04-27T10:00:00'] # SCIDAR data also goes from 00:00 to 10:00 UT
ds_mean_Cn2 = ds_pl.mean(dim = 'time')

Cn2_value = ds_seeing.J.loc['2016-04-27T00:00:00':'2016-04-27T10:00:00'].mean(dim = 'time').values

fig, a = plt.subplots(2,2,figsize=(15,7))
ds_pl.plot(ax = a[0][0])
a[0][0].invert_yaxis()
a[0][0].set_title(r'C$_{n}^2$ night profile')

ds_mean_Cn2.plot.line( '-ko',y = 'level',ax = a[0][1])
a[0][1].invert_yaxis()
a[0][1].set_title(r'Cn2 integral = ${}$'.format(Astroclimate_function_pool.as_si(Cn2_value, 2)))
a[0][1].set_xlabel(r'C$_{n}^2$ [m$^{1/3}$]')
a[0][1].set_ylabel('pressure level [hPa]')
a[0][1].set_xscale('log')

plt.tight_layout()
plt.savefig('./sites/Paranal/Output/Plots/seeing/SCIDAR/Cn2_Profile/2016-04-26_ERA5_Cn2_Profile.pdf')
plt.show()
# add scidar data
fig, a = plt.subplots()
ds_cn2_scidar.loc['2016-04-27T00:00:00':'2016-04-27T90:00:00', :].plot(ax = a, y = 'altitude') #(ax = a[1][0])

# drop nan
ds_cn2_scidar_nonan = ds_cn2_scidar.where(ds_cn2_scidar.loc['2016-04-27', :] != np.nan, drop = True)

import plotly.express as px
fig = px.imshow(ds_cn2_scidar, color_continuous_scale='RdBu_r')
fig.show()

plt.imshow(ds_cn2_scidar.loc['2016-04-27', :])


plt.imshow(ds_cn2_scidar_nonan)


#%% TEST for Cn2 Integral

def sum_Cn2_func(df_z_0, df_z_m1, df_z_p1, Cn2):
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1 #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  # print(f'delta_z = {delta_z.values}')
  return Cn2 * delta_z

ds_mean_geopot = ds_Cn2_profile_SCIDAR.geopotential_height.loc[:,'2016-04-27T00:00:00':'2016-04-27T10:00:00'].mean(dim = 'time') # SCIDAR data also goes from 00:00 to 10:00 UT

Cn2_integral = 0
for i in range(0, len(pr_levels[:-2])):
  if i == 0:
    df_z_m1 = 0
  else:
    df_z_m1 = ds_mean_geopot[i-1]
  df_z_0 = ds_mean_geopot[i]
  df_z_p1 = ds_mean_geopot[i+1]
  Cn2 = ds_mean_Cn2[i]

  Cn2_integral = Cn2_integral + sum_Cn2_func(df_z_0, df_z_m1, df_z_p1, Cn2)
print(Cn2_integral.values) # = 1.3592929181963962e-13

# test for Cn2 integral (integral from 0 to 950 hPa!): 
# ds_seeing.J.loc['2016-04-27T00:00:00':'2016-04-27T10:00:00'].mean(dim = 'time').values = 2.07897056e-13 (50 to 950hPa)

# 50 to 800hPa:
# J = 1.35931578e-13

# fig, ax = plt.subplots()
# plt.plot(Profile_1[:,5], geopot_1['z'][:-2], '-o')
# ax.set_xscale('log')


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

ds_free_seeing = xr.Dataset({"free_seeing": epsilon(J)})

#%%
# next step: J_free to xarray with variable J, time, lon, lat
# then, calculate epsilon(J).loc['2016-04-26'])

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
# ds_seeing = xr.Dataset({"seeing": epsilon(J)})
#ds_seeing = ds_J.assign({"J": J,"seeing": epsilon(J)})
df_seeing = ds_seeing.to_dask_dataframe()

# df_comp_seeing = df_prep(df_seeing, 'seeing', 'seeing') 

# make sure that J (second column) is not getting lost
df_seeing = df_seeing[['time', 'seeing', 'J']]
# df_seeing = df_seeing.set_index('time')
df_comp_seeing = df_seeing.compute()

df_nonan = df_comp_seeing.dropna(axis = 0, how = 'any') #dropnan's

# save to csv
df_comp_seeing.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_Cn2_2000to2020.csv', header = True, index = True)

df_seeing.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_dask_Seeing_integral_Cn2_2000to2020.csv', header = True)

df_nonan.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_Cn2_2000to2020_dropnan.csv', header = True, index = True)

# get value for 2017-01-10
# df_comp_seeing.loc[df_comp_seeing.index == dt.datetime(2017, 1, 10)]
# seeing value is: 0.261417
# J is 7.346058e-14

# hydrostat
# df_seeing_hydrost = ds_seeing_hydrost.to_dask_dataframe()
# df_comp_seeing_hydrost = df_prep(df_seeing_hydrost, 'seeing_hydro', 'seeing_hydro')

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#open in-situ measurements as pandas dataframe
seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv')
free_seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv')

#%% prepare Paranal calculated data

seeing_hourly_preped = mes_prep(seeing_hourly) # timezone = None, already in UTC
free_seeing_hourly_preped = mes_prep(free_seeing_hourly) # timezone = None, already in UTC

#%%
# # plot without computing dataframe
# plt.plot(free_seeing_hourly_preped.Cn2 * 10**(-15))
# plt.plot(ds_seeing.J.loc['2016-07-01':])
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
['Seeing Paranal'], yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_seeing = 'seeing', Era5_seeing_hydrostatic = 'seeing_hydro')



# Cn2
plot_timeseries_long('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_Cn2_2016to2020_long.pdf', yearly_grouped_seeing,
['Cn2'], yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_Cn2_integrated = 'J')


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


#%%
#################################################### CN2 ##########################3
#%% Cn2

#open in-situ measurements as pandas dataframe
free_seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv')

# prepare Paranal calculated data
free_seeing_hourly_preped = mes_prep(free_seeing_hourly) # timezone = None, already in UTC

# multiply Cn2 by 10**(-15)
free_seeing_hourly_preped['Cn2_15'] = free_seeing_hourly_preped['Cn2']*10**(-15)

# prepare df_comp_seeing
df_comp_seeing['time'] = pd.to_datetime(df_comp_seeing['time'])
df_comp_seeing.set_index('time', inplace=True)

#%%
# merge datasets
merged_df_seeing, seasonal_seeing, diurnal_seeing, monthly_grouped_seeing, yearly_grouped_seeing  = merge_df(free_seeing_hourly_preped,
df_seeing_dataframe, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_seeing_era5, monthly_grouped_seeing_era5, yearly_grouped_seeing_era5  = merge_df_long(df_seeing_dataframe)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_seeing,'diurnal cycle Cerro Paranal', 'Cn2_15', 
           './sites/Paranal/Output/Plots/seeing/diurnal_cycle_UTC_Cn2_instantaneous_Paranal_2000to2020.pdf', 
           'J')
           
# %% seasonal cycle
plot_cycle(seasonal_seeing,'seasonal cycle Cerro Paranal', 'Cn2_15', 
           './sites/Paranal/Output/Plots/seeing/seasonal_cycle_UTC_Cn2_instantaneous_Paranal_2000to2020.pdf', 
           'J')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_free_seeing_2015to2019_movav.pdf', yearly_grouped_free_seeing,
# In_situ = 'seeing Paranal', Era5_FA_seeing = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_Cn2_instantaneous_2000to2020_long.pdf', yearly_grouped_seeing,
['Cn2_15'], yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_Cn2_integrated_to_800hPa = 'J')


# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'J_5'
insitu_param = 'Cn2_15'
path = './sites/Paranal/Output/Plots/seeing/correlation_Paranal_Cn2' + parameter +'_2000to2020_'
xax = r'Era 5 FA seeing [$m^{1/3}$]'
yax = r'in-situ FA seeing [$m^{1/3}$]'

corr_plots_hourly_monthly_yearly(path, merged_df_seeing, monthly_grouped_seeing, 
yearly_grouped_seeing, parameter, insitu_param, xax, yax)



