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

Seeing_pressure_levels = []
 
# in hPa
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875]

for i in range(0,len(pr_levels)):
    path = './sites/Paranal/Era5_data/seeing/' + str(pr_levels[i]) + 'hPa/*.nc'
    ds = xr.open_mfdataset(path, combine = 'by_coords')
    ds_sel = ds.sel(expver = 1 ,longitude= -70.25,latitude= -24.75,method='nearest') # green point
    # select only UTC 00:00 to 09:00 (nighttime hours)
    ds_sel_nighttime = ds_sel.where(ds_sel['time'].dt.hour <= 9) 
    # create new coordinate pressure "level"
    ds_P = ds_sel_nighttime.assign_coords({"level": pr_levels[i]})

    # concatenate DataArrays along new dimension 'level'
    if i == 0:
      ds_full = ds_P
    else:
      ds_full = xr.concat([ds_full, ds_P], 'level')

    # ds_full is a dataArray that can be accessed via ds_full.variable[pressure level index][time index]
    # for example: ds_full.u[10][0]

    #ds_sel = ds_sel.resample(time = 'M').mean()
    Seeing_pressure_levels.append(ds_sel_nighttime)

#%% select only nighttime values

time = pd.date_range('2000-01-01', freq='6H', periods=365 * 4)
ds = xr.Dataset({'foo': ('time', np.arange(365 * 4)), 'time': time})

ds.sel(time = dt.time(9))

ds.sel(time = dt.time(9))
ds.loc['2000-01-01':'2000-01-02']

nov_1_10 = ds['foo'][(ds['time'].dt.hour <= 9)]
test = ds.where(ds['time'].dt.hour <= 9) #.dropna('time', how = 'all')
#ds_sel.where(ds.time < ds.dt.time(9))
#ds_sel.sel(time.dt.hour = [0:10])
#ds_sel.time.dt.hour[0:10]

#%% put pressure back to xarray and do not create a list

# create new coordinate
ds_sel_nighttime.assign_coords({"level": 2})

# concatenate along time
ds_1 = Seeing_pressure_levels[0]
ds_2 = Seeing_pressure_levels[1]

xr.concat([ds_1, ds_2], 'level')
result = [xr.concat([ds_1, ds_2], 'level') for ds_1, ds_2 in Seeing_pressure_levels]


#%%
#open in-situ measurements as pandas dataframe
seeing_hourly = pd.read_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv')


#%% formula implementation native
# gravitational acceleration ERA5
g = 9.80665 #m/s^2, from Era5 documentation
k_var = 6
RCp = 0.286 # R/cp
P_0 = 1000 #mbar

def theta(T, P):
  return(T * (P_0 / P)**RCp)

def E(u_i0, u_i1, v_i0, v_i1, delta_z):
  return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )

def Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, delta_z, T1, P1): # P1 and T1 stand for the values from the next level, i+1
  Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  return(Cn2_var)

def delta_z_now(i, df_z_0, df_z_m1, df_z_p1): # df_z is the geopotential height (variable z/g in Era5) at one time, i describes the pressure level
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1#df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  return(delta_z)  

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)


#%% calculate seeing for big dataset
#Seeing_pressure_levels[i].u[k].values

# in hPa
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]

Seeing_Era5 = []
Seeing_time = []
cont_J = []
cont_delta_z = []

for k in range(0, 20):#len(Seeing_pressure_levels[0].time)): # sample over time
  J = 0
  
  # print('T ', Seeing_pressure_levels[i].t[k].values)
  # print('P', pr_levels[i])
  # print('u_i ', Seeing_pressure_levels[i].u[k].values)
  # print('u_i+1 ', Seeing_pressure_levels[int(i+1)].u[k].values)
  # print('v_i ', Seeing_pressure_levels[i].v[k].values)
  # print('v_i+1 ' , Seeing_pressure_levels[int(i+1)].v[k].values)
  # print( 'T_i+1', Seeing_pressure_levels[int(i+1)].t[k].values)
  # print('P_i+1', pr_levels[int(i+1)])
  # print('theta(T,P) ', theta(Seeing_pressure_levels[i].t[k].values , pr_levels[i]))
  # print('E', E(Seeing_pressure_levels[i].u[k].values, Seeing_pressure_levels[int(i+1)].u[k].values, Seeing_pressure_levels[i].v[k].values, Seeing_pressure_levels[int(i+1)].v[k].values, delta_z))
  # print('dtheta/dz ',(theta( Seeing_pressure_levels[int(i+1)].t[k].values, pr_levels[int(i+1)])
  # - theta(Seeing_pressure_levels[i].t[k].values , pr_levels[i]))/ delta_z)
  # T = 0
  # P = 0
  # i = 0
  # df_z_0 = 0
  # df_z_m1 = 0
  # df_z_p1 = 0
  # u_i0 = 0
  # u_i1 = 0
  # v_i0 = 0
  # v_i1 = 0
  # delta_z = 0
  # T1 = 0
  # P1 = 0

  # integrate until that closest_value
  for i in range(0, len(pr_levels[:-2])):
    # sample over pressure levels, up to 800hPa for a start 
    # integrate
    delta_z = delta_z_now(i, Seeing_pressure_levels[i].z[k].values/g, Seeing_pressure_levels[int(i-1)].z[k].values/g, Seeing_pressure_levels[int(i+1)].z[k].values/g)
    J = J + Cn2_func(Seeing_pressure_levels[i].t[k].values, pr_levels[i] , 
    Seeing_pressure_levels[i].u[k].values, Seeing_pressure_levels[int(i+1)].u[k].values, Seeing_pressure_levels[i].v[k].values,
    Seeing_pressure_levels[int(i+1)].v[k].values, delta_z, 
    Seeing_pressure_levels[int(i+1)].t[k].values, pr_levels[int(i+1)]) * delta_z

    # save development
    cont_delta_z.append(delta_z)
  
  cont_J.append(J)
 # k selects the time, i selects the level
    
  Seeing_Era5.append(epsilon(J))
  print('epsilon(J) = {}'.format(epsilon(J))) # fill list with integral
  Seeing_time.append(Seeing_pressure_levels[i].u[k].time.values) # fill list with time

#%% ######################################## VARIATION ####################################### 
# VARIATION of formula implementation via xarray ds_full

# gravitational acceleration ERA5
g = 9.80665 #m/s^2, from Era5 documentation
k_var = 6
RCp = 0.286 # R/cp
P_0 = 1000 #mbar

def theta(T, P):
  return(T * (P_0 / P)**RCp)

def E(u_i0, u_i1, v_i0, v_i1, delta_z):
  return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )



def delta_z_now(i, df_z_0, df_z_m1, df_z_p1): # df_z is the geopotential height (variable z/g in Era5) at one time, i describes the pressure level
  if int(i) == 0:
    delta_z = df_z_0 - df_z_p1#df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  else:
    delta_z =  0.5 * (df_z_m1 - df_z_p1) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  return(delta_z)  

def Cn2_func_dz(ds, pr_index, time_index): # P1 and T1 stand for the values from the next level, i+1
  T = ds.t[pr_index][time_index]
  P = ds.level[pr_index]
  T1 = ds.t[int(pr_index + 1)][time_index]
  P1 = ds.level[int(pr_index + 1)]
  delta_z = delta_z_now(pr_index, ds.z[pr_index][time_index]/g, ds.z[int(pr_index - 1)][time_index]/g, ds.z[int(pr_index + 1)][time_index]/g)
  E_value = E(ds.u[pr_index][time_index], ds.u[int(pr_index + 1)][time_index], ds.v[pr_index][time_index], ds.v[int(pr_index + 1)][time_index], delta_z) 
  Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E_value / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
  J_contribution = Cn2_var * delta_z
  return(J_contribution)

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)

#%% ####################################### integral of VARIATION formulae ####################
# calculate seeing for big dataset

# variation of seeing calculation verified!

# in hPa
pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]

Seeing_Era5 = []
Seeing_time = []
cont_J = []
cont_delta_z = []

for k in range(0, 20):#len(Seeing_pressure_levels[0].time)): # sample over time
  J = 0
  # why does this not work?
  results = [Cn2_func_dz(ds_full, item, k) for item in range(0, len(pr_levels[:-2]))]
  J = sum(results)
  #integrate until closest_value
  # sample over pressure levels, up to 800hPa for a start

  # for i in range(0, len(pr_levels[:-2])):
  #   # integrate (sum)
  #   J = J + Cn2_func_dz(ds_full, i, k)

  
  cont_J.append(J)
 # k selects the time, i selects the level
    
  Seeing_Era5.append(epsilon(J))
  print('epsilon(J) = {}'.format(epsilon(J))) # fill list with integral values
  # Seeing_time.append(ds_full.u[k].time.values) # fill list with time

#%% find error
#plt.plot(Seeing_time, Seeing_Era5)
# def Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, delta_z, T1, P1): # P1 and T1 stand for the values from the next level, i+1
#   #Cn2 = (80*10**(-6) * P / (T * theta(T, P)))**2 * k * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
#   print((80*10**(-6) * P / (T * theta(T, P)))**2)
#   print((2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2)
#   #return(Cn2)
#   print(((80*10**(-6) * P / (T * theta(T, P)))**2) * k * ((2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2))
#   print(1.7173995936403701e-13 * 7.226296881589113e-07)
#   return((80*10**(-6) * P / (T * theta(T, P)))**2 * k * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2)

i = 10
k = 0
delta_z = delta_z_now(i, Seeing_pressure_levels[i].z[k].values/g, Seeing_pressure_levels[int(i-1)].z[k].values/g, Seeing_pressure_levels[int(i+1)].z[k].values/g)
# print('T ', Seeing_pressure_levels[i].t[k].values)
# print('P', pr_levels[i])
# print('u_i ', Seeing_pressure_levels[i].u[k].values)
# print('u_i+1 ', Seeing_pressure_levels[int(i+1)].u[k].values)
# print('v_i ', Seeing_pressure_levels[i].v[k].values)
# print('v_i+1 ' , Seeing_pressure_levels[int(i+1)].v[k].values)
# print( 'T_i+1', Seeing_pressure_levels[int(i+1)].t[k].values)
# print('P_i+1', pr_levels[int(i+1)])
# print('theta(T,P) ', theta(Seeing_pressure_levels[i].t[k].values , pr_levels[i]))
# print('E', E(Seeing_pressure_levels[i].u[k].values, Seeing_pressure_levels[int(i+1)].u[k].values, Seeing_pressure_levels[i].v[k].values, Seeing_pressure_levels[int(i+1)].v[k].values, delta_z))
# print('dtheta/dz ',(theta( Seeing_pressure_levels[int(i+1)].t[k].values, pr_levels[int(i+1)])
# - theta(Seeing_pressure_levels[i].t[k].values , pr_levels[i]))/ delta_z)

Cn2_func(Seeing_pressure_levels[i].t[k].values, pr_levels[i] , 
    Seeing_pressure_levels[i].u[k].values, Seeing_pressure_levels[int(i+1)].u[k].values, Seeing_pressure_levels[i].v[k].values,
    Seeing_pressure_levels[int(i+1)].v[k].values, delta_z, 
    Seeing_pressure_levels[int(i+1)].t[k].values, pr_levels[int(i+1)])


#%%
# try to vectorise computation

result = [Cn2_func(item) for item in Seeig_pressure_levels]


#%% reproduce result of '2000-11-21 14:00:00', epsilon = 23.039622

# Seeing_Nov = []

# # in hPa
# pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875]

# for i in range(0,len(pr_levels)):
#     path = './sites/Paranal/Era5_data/seeing/' + str(pr_levels[i]) + 'hPa/*.nc'
#     ds = xr.open_mfdataset(path, combine = 'by_coords')
#     ds_sel = ds.sel(longitude= -70.25,latitude= -24.75,method='nearest') # green point
#     ds_sel_time = ds_sel.sel(time = '2000-11-21 14:00:00')
#     Seeing_Nov.append(ds_sel_time)
# J = 0
# k = 0
# for i in range(0, len(pr_levels[:-2])): # sample over pressure levels, up to 800hPa for a start   
#   # integrate
#   delta_z = delta_z_now(i, Seeing_Nov[i].z[k].values/g, Seeing_Nov[int(i-1)].z[k].values/g, Seeing_Nov[int(i+1)].z[k].values/g)

#   J = J + Cn2_func(Seeing_Nov[i].t[k].values, pr_levels[i] , 
#   Seeing_Nov[i].u[k].values, Seeing_Nov[int(i+1)].u[k].values, Seeing_Nov[i].v[k].values,
#   Seeing_Nov[int(i+1)].v[k].values, delta_z, 
#   Seeing_Nov[int(i+1)].t[k].values, pr_levels[int(i+1)]) * delta_z
#   print(J)

# epsilon(J)
#%%
# put integrated values into dataframe for plotting

column_names = ["Seeing integral"]
df_Seeing_integral = pd.DataFrame(Seeing_Era5, index = Seeing_time, columns = column_names) 
df_Seeing_integral = df_Seeing_integral.rename_axis(index = 'time')
df_Seeing_integral.to_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2.csv', header = True, index = True)

#%% plot seeing integral

plt.plot(df_Seeing_integral.index, df_Seeing_integral['Seeing integral'])
#%% prepare Paranal calculated data

seeing_hourly_preped = mes_prep(seeing_hourly) # timezone = None, already in UTC

#%%
# merge datasets
merged_df_seeing, seasonal_seeing, diurnal_seeing, monthly_grouped_seeing, yearly_grouped_seeing  = merge_df(seeing_hourly_preped,
df_Seeing_integral, dropnan = True)

# create special merged dataframe for ERA 5 data (which goes back to 1979)
merged_df_seeing_era5, monthly_grouped_seeing_era5, yearly_grouped_seeing_era5  = merge_df_long(df_Seeing_integral)

# %% diurnal cycle
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
plot_cycle(diurnal_seeing,'diurnal cycle Cerro Paranal', 'Seeing Paranal', 
           './sites/Paranal/Output/Plots/seeing/diurnal_cycle_UTC_seeing_Paranal_2000.pdf', 
           'Seeing integral')
           
# %% seasonal cycle
plot_cycle(seasonal_seeing,'seasonal cycle Cerro Paranal', 'Seeing Paranal', 
           './sites/Paranal/Output/Plots/seeing/seasonal_cycle_UTC_seeing_Paranal_2000.pdf', 
           'Seeing integral')

# %%
# plot timeseries, moving average

# does not make sense to calculate the moving average for a timespan of only 5 years
# plot_timeseries_movav('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_seeing_2015to2019_movav.pdf', yearly_grouped_seeing,
# In_situ = 'seeing Paranal', Era5_seeing = 'tcw')

# use merged_df
# plot_timeseries_long(filename, yearly_grouped_insitu ,insitu_param, yearly_grouped_era5,**kwargs
plot_timeseries_long('./sites/Paranal/Output/Plots/seeing/timeseries_Paranal_seeing_2000Feb_long.pdf', yearly_grouped_seeing,
'Seeing Paranal', yearly_grouped_seeing_era5.loc[:'2019-12-31'], moving = False, Era5_seeing = 'Seeing integral')


# %%
# correlation plot: alpha = 0.05
# hourly

parameter = 'Seeing integral'
path = './sites/Paranal/Output/Plots/seeing/correlation_Paranal_' + parameter +'_2000Feb_'
xax = 'Era 5 Seeing integral [arcsec]'
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

# # yearly
# correlation_plot(path + 'yearly.pdf',
# 'yearly means',
# yearly_grouped_seeing[parameter]['mean'],  xax,
# yearly_grouped_seeing['Seeing Paranal']['mean'], yax)


#%%
# start at 1hPa and end at 800hPa (then, T1 and P1 are well defined)
# first, write for loop, then use list comprehensions / lambda functions
J = 0
cont = []
contribution = []
for i in range(0, len(df_seeing_era5_sort['level'][:-8])):
  # if i == 0:
  #   delta_z = df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
  # else:
  #   delta_z =  0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  # # calculate integral
  # print('J= ', J)
  # print(delta_z)
  # print('T ', df_seeing_era5_sort['t'][i])
  # print('P', df_seeing_era5_sort['level'][i])
  # print('u_i ', df_seeing_era5_sort['u'][i])
  # print('u_i+1 ',df_seeing_era5_sort['u'][int(i+1)])
  # print('v_i ',df_seeing_era5_sort['v'][i])
  # print('v_i+1 ' ,df_seeing_era5_sort['v'][int(i+1)])
  # print( 'T_i+1',df_seeing_era5_sort['t'][int(i+1)])
  # print('P_i+1',df_seeing_era5_sort['level'][int(i+1)])
  # print('theta(T,P) ', theta(df_seeing_era5_sort['t'][i] ,df_seeing_era5_sort['level'][i]))
  # print('E', E(df_seeing_era5_sort['u'][i], df_seeing_era5_sort['u'][int(i+1)], df_seeing_era5_sort['v'][i], df_seeing_era5_sort['v'][int(i+1)], delta_z))
  # print('dtheta/dz ',(theta(df_seeing_era5_sort['t'][int(i+1)], df_seeing_era5_sort['level'][int(i+1)])
  # - theta(df_seeing_era5_sort['t'][i] ,df_seeing_era5_sort['level'][i]))/ delta_z)
  
  J = J + Cn2_func(df_seeing_era5_sort['t'][i],df_seeing_era5_sort['level'][i] , df_seeing_era5_sort['u'][i], df_seeing_era5_sort['u'][int(i+1)], df_seeing_era5_sort['v'][i],
   df_seeing_era5_sort['v'][int(i+1)], delta_z_now(i), df_seeing_era5_sort['t'][int(i+1)], df_seeing_era5_sort['level'][int(i+1)]) * delta_z
  cont.append(J)
  contribution.append(Cn2_func(df_seeing_era5_sort['t'][i],df_seeing_era5_sort['level'][i] , df_seeing_era5_sort['u'][i], df_seeing_era5_sort['u'][int(i+1)], df_seeing_era5_sort['v'][i],
   df_seeing_era5_sort['v'][int(i+1)], delta_z_now(i), df_seeing_era5_sort['t'][int(i+1)], df_seeing_era5_sort['level'][int(i+1)]) * delta_z)


print(f'J = {J}, epsilon = {epsilon(J)}')

#%%
plt.plot(df_seeing_era5_sort['level'][:-8], cont, '-o', label = 'integral')
#plt.bar(df_seeing_era5_sort['level'][:-8], contribution, 1)#, label = 'absolut contribution')
plt.xlabel('pressure level [hPa]')

plt.ylabel(r'J [$m^(1/3)$]')
plt.title(r'J = $\int C_n^2(z) dz $')
plt.legend()
plt.savefig('./sites/Paranal/Output/Plots/seeing/integral_2017-01-10_osborn_formula.pdf')


#%%
# open NETCDF files on 600hPa to 750hPa
#ds_wind = xr.open_mfdataset('./sites/Paranal/Data/Era_5/seeing/200hPa/*.nc', combine = 'by_coords')

# convert netcdf to dataframe, label pressure levels
# change coords!!!!!!!!!

# not nearest,  but 'green' point, higher elevated
df_wind = netcdf_to_df(ds_wind, -70.25, -24.75)
df_wind_prep = df_prep(df_wind, 'u', 'u')

print('netcdf to df done')
#%%
# create column with u**2
df_wind_prep['u_square'] = df_wind_prep['u'] **2
#df_wind_prep['u_square'] = (df_wind_prep['u'] + df_wind_prep['v']) **2

#%% try out formula from Hellemeier 2018

wv = 500 * 10**(-9) # wavelength 500nm
k = 2 * np.pi /wv # wavenumber k
# seeing = 1.1 # for example
# u_square = 200 # for example

# epsilon = 206265 * 0.976 * wv * (0.423 * k**2 * u_square)**(3/5) # rad * 206265 = arcsec
A = 10**(-15) # m^(1/3)
df_wind_prep['epsilon'] = 206265 * 0.976 * wv / (0.423 * k**2 *A * df_wind_prep['u_square'])**(-3/5) # rad * 206265 = arcsec

# A = (206265 * 0.976 * wv / seeing) ** (-5/3) / (0.423 * k**2 * u_square)

#%% try out formula from https://academic.oup.com/mnras/article/480/1/1278/5056187 
# open NETCDF files
ds_seeing_era5 = xr.open_mfdataset('./sites/Paranal/Data/Era_5/seeing/Era5_seeing_test_2017-01-10.nc', combine = 'by_coords')

# not nearest,  but 'green' point, higher elevated
df_seeing_era5 = netcdf_to_df(ds_seeing_era5, -70.25, -24.75)
df_seeing_era5_prep = df_prep(df_seeing_era5, 'u', 'u')

print('netcdf to df done')

# gravitational acceleration ERA5
g = 9.80665 #m/s^2

df_seeing_era5_sort = df_seeing_era5_prep.loc['2017-01-10 10:00:00'].sort_values('level')
df_seeing_era5_sort['z/g'] = df_seeing_era5_sort['z']/g
