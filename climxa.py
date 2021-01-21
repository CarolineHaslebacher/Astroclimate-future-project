# %% 
  
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr

import xarray.plot as xplt 
#import sunpy.timeseries as ts 
#

########## for cycle ##############
from matplotlib import dates as d
import datetime as dt
import pytz
import time

from itertools import cycle

from functools import reduce
from scipy import stats

import csv

import seaborn as sns
sns.set()

from matplotlib import cycler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import skill_metrics as sm
import cartopy.crs as ccrs

import copy
import pickle

import webcolors
import math
from matplotlib.colors import to_hex

import os

# inline labels
from labellines import *

#%%

# color (from Brett)
# from matplotlib.colors import to_hex
# [to_hex(plt.cm.viridis(i / 10)) for i in range(10)] 


# function for reading in ERA5 data on pressure levels
# the path is always composed in the same way
def read_ERA5_pr_level_data(site, ls_pr_levels, variable, ERA5_var, ERA5_path=None):
    
    ds_list = []
    for pressure in ls_pr_levels:
        if ERA5_path == None:
            if variable == 'T':
                path = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/T/pressure_levels/' + str(pressure) + 'hPa/*.nc'
            else:
                path = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/' + variable + '/'+ str(pressure) + 'hPa/*.nc'
        else:
            path = ERA5_path # only one pressure level (or rewrite code)

        ds = xr.open_mfdataset(path, combine = 'by_coords')
        ds = ds.assign_coords({'level': pressure})
        # not in all datasets, expver exists, so check if it is a coordinate
        if 'expver' in ds.coords:
            # expver 5 is the most complete version
            ds = ds.sel(expver = 5)
            ds = ds.drop_vars(['expver']) # to prevent conflicts
        # what is happening here????? (for specific humidity, Paranal, reading in of datasets is wrong)
        # print(ds['q'].sel(longitude=lon, latitude=lat, method='nearest').load())
        ds_list.append(ds)
    ds_ERA5_pr = xr.concat(ds_list, dim = 'level')
    ds_ERA5_pr.level.attrs['units'] = 'hPa'

    if variable == 'T':
        # convert Kelvin to celsius
        for var in ERA5_var:
            ds_ERA5_pr[var] = ds_ERA5_pr[var] - 273.15
            ds_ERA5_pr[var].attrs['units'] = 'deg C' #r'^{\circ}'
            ds_ERA5_pr[var].attrs['long_name'] = 'Temperature'

    return ds_ERA5_pr

def read_ERA5_sg_level_data(site, variable, single_lev_var, ERA5_path=None):
    if variable == 'T':
        path =  '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/T/single_levels/*.nc'
    elif ERA5_path == None:
        path = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/' + variable + '/*.nc'
    else:
        path = ERA5_path

    ds = xr.open_mfdataset(path, combine = 'by_coords')
    if 'expver' in ds.coords:
        # expver 5 is the most complete version
        ds = ds.sel(expver = 5)

    if variable == 'T':
        # convert Kelvin to celsius
        for var in single_lev_var:
            ds[var] = ds[var] - 273.15
            ds[var].attrs['units'] = 'deg C' #r'^{\circ}'
            ds[var].attrs['long_name'] = 'Temperature'
        
    return ds

#%%

def get_PRIMAVERA(d_model, clim_key, site, pressure_level=False, single_level=False):
    # 
    # read pressure level data:
    if pressure_level:
        # empty list to combine datasets from different forcings
        forcing_ls = []
        for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
             
            for clim_var in d_model[clim_key]['clim_var']: # in case there are more than one climate variables, function is not fully prepared (assume that there is only one)
                try:
                    # path = '/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables/' + clim_var + '/Amon/' + clim_key + '/' + folder + '/*.nc'
                    path = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + clim_var + '/Amon/' + clim_key + '/' + forcing + '.nc'
                    # print(path)
                    #ds = xr.open_mfdataset(path, combine = 'by_coords')
                    ds = xr.open_dataset(path)

                    # check for lon/lat instead of longitude/latitude
                    if 'lon' in ds.coords:
                        ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})
                    
                        # rename coordinates to make it compatible with ERA5 datasets
                        # rename plev, lev, ... to level
                        if 'plev' in ds.coords:
                            ds = ds.rename({'plev': 'level'})
                        elif 'lev' in ds.coords:
                            ds = ds.rename({'lev': 'level'})

                        # pressure level in model data is in Pa. convert it to hPa
                        ds['level'] = ds['level']/100
                        ds['level'] = ds['level'].astype('int64')
                        ds['level'].attrs['units'] = 'hPa'

                    if clim_var == 'ta' or clim_var == 'tas':
                        # convert Kelvin to celsius
                        ds[clim_var] = ds[clim_var] - 273.15
                        ds[clim_var].attrs['units'] = 'deg C' #r'^{\circ}'
                    
                    
                    # copy only clim_var to new dataset (and leave behind time_bnds,...)
                    # this step makes it possible to cycle through data variables (for var in ds.data_vars: do something)
                    ds = xr.Dataset({clim_var:  ds[clim_var]}) 
                    # now rename variable to be able to distinct between different forcings   
                    ds = ds.rename_vars({clim_var: clim_var + ' ' + forcing})

                    forcing_ls.append(ds.load())

                except (FileNotFoundError, KeyError):
                    print('I could not find data for {}, {}, {}'.format(clim_key, forcing, clim_var))

        # I had a problem with loading, loading relative and specific humidity was extremely slow (but Temperature was fine)
        # merging took such a long time!
        # I solved it by loading the dataset (forcing_ls.append(ds.load()))
        ds_different_forcings = xr.merge(forcing_ls, join='outer', compat='equals') # 'equals': all values and dimensions must be the same (but not attributes)

    # read single level data
    if single_level:
        # empty list to combine datasets from different forcings
        forcing_ls = []
        for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
        # come back later to complete functions for more than one folder (vor allem plotting)
        # maybe I can merge or append the datasets (hist, present, SSTfuture, future)
            
            for clim_var in d_model[clim_key]['single_lev_var']: # in case there are more than one climate variables
                try:    
                    path = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + clim_var + '/Amon/' + clim_key + '/' + forcing + '.nc'
                    #ds = xr.open_mfdataset(path, combine = 'by_coords')
                    ds = xr.open_dataset(path)

                    # check for lon/lat instead of longitude/latitude
                    if 'lon' in ds.coords:
                        ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})

                    if clim_var == 'ta' or clim_var == 'tas':
                        # convert Kelvin to celsius
                        ds[clim_var] = ds[clim_var] - 273.15
                        ds[clim_var].attrs['units'] = 'deg C' #r'^{\circ}'

                    if clim_var == 'clt':
                        # cloud cover is stored in %, but we want decimal
                        ds[clim_var] = ds[clim_var]/100

                    # copy only clim_var to new dataset (and leave behind time_bnds,...)
                    # this step makes it possible to cycle through data variables (for var in ds.data_vars: do something)
                    ds = xr.Dataset({clim_var:  ds[clim_var]}) 
                    # now rename variable to be able to distinct between different forcings   
                    ds = ds.rename_vars({clim_var: clim_var + ' ' + forcing})

                    forcing_ls.append(ds)

                except (FileNotFoundError, KeyError):
                    print('I could not find data for {}, {}, {}'.format(clim_key, forcing, clim_var))

            # merge different forcings into one dataset
            ds_different_forcings = xr.merge(forcing_ls, join='outer') #, compat = 'no_conflicts') 

    return ds_different_forcings
        

# function that reads data (column named 'time' must exist and works as a header) converts dataframe to xarray
def df_to_xarray(path):

    df = pd.read_csv(path, parse_dates=True, index_col='time')
    ds = df.to_xarray()

    return ds

# merges datasets and select time slice
def merge_ds_time_slicer(list_of_datasets, time_slice = slice(None)):
    
    # merge datasets
    ds_merged = xr.merge(list_of_datasets) # remove ', compat='override'' immediatly after testing HRCM!!

    
    # select time slice that they both have in common
    # if time_slice = slice(None), all values get selected
    ds_time_sliced = ds_merged.sel(time = time_slice)# e.g. slice('1998-01-01','2019-12-31'))

    # create different dataset for timeseries 
    new_ls_ds = []
    for i, ds in enumerate(list_of_datasets):
        
        # for in-situ dataset, select time_slice
        # !!!! ?? what if I do not have insitu data??? --> do not use this code then !!!
        if i == 0:
            ds = ds.sel(time = time_slice)
            new_ls_ds.append(ds)
        
        else: # for ERA5 dataset, select data from 1979 until 2019-12-31
            ds = ds.sel(time = slice(None, '2019-12-31'))
            new_ls_ds.append(ds)


    # merge the new list of datasets
    ds_merged_full = xr.merge(new_ls_ds) # remove ', compat='override'' immediatly after testing HRCM!!

    # use ds_time_sliced for cycles and ds_merged_full for timeseries
    return ds_time_sliced, ds_merged_full
 
def xr_sel(d_ds, key_to_sel , my_lon, lat, obs=False):
    # select the dataset with the key string given to the function
    ds = d_ds[key_to_sel]

    # for ERA5 data, longitude is needed in -180 to 180
    if obs == True:
        if my_lon > 180:
            my_lon = my_lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
            print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_lon))
    # or better: lookup lon&lat in orog dict

    # select lon and lat
    try:
        ls_level = d_ds['Plev']
        ds_sel = ds.sel(longitude= my_lon,latitude= lat , level = ls_level ,method='nearest')
    except (ValueError, KeyError):
        print('no pressure levels to select')
        try:
            ds_sel = ds.sel(longitude= my_lon,latitude= lat ,method='nearest')
        except ValueError:
            print('no lon/lat to select for {}'.format(key_to_sel))
            ds_sel = ds
            # print(ds)
    # do the following check only later!
    # # check that ds_sel is not totally empty for every pressure level (important for climate model data, where we always have pressure levels)
    # if 'Plev' in d_ds.keys():
    #     print('yes')
    #     for PressureLevel in d_ds['Plev']:
    #         if clim_var != None:
    #             print('yes, clim_var is not none, Plev is {}'.format(PressureLevel))
    #             ds_check = ds_sel.where(xr.ufuncs.isnan(ds_sel[clim_var].sel(level= PressureLevel)) != True, drop = True)
    #             print(ds_check['time'].size)
    #             if ds_check['time'].size == 0:
    #                 print('I found an empty array and drop it now: plev = {}'.format(PressureLevel))
    #                 # if array is empty after dropping all nan's, delete that pressure level
    #                 ds_sel = ds_sel.drop_sel(level=PressureLevel)
    return ds_sel

# define the classification function
def skill_score_classification(skill):
    skill = round(skill, 2)
    if skill < 0.32:
        return 'Poor'
    elif skill < 0.63: # only skills that are higher than 0.32 enter here!
        return 'Mediocre'
    elif skill < 0.91:
        return 'Good'
    elif skill <= 1.0:
        return 'Excellent'
    else:
        raise Exception('WARNING: no valid skill entered: {}'.format(skill))

def group_mean(time_freq_string, d_ds, ds_sel, std=False, return_mean_ds=False):
    # groupby time  
    ds_mean = ds_sel.groupby(time_freq_string).mean(dim='time')
    
    if std == True: # calculate standard deviaton
        ds_std = ds_sel.groupby(time_freq_string).std(dim='time')
        
    if return_mean_ds == True: # we have to return ds_mean and ds_std separately
        # (maybe needed because ds_mean_month already exists)
        if std == True:
            return ds_mean, ds_std
        else:
            return ds_mean
    else: # else, create ds_mean_month/ds_mean_year and return whole dictionary
        d_ds["ds_mean_" + time_freq_string[5:]] = ds_mean
        if std == True:
            d_ds["ds_std_" + time_freq_string[5:]] = ds_std
        return d_ds

def linreg(varx, vary):
    # for the coefficient of determination, R^2, simply calculate: r_value ** 2
    # (r_value is the correlation coefficient)
    
    # sort out nan values
    mask = ~np.isnan(varx) & ~np.isnan(vary)

    # calculate linear regressions statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

    return slope , intercept, r_value, p_value, std_err


def linreg_slope(varx, vary):
    # for the coefficient of determination, R^2, simply calculate: r_value ** 2
    # (r_value is the correlation coefficient)
    try:
        # sort out nan values
        mask = ~np.isnan(varx) & ~np.isnan(vary)

        # calculate linear regressions statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

        return slope #, intercept, r_value, p_value, std_err
    except ValueError:
        print('for {}, {}, there was a Value Error'.format(varx, vary))
        return np.nan

def linreg_r_value(varx, vary):
    try:
        # sort out nan values
        mask = ~np.isnan(varx) & ~np.isnan(vary)

        # calculate linear regressions statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

        return r_value #, intercept, r_value, p_value, std_err
    except ValueError:
        print('for {}, {}, there was a Value Error'.format(varx, vary))
        return np.nan

def linreg_p_value(varx, vary):
    try:
        # sort out nan values
        mask = ~np.isnan(varx) & ~np.isnan(vary)

        # calculate linear regressions statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

        return p_value #, intercept, r_value, p_value, std_err
    except ValueError:
        print('for {}, {}, there was a Value Error'.format(varx, vary))
        return np.nan

def linreg_std_err_slope(varx, vary):
    try:
        # sort out nan values
        mask = ~np.isnan(varx) & ~np.isnan(vary)

        # calculate linear regressions statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

        return std_err #, intercept, r_value, p_value, std_err
    except ValueError:
        print('for {}, {}, there was a Value Error'.format(varx, vary))
        return np.nan

def linreg_intercept(varx, vary):
    try:
        # sort out nan values
        mask = ~np.isnan(varx) & ~np.isnan(vary)

        # calculate linear regressions statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

        return intercept #, intercept, r_value, p_value, std_err
    except ValueError:
        print('for {}, {}, there was a Value Error'.format(varx, vary))
        return np.nan


def d_5year_rolling_mean(ds_mean_year, intercept=False):
    # all stats are based on a 5-year rolling mean (for smoother averages)

    # 5 year rolling mean, the value is assigned to the center
    ds_rolling = ds_mean_year.rolling(year = 5, center = True, min_periods = 5).mean()

    # use xr.apply_ufunc (may return only one array) to whole dataset.
    # the name of the dataset tells you which stats is stored in it
    ds_slope = xr.apply_ufunc(linreg_slope, ds_rolling.year, ds_rolling, input_core_dims=[['year'], ['year']], vectorize=True)
    ds_r_value = xr.apply_ufunc(linreg_r_value, ds_rolling.year, ds_rolling, input_core_dims=[['year'], ['year']], vectorize=True)
    ds_p_value = xr.apply_ufunc(linreg_p_value, ds_rolling.year, ds_rolling, input_core_dims=[['year'], ['year']], vectorize=True)
    ds_std_err = xr.apply_ufunc(linreg_std_err_slope, ds_rolling.year, ds_rolling, input_core_dims=[['year'], ['year']], vectorize=True)
    if intercept: # for plotting the trendlines
        ds_intercept = xr.apply_ufunc(linreg_intercept, ds_rolling.year, ds_rolling, input_core_dims=[['year'], ['year']], vectorize=True)

    # check with code below 
    # slope, intercept, r_value, p_value, std_err, varx, vary = linreg(ds_rolling.year, ds_rolling[clim_var].sel(longitude=lon, latitude=lat, level= Plev,method='nearest'))

    # for a rough trend, just take latest 5years minus first 5 years
    ds_rough_trend = ds_rolling.isel(year=-3) - ds_rolling.isel(year=3) # 3 because it's the rolling mean

    if intercept:
        return ds_rough_trend, ds_slope, ds_r_value, ds_p_value, ds_std_err, ds_intercept

    else:
        return ds_rough_trend, ds_slope, ds_r_value, ds_p_value, ds_std_err

#%% seeing

def read_ERA5_seeing_data(base_path, lon, lat):

    # in hPa
    #pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
    # pr_levels = pressure_levels_list

    # for i in range(0, len(pr_levels)):
    # better: search for folders in base path and iterate through them
    pr_levels_strings = os.listdir(path = base_path)

    # extract integers and sort list
    pr_levels = [int(x[:-3]) for x in pr_levels_strings]
    pr_levels.sort()

    for i in range(0, len(pr_levels)):
        #'./sites/Paranal/Era5_data/seeing/'
        path = str(base_path) + str(pr_levels[i]) + 'hPa/*.nc'

        ds = xr.open_mfdataset(path, combine = 'by_coords')
        
        if 'expver' in ds.coords:
            ds = ds.sel(expver=5)
            # drop expver
            ds = ds.reset_coords('expver')
            ds = ds.drop('expver')

        ds_sel = ds.sel(longitude= lon,latitude= lat, method='nearest') # green point
        
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

    return ds_full

def theta(T, P):
  RCp = 0.286 # R/cp
  P_0 = 1000 #mbar

  return(T * (P_0 / P)**RCp)

def E(u_i0, u_i1, v_i0, v_i1, delta_z):
  return( ((u_i1-u_i0)/delta_z)**2 +  ((v_i1-v_i0)/delta_z)**2 )

def Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1): # P1 and T1 stand for the values from the next level, i+1
    g = 9.80665 #m/s^2, from Era5 documentation 
    # k_var = 6 # from Osborn 2018
    k_var = 1 # calibrate later
    if int(i) == 0:
        delta_z = abs(df_z_0 - df_z_p1) #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
    else:
        delta_z =  abs(0.5 * (df_z_m1 - df_z_p1)) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  
    Cn2_var = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
    return(Cn2_var * delta_z)

def Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1): # P1 and T1 stand for the values from the next level, i+1
    g = 9.80665 #m/s^2, from Era5 documentation 
    # k_var = 6 # from Osborn 2018  
    k_var = 1 # calibrate later!
    if int(i) == 0:
        delta_z = abs(df_z_0 - df_z_p1)
    else:
        delta_z =  abs(0.5 * (df_z_m1 - df_z_p1)) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])
  
    Cn2_var_profile = (80*10**(-6) * P / (T * theta(T, P)))**2 * k_var * (2 * E(u_i0, u_i1, v_i0, v_i1, delta_z) / (g/theta(T, P) * abs(theta(T1, P1)- theta(T, P))/ delta_z))**(2/3) * ((theta(T1, P1)- theta(T, P))/ delta_z)**2
    return(Cn2_var_profile)

def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)

def invert_epsilon(epsilon):
    wv = 500 * 10**(-9) # wavelength 500nm
    k_wave = 2 * np.pi /wv # wavenumber k
    J = (epsilon/(206265 * 0.976 * wv * (0.423 * k_wave**2)**(3/5)))**(5/3)
    # is this fully correct? I get 2*10**(-28)! instead of 10**(-16)
    return J


#integrate until closest_value
# sample over pressure levels, up to end of pr_levels_list
def ERA5_seeing_calc(ds_full, mean_insitu, pr_levels_list):
    g = 9.80665 #m/s^2, from Era5 documentation 
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

        # (for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension)
        # do that!! (and only if J_add is not nan)
        ds_Cn2 = Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)
        ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels_list[i]})

        # concatenate DataArrays along new dimension 'level'
        if i == 0:
            ds_Cn2_profile = ds_Cn2
        else:
            ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')

    # (for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension)

    # calibration
    calib_factor = mean_insitu/np.mean(epsilon(J))

    # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
    ds_seeing = xr.Dataset({"seeing": calib_factor * epsilon(J)})

    ds_Cn2_profile = xr.Dataset({"Cn2": calib_factor * ds_Cn2_profile})
    ds_Cn2_profile = ds_Cn2_profile.reset_coords(drop=True)

    return ds_seeing, ds_Cn2_profile, calib_factor.values

# funtion for wind speed seeing
def ERA5_seeing_wind(ds_full, mean_insitu, PRIMAVERA = False, U_clim_var=None, V_clim_var=None, idx=None):
    g = 9.80665 #m/s^2, from Era5 documentation 
    # A = 10**(-15) # calibration factor, we just take something to get values in the correct order
    # A = 5 * 10**(-16) # after playing around with the surface layer, I found that this A fits better the data (at least for Chile)
    # the calibration factor influences the model skill. we therefore need to calibrate each site individually
    # for this, we take the mean of the insitu data (the seeing)

    if PRIMAVERA == True:
        i = idx
        # check index
        if ds_full[U_clim_var][:,i].level != 200:
            raise Exception('index of 5 does not correspond to 200hPa level. check data.')

        # calculate J
        J = (ds_full[U_clim_var][:,i]**2 + ds_full[V_clim_var][:,i]**2)

    else:
        i = 5 # index of 5 should be equal to 200hPa for ERA5 data

        # check if index corresponds really to 200hPa level
        if ds_full.u[i].level != 200:
            raise Exception('index of 5 does not correspond to 200hPa level. check data.')
        
        # calculate J
        J = (ds_full.u[i]**2 + ds_full.v[i]**2)
    # maybe it makes sense to load J into memory here. Otherwise it may be that it has to be loaded twice (once for the mean and once for writing to netcdf)!

    # calibration
    calib_factor = mean_insitu/np.mean(epsilon(J))

    # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
    ds_seeing = xr.Dataset({"wind speed seeing": calib_factor * epsilon(J)})

    return ds_seeing, calib_factor.values


def get_PRIMAVERA_surface_pressure_level(model_name_clim_key, site_name_folder, lon, lat):

    pr_levels_model = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000

    # for model surface pressure level 
    d_model_temp = {"HadGEM": {"folders": ['hist'],"taylor_folder": ['hist'], "name": "HadGEM3-GC31-HM"},
                "EC-Earth": {"folders": ['hist'],"taylor_folder": ['hist'], "name": "EC-Earth3P-HR"} ,
                "CNRM": {"folders": ['hist'], "taylor_folder": ['hist'],"name": "CNRM-CM6-1-HR"},
                "MPI": {"folders": ['hist'],"taylor_folder": ['hist'], "name": "MPI-ESM1-2-XR"},
                "CMCC": {"folders": ['hist'],"taylor_folder": ['hist'], "name": "CMCC-CM2-VHR4"},
                "ECMWF": {"folders": ['hist'],"taylor_folder": ['hist'], "name": "ECMWF-IFS-HR"} }
    
    # for check, write individual surface pressure mean values into 'Plev'
    clim_key = model_name_clim_key

    # we must define single_lev_var as 'ps' to load in surface pressure (change back in the end)
    d_model_temp[clim_key]['single_lev_var'] = ['ps']
    # load surface pressure data... NOT FINISHED!!
    d_model_temp[clim_key]['ds_surface_pressure'] = get_PRIMAVERA(d_model_temp, clim_key, site_name_folder, single_level=True)
    # convert to hPa
    d_model_temp[clim_key]['ds_surface_pressure']['ps hist'] = d_model_temp[clim_key]['ds_surface_pressure']['ps hist']/100
    # select lon/lat
    d_model_temp[clim_key]['ds_surface_pressure'] = xr_sel(d_model_temp[clim_key], 'ds_surface_pressure', lon, lat)
    # calculate the mean (and integrate until this value (otherwise, I cannot shorten the for loop by time))
    d_model_temp[clim_key]['surface_pressure_mean'] = d_model_temp[clim_key]['ds_surface_pressure']['ps hist'] .mean()
    print(d_model_temp[clim_key]['surface_pressure_mean'])
    # also check standard deviation!

    #  define until which pressure level we integrate
    given_value = d_model_temp[clim_key]['surface_pressure_mean'] 
    # find closest value available in pressure levels
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(pr_levels_model, key=absolute_difference_function)

    return closest_value


# for seeing value, integrate Cn2 over all pressure levels
# pr_levels_list = pr_levels[:-2]
def PRIMAVERA_calc_seeing(ds_full, mean_insitu, lon, lat, T_clim_var, U_clim_var, V_clim_var, Z_clim_var, pr_levels_list, site, clim_key, forcing, closest_value):
    J = 0

    for i in range(0, len(pr_levels_list)-1):
        # print(pr_levels_list[i])
        T = ds_full[T_clim_var].sel(level=pr_levels_list[i])
        P = ds_full.level[i]
        u_i0 = ds_full[U_clim_var].sel(level=pr_levels_list[i])
        u_i1 = ds_full[U_clim_var].sel(level=pr_levels_list[i+1])
        v_i0 = ds_full[V_clim_var].sel(level=pr_levels_list[i]) 
        v_i1 = ds_full[V_clim_var].sel(level=pr_levels_list[i+1])
        T1 = ds_full[T_clim_var].sel(level=pr_levels_list[i+1])
        P1 = ds_full.level[int(i+1)] 
        df_z_0 = ds_full[Z_clim_var].sel(level=pr_levels_list[i]) # do not divide by g, it is already in m (model data)
        if i == 0:
            df_z_m1 = 0
        else:
            df_z_m1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i-1])
        df_z_p1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i+1])
        # integrate (sum)
        J_add = Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)

        # test if climxa.Cn2_func() doesn't return NaN
        ds_check = J_add.where(xr.ufuncs.isnan(J_add) != True, drop = True)
        # print(ds_check['time'].size)
        if ds_check['time'].size == 0:
            print('nan array for {}'.format(i))
        else:
            J = J + J_add
        
            # (for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension)
            # do that!! (and only if J_add is not nan)
            ds_Cn2 = Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)
            ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels_list[i]})

            # concatenate DataArrays along new dimension 'level'
            if i == 0:
                ds_Cn2_profile = ds_Cn2
            else:
                ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')

    if J.all() != 0: # otherwise, all were nan

        # calculate calibration factor
        calib_factor = mean_insitu/np.mean(epsilon(J))

        # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
        ds_seeing = xr.Dataset({"seeing": calib_factor * epsilon(J)})
        ds_Cn2_profile = xr.Dataset({"Cn2": calib_factor * ds_Cn2_profile})
        ds_Cn2_profile = ds_Cn2_profile.reset_coords(drop=True)

        ds_seeing["seeing"].attrs['long_name'] = 'Seeing'
        ds_seeing["seeing"].attrs['units'] = 'arcsec'
        ds_seeing['seeing'].attrs['calibration_factor'] = calib_factor.values # median is already a float (somehow)

        ds_Cn2_profile["Cn2"].attrs['long_name'] = 'Cn2'
        ds_Cn2_profile["Cn2"].attrs['units'] = 'm^(1/3)'
        ds_Cn2_profile['Cn2'].attrs['calibration_factor'] = calib_factor.values

        # add coordinate 'level'
        # drop 'level' now, otherwise we have 'level=200'
        # ds_seeing = ds_seeing.reset_coords(names='level', drop=True)
        # --> do not drop anything here, keep 200hPa (makes it easier to read in the data!!)
        ds_seeing["seeing"] = ds_seeing["seeing"].assign_coords(level=closest_value, longitude=lon, latitude=lat)
        # add dimension 'level'
        ds_seeing["seeing"] = ds_seeing["seeing"].expand_dims(dim=['level', 'longitude', 'latitude'])
        # now, the level can be selected with e.g. ds.sel(level=775)
        # add coords lon and lat

        # define path
        path = './sites/'+ site + '/Data/HighResMIP/seeing/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files

        # make directory if not available
        # if os.path.exists(path):
        #     os.remove(path)
        # else:
        os.makedirs(os.path.dirname(path), exist_ok=True) 

        ds_seeing.to_netcdf(path)

        # path for Cn2 profile
        path_Cn2 = './sites/'+ site + '/Data/HighResMIP/Cn2/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files
        
        # if os.path.exists(path_Cn2):
        #     os.remove(path_Cn2)
        # else:
        os.makedirs(os.path.dirname(path_Cn2), exist_ok=True)  

        ds_Cn2_profile.to_netcdf(path_Cn2)

    else:
        print('Check dataset. Input seems to be nan.')

    return np.array(calib_factor.reset_coords(drop=True))

#%% define skill_score_classification_to_table


def skill_score_classification_to_table(file_name, folder_name, title_var, variable, pressure_level = False, ERA5_is_ref=False):
   
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    
    ls_site = []
    # ERA5
    ls_ERA5_classif = []
    ls_ERA5_variable = []
    ls_ERA5_raw_skill = []
    ls_ERA5_raw_corr = []

    ls_PRIMAVERA_classif = []
    ls_PRIMAVERA_variable = []
    ls_PRIMAVERA_raw_skill = []
    ls_PRIMAVERA_raw_corr = []

    ls_best_match_ensemble = []
    ls_diff_match_ensemble = []


    # and also create nested list for the check of the length (it gets filled as the list gets bigger!)
    nested_list_dict = {'ls_ERA5_classif': ls_ERA5_classif, 'ls_ERA5_variable': ls_ERA5_variable,
                        'ls_PRIMAVERA_classif': ls_PRIMAVERA_classif, 'ls_PRIMAVERA_variable': ls_PRIMAVERA_variable}

    # fill dict with lists
    # table_dict = {'site': [], 'ERA5_'}

    for idx in range(0, 8):

        # get lon/lat and site_name from d_site_lonlat_data dict
        site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
        # print(site_name_folder)

        # change names for plotting
        if site_name_folder == 'MaunaKea':
            site_noUnderline = 'Mauna Kea'
        elif site_name_folder == 'siding_spring':
            site_noUnderline = 'Siding Spring'
        else:
            site_noUnderline = site_name_folder.replace('_', ' ')

        # set counter for ERA5 and PRIMAVERA ensemble means to zero for each site
        count_PRIMAVERA = 0
        count_ERA5 = 0

        
        # fill list of sites (before skipping sites where we have no data)
        ls_site.append(site_noUnderline)

        # if we do not compare ERA5 to PRIMAVERA, we need to specify for which sites we do not have measurements ('nm')
        if ERA5_is_ref == False:

            if folder_name == 'total_cloud_cover':

                # for sites where we have no in situ reference, we have to exclude them here!
                if idx == 6 or idx == 7:
                    # append 'nm' for 'no measurement' to list, so that still all sites are represented
                    ls_ERA5_classif.append(' ')
                    ls_ERA5_variable.append('nm')
                    ls_ERA5_raw_corr.append(np.nan)
                    ls_ERA5_raw_skill.append(np.nan)

                    ls_PRIMAVERA_classif.append(' ')
                    ls_PRIMAVERA_variable.append('nm')
                    ls_PRIMAVERA_raw_corr.append(np.nan)
                    ls_PRIMAVERA_raw_skill.append(np.nan)

                    continue      
                
            elif folder_name == 'TCW':
                if idx == 3 or idx == 4 or idx == 5 or idx == 6 or idx == 7: # for all indices greater than 2, we have no data
                    # append 'nm' for 'no measurement' to list, so that still all sites are represented
                    ls_ERA5_classif.append(' ')
                    ls_ERA5_variable.append('nm')
                    ls_ERA5_raw_corr.append(np.nan)
                    ls_ERA5_raw_skill.append(np.nan)

                    ls_PRIMAVERA_classif.append(' ')
                    ls_PRIMAVERA_variable.append('nm')
                    ls_PRIMAVERA_raw_corr.append(np.nan)
                    ls_PRIMAVERA_raw_skill.append(np.nan)

                    continue

            elif folder_name == 'seeing_nc':
                # for sites where we have no in situ reference, we have to exclude them here!
                if idx == 5 or idx == 6:
                    # append 'nm' for 'no measurement' to list, so that still all sites are represented
                    ls_ERA5_classif.append(' ')
                    ls_ERA5_variable.append('nm')
                    ls_ERA5_raw_corr.append(np.nan)
                    ls_ERA5_raw_skill.append(np.nan)

                    ls_PRIMAVERA_classif.append(' ')
                    ls_PRIMAVERA_variable.append('nm')
                    ls_PRIMAVERA_raw_corr.append(np.nan)
                    ls_PRIMAVERA_raw_skill.append(np.nan)

                    continue

        # # load in dataset
        # if variable == 'seeing model':
        #     df_skill = pd.read_csv('./Model_evaluation/'+ folder_name + '/csv_info/' + site_name_folder + file_name + 'osborn_sorted_skill_dict.csv', header=None)

        # elif variable == '200hPa-wind-speed seeing':
        #     df_skill = pd.read_csv('./Model_evaluation/'+ folder_name + '/csv_info/' + site_name_folder + file_name + '200hPa_wind_speed_sorted_skill_dict.csv', header=None)

        # else: # read in standard format
        df_skill = pd.read_csv('./Model_evaluation/'+ folder_name + '/csv_info/' + site_name_folder + file_name + 'sorted_skill_dict.csv', header=None)

        # go through rows of dataframe
        for index, row in df_skill.iterrows():
            # print(row[0])

            # ERA5: stop if ERA5 is in first column of that row
            if ERA5_is_ref == False: # otherwise, ERA5 is reference!
                if 'ERA5' in row[0] and count_ERA5 == 0:
                    # make sure that only the first (and best) ERA5 entry gets selected
                    count_ERA5 = 1 
                    # append classification to the classification list of the 200hPa wind speed seeing
                    ls_ERA5_classif.append(skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                    # append string to the variable list
                    if pressure_level == True:
                        # get pressure level from label
                        label_split = row[0].split(' ') # is a list
                        pr_lev = label_split[-1]

                        ls_ERA5_variable.append(pr_lev + 'hPa')
                    else:
                        ls_ERA5_variable.append('tcc') # total cloud cover
                    # to calculate the mean in the end
                    ls_ERA5_raw_skill.append(row[1])
                    ls_ERA5_raw_corr.append(row[2])

            # for the PRIMAVERA ensemble means, there are more than two in each dataset, but we want only the best one! (either 'hist' or 'present')
            if 'Ensemble' in row[0] and count_PRIMAVERA == 0: # we only want the Ensemble skills

                # increase counter of PRIMAVERA ensemble
                count_PRIMAVERA =  1

                # separate the string by spaces and save the last word ('hist', 'present', ...)
                label_split = row[0].split(' ') # is a list
                forcing = label_split[-1]
                # assign correct string
                if forcing=='present':
                    my_label = 'SST present'
                elif forcing == 'SSTfuture': # should have only 'present' and 'hist' up to now..
                    my_label = 'SST future'
                else:
                    my_label = forcing

                # get the skill score classification and compose string
                ls_PRIMAVERA_classif.append(skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                
                # append the forcing
                if pressure_level == True:
                    # get pressure level from label
                    pr_lev = label_split[1]
                    if '0' in pr_lev or '5' in pr_lev: 
                        # then, it is really a pressure level
                        # otherwise, it could be tas, the surface temperature!
                        ls_PRIMAVERA_variable.append(pr_lev + 'hPa ' + my_label)
                    else:
                        single_lev = pr_lev
                        ls_PRIMAVERA_variable.append(single_lev+ ' ' + my_label)

                else:
                    ls_PRIMAVERA_variable.append(my_label)

                # for mean
                ls_PRIMAVERA_raw_skill.append(row[1])
                ls_PRIMAVERA_raw_corr.append(row[2])
            

            elif 'Ensemble' in row[0] and count_PRIMAVERA == 1: # we want to know the difference between the best and second best Ensemble projection
                # print(row)

                count_PRIMAVERA =  2
                
                # separate the string by spaces and save the last word ('hist', 'present', ...)
                label_split = row[0].split(' ') # is a list
                forcing = label_split[-1]
                # assign correct 1st simulation
                if forcing=='present':
                    # then, hist is first entry
                    best_match_ensemble = 'hist'
                elif forcing=='hist': # should have only 'present' and 'hist' up to now..
                    # then, SST-present is first entry
                    best_match_ensemble = 'present'
                elif forcing == 'SSTfuture':
                    best_match_ensemble = 'future'
                elif forcing == 'future':
                    best_match_ensemble = 'SSTfuture'

                # calculate difference between best match and second best match
                # this works because every skill_score_dict has exactly two Ensemble entries
                # row[1] is now skill score of second best Ensemble
                # ls_PRIMAVERA_raw_skill[idx] is skill score of best match
                diff_match_ensemble = ls_PRIMAVERA_raw_skill[idx] - row[1]
                
                # write to list 
                # 1st row: site
                # (ls_site is already there)
                # 2nd row: name of best match (e.g. 'SST present')
                ls_best_match_ensemble.append(best_match_ensemble)
                # 3rd row: difference in skill score to second best match
                ls_diff_match_ensemble.append(diff_match_ensemble)

                                


    # check length of lists!
    for ls_key in nested_list_dict.keys():
        if len(nested_list_dict[ls_key]) > 8:
            raise Exception('investigate {}'.format(ls_key))


    # compose dataframe
    df_ERA5 = pd.DataFrame(list(zip(ls_site, ls_ERA5_variable, ls_ERA5_classif)),
                        columns = ['Site', title_var, 'classification'])

    df_PRIMAVERA = pd.DataFrame(list(zip(ls_site, ls_PRIMAVERA_variable, ls_PRIMAVERA_classif)),
                        columns = ['Site', title_var, 'classification'])



    # compose df_corr with classications!!!)
    if ERA5_is_ref == False:
        df_class = pd.DataFrame(list(zip(ls_site, ls_ERA5_raw_skill, ls_PRIMAVERA_raw_skill)),
                            columns = ['Site', 'ERA5', 'PRIMAVERA'])
        df_corr = pd.DataFrame(list(zip(ls_site, ls_ERA5_raw_corr, ls_PRIMAVERA_raw_corr)),
                            columns = ['Site', 'ERA5', 'PRIMAVERA'])
    else: # ERA5 is empty, because it is reference
        df_class = pd.DataFrame(list(zip(ls_site, ls_PRIMAVERA_raw_skill)),
                            columns = ['Site', 'PRIMAVERA'])
        df_corr = pd.DataFrame(list(zip(ls_site, ls_PRIMAVERA_raw_corr)),
                            columns = ['Site', 'PRIMAVERA'])

        # compose dataframe for difference in PRIMAVERA ensemble
        # we take PRIMAVERA to ERA5 to define this!
        df_ensemble_match_diff = pd.DataFrame(list(zip(ls_site, ls_best_match_ensemble, ls_diff_match_ensemble)),
                                            columns = ['Site', 'Best Simulation', 'Difference to second best simulation'])

    # calculations with corr
    # drop nan first
    df_corr_nonan = df_corr.dropna(how='any')
    # calc mean, max, min
    df_corr_nonan.max(axis=0) # max values of different approaches
    df_corr_nonan.max(axis=1) # max values of each site
    df_corr_nonan.min(axis=0) # min values of different approaches
    df_corr_nonan.min(axis=1) # min values of each site


    # calculate mean skill (just for me)
    print('mean of ERA5 classification = {}'.format(np.mean(ls_ERA5_raw_skill)))
    print('mean of PRIMAVERA classification = {}'.format(np.mean(ls_PRIMAVERA_raw_skill)))

    # return datasets
    if ERA5_is_ref == False:
        return df_ERA5, df_PRIMAVERA, df_class
    else:
        return df_ERA5, df_PRIMAVERA, df_class, df_ensemble_match_diff


def plot_axspan_classification(ax):
    # draw shaded areas for 'poor', 'mediocre', 'good', 'excellent'
    # sample color from 'plasma' colorbar
    # needs 'from matplotlib.colors import to_hex'
    # ls_col_class = list(reversed([to_hex(plt.cm.copper(i / 5)) for i in range(5)]))
    ls_col_class = [to_hex(plt.cm.copper(i / 5)) for i in range(5)]
    # poor
    ax.axvspan(0, 0.314, alpha=0.45, color=ls_col_class[0])
    # mediocre
    ax.axvspan(0.315, 0.624, alpha=0.45, color=ls_col_class[1])
    # good
    ax.axvspan(0.625, 0.904, alpha=0.45, color=ls_col_class[2])
    # excellent
    ax.axvspan(0.905, 1, alpha=0.45, color=ls_col_class[3])
    # add text on top (=0.9) of plot
    ax.text(0.16, 1, 'Poor', fontsize=12, ha="center", 
         transform=ax.transAxes)
    ax.text(0.5, 1, 'Mediocre', fontsize=12, ha="center", 
         transform=ax.transAxes)
    ax.text(0.76, 1, 'Good', fontsize=12, ha="center", 
         transform=ax.transAxes)
    ax.text(0.95, 1, 'Excellent', fontsize=12, ha="center", 
         transform=ax.transAxes)

    return

#%% 
def Prlevel_attributes_T_RH_SH():
    # generate attribute list automatically
    attrib = []
    attrib_ERA5 = []

    # load d_site_lonlat_data
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    for idx in range(0,8):
        # get pressure levels saved in d_site_lonlat_data for ERA5 and PRIMAVERA
        ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
        ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
        # append to list
        # [0] because there is only one pressure level (or should be!)
        attrib.append('_' + str(ls_pr_levels_clim_model[0]))
        attrib_ERA5.append('_' + str(ls_pr_levels_ERA5[0]))
    
    return attrib_ERA5, attrib

# attrib_ERA5, attrib = Prlevel_attributes_T_RH_SH()


#%% define function

def trend_analysis_plot(variable, gs, fig, fig_idx):

    # custom parameters for individual variables
    if variable == 'wind_speed_seeing':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/wind_speed/"
        attrib = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        unit = 'arcsec'
        title = '200hPa-wind-speed seeing'

    elif variable == 'seeing_osborn':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/osborn/"
        attrib = ['_200', '_200', '_200', '_200','_200', '_200', '_200', '_200']
        attrib_ERA5 = ['_800', '_900', '_825', '_825','_975', '_950', '_850', '_850']
        unit = 'arcsec'
        title = 'seeing model'

    elif variable == 'SH':
        base_path =  "./Model_evaluation/SH/future_trends/"
        # attrib = ['_700', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib_ERA5 = ['_600', '_775', '_775', '_775','_800', '_950', '_850', '_750']
        unit = '(g/kg)'
        title = 'Specific Humidity'

    elif variable == 'RH':
        base_path =  "./Model_evaluation/RH/future_trends/"
        # attrib = ['_600', '_700', '_850', '_850','_850', '_1000', '_850', '_700']
        # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib_ERA5 = ['_600', '_750', '_775', '_750','_850', '_950', '_875', '_750']
        unit = '%'
        title = 'Relative Humidity'

    elif variable == 'T':
        base_path =  "./Model_evaluation/T/future_trends/"
        # attrib = ['_single_level', '_single_level', '_850', '_single_level','_850', '_925', '_single_level', '_700']
        # attrib = ['_600', '_700', '_850', '_850','_700', '_single_level', '_single_level', '_700']
        # attrib_ERA5 = ['_600', '_750', '_750', '_775','_750', '_900', '_875', '_750']
        unit = '°C'
        title = 'Temperature'

    elif variable == 'total_cloud_cover':
        base_path =  "./Model_evaluation/total_cloud_cover/future_trends/"
        attrib = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
        attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
        unit = '%'
        title = 'Total cloud cover'

    elif variable == 'TCW':
        base_path = "./Model_evaluation/TCW/future_trends/"
        attrib = ['_600', '_700', '_700', '_850','_700', '_925', '_850', '_700']
        attrib_ERA5 = ['_600', '_750', '_775', '_775','_775', '_900', '_825', '_750']
        unit = 'mmH2O'
        title = 'Precipitable water vapor'

    # load d_site_lonlat_data
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    # for T, RH and SH, we have fixed Plevs
    if variable == 'T' or variable == 'RH' or variable == 'SH':
        attrib_ERA5, attrib = Prlevel_attributes_T_RH_SH()

    ls_site_names = []
    line_list = []

    ls_hex = [to_hex(plt.cm.terrain(i / 8)) for i in range(8)] 

    # fig, ax = plt.subplots(figsize=(8,4))
    # add subplot to figure
    ax = fig.add_subplot(gs[int((fig_idx - (fig_idx%2))/2), fig_idx%2])

    # for every variable, read in csv that tells us how big the difference between PRIMAVERA Ensemble best and second best match is
    # we want to display this in the trend analysis plot so that we know which scored best
    df_ensemble_match_diff = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_difference_' + variable + '.csv')


    for idx in range(0, 8):
        
        site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
        # "Mauna_Kea", "Cerro_Paranal", "La_Silla", "Cerro_Tololo", "La_Palma", "Siding_Spring", "Sutherland", "SPM"
        # change site name if needed
        if idx == 0: # mauna kea
            site_name_folder = 'Mauna_Kea'
        if idx == 1:
            site_name_folder = 'Cerro_Paranal'
        
        ls_site_names.append(site_name_folder.replace('_', ' '))


        ls_forcing = ['hist', 'present', 'future', 'SSTfuture', 'ERA5'] # 'future', 'SSTfuture', and eventually 'ERA5'
        for forcing in ls_forcing:

            ########## ERA5
            if forcing == 'ERA5':
                if variable == 'wind_speed_seeing' or variable == 'seeing_osborn':
                    ERA5_path = "./Model_evaluation/seeing_nc/ERA5_trends/"
                else:
                    ERA5_path = "./Model_evaluation/" + variable + "/ERA5_trends/"
                df = pd.read_csv(ERA5_path + site_name_folder + attrib_ERA5[idx] + variable + '_' + '_ERA5_Projections_Bayesian_model_map2stan.csv')

            else:
                # read PRIMAVERA csv
                df = pd.read_csv(base_path + site_name_folder + '_' + forcing + attrib[idx] + variable + '_' + '_PRIMAVERA_Projections_Bayesian_model_map2stan.csv')

            # rename 
            df = df.rename(columns={'Unnamed: 0': 'parameter'})

            # plot b (the gradient of the slope if the seasonal variation was excluded)
            # use idx as x axis (idx and idx+0.5)
            # get marker from function
            # df['mean'][1] is the gradient b
            # multiply with 10 to get it in unit/decade
            if forcing == 'future' or forcing == 'SSTfuture':
                
                color = '#009e73'

                if forcing == 'future':
                    x_idx = idx + 0.8
                else:
                    x_idx = idx + 0.95
                
            else:
                if forcing == 'ERA5':
                    color = 'k' # ERA5 is the new ground truth and should therefore be black # '#0072b2' # '#56b4e9'
                    x_idx = idx + 0.1               

                else: # hist or present
                    color = '#e69f00'#  
                    if forcing == 'hist':
                        x_idx = idx + 0.3
                    else:
                        x_idx = idx + 0.45
            # get color from climxa.return_model_color(Mname)
            # color = climxa.return_model_color(forcing)[0] # is list...
            # DO NOT get color from climxa, because then, coupled have the same color, but here I want future to have the same color!

            # get marker for individual forcings (and ERA5)
            marker = trend_marker(forcing)
            markersize_scale = 20

            # not for ERA5
            if forcing != 'ERA5': # only for PRIMAVERA trends
                if forcing == 'present' or forcing == 'hist': # 
                    # get markersize and alpha-value depending on difference 
                    if df_ensemble_match_diff['Best Simulation'][idx] == forcing:
                        # if the best simulation matches the current forcing, apply no special markersize and alpha
                        # alpha = 0.9
                        markersize = 11
                        # but we have to store somehow if coupled or atmos-only is best match!
                        # atmos-only (SST present, SST future)
                        if forcing == 'present':
                            sim_keyword = 'atmos-only'
                            # sim-keyword indicateds best match!
                        elif forcing == 'hist': # forcing == 'hist' or 'future'
                            sim_keyword = 'coupled'
                    else: # df_ensemble_match_diff['Best Simulation'][idx] != forcing
                        # smaller markersize for second best match, alpha corresponding to difference
                        # e.g. if best match is 'present', then we have 'hist' here
                        markersize = 9 - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]
                    
                        # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]

                    # elif df_ensemble_match_diff['Best Simulation'][idx] != forcing:
                    #     # but we have to store somehow if coupled or atmos-only is best match!
                    #     # atmos-only (SST present, SST future)
                    #     if forcing == 'present' or forcing == 'SSTfuture':
                    #         sim_keyword = 'atmos-only'
                    #         # sim-keyword indicateds best match!
                    #     else: # forcing == 'hist' or 'future'
                    #         sim_keyword = 'coupled'
                else: # forcing == 'future' or 'SSTfuture'
                    if sim_keyword == 'coupled': # best match is 'hist', 'future' should be big, 'SST future' small
                        if forcing == 'future':
                            # display big
                            # alpha = 0.9
                            markersize = 11
                        else: # forcing == 'SSTfuture'
                            # smaller
                            markersize = 9  - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]
                    
                            # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]
                    
                    elif sim_keyword == 'atmos-only':
                        if forcing == 'SSTfuture':
                            # display big
                            
                            markersize = 11
                        else: # forcing == 'future'
                            # smaller
                            markersize = 9  - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]
                    
                            # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]


            else: # forcing == 'ERA5'
                # alpha = 0.9
                markersize = 10 # ERA5

            # alpha = 0.9

            #plt.scatter(x_idx, df['mean'][1]*120, marker = marker, s = 130, c=color, alpha = 0.85)
            
            # plot black errorbars
            # errorbars must be positive (one goes in minus y direction, the other in plus)
            # this line also plots the data!
            # yerr = df['sd'][1] # if I want to show the standard deviation

            ##### plot
            ax.errorbar(x_idx, df['mean'][1]*120, yerr=np.array([[abs(df['mean'][1]*120 - df['5.5%'][1]*120), abs(df['mean'][1]*120 - df['94.5%'][1]*120)]]).T, 
                        c=color, markeredgecolor = 'k', ecolor=color, markersize = markersize, marker=marker ) # , alpha=0.8
                    # ecolor='k'

            # for better visibility, plot vertical background color that fill the space of one site
            # if idx%2 != 0:
            #     ax.axvspan(idx, idx + 1, alpha=0.02, color='red')

            # individual color for every site
            ax.axvspan(idx, idx + 1, alpha=0.07, color=ls_hex[idx])
                        
            # for legend
            if idx == 0:
                if forcing=='present':
                    my_label = 'SST present (atmos-only)'
                elif forcing == 'SSTfuture':
                    my_label = 'SST future (atmos-only)'
                elif forcing == 'hist':
                    my_label = 'hist (coupled)'
                else: # forcing == future
                    my_label = 'future (coupled)'
                # for legend
                line_list.append(Line2D([0], [0], linestyle = '', color = color, marker = marker, label = my_label))
    
    # add line at y = 0.0 to emphasize the state zero
    ax.axhline(y=0, xmin=0, xmax=8, color = 'red')

    # append errorbar to legend
    line_list.append(Line2D([0], [0], linestyle = '-', color = 'k', label = '89-percentile interval'))


    # legend only for last plot!
    if fig_idx == 6:
        ax.legend(handles=line_list, loc='lower left', bbox_to_anchor= (1.2, 0))
    
    ax.set_xticks(np.arange(0, 8)) #, rotation = 60

    # for seeing, set same ylimits
    if fig_idx == 4 or fig_idx == 5:
        ax.set_ylim(-0.03, 0.05)

    # # if labels should only be displayed at the bottom of the plot
    # if fig_idx == 5 or fig_idx == 6:
    #     ax.set_xticklabels(ls_site_names)
    # else:
    #     plt.setp(ax.get_xticklabels(), visible=False)

    # else, plot x-labels for all variables
    # set labels for the x axis (site names)
    ax.set_xticklabels(ls_site_names)

    # shrink axis to make space for xlabels
    box = ax.get_position()
    ax.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

    # set xlim (otherwise there is unused space left and right of the plots)
    ax.set_xlim(0, 8)

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-30, ha="left" )

    ax.set_ylabel(unit + ' per decade')

    # write a), b) 
    if title != None:
        ax.set_title(alphabet_from_idx(fig_idx) + ') Trends of ' + title)
    else:
        ax.set_title(alphabet_from_idx(fig_idx) + ') Trends of ' + variable.replace('_', ' '))


    # save fig
    # fig.savefig(base_path + 'Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)
    # # also save it into figures folder for publication
    # fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/figures/' + variable + '_Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)

    # plt.show()

    return fig

#%%

def d_color_generator():
     # creates dictionary that assigns a distinct color to each pressure level
    # d_color = {'100': 'blueviolet', '125': 'tomato',
    #         '150': 'lightsalmon', '175': 'peru', '200': 'khaki',
    #         '225': 'rosybrown', '250': 'sandybrown', '300': 'darkseagreen',
    #         '350': 'palegreen', '400': 'pink', '450': 'brown',
    #         '500': 'indianred', '550': 'darkred', 
    d_color = {'200': 'darkseagreen','600': '#e69f00',#'red', 
        '650': 'blue', '700': 'maroon' ,'750': '#56b4e9', #'750': 'darkgoldenrod', # '700': 'darkorange'
        '775': '#009e73', '800': 'blueviolet', '825': 'palegreen', # '800': 'cadetblue'
        '850': 'olive', '875': 'violet', '900': 'royalblue', # '800': 'darkgreen', '875': 'darkcyan'
        '925': 'indigo', '950': 'darkgreen', '975': 'yellowgreen', # '900': 'indigo', '950': 'violet','975':'palevioletred'
        '1000': 'lightsalmon', # 
        'prw':  'navy', 'tcw': '#0072b2', 't2m': '#d55e00', 'tas': '#d55e00', 'q_integrated': 'peru',
        'wind_speed_seeing': 'tomato', 'wind speed seeing': 'tomato'} # single level variables
# prw: 'navy',

    #inferno_colors = [to_hex(plt.cm.gist_ncar(i / 14)) for i in range(14)] 
    # inferno_colors = inferno_colors[2:] # do not use the first to, since they are almost black
    # # 14 pressure levels in use
    # viridis_colors = [to_hex(plt.cm.viridis(i / 8)) for i in range(8)] 
    # viridis_colors = viridis_colors[:-1] # do not take last oney, since they are also almost yellow

    # d_color = {'600': inferno_colors[0],#'red', 
    #         '650': inferno_colors[1], '700': inferno_colors[2] , '750': inferno_colors[3], #'750': 'darkgoldenrod', # '700': 'darkorange'
    #         '775': inferno_colors[4], '800': inferno_colors[5], '825': inferno_colors[6], # '800': 'cadetblue'
    #         '850': inferno_colors[7], '875': inferno_colors[8], '900': inferno_colors[9], # '800': 'darkgreen', '875': 'darkcyan'
    #         '925': inferno_colors[10], '950': inferno_colors[11], '975': inferno_colors[12], # '900': 'indigo', '950': 'violet','975':'palevioletred'
    #         '1000': inferno_colors[13], # 
    #         # single level variables
            # 'prw': '#56b4e9', 'tcw': '#0072b2', 't2m': 'yellowgreen', 'tas': 'yellowgreen', 'q_integrated': '#009e73'} # single level variables
    
    return d_color

def d_color_generator_ind_model(PRIMAVERA=True, second_pub_dataset=False):

    if PRIMAVERA == True:
        if second_pub_dataset == True:
            d_color = {'ERA5': 'tomato', 'PRIMAVERA coupled': 'orange', 'PRIMAVERA atmos-only': 'blue'}
        else:
            d_color = {'ERA5': 'maroon', 'PRIMAVERA coupled': '#d55e00', 'PRIMAVERA atmos-only': 'navy'}
        
    else:
        d_color = {'ERA5': 'k'} # ERA5 changed from 'maroon' to 'k', because it is new ground truth

    return d_color


def trend_marker(forcing):
    # coupled (hist and future) and atmos-only (present and SSTfuture) should have the same marker
    mar_collection = {'ERA5': '*',
                        'hist': 's', # '^',
                        'present': 'h', # ,
                        'future': 's',
                        'SSTfuture': 'h'} # 'd'
                        
    trend_marker = mar_collection[forcing]
    return trend_marker

def clim_key_marker(clim_key):

    mar_collection_clim_models = {'HadGEM': 'p',
                                'EC-Earth': 'h',
                                'CNRM': 's',
                                'MPI': 'd',
                                'CMCC': '^',
                                'ECMWF': 'v'}
    clim_marker = mar_collection_clim_models[clim_key]
    return clim_marker

def P_lev_color(P_lev):
    # calls d_color_generator for the color-dict
    d_color = d_color_generator()
    try:
        P_color = d_color[str(P_lev)]
    except KeyError:
        print(f'no color assigned for Pressure level {P_lev}. I just assign navy.')
        P_color = 'navy'
    return P_color

def return_model_color(Mname, second_pub_dataset=False):
    # read in dict (include PRIMAVERA)
    
    # if second_pub_dataset==True:
    #     d_color = d_color_generator_ind_model(PRIMAVERA=True, second_pub_dataset=True)
    # else:
    d_color = d_color_generator_ind_model(PRIMAVERA=True, second_pub_dataset=second_pub_dataset)


    # Mname might be the forcing
    # if so, assign correct label (coupled or atmos-only)
    if Mname == 'hist' or Mname == 'future':
        Mname = 'PRIMAVERA coupled'
    elif Mname == 'present' or Mname == 'SSTfuture':
        Mname = 'PRIMAVERA atmos-only'

    # extract color for model of input
    try:
        Mcolor = d_color[Mname]
    except KeyError: # if Mname not in d_color
        Mcolor = 'tomato'
        print(f'no color assigned for model {Mname}. I just assign tomato.')
    
    return [Mcolor]

#%%
# implement taylor plot into DSCT
    # input: d_obs['ds] already merged with ERA5 data (and loaded)
    # input: d_model (ds_Amon for monthly Taylor diagram)

def time_intersect(d_obs=None, d_model = None):
    merge_ls = []

    if d_obs!=None:
        # resample dataset with ERA5 and observations monthly (for monthly taylor diagram)
        d_obs['ds_taylor'] = d_obs['ds_sel'].resample(time = '1m', keep_attrs=True).mean() # lon/lat selected
        # if it is already resampled, nothing should happen
        
        # take only year and month from datetime index (because in ds_Amon, the value is assigned to YYYY-MM-16)
        d_obs['ds_taylor']['time'] = d_obs['ds_taylor']['time'].dt.strftime('%Y-%m')
        d_obs['ds_taylor']['time'] = pd.to_datetime(d_obs['ds_taylor'].indexes['time'])

        # list for merger
        # reset coordinates (drop all coordinates that are non-index)
        # therefore, select first level (so that this dimension goes away)
        # and hope, that time is the same for all levels (this was the case so far) -->but not anymore, so drop all nans in all pressure levels
        if 'level' in d_obs['ds_taylor'].coords:
            # loop through pressure levels and drop nan's
            try:
                merge_ls = [d_obs['ds_taylor'].reset_coords(drop = True).isel(level=0)]
            except ValueError: # happens for wind speed seeing, which has a level, but the level is already 'selected', and I ingest it as single level data
                print('WARNING: dimensions level do not exist!')
                merge_ls = [d_obs['ds_taylor'].reset_coords(drop = True)] 
        else: # no need to select any levels
            merge_ls = [d_obs['ds_taylor'].reset_coords(drop = True)] 


    if d_model != None:
        for i, clim_key in enumerate(d_model.keys()):
            #if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
            # Since resampling doesn't work (for HadGEM, time is object, not convertable to datetime), make a deep copy of ds_sel_cycle
            # d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_sel_cycle'].resample(time='1m').mean()
            d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_sel_cycle'].copy(deep=True) # makes a deep copy (so that datasets do not change in the same way)

            # take out only YYYY-MM from CF time index (change for whole dataset)
            d_model[clim_key]['ds_taylor']['time'] = d_model[clim_key]['ds_taylor']['time'].dt.strftime('%Y-%m')
            d_model[clim_key]['ds_taylor']['time'] = pd.to_datetime(d_model[clim_key]['ds_taylor'].indexes['time'])
            
            if 'clim_var' in d_model[clim_key].keys():
                for clim_var in d_model[clim_key]['clim_var']: #d_model[clim_key]['clim_var']:  d_model[clim_key]['ds_taylor'].data_vars
                    # loop through different forcings:
                    # only append those in 'taylor_folders', since 'time_intersect' is only for Taylor diagram and seasonal cycle
                    
                    if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                        for forcing in d_model[clim_key]['taylor_folder']: # e.g. 'hist'
                            forced_clim_var = clim_var + ' ' + forcing
                            # reset counter for every forcing
                            counter_plev_empty_forcing = 0 
                            # copy only clim_var to new dataset (and leave behind time_bnds,...)
                            # select first level (so that this dimension also goes away)

                            # loop through pressure levels and drop all nans
                            for Plev in d_model[clim_key]['Plev']:
                                # check if array is empty
                                ds_check = d_model[clim_key]['ds_taylor'][forced_clim_var].sel(level=Plev).dropna(how='all', dim='time')
                                # print(ds_check['time'].size)
                                if ds_check['time'].size != 0:
                                    d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_taylor'][forced_clim_var].sel(level=Plev).dropna(how='all', dim='time')}) 
                                    # d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_taylor'][forced_clim_var].isel(level=0)}) 
                                    
                                    # append dataset to merge_ls (list with datasets that are going to be merged), 
                                    # but rename the variable (otherwise they are all called the same)
                                    # include pressure level in new name, and drop level-coord
                                    # and drop coordinates that are non-index cotime_intersectr'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)+ str(Plev)}))
                                    merge_ls.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)+ str(Plev)}))
                                else:
                                    # if it is empty, the corresponding forcing needs to be removed from 'taylor_folder', 
                                    # only if it is empty for all pressure levels (check with a counter)
                                    # so that it doesn't lead to conflicts when plotting (and appending ref_pred_ls)
                                    print('I do not append merge_ls for {} of {}'.format(forced_clim_var, clim_key))
                                    # append list of forced_clim_var + pressure level to filter it later

                                    # increase counter
                                    counter_plev_empty_forcing = counter_plev_empty_forcing + 1
                                    # if counter reaches number of pressure levels, so if forcing is empty for all pressure levels, remove forcing from taylor_folder
                                    if counter_plev_empty_forcing == len(d_model[clim_key]['Plev']): # if array is empty for all pressure levels
                                        print('forcing contains empty folders for all pressure levels')
                                        if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                                            d_model[clim_key]['taylor_folder'].remove(forcing)

            if 'single_lev_var' in d_model[clim_key].keys():
                for clim_var in d_model[clim_key]['single_lev_var']:
                    # loop through different forcings:
                    if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                        for forcing in d_model[clim_key]['taylor_folder']: # e.g. 'hist'
                            forced_clim_var = clim_var + ' ' + forcing
                            # copy only clim_var to new dataset (and leave behind time_bnds,...)
                            d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_taylor'][forced_clim_var]}) 
                            
                            # append dataset to merge_ls (list with datasets that are going to be merged), 
                            # but rename the variable (otherwise they are all called the same)
                            # and drop coordinates that are non-index coordinates (like lon and lat, because there is only one selected)
                            merge_ls.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)}))

    # merge datasets to get intersected time index with join=outer to use the union of object indexes as a first step (before choosing intersecting values only)!
    ds_Ty_merged = xr.merge(merge_ls, join='outer', compat = 'no_conflicts') # compat = override is needed e.g. for 
                                                                    # (all datasets have different longitude selected, but that doesn't interest me right now, since I am only interested in the intersecting time)
    # fast way: .where(xr.ufuncs.isnan(ds_taylor['La_Palma Specific Humidity']) != True, drop = True).where(xr.ufuncs.isnan(ds_taylor['q']) != True, drop = True).where(xr.ufuncs.isnan(ds_Amon['hus']) != True, drop = True)
    # drop nans in every variable
    # drop nans for insitu data
    if d_obs!=None:
        if 'insitu_var' in d_obs.keys():
            for insitu_param in d_obs['insitu_var']:
                ds_Ty_merged = ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[insitu_param]) != True, drop = True)
        
        # drop nans for ERA5 data
        if 'ERA5_var' in d_obs.keys():
            # loop through climate variables to plot
            for clim_var in d_obs['ERA5_var']:
                ds_Ty_merged = ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[clim_var]) != True, drop = True)
        # don't forget single level data
        if 'single_lev' in d_obs.keys():
            for clim_var in d_obs['single_lev']:
                ds_Ty_merged = ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[clim_var]) != True, drop = True)
        
    # drop nans for climate model data
    if d_model != None:
        for i, clim_key in enumerate(d_model.keys()):
            # loop through different climate variables
            if 'clim_var' in d_model[clim_key].keys():
                for clim_var in d_model[clim_key]['clim_var']:
                    # now only loop through folders that should be taken for the taylor diagram (otherwise, ds_Ty_merged is zero!)
                    if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                        for forcing in d_model[clim_key]['taylor_folder']: # e.g. 'hist'
                            forced_clim_var = clim_var + ' ' + forcing
                            for Plev in d_model[clim_key]['Plev']:
                                if forced_clim_var + str(i)+ str(Plev) in ds_Ty_merged.data_vars:
                                    # print(ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[clim_var + str(i)]) != True, drop = True))
                                    ds_Ty_merged = ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[forced_clim_var + str(i)+ str(Plev)]) != True, drop = True)
                
            if 'single_lev_var' in d_model[clim_key].keys():
                for clim_var in d_model[clim_key]['single_lev_var']:
                    # now only loop through folders that should be taken for the taylor diagram (otherwise, ds_Ty_merged is zero!)
                    if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                        for forcing in d_model[clim_key]['taylor_folder']: # e.g. 'hist'
                            forced_clim_var = clim_var + ' ' + forcing
                            ds_Ty_merged = ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[forced_clim_var + str(i)]) != True, drop = True)

    # ds_Ty_merged has one flaw, therefore it cannot be used for other things than the intersected time
    # the flaw is the merging of the pressure levels, which leaves behind only the intersecting pressure levels
    
    # return ds_Ty_merged['time'] with reset coordinates
    # otherwise, it has a level and automatically destroys levels in other datasets
    return ds_Ty_merged['time'].reset_coords(drop = True)

# # compare d_obs, ds_merged and ds_Ty_merged
# d_obs['ds_merged']['q'].sel(level=700).plot(marker='o', markersize=2, label='ds_merged')
# d_obs['ds']['q'].sel(level=700).plot(marker='o', markersize=2, label='ds')
# ds_Ty_merged['q'].sel(level=700).plot(marker='o', markersize=2, label='ds_Ty_merged')
# plt.legend()



# define a function solely for the Ensemble mean of the monthly timeseries of the PRIMAVERA data
def PRIMAVERA_time_intersect(d_model):
    # for the PRIMAVERA model ensemble, we need all forcings, and all available intersecting months
    # we use 'ds_sel' 
    merge_ls_hist = []
    merge_ls_future = []
    # merge for hist/present and future/SSTfuture separately! (otherwise there is no intersection!)

    for i, clim_key in enumerate(d_model.keys()):
        #if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
        # Since resampling doesn't work (for HadGEM, time is object, not convertable to datetime), make a deep copy of ds_sel_cycle
        
        d_model[clim_key]['ds_monthly_timeseries_intersect'] = d_model[clim_key]['ds_sel'].copy(deep=True) # makes a deep copy (so that datasets do not change in the same way)

        # take out only YYYY-MM from CF time index (change for whole dataset)
        # change time from e.g. 1950-01-16 to 1950-01-01
        d_model[clim_key]['ds_monthly_timeseries_intersect']['time'] = d_model[clim_key]['ds_monthly_timeseries_intersect']['time'].dt.strftime('%Y-%m')
        d_model[clim_key]['ds_monthly_timeseries_intersect']['time'] = pd.to_datetime(d_model[clim_key]['ds_monthly_timeseries_intersect'].indexes['time'])
        
        if 'clim_var' in d_model[clim_key].keys():
            for clim_var in d_model[clim_key]['clim_var']: #d_model[clim_key]['clim_var']:  d_model[clim_key]['ds_taylor'].data_vars
                # loop through different forcings:
                # only append those in 'taylor_folders', since 'time_intersect' is only for Taylor diagram and seasonal cycle
                
                if bool(d_model[clim_key]['folders']) == True: # if list is not empty
                    for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                        forced_clim_var = clim_var + ' ' + forcing
                        # reset counter for every forcing
                        counter_plev_empty_forcing = 0 
                        # copy only clim_var to new dataset (and leave behind time_bnds,...)
                        # select first level (so that this dimension also goes away)

                        # loop through pressure levels and drop all nans
                        for Plev in d_model[clim_key]['Plev']:
                            # check if array is empty
                            ds_check = d_model[clim_key]['ds_monthly_timeseries_intersect'][forced_clim_var].sel(level=Plev).dropna(how='all', dim='time')
                            # print(ds_check['time'].size)
                            if ds_check['time'].size != 0:
                                d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_monthly_timeseries_intersect'][forced_clim_var].sel(level=Plev).dropna(how='all', dim='time')}) 
                                # d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_monthly_timeseries_intersect'][forced_clim_var].isel(level=0)}) 
                                
                                # append dataset to merge_ls (list with datasets that are going to be merged), 
                                # but rename the variable (otherwise they are all called the same)
                                # include pressure level in new name, and drop level-coord
                                # and drop coordinates that are non-index cotime_intersectr'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)+ str(Plev)}))
                                if forcing == 'hist' or forcing == 'present':
                                    merge_ls_hist.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)+ str(Plev)}))
                                else:
                                    merge_ls_future.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)+ str(Plev)}))
                            else:
                                # if it is empty, the corresponding forcing needs to be removed from 'taylor_folder', 
                                # only if it is empty for all pressure levels (check with a counter)
                                # so that it doesn't lead to conflicts when plotting (and appending ref_pred_ls)
                                print('I do not append merge_ls for {} of {}'.format(forced_clim_var, clim_key))
                                # append list of forced_clim_var + pressure level to filter it later

                                # increase counter
                                counter_plev_empty_forcing = counter_plev_empty_forcing + 1
                                # if counter reaches number of pressure levels, so if forcing is empty for all pressure levels, remove forcing from taylor_folder
                                if counter_plev_empty_forcing == len(d_model[clim_key]['Plev']): # if array is empty for all pressure levels
                                    print('forcing contains empty folders for all pressure levels')
                                    if bool(d_model[clim_key]['folders']) == True: # if list is not empty
                                        d_model[clim_key]['folders'].remove(forcing)

        if 'single_lev_var' in d_model[clim_key].keys():
            for clim_var in d_model[clim_key]['single_lev_var']:
                # loop through different forcings:
                if bool(d_model[clim_key]['folders']) == True: # if list is not empty
                    for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                        forced_clim_var = clim_var + ' ' + forcing
                        # copy only clim_var to new dataset (and leave behind time_bnds,...)
                        d_model[clim_key]['tempTaylor'] = xr.Dataset({forced_clim_var:  d_model[clim_key]['ds_monthly_timeseries_intersect'][forced_clim_var]}) 
                        
                        # append dataset to merge_ls (list with datasets that are going to be merged), 
                        # but rename the variable (otherwise they are all called the same)
                        # and drop coordinates that are non-index coordinates (like lon and lat, because there is only one selected)
                        if forcing == 'hist' or forcing == 'present':
                            merge_ls_hist.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)}))
                        else:
                            merge_ls_future.append(d_model[clim_key]['tempTaylor'].reset_coords(drop = True).rename({forced_clim_var: forced_clim_var + str(i)}))
    # merge datasets to get intersected time index with join=outer to use the union of object indexes as a first step (before choosing intersecting values only)!
    ds_Ty_merged_hist = xr.merge(merge_ls_hist, join='outer', compat = 'no_conflicts') # compat = override is needed e.g. for 
                                                                    # (all datasets have different longitude selected, but that doesn't interest me right now, since I am only interested in the intersecting time)
    ds_Ty_merged_future = xr.merge(merge_ls_future, join='outer', compat = 'no_conflicts')
    # fast way: .where(xr.ufuncs.isnan(ds_taylor['La_Palma Specific Humidity']) != True, drop = True).where(xr.ufuncs.isnan(ds_taylor['q']) != True, drop = True).where(xr.ufuncs.isnan(ds_Amon['hus']) != True, drop = True)
    # drop nans in every variable

    #### filter nanas
    for i, clim_key in enumerate(d_model.keys()):
        # loop through different climate variables
        if 'clim_var' in d_model[clim_key].keys():
            for clim_var in d_model[clim_key]['clim_var']:
                # now only loop through folders that should be taken for the taylor diagram (otherwise, ds_Ty_merged is zero!)
                if bool(d_model[clim_key]['folders']) == True: # if list is not empty
                    for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                        forced_clim_var = clim_var + ' ' + forcing
                        for Plev in d_model[clim_key]['Plev']:
                            if forced_clim_var + str(i)+ str(Plev) in ds_Ty_merged_hist.data_vars:
                                # print(ds_Ty_merged.where(xr.ufuncs.isnan(ds_Ty_merged[clim_var + str(i)]) != True, drop = True))
                                ds_Ty_merged_hist = ds_Ty_merged_hist.where(xr.ufuncs.isnan(ds_Ty_merged_hist[forced_clim_var + str(i)+ str(Plev)]) != True, drop = True)
                            elif forced_clim_var + str(i)+ str(Plev) in ds_Ty_merged_future.data_vars:
                                ds_Ty_merged_future = ds_Ty_merged_future.where(xr.ufuncs.isnan(ds_Ty_merged_future[forced_clim_var + str(i)+ str(Plev)]) != True, drop = True)
            
        if 'single_lev_var' in d_model[clim_key].keys():
            for clim_var in d_model[clim_key]['single_lev_var']:
                # now only loop through folders that should be taken for the taylor diagram (otherwise, ds_Ty_merged is zero!)
                if bool(d_model[clim_key]['folders']) == True: # if list is not empty
                    for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                        forced_clim_var = clim_var + ' ' + forcing
                        if forced_clim_var + str(i) in ds_Ty_merged_hist.data_vars:
                            ds_Ty_merged_hist = ds_Ty_merged_hist.where(xr.ufuncs.isnan(ds_Ty_merged_hist[forced_clim_var + str(i)]) != True, drop = True)
                        elif forced_clim_var + str(i) in ds_Ty_merged_future.data_vars:
                            ds_Ty_merged_future = ds_Ty_merged_future.where(xr.ufuncs.isnan(ds_Ty_merged_future[forced_clim_var + str(i)]) != True, drop = True)

    # return the time of both datasets
    # order: hist, future
    return ds_Ty_merged_hist['time'].reset_coords(drop = True), ds_Ty_merged_future['time'].reset_coords(drop = True)



def SH_int_func(i, delta_p, specific_humidity):
    # define gravitational acceleration (implemented in ERA5)
    g = 9.81

    return delta_p * specific_humidity / g # k selects the time, i selects the level

def Euler_centred_PWV(ds_SH_pressure_levels, pr_max_idx, variable, tcw_profile=False):
    # integrate with Euler scheme (Centred in Space)
    summe = 0
    ds_tcw_profile = xr.Dataset()
    for i in range(0, pr_max_idx + 1): # go until closest pressure value (that is, until value of 'SH_integral_pressure')
        Pr_i0 = ds_SH_pressure_levels.level[i] # starts at 1hPa (model data) and 250hPa (ERA5 data)
        
        try: # try to take i+1 level
            Pr_ip1 = ds_SH_pressure_levels.level[i+1] 

            # find delta_p
            if i == 0:
                # 100* because for conversion to Pa (otherwise, integral is wrong)
                delta_p = 100 * (Pr_ip1 - Pr_i0) # pr_levels_Pa[int(i+1)]  - pr_levels_Pa[i]
            
            # elif i == (len(pr_levels_Pa) - 1): # highest index of pr_levels_Pa is 1 smaller than length
            #     delta_p = pr_levels_Pa[i] - pr_levels_Pa[int(i-1)]
            else:
                Pr_im1 = ds_SH_pressure_levels.level[i-1] 
                delta_p =  100 * (0.5 * (Pr_ip1 - Pr_im1)) #  0.5 * (pr_levels_Pa[int(i+1)] - pr_levels_Pa[int(i-1)])
                # * 100 for transformation from hPa to Pa
        except IndexError: # then we reached the end of the possible pressure levels (above 1000hPa)
            # use Euler backward
            Pr_im1 = ds_SH_pressure_levels.level[i-1] 
            delta_p = 100* (Pr_i0 - Pr_im1)

        specific_humidity = ds_SH_pressure_levels[variable].isel(level=i) # ERA5 specific humidity is always called 'q', but you can also pass 'ERA5_var'


        # if summe != np.nan: # prevent nan values from going into 'summe', otherwise whole summe is nan
        summe = summe + SH_int_func(i, delta_p, specific_humidity) # this way, 'summe' becomes a DataArray
        # else:
        #     print('found nan')

        if tcw_profile:
            # vertical profile: write individual values into xarray
            dA_tcw_profile = SH_int_func(i, delta_p, specific_humidity) # this way, 'summe' becomes a DataArray
            dA_tcw_profile = dA_tcw_profile.assign_coords({"level": Pr_i0})
            dA_tcw_profile = dA_tcw_profile.rename('q_integrated_profile')
            dA_tcw_profile = dA_tcw_profile.reset_coords(['longitude', 'latitude'], drop =True)
            #ds_tcw_profile = ds_tcw_profile.reset_coords(['longitude', 'latitude'], drop =True)
            ds_tcw_profile_from_dA = dA_tcw_profile.to_dataset()

            # concatenate DataArrays along new dimension 'level'
            if i == 0:
                ds_tcw_profile = ds_tcw_profile_from_dA
            else:
                ds_tcw_profile = xr.concat([ds_tcw_profile, ds_tcw_profile_from_dA], 'level')

            # ds_tcw_profile = xr.Dataset({"q_integrated_profile": ds_tcw_profile}) # --> is already a dataset

            return summe, ds_tcw_profile

        else:
            return summe


def SH_integral_to_TCW(SH_integral_pressure, site, my_ERA5_lon, lat):
    # this function calculates the integral of the specific humidity over the pressure and stores this
    # in a separate dataset, which should be merged with d_obs

    SH_pressure_levels = []
    # in hPa
    pr_levels_all = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa
    # read in data (only for downloaded pressure levels)
    # find upper max index that corresponds to given pressure level
    pr_max_idx = pr_levels_all.index(SH_integral_pressure)

    for i in range(0,pr_max_idx + 2): # go one further than required, to make integration more accurate
        path =  '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/SH/'+ str(pr_levels_all[i]) + 'hPa' +'/*.nc'
        ds = xr.open_mfdataset(path)
        ds_sel = ds.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
        if 'expver' in ds_sel.coords:
            ds_sel = ds_sel.sel(expver=5)
        # # line below: commented for generating hourly data for diurnal cycle
        # ds_sel = ds_sel.resample(time = '1m', keep_attrs=True).mean()
        
        # assign pressure level
        ds_sel = ds_sel.assign_coords({"level": pr_levels_all[i]})

        SH_pressure_levels.append(ds_sel)
    
    ds_SH_pressure_levels = xr.concat(SH_pressure_levels, dim = 'level')

    # do Integral
    summe, ds_tcw_profile = Euler_centred_PWV(ds_SH_pressure_levels, pr_max_idx, 'q')

    # write integral to dataset, let's call the variable 'q_integrated',
    # 'q_integrated' must be passed as 'ERA5_variable' to make sure that it is plotted afterwards
    ds_q_integrated = xr.Dataset({"q_integrated": summe})

    return ds_q_integrated, ds_tcw_profile

    
def calc_d_model_ensemble(d_model, d_Ensemble):
    
    # empty list for merging all forcings together
    ensemble_mean_year = []
    ensemble_mean_month = []
    ensemble_mean_month_comparison = []
    ensemble_mean_taylor = []
    ensemble_mean_monthly_timeseries = []

    # pressure level variables

    # write 'clim_var' to be plotted with the ensemble into d_Ensemble 
    if 'clim_var' in d_Ensemble.keys():
        for clim_var in d_Ensemble['clim_var']:
            # go through forcings
            for forc_folder in d_Ensemble['folders']:
                variable = clim_var + ' ' + forc_folder
                # empty list for calculating mean
                av_ls_year = []
                av_ls_month = []
                av_ls_month_comparison = []
                av_ls_taylor = []
                av_ls_monthly_timeseries = []

                time_monthly_timeseries_hist, time_monthly_timeseries_future = PRIMAVERA_time_intersect(d_model)

                for clim_idx, clim_key in enumerate(d_model.keys()):
                    # only include in mean if the simulation of the forcing really exists in this dataset
                    if forc_folder in d_model[clim_key]['folders']:
                        # append only if not empty!
                        ds_check = d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year')
                        if ds_check.year.size!=0:
                            av_ls_year.append(ds_check)
                        else:
                            print('Empty array for {}, {}, year'.format(clim_key, variable))
                        if 'ds_mean_month' in d_model[clim_key].keys():
                            ds_check = d_model[clim_key]['ds_mean_month'][variable]
                            if ds_check.month.size!=0:
                                av_ls_month.append(ds_check)
                                av_ls_month_comparison.append(d_model[clim_key]['ds_mean_month_comparison'][variable])
                            else:
                                print('Empty array for {}, {}, month'.format(clim_key, variable))
                        
                        if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                            if forc_folder in d_model[clim_key]['taylor_folder']:
                                ds_check = d_model[clim_key]['ds_taylor'][variable].dropna(dim='time')
                                # print(ds_check)
                                if ds_check.time.size != 0:
                                    av_ls_taylor.append(ds_check)
                                    time = ds_check.time # assign time (same for all clim_keys) for new dataset
                                else:
                                    print('Empty array for {}, {}, taylor'.format(clim_key, variable))
                
                        # check ds_monthly_timeseries_intersect (deep copy of ds_sel, with changed month indexes)
                        if forc_folder == 'hist' or forc_folder == 'present':
                            # there are some datasets that have the year 2015 in the historical simulations! 
                            # therefore, select only the years up to 2014
                            ds_check = d_model[clim_key]['ds_monthly_timeseries_intersect'][variable].sel(time=slice('1950-01', '2014-12')).dropna(dim='time')
                            time_monthly_timeseries = time_monthly_timeseries_hist
                        else:
                            ds_check = d_model[clim_key]['ds_monthly_timeseries_intersect'][variable].dropna(dim='time')
                            time_monthly_timeseries = time_monthly_timeseries_future
                        
                        if ds_check.time.size!=0:
                            # select the intersecting months (calculated with 'PRIMAVERA_time_intersect')
                            ds_check_sel = ds_check.sel(time=time_monthly_timeseries)

                            # get time out of checked 
                            # time_monthly_timeseries = ds_check.time
                            av_ls_monthly_timeseries.append(ds_check_sel)
                            
                        

                # watch out in this section!!! clim_key is just the last clim_key in d_model. Now if for the last model, the taylor_folder is empty ([]), we have a problem here!
                # assign time and level (all the same)
                level = d_Ensemble['Plev']
                if forc_folder == 'hist' or forc_folder == 'present':
                    year = np.arange(1950,2015,1)
                elif forc_folder == 'future' or forc_folder == 'SSTfuture':
                    year = np.arange(2015,2051,1)
                else:
                    raise Exception('array of year cannot be defined.')

                month = np.arange(1,13,1) # [1,2,3,4,5,6,7,8,9,10,11,12]

                # calculate Ensemble mean
                np_mean_year = np.mean(av_ls_year, axis=0)
                if 'ds_mean_month' in d_model[clim_key].keys():
                    np_mean_month = np.mean(av_ls_month, axis=0)
                    np_mean_month_comparison = np.mean(av_ls_month_comparison, axis=0)
                
                if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                    if forc_folder in d_model[clim_key]['taylor_folder']:
                        np_mean_taylor =  np.mean(av_ls_taylor, axis=0)

                # 2020-11-02: error "could not broadcast input array from shape (780) into shape (1)" was weird because it still produced np_mean
                # reason was, that only the av_ls_monthly_timeseries[4] was corrupt
                # (EC-Earth not included here, length of list was 5, since forcing was 'present' and EC-Earth has no 'present')
                # problem was, that av_ls_monthly_timeseries[4] had 780 instead of 777 month values
                # solution: take 'time_monthly_timeseries' from first clim_key and select the exact same time for the following clim_keys
                np_mean_monthly_timeseries = np.mean(av_ls_monthly_timeseries, axis=0)
                np_std_monthly_timeseries = np.std(av_ls_monthly_timeseries, axis=0)
                    
                # create Dataset
                ds_forc_year = xr.Dataset({variable + ' mean':  (("year", "level"), np_mean_year) , variable + ' std': (("year", "level"), np.std(av_ls_year, axis=0) )}, coords={'level': level, 'year': year })
                if 'ds_mean_month' in d_model[clim_key].keys():
                    ds_forc_month = xr.Dataset({variable + ' mean':  (("month", "level"), np_mean_month), variable + ' std': (("month", "level"), np.std(av_ls_month, axis=0) )}, coords={'level': level, 'month': month })
                    ds_forc_month_comparison = xr.Dataset({variable + ' mean':  (("month", "level"), np_mean_month_comparison), variable + ' std': (("month", "level"), np.std(av_ls_month_comparison, axis=0) )}, coords={'level': level, 'month': month })
                if forc_folder in d_model[clim_key]['taylor_folder']: # bool not needed, since even if list is empty, we can still check
                    try:
                        ds_forc_taylor = xr.Dataset({variable + ' mean':  (("time", "level"), np_mean_taylor) , variable + ' std': (("time", "level"), np.std(av_ls_taylor, axis=0) )}, coords={'level': level, 'time': time}) # level
                    except ValueError: # change order of dimensions (I think I caused this error by dropping 'level' and adding it again in Model_SH_Integral.py)
                        # print('got Value Error. Change time and level and try again.')
                        # print('crashed at {}, {}; if it stopped here (mean taylor)'.format(clim_key, forc_folder))
                        ds_forc_taylor = xr.Dataset({variable + ' mean':  (("level", "time"), np_mean_taylor) , variable + ' std': (("level", "time"), np.std(av_ls_taylor, axis=0) )}, coords={'level': level, 'time': time})
                try:
                    ds_forc_monthly_timeseries = xr.Dataset({variable + ' mean': (("time", "level"), np_mean_monthly_timeseries), variable + ' std': (("time", "level"), np_std_monthly_timeseries)}, coords={'level': level, 'time': time_monthly_timeseries})
                except ValueError: # change order of dimensions
                    # print('crashed at {}, {}; if it stopped here (monthly timeseries)'.format(clim_key, forc_folder))
                    ds_forc_monthly_timeseries = xr.Dataset({variable + ' mean': (("level", "time"), np_mean_monthly_timeseries), variable + ' std': (("level", "time"), np_std_monthly_timeseries)}, coords={'level': level, 'time': time_monthly_timeseries})
                
                # append for merging    
                ensemble_mean_year.append(ds_forc_year)
                if 'ds_mean_month' in d_model[clim_key].keys():
                    ensemble_mean_month.append(ds_forc_month)
                    ensemble_mean_month_comparison.append(ds_forc_month_comparison)
                if forc_folder in d_model[clim_key]['taylor_folder']:
                    ensemble_mean_taylor.append(ds_forc_taylor)
                ensemble_mean_monthly_timeseries.append(ds_forc_monthly_timeseries)

    # single level variables (no 'level', otherwise same)
    if 'single_lev_var' in d_Ensemble.keys():
        for clim_var in d_Ensemble['single_lev_var']:
            # go through forcings
            for forc_folder in d_Ensemble['folders']:
                variable = clim_var + ' ' + forc_folder
                # empty list for calculating mean
                av_ls_year = []
                av_ls_month = []
                av_ls_month_comparison = []
                av_ls_taylor = [] # added on 14.10.2020
                av_ls_monthly_timeseries = [] # addedd on 20.10.2020           

                time_monthly_timeseries_hist, time_monthly_timeseries_future = PRIMAVERA_time_intersect(d_model)         

                for clim_key in d_model.keys():
                    # only include in mean if the simulation of the forcing really exists in this dataset
                    if forc_folder in d_model[clim_key]['folders']:
                        # append only if not empty
                        ds_check = d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year')
                        if ds_check.year.size!=0:
                            av_ls_year.append(ds_check)
                        else:
                            print('Empty array for {}, {}, year'.format(clim_key, variable))
                        if 'ds_mean_month' in d_model[clim_key].keys():
                            ds_check = d_model[clim_key]['ds_mean_month'][variable]
                            if ds_check.month.size!=0:
                                av_ls_month.append(ds_check)
                                av_ls_month_comparison.append(d_model[clim_key]['ds_mean_month_comparison'][variable])
                            else:
                                print('Empty array for {}, {}, month'.format(clim_key, variable))
                        
                        if forc_folder in d_model[clim_key]['taylor_folder']:
                            ds_check = d_model[clim_key]['ds_taylor'][variable].dropna(dim='time')
                            # print(ds_check)
                            if ds_check.time.size != 0:
                                av_ls_taylor.append(ds_check)
                                time = ds_check.time # assign time (same for all clim_keys) for new dataset
                            else:
                                print('Empty array for {}, {}, taylor'.format(clim_key, variable))

                        # check ds_monthly_timeseries_intersect for monthly timeseries (deep copy of ds_sel, but with changed month indexes)
                        if forc_folder == 'hist' or forc_folder == 'present':
                            # there are some datasets that have the year 2015 in the historical simulations! 
                            # therefore, select only the years up to 2014
                            ds_check = d_model[clim_key]['ds_monthly_timeseries_intersect'][variable].sel(time=slice('1950-01', '2014-12')).dropna(dim='time')
                            time_monthly_timeseries = time_monthly_timeseries_hist
                        else:
                            ds_check = d_model[clim_key]['ds_monthly_timeseries_intersect'][variable].dropna(dim='time')
                            time_monthly_timeseries = time_monthly_timeseries_future
                        if ds_check.time.size!=0:
                            ds_check_sel = ds_check.sel(time=time_monthly_timeseries)
                            av_ls_monthly_timeseries.append(ds_check_sel)


                # assign time (all the same)
                if forc_folder == 'hist' or forc_folder == 'present':
                    year = np.arange(1950,2015,1)
                elif forc_folder == 'future' or forc_folder == 'SSTfuture':
                    year = np.arange(2015,2051,1)
                else:
                    raise Exception('array of year cannot be defined.')

                month = np.arange(1,13,1) # [1,2,3,4,5,6,7,8,9,10,11,12]

                # calculate Ensemble mean
                np_mean_year = np.mean(av_ls_year, axis=0)
                if 'ds_mean_month' in d_model[clim_key].keys():
                    np_mean_month = np.mean(av_ls_month, axis=0)
                    np_mean_month_comparison = np.mean(av_ls_month_comparison, axis=0)
                if forc_folder in d_model[clim_key]['taylor_folder']:
                    np_mean_taylor =  np.mean(av_ls_taylor, axis=0)
                np_mean_monthly_timeseries = np.mean(av_ls_monthly_timeseries, axis=0)

                ds_forc_year = xr.Dataset({variable + ' mean':  (("year"), np_mean_year) , variable + ' std': (("year"), np.std(av_ls_year, axis=0) )}, coords={'year': year })
                if 'ds_mean_month' in d_model[clim_key].keys():
                    ds_forc_month = xr.Dataset({variable + ' mean':  (("month"), np_mean_month), variable + ' std': (("month"), np.std(av_ls_month, axis=0) )}, coords={'month': month })
                    ds_forc_month_comparison = xr.Dataset({variable + ' mean':  (("month"), np_mean_month_comparison), variable + ' std': (("month"), np.std(av_ls_month_comparison, axis=0) )}, coords={'month': month })
                if forc_folder in d_model[clim_key]['taylor_folder']:
                    ds_forc_taylor = xr.Dataset({variable + ' mean':  (("time"), np_mean_taylor) , variable + ' std': (("time"), np.std(av_ls_taylor, axis=0) )}, coords={'time': time })
                # monthly mean timeseries
                ds_forc_monthly_timeseries = xr.Dataset({variable + ' mean': (("time"), np_mean_monthly_timeseries), variable + ' std': (("time"), np.std(av_ls_monthly_timeseries, axis=0))}, coords={'time': time_monthly_timeseries})


                ensemble_mean_year.append(ds_forc_year)
                if 'ds_mean_month' in d_model[clim_key].keys():
                    ensemble_mean_month.append(ds_forc_month)   
                    ensemble_mean_month_comparison.append(ds_forc_month_comparison)  
                if forc_folder in d_model[clim_key]['taylor_folder']:
                    ensemble_mean_taylor.append(ds_forc_taylor)
                ensemble_mean_monthly_timeseries.append(ds_forc_monthly_timeseries)

    d_Ensemble['ds_ensemble_mean_year'] = xr.merge(ensemble_mean_year)
    if 'ds_mean_month' in d_model[clim_key].keys():
        d_Ensemble['ds_ensemble_mean_month'] = xr.merge(ensemble_mean_month) 
        d_Ensemble['ds_ensemble_mean_month_comparison'] = xr.merge(ensemble_mean_month_comparison)
    d_Ensemble['ds_taylor'] =  xr.merge(ensemble_mean_taylor)
    d_Ensemble['ds_ensemble_monthly_timeseries'] = xr.merge(ensemble_mean_monthly_timeseries, compat='override') # longitudes were not exactly the same!

    return

#%%

def plot_line(ds, var ,my_ax, time_freq, P_lev = None, alpha=0.85):
    if P_lev != None:
        hin = ds[var].sel(level = P_lev).plot.line(x=time_freq, markersize = 4, ax=my_ax, add_legend=False)
    else:
        hin = ds[var].plot.line(x=time_freq, markersize = 4, ax=my_ax)
    return hin # required for hin[0].get_color()


def xr_plot_cycles_timeseries(d_obs, site, variable, lon, lat, d_model = None, 
                                diurnal=False, fig_diurnal=None, d_Ensemble=None, MasterFig=None, ax_ref = None):
    # initialize timer
    # # start clock
    # start_time = time.time() # measure elapsed time

    # check time with: print("--- %s seconds ---" % (time.time() - start_time))
    
    # define string of site that has no underlines, but white spaces
    if site == 'MaunaKea':
        site_noUnderline = 'Mauna Kea'
    elif site == 'siding_spring':
        site_noUnderline = 'Siding Spring'
    else:
        site_noUnderline = site.replace('_', ' ')
    
    # initialize figure        
    # fig, (ax0, ax1) = plt.subplots(ncols=2, sharey = True, figsize= (10,4))
    if MasterFig == None:
        fig = plt.figure(figsize = (25, 4),constrained_layout=True) # (this is not compatible with tight_layout)
        idx = 0 # so that if-conditions for siding_spring can be evaluated

        if diurnal: # if diurnal == True
            gs = fig.add_gridspec(1, 5)
            # sorry for the confusing numbering. I wrote the code first so that the diurnal cycle comes at second position, but later I wanted to change it
            ax1 = fig.add_subplot(gs[0, 0]) # diurnal cycle
            ax0 = fig.add_subplot(gs[0, 1]) # seasonal cycle
            ax3 = fig.add_subplot(gs[0, 2:-1]) # span timeseries over two subplot lengths
            ax4 = fig.add_subplot(gs[0, 4]) # taylor diagram
        else:
            gs = fig.add_gridspec(1, 4)
            # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
            ax3 = fig.add_subplot(gs[0, 0:-2]) # span timeseries over two subplot lengths
            ax0 = fig.add_subplot(gs[0, 2], sharey=ax3) # seasonal cycle
            ax4 = fig.add_subplot(gs[0, 3])  # taylor diagram

    else: # take MasterFig as figure, only append axes to it
        print('MasterFig is getting unpacked!')
        # MasterFig is a tuple (fig, idx)
        fig, idx, max_idx, gs, Plev_list, model_color = MasterFig # unpack tuple


        if ax_ref != None:
            # take the reference axis
            ax_ref0, ax_ref3 = ax_ref
            # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
            ax3 = fig.add_subplot(gs[idx, 0:-2],sharex=ax_ref3) # span timeseries over two subplot lengths
            ax0 = fig.add_subplot(gs[idx, 2], sharey=ax3, sharex=ax_ref0) # seasonal cycle
            ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram

        else:
            # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
            ax3 = fig.add_subplot(gs[idx, 0:-2]) # span timeseries over two subplot lengths
            ax0 = fig.add_subplot(gs[idx, 2], sharey=ax3) # seasonal cycle
            ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram

        if diurnal:
            columns = 3
            rows = math.ceil((max_idx + 1)/3) # calculate number of rows dependent on max_idx, round up with ceil
            if ((max_idx + 1)%3) == 0: # add a row for the legend
                rows = rows + 1
                print('row is added. we now have {} rows'.format(rows))
            # add a new subplot (fig_diurnal may not be None!)
            ax1 = fig_diurnal.add_subplot(rows, columns, idx + 1)

    # print("--- %s seconds for initializing figure ---" % (time.time() - start_time))
    # start_time = time.time()

    # create empty lists for handlebars for the legend
    Patch_list = [] # colors of different pressure levels and single level variables
    line_list = [] # markers for different models (including ERA5 and in-situ data)
    Ensemble_insitu_list = [] # labels of in situ data only
    Ensemble_insitu_labels = []
    line_list_ensemble = [] # labels of other climate data (ERA5 mainly)
    forcing_line_list = [] # line styles for different forcings

    # for taylor diagram (only labels of predictions, not the reference)
    taylor_label = [] # 
    # marker dictionary
    marD = {} #(e.g. marD = {'d': ['b','k', 'y'], '+': ['g']}# 'd' for ERA5, '+' for HadGEM)
    # initialize marker collection
    # mar_collection = cycle(['*','p','h', 's','d','^','v'])

    # edgecol_collection = cycle(['w' ,'k']) # there are two different forcings, so take black and white

    # assign reference (insitu) and predictions (ERA5 and model data)
    # empty list for taylor diagram (reference and predictions)
    ref_pred_ls = []
    name_ref_pred = []

    
    ########### in situ

    if idx == 5 and variable == 'seeing_nc' and MasterFig != None and 'ds_siding_spring_yearly' in d_obs.keys():  
        # add insitu_var for siding spring here (so that climxa ignores in situ dataset until now)
        d_obs['insitu_var'] = ['Seeing Siding Spring']

    # check if there is in-situ data
    if 'insitu_var' in d_obs.keys():
        # set in-situ color cycler
        insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p'])  * cycler(markersize=[12]))
        ax0.set_prop_cycle(insitu_cycler)
        if diurnal:
            ax1.set_prop_cycle(insitu_cycler)
        ax3.set_prop_cycle(insitu_cycler)

        for i, insitu_param in enumerate(d_obs['insitu_var']):
            if idx == 5 and variable == 'seeing_nc' and MasterFig != None and 'ds_siding_spring_yearly' in d_obs.keys():
                # first, add ds_mean_year of siding_spring insitu data... you can store the dataset in d_obs somewhere!
                d_obs['ds_mean_year']['Seeing Siding Spring'] = d_obs['ds_siding_spring_yearly']['ds_mean_year']
                d_obs['ds_std_year']['Seeing Siding Spring'] = d_obs['ds_siding_spring_yearly']['ds_std_year']

                
                # timeseries
                hin = plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
                ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
                                alpha=.2, facecolor='k')
            else:
                #seasonal cycle
                # seasonal
                hin = plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
                ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
                                alpha=.25, facecolor='k')
                #diurnal
                if diurnal:
                    plot_line(d_obs['ds_mean_hour'], insitu_param , ax1, 'hour')
                    ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][insitu_param] - d_obs['ds_std_hour'][insitu_param]), (d_obs['ds_mean_hour'][insitu_param] + d_obs['ds_std_hour'][insitu_param]),
                                    alpha=.2, facecolor='k')
                # timeseries
                plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
                ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
                                alpha=.2, facecolor='k')
                # for taylor plot (very first entry of ref_pred_ls is reference!)
                # if there are more than one in-situ variables, take only the first in the list_of_insitu_vars
                if i == 0:
                    ref_pred_ls.append({"data" : d_obs['ds_taylor'][insitu_param]})
                    name_ref_pred.append('in-situ ' + insitu_param)

            # if no model data is present, compare in situ data against each other
            # elif 'ERA5_var' not in d_obs.keys() and 'single_lev' not in d_obs.keys() and d_model == None:
            #     ref_pred_ls.append({"data" : d_obs['ds_taylor'][insitu_param]})
            #     name_ref_pred.append('in-situ ' + insitu_param)
            #     if i == 1: # only do once
            #         marD['o'] = {}
            #         marD['o']['w'] = 'gray'
            #     if i == 2:
            #         marD['o']['w'] = 'black'

            
            # append to list for legend inside each subplot
            if MasterFig != None:
                if model_color == True or diurnal == True:
                    # add here specific labels such as 'Downtime' for clouds 
                    # downtime for Tololo, Mauna Kea and La Palma
                    if variable == 'total_cloud_cover':
                        if idx == 0 or idx == 3 or idx == 4:
                            my_insitu_label = 'in-situ downtime'
                        elif idx == 1 or idx == 2: # paranal and la silla
                            my_insitu_label = 'in-situ photometric nights'
                        elif idx == 5: # siding spring
                            my_insitu_label = 'in-situ time lost due to bad weather'
                    
                    # seeing
                    if variable == 'seeing_nc':
                        # MASS or DIMM?

                        # [0]: Mauna Kea: MASS (DIMM available)
                        # [1]: Cerro Paranal: DIMM (MASS available from 2016-2020)
                        # [2]: La Silla: DIMM
                        # [3]: Cerro Tololo: DIMM
                        # [4]: La Palma: RoboDIMM
                        # [5]: Siding Spring: yearly reports
                        # [6]: Sutherland: no data
                        # [7]: SPM: MASS (DIMM is available!)
                        if idx == 0 or idx == 7:
                            my_insitu_label = 'in-situ MASS seeing'
                        elif idx == 4:
                            my_insitu_label = 'in-situ RoboDIMM seeing'
                        else:
                            my_insitu_label = 'in-situ DIMM seeing'

                    else: # if it is clear what data we have
                        my_insitu_label= 'in-situ'

                    # do not append to 'line_list', because we already have it in the subplot's legend
                    Ensemble_insitu_list.append(Line2D([0], [0], color = 'k',marker = hin[0].get_marker(), label = my_insitu_label))  # + r' $\pm$ std. dev.'
                    Ensemble_insitu_labels.append(my_insitu_label)



            else:
                line_list.append(Line2D([0], [0], color = 'k',marker = hin[0].get_marker(), label = 'in-situ ' + insitu_param.replace('_', ' ')))        #  + r' $\pm$ std. dev.'
                Ensemble_insitu_list.append(Line2D([0], [0], color = 'k',marker = hin[0].get_marker(), label = 'in-situ ' + insitu_param.replace('_', ' ') ))  # + r' $\pm$ std. dev.'
                Ensemble_insitu_labels.append('in-situ ' + insitu_param.replace('_', ' '))
        print('in-situ plotting done.')

        # print("--- %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()

      
    if d_model != None and d_Ensemble != None:
        # check if we are in 'model_color' publication mode.
        # if so, it can be that we want to plot two datasets
        # this is the case for seeing, up to now
        # I hardcode this case here
        if MasterFig != None and model_color == True:
            if variable == 'seeing_nc' and 'single_lev_var' in d_Ensemble.keys():
                print('I detected a second publication dataset that needs special colors.')
                # then, 200hPa-wind-speed seeing should be in d_Ensemble
                # we update 'second_pub_dataset' to True
                # this new variable appears in return_model_color and d_color_generator_ind_model
                second_pub_dataset = True
                # below, when we call one of these functions, we pass second_pub_dataset = second_pub_dataset, because it is set now
    
            else: # in any other case, set to False
                second_pub_dataset = False

        else: # in any other case, set to False
            second_pub_dataset = False

    else: # in any other case, set to False
        second_pub_dataset = False


    ########## ERA 5

    # create entry in legend for linestyle of ERA5

    ###################
    if 'ERA5_var' in d_obs.keys(): 
        # taylor label for ERA5 
        # append label list only, if ERA5 should not be the reference,
        # meaning if there is insitu data
        # if 'insitu_var' in d_obs.keys():
        #     taylor_label.append('ERA5')

        # else, do not append list

        marD_ERA5 =  '*'
        marD[marD_ERA5] = {} # initialize subdict, for edgecolors
        # get color
        if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
            if d_model != None:
                col = return_model_color('ERA5') # returns a list with one element, but that's fine
                # we assume here that Plev data has normal model_color s and single level data has colors of second pub dataset!

        else: # else, get colors for individual pressure levels
            col = [P_lev_color(x) for x in d_obs['Plev']]
        # store colors in marker dict
        # only append marD if ERA5 pressure level data doesn't have to be reference
        if 'insitu_var' not in d_obs.keys() or (idx == 5 and variable == 'seeing_nc' and MasterFig != None): # then, ERA5 has to give a reference
            # check if list of colors has length greater than 1, 
            # then, append marD with colors except first pressure level (which serves now as a reference)
            # WAIT: why first pressure level and not CLOSEST pressure level? (okey, it works for tcw, because there is only one pressure level (until now))
            if len(col) > 1:
               marD[marD_ERA5]['w'] = col[1:]

        else:
            marD[marD_ERA5]['w'] = col # take white ('w') as edgecolor for ERA5

        # set cycler for PRESSURE level data
        Era5_cycler = (cycler(color = col) * cycler(linestyle=['-']) * cycler(marker = ['*']))

        ax0.set_prop_cycle(Era5_cycler)
        if diurnal: # if diurnal == True
            ax1.set_prop_cycle(Era5_cycler)
        ax3.set_prop_cycle(Era5_cycler)

        # loop through climate variables to plot
        
        for clim_var in d_obs['ERA5_var']:
            for P_lev in d_obs['Plev']:  
                # seasonal
                hin = plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month', P_lev)
                # std dev
                ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][clim_var].sel(level=P_lev) - d_obs['ds_std_month'][clim_var].sel(level=P_lev)), (d_obs['ds_mean_month'][clim_var].sel(level=P_lev) + d_obs['ds_std_month'][clim_var].sel(level=P_lev)),
                                alpha=.25)
                #diurnal
                if diurnal: # if diurnal == True
                    plot_line(d_obs['ds_mean_hour'], clim_var ,ax1, 'hour', P_lev)
                    
                    # if variable == 'TCW': # plot standard deviations if there is no in situ data
                    #     if (site != 'MaunaKea') and (site != 'Paranal') and (site != 'La_Silla'):

                    # plot std dev anyway!
                    ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][clim_var].sel(level=P_lev) - d_obs['ds_std_hour'][clim_var].sel(level=P_lev))
                                        , (d_obs['ds_mean_hour'][clim_var].sel(level=P_lev) + d_obs['ds_std_hour'][clim_var].sel(level=P_lev)),
                                        alpha=.25)
                            
                # timeseries
                plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year', P_lev)
                # plot std dev
                ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][clim_var].sel(level=P_lev) - d_obs['ds_std_year'][clim_var].sel(level=P_lev)), (d_obs['ds_mean_year'][clim_var].sel(level=P_lev) + d_obs['ds_std_year'][clim_var].sel(level=P_lev)),
                                alpha=.2)
                # print(hin[0].get_color())
                # taylor
                ref_pred_ls.append({"data" : d_obs['ds_taylor'][clim_var].sel(level = P_lev)})
                name_ref_pred.append('ERA5 ' + str(P_lev))

                Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = str(P_lev) + 'hPa'))
                # append to list to plot inside each subplot (append models only if model_color == True)
                if MasterFig != None:
                    if model_color == True:
                        if variable == 'seeing_nc':
                            # pressure level
                            my_insitu_label = 'ERA5 seeing model'
                        # for model data
                        # if pressure level data (= obsorn seeing)
                        # elif single level data (= 200hPa-wind-speed seeing)
                        else:
                            my_insitu_label = 'ERA5 ' +  str(P_lev) + 'hPa'

                        Ensemble_insitu_list.append(Line2D([0], [0], color = hin[0].get_color() ,marker = hin[0].get_marker(), label = my_insitu_label))  
                        Ensemble_insitu_labels.append(my_insitu_label)
                    # print("--- %s seconds for one pressure level" % (time.time() - start_time))
                    # start_time = time.time()


    if 'single_lev' in d_obs.keys() :
        # if there are no Pressure level data, you have to assign the ERA5 marker
        if 'Plev' not in d_obs.keys() or d_obs['Plev'] == None:
            marD_ERA5 =  '*'
            if 'insitu_var' in d_obs.keys(): # then we already have a reference (else, we do not need marD_ERA5 at all)
                marD[marD_ERA5] = {}

        # # append taylor label only if 'ERA5' not already in list
        # if 'ERA5' not in taylor_label and 'insitu_var' in d_obs.keys():
        #     taylor_label.append('ERA5')

        # get color
        if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
            if d_model != None:
                if second_pub_dataset == True:
                    # pass keyword second_pub_dataset = True
                    col = return_model_color('ERA5', second_pub_dataset = True) # returns a list with one element, but that's fine
                else: # normal single level procedure for model_color (maroon)
                    col = return_model_color('ERA5') # returns a list with one element, but that's fine

        else:
            # set cycler for SINGLE level data
            col = [P_lev_color(x) for x in d_obs['single_lev']] # ['m','lightcoral', 'navy'] # probably not more than 3 different single level variables
        
        # special case of siding_spring, where in situ data cannot serve for taylor diagram (only yearly data!)
        if MasterFig != None and idx == 5 and variable == 'seeing_nc':
            # do not add to marD if there is also no Plev ERA5 data
            if 'Plev' not in d_obs.keys() or d_obs['Plev'] == None:
                print('ERA5 single level data is now the reference.')

            # if there is pressure level data, we still need to add single level data to marD for Taylor diagram
            else: 
                marD_ERA5 =  '*'
                marD[marD_ERA5] = {}
                marD[marD_ERA5]['w'] = col[:len(d_obs['single_lev'])]

        # append ERA5 col
        elif 'Plev' not in d_obs.keys() or d_obs['Plev'] == None:
                if 'insitu_var' not in d_obs.keys():
                    # now, ERA5 single level has to be the reference for the taylor diagram
                    print('ERA5 single level data is now the reference')
                else:       
                    marD[marD_ERA5]['w'] = col[:len(d_obs['single_lev'])] # only put in as many single level variables as we have
        

        # if ERA5 must be reference for taylor diagram (insitu_var not in d_obs.keys), 
        # then it might also be that a pressure level has already been used as a ref, 
        # and marD['*']['w'] doesnt exist anymore
        elif 'insitu_var' not in d_obs.keys() and len(d_obs['Plev']) == 1:
            marD[marD_ERA5]['w'] = col[:len(d_obs['single_lev'])]
        else: # if there are pressure levels, then we can simply add the list with +=
            marD[marD_ERA5]['w'] += col[:len(d_obs['single_lev'])] # only put in as many single level variables as we have


        Era5_SG_cycler = (cycler(color = col) * cycler(linestyle=['-'])) * plt.cycler(marker='*') # cycler(linestyle=['-']))
        ax0.set_prop_cycle(Era5_SG_cycler)
        if diurnal: # if diurnal == True
            ax1.set_prop_cycle(Era5_SG_cycler)
        ax3.set_prop_cycle(Era5_SG_cycler)

        for clim_var in d_obs['single_lev']:
            # seasonal
            hin = plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month')
            ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][clim_var] - d_obs['ds_std_month'][clim_var]), (d_obs['ds_mean_month'][clim_var] + d_obs['ds_std_month'][clim_var]),
                                alpha=.25)
            #diurnal
            if diurnal: # if diurnal == True
                plot_line(d_obs['ds_mean_hour'], clim_var ,ax1, 'hour')
                ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][clim_var] - d_obs['ds_std_hour'][clim_var])
                                        , (d_obs['ds_mean_hour'][clim_var] + d_obs['ds_std_hour'][clim_var]),
                                        alpha=.25)

            # timeseries
            plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year')
            # plot std dev
            ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][clim_var] - d_obs['ds_std_year'][clim_var]), (d_obs['ds_mean_year'][clim_var] + d_obs['ds_std_year'][clim_var]),
                                alpha=.2)

            # taylor
            ref_pred_ls.append({"data" : d_obs['ds_taylor'][clim_var]})
            name_ref_pred.append('ERA5 ' + clim_var)
            
            Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = clim_var + ' single level'))

            # append to list to plot inside each subplot (append models only if model_color == True)
            if MasterFig != None:
                if model_color == True:
                    if variable == 'seeing_nc':
                        # pressure level
                        my_insitu_label = 'ERA5 200hPa-wind-speed seeing'
                    # for model data
                    # if pressure level data (= obsorn seeing)
                    # elif single level data (= 200hPa-wind-speed seeing)
                    else:
                        my_insitu_label = 'ERA5 ' +  clim_var + ' single level'

                    Ensemble_insitu_list.append(Line2D([0], [0], color = hin[0].get_color() ,marker = hin[0].get_marker(), label = my_insitu_label))  
                    Ensemble_insitu_labels.append(my_insitu_label)

    # ERA 5 label
    if MasterFig != None:
        if model_color == False:
            line_list.append(Line2D([0], [0], linestyle = '-', marker = '*', markersize = 12, color = 'k', label = 'ERA5'))
            line_list_ensemble.append(Line2D([0], [0], linestyle = '-', marker = '*', markersize = 12, color = 'k', label = 'ERA5'))

    
    print('ERA5 plotting done.')


    ########## climate models


    # initialize linestyles for at least 6 different climate models
    # lin_st = ['dashed', (0, (1, 1)), (0,(5,2,5,5,1,4)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0,(5, 2, 20, 2)) ] # (0, (3, 5, 1, 5, 1, 5)) dashed, dotted, densely dotted, dashdotted, densely dashdotdotted, dashdotdotted
    lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
    
    #--> instead of code below, use same marker for same model! (use climxa.clim_key_marker(clim_key))
    # mar_collection_clim_models_only = ['p','h', 's','d','^','v']

    if d_model != None:
        # calculate Ensemble values, if d_Ensemble exists
        if d_Ensemble != None:
            calc_d_model_ensemble(d_model, d_Ensemble)
            # I have to append in the order of marD!
            # therefore, I initialize lists for Ensemble datasets 
            # that can be appended after the first iteration
            ref_pred_ls_Ensemble = []
            name_ref_pred_Ensemble = []


        # use different markers (same as for taylor diagram) for models
        for i, clim_key in enumerate(d_model.keys()):

            # scan here for empty arrays,
            # loop through pressure levels
            # delete pressure levels for totally (all forcings) empty arrays, 
            # so that they do not lead to conflicts later
            # do this only if 'ds_taylor' is available
            if 'Plev' in d_model[clim_key].keys() and 'ds_taylor' in d_model[clim_key].keys():
                for PressureLevel in d_model[clim_key]['Plev']:
                    # this is a bit ugly, since it is inside the forcing-loop
                    # but I need the new marD .. why? --> marD is based on pressure levels

                    # set counter to zero for one pressure level
                    counter_empty_forced_arrays = 0
                    # only loop through forcings that should be used for taylor_folder
                    for forc_folder in d_model[clim_key]['taylor_folder']:
                        clim_var_loop = d_model[clim_key]['clim_var'][0] + ' ' + forc_folder # [0] because it is a list (with one entry)
                        # select the variable 'clim_var_loop' drop all nan's and write to dataset
                        ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)
                        # print('this is where I check the dataset ds_taylor (line 1809). Now I have {}.'.format(forc_folder))

                        if ds_check['time'].size == 0:
                            print('I found an empty array (line 1812): model = {}, plev = {}'.format(clim_key, PressureLevel))
                            # if array is empty after dropping all nan's, delete the pressure level entry in ['Plev'] (delete the color entry in marD for this pressure level)
                            # delete only, if all forcings for this pressure level are empty (check with counter)
                            # add to counter for the current pressure level
                            counter_empty_forced_arrays = counter_empty_forced_arrays + 1
                            print(counter_empty_forced_arrays)
                            
                            if counter_empty_forced_arrays == len(d_model[clim_key]['folders']):
                                # commented on July 20 (only need to delete a pressure level, nothing else (pressure level defines color entry)) 
                                # d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)
                                
                                if len(d_model[clim_key]['ds_taylor'][clim_var_loop].level) > 1:
                                    # if the empty dataset has in reality more than one pressure levels left, then only delete the selection
                                    # remove pressure level from d_model[clim_key]['Plev']
                                    print('I have {} Pressure Levels'.format(len(d_model[clim_key]['ds_taylor'][clim_var_loop].level)))
                                    if PressureLevel in d_model[clim_key]['Plev']:
                                        print('I remove {} from {}.'.format(PressureLevel, clim_key))
                                        d_model[clim_key]['Plev'].remove(PressureLevel)
                                    # d_model[clim_key]['ds_seds_taylorl_cycle'][clim_var_loop] = d_model[clim_key]['ds_taylor'][clim_var_loop].drop_sel(level = PressureLevel)
                                # if len(d_model[clim_key]['ds_taylor'][clim_var_loop].level) == 1::
                                #     # also drop the variable
                                #     d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_taylor'].drop_vars(clim_var_loop)


            line_list.append(Line2D([0], [0], linestyle = '', color = 'k', marker = clim_key_marker(clim_key), label = d_model[clim_key]['name']))

            # 1 marker for one climate model
            if 'ds_taylor' in d_model[clim_key].keys(): # only create entry in marD if ds_taylor available
                marD_clim = clim_key_marker(clim_key) # next(mar_collection)
                marD[marD_clim]= {} # initialize subdict

            # reinitialize edgecolor cycle
            edgecol_collection = cycle(['w' ,'k']) # there are two different forcings, so I take black and white to distinguish them

            # loop through different forcings
            for k, forcing in enumerate(d_model[clim_key]['folders']):
                if 'clim_var' in d_model[clim_key].keys():
                    # clim_var was renamed in get_PRIMAVERA
                    clim_var = d_model[clim_key]['clim_var'][0] + ' ' + forcing

                # specify line style (different for all forcings)
                if forcing=='hist':
                    forced_linestyle = lin_st[0]
                elif forcing=='present':
                    forced_linestyle = lin_st[1]
                elif forcing=='future':
                    forced_linestyle = lin_st[2]
                elif forcing=='SSTfuture': 
                    forced_linestyle = lin_st[3]

                if MasterFig != None:
                    if model_color == True and (forcing == 'hist' or forcing == 'present'): # define '-' as the linestyle for all forcings!
                        forced_linestyle = '-'
                        # but for future, we must keep the plotted linestyles (at least for the seasonal cycle!)


                # we need to define edgecolor for forcing, but only for forcings that go into taylor diagram
                if forcing in d_model[clim_key]['taylor_folder']:
                    if 'ds_taylor' in d_model[clim_key].keys():
                        marD_forcing = next(edgecol_collection) # not guaranteed that the same forcing has the same edgecolor 
                        # works only if we have the same taylor_folders (same order, but second or both forcings can be missing)
                        # (edgecol_collection is reinitialized for every new model)

                # create list of forcing labels, so that we can check if the label is already in the linestyle list
                list_of_forcing_labels = []
                for element in forcing_line_list:
                    # append to list of labels
                    list_of_forcing_labels.append(element.get_label())
                
                if forcing=='present':
                    my_label = 'SST present'
                elif forcing == 'SSTfuture':
                    my_label = 'SST future'
                else:
                    my_label = forcing

                if MasterFig != None:
                    if my_label not in list_of_forcing_labels: # if the current forcing is not already in the label list for the legend, do so now
                        if model_color == True and (forcing == 'future' or forcing == 'SSTfuture'):
                            # for publication, legend is needed only for future and SSTfuture linestyles in seasonal cycle (others are clear)
                            forcing_line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = return_model_color(forcing, second_pub_dataset = False)[0], label = my_label))

                            if second_pub_dataset == True:
                                forcing_line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = return_model_color(forcing, second_pub_dataset = second_pub_dataset)[0], label = my_label))

                        # elif model_color == True and (forcing == 'hist' or forcing == 'present'):
                        #     print('-')
                    elif model_color == False:
                        forcing_line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = 'k', label = my_label))
                
                # append taylor label only if forcing is not already in list
                # and only for forcings that are going into taylor diagram
                for folder in d_model[clim_key]['taylor_folder']:
                    # watch out here! I always use the coupled first in my 'taylor_folder'
                    if 'coupled' not in taylor_label:
                        if folder == 'hist' or folder == 'future':
                            folder_label= 'coupled'
                            taylor_label.append(folder_label)
                    elif 'atmosphere-land' not in taylor_label:
                        if folder == 'present' or folder == 'SSTfuture':
                            folder_label = 'atmosphere-land'
                            taylor_label.append(folder_label)

                # pressure level data
                if 'Plev' in d_model[clim_key].keys():

                    # create cycler for all clim_vars (from different climate models!): P_lev_color * linestyle

                    # get color
                    if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
                        # we need either coupled ('hist' and 'future') or atmosphere-only ('present', 'SSTfuture')
                        col_mod = return_model_color(forcing, second_pub_dataset = False) # returns a list with one element, but that's fine
                        # ATTENTION: we assume here that if second_pub_dataset is True, we want to plot the sg_lev var in special colors
                        # therefore, we set second_pub_dataset to False for Plevs
                    else: # else, get colors for individual pressure levels
                        col_mod = [P_lev_color(x) for x in d_model[clim_key]['Plev']]

                    col_mod_Ensemble = copy.deepcopy(col_mod)
                    # clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[lin_st[i]])) # we need a list here

                    if d_Ensemble == None:
                        clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[forced_linestyle]) * plt.cycler(marker=[clim_key_marker(clim_key)])) # we need a list here
                    else: # no marker at all for plotting Ensemble mean
                        clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[forced_linestyle])) # we need a list here

                    if forcing in d_model[clim_key]['taylor_folder']:
                        if 'ds_taylor' in d_model[clim_key].keys():
                            # store colors in marker dict
                            marD[marD_clim][marD_forcing] = col_mod
                        if d_Ensemble != None and i == 0: # only once
                            if 'o' not in marD.keys(): # otherwise, I overwrite it in the for loop!!
                                marD['o'] = {} # initialize Ensemble entry in marD
                            # and fill dict with red (should be a big red dot)
                            marD['o'][marD_forcing] = col_mod_Ensemble

                    ax0.set_prop_cycle(clim_model_cycler)

                    # # scan here for empty arrays
                    # if k == 0: # do this only once (this is really ugly now...other way would be to also do marD out of all loops)
                    #     for idx3, PressureLevel in enumerate(d_model[clim_key]['Plev']):
                    #         # this is a bit ugly, since it is inside the forcing-loop
                    #         # but I need the new marD .. why?

                    #         # set counter to zero for one pressure level
                    #         counter_empty_forced_arrays = 0
                    #         for forc_folder in d_model[clim_key]['taylor_folder']:
                    #             clim_var_loop = d_model[clim_key]['clim_var'][0] + ' ' + forc_folder

                    #             ds_check = d_model[clim_key]['ds_sel_cycle'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_sel_cycle'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)
                    #             # print(ds_check['time'].size)
                    #             if ds_check['time'].size != 0:
                    #                 # print('check done, array is not empty')
                    #                 # only append pressure level dataarray if array is not empty
                    #                 if forcing in d_model[clim_key]['taylor_folder']:
                    #                     # only append to list that is fed into taylor diagram (data) if forcing is in taylor_folder
                    #                     # otherwise, we do not want this data compared (and it is not in ds_taylor anyway)
                    #                     # print('I append for model {} and forcing {} and pressure level {}'.format(clim_key, forc_folder, PressureLevel)) 
                    #                     ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level = PressureLevel)})
                    #                     name_ref_pred.append(clim_key + ' ' + str(PressureLevel) + ' ' + clim_var_loop)
                    #             else:
                    #                 print('I found an empty array: model = {}, plev = {}'.format(clim_key, PressureLevel))
                    #                 # if array is empty after dropping all nan's, delete the color entry in marD for this pressure level
                    #                 # delete color only, if all forcings for this pressure level are empty
                    #                 # add to counter for the current pressure level
                    #                 counter_empty_forced_arrays = counter_empty_forced_arrays + 1
                    #                 print(counter_empty_forced_arrays)
                    #                 if counter_empty_forced_arrays == len(d_model[clim_key]['folders']): 
                    #                     # remove color from that specific pressure level from marD
                    #                     # if it is not already removed!!
                    #                     if P_lev_color(PressureLevel) in marD[marD_clim][marD_forcing]:
                    #                         marD[marD_clim][marD_forcing].remove(P_lev_color(PressureLevel))
                    #                 #   marD[marD_clim][marD_forcing].remove(marD[marD_clim][marD_forcing][idx2])


                    if diurnal: # if diurnal == True
                        ax1.set_prop_cycle(clim_model_cycler)
                    ax3.set_prop_cycle(clim_model_cycler)

                    # # append taylor label list for every model, use model name from dict d_model
                    # taylor_label.append(d_model[clim_key]['name'])
                            
                    # print("--- %s seconds before starting 1st for loop of climate model ---" % (time.time() - start_time))
                    # start_time = time.time()

                    # no loop needed below, since it is only one climate variable (always!)
                    # # loop through different climate variables
                    # for clim_var in d_model[clim_key]['clim_var']:

                    # # label for linestyle
                    # line_list.append(Line2D([0], [0], linestyle = lin_st[i], color = 'k', label = d_model[clim_key]['name']))

                    for P_lev in d_model[clim_key]['Plev']: 
                        if d_Ensemble == None: # plot each model individually
                            # seasonal
                            if 'ds_mean_month' in d_model[clim_key].keys():
                                hin = plot_line(d_model[clim_key]['ds_mean_month'], clim_var ,ax0, 'month', P_lev)
                            #timeseries
                            hin = plot_line(d_model[clim_key]['ds_mean_year'], clim_var ,ax3, 'year', P_lev)

                        else: # plot Ensemble values
                            if i == 0: # plot Ensemble (only for 1 clim_key (we need only d_Ensemble))
                                # seasonal
                                hin = plot_line(d_Ensemble['ds_ensemble_mean_month'], clim_var + ' mean' ,ax0, 'month', P_lev)
                                    # std deviation seasonal cycle
                                ax0.fill_between(d_Ensemble['ds_ensemble_mean_month'].month, (d_Ensemble['ds_ensemble_mean_month'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_month'][clim_var + ' std'].sel(level=P_lev)),  
                                    (d_Ensemble['ds_ensemble_mean_month'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_month'][clim_var + ' std'].sel(level=P_lev)),
                                    alpha=.2)

                                # seasonal, other than in taylor_folder, for comparison
                                if forcing not in d_model[clim_key]['taylor_folder']:
                                    # problem: cycler goes on with colors!
                                    # --> solution: specify color separately
    
                                    # get color
                                    if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
                                        # we need either coupled ('hist' and 'future') or atmosphere-only ('present', 'SSTfuture')
                                        color = return_model_color(forcing, second_pub_dataset = False)[0] # unpack list!
                                        # ATTENTION: we assume here that if second_pub_dataset is True, we want to plot the sg_lev var in special colors
                                        # therefore, we set second_pub_dataset to False for Plevs

                                    else: # else, get colors for individual pressure levels
                                        color = P_lev_color(P_lev)

                                    
                                    d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level = P_lev).plot.line(x='month', ax=ax0, add_legend=False, color=color)
                                        # std deviation
                                    ax0.fill_between(d_Ensemble['ds_ensemble_mean_month_comparison'].month, (d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' std'].sel(level=P_lev)),  
                                    (d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' std'].sel(level=P_lev)),
                                    alpha=.2, color=color)

                                else: # if forcing in taylor folder, add to taylor diagram
                                    # add to taylor diagram
                                    ref_pred_ls_Ensemble.append({"data": d_Ensemble['ds_taylor'][clim_var  + ' mean'].sel(level = P_lev)})
                                    name_ref_pred_Ensemble.append('d_Ensemble' + ' ' + str(P_lev) + ' ' + clim_var)
                                                           
                                # timeseries
                                plot_line(d_Ensemble['ds_ensemble_mean_year'], clim_var + ' mean' ,ax3, 'year', P_lev)
                                    # std deviation timeseries
                                ax3.fill_between(d_Ensemble['ds_ensemble_mean_year'].year, (d_Ensemble['ds_ensemble_mean_year'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_year'][clim_var + ' std'].sel(level=P_lev)),  
                                    (d_Ensemble['ds_ensemble_mean_year'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_year'][clim_var + ' std'].sel(level=P_lev)),
                                    alpha=.2)
                        #diurnal
                        if diurnal: # if diurnal == True
                            plot_line(d_model[clim_key]['ds_mean_hour'], clim_var ,ax1, 'hour', P_lev)

                        # taylor
                        # check here if array for this pressure level is not empty
                        # NO --> do this out of this loop

                        # create list of patch labels to check if we need to append the legend or not
                        list_of_patch_labels = []
                        for element in Patch_list:
                            # append to list of labels
                            list_of_patch_labels.append(element.get_label())

                        if str(P_lev) + 'hPa' not in list_of_patch_labels: # if the current Pressure level (e.g. 700hPa) is not already drawn to the legend, do so now
                            print('I append Patch_list')
                            print('because {} is not in {}'.format(str(P_lev), list_of_patch_labels))
                            Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = str(P_lev) + 'hPa'))
                        
                        # append to list to plot inside each subplot (append models only if model_color == True)
                        # for PRIMAVERA, we also need to constrain the forcing
                        # because we need only the pressure level, we can try to combine the two 
                        # if model_color == True:
                        #     Ensemble_insitu_list.append(Line2D([0], [0], color = hin[0].get_color() ,marker = hin[0].get_marker(), label = 'PRIMAVERA ' + str(P_lev) + 'hPa'))  
                        # here, just save pressure level! (or later, store single level label)
                        if variable == 'seeing_nc':
                            # single level
                            model_color_PRIMAVERA_label = 'PRIMAVERA seeing model'
                        # for model data
                        # if pressure level data (= obsorn seeing)
                        # elif single level data (= 200hPa-wind-speed seeing)
                        else:
                            model_color_PRIMAVERA_label = 'PRIMAVERA ' +  str(P_lev) + 'hPa'

                            
                        
                        # check again for empty arrays, 
                        # only append ref_pred_ls, if not empty
                        if 'ds_taylor' in d_model[clim_key].keys() and forcing in d_model[clim_key]['taylor_folder']:
                            ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var].sel(level= P_lev)) != True, drop = True)

                            if ds_check['time'].size != 0:
                                ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][clim_var].sel(level = P_lev)})
                                name_ref_pred.append(clim_key + ' ' + str(P_lev) + ' ' + clim_var)

                            # if array is empty   
                            elif ds_check['time'].size == 0:   
                                print('empty array {}, {}'.format(clim_key, P_lev))
                                # remove color from that specific pressure level from marD
                                # if it is not already removed!!
                                if P_lev_color(P_lev) in marD[marD_clim][marD_forcing]:
                                    print('I remove from dict: {} for {}, {}'.format(P_lev, marD_clim, marD_forcing))
                                    marD[marD_clim][marD_forcing].remove(P_lev_color(P_lev))
                                # if model_color == True, remove other
                                if return_model_color(forcing, second_pub_dataset = False)[0] in marD[marD_clim][marD_forcing]:
                                    print('I remove from dict: {} for {}, {}'.format(return_model_color(forcing, second_pub_dataset = False)[0], marD_clim, marD_forcing))
                                    marD[marD_clim][marD_forcing].remove(return_model_color(forcing, second_pub_dataset = False)[0])
                                    # ATTENTION: we assume here that if second_pub_dataset is True, we want to plot the sg_lev var in special colors
                                    # therefore, we set second_pub_dataset to False for Plevs

                        # print("--- %s seconds for one pressure level of one climate model ---" % (time.time() - start_time))
                        # start_time = time.time()

                        
                
                # single level data
                if 'single_lev_var' in d_model[clim_key].keys():
                    # we assume that there is only one single level var (might have to be changed in later versions!)
                    sg_clim_var = d_model[clim_key]['single_lev_var'][0] + ' ' + forcing

                    # set cycler for SINGLE level data

                    # get color
                    if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
                        # we need either coupled ('hist' and 'future') or atmosphere-only ('present', 'SSTfuture')
                        col = return_model_color(forcing, second_pub_dataset = second_pub_dataset) # col is a list

                    else: # else, get colors for individual pressure levels
                        # take same colors as for ERA5 single level variables
                        col = [P_lev_color(x) for x in d_model[clim_key]['single_lev_var']] # ['m','lightcoral', 'navy']
                        # take same colors as for ERA5 single level variables
                    
                    # clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[lin_st[i]])) # we need a list here
                    if d_Ensemble == None:
                        clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[forced_linestyle]) * plt.cycler(marker=[clim_key_marker(clim_key)])) # we need a list here
                    else: # no marker
                        clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[forced_linestyle])) # we need a list here

                    ax0.set_prop_cycle(clim_model_cycler)
                    if diurnal: # if diurnal == True
                        ax1.set_prop_cycle(clim_model_cycler)
                    ax3.set_prop_cycle(clim_model_cycler)

                    # # if there are no Pressure level data, you have to assign the climate model marker
                    # if 'Plev' not in d_model[clim_key].keys():                          
                        # marD_clim = next(mar_collection)

                    # append marker col
                    if 'ds_taylor' in d_model[clim_key].keys():
                        if 'Plev' not in d_model[clim_key].keys(): # if there is no pressure level data
                            if forcing in d_model[clim_key]['taylor_folder']:
                                # store colors in marker dict
                                marD[marD_clim][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
                        
                            #marD[marD_clim] = col[:len(d_model[clim_key]['single_lev_var'])] # only put in as many single level variables as we have
                        else: # if there are pressure levels, then we can simply add the list with +=
                            if forcing in d_model[clim_key]['taylor_folder']:
                                marD[marD_clim][marD_forcing] += col[:len(d_model[clim_key]['single_lev_var'])]
                            # marD[marD_clim] += col[:len(d_model[clim_key]['single_lev_var'])] # only put in as many single level variables as we have

                        # WHY IS THIS NOT NEEDED??????? I DON'T UNDERSTAND --> it is needed; I've had problems with deep and shallow copy of list
                        if d_Ensemble != None:
                            if forcing in d_model[clim_key]['taylor_folder'] and i == 0:
                                if 'o' not in marD.keys(): # otherwise, I overwrite it in the for loop!!
                                    marD['o'] = {} # initialize Ensemble entry in marD
                                # and fill dict with color
                                if bool(marD['o']) == False: # if dict is empty
                                    marD['o'][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
                                elif 'clim_var' not in d_Ensemble.keys(): # if there is only the single level var
                                    marD['o'][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
                                else: # append!
                                    marD['o'][marD_forcing] += col[:len(d_model[clim_key]['single_lev_var'])]

                    # append taylor label only if model name is not already in list
                    # if  d_model[clim_key]['name'] not in taylor_label:
                    #     taylor_label.append(d_model[clim_key]['name'])
                            
                # # plot single level model data
                # for sg_clim_var in d_model[clim_key]['single_lev_var']:
                    
                    # # add to model legend only if there is no pressure level data
                    # if 'Plev' not in d_model[clim_key].keys(): 
                    #     line_list.append(Line2D([0], [0], linestyle = lin_st[i], color = 'k', label = d_model[clim_key]['name']))
                    if d_Ensemble == None: # plot each model individually
                        # seasonal
                        if 'ds_mean_month' in d_model[clim_key].keys():
                            hin = plot_line(d_model[clim_key]['ds_mean_month'], sg_clim_var ,ax0, 'month')

                        #timeseries
                        hin = plot_line(d_model[clim_key]['ds_mean_year'], sg_clim_var ,ax3, 'year')
                    
                    else: # plot Ensemble, d_Ensemble is not None
                        if i == 0: # (only for 1 clim_key (we need only d_Ensemble))
                            # seasonal
                            hin = plot_line(d_Ensemble['ds_ensemble_mean_month'], sg_clim_var + ' mean' ,ax0, 'month')
                            ax0.fill_between(d_Ensemble['ds_ensemble_mean_month'].month, (d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' std']),  
                                    (d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' std']),
                                    alpha=.2)                            

                            # seasonal, other than in taylor_folder, for comparison
                            # if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                            if forcing not in d_model[clim_key]['taylor_folder']:
                                # problem: cycler goes on with colors!
                                # --> solution: specify color separately
                                color = col[0] # assume there is only one single level data!
                                d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'].plot.line(x='month', ax=ax0, add_legend=False, color=color)
                                    # std deviation
                                ax0.fill_between(d_Ensemble['ds_ensemble_mean_month_comparison'].month, (d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' std']),  
                                (d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' std']),
                                alpha=.2, color=color)
                            
                            else: # if forcing in taylor folder, add to taylor diagram
                                # add to taylor diagram
                                if bool(d_model[clim_key]['taylor_folder']) == True:
                                    print('data to ref_pred_ls_Ensemble for {}, {}'.format(forcing, sg_clim_var))
                                    ref_pred_ls_Ensemble.append({"data": d_Ensemble['ds_taylor'][sg_clim_var  + ' mean']})
                                    name_ref_pred_Ensemble.append('d_Ensemble' + ' ' + sg_clim_var)
                                                    

                            # timeseries
                            plot_line(d_Ensemble['ds_ensemble_mean_year'], sg_clim_var + ' mean' ,ax3, 'year')
                            ax3.fill_between(d_Ensemble['ds_ensemble_mean_year'].year, (d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' std']),  
                                    (d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' std']),
                                    alpha=.2)

                    #diurnal
                    if diurnal: # if diurnal == True
                        plot_line(d_model[clim_key]['ds_mean_hour'], sg_clim_var ,ax1, 'hour')
                    
                    # taylor
                    # check here if array for this pressure level is not empty
                    # because in 'time_intersect', I only selected one pressure level (not true anymore)
                    if forcing in d_model[clim_key]['taylor_folder']: # check only for taylor folders (others are nan by construction)
                        if 'ds_taylor' in d_model[clim_key].keys():
                            ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][sg_clim_var]) != True, drop = True)
                            # print(ds_check['time'].size)
                            # print('this is where I check the dataset ds_taylor (line 2201). Now I have {}.'.format(sg_clim_var))
                            if ds_check['time'].size != 0:
                                # only append pressure level dataarray if array is not empty
                                # print('check done, array is not empty')
                                if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                                    if forcing in d_model[clim_key]['taylor_folder']:
                                        # only append if forcing is in taylor_folder
                                        ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][sg_clim_var]})
                                        name_ref_pred.append(clim_key + ' ' + sg_clim_var)
                            else:
                                print('I found an empty array: model = {}'.format(clim_key))
                                # # if array is empty after dropping all nan's, delete that pressure level
                                # marD[marD_clim].remove(marD[marD_clim][idx2]) ?? think about this case (but I do not think it should ever happen for single level data)
                            
                    # create list of patch labels to check if we need to append the legend or not
                    list_of_patch_labels = []
                    for element in Patch_list:
                        # append to list of labels
                        list_of_patch_labels.append(element.get_label())

                    if d_model[clim_key]['single_lev_var'][0] + ' single level' not in list_of_patch_labels and d_model[clim_key]['single_lev_var'][0] + ' single level (PRIMAVERA)' not in list_of_patch_labels: # if the current Pressure level (e.g. 700hPa) is not already drawn to the legend, do so now
                        print('I append Patch_list')                   
                        Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = d_model[clim_key]['single_lev_var'][0] + ' single level (PRIMAVERA)'))           
                    

                    # here, just store single level label for the sublegend (for model_color == True only, but doesn't matter really)
                    # only take first three characters!!
                    if variable == 'seeing_nc':
                        # single level
                        model_color_PRIMAVERA_label_SG = 'PRIMAVERA 200hPa-wind-speed seeing'
                    # for model data
                    # if pressure level data (= obsorn seeing)
                    # elif single level data (= 200hPa-wind-speed seeing)
                    else:
                        model_color_PRIMAVERA_label = 'PRIMAVERA ' +  sg_clim_var[:3] # + ' single level'


            # now here, append ensemble taylor data to ref_pred_ls (to keep order of marD) -- well, dict is not ordered anyway..!?
            if d_Ensemble != None:
                if i == 0:
                    print('APPENDING\n')
                    for ensemble_taylor_dataset in ref_pred_ls_Ensemble:
                        ref_pred_ls.append(ensemble_taylor_dataset)
                    for name_ensemble_taylor_dataset in name_ref_pred_Ensemble:
                        name_ref_pred.append(name_ensemble_taylor_dataset)


            print('model {} done'.format(i))
    # add 'o' for Ensemble in the end
    if d_Ensemble != None:
        line_list.append(Line2D([0], [0], linestyle = '', color = 'k', marker = 'o',markersize = 15, label = 'Ensemble mean'))

    # append to list to plot inside each subplot (append models only if model_color == True)
    # for PRIMAVERA, we also need to constrain the forcing
    # because we need only the pressure level, we can try to combine the two 
    if MasterFig != None and d_Ensemble != None:
        if model_color == True:
            # I combine the colors for coupled and atmos-only into one Patch!
            # with the help of stackoverflow: https://stackoverflow.com/questions/57789191/matplotlib-patches-one-patch-with-mulitple-colors
            # use now 'model_color_PRIMAVERA_label'

            # below commented on 2020-12-11
            # colors = [ return_model_color('PRIMAVERA coupled')[0], return_model_color('PRIMAVERA atmos-only')[0]]
            # # attach to list with list comprehension
            # Ensemble_insitu_list.append([Patch([0], [0], color=c, label = model_color_PRIMAVERA_label) for c in colors])
            # Ensemble_insitu_labels.append(model_color_PRIMAVERA_label)

            # another option is to write explicitly which color corresponds to which kind of simulation
            # this way, the legend under the taylor diagram can be removed!
            # append coupled label and handle
            Ensemble_insitu_list.append(Patch([0], [0], color=return_model_color('PRIMAVERA coupled', second_pub_dataset = False)[0], label = model_color_PRIMAVERA_label + ' coupled'))
            Ensemble_insitu_labels.append(model_color_PRIMAVERA_label + ' coupled')
            # append atmos-only label and handle
            Ensemble_insitu_list.append(Patch([0], [0], color=return_model_color('PRIMAVERA atmos-only', second_pub_dataset = False)[0], label = model_color_PRIMAVERA_label + ' atmos-only'))
            Ensemble_insitu_labels.append(model_color_PRIMAVERA_label + ' atmos-only')

            if second_pub_dataset == True:
                # then we must append more Patches, because we have two datasets.
                # we pass second_pub_dataset=True
                Ensemble_insitu_list.append(Patch([0], [0], color=return_model_color('PRIMAVERA coupled', second_pub_dataset = True)[0], label = model_color_PRIMAVERA_label_SG + ' coupled'))
                Ensemble_insitu_labels.append(model_color_PRIMAVERA_label_SG + ' coupled')
                # append atmos-only label and handle
                Ensemble_insitu_list.append(Patch([0], [0], color=return_model_color('PRIMAVERA atmos-only', second_pub_dataset = True)[0], label = model_color_PRIMAVERA_label_SG + ' atmos-only'))
                Ensemble_insitu_labels.append(model_color_PRIMAVERA_label_SG + ' atmos-only')



    
    print('climate models plotting done.')

    # plot again insitu data, otherwise it is hidden behind climate model data
    if 'insitu_var' in d_obs.keys():
        # set in-situ color cycler
        insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p']))
        ax0.set_prop_cycle(insitu_cycler)
        if diurnal:
            ax1.set_prop_cycle(insitu_cycler)
        ax3.set_prop_cycle(insitu_cycler)

        for i, insitu_param in enumerate(d_obs['insitu_var']):
            if idx == 5 and variable == 'seeing_nc' and MasterFig != None:
                # timeseries
                plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
                ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
                                alpha=.2, facecolor='k')

            else:
                #seasonal cycle
                # seasonal
                hin = plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
                ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
                                alpha=.25, facecolor='k')
                #diurnal
                if diurnal:
                    plot_line(d_obs['ds_mean_hour'], insitu_param , ax1, 'hour')
                    ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][insitu_param] - d_obs['ds_std_hour'][insitu_param]), (d_obs['ds_mean_hour'][insitu_param] + d_obs['ds_std_hour'][insitu_param]),
                                    alpha=.2, facecolor='k')
                # timeseries
                plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
                ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
                                alpha=.2, facecolor='k')


    ########### Taylor Diagram

    # only now, if ERA5 is the reference, leave marD['*'] behind ('*' is used as the ERA5 marker)
    # only delete if there is only one entry for ERA5

    # commented on 24. july 2020: if 'insitu_var' not in d_obs.keys() and 'Plev' not in d_obs.keys() and len(marD['*']['w']) == 1:
    if '*' in marD.keys():
        if marD['*'] == {}: # check if marD['ERA5] is entirely empty
            del(marD['*'])

    # before calling 'taylor_statistics', 
    # make sure that ref_pred_ls has the same length 
    # than the colors
    print(marD)
    counter_colors = 0
    for mod in marD.keys():
        for forcVal in marD[mod].keys():
            for colVal in marD[mod][forcVal]:
                counter_colors += 1
    if counter_colors != (len(ref_pred_ls) - 1): # -1 for reference that is not plotted with a color
        raise Exception('Number of colors in marker dict (marD) is not the same as length of list of predictions: {} != {}'.format(counter_colors, (len(ref_pred_ls) - 1)))

    # list comprehension for list ref_pred_ls, to get taylor statistics (compare every prediction to the reference (first entry))
    taylor_stats = [sm.taylor_statistics(pred, ref_pred_ls[0],'data') for pred in ref_pred_ls[1:]]

    # # for testing length of ref_pred_ls
    # for idx4, arr in enumerate(ref_pred_ls):
    #     print(idx4, len(arr['data']['time']))


    # Store statistics in arrays
    # with list comprehension
    #
    if not diurnal: # only calculate if diurnal is not selected (if we have no d_obs in situ data, this does not work!)
        # first entry is special
        ls_sdev = []
        ls_crmsd = []
        ls_ccoef = []
        ls_sdev.append(taylor_stats[0]['sdev'][0]/taylor_stats[0]['sdev'][0]) # is it okey, if I just divide at the end to normalize?
        ls_crmsd.append(taylor_stats[0]['ccoef'][0]/taylor_stats[0]['sdev'][0] )
        ls_ccoef.append(taylor_stats[0]['ccoef'][0])

        # print("--- %s seconds for initializing ls_sdev,... lists ---" % (time.time() - start_time))
        # start_time = time.time()

        # expand
        ls_sdev += [taylor_stats[int(i)]['sdev'][1]/taylor_stats[0]['sdev'][0] for i in range(0,int(len(taylor_stats)))] # first is not included; normalize
        ls_crmsd += [taylor_stats[int(i)]['crmsd'][1]/taylor_stats[0]['sdev'][0]  for i in range(0,int(len(taylor_stats)))]
        ls_ccoef += [taylor_stats[int(i)]['ccoef'][1] for i in range(0,int(len(taylor_stats)))]

        # print("--- %s seconds for filling ls_sdev,...  ---" % (time.time() - start_time))
        # start_time = time.time()

        sdev = np.array(ls_sdev) # Standard deviations
        crmsd = np.array(ls_crmsd) # Centered Root Mean Square Difference 
        ccoef = np.array(ls_ccoef) # Correlation


        # RMSs(i) = sqrt(STDs(i).^2 + STDs(0)^2 - 2*STDs(i)*STDs(0).*CORs(i)) is true (law of cosine)
        
        # rank models
        # use skill score from Karl Taylor (2001)
        # write up dict with name of model and rank
        # model name from list that is written right after each element that is appended to ref_pred_ls
        
        # empty dict
        skill_dict = {}

        # define R_zero, which describes the highest possible correlation of any climate model
        # it is never equal to 1 (perfect correlation), due to internal variability (different initial conditions)
        R_zero = 0.995 # take 0.995 for now, but think about it (maybe take ERA5 correlation?)

        for i, corR in enumerate(ccoef):
            Skill = 4*(1 + corR)**4 / ( (sdev[i] + 1/sdev[i])**2 * (1 + R_zero)**4 )
            # write into dictionary
            skill_dict[name_ref_pred[i]] = [Skill, ccoef[i], sdev[i], crmsd[i]]
            # add correlation, sdev and centred root mean square error (crmsd)


        # sort dict after values (from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value)
        sorted_skill_dict = {k: v for k, v in sorted(skill_dict.items(), key=lambda item: item[1], reverse=True)}
        # print it to the screen
        # # print(sorted_skill_dict)
        # save sorted_skill_dict to csv
        # (maybe together with linear regression analysis?)
        # do later; use following
        # (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
        #     .to_csv(path_skill_folder + 'sorted_skill_dict.csv', header=False))

        # print("--- %s seconds for np.array(ls_sdev),... ---" % (time.time() - start_time))
        # start_time = time.time()

        # print(sdev)
        # print(crmsd)
        # print(ccoef)
    
    else: # if diurnal is selected, we need to set sorted_skill dict to an empty dict, so that it can be returned
        sorted_skill_dict = {}

    # plot taylor diagram on ax4
    if not diurnal: # else, we have conflicts with plotting (I don't know why exactly)
        if MasterFig != None: # for the masterfigure, we need to set option['legendCH'] to True for last row
            
            if idx == max_idx: # last line (space to plot legends)
                if model_color == True: # then we should take the colors for individual models only, and plot it separately
                    sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                                colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                                titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                                MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4, legendCH=False) 
                    # LEGEND no longer needed at all! (if it works, it can be ingested into if's below)

                    # edit: no legend needed because we already have the color legend under the timeseries!
                    # ##### add legend here for publication
                    # # if for publication
                    # Tay_Patch_list = []
                    
                    # if d_model != None:
                    #     d_Tay_color = d_color_generator_ind_model(PRIMAVERA=True)
                    # else:
                    #     d_Tay_color = d_color_generator_ind_model(PRIMAVERA=False)
                    
                    # for key, val in d_Tay_color.items():
                    #     # append patches to list for Taylor diagram legend
                    #     Tay_Patch_list.append(Patch(facecolor = val, edgecolor =val, label=key))

                    # ax4.legend(handles=Tay_Patch_list, loc='upper left', ncol=1, bbox_to_anchor= (0, -0.3))
                    # ######

                else: # plot legend with edgecolors normally, if not for publication
                    sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                                colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                                titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                                MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4, legendCH=True) 
            
            else: # set legendCH to False
                sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                    colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                    titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                    MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4, legendCH=False)
            
            # shrink axis
            # and move a bit to the left (x0-0.03) to get rid of white space

            box = ax4.get_position()
            ax4.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

        else:
            sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                MarkerDictCH=marD, alpha=0.7, markerSize= 9)



    # plt.gca().set_title('Taylor Diagram (Frequency = Monthly)')
    # plt.gca().text(0, 1.7, 'Taylor Diagram (Frequency = Monthly)', fontsize=12)

    print('Taylor diagram done.')
    # print("--- %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()

    # set labels and legends
    ax0.set_xlabel('time [months]')
    ax0.set_xticks(np.arange(1, 13, step=1))

    # --> commented on 2020-08-04, because I introduced sharey with ax0
    # # check if pressure level data (ERA5_var) or single_level data is available
    # if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
    #     ax0.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
    # elif 'single_lev' in d_obs.keys():
    #     ax0.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit
        
    if diurnal: # if diurnal == True
        ax1.set_xlabel('time [hours]')
        # check if pressure level data (ERA5_var) or single_level data is available
        if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
            ax1.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
        elif 'single_lev' in d_obs.keys():
            ax1.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit
        ax1.set_xticks(np.arange(2, 25, step=2))
        # label only every second hour
        ax1.set_xticklabels(np.arange(2, 25, step=2), rotation=45, fontsize='small')


    ax3.set_xlabel('time [years]')
    start, end = ax3.get_xlim()
    ax3.set_xticks(np.arange(int(round(start)), int(round(end)), step=4))
    ax3.set_xticklabels(np.arange(int(round(start)), int(round(end)), step=4), rotation=45)

    
    # check if pressure level data (ERA5_var) or single_level data is available
    if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
        ax3.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
    elif 'single_lev' in d_obs.keys():
        ax3.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit

    # ax4.set_title('Taylor Diagram (Frequency = Monthly)')
    
    if MasterFig == None: 
        # seasonal cycle
        # same y axis, same legend
        ax0.set_title('seasonal cycle (with data from  {} months)'.format(d_obs['ds_taylor']['time'].size))
        if diurnal: # if diurnal == True
            ax1.set_title('diurnal cycle')
        ax3.set_title('timeseries')

        ax0.legend(handles=line_list, loc='upper left', ncol=2, bbox_to_anchor= (0, -0.3))
        leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.3)) # legend of forcings (under timeseries)
        leg2 = ax3.legend(handles=Patch_list , loc='upper right', bbox_to_anchor=(0.7, -0.3))
        # because leg1 will be automatically removed, I have to add it again as an artist
        ax3.add_artist(leg1)

        # check if pressure level data (ERA5_var) or single_level data is available
        if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
            fig.suptitle(f'{site_noUnderline} ' + d_obs['ds'][d_obs['ERA5_var'][0]].long_name + f', longitude = {lon:.2f}, latitude = {lat:.2f}', fontsize = '14')
        elif 'single_lev' in d_obs.keys():
            fig.suptitle(f'{site_noUnderline} ' + d_obs['ds'][d_obs['single_lev'][0]].long_name + f', longitude = {lon:.2f}, latitude = {lat:.2f}', fontsize = '14')

    if MasterFig != None: # same as 'else'
        # ylabel visibile = False
        ax0.set_ylabel('')
        plt.setp(ax0.get_yticklabels(), visible=False)
        
        # title: site (over timeseries)
        ax3.set_title(f'{alphabet_from_idx(idx)}) {site_noUnderline}')

        # no title (otherwise, we get automatically the lon/lat xarray title)
        # ax3.title.set_visible(False)
        ax0.title.set_visible(False)

        # plot legend into timeline, indicating the in situ data specs (e.g. in situ Temperature 2m)
        # 
        # leg0 = ax3.legend(handles=Ensemble_insitu_list, labels = Ensemble_insitu_labels, loc='lower left', 
        #                     ncol=len(Ensemble_insitu_list), handler_map = {list: HandlerTuple(None)}) # bbox_to_anchor= (0, 0),loc='upper left'
        if second_pub_dataset == True:
            # we need ncol = 2, otherwise legend is too big
            leg0 = ax3.legend(handles=Ensemble_insitu_list, labels = Ensemble_insitu_labels, loc='lower left', 
                            ncol=2, bbox_to_anchor= (0.1, -0.05)) # bbox_to_anchor= (0, 0),loc='upper left'
                


        else: # if we have only one dataset, ncol = 1
            leg0 = ax3.legend(handles=Ensemble_insitu_list, labels = Ensemble_insitu_labels, loc='lower left', 
                            ncol=1, bbox_to_anchor= (0.68, -0.05)) # bbox_to_anchor= (0, 0),loc='upper left'
                
        # print(Ensemble_insitu_list)
        # print(Ensemble_insitu_labels)
        # add to Ensemble_insitu_list if model_color == True, so for the publication
        # we want to indicate the pressure level in the individual legend in each subplot
        # it should not be overwritten by other legends, make sure for that by adding an artist
        # just append where necessary
        # if model_color == True:
        #     leg0 = ax3.legend(handles=Ensemble_insitu_list, loc='lower left', ncol=len(Ensemble_insitu_list)) # bbox_to_anchor= (0, 0),loc='upper left'


        if idx != max_idx: # for last row, draw axis labels (so this is true for all rows except last row)
            # plot in situ line style (provides information about measurements)
            # how to find out if there are more than one in situ data?
            
            # set label visibility to False
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax0.get_xticklabels(), visible=False)
            ax3.xaxis.label.set_visible(False)
            ax0.xaxis.label.set_visible(False)
            # ax3.xaxis.set_visible(False) # turns off helping lines as well...
            # ax0.xaxis.set_visible(False) 
        
        elif model_color == False: # same as: if idx == max_idx and model_color == False
            # for last entry, plot the legends
            # get all pressure level entries used (600-1000hPa)
            d_color = d_color_generator() # get all defined colors from climxa
            Patch_list_ensemble = []

            if d_model != None:
                clim_key_list = list(d_model.keys()) # make a list out of the model keys, so that I am able to select the first one (since they all have the same variable names stored)
            for key, val in d_color.items():
                # fill Patch list wit items in defined color dictionary

                # if '0' in key or '5' in key: # search for pressure levels (always have either 0 or 5 at the end)
                #     Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                
                for Plev in Plev_list:
                    if str(Plev) == key:
                        Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                                    
                if 'single_lev' in d_obs.keys():
                    if key in d_obs['single_lev']: # else, do not add 'hPa'
                        # if key == 'tcw':
                        #     key = 'total column water'
                        # elif key == 't2m':
                        #     key = 'two-metre temperature'

                        # append Patch list with single level keys
                        Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                    
                elif d_model!=None:
                    if 'single_lev_var' in d_model[clim_key].keys():
                        if key in d_model[clim_key_list[0]]['single_lev_var']:
                            # if key == 'prw':
                            #     key = 'precipitable rain water'
                            if key == 'q_integrated':
                                key = 'PWV from integral'
                            # elif key == 'tas':
                            #     key = 'two-metre temperature'

                            # append Patch list with single level keys
                            Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))

            ax0.legend(handles=line_list, loc='upper left', ncol=2, bbox_to_anchor= (0, -0.3))
            leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.3)) # legend of forcings (under timeseries)
            leg2 = ax3.legend(handles=Patch_list_ensemble,loc='upper right', bbox_to_anchor=(0.7, -0.3), ncol=3)
            # because leg1 will be automatically removed, I have to add it again as an artist
            ax3.add_artist(leg1)
            ax3.add_artist(leg0) # don't forget insitu plot!
        
        else: # same as: if idx == max_idx and model_color == True
            
            # COMMENTED ON 2021-01-07
            # Patch_list_ensemble = []

            # if d_model != None:
            #     d_color = d_color_generator_ind_model(PRIMAVERA=True, second_pub_dataset = False) # get defined colors for ERA5, PRIMAVERA coupled, PRIMAVERA atmos-only

            # else:
            #     d_color = d_color_generator_ind_model(PRIMAVERA=False, second_pub_dataset = False) # get defined color for ERA5

            # for key, val in d_color.items():
            #     # fills Patch_list with colors of ERA5 and PRIMAVERA to plot a legend
            #     Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))

            ax0.legend(handles=line_list, loc='upper left', ncol=3 , bbox_to_anchor= (0, -0.2))
            # legend of forcing linestyles (under timeseries)
            leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.2)) 
            # ncol = 1 (only 3 entries!)
            # leg2 = ax3.legend(handles=Patch_list_ensemble,loc='upper right', bbox_to_anchor=(1, -0.3), ncol=1)
            # because leg1 will be automatically removed, I have to add it again as an artist
            # ax3.add_artist(leg1)
            ax3.add_artist(leg0)

        #### diurnal cycle, Masterfigure    
        if diurnal:
            ax1.set_title(f'{alphabet_from_idx(idx)}) {site_noUnderline}')
            # insitu variables legend should be shown in every subplot
            ax1.legend(handles=Ensemble_insitu_list, loc='lower left') #loc='upper left', bbox_to_anchor= (0, -0.25))

            # ylabel only for first image in row
            if (idx%3) != 0:
                ax1.yaxis.label.set_visible(False)

            if idx != max_idx and idx != (max_idx-1) and idx != (max_idx-2):
                # hide ticks except for last row
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax1.xaxis.label.set_visible(False)
               
            elif idx == max_idx: # plot legend into last subplot
                d_color = d_color_generator() # get all defined colors from climxa
                Patch_list_ensemble = []
                
                if d_model != None:
                    clim_key_list = list(d_model.keys()) # make a list out of the model keys, so that I am able to select the first one (since they all have the same variable names stored)
                for key, val in d_color.items():
                    # fill Patch list wit items in defined color dictionary

                    # if '0' in key or '5' in key: # search for pressure levels (always have either 0 or 5 at the end)
                    #     Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                    
                    for Plev in Plev_list:
                        if str(Plev) == key:
                            Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                        
                    if 'single_lev' in d_obs.keys():
                        if key in d_obs['single_lev']: # else, do not add 'hPa'
                            # if key == 'tcw':
                            #     key = 'total column water'
                            # elif key == 't2m':
                            #     key = 'two-metre temperature'

                            # append Patch list with single level keys
                            Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                        
                    elif d_model!=None:
                        if 'single_lev_var' in d_model[clim_key].keys():
                            if key in d_model[clim_key_list[0]]['single_lev_var']:
                                # if key == 'prw':
                                #     key = 'precipitable rain water'
                                if key == 'q_integrated':
                                    key = 'PWV from integral'
                                # elif key == 'tas':
                                #     key = 'two-metre temperature'

                                # append Patch list with single level keys
                                Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                                
                # plot legend on last axis (create extra subplot for it)
                
                ax = fig_diurnal.add_subplot(rows, columns, idx+2)

                # plt.gca().set_aspect('equal', adjustable='datalim')

                # ERA5 legend
                leg1 = ax.legend(handles = line_list_ensemble, loc='upper left', bbox_to_anchor=(0,1))
                leg2 = ax.legend(handles = Patch_list_ensemble, loc = 'upper left', bbox_to_anchor= (0, 0.9), ncol=2)
                ax.add_artist(leg1)
                
                # set off whole subplot
                ax.axis('off')

    print(lon)

    # print("--- %s seconds for plotting attributes (title, axis) ---" % (time.time() - start_time))
    # start_time = time.time()

    if MasterFig == None:
        # save
        path_savefig = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Output/Plots/' + variable + '/'
        os.makedirs(os.path.dirname(path_savefig), exist_ok=True) 
        fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Output/Plots/' + variable + '/'+ site +'_DSC_T.pdf') #, bbox_inches='tight')
        # show and close
        plt.show()
        plt.close()

    # print("--- %s seconds for drawing and saving ---" % (time.time() - start_time))

    # return the figure so it can be saved or modified
    return fig, sorted_skill_dict, ax0, ax3
































#%%
# def plot_line(ds, var ,my_ax, time_freq, P_lev = None, alpha=0.85):
#     if P_lev != None:
#         hin = ds[var].sel(level = P_lev).plot.line(x=time_freq, markersize = 4, ax=my_ax, add_legend=False)
#     else:
#         hin = ds[var].plot.line(x=time_freq, markersize = 4, ax=my_ax)
#     return hin # required for hin[0].get_color()

# def xr_plot_cycles_timeseries_2(d_obs, site, variable, lon, lat, d_model = None, 
#                                 diurnal=False, fig_diurnal=None, d_Ensemble=None, MasterFig=None, ax_ref = None):
#     # initialize timer
#     # # start clock
#     # start_time = time.time() # measure elapsed time

#     # check time with: print("--- %s seconds ---" % (time.time() - start_time))
    
#     # define string of site that has no underlines, but white spaces
#     if site == 'MaunaKea':
#         site_noUnderline = 'Mauna Kea'
#     elif site == 'siding_spring':
#         site_noUnderline = 'Siding Spring'
#     else:
#         site_noUnderline = site.replace('_', ' ')
    
#     # initialize figure        
#     # fig, (ax0, ax1) = plt.subplots(ncols=2, sharey = True, figsize= (10,4))
#     if MasterFig == None:
#         fig = plt.figure(figsize = (25, 4),constrained_layout=True) # (this is not compatible with tight_layout)

#         if diurnal: # if diurnal == True
#             gs = fig.add_gridspec(1, 5)
#             # sorry for the confusing numbering. I wrote the code first so that the diurnal cycle comes at second position, but later I wanted to change it
#             ax1 = fig.add_subplot(gs[0, 0]) # diurnal cycle
#             ax0 = fig.add_subplot(gs[0, 1]) # seasonal cycle
#             ax3 = fig.add_subplot(gs[0, 2:-1]) # span timeseries over two subplot lengths
#             ax4 = fig.add_subplot(gs[0, 4]) # taylor diagram
#         else:
#             gs = fig.add_gridspec(1, 4)
#             # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
#             ax3 = fig.add_subplot(gs[0, 0:-2]) # span timeseries over two subplot lengths
#             ax0 = fig.add_subplot(gs[0, 2], sharey=ax3) # seasonal cycle
#             ax4 = fig.add_subplot(gs[0, 3])  # taylor diagram

#     else: # take MasterFig as figure, only append axes to it
#         print('MasterFig is getting unpacked!')
#         # MasterFig is a tuple (fig, idx)
#         fig, idx, max_idx, gs, Plev_list = MasterFig # unpack tuple

#         if ax_ref != None:
#             # take the reference axis
#             ax_ref0, ax_ref3 = ax_ref
#             # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
#             ax3 = fig.add_subplot(gs[idx, 0:-2],sharex=ax_ref3) # span timeseries over two subplot lengths
#             ax0 = fig.add_subplot(gs[idx, 2], sharey=ax3, sharex=ax_ref0) # seasonal cycle
#             ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram

#         else:
#             # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
#             ax3 = fig.add_subplot(gs[idx, 0:-2]) # span timeseries over two subplot lengths
#             ax0 = fig.add_subplot(gs[idx, 2], sharey=ax3) # seasonal cycle
#             ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram

#         if diurnal:
#             columns = 3
#             rows = math.ceil((max_idx + 1)/3) # calculate number of rows dependent on max_idx, round up with ceil
#             if ((max_idx + 1)%3) == 0: # add a row for the legend
#                 rows = rows + 1
#                 print('row is added. we now have {} rows'.format(rows))
#             # add a new subplot (fig_diurnal may not be None!)
#             ax1 = fig_diurnal.add_subplot(rows, columns, idx + 1)

#     # print("--- %s seconds for initializing figure ---" % (time.time() - start_time))
#     # start_time = time.time()

#     # create empty lists for handlebars for the legend
#     Patch_list = [] # colors of different pressure levels and single level variables
#     line_list = [] # markers for different models (including ERA5 and in-situ data)
#     Ensemble_insitu_list = [] # labels of in situ data only
#     line_list_ensemble = [] # labels of other climate data (ERA5 mainly)
#     forcing_line_list = [] # line styles for different forcings

#     # for taylor diagram (only labels of predictions, not the reference)
#     taylor_label = [] # 
#     # marker dictionary
#     marD = {} #(e.g. marD = {'d': ['b','k', 'y'], '+': ['g']}# 'd' for ERA5, '+' for HadGEM)
#     # initialize marker collection
#     # mar_collection = cycle(['*','p','h', 's','d','^','v'])

#     # edgecol_collection = cycle(['w' ,'k']) # there are two different forcings, so take black and white

#     # assign reference (insitu) and predictions (ERA5 and model data)
#     # empty list for taylor diagram (reference and predictions)
#     ref_pred_ls = []
#     name_ref_pred = []

    
#     ########### in situ

#     # check if there is in-situ data
#     if 'insitu_var' in d_obs.keys():
#         # set in-situ color cycler
#         insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p'])  * cycler(markersize=[12]))
#         ax0.set_prop_cycle(insitu_cycler)
#         if diurnal:
#             ax1.set_prop_cycle(insitu_cycler)
#         ax3.set_prop_cycle(insitu_cycler)

#         for i, insitu_param in enumerate(d_obs['insitu_var']):
#             #seasonal cycle
#             # seasonal
#             hin = plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
#             ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
#                             alpha=.25, facecolor='k')
#             #diurnal
#             if diurnal:
#                 plot_line(d_obs['ds_mean_hour'], insitu_param , ax1, 'hour')
#                 ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][insitu_param] - d_obs['ds_std_hour'][insitu_param]), (d_obs['ds_mean_hour'][insitu_param] + d_obs['ds_std_hour'][insitu_param]),
#                                 alpha=.2, facecolor='k')
#             # timeseries
#             plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
#             ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
#                             alpha=.2, facecolor='k')
#             # for taylor plot (very first entry of ref_pred_ls is reference!)
#             # if there are more than one in-situ variables, take only the first in the list_of_insitu_vars
#             if i == 0:
#                 ref_pred_ls.append({"data" : d_obs['ds_taylor'][insitu_param]})
#                 name_ref_pred.append('in-situ ' + insitu_param)

#             # if no model data is present, compare in situ data against each other
#             # elif 'ERA5_var' not in d_obs.keys() and 'single_lev' not in d_obs.keys() and d_model == None:
#             #     ref_pred_ls.append({"data" : d_obs['ds_taylor'][insitu_param]})
#             #     name_ref_pred.append('in-situ ' + insitu_param)
#             #     if i == 1: # only do once
#             #         marD['o'] = {}
#             #         marD['o']['w'] = 'gray'
#             #     if i == 2:
#             #         marD['o']['w'] = 'black'

#             line_list.append(Line2D([0], [0], color = 'k',marker = hin[0].get_marker(), label = 'in-situ ' + insitu_param.replace('_', ' ')))        #  + r' $\pm$ std. dev.'
#             Ensemble_insitu_list.append(Line2D([0], [0], color = 'k',marker = hin[0].get_marker(), label = 'in-situ ' + insitu_param.replace('_', ' ') ))  # + r' $\pm$ std. dev.'
#         print('in-situ plotting done.')
#         # print("--- %s seconds ---" % (time.time() - start_time))
#         # start_time = time.time()

      
    
#     ########## ERA 5

#     # create entry in legend for linestyle of ERA5

#     ###################
#     if 'ERA5_var' in d_obs.keys(): 
#         # taylor label for ERA5 
#         # append label list only, if ERA5 should not be the reference,
#         # meaning if there is insitu data
#         # if 'insitu_var' in d_obs.keys():
#         #     taylor_label.append('ERA5')

#         # else, do not append list

#         marD_ERA5 =  '*'
#         marD[marD_ERA5] = {} # initialize subdict, for edgecolors

#         # get color
#         if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
#             # we need either coupled ('hist' and 'future') or atmosphere-only ('present', 'SSTfuture')
#             color = return_model_color('ERA5') # unpack list!

#         else: # else, get colors for individual pressure levels
#             col = [P_lev_color(x) for x in d_obs['Plev']]

#         # store colors in marker dict
#         # only append marD if ERA5 pressure level data doesn't have to be reference
#         if 'insitu_var' not in d_obs.keys(): # then, ERA5 has to give a reference
#             # check if list of colors has length greater than 1, 
#             # then, append marD with colors except first pressure level (which serves now as a reference)
#             # WAIT: why first pressure level and not CLOSEST pressure level? (okey, it works for tcw, because there is only one pressure level (until now))
#             if len(col) > 1:
#                marD[marD_ERA5]['w'] = col[1:]

#         else:
#             marD[marD_ERA5]['w'] = col # take white ('w') as edgecolor for ERA5

#         # set cycler for PRESSURE level data
#         Era5_cycler = (cycler(color = col) * cycler(linestyle=['-']) * cycler(marker = ['*']))

#         ax0.set_prop_cycle(Era5_cycler)
#         if diurnal: # if diurnal == True
#             ax1.set_prop_cycle(Era5_cycler)
#         ax3.set_prop_cycle(Era5_cycler)

#         # loop through climate variables to plot
        
#         for clim_var in d_obs['ERA5_var']:
#             for P_lev in d_obs['Plev']:  
#                 # seasonal
#                 hin = plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month', P_lev)
#                 #diurnal
#                 if diurnal: # if diurnal == True
#                     plot_line(d_obs['ds_mean_hour'], clim_var ,ax1, 'hour', P_lev)
                    
#                     if variable == 'TCW': # plot standard deviations if there is no in situ data
#                         if (site != 'MaunaKea') and (site != 'Paranal') and (site != 'La_Silla'):
#                             ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][clim_var].sel(level=P_lev) - d_obs['ds_std_hour'][clim_var].sel(level=P_lev))
#                                         , (d_obs['ds_mean_hour'][clim_var].sel(level=P_lev) + d_obs['ds_std_hour'][clim_var].sel(level=P_lev)),
#                                         alpha=.25)
                            
#                 # timeseries
#                 plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year', P_lev)
#                 # print(hin[0].get_color())
#                 # taylor
#                 ref_pred_ls.append({"data" : d_obs['ds_taylor'][clim_var].sel(level = P_lev)})
#                 name_ref_pred.append('ERA5 ' + str(P_lev))

#                 Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = str(P_lev) + 'hPa'))
                
#                 # print("--- %s seconds for one pressure level" % (time.time() - start_time))
#                 # start_time = time.time()


#     if 'single_lev' in d_obs.keys() :
#         # if there are no Pressure level data, you have to assign the ERA5 marker
#         if 'Plev' not in d_obs.keys() or d_obs['Plev'] == None:
#             marD_ERA5 =  '*'
#             if 'insitu_var' in d_obs.keys(): # then we already have a reference (else, we do not need marD_ERA5 at all)
#                 marD[marD_ERA5] = {}

#         # # append taylor label only if 'ERA5' not already in list
#         # if 'ERA5' not in taylor_label and 'insitu_var' in d_obs.keys():
#         #     taylor_label.append('ERA5')

#         # set cycler for SINGLE level data
#         # get color
#         if MasterFig != None and model_color == True: # then, we only have one ERA5 dataset!
#             # we need either coupled ('hist' and 'future') or atmosphere-only ('present', 'SSTfuture')
#             color = return_model_color('ERA5') # unpack list!

#         else: # else, get colors for individual pressure levels
#             col = [P_lev_color(x) for x in d_obs['single_lev']] # ['m','lightcoral', 'navy'] # probably not more than 3 different single level variables

#         # append ERA5 col
#         if 'Plev' not in d_obs.keys() or d_obs['Plev'] == None:
#                 if 'insitu_var' not in d_obs.keys():
#                     # now, ERA5 single level has to be the reference for the taylor diagram
#                     print('ERA5 single level data is now the reference')
#                 else:       
#                     marD[marD_ERA5]['w'] = col[:len(d_obs['single_lev'])] # only put in as many single level variables as we have

#         # if ERA5 must be reference for taylor diagram (insitu_var not in d_obs.keys), 
#         # then it might also be that a pressure level has already been used as a ref, 
#         # and marD['*']['w'] doesnt exist anymore
#         elif 'insitu_var' not in d_obs.keys() and len(d_obs['Plev']) == 1:
#             marD[marD_ERA5]['w'] = col[:len(d_obs['single_lev'])]
#         else: # if there are pressure levels, then we can simply add the list with +=
#             marD[marD_ERA5]['w'] += col[:len(d_obs['single_lev'])] # only put in as many single level variables as we have


#         Era5_SG_cycler = (cycler(color = col) * cycler(linestyle=['-'])) * plt.cycler(marker='*') # cycler(linestyle=['-']))
#         ax0.set_prop_cycle(Era5_SG_cycler)
#         if diurnal: # if diurnal == True
#             ax1.set_prop_cycle(Era5_SG_cycler)
#         ax3.set_prop_cycle(Era5_SG_cycler)

#         for clim_var in d_obs['single_lev']:
#             # seasonal
#             hin = plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month')
#             #diurnal
#             if diurnal: # if diurnal == True
#                 plot_line(d_obs['ds_mean_hour'], clim_var ,ax1, 'hour')
#             # timeseries
#             plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year')
#             # taylor
#             ref_pred_ls.append({"data" : d_obs['ds_taylor'][clim_var]})
#             name_ref_pred.append('ERA5 ' + clim_var)
            
#             Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = clim_var + ' single level'))

#     # ERA 5 label
#     line_list.append(Line2D([0], [0], linestyle = '-', marker = '*', markersize = 12, color = 'k', label = 'ERA5'))
#     line_list_ensemble.append(Line2D([0], [0], linestyle = '-', marker = '*', markersize = 12, color = 'k', label = 'ERA5'))
#     print('ERA5 plotting done.')


#     ########## climate models


#     # initialize linestyles for at least 6 different climate models
#     # lin_st = ['dashed', (0, (1, 1)), (0,(5,2,5,5,1,4)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0,(5, 2, 20, 2)) ] # (0, (3, 5, 1, 5, 1, 5)) dashed, dotted, densely dotted, dashdotted, densely dashdotdotted, dashdotdotted
#     lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
    
#     #--> instead of code below, use same marker for same model! (use climxa.clim_key_marker(clim_key))
#     # mar_collection_clim_models_only = ['p','h', 's','d','^','v']

#     if d_model != None:
#         # calculate Ensemble values, if d_Ensemble exists
#         if d_Ensemble != None:
#             calc_d_model_ensemble(d_model, d_Ensemble)
#             # I have to append in the order of marD!
#             # therefore, I initialize lists for Ensemble datasets 
#             # that can be appended after the first iteration
#             ref_pred_ls_Ensemble = []
#             name_ref_pred_Ensemble = []

#         # use different markers (same as for taylor diagram) for models
#         for i, clim_key in enumerate(d_model.keys()):

#             # scan here for empty arrays,
#             # loop through pressure levels
#             # delete pressure levels for totally (all forcings) empty arrays, 
#             # so that they do not lead to conflicts later
#             # do this only if 'ds_taylor' is available
#             if 'Plev' in d_model[clim_key].keys() and 'ds_taylor' in d_model[clim_key].keys():
#                 for PressureLevel in d_model[clim_key]['Plev']:
#                     # this is a bit ugly, since it is inside the forcing-loop
#                     # but I need the new marD .. why? --> marD is based on pressure levels

#                     # set counter to zero for one pressure level
#                     counter_empty_forced_arrays = 0
#                     # only loop through forcings that should be used for taylor_folder
#                     for forc_folder in d_model[clim_key]['taylor_folder']:
#                         clim_var_loop = d_model[clim_key]['clim_var'][0] + ' ' + forc_folder # [0] because it is a list (with one entry)
#                         # select the variable 'clim_var_loop' drop all nan's and write to dataset
#                         ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)

#                         if ds_check['time'].size == 0:
#                             print('I found an empty array: model = {}, plev = {}'.format(clim_key, PressureLevel))
#                             # if array is empty after dropping all nan's, delete the pressure level entry in ['Plev'] (delete the color entry in marD for this pressure level)
#                             # delete only, if all forcings for this pressure level are empty (check with counter)
#                             # add to counter for the current pressure level
#                             counter_empty_forced_arrays = counter_empty_forced_arrays + 1
#                             print(counter_empty_forced_arrays)
                            
#                             if counter_empty_forced_arrays == len(d_model[clim_key]['folders']):
#                                 # commented on July 20 (only need to delete a pressure level, nothing else (pressure level defines color entry)) 
#                                 # d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)
                                
#                                 if len(d_model[clim_key]['ds_taylor'][clim_var_loop].level) > 1:
#                                     # if the empty dataset has in reality more than one pressure levels left, then only delete the selection
#                                     # remove pressure level from d_model[clim_key]['Plev']
#                                     print('I have {} Pressure Levels'.format(len(d_model[clim_key]['ds_taylor'][clim_var_loop].level)))
#                                     if PressureLevel in d_model[clim_key]['Plev']:
#                                         print('I remove {} from {}.'.format(PressureLevel, clim_key))
#                                         d_model[clim_key]['Plev'].remove(PressureLevel)
#                                     # d_model[clim_key]['ds_seds_taylorl_cycle'][clim_var_loop] = d_model[clim_key]['ds_taylor'][clim_var_loop].drop_sel(level = PressureLevel)
#                                 # if len(d_model[clim_key]['ds_taylor'][clim_var_loop].level) == 1::
#                                 #     # also drop the variable
#                                 #     d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_taylor'].drop_vars(clim_var_loop)


#             line_list.append(Line2D([0], [0], linestyle = '', color = 'k', marker = clim_key_marker(clim_key), label = d_model[clim_key]['name']))

#             # 1 marker for one climate model
#             if 'ds_taylor' in d_model[clim_key].keys(): # only create entry in marD if ds_taylor available
#                 marD_clim = clim_key_marker(clim_key) # next(mar_collection)
#                 marD[marD_clim]= {} # initialize subdict

#             # reinitialize edgecolor cycle
#             edgecol_collection = cycle(['w' ,'k']) # there are two different forcings, so I take black and white to distinguish them

#             # loop through different forcings
#             for k, forcing in enumerate(d_model[clim_key]['folders']):
#                 if 'clim_var' in d_model[clim_key].keys():
#                     # clim_var was renamed in get_PRIMAVERA
#                     clim_var = d_model[clim_key]['clim_var'][0] + ' ' + forcing

#                 # specify line style (different for all forcings)
#                 if forcing=='hist':
#                     forced_linestyle = lin_st[0]
#                 elif forcing=='present':
#                     forced_linestyle = lin_st[1]
#                 elif forcing=='future':
#                     forced_linestyle = lin_st[2]
#                 elif forcing=='SSTfuture': 
#                     forced_linestyle = lin_st[3]
                

#                 # we need to define edgecolor for forcing, but only for forcings that go into taylor diagram
#                 if forcing in d_model[clim_key]['taylor_folder']:
#                     if 'ds_taylor' in d_model[clim_key].keys():
#                         marD_forcing = next(edgecol_collection) # not guaranteed that the same forcing has the same edgecolor 
#                         # works only if we have the same taylor_folders (same order, but second or both forcings can be missing)
#                         # (edgecol_collection is reinitialized for every new model)

#                 # create list of forcing labels, so that we can check if the label is already in the linestyle list
#                 list_of_forcing_labels = []
#                 for element in forcing_line_list:
#                     # append to list of labels
#                     list_of_forcing_labels.append(element.get_label())
                
#                 if forcing=='present':
#                     my_label = 'SST present'
#                 elif forcing == 'SSTfuture':
#                     my_label = 'SST future'
#                 else:
#                     my_label = forcing

#                 if my_label not in list_of_forcing_labels: # if the current forcing is not already in the label list for the legend, do so now
#                     forcing_line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = 'k', label = my_label))
                
#                 # append taylor label only if forcing is not already in list
#                 # and only for forcings that are going into taylor diagram
#                 for folder in d_model[clim_key]['taylor_folder']:
#                     # watch out here! I always use the coupled first in my 'taylor_folder'
#                     if 'coupled' not in taylor_label:
#                         if folder == 'hist' or folder == 'future':
#                             folder_label= 'coupled'
#                             taylor_label.append(folder_label)
#                     elif 'atmosphere-land' not in taylor_label:
#                         if folder == 'present' or folder == 'SSTfuture':
#                             folder_label = 'atmosphere-land'
#                             taylor_label.append(folder_label)

#                 # pressure level data
#                 if 'Plev' in d_model[clim_key].keys():

#                     # create cycler for all clim_vars (from different climate models!): P_lev_color * linestyle
#                     col_mod = [P_lev_color(x) for x in d_model[clim_key]['Plev']]
#                     col_mod_Ensemble = copy.deepcopy(col_mod)
#                     # clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[lin_st[i]])) # we need a list here

#                     if d_Ensemble == None:
#                         clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[forced_linestyle]) * plt.cycler(marker=[clim_key_marker(clim_key)])) # we need a list here
#                     else: # no marker at all for plotting Ensemble mean
#                         clim_model_cycler = (plt.cycler(color = col_mod) * plt.cycler(linestyle=[forced_linestyle])) # we need a list here

#                     if forcing in d_model[clim_key]['taylor_folder']:
#                         if 'ds_taylor' in d_model[clim_key].keys():
#                             # store colors in marker dict
#                             marD[marD_clim][marD_forcing] = col_mod
#                         if d_Ensemble != None and i == 0: # only once
#                             if 'o' not in marD.keys(): # otherwise, I overwrite it in the for loop!!
#                                 marD['o'] = {} # initialize Ensemble entry in marD
#                             # and fill dict with red (should be a big red dot)
#                             marD['o'][marD_forcing] = col_mod_Ensemble

#                     ax0.set_prop_cycle(clim_model_cycler)

#                     # # scan here for empty arrays
#                     # if k == 0: # do this only once (this is really ugly now...other way would be to also do marD out of all loops)
#                     #     for idx3, PressureLevel in enumerate(d_model[clim_key]['Plev']):
#                     #         # this is a bit ugly, since it is inside the forcing-loop
#                     #         # but I need the new marD .. why?

#                     #         # set counter to zero for one pressure level
#                     #         counter_empty_forced_arrays = 0
#                     #         for forc_folder in d_model[clim_key]['taylor_folder']:
#                     #             clim_var_loop = d_model[clim_key]['clim_var'][0] + ' ' + forc_folder

#                     #             ds_check = d_model[clim_key]['ds_sel_cycle'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_sel_cycle'][clim_var_loop].sel(level= PressureLevel)) != True, drop = True)
#                     #             # print(ds_check['time'].size)
#                     #             if ds_check['time'].size != 0:
#                     #                 # print('check done, array is not empty')
#                     #                 # only append pressure level dataarray if array is not empty
#                     #                 if forcing in d_model[clim_key]['taylor_folder']:
#                     #                     # only append to list that is fed into taylor diagram (data) if forcing is in taylor_folder
#                     #                     # otherwise, we do not want this data compared (and it is not in ds_taylor anyway)
#                     #                     # print('I append for model {} and forcing {} and pressure level {}'.format(clim_key, forc_folder, PressureLevel)) 
#                     #                     ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][clim_var_loop].sel(level = PressureLevel)})
#                     #                     name_ref_pred.append(clim_key + ' ' + str(PressureLevel) + ' ' + clim_var_loop)
#                     #             else:
#                     #                 print('I found an empty array: model = {}, plev = {}'.format(clim_key, PressureLevel))
#                     #                 # if array is empty after dropping all nan's, delete the color entry in marD for this pressure level
#                     #                 # delete color only, if all forcings for this pressure level are empty
#                     #                 # add to counter for the current pressure level
#                     #                 counter_empty_forced_arrays = counter_empty_forced_arrays + 1
#                     #                 print(counter_empty_forced_arrays)
#                     #                 if counter_empty_forced_arrays == len(d_model[clim_key]['folders']): 
#                     #                     # remove color from that specific pressure level from marD
#                     #                     # if it is not already removed!!
#                     #                     if P_lev_color(PressureLevel) in marD[marD_clim][marD_forcing]:
#                     #                         marD[marD_clim][marD_forcing].remove(P_lev_color(PressureLevel))
#                     #                 #   marD[marD_clim][marD_forcing].remove(marD[marD_clim][marD_forcing][idx2])


#                     if diurnal: # if diurnal == True
#                         ax1.set_prop_cycle(clim_model_cycler)
#                     ax3.set_prop_cycle(clim_model_cycler)

#                     # # append taylor label list for every model, use model name from dict d_model
#                     # taylor_label.append(d_model[clim_key]['name'])
                            
#                     # print("--- %s seconds before starting 1st for loop of climate model ---" % (time.time() - start_time))
#                     # start_time = time.time()

#                     # no loop needed below, since it is only one climate variable (always!)
#                     # # loop through different climate variables
#                     # for clim_var in d_model[clim_key]['clim_var']:

#                     # # label for linestyle
#                     # line_list.append(Line2D([0], [0], linestyle = lin_st[i], color = 'k', label = d_model[clim_key]['name']))

#                     for P_lev in d_model[clim_key]['Plev']: 
#                         if d_Ensemble == None: # plot each model individually
#                             # seasonal
#                             if 'ds_mean_month' in d_model[clim_key].keys():
#                                 hin = plot_line(d_model[clim_key]['ds_mean_month'], clim_var ,ax0, 'month', P_lev)
#                             #timeseries
#                             hin = plot_line(d_model[clim_key]['ds_mean_year'], clim_var ,ax3, 'year', P_lev)

#                         else: # plot Ensemble values
#                             if i == 0: # plot Ensemble (only for 1 clim_key (we need only d_Ensemble))
#                                 # seasonal
#                                 hin = plot_line(d_Ensemble['ds_ensemble_mean_month'], clim_var + ' mean' ,ax0, 'month', P_lev)
#                                     # std deviation seasonal cycle
#                                 ax0.fill_between(d_Ensemble['ds_ensemble_mean_month'].month, (d_Ensemble['ds_ensemble_mean_month'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_month'][clim_var + ' std'].sel(level=P_lev)),  
#                                     (d_Ensemble['ds_ensemble_mean_month'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_month'][clim_var + ' std'].sel(level=P_lev)),
#                                     alpha=.2)

#                                 # seasonal, other than in taylor_folder, for comparison
#                                 if forcing not in d_model[clim_key]['taylor_folder']:
#                                     # problem: cycler goes on with colors!
#                                     # --> solution: specify color separately
#                                     color = P_lev_color(P_lev)
#                                     d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level = P_lev).plot.line(x='month', ax=ax0, add_legend=False, color=color)
#                                         # std deviation
#                                     ax0.fill_between(d_Ensemble['ds_ensemble_mean_month_comparison'].month, (d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' std'].sel(level=P_lev)),  
#                                     (d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_month_comparison'][clim_var + ' std'].sel(level=P_lev)),
#                                     alpha=.2, color=color)

#                                 else: # if forcing in taylor folder, add to taylor diagram
#                                     # add to taylor diagram
#                                     ref_pred_ls_Ensemble.append({"data": d_Ensemble['ds_taylor'][clim_var  + ' mean'].sel(level = P_lev)})
#                                     name_ref_pred_Ensemble.append('d_Ensemble' + ' ' + str(P_lev) + ' ' + clim_var)
                                                           
#                                 # timeseries
#                                 plot_line(d_Ensemble['ds_ensemble_mean_year'], clim_var + ' mean' ,ax3, 'year', P_lev)
#                                     # std deviation timeseries
#                                 ax3.fill_between(d_Ensemble['ds_ensemble_mean_year'].year, (d_Ensemble['ds_ensemble_mean_year'][clim_var + ' mean'].sel(level=P_lev) - d_Ensemble['ds_ensemble_mean_year'][clim_var + ' std'].sel(level=P_lev)),  
#                                     (d_Ensemble['ds_ensemble_mean_year'][clim_var + ' mean'].sel(level=P_lev) + d_Ensemble['ds_ensemble_mean_year'][clim_var + ' std'].sel(level=P_lev)),
#                                     alpha=.2)
#                         #diurnal
#                         if diurnal: # if diurnal == True
#                             plot_line(d_model[clim_key]['ds_mean_hour'], clim_var ,ax1, 'hour', P_lev)

#                         # taylor
#                         # check here if array for this pressure level is not empty
#                         # NO --> do this out of this loop

#                         # create list of patch labels to check if we need to append the legend or not
#                         list_of_patch_labels = []
#                         for element in Patch_list:
#                             # append to list of labels
#                             list_of_patch_labels.append(element.get_label())

#                         if str(P_lev) + 'hPa' not in list_of_patch_labels: # if the current Pressure level (e.g. 700hPa) is not already drawn to the legend, do so now
#                             print('I append Patch_list')
#                             print('because {} is not in {}'.format(str(P_lev), list_of_patch_labels))
#                             Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = str(P_lev) + 'hPa'))
                        
#                         # check again for empty arrays, 
#                         # only append ref_pred_ls, if not empty
#                         if 'ds_taylor' in d_model[clim_key].keys() and forcing in d_model[clim_key]['taylor_folder']:
#                             ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][clim_var].sel(level= P_lev)) != True, drop = True)

#                             if ds_check['time'].size != 0:
#                                 ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][clim_var].sel(level = P_lev)})
#                                 name_ref_pred.append(clim_key + ' ' + str(P_lev) + ' ' + clim_var)

#                             # if array is empty   
#                             elif ds_check['time'].size == 0:   
#                                 print('empty array {}, {}'.format(clim_key, P_lev))
#                                 # remove color from that specific pressure level from marD
#                                 # if it is not already removed!!
#                                 if P_lev_color(P_lev) in marD[marD_clim][marD_forcing]:
#                                     print('I remove from dict: {} for {}, {}'.format(P_lev, marD_clim, marD_forcing))
#                                     marD[marD_clim][marD_forcing].remove(P_lev_color(P_lev))

#                         # print("--- %s seconds for one pressure level of one climate model ---" % (time.time() - start_time))
#                         # start_time = time.time()
                
#                 # single level data
#                 if 'single_lev_var' in d_model[clim_key].keys():
#                     sg_clim_var = d_model[clim_key]['single_lev_var'][0] + ' ' + forcing

#                     # set cycler for SINGLE level data
#                     # take same colors as for ERA5 single level variables
#                     col = [P_lev_color(x) for x in d_model[clim_key]['single_lev_var']] # ['m','lightcoral', 'navy']
#                     # take same colors as for ERA5 single level variables
#                     # clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[lin_st[i]])) # we need a list here
#                     if d_Ensemble == None:
#                         clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[forced_linestyle]) * plt.cycler(marker=[clim_key_marker(clim_key)])) # we need a list here
#                     else: # no marker
#                         clim_model_cycler = (plt.cycler(color = col) * plt.cycler(linestyle=[forced_linestyle])) # we need a list here

#                     ax0.set_prop_cycle(clim_model_cycler)
#                     if diurnal: # if diurnal == True
#                         ax1.set_prop_cycle(clim_model_cycler)
#                     ax3.set_prop_cycle(clim_model_cycler)

#                     # # if there are no Pressure level data, you have to assign the climate model marker
#                     # if 'Plev' not in d_model[clim_key].keys():                          
#                         # marD_clim = next(mar_collection)

#                     # append marker col
#                     if 'ds_taylor' in d_model[clim_key].keys():
#                         if 'Plev' not in d_model[clim_key].keys(): # if there is no pressure level data
#                             if forcing in d_model[clim_key]['taylor_folder']:
#                                 # store colors in marker dict
#                                 marD[marD_clim][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
                        
#                             #marD[marD_clim] = col[:len(d_model[clim_key]['single_lev_var'])] # only put in as many single level variables as we have
#                         else: # if there are pressure levels, then we can simply add the list with +=
#                             if forcing in d_model[clim_key]['taylor_folder']:
#                                 marD[marD_clim][marD_forcing] += col[:len(d_model[clim_key]['single_lev_var'])]
#                             # marD[marD_clim] += col[:len(d_model[clim_key]['single_lev_var'])] # only put in as many single level variables as we have

#                         # WHY IS THIS NOT NEEDED??????? I DON'T UNDERSTAND --> it is needed; I've had problems with deep and shallow copy of list
#                         if d_Ensemble != None:
#                             if forcing in d_model[clim_key]['taylor_folder'] and i == 0:
#                                 if 'o' not in marD.keys(): # otherwise, I overwrite it in the for loop!!
#                                     marD['o'] = {} # initialize Ensemble entry in marD
#                                 # and fill dict with color
#                                 if bool(marD['o']) == False: # if dict is empty
#                                     marD['o'][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
#                                 elif 'clim_var' not in d_Ensemble.keys(): # if there is only the single level var
#                                     marD['o'][marD_forcing] = col[:len(d_model[clim_key]['single_lev_var'])]
#                                 else: # append!
#                                     marD['o'][marD_forcing] += col[:len(d_model[clim_key]['single_lev_var'])]

#                     # append taylor label only if model name is not already in list
#                     # if  d_model[clim_key]['name'] not in taylor_label:
#                     #     taylor_label.append(d_model[clim_key]['name'])
                            
#                 # # plot single level model data
#                 # for sg_clim_var in d_model[clim_key]['single_lev_var']:
                    
#                     # # add to model legend only if there is no pressure level data
#                     # if 'Plev' not in d_model[clim_key].keys(): 
#                     #     line_list.append(Line2D([0], [0], linestyle = lin_st[i], color = 'k', label = d_model[clim_key]['name']))
#                     if d_Ensemble == None: # plot each model individually
#                         # seasonal
#                         if 'ds_mean_month' in d_model[clim_key].keys():
#                             hin = plot_line(d_model[clim_key]['ds_mean_month'], sg_clim_var ,ax0, 'month')

#                         #timeseries
#                         hin = plot_line(d_model[clim_key]['ds_mean_year'], sg_clim_var ,ax3, 'year')
                    
#                     else: # plot Ensemble, d_Ensemble is not None
#                         if i == 0: # (only for 1 clim_key (we need only d_Ensemble))
#                             # seasonal
#                             hin = plot_line(d_Ensemble['ds_ensemble_mean_month'], sg_clim_var + ' mean' ,ax0, 'month')
#                             ax0.fill_between(d_Ensemble['ds_ensemble_mean_month'].month, (d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' std']),  
#                                     (d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_month'][sg_clim_var + ' std']),
#                                     alpha=.2)                            

#                             # seasonal, other than in taylor_folder, for comparison
#                             # if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
#                             if forcing not in d_model[clim_key]['taylor_folder']:
#                                 # problem: cycler goes on with colors!
#                                 # --> solution: specify color separately
#                                 color = col[0] # assume there is only one single level data!
#                                 d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'].plot.line(x='month', ax=ax0, add_legend=False, color=color)
#                                     # std deviation
#                                 ax0.fill_between(d_Ensemble['ds_ensemble_mean_month_comparison'].month, (d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' std']),  
#                                 (d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_month_comparison'][sg_clim_var + ' std']),
#                                 alpha=.2, color=color)
                            
#                             else: # if forcing in taylor folder, add to taylor diagram
#                                 # add to taylor diagram
#                                 if bool(d_model[clim_key]['taylor_folder']) == True:
#                                     print('data to ref_pred_ls_Ensemble for {}, {}'.format(forcing, sg_clim_var))
#                                     ref_pred_ls_Ensemble.append({"data": d_Ensemble['ds_taylor'][sg_clim_var  + ' mean']})
#                                     name_ref_pred_Ensemble.append('d_Ensemble' + ' ' + sg_clim_var)
                                                    

#                             # timeseries
#                             plot_line(d_Ensemble['ds_ensemble_mean_year'], sg_clim_var + ' mean' ,ax3, 'year')
#                             ax3.fill_between(d_Ensemble['ds_ensemble_mean_year'].year, (d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' mean'] - d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' std']),  
#                                     (d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' mean'] + d_Ensemble['ds_ensemble_mean_year'][sg_clim_var + ' std']),
#                                     alpha=.2)

#                     #diurnal
#                     if diurnal: # if diurnal == True
#                         plot_line(d_model[clim_key]['ds_mean_hour'], sg_clim_var ,ax1, 'hour')
                    
#                     # taylor
#                     # check here if array for this pressure level is not empty
#                     # because in 'time_intersect', I only selected one pressure level (not true anymore)
#                     if 'ds_taylor' in d_model[clim_key].keys():
#                         ds_check = d_model[clim_key]['ds_taylor'].where(xr.ufuncs.isnan(d_model[clim_key]['ds_taylor'][sg_clim_var]) != True, drop = True)
#                         # print(ds_check['time'].size)
#                         if ds_check['time'].size != 0:
#                             # only append pressure level dataarray if array is not empty
#                             # print('check done, array is not empty')
#                             if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
#                                 if forcing in d_model[clim_key]['taylor_folder']:
#                                     # only append if forcing is in taylor_folder
#                                     ref_pred_ls.append({"data": d_model[clim_key]['ds_taylor'][sg_clim_var]})
#                                     name_ref_pred.append(clim_key + ' ' + sg_clim_var)
#                         else:
#                             print('I found an empty array: model = {}'.format(clim_key))
#                             # # if array is empty after dropping all nan's, delete that pressure level
#                             # marD[marD_clim].remove(marD[marD_clim][idx2]) ?? think about this case (but I do not think it should ever happen for single level data)
                        
#                     # create list of patch labels to check if we need to append the legend or not
#                     list_of_patch_labels = []
#                     for element in Patch_list:
#                         # append to list of labels
#                         list_of_patch_labels.append(element.get_label())

#                     if d_model[clim_key]['single_lev_var'][0] + ' single level' not in list_of_patch_labels and d_model[clim_key]['single_lev_var'][0] + ' single level (PRIMAVERA)' not in list_of_patch_labels: # if the current Pressure level (e.g. 700hPa) is not already drawn to the legend, do so now
#                         print('I append Patch_list')                   
#                         Patch_list.append(Patch(facecolor=hin[0].get_color(), edgecolor=hin[0].get_color(), label = d_model[clim_key]['single_lev_var'][0] + ' single level (PRIMAVERA)'))           
            
#             # now here, append ensemble taylor data to ref_pred_ls (to keep order of marD) -- well, dict is not ordered anyway..!?
#             if d_Ensemble != None:
#                 if i == 0:
#                     print('APPENDING\n')
#                     for ensemble_taylor_dataset in ref_pred_ls_Ensemble:
#                         ref_pred_ls.append(ensemble_taylor_dataset)
#                     for name_ensemble_taylor_dataset in name_ref_pred_Ensemble:
#                         name_ref_pred.append(name_ensemble_taylor_dataset)


#             print('model {} done'.format(i))
#     # add 'o' for Ensemble in the end
#     if d_Ensemble != None:
#         line_list.append(Line2D([0], [0], linestyle = '', color = 'k', marker = 'o',markersize = 15, label = 'Ensemble mean'))

            
#     print('climate models plotting done.')

#     # plot again insitu data, otherwise it is hidden behind climate model data
#     if 'insitu_var' in d_obs.keys():
#         # set in-situ color cycler
#         insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p']))
#         ax0.set_prop_cycle(insitu_cycler)
#         if diurnal:
#             ax1.set_prop_cycle(insitu_cycler)
#         ax3.set_prop_cycle(insitu_cycler)

#         for i, insitu_param in enumerate(d_obs['insitu_var']):
#             #seasonal cycle
#             # seasonal
#             hin = plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
#             ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
#                             alpha=.25, facecolor='k')
#             #diurnal
#             if diurnal:
#                 plot_line(d_obs['ds_mean_hour'], insitu_param , ax1, 'hour')
#                 ax1.fill_between(d_obs['ds_std_hour'].hour, (d_obs['ds_mean_hour'][insitu_param] - d_obs['ds_std_hour'][insitu_param]), (d_obs['ds_mean_hour'][insitu_param] + d_obs['ds_std_hour'][insitu_param]),
#                                  alpha=.2, facecolor='k')
#             # timeseries
#             plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
#             ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
#                             alpha=.2, facecolor='k')


#     ########### Taylor Diagram

#     # only now, if ERA5 is the reference, leave marD['*'] behind ('*' is used as the ERA5 marker)
#     # only delete if there is only one entry for ERA5

#     # commented on 24. july 2020: if 'insitu_var' not in d_obs.keys() and 'Plev' not in d_obs.keys() and len(marD['*']['w']) == 1:
#     if '*' in marD.keys():
#         if marD['*'] == {}: # check if marD['ERA5] is entirely empty
#             del(marD['*'])

#     # before calling 'taylor_statistics', 
#     # make sure that ref_pred_ls has the same length 
#     # than the colors
#     print(marD)
#     counter_colors = 0
#     for mod in marD.keys():
#         for forcVal in marD[mod].keys():
#             for colVal in marD[mod][forcVal]:
#                 counter_colors += 1
#     if counter_colors != (len(ref_pred_ls) - 1): # -1 for reference that is not plotted with a color
#         raise Exception('Number of colors in marker dict (marD) is not the same as length of list of predictions: {} != {}'.format(counter_colors, (len(ref_pred_ls) - 1)))

#     # list comprehension for list ref_pred_ls, to get taylor statistics (compare every prediction to the reference (first entry))
#     taylor_stats = [sm.taylor_statistics(pred, ref_pred_ls[0],'data') for pred in ref_pred_ls[1:]]

#     # # for testing length of ref_pred_ls
#     # for idx4, arr in enumerate(ref_pred_ls):
#     #     print(idx4, len(arr['data']['time']))


#     # Store statistics in arrays
#     # with list comprehension
#     #
#     # first entry is special
#     ls_sdev = []
#     ls_crmsd = []
#     ls_ccoef = []
#     ls_sdev.append(taylor_stats[0]['sdev'][0]/taylor_stats[0]['sdev'][0]) # is it okey, if I just divide at the end to normalize?
#     ls_crmsd.append(taylor_stats[0]['ccoef'][0]/taylor_stats[0]['sdev'][0] )
#     ls_ccoef.append(taylor_stats[0]['ccoef'][0])

#     # print("--- %s seconds for initializing ls_sdev,... lists ---" % (time.time() - start_time))
#     # start_time = time.time()

#     # expand
#     ls_sdev += [taylor_stats[int(i)]['sdev'][1]/taylor_stats[0]['sdev'][0] for i in range(0,int(len(taylor_stats)))] # first is not included; normalize
#     ls_crmsd += [taylor_stats[int(i)]['crmsd'][1]/taylor_stats[0]['sdev'][0]  for i in range(0,int(len(taylor_stats)))]
#     ls_ccoef += [taylor_stats[int(i)]['ccoef'][1] for i in range(0,int(len(taylor_stats)))]

#     # print("--- %s seconds for filling ls_sdev,...  ---" % (time.time() - start_time))
#     # start_time = time.time()

#     sdev = np.array(ls_sdev) # Standard deviations
#     crmsd = np.array(ls_crmsd) # Centered Root Mean Square Difference 
#     ccoef = np.array(ls_ccoef) # Correlation


#     # RMSs(i) = sqrt(STDs(i).^2 + STDs(0)^2 - 2*STDs(i)*STDs(0).*CORs(i)) is true (law of cosine)
    
#     # rank models
#     # use skill score from Karl Taylor (2001)
#     # write up dict with name of model and rank
#     # model name from list that is written right after each element that is appended to ref_pred_ls
    
#     # empty dict
#     skill_dict = {}

#     # define R_zero, which describes the highest possible correlation of any climate model
#     # it is never equal to 1 (perfect correlation), due to internal variability (different initial conditions)
#     R_zero = 0.995 # take 0.995 for now, but think about it (maybe take ERA5 correlation?)

#     for i, corR in enumerate(ccoef):
#         Skill = 4*(1 + corR)**4 / ( (sdev[i] + 1/sdev[i])**2 * (1 + R_zero)**4 )
#         # write into dictionary
#         skill_dict[name_ref_pred[i]] = [Skill, ccoef[i], sdev[i], crmsd[i]]
#         # add correlation, sdev and centred root mean square error (crmsd)


#     # sort dict after values (from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value)
#     sorted_skill_dict = {k: v for k, v in sorted(skill_dict.items(), key=lambda item: item[1], reverse=True)}
#     # print it to the screen
#     # # print(sorted_skill_dict)
#     # save sorted_skill_dict to csv
#     # (maybe together with linear regression analysis?)
#     # do later; use following
#     # (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
#     #     .to_csv(path_skill_folder + 'sorted_skill_dict.csv', header=False))

#     # print("--- %s seconds for np.array(ls_sdev),... ---" % (time.time() - start_time))
#     # start_time = time.time()

#     # print(sdev)
#     # print(crmsd)
#     # print(ccoef)

#     # plot taylor diagram on ax4
#     if not diurnal: # else, we have conflicts with plotting (I don't know why exactly)
#         if MasterFig != None: # for the masterfigure, we need to set option['legendCH'] to True for last row
#             if idx == max_idx: # plot legend
#                 sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
#                                 colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
#                                 titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
#                                 MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4, legendCH=True) 
#             else: # set legendCH to False
#                 sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
#                     colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
#                     titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
#                     MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4, legendCH=False)
            
#             # shrink axis
#             # and move a bit to the left (x0-0.03) to get rid of white space

#             box = ax4.get_position()
#             ax4.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

#         else:
#             sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
#                 colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
#                 titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
#                 MarkerDictCH=marD, alpha=0.7, markerSize= 9)



#     # plt.gca().set_title('Taylor Diagram (Frequency = Monthly)')
#     # plt.gca().text(0, 1.7, 'Taylor Diagram (Frequency = Monthly)', fontsize=12)

#     print('Taylor diagram done.')
#     # print("--- %s seconds ---" % (time.time() - start_time))
#     # start_time = time.time()

#     # set labels and legends
#     ax0.set_xlabel('time [months]')
#     ax0.set_xticks(np.arange(1, 13, step=1))

#     # --> commented on 2020-08-04, because I introduced sharey with ax0
#     # # check if pressure level data (ERA5_var) or single_level data is available
#     # if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
#     #     ax0.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
#     # elif 'single_lev' in d_obs.keys():
#     #     ax0.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit
        
#     if diurnal: # if diurnal == True
#         ax1.set_xlabel('time [hours]')
#         # check if pressure level data (ERA5_var) or single_level data is available
#         if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
#             ax1.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
#         elif 'single_lev' in d_obs.keys():
#             ax1.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit
#         ax1.set_xticks(np.arange(2, 25, step=2))
#         # label only every second hour
#         ax1.set_xticklabels(np.arange(2, 25, step=2), rotation=45, fontsize='small')


#     ax3.set_xlabel('time [years]')
#     start, end = ax3.get_xlim()
#     ax3.set_xticks(np.arange(int(round(start)), int(round(end)), step=4))
#     ax3.set_xticklabels(np.arange(int(round(start)), int(round(end)), step=4), rotation=45)

    
#     # check if pressure level data (ERA5_var) or single_level data is available
#     if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
#         ax3.set_ylabel(d_obs['ds'][d_obs['ERA5_var'][0]].long_name + ' [' + d_obs['ds'][d_obs['ERA5_var'][0]].units + ']') # construct y-label with variable name and unit
#     elif 'single_lev' in d_obs.keys():
#         ax3.set_ylabel(d_obs['ds'][d_obs['single_lev'][0]].long_name + ' [' + d_obs['ds'][d_obs['single_lev'][0]].units + ']') # construct y-label with variable name and unit

#     # ax4.set_title('Taylor Diagram (Frequency = Monthly)')
    
#     if MasterFig == None: 
#         # seasonal cycle
#         # same y axis, same legend
#         ax0.set_title('seasonal cycle (with data from  {} months)'.format(d_obs['ds_taylor']['time'].size))
#         if diurnal: # if diurnal == True
#             ax1.set_title('diurnal cycle')
#         ax3.set_title('timeseries')

#         ax0.legend(handles=line_list, loc='upper left', ncol=2, bbox_to_anchor= (0, -0.3))
#         leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.3)) # legend of forcings (under timeseries)
#         leg2 = ax3.legend(handles=Patch_list , loc='upper right', bbox_to_anchor=(0.7, -0.3))
#         # because leg1 will be automatically removed, I have to add it again as an artist
#         ax3.add_artist(leg1)

#         # check if pressure level data (ERA5_var) or single_level data is available
#         if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
#             fig.suptitle(f'{site_noUnderline} ' + d_obs['ds'][d_obs['ERA5_var'][0]].long_name + f', longitude = {lon:.2f}, latitude = {lat:.2f}', fontsize = '14')
#         elif 'single_lev' in d_obs.keys():
#             fig.suptitle(f'{site_noUnderline} ' + d_obs['ds'][d_obs['single_lev'][0]].long_name + f', longitude = {lon:.2f}, latitude = {lat:.2f}', fontsize = '14')

#     if MasterFig != None: # same as 'else'
#         # ylabel visibile = False
#         ax0.set_ylabel('')
#         plt.setp(ax0.get_yticklabels(), visible=False)
        
#         # title: site (over timeseries)
#         ax3.set_title(f'{alphabet_from_idx(idx)}) {site_noUnderline}')

#         # no title (otherwise, we get automatically the lon/lat xarray title)
#         # ax3.title.set_visible(False)
#         ax0.title.set_visible(False)

#         # plot legend into timeline, indicating the in situ data specs (e.g. in situ Temperature 2m)
#         ax3.legend(handles=Ensemble_insitu_list, loc='lower left', ncol=len(Ensemble_insitu_list)) # bbox_to_anchor= (0, 0),loc='upper left'

#         if idx != max_idx: # for last row, draw axis labels (so this is true for all rows except last row)
#             # plot in situ line style (provides information about measurements)
#             # how to find out if there are more than one in situ data?
            
#             # set label visibility to False
#             plt.setp(ax3.get_xticklabels(), visible=False)
#             plt.setp(ax0.get_xticklabels(), visible=False)
#             ax3.xaxis.label.set_visible(False)
#             ax0.xaxis.label.set_visible(False)
#             # ax3.xaxis.set_visible(False) # turns off helping lines as well...
#             # ax0.xaxis.set_visible(False) 

#         else: # for last entry, plot the legends
#             # get all pressure level entries used (600-1000hPa)
#             d_color = d_color_generator() # get all defined colors from climxa
#             Patch_list_ensemble = []

#             if d_model != None:
#                 clim_key_list = list(d_model.keys()) # make a list out of the model keys, so that I am able to select the first one (since they all have the same variable names stored)
#             for key, val in d_color.items():
#                 # fill Patch list wit items in defined color dictionary

#                 # if '0' in key or '5' in key: # search for pressure levels (always have either 0 or 5 at the end)
#                 #     Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                
#                 for Plev in Plev_list:
#                     if str(Plev) == key:
#                         Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                                    
#                 if 'single_lev' in d_obs.keys():
#                     if key in d_obs['single_lev']: # else, do not add 'hPa'
#                         # if key == 'tcw':
#                         #     key = 'total column water'
#                         # elif key == 't2m':
#                         #     key = 'two-metre temperature'

#                         # append Patch list with single level keys
#                         Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                    
#                 elif d_model!=None:
#                     if 'single_lev_var' in d_model[clim_key].keys():
#                         if key in d_model[clim_key_list[0]]['single_lev_var']:
#                             # if key == 'prw':
#                             #     key = 'precipitable rain water'
#                             if key == 'q_integrated':
#                                 key = 'PWV from integral'
#                             # elif key == 'tas':
#                             #     key = 'two-metre temperature'

#                             # append Patch list with single level keys
#                             Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))

#             ax0.legend(handles=line_list, loc='upper left', ncol=2, bbox_to_anchor= (0, -0.3))
#             leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.3)) # legend of forcings (under timeseries)
#             leg2 = ax3.legend(handles=Patch_list_ensemble,loc='upper right', bbox_to_anchor=(0.7, -0.3), ncol=3)
#             # because leg1 will be automatically removed, I have to add it again as an artist
#             ax3.add_artist(leg1)

#         #### diurnal cycle, Masterfigure    
#         if diurnal:
#             ax1.set_title(f'{alphabet_from_idx(idx)}) {site_noUnderline}')
#             # insitu variables legend should be shown in every subplot
#             ax1.legend(handles=Ensemble_insitu_list, loc='lower left') #loc='upper left', bbox_to_anchor= (0, -0.25))

#             # ylabel only for first image in row
#             if (idx%3) != 0:
#                 ax1.yaxis.label.set_visible(False)

#             if idx != max_idx and idx != (max_idx-1) and idx != (max_idx-2):
#                 # hide ticks except for last row
#                 plt.setp(ax1.get_xticklabels(), visible=False)
#                 ax1.xaxis.label.set_visible(False)
               
#             elif idx == max_idx: # plot legend into last subplot
#                 d_color = d_color_generator() # get all defined colors from climxa
#                 Patch_list_ensemble = []
                
#                 if d_model != None:
#                     clim_key_list = list(d_model.keys()) # make a list out of the model keys, so that I am able to select the first one (since they all have the same variable names stored)
#                 for key, val in d_color.items():
#                     # fill Patch list wit items in defined color dictionary

#                     # if '0' in key or '5' in key: # search for pressure levels (always have either 0 or 5 at the end)
#                     #     Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                    
#                     for Plev in Plev_list:
#                         if str(Plev) == key:
#                             Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
                        
#                     if 'single_lev' in d_obs.keys():
#                         if key in d_obs['single_lev']: # else, do not add 'hPa'
#                             # if key == 'tcw':
#                             #     key = 'total column water'
#                             # elif key == 't2m':
#                             #     key = 'two-metre temperature'

#                             # append Patch list with single level keys
#                             Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                        
#                     elif d_model!=None:
#                         if 'single_lev_var' in d_model[clim_key].keys():
#                             if key in d_model[clim_key_list[0]]['single_lev_var']:
#                                 # if key == 'prw':
#                                 #     key = 'precipitable rain water'
#                                 if key == 'q_integrated':
#                                     key = 'PWV from integral'
#                                 # elif key == 'tas':
#                                 #     key = 'two-metre temperature'

#                                 # append Patch list with single level keys
#                                 Patch_list_ensemble.append(Patch(facecolor=val,edgecolor=val, label = key))
                                
#                 # plot legend on last axis (create extra subplot for it)
                
#                 ax = fig_diurnal.add_subplot(rows, columns, idx+2)

#                 # plt.gca().set_aspect('equal', adjustable='datalim')

#                 # ERA5 legend
#                 leg1 = ax.legend(handles = line_list_ensemble, loc='upper left', bbox_to_anchor=(0,1))
#                 leg2 = ax.legend(handles = Patch_list_ensemble, loc = 'upper left', bbox_to_anchor= (0, 0.9), ncol=2)
#                 ax.add_artist(leg1)
                
#                 # set off whole subplot
#                 ax.axis('off')

#     print(lon)

#     # print("--- %s seconds for plotting attributes (title, axis) ---" % (time.time() - start_time))
#     # start_time = time.time()

#     if MasterFig == None:
#         # save
#         path_savefig = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Output/Plots/' + variable + '/'
#         os.makedirs(os.path.dirname(path_savefig), exist_ok=True) 
#         fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Output/Plots/' + variable + '/'+ site +'_DSC_T.pdf') #, bbox_inches='tight')
#         # show and close
#         plt.show()
#         plt.close()

#     # print("--- %s seconds for drawing and saving ---" % (time.time() - start_time))

#     # return the figure so it can be saved or modified
#     return fig, sorted_skill_dict, ax0, ax3

def alphabet_from_idx(idx):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    return alphabet[idx]
    
# fig = plt.figure( figsize = (28, 5),constrained_layout=True)
# gs = fig.add_gridspec(1, 5)
#         # sorry for the confusing numbering. I wrote the code first so that the diurnal cycle comes at second position, but later I wanted to change it
# ax1 = fig.add_subplot(gs[0, 0]) # diurnal cycle
# ax0 = fig.add_subplot(gs[0, 1]) # seasonal cycle
# ax3 = fig.add_subplot(gs[0, 2:-1]) # span timeseries over two subplot lengths
# ax4 = fig.add_subplot(gs[0, 4]) # taylor diagram
# for insitu_param in d_obs['insitu_var']:
#     plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
# start, end = ax3.get_xlim()
# ax3.set_xticks(np.arange(int(round(start)), int(round(end)), step=2))
# ax3.set_xticklabels(np.arange(int(round(start)), int(round(end)), step=2), rotation=45)



#%% OROGRAPHY #################################################################################################################################

#%% definition of functions for maps (orography, land sea mask, but also variables (T,SH,...))

def load_models_orography():
    # use nested dictionary
    d_oro_model = {'HadGEM': {"name": 'HadGEM3-GC31-HM'},
                'EC-Earth': {"name": 'EC-Earth-3-HR'},
                'MPI': {"name": 'MPI-ESM1-2-XR'},
                'ECMWF': {"name": 'ECMWF-IFS-HR'},
                'CMCC': {"name": 'CMCC-CM2-VHR4'},
                'CNRM': {"name": 'CNRM-CM6-1-HR'},
                'ERA5': {"name": 'ERA5'}}

    # path to orography files
    path_orog = '/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/orog/'

    # read in land-sea-mask (lsm)
    d_oro_model['HadGEM']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/HadGEM_n512e_eorca025_frac_land_sea_mask_no-unlim.nc') #
    d_oro_model['EC-Earth']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/lsm.EC-Earth-3-HR.v2.nc') # 
    d_oro_model['MPI']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/MPI_XR-landseamask.nc') # 
    d_oro_model['ECMWF']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/lsm.ECMWF-IFS-HR.v2.nc') # 
    d_oro_model['CMCC']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/masks_CMCC-CM2_VHR4_AGCM.nc') # 
    d_oro_model['CNRM']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/masks_CNRM-CM6-HR_AGCM_v1.nc') #  
    d_oro_model['ERA5']['lsm'] = xr.open_dataset(path_orog + 'land-sea-mask/ERA5_land_sea_mask.nc') #

    # select variable
    d_oro_model['HadGEM']['lsm'] = d_oro_model['HadGEM']['lsm'].land_area_fraction
    d_oro_model['EC-Earth']['lsm'] = d_oro_model['EC-Earth']['lsm'].lsm
    d_oro_model['MPI']['lsm'] = d_oro_model['MPI']['lsm'].SLM
    d_oro_model['ECMWF']['lsm'] = d_oro_model['ECMWF']['lsm'].lsm
    d_oro_model['CMCC']['lsm'] = d_oro_model['CMCC']['lsm'].mask # there is also area and frac. But I think I need mask.
    d_oro_model['CNRM']['lsm'] = d_oro_model['CNRM']['lsm'].mask_gr # downloaded data is 'regridded on a 360 gaussian grid'. We take the mask_gr, because it is on a regular lat-lon grid, correct??
    d_oro_model['ERA5']['lsm'] = d_oro_model['ERA5']['lsm'].lsm


    # read in model orography
    d_oro_model['HadGEM']['orog'] = xr.open_dataset(path_orog + 'orog_HadGEM3-GC31-HM.nc') # 0-360
    d_oro_model['EC-Earth']['orog']  = xr.open_dataset(path_orog + 'orog.EC-Earth-3-HR.nc') # 0-360 (lon/lat)
    d_oro_model['MPI']['orog']  = xr.open_dataset(path_orog + 'MPIESM-XR-meanOROGRAPHY.nc') # 0-360 (lon/lat)
    d_oro_model['ECMWF']['orog']  = xr.open_dataset(path_orog + 'orog_ECMWF-IFS-HR_tc0399_0.5x0.5.nc') # 0-360
    d_oro_model['CMCC']['orog']  = xr.open_dataset(path_orog + 'orog_fx_CMCC-CM2-VHR4_highresSST-present_r1i1p1f1_gn.nc') # 0-360 (lon/lat)
    d_oro_model['CNRM']['orog']  = xr.open_dataset(path_orog + 'orog_fx_CNRM-CM6-1-HR_highresSST-present_r1i1p1f2_gr.nc') #  0-360 (lon/lat)
    d_oro_model['ERA5']['orog']  = xr.open_dataset('/home/haslebacher/chaldene/cds_data_ERA5/single_level/Era5_orography_singleLevel.nc') # 0-360

    # select variable

    d_oro_model['HadGEM']['orog']  = d_oro_model['HadGEM']['orog'].surface_altitude
    d_oro_model['EC-Earth']['orog']  = d_oro_model['EC-Earth']['orog'].orog
    d_oro_model['MPI']['orog']  = d_oro_model['MPI']['orog'].OROMEA
    d_oro_model['ECMWF']['orog']  = d_oro_model['ECMWF']['orog'].z
    d_oro_model['CMCC']['orog']  = d_oro_model['CMCC']['orog'].orog
    d_oro_model['CNRM']['orog']  = d_oro_model['CNRM']['orog'].orog
    # for ERA5, first select variable, then select time
    d_oro_model['ERA5']['orog']  = d_oro_model['ERA5']['orog'].z_0001/9.81 # divide geopotential by gravitational acceleration
    d_oro_model['ERA5']['orog']  = d_oro_model['ERA5']['orog'].sel(time = '2019-03-01')

    # for d_oro_model['ECMWF']['orog'], d_oro_model['EC-Earth-3-HR']['lsm'] and ERA5 lsm, we also have to select the time
    d_oro_model['ECMWF']['orog'] = d_oro_model['ECMWF']['orog'].sel(time='1950-01-01')
    d_oro_model['ECMWF']['lsm'] = d_oro_model['ECMWF']['lsm'].sel(time='1950-01-01')
    d_oro_model['EC-Earth']['lsm'] = d_oro_model['EC-Earth']['lsm'].sel(time='1950-01-01')
    d_oro_model['ERA5']['lsm'] = d_oro_model['ERA5']['lsm'].sel(time='1979-01-01')

    # land-sea mask of some models is special (it has i and j as dimensions, not lon and lat). fix it.
    
    # this is not yet working!
    
    # for EC-EARTH
    daf = d_oro_model['EC-Earth']['lsm'].to_dataframe()
    daf_ind = daf.set_index(['latitude','longitude'])
    d_oro_model['EC-Earth']['lsm'] = daf_ind.lsm.to_xarray() # select 'lsm' to get a DataArray (instead of Dataset, which has no attribute 'contour')
    # for CMCC
    daf = d_oro_model['CMCC']['lsm'].to_dataframe()
    daf_ind = daf.set_index(['yc', 'xc'])
    d_oro_model['CMCC']['lsm']= daf_ind['mask'].to_xarray()
    d_oro_model['CMCC']['lsm'] = d_oro_model['CMCC']['lsm'].rename({'xc': 'longitude', 'yc': 'latitude'})
    # for CNRM
    daf = d_oro_model['CNRM']['lsm'].to_dataframe()
    daf_ind = daf.set_index(['lat_gr', 'lon_gr'])
    d_oro_model['CNRM']['lsm'] = daf_ind.mask_gr.to_xarray()
    # and rename lon_gr and lat_gr
    d_oro_model['CNRM']['lsm'] = d_oro_model['CNRM']['lsm'].rename({'lon_gr': 'longitude', 'lat_gr': 'latitude'})

    return d_oro_model


def check_order(model,x, y, lower_lon, upper_lon, lower_lat, upper_lat):
    # Note that xarray is fiddly with indexing - if x or y values are ordered
    # high to low, then the slice bounds need to be reversed.  So check that
    x_ordered_low2high = model[x].values[-1] - model[x].values[0] > 0
    y_ordered_low2high = model[y].values[-1] - model[y].values[0] > 0

    if x_ordered_low2high:
        x_index = slice(lower_lon, upper_lon)
    else:
        x_index = slice(upper_lon, lower_lon)

    if y_ordered_low2high:
        y_index = slice(lower_lat, upper_lat)
    else:
        y_index = slice(upper_lat, lower_lat)

    return x_index, y_index
    #subset = model.sel(x=x_index, y=y_index)

def as_si(x, ndp):
    if 'e' in str(x):
        s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
        m, e = s.split('e')
        return r'{m:s} \cdot 10^{{{e:d}}}'.format(m=m, e=int(e))
    elif x == 'intercept':
        return(np.sign(x) * abs(x))
    else:
        return(x)

def plot_oro(ax, model, index, param, list_vars, oro_name, projex, lon1, lon2, lat1, lat2, lon_obs, lat_obs, pressure, cmap1, norm1, observatory_name):
    # plot model
    orography = model[param].plot(ax=ax, transform=projex, cmap=cmap1, norm=norm1,
                     add_labels=False, add_colorbar=False)
    # add land-sea mask if it is in list_vars
    if 'lsm' in list_vars:
        # print(model['lsm'])
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1, 1, projection=projex)
        CS = model['lsm'].plot.contour(ax=ax, transform=projex,
                     add_labels=False, add_colorbar=False)
        # draw labels inline, but do not draw colorbar
        plt.clabel(CS, inline=1, fontsize=10)
        # plt.show()

    # orography = model[param].plot.contour(ax=ax, transform=projex,
    #              add_labels=False, add_colorbar=False)
    # ax.clabel(CS, inline=1, fontsize=10)

    # ax.set_global()
    ax.coastlines()
    ax.set_extent([lon1 , lon2, lat1, lat2])

    MK_x_ticks = np.arange(lon1, lon2, 0.25)
    ax.set_xticks(MK_x_ticks, crs=projex, minor = True)
    MK_y_ticks = np.arange(lat1, lat2, 0.25)
    ax.set_yticks(MK_y_ticks,crs=projex)

    # omit colorbar for subplot, instead, create global colorbar in the end

    #ax.xticks(rotation=90)
    # ax.gridlines(draw_labels=True)
    #observatory (google maps) Mauna Kea
    lon = lon_obs
    lat = lat_obs
    #x,y = m2(lon, lat)
    plt.plot(lon, lat, 'ko', markersize=6, label = observatory_name + ', ' + str(pressure) + 'hPa', transform=projex)
    
    # plot nearest gridpoint
    if 'longitude' in model['nearest'].coords:
        lon_m = model['nearest'].longitude
        lat_m = model['nearest'].latitude
    elif 'lon' in model['nearest'].coords:
        lon_m = model['nearest'].lon
        lat_m = model['nearest'].lat

    plt.plot(lon_m, lat_m,'ro', label='nearest gridpoint', transform=projex)
    # print also onto plot with textbox for overview
    if param == 'orog':
        print('nearest: {:.1f} m'.format(model['nearest'].values))
        plt.text( 0.1 , 0.1 , 'nearest: {:.1f} m'.format(model['nearest'].values), #(surface_ele),
        transform=ax.transAxes,  bbox=dict(facecolor='white', alpha = 0.6))
    elif param == '_slope':
        if model['_p_value'].values < 0.001:
            p_value = '<0.001'
        else:
            p_value = '=' +  '{:.2f}'.format(model['_p_value'].values)

        plt.text( 0.1 , 0.1 , 'nearest:\nslope={:.3f}'.format(model['nearest'].values)+ r'$\pm$' + '{:.3f}\np-value{}\n'.format(model['_std_err'].values, p_value) + r'$r^2$=' + '{:.2f}'.format((model['_r_value'].values)**2), #(surface_ele),
        transform=ax.transAxes, bbox=dict(facecolor='white', alpha = 0.6))

    ax.set_title(alphabet_from_idx(index) + ') ' + oro_name)

    return orography, ax

def min_max_ele(d_oro_model, param):
    # this function finds the minimum and maximum elevation (for use in colorbar)
    global_max = -4000
    global_min = 4000

    for super_model in d_oro_model.values():
        model = super_model[param] # e.g. param = 'orog'

        min_model = model.min()
        max_model = model.max()
        if max_model > global_max:
            global_max = max_model
        if min_model < global_min:
            global_min = min_model

    return  global_min.values, global_max.values


#%%
# create plot
def plot_oro_site(list_vars, d_oro_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, pressure, site_name, observatory_name, fx_var = None):
    # define Projection
    projex = ccrs.PlateCarree()

    # define colormap
    cmap1 = mpl.cm.cividis

    # fx_var defines which parameter of d_oro_model should be taken for defining vmin and vmax (for colorbar)
    if fx_var == None:
        fx_var = 'orog'
    vmin, vmax = min_max_ele(d_oro_model, fx_var)

    # # or use global min and max
    # vmin = -150
    # vmax = 2500
    norm1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # define plot array's arrangement
    columns = 2
    rows = 4

    # create figure with figsize big enough to accomodate all maps, labels, etc.
    fig = plt.figure(figsize=(8, 15)) #, tight_layout=False), sharey=True, sharex=True, 

    # main plotting loop
    for index, (oro_name, model) in enumerate(d_oro_model.items()):
        # if oro_name == 'ERA5':
        #     ax = fig.add_subplot(rows, columns, index + 2, projection=projex)
        #     plt.legend(loc = 'upper left', bbox_to_anchor= (-0.3, -0.3))
        # plot
        if index == 0:
            ax = fig.add_subplot(rows, columns, index + 1, projection=projex)
        else:
            ax = fig.add_subplot(rows, columns, index + 1, projection=projex, sharex=ax, sharey=ax)
        # print(ax.get_position())
        # try: 
            # orography = model.plot(ax=ax, transform=projex) #, cmap = 'cividis'
            # plot orography

        orography, ax = plot_oro(ax, model, index, fx_var, list_vars, oro_name, projex, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs,pressure, cmap1, norm1, observatory_name)
        
        if (index%2) != 0: # for plots on the right handside
            plt.setp(ax.get_yticklabels(), visible=False)
            # ax.xaxis.label.set_visible(False)
        if index != 5 and index != 6: 
            plt.setp(ax.get_xticklabels(), visible=False)
        else: # for last plots in column, plot xaxis labels
            plt.setp(ax.get_xticklabels(), visible=True)
        # plot_lsm

        # axe.iternext()
        # get handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # except KeyError: # if 
        #     print('variable {} not in dataset of model {}'.format(fx_var, oro_name))
        #     # pass

    # plot legend on last axis
    ax = fig.add_subplot(rows, columns, index + 2, projection=projex)
    # ax.set_anchor = ('S')
    # left, bottom, width, height = 0, 0.125, 0.455, 0.2891
    # ax.set_position([left, bottom, width, height])

    plt.gca().set_aspect('equal', adjustable='datalim')

    plt.legend(handles, labels, loc = 'upper left', bbox_to_anchor= (-0.1, 1))

    # plot world map with dot showing observatory on last axis
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.plot(lon_obs, lat_obs, 'ko',  markersize=9, transform=projex)

    # add a subplot for plotting a vertical colorbar
    bottom, top = 0.1, 0.95
    left, right = 0.1, 0.8
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.1)

    cbar_ax = fig.add_axes([0.85, bottom, 0.05, top-bottom])
    cbar_ax = fig.colorbar(orography, cax=cbar_ax)  # plot colorbar
    # orography.set_clim(-2.0, 2.0)
    if fx_var == 'orog':
        cbar_ax.set_label('Surface Elevation [m]', fontsize = 15)
    # cbar_ax.set_yticklabels(['-1', '0', '100'])
    elif fx_var == '_rough_trend':
        cbar_ax.set_label('average of 5 latest years minus average of 5 first years', fontsize = 15)
    elif fx_var == '_slope':
        cbar_ax.set_label('slope of trendline (based on 5-year rolling mean)' , fontsize = 15)

    fig.suptitle(site_name, fontsize = 15)
    fig.tight_layout()

    if fx_var == 'orog':
        fig.savefig(site_name.replace(' ', '_') + '_orography.pdf', bbox_inches='tight') # save to 'orog' folder
        fig.savefig(site_name.replace(' ', '_') + '_orography.png', bbox_inches='tight', dpi=400) # save a png as well
    # for stats in ['_rough_trend', '_slope']     
    #  
    plt.show() 

    return fig



#%%
def PRIMAVERA_sel(list_vars, d_oro_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, incr = None):
    for oro_name, model in d_oro_model.items():
        for var in list_vars: #['orog']: #['orog', 'lsm']:
            try:
                print(oro_name)
                # add a small increment to the lat/lon boundaries to avoid white space in the plots
                # unfortunately, this creates a bias of the colorbar. To also avoid this, you would have to select twice
                # we just apply the smallest possible increment so that the colorbar is not disturbed too much, but the plots do not show white space
                if var == 'orog':
                    incr = 0.2
                elif var == 'lsm':
                    incr = 0.5 # for the land sea mask, which does not have a colorbar, the can take a higher increment
                else:
                    incr = incr
                x_index, y_index = check_order(model[var], 'longitude', 'latitude', lower_lon - incr, upper_lon + incr, lower_lat - incr, upper_lat + incr)
                d_oro_model[oro_name][var] = model[var].sel(longitude=x_index,latitude= y_index)
                # new dict entry with surface elevation of nearest gridpoint
                if var == 'orog':
                    d_oro_model[oro_name]['nearest'] = model[var].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')
            except KeyError:
                x_index, y_index = check_order(model[var], 'lon', 'lat', lower_lon - incr, upper_lon + incr, lower_lat - incr, upper_lat + incr)
                d_oro_model[oro_name][var] = model[var].sel(lon= x_index,lat= y_index)
                # new dict entry with surface elevation of nearest gridpoint
                if var == 'orog':
                    d_oro_model[oro_name]['nearest'] = model[var].sel(lon=lon_obs, lat=lat_obs, method='nearest')
        
    return d_oro_model

def main_oro(list_vars, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, pressure, site_name, observatory_name):
    # load models
    d_oro_model = load_models_orography()

    # select model grid
    d_oro_model = PRIMAVERA_sel(list_vars, d_oro_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs)

    # call plot function
    plot_oro_site(list_vars, d_oro_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, pressure, site_name, observatory_name)



# trend_map

# ['folders'] must be the same for all models (otherwise, do what?)


# function takes in: idx, path, d_model (['folders'], ['clim_var'], ['Plev'], ['single_lev_var'])

def trend_map(d_model, d_obs, variable, site, idx, path, SH_integral=False):

    # open site specific dictionary
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))
    
    # [0]: Mauna Kea
    # [1]: Cerro Paranal
    # [2]: La Silla
    # [3]: Cerro Tololo
    # [4]: La Palma
    # [5]: Siding Spring
    # [6]: Sutherland
    # [7]: SPM

    # index is used for getting the correct site
    print(d_site_lonlat_data['site_name'][idx])
                                                        # for zoomed out plots
    upper_lon = d_site_lonlat_data['upper_lon'][idx] # + 3
    lower_lon = d_site_lonlat_data['lower_lon'][idx] # - 3
    lower_lat = d_site_lonlat_data['lower_lat'][idx] # - 3
    upper_lat = d_site_lonlat_data['upper_lat'][idx] # + 3
    lon_obs = d_site_lonlat_data['lon_obs'][idx]
    lat_obs = d_site_lonlat_data['lat_obs'][idx]
    site_name = d_site_lonlat_data['site_name'][idx]
    observatory_name = d_site_lonlat_data['observatory_name'][idx]
    pressure = d_site_lonlat_data['pressure [hPa]'][idx] 

    ####### load PRIMAVERA data
    for clim_key in d_model.keys(): # go through all climate models
        if SH_integral == True:
            merge_model_plev = []
            for Plev in d_model[clim_key]['Plev']: # Plev gives us in this case the limit of integration; there can be more than one datasets
                # load in 'q_integrated' dataset
                path_q_integrated_model = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(Plev) + 'hPa.nc' # where files are stored
                ds_q_integrated_model = xr.open_dataset(path_q_integrated_model) 
                merge_model_plev.append(ds_q_integrated_model)
            # combine datasets on dimension 'level'
            d_model[clim_key]['ds_Amon_Pr'] = xr.merge(merge_model_plev, join='outer') # 'outer' uses union of object indexes
            
            if 'single_lev_var' not in d_model[clim_key].keys():
                print(clim_key)
                print('create ds_Amon')
                d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Pr']

        # pressure level data
        elif 'clim_var' in d_model[clim_key].keys() and d_model[clim_key]['clim_var'] != None:
            # function that loads in model data from .../chaldene/Astroclimate_Project/HighResMIP/variables/.../Amon/...
            d_model[clim_key]['ds_Amon_Pr'] = get_PRIMAVERA(d_model, clim_key, site, pressure_level=True)

            if 'single_lev_var' not in d_model[clim_key].keys():
            # if there is no single level data, ds_Amon_Pr is equal to ds_Amon
                d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Pr']
            
        # single level data
        if 'single_lev_var' in d_model[clim_key].keys():
            # function that loads in model data from .../chaldene/Astroclimate_Project/HighResMIP/variables/.../Amon/...
            d_model[clim_key]['ds_Amon_Sg'] = get_PRIMAVERA(d_model, clim_key, site, single_level=True)
            
            # if there is no pressure level data
            if 'clim_var' not in d_model[clim_key].keys():
            # if there is no pressure level data, ds_Amon_Sg is equal to ds_Amon
                d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Sg']
            
        # case where we have single and pressure level data --> combine now into one dataset, called ds_Amon
        if 'clim_var' in d_model[clim_key].keys() and 'single_lev_var' in d_model[clim_key].keys():
            d_model[clim_key]['ds_Amon'] = xr.merge([d_model[clim_key]['ds_Amon_Pr'], d_model[clim_key]['ds_Amon_Sg']])

    ###### prepare 'ds_Amon_year' and remove empty arrays entirely from dict

    clim_key_list = [] # empty list for not entirely empty arrays
    for clim_key in d_model.keys():
        # check if 'ds_Amon' is entirely empty
        if len(d_model[clim_key]['ds_Amon'].variables) != 0: # if array was empty, there are no variables at all
            ############# for 5year rolling mean function
            # select pressure levels of ds_Amon
            if 'clim_var' in d_model[clim_key].keys() and d_model[clim_key]['clim_var'] != None:
                d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon'].sel(level = d_model[clim_key]['Plev'])
            # group yearly
            d_model[clim_key]['ds_Amon_year'] = d_model[clim_key]['ds_Amon'].groupby('time.year').mean(dim='time')
            
        else:
            print('ds_Amon of {} is entirely empty'.format(clim_key))
            clim_key_list.append(clim_key)
    for clim_key in clim_key_list: # if I would do this inside the loop above, I can't because dict may not change its size
        del(d_model[clim_key])

    # prepare 'ds_rough_trend_Pr' (5 latest years minus 5 first years)
    for clim_key in d_model.keys():
        # pressure level data
        if 'clim_var' in d_model[clim_key].keys() and d_model[clim_key]['clim_var'] != None:
            # empty list for merging again after calculating d_5year_rolling
            rolling_list = []

            for clim_var in d_model[clim_key]['clim_var']:
                for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                    
                    forced_clim_var = clim_var + ' ' + forcing
                    ds_temp = d_model[clim_key]['ds_Amon_year'][forced_clim_var].dropna(dim='year', how = 'all') # drop year only if ALL values are nan for that year
                    ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err = d_5year_rolling_mean(ds_temp)
                    # write dataArrays into one dataset, assign name of variables
                    ds_combined_temp = xr.Dataset(data_vars={forced_clim_var + '_rough_trend': ds_rough_temp, 
                                        forced_clim_var + '_slope': ds_slope,
                                        forced_clim_var + '_r_value': ds_r_value,
                                        forced_clim_var + '_p_value': ds_p_value,
                                        forced_clim_var + '_std_err': ds_std_err})
                    # # rename dataset variables
                    # ds_rough_temp = ds_rough_temp.rename(forced_clim_var)

                    rolling_list.append(ds_combined_temp) # dimension 'time' goes away, so can be merged again
            d_model[clim_key]['ds_rough_trend_Pr'] = xr.merge(rolling_list)

    ###### load ERA5 data (copied from main_plotting_routine())
    if 'ERA5_var' in d_obs.keys():
        ERA5_var = d_obs['ERA5_var'][0] # get ERA5_variable (there should be only one entry in list)
    if 'single_lev' in d_obs.keys() :
        ERA5_var_Sg = d_obs['single_lev'][0]
    # define longitude for ERA5
    if lon_obs > 180:
        my_ERA5_lon = lon_obs - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
        print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
    else:
        my_ERA5_lon = lon_obs

    load_ds = []
    chile_grid = ['Tololo', 'Pachon', 'Silla']
    if any(x in site for x in chile_grid):
        site_ERA = 'Paranal'
    else:
        site_ERA = site

    # get data for q_integrated (precipitable water vapor)
    # if SH_integral_pressure != None: # think of something else than SH_integral_pressure
    if SH_integral == True: # stopp, q_integrated was calculated only for 'nearest'
        for P_lev in d_obs['Plev']:
            # load already integrated datasets
            path_q_integrated = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/q_integrated/ds_ERA5_q_integrated_monthly_resampled_nearest_' + str(P_lev) + 'hPa.nc' # where files are stored
    
            ds_q_integrated = xr.open_dataset(path_q_integrated)
            print(ds_q_integrated.load())
            if 'expver' in ds_q_integrated.coords:
                ds_q_integrated = ds_q_integrated.sel(expver = 5)

            load_ds.append(ds_q_integrated.load())

    else: # no SH_integral to load --> read in ERA5 data normally
        # get ERA5 data    
        # on pressure levels
        if 'Plev' in d_obs.keys(): # Plev is used to find corresponding folders
            ds_ERA5_Plev = read_ERA5_pr_level_data(site_ERA, d_obs['Plev'], variable, d_obs['ERA5_var'], ERA5_path = ERA5_path) # append list of datasets that are merged together
            # resample monthly
            ds_ERA5_Plev = ds_ERA5_Plev.resample(time = '1m', keep_attrs=True).mean()
                            
            load_ds.append(ds_ERA5_Plev)

    # on single levels
    if 'single_lev' in d_obs.keys() :
        ds_ERA5_Sglev = read_ERA5_sg_level_data(site_ERA, variable, d_obs['single_lev'])

        # resample monthly
        ds_ERA5_Sglev = ds_ERA5_Sglev.resample(time = '1m', keep_attrs=True).mean()

        load_ds.append(ds_ERA5_Sglev) # 

    # merge all ERA5 datasets
    d_obs['ds_ERA5'] = xr.merge(load_ds)
    # select time only until end of 2019. otherwise, trend is biased because we have only first part of 2020!
    d_obs['ds_ERA5'] = d_obs['ds_ERA5'].sel(time = slice(None, '2019-12-31'))
    # group yearly, take mean         
    d_obs['ds_ERA5_year'] = d_obs['ds_ERA5'].groupby('time.year').mean(dim='time')
    # create 'ds_rough_trend_Pr'
    if 'Plev' in d_obs.keys():
        rolling_list = []
        for ERA5_var in d_obs['ERA5_var']:
            ds_temp = d_obs['ds_ERA5_year'][ERA5_var].dropna(dim='year', how = 'all') # drop year only if ALL values are nan for that year
            ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err = d_5year_rolling_mean(ds_temp.load())
            # write dataArrays into one dataset, assign name of variables
            ds_combined_temp = xr.Dataset(data_vars={ERA5_var + '_rough_trend': ds_rough_temp, 
                                ERA5_var + '_slope': ds_slope,
                                ERA5_var + '_r_value': ds_r_value,
                                ERA5_var + '_p_value': ds_p_value,
                                ERA5_var + '_std_err': ds_std_err})
            # # rename dataset variables
            # ds_rough_temp = ds_rough_temp.rename(forced_clim_var)

            rolling_list.append(ds_combined_temp) # dimension 'time' goes away, so can be merged again
        d_obs['ds_rough_trend_ERA5_Pr'] = xr.merge(rolling_list)

    # plots for pressure levels
    if 'clim_var' in d_model[clim_key].keys() and d_model[clim_key]['clim_var'] != None:
        # loop through pressure levels and build 'map' out of 'ds_rough_trend_Pr'
        for clim_var in d_model[clim_key]['clim_var']:
            for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                clim_var_map = clim_var + ' ' + forcing
                for stats in ['_rough_trend', '_slope']: # good enough for now (think later about title)
                    for Plev in d_model[clim_key]['Plev']:
                        for clim_key in d_model.keys():
                            # create DataArray 'map' by selecting variable and pressure levels
                            d_model[clim_key][stats] = d_model[clim_key]['ds_rough_trend_Pr'][clim_var_map + stats].sel(level=Plev)
                            d_model[clim_key]['nearest'] = d_model[clim_key][stats].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')                        
                            d_model[clim_key]['_r_value'] = d_model[clim_key]['ds_rough_trend_Pr'][clim_var_map + '_r_value'].sel(longitude=lon_obs, latitude=lat_obs, level=Plev, method='nearest') 
                            d_model[clim_key]['_p_value'] = d_model[clim_key]['ds_rough_trend_Pr'][clim_var_map + '_p_value'].sel(longitude=lon_obs, latitude=lat_obs, level=Plev, method='nearest') 
                            d_model[clim_key]['_std_err'] = d_model[clim_key]['ds_rough_trend_Pr'][clim_var_map + '_std_err'].sel(longitude=lon_obs, latitude=lat_obs, level=Plev, method='nearest') 
                        # add ERA5 here
                        # note: for ERA5, we do not have the same pressure levels, but we still just take the 'nearest'
                        if 'Plev' in d_obs.keys():
                            d_model['ERA5'] = {}
                            d_model['ERA5'][stats] = d_obs['ds_rough_trend_ERA5_Pr'][ERA5_var + stats].sel(level=Plev, method='nearest')
                            d_model['ERA5']['nearest'] = d_model['ERA5'][stats].sel(longitude=my_ERA5_lon, latitude=lat_obs, method='nearest')
                            d_model['ERA5']['_r_value'] = d_obs['ds_rough_trend_ERA5_Pr'][ERA5_var + '_r_value'].sel(longitude=my_ERA5_lon, latitude=lat_obs, level=Plev, method='nearest')
                            d_model['ERA5']['_p_value'] = d_obs['ds_rough_trend_ERA5_Pr'][ERA5_var + '_p_value'].sel(longitude=my_ERA5_lon, latitude=lat_obs, level=Plev, method='nearest')
                            d_model['ERA5']['_std_err'] = d_obs['ds_rough_trend_ERA5_Pr'][ERA5_var + '_std_err'].sel(longitude=my_ERA5_lon, latitude=lat_obs, level=Plev, method='nearest')
                            
                        list_vars = [clim_var_map]
                        fx_var = stats
                        fig_map = plot_oro_site(list_vars, d_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, pressure, site_name, observatory_name, fx_var = fx_var)
                        # change title
                        fig_map.suptitle(site_name + ': ' + clim_var_map + stats + ', ' + str(Plev), fontsize=12)

                        # save here, because fig_map is overplotted
                        filename = site + '_' +  clim_var_map + '_' + str(Plev) +  'hPa.png'
                        fig_map.savefig(path + filename, bbox_inches='tight')

                        # delete ERA5 again here (otherwise it is also in d_model.keys())
                        if 'Plev' in d_obs.keys():
                            del(d_model['ERA5'])

    # plots for single level data
    for clim_key in d_model.keys(): # go through all climate models
        # single level data
        if 'single_lev_var' in d_model[clim_key].keys():
            # empty list for merging again after calculating d_5year_rolling
            rolling_list = []

            for sg_clim_var in d_model[clim_key]['single_lev_var']:
                for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                    forced_clim_var = sg_clim_var + ' ' + forcing
                    ds_temp = d_model[clim_key]['ds_Amon_year'][forced_clim_var].dropna(dim='year', how = 'all')
                    ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err = d_5year_rolling_mean(ds_temp)
                    # write dataArrays into one dataset, assign name of variables
                    ds_combined_temp = xr.Dataset(data_vars={forced_clim_var + '_rough_trend': ds_rough_temp, 
                                        forced_clim_var + '_slope': ds_slope,
                                        forced_clim_var + '_r_value': ds_r_value,
                                        forced_clim_var + '_p_value': ds_p_value,
                                        forced_clim_var + '_std_err': ds_std_err})
                    # # rename dataset variables
                    # ds_rough_temp = ds_rough_temp.rename(forced_clim_var)
                    rolling_list.append(ds_combined_temp)
            d_model[clim_key]['ds_rough_trend_Sg'] = xr.merge(rolling_list)

    ##### ERA5 single level var
    # create 'ds_rough_trend_Sg'
    if 'single_lev' in d_obs.keys() :
        rolling_list = []
        for ERA5_var_Sg in d_obs['single_lev']:
            ds_temp = d_obs['ds_ERA5_year'][ERA5_var_Sg].dropna(dim='year', how = 'all') # drop year only if ALL values are nan for that year
            ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err = d_5year_rolling_mean(ds_temp.load())
            # write dataArrays into one dataset, assign name of variables
            ds_combined_temp = xr.Dataset(data_vars={ERA5_var_Sg + '_rough_trend': ds_rough_temp, 
                                ERA5_var_Sg + '_slope': ds_slope,
                                ERA5_var_Sg + '_r_value': ds_r_value,
                                ERA5_var_Sg + '_p_value': ds_p_value,
                                ERA5_var_Sg + '_std_err': ds_std_err})

            rolling_list.append(ds_combined_temp) 
        d_obs['ds_rough_trend_ERA5_Sg'] = xr.merge(rolling_list)

    # Single level
    if 'single_lev_var' in d_model[clim_key].keys(): # wait: clim_key is the last value from above! (no loop here)

        for sg_clim_var in d_model[clim_key]['single_lev_var']:
            for forcing in d_model[clim_key]['folders']: # e.g. 'hist'
                clim_var_map = sg_clim_var + ' ' + forcing
                for stats in ['_rough_trend', '_slope']:
                    for clim_key in d_model.keys():
                        # loop through Pressure levels, and clim_vars
                        d_model[clim_key][stats] = d_model[clim_key]['ds_rough_trend_Sg'][clim_var_map + stats]
                        d_model[clim_key]['nearest'] = d_model[clim_key][stats].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')
                        d_model[clim_key]['_r_value'] = d_model[clim_key]['ds_rough_trend_Sg'][clim_var_map + '_r_value'].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')
                        d_model[clim_key]['_p_value'] = d_model[clim_key]['ds_rough_trend_Sg'][clim_var_map + '_p_value'].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')
                        d_model[clim_key]['_std_err'] = d_model[clim_key]['ds_rough_trend_Sg'][clim_var_map + '_std_err'].sel(longitude=lon_obs, latitude=lat_obs, method='nearest')
                    
                    # add ERA5 here
                    if 'single_lev' in d_obs.keys() :
                        d_model['ERA5'] = {}
                        d_model['ERA5'][stats] = d_obs['ds_rough_trend_ERA5_Sg'][ERA5_var_Sg + stats]
                        d_model['ERA5']['nearest'] = d_model['ERA5'][stats].sel(longitude=my_ERA5_lon, latitude=lat_obs, method='nearest')
                        d_model['ERA5']['_r_value'] = d_obs['ds_rough_trend_ERA5_Sg'][ERA5_var_Sg + '_r_value'].sel(longitude=my_ERA5_lon, latitude=lat_obs, method='nearest')
                        d_model['ERA5']['_p_value'] = d_obs['ds_rough_trend_ERA5_Sg'][ERA5_var_Sg + '_p_value'].sel(longitude=my_ERA5_lon, latitude=lat_obs, method='nearest')
                        d_model['ERA5']['_std_err'] = d_obs['ds_rough_trend_ERA5_Sg'][ERA5_var_Sg + '_std_err'].sel(longitude=my_ERA5_lon, latitude=lat_obs, method='nearest')
                        
                    list_vars = [clim_var_map]
                    fx_var = stats
                    fig_map = plot_oro_site(list_vars, d_model, upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, pressure, site_name, observatory_name, fx_var = fx_var)
                    fig_map.suptitle(site_name + ': ' + clim_var_map, fontsize=12)

                    # save here, because fig_map is overplotted
                    filename = site + '_' +  clim_var_map + '.png'
                    fig_map.savefig(path + filename, bbox_inches='tight')

                    # delete ERA5 again here
                    if 'single_lev' in d_obs.keys() :
                        del(d_model['ERA5'])


#%%

def round_significant(number_to_round, significant_digits):
    # from https://www.kite.com/python/answers/how-to-round-a-number-to-significant-digits-in-python#:~:text=Use%20round()%20to%20round,digits%20minus%20(int(math.)
    rounded_number =  round(number_to_round, significant_digits - int(math.floor(math.log10(abs(number_to_round)))) - 1)
    return rounded_number

def fill_d_future_trends(d_future_trends, clim_var, Plev, single_lev_var, site_name_folder):
    # initialize subdict for the input site
    d_future_trends[site_name_folder] = {}
    # fill this subdict with site specific information
    d_future_trends[site_name_folder]['clim_var'] = clim_var
    d_future_trends[site_name_folder]['Plev'] = Plev
    d_future_trends[site_name_folder]['single_lev_var'] = single_lev_var

    return d_future_trends

def idx_to_site(idx):
    if idx == 0:
        return 'Mauna_Kea'
    elif idx == 1:
        return 'Cerro_Paranal'
    elif idx == 2:
        return 'La_Silla'
    elif idx == 3:
        return 'Cerro_Tololo'
    elif idx == 4:
        return 'La_Palma'
    elif idx == 5:
        return 'Siding_Spring'
    elif idx == 6:
        return 'Sutherland'
    elif idx == 7:
        return 'SPM'
    else:
        raise Exception('idx is not well defined: {}'.format(idx))

def future_trendlines(d_future_trends, units, unit, variable, significant_digits, ylabel):

    # open site specific dictionary
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    # [0]: Mauna Kea
    # [1]: Cerro Paranal
    # [2]: La Silla
    # [3]: Cerro Tololo
    # [4]: La Palma
    # [5]: Siding Spring
    # [6]: Sutherland
    # [7]: SPM

    # initialize figure
    fig, ax = plt.subplots()

    lines_legend = []

    # colors for sites, take from colorbar
    inferno_colors = [to_hex(plt.cm.viridis(i / 8)) for i in range(8)] 

    # loop through sites
    for idx in range(0, 8):

        site = d_site_lonlat_data['site_name_folder'][idx]
        print(site)

        d_future_site = d_future_trends[site] # select correct dictionary

        d_Ensemble = pickle.load(open('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + variable + '_d_Ensemble.pkl', "rb"))

        rolling_list = []
        clim_var_list = [] 

        if 'clim_var' in d_future_site.keys() and d_future_site['clim_var'] != None:
            for clim_var in d_future_site['clim_var']:
                for P_lev in d_future_site['Plev']:
                    for forcing in ['future', 'SSTfuture']:
                        forced_clim_var = clim_var + ' ' + forcing + ' mean'
                        clim_var_list.append(forced_clim_var + ' ' + str(P_lev)) # for writing .csv file

                        ds_temp = d_Ensemble['ds_ensemble_mean_year'][forced_clim_var].sel(level=P_lev).dropna(dim='year', how = 'all')
                        
                        # write 5-year rolling into dataset that can be written to .dat file
                        ds_rolling = ds_temp.rolling(year = 5, center = True, min_periods = 5).mean()
                        ds_rolling.name = 'temp' # assign a name so that .to_dataframe() works
                        path_5rolling = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/' + idx_to_site(idx) +  forcing + '_' + str(P_lev) + '_5year_rolling_dataset.csv'
                        ds_rolling.dropna(dim='year').to_dataframe().to_csv(path_5rolling) # drop 2015,2016 and 2049, 2050
 
                        
                        ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err, ds_intercept = d_5year_rolling_mean(ds_temp, intercept=True)
                        ds_combined_temp = xr.Dataset(data_vars={forced_clim_var + ' ' + str(P_lev) + ' rough_trend': ds_rough_temp, 
                                                            forced_clim_var + ' ' + str(P_lev) + ' slope': ds_slope,
                                                            forced_clim_var + ' ' + str(P_lev) + ' r_value': ds_r_value,
                                                            forced_clim_var + ' ' + str(P_lev) + ' p_value': ds_p_value,
                                                            forced_clim_var + ' ' + str(P_lev) + ' std_err': ds_std_err,
                                                            forced_clim_var + ' ' + str(P_lev) + ' intercept': ds_intercept})

                        rolling_list.append(ds_combined_temp.reset_coords(drop = True))

            # d_Ensemble['ds_linregress'] = xr.merge(rolling_list)

        if 'single_lev_var' in d_future_site.keys() and d_future_site['single_lev_var'] != None:
            for clim_var in d_future_site['single_lev_var']:
                for forcing in ['future', 'SSTfuture']:
                    forced_clim_var = clim_var + ' ' + forcing + ' mean'
                    clim_var_list.append(forced_clim_var) # for .csv

                    ds_temp = d_Ensemble['ds_ensemble_mean_year'][forced_clim_var].dropna(dim='year', how = 'all')
                    
                    # write 5-year rolling into dataset that can be written to .dat file
                    ds_rolling = ds_temp.rolling(year = 5, center = True, min_periods = 5).mean()
                    ds_rolling.name = 'temp' # assign a name so that .to_dataframe() works
                    path_5rolling = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/' + idx_to_site(idx) +  forcing + '_single_level_5year_rolling_dataset.csv'
                    ds_rolling.dropna(dim='year').to_dataframe().to_csv(path_5rolling) # drop 2015,2016 and 2049, 2050
                    
                    ds_rough_temp, ds_slope, ds_r_value, ds_p_value, ds_std_err, ds_intercept = d_5year_rolling_mean(ds_temp, intercept=True)
                    ds_combined_temp = xr.Dataset(data_vars={forced_clim_var + ' rough_trend': ds_rough_temp, 
                                                        forced_clim_var + ' slope': ds_slope,
                                                        forced_clim_var + ' r_value': ds_r_value,
                                                        forced_clim_var + ' p_value': ds_p_value,
                                                        forced_clim_var + ' std_err': ds_std_err,
                                                        forced_clim_var + ' intercept': ds_intercept})

                    rolling_list.append(ds_combined_temp)

        # merge everything in the end
        d_Ensemble['ds_linregress'] = xr.merge(rolling_list)

        # write to .csv

        # save to file (variable in filename)
        # header = ['site', 'slope [units]', 'r^2', 'p_value', 'std_dev']

        path_linregress_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/'
        os.makedirs(os.path.dirname(path_linregress_folder), exist_ok=True) 

        # round to significant digits (from https://www.kite.com/python/answers/how-to-round-a-number-to-significant-digits-in-python#:~:text=Use%20round()%20to%20round,digits%20minus%20(int(math.)
        # a_number = 123.45
        # significant_digits = 4
        # rounded_number =  round(a_number, significant_digits - int(math.floor(math.log10(abs(a_number)))) - 1)


        if idx == 0:
            mode = 'w' # overwrite old file
        else:
            mode = 'a' # append to file

        with open(path_linregress_folder + variable + '_future_trends_linregress.csv', mode=mode) as csvfile:
            linregress_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            if idx == 0: # print header only once
                linregress_writer.writerow(['variable','site', 'forcing', 'slope ' + units, 'intercept ' + unit, 'R$^2$', 'p value'])
            
            # for all clim_vars that got saved above
            for clim_var in clim_var_list:
                # split clim_var string (e.g. 'ta future mean 700') to use it for labels and table
                split_list = clim_var.split(' ') # returns a list
                var = split_list[0]
                forcing = split_list[1]
                if len(split_list) == 4:
                    P_lev = split_list[3] + 'hPa'
                    var = var + ' ' + P_lev

                slope = d_Ensemble['ds_linregress'][clim_var + ' slope'].values * 10
                intercept = d_Ensemble['ds_linregress'][clim_var + ' intercept'].values * 1
                intercept_2017 = slope/10 * 2017 + intercept
                slope_error = d_Ensemble['ds_linregress'][clim_var + ' std_err'].values * 10
                r_squared_value = d_Ensemble['ds_linregress'][clim_var + ' r_value'].values ** 2
                p_value = d_Ensemble['ds_linregress'][clim_var + ' p_value'].values * 1
                # calculate trend for 10 decades
                linregress_writer.writerow([# only for myself
                                            var.replace('_', ' '),
                                            site.replace('_', ' '),
                                            forcing,
                                            # slope plus/minus std deviation
                                            str(round_significant(slope, significant_digits)) 
                                            + '$\pm$' + str(round_significant(slope_error, significant_digits)), 
                                            # intercept
                                            round_significant(intercept_2017, significant_digits),
                                            # r_value ** 2
                                            round_significant(r_squared_value, significant_digits),
                                            # p_value
                                            round_significant(p_value, significant_digits)])
                                            
                x = np.arange(2017,2051,1)
                color = inferno_colors[idx]
                if forcing == 'future':
                    linestyle = 'dotted'
                else:
                    linestyle = (0, (3, 1, 1, 1, 1, 1))

                ax.plot(x, slope/10*(x-2017) + intercept_2017, color=color, linestyle=linestyle,
                            label=site)# label = site + ' ' + var + ' ' + forcing + ', R^2=' + str(round_significant(r_squared_value, significant_digits)))
                lines_legend.append(Line2D([0], [0], linestyle = linestyle, color = color, label = site + ' ' + var + ' ' + forcing + ', R^2=' + str(round_significant(r_squared_value, significant_digits))))

    labelLines(plt.gca().get_lines(),zorder=2.5, fontsize = 5) # , xvals = [2015]*16

    ax.legend(handles = lines_legend,loc='lower left', bbox_to_anchor = (1.05, 0))
    ax.set_xlabel('time [years]')
    ax.set_ylabel(ylabel)
    fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/' + variable + '_Trendlines.pdf', bbox_inches='tight')
    fig.show()
            

def ds_monthly_PRIMAVERA_to_csv(d_future_trends, variable):
    # open site specific dictionary
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    # [0]: Mauna Kea
    # [1]: Cerro Paranal
    # [2]: La Silla
    # [3]: Cerro Tololo
    # [4]: La Palma
    # [5]: Siding Spring
    # [6]: Sutherland
    # [7]: SPM

    # loop through sites
    for idx in range(0, 8):

        site = d_site_lonlat_data['site_name_folder'][idx]
        print(site)

        d_future_site = d_future_trends[site] # select correct dictionary

        d_Ensemble = pickle.load(open('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + variable + '_d_Ensemble.pkl', "rb"))

        if 'clim_var' in d_future_site.keys() and d_future_site['clim_var'] != None:
            for clim_var in d_future_site['clim_var']:
                for P_lev in d_future_site['Plev']:
                    for forcing in ['hist', 'present' ,'future', 'SSTfuture']:
                        forced_clim_var = clim_var + ' ' + forcing + ' mean'

                        ds_temp = d_Ensemble['ds_ensemble_monthly_timeseries'][forced_clim_var].sel(level=P_lev).dropna(dim='time', how = 'all')
                        # rename variable to temp (for R)
                        ds_temp = ds_temp.rename('temp')
                        # save to csv
                        path_ds_ensemble_monthly_timeseries= '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/' + idx_to_site(idx) + '_' +  forcing + '_' + str(P_lev) + '_monthly_timeseries.csv'
                        ds_temp.dropna(dim='time').to_dataframe().to_csv(path_ds_ensemble_monthly_timeseries) # drop 2015,2016 and 2049, 2050


        if 'single_lev_var' in d_future_site.keys() and d_future_site['single_lev_var'] != None:
            for clim_var in d_future_site['single_lev_var']:
                for forcing in ['hist', 'present' ,'future', 'SSTfuture']:
                    forced_clim_var = clim_var + ' ' + forcing + ' mean'

                    ds_temp = d_Ensemble['ds_ensemble_monthly_timeseries'][forced_clim_var].dropna(dim='time', how = 'all')
                    # rename variable to temp (for R)
                    ds_temp = ds_temp.rename('temp')                    
                    # save to csv
                    path_ds_ensemble_monthly_timeseries = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/future_trends/' + idx_to_site(idx) + '_' +   forcing + '_single_level_monthly_timeseries.csv'
                    ds_temp.dropna(dim='time').to_dataframe().to_csv(path_ds_ensemble_monthly_timeseries) # drop 2015,2016 and 2049, 2050
    
    return

def ds_monthly_ERA5_to_csv(variable):
    # open site specific dictionary
    d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

    # [0]: Mauna Kea
    # [1]: Cerro Paranal
    # [2]: La Silla
    # [3]: Cerro Tololo
    # [4]: La Palma
    # [5]: Siding Spring
    # [6]: Sutherland
    # [7]: SPM

    # loop through sites
    for idx in range(0, 8):

        site = d_site_lonlat_data['site_name_folder'][idx]
        print(site)

        d_obs = pickle.load(open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site + '_' + variable + '_d_obs_ERA5_and_insitu.pkl', "rb"))

        if 'ERA5_var' in d_obs.keys() and d_obs['ERA5_var'] != None:
            for clim_var in d_obs['ERA5_var']:
                for P_lev in d_obs['Plev']:
                    
                    ds_temp = d_obs['ds_merged'][clim_var].sel(level=P_lev).dropna(dim='time', how = 'all')
                    # rename variable to temp (for R)
                    ds_temp = ds_temp.rename('temp')
                    # resample monthly
                    ds_temp = ds_temp.resample(time='1m').mean()
                    # save to csv
                    path_ERA5_monthly_timeseries= '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/ERA5_trends/' + idx_to_site(idx) + '_' + str(P_lev) + '_monthly_timeseries_ERA5.csv'
                    os.makedirs(os.path.dirname(path_ERA5_monthly_timeseries), exist_ok=True) 
                    ds_temp.dropna(dim='time').to_dataframe().to_csv(path_ERA5_monthly_timeseries) # drop 2015,2016 and 2049, 2050


        if 'single_lev' in d_obs.keys() and d_obs['single_lev'] != None:
            for clim_var in d_obs['single_lev']:
                    
                ds_temp = d_obs['ds_merged'][clim_var].dropna(dim='time', how = 'all')
                # rename variable to temp (for R)
                ds_temp = ds_temp.rename('temp')
                # resample monthly
                ds_temp = ds_temp.resample(time='1m').mean()
                # save to csv
                path_ERA5_monthly_timeseries= '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/ERA5_trends/' + idx_to_site(idx) + '_single_level_monthly_timeseries_ERA5.csv'
                os.makedirs(os.path.dirname(path_ERA5_monthly_timeseries), exist_ok=True) 
                ds_temp.dropna(dim='time').to_dataframe().to_csv(path_ERA5_monthly_timeseries) # drop 2015,2016 and 2049, 2050

    return

#%%
# d_obs and d_model are dictionaries, e.g. 
# d_obs = {"ds": ds_hourly, "insitu_var":['La_Palma Specific Humidity'], "ERA5_var": ['q'], "Plev": [700, 750, 775, 800, 850]}
#d_model = {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [700, 850]}

# for testing:
# diurnal = False
# time_slice_var = time_slice_var_meteo or time_slice_var_PWV
# site = site_name_folder
# SH_integral = False

def main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, 
                        diurnal=False, fig_diurnal=None, d_model = None, SH_integral=False, ERA5_path = None,
                         d_Ensemble=None, MasterFig = None, ax_ref = None, nighttime=False):
    
    # start clock to check performance
    start_time = time.time()

    # always keep attributes
    xr.set_options(keep_attrs=True)

    # if there is no in-situ data (e.g. precipitable water vapor), there is no d_obs['ds']
    if 'ds' in d_obs.keys():
        # diurnal cycle is only plotted if diurnal = True
        # if it is not plotted, you can resample monthly from the beginning, to speed up the code
        if diurnal==False:
            d_obs['ds'] = d_obs['ds'].resample(time = '1m', keep_attrs=True).mean()

        # check if there is already everything prepared in d_obs and d_model
        # assume that if ds_merged is there, the rest is there as well (for simplicity reasons)
    if 'ds_merged' not in d_obs.keys():
        print('ds_merged is not available, so I create it\n')

        # check if there is in-situ data available
        if 'ds' in d_obs.keys():
            load_ds = [d_obs['ds']]

        # check if insitu data is available (not often the case for PWV)
        if 'ds' not in d_obs.keys():
            # in this case, only load in ERA5 data
            # and start with an empty list
            load_ds = []       
            # if there is no insitu data (and only ERA5 data, select whole time range of ERA5 data)
            time_slice_var = slice('1979-01-01', '2020-01-01', None)    
            # later: think about case where there is also no ERA5 data 
        
        # read in ERA 5 data (data is stored for every site in the corresponding path)
        # the data for all sites in Chile is stored in folder structure of site 'Paranal'
        chile_grid = ['Tololo', 'Pachon', 'Silla']

        if any(x in site for x in chile_grid):
            site_ERA = 'Paranal'
        else:
            site_ERA = site

        if lon > 180:
            my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
            print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
        else:
            my_ERA5_lon = lon
        
        print(site_ERA)

        # get data for q_integrated (precipitable water vapor)
        # if SH_integral_pressure != None: # think of something else than SH_integral_pressure
        if SH_integral == True:
            if 'Plev' in d_obs.keys():
                for P_lev in d_obs['Plev']:
                    # function of SH_integral_pressure(SH_integral_pressure)
                    # ds_q_integrated = SH_integral_to_TCW(SH_integral_pressure, site_ERA, my_ERA5_lon, lat)
                    
                    # load already integrated datasets
                    if diurnal:
                        path_q_integrated = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/q_integrated/ds_ERA5_q_integrated_hourly_nearest_' + str(P_lev) + 'hPa.nc' # where files are stored
                    else:
                        path_q_integrated = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/q_integrated/ds_ERA5_q_integrated_monthly_resampled_nearest_' + str(P_lev) + 'hPa.nc' # where files are stored
            
                    ds_q_integrated = xr.open_dataset(path_q_integrated)
                    # print(ds_q_integrated)
                    if 'expver' in ds_q_integrated.dims:
                        ds_q_integrated = ds_q_integrated.sel(expver = 5)

                    # take nighttime values only, if nighttime==True
                    if nighttime:
                        print('nighttime selected')
                        # take ERA5 nighttime values (local), 19:00-7:00
                        if site == 'Paranal' or site == 'La_Silla' or site == 'Cerro_Tololo':
                            time_delta_to_UTC = 3
                            
                        elif site == 'MaunaKea':
                            time_delta_to_UTC = 10
                        
                        elif site == 'La_Palma':
                            time_delta_to_UTC = 0 # la palma belongs to spain, but local nighttime is rather in the timezone +0
                        
                        elif site == 'Sutherland':
                            time_delta_to_UTC = (-2)

                        elif site == 'siding_spring':
                            time_delta_to_UTC = (-10)

                        elif site == 'SPM':
                            time_delta_to_UTC = 6
                        
                        ds_q_integrated = ds_q_integrated.where((ds_q_integrated['time'].dt.hour <= (7 + time_delta_to_UTC)%24) | (ds_q_integrated['time'].dt.hour > (19 + time_delta_to_UTC)%24))
                        
                        print(ds_q_integrated)

                    load_ds.append(ds_q_integrated.load())

        elif variable == 'seeing_nc':
            if 'ERA5_var' in d_obs.keys():
                # check if there is the pressure level variable 'seeing' (integrated until ERA5 surface level)
                if 'seeing' in d_obs['ERA5_var']:
                    for P_lev in d_obs['Plev']:
                        path_seeing = './sites/'+ site + '/Data/Era_5/seeing_nc/ds_ERA5_seeing_hourly_nearest_' + str(P_lev) + 'hPa.nc' # exact name of file (because it could have more files in one folder)

                        ds_seeing = xr.open_dataset(path_seeing)

                        if nighttime:
                            # take ERA5 nighttime values (local), 19:00-7:00
                            if site == 'Paranal' or site == 'La_Silla' or site == 'Cerro_Tololo':
                                time_delta_to_UTC = 3
                                
                            elif site == 'MaunaKea':
                                time_delta_to_UTC = 10
                            
                            elif site == 'La_Palma':
                                time_delta_to_UTC = 0 # la palma belongs to spain, but local nighttime is rather in the timezone +0
                            
                            elif site == 'Sutherland':
                                time_delta_to_UTC = (-2)

                            elif site == 'siding_spring':
                                time_delta_to_UTC = (-10)

                            elif site == 'SPM':
                                time_delta_to_UTC = 6
                            
                            ds_seeing = ds_seeing.where((ds_seeing['time'].dt.hour <= (7 + time_delta_to_UTC)%24) | (ds_seeing['time'].dt.hour > (19 + time_delta_to_UTC)%24))
                            
                            print(ds_seeing)

                        load_ds.append(ds_seeing.load())

                        print('seeing loaded')


        else: # no SH_integral to load --> read in ERA5 data normally
            # get ERA5 data    
            # on pressure levels
            if 'Plev' in d_obs.keys(): # Plev is used to find corresponding folders
                ds_ERA5_Plev = read_ERA5_pr_level_data(site_ERA, d_obs['Plev'], variable, d_obs['ERA5_var'],  ERA5_path=ERA5_path) # append list of datasets that are merged together
                # select lon/lat here (be careful with 
                ds_ERA5_Plev = ds_ERA5_Plev.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
                
                # take nighttime values only, if nighttime==True
                if nighttime:
                    # take ERA5 nighttime values (local), 19:00-7:00
                    if site == 'Paranal' or site == 'La_Silla' or site == 'Cerro_Tololo':
                        time_delta_to_UTC = 3
                        
                    elif site == 'MaunaKea':
                        time_delta_to_UTC = 10
                    
                    elif site == 'La_Palma':
                        time_delta_to_UTC = 0 # la palma belongs to spain, but local nighttime is rather in the timezone +0
                    
                    elif site == 'Sutherland':
                        time_delta_to_UTC = (-2)

                    elif site == 'siding_spring':
                        time_delta_to_UTC = (-10)

                    elif site == 'SPM':
                        time_delta_to_UTC = 6
                       
                    ds_ERA5_Plev = ds_ERA5_Plev.where((ds_ERA5_Plev['time'].dt.hour <= (7 + time_delta_to_UTC)%24) | (ds_ERA5_Plev['time'].dt.hour > (19 + time_delta_to_UTC)%24))
                    
                    print(ds_ERA5_Plev)

                if diurnal==False: # resample monthly, if hourly data is not needed for diurnal cycle
                    ds_ERA5_Plev = ds_ERA5_Plev.resample(time = '1m', keep_attrs=True).mean()
                                
                load_ds.append(ds_ERA5_Plev.load()) # load dataset into memory for faster processing afterwards
            
        # on single levels
        if 'single_lev' in d_obs.keys():

            # calculated cloud cover, read in here
            if variable == 'calc_cloud_cover':
                print('calc cloud cover reading')
                ds_calc_cloud_cover = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/Era_5/calc_cloud_cover/calc_cloud_cover.nc')
                load_ds.append(ds_calc_cloud_cover.load())
                
            
            elif variable == 'seeing_nc':
                # check if there is a single lev var called 'wind speed seeing'
                # if so, load in files stored in: path = './sites/'+ site_name_folder + '/Data/Era_5/seeing_nc/ds_ERA5_200hPa_wind_speed_related_seeing_hourly_nearest.nc'
                if 'wind speed seeing' in d_obs['single_lev']:
                    path_wind_seeing = './sites/'+ site + '/Data/Era_5/seeing_nc/ds_ERA5_200hPa_wind_speed_related_seeing_hourly_nearest.nc'

                    ds_wind_seeing = xr.open_dataset(path_wind_seeing)

                    if nighttime:
                        # take ERA5 nighttime values (local), 19:00-7:00
                        if site == 'Paranal' or site == 'La_Silla' or site == 'Cerro_Tololo':
                            time_delta_to_UTC = 3
                            
                        elif site == 'MaunaKea':
                            time_delta_to_UTC = 10
                        
                        elif site == 'La_Palma':
                            time_delta_to_UTC = 0 # la palma belongs to spain, but local nighttime is rather in the timezone +0
                        
                        elif site == 'Sutherland':
                            time_delta_to_UTC = (-2)

                        elif site == 'siding_spring':
                            time_delta_to_UTC = (-10)

                        elif site == 'SPM':
                            time_delta_to_UTC = 6
                        
                        ds_wind_seeing = ds_wind_seeing.where((ds_wind_seeing['time'].dt.hour <= (7 + time_delta_to_UTC)%24) | (ds_wind_seeing['time'].dt.hour > (19 + time_delta_to_UTC)%24))
                        
                        print(ds_wind_seeing)

                    load_ds.append(ds_wind_seeing.load())

                elif 'surface_seeing' in d_obs['single_lev']:
                    path_surface_seeing = './sites/'+ site + '/Data/Era_5/seeing_nc/ds_surface_seeing.nc'

                    ds_surface_seeing = xr.open_dataset(path_surface_seeing)

                    load_ds.append(ds_surface_seeing.load())
                    print(ds_surface_seeing)


            else:
                ds_ERA5_Sglev = read_ERA5_sg_level_data(site_ERA, variable, d_obs['single_lev'])
                # select lon/lat (wrong; lon/lat are not -180to180 anymore)

                ds_ERA5_Sglev = ds_ERA5_Sglev.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')

                if nighttime:
                    # take ERA5 nighttime values (local), 19:00-7:00
                    if site == 'Paranal' or site == 'La_Silla' or site == 'Cerro_Tololo':
                        time_delta_to_UTC = 3
                        
                    elif site == 'MaunaKea':
                        time_delta_to_UTC = 10
                    
                    elif site == 'La_Palma':
                        time_delta_to_UTC = 0 # la palma belongs to spain, but local nighttime is rather in the timezone +0
                    
                    elif site == 'Sutherland':
                        time_delta_to_UTC = (-2)

                    elif site == 'siding_spring':
                        time_delta_to_UTC = (-10)

                    elif site == 'SPM':
                        time_delta_to_UTC = 6
                       
                    ds_ERA5_Sglev = ds_ERA5_Sglev.where((ds_ERA5_Sglev['time'].dt.hour <= (7 + time_delta_to_UTC)%24) | (ds_ERA5_Sglev['time'].dt.hour > (19 + time_delta_to_UTC)%24))
                    
                    print(ds_ERA5_Sglev)


                if diurnal==False: # resample monthly
                    ds_ERA5_Sglev = ds_ERA5_Sglev.resample(time = '1m', keep_attrs=True).mean()

                load_ds.append(ds_ERA5_Sglev.load()) # 

        # merge and slice in time, for cycles
        try:
            d_obs['ds'], d_obs['ds_merged'] = merge_ds_time_slicer(load_ds, time_slice = time_slice_var)
        except KeyError:
            # try to sort ds_hourly
            load_ds[0] = load_ds[0].sortby('time')
            d_obs['ds'], d_obs['ds_merged'] = merge_ds_time_slicer(load_ds, time_slice = time_slice_var)
        # now we have created d_obs['ds']


        # print("--- %s seconds for creating ds_merged ---" % (time.time() - start_time))
        # start_time = time.time()

    else:
        print('ds_merged is already created.')

    # prepare for taylor diagram
    d_obs['ds_sel'] = d_obs['ds'] # shallow copy is sufficient here 
    # already selected # not needed anymore: .xr_sel(d_obs, 'ds' , lon, lat, obs=True)


    ######## load in model data (extracted regions)
    if d_model != None:
        for clim_key in d_model.keys(): # go through all climate models
            print(time.time() - start_time)
            print(clim_key)

            if SH_integral == True:
                merge_model_plev = []
                for Plev in d_model[clim_key]['Plev']: # Plev gives us in this case the limit of integration; there can be more than one datasets
                    # load in 'q_integrated' dataset

                    path_q_integrated_model = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(Plev) + 'hPa.nc' # where files are stored
                    # # until surface pressure:
                    # path_q_integrated_model = '/home/haslebacher/chaldene/Astroclimate_Project//sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(Plev) + 'hPa_surface_pressure.nc' # where to save the files

                    ds_q_integrated_model = xr.open_dataset(path_q_integrated_model) 
                    merge_model_plev.append(ds_q_integrated_model)
                # combine datasets on dimension 'level'
                d_model[clim_key]['ds_Amon_Pr'] = xr.merge(merge_model_plev, join='outer') # 'outer' uses union of object indexes
                
                if 'single_lev_var' not in d_model[clim_key].keys():
                    print(clim_key)
                    print('create ds_Amon')
                    d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Pr']

            # pressure level data
            elif 'clim_var' in d_model[clim_key].keys() and d_model[clim_key]['clim_var'] != None:
                # function that loads in model data from .../chaldene/Astroclimate_Project/HighResMIP/variables/.../Amon/...
                d_model[clim_key]['ds_Amon_Pr'] = get_PRIMAVERA(d_model, clim_key, site, pressure_level=True)
                print(time.time() - start_time)

                if 'single_lev_var' not in d_model[clim_key].keys():
                # if there is no single level data, ds_Amon_Pr is equal to ds_Amon
                    d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Pr']
            

            # single level data
            if 'single_lev_var' in d_model[clim_key].keys():
                # function that loads in model data from .../chaldene/Astroclimate_Project/HighResMIP/variables/.../Amon/...
                d_model[clim_key]['ds_Amon_Sg'] = get_PRIMAVERA(d_model, clim_key, site, single_level=True)

                # if there is no pressure level data
                if 'clim_var' not in d_model[clim_key].keys():
                # if there is no pressure level data, ds_Amon_Sg is equal to ds_Amon
                    d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon_Sg']
                
            # case where we have single and pressure level data --> combine now into one dataset, called ds_Amon
            if 'clim_var' in d_model[clim_key].keys() and 'single_lev_var' in d_model[clim_key].keys():
                d_model[clim_key]['ds_Amon'] = xr.merge([d_model[clim_key]['ds_Amon_Pr'], d_model[clim_key]['ds_Amon_Sg']])


        print("--- %s seconds for getting PRIMAVERA data ---" % (time.time() - start_time))
        start_time = time.time()


    # select time slice of model data for diurnal/seasonal cycle
    if d_model != None:
        for clim_key in d_model.keys(): # go through all climate models
            if 'ds_sel' not in d_model[clim_key].keys(): # check if ds_sel is already there

                # select lon/lat 
                # use ds_sel for timeseries, where only lon/lat is selected (and not time)
                d_model[clim_key]['ds_sel'] = xr_sel(d_model[clim_key], 'ds_Amon', lon, lat) # pressure levels are selected with this function
  
                # load ds_sel into memory for faster processing (not really needed now, since regions are extracted)
                d_model[clim_key]['ds_sel'] = d_model[clim_key]['ds_sel'].load()
                # --> try to load at a later point (and load in only variable we need (and not e.g. lot_bnds))

            if 'ds_sel_cycle' not in d_model[clim_key].keys(): # check if ds_sel is already there
                
                try: # use ds_sel_cycle for diurnal cycle (this is now the only use left)
                    d_model[clim_key]['ds_sel_cycle'] = d_model[clim_key]['ds_sel'].sel(time = time_slice_var)
                except ValueError:
                    print('invalid time_slice_var for ds_model. This often happens if you pass the 31th day of a month. I am guessing now that you passed day 31 and I am changing that to day 30.')
                    try:
                        time_slice_var_cf = slice(time_slice_var.start ,time_slice_var.stop[:-1] + '0') # change 31 to 30
                        d_model[clim_key]['ds_sel_cycle'] = d_model[clim_key]['ds_sel'].sel(time = time_slice_var_cf)
                    except ValueError:
                        print('It seems that my guess is wrong. I just go on with the full timeframe.')
                        d_model[clim_key]['ds_sel_cycle'] = d_model[clim_key]['ds_sel']

            # print('ds_sel and ds_sel_cycle of climate model {} done.'.format(clim_key))
            # print("--- %s seconds for selecting lon/lat/Plev/time (and loading) of climate_model ---" % (time.time() - start_time))
            # start_time = time.time()

    # check size of time
    # if size is zero (meaning that the insitu data and data from current selected forcing (e.g. hist) have no intersecting time)
    # then, exclude from Taylor diagram and seasonal cycle
    # for now , assume that there is only one forcing

    temporary_None = False # keyword for knowing if d_model was None from beginning; 
    # can be changed in next code snippet

    # just take last clim_key. Time range is the same for all models (and same forcing)
    # STOP: ds_sel_cycle is not zero for models that have future and hist forcing. 
    # I was lucky that the last model ('ECMWF') only has hist members!!
    # change behaviour, so that we test the taylor-folder variables
    if d_model != None:
        clim_key_list = list(d_model.keys())
        # take first model (which is usually hadgem, which has all forcings in it)
        # (otherwise, it has at least the taylor folders in it)
        # check for 1 variable
        data_vars_list = list(d_model[clim_key_list[0]]['ds_sel_cycle'].data_vars)
        # but variable must be from 'taylor_folder' (take first taylor folder element)

        folder_var = d_model[clim_key_list[0]]['taylor_folder'][0]
        for dvar in data_vars_list:
            if folder_var in dvar:
                print(dvar)
                my_taylor_var = dvar
            else:
                print('folder_var not in dvar: {} not in {}'.format(folder_var, dvar))

        # drop nan in dataset (otherwise also not zero)
        # use my_taylor_var
        ds_check = d_model[clim_key_list[0]]['ds_sel_cycle'][my_taylor_var].dropna(dim='time', how='all')['time'] # how='all' deletes time only, if all data (on all pressure levels) are nan (important e.g. for siding spring, RH)

        if ds_check.size == 0:
        # or: if time_slice_var.start or time_slice_var.end in taylor-forcings
            Intersected_time_taylor = time_intersect(d_obs, d_model=None)
            # make deep copy of d_model dictionary, so that d_model can be set to None, and later changed again do the dict
            d_model_copy = copy.deepcopy(d_model)
            d_model = None
            print('WARNING: d_model is now None, because time intersection is zero.')
            # we also need a keyword that tells us that d_model was not None from the very
            # beginning of the function!
            temporary_None = True
            

        # for seasonal cycle (and taylor diagram)
        else:
            Intersected_time_taylor = time_intersect(d_obs, d_model)
            # note: d_obs gets resampled monthly in 'time_intersect'
    else: # 
        Intersected_time_taylor = time_intersect(d_obs)


    # create datasets with selected 'Intersected_time_taylor' as time
    # d_obs['ds_taylor'] is already created, and can be used
    # select Intersected time for d_obs

    d_obs['ds_taylor'] = d_obs['ds_taylor'].sel(time=Intersected_time_taylor)
    # # line above (resampling) did not work because of Intersected_time_taylor also had pressure level
    # if 'insitu_var' in d_obs.keys():
    #     for insitu_var in d_obs['insitu_var']:
    #         d_obs['ds_taylor'][insitu_var] = d_obs['ds_taylor'][insitu_var].sel(time=Intersected_time_taylor)
    # if 'ERA5_var' in d_obs.keys():
    #     for clim_var in d_obs['ERA5_var']:
    #         print(clim_var)
    #         for Plev in d_obs['Plev']:
    #             d_obs['ds_taylor'][clim_var] = d_obs['ds_taylor'][clim_var].sel(level = Plev, time=Intersected_time_taylor)
    # if 'single_lev' in d_obs.keys() :
    #     for clim_var in d_obs['single_lev']:
    #         d_obs['ds_taylor'][clim_var] = d_obs['ds_taylor'][clim_var].sel(time=Intersected_time_taylor)

    # select Intersected time for climate models
    if d_model != None:
        for clim_key in d_model.keys(): # go through all climate models
            if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                d_model[clim_key]['ds_taylor'] = d_model[clim_key]['ds_taylor'].sel(time=Intersected_time_taylor)
    # use these datasets for ds_mean_month and taylor diagram

    # prepare for diurnal/seasonal cycle
    if diurnal == False: # group only monthly if the option 'diurnal' is False
        frequencies =  ['time.month']
    else:
        frequencies =  ['time.hour', 'time.month']

    for time_freq_string in frequencies: # loop through frequencies
        # group observational dict
        if time_freq_string == 'time.hour':
            # take ds_sel for diurnal cycle (not resampled)
            d_obs = group_mean(time_freq_string, d_obs, d_obs['ds_sel'], std=True)
        elif time_freq_string == 'time.month':
            # take ds_taylor for seasonal cycle
            d_obs = group_mean(time_freq_string, d_obs, d_obs['ds_taylor'], std=True)
        
        # group model data
        if d_model != None:
            for clim_key in d_model.keys(): # go through all climate models
                # the function 'group_mean' returns the whole dict, 
                # with new entries according to the grouping
                if time_freq_string == 'time.hour':
                    d_model[clim_key] = group_mean(time_freq_string, d_model[clim_key], d_model[clim_key]['ds_sel_cycle'], std=False)
                elif time_freq_string == 'time.month':
                    # if bool(d_model[clim_key]['taylor_folder']) == True: # if list is not empty
                    
                    # use d_model[clim_key]['ds_taylor']
                    d_model[clim_key] = group_mean(time_freq_string, d_model[clim_key], d_model[clim_key]['ds_taylor'], std=False)
                    # print('d_model ds_mean_month')
                    if d_Ensemble != None: 
                        # then, group also ds_sel to monthly data 
                        # (and take only forcings that are not already plotted with ds_taylor)
                        d_model[clim_key]['ds_mean_month_comparison'], d_model[clim_key]['ds_std_month_comparison'] = group_mean(time_freq_string, d_model[clim_key], d_model[clim_key]['ds_sel'], std=True, return_mean_ds=True)


    # prepare for timeseries (apply a time slice to data, and group)
    time_freq_string = 'time.year'
    # ds_mean_year for d_obs:
    # I need the ERA5 data since 1979 until 2019 December, and the observational data since time_slice_var.start
    d_obs = group_mean(time_freq_string, d_obs, xr_sel(d_obs, 'ds_merged' ,lon, lat, obs=True).resample(time='m').mean() , std=True) # for the timeseries, take ds_merged, which includes whole range of ERA5 data. BUT: you need to resample monthly for a realistic std dev!!!
    
    # now is the time to set d_model again equal to the dict, if no time intersection
    if d_model == None and temporary_None == True:
        # if d_model_copy[clim_key]['ds_sel_cycle']['time'].size == 0: # not needed, right? (2020-08-13)
        del(d_model)
        d_model = d_model_copy

    # ds_mean_year for d_model: (and preparation for map)
    if d_model != None:
        for clim_key in d_model.keys(): # go through all climate models
            d_model[clim_key] = group_mean(time_freq_string, d_model[clim_key], d_model[clim_key]['ds_sel'], std=False)

    # ( ########### 5-year rolling mean (ds_rough_trend) )

    print(lon)
    # show seasonal cycle as well as timeseries and taylor diagram in one plot, (optional: diurnal cycle)
    fig, sorted_skill_dict, ax0, ax3 = xr_plot_cycles_timeseries(d_obs, site, variable, lon, lat, d_model, 
                                                            diurnal=diurnal, fig_diurnal=fig_diurnal, d_Ensemble=d_Ensemble, 
                                                            MasterFig=MasterFig, ax_ref = ax_ref)

    # return dictionary because of loaded datasets, check when loading, if it already exists


    return d_obs, d_model, fig, sorted_skill_dict, ax0, ax3

# %%
# test_model = d_model[clim_key]['ds_sel'].resample(time='1m').mean()
# test_obs = d_obs['ds_sel']
# # both datasets are resampled
# # try to merge
# xr.merge([test_obs, test_model], compat='override') # join='outer', compat = 'no_conflicts'
# # would work!! but lon, lat cannot be conserved

# %%