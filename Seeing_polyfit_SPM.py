# this script plots the vertical profile of the astronomical seeing


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
import copy
 
from matplotlib.lines import Line2D

import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa
import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


#%% RELOAD CLIMXA

#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file

import climxa

import importlib
importlib.reload(climxa)

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')


#%% load in site specific data

# [0]: Mauna Kea
# [1]: Cerro Paranal
# [2]: La Silla
# [3]: Cerro Tololo
# [4]: La Palma
# [5]: Siding Spring
# [6]: Sutherland
# [7]: SPM

# d_site_lonlat_data = pd.read_csv('/home/haslebacher/chaldene/Astroclimate_Project/Sites_lon_lat_ele_data.csv')
d_site_lonlat_data = pickle.load( open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))


#%%
# read in seeing vars (t, u, v)
# only ERA5!
def get_seeing_variables(idx, d_site_lonlat_data):

    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    chile_grid = ['Tololo', 'Pachon', 'Silla']
    if any(x in site_name_folder for x in chile_grid):
        site_ERA = 'Paranal'
    else:
        site_ERA = site_name_folder

    if lon > 180:
        my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
        print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
    else:
        my_ERA5_lon = lon

    # use function which loads in all specific humidity datasets 
    # and integrates them to specific humidity

    if site_ERA == 'Paranal': # Paranal
        seeing_data_path =  './sites/' + site_ERA + '/Era5_data/seeing/'
    else:
        seeing_data_path =  './sites/' + site_ERA + '/Data/Era_5/seeing/'

    ds_seeing_vars = climxa.read_ERA5_seeing_data(seeing_data_path, my_ERA5_lon, lat)
    ds_seeing_vars = ds_seeing_vars.load() # load here to prevent program from running 

    # WE NEED TO INVERSE Also the pressure levels in the array
    ds_seeing_vars = ds_seeing_vars.sortby('level', ascending=False)

    return ds_seeing_vars

#%%
# PRIMAVERA
def get_PRIMAVERA_seeing_vars(idx):
    
    # or define index for one iteration only
    # idx = 0

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    # for calibration of the wind speed seeing and the osborn seeing, read in in situ data
    list_of_insitu_vars = ['Seeing ' + site_name_folder]

    # pressure level of seeing integration
    path_seeing = d_site_lonlat_data['path_ds_seeing'][idx] 

    # special case of siding_spring, where we have only yearly data:
    if idx == 5:
        df_siding_spring = pd.read_csv(path_seeing, index_col='year', delimiter='\t')
        ds_siding_spring = df_siding_spring.to_xarray()
        mean_insitu = np.mean(ds_siding_spring['ds_mean_year'])


    elif idx == 6: # sutherland: no insitu data

        mean_insitu = 1.32 # from Optical turbulence characterization at the SAAO Sutherland site (L. Catala)

    else:
        # read in ds_hourly (in-situ data)
        # ds_hourly = climxa.df_to_xarray('./sites/Paranal/Data/    # attention! taylor folders can change! think about that in the return...in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
        ds_hourly = climxa.df_to_xarray(path_seeing)
    
        mean_insitu = np.mean(ds_hourly[list_of_insitu_vars[0]])

    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                    "EC-Earth": {"folders": ['hist', 'future'], 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'],  'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'], 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

    for clim_key in d_model.keys():
        start_time = time.time() # measure elapsed time

        print('reading data for {}, {}'.format(clim_key, site_name_folder))
        # do not forget forcing --> get_PRIMAVERA does that automatically
        ls_seeing_vars = []
        for seeing_var in ['ua', 'va', 'zg', 'ta']:
            # read model data
            d_model[clim_key]['clim_var'] = [seeing_var]
            # append dataset to list so that all seeing variables are in one dataset
            ds_temp = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, pressure_level=True)
            
            # ds_notfind = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/La_Palma/Data/HighResMIP/ua/Amon/CNRM/hist_notfind.nc')
            # ds = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/La_Palma/Data/HighResMIP/ua/Amon/CNRM/hist.nc')
            # ds.sel(lon=lon, lat=lat, method='nearest').dropna(dim='time')

            # problem: ta and zg do not have the same lon/lat values than ua and va!!!

            # --> select lon and lat and drop lon,lat coords
            ds_temp = ds_temp.sel(longitude=lon, latitude=lat, method='nearest')
            # print(ds_temp[seeing_var + ' hist'].dropna(dim='time'))

            if seeing_var == 'ta':
                ds_temp = ds_temp + 273.15 # to Kelvin
            # drop
            ds_temp = ds_temp.reset_coords(drop=True)
            # print(ds_temp)

            ls_seeing_vars.append(ds_temp)

        # compose dataset with ua, va and zg which we need to calculate the seeing 
        # for this, nothing gets computed!!
        d_model[clim_key]['ds_seeing'] = xr.merge(ls_seeing_vars, join='outer')

    return d_model


#%%
def seeing_vars_polyfit(x_in, y_in,deg=5, num=100):

    coefficients = np.polyfit(x_in, y_in, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    return poly, new_x, new_y


def polyfit_seeing_integral(ds_full):

    start_time = time.time()

    # define constants
    RCp = 0.286 # R/cp
    P_0 = 1000 #mbar

    ls_J_poly = []
    ls_time_idx = [] # just to make sure we take the correct time index
    ls_Cn2 = []


    Cn2_array = np.zeros((len(ds_full.time), 100))

    counter = 0

    for count_idx, time_idx in enumerate(ds_full.time):

        # save to dict
        d_polyfits = {}

        for y_var in ['u', 'v', 't', 'level']:

            # our base is the geopotential height z
            # select a 1d-array (select time)
            x_in = ds_full.z.sel(time = time_idx)

            if y_var == 'level':
                # has no time coordinate!
                y_in = ds_full[y_var]
            else:
                y_in = ds_full[y_var].sel(time = time_idx)
            
            poly, new_x, new_y = seeing_vars_polyfit(x_in, y_in, deg=5, num=100)

            # calculate the derivative
            # polyderiv = poly(np.polyder(poly)) # this is just wrong!
            polyderiv = np.polyder(poly)
            der_y = polyderiv(new_x)

            d_polyfits[y_var] = {}
            d_polyfits[y_var]['poly'] = poly
            d_polyfits[y_var]['100z'] = new_x # aber wir wollen ableitung nach dz
            d_polyfits[y_var]['100vars'] = new_y
            d_polyfits[y_var]['deriv'] = der_y

        # calc Theta
        P = d_polyfits['level']['100vars']
        T = d_polyfits['t']['100vars']

        Theta_var = T*(P_0 /P)**(RCp)
        Theta_deriv = 1/P * ( (P_0/P )**(RCp) * (-RCp * T * d_polyfits['level']['deriv'] + P * d_polyfits['t']['deriv']) )

        # calculate Cn2
        k_var = 1
        g = 9.80665

        dudz = d_polyfits['u']['deriv']
        dvdz = d_polyfits['v']['deriv']
        E = dudz**2 + dvdz**2

        Cn2_poly = (80*10**(-6) * P / (T * Theta_var))**2 * k_var * (2 * E / (g/Theta_var * Theta_deriv))**(2/3) * Theta_deriv**2
        # Cn2_poly_forxr = np.expand_dims(Cn2_poly, axis=1)
        
        Cn2_array[count_idx] = Cn2_poly

        # Cn2_ls.append([Cn2_poly, time_idx.values, d_polyfits['level']['100vars']]) 
        # np.array([[[Cn2_poly]], [[time_idx.values]], [[d_polyfits['level']['100vars']]]]])

        J_trapz = np.trapz(Cn2_poly, x=d_polyfits['level']['100z'])

        # save to xarray dataset
        # take pressure level as basis, since this is always the same
        # ds_Cn2_poly = xr.Dataset(data_vars={'Cn2_poly': (('Plevs', 'time'), Cn2_poly_forxr)}, coords={'Plevs': d_polyfits['level']['100vars'], 'time': np.atleast_1d(time_idx)})
        # ds_Cn2_poly = xr.Dataset(data_vars={'Cn2_poly': Cn2_poly}, coords={'time': np.atleast_1d(time_idx)})
        
        # ds = xr.Dataset(data_vars={y_var: ('time', new_y), y_var + '_deriv': ('time', der_y), 'poly_z': ('time', new_x)}, coords={'time': time_idx}) #, new_y, der_y} , 
        # ds_J_poly = xr.Dataset({'J_trapz' : (time_idx, J_trapz)})
        # merge with other Cn2's along level

        # if counter == 0:
        #     ds_Cn2_conc = ds_Cn2_poly
        #     counter = 1
        # else:
        #     ds_Cn2_conc = xr.concat([ds_Cn2_conc, ds_Cn2_poly], 'Plevs')

        ls_J_poly.append(J_trapz)
        ls_time_idx.append(time_idx.values)

        # ls_Cn2.append(ds_Cn2_poly)
        

    # compose 
    ds_J_poly = xr.Dataset(data_vars = {'J_poly': ('time', ls_J_poly)}, coords={'time': ls_time_idx})

    # concat ls_Cn2
    # doesn't work as well because Plevs also have minimal differences!
    # shit, I can't do it. We get 1000 Plevs!
    # ds_Cn2_full = xr.merge(ls_Cn2, compat='override')
    ds_Cn2_full = xr.Dataset(data_vars={'Cn2_poly': (('time','Plevs'), Cn2_array)}, coords={'time': np.array(ls_time_idx), 'Plevs': d_polyfits['level']['100vars']})


    print("--- %s seconds ---" % (time.time() - start_time))

    # 0.20 seconds if we concat in each loop
    # 0.18 seconds if we append to list and concat afterwards
    # 0.15 seconds if we append lists also for Cn2

    # for 1000 iterations: 14 seconds
    # --> for 361798 = 5065 seconds = 1.4 hours!!!
    # --> resample first monthly!
    # 8 seconds!

    return ds_J_poly, ds_Cn2_full



#%% get something to work with

idx = 7 # SPM

site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

d_model = get_PRIMAVERA_seeing_vars(idx)

ds_full = get_seeing_variables(idx, d_site_lonlat_data)

# resample monthly!
ds_full_resampled = ds_full.resample(time = '1m', keep_attrs=True).mean()
ds_full_2019 = ds_full_resampled.sel(time=slice(None,'2019-12-31'))


#%% polyfit

# calibration factors
calib_ERA5 = 2.92
calib_PRIMAVERA = 5

#%%

# get J_poly and Cn2_poly back
ds_J_poly_ERA5, ds_Cn2_full_ERA5 =  polyfit_seeing_integral(ds_full_2019)

# PRIMAVERA: for every clim_key, ...
# or just for HadGEM?

ds_J_poly_PRIMAVERA, ds_Cn2_full_PRIMAVERA =  polyfit_seeing_integral(ds_full_2019)




#%% save and do comparison in xarray seeing directly

# ERA5
# calculate epsilon
ds_epsilon =  xr.Dataset({"seeing": calib_ERA5 * climxa.epsilon(ds_J_poly_ERA5['J_poly'])})


#%%

# define path for polyfit osborn seeing
path = './sites/'+ site_name_folder + '/Data/Era_5/seeing_pf/ds_ERA5_seeing_monthly_polyfit_integral.nc' # where to save the files
# make directory if not available
os.makedirs(os.path.dirname(path), exist_ok=True) 
# save array to netcdf file, store it 
ds_J_poly_ERA5.to_netcdf(path)

#%%

# PRIMAVERA
# but which folder? J_poly?
path = './sites/'+ site + '/Data/HighResMIP/J_poly/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files

# make directory if not available
# if os.path.exists(path):
#     os.remove(path)
# else:
os.makedirs(os.path.dirname(path), exist_ok=True) 

ds_seeing.to_netcdf(path)



