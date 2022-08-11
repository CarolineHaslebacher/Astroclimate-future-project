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




#%% get something to work with

idx = 7 # SPM

site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

d_model = get_PRIMAVERA_seeing_vars(idx)

#### ERA5
# load data
median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'
ds_ERA5_median = xr.open_dataset(median_path).load()

# divide ERA5 by 10
ds_ERA5_median['z'] = ds_ERA5_median['z']/10 

pr_levels_list = [850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
ds_sel_ERA5 = ds_ERA5_median.sel(level= pr_levels_list)



#%% polyfit

# y =ds_ERA5_median.u
# x = ds_ERA5_median.level



def seeing_vars_polyfit(x_in, y_in,deg=5, num=100):

    coefficients = np.polyfit(x_in, y_in, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    return poly, new_x, new_y

# ds_sel_ERA5
# y =ds_sel_ERA5.v
# x = ds_sel_ERA5.level

# empty dict for functions
d_polyfits = {}

# for y in [ds_sel_ERA5.u, ds_sel_ERA5.v, ds_sel_ERA5.t, ds_sel_ERA5.z]:
for y_var in ['u', 'v', 't', 'level']:

    x = ds_sel_ERA5.z
    y = ds_sel_ERA5[y_var]

    # not needed!
    # if y_var == 'level':
    #     # then it is crucial to reverse the order of the Plevs, 
    #     # otherwise we have the same problem!
    #     y = y.sortby('level', ascending=False)

    
    poly, new_x, new_y = seeing_vars_polyfit(x, y, deg=5, num=100)

    # calculate the derivative
    # polyderiv = poly(np.polyder(poly)) # this is just wrong!
    polyderiv = np.polyder(poly)
    der_y = polyderiv(new_x)

    d_polyfits[y_var] = {}
    d_polyfits[y_var]['poly'] = poly
    d_polyfits[y_var]['100z'] = new_x # aber wir wollen ableitung nach dz
    d_polyfits[y_var]['100vars'] = new_y
    d_polyfits[y_var]['deriv'] = der_y

#%%
# for xarray with many time entries
x = ds_sel_ERA5.z[:,3000:4000]
y = ds_sel_ERA5[y_var][:,3000:4000]

xr.apply_ufunc(seeing_vars_polyfit, x, y, input_core_dims=[['level'], ['level']], 
                                     output_core_dims=['poly', 'new_x', 'new_y'], vectorize=True)

### we can groupby level, do the calculation
# from https://stackoverflow.com/questions/38960903/applying-numpy-polyfit-to-xarray-dataset


#%%
# give back dataset

# for testing
x_in = ds_sel_ERA5.z
y_in = ds_sel_ERA5[y_var]

def myRMSE(df_P, df_O):
    # P for predictions
    # O for observations
    return(np.sqrt(np.sum((df_P - df_O)**2)/len(df_P)))
    
def seeing_vars_polyfit(x_in, y_in,deg=5, num=100):

    coefficients = np.polyfit(x_in, y_in, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)

    # also calc derivative here
    polyderiv = np.polyder(poly)
    der_y = polyderiv(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    # for each time, we want to have the new_x, new_y and the der_y
    # 100 values for each of them
    # data_vars = {variable_name: (dimension name, data_array)}
    # coords = {name: array}
    ds = xr.Dataset(data_vars={y_var: ('poly_z', new_y), y_var + '_deriv': ('poly_z', der_y)}, coords={'poly_z': new_x}) #, new_y, der_y} , 

    # I do not need the 'poly' polynomial coefficients!

    # calculate residuals between polynomial and our data points
    calc_y = poly(x_in)

    plt.plot(x_in, y_in)
    plt.plot(x_in, calc_y)

    residuals = y_in - calc_y
    RMSE = myRMSE(calc_y, y_in)
    # RMSE

    # add as attribute
    ds[y_var] = ds[y_var].assign_attrs({'RMSE': RMSE.values})

    return ds


#%% plot some 

y_var = 'u'

plt.plot(d_polyfits[y_var]['100z'], d_polyfits[y_var]['100vars'])
plt.plot(d_polyfits[y_var]['100z'], d_polyfits[y_var]['deriv'])

plt.plot(d_polyfits[y_var]['100vars'], d_polyfits[y_var]['100z'])

# plev versus z: CHECK
plt.plot(d_polyfits['level']['100z'], d_polyfits['level']['100vars'])
plt.gca().invert_yaxis()



#%%

# P = d_polyfits['t']['100z'] # does not matter of what!
P = d_polyfits['level']['100vars']
T = d_polyfits['t']['100vars']

RCp = 0.286 # R/cp
P_0 = 1000 #mbar

Theta = T*(P_0 /P)**(RCp)
# plt.plot(d_polyfits['t']['100z'], Theta)
# !! dtheta nach dz (und nicht dp)
dTheta_man = d_polyfits['t']['deriv'] * (P_0 /P)**(RCp) - RCp * d_polyfits['level']['deriv'] * T *1/P *(P_0 /P)**(RCp)
dTheta_wolf = 1/P * ( (P_0/P )**(RCp) * (-RCp * T * d_polyfits['level']['deriv'] + P * d_polyfits['t']['deriv']) )

# plt.plot( d_polyfits['t']['100z'], dTheta_man)
# plt.plot( d_polyfits['t']['100z'], dTheta_wolf)

###### poly of Theta for derivative
x = ds_sel_ERA5.z
y = ds_sel_ERA5['t'] * (P_0 / ds_sel_ERA5.level)**RCp
# try:  ds_sel_ERA5['level'].sortby('level', ascending=False)

# plt.plot(x,y)
# plt.plot(x, ds_sel_ERA5.level)

poly2, new_x, new_y = seeing_vars_polyfit(x, y, deg=5, num=100)

# plt.plot(x, y, "o", new_x, new_y)

# calculate the derivative
polyderiv = np.polyder(poly2)
der_y = polyderiv(new_x)

# plt.plot( d_polyfits['t']['100z'], dTheta_wolf)
# plt.plot(new_x, der_y) 
# --> calculating theta out of the 11 values and then calculate the polynomial
# or do the derivative by hand and put in the 100vars derived from the polynomial is almost equal!


#####

dTheta_dz = der_y

# test dTheta/dz
th0 = 303.28925207 # or: new_y[0]
th1 = 304.17272728
z0 = 1479.23059082 # or: new_x[0]
z1 = 1668.53565193

test_dTheta_dz = (th1-th0)/(z1-z0) # = 0.0046669391976090135
# but dTheta_dz[0] = 296.3536778452241 # before I removed 'poly' before np.polyder()
# now, dTheta_dz[0] = 0.004683882856482026

# append to dict
y_var = 'Theta'
d_polyfits[y_var] = {}
d_polyfits[y_var]['poly'] = poly
d_polyfits[y_var]['100z'] = new_x # aber wir wollen ableitung nach dz
d_polyfits[y_var]['100vars'] = new_y
d_polyfits[y_var]['deriv'] = der_y

k_var = 1
g = 9.80665

dudz = d_polyfits['u']['deriv']
dvdz = d_polyfits['v']['deriv']
E = dudz**2 + dvdz**2
Theta_var = new_y # d_polyfits['Theta']['100vars']
Theta_deriv = der_y # d_polyfits['Theta']['deriv']


Cn2_poly = (80*10**(-6) * P / (T * Theta_var))**2 * k_var * (2 * E / (g/Theta_var * Theta_deriv))**(2/3) * Theta_deriv**2

J_trapz = np.trapz(Cn2_poly, x=d_polyfits['level']['100z'])

#%%

# plt.plot(d_polyfits['level']['100z'], d_polyfits['level']['100vars'])

# plot Cn2 profile
plt.plot(Cn2_poly, d_polyfits['level']['100z'])

# or with pressure as basis
# STOP: you can do that only if function was linear!
# No no, these values we have here are 'not evenly spaced', for Plevs!
# but evenly spaced for z
plt.plot(Cn2_poly, d_polyfits['level']['100vars']) 
plt.gca().invert_yaxis()
# first, change basis

# but why minimum at 200hPa??
# we have maximum wind speed there...
plt.plot(E, d_polyfits['level']['100vars'] ) 
# E has minimum at 200! because the derivative is exactly zero!! turning point.

plt.plot(dudz, d_polyfits['level']['100vars'] )

plt.plot(d_polyfits['u']['100vars'], d_polyfits['level']['100vars'])

# NOTE: the osborn formula is not depending on the wind speed, but of the derivative of the wind speed!!!


#%% integral for J

# J = np.polyint(Cn2_poly) # 'antiderivative' polynom
# plt.plot(J[:100],d_polyfits['level']['100z'])
# # to pressure
# plt.plot(J[:100],d_polyfits['level']['100vars'])
# plt.gca().invert_yaxis()

# use numpy trapezoidal rule
J_trapz = np.trapz(Cn2_poly, x=d_polyfits['level']['100z']) # integrate over dz
# = 7.721706859470486e-15
# right between if I integrate from 50 to 700 and if I integrate from 850 to 100!

#%% find a solution to integrate xarray
# but anyway, there would only be one loop over time. might be okey!
# no! it is taking too long

# Test here if derivative after level would work
# du/dz = du/dP * 1/ dz/dP

def seeing_vars_polyfit(x_in, y_in,deg=5, num=100):

    coefficients = np.polyfit(x_in, y_in, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    return poly, new_x, new_y

# ds_sel_ERA5
# y =ds_sel_ERA5.v
# x = ds_sel_ERA5.level

# empty dict for functions
d_polyfits = {}

# for y in [ds_sel_ERA5.u, ds_sel_ERA5.v, ds_sel_ERA5.t, ds_sel_ERA5.z]:
for y_var in ['u', 'v', 't', 'z']:

    x = ds_sel_ERA5.level
    y = ds_sel_ERA5[y_var]

    # not needed!
    # if y_var == 'level':
    #     # then it is crucial to reverse the order of the Plevs, 
    #     # otherwise we have the same problem!
    #     y = y.sortby('level', ascending=False)

    
    poly, new_x, new_y = seeing_vars_polyfit(x, y, deg=5, num=100)

    # calculate the derivative
    # polyderiv = poly(np.polyder(poly)) # this is just wrong!
    polyderiv = np.polyder(poly)
    der_y = polyderiv(new_x)

    d_polyfits[y_var] = {}
    d_polyfits[y_var]['poly'] = poly
    d_polyfits[y_var]['100Plevs'] = new_x # aber wir wollen ableitung nach dz
    d_polyfits[y_var]['100vars'] = new_y
    d_polyfits[y_var]['deriv'] = der_y

# do calculations

k_var = 1
g = 9.80665


#### theta

RCp = 0.286 # R/cp
P_0 = 1000 #mbar

P = d_polyfits['t']['100Plevs']
T = d_polyfits['t']['100vars']

Theta = T*(P_0 /P)**(RCp)
# dTheta_dP = - T * 1/P *(P_0/P)**(RCp) # 1/P * ( (P_0/P )**(RCp) * (-RCp * T * d_polyfits['level']['deriv'] + P * d_polyfits['t']['deriv']) )

# dTheta_dz = dTheta_dP * 1/d_polyfits['z']['deriv']


##### use partial derivatives
#  dTheta/dz = de Theta/ de P * 1/ de z/de P + de Theta/de T * de T/de z

deTheta_deP = - T * 1/P *(P_0/P)**(RCp) # 1/P * ( (P_0/P )**(RCp) * (-RCp * T * d_polyfits['level']['deriv'] + P * d_polyfits['t']['deriv']) )
deTheta_deT = (P_0/P)**(RCp)

d_Theta_dz_mitdP = deTheta_deP * 1/d_polyfits['z']['deriv'] + deTheta_deT * d_polyfits['t']['deriv'] * 1/d_polyfits['z']['deriv']

plt.plot(Theta, d_polyfits['z']['100Plevs'])
plt.plot(d_Theta_dz_mitdP, d_polyfits['z']['100vars'])

######

dudz = d_polyfits['u']['deriv']*1/d_polyfits['z']['deriv'] # du/dP*1/(dP/dz)
dvdz = d_polyfits['v']['deriv']*1/d_polyfits['z']['deriv']
E = dudz**2 + dvdz**2

Theta_var = Theta # d_polyfits['Theta']['100vars']
Theta_deriv = d_Theta_dz_mitdP # d_polyfits['Theta']['deriv']


Cn2_poly = (80*10**(-6) * P / (T * Theta_var))**2 * k_var * (2 * E / (g/Theta_var * Theta_deriv))**(2/3) * Theta_deriv**2

J_trapz_dP = np.trapz(Cn2_poly * d_polyfits['z']['deriv'], x=d_polyfits['u']['100Plevs'])
# 1.36e-13
# J_trapz_dP = np.trapz(Cn2_poly, x=d_polyfits['z']['100vars'])
# = 1.806e-13

# plt.plot(E, d_polyfits['u']['100Plevs'] ) 
# # E has minimum at 200! because the derivative is exactly zero!! turning point.

# plt.plot(dudz, d_polyfits['u']['100Plevs'] )


#%%

# load one ds_full
idx = 7

print(d_site_lonlat_data['site_name'][idx])
# lon_obs and lat_obs are in 0-360 format!!
lon = d_site_lonlat_data['lon_obs'][idx]
lat = d_site_lonlat_data['lat_obs'][idx]
site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
print(surface_pressure_observation)
# check available pressure for ERA5 
absolute_difference_function = lambda list_value : abs(list_value - given_value)
pr_levels_ERA5 = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000] # siding spring is the upper limit with 892hPa
# 2021-02-22: inverse pressure levels to include a big ground layer!
pr_levels_ERA5.reverse() # NEW!!!

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
# ds_seeing_vars = ds_seeing_vars.load() # load here for faster performance

# WE NEED TO INVERSE Also the pressure levels in the array
ds_seeing_vars = ds_seeing_vars.sortby('level', ascending=False)

#%%
import copy

ds_full = copy.deepcopy(ds_seeing_vars)

#%% testing polyfit with ds_full

def myRMSE(df_P, df_O):
    # P for predictions
    # O for observations
    return(np.sqrt(np.sum((df_P - df_O)**2)/len(df_P)))
    
def seeing_vars_polyfit_xr(x_in, y_in, y_var, time_idx, deg=5, num=100):

    # calculate polynomial out of input variables
    coefficients = np.polyfit(x_in, y_in, deg=deg)
    poly = np.poly1d(coefficients)

    # define new basis (geopot height)
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    # retrieve the y-values based on the polynomial fit
    new_y = poly(new_x)

    # also calc derivative here
    polyderiv = np.polyder(poly)
    der_y = polyderiv(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    # put output into a dataset
    # for each time, we want to have the new_x, new_y and the der_y
    # 100 values for each of them
    # data_vars = {variable_name: (dimension name, data_array)}
    # coords = {name: array}
    ds = xr.Dataset(data_vars={y_var: ('poly_z', new_y), y_var + '_deriv': ('poly_z', der_y)}, coords={'poly_z': new_x, 'time': time_idx}) #, new_y, der_y} , 
    # ds = xr.Dataset(data_vars={y_var: ('time', new_y), y_var + '_deriv': ('time', der_y), 'poly_z': ('time', new_x)}, coords={'time': time_idx}) #, new_y, der_y} , 

    # I do not need the 'poly' polynomial coefficients!

    # calculate residuals between polynomial and our data points
    # calc_y are the values we get for our 11 or 19 pressure levels/geopot heights (x_in)
    calc_y = poly(x_in)

    # plt.plot(x_in, y_in)
    # plt.plot(x_in, calc_y)

    residuals = y_in - calc_y
    RMSE = myRMSE(calc_y, y_in)
    # RMSE

    # add RMSE as attribute
    ds[y_var] = ds[y_var].assign_attrs({'RMSE': RMSE.values})

    return ds

#%% loop



# it is too slow! 10 time indices need 45 seconds!!
# resample monthly first!
# or, what if I load it into memory first?
# then, 10 time indices need 0.399 seconds!


import time
start_time = time.time() # measure elapsed time

# ls_con = []

for time_idx in ds_full.time[:10]:

    # save to dict
    dp = {}

    for y_var in ['u', 'v', 't', 'level']:

        # print(y_var)

        # our base is the geopotential height z
        # select a 1d-array (select time)
        x_in = ds_full.z.sel(time = time_idx)

        # select our y values, select time if possible
        if y_var == 'level':
            # has no time coordinate!
            y_in = ds_full[y_var]
        else:
            y_in = ds_full[y_var].sel(time = time_idx)

        ds = seeing_vars_polyfit_xr(x_in, y_in, y_var, time_idx)

        # ls_con.append(ds)
        # ls_time.append(time_idx)

        # I want to make a new dataset where each element of the list corresponds to one point in time
        # ds_timeline = xr.Dataset(ls_con, ls_time)
        dp[y_var] = xr.concat(ls_con, dim='time')

    # for every time index, compute Cn2 and J

#%%

def seeing_vars_polyfit(x_in, y_in,deg=5, num=100):

    coefficients = np.polyfit(x_in, y_in, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x_in[0], x_in[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)

    # plt.plot(x, y, "o", new_x, new_y)

    return poly, new_x, new_y


start_time = time.time()

# define constants
RCp = 0.286 # R/cp
P_0 = 1000 #mbar

ls_J_poly = []
ls_time_idx = [] # just to make sure we take the correct time index
ls_Cn2 = []

# resample monthly!
ds_full_resampled = ds_full.resample(time = '1m', keep_attrs=True).mean()
ds_full_2019 = ds_full_resampled.sel(time=slice(None,'2019-12-31'))

Cn2_array = np.zeros((len(ds_full_2019.time), 100))

counter = 0

for count_idx, time_idx in enumerate(ds_full_2019.time):

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

#%% save and do comparison in xarray seeing directly

#%%
# for testing output
plt.plot(ds_Cn2_conc['Cn2_poly'].isel(time=3), ds_Cn2_conc.Plevs)
plt.gca().invert_yaxis()
plt.xscale('log')

plt.plot(ds_Cn2_full['Cn2_poly'].isel(time=3), ds_Cn2_conc.Plevs)
plt.gca().invert_yaxis()
plt.xscale('log')

#%%
# MAYBE new_x is not the same? --> yes!!
# there are many nan values after concatenating
# --> Calculate Cn2 right in every loop



#%%


ds_full.resample(time = '1m', keep_attrs=True).mean()

#%% 

x_multi = ds_full.level
y_multi = ds_full.u

poly_multi = np.polynomial.polynomial.polyfit(x_multi, y_multi, 5)












#%%
# I did that in a separate file!

# for idx in range(0,8):

#     if idx == 1:
#         # already saved
#         continue

#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

#     # read data
#     ds_u_v_t = get_seeing_variables(idx, d_site_lonlat_data)

#     # calculate time median and store! (load next time)
#     ds_median = ds_u_v_t.median(dim='time')

#     # save to netcdf
#     median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'

#     ds_median.to_netcdf(median_path)

#%%


    # # load
    # median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'

    # ds_median_load = xr.open_dataset(median_path).load()



#%%

# fig = plot_seeing_vars_vertical_profile(variable, list_of_clim_vars, list_of_model_clim_params, my_xlabel)

# # save fig
# fig.savefig('./Model_evaluation/'+ variable + '/All_Sites' + variable + '_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

# # fig.savefig('./publication/figures/All_Sites_SH_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

# plt.show()

#%% only plot after loading in data

# fig = plt.figure(figsize = (8, 20),constrained_layout=True) # (this is not compatible with tight_layout)

# gs = fig.add_gridspec(4, 2)

# ax = fig.add_subplot(gs[int((idx - (idx%2))/2), idx%2])

# pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
# pr_levels_list.reverse() # exactly opposite to seeing, how it is saved!
# lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
# line_list = []

# for forcing in d_Ensemble['folders']:

#     # choose the right linestyle
#     if forcing=='hist':
#         forced_linestyle = lin_st[0]
#         color = 'navy'
#     elif forcing=='present':
#         forced_linestyle = lin_st[1]
#         color = 'navy'
#     elif forcing=='future':
#         forced_linestyle = lin_st[2]
#         color = 'red'
    
#     ax.plot(d_Ensemble['hus ' + forcing], pr_levels_list[0:(len(d_Ensemble['hus ' + forcing]))], 
#                     linestyle=forced_linestyle, color=color)

#     # for legend
#     line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = color, label = 'PRIMAVERA specific humidity ' + forcing))



# ax.plot(ds_ERA5_SH['q'].median(dim='time'), ds_ERA5_SH.level, linestyle = '-', color = '#009e73') # goes until 50hPa (glad I included it until 50hPa!)

# # for legend
# line_list.append(Line2D([0], [0], linestyle = '-', color = '#009e73', label = 'ERA5 specific humidity'))



# ax.set_xscale('log')
# ax.set_yticks(np.arange(0, 1000, 100)) # pressure levels
# plt.gca().invert_yaxis()
# ax.set_xlabel('Specific humidity [kg/kg]')
# ax.set_ylabel('Pressure [hPa]')
# ax.set_title(site_name_folder)
# plt.show()

#%%

# to find out why there is a peak at 70hPa (18641.822 m), look at raw data (ds_full from 'xarray_prepare_seeing_data_PRIMAVERA')
# plot             T_clim_var = 'ta ' + forcing
            # U_clim_var = 'ua ' + forcing
            # V_clim_var = 'va ' + forcing
            # Z_clim_var = 'zg ' + forcing
# plt.plot(ds_full['ua hist'].median(dim='time'), ds_full['ua hist'].level)
# plt.plot(ds_full['va hist'].median(dim='time'), ds_full['va hist'].level)
# plt.plot(ds_full['ta hist'].median(dim='time'), ds_full['ta hist'].level)
# plt.plot(ds_full['zg hist'].median(dim='time'), ds_full['zg hist'].level)

# 200hPa ~= 12401.16m

# # vertical profile
# plt.plot(ds_Cn2_profile["Cn2"].median(dim='time'), ds_Cn2_profile["Cn2"].level)
# plt.gca().invert_yaxis()
# # --> plot looks strange, peak at 150hPa!?
# # write function that plots all vertical profiles!


# # plot
# ds_median_tcw = ds_tcw_profile.median(dim = 'time')
# ds_median_tcw['q_integrated hist'].plot.line(y='level', label = 'hist')
# ds_median_tcw['q_integrated present'].plot.line(y='level', label = 'SST present')
# ds_median_tcw['q_integrated future'].plot.line(y='level', label = 'future')
# ds_median_tcw['q_integrated SSTfuture'].plot.line(y='level', label = 'SST future')
# plt.gca().invert_yaxis()
# plt.xlabel('precipitable water vapor (integral of specific humidity) [mmH2O]')
# plt.title('vertical profile of precipitable water vapor (Cerro Tololo; CMCC hist)')
# plt.legend()
# plt.savefig('./Model_evaluation/' + variable + '/' + site_name_folder + '_' + variable + '_CMCC_vertical_integral.png', dpi=400)

# %% ERA5

# # loop through sites
# for idx in range(0, 8):
#     # or define index for one iteration only
#     # idx = 0

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]

#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

#     list_of_single_level_vars = ['Cn2']

#     # define d_obs for reading in data
#     d_obs = {"single_lev": list_of_single_level_vars}

#     # read in ds_Cn2 for every model
#     d_model[clim_key]['ds_Cn2'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)

#     # plot vertical profile (median)
#     plt.plot(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].median(dim='time'), d_model[clim_key]['ds_Cn2'].level)
#     plt.xscale('log')
    
#     plt.gca().invert_yaxis()
#     plt.xlabel('Cn2 [m^{1/3}]')
#     plt.ylabel('Pressure [hPa]')


