# 2020-08-29
# this function integrates the specific humidity 
# from the model data sets 
# and saves them into a separate netcdf file

#%%

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr

import xarray.plot as xplt

from matplotlib import dates as d
import datetime as dt
import time

from itertools import cycle
from functools import reduce
from scipy import stats

import csv

import math

import seaborn as sns
sns.set()
from matplotlib import cycler

import pickle

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

#%% global parameters

# list_of_single_level_vars = ['seeing']
# variable = 'seeing_nc'
# # list_of_clim_vars = ['seeing']
# list_of_model_clim_params = ['seeing']
# # list_of_single_level_model_clim_params = ['seeing']


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




#%% seeing

# # load in orography and get elevation from there (for integral end)
# oro_ERA5 = xr.open_dataset('/home/haslebacher/chaldene/cds_data_ERA5/single_level/Era5_orography_singleLevel.nc') # 0-360
# oro_ERA5 = oro_ERA5.z_0001/9.81 # divide geopotential by gravitational acceleration
# oro_ERA5 = oro_ERA5.sel(time = '2019-03-01')



# ERA5, seeing

ls_calibration = []

importlib.reload(climxa)

for idx in range(0, 8): # !!!!! check !!

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
    ds_seeing_vars = ds_seeing_vars.load() # load here for faster performance

    # WE NEED TO INVERSE Also the pressure levels in the array
    ds_seeing_vars = ds_seeing_vars.sortby('level', ascending=False)

    # TAKE MODEL SURFACE ELEVATION (surface pressure)!!!
    path_surf_pr = './sites/' + site_ERA + '/Data/Era_5/surface_pressure/*.nc'
    surf_pr_ERA5 = xr.open_mfdataset(path_surf_pr, combine = 'by_coords')

    # note: here we need to take 'lon' and not 'my_ERA5_lon'
    my_ERA5_elevation = surf_pr_ERA5.sel(longitude = my_ERA5_lon, latitude=lat, method='nearest')
    my_ERA5_elevation_mean = my_ERA5_elevation.sp.mean(dim='time').values/100 # to hPa
    print('ERA5 model surface pressure: {}'.format(my_ERA5_elevation_mean))


    # get closest matching ERA5 pressure level (to mean of ERA5 surface pressure)
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    given_value = my_ERA5_elevation_mean
    if isinstance(my_ERA5_elevation_mean, (list, tuple, np.ndarray)) == True:
        given_value = my_ERA5_elevation_mean[0]

    closest_value = min(pr_levels_ERA5, key=absolute_difference_function)
    my_seeing_indx = pr_levels_ERA5.index(closest_value)
    print('closest value: {}'.format(closest_value))
    
    # if the index is the maximal index, make it minus one, because the integration is centred Euler forward
    # if len(pr_levels_ERA5) == len(pr_levels_ERA5[:(my_seeing_indx+1)]):
    #     my_seeing_indx = my_seeing_indx - 1
    # COMMENTED because it is not needed anymore, since we integrate until 50hPa for all sites

    #### calculate seeing mean value for calibration
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
        # use mean of other calib factors
        # A = 5 * 10^(-16)
        # k = 6

        mean_insitu = 1.32 # from Optical turbulence characterization at the SAAO Sutherland site (L. Catala)

    else:
        # read in ds_hourly (in-situ data)
        # ds_hourly = climxa.df_to_xarray('./sites/Paranal/Data/    # attention! taylor folders can change! think about that in the return...in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
        ds_hourly = climxa.df_to_xarray(path_seeing)
    
        mean_insitu = np.mean(ds_hourly[list_of_insitu_vars[0]])



    #########
    # MAIN: calculate seeing, integrate until model surface pressure level
    ds_seeing_integrated, ds_Cn2_profile, calib_factor = climxa.ERA5_seeing_calc(ds_seeing_vars, mean_insitu, pr_levels_ERA5[my_seeing_indx:]) # CHANGED from: pr_levels_ERA5[:(my_seeing_indx+1)]
    ds_seeing_wind, calib_factor_wind = climxa.ERA5_seeing_wind(ds_seeing_vars, mean_insitu)


    # save calib_factor for writing to file
    ls_calibration.append(f'Site: {site_name_folder}, calibration_factor for seeing model: {np.around(calib_factor, 3)}, for 200hPa-wind-speed seeing: {climxa.as_si(calib_factor_wind, 3)}')

    # add attributes
    ds_seeing_integrated['seeing'].attrs['long_name'] = 'Seeing'
    ds_seeing_integrated['seeing'].attrs['units'] = 'arcsec' # arcseconds
    ds_seeing_integrated['seeing'].attrs['calibration_factor'] = calib_factor


    ds_seeing_wind['wind speed seeing'].attrs['long_name'] = '200hPa-Wind Speed related Seeing'
    ds_seeing_wind['wind speed seeing'].attrs['units'] = 'arcsec'
    ds_seeing_wind['wind speed seeing'].attrs['calibration_factor'] = calib_factor_wind


    ds_Cn2_profile["Cn2"].attrs['long_name'] = 'Cn2'
    ds_Cn2_profile["Cn2"].attrs['units'] = 'm^(1/3)'

    # path for Cn2 profile
    path_Cn2 = './sites/'+ site_name_folder + '/Data/Era_5/Cn2/ds_ERA5_Cn2_profile_hourly_nearest.nc'
    os.makedirs(os.path.dirname(path_Cn2), exist_ok=True)  
    ds_Cn2_profile.to_netcdf(path_Cn2)

    # add coordinate 'level'
    ds_seeing_integrated['seeing'] = ds_seeing_integrated['seeing'].assign_coords(level=closest_value)
    # add dimension 'level'
    ds_seeing_integrated['seeing'] = ds_seeing_integrated['seeing'].expand_dims(dim='level')
    # now, the level can be selected with e.g. ds.sel(level=775)

    # define path for Osborn seeing
    path = './sites/'+ site_name_folder + '/Data/Era_5/seeing_nc/ds_ERA5_seeing_hourly_nearest_' + str(closest_value) + 'hPa.nc' # where to save the files
    # make directory if not available
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    # save array to netcdf file, store it 
    ds_seeing_integrated.to_netcdf(path)


    # define path for wind related seeing
    path = './sites/'+ site_name_folder + '/Data/Era_5/seeing_nc/ds_ERA5_200hPa_wind_speed_related_seeing_hourly_nearest.nc' # where to save the files
    # make directory if not available
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    # save array to netcdf file, store it 
    ds_seeing_wind.to_netcdf(path)

# calculate Cn2 profile!!

# save calibration factors
df = pd.DataFrame(ls_calibration)
df.to_csv('./Model_evaluation/seeing_nc/calibration_factors_ERA5.csv')


#%% correct 'level' to the one in the filename
# you can run it as many times as you wish

for idx in range(0, 8): # !!!!! check !!

    print(d_site_lonlat_data['site_name'][idx])

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_level_seeing'][idx]

    # load nc file
    path = './sites/'+ site_name_folder + '/Data/Era_5/seeing_nc/ds_ERA5_seeing_hourly_nearest_' + str(ls_pr_levels_ERA5[0]) + 'hPa.nc'


    with xr.open_dataset(path) as ds:
        data_ds = ds.load()

    print(data_ds.level)
    if data_ds.level == 100:
        print(idx)
        # drop 'level'
        data_ds = data_ds.reset_index('level', drop = True)
        data_ds = data_ds.squeeze('level') # ds_re.reset_coords(drop=True)

        data_ds = data_ds.assign_coords(level=ls_pr_levels_ERA5[0])
        ds_seeing_integrated = data_ds.expand_dims(dim='level')
        # now, the level can be selected with e.g. ds.sel(level=775

        print(ds_seeing_integrated.level)

        # load, otherwise we cannot overwrite
        ds_seeing_integrated = ds_seeing_integrated.load()

        # save again
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        ds_seeing_integrated.to_netcdf(path, mode='w')



#%%
# quickly load wind seeing
# ds_wind = xr.open_dataset(path)


#%%
# # for free seeing, integrate only up to an altitude of 3.6km

# # # first, find pressure level which is nearest to 3600m
# # for p_index in range(0, len(pr_levels)):
# #   if (ds_full.z[p_index]/g).mean() < 3600:
# #     print('the value of altitude which is smaller than 3600m is {}'.format((ds_full.z[p_index]/g).mean().values))
# #     print('the index is {}'.format(p_index))
# #     break

# #%% wind direction (for Paranal)

# # read ERA5 single level wind speed data
# site = 'Paranal'
# variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']
# lon = -70.40
# lat = -24.63

# ls_merge = []

# for variable in variables: 
#     ERA5_path = './sites/Chile/Data/Era_5/' + variable + '/*.nc'
#     ds_temp = climxa.read_ERA5_sg_level_data(site, variable, variable, ERA5_path=None)

#     ls_merge.append(ds_temp)

# ds_sg_wind = xr.merge(ls_merge)

# # select lon and lat
# ds_sg_wind = ds_sg_wind.sel(longitude= lon,latitude= lat ,method='nearest')

# ds_sg_wind = ds_sg_wind.load()

# #%%
# # resample monthly to speed up computation
# ds_sg_wind = ds_sg_wind.resample(time='m').mean()

# # ds_sg_wind['alpha'] = [math.atan(u/v) for u, v in zip(ds_sg_wind['u10'], ds_sg_wind['v10'])]

# # wind direction is in radians (minus pi/2 and pi/2)

# # if u > 0, v > 0 --> wind dir = alpha
# # if u < 0, v > 0 --> wind dir = 180°-alpha = pi - alpha
# # if u > 0, v < 0 --> wind dir = 360° - alpha = 2pi - alpha
# # if u < 0, v < 0 --> wind dir = 180° + alpha = pi + aloha

# # ds_wind_direction = [math.atan(u/v) for u, v in zip(ds_sg_wind['u10'], ds_sg_wind['v10']) if ]
# #%%
# ls_wind_dir = []
# ls_time = []

# for idx, (u, v) in enumerate(zip(ds_sg_wind['u10'], ds_sg_wind['v10'])):
#     alpha = math.atan(u/v)
#     if u >= 0 and v >= 0:
#         ls_wind_dir.append(alpha)
#         ls_time.append(ds_sg_wind['u10'].time[idx].values)
#     elif u <= 0 and v > 0:
#         ls_wind_dir.append(np.pi - alpha)
#         ls_time.append(ds_sg_wind['u10'].time[idx].values)
#     elif u >= 0 and v < 0:
#         ls_wind_dir.append(2*np.pi - alpha)
#         ls_time.append(ds_sg_wind['u10'].time[idx].values)
#     elif u <= 0 and v < 0:
#         ls_wind_dir.append(np.pi + alpha)
#         #ds_sg_wind['wind direction'] = np.pi + alpha
#         ls_time.append(ds_sg_wind['u10'].time[idx].values)

# # compose dataset ???!!???

# xr.DataArray({"wind direction": np.array(ls_wind_dir)}, dims='time', coords={'time': np.array(ls_time)} )

# # ds_wind_direction = [math.atan(u/v) for u, v in zip(ds_sg_wind['u10'], ds_sg_wind['v10']) if u >= 0 and v >= 0]

# plt.plot(ls_time, ls_wind_dir)

# #%% group yearly:

# ds_sg_wind_yearly = ds_sg_wind.resample(time='y').mean()

# ls_wind_dir_yearly = []
# ls_time_yearly = []

# for idx, (u, v) in enumerate(zip(ds_sg_wind_yearly['u10'], ds_sg_wind_yearly['v10'])):
#     alpha = math.atan(u/v)
#     if u >= 0 and v >= 0:
#         ls_wind_dir_yearly.append(alpha)
#         ls_time_yearly.append(ds_sg_wind_yearly['u10'].time[idx].values)
#     elif u <= 0 and v > 0:
#         ls_wind_dir_yearly.append(np.pi - alpha)
#         ls_time_yearly.append(ds_sg_wind_yearly['u10'].time[idx].values)
#     elif u >= 0 and v < 0:
#         ls_wind_dir_yearly.append(2*np.pi - alpha)
#         ls_time_yearly.append(ds_sg_wind_yearly['u10'].time[idx].values)
#     elif u <= 0 and v < 0:
#         ls_wind_dir_yearly.append(np.pi + alpha)
#         #ds_sg_wind['wind direction'] = np.pi + alpha
#         ls_time_yearly.append(ds_sg_wind_yearly['u10'].time[idx].values)

# plt.plot(ls_time_yearly, ls_wind_dir_yearly)

#%%

# # climate models: SH integral

# importlib.reload(climxa)


# pr_levels_model = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# # CNRM:     100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# # CMCC:     100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100 
# # EC-Earth: 100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100 
# # ECMWF:    100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# # HadGEM:   100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# # MPI:      100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100
# # --> all the same!


# list_of_model_clim_params = ['hus']
# list_of_single_level_model_clim_params = None
# ls_pr_levels_clim_model = pr_levels_model # load in all pressure levels
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
# d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

# # EC-Earth, 'present' --> but only leads to errors (I think data download was issue) --> run again without 'present'
# d_model = {'EC-Earth': {"folders": ['hist', 'future'], "taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"}}

# for idx in range(0, 8):

#     # idx = 5

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]
#     site = d_site_lonlat_data['site_name_folder'][idx]

#     surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
#     print(surface_pressure_observation)
#     # check available pressure for climate models 
#     absolute_difference_function = lambda list_value : abs(list_value - given_value)

#     given_value = surface_pressure_observation
#     closest_value = min(pr_levels_model, key=absolute_difference_function)

#     S_integral_pressure = closest_value # find nearest pressure

#     print('closest match: {}'.format(S_integral_pressure))


#     for clim_key in d_model.keys():

#         # use function which loads in all specific humidity datasets 
#         d_model[clim_key]['ds_hus'] = climxa.get_PRIMAVERA(d_model, clim_key, site, pressure_level=True)

#         # select lon/lat
#         d_model[clim_key]['ds_sel'] = climxa.xr_sel(d_model[clim_key], 'ds_hus', lon, lat)

#         # find maximal index 
#         pr_max_idx = pr_levels_model.index(S_integral_pressure)

#         # initialize dataset
#         ds_seeing_integrated = xr.Dataset()

#         # integrate
#         for forc_idx, forcing in enumerate(d_model[clim_key]['folders']):
#             summe = climxa.Euler_centred_PWV(d_model[clim_key]['ds_sel'], pr_max_idx, 'hus ' + forcing)

#             # write integral to dataset
#             ds_seeing_integrated["seeing " + forcing] = summe

#             # add attributes
#             ds_seeing_integrated["seeing " + forcing].attrs['long_name'] = 'Precipitable Water Vapor'
#             ds_seeing_integrated["seeing " + forcing].attrs['units'] = 'mmH2O' # = 1kg/m^2

#             # add coordinate 'level'
#             if forc_idx == 0:
#                 ds_seeing_integrated = ds_seeing_integrated.drop('level') # drop level, otherwise it is not overwritten properly sometimes
            
#             ds_seeing_integrated["seeing " + forcing] = ds_seeing_integrated["seeing " + forcing].assign_coords(level=S_integral_pressure)
#             # add dimension 'level'
#             ds_seeing_integrated["seeing " + forcing] = ds_seeing_integrated["seeing " + forcing].expand_dims(dim='level')
#             # now, the level can be selected with e.g. ds.sel(level=775)

#         # define path
#         path = './sites/'+ site + '/Data/HighResMIP/seeing/Amon/' + clim_key + '/ds_' + clim_key + '_seeing_monthly_resampled_nearest_' + site + '_' + str(S_integral_pressure) + 'hPa.nc' # where to save the files

#         # make directory if not available
#         os.makedirs(os.path.dirname(path), exist_ok=True) 

#         # save array to netcdf file, store it 
#         ds_seeing_integrated.to_netcdf(path)

#%%