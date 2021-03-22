# 2020-08-29
# this function reads in the ERA5 and PRIMAVERA model data 
# and calculates the seeing values out of u,v,z and t
# it stores it in a netcdf file
# (code snippets from 'Seeing_nested_for_loop_simplify.py')

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
 
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa
import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


#%% RELOAD CLIMXA
#%%
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


#%% ERA5

# # PARANAL (already done)
# longitude= -70.25
# latitude= -24.75

# # use functions
# pr_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875]
# ds_paranal = climxa.read_ERA5_seeing_data(pr_levels, './sites/Paranal/Era5_data/seeing/', longitude, latitude)
# ds_seeing = climxa.ERA5_seeing_calc(ds_paranal, pr_levels[:-2])

# # save dataset
# ds_seeing.to_netcdf('...path...')


#%% function for defining the model surface pressure level



#%% climate models
# ingest data to get ds_full (including all relevant variables)

# # investigate error with la palma
# clim_key = 'CNRM'
# idx = 4
# integration until 1000hPa could be the problem. Maybe there are too many nans?
# I think I remember that I changed somewhen that I set the integration max to 925 in case that it is 1000hPa.
# try this again! --> yes!!!!

importlib.reload(climxa)


ls_calibration = []

for idx in range(0, 8): #
    
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
        # use mean of other calib factors
        # A = 5 * 10^(-16)
        # k = 6

        mean_insitu = 1.32 # from Optical turbulence characterization at the SAAO Sutherland site (L. Catala)

    else:
        # read in ds_hourly (in-situ data)
        # ds_hourly = climxa.df_to_xarray('./sites/Paranal/Data/    # attention! taylor folders can change! think about that in the return...in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
        ds_hourly = climxa.df_to_xarray(path_seeing)
    
        mean_insitu = np.mean(ds_hourly[list_of_insitu_vars[0]])


    #d_model = {"HadGEM": {"folders": ['present'],"taylor_folder": ['present'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"}}

    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                    "EC-Earth": {"folders": ['hist', 'future'], 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'],  'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'], 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }
    
    # # investigation of CNRM error for la palma
    # d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
    #                 "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'],  'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"}}
    
    # d_model['CNRM']['clim_var'] = ['ua']
    # ua_cnrm = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, pressure_level=True)
    # ua_cnrm_sel = ua_cnrm.sel(longitude=lon, latitude=lat, level=1000, method='nearest')
    # ua_cnrm_sel = ua_cnrm.sel(longitude=lon, latitude=lat, level=200, method='nearest')
    # ua_cnrm_sel['ua hist'].dropna(dim='time')
    # --> the 1000hPa dataset forces the other data values to drop
    # in other words, only the 1000hPa has a lot of missing values because it is so close to the surface pressure which also varies!

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

        ############## second part: calculate seeing
        print('calculating dataset now.')
        # define variables for function
        ds_full = d_model[clim_key]['ds_seeing'].load()
        # calc length of pr_levels_list with closest pressure level
        pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
        # NEW: INVERSE LIST HERE
        pr_levels_list.reverse() # NEW!!! we integrate from bottom of model surface to top of atmosphere

 
        site = site_name_folder        

        PRIMAVERA_surface_pressure_value = climxa.get_PRIMAVERA_surface_pressure_level(clim_key, site_name_folder, lon, lat)
        ls_idx = pr_levels_list.index(PRIMAVERA_surface_pressure_value) # + 1 # not needed anymore!!
        # update pr_levels_list
        if ls_idx == 0: # if idx == 19 and pressure ≃ 1000 (la palma) --> integrate only until 925hPa since too many entries are nan for 1000hPa
            # take one away from idx
            ls_idx = ls_idx + 1 # update: from -1 to + 1
        # pr_levels_list = pr_levels_list[:ls_idx] # error in first iteration: integrated all until 925hPa
        
        # correct:
        pr_levels_list = pr_levels_list[ls_idx:]

        # loop through forcings
        for forcing in d_model[clim_key]['folders']:

            print(forcing)

            T_clim_var = 'ta ' + forcing
            U_clim_var = 'ua ' + forcing
            V_clim_var = 'va ' + forcing
            Z_clim_var = 'zg ' + forcing

            # importlib.reload(climxa)

            # compute seeing (saved in PRIMAVERA_calc_seeing)
            k_factor = climxa.PRIMAVERA_calc_seeing(ds_full, mean_insitu, lon, lat, T_clim_var, U_clim_var, V_clim_var, Z_clim_var, pr_levels_list, site, clim_key, forcing, PRIMAVERA_surface_pressure_value)
            # wind_speed_seeing
            idx_200hPa = list(ds_full.level).index(200)
            ds_seeing_wind, calib_factor_wind = climxa.ERA5_seeing_wind(ds_full, mean_insitu, PRIMAVERA = True, U_clim_var=U_clim_var, V_clim_var=V_clim_var, idx=idx_200hPa)
            
            print('k_factor = {}, calib factor = {}'.format(k_factor, calib_factor_wind))
            # save calib_factor for writing to file
            ls_calibration.append(f'Site: {site_name_folder}, clim_key: {clim_key}, forcing: {forcing}, calibration_factor for seeing model: {np.around(k_factor, 3)}, for 200hPa-wind-speed seeing: {climxa.as_si(calib_factor_wind, 3)}')

            
            # rename variable from 'wind speed seeing' to 'wind_speed_seeing'
            ds_seeing_wind = ds_seeing_wind.rename_vars({'wind speed seeing': 'wind_speed_seeing'})

            # save wind_speed seeing
            ds_seeing_wind['wind_speed_seeing'].attrs['long_name'] = '200hPa-Wind Speed related Seeing'
            ds_seeing_wind['wind_speed_seeing'].attrs['units'] = 'arcsec'
            ds_seeing_wind['wind_speed_seeing'].attrs['calibration_factor'] = calib_factor_wind

            # define path where to save the wind_speed_seeing
            path = './sites/'+ site + '/Data/HighResMIP/wind_speed_seeing/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files
            # make directory if not available
            # if os.path.exists(path):
            #     print('path exists, delete file')
            #     os.remove(path)
            # else:
            os.makedirs(os.path.dirname(path), exist_ok=True) 
                # print('make directory')
            
            # save to netcdf
            ds_seeing_wind.to_netcdf(path)

        print("--- %s seconds ---" % (time.time() - start_time))

# save calibration factors
df = pd.DataFrame(ls_calibration)
df.to_csv('./Model_evaluation/seeing_nc/calibration_factors_PRIMAVERA.csv')

#%% get pressure levels to know where we integrated to

# for idx in range(0, 8): #

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]
#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
#     ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]

#     d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
#                     "EC-Earth": {"folders": ['hist', 'future'], 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
#                     "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'],  'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
#                     "MPI": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
#                     "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'], 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
#                     "ECMWF": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }
    
#     for clim_key in d_model.keys():
#         PRIMAVERA_surface_pressure_value = climxa.get_PRIMAVERA_surface_pressure_level(clim_key, site_name_folder, lon, lat)
#         print(f'clim model: {clim_key}')
#         print(f'surface pressure value: {PRIMAVERA_surface_pressure_value}')

#     print('\n')

#%% plotting

# # plot to compare to old dataset and ERA5
# plt.plot(ds_seeing_wind['wind speed seeing'].time, ds_seeing_wind['wind speed seeing']) 
# plt.plot(d_model['MPI']['ds_Amon']['wind_speed_seeing hist'].time, d_model['MPI']['ds_Amon']['wind_speed_seeing hist'])

# # plot some data to see influence of calibration parameters
# plt.plot(ds_seeing_wind['wind_speed_seeing'].time, ds_seeing_wind['wind_speed_seeing'])
# plt.plot(ds_seeing.time, ds_seeing['seeing'].sel(level=925, longitude=204.5, latitude=19.82, method='nearest'))

# # vertical profile
# plt.plot(ds_Cn2_profile["Cn2"].mean(dim='time'), ds_Cn2_profile["Cn2"].level)
# plt.gca().invert_yaxis()
# # --> plot looks strange, peak at 150hPa!?
# # write function that plots all vertical profiles!

#%%
# # try out ERA5_seeing_calc
# ds_full = d_model[clim_key]['ds_seeing']
# # below not needed, if I change function to .isel[level=i]
# # ds_full = ds_full.transpose('level', 'time') # so that ds_full[var][i] selects the level

# # ds_check = ds_full.where(xr.ufuncs.isnan(ds_full['ta present']) != True, drop = True)

# pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
# # pr_levels_list = [850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]

# # also find closest match with surface pressure level...!!!???!!!???
# # use ps data

# forcing = 'present' # CHANGE later, when more data is available

# T_clim_var = 'ta '
# U_clim_var = 'ua '
# V_clim_var = 'va '
# Z_clim_var = 'zg '

# # for seeing value, integrate Cn2 over all pressure levels
# # pr_levels_list = pr_levels[:-2]
# def PRIMAVERA_calc_seeing(ds_full, lon, lat, T_clim_var, U_clim_var, V_clim_var, Z_clim_var, pr_levels_list, site):
#     J = 0

#     for i in range(0, len(pr_levels_list)-1):
#         print(pr_levels_list[i])
#         T = ds_full[T_clim_var].sel(level=pr_levels_list[i])
#         P = ds_full.level[i]
#         u_i0 = ds_full[U_clim_var].sel(level=pr_levels_list[i])
#         u_i1 = ds_full[U_clim_var].sel(level=pr_levels_list[i+1])
#         v_i0 = ds_full[V_clim_var].sel(level=pr_levels_list[i]) 
#         v_i1 = ds_full[V_clim_var].sel(level=pr_levels_list[i+1])
#         T1 = ds_full[T_clim_var].sel(level=pr_levels_list[i+1])
#         P1 = ds_full.level[int(i+1)] 
#         df_z_0 = ds_full[Z_clim_var].sel(level=pr_levels_list[i]) # do not divide by g, it is already in m (model data)
#         if i == 0:
#             df_z_m1 = 0
#         else:
#             df_z_m1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i-1])
#         df_z_p1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i+1])
#         # integrate (sum)
#         J_add = climxa.Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)

#         # test if climxa.Cn2_func() doesn't return NaN
#         ds_check = J_add.where(xr.ufuncs.isnan(J_add) != True, drop = True)
#         print(ds_check['time'].size)
#         if ds_check['time'].size == 0:
#             print('nan array for {}'.format(i))
#         else:
#             J = J + J_add
        
#             # (for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension)
#             # do that!! (and only if J_add is not nan)
#             ds_Cn2 = Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1, k_var)
#             ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels_list[i]})

#             # concatenate DataArrays along new dimension 'level'
#             if i == 0:
#                 ds_Cn2_profile = ds_Cn2
#             else:
#                 ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')

    
#     # calculate seeing (epsilon) and convert to xarray dataset with only time as a dimension
#     ds_seeing = xr.Dataset({"seeing": climxa.epsilon(J)})
#     ds_seeing = ds_seeing.reset_coords(drop=True)

#     ds_Cn2_profile = xr.Dataset({"Cn2": ds_Cn2_profile})
#     ds_Cn2_profile = ds_Cn2_profile.reset_coords(drop=True)

#     ds_seeing["seeing"].attrs['long_name'] = 'Seeing'
#     ds_seeing["seeing"].attrs['units'] = 'arcsec'

#     ds_Cn2_profile["Cn2"].attrs['long_name'] = 'Cn2'
#     ds_Cn2_profile["Cn2"].attrs['units'] = 'm^(1/3)'

#     # add coordinate 'level'
#     ds_seeing["seeing"] = ds_seeing["seeing"].assign_coords(level=pr_levels_list[i], longitude=lon, latitude=lat)
#     # add dimension 'level'
#     ds_seeing["seeing"] = ds_seeing["seeing"].expand_dims(dim=['level', 'longitude', 'latitude'])
#     # now, the level can be selected with e.g. ds.sel(level=775)
#     # add coords lon and lat

#     # define path
#     path = './sites/'+ site + '/Data/HighResMIP/seeing/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files

#     # make directory if not available
#     os.makedirs(os.path.dirname(path), exist_ok=True) 

#     ds_seeing.to_netcdf(path)

#     # path for Cn2 profile
#     path_Cn2 = './sites/'+ site + '/Data/HighResMIP/Cn2/Amon/' + clim_key + '/' + forcing +'.nc' # where to save the files
#     os.makedirs(os.path.dirname(path_Cn2), exist_ok=True) 

#     ds_Cn2_profile.to_netcdf(path_Cn2)


# %%
