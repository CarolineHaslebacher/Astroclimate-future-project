# this script loads in climate model data, 
# extracts the lon/lat grid (from site_dict)
# and stores the files as netcdf files again
# this will shorten the time for loading and processing the climate model data immensely

#%% import libraries
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr

import pickle
import os

import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa

import matplotlib.pyplot as plt
import time

#%%

import importlib
importlib.reload(climxa)

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

# load in site_dict
d_site_lonlat_data = pickle.load( open( "./d_site_lonlat_data.pkl", "rb" ))

#%%

def PRIMAVERA_extract(model, var,  upper_lon, lower_lon, lower_lat, upper_lat, incr = None):
    # longitudes must all be in 0-360 format
    try:
        # add a small increment to the lat/lon boundaries to avoid white space in the plots
        # unfortunately, this creates a bias of the colorbar. To also avoid this, you would have to select twice
        # we just apply the smallest possible increment so that the colorbar is not disturbed too much, but the plots do not show white space
        incr = increment
        x_index, y_index = climxa.check_order(model[var], 'longitude', 'latitude', lower_lon - incr, upper_lon + incr, lower_lat - incr, upper_lat + incr)
        model_sel = model.sel(longitude=x_index,latitude= y_index)

    except KeyError:
        x_index, y_index = climxa.check_order(model[var], 'lon', 'lat', lower_lon - incr, upper_lon + incr, lower_lat - incr, upper_lat + incr)
        model_sel = model.sel(lon= x_index,lat= y_index)

    return model_sel

#%%

# load in model data for all model names

# define model names and variables

# ls_model_name = ['EC-Earth'] #, 'MPI', 'CMCC', 'CNRM', 'HadGEM', 'ECMWF'] # 
# #variables = ['ps']
# # ls_model_name = ['HadGEM']
# variables = ['clt'] # 'hur', 'hus', 'prw', 'ta', 'tas'

# seeing
ls_model_name = ['MPI', 'CMCC', 'CNRM', 'HadGEM', 'ECMWF', 'EC-Earth'] # 
variables = ['ua', 'va', 'zg']


# define increment of gridboxes
increment = 0.5 # enough for contourplot
# note: if I see that it is quite fast, I can do an increment of 3, to be able to generate a zoomed out plot
# but: I do not have these data for ERA5

# make sure that we keep all attributes and do not loose them
xr.set_options(keep_attrs=True)

# load in dataset for all folders
for model_folder in ls_model_name:
    for variable in variables:

        path = './HighResMIP/variables/' + variable + '/Amon/' + model_folder 
        print(path)
        # os.listdir gives me the subdirectories (not always the same for every model)
        
        # CHOOSE LINES HERE
        time_folders = os.listdir(path = path)
        # time_folders = ['present'] # for only one new folder (EC-Earth, present)

        for time_folder in time_folders: # now cycle through the folders (hist, present, ..)
            print(time_folder)
            subdir = path + '/' +  time_folder + '/'
            
            # mfdataset causes problems if there is only 1 single .nc file (this is the case for 
            # CMCC, tas, hist, for example)
            if len(os.listdir(path = subdir)) <= 3: # there is always the bash file (.sh) and the status file (.status) 
                # open the file that ends with .nc
                for my_file in os.listdir(path = subdir):
                    if my_file.endswith(".nc"):
                        print(subdir + my_file)
                        ds = xr.open_dataset(subdir + my_file)
            else: # if there is only one .nc file
                ds = xr.open_mfdataset(path + '/' +  time_folder + '/*.nc', combine = 'by_coords')
            
            for idx in range(4, 5): # (0, 8)
        
                # start clock
                start_time = time.time() # measure elapsed time

                # check time with: print("--- %s seconds ---" % (time.time() - start_time))
                
                print(d_site_lonlat_data['site_name_folder'][idx])
                                                                
                upper_lon = d_site_lonlat_data['upper_lon'][idx] 
                lower_lon = d_site_lonlat_data['lower_lon'][idx] 
                lower_lat = d_site_lonlat_data['lower_lat'][idx] 
                upper_lat = d_site_lonlat_data['upper_lat'][idx] 
                site = d_site_lonlat_data['site_name_folder'][idx]

                # select grids (from site dict, plus 1 deg for maps plotting)
                ds_sel = PRIMAVERA_extract(ds, variable, upper_lon, lower_lon, lower_lat, upper_lat, incr = increment)
                
                # select only relevant variable (and drop others, like 'time_bnds')
                # (new Dataset)
                ds_extracted = xr.Dataset({variable: ds_sel[variable]})

                # load this dataset now into memory (I think this helps, but it surely takes a lot of time)
                # check if it is already faster without loading

                # save to netcdf
                # define where to save the files
                path_to_save = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + variable + '/Amon/' + model_folder + '/' 
                # make directory if not already created
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                if os.path.exists(path_to_save + time_folder + '.nc') == True: 
                    print('file already exists') 
                    continue
                else:
                    ds_extracted.to_netcdf(path = path_to_save + time_folder + '.nc') # option: compute=True

                print("--- %s seconds for one site ---" % (time.time() - start_time))

# note: other option is not to combine all netcdfs
# note: it makes sense to take all pressure levels, so don't touch them

# (check if it loads faster!)

#%%
# # run file only for 'hist' folder
# # for testing performance
# variable = 'hus'
# time_folder = 'hist'
# for model_folder in ls_model_name:
#     path = './HighResMIP/variables/' + variable + '/Amon/' + model_folder
#     print(path)

#     # correct part
#     ds = xr.open_mfdataset(path + '/' +  time_folder + '/*.nc', combine = 'by_coords')
#     print(time_folder)
#     for idx in range(0, len(d_site_lonlat_data['site_name_folder'])): 
#         # start clock
#         start_time = time.time() # measure elapsed time

#         # check time with: print("--- %s seconds ---" % (time.time() - start_time))
        
#         print(d_site_lonlat_data['site_name_folder'][idx])
                                                        
#         upper_lon = d_site_lonlat_data['upper_lon'][idx] 
#         lower_lon = d_site_lonlat_data['lower_lon'][idx] 
#         lower_lat = d_site_lonlat_data['lower_lat'][idx] 
#         upper_lat = d_site_lonlat_data['upper_lat'][idx] 
#         site = d_site_lonlat_data['site_name_folder'][idx]

#         # select grids (from site dict, plus 1 deg for maps plotting)
#         ds_sel = PRIMAVERA_extract(ds, variable, upper_lon, lower_lon, lower_lat, upper_lat, incr = increment)
        
#         # select only relevant variable (and drop others, like 'time_bnds')
#         # (new Dataset)
#         ds_extracted = xr.Dataset({variable: ds_sel[variable]})

#         # load this dataset now into memory (I think this helps, but it surely takes a lot of time)
#         # check if it is already faster without loading

#         # save to netcdf
#         # define where to save the files
#         path_to_save = '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site +'/Data/HighResMIP/' + variable + '/Amon/' + model_folder + '/' 
#         # make directory if not already created
#         os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
#         if os.path.exists(path_to_save + time_folder + '.nc') == True: 
#             print('file already exists') 
#             continue
#         else:
#             ds_extracted.to_netcdf(path = path_to_save + time_folder + '.nc') # option: compute=True

#         print("--- %s seconds for one site ---" % (time.time() - start_time))




# %%
# # compare reading ERA5 and HadGEM data

# ds_ERA5 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/650hPa/*.nc')
# ds_HadGEM = xr.open_dataset('./sites/MaunaKea/Data/HighResMIP/hus/Amon/HadGEM/hist.nc')



# # %%

# plt.plot(ds_ERA5['q'].sel(latitude= 20, longitude=-156))

# #%%
# plt.plot(ds_HadGEM['hus'].sel(lat=20, lon = 360-156, plev = 60000, method='nearest'))



# %%
