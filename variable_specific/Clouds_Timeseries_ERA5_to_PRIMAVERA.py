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

import seaborn as sns
sns.set()
from matplotlib import cycler

import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

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

#%% global parameters

list_of_single_level_vars = ['tcc']
variable = 'total_cloud_cover'
# list_of_single_level_vars = ['hcc']
# variable = 'high_cloud_cover'
# list_of_single_level_vars = ['calc_cloud_cover']
# variable = 'calc_cloud_cover'

# list_of_clim_vars = ['seeing']
# list_of_model_clim_params = ['q_integrated']
list_of_single_level_model_clim_params = ['clt']

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


#%% main

# choose max_idx
max_idx = 7 # starts at 0 (max is 7)

diurnal=False
Ensemble = True # problem if only single lev PRIMAVERA var!! taylor folder vars....
masterfigure = True
save = True
model = True
nighttime = True
future=True

if masterfigure == True:
    master_fig = plt.figure( figsize = (25, 4*(max_idx+1))) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
    gs = master_fig.add_gridspec(max_idx + 1, 4) # first entry: rows, second entry: columns
    # initialize pressure level list (for legend; plot only those in use)
    Plev_list = []
else:
    MasterFig = None

if diurnal:
    # create a whole new figure only for the diurnal cycles
    fig_diurnal = plt.figure(figsize=(15, 2*(max_idx+1))) # max 16 height 
    # fig_diurnal, ax1 = plt.subplots(sharex=True, sharey=True) 
else:
    fig_diurnal = None

# for idx in [4]: # range(0, max_idx+1):
for idx in range(0, max_idx+1):

    if future:
        file_name = '_ERA5toPRIMAVERA_FUTURE_' # at least, type '_' 
    else:
        file_name = '_ERA5toPRIMAVERA_HIST_'

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    time_slice_var_clouds = slice('1979-01-01', '2014-12-31')

    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files
    
    
    list_of_insitu_vars = None
    path_ds_clouds = None

    d_obs = {"single_lev": list_of_single_level_vars} #, "single_lev": list_of_single_level_vars}

    # define observational dict
    # d_obs = {"ds": ds_hourly, "insitu_var": ['percentage_of_time_lost'], "single_lev": list_of_single_level_vars}
    # 

    if model:
        if future==True: # for Mauna Kea, the cloud data starts in 2015 only
            taylor_folders = ['future', 'SSTfuture']

            # if I would like to include all models, I would have to implement 'if taylor_folder is empty...'
            d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
                #"EC-Earth": {"folders": ['hist','present', 'future'], "taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'],"taylor_folder": [] ,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
                    "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": [] ,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"}}
            
        else:

            taylor_folders = ['hist', 'present'] # 
        
            # d_model = {"EC-Earth": {"folders": ['hist'], "taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"}}
            
            # EC-Earth is not running ('hist' is fine, the others are not). Download issues? there are nan values for the selected lon/lat
            d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
                #"EC-Earth": {"folders": ['hist','present', 'future'], "taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"}}
            
    else:
        d_model = None

    if Ensemble:
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "single_lev_var": list_of_single_level_model_clim_params}
    else:
        d_Ensemble = None

    importlib.reload(climxa)

    start_time = time.time() # measure elapsed time
    if masterfigure:
        if idx == 0:
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_clouds, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig, nighttime=nighttime)
        else: # now we have an axis we can refer to fore sharing the xaxis
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_clouds, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                        ax_ref = (ax0_t, ax3_t), nighttime=nighttime)
    else:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_clouds, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig, nighttime=nighttime)


    print("--- %s seconds ---" % (time.time() - start_time))
    
    if save:
        # save d_obs
        with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_d_obs_ERA5_and_insitu.pkl', "wb") as myfile:
                pickle.dump(d_obs, myfile)

        # save d_model and d_obs
        if model:
            with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_'+ variable + '_d_model.pkl', "wb") as myfile:
                pickle.dump(d_model, myfile)
                
        if masterfigure:
            if idx == max_idx:
                fig.savefig('./Model_evaluation/' + variable + '/'  + variable + file_name +  '_overview_Ensemble_DSC_T.pdf', bbox_inches = 'tight', pad_inches=0.0)
        
        if diurnal:
            if idx == max_idx:
                fig_diurnal.savefig('./Model_evaluation/' + variable + '/' + variable + '_overview_Diurnal_ERA5.pdf', bbox_inches = 'tight', pad_inches=0.0)

        if masterfigure == False:
            print('gets saved')
            fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)
        
        if Ensemble:
            with open('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site_name_folder +'/Data/HighResMIP/' + variable + '_d_Ensemble.pkl', "wb") as myfile:
                pickle.dump(d_Ensemble, myfile)

        # fig_ss.savefig('./Model_evaluation/' + variable + '/' + site_name_folder + '_' + variable + '_DSC_T.png', dpi=400)

        # save dict to csv
        path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/'+ variable + '/csv_info/'
        os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
        (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
            .to_csv(path_skill_folder + site_name_folder + file_name +'_sorted_skill_dict.csv', header=False))


# %%
