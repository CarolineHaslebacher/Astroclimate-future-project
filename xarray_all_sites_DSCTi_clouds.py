# ADD THIS CODE TO SAVE d_obs and d_model FOR LONGTERM in the folder Astroclimate_outcome
        # # save d_obs
        # with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_d_obs_ERA5_and_insitu.pkl', "wb") as myfile:
        #         pickle.dump(d_obs, myfile)

        # # save d_model and d_obs
        # if model:
        #     with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_'+ variable + '_d_model.pkl', "wb") as myfile:
        #         pickle.dump(d_model, myfile)

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

# for calculated cloud cover:
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
Ensemble = False # problem if only single lev PRIMAVERA var!! taylor folder vars....
masterfigure = False
save = False
model = True
nighttime = False
model_color = False

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
# for idx in range(0, max_idx+1):

for idx in range(4, 5):

    file_name = '' 
    if nighttime:
        file_name = '_nighttime'

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list, model_color) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    # Mauna Loa
    # lon = 360-155.5
    # lat = 19.5

    path_ds_clouds = d_site_lonlat_data['path_ds_clouds'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    time_slice_var_clouds = d_site_lonlat_data['time_slice_var_clouds'][idx] # !!!!
    list_of_insitu_vars = [site_name_folder + ' Clouds']

    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files
    
    
    if d_site_lonlat_data['path_ds_clouds'][idx] != 0:
        # read in ds_hourly (in-situ data) 
        ds_hourly = climxa.df_to_xarray(path_ds_clouds)

        if idx == 0: # Mauna Kea
            list_of_insitu_vars = ['Mauna Kea Clouds']
            ### high cloud cover for Mauna Kea
            # list_of_single_level_vars = ['hcc']
            # variable = 'high_cloud_cover'

        if idx ==5:
            list_of_insitu_vars =  ['percentage_of_time_lost']

        ds_hourly_sel = xr.Dataset({list_of_insitu_vars[0]: ds_hourly[list_of_insitu_vars[0]] })
        # if more than one element in list_of_insitu_vars, use 'assign' to add variables to dataset
        if len(list_of_insitu_vars) > 1:
            for var in list_of_insitu_vars[1:]:
                ds_hourly_sel = ds_hourly_sel.assign({var: ds_hourly[var]})

        # define observational dict
        d_obs = {"ds": ds_hourly_sel, "insitu_var": list_of_insitu_vars, "single_lev": list_of_single_level_vars}

    else:
        list_of_insitu_vars = None
        path_ds_clouds = None

        d_obs = {"single_lev": list_of_single_level_vars} #, "single_lev": list_of_single_level_vars}

    # define observational dict
    # d_obs = {"ds": ds_hourly, "insitu_var": ['percentage_of_time_lost'], "single_lev": list_of_single_level_vars}
    # 

    if model:
        if idx == 0: # for Mauna Kea, the cloud data starts in 2015 only
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

    # if model:
    #     d_model = {"HadGEM": {"folders": ['hist', 'present', 'future'],"taylor_folder": ['hist', 'present'],"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"}}
    # else:
    #     d_model = None

    # if Ensemble:
    #     d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params}
    # else:
    #     d_Ensemble = None

    # if masterfigure:
    #     # save pressure levels to list, for legend (plot only pressure levels that are used)
    #     for P_lev in d_obs['Plev']:
    #         if P_lev not in Plev_list:
    #             Plev_list.append(P_lev)
    #     if Ensemble:
    #         for P_lev in d_Ensemble['Plev']:
    #             if P_lev not in Plev_list:
    #                 Plev_list.append(P_lev)
    #     elif d_model != None: # loop through climate models
    #         for clim_key in d_model.keys():
    #             for P_lev in d_model[clim_key]['Plev']:
    #                 if P_lev not in Plev_list:
    #                     Plev_list.append(P_lev)

    importlib.reload(climxa)

    # import time
    # start_time = time.time() # measure elapsed time
    # d_obs_ss, d_model_ss, fig_ss, sorted_skill_dict_ss = climxa.main_plotting_routine(site_name_folder, variable, 
    #                                 time_slice_var, d_obs, lon, lat, path, diurnal=False,
    #                                     d_model = d_model, SH_integral=False)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # # save as png with high resolution. Only save as pdf in the 'last round'
    # # fig_ss.savefig('./Model_evaluation/' + variable + '/' + folder_for_path + '/' + site_name_folder + '_' + variable + '_DSC_T.png', dpi=400)



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


        fig.savefig('./Model_evaluation/' + variable + '/' + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400)

        # save dict to csv
        path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/'+ variable + '/csv_info/'
        os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
        (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
            .to_csv(path_skill_folder + site_name_folder + file_name +'_sorted_skill_dict.csv', header=False))

#%%
# map
########## plot trend map
path_map = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/maps/'
os.makedirs(os.path.dirname(path_map), exist_ok=True) 

# d_model = {"HadGEM": {"folders": folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
            #"EC-Earth": {"folders": ['hist','present', 'future'], "taylor_folder": taylor_folders, "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
            "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
            "MPI": {"folders": ['hist', 'present'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
            "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
            "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": taylor_folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"}}
    

climxa.trend_map(d_model, d_obs, variable, site_name_folder, idx, path_map, SH_integral=False)


#%% testing Hawaian regional climate model


ds_HRCM = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/HRCM/tcc/Hawaii_Regional_Climate_Model_Simulations_Hourly_3km_domain_Present_1990_2009_tcc.nc')
ds_HRCM_sel = ds_HRCM.sel(lon=lon, lat=lat, method='nearest')


ds_HRCM_yearly = ds_HRCM_sel['cldt'].groupby('time.year').mean()
ds_HRCM_monthly = ds_HRCM_sel['cldt'].groupby('time.month').mean()

# plot onto fig (which was generated above)
ax3_t.plot(ds_HRCM_yearly.year, ds_HRCM_yearly)
ax0_t.plot(ds_HRCM_monthly.month, ds_HRCM_monthly)
# add to d_obs as insitu param?

# save fig
file_name = '_HRCM_cldt_'
fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_correctcoords_DSC_T.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)
        

# %% TESTING formula from Hellemeier

# siding spring ERA5 elevation: ~500m
# siding spring observatory elevation: ~1134m
# f_c = (2000m - 1134m)/2000m * f_LCC + f_MCC + f_HCC
insitu_elevation = 1134

# Mauna Kea
insitu_elevation = 4200

ds_LCC = climxa.read_ERA5_sg_level_data(site_name_folder, 'low_cloud_cover', 'lcc')
ds_MCC = climxa.read_ERA5_sg_level_data(site_name_folder, 'medium_cloud_cover', 'mcc')
ds_HCC = climxa.read_ERA5_sg_level_data(site_name_folder, 'high_cloud_cover', 'hcc')
ds_base = climxa.read_ERA5_sg_level_data(site_name_folder, 'cloud_base_height', 'cbh')

#%%

if lon > 180:
    my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
    print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
else:
    my_ERA5_lon = lon
    
ds_LCC_sel = ds_LCC.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
ds_MCC_sel = ds_MCC.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
ds_HCC_sel = ds_HCC.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
ds_base_sel = ds_base.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')

ds_test = ds_LCC_sel.lcc + ds_HCC_sel.hcc + ds_MCC_sel.mcc # --> divide by 3

# siding spring
# ds_Hellemeier = ((2000-insitu_elevation)/2000*ds_LCC_sel.lcc + ds_MCC_sel.mcc + ds_HCC_sel.hcc)/3

# mauna kea
# mcc goes from 2km to 6km
ds_Hellemeier = ((6000-insitu_elevation)/6000*ds_MCC_sel.mcc + ds_HCC_sel.hcc)/2


ds_cloud_calc = xr.Dataset({"calc_cloud_cover": ds_Hellemeier})
ds_cloud_calc["calc_cloud_cover"].attrs['long_name'] = 'calculated cloud cover'
ds_cloud_calc["calc_cloud_cover"].attrs['units'] = 'fraction'

# folder 'calc_cloud_cover' must be prepared
ds_cloud_calc.to_netcdf('./sites/' + site_name_folder + '/Data/Era_5/calc_cloud_cover/calc_cloud_cover.nc')


# %%
