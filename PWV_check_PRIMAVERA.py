# 2020-05-07
# attempt to optimize Astroclimate_function pool by only using xarray Datasets (in the outlook of climate model data (PRIMAVERA))
# goal is to have 1 script for 1 site, and all parameters. 
# functions should be as flexible as possible

# THIS SCRIPT INCLUDES:
    # PRIMAVERA MODEL CHECK UNTIL SURFACE PRESSURE
    # ERA5 TCW PROFILE

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

import skill_metrics as sm

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

list_of_single_level_vars = ['tcw']
variable = 'TCW'
list_of_clim_vars = ['q_integrated']
list_of_model_clim_params = ['q_integrated']


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

max_idx = 0 # starts at 0 (max is 7)

diurnal= False
Ensemble = False
masterfigure = True
save = False
model = True
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

for idx in range(0, max_idx+1):
    # or define index for one iteration only
    # idx = 3

    file_name = 'noTCW_' # at least, type '_' 

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list, model_color) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    time_slice_var_PWV = d_site_lonlat_data['time_slice_var_PWV'][idx]
    
    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files
    
    # get observed pressure
    surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
    print(surface_pressure_observation)
    # check available pressure for ERA5,
    # choose closest match
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    pr_levels_ERA5 = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa
    pr_levels_model = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    given_value = surface_pressure_observation
    SH_integral_pressure_ERA5 = min(pr_levels_ERA5, key=absolute_difference_function)
    SH_integral_pressure_model = min(pr_levels_model, key=absolute_difference_function)
    # SH_integral_pressure can be written into d_obs['Plev'], so that plotting function selects pressure level
    # use SH_integral_pressure as pressure for the data (ERA5 and model data)
    # the datasets are stored so that the pressure can actually be selected, 
    # even if there is only one pressure
    ls_pr_levels_ERA5 = [SH_integral_pressure_ERA5]
    ls_pr_levels_clim_model = [SH_integral_pressure_model]

    ###########

    # for check of ps integral 
    # ps: also change 'path_q_integrated' (add surface_pressure) in climxa.py
    list_of_single_level_model_clim_params = ['prw']

    d_model = {"HadGEM": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "HadGEM3-GC31-HM"},
                "EC-Earth": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "EC-Earth3P-HR"} ,
                 "CNRM": {"folders": ['hist'], "taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "CNRM-CM6-1-HR"},
                 "MPI": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "MPI-ESM1-2-XR"},
                 "CMCC": {"folders": ['hist'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, "name": "CMCC-CM2-VHR4"},
                 "ECMWF": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "ECMWF-IFS-HR"} }
    # for check, write individual surface pressure mean values into 'Plev'
    for clim_key in d_model.keys():
        # we must define single_lev_var as 'ps' to load in surface pressure (change back in the end)
        d_model[clim_key]['single_lev_var'] = ['ps']
        # load surface pressure data... NOT FINISHED!!
        d_model[clim_key]['ds_surface_pressure'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)
        # convert to hPa
        d_model[clim_key]['ds_surface_pressure']['ps hist'] = d_model[clim_key]['ds_surface_pressure']['ps hist']/100
        # select lon/lat
        d_model[clim_key]['ds_surface_pressure'] = climxa.xr_sel(d_model[clim_key], 'ds_surface_pressure', lon, lat)
        # calculate the mean (and integrate until this value (otherwise, I cannot shorten the for loop by time))
        d_model[clim_key]['surface_pressure_mean'] = d_model[clim_key]['ds_surface_pressure']['ps hist'] .mean()
        print(d_model[clim_key]['surface_pressure_mean'])
        # also check standard deviation!

        #  define until which pressure level we integrate
        given_value = d_model[clim_key]['surface_pressure_mean'] 
        # find closest value available in pressure levels
        closest_value = min(pr_levels_model, key=absolute_difference_function)

        # write pressure into ['Plev']
        d_model[clim_key]['Plev'] = [closest_value]

        if clim_key == 'CMCC':
            # do not plot 'prw'
            del(d_model[clim_key]['single_lev_var'])

        else:
            d_model[clim_key]['single_lev_var'] = ['prw']


    ############

    # I only have PWV data for Mauna Kea, La Palma, Paranal and La Silla
    # 'path_ds_PWV' for other sites are set to zero
    if d_site_lonlat_data['path_ds_PWV'][idx] != 0:
        if idx == 2 and diurnal == True: # la silla: use upsampled hourly data
            path_ds_PWV = './sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_upsampled_Linear_interpolation_La_Silla_PWV.csv'
        else:
            path_ds_PWV = d_site_lonlat_data['path_ds_PWV'][idx]
        list_of_insitu_vars = [site_name_folder + ' PWV']

        # read in ds_hourly (in-situ data)
        ds_hourly = climxa.df_to_xarray(path_ds_PWV)

        if idx == 0: # Mauna Kea
            ds_hourly = ds_hourly.rename_vars({'PWV': list_of_insitu_vars[0]})
            # for MaunaKea, dataset needs to be filtered 
            # (there are values equal to 99, which are clearly measurement errors)
            mask_PWV_MK = (ds_hourly['MaunaKea PWV'] != 99)
            ds_temp= ds_hourly['MaunaKea PWV'][mask_PWV_MK]
            # delete ds_hourly
            del(ds_hourly)
            # and write ds_temp again into ds_hourly
            ds_hourly = xr.Dataset({'MaunaKea PWV': ds_temp})


        # instead of dropping variables, like this:
        # ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Palma Relative Humidity', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Temperature', 'La_Palma Pressure'])    
        # better compose new dataset with list_of_insitu_vars
        
        ds_hourly_sel = xr.Dataset({list_of_insitu_vars[0]: ds_hourly[list_of_insitu_vars[0]] })
        # if more than one element in list_of_insitu_vars, use 'assign' to add variables to dataset
        if len(list_of_insitu_vars) > 1:
            for var in list_of_insitu_vars[1:]:
                ds_hourly_sel = ds_hourly_sel.assign({var: ds_hourly[var]})

        # define observational dict
        d_obs = {"ds": ds_hourly_sel, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}


    else:
        list_of_insitu_vars = None
        path_ds_PWV = None

        d_obs = {"ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5} #, "single_lev": list_of_single_level_vars}

    if idx == 0: # Mauna Kea again
        # in this dataset, there are duplicate values. I don't know why, but I am going to remove them
        _, index, counts = np.unique(d_obs['ds']['time'], return_index=True, return_counts=True)

        d_obs['ds'] = d_obs['ds'].isel(time=index)


    # append d_site_lonlat_data with SH_integral_pressure

    # define model dict

    # exclude EC-Earth from present (has no 'present')
    # exclude ECMWF and MPI from 'future'
    # exclude EC-Earth, MPI, ECMWF from SSTfuture
    # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
    
    # # change here: folders
    # folder = ['future'] 'ds_ensemble_mean_month_comparison'
    # folder_for_path = 'future'
    # d_model = {"HadGEM": {"folders": folder, "taylor_folder": folder,"single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,  "name": "HadGEM3-GC31-HM"},
    #            "EC-Earth": {"folders": folder,"taylor_folder": folder, "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,  "name": "EC-Earth3P-HR"},
    #             "CNRM": {"folders": folder, "taylor_folder": folder,"single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,  "name": "CNRM-CM6-1-HR"}} #,
    #             # "MPI": {"folders": folder, "taylor_folder": folder,"single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,  "name": "MPI-ESM1-2-XR"},
    #             #"CMCC": {"folders": folder, "taylor_folder": folder,"single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,  "name": "CMCC-CM2-VHR4"},
    #             # "ECMWF": {"folders": folderpresent

    # if model:
    #     # d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
    #     #             "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
    #     #             # "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
    #     #             "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
    #     #             "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
    #     #             "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }
        
    #     if idx == 1: # Paranal data intersects only with future 
    #         # take future for taylor folder
    #         d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
    #                 "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['future'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
    #                 "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
    #                 # "MPI": {"folders": ['hist', 'present'],"taylor_folder": [],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
    #                 "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['future', 'SSTfuture'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"}} #,
    #                 # "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": [],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

    #                 # ADD MPI AND ECMWF LATER, WHEN YOU HAVE DATA!!!!!!
    #     else:
    #         d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
    #                 "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
    #                 "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"},
    #                 "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
    #                 "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
    #                 "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
    #                 "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }
        
    # else:
    #     d_model = None

    # # for CMCC analysis only
    # ls_pr_levels_clim_model = [1000]
    # d_model = {"CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"}}
    # d_obs = {"single_lev": list_of_single_level_vars}
    # # fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/TCW/CMCC_prw_vs_integral.png')

    if Ensemble:
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model} # , "single_lev_var": list_of_single_level_model_clim_params
    else:
        d_Ensemble = None

    # only with the keyword 'SH_integral=True', climxa doesn't search for 'q_integrated'
    # but reads in the prepared datasets directly

    if masterfigure:
        # save pressure levels to list, for legend (plot only pressure levels that are used)
        for P_lev in d_obs['Plev']:
            if P_lev not in Plev_list:
                Plev_list.append(P_lev)
        if Ensemble:
            for P_lev in d_Ensemble['Plev']:
                if P_lev not in Plev_list:
                    Plev_list.append(P_lev)
        elif d_model != None: # loop through climate models
            for clim_key in d_model.keys():
                for P_lev in d_model[clim_key]['Plev']:
                    if P_lev not in Plev_list:
                        Plev_list.append(P_lev)

    # main plotting routine
    importlib.reload(climxa)

    start_time = time.time() # measure elapsed time

    if masterfigure:
        if idx == 0:
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_PWV, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, SH_integral=True, d_Ensemble=d_Ensemble, MasterFig=MasterFig)
        else: # now we have an axis we can refer to fore sharing the xaxis
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_PWV, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, SH_integral=True, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                        ax_ref = (ax0_t, ax3_t))
    else:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_PWV, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, SH_integral=True, d_Ensemble=d_Ensemble, MasterFig=MasterFig)


    print("--- %s seconds ---" % (time.time() - start_time))
    
    # save as .pdf
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
                fig.savefig('./Model_evaluation/' + variable + '/'  + variable +'_overview_Ensemble_DSC_T.pdf', bbox_inches = 'tight', pad_inches=0.0)
        if diurnal:
            if idx == max_idx:
                fig_diurnal.savefig('./Model_evaluation/' + variable + '/' + variable + '_overview_Diurnal_ERA5.pdf', bbox_inches = 'tight', pad_inches=0.0)

        # save as .png
        else:
            fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)

        if Ensemble:
            with open('/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site_name_folder +'/Data/HighResMIP/' + variable + '_d_Ensemble.pkl', "wb") as myfile:
                pickle.dump(d_Ensemble, myfile)
                
        # save dict to csv
        # [Skill, ccoef[i], sdev[i], crmsd[i]]
        path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/csv_info/'
        os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
        (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
            .to_csv(path_skill_folder + site_name_folder + file_name + 'sorted_skill_dict.csv', 
                    header=['Skill', 'corr_coef', 'st_dev', 'crms_error']))

    # ########## plot trend map
    # path_map = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/maps/'
    # os.makedirs(os.path.dirname(path_map), exist_ok=True) 

    # # no q_integrated right now
    # # d_obs = {"ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}
    # del(d_obs['ERA5_var'])
    # del(d_obs['Plev'])

    # folders_list = [['hist'], ['present'], ['future'], ['SSTfuture']]
    # for folders in folders_list:
    
    #     d_model = {"HadGEM": {"folders": folders,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
    #             "EC-Earth": {"folders": folders,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
    #                 "CNRM": {"folders":folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
    #                 "MPI": {"folders": folders,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
    #                 "CMCC": {"folders": folders ,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
    #                 "ECMWF": {"folders": folders,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

    #     climxa.trend_map(d_model, d_obs, variable, site_name_folder, idx, path_map, SH_integral=False)


#%% check-up: individual Taylor-diagrams

def get_ps_check_d_model(clim_key):

    absolute_difference_function = lambda list_value : abs(list_value - given_value)

    pr_levels_model = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


    # we must define single_lev_var as 'ps' to load in surface pressure (change back in the end)
    d_model[clim_key]['single_lev_var'] = ['ps']
    # load surface pressure data... NOT FINISHED!!
    d_model[clim_key]['ds_surface_pressure'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)
    # convert to hPa
    d_model[clim_key]['ds_surface_pressure']['ps hist'] = d_model[clim_key]['ds_surface_pressure']['ps hist']/100
    # select lon/lat
    d_model[clim_key]['ds_surface_pressure'] = climxa.xr_sel(d_model[clim_key], 'ds_surface_pressure', lon, lat)
    # calculate the mean (and integrate until this value (otherwise, I cannot shorten the for loop by time))
    d_model[clim_key]['surface_pressure_mean'] = d_model[clim_key]['ds_surface_pressure']['ps hist'] .mean()
    
    # print if needed
    # print(d_model[clim_key]['surface_pressure_mean'])
    
    # also check standard deviation!

    #  define until which pressure level we integrate
    given_value = d_model[clim_key]['surface_pressure_mean'] 
    # find closest value available in pressure levels
    closest_value = min(pr_levels_model, key=absolute_difference_function)
    print(closest_value)

    # write pressure into ['Plev']
    d_model[clim_key]['Plev'] = [closest_value]

    # change single level variable to precipitable rain water
    d_model[clim_key]['single_lev_var'] = ['prw']

    ####### get PRIMAVERA data

    # load q_integrated
    path_q_integrated_model = '/home/haslebacher/chaldene/Astroclimate_Project//sites/'+ site_name_folder + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site_name_folder + '_' + str(closest_value) + 'hPa_surface_pressure.nc' # for check

    d_model[clim_key]['ds_Amon_Pr'] = xr.open_dataset(path_q_integrated_model) 

    # load prw (single level data)
    d_model[clim_key]['ds_Amon_Sg'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)

    # merge
    d_model[clim_key]['ds_Amon'] = xr.merge([d_model[clim_key]['ds_Amon_Pr'], d_model[clim_key]['ds_Amon_Sg']])

    # load into memory
    d_model[clim_key]['ds_Amon'] = d_model[clim_key]['ds_Amon'].load()

    # select lon/lat 
    # use ds_sel for timeseries, where only lon/lat is selected (and not time)
    d_model[clim_key]['ds_sel'] = climxa.xr_sel(d_model[clim_key], 'ds_Amon', lon, lat) # pressure levels are selected with this function

    return closest_value


#%%

# choose max index
max_idx = 7

# loop through sites
for idx in range(0, max_idx+1):

    # initialize one figure for each site
    fig = plt.figure(figsize = (16, 28)) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
    gs = fig.add_gridspec(6, 2) # first entry: rows, second entry: columns

    print(d_site_lonlat_data['site_name'][idx])
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    # define d_model and load data
    list_of_single_level_model_clim_params = ['prw']

    d_model = {"HadGEM": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "HadGEM3-GC31-HM"},
                "EC-Earth": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "EC-Earth3P-HR"} ,
                 "CNRM": {"folders": ['hist'], "taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "CNRM-CM6-1-HR"},
                 "MPI": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "MPI-ESM1-2-XR"},
                 "CMCC": {"folders": ['hist'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, "name": "CMCC-CM2-VHR4"},
                 "ECMWF": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "ECMWF-IFS-HR"} }
    
    # for check, write individual surface pressure mean values into 'Plev'
    for clim_ind, clim_key in enumerate(d_model.keys()):

        # for check of ps integral 
        closest_value = get_ps_check_d_model(clim_key)

        # for clim_key in d_model.keys():

        # add subplot
        # gs = fig.add_gridspec(1, 2)
        ax0 = fig.add_subplot(gs[clim_ind, 0]) # correlation plot
        ax1 = fig.add_subplot(gs[clim_ind, 1]) # Taylor diagram

        # if clim_key == 'CNRM' and site_name_folder == 'MaunaKea':
        #     continue
    
        #else:

        # sometimes, the integrated until surface pressure datasets are 'nan' only
        try:

            ref_pred_ls = []
            # first entry is 'prw hist', the ground truth
            # second entry is 'q_integrated hist', the integrated specific humidity
            # we want to see how close they are, if we compare monthly values
            ref_pred_ls.append({"data": d_model[clim_key]['ds_sel']['prw hist']})
            ref_pred_ls.append({"data": d_model[clim_key]['ds_sel']['q_integrated hist'].sel(level=closest_value)})

            taylor_label = ['integral']

            taylor_stats = [sm.taylor_statistics(pred, ref_pred_ls[0],'data') for pred in ref_pred_ls[1:]]


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

            # fig, ax = plt.subplots(1,2, figsize=(8,5))

            sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                                colOBS = 'k', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                                titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                                alpha=0.7, markerSize= 9, legendCH=False) # ,  my_axis=ax1 # doesn't help!

            # plot simple correlation diagram
            ax0.scatter(ref_pred_ls[0]['data'], ref_pred_ls[1]['data'])
            ax0.plot(range(0,40), range(0,40), color='k')
            ax0.set_xlabel('prw [mmH2O]')
            ax0.set_ylabel('specific humidity integral [mmH2O] \nuntil surface pressure closest to {}'.format(round(float(d_model[clim_key]['surface_pressure_mean'].values),2)))
            ax0.set_title(f'{clim_key}')

            # box = ax1.get_position()
            # ax1.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

        except ValueError:
            print('Value Error for {}'.format(clim_key))
            continue
                
    fig.suptitle(f'{site_name_folder}')

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.show()

    # save for every site
    fig.savefig('./Model_evaluation/TCW/check_PWV_integral_until_surface_pressure/' + site_name_folder + '_corr_and_Taylor.pdf', bbox_inches = 'tight', pad_inches=0.0)



# %%
