# 2020-05-07
# attempt to optimize Astroclimate_function pool by only using xarray Datasets (in the outlook of climate model data (PRIMAVERA))
# goal is to have 1 script for 1 site, and all parameters.
# functions should be as flexible as possible

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
# sns.set()
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

list_of_single_level_vars = ['tcw']
variable = 'TCW'
list_of_clim_vars = ['q_integrated']
list_of_model_clim_params = ['q_integrated']
list_of_single_level_model_clim_params = ['prw']

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

max_idx = 2 # for this special PWV cycle, for which we by chance only have data for Mauna Kea, Paranal and La Silla, exactly the first three sites!
# NOTE that this must be changed if I change the order or the indices!

diurnal=True
Ensemble = False
masterfigure = True
model_color =False

if masterfigure == True:
    master_fig = plt.figure( figsize = (20, 4*(max_idx+1))) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
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

    file_name = '_' # at least, type '_'

    if idx == 3: # exclude Cerro Tololo for diurnal cycle (and stop at max_idx = 4)
        continue

    if masterfigure:
        MasterFig = (master_fig, idx, 3, gs, Plev_list, model_color) # or None

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

    if idx == 4: # La palma: add 750 hPa
        ls_pr_levels_ERA5.append(800)

    # # for check of ps integral
    # # ps: also change 'path_q_integrated' (add surface_pressure) in climxa.py
    # d_model = {"HadGEM": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "HadGEM3-GC31-HM"},
    #             "EC-Earth": {"folders": ['hist'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "EC-Earth3P-HR"} ,
    #              "CNRM": {"folders": ['hist'], "taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, "name": "CNRM-CM6-1-HR"},
    #              "MPI": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "MPI-ESM1-2-XR"},
    #              "CMCC": {"folders": ['hist'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, "name": "CMCC-CM2-VHR4"},
    #              "ECMWF": {"folders": ['hist'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, "name": "ECMWF-IFS-HR"} }
    # # for check, write individual surface pressure mean values into 'Plev'
    # for clim_key in d_model.keys():
    #     # we must define single_lev_var as 'ps' to load in surface pressure (change back in the end)
    #     d_model[clim_key]['single_lev_var'] = ['ps']
    #     # load surface pressure data... NOT FINISHED!!
    #     d_model[clim_key]['ds_surface_pressure'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)
    #     # convert to hPa
    #     d_model[clim_key]['ds_surface_pressure']['ps hist'] = d_model[clim_key]['ds_surface_pressure']['ps hist']/100
    #     # select lon/lat
    #     d_model[clim_key]['ds_surface_pressure'] = climxa.xr_sel(d_model[clim_key], 'ds_surface_pressure', lon, lat)
    #     # calculate the mean (and integrate until this value (otherwise, I cannot shorten the for loop by time))
    #     d_model[clim_key]['surface_pressure_mean'] = d_model[clim_key]['ds_surface_pressure']['ps hist'] .mean()
    #     print(d_model[clim_key]['surface_pressure_mean'])
    #     # also check standard deviation!

    #     #  define until which pressure level we integrate
    #     given_value = d_model[clim_key]['surface_pressure_mean']
    #     # find closest value available in pressure levels
    #     closest_value = min(pr_levels_model, key=absolute_difference_function)

    #     # write pressure into ['Plev']
    #     d_model[clim_key]['Plev'] = [closest_value]

        # if clim_key == 'CMCC':
        #     # do not plot 'prw'
        #     del(d_model[clim_key]['single_lev_var'])

        # else:
        #     d_model[clim_key]['single_lev_var'] = ['prw']

    # I only have PWV data for Mauna Kea, Paranal and La Silla
    # note: la palma excluded due to unreliable data
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

        d_obs = {"ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}

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

    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                 "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                 "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                 "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                 "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

    # or:
    d_model = None

    # for CMCC analysis only
    # d_model = {"CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"}}
    # d_obs = {"single_lev": list_of_single_level_vars}

    # d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "single_lev_var": list_of_single_level_model_clim_params}
    d_Ensemble = None

    # only with the keyword 'SH_integral=True', climxa doesn't search for 'q_integrated'
    # but reads in the prepared datasets directly

    if masterfigure:
        # save pressure levels to list, for legend (plot only pressure levels that are used)
        for P_lev in d_obs['Plev']:
            print(d_obs['Plev'])
            if P_lev not in Plev_list:
                print('I append list with {}'.format(P_lev))
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

    # if idx == 4: # otherwise we see a gap
    #     idx = 3
    #     max_idx = 3
    #     MasterFig = (master_fig, idx, max_idx, gs, Plev_list)

    if idx == 0:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable,
                                    time_slice_var_PWV, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                     d_model = None, SH_integral=True, d_Ensemble=d_Ensemble, MasterFig=MasterFig)
    else: # now we have an axis we can refer to fore sharing the xaxis
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable,
                                    time_slice_var_PWV, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                    d_model = None, SH_integral=True, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                    ax_ref = (ax0_t, ax3_t))

    print("--- %s seconds ---" % (time.time() - start_time))

    # save as .pdf

    # if diurnal:
    #     if idx == max_idx:
    #         fig_diurnal.savefig('./Model_evaluation/' + variable + '/' + variable + '_overview_Diurnal_ERA5.pdf', bbox_inches = 'tight', pad_inches=0.0)

    # # save as .png
    # else:
    #     fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400)

    # save as .pdf
    if masterfigure:
        if idx == max_idx:
            fig.savefig('./Model_evaluation/' + variable + '/'  + variable +'_overview_Ensemble_DSC_T_ERA5_vs_insitu_only.pdf', bbox_inches = 'tight', pad_inches=0.0)
    if diurnal:
        if idx == max_idx:
            fig_diurnal.savefig('./Model_evaluation/' + variable + '/' + variable + '_overview_Diurnal_ERA5.pdf', bbox_inches = 'tight', pad_inches=0.0)

    # save as .png
    else:
        fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400)

    # save d_obs as pickle file
    with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_d_obs_ERA5_and_insitu_diurnal.pkl', "wb") as myfile:
            pickle.dump(d_obs_ss, myfile)

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




#%%


#%% integrate SH to precipitable water vapor
# do this only once, save it as netcdf file and load it next time

# # ERA5, PWV

# importlib.reload(climxa)

# for idx in range(0, 8):

#     # idx = 5

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]
#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

#     surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
#     print(surface_pressure_observation)
#     # check available pressure for ERA5
#     absolute_difference_function = lambda list_value : abs(list_value - given_value)
#     pr_levels_ERA5 = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa

#     given_value = surface_pressure_observation
#     closest_value = min(pr_levels_ERA5, key=absolute_difference_function)

#     SH_integral_pressure = closest_value # find nearest pressure

#     print('closest match: {}'.format(SH_integral_pressure))

#     chile_grid = ['Tololo', 'Pachon', 'Silla']
#     if any(x in site_name_folder for x in chile_grid):
#         site_ERA = 'Paranal'
#     else:
#         site_ERA = site_name_folder

#     if lon > 180:
#         my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
#         print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
#     else:
#         my_ERA5_lon = lon

#     # use function which loads in all specific humidity datasets
#     # and integrates them to specific humidity

#     ds_q_integrated, ds_tcw_profile = climxa.SH_integral_to_TCW(SH_integral_pressure, site_ERA, my_ERA5_lon, lat)

#     # add attributes
#     ds_q_integrated['q_integrated'].attrs['long_name'] = 'Precipitable Water Vapor'
#     ds_q_integrated['q_integrated'].attrs['units'] = 'mmH2O' # = 1kg/m^2

#     # add coordinate 'level'
#     ds_q_integrated['q_integrated'] = ds_q_integrated['q_integrated'].assign_coords(level=SH_integral_pressure)
#     # add dimension 'level'
#     ds_q_integrated['q_integrated'] = ds_q_integrated['q_integrated'].expand_dims(dim='level')
#     # now, the level can be selected with e.g. ds.sel(level=775)

#     # define path
#     path = './sites/'+ site_name_folder + '/Data/Era_5/q_integrated/ds_ERA5_q_integrated_hourly_nearest_' + str(SH_integral_pressure) + 'hPa.nc' # where to save the files

#     # make directory if not available
#     os.makedirs(os.path.dirname(path), exist_ok=True)

#     # save array to netcdf file, store it
#     ds_q_integrated.to_netcdf(path)

#%%
# climate models: SH integral

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
# list_of_single_level_model_clim_params = ['ps']

# ls_pr_levels_clim_model = pr_levels_model # load in all pressure levels
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
# # d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
# #             "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
# #             "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
# #             "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
# #             "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
# #             "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

# # take only hist for integrated water vapor until surface pressure
# d_model = {"HadGEM": {"folders": ['hist'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist'], "taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }


# for idx in range(0, 8):
#     # idx = 3 # Cerro Tololo

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]
#     site = d_site_lonlat_data['site_name_folder'][idx]

#     surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx] # or 1000 # go until surface (where is this?) #  d_site_lonlat_data['pressure [hPa]'][idx]
#     print(surface_pressure_observation)
#     # check available pressure for climate models
#     absolute_difference_function = lambda list_value : abs(list_value - given_value)

#     # # commented on 2020-08-09 because I wanted to check if my integrated values are the same as the variable 'prw'
#     # given_value = surface_pressure_observation]
#     # closest_value = min(pr_levels_model, key=absolute_difference_function)

#     # SH_integral_pressure = closest_value # find nearest pressure

#     # print('closest match: {}'.format(SH_integral_pressure))


#     for clim_key in d_model.keys():

#         # load surface pressure data... NOT FINISHED!!
#         d_model[clim_key]['ds_surface_pressure'] = climxa.get_PRIMAVERA(d_model, clim_key, site, single_level=True)
#         # convert to hPa
#         d_model[clim_key]['ds_surface_pressure']['ps hist'] = d_model[clim_key]['ds_surface_pressure']['ps hist']/100
#         # select lon/lat
#         d_model[clim_key]['ds_surface_pressure'] = climxa.xr_sel(d_model[clim_key], 'ds_surface_pressure', lon, lat)
#         # calculate the mean (and integrate until this value (otherwise, I cannot shorten the for loop by time))
#         d_model[clim_key]['surface_pressure_mean'] = d_model[clim_key]['ds_surface_pressure']['ps hist'] .mean()
#         print(d_model[clim_key]['surface_pressure_mean'])
#         # also check standard deviation!

#         #  define until which pressure level we integrate
#         given_value = d_model[clim_key]['surface_pressure_mean']
#         # find closest value available in pressure levels
#         closest_value = min(pr_levels_model, key=absolute_difference_function)

#         SH_integral_pressure = closest_value # find nearest pressure

#         print('closest match: {}'.format(SH_integral_pressure))

#         # use function which loads in all specific humidity datasets
#         d_model[clim_key]['ds_hus'] = climxa.get_PRIMAVERA(d_model, clim_key, site, pressure_level=True)

#         # select lon/lat
#         d_model[clim_key]['ds_sel'] = climxa.xr_sel(d_model[clim_key], 'ds_hus', lon, lat)

#         # find maximal index
#         pr_max_idx = pr_levels_model.index(SH_integral_pressure)

#         # initialize dataset
#         ds_q_integrated = xr.Dataset()

#         # initialize list for merging
#         list_forced_profiles = []

#         for forcing in d_model[clim_key]['folders']:
#             # integrate
#             summe, ds_tcw_profile = climxa.Euler_centred_PWV(d_model[clim_key]['ds_sel'], pr_max_idx, 'hus ' + forcing)
#             var_name = "q_integrated " + forcing
#             list_forced_profiles.append(ds_tcw_profile.rename({'q_integrated_profile': var_name}))
#             # write integral to dataset
#             ds_q_integrated["q_integrated " + forcing] = summe

#             # add attributes
#             ds_q_integrated["q_integrated " + forcing].attrs['long_name'] = 'Precipitable Water Vapor'
#             ds_q_integrated["q_integrated " + forcing].attrs['units'] = 'mmH2O' # = 1kg/m^2

#             # add coordinate 'level'
#             ds_q_integrated["q_integrated " + forcing] = ds_q_integrated["q_integrated " + forcing].assign_coords(level=SH_integral_pressure)
#             # add dimension 'level'
#             ds_q_integrated["q_integrated " + forcing] = ds_q_integrated["q_integrated " + forcing].expand_dims(dim='level')
#             # now, the level can be selected with e.g. ds.sel(level=775)

#         # merge list_forced_profiles
#         ds_tcw_profile = xr.merge(list_forced_profiles)

#         # define path
#         # path = './sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(SH_integral_pressure) + 'hPa.nc' # where to save the files
#         path = './sites/'+ site + '/Data/HighResMIP/q_integrated/Amon/' + clim_key + '/ds_' + clim_key + '_q_integrated_monthly_resampled_nearest_' + site + '_' + str(SH_integral_pressure) + 'hPa_surface_pressure.nc' # where to save the files

#         # make directory if not available
#         os.makedirs(os.path.dirname(path), exist_ok=True)

#         # save array to netcdf file, store it
#         ds_q_integrated.to_netcdf(path)


#%% plot tcw profile

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

#%% function for map
# only ERA5 as input, whole grid

# get ERA5 data
# in d_obs['ds_merged']


# # group by year
# time_freq_string = 'time.year'

# d_obs = group_mean(time_freq_string, d_obs, lon, lat, std=False, ds_merged=True, sel=False)
# # take (mean of 1979, 1980, 1981, 1982 and 1983) - (mean of 2015, 2016, 2017, 2018, 2019)
# # idea: calculate rolling mean, then take first and last entry

# #d_obs['ds_mean_year']['t'].rolling(year=5, center=True).mean()
# d_obs['ds_roll_test'] = d_5year_rolling_mean(d_obs['ds_mean_year'])


# (d_obs['ds_roll_test'].sel(level=700, year = 2017)['t']- d_obs['ds_roll_test'].sel(level=700, year = 1981)['t']).plot()
# # plot point of observatory
# lon = -17.89
# lat = 28.75
# plt.plot(lon, lat, 'ko', markersize=5, label = 'TNG')
# plt.text(lon, lat + 0.05, 'TNG: 2370m',
#          horizontalalignment='left')

#%%


# #%% ################################################################################################################

# #%% LA PALMA
# #open in-situ measurements as pandas dataframe

# path_ds = './sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)

# # apply specific changes to dataset
# # add coordinates
# ds_hourly = ds_hourly.assign_coords({'latitude': 28.76, 'longitude': -17.88})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'La_Palma Specific Humidity'})

# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Palma Relative Humidity', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Pressure'])




# #%%
# # main parameters
# site = 'La_Palma'
# ls_pr_levels = [700, 750, 775, 800] # code reads in automatically data for these levels
# # variable = 'T'
# time_slice_var = slice('1998-01-01','2019-12-31')
# list_of_insitu_vars = [site + ' Temperature']
# lon =  -17.88
# lat =  28.76
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # path = './sites/La_Palma/Output/Plots/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# # maybe I have more than one clim model (use nested dict)
# # d_model = {"clim_model_1": {"ds": ds_HadGEM, "clim_var": ['hus'], 'Plev': [700, 850], "name": "Met Office", 'ds_timeseries': ds_Amon}, "clim_model_2": {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [925], "name": 'other office', 'ds_timeseries': ds_Amon}}

# #%% main plotting routine

# fig_La_Palma, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)

# # d_test = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# site='La_Palma'
# fig_La_Palma.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')






# #%% ########################################### siding spring #################################################

# path_ds = './sites/siding_spring/Data/in-situ/Specific_humidity_siding_spring_2003to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  149.07
# lat =  -31.28
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'siding spring Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'siding spring Pressure'])

# #%%
# # main parameters
# site = 'siding_spring' # refers to folder structure
# ls_pr_levels = [850, 875, 900] # code reads in automatically data for these levels

# time_slice_var = slice('2004-01-01','2019-12-31')
# list_of_insitu_vars = ['siding spring Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/'  # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_siding_spring, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# site = 'siding_spring'
# fig_siding_spring.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')

# # %% ###############################################  Mauna Kea ####################################################3333

# path_ds = './sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -155.53
# lat =  19.83
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Mauna Kea Specific Humidity'})
# ds_hourly = ds_hourly.rename_vars({'Temp': 'Mauna Kea Temperature'})


# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['P', 'RH'])


# #%%
# # main parameters
# site = 'MaunaKea' # refers to folder structure
# ls_pr_levels = [600, 650, 700, 750] # code reads in automatically data for these levels

# time_slice_var = slice('2000-01-01','2019-12-31')
# list_of_insitu_vars = ['Mauna Kea Temperature']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# # in this dataset, there are duplicate values. I don't know why, but I am going to remove them
# _, index, counts = np.unique(d_obs['ds']['time'], return_index=True, return_counts=True)

# d_obs['ds'] = d_obs['ds'].isel(time=index)

# #%% main plotting routine

# fig_MaunaKea, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_MaunaKea.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')

# # %% ###################################### SPM ##############################################################

# path_ds = './sites/SPM/Data/in-situ/hourly_meteo/Specific_humidity_SPM_2006to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -115.46 # change!!
# lat =  31.04 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'SPM Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['SPM Pressure', 'SPM Relative Humidity'])

# # %%
# # main parameters
# site = 'SPM' # refers to folder structure
# ls_pr_levels = [700, 750, 775] # code reads in automatically data for these levels

# time_slice_var = slice('2007-01-01','2019-12-31')
# list_of_insitu_vars = ['SPM Temperature']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_SPM, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_SPM.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')




# # %% ######################################### CERRO PARANAL ################################################################################

# path_ds = './sites/Paranal/Data/in-situ/SH_calculated/Specific_humidity_Paranal_ESO_2000to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -70.4 # change!!
# lat =  -24.63 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Cerro Paranal Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Paranal RH 2m', 'Paranal RH -20m', 'Paranal Pressure' ])

# # take time slice manually of T-20m, since it shows very high values in 2006
# ds_hourly['Paranal T -20m'] = ds_hourly['Paranal T -20m'].sel(time = slice('2007-01-01', None))

# # %%
# # main parameters
# site = 'Paranal' # refers to folder structure
# ls_pr_levels = [700, 750, 800] # code reads in automatically data for these levels

# time_slice_var = slice('2000-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Paranal T 2m', 'Paranal T -20m']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_Paranal, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_Paranal.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')



# # %% ############################################ Cerro Tololo #######################################################################

# path_ds = './sites/Cerro_Tololo/Data/in-situ/Specific_humidity_Cerro_Tololo_2002to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -70.81 # change!!,
# lat =  -30.17 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Cerro Tololo Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Cerro Tololo Relative Humidity', 'Cerro Tololo Pressure (hPa)'])

# # %%
# # main parameters
# site = 'Cerro_Tololo' # refers to folder structure
# ls_pr_levels = [750, 775, 800] # code reads in automatically data for these levels

# time_slice_var = slice('2002-01-01','2018-12-31') # CHANGE!
# list_of_insitu_vars = ['Cerro Tololo Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Tololo, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_Tololo.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')





# # %% ############################################ Cerro Pachon #######################################################################

# # no specific humidity data due to pressure missing!
# # worth analysing?




# # %% ############################################ La Silla #######################################################################



# idx = 2

# path_ds = './sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_PWV.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates

# lon =  df_site_lonlat_data['lon_obs'][idx] - 360
# lat =  df_site_lonlat_data['lat_obs'][idx] # Change!!
# # ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# # %%
# # main parameters
# site = 'La_Silla' # refers to folder structure
# ls_pr_levels = None # code reads in automatically data for these levels

# time_slice_var = slice('2001-01-01','2007-12-31') # CHANGE!
# list_of_insitu_vars = ['PWV La Silla']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": None, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine
# importlib.reload(climxa)

# d_obs, d_model, fig_ct = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, diurnal=False, d_model = None)


# fig_ct.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')

# # try np.savez to save figures




# # %% ######################################## Sutherland #######################################################################

# path_ds = './sites/Sutherland/Data/in-situ/hourly_meteo/Specific_humidity_Sutherland.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  20.81 # change!!, ,
# lat =  -32.38 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})


# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Sutherland Relative Humidity', 'Sutherland Pressure', 'Sutherland Specific Humidity'])

# # %%
# # main parameters
# site = 'Sutherland' # refers to folder structure
# ls_pr_levels = [800, 825, 850, 875] # code reads in automatically data for these levels

# time_slice_var = slice('2013-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Sutherland Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Sutherland, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_Sutherland.savefig('./ERA5_validation/' + site + '_' + variable + '_DSCT.pdf')






# %%
