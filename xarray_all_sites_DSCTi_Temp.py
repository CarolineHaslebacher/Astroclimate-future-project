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
sns.set()
from matplotlib import cycler

import pickle
import time

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

#%%
#'global parameters

variable = 'T'
list_of_clim_vars = ['t']
list_of_single_level_vars = ['t2m']
list_of_model_clim_params = ['ta']
list_of_single_level_model_clim_params = ['tas']
# there are no pressure levels for 't2m', how to solve? try except?

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

# masterfigure

max_idx = 7 # starts at 0 (max is 7)

diurnal=False
Ensemble = True
masterfigure = True
save = False
model = True
model_color = False

if masterfigure == True:
    # master_fig = plt.figure( figsize = (25, 4*(max_idx+1))) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
    master_fig = plt.figure( figsize = (22, 3.2*(max_idx+1)))
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
#for idx in range(6, 7):
    # or define index for one iteration only
    # idx = 3

    file_name = '_' # at least, type '_' 

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list, model_color) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    # lon = 360-70.2
    lat = d_site_lonlat_data['lat_obs'][idx]

    path_ds_SH_RH_T = d_site_lonlat_data['path_ds_SH_RH_T'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    time_slice_var_meteo = d_site_lonlat_data['time_slice_var_meteo'][idx]
    list_of_insitu_vars = [site_name_folder + ' Temperature']

    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files
    
    # read in ds_hourly (in-situ data) 
    ds_hourly = climxa.df_to_xarray(path_ds_SH_RH_T)

    if idx == 0: # Mauna Kea
        list_of_insitu_vars = ['Mauna Kea Temperature']
        ds_hourly = ds_hourly.rename_vars({'Temp': 'Mauna Kea Temperature'})

        # ls_pr_levels_ERA5 = [600] # change back after layout is fine
    if idx == 1: # Paranal
        list_of_insitu_vars = ['Paranal T 2m', 'Paranal T -20m']
        ds_hourly['Paranal T -20m'] = ds_hourly['Paranal T -20m'].sel(time=slice('2007-01-01', None))
        # ls_pr_levels_ERA5 = [750] # change back after layout is fine
     
    if idx == 2: # La Silla
        list_of_insitu_vars = ['La_Silla T 2m', 'La_Silla T 30m'] # , 'La_Silla T ground' (it doesn't add useful information)
        
        # # crop out missing data in year 2010
        # #ds_hourly['La_Silla T 2m'] = ds_hourly['La_Silla T 2m'].sel(time=slice('2007-01-01', None))
        # ds_hourly['La_Silla T 2m'].sel(time='2010').where(ds_hourly['La_Silla T 2m'] != np.nan)
    
    if idx == 3: # Cerro Tololo
        list_of_insitu_vars = ['Cerro Tololo Temperature']
    if idx == 4: # La Palma
        ls_pr_levels_ERA5 = [700, 750, 775, 800]
    if idx == 5: # siding spring
        list_of_insitu_vars = ['siding spring Temperature']
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

    if idx == 0: # Mauna Kea again
        # in this dataset, there are duplicate values. I don't know why, but I am going to remove them
        _, index, counts = np.unique(d_obs['ds']['time'], return_index=True, return_counts=True)

        d_obs['ds'] = d_obs['ds'].isel(time=index)

    # define model dict
    # change here: folders
    # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
    if model:
        d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
               "EC-Earth": {"folders": ['hist','present', 'future'],"taylor_folder": ['hist', 'present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
                "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
                "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
                "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
                "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"}}
        
    else:
        d_model = None

    if Ensemble:
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params}
    else:
        d_Ensemble = None

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
                                        time_slice_var_meteo, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig)
        else: # now we have an axis we can refer to fore sharing the xaxis
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_meteo, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                        ax_ref = (ax0_t, ax3_t))
    else:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var_meteo, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig)


    print("--- %s seconds ---" % (time.time() - start_time))
    
    # 
    if save: # save as .pdf
        # save d_obs
        with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_d_obs_ERA5_and_insitu.pkl', "wb") as myfile:
                pickle.dump(d_obs, myfile)
        # Mauna Kea HRCM
        # with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_HRCM_d_obs_ERA5_and_insitu.pkl', "wb") as myfile:
        #         pickle.dump(d_obs, myfile)

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
        
    # bottom, top = 0, 1
    # left, right = 0, 1
    # fig.subplots_adjust(hspace=0.2)

    # save dict to csv
    if save:
        # [Skill, ccoef[i], sdev[i], crmsd[i]]
        path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/csv_info/'
        os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
        (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index').to_csv(path_skill_folder + site_name_folder + file_name + 'sorted_skill_dict.csv', header=False))


    # ########## plot trend map
    # path_map = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/maps/'
    # os.makedirs(os.path.dirname(path_map), exist_ok=True) 

                ## WHAT is this doing here? did I copy it by accident? Or worse, cut it?
                    # # append taylor label only if forcing is not already in list
                    # # for folder in d_model[clim_key]['taylor_folder']:
                    # #     if folder not in taylor_label:
                    # #         taylor_label.append(folder)
                    # for folder in d_model[clim_key]['taylor_folder']:
                    #     print(taylor_label)
                    #     if 'atmosphere-land' not in taylor_label:
                    #         if folder == 'present' or folder == 'SSTfuture':
                    #             folder_label = 'atmosphere-land'
                    #             taylor_label.append(folder_label)
                    #             print(taylor_label)
                    #     if 'coupled' not in taylor_label:
                    #         if folder == 'future' or folder == 'SSTfuture':
                    #             folder_label= 'coupled'
                    #             taylor_label.append(folder_label)
                    #             print(taylor_label)
    
    #     d_model = {"HadGEM": {"folders": folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
    #                 "MPI": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
    #                 "CMCC": {"folders": folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
    #                 "ECMWF": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

    #     climxa.trend_map(d_model, d_obs, variable, site_name_folder, idx, path_map, SH_integral=False)

#%% Mauna Kea quickly look at regional model

ds_HRCM = xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/HRCM/t2m/Hawaii_Regional_Climate_Model_Simulations_Hourly_3km_domain_Present_1990_2009_t2m.nc')
ds_HRCM_sel = ds_HRCM.sel(lon=lon, lat=lat, method='nearest')

xr.open_dataset('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/T/single_levels_HRCM/HRCM_2000to2009.nc')

# temperature is in kelvin
ds_HRCM_sel['t2m'] = ds_HRCM_sel['t2m'] - 273.15

ds_HRCM_yearly = ds_HRCM_sel['t2m'].groupby('time.year').mean()
ds_HRCM_monthly = ds_HRCM_sel['t2m'].groupby('time.month').mean()

ax3_t.plot(ds_HRCM_yearly.year, ds_HRCM_yearly)
ax0_t.plot(ds_HRCM_monthly.month, ds_HRCM_monthly)

# save fig
fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + '_HRCM_implanted_DSC_T.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)
        

# add to d_obs as insitu param?

# quickly plot ds_hourly_sel and ds_HRCM

# idea: save HRCM t2m dataset where ERA5 t2m is saved and run code!!
# for this to work, save two different datasets
# longitude and latitude
ds_HRCM = ds_HRCM.rename({'lon': 'longitude', 'lat': 'latitude'})
# drop variables that are not used
ds_HRCM = ds_HRCM.drop(['lat_bnds', 'lon_bnds'])
# first dataset
ds_HRCM_1990 = ds_HRCM.sel(time=slice('1990-01-01', '1999-12-31'))
ds_HRCM_2000 = ds_HRCM.sel(time=slice('2000-01-01', '2009-12-31'))
# save to netcdf
ds_HRCM_1990.to_netcdf('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/HRCM/HRCM_1990to1999.nc')
ds_HRCM_2000.to_netcdf('/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/HRCM/HRCM_2000to2009.nc')


#%% PRESSURE
# # pressure monthly resampling code

# with open('/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/P/Pressure_all_sites_monthly_averaged.csv', 'w') as press_file:

#     linregress_writer = csv.writer(press_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     linregress_writer.writerow(['site', 'mean', 'std'])
            
#     for idx in range(0,8):

#         site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
#         path_ds_SH_RH_T = d_site_lonlat_data['path_ds_SH_RH_T'][idx]

#         ds_hourly = climxa.df_to_xarray(path_ds_SH_RH_T)

#         if idx == 0:
#             ds_hourly_Array = ds_hourly['P']
#         elif idx == 3:
#             ds_hourly_Array = ds_hourly['Cerro Tololo Pressure (hPa)']
#         elif idx == 5:
#             ds_hourly_Array = ds_hourly['siding spring Pressure']
#         else:
#             ds_hourly_Array = ds_hourly[site_name_folder + ' Pressure']
            
#         # resample monthly and take mean and std deviation
#         mean = ds_hourly_Array.resample(time='1m').mean().mean(dim='time')
#         std = ds_hourly_Array.resample(time='1m').mean().std(dim='time')
        
#         linregress_writer.writerow([site_name_folder, str(mean), str(std)])

# for idx in range(0,8):

#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
#     print(site_name_folder)
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]
#     print('lon = {}'.format(lon))
#     print('lat = {}\n'.format(lat))



#%% fill plot during for loop python
# (I solved missbehaving layout by renaming output of xr_plot_cycles_timeseries from fig_ct to fig)
# I do not know why it did not run otherwise..

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
 
# import netCDF4
# import xarray as xr

# import xarray.plot as xplt
# import skill_metrics as sm


# # load xarray dataset
# airtemps = xr.tutorial.open_dataset("air_temperature")
# air = airtemps.air - 273.15
# # air.isel(lat=10, lon=10)

# def plot_fig(MasterFig=None, ax_ref = None):
#     if MasterFig != None:
#         print('MasterFig is getting unpacked!')
#         # MasterFig is a tuple (fig, idx)
#         fig, idx, gs = MasterFig # unpack tuple

#         if ax_ref != None:
#             # take the reference axis
#             ax_ref0, ax_ref3 = ax_ref
#             # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
#             ax0 = fig.add_subplot(gs[idx, 2], sharex=ax_ref0) # seasonal cycle
#             ax3 = fig.add_subplot(gs[idx, 0:-2], sharex=ax_ref3) # span timeseries over two subplot lengths
#             ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram
#         else:
#             # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
#             ax0 = fig.add_subplot(gs[idx, 2]) # seasonal cycle
#             ax3 = fig.add_subplot(gs[idx, 0:-2]) # span timeseries over two subplot lengths
#             ax4 = fig.add_subplot(gs[idx, 3])  # taylor diagram
#     if idx == 0:
#         ax0.plot([10,20,15, 13], [1,2,3,5])

#         ax3.plot([1,3], [6,3])
#         # air.isel(lat=10, lon=10).plot.line("-o" ,ax=ax3)
#         # air.isel(lat=6, lon=15).plot.line(color="purple", marker="o", ax=ax0)
#         # air.isel(lat=10, lon=10).plot.line("-o" ,ax=ax3)

#         # # set in-situ color cycler
#         # insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p']))
        
#         # ax0.set_prop_cycle(insitu_cycler)
#         # ax3.set_prop_cycle(insitu_cycler)

#         # for i, insitu_param in enumerate(d_obs['insitu_var']):
#         #     #seasonal cycle
#         #     # seasonal
#         #     hin = climxa.plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
#         #     # ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
#         #     #                 alpha=.25, facecolor='k')
#         #     # timeseries
#         #     climxa.plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
#         #     # ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
#         #     #                 alpha=.2, facecolor='k')

#         # plt.plot([6,3,4,3], [90,60,40,100]) # this works (last plot is just drawn on last gridspec axis)
        
#         # # ERA5
#         # # get color for P_levels
#         # col = [climxa.P_lev_color(x) for x in d_obs['Plev']]

#         # # set cycler for PRESSURE level data
        
#         # Era5_cycler = (cycler(color = col) * cycler(linestyle=['-']) * cycler(marker = ['*']))

#         # ax0.set_prop_cycle(Era5_cycler)
#         # ax3.set_prop_cycle(Era5_cycler)

#         # # loop through climate variables to plot
        
#         # for clim_var in d_obs['ERA5_var']:
#         #     for P_lev in d_obs['Plev']:  
#         #         # seasonal
#         #         hin = climxa.plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month', P_lev)
#         #         # timeseries
#         #         climxa.plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year', P_lev)

#         # # SINGLE LEVEL

#         # # set cycler for SINGLE level data
#         # col = ['m','lightcoral', 'navy'] # probably not more than 3 different single level variables

#         # Era5_SG_cycler = (cycler(color = col) * cycler(linestyle=['-'])) * plt.cycler(marker='*') # cycler(linestyle=['-']))
#         # ax0.set_prop_cycle(Era5_SG_cycler)

#         # ax3.set_prop_cycle(Era5_SG_cycler)

#         # for clim_var in d_obs['single_lev']:
#         #     # seasonal
#         #     hin = climxa.plot_line(d_obs['ds_mean_month'], clim_var ,ax0, 'month')
#         #     # timeseries
#         #     climxa.plot_line(d_obs['ds_mean_year'], clim_var, ax3, 'year')

#         sdev = [1.,         0.79489189, 0.72909418, 0.67983078, 0.66597742, 0.81855761]
#         crmsd = [0.49512275, 0.35768422, 0.4479255,  0.54590334, 0.56531812, 0.49595067]
#         ccoef = [1.,         0.94598721, 0.91273604, 0.85621265, 0.84382842, 0.86986516]

#         sdev = np.array(sdev)
#         crmsd = np.array(crmsd)
#         ccoef = np.array(ccoef)

#         marD = {'*': {'w': ['red', 'maroon', 'darkorange', 'darkgoldenrod', 'm']}}

#         taylor_label = ['hist', 'present']

#         sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
#                             colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
#                             titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
#                             MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4) 

#         # shrink axis height of taylor diagram
#         box = ax4.get_position()
#         ax4.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

#         # aspect = 0.5
#         # ax4.set_aspect(aspect)

#         ax3.set_xlabel('time [years]')
#         # ax3.set_xticks([1,3,4])
#         # ax3.xaxis.set_visible(False) 
#         plt.setp(ax3.get_xticklabels(), visible=False)
#         ax0.xaxis.set_visible(False) 
#         ax3.title.set_visible(False)

#     else: # for testing sharex (ax3 changed with ax0)
#         ax3.plot([10,20,15, 13], [1,2,3,5])
#         # air.isel(lat=14, lon=10).plot.line("b-^", ax=ax3)
#         # air.isel(lat=4, lon=5).plot.line(color="purple", marker="o", ax=ax0)
#         # # replot
#         air.isel(lat=14, lon=10).plot.line("b-^", ax=ax3)

#         # insitu_cycler = (cycler(color=['k']) * cycler(marker=['o', 'X', 'd', 'p']))
        
#         # ax0.set_prop_cycle(insitu_cycler)
#         # ax3.set_prop_cycle(insitu_cycler)

#         # for i, insitu_param in enumerate(d_obs['insitu_var']):
#         #     #seasonal cycle
#         #     # seasonal
#         #     hin = climxa.plot_line(d_obs['ds_mean_month'], insitu_param , ax0, 'month')
#         #     # ax0.fill_between(d_obs['ds_std_month'].month, (d_obs['ds_mean_month'][insitu_param] - d_obs['ds_std_month'][insitu_param]), (d_obs['ds_mean_month'][insitu_param] + d_obs['ds_std_month'][insitu_param]),
#         #     #                 alpha=.25, facecolor='k')
#         #     # timeseries
#         #     climxa.plot_line(d_obs['ds_mean_year'], insitu_param , ax3, 'year')
#         #     # ax3.fill_between(d_obs['ds_std_year'].year, (d_obs['ds_mean_year'][insitu_param] - d_obs['ds_std_year'][insitu_param]), (d_obs['ds_mean_year'][insitu_param] + d_obs['ds_std_year'][insitu_param]),
#         #     #                 alpha=.2, facecolor='k')

#         ax0.plot([1,3], [6,3])
        
#         # this works! I don't understand why it doesn't work for the taylor diagram
#         #cax = plt.gca()
#         #print(cax)
#         #cax.plot([6,3,4,3], [90,60,40,100]) # this works (last plot is just drawn on last gridspec axis)
        
#         # minimal example for taylor diagram
#         sdev = [1.,         0.79489189, 0.72909418, 0.67983078, 0.66597742, 0.81855761]
#         crmsd = [0.49512275, 0.35768422, 0.4479255,  0.54590334, 0.56531812, 0.49595067]
#         ccoef = [1.,         0.94598721, 0.91273604, 0.85621265, 0.84382842, 0.86986516]

#         sdev = np.array(sdev)
#         crmsd = np.array(crmsd)
#         ccoef = np.array(ccoef)

#         marD = {'*': {'w': ['red', 'maroon', 'darkorange', 'darkgoldenrod', 'm']}}

#         taylor_label = ['hist', 'present']

#         sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
#                             colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
#                             titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
#                             MarkerDictCH=marD, alpha=0.7, markerSize= 9, my_axis=ax4) 

#         # shrink axis
#         box = ax4.get_position()
#         ax4.set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

#         plt.setp(ax3.get_xticklabels(), visible=False)

#     # bottom, top = 0.1, 0.95
#     # left, right = 0.1, 0.8
#     # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)


#     # cbar_ax = fig.add_axes([0.85, bottom, 0.05, top-bottom])
#     # cbar_ax = fig.colorbar(orography, cax=cbar    # cbar_ax = fig.add_axes([0.85, bottom, 0.05, top-bottom])
#     # cbar_ax = fig.colorbar(orography, cax=cbar_ax)  # plot colorbar_ax)  # plot colorbar

#     return fig, ax0, ax3


# def tf_fig(MasterFig=None, ax_ref=None):

#     fig, ax0, ax3 = plot_fig(MasterFig=MasterFig, ax_ref=ax_ref)

#     return fig, ax0, ax3

# master_fig = plt.figure( figsize = (17, 8)) #,constrained_layout=True)
# gs = master_fig.add_gridspec(3, 4) #, width_ratios=[1, 1, 1, 1.3])

# for idx in range(0,3):
#     MasterFig = (master_fig, idx, gs)
#     if idx == 0:
#         fig_t, ax0_t, ax3_t = tf_fig(MasterFig=MasterFig)
#     else:
#         fig_t, ax0_t, ax3_t = tf_fig(MasterFig=MasterFig, ax_ref=(ax0_t, ax3_t))
    
# # gs.tight_layout(fig_t)


#%% pass axis to new figure (works, but shifts whole figure)

# ax0.remove()

# fig2 = plt.figure( figsize = (20, 5),constrained_layout=True)
# gs = fig2.add_gridspec(1, 4)
# # do not create an axis with  name 'ax1' (where diurnal cycle is plotted normally)
# ax_seasonal = fig2.add_subplot(gs[0, 2]) # seasonal cycle
# ax_timeseries = fig2.add_subplot(gs[0, 0:-2]) # span timeseries over two subplot lengths
# ax_taylor = fig2.add_subplot(gs[0, 3])  # taylor diagram


# ax0.figure=fig2
# fig2.axes.append(ax0)



# # dummy = fig2.add_subplot(111)
# ax0.set_position(ax_seasonal.get_position())
# fig2.add_axes(ax0)
# ax_seasonal.remove()

# plt.show()


#%% ensemble of d_model, try out here

# sum_year = 0
# sum_month = 0
# counter = 0

# def calc_d_model_ensemble(d_model, d_Ensemble):
    
#     # empty list for merging all forcings together
#     ensemble_mean_year = []
#     ensemble_mean_month = []

#     # pressure level variables

#     # write 'clim_var' to be plotted with the ensemble into d_Ensemble 
#     if 'clim_var' in d_Ensemble.keys():
#         for clim_var in d_Ensemble['clim_var']:
#             # go through forcings
#             for forc_folder in d_Ensemble['folders']:
#                 variable = clim_var + ' ' + forc_folder
#                 # empty list for calculating mean
#                 av_ls_year = []
#                 av_ls_month = []

#                 for clim_key in d_model.keys():
#                     # only include in mean if the simulation of the forcing really exists in this dataset
#                     if forc_folder in d_model[clim_key]['folders']:
#                         av_ls_year.append(d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year'))
#                         av_ls_month.append(d_model[clim_key]['ds_mean_month'][variable])

#                 # assign and level (all the same)
#                 level = d_Ensemble['Plev']
#                 if forc_folder == 'hist' or forc_folder == 'present':
#                     year = np.arange(1950,2015,1)
#                 elif forc_folder == 'future' or forc_folder == 'SSTfuture':
#                     year = np.arange(2015,2051,1)
#                 else:
#                     raise Exception('array of year cannot be defined.')

#                 month = np.arange(1,13,1) # [1,2,3,4,5,6,7,8,9,10,11,12]

#                 # calculate Ensemble mean
#                 np_mean_year = np.mean(av_ls_year, axis=0)
#                 np_mean_month = np.mean(av_ls_month, axis=0)

#                 ds_forc_year = xr.Dataset({variable + ' mean':  (("year", "level"), np_mean_year) , variable + ' std': (("year", "level"),np.std(av_ls_year, axis=0) )}, coords={'level': level, 'year': year })
#                 ds_forc_month = xr.Dataset({variable + ' mean':  (("month", "level"), np_mean_month), variable + ' std': (("year", "level"),np.std(av_ls_month, axis=0) )}, coords={'level': level, 'month': month })
                    
#                 ensemble_mean_year.append(ds_forc_year)
#                 ensemble_mean_month.append(ds_forc_month)

#     # single level variables (no 'level', otherwise same)
#     if 'single_lev_var' in d_Ensemble.keys():
#         for clim_var in d_Ensemble['single_lev_var']:
#             # go through forcings
#             for forc_folder in d_Ensemble['folders']:
#                 variable = clim_var + ' ' + forc_folder
#                 # empty list for calculating mean
#                 av_ls_year = []
#                 av_ls_month = []

#                 for clim_key in d_model.keys():
#                     # only include in mean if the simulation of the forcing really exists in this dataset
#                     if forc_folder in d_model[clim_key]['folders']:
#                         av_ls_year.append(d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year'))
#                         av_ls_month.append(d_model[clim_key]['ds_mean_month'][variable])

#                 # assign and level (all the same)
#                 if forc_folder == 'hist' or forc_folder == 'present':
#                     year = np.arange(1950,2015,1)
#                 elif forc_folder == 'future' or forc_folder == 'SSTfuture':
#                     year = np.arange(2015,2051,1)
#                 else:
#                     raise Exception('array of year cannot be defined.')

#                 month = np.arange(1,13,1) # [1,2,3,4,5,6,7,8,9,10,11,12]

#                 # calculate Ensemble mean
#                 np_mean_year = np.mean(av_ls_year, axis=0)
#                 np_mean_month = np.mean(av_ls_month, axis=0)

#                 ds_forc_year = xr.Dataset({variable + ' mean':  (("year"), np_mean_year) , variable + ' std': (("year"), np.std(av_ls_year, axis=0) )}, coords={'year': year })
#                 ds_forc_month = xr.Dataset({variable + ' mean':  (("month",), np_mean_month), variable + ' std': (("year"), np.std(av_ls_month, axis=0) )}, coords={'month': month })
                    
#                 ensemble_mean_year.append(ds_forc_year)
#                 ensemble_mean_month.append(ds_forc_month)    


#     d_Ensemble['ds_ensemble_mean_year'] = xr.merge(ensemble_mean_year)
#     d_Ensemble['ds_ensemble_mean_month'] = xr.merge(ensemble_mean_month) 

#     return

#     # single level variables

#     for clim_key in d_model.keys():
#         variable = 'ta hist'
#         # append sum
#         print(clim_key)
#         av_ls.append(d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year'))

#         # sum_year += d_model[clim_key]['ds_mean_year'][variable].dropna(dim='year')
#         # sum_month += d_model[clim_key]['ds_mean_month'][variable].dropna(dim='month')
#         # counter += 1

#     average_year = sum_year/counter

#     np.std(av_ls, axis=0)


# plt.plot(average_year.year, average_year.sel(level=700))

# d_model['HadGEM']['ds_mean_year']['ta hist'] + d_model['EC-Earth']['ds_mean_year']['ta hist']


#%%


#%% ################################################################################################################

# #%% LA PALMA
# #open in-situ measurements as pandas dataframe

# path_ds = './sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)

# # apply specific changes to dataset
# # add coordinates
# ds_hourly = ds_hourly.assign_coords({'latitude': 28.76, 'longitude': -17.88})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'La_Palma Specific Humidity'})

# # ds_hourly['La_Palma Relative Humidity'].attrs['units'] = '%'
# # ds_hourly['La_Palma Pressure'].attrs['units'] = 'hPa'
# # ds_hourly['La_Palma Specific Humidity'].attrs['units'] = 'kg/kg'
# # ds_hourly['La_Palma Temperature'].attrs['units'] = '$^{\circ}$C'

# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Palma Relative Humidity', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Pressure'])


# #%% read in HadGEM data
# ds_HadGEM = xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_3hr_La_Palma_grid.nc')
# ds_HadGEM = ds_HadGEM.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_HadGEM['longitude'] = ds_HadGEM['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_HadGEM['level'] = ds_HadGEM['level']/100

# # ds_sel = ds_HadGEM.sel(longitude= lon, latitude = lat, method = 'nearest')
# # ds_sel_res = ds_sel.groupby('time.month').mean(dim='time')

# # read Amon data from HadGEM
# ds_Amon = xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_Amon_La_Palma_lon342_2_lat28_7_full.nc')
# ds_Amon = ds_Amon.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_Amon['longitude'] = ds_Amon['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_Amon['level'] = ds_Amon['level']/100


# #%%
# # main parameters
# site = 'La_Palma'
# ls_pr_levels = [700, 750, 775, 800] # code reads in automatically data for these levels
# # variable = 'T'
# time_slice_var_meteo = slice('1998-01-01','2019-12-31')
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

# fig_La_Palma, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)

# # d_test = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# site='La_Palma'
# fig_La_Palma.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')



# #%% function for map
# # only ERA5 as input, whole grid

# # get ERA5 data    
# # in d_obs['ds_merged']


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

# time_slice_var_meteo = slice('2004-01-01','2019-12-31')
# list_of_insitu_vars = ['siding spring Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/'  # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_siding_spring, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


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

# time_slice_var_meteo = slice('2000-01-01','2019-12-31')
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

# fig_MaunaKea, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


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

# time_slice_var_meteo = slice('2007-01-01','2019-12-31')
# list_of_insitu_vars = ['SPM Temperature']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_SPM, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


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

# time_slice_var_meteo = slice('2000-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Paranal T 2m', 'Paranal T -20m']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_Paranal, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


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

# time_slice_var_meteo = slice('2002-01-01','2018-12-31') # CHANGE!
# list_of_insitu_vars = ['Cerro Tololo Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Tololo, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# fig_Tololo.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')





# # %% ############################################ Cerro Pachon #######################################################################

# # no specific humidity data due to pressure missing!
# # worth analysing?




# # %% ############################################ La Silla #######################################################################

# path_ds = './sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_30m_La_Silla_ESO_2000to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -70.74 # change!!, , 
# lat =  -29.26 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'La Silla Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Silla RH 2m', 'La_Silla Pressure'])

# # %%
# # main parameters
# site = 'La_Silla' # refers to folder structure
# ls_pr_levels = [750, 775, 800] # code reads in automatically data for these levels

# time_slice_var_meteo = slice('2000-01-01','2015-12-31') # CHANGE!
# list_of_insitu_vars = ['La_Silla T 30m', 'La_Silla T 2m', 'La_Silla T ground']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_LaSilla, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# fig_LaSilla.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')

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

# time_slice_var_meteo = slice('2013-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Sutherland Temperature']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Sutherland, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# fig_Sutherland.savefig('./ERA5_validation/' + site + '_' + variable + '_DSCT.pdf')






# %%
