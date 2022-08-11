# add this in future scripts. save d_Ensemble also in Astroclimate outcome!

#        if Ensemble:
            # with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/' + variable + '_d_Ensemble.pkl', "wb") as myfile:
            #     pickle.dump(d_Ensemble, myfile)

# save also the sorted_skill_dict
        # as pickle file
        # with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/' + site_name_folder + '_' + variable + '_d_skill_scores.pkl', "wb") as myfile:
        #     pickle.dump(sorted_skill_dict_ss, myfile)
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

# list_of_single_level_vars = ['surface_seeing'] # ERA5 datasets only for Chilean sites
list_of_single_level_vars = ['wind speed seeing']
list_of_clim_vars = ['seeing']
variable = 'seeing_nc'
# list_of_single_level_model_clim_params = ['seeing']

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
d_site_lonlat_data = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))


#%%

max_idx = 7 # starts at 0 (max is 7)

diurnal=False
Ensemble = True
masterfigure = True
save = True
model = True
nighttime = True
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

    # if idx == 6:
    #     continue

    # CHOOSE FILE NAME
    file_name = '_calibrated_'
    # file_name = ''

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list, model_color) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    # ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]

    list_of_insitu_vars = ['Seeing ' + site_name_folder]

    # pressure level of seeing integration
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_level_seeing'][idx]
    time_slice_var = d_site_lonlat_data['time_slice_var_seeing'][idx] #
    path_seeing = d_site_lonlat_data['path_ds_seeing'][idx] 

    path = './sites/'+ site_name_folder + '/Output/Plots/seeing/' # where to save the files

    # special case of siding_spring, where we have only yearly data:
    if idx == 5:
        df_siding_spring = pd.read_csv(path_seeing, index_col='year', delimiter='\t')
        ds_siding_spring = df_siding_spring.to_xarray()

        # df_siding_spring

        d_obs = {'ds_siding_spring_yearly': ds_siding_spring, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}


    elif idx == 6: # sutherland: no insitu data
        d_obs = {"ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}


    else:
        # read in ds_hourly (in-situ data)
        # ds_hourly = climxa.df_to_xarray('./sites/Paranal/Data/    # attention! taylor folders can change! think about that in the return...in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
        ds_hourly = climxa.df_to_xarray(path_seeing)
        # CHANGE PATH, append d_sites_lon_lat

        ds_hourly_sel = xr.Dataset({list_of_insitu_vars[0]: ds_hourly[list_of_insitu_vars[0]] })
        # if more than one element in list_of_insitu_vars, use 'assign' to add variables to dataset
        if len(list_of_insitu_vars) > 1:
            for var in list_of_insitu_vars[1:]:
                ds_hourly_sel = ds_hourly_sel.assign({var: ds_hourly[var]})

        # mauna kea: shift time by 10
        if idx == 0:
            # ds_hourly_sel['time'] = ds_hourly_sel['time'] + dt.timedelta(hours=10)

            # convert to dataframe
            ds_temp = ds_hourly_sel['Seeing MaunaKea'].to_dataframe()

            # shift index
            ds_temp.index = ds_temp.index + dt.timedelta(hours=10)
            # define shifted ds as new ds_hourly_sel
            ds_hourly_sel = xr.Dataset(ds_temp) # xr.Dataset(data_vars = {'Seeing MaunaKea' : ds_hourly_sel['Seeing MaunaKea'].to_dataframe()}, coords = {'time': ds_hourly_sel['time'].to_index() + dt.timedelta(hours=10)})

            # filter dataset (there are values below zero)
            ds_hourly_sel = ds_hourly_sel.where(ds_hourly_sel['Seeing MaunaKea'] > 0, drop = True)

        # filter all data
        # plotting has shown that a cutoff at 10 arcsec would filter some outliers
        ds_hourly_sel = ds_hourly_sel.where(ds_hourly_sel[list_of_insitu_vars[0]] < 10, drop = True)

        # define observational dict
        # d_obs = {"ds": ds_hourly, "insitu_var": ['MASS_DIMM_Seeing', 'free_atmosphere_seeing'], "single_lev": list_of_single_level_vars}
        # d_obs = {"single_lev": list_of_single_level_vars} # only ERA5
        d_obs = {'ds': ds_hourly_sel, "insitu_var":  list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5, "single_lev": list_of_single_level_vars}


    # # define climate model levels
    if model:
        ls_pr_levels_clim_model = [200] # unfortunately, all pressure levels are saved as 200
        list_of_single_level_model_clim_params = ['wind_speed_seeing']

        # if idx != 4:
            # d_model = {"HadGEM": {"folders": ['present'], "taylor_folder": ['present'],  "name": "HadGEM3-GC31-HM"}}
        d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
                        "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
                        "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], ,  "single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"} ,
                        "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
                        "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
                        "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

        # else: # for la palma, the model 'CNRM' does throw errors
        #     d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
        #                 "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
        #                 "CNRM": {"folders": ['future', 'SSTfuture'], "taylor_folder": [], ,  "single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"} ,
        #                 "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
        #                 "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
        #                 "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],  "single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }


        # # only wind speed seeing (single level)
        # d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "HadGEM3-GC31-HM"},
        #             "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'], "single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"},
        #             "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
        #             "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
        #             "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
        #             "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }
                
  
    else:
        d_model = None

    if Ensemble:
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'],   "single_lev_var": list_of_single_level_model_clim_params}
        # # only wind speed seeing
        # d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "single_lev_var": list_of_single_level_model_clim_params}
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

    importlib.reload(climxa)

    import time
    start_time = time.time() # measure elapsed time

    if masterfigure:
        if idx == 0:
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig, nighttime=nighttime)
        else: # now we have an axis we can refer to fore sharing the xaxis
            d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                        ax_ref = (ax0_t, ax3_t), nighttime=nighttime)
    else:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable, 
                                        time_slice_var, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                        d_model = d_model, d_Ensemble=d_Ensemble, MasterFig=MasterFig, nighttime=nighttime)

    print("--- %s seconds ---" % (time.time() - start_time))

    # 
    if save: # save as .pdf

        # save d_obs
        with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_' +  variable + '_d_obs_ERA5_and_insitu.pkl', "wb") as myfile:
                pickle.dump(d_obs, myfile)

        # save d_model and d_obs
        if model:
            with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/'+ site_name_folder + '_'+ variable + '_d_model.pkl', "wb") as myfile:
                pickle.dump(d_model, myfile)
                
        if masterfigure:
            if idx == max_idx:
                fig.savefig('./Model_evaluation/' + variable + '/'  + variable + file_name + '_overview_Ensemble_DSC_T.pdf', bbox_inches = 'tight', pad_inches=0.0)
        
        if diurnal:
            if idx == max_idx:
                fig_diurnal.savefig('./Model_evaluation/' + variable + '/' + variable + '_overview_Diurnal_ERA5.pdf', bbox_inches = 'tight', pad_inches=0.0)

        # save as .png
        else:
            fig.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400, bbox_inches = 'tight', pad_inches=0.0)
        
        if Ensemble:
            with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/' + variable + '_d_Ensemble.pkl', "wb") as myfile:
                pickle.dump(d_Ensemble, myfile)
            # and save here (needed for 'future_and_past_trends.py')
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
        # and as pickle file
        with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/' + site_name_folder + '_' + variable + '_d_skill_scores.pkl', "wb") as myfile:
            pickle.dump(sorted_skill_dict_ss, myfile)


#%% la palma read pickle file (from when it worked, check differences)

# ds_lapalma = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/uncalibrated_seeing_data/La_Palma_seeing_nc_d_model.pkl", "rb" ))

# # notes: 7, 8 and 9 are missing months
# # ds_Amon looks exactly the same (same entries! although d_model should be calibrated)
# # ds_sel shows different values!
# dataset = 'ds_Amon_Pr'
# d_model['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest').dropna(dim='time') == ds_lapalma['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest').dropna(dim='time')
# # THIS IS TRUE! so integration was not the problem! STOP why is it the same despite the new calibration??
# # BUT: time = 393 IN BOTH DATASETS!

# # plot?
# # test if xarrays are the same: 
# dataset = 'ds_sel'
# d_model['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).dropna(dim='time')

# ds_lapalma['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).dropna(dim='time')

# # somehow, the ds_Amon_Pr is the same, but later it is not! could it be that ds_lapalma was overwritten with the calibrated dataset?
# # because ds_lapalma has 780 entries on ds_sel
# # either it is in climxa or it is in xarray_prepare_seeing_data_PRIMAVERA!
# # fact is, that in d_model, there are way too many nan's. check integration!

# # 'ua hist' also only has 393 values! have a look at it with ncdump

# # ds_sel has no entry that is the same!
# # d_model has 393 entries (nan's dropped), whereas ds_lapalma has 780 entries!?
# # compare times!
# d_model['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).dropna(dim='time').time

# ds_lapalma['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).dropna(dim='time').time
# # now, why are there so many months missing?
# # is integration sometimes zero, because of integration until 1000hPa? check integration

# # groupby ds_Amon month
# dataset = 'ds_Amon_Pr'
# d_model['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest').dropna(dim='time').groupby('time.month').mean()
# ds_lapalma['CNRM'][dataset]['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest').dropna(dim='time').groupby('time.month').mean()


# # is TRUE for all 780 entries!! for HadGEM
# # but for CNRM, it is true and false mixed!! only because of nan?
# # yes! only because of nan!

# d_model['CNRM']['ds_sel']['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest') == ds_lapalma['CNRM']['ds_Amon_Pr']['seeing hist'].sel(time = slice('1950-01-01', '2015-01-01')).sel(level=200, longitude=342.1, latitude=28.76, method='nearest')

# # found problem by setting Ensemble=False
# # in the insitu data, the months 7, 8 and 9 are missing.
# # BUT in the plot of Nov 2, it is NOT missing. INVESTIGATE!!


# dobs_lapalma = pickle.load(open( "/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/uncalibrated_seeing_data/La_Palma_seeing_nc_d_obs_ERA5_and_insitu.pkl", "rb" ))

# d_obs.keys()

# # in ds_mean_month, first and second month are same, afterwards it is different
# # ds_taylor starts in 2008-10-01 for d_obs and 2008-04-01 for dobs_lapalma. why?

# # plot d_obs['ds']
# plt.plot(d_obs['ds']['Seeing La_Palma'].time, d_obs['ds']['Seeing La_Palma'])
# plt.plot(dobs_lapalma['ds']['Seeing La_Palma'].time, dobs_lapalma['ds']['Seeing La_Palma'])

# # check if 'ds' is the same
# d_obs['ds']['Seeing La_Palma'].sel(time=slice('2009-01-01', '2014-12-01')) == dobs_lapalma['ds']['Seeing La_Palma'].sel(time=slice('2009-01-01', '2014-12-01'))

# # drop nan! (there are so many!)

# # groupby month: dobs_lapalma['ds']['Seeing La_Palma'].groupby('time.month').mean(dim='time') and d_obs['ds']['Seeing La_Palma'].groupby('time.month').mean(dim='time')
# # are the same!! and have 12 months
# # --> problem arises somewhere later

# # ds_merged grouped by month is also the same!
# # ds_sel grouped by month also!
# # BUT ds_taylor not the same! 7,8,9 are missing in d_obs.

# # check d_obs ERA5 ds_sel

# # what if I resample monthly?
# # ds_sel still the same!


# # found that dataset is wrong (975hPa vs 1000hPa!) try again with 1000hPa
# # changed nothing despite calibration! months are still missing!

# # set d_model to false --> works! so d_obs is not the problem!
# # it must have something to do with what happens with d_Ensemble!
# # strange thing is: if I set Ensemble to false, 3 months still are missing 
# # --> it must be due to PRIMAVERA seeing data. 
# # could it be related to integration 975 vs. 1000?

# # maybe HadGEM is not the problem! try to find which model has a problem by sorting it out
# # CNRM is the problem!! but why?

# # %% siding spring code (only yearly seeing values are available)

# # add ds_mean_year of siding spring to d_obs['ds_mean_year']
# d_obs['ds_mean_year']['Seeing siding_spring'] = ds_siding_spring['ds_mean_year']
# d_obs['ds_std_year']['Seeing siding_spring'] = ds_siding_spring['ds_std_year']

# d_obs['insitu_var'] = ['Seeing siding_spring']

# importlib.reload(climxa)

# masterfigure = True

# idx = 5
# max_idx = 5



# if masterfigure == True:
#     master_fig = plt.figure( figsize = (25, 4*(max_idx+1))) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
#     gs = master_fig.add_gridspec(max_idx + 1, 4) # first entry: rows, second entry: columns
#     # initialize pressure level list (for legend; plot only those in use)
#     Plev_list = []

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

# if masterfigure:
#     MasterFig = (master_fig, idx, max_idx, gs, Plev_list) 


# # plot
# climxa.xr_plot_cycles_timeseries(d_obs, site, variable, lon, lat, d_model = None, 
#                                 diurnal=False, fig_diurnal=None, d_Ensemble=None, MasterFig=MasterFig, ax_ref = None)



# %%
