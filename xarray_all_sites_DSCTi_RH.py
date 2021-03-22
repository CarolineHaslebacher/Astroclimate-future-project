# 2020-05-07
# attempt to optimize Astroclimate_function pool by only using xarray Datasets (in the outlook of climate model data (PRIMAVERA))
# goal is to have 1 script for 1 site, and all parameters. 
# functions should be as flexible as possible

#%%
import os
import psutil
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


import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa


# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


#%% RELOAD CLIMXA

import importlib
importlib.reload(climxa)

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#%% global parameters

list_of_single_level_vars = None
variable = 'RH'
list_of_clim_vars = ['r']
list_of_model_clim_params = ['hur']

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

max_idx = 7 # starts at 0 (max is 7)

diurnal=False
Ensemble = True
masterfigure = False
save = True
model = True

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
#for idx in range(3,4):

    # or define index for one iteration only
    # idx = 3

    file_name = '_' # at least, type '_' 

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    # Mauna Kea, other gridpoint
    # lat = 19.50

    path_ds_SH_RH_T = d_site_lonlat_data['path_ds_SH_RH_T'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    time_slice_var_meteo = d_site_lonlat_data['time_slice_var_meteo'][idx]
    list_of_insitu_vars = [site_name_folder + ' Relative Humidity']

    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files
    
    # read in ds_hourly (in-situ data)
    ds_hourly = climxa.df_to_xarray(path_ds_SH_RH_T)
    if idx == 0: # for Mauna Kea
        list_of_insitu_vars = ['Mauna Kea Relative Humidity']
        ds_hourly = ds_hourly.rename_vars({'RH': 'Mauna Kea Relative Humidity'})
    if idx == 1: # Paranal
        list_of_insitu_vars = ['Paranal RH 2m', 'Paranal RH -20m'] # solve problems here
    # ds_hourly = ds_hourly.rename_vars({'specific_humidity': list_of_insitu_vars[0]})
    if idx == 2: # La Silla
        list_of_insitu_vars = ['La_Silla RH 2m']
    if idx == 3: # Cerro Tololo
        list_of_insitu_vars = ['Cerro Tololo Relative Humidity']
    if idx == 5: # siding spring
        list_of_insitu_vars = ['siding spring Relative Humidity']

    # instead of dropping variables, like this:
    # ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Palma Relative Humidity', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Temperature', 'La_Palma Pressure'])    
    # better compose new dataset with list_of_insitu_vars
    
    ds_hourly_sel = xr.Dataset({list_of_insitu_vars[0]: ds_hourly[list_of_insitu_vars[0]] })
    # if more than one element in list_of_insitu_vars, use 'assign' to add variables to dataset
    if len(list_of_insitu_vars) > 1:
        for var in list_of_insitu_vars[1:]:
            ds_hourly_sel = ds_hourly_sel.assign({var: ds_hourly[var]})


    # define observational dict
    d_obs = {"ds": ds_hourly_sel, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels_ERA5}

    if idx == 0: # Mauna Kea again
        # in this dataset, there are duplicate values. I don't know why, but I am going to remove them
        _, index, counts = np.unique(d_obs['ds']['time'], return_index=True, return_counts=True)

        d_obs['ds'] = d_obs['ds'].isel(time=index)

    # define model dict
    # change here: folders
    # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
    if model:
        d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                    "EC-Earth": {"folders": ['hist', 'present', 'future'],"taylor_folder": ['hist', 'present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }
        
        # For sutherland, testing skill scores for future simulations
        # d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['future','SSTfuture'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
        #        "EC-Earth": {"folders": ['hist','present', 'future'],"taylor_folder": ['future'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"},
        #         "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['future','SSTfuture'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
        #         #"MPI": {"folders": ['hist', 'present'],"taylor_folder": [] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
        #         "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['future','SSTfuture'] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"}}#,
        #         # "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": [] ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"}}
    
    
    else:
        d_model = None

    if Ensemble:
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model}
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
                
    # # save as png with high resolution. Only save as pdf in the 'last round'
    # fig_ss.savefig('./Model_evaluation/' + variable + '/'  + site_name_folder + '_' + variable + file_name + '_DSC_T.png', dpi=400)

        # save dict to csv
        # [Skill, ccoef[i], sdev[i], crmsd[i]]
        path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/csv_info/'
        os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True) 
        (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
            .to_csv(path_skill_folder + site_name_folder + file_name + 'sorted_skill_dict.csv', header=False))

    # ########## plot trend map
    # path_map = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/maps/'
    # os.makedirs(os.path.dirname(path_map), exist_ok=True) 

    # folders_list = [['hist'], ['present'], ['future'], ['SSTfuture']]
    # for folders in folders_list:
    
    #     d_model = {"HadGEM": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
    #             "EC-Earth": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"},
    #                 "CNRM": {"folders":folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
    #                 "MPI": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"name": "MPI-ESM1-2-XR"},
    #                 "CMCC": {"folders": folders ,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model,"name": "CMCC-CM2-VHR4"},
    #                 "ECMWF": {"folders": folders,"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

    #     climxa.trend_map(d_model, d_obs, variable, site_name_folder, idx, path_map, SH_integral=False)





#%%

#%% investigate error behaviour

# # ds_taylor must have been created
# # test for Mauna Kea, RH, 5 years 
# time_freq_string = 'time.month'
# d_test_error = d_obs['ds_taylor']['Mauna Ke/home/haslebacher/chaldene/Astroclimate_Project/sites/xarray_all_sites_DSCTi_RH.pya Relative Humidity'].sel(time=slice('2000-01-01', '2004-12-31'))

# d_test = climxa.group_mean(time_freq_string, d_obs, d_test_error, std=True)

# d_test['ds_mean_month']
# d_test['ds_std_month']

# df = d_test_error.to_dataframe()

# # test passed!

# # test ds_std_year
# time_freq_string = 'time.year'

# d_test_error = d_obs['ds_taylor']['Mauna Kea Relative Humidity'].sel(time=slice('2000-01-01', '2004-12-31'))

# d_test = climxa.group_mean(time_freq_string, d_obs, d_test_error, std=True)

# d_test['ds_mean_year']
# d_test['ds_std_year']

# df = d_test_error.to_dataframe()

# summ = df['Mauna Kea Relative Humidity'][0] + df['Mauna Kea Relative Humidity'][1] + df['Mauna Kea Relative Humidity'][2] + df['Mauna Kea Relative Humidity'][3] + df['Mauna Kea Relative Humidity'][4] + df['Mauna Kea Relative Humidity'][5] + df['Mauna Kea Relative Humidity'][6] + df['Mauna Kea Relative Humidity'][7] + df['Mauna Kea Relative Humidity'][8] + df['Mauna Kea Relative Humidity'][9] + df['Mauna Kea Relative Humidity'][10] + df['Mauna Kea Relative Humidity'][11]
 
# mean = summ/12 # = 31.8031001051871

# # std
# sums = (df['Mauna Kea Relative Humidity'][0]-mean)**2 + (df['Mauna Kea Relative Humidity'][1]- mean)**2 + (df['Mauna Kea Relative Humidity'][2]- mean)**2 + (df['Mauna Kea Relative Humidity'][3]- mean)**2 + (df['Mauna Kea Relative Humidity'][4]- mean)**2 + (df['Mauna Kea Relative Humidity'][5]- mean)**2 + (df['Mauna Kea Relative Humidity'][6]- mean)**2 + (df['Mauna Kea Relative Humidity'][7]- mean)**2 + (df['Mauna Kea Relative Humidity'][8]- mean)**2 + (df['Mauna Kea Relative Humidity'][9]- mean)**2 + (df['Mauna Kea Relative Humidity'][10]- mean)**2 + (df['Mauna Kea Relative Humidity'][11]- mean)**2
 
# std = np.sqrt(sums/12) # = 8.652417582150214

# # test passed!

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

# # ds_hourly['La_Palma Relative Humidity'].attrs['units'] = '%'
# # ds_hourly['La_Palma Pressure'].attrs['units'] = 'hPa'
# # ds_hourly['La_Palma Specific Humidity'].attrs['units'] = 'kg/kg'
# # ds_hourly['La_Palma Temperature'].attrs['units'] = '$^{\circ}$C'

# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Temperature', 'La_Palma Pressure'])

# #%%

# from pyesgf.search import SearchConnection
# import xarray as xr
# import os

# from pyesgf.logon import LogonManager



# # experiment_id : 'highresSST-present'
# # source_id ='HadGEM3-GC31-HM'
# def get_clim_model_ds(variable, frequency, experiment_id, source_id, table_id = None):
#     # https://esgf-pyclient.readthedocs.io/en/latest/notebooks/demo/subset-cmip6.html
#     # username: chaslebacher
#     # password: see notebook
#     conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)


#     ctx = conn.new_context(
#         project='CMIP6',
#         source_id=source_id,
#         experiment_id=experiment_id,
#         variable= variable,
#         frequency=frequency,
#         table_id=table_id,
#         variant_label='r1i1p1f1',
#         data_node='esgf-data3.ceda.ac.uk')

#     print('frequencies:', ctx.facet_counts['frequency'])
#     print('table ID:', ctx.facet_counts['table_id'])
#     ctx.hit_count

#     #%%
#     result = ctx.search()[0]
#     result.dataset_id

#     #%%
#     files = result.file_context().search()

#     # ds_agg = xr.open_mfdataset([myfile.opendap_url for myfile in files], chunks={'time': 120}, combine='nested', concat_dim='time')
#     ds_agg = xr.open_mfdataset([myfile.opendap_url for myfile in files], chunks={'time': 120}, combine='nested', concat_dim='time')
    
#     return ds_agg

# def modify_clim_ds(clim_ds):
#     clim_ds = clim_ds.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
#     clim_ds['longitude'] = clim_ds['longitude'] - 360 # to make it identical to ERA 5 data
#     #  pr level to hPa
#     clim_ds['level'] = clim_ds['level']/100
#     clim_ds = clim_ds.drop_vars(['lat_bnds', 'lon_bnds'])

#     return clim_ds

# #%%

# # make sure you're logged on and start aggregation
# lm = LogonManager()
# if lm.is_logged_on() == False:

#     # log on with openID
#     OPENID = 'https://ceda.ac.uk/openid/Caroline.Haslebacher'

#     lm.logon_with_openid(openid=OPENID, password=None, bootstrap=True)
#     # home node: ESGF-INDEX1.CEDA.AC.UK
#     lm.is_logged_on()

# variable = 'hur'
# frequency = '3hr'
# experiment_id = 'highresSST-present'
# source_id ='HadGEM3-GC31-HM'

# ds_HadGEM_SST_present = get_clim_model_ds(variable, frequency, experiment_id, source_id, table_id = None)

# ds_HadGEM_SST_present = modify_clim_ds(ds_HadGEM_SST_present)

# frequency = 'mon'
# table_id = 'Amon'
# ds_Amon_SST_present = get_clim_model_ds(variable, frequency, experiment_id, source_id, table_id = table_id)

# ds_Amon_SST_present = modify_clim_ds(ds_Amon_SST_present)



# #%%

# variable = 'hur'
# frequency = '3hr'
# experiment_id = 'hist-1950'
# source_id ='HadGEM3-GC31-HM'

# ds_Had_GEM_hist = get_clim_model_ds(variable, frequency, experiment_id, source_id, table_id = None)

# ds_Had_GEM_hist = modify_clim_ds(ds_Had_GEM_hist)

# experiment_id = 'hist-1950'
# source_id ='HadGEM3-GC31-HM'
# frequency = 'mon'
# table_id = 'Amon'

# ds_Amon_hist = get_clim_model_ds(variable, frequency, experiment_id, source_id, table_id = table_id)

# # ds_Amon_hist = modify_clim_ds(ds_Amon_hist)

# #%% check why netcdf access failure



# ds_sel = ds_Had_GEM_hist.sel(plev=[70000],lat=lon, lon = lat, method='nearest')

# ds_loaded = ds_sel.load()
# # try resample

# #%% source_id ='EC-Earth3P-HR'
# #source_id ='EC-Earth3P-HR'




# #%% read in HadGEM data

# # ds_HadGEM = xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_3hr_La_Palma_grid.nc')
# # ds_HadGEM = ds_HadGEM.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# # ds_HadGEM['longitude'] = ds_HadGEM['longitude'] - 360 # to make it identical to ERA 5 data
# # #  pr level to hPa
# # ds_HadGEM['level'] = ds_HadGEM['level']/100

# # ds_sel = ds_HadGEM.sel(longitude= lon, latitude = lat, method = 'nearest')
# # ds_sel_res = ds_sel.groupby('time.month').mean(dim='time')

# #%%
# # read Amon data from HadGEM
# ds_Amon = xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_Amon_La_Palma_lon342_2_lat28_7_full.nc')
# ds_Amon = ds_Amon.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_Amon['longitude'] = ds_Amon['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_Amon['level'] = ds_Amon['level']/100
# ds_Amon['level'] = ds_Amon['level'].astype('int64')
# ds_Amon['level'].attrs['units'] = 'hPa'

# #%%
# # # prepare ds_agg
# # ds_agg = ds_agg.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})

# # ds_agg['longitude'] = ds_agg['longitude'] - 360
# # #  pr level to hPa
# # ds_agg['level'] = ds_agg['level']/100


# #%%
# # main parameters
# site = 'La_Palma'
# ls_pr_levels = [700, 750, 775, 800, 850, 900, 950] # code reads in automatically data for these levels
# variable = 'RH'
# time_slice_var_meteo = slice('1998-01-01','2019-12-31')
# list_of_insitu_vars = ['La_Palma Relative Humidity']
# list_of_single_level_vars = None
# list_of_clim_vars = ['r']
# lon =  -17.88
# lat =  28.76
# path = './sites/'+ site + '/Output/Plots/' + variable + '/ds_agg_directly_HadGEM_present_hist' # where to save the files
# # ds_clim_model_param = ds_agg # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# d_model = None

# #%%
# # maybe I have more than one clim model (use nested dict)
# d_model = {"clim_model_1": {"ds": ds_HadGEM_SST_present, "clim_var": ['hus'], 'Plev': [700, 850], "name": "Met Office, SST-present", 'ds_timeseries': ds_Amon_SST_present}, "clim_model_2": {"ds": ds_Had_GEM_hist, "clim_var": ['hus'], 'Plev': [700, 850], "name": "Met Office, hist-1950", 'ds_timeseries': ds_Amon_hist}}

# #%% main plotting routine

# fig_La_Palma, ax0_Palma, ax1_Palma, ax3_Palma = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = d_model)


# fig_La_Palma.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')

# # save to netcdf (or save dict as pickle)


# #%% test taylor plot

# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import skill_metrics as sm
# from sys import version_info




# ds_taylor = d_obs['ds'].resample(time = '1m').mean().where(xr.ufuncs.isnan(d_obs['ds']['La_Palma Relative Humidity']) != True, drop = True)

# # take only year and month from datetime index (because in ds_Amon, the value is assigned to YYYY-MM-16)
# ds_taylor['time'] = ds_taylor['time'].dt.strftime('%Y-%m')
# ds_taylor['time'] = pd.to_datetime(ds_taylor.indexes['time'])

# ds_Amon['time'] = ds_Amon['time'].dt.strftime('%Y-%m')
# ds_Amon['time'] = pd.to_datetime(ds_Amon.indexes['time'])

# ds_Ty_merged = xr.merge([ds_taylor, ds_Amon], join='outer').where(xr.ufuncs.isnan(ds_taylor['La_Palma Relative Humidity']) != True, drop = True).where(xr.ufuncs.isnan(ds_taylor['r']) != True, drop = True).where(xr.ufuncs.isnan(ds_Amon['hus']) != True, drop = True)


# ref = {"data" : ds_Ty_merged['La_Palma Relative Humidity']}
# pred1 = {"data" : ds_Ty_merged['r'].sel(latitude = lat, longitude = lon, level = 800, method = 'nearest')}
# pred2 = {"data" : ds_Ty_merged['r'].sel(latitude = lat, longitude = lon, level = 850, method = 'nearest')}
# pred3 = {"data" : ds_taylor['r'].sel(time=ds_Amon['time'].dt.strftime('%Y-%m'), method='nearest').sel(latitude = lat, longitude = lon, level = 900, method = 'nearest')}
# # pred4 = {"data" : ds_taylor['r'].sel(latitude = lat, longitude = lon, level = 700, method = 'nearest')}
# # hadGEM





# #slice('1998-01', '2019-12')
# pred4 = {"data": ds_Ty_merged.hus.sel(level = 850.0, time = ds_Ty_merged.time)}
# # # only needed when convertion to datetime index is needed for plotting on the same axis
# # # datetimeindex = ds_res_model.indexes['time'].to_datetimeindex()
# # # ds_res_model['time'] = datetimeindex

# # Calculate statistics for Taylor diagram
# # The first array element (e.g. taylor_stats1[0]) corresponds to the 
# # reference series while the second and subsequent elements
# # (e.g. taylor_stats1[1:]) are those for the predicted series.
# taylor_stats1 = sm.taylor_statistics(pred1, ref,'data')
# taylor_stats2 = sm.taylor_statistics(pred2, ref,'data')
# taylor_stats3 = sm.taylor_statistics(pred3, ref,'data')
# taylor_stats4 = sm.taylor_statistics(pred4, ref,'data')

# # Store statistics in arrays
# sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], 
#                     taylor_stats2['sdev'][1], taylor_stats3['sdev'][1], taylor_stats4['sdev'][1]])
# crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], 
#                     taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1], taylor_stats4['crmsd'][1]])
# ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], 
#                     taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], taylor_stats3['ccoef'][1]])

# '''
# Produce the Taylor diagram
# Note that the first index corresponds to the reference series for 
# the diagram. For example sdev[0] is the standard deviation of the 
# reference series and sdev[1:4] are the standard deviations of the 
# other 3 series. The value of sdev[0] is used to define the origin 
# of the RMSD contours. The other values are used to plot the points 
# (total of 3) that appear in the diagram.
# '''
# label = ['obs','800hPa', '850hPa', '900hPa', 'hadgem']

# sm.taylor_diagram(sdev,crmsd,ccoef,markerLabel = label, styleOBS = '-', 
#                     colOBS = 'k', markerobs = 'o', 
#                     titleOBS = 'observation')
# # I should try to also plot the confidence interval and the p-value (errorbar and color of errorbar for example)

# # Write plot to file
# plt.savefig('./ERA5_validation/' + site + '_' + variable + '_taylorPlot.pdf')


# # Show plot
# plt.show()


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
# ls_pr_levels = [850, 875, 900, 950, 1000] # code reads in automatically data for these levels
# variable = 'RH'
# time_slice_var_meteo = slice('2004-01-01','2019-12-31')
# list_of_insitu_vars = ['siding spring Relative Humidity']
# list_of_clim_vars = ['r']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": None}

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
# ds_hourly = ds_hourly.rename_vars({'RH': 'Mauna Kea Relative Humidity'})


# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Temp', 'P',])


# #%%
# # main parameters
# site = 'MaunaKea' # refers to folder structure
# ls_pr_levels = [600, 650, 700, 750] # code reads in automatically data for these levels

# time_slice_var_meteo = slice('2000-01-01','2019-12-31')
# list_of_insitu_vars = ['Mauna Kea Relative Humidity']

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
# ds_hourly = ds_hourly.drop_vars(['SPM Pressure', 'SPM Temperature'])

# # %%
# # main parameters
# site = 'SPM' # refers to folder structure
# ls_pr_levels = [700, 750, 775, 850, 875] # code reads in automatically data for these levels

# time_slice_var_meteo = slice('2007-01-01','2019-12-31')
# list_of_insitu_vars = ['SPM Relative Humidity']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/'# where to save the files
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
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Paranal T 2m', 'Paranal T -20m', 'Paranal Pressure' ])

# # %%
# # main parameters
# site = 'Paranal' # refers to folder structure
# ls_pr_levels = [700, 750, 800] # code reads in automatically data for these levels
# variable = 'RH'
# time_slice_var_meteo = slice('2000-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Paranal RH 2m', 'Paranal RH -20m']
# list_of_clim_vars = ['r']
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
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Cerro Tololo Temperature', 'Cerro Tololo Pressure (hPa)'])

# # %%
# # main parameters
# site = 'Cerro_Tololo' # refers to folder structure
# ls_pr_levels = [650, 700, 750, 775, 800] # code reads in automatically data for these levels
# variable = 'RH'
# time_slice_var_meteo = slice('2002-01-01','2018-12-31') # CHANGE!
# list_of_insitu_vars = ['Cerro Tololo Relative Humidity']
# list_of_clim_vars = ['r']
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
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Silla T 30m', 'La_Silla T 2m', 'La_Silla T ground', 'La_Silla Pressure'])

# # %%
# # main parameters
# site = 'La_Silla' # refers to folder structure
# ls_pr_levels = [750, 775, 800] # code reads in automatically data for these levels
# variable = 'RH'
# time_slice_var_meteo = slice('2000-01-01','2015-12-31') # CHANGE!
# list_of_insitu_vars = ['La_Silla RH 2m']
# list_of_clim_vars = ['r']
# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_LaSilla, ax0_Silla, ax1_Silla, ax3_Silla = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# fig_LaSilla.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')



# climxa.group_mean('time.month', d_obs, climxa.xr_sel(d_obs, 'ds' ,lon, lat), std=True)






# # %% ######################################## Sutherland #######################################################################

# path_ds = './sites/Sutherland/Data/in-situ/hourly_meteo/Specific_humidity_Sutherland.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  20.81 # change!!, , 
# lat =  -32.38 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})


# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Sutherland Temperature', 'Sutherland Pressure', 'Sutherland Specific Humidity'])

# # %%
# # main parameters
# site = 'Sutherland' # refers to folder structure
# ls_pr_levels = [800, 825, 850, 875] # code reads in automatically data for these levels

# time_slice_var_meteo = slice('2013-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Sutherland Relative Humidity']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Sutherland, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var_meteo, d_obs, lon, lat, path, d_model = None)


# fig_Sutherland.savefig('./ERA5_validation/' + site + '_' + variable + '_DSCT.pdf')
