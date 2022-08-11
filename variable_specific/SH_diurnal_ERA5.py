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
import time

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
# global
list_of_single_level_vars = None
variable = 'SH'
list_of_clim_vars = ['q']
list_of_model_clim_params = ['hus']

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

diurnal=True
Ensemble = False
masterfigure = True
model_color = False

if masterfigure == True:
    master_fig = plt.figure( figsize = (20, 4*(max_idx+1))) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
    gs = master_fig.add_gridspec(max_idx + 1, 4) # first entry: rows, second entry: columns
    # initialize pressure level list (for legend; plot only those in use)
    Plev_list = []
else:
    MasterFig = None

if diurnal:
    # create a whole new figure only for the diurnal cycles
    fig_diurnal = plt.figure(figsize=(17, 2.3*(max_idx+1))) # max 16 height
    # fig_diurnal, ax1 = plt.subplots(sharex=True, sharey=True)
else:
    fig_diurnal = None

for idx in range(0, max_idx+1):
    # or define index for one iteration only
    # idx = 3

    file_name = '_' # at least, type '_'

    if masterfigure:
        MasterFig = (master_fig, idx, max_idx, gs, Plev_list, model_color) # or None

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    path_ds_SH_RH_T = d_site_lonlat_data['path_ds_SH_RH_T'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    time_slice_var_meteo = d_site_lonlat_data['time_slice_var_meteo'][idx]
    list_of_insitu_vars = [site_name_folder + ' Specific Humidity']

    path = './sites/'+ site_name_folder + '/Output/Plots/'+ variable + '/' # where to save the files

    # read in ds_hourly (in-situ data)
    ds_hourly = climxa.df_to_xarray(path_ds_SH_RH_T)
    ds_hourly = ds_hourly.rename_vars({'specific_humidity': list_of_insitu_vars[0]})

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
    # ATTENTION: 'taylor_folders' must be the same for all models, otherwise the edgecolor is wrong! (the first entry has no edgecolor, the second has black edgecolor)
    # hist and present cover the time until end of 2014, future goes on after 2014
    # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored
    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                 "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                 "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                 "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                 "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

    d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model}
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

    if idx == 0:
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable,
                                    time_slice_var_meteo, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                     d_model = None, d_Ensemble=d_Ensemble, MasterFig=MasterFig)
    else: # now we have an axis we can refer to fore sharing the xaxis
        d_obs_ss, d_model_ss, fig, sorted_skill_dict_ss, ax0_t, ax3_t = climxa.main_plotting_routine(site_name_folder, variable,
                                    time_slice_var_meteo, d_obs, lon, lat, path, diurnal=diurnal, fig_diurnal=fig_diurnal,
                                    d_model = None, d_Ensemble=d_Ensemble, MasterFig=MasterFig,
                                    ax_ref = (ax0_t, ax3_t))

    print("--- %s seconds ---" % (time.time() - start_time))

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

    # save dict to csv
    # path_skill_folder = '/home/haslebacher/chaldene/Astroclimate_Project/Model_evaluation/' + variable + '/csv_info/'
    # os.makedirs(os.path.dirname(path_skill_folder), exist_ok=True)
    # (pd.DataFrame.from_dict(data=sorted_skill_dict_ss, orient='index')
    #     .to_csv(path_skill_folder + site_name_folder + file_name + 'sorted_skill_dict.csv', header=False))


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
#%% tests for saving as json file
# import json

# with open('testjson.json', 'w') as fp:
#     json.dump(d_obs_ss, fp, indent=4)

# doesn't work for xarray datasets!
# so for xarray datasets, I could wirte to netcdf
# or just leave it with pickle files
# try to load one!
# import pickle
# with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/Sutherland_total_cloud_cover_d_obs_ERA5_and_insitu.pkl', 'rb') as myfile:
#     dload = pickle.load(myfile)

#%% investigate error behaviour
# # ds_taylor must have been created
# # test for La_Palma, 5 years

# d_test_error = d_obs['ds_taylor']['La_Palma Specific Humidity'].sel(time=slice('1998-01-01', '2001-12-31'))
# time_freq_string = 'time.month'
# d_test = climxa.group_mean(time_freq_string, d_obs, d_test_error, std=True)

#%% quickly test Taylor Diagram
# restart kernel

# import skill_metrics as sm
# # for a quick test of changes:
# path_ds = './sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)

# ds_hourly = ds_hourly.assign_coords({'latitude': 28.76, 'longitude': -17.88})
# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'La_Palma Specific Humidity'})
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Pressure'])

# list_of_single_level_vars = None
# variable = 'SH'
# list_of_clim_vars = ['q']
# site = 'La_Palma'
# ls_pr_levels = [700, 750, 775, 800, 850] # code reads in automatically data for these levels
# time_slice_var = slice('1998-01-01','2019-12-31')
# list_of_insitu_vars = ['La_Palma Specific Humidity']

# lon =  -17.88
# lat =  28.76

# # filter nan values
# ds_hourly = ds_hourly.where(xr.ufuncs.isnan(ds_hourly['La_Palma Relative Humidity']) != True, drop = True).where(xr.ufuncs.isnan(ds_hourly['La_Palma Temperature']) != True, drop=True)

# # write dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# ref = {"data" : d_obs['ds']['La_Palma Relative Humidity'][10:]}
# pred1 = {"data" : d_obs['ds']['La_Palma Relative Humidity'][:-10]}
# pred2 = {"data" : d_obs['ds']['La_Palma Temperature'][:-10]}

# taylor_stats1 = sm.taylor_statistics(pred1, ref,'data')
# taylor_stats2 = sm.taylor_statistics(pred2, ref,'data')

# # Store statistics in arrays
# sdev = np.array([taylor_stats1['sdev'][0]/taylor_stats1['sdev'][0], taylor_stats1['sdev'][1]/taylor_stats1['sdev'][0], taylor_stats2['sdev'][1]/taylor_stats1['sdev'][0]])
# crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], taylor_stats2['crmsd'][1]])
# ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], taylor_stats2['ccoef'][1]])

# marD = {'+': ['b', 'k']} #, 'd': ['r']}

# # mlabel =  {'ERA5 800hPa': 'k'}
# label = ['Era5'] #, 'clim model']
# sm.taylor_diagram(sdev,crmsd,ccoef, checkstats='on', styleOBS = '-', markerLabel = label,
#                     colOBS = 'r', markerobs = 'o', markerLegend = 'off', stylerms ='-',colRMS='grey',
#                     titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey',
#                     MarkerDictCH=marD)
# plt.gca().text(0, 1.1, 'Taylor Diagram (Frequency = Monthly)', fontsize=12)

#%% edgecolor
# mar_collection_clim_models_only = ['p','h', 's','d','^','v']

# import numpy as np
# import matplotlib.pyplot as plt

# size = 256,16
# dpi = 72.0
# figsize= size[0] / float(dpi), size[1] / float(dpi)
# fig = plt.figure(figsize=figsize, dpi=dpi)
# fig.patch.set_alpha(0)
# plt.axes([0, 0, 1, 1], frameon=False)

# for i in range(1, 11):
#     r, g, b = np.random.uniform(0, 1, 3)
#     plt.plot([i, ], [1, ], 'h', markersize=9, markerfacecolor='r',
#              markeredgewidth=1.2, markeredgecolor='k')


# plt.xlim(0, 11)
# plt.xticks([])
# plt.yticks([])

# plt.show()

#%% ################################################################################################################

# #%% LA PALMA
# #open in-situ measurements as pandas dataframe

# path_ds = './sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv'

# ds_hourly = climxa.df_to_xarray(path_ds)

# # apply specific changes to dataset
# # add coordinates
# # ds_hourly = ds_hourly.assign_coords({'latitude': 28.76, 'longitude': -17.88})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'La_Palma Specific Humidity'})

# # ds_hourly['La_Palma Relative Humidity'].attrs['units'] = '%'
# # ds_hourly['La_Palma Pressure'].attrs['units'] = 'hPa'
# # ds_hourly['La_Palma Specific Humidity'].attrs['units'] = 'kg/kg'
# # ds_hourly['La_Palma Temperature'].attrs['units'] = '$^{\circ}$C'

# # for better performance, drop unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Palma Relative Humidity', 'WindDirectionDeg', 'WindSpeedMS', 'La_Palma Temperature', 'La_Palma Pressure'])


# #%% read in HadGEM data from Amon Download (Juni 2020)
# ds_HadGEM = xr.open_mfdataset('./HighResMIP/variables/hus/Amon/HadGEM/future/*.nc')
# ds_HadGEM = ds_HadGEM.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_HadGEM['longitude'] = ds_HadGEM['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_HadGEM['level'] = ds_HadGEM['level']/100


# #%% read in HadGEM data
# ds_HadGEM = xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_3hr_La_Palma_grid.nc')
# ds_HadGEM = ds_HadGEM.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_HadGEM['longitude'] = ds_HadGEM['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_HadGEM['level'] = ds_HadGEM['level']/100

# # ds_sel = ds_HadGEM.sel(longitude= lon, latitude = lat, method = 'nearest')
# # ds_sel_res = ds_sel.groupby('time.month').mean(dim='time')

# #%%
# # read Amon data from HadGEM
# #ds_Amon3= xr.open_dataset('./HighResMIP/HadGEM3/hus_opendap/HadGEM_SST-present_hus_Amon_La_Palma_lon342_2_lat28_7_full.nc')
# # juni 2020
# ds_Amon = xr.open_mfdataset('./HighResMIP/variables/hus/Amon/HadGEM/future/*.nc')

# ds_Amon = ds_Amon.rename({'plev': 'level', 'lon': 'longitude', 'lat': 'latitude'})
# ds_Amon['longitude'] = ds_Amon['longitude'] - 360 # to make it identical to ERA 5 data
# #  pr level to hPa
# ds_Amon['level'] = ds_Amon['level']/100
# ds_Amon['level'] = ds_Amon['level'].astype('int64')
# ds_Amon['level'].attrs['units'] = 'hPa'


# #%%
# # main parameters
# site = 'La_Palma'
# ls_pr_levels = [700, 750, 775, 800, 850] # code reads in automatically data for these levels

# time_slice_var = slice('1998-01-01','2019-12-31')
# list_of_insitu_vars = ['La_Palma Specific Humidity']

# lon =  -17.88
# lat =  28.76
# path = './sites/La_Palma/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%%

# # maybe I have more than one clim model (use nested dict)
# d_model = {"HadGEM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700, 850], "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "ECMWF-IFS-HR"} } #, "clim_model_2": {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [925], "name": 'other office', 'ds_timeseries': ds_Amon}}
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored

# #%% main plotting routine
# importlib.reload(climxa)

# import time
# start_time = time.time() # measure elapsed time
# d_obs_LaPalma, d_model_LaPalma, fig_La_Palma = climxa.main_plotting_routine(site, variable,
#                                 time_slice_var, d_obs, lon, lat, path, d_model = d_model)
# print("--- %s seconds ---" % (time.time() - start_time))

# fig_La_Palma.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')





# #%% timeseries
# # # compare regrouped 3hr climate model data to monthly data

# # ds_model_3hr = ds_HadGEM['hus'].sel(longitude= lon,latitude=lat, level = ['700', '850'],method='nearest').groupby('time.year').mean()

# # #ds_model_sel = ds_clim_model_param['hus'].sel(longitude= lon,latitude= lat, level = '700',method='nearest')
# # ds_Amon_sel = ds_Amon['hus'].sel(level = ['700', '850']).groupby('time.year').mean()
# # # attention! check differences between calendars

# # # only needed when convertion to datetime index is needed for plotting on the same axis
# # # datetimeindex = ds_res_model.indexes['time'].to_datetimeindex()
# # # ds_res_model['time'] = datetimeindex

# # ds_yearly.q.plot.line(x='year')
# # ds_yearly['La_Palma Specific Humidity'].plot(color='k')

# # ds_Amon_sel.plot.line(x='year')
# # plt.savefig('./HighResMIP/Amon_long_timeseries_hus_La_Palma.pdf')

# # ds_model_3hr.plot.line(x='year')
# # plt.legend(['Amon 700hPa', 'Amon 850hPa', '3hr 700hPa', '3hr 850hPa'])
# # plt.savefig('./HighResMIP/Amon_vs_3hr_groupby_year_hus_La_Palma.pdf')

# # #ds_res_model.plot.line(x='time')
# # #plt.plot(ds_res_model.time, ds_res_model.values)

# # ds_model.plot.line(x='year')

# # plt.plot(ds_res_model.time, ds_res_model.dropna(dim='time'))
# # # resample monthly to get YYYY-MM

# # # how to filter nan from xarray:
# # # ds_model.where(xr.ufuncs.isnan(ds_model) == True, drop = True)
# # ds_model_3hr_700 = ds_HadGEM['hus'].sel(longitude= lon,latitude= lat, level = '700',method='nearest')

# # ds_clim_nan = ds_model_3hr_700.where(xr.ufuncs.isnan(ds_model_3hr_700.hus) == True, drop = True)





# #%% ########################################### siding spring #################################################

# path_ds = './sites/siding_spring/Data/in-situ/Specific_humidity_siding_spring_2003to2020.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  149.07
# lat =  -31.28
# # ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'siding spring Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'siding spring Pressure'])

# #%%
# # main parameters
# site = 'siding_spring' # refers to folder structure
# ls_pr_levels = [850, 875, 900, 950, 975, 1000] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2004-01-01','2019-12-31')
# list_of_insitu_vars = ['siding spring Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/siding_spring/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# list_of_single_level_vars = None
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%% main plotting routine

# fig_siding_spring, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# site = 'siding_spring'
# fig_siding_spring.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')


# del d_obs

# # %% ###############################################  Mauna Kea ####################################################3333

# path_ds = './sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -155.53
# lat =  19.83
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Mauna Kea Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Temp', 'P', 'RH'])


# #%%
# # main parameters
# site = 'MaunaKea' # refers to folder structure
# ls_pr_levels = [600, 650, 700, 750] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2000-01-01','2019-12-31')
# list_of_insitu_vars = ['Mauna Kea Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/MaunaKea/Output/Plots/SH/' # where to save the files
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
# # ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'SPM Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['SPM Pressure', 'SPM Relative Humidity', 'SPM Temperature'])

# # %%
# # main parameters
# site = 'SPM' # refers to folder structure
# ls_pr_levels = [700, 750, 775, 850] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2007-01-01','2019-12-31')
# list_of_insitu_vars = ['SPM Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/'+ site + '/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%%

# d_model = {"HadGEM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700, 850], "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "ECMWF-IFS-HR"} } #, "clim_model_2": {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [925], "name": 'other office', 'ds_timeseries': ds_Amon}}
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored

# #%% main plotting routine

# import time
# start_time = time.time() # measure elapsed time
# d_obs_SPM, d_model_SPM, fig_SPM = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = d_model)
# print("--- %s seconds for main plotting routinge ---" % (time.time() - start_time))


# fig_SPM.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')




# # %% ######################################### CERRO PARANAL ################################################################################

# path_ds = './sites/Paranal/Data/in-situ/SH_calculated/Specific_humidity_Paranal_ESO_2000to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -70.4 # change!!
# lat =  -24.63 # Change!!
# # ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Cerro Paranal Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Paranal T 2m', 'Paranal T -20m', 'Paranal RH 2m', 'Paranal RH -20m', 'Paranal Pressure' ])

# # %%
# # main parameters
# site = 'Paranal' # refers to folder structure
# ls_pr_levels = [700, 750, 800] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2000-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Cerro Paranal Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/'+ site + '/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# #%%

# # maybe I have more than one clim model (use nested dict)
# d_model = {"HadGEM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700, 850], "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "ECMWF-IFS-HR"} } #, "clim_model_2": {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [925], "name": 'other office', 'ds_timeseries': ds_Amon}}
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored

# #%% main plotting routine

# import time
# start_time = time.time() # measure elapsed time

# d_obs_Paranal, d_model_Paranal, fig_Paranal = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = d_model)
# print("--- %s seconds for main plotting routinge ---" % (time.time() - start_time))


# fig_Paranal.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')



# # %% ############################################ Cerro Tololo #######################################################################

# path_ds = './sites/Cerro_Tololo/Data/in-situ/Specific_humidity_Cerro_Tololo_2002to2019.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  -70.81 # change!!,
# lat =  -30.17 # Change!!
# # ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})

# ds_hourly = ds_hourly.rename_vars({'specific_humidity': 'Cerro Tololo Specific Humidity'})

# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Cerro Tololo Temperature', 'Cerro Tololo Relative Humidity', 'Cerro Tololo Pressure (hPa)'])

# # %%
# # main parameters
# site = 'Cerro_Tololo' # refers to folder structure
# ls_pr_levels = [750, 775, 800] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2002-01-01','2018-12-31') # CHANGE!
# list_of_insitu_vars = ['Cerro Tololo Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/'+ site + '/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}

# d_model = {"HadGEM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700, 850], "name": "HadGEM3-GC31-HM"},
#             "EC-Earth": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "EC-Earth3P-HR"},
#             "CNRM": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CNRM-CM6-1-HR"},
#             "MPI": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "MPI-ESM1-2-XR"},
#             "CMCC": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "CMCC-CM2-VHR4"},
#             "ECMWF": {"folders": ['hist'],"clim_var": ['hus'], 'Plev': [700,850], "name": "ECMWF-IFS-HR"} } #, "clim_model_2": {"ds": ds_clim_model_param, "clim_var": ['hus'], 'Plev': [925], "name": 'other office', 'ds_timeseries': ds_Amon}}
# # the name of the subdict (e.g. HadGEM) must be equal to the folder name where the model data is stored


# #%% main plotting routine

# import time
# start_time = time.time() # measure elapsed time

# d_obs, d_model, fig= climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = d_model)
# print("--- %s seconds for main plotting routine ---" % (time.time() - start_time))


# fig.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')





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
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'La_Silla T 30m', 'La_Silla T 2m', 'La_Silla T ground', 'La_Silla RH 2m', 'La_Silla Pressure'])

# # %%
# # main parameters
# site = 'La_Silla' # refers to folder structure
# ls_pr_levels = [750, 775, 800] # code reads in automatically data for these levels
# variable = 'SH'
# time_slice_var = slice('2000-01-01','2015-12-31') # CHANGE!
# list_of_insitu_vars = ['La Silla Specific Humidity']
# list_of_clim_vars = ['q']
# path = './sites/'+ site + '/Output/Plots/SH/' # where to save the files
# # ds_clim_model_param = ds_HadGEM.sel(time=slice('1998-01-01','2019-12-30')) # select time slice for diurnal/seasonal cycle
# # list_of_pr_level_clim_model = [700, 850]
# # list_of_model_clim_params = ['hus']

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_LaSilla, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)


# fig_LaSilla.savefig('./ERA5_validation/' + site + '_' + variable + '_DSC_T.pdf')





# # %% ######################################## Sutherland #######################################################################

# path_ds = './sites/Sutherland/Data/in-situ/hourly_meteo/Specific_humidity_Sutherland.csv'
# ds_hourly = climxa.df_to_xarray(path_ds)
# # apply specific changes to dataset
# # add coordinates
# lon =  20.81 # change!!, ,
# lat =  -32.38 # Change!!
# ds_hourly = ds_hourly.assign_coords({'latitude': lat, 'longitude': lon})


# # get rid of unused variables
# ds_hourly = ds_hourly.drop_vars(['Unnamed: 0', 'Sutherland Temperature', 'Sutherland Pressure', 'Sutherland Relative Humidity'])

# # %%
# # main parameters
# site = 'Sutherland' # refers to folder structure
# ls_pr_levels = [800, 825, 850, 875] # code reads in automatically data for these levels

# time_slice_var = slice('2013-01-01','2019-12-31') # CHANGE!
# list_of_insitu_vars = ['Sutherland Specific Humidity']

# path = './sites/'+ site + '/Output/Plots/' + variable + '/' # where to save the files

# #%% write main params into dictionary
# d_obs = {"ds": ds_hourly, "insitu_var": list_of_insitu_vars, "ERA5_var": list_of_clim_vars, "Plev": ls_pr_levels, "single_lev": list_of_single_level_vars}


# #%% main plotting routine

# fig_Sutherland, ax0, ax1, ax3 = climxa.main_plotting_routine(site, variable, time_slice_var, d_obs, lon, lat, path, d_model = None)

# fig_Sutherland.savefig('./ERA5_validation/' + site + '_' + variable + '_DSCT.pdf')




# %%
