# this script plots the vertical profile of the astronomical seeing


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
import copy

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# revision: for xticks in log scale
import matplotlib.ticker

import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa
import pickle

# restore matplotlib with
# import importlib
# import matplotlib as mpl
# importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)


#%% RELOAD CLIMXA

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

#%% seeing (Cn2 profile)

# loop through sites
# create a big plot (all 8 sites on one pdf page)
fig = plt.figure(figsize = (17, 20)) #,constrained_layout=True) # (this is not compatible with tight_layout)
# 14, 20
gs = fig.add_gridspec(3, 3)

# ax1 = fig.add_subplot(gs[0, 0]) # diurnal cycle
# ax0 = fig.add_subplot(gs[0, 1]) # seasonal cycle
# ax3 = fig.add_subplot(gs[1, 0]) # span timeseries over two subplot lengths
# ax4 = fig.add_subplot(gs[1, 1]) # taylor diagram
# ax5 = fig.add_subplot(gs[2, 0]) # span timeseries over two subplot lengths
# ax6 = fig.add_subplot(gs[2, 1]) # taylor diagram
# ax7 = fig.add_subplot(gs[3, 0]) # span timeseries over two subplot lengths
# ax8 = fig.add_subplot(gs[3, 1]) # taylor diagram

for idx in range(0, 8):
    # or define index for one iteration only
    # idx = 0

    # define axis on which to plot
    ax = fig.add_subplot(gs[int((idx - (idx%3))/3), idx%3]) # int((i idx - (idx%2))/2), idx%2

    print(d_site_lonlat_data['site_name'][idx])
    # lon_obs and lat_obs are in 0-360 format!!
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    list_of_single_level_model_clim_params = ['Cn2']

    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params,  "name": "HadGEM3-GC31-HM"},
                    "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist'],"single_lev_var": list_of_single_level_model_clim_params, "name": "EC-Earth3P-HR"} ,
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "single_lev_var": list_of_single_level_model_clim_params, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"single_lev_var": list_of_single_level_model_clim_params, "name": "ECMWF-IFS-HR"} }

    # calculate d_Ensemble (there is no d_Ensemble of Cn2 stored somewhere up to now)
    d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "single_lev_var": list_of_single_level_model_clim_params}

    for clim_key in d_model.keys():

        # read in ds_Cn2 for every model
        d_model[clim_key]['ds_Cn2'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)

    # calculate ensemble mean of time medians
    for forcing in d_Ensemble['folders']:

        ls_Cn2_primavera = []
        ls_Cn2_75iqr_primavera = []
        ls_Cn2_25iqr_primavera = []

        for clim_key in d_model.keys():

            # # plot vertical profile (median)
            # plt.plot(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].median(dim='time'), d_model[clim_key]['ds_Cn2'].level)

            # calculate ensemble for every forcing, take time median
            if forcing in d_model[clim_key]['folders']:
                ls_Cn2_primavera.append(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].median(dim='time'))
                # added Interquartile range on May 05 2021
                ls_Cn2_25iqr_primavera.append(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].quantile(q=0.25, dim='time'))
                ls_Cn2_75iqr_primavera.append(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].quantile(q=0.75, dim='time'))


        # compose d_Ensemble dataset
        d_Ensemble['Cn2 ' + forcing] = np.mean(ls_Cn2_primavera, axis=0)
        d_Ensemble['Cn2 ' + forcing+ ' IQR 0.25'] = np.mean(ls_Cn2_25iqr_primavera, axis=0)
        d_Ensemble['Cn2 ' + forcing+ ' IQR 0.75'] = np.mean(ls_Cn2_75iqr_primavera, axis=0)


    # plotting
    pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
    pr_levels_list.reverse()
    lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
    line_list = []

    for forcing in d_Ensemble['folders']:

        # choose the right linestyle
        if forcing=='hist':
            forced_linestyle = lin_st[0]
            color = 'navy'
        elif forcing=='present':
            forced_linestyle = lin_st[1]
            color = 'navy'
        elif forcing=='future':
            forced_linestyle = lin_st[2]
            color = 'red'
        # revision: I just forgot the below line!!!!
        elif forcing=='SSTfuture':
            forced_linestyle = lin_st[3]
            color = 'red'
        else:
            raise Warning('no corresponding linestyle selected!')

        ax.plot(d_Ensemble['Cn2 ' + forcing], pr_levels_list[0:(len(d_Ensemble['Cn2 ' + forcing]))],
                        linestyle=forced_linestyle, color=color) # , marker='o'
        # IQR fill between x-axis
        ax.fill_betweenx(pr_levels_list[0:(len(d_Ensemble['Cn2 ' + forcing]))], d_Ensemble['Cn2 ' + forcing + ' IQR 0.25'],
                                            d_Ensemble['Cn2 ' + forcing + ' IQR 0.75'], facecolor= color, alpha=0.1)


        # for legend
        line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = color, label = r'PRIMAVERA $C_n^2$ ' + forcing))


    #### ERA5
    # load data
    path_Cn2 = './sites/'+ site_name_folder + '/Data/Era_5/Cn2/ds_ERA5_Cn2_profile_hourly_nearest.nc'
    ds_ERA5_Cn2 = xr.open_dataset(path_Cn2)

    # plot Cn2 profile
    ax.plot(ds_ERA5_Cn2['Cn2'].median(dim='time'), ds_ERA5_Cn2.level, linestyle = '-', color = '#009e73') # , marker='o' # goes until 50hPa (glad I included it until 50hPa!)
    # plot interquartile range (fill between x-axis!)
    ax.fill_betweenx(ds_ERA5_Cn2.level, ds_ERA5_Cn2['Cn2'].quantile(q=0.25, dim='time'), ds_ERA5_Cn2['Cn2'].quantile(q=0.75, dim='time'), facecolor= '#009e73', alpha=0.2) # , label='IQR'

    # for legend
    line_list.append(Line2D([0], [0], linestyle = '-', color = '#009e73', label = r'ERA5 $C_n^2$'))
    # iqr (for all data! therefore, take black as the facecolor)
    line_list.append(Patch(facecolor = 'k', label = 'IQR', alpha=0.1))


    # maybe I have to cut off the integration of PRIMAVERA to get rid of the peak at 70hPa to increase correlation of osborn seeing?

    # change site label
    if site_name_folder == 'MaunaKea':
        site_noUnderline = 'Mauna Kea'
    elif site_name_folder == 'siding_spring':
        site_noUnderline = 'Siding Spring'
    else:
        site_noUnderline = site_name_folder.replace('_', ' ')

    ax.set_xscale('log')
    # revision: set xticks for log scale. From here: https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
    x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 8)
    ax.xaxis.set_major_locator(x_major)
    x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_yticks(np.arange(0, 1100, 100)) # revision: until 1100 instead of 1000, otherwise '1000hPa' is not plotted
    # ax.set_yticks(ds_ERA5_Cn2.level)
    plt.gca().invert_yaxis()
    ax.set_xlabel(r'$C_n^2$ $[m^{1/3}]$')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_title(climxa.alphabet_from_idx(idx) + ') ' + site_noUnderline)
    # ax.set_xlim(1e-19, 5e-17)

    # revision: no legend!
    # if idx == 7:
    #     # plot legend into empty spot
    #     # ax_legend = fig.add_subplot(gs[2,2]) # int((i idx - (idx%2))/2), idx%2
    #     ax.legend(handles=line_list, loc='upper left', bbox_to_anchor= (1.1, 1))

# revision: slightly adjust plot (not sns.set() anymore)
fig.subplots_adjust(hspace=0.3,
                    wspace=0.3)

# save fig
fig.savefig('./Model_evaluation/seeing_nc/All_Sites_Cn2_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)
fig.savefig('./publication/revision/figures/All_Sites_Cn2_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

plt.show()

#%% define function

def vertical_profile_ERA5_and_PRIMAVERA_SH(variable, list_of_clim_vars, list_of_model_clim_params, my_xlabel):

    # loop through sites
    # create a big plot (all 8 sites on one pdf page)
    fig = plt.figure(figsize = (14, 20)) # (this is not compatible with tight_layout)
    # (8, 20)
    gs = fig.add_gridspec(3, 3)

    # ax1 = fig.add_subplot(gs[0, 0]) # diurnal cycle
    # ax0 = fig.add_subplot(gs[0, 1]) # seasonal cycle
    # ax3 = fig.add_subplot(gs[1, 0]) # span timeseries over two subplot lengths
    # ax4 = fig.add_subplot(gs[1, 1]) # taylor diagram
    # ax5 = fig.add_subplot(gs[2, 0]) # span timeseries over two subplot lengths
    # ax6 = fig.add_subplot(gs[2, 1]) # taylor diagram
    # ax7 = fig.add_subplot(gs[3, 0]) # span timeseries over two subplot lengths
    # ax8 = fig.add_subplot(gs[3, 1]) # taylor diagram

    for idx in range(0, 8):
        # or define index for one iteration only
        # idx = 0

        # define axis on which to plot; runs up from [0,0] to [3,1]
        ax = fig.add_subplot(gs[int((idx - (idx%3))/3), idx%3])

        print(d_site_lonlat_data['site_name'][idx])
        # lon_obs and lat_obs are in 0-360 format!!
        lon = d_site_lonlat_data['lon_obs'][idx]
        lat = d_site_lonlat_data['lat_obs'][idx]

        ls_pr_levels_clim_model = None # we take all pressure levels!
        site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

        d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                        "EC-Earth": {"folders": ['hist', 'future'],"taylor_folder": ['hist', 'present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                        "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'], "taylor_folder": ['hist','present'], "clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                        "MPI": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                        "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                        "ECMWF": {"folders": ['hist', 'present'],"taylor_folder": ['hist','present'],"clim_var": list_of_model_clim_params, 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

        # calculate d_Ensemble (there is no d_Ensemble of Cn2 stored somewhere up to now)
        d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture'], "clim_var": list_of_model_clim_params}

        # closest pressure level (I integrated the TCW until there)
        if idx == 5: # siding Spring
            closest_pr_lev = 925

        elif idx == 1 or idx == 2 or idx == 4 or idx == 7: # SPM, Paranal, La Silla, La Palma
            closest_pr_lev = 700

        elif idx == 3 or idx == 6: # Tololo, sutherland
            closest_pr_lev = 850

        elif idx == 0: # Mauna Kea
            closest_pr_lev = 600

        # plotting
        pr_levels_list = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1] # , 925, 1000
        pr_levels_list = pr_levels_list[pr_levels_list.index(closest_pr_lev):]

        for clim_key in d_model.keys():
            print(clim_key)
            # read in ds_SH for every model
            d_model[clim_key]['ds_SH'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, pressure_level=True)
            # select lon/lat
            d_model[clim_key]['ds_SH_sel'] = d_model[clim_key]['ds_SH'].sel(level=pr_levels_list, longitude=lon, latitude=lat, method='nearest')

        # calculate ensemble mean of time medians
        for forcing in d_Ensemble['folders']:

            ls_SH_primavera = []
            ls_SH_25iqr_primavera = []
            ls_SH_75iqr_primavera = []

            for clim_key in d_model.keys():

                # # plot vertical profile (median)
                # plt.plot(d_model[clim_key]['ds_SH']["Cn2 " + forcing].median(dim='time'), d_model[clim_key]['ds_Cn2'].level)

                # calculate ensemble for every forcing, take time median
                if forcing in d_model[clim_key]['folders']:
                    ls_SH_primavera.append(d_model[clim_key]['ds_SH_sel'][list_of_model_clim_params[0] + " " + forcing].median(dim='time'))
                    # added Interquartile range on May 05 2021
                    ls_SH_25iqr_primavera.append(d_model[clim_key]['ds_SH_sel'][list_of_model_clim_params[0] + " " + forcing].quantile(q=0.25, dim='time'))
                    ls_SH_75iqr_primavera.append(d_model[clim_key]['ds_SH_sel'][list_of_model_clim_params[0] + " " + forcing].quantile(q=0.75, dim='time'))

            # compose d_Ensemble dataset
            d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing] = np.mean(ls_SH_primavera, axis=0)
            d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing+ ' IQR 0.25'] = np.mean(ls_SH_25iqr_primavera, axis=0)
            d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing+ ' IQR 0.75'] = np.mean(ls_SH_75iqr_primavera, axis=0)



        lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
        line_list = []

        for forcing in d_Ensemble['folders']:

            # choose the right linestyle
            if forcing=='hist':
                forced_linestyle = lin_st[0]
                color = 'navy'
            elif forcing=='present':
                forced_linestyle = lin_st[1]
                color = 'navy'
            elif forcing=='future':
                forced_linestyle = lin_st[2]
                color = 'red'
            # revision: I just forgot the below line!!!!
            elif forcing=='SSTfuture':
                forced_linestyle = lin_st[3]
                color = 'red'
            else:
                raise Warning('no corresponding linestyle selected!')


            ax.plot(d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing], pr_levels_list,
                            linestyle=forced_linestyle, color=color)
            # IQR fill between x-axis
            ax.fill_betweenx(pr_levels_list, d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing + ' IQR 0.25'],
                                                d_Ensemble[list_of_model_clim_params[0] + ' ' + forcing + ' IQR 0.75'], facecolor= color, alpha=0.1)

            # for legend
            line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = color, label = 'PRIMAVERA specific humidity ' + forcing))



        #### ERA5
        # load data (code from 'calc_PWV_ERA5.py' and climxa.SH_integral_to_TCW(...))
        surface_pressure_observation = d_site_lonlat_data['pressure [hPa]'][idx]
        print(surface_pressure_observation)
        # check available pressure for ERA5
        absolute_difference_function = lambda list_value : abs(list_value - given_value)
        pr_levels_ERA5 = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa

        given_value = surface_pressure_observation
        closest_value = min(pr_levels_ERA5, key=absolute_difference_function)

        SH_integral_pressure = closest_value # find nearest pressure

        print('closest match: {}'.format(SH_integral_pressure))

        chile_grid = ['Tololo', 'Pachon', 'Silla']
        if any(x in site_name_folder for x in chile_grid):
            site_ERA = 'Paranal'
        else:
            site_ERA = site_name_folder

        if lon > 180:
            my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
            print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
        else:
            my_ERA5_lon = lon

        SH_pressure_levels = []
        # in hPa
        pr_levels_all = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925] # siding spring is the upper limit with 892hPa
        # read in data (only for downloaded pressure levels)
        # find upper max index that corresponds to given pressure level
        pr_max_idx = pr_levels_all.index(SH_integral_pressure)

        for i in range(0,pr_max_idx + 1): #
            print(i)
            path =  '/home/haslebacher/chaldene/Astroclimate_Project/sites/'+ site_ERA + '/Data/Era_5/'+ variable + '/'+ str(pr_levels_all[i]) + 'hPa' +'/*.nc'
            ds = xr.open_mfdataset(path)
            ds_sel = ds.sel(longitude= my_ERA5_lon,latitude= lat ,method='nearest')
            if 'expver' in ds_sel.coords:
                ds_sel = ds_sel.sel(expver=5)
            # # line below: commented for generating hourly data for diurnal cycle
            # ds_sel = ds_sel.resample(time = '1m', keep_attrs=True).mean()

            # assign pressure level
            ds_sel = ds_sel.assign_coords({"level": pr_levels_all[i]})

            SH_pressure_levels.append(ds_sel)

        ds_ERA5_SH = xr.concat(SH_pressure_levels, dim = 'level')
        # load xarray dataset
        ds_ERA5_SH = ds_ERA5_SH.load()

        # plot Cn2 profile
        ax.plot(ds_ERA5_SH[list_of_clim_vars[0]].median(dim='time'), ds_ERA5_SH.level, linestyle = '-', color = '#009e73') # goes until 50hPa (glad I included it until 50hPa!)
        # plot interquartile range (fill between x-axis!)
        ax.fill_betweenx(ds_ERA5_SH.level, ds_ERA5_SH[list_of_clim_vars[0]].quantile(q=0.25, dim='time'), ds_ERA5_SH[list_of_clim_vars[0]].quantile(q=0.75, dim='time'), facecolor= '#009e73', alpha=0.2) # , label='IQR'

        # for legend
        line_list.append(Line2D([0], [0], linestyle = '-', color = '#009e73', label = 'ERA5 specific humidity'))
        # iqr (for all data! therefore, take black as the facecolor)
        line_list.append(Patch(facecolor = 'k', label = 'IQR', alpha=0.1))

        # maybe I have to cut off the integration of PRIMAVERA to get rid of the peak at 70hPa to increase correlation of osborn seeing?

        # revision: no legend!!
        # if idx == 7:
        #     ax.legend(handles=line_list, loc='upper left', bbox_to_anchor= (1.1, 1)) # (0, -0.2)

        # change site label
        if site_name_folder == 'MaunaKea':
            site_noUnderline = 'Mauna Kea'
        elif site_name_folder == 'siding_spring':
            site_noUnderline = 'Siding Spring'
        else:
            site_noUnderline = site_name_folder.replace('_', ' ')

        ax.set_xscale('log')
        # revision: set xticks for log scale. From here: https://stackoverflow.com/questions/30887920/how-to-show-minor-tick-labels-on-log-scale-with-matplotlib
        x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 8)
        ax.xaxis.set_major_locator(x_major)
        x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        # ax.set_yticks(np.arange(0, 1000, 100)) # pressure levels
        plt.gca().invert_yaxis()
        ax.set_title(climxa.alphabet_from_idx(idx) + ') ' + site_noUnderline)

        ax.set_xlabel(my_xlabel)
        ax.set_ylabel('Pressure [hPa]')

        # revision: adjust slightly
        fig.subplots_adjust(hspace=0.3,
                    wspace=0.3)

    return fig

#%%

# same, just select SH as input --> I could make function out of it

# list_of_single_level_vars = None
variable = 'SH'
list_of_clim_vars = ['q']
list_of_model_clim_params = ['hus']
my_xlabel = 'Specific humidity [kg/kg]'

fig = vertical_profile_ERA5_and_PRIMAVERA_SH(variable, list_of_clim_vars, list_of_model_clim_params, my_xlabel)


# save fig
fig.savefig('./Model_evaluation/SH/All_Sites_SH_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)
fig.savefig('./publication/revision/figures/All_Sites_SH_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

plt.show()

#%% temperature

variable = 'T'
list_of_clim_vars = ['t']
list_of_model_clim_params = ['ta']
my_xlabel = r'Temperature [$^{\circ}$ C]'

#%%
# read in seeing vars (t, u, v)
# only ERA5!
def get_seeing_variables(idx, d_site_lonlat_data):

    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]
    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    chile_grid = ['Tololo', 'Pachon', 'Silla']
    if any(x in site_name_folder for x in chile_grid):
        site_ERA = 'Paranal'
    else:
        site_ERA = site_name_folder

    if lon > 180:
        my_ERA5_lon = lon - 360 # e.g. 360-17.88 = 342.12 > 180 --> lon = 342.12 - 360 = -17.88
        print('I adapted lon to -180/180 lon. new lon is: {}'.format(my_ERA5_lon))
    else:
        my_ERA5_lon = lon

    # use function which loads in all specific humidity datasets
    # and integrates them to specific humidity

    if site_ERA == 'Paranal': # Paranal
        seeing_data_path =  './sites/' + site_ERA + '/Era5_data/seeing/'
    else:
        seeing_data_path =  './sites/' + site_ERA + '/Data/Era_5/seeing/'

    ds_seeing_vars = climxa.read_ERA5_seeing_data(seeing_data_path, my_ERA5_lon, lat)
    ds_seeing_vars = ds_seeing_vars.load() # load here to prevent program from running

    return ds_seeing_vars

# PRIMAVERA
def get_PRIMAVERA_seeing_vars(idx):

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

        mean_insitu = 1.32 # from Optical turbulence characterization at the SAAO Sutherland site (L. Catala)

    else:
        # read in ds_hourly (in-situ data)
        # ds_hourly = climxa.df_to_xarray('./sites/Paranal/Data/    # attention! taylor folders can change! think about that in the return...in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv') # hourly_Paranal_Seeing.csv'), ['Seeing Paranal']
        ds_hourly = climxa.df_to_xarray(path_seeing)

        mean_insitu = np.mean(ds_hourly[list_of_insitu_vars[0]])

    d_model = {"HadGEM": {"folders": ['hist','present', 'future', 'SSTfuture'],'Plev': ls_pr_levels_clim_model, "name": "HadGEM3-GC31-HM"},
                    "EC-Earth": {"folders": ['hist', 'future'], 'Plev': ls_pr_levels_clim_model, "name": "EC-Earth3P-HR"} ,
                    "CNRM": {"folders": ['hist','present', 'future', 'SSTfuture'],  'Plev': ls_pr_levels_clim_model, "name": "CNRM-CM6-1-HR"},
                    "MPI": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "MPI-ESM1-2-XR"},
                    "CMCC": {"folders": ['hist','present', 'future', 'SSTfuture'], 'Plev': ls_pr_levels_clim_model, "name": "CMCC-CM2-VHR4"},
                    "ECMWF": {"folders": ['hist', 'present'], 'Plev': ls_pr_levels_clim_model, "name": "ECMWF-IFS-HR"} }

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

    return d_model


#%%

# other function: plot t, u, v profile
def plot_seeing_vars_vertical_profile(idx, d_model):
    # loop through sites
    # create a big plot (all 8 sites on one pdf page)
    # fig = plt.figure(figsize = (8, 6)) # (this is not compatible with tight_layout)
    # (8, 20)
    # gs = fig.add_gridspec(1, 3)

    fig, ax = plt.subplots(1,4, figsize = (16, 6), tight_layout = True)

    ######

    # if we want to have separate plots:
    # define axis (use idx from site)
    # ax_u = fig.add_subplot(gs_u[int((idx - (idx%3))/3), idx%3])
    # ax_v = fig.add_subplot(gs_v[int((idx - (idx%3))/3), idx%3])
    # ax_t = fig.add_subplot(gs_t[int((idx - (idx%3))/3), idx%3])
    # ax_z = fig.add_subplot(gs_z[int((idx - (idx%3))/3), idx%3])


    # fig = plt.figure(figsize = (14, 20)) # (this is not compatible with tight_layout)
    # # (8, 20)
    # gs = fig.add_gridspec(3, 3)
    # define axis on which to plot; runs up from [0,0] to [3,1]
    # ax = fig.add_subplot(gs[int((idx - (idx%3))/3), idx%3])

    ###### above in development

    # ax1 = fig.add_subplot(gs[0,0]) # t
    # ax2 = fig.add_subplot(gs[0,1]) # u
    # ax3 = fig.add_subplot(gs[0,2]) # v

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    lsclimvars = ['ta', 'ua', 'va', 'zg']
    ls_labels = [r'Temperature [$^{\circ}$ C]', 'eastward wind speed [m/s]', 'northward wind speed [m/s]', 'geopotential height [m]']
    # initialize d_Ensemble dictionary
    d_Ensemble = {"folders": ['hist','present', 'future', 'SSTfuture']} # , "clim_var": clim_var


    for climind, clim_var in enumerate(lsclimvars):
        print(climind)
        # one ax for one clim_var
        # ax = ax[climind]

        # calculate d_Ensemble (there is no d_Ensemble of Cn2 stored somewhere up to now)
        # feed clim_var and Plev

        # calculate ensemble mean of time medians

        # old code below:
        # for clim_key in d_model.keys():
        #     for forcing in d_model[clim_key]['folders']:
                # old: plot vertical profile (median) for every individual forcing and model
                # ax[climind].plot(d_model[clim_key]['ds_seeing'][clim_var + " " + forcing].median(dim='time'), d_model[clim_key]['ds_seeing'].level,
                #         linestyle = 'dotted', label='PRIMAVERA ' + clim_key + ' ' + forcing)

        for forcing in d_Ensemble['folders']:

            ls_uvtz_primavera = []
            ls_uvtz_25iqr_primavera = []
            ls_uvtz_75iqr_primavera = []

            for clim_key in d_model.keys():
                ### new: calculate ensemble for every forcing, take time median
                if forcing in d_model[clim_key]['folders']:

                    # problem: not all models go down to 1000hP or other limits...
                    # e.g. for Cerro Tololo, the first model stops at 700hPa
                    # don't take the median
                    ls_uvtz_primavera.append(d_model[clim_key]['ds_seeing'][clim_var + " " + forcing].median(dim='time')) # .rename(clim_key) .dropna(dim='level')

                    ls_uvtz_25iqr_primavera.append(d_model[clim_key]['ds_seeing'][clim_var + " " + forcing].quantile(q=0.25, dim='time'))
                    ls_uvtz_75iqr_primavera.append(d_model[clim_key]['ds_seeing'][clim_var + " " + forcing].quantile(q=0.75, dim='time'))

                    # print(ls_uvtz_primavera)

            # compose d_Ensemble dataset
            # merge list to xarray!
            # xr_uvtz_median = xr.merge(ls_uvtz_primavera)
            # I take the mean of the medians! if Nan, exclude...
            # for this purpose, use np.nanmean: this ignores nan values and just takes the mean over the rest!
            d_Ensemble[clim_var + ' ' + forcing] = np.nanmean(ls_uvtz_primavera, axis=0)
            d_Ensemble[clim_var + ' ' + forcing + ' IQR 0.25'] = np.nanmean(ls_uvtz_25iqr_primavera, axis=0)
            d_Ensemble[clim_var + ' ' + forcing + ' IQR 0.75'] = np.nanmean(ls_uvtz_75iqr_primavera, axis=0)

        # add level (just take one clim_key, the pressure levels are all the same!)
        d_Ensemble['Plev'] = d_model[clim_key]['ds_seeing'].level

        # plotting
        # pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
        lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
        line_list = []

        for forcing in d_Ensemble['folders']:

            # choose the right linestyle
            if forcing=='hist':
                forced_linestyle = lin_st[0]
                color = 'navy'
            elif forcing=='present':
                forced_linestyle = lin_st[1]
                color = 'navy'
            elif forcing=='future':
                forced_linestyle = lin_st[2]
                color = 'red'
            # revision: I just forgot the below line!!!!
            elif forcing=='SSTfuture':
                forced_linestyle = lin_st[3]
                color = 'red'
            else:
                raise Warning('no corresponding linestyle selected!')

            ax[climind].plot(d_Ensemble[clim_var + ' ' + forcing], d_Ensemble['Plev'], # pr_levels_list[0:(len(d_Ensemble[clim_var + ' ' + forcing]))],
                            linestyle=forced_linestyle, color=color)
            # IQR fill between x-axis
            ax[climind].fill_betweenx(d_Ensemble['Plev'], d_Ensemble[clim_var + ' ' + forcing + ' IQR 0.25'],
                                            d_Ensemble[clim_var + ' ' + forcing + ' IQR 0.75'], facecolor= color, alpha=0.1)

            # for legend
            line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = color, label = 'PRIMAVERA ' + forcing))


        #### ERA5
        # load data
        median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'
        iqr_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_IQR_ERA5_u_v_t_z.nc'

        ds_ERA5_iqr = xr.open_dataset(iqr_path).load()
        ds_ERA5_median = xr.open_dataset(median_path).load()

        if clim_var == 'zg':
            # divide ERA5 by 10
            ds_ERA5_median['z'] = ds_ERA5_median['z']/10
            ds_ERA5_iqr['z'] = ds_ERA5_iqr['z']/10

        # plot profile
        # note: clim_var[0] only takes the first character ('u', 'v', 't')
        ax[climind].plot(ds_ERA5_median[clim_var[0]], ds_ERA5_median.level, linestyle = '-', color = '#009e73') # , label='ERA5' # goes until 50hPa (glad I included it until 50hPa!)
        # plot interquartile range (fill between x-axis!)
        ax[climind].fill_betweenx(ds_ERA5_iqr.level, ds_ERA5_iqr[clim_var[0]].sel(quantile=0.25), ds_ERA5_iqr[clim_var[0]].sel(quantile=0.75), facecolor= '#009e73', alpha=0.2) # , label='IQR'

        # for legend
        # median
        line_list.append(Line2D([0], [0], linestyle = '-', color = '#009e73', label = 'ERA5'))
        # iqr (for all data! therefore, take black as the facecolor)
        line_list.append(Patch(facecolor = 'k', label = 'IQR', alpha=0.1))
        # plot a red line at 70hPa
        # ax[climind].axhline(y=70, xmin=-20, xmax=300, color = 'red', label='70hPa line')

        # maybe I have to cut off the integration of PRIMAVERA to get rid of the peak at 70hPa to increase correlation of osborn seeing?

        # change site label
        if site_name_folder == 'MaunaKea':
            site_noUnderline = 'Mauna Kea'
        elif site_name_folder == 'siding_spring':
            site_noUnderline = 'Siding Spring'
        else:
            site_noUnderline = site_name_folder.replace('_', ' ')

        # ax.set_xscale('log')
        # ax.set_yticks(np.arange(0, 1000, 100))
        ax[climind].invert_yaxis()
        # ax[climind].set_xlabel(clim_var)
        ax[climind].set_ylabel('Pressure [hPa]')
        # ax.set_title(site_noUnderline)
        ax[climind].set_xlabel(ls_labels[climind])

        # revision: no legend!!
        # if climind == (len(lsclimvars)-1):
        #     ax[climind].legend(handles=line_list, loc = 'upper left', bbox_to_anchor = (1.1, 1))

        fig.suptitle(site_noUnderline) # f'{climxa.alphabet_from_idx(idx)}) {site_noUnderline}')

        # if idx == 7:
        #     # plot legend into empty spot
        #     # ax_legend = fig.add_subplot(gs[2,2]) # int((i idx - (idx%2))/2), idx%2
        #     ax.legend(handles=line_list, loc='upper left', bbox_to_anchor= (1.1, 1))

    # save fig
    # fig.savefig('./Model_evaluation/seeing_nc/All_Sites_Cn2_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)
    # fig.savefig('./publication/figures/All_Sites_Cn2_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

    # plt.show()

    return fig


#%% MAIN: vertical profile of
# loop through sites

# I've had this idea, but it is not so good.. just plot single figs
# # plot to individual figures for individual variables
# fig_u = plt.figure(figsize = (14, 20))
# fig_v = plt.figure(figsize = (14, 20))
# fig_t = plt.figure(figsize = (14, 20))
# fig_z = plt.figure(figsize = (14, 20))

# # define gridspec
# gs_u = fig.add_gridspec(3, 3)
# gs_v = fig.add_gridspec(3, 3)
# gs_t = fig.add_gridspec(3, 3)
# gs_z = fig.add_gridspec(3, 3)

for idx in range(0,8):

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

    d_model = get_PRIMAVERA_seeing_vars(idx)

    # include a step to calculate median and IQR for PRIMAVERA! just take median of 6 models? No!
    #  take median of every model and then take mean?
    # calc_PRIMAVERA_median
    # pass figures back and forth!
    fig = plot_seeing_vars_vertical_profile(idx, d_model)

    # save fig
    fig.savefig('./Model_evaluation/seeing_nc/vertical_profile_u_v_t/'+ site_name_folder + '_vertical_profile_median_IQR_u_v_t.png', bbox_inches = 'tight', pad_inches=0.0, dpi=400)

# remember: ua: eastward wind (going to the east (jet stream!)), va: northward wind


#%% END OF MAIN!

#%%
# plot Cn2 (possible?)
# calc cn2 out of medians


# PRIMAVERA = True
# ERA5 = True # only change ERA5

# k_factor = climxa.PRIMAVERA_calc_seeing(ds_full, mean_insitu, lon, lat, T_clim_var, U_clim_var, V_clim_var, Z_clim_var, pr_levels_list, site, clim_key, forcing, PRIMAVERA_surface_pressure_value)

save = False

fig, ax = plt.subplots(1,5, figsize = (16, 6), tight_layout = True)

for mod in ['ERA5', 'PRIMAVERA']:
# for mod in ['ERA5']:

    # PRIMAVERA
    if mod == 'PRIMAVERA':
        ds_full = d_model['HadGEM']['ds_seeing'].median(dim='time')
        # pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000

        forcing = 'hist'

        T_clim_var = 'ta ' + forcing
        U_clim_var = 'ua ' + forcing
        V_clim_var = 'va ' + forcing
        Z_clim_var = 'zg ' + forcing

    # ERA5
    if mod == 'ERA5':
        ds_full = copy.deepcopy(ds_ERA5_median) # note: run once inside of plot_seeing_vars_vertical_profile for one idx to get one ERA5 median dataset
        # ds_full['z'] = ds_full['z']/10

        ls_P_ERA5 = list(ds_ERA5_median.level.values)
        # ls_P_ERA5.reverse()

        T_clim_var = 't'
        U_clim_var = 'u'
        V_clim_var = 'v'
        Z_clim_var = 'z'

    # pr_levels_list = ls_P_ERA5

    # use same list for PRIMAVERA AND ERA5 to compare both!
    # pr_levels_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850] # , 925, 1000
    pr_levels_list = [850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50] # , 925, 1000
    # I just tried to see what happens if I integrate in the other direction: looks shifted (I would need to assign different pressure levels)
    # since we start at 850hPa, we get higher Cn2 values!

    J = 0

    ls_pr = []
    ls_cn2 = []
    ls_dudz = []
    ls_theta = []
    ls_dtheta_dz = []
    ls_E = []
    ls_dz = []

    # for i in range(0, len(pr_levels_list)-1):
    # maybe even refine it more by carefully thinking about the 'end'
    # otherwise we loose information!
    for i in range(0, len(pr_levels_list)-1):
        # print(pr_levels_list[i])
        T = ds_full[T_clim_var].sel(level=pr_levels_list[i])
        # P = ds_full.level[i] # WRONG!!!
        P = ds_full.level.sel(level=pr_levels_list[i]) #[i]
        u_i0 = ds_full[U_clim_var].sel(level=pr_levels_list[i])
        u_i1 = ds_full[U_clim_var].sel(level=pr_levels_list[i+1])
        v_i0 = ds_full[V_clim_var].sel(level=pr_levels_list[i])
        v_i1 = ds_full[V_clim_var].sel(level=pr_levels_list[i+1])
        T1 = ds_full[T_clim_var].sel(level=pr_levels_list[i+1])
        # P1 = ds_full.level[int(i+1)]  # wrong!
        P1 = ds_full.level.sel(level=pr_levels_list[i+1])
        df_z_0 = ds_full[Z_clim_var].sel(level=pr_levels_list[i]) # do not divide by g, it is already in m (model data)
        if i == 0:
            df_z_m1 = 0 # not used, but needs a value
        else:
            df_z_m1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i-1])

        # if i == (len(pr_levels_list) - 1):
        #     df_z_p1 = None
        # else:
        df_z_p1 = ds_full[Z_clim_var].sel(level=pr_levels_list[i+1])

        # print(T, P)

        # integrate (sum)
        J_add = climxa.Cn2_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)

        # test if climxa.Cn2_func() doesn't return NaN
        ds_check = J_add.where(xr.ufuncs.isnan(J_add) != True, drop = True)
        # print(ds_check['time'].size)

        J = J + J_add


        # (for Cn2 profile, do not sum over Cn2, but store Cn2 on pressure level dimension)
        # do that!! (and only if J_add is not nan)
        ds_Cn2 = climxa.Cn2_profile_func(T, P, u_i0, u_i1, v_i0, v_i1, T1, P1, i, df_z_0, df_z_m1, df_z_p1)
        ls_pr.append(pr_levels_list[i])
        ls_cn2.append(ds_Cn2.values)

        # calculate du/dz as a toy model
        # if int(i) == 0:
        #     delta_z = abs(df_z_0 - df_z_p1) #df_seeing_era5_sort['z/g'][i] - df_seeing_era5_sort['z/g'][int(i+1)]
        # # elif df_z_p1 == None: # if we have reached the end of the pressure level list
        # #     delta_z = abs(df_z_m1 - df_z_0)
        # else:
        #     delta_z =  abs(0.5 * (df_z_m1 - df_z_p1)) #0.5 * (df_seeing_era5_sort['z/g'][i-1] - df_seeing_era5_sort['z/g'][int(i+1)])

        delta_z = abs(df_z_0 - df_z_p1)
        ls_dz.append(delta_z)

        ls_dudz.append(((u_i1-u_i0) / delta_z)**2)

        ls_theta.append(climxa.theta(T, P))
        ls_dtheta_dz.append(abs(climxa.theta(T1, P1)- climxa.theta(T, P))/ delta_z)
        ls_E.append(climxa.E(u_i0, u_i1, v_i0, v_i1, delta_z))


        # ds_Cn2 = ds_Cn2.assign_coords({"level": pr_levels_list[i]})

        # print(f'Cn2 = {ds_Cn2}')
        # print(f'pressure level = {pr_levels_list[i]}')

        # # concatenate DataArrays along new dimension 'level'
        # if i == 0:
        #     ds_Cn2_profile = ds_Cn2
        # else:
        #     ds_Cn2_profile = xr.concat([ds_Cn2_profile, ds_Cn2], 'level')

        # just use list to append

    print(ls_cn2)
    print(J)

    if mod == 'ERA5':
        ds_sel_ERA5 = ds_ERA5_median.sel(level= pr_levels_list)
        ax[0].plot(ls_cn2, ls_pr)
        ax[1].plot(ds_sel_ERA5.t, pr_levels_list)
        ax[2].plot(ds_sel_ERA5.u, pr_levels_list)
        ax[3].plot(ds_sel_ERA5.v, pr_levels_list)
        ax[4].plot(ds_sel_ERA5.z, pr_levels_list, label='ERA5')

    # ls_cn2_PRIMAVERA = copy.deepcopy(ls_cn2)
    # plt.plot(ls_cn2_PRIMAVERA, ls_pr)
    if mod == 'PRIMAVERA':
        ds_sel_PRIM = ds_full.sel(level= pr_levels_list)
        ax[0].plot(ls_cn2, ls_pr)
        ax[1].plot(ds_sel_PRIM['ta hist'], pr_levels_list)
        ax[2].plot(ds_sel_PRIM['ua hist'], pr_levels_list)
        ax[3].plot(ds_sel_PRIM['va hist'], pr_levels_list)
        ax[4].plot(ds_sel_PRIM['zg hist'], pr_levels_list, label='PRIMAVERA (HadGEM hist)')



# invert axis

# set labels
ls_labels = [r'Cn2 [$m^{1/3}$]' ,r'Temperature [$^{\circ}$ C]', 'eastward wind speed [m/s]', 'northward wind speed [m/s]', 'geopotential height [m]']

for ix, lab in enumerate(ls_labels):
    ax[ix].set_xlabel(lab)
    ax[ix].invert_yaxis()
    ax[ix].set_yticks(pr_levels_list)

ax[0].set_xscale('log') # logarithmic scale
ax[0].set_ylabel('Pressure [hPa]')
# ax.set_title(site_noUnderline)
# ax[4].set_xlabel(ls_labels[climind])
ax[4].legend(loc = 'upper left', bbox_to_anchor = (1.1, 1))

# for integration comparison:
ax[0].set_ylim(900, 50)
ax[0].set_xlim(0.5e-19, 4e-18)

# save
if save:
    fig.savefig('./Model_evaluation/seeing_nc/vertical_profile_u_v_t/'+ 'SPM' + '_vertical_profile_u_v_t_z_Cn2_HadGEM_ERA5.png', bbox_inches = 'tight', pad_inches=0.0, dpi=400)

#%%

plt.plot(ls_dudz, ls_pr)
plt.show()


#%%
# notes: if I change delta_z to delta_z = abs(df_z_0 - df_z_p1), then the vertical profile of du/dz**2 looks the same for both integration ways
# note2: my Cn2 profile looks not quite right. We expect it to have a maximum at 200hPa, but there is a minimum
# check whole formula! for that, create dataframe with one dataset of u,v,t,z
# # save as csv and import to Libreoffice
#  do integration by hand
df_sel_ERA5 = ds_sel_ERA5.to_dataframe()
# to csv
df_sel_ERA5.to_csv('./Model_evaluation/seeing_nc/vertical_profile_u_v_t/'+ 'SPM' + '_dataframe_testing_u_v_t_z_ERA5.csv')

df_cn2_ERA5 = pd.DataFrame([ls_pr, ls_cn2])
df_cn2_ERA5.to_csv('./Model_evaluation/seeing_nc/vertical_profile_u_v_t/'+ 'SPM' + '_dataframe_testing_Cn2_ERA5.csv')

#%% polyfit

# y =ds_ERA5_median.u
# x = ds_ERA5_median.level



def seeing_vars_polyfit(x,y,deg=5, num=100):

    coefficients = np.polyfit(x, y, deg=deg)

    poly = np.poly1d(coefficients)

    # plot
    new_x = np.linspace(x[0], x[-1], num=100) # now we have '100 pressure levels'!!
    new_y = poly(new_x)


    # plt.plot(x, y, "o", new_x, new_y)

    return poly, new_x, new_y

# ds_sel_ERA5
# y =ds_sel_ERA5.v
# x = ds_sel_ERA5.level

# empty dict for functions
d_polyfits = {}

# for y in [ds_sel_ERA5.u, ds_sel_ERA5.v, ds_sel_ERA5.t, ds_sel_ERA5.z]:
for y_var in ['u', 'v', 't', 'level']:

    x = ds_sel_ERA5.z
    y = ds_sel_ERA5[y_var]

    if y_var == 'level':
        # then it is crucial to reverse the order of the Plevs,
        # otherwise we have the same problem!
        y = y.sortby('level', ascending=False)


    poly, new_x, new_y = seeing_vars_polyfit(x, y, deg=5, num=100)

    # calculate the derivative
    # polyderiv = poly(np.polyder(poly)) # this is just wrong!
    polyderiv = np.polyder(poly)
    der_y = polyderiv(new_x)

    d_polyfits[y_var] = {}
    d_polyfits[y_var]['poly'] = poly
    d_polyfits[y_var]['100z'] = new_x # aber wir wollen ableitung nach dz
    d_polyfits[y_var]['100vars'] = new_y
    d_polyfits[y_var]['deriv'] = der_y

#%% plot some

y_var = 'u'

plt.plot(d_polyfits[y_var]['100z'], d_polyfits[y_var]['100vars'])
plt.plot(d_polyfits[y_var]['100z'], d_polyfits[y_var]['deriv'])

plt.plot(d_polyfits[y_var]['100vars'], d_polyfits[y_var]['100z'])

# plev versus z: CHECK
plt.plot(d_polyfits['level']['100z'], d_polyfits['level']['100vars'])
plt.gca().invert_yaxis()

#%%

# P = d_polyfits['t']['100z'] # does not matter of what!
P = d_polyfits['level']['100vars']
T = d_polyfits['t']['100vars']

RCp = 0.286 # R/cp
P_0 = 1000 #mbar

Theta = T*(P_0 /P)**(RCp)
# plt.plot(d_polyfits['t']['100z'], Theta)
# !! dtheta nach dz (und nicht dp)
dTheta_man = d_polyfits['t']['deriv'] * (P_0 /P)**(RCp) - RCp * d_polyfits['level']['deriv'] * T *1/P *(P_0 /P)**(RCp)
dTheta_wolf = 1/P * ( (P_0/P )**(RCp) * (-RCp * T * d_polyfits['level']['deriv'] + P * d_polyfits['t']['deriv']) )

# plt.plot( d_polyfits['t']['100z'], dTheta_man)
# plt.plot( d_polyfits['t']['100z'], dTheta_wolf)

###### poly of Theta for derivative
x = ds_sel_ERA5.z
y = ds_sel_ERA5['t'] * (P_0 / ds_sel_ERA5.level)**RCp
# try:  ds_sel_ERA5['level'].sortby('level', ascending=False)

# plt.plot(x,y)
# plt.plot(x, ds_sel_ERA5.level)

poly2, new_x, new_y = seeing_vars_polyfit(x, y, deg=5, num=100)

# plt.plot(x, y, "o", new_x, new_y)

# calculate the derivative
polyderiv = np.polyder(poly2)
der_y = polyderiv(new_x)

# plt.plot( d_polyfits['t']['100z'], dTheta_wolf)
# plt.plot(new_x, der_y)
# --> calculating theta out of the 11 values and then calculate the polynomial
# or do the derivative by hand and put in the 100vars derived from the polynomial is almost equal!


#####

dTheta_dz = der_y

# test dTheta/dz
th0 = 303.28925207 # or: new_y[0]
th1 = 304.17272728
z0 = 1479.23059082 # or: new_x[0]
z1 = 1668.53565193

test_dTheta_dz = (th1-th0)/(z1-z0) # = 0.0046669391976090135
# but dTheta_dz[0] = 296.3536778452241 # before I removed 'poly' before np.polyder()
# now, dTheta_dz[0] = 0.004683882856482026

# dTheta_wolf is

# append to dict
y_var = 'Theta'
d_polyfits[y_var] = {}
d_polyfits[y_var]['poly'] = poly
d_polyfits[y_var]['100z'] = new_x # aber wir wollen ableitung nach dz
d_polyfits[y_var]['100vars'] = new_y
d_polyfits[y_var]['deriv'] = der_y

k_var = 1
g = 9.80665

dudz = d_polyfits['u']['deriv']
dvdz = d_polyfits['v']['deriv']
E = dudz**2 + dvdz**2
Theta_var = new_y # d_polyfits['Theta']['100vars']
Theta_deriv = der_y # d_polyfits['Theta']['deriv']


Cn2_poly = (80*10**(-6) * P / (T * Theta_var))**2 * k_var * (2 * E / (g/Theta_var * Theta_deriv))**(2/3) * Theta_deriv**2

#%%

# plt.plot(d_polyfits['level']['100z'], d_polyfits['level']['100vars'])

# plot Cn2 profile
plt.plot(Cn2_poly, d_polyfits['level']['100z'])

# or with pressure as basis
plt.plot(Cn2_poly, d_polyfits['level']['100vars'])
plt.gca().invert_yaxis()

# but why minimum at 200hPa??
# we have maximum wind speed there...
plt.plot(E, d_polyfits['level']['100vars'] )
# E has minimum at 200! because the derivative is exactly zero!! turning point.

plt.plot(dudz, d_polyfits['level']['100vars'] )

plt.plot(d_polyfits['u']['100vars'], d_polyfits['level']['100vars'])

# NOTE: the osborn formula is not depending on the wind speed, but of the derivative of the wind speed!!!


#%% integral for J

# J = np.polyint(Cn2_poly) # 'antiderivative' polynom
# plt.plot(J[:100],d_polyfits['level']['100z'])
# # to pressure
# plt.plot(J[:100],d_polyfits['level']['100vars'])
# plt.gca().invert_yaxis()

# use numpy trapezoidal rule
J_trapz = np.trapz(Cn2_poly, x=d_polyfits['level']['100z']) # integrate over dz
# = 7.721706859470486e-15
# right between if I integrate from 50 to 700 and if I integrate from 850 to 100!

#%% find a solution to integrate xarray
# but anyway, there would only be one loop over time. might be okey!


#%%
# I did that in a separate file!

# for idx in range(0,8):

#     if idx == 1:
#         # already saved
#         continue

#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

#     # read data
#     ds_u_v_t = get_seeing_variables(idx, d_site_lonlat_data)

#     # calculate time median and store! (load next time)
#     ds_median = ds_u_v_t.median(dim='time')

#     # save to netcdf
#     median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'

#     ds_median.to_netcdf(median_path)

#%%


    # # load
    # median_path = './Astroclimate_outcome/median_nc_u_v_t/' + site_name_folder + '_median_ERA5_u_v_t_z.nc'

    # ds_median_load = xr.open_dataset(median_path).load()



#%%

fig = plot_seeing_vars_vertical_profile(variable, list_of_clim_vars, list_of_model_clim_params, my_xlabel)

# save fig
fig.savefig('./Model_evaluation/'+ variable + '/All_Sites' + variable + '_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

# fig.savefig('./publication/figures/All_Sites_SH_vertical_profile.pdf', bbox_inches = 'tight', pad_inches=0.0)

plt.show()

#%% only plot after loading in data

# fig = plt.figure(figsize = (8, 20),constrained_layout=True) # (this is not compatible with tight_layout)

# gs = fig.add_gridspec(4, 2)

# ax = fig.add_subplot(gs[int((idx - (idx%2))/2), idx%2])

# pr_levels_list = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] # , 925, 1000
# pr_levels_list.reverse() # exactly opposite to seeing, how it is saved!
# lin_st = ['dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
# line_list = []

# for forcing in d_Ensemble['folders']:

#     # choose the right linestyle
#     if forcing=='hist':
#         forced_linestyle = lin_st[0]
#         color = 'navy'
#     elif forcing=='present':
#         forced_linestyle = lin_st[1]
#         color = 'navy'
#     elif forcing=='future':
#         forced_linestyle = lin_st[2]
#         color = 'red'

#     ax.plot(d_Ensemble['hus ' + forcing], pr_levels_list[0:(len(d_Ensemble['hus ' + forcing]))],
#                     linestyle=forced_linestyle, color=color)

#     # for legend
#     line_list.append(Line2D([0], [0], linestyle = forced_linestyle, color = color, label = 'PRIMAVERA specific humidity ' + forcing))



# ax.plot(ds_ERA5_SH['q'].median(dim='time'), ds_ERA5_SH.level, linestyle = '-', color = '#009e73') # goes until 50hPa (glad I included it until 50hPa!)

# # for legend
# line_list.append(Line2D([0], [0], linestyle = '-', color = '#009e73', label = 'ERA5 specific humidity'))



# ax.set_xscale('log')
# ax.set_yticks(np.arange(0, 1000, 100)) # pressure levels
# plt.gca().invert_yaxis()
# ax.set_xlabel('Specific humidity [kg/kg]')
# ax.set_ylabel('Pressure [hPa]')
# ax.set_title(site_name_folder)
# plt.show()

#%%

# to find out why there is a peak at 70hPa (18641.822 m), look at raw data (ds_full from 'xarray_prepare_seeing_data_PRIMAVERA')
# plot             T_clim_var = 'ta ' + forcing
            # U_clim_var = 'ua ' + forcing
            # V_clim_var = 'va ' + forcing
            # Z_clim_var = 'zg ' + forcing
# plt.plot(ds_full['ua hist'].median(dim='time'), ds_full['ua hist'].level)
# plt.plot(ds_full['va hist'].median(dim='time'), ds_full['va hist'].level)
# plt.plot(ds_full['ta hist'].median(dim='time'), ds_full['ta hist'].level)
# plt.plot(ds_full['zg hist'].median(dim='time'), ds_full['zg hist'].level)

# 200hPa ~= 12401.16m

# # vertical profile
# plt.plot(ds_Cn2_profile["Cn2"].median(dim='time'), ds_Cn2_profile["Cn2"].level)
# plt.gca().invert_yaxis()
# # --> plot looks strange, peak at 150hPa!?
# # write function that plots all vertical profiles!


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

# %% ERA5

# # loop through sites
# for idx in range(0, 8):
#     # or define index for one iteration only
#     # idx = 0

#     print(d_site_lonlat_data['site_name'][idx])
#     # lon_obs and lat_obs are in 0-360 format!!
#     lon = d_site_lonlat_data['lon_obs'][idx]
#     lat = d_site_lonlat_data['lat_obs'][idx]

#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

#     list_of_single_level_vars = ['Cn2']

#     # define d_obs for reading in data
#     d_obs = {"single_lev": list_of_single_level_vars}

#     # read in ds_Cn2 for every model
#     d_model[clim_key]['ds_Cn2'] = climxa.get_PRIMAVERA(d_model, clim_key, site_name_folder, single_level=True)

#     # plot vertical profile (median)
#     plt.plot(d_model[clim_key]['ds_Cn2']["Cn2 " + forcing].median(dim='time'), d_model[clim_key]['ds_Cn2'].level)
#     plt.xscale('log')

#     plt.gca().invert_yaxis()
#     plt.xlabel('Cn2 [m^{1/3}]')
#     plt.ylabel('Pressure [hPa]')


