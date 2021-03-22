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
from matplotlib.colors import to_hex

import pickle

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
# note: check if filename is the up to date!!

#%%

# idx = 1 # test for Paranal

# for idx in range(0, 8):

#     if idx == 4:
#         continue
#     site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
#     print(site_name_folder)
#     ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_level_seeing'][idx]

#     # try with seeing, compare calibrated to uncalibrated
#     file_name = ''
#     df_skill_seeing = pd.read_csv('./Model_evaluation/seeing_nc/csv_info/uncalibrated/' + site_name_folder + file_name + 'sorted_skill_dict.csv', header=None)

#     file_name = '_calibrated_'
#     df_skill_seeing_calib = pd.read_csv('./Model_evaluation/seeing_nc/csv_info/' + site_name_folder + file_name + 'sorted_skill_dict.csv', header=None)

#     ls_df = [df_skill_seeing, df_skill_seeing_calib]


#     # define mod list
#     # list with all relevant 
#     ls_mod = list(df_skill_seeing[0][1:])
#     ls_mod_calib = list(df_skill_seeing_calib[0][1:])

#     # choose which statistics from taylor diagram we use here
#     # 1 = skill
#     # 2 = corr # should be the same
#     # 3 = std dev
#     taylor_idx = 3

#     for idx_mod, mod in enumerate(ls_mod):
#         # uncalibrated in red
#         color='r'
#         marker='o'
#         if 'Ensemble' in mod:
#             marker = 'd'
#         elif 'ERA5' in mod:
#             marker = 'p'
#         plt.scatter(idx_mod, df_skill_seeing.loc[df_skill_seeing[0] == mod][taylor_idx], color=color, marker=marker)


#         # calibrated in blue
#         color='b'
#         marker='o'
#         if 'Ensemble' in mod:
#             marker = 'd'
#         elif 'ERA5' in mod:
#             marker = 'p'
#         plt.scatter(idx_mod, df_skill_seeing_calib.loc[df_skill_seeing_calib[0] == mod][taylor_idx], color=color, marker=marker)

#         plt.legend(['uncalibrated', 'calibrated'])

#     plt.title(site_name_folder + ' std dev')
#     plt.show()


# OR: plot taylor diagram...

# plot skill


#%% MAIN

# first, do it for version with in-situ as reference,
# next, do it for version with ERA5 as reference
# plot all in one plot

# choose if we should compute the tables or read in saved csv files
compute = False

# prepare figure
fig = plt.figure(figsize = (16, 28)) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
gs = fig.add_gridspec(7, 2) # first entry: rows, second entry: columns


# for file_name in ['_', '_ERA5_to_PRIMAVERA_']:
ls_var = ['T','RH','SH', 'TCW', 'wind_speed_seeing', 'seeing_osborn', 'total_cloud_cover']
# ls_var = ['T','RH','SH']
# ls_var = ['wind_speed_seeing', 'seeing_osborn']

max_idx = len(ls_var) - 1

for idx, var in enumerate(ls_var):

    if var == 'total_cloud_cover':
        # clouds
        folder_name = 'total_cloud_cover'
        # file_name = '_nighttime_'
        pressure_level = False
        # pressure_level = True
        title_var = 'cloud cover'
        variable = 'cloud cover'
    elif var == 'T':
        # Temperature
        folder_name = 'T'
        pressure_level = True
        title_var = 'T'
        variable = 'Temperature'
    elif var == 'SH':
        # specific humidity
        folder_name = 'SH'
        pressure_level = True
        title_var = 'SH'
        variable = 'Specific Humidity'
    elif var == 'RH':
        # relative humidity
        folder_name = 'RH'
        pressure_level = True
        title_var = 'RH'
        variable = 'Relative Humidity'
    elif var == 'TCW':
        # precipitable water vapor
        folder_name = 'TCW'
        pressure_level = True
        title_var = 'PWV'
        variable = 'Precipitable Water Vapor'

        # (seeing only works for PRIMAVERA to ERA5!
        # others are not separated scripts..
        # but I could do!) --> I did
        # 200hPa-wind-speed seeing
    # seeing vars have the same folder name! --> include variable instead of folder_name!
    elif var == 'wind_speed_seeing':
        folder_name = 'seeing_nc'
        pressure_level = False
        title_var = '200hPa-seeing'
        variable = '200hPa-wind-speed seeing'
    elif var == 'seeing_osborn':
        # osborn seeing
        folder_name = 'seeing_nc'
        pressure_level = True
        title_var = 'seeing model'
        variable = 'seeing model'

    print(var)

    # changes are not permament yet..
    # save new skill dicts of workflow 4.1 as '_pub_'

    #### in-situ as ref
    if folder_name == 'T' or folder_name == 'SH' or folder_name == 'RH'or folder_name == 'total_cloud_cover':
        file_name = '_pub_' # workflow 4.1 (SH, T, RH only!)

    elif var  == 'wind_speed_seeing':
        file_name = '_200hPa_wind_speed_pub_'
    elif var == 'seeing_osborn':
        file_name = '_osborn_pub_'

    else:
        file_name = '_'
        # file_name = '_trade_off_fixed_Plev_'

    # else: # file_name = '_nighttime_'
    #     file_name = '_nighttime_'

    if compute == True:

        df_ERA5_insitu, df_PRIMAVERA_insitu, df_class_insitu = climxa.skill_score_classification_to_table(file_name, folder_name, title_var, variable, pressure_level = pressure_level)

        # save to .csv
        # if we do not compare ERA5 to PRIMAVERA (but if we have in-situ as reference)
        df_ERA5_insitu.to_csv('./Astroclimate_outcome/skill_score_classification/ERA5_skill_score_classification_' + var + '.csv', index=False)
        df_PRIMAVERA_insitu.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_skill_score_classification_' + var + '.csv', index=False)
        df_class_insitu.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_and_ERA5_to_insitu_skill_score_for_plotting_' + var + '.csv', index=False)
    
    else:
        df_class_insitu = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_and_ERA5_to_insitu_skill_score_for_plotting_' + var + '.csv')


    #### ERA5 as ref
    # if folder_name == 'total_cloud_cover':
    #     file_name = '_ERA5_to_PRIMAVERA__'
    if var  == 'wind_speed_seeing':
        file_name = '_ERA5_to_PRIMAVERA_200hPa_wind_speed_'
    elif var == 'seeing_osborn':
        file_name = '_ERA5_to_PRIMAVERA_osborn_'
    else:
        file_name = '_ERA5_to_PRIMAVERA_'
    # file_name = '_ERA5_to_PRIMAVERA_allPrlevels'

    if compute == True:
        # , ERA5_is_ref=True
        df_ERA5, df_PRIMAVERA_to_ERA5, df_class_PRIMAVERA_to_ERA5, df_ensemble_match_diff = climxa.skill_score_classification_to_table(file_name, folder_name, title_var, variable, pressure_level = pressure_level, ERA5_is_ref=True)
        # note: df_ERA5 is empty (ignore)
        # save; add ERA5_to_PRIMAVERA to saved filename
        df_PRIMAVERA_to_ERA5.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_skill_score_classification_ERA5_to_PRIMAVERA_' + var + '.csv', index=False)

        # we need to save best matching simulation (hist or SST present?)
        df_ensemble_match_diff.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_difference_' + var + '.csv', index=False)
    
        # save df_class_PRIMAVERA_to_ERA5 for plotting
        df_class_PRIMAVERA_to_ERA5.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_for_plotting_' + var + '.csv', index=False)
    
    else: # read in files
        df_class_PRIMAVERA_to_ERA5 = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_for_plotting_' + var + '.csv')

    # OR: read from files (would be faster)
    #
    # PLOTTING
    # show skill score classifications

    # add subplot to figure
    ax = fig.add_subplot(gs[int((idx - (idx%2))/2), idx%2])

    ######### do not sort in second approach (where we combine both)

    # draw barplots with skill scores (sorted after PRIMAVERA)
    ax.barh(2* df_class_insitu.index, df_class_insitu['ERA5'], height=0.4,  align='center', color = '#0072b2', label='ERA5 to in-situ')
    ax.barh(2* df_class_insitu.index + 0.4, df_class_insitu['PRIMAVERA'], height=0.4, align='center', color = '#e69f00', label='PRIMAVERA to in-situ')

    # add PRIMAVERA to ERA5 (df_class_PRIMAVERA_to_ERA5)
    ax.barh(2* df_class_PRIMAVERA_to_ERA5.index + 0.8, df_class_PRIMAVERA_to_ERA5['PRIMAVERA'], height=0.4,  align='center', color = 'red', label='PRIMAVERA to ERA5')

    # define from where we take index and site (shouldn't matter)
    df_class_sorted = df_class_insitu

    # set yticks and labels (sites) and invert yaxis so that Mauna Kea is on top
    ax.set_yticks(2* df_class_sorted.index)
    ax.set_yticklabels(df_class_sorted['Site'])
    ax.invert_yaxis()  # labels read top-to-bottom

    # ax.set_title(folder_name)
    # x label
    ax.set_xlabel('skill score classification', fontsize=10)
    ax.set_xlim(0,1)

    # set legend only for last subplot
    if var == ls_var[max_idx]: # only true for last index
        ax.legend(loc='lower left', bbox_to_anchor= (1.2, 0))
        
    # set title
    # ax (gca()) and not plt.gcf()
    # plt.gca().text(0.5, 1.2, variable, horizontalalignment='center', fontsize=16)
    ax.set_title(climxa.alphabet_from_idx(idx) + ') ' + variable, y = 1.07, fontsize=14)

    # shrink axis to make space for xlabels
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

    # call function to plot classification axspans (Poor, Mediocre, Good, Excellent)
    climxa.plot_axspan_classification(ax)

    # save fig
    if var == ls_var[max_idx]: # only true for last index
        # png for powerpoint
        # fig.savefig('./Model_evaluation/' + folder_name + '/'  + var + '_PRIMAVERA_to_ERA5_skill_scores.png', bbox_inches = 'tight', pad_inches=0.0, dpi=500)
        # pdf for publication
        fig.savefig('./publication/figures/'  + 'All_vars' + '_PRIMAVERA_to_ERA5_skill_scores.pdf', bbox_inches = 'tight', pad_inches=0.0)






#%% compose new table for comparison (and for latex!)
# seeing

# fill a new dataframe (that looks already like a table)
# that is, fill list that you in the end stick together to a dataframe

ls_site = []
# ERA5
ls_ERA5_classif_osborn = []
ls_ERA5_variable_osborn = []
ls_ERA5_raw_skill_osborn = []
ls_ERA5_raw_corr_osborn = []
ls_ERA5_classif_wind = []
ls_ERA5_variable_wind = []
ls_ERA5_raw_skill_wind = []
ls_ERA5_raw_corr_wind = []

# PRIMAVERA
ls_PRIMAVERA_classif_osborn = []
ls_PRIMAVERA_variable_osborn = []
ls_PRIMAVERA_raw_skill_osborn = []
ls_PRIMAVERA_raw_corr_osborn = []
ls_PRIMAVERA_classif_wind = []
ls_PRIMAVERA_variable_wind = []
ls_PRIMAVERA_raw_skill_wind = []
ls_PRIMAVERA_raw_corr_wind = []

# and also create nested list for the check of the length (it gets filled as the list gets bigger!)
nested_list_dict = {'ls_ERA5_classif_osborn': ls_ERA5_classif_osborn, 'ls_ERA5_variable_osborn': ls_ERA5_variable_osborn,
                    'ls_ERA5_classif_wind': ls_ERA5_classif_wind, 'ls_ERA5_variable_wind': ls_ERA5_variable_wind,
                    'ls_PRIMAVERA_classif_osborn': ls_PRIMAVERA_classif_osborn, 'ls_PRIMAVERA_variable_osborn': ls_PRIMAVERA_variable_osborn,
                    'ls_PRIMAVERA_classif_wind': ls_PRIMAVERA_classif_wind, 'ls_PRIMAVERA_variable_wind': ls_PRIMAVERA_variable_wind }

# fill dict with lists
# table_dict = {'site': [], 'ERA5_'}


for idx in range(0, 8):

    site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
    print(site_name_folder)
    ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_level_seeing'][idx]
    lon = d_site_lonlat_data['lon_obs'][idx]
    lat = d_site_lonlat_data['lat_obs'][idx]

    if site_name_folder == 'MaunaKea':
        site_noUnderline = 'Mauna Kea'
    elif site_name_folder == 'siding_spring':
        site_noUnderline = 'Siding Spring'
    else:
        site_noUnderline = site_name_folder.replace('_', ' ')

    # set counter for PRIMAVERA ensemble means to zero for each site
    count_wind = 0
    count_osborn = 0

    # fill list of sites
    ls_site.append(site_noUnderline)

    # for sites where we have no in situ reference, we have to exclude them here!
    if idx == 5 or idx == 6:
        # append 'nm' for 'no measurement' to list, so that still all sites are represented
        ls_ERA5_classif_osborn.append(' ')
        ls_ERA5_variable_osborn.append('nm')
        ls_ERA5_raw_corr_osborn.append(np.nan)
        ls_ERA5_classif_wind.append(' ')
        ls_ERA5_variable_wind.append('nm')
        ls_ERA5_raw_corr_wind.append(np.nan)

        ls_PRIMAVERA_classif_osborn.append(' ')
        ls_PRIMAVERA_variable_osborn.append('nm')
        ls_PRIMAVERA_raw_corr_osborn.append(np.nan)
        ls_PRIMAVERA_classif_wind.append(' ')
        ls_PRIMAVERA_variable_wind.append('nm')
        ls_PRIMAVERA_raw_corr_wind.append(np.nan)

        # also append to 'raw_skill', for plotting (otherwise we mix indices)
        ls_ERA5_raw_skill_wind.append(np.nan)
        ls_ERA5_raw_skill_osborn.append(np.nan)
        ls_PRIMAVERA_raw_skill_osborn.append(np.nan)
        ls_PRIMAVERA_raw_skill_wind.append(np.nan)

        continue      



    # load in dataset
    file_name = '_calibrated_'
    df_skill_seeing_calib = pd.read_csv('./Model_evaluation/seeing_nc/csv_info/' + site_name_folder + file_name + 'sorted_skill_dict.csv', header=None)

    # go through rows of dataframe
    for index, row in df_skill_seeing_calib.iterrows():
        # print(row[0])

        # ERA5: stop if ERA5 is in first column of that row
        if 'ERA5' in row[0]:
            # print(row)
            if 'wind' in row[0]:
                # append classification to the classification list of the 200hPa wind speed seeing
                ls_ERA5_classif_wind.append(climxa.skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                # append string to the variable list
                ls_ERA5_variable_wind.append('200hPa')
                # to calculate the mean in the end
                ls_ERA5_raw_skill_wind.append(row[1])
                ls_ERA5_raw_corr_wind.append(row[2])

            else: # osborn seeing
                # append classification to the classification list of the osborn seeing
                ls_ERA5_classif_osborn.append(climxa.skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                # append string to the variable list
                # [:5] to strip of 'ERA5
                ls_ERA5_variable_osborn.append(row[0][5:] + 'hPa') # for ERA5, the pressure level is written correctly in the label
                # to calculate the mean in the end
                ls_ERA5_raw_skill_osborn.append(row[1])
                ls_ERA5_raw_corr_osborn.append(row[2])

        # for the PRIMAVERA ensemble means, there are more than two in each dataset, but we want only the best one! (either 'hist' or 'present')
        elif 'Ensemble' in row[0]: # we only want the Ensemble skills
            # print(row)
            # get the integral limit (just take one clim_key)
            clim_key = 'HadGEM'
            PRIMAVERA_surface_pressure_value = climxa.get_PRIMAVERA_surface_pressure_level(clim_key, site_name_folder, lon, lat)
            # print('PRIMAVERA_surface_pressure_value')

            # separate the string by spaces and save the last word ('hist', 'present', ...)
            label_split = row[0].split(' ') # is a list
            forcing = label_split[-1]
            # assign correct string
            if forcing=='present':
                my_label = 'SST present'
            elif forcing == 'SSTfuture': # should have only 'present' and 'hist' up to now..
                my_label = 'SST future'
            else:
                my_label = forcing

            if 'wind' in row[0] and count_wind == 0:
                # get the skill score classification and compose string
                ls_PRIMAVERA_classif_wind.append(climxa.skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                # append the forcing
                ls_PRIMAVERA_variable_wind.append('200hPa ' + my_label)
                # for mean
                ls_PRIMAVERA_raw_skill_wind.append(row[1])
                ls_PRIMAVERA_raw_corr_wind.append(row[2])
                # increase counter of the wind speed seeing to 1 so that the next (poorer) wind speed seeing does not get into the lists
                count_wind = 1


            elif 'wind' not in row[0] and count_osborn == 0: # osborn seeing
                ls_PRIMAVERA_classif_osborn.append(climxa.skill_score_classification(row[1]) + ' ({})'.format(round(row[1],2)))
                ls_PRIMAVERA_variable_osborn.append(str(PRIMAVERA_surface_pressure_value) + 'hPa ' + my_label) 
                ls_PRIMAVERA_raw_skill_osborn.append(row[1])
                ls_PRIMAVERA_raw_corr_osborn.append(row[2])
                count_osborn = 1

# check length of lists!
for ls_key in nested_list_dict.keys():
    if len(nested_list_dict[ls_key]) > 8:
        raise Exception('investigate {}'.format(ls_key))


# compose dataframe
df_ERA5 = pd.DataFrame(list(zip(ls_site, ls_ERA5_variable_wind, ls_ERA5_classif_wind, ls_ERA5_variable_osborn, ls_ERA5_classif_osborn)),
                    columns = ['Site', '200hPa seeing', 'classification', 'seeing model', 'classification'])

df_PRIMAVERA = pd.DataFrame(list(zip(ls_site, ls_PRIMAVERA_variable_wind, ls_PRIMAVERA_classif_wind, ls_PRIMAVERA_variable_osborn, ls_PRIMAVERA_classif_osborn)),
                    columns = ['Site', '200hPa seeing', 'classification', 'seeing model', 'classification'])

# save to .csv
df_ERA5.to_csv('./Astroclimate_outcome/skill_score_classification/ERA5_seeing.csv', index=False)
df_PRIMAVERA.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_seeing.csv', index=False)

# show correlations (for my information rather than for latex)
df_corr = pd.DataFrame(list(zip(ls_site, ls_ERA5_raw_corr_wind, ls_ERA5_raw_corr_osborn, ls_PRIMAVERA_raw_corr_wind, ls_PRIMAVERA_raw_corr_osborn)),
                        columns = ['Site', 'ERA5 wind speed', 'ERA5 osborn', 'PRIMAVERA wind', 'PRIMAVERA osborn'])
# store classifications in dataframe for plotting
df_skill_seeing_insitu = pd.DataFrame(list(zip(ls_site, ls_ERA5_raw_skill_wind, ls_ERA5_raw_skill_osborn, ls_PRIMAVERA_raw_skill_wind, ls_PRIMAVERA_raw_skill_osborn)),
                        columns = ['Site', 'ERA5 wind speed', 'ERA5 osborn', 'PRIMAVERA wind', 'PRIMAVERA osborn'])


# calculations with corr
# drop nan first
df_corr_nonan = df_corr.dropna(how = 'any')
# calc mean, max, min
df_corr_nonan.max(axis=0) # max values of different approaches
df_corr_nonan.max(axis=1) # max values of each site
df_corr_nonan.min(axis=0) # min values of different approaches
df_corr_nonan.min(axis=1) # min values of each site


# calculate mean skill (just for me)
print('mean of ERA5 wind speed seeing classification = {}'.format(np.mean(ls_ERA5_raw_skill_wind)))
print('mean of ERA5 osborn seeing classification = {}'.format(np.mean(ls_ERA5_raw_skill_osborn)))
print('mean of PRIMAVERA wind speed seeing classification = {}'.format(np.mean(ls_PRIMAVERA_raw_skill_wind)))
print('mean of PRIMAVERA osborn seeing classification = {}'.format(np.mean(ls_PRIMAVERA_raw_skill_osborn)))

#%% seeing: PRIMAVERA to ERA5

file_name = '_ERA5_to_PRIMAVERA_'

# osborn seeing
folder_name = 'seeing_nc'
pressure_level = True
title_var = 'seeing model'
variable = 'seeing model'

df_ERA5, df_PRIMAVERA_to_ERA5, df_class_PRIMAVERA_to_ERA5_osborn = skill_score_classification_to_table(file_name, folder_name, title_var, variable, pressure_level = False)
# note: df_ERA5 is empty (ignore)
# save; add ERA5_to_PRIMAVERA to saved filename
# df_PRIMAVERA_to_ERA5.to_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_skill_score_classification_ERA5_to_PRIMAVERA_' + folder_name + '.csv', index=False)
# note: I saved it once already

folder_name = 'seeing_nc'
pressure_level = False
title_var = '200hPa-seeing'
variable = '200hPa-wind-speed seeing'

df_ERA5, df_PRIMAVERA_to_ERA5, df_class_PRIMAVERA_to_ERA5_wind_speed = skill_score_classification_to_table(file_name, folder_name, title_var, variable, pressure_level = False)

#%% plot seeing (separate, because we have two approaches)

folder_name = 'seeing_nc'
seeing_title = 'Seeing'

fig, ax = plt.subplots(figsize=(5,8))
# draw barplots with skill scores (sorted after PRIMAVERA)

fact_stretch = 10
height = 1.1

# to compare plots, plot first ERA5 osborn, then ERA5 wind speed, and so on
# ERA5
ax.barh(fact_stretch* df_skill_seeing_insitu.index - 3*height, df_skill_seeing_insitu['ERA5 osborn'], edgecolor = 'k', linewidth= 1.5, height=height,  align='center', color = '#0072b2', label='ERA5 seeing model to in-situ')
ax.barh(fact_stretch* df_skill_seeing_insitu.index - 2*(height- 0.2), df_skill_seeing_insitu['ERA5 wind speed'], edgecolor = 'r', linewidth= 1, height=height,  align='center', color = '#0072b2', label='ERA5 200hPa-wind-speed seeing to in-situ')

ax.barh(fact_stretch* df_skill_seeing_insitu.index - 0.5*height, df_skill_seeing_insitu['PRIMAVERA osborn'], edgecolor = 'k', linewidth= 1.5,  height=height, align='center', color = '#e69f00', label='PRIMAVERA seeing model to in-situ')
ax.barh(fact_stretch* df_skill_seeing_insitu.index + 0.5*(height + 0.2), df_skill_seeing_insitu['PRIMAVERA wind'], edgecolor = 'r', linewidth= 1, height=height, align='center', color = '#e69f00', label='PRIMAVERA 200hPa-wind-speed seeing to in-situ')

# add PRIMAVERA to ERA5 (df_class_PRIMAVERA_to_ERA5)
# osborn
ax.barh(fact_stretch* df_class_PRIMAVERA_to_ERA5_wind_speed.index + 2*height, df_class_PRIMAVERA_to_ERA5_osborn['PRIMAVERA'], edgecolor = 'k', linewidth= 1.5, height=height,  align='center', color = 'red', label='seeing model PRIMAVERA to ERA5')
# wind speed
ax.barh(fact_stretch* df_class_PRIMAVERA_to_ERA5_wind_speed.index + 3* (height + 0.1), df_class_PRIMAVERA_to_ERA5_wind_speed['PRIMAVERA'], edgecolor = 'r', linewidth= 1, height=height,  align='center', color = 'red', label='200hPa-wind-speed PRIMAVERA to ERA5')


df_class_sorted = df_skill_seeing_insitu

ax.set_yticks(fact_stretch* df_class_sorted.index)
ax.set_yticklabels(df_class_sorted['Site'])
ax.invert_yaxis()  # labels read top-to-bottom

plot_axspan_classification(ax)

# set title
plt.gcf().text(0.36, 0.95, seeing_title, fontsize=16)
# ax.set_title(folder_name)
# x label
ax.set_xlabel('skill score classification')
ax.legend(loc='upper left', bbox_to_anchor=(0,-0.1))

# save fig
# png for powerpoint
fig.savefig('./Model_evaluation/' + folder_name + '/'  + folder_name + '_PRIMAVERA_to_ERA5_skill_scores.png', bbox_inches = 'tight', pad_inches=0.0, dpi=500)

# for inkscape, as pdf
# fig.savefig('./Model_evaluation/' + folder_name + '/'  + folder_name + '_PRIMAVERA_to_ERA5_skill_scores.pdf', bbox_inches = 'tight', pad_inches=0.0)



#%%

#################


# # plot PRIMAVERA and ERA5 skill score
# ### plot PRIMAVERA skill score (separately and together!)
# # reset index in order to use the index for plotting
# df_class_sorted_PRIMAVERA = df_class_insitu.sort_values('PRIMAVERA').reset_index(drop=True)
# df_class_sorted_ERA5 = df_class_insitu.sort_values('ERA5').reset_index(drop=True)
# # plot
# fig, ax = plt.subplots()
# # draw barplots with skill scores (sorted after PRIMAVERA)
# # sorted after PRIMAVERA

# # ax.barh(2* df_class_sorted_PRIMAVERA.index, df_class_sorted_PRIMAVERA['PRIMAVERA'], height=0.5, align='center', color = '#e69f00', label='PRIMAVERA')
# # ax.barh(2* df_class_sorted_PRIMAVERA.index + 0.5, df_class_sorted_PRIMAVERA['ERA5'], height=0.5,  align='center', color = '#0072b2', label='ERA5')

# # sorted after ERA5
# ax.barh(2* df_class_sorted_ERA5.index, df_class_sorted_ERA5['PRIMAVERA'], height=0.5, align='center', color = '#e69f00', label='PRIMAVERA')
# ax.barh(2* df_class_sorted_ERA5.index + 0.5, df_class_sorted_ERA5['ERA5'], height=0.5,  align='center', color = '#0072b2', label='ERA5')


# df_class_sorted = df_class_sorted_ERA5

# ax.set_yticks(2* df_class_sorted.index)
# ax.set_yticklabels(df_class_sorted['Site'])
# ax.invert_yaxis()  # labels read top-to-bottom

# # draw shaded areas for 'poor', 'mediocre', 'good', 'excellent'
# # sample color from 'plasma' colorbar
# # needs 'from matplotlib.colors import to_hex'
# # ls_col_class = list(reversed([to_hex(plt.cm.copper(i / 5)) for i in range(5)]))
# ls_col_class = [to_hex(plt.cm.copper(i / 5)) for i in range(5)]
# # poor
# ax.axvspan(0, 0.314, alpha=0.45, color=ls_col_class[0])
# # mediocre
# ax.axvspan(0.315, 0.624, alpha=0.45, color=ls_col_class[1])
# # good
# ax.axvspan(0.625, 0.904, alpha=0.45, color=ls_col_class[2])
# # excellent
# ax.axvspan(0.905, 1, alpha=0.45, color=ls_col_class[3])
# # ax.text(0.2, -0.3, 'Poor')
# plt.gcf().text(0.2, 0.9, 'Poor', fontsize=12)
# plt.gcf().text(0.4, 0.9, 'Mediocre', fontsize=12)
# plt.gcf().text(0.65, 0.9, 'Good', fontsize=12)
# plt.gcf().text(0.8, 0.9, 'Excellent', fontsize=12)

# # set title
# plt.gcf().text(0.4, 1, variable, fontsize=16)
# # ax.set_title(folder_name)
# # x label
# ax.set_xlabel('skill score classification')
# ax.legend(loc='upper left', bbox_to_anchor=(1,1))
     



# else: # no ERA5
#     df_corr = pd.DataFrame(list(zip(ls_site, ls_PRIMAVERA_raw_corr)),
#                         columns = ['Site', 'PRIMAVERA'])

#     ### plot PRIMAVERA skill score
#     # sort and reset index
#     df_corr_sorted = df_corr.sort_values('PRIMAVERA').reset_index(drop=True)
#     # plot
#     fig, ax = plt.subplots()
#     ax.barh(df_corr_sorted.index, df_corr_sorted['PRIMAVERA'],  align='center')
#     ax.set_yticks(df_corr_sorted.index)
#     ax.set_yticklabels(df_corr_sorted['Site'])
#     ax.invert_yaxis()  # labels read top-to-bottom
#     # draw shaded areas for 'poor', 'mediocre', 'good', 'excellent'
#     # sample color from 'plasma' colorbar
#     # needs 'from matplotlib.colors import to_hex'
#     # ls_col_class = list(reversed([to_hex(plt.cm.copper(i / 5)) for i in range(5)]))
#     ls_col_class = [to_hex(plt.cm.copper(i / 5)) for i in range(5)]
#     # poor
#     ax.axvspan(0, 0.314, alpha=0.5, color=ls_col_class[0])
#     # mediocre
#     ax.axvspan(0.315, 0.624, alpha=0.5, color=ls_col_class[1])
#     # good
#     ax.axvspan(0.625, 0.904, alpha=0.5, color=ls_col_class[2])
#     # excellent
#     ax.axvspan(0.905, 1, alpha=0.5, color=ls_col_class[3])
#     # ax.text(0.2, -0.3, 'Poor')
#     plt.gcf().text(0.2, 0.9, 'Poor', fontsize=12)
#     plt.gcf().text(0.4, 0.9, 'Mediocre', fontsize=12)
#     plt.gcf().text(0.65, 0.9, 'Good', fontsize=12)
#     plt.gcf().text(0.8, 0.9, 'Excellent', fontsize=12)
#     # set title
#     plt.gcf().text(0.4, 1, variable, fontsize=16)
#     # ax.set_title(folder_name)
#     # x label
#     ax.set_xlabel('skill score classification')



# %%
