# 2020-12-22
# produce a plot that summarizes all my results

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

from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

import csv

import seaborn as sns
# sns.set()
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


#%%
def Prlevel_attributes_T_RH_SH():
    # generate attribute list automatically
    attrib = []
    attrib_ERA5 = []

    for idx in range(0,8):
        # get pressure levels saved in d_site_lonlat_data for ERA5 and PRIMAVERA
        ls_pr_levels_clim_model = d_site_lonlat_data['ls_pr_levels_clim_model'][idx]
        ls_pr_levels_ERA5 = d_site_lonlat_data['ls_pr_levels_ERA5'][idx] # code reads in automatically data for these levels
        # append to list
        # [0] because there is only one pressure level (or should be!)
        attrib.append('_' + str(ls_pr_levels_clim_model[0]))
        attrib_ERA5.append('_' + str(ls_pr_levels_ERA5[0]))

    return attrib_ERA5, attrib

# attrib_ERA5, attrib = Prlevel_attributes_T_RH_SH()


#%% define function

def trend_analysis_plot(variable, gs, fig, fig_idx):
    # custom parameters for individual variables
    if variable == 'wind_speed_seeing':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/wind_speed/"
        attrib = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        unit = 'arcsec'
        title = '200hPa-wind-speed seeing'

    elif variable == 'seeing_osborn':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/osborn/"
        attrib = ['_5', '_5', '_5', '_5','_5', '_5', '_5', '_5'] # ['_200', '_200', '_200', '_200','_200', '_200', '_200', '_200']
        attrib_ERA5 = ['_800', '_900', '_825', '_825','_975', '_950', '_850', '_850']
        unit = 'arcsec'
        title = 'seeing model'

    elif variable == 'SH':
        base_path =  "./Model_evaluation/SH/future_trends/"
        # attrib = ['_700', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib_ERA5 = ['_600', '_775', '_775', '_775','_800', '_950', '_850', '_750']
        unit = '(g/kg)'
        title = 'Specific Humidity'

    elif variable == 'RH':
        base_path =  "./Model_evaluation/RH/future_trends/"
        # attrib = ['_600', '_700', '_850', '_850','_850', '_1000', '_850', '_700']
        # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
        # attrib_ERA5 = ['_600', '_750', '_775', '_750','_850', '_950', '_875', '_750']
        unit = '%'
        title = 'Relative Humidity'

    elif variable == 'T':
        base_path =  "./Model_evaluation/T/future_trends/"
        # attrib = ['_single_level', '_single_level', '_850', '_single_level','_850', '_925', '_single_level', '_700']
        # attrib = ['_600', '_700', '_850', '_850','_700', '_single_level', '_single_level', '_700']
        # attrib_ERA5 = ['_600', '_750', '_750', '_775','_750', '_900', '_875', '_750']
        unit = '°C'
        title = 'Temperature'

    elif variable == 'total_cloud_cover':
        base_path =  "./Model_evaluation/total_cloud_cover/future_trends/"
        attrib = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
        attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
        unit = '%'
        title = 'Total cloud cover'

    elif variable == 'TCW':
        base_path = "./Model_evaluation/TCW/future_trends/"
        attrib = ['_600', '_700', '_700', '_850','_700', '_925', '_850', '_700']
        attrib_ERA5 = ['_600', '_750', '_775', '_775','_775', '_900', '_825', '_750']
        unit = 'mmH2O'
        title = 'Precipitable water vapor'


    # for T, RH and SH, we have fixed Plevs
    if variable == 'T' or variable == 'RH' or variable == 'SH':
        attrib_ERA5, attrib = Prlevel_attributes_T_RH_SH()

    ls_site_names = []
    line_list = []

    ls_hex = [to_hex(plt.cm.terrain(i / 8)) for i in range(8)]

    # fig, ax = plt.subplots(figsize=(8,4))
    # add subplot to figure
    ax = fig.add_subplot(gs[int((fig_idx - (fig_idx%2))/2), fig_idx%2])

    # for every variable, read in csv that tells us how big the difference between PRIMAVERA Ensemble best and second best match is
    # we want to display this in the trend analysis plot so that we know which scored best
    df_ensemble_match_diff = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_difference_' + variable + '.csv')


    for idx in range(0, 8):

        site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
        # "Mauna_Kea", "Cerro_Paranal", "La_Silla", "Cerro_Tololo", "La_Palma", "Siding_Spring", "Sutherland", "SPM"
        # change site name if needed
        if idx == 0: # mauna kea
            site_name_folder = 'Mauna_Kea'
        if idx == 1:
            site_name_folder = 'Cerro_Paranal'

        ls_site_names.append(site_name_folder.replace('_', ' '))



        ls_forcing = ['hist', 'present', 'future', 'SSTfuture', 'ERA5'] # 'future', 'SSTfuture', and eventually 'ERA5'
        for forcing in ls_forcing:

            ########## ERA5
            if forcing == 'ERA5':
                if variable == 'wind_speed_seeing' or variable == 'seeing_osborn':
                    ERA5_path = "./Model_evaluation/seeing_nc/ERA5_trends/"
                else:
                    ERA5_path = "./Model_evaluation/" + variable + "/ERA5_trends/"
                df = pd.read_csv(ERA5_path + site_name_folder + attrib_ERA5[idx] + variable + '_' + '_ERA5_Projections_Bayesian_model_map2stan.csv')

            else:
                # read PRIMAVERA csv
                df = pd.read_csv(base_path + site_name_folder + '_' + forcing + attrib[idx] + variable + '_' + '_PRIMAVERA_Projections_Bayesian_model_map2stan.csv')

            # rename
            df = df.rename(columns={'Unnamed: 0': 'parameter'})

            # plot b (the gradient of the slope if the seasonal variation was excluded)
            # use idx as x axis (idx and idx+0.5)
            # get marker from function
            # df['mean'][1] is the gradient b
            # multiply with 10 to get it in unit/decade
            if forcing == 'future' or forcing == 'SSTfuture':

                color = '#009e73'

                if forcing == 'future':
                    x_idx = idx + 0.8
                else:
                    x_idx = idx + 0.95

            else:
                if forcing == 'ERA5':
                    color = 'k' # ERA5 is the new ground truth and should therefore be black # '#0072b2' # '#56b4e9'
                    x_idx = idx + 0.1

                else: # hist or present
                    color = '#e69f00'#
                    if forcing == 'hist':
                        x_idx = idx + 0.3
                    else:
                        x_idx = idx + 0.45
            # get color from climxa.return_model_color(Mname)
            # color = climxa.return_model_color(forcing)[0] # is list...
            # DO NOT get color from climxa, because then, coupled have the same color, but here I want future to have the same color!

            # get marker for individual forcings (and ERA5)
            marker = climxa.trend_marker(forcing)
            markersize_scale = 20

            # not for ERA5
            if forcing != 'ERA5': # only for PRIMAVERA trends
                if forcing == 'present' or forcing == 'hist': #
                    # get markersize and alpha-value depending on difference
                    if df_ensemble_match_diff['Best Simulation'][idx] == forcing:
                        # if the best simulation matches the current forcing, apply no special markersize and alpha
                        # alpha = 0.9
                        markersize = 11
                        # but we have to store somehow if coupled or atmos-only is best match!
                        # atmos-only (SST present, SST future)
                        if forcing == 'present':
                            sim_keyword = 'atmos-only'
                            # sim-keyword indicateds best match!
                        elif forcing == 'hist': # forcing == 'hist' or 'future'
                            sim_keyword = 'coupled'
                    else: # df_ensemble_match_diff['Best Simulation'][idx] != forcing
                        # smaller markersize for second best match, alpha corresponding to difference
                        # e.g. if best match is 'present', then we have 'hist' here
                        markersize = 9 - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]

                        # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]

                    # elif df_ensemble_match_diff['Best Simulation'][idx] != forcing:
                    #     # but we have to store somehow if coupled or atmos-only is best match!
                    #     # atmos-only (SST present, SST future)
                    #     if forcing == 'present' or forcing == 'SSTfuture':
                    #         sim_keyword = 'atmos-only'
                    #         # sim-keyword indicateds best match!
                    #     else: # forcing == 'hist' or 'future'
                    #         sim_keyword = 'coupled'
                else: # forcing == 'future' or 'SSTfuture'
                    if sim_keyword == 'coupled': # best match is 'hist', 'future' should be big, 'SST future' small
                        if forcing == 'future':
                            # display big
                            # alpha = 0.9
                            markersize = 11
                        else: # forcing == 'SSTfuture'
                            # smaller
                            markersize = 9  - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]

                            # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]

                    elif sim_keyword == 'atmos-only':
                        if forcing == 'SSTfuture':
                            # display big

                            markersize = 11
                        else: # forcing == 'future'
                            # smaller
                            markersize = 9  - markersize_scale*df_ensemble_match_diff['Difference to second best simulation'][idx]

                            # alpha = 1 - 2*df_ensemble_match_diff['Difference to second best simulation'][idx]


            else: # forcing == 'ERA5'
                # alpha = 0.9
                markersize = 10 # ERA5

            # alpha = 0.9

            #plt.scatter(x_idx, df['mean'][1]*120, marker = marker, s = 130, c=color, alpha = 0.85)

            # plot black errorbars
            # errorbars must be positive (one goes in minus y direction, the other in plus)
            # this line also plots the data!
            # yerr = df['sd'][1] # if I want to show the standard deviation

            ax.errorbar(x_idx, df['mean'][1]*120, yerr=np.array([[abs(df['mean'][1]*120 - df['5.5%'][1]*120), abs(df['mean'][1]*120 - df['94.5%'][1]*120)]]).T,
                        c=color, markeredgecolor = 'k', ecolor=color, markersize = markersize, marker=marker ) # , alpha=0.8
                    # ecolor='k'

            # for better visibility, plot vertical background color that fill the space of one site
            # if idx%2 != 0:
            #     ax.axvspan(idx, idx + 1, alpha=0.02, color='red')

            # individual color for every site
            ax.axvspan(idx, idx + 1, alpha=0.07, color=ls_hex[idx])

            # for legend
            if idx == 0:
                if forcing=='present':
                    my_label = 'atmos-past'
                elif forcing == 'SSTfuture':
                    my_label = 'atmos-future'
                elif forcing == 'hist':
                    my_label = 'coupled-past'
                elif forcing == 'future': # forcing == future
                    my_label = 'coupled-future'
                else: # ERA5
                    my_label = 'ERA5'
                # for legend
                line_list.append(Line2D([0], [0], linestyle = '', color = color, marker = marker, label = my_label))

    # add line at y = 0.0 to emphasize the state zero
    ax.axhline(y=0, xmin=0, xmax=8, color = 'red')

    # append errorbar to legend
    line_list.append(Line2D([0], [0], linestyle = '-', color = 'k', label = '89% percentile interval')) # percentile


    # revision: no legend!
    # revision 2: legend!
    # legend only for last plot!
    if fig_idx == 13: # change..
        ax.legend(handles=line_list, loc='upper left', ncol = 2, bbox_to_anchor= (0, -0.35))

    ax.set_xticks(np.arange(0, 8)) #, rotation = 60

    # for seeing, set same ylimits
    if fig_idx == 11 or fig_idx == 13:
        ax.set_ylim(-0.03, 0.05)

    # # if labels should only be displayed at the bottom of the plot
    # if fig_idx == 5 or fig_idx == 6:
    #     ax.set_xticklabels(ls_site_names)
    # else:
    #     plt.setp(ax.get_xticklabels(), visible=False)

    # else, plot x-labels for all variables
    # set labels for the x axis (site names)
    ax.set_xticklabels(ls_site_names, fontsize=12)

    # shrink axis to make space for xlabels
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

    # set xlim (otherwise there is unused space left and right of the plots)
    ax.set_xlim(0, 8.1)

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-25, ha="left")

    ax.set_ylabel(unit + ' per decade')

    # write a), b)
    if title != None:
        ax.set_title(climxa.alphabet_from_idx(fig_idx) + ') trends in ' + title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(climxa.alphabet_from_idx(fig_idx) + ') trends in ' + variable.replace('_', ' '), fontsize=12, fontweight='bold')


    # save fig
    # fig.savefig(base_path + 'Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)
    # # also save it into figures folder for publication
    # fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/figures/' + variable + '_Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)

    # plt.show()

    return fig

#%% main plot

# first, do it for version with in-situ as reference,
# next, do it for version with ERA5 as reference
# plot all in one plot

# choose if we should compute the tables or read in saved csv files (skill scores)
compute = False

# prepare figure
fig = plt.figure(figsize = (18, 26)) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
gs = fig.add_gridspec(7, 2) # first entry: rows, second entry: columns


# for file_name in ['_', '_ERA5_to_PRIMAVERA_']:
ls_var = ['T','RH','SH', 'TCW', 'total_cloud_cover', 'wind_speed_seeing', 'seeing_osborn']
# ls_var = ['T','RH','SH']
# ls_var = ['wind_speed_seeing', 'seeing_osborn']

max_idx = len(ls_var) - 1

# for fig_idx in range(0,14):

# for idx, var in enumerate(ls_var):

for fig_idx in range(0,14):
    # we plot two subplots per variable

    if fig_idx % 2 == 0:
        # then we are in first column, where we want to plot the skill score classification

        idx = int(fig_idx/2)
        var = ls_var[idx]

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
        ax = fig.add_subplot(gs[int((fig_idx - (fig_idx%2))/2), fig_idx%2])

        ######### do not sort in second approach (where we combine both)

        # draw barplots with skill scores (sorted after PRIMAVERA)
        ax.barh(2* df_class_insitu.index, df_class_insitu['ERA5'], height=0.4,  align='center', color = '#0072b2', label='ERA5 versus in-situ')
        ax.barh(2* df_class_insitu.index + 0.6, df_class_insitu['PRIMAVERA'], height=0.4, align='center', color = '#e69f00', label='PRIMAVERA versus in-situ')

        # add PRIMAVERA to ERA5 (df_class_PRIMAVERA_to_ERA5)
        ax.barh(2* df_class_PRIMAVERA_to_ERA5.index + 1.0, df_class_PRIMAVERA_to_ERA5['PRIMAVERA'], height=0.4,  align='center', color = 'red', label='PRIMAVERA versus ERA5')

        # define from where we take index and site (shouldn't matter)
        df_class_sorted = df_class_insitu

        # set yticks and labels (sites) and invert yaxis so that Mauna Kea is on top
        ax.set_yticks(2* df_class_sorted.index)
        ax.set_yticklabels(df_class_sorted['Site'])
        ax.invert_yaxis()  # labels read top-to-bottom

        # ax.set_title(folder_name)
        # x label
        ax.set_xlabel('skill score', fontsize=12) # , fontsize=10 , # y = 0 is ignored!
        ax.xaxis.set_label_coords(0.5, -0.025)

        ax.set_xlim(0,1)

        # revision: no legend!
        # revision2: legend!
        # set legend only for last subplot
        if var == ls_var[max_idx]: # only true for last index
            ax.legend(loc='upper left', bbox_to_anchor= (0, -0.25))

        # set title
        # ax (gca()) and not plt.gcf()
        # plt.gca().text(0.5, 1.2, variable, horizontalalignment='center', fontsize=16)

        ax.set_title(climxa.alphabet_from_idx(fig_idx) + ') model evaluation of ' + variable, y = 1.1, fontsize=12, fontweight='bold')

        # shrink axis to make space for xlabels
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

        # call function to plot classification axspans (Poor, Mediocre, Good, Excellent)
        climxa.plot_axspan_classification(ax)

    # # save fig
    # if var == ls_var[max_idx]: # only true for last index
    #     # png for powerpoint
    #     # fig.savefig('./Model_evaluation/' + folder_name + '/'  + var + '_PRIMAVERA_to_ERA5_skill_scores.png', bbox_inches = 'tight', pad_inches=0.0, dpi=500)
    #     # pdf for publication
    #     fig.savefig('./publication/figures/'  + 'All_vars' + '_skillScores_and_Trends.pdf', bbox_inches = 'tight', pad_inches=0.0)

    else: # second column: trends

        idx = int((fig_idx - 1)/2)
        var = ls_var[idx]

        # we use the function 'trend_analysis_plot' defined above, and do not take the very similar one from climxa! (only indices differ for legend and seeing ylims)
        fig = trend_analysis_plot(var, gs, fig, fig_idx)

# save figure after loop
fig.savefig('./publication/revision2/figures/'  + 'All_vars' + '_skillScores_and_Trends.pdf', bbox_inches = 'tight', pad_inches=0.0)
# save as png for powerpoint
fig.savefig('./publication/revision2/figures/'  + 'All_vars' + '_skillScores_and_Trends.png', bbox_inches = 'tight', pad_inches=0.0, dpi=600)


# %%
