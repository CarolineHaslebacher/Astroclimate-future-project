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

# read in all relevant csv files
# generate table like this:
#      |                     variable 1                                      | variable 2 | ...
# site | ERA5 classification | PRIMAVERA classification (plus coupled/atmos) | .. | ..    |



ls_var = ['T','RH','SH', 'TCW', 'total_cloud_cover', 'wind_speed_seeing', 'seeing_osborn']
# ls_var = ['T','RH','SH']
# ls_var = ['wind_speed_seeing', 'seeing_osborn']
# ls_var = ['total_cloud_cover']
# ls_var = ['seeing_osborn']

#%%


for idx, var in enumerate(ls_var):

    #### read tables
    # ERA5 to in-situ
    df_ERA5_insitu = pd.read_csv('./Astroclimate_outcome/skill_score_classification/ERA5_skill_score_classification_' + var + '.csv')

    # PRIMAVERA to in-situ
    df_PRIMAVERA_insitu = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_skill_score_classification_' + var + '.csv')

    # PRIMAVERA to ERA5
    df_PRIMAVERA_to_ERA5 = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_skill_score_classification_ERA5_to_PRIMAVERA_' + var + '.csv')

    #####

    # for every variable, initialize table
    skill_score_master_table = pd.DataFrame(columns = ['Site'], data=df_ERA5_insitu['Site'])


    if var == 'TCW':
        var = 'PWV'
    elif var == 'total_cloud_cover':
        var = 'cloud cover'
    elif var == 'wind_speed_seeing':
        var = '200hPa-seeing'
    elif var == 'seeing_osborn':
        var = 'seeing model'


    # append columns to table
    # we do not need the column where pressure levels are defined!
    # skill_score_master_table[var + ' ERA5 classification'] = df_ERA5_insitu['classification']
    skill_score_master_table['E to i'] = df_ERA5_insitu['classification'].replace(' ', '...')


    # PRIMAVERA to in-situ
    if var == 'cloud cover' or var == '200hPa-seeing': # single level var, no need to strip pressure level from 'hist' or 'present'
        # skill_score_master_table[var + ' PRIMAVERA to in-situ data'] = df_PRIMAVERA_insitu[var]
        skill_score_master_table['P to i sim'] = df_PRIMAVERA_insitu[var].replace('nm', '...')
    elif var == 'seeing model':
        # now we have only 5hPa instead of 200hPa
        skill_score_master_table['P to i sim'] = [x[5:] for x in df_PRIMAVERA_insitu[var].replace('nm', '..........')]
    
    else:
        # skill_score_master_table[var + ' PRIMAVERA to in-situ data'] = [x[7:] for x in df_PRIMAVERA_insitu[var]]
        skill_score_master_table['P to i sim'] = [x[7:] for x in df_PRIMAVERA_insitu[var].replace('nm', '..........')]
    
    # skill_score_master_table[var + ' PRIMAVERA to in-situ classification'] = df_PRIMAVERA_insitu['classification']
    skill_score_master_table['P to i'] = df_PRIMAVERA_insitu['classification'].replace(' ', '...')


    # PRIMAVERA to ERA5
    if var == 'cloud cover' or var == '200hPa-seeing':
        # column has no pressure level attribute (single level data!)
        # skill_score_master_table[var + ' PRIMAVERA to ERA5 data'] = df_PRIMAVERA_to_ERA5[var]
        skill_score_master_table['P to E sim'] = df_PRIMAVERA_to_ERA5[var] 
    elif var == 'seeing model':
        # now we have only 5hPa instead of 200hPa
        skill_score_master_table['P to E sim'] = [x[5:] for x in df_PRIMAVERA_to_ERA5[var]]
    else:
        # skill_score_master_table[var + ' PRIMAVERA to ERA5 data'] = [x[7:] for x in df_PRIMAVERA_to_ERA5[var]]
        skill_score_master_table['P to E sim'] = [x[7:] for x in df_PRIMAVERA_to_ERA5[var]]
    # skill_score_master_table[var + ' PRIMAVERA to ERA5 classification'] = df_PRIMAVERA_to_ERA5['classification']
    skill_score_master_table['P to E'] = df_PRIMAVERA_to_ERA5['classification']

    
# save to csv
# can be imported to libreoffice or outlook for example
# skill_score_master_table.to_csv('./Astroclimate_outcome/skill_score_classification/Skill_score_master_table_all_vars.csv')

    # Table is way too big.
    # make it shorter with shorter column headers and save for every variable
    skill_score_master_table.to_csv('./publication/tables/to_appendix/Skill_Score_master_table_'+ var.replace(' ', '_') + '.csv', index = False) #, quoting=0) #, quoting=csv.QUOTE_NONE, quotechar='', escapechar='' )

#%% informal skill_scores table

for idx, var in enumerate(ls_var):

    df_class_PRIMAVERA_to_ERA5 = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_Ensemble_skill_score_for_plotting_' + var + '.csv')
    df_class_insitu = pd.read_csv('./Astroclimate_outcome/skill_score_classification/PRIMAVERA_and_ERA5_to_insitu_skill_score_for_plotting_' + var + '.csv')


    if var == 'TCW':
        var = 'PWV'
    elif var == 'total_cloud_cover':
        var = 'cloud cover'
    elif var == 'wind_speed_seeing':
        var = '200hPa-seeing'
    elif var == 'seeing_osborn':
        var = 'seeing model'

    if idx == 0:
        # initialize table (pandas dataframe) for informal calculations
        informal_skill_score = pd.DataFrame(columns = ['Site'], data=df_class_insitu['Site'])

    # append to table
    informal_skill_score[var + ' E to i'] = df_class_insitu['ERA5']
    informal_skill_score[var + ' P to i'] = df_class_insitu['PRIMAVERA']
    informal_skill_score[var + ' P to E'] = df_class_PRIMAVERA_to_ERA5['PRIMAVERA']


# save as csv
informal_skill_score.to_csv('./publication/tables/Skill_score_informal_all_vars.csv', index = False)

# only osborn seeing
# informal_skill_score.to_csv('./publication/tables/Skill_score_informal_osborn_seeing.csv', index = False)


# %%
# TRENDS
# use climxa.round_significant(number_to_round, significant_digits) to round numbers

# initialize table
# trends_master_table = pd.DataFrame(columns = ['Site'], data=['Mauna Kea', 'Paranal', 'La Silla', 'Cerro Tololo', 'La Palma', 'Siding Spring', 'Sutherland', 'SPM'])
informal_trends_table = pd.DataFrame(columns = ['Site'], data=['Mauna Kea', 'Paranal', 'La Silla', 'Cerro Tololo', 'La Palma', 'Siding Spring', 'Sutherland', 'SPM'])
informal_trends_and_errorbars = pd.DataFrame(columns = ['Site'], data=['Mauna Kea', 'Paranal', 'La Silla', 'Cerro Tololo', 'La Palma', 'Siding Spring', 'Sutherland', 'SPM'])
ls_informal_errorbars = []

for idx, variable in enumerate(ls_var):

    # initialize master table for every variable
    trends_master_table = pd.DataFrame(columns = ['Site'], data=['Mauna Kea', 'Paranal', 'La Silla', 'Cerro Tololo', 'La Palma', 'Siding Spring', 'Sutherland', 'SPM'])

    # custom parameters for individual variables
    if variable == 'wind_speed_seeing':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/wind_speed/"
        attrib = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
        unit = 'arcsec'
        title = '200hPa-wind-speed seeing'

    elif variable == 'seeing_osborn':
        base_path =  "./Model_evaluation/seeing_nc/future_trends/osborn/"
        attrib = ['_5', '_5', '_5', '_5','_5', '_5', '_5', '_5']
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
        attrib_ERA5, attrib = climxa.Prlevel_attributes_T_RH_SH()

    ls_site_names = []
    line_list = []


    ls_forcing = ['ERA5', 'hist', 'present', 'future', 'SSTfuture'] # 'future', 'SSTfuture', and eventually 'ERA5'
    for forcing in ls_forcing:

        # initialize lists for every forcing
        ls_trends = []
        ls_errorbar = []
        ls_error_below = []
        ls_error_above = []

        for idx in range(0, 8):
            
            site_name_folder = d_site_lonlat_data['site_name_folder'][idx]

            if idx == 0: # mauna kea
                site_name_folder = 'Mauna_Kea'
            if idx == 1:
                site_name_folder = 'Cerro_Paranal'

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

            # change string of forcing
            if forcing == 'present':
                my_forcing_label = 'SST present'
            elif forcing == 'SSTfuture':
                my_forcing_label = 'SST future'
            else:
                my_forcing_label = forcing

            # get slope and errorbars
            my_slope = df['mean'][1]*120
            my_error_below = df['5.5%'][1]*120# 89 percentile interval
            my_error_above = df['94.5%'][1]*120

            # ax.errorbar(x_idx, df['mean'][1]*120, yerr=np.array([[abs(df['mean'][1]*120 - df['5.5%'][1]*120), abs(df['mean'][1]*120 - df['94.5%'][1]*120)]]).T, 
            #             c=color, markeredgecolor = 'k', ecolor=color, markersize = markersize, marker=marker ) # , alpha=0.8
            #         # ecolor='k'

            # fill list (round numbers!)
            
            # for FORMAL LATEX:
            ls_trends.append(r'\makecell[l]{' +  str(climxa.round_significant(my_slope, 2)) + r'\\ (' + str(climxa.round_significant(my_error_below, 2)) + r'; ' + str( climxa.round_significant(my_error_above, 2)) + ')}' )
            
            # for informal, UNCOMMENT:
            # ls_trends.append(climxa.round_significant(my_slope, 2))

            # ls_errorbar.append(( climxa.round_significant(my_error_below, 2),  climxa.round_significant(my_error_above, 2)))
            ls_error_below.append(abs(df['mean'][1]*120 - df['5.5%'][1]*120))
            ls_error_above.append(abs(df['mean'][1]*120 - df['94.5%'][1]*120))

        # append to table after every iteration of a forcing
        trends_master_table[my_forcing_label] = ls_trends # variable + ' ' + forcing + ' trend [' + unit + ' per decade]'
        # trends_master_table[forcing + r' 89\%-interval'] = ls_errorbar # variable + ' ' + forcing + ' 89-percentile interval'

        # generate one table without 89% interval (for my analysis!)
        # use table to calculate averages
        informal_trends_table[variable + ' ' + forcing + ' trend [' + unit + ' per decade]'] = ls_trends

        # generate another table with errors
        informal_trends_and_errorbars[variable + ' ' + forcing + ' trend [' + unit + ' per decade]'] = ls_trends
        informal_trends_and_errorbars[variable + ' ' + forcing + ' error below [' + unit + ' per decade]'] = ls_error_below
        informal_trends_and_errorbars[variable + ' ' + forcing + ' error above [' + unit + ' per decade]'] = ls_error_above

        # list only for headers
        ls_informal_errorbars.append([variable + ' ' + forcing + ' error below [' + unit + ' per decade]'])
        ls_informal_errorbars.append([variable + ' ' + forcing + ' error above [' + unit + ' per decade]'])
        

# save to .csv
# trends_master_table.to_csv('./Astroclimate_outcome/Trends_master_table_all_forcings_all_vars.csv')

    # Table is waaay too long!
    # save every variable in one file..!
    # and shorten titles!
    # escapechar because we have a comma!!
    # 
    # 
    trends_master_table.to_csv('./publication/tables/to_appendix/Trends_master_table_all_forcings_'+ variable + '.csv', index = False) #, quoting=0) #, quoting=csv.QUOTE_NONE, quotechar='', escapechar='' )


# %% calculate mean trends with informal table 'informal_trends_table'
# also implement gaussian error propagation right here

# EDIT: calculating a mean makes no sense for these eight sites. It does not add value to the paper.
# therefore, give a range (min to max) for every variable. With this, we also see the spread clearly.

# df_average = pd.DataFrame(columns=informal_trends_table.columns)
# df_gaussian_error = pd.DataFrame(columns=informal_trends_table.columns)

# # this only considers columns in 'informal_trends_table', where no errors are included
# for col in informal_trends_table.columns:
#     if col == 'Site':
#         df_average[col] = ['Average trend']
#         continue

#     df_average[col] = np.mean(informal_trends_table[col])
#     # df_average.append(np.mean(informal_trends_table[col]))

# # this considers only errors and calculates the gaussian error propagation of one column
# for col in ls_informal_errorbars:

#     ls_squares = [x**2 for x in informal_trends_and_errorbars[col]]

#     df_gaussian_error[col] = np.sum(ls_squares) # sum of squares.....

#     # yerr_avg = 0.5 * np.sqrt(yerr_0**2 + yerr_1**2)

# # trends_master_table.append(df_average)

# informal_trends_average = pd.concat([informal_trends_and_errorbars, df_average])

# # save to .csv
# informal_trends_average.to_csv('./publication/Trends_master_table_all_forcings_all_vars_informal.csv')

df_min = pd.DataFrame(columns=informal_trends_table.columns) 
df_max = pd.DataFrame(columns=informal_trends_table.columns)
# df_2050 = pd.DataFrame(columns=informal_trends_table.columns) # until 2050 (= 3 decades from now)

# this only considers columns in 'informal_trends_table', where no errors are included
for col in informal_trends_table.columns:
    if col == 'Site':
        df_min[col] = ['Minimum']
        df_max[col] = ['Maximum']
        # df_2050[col] = ['Until 2050']
        continue

    df_min[col] = np.min(informal_trends_table[col]) * 3
    df_max[col] = np.max(informal_trends_table[col]) * 3
    # df_2050_min[col] = np.max(informal_trends_table[col])
    # df_average.append(np.mean(informal_trends_table[col])

# concatenate (append rows)
informal_trends_min_max = pd.concat([informal_trends_table, df_min, df_max]) # , df_2050

# save to .csv
# informal_trends_min_max.to_csv('./publication/tables/Trends_master_table_all_forcings_all_vars_informal_min_max.csv')

# only osborn seeing
informal_trends_min_max.to_csv('./publication/tables/Trends_master_table_all_forcings_Seeing_osborn_corrected_informal_min_max.csv')



# %%
