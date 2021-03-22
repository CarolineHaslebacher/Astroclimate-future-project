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

from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

from itertools import cycle
from functools import reduce
from scipy import stats

import csv

import seaborn as sns
sns.set()
from matplotlib import cycler

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



#%% MAIN

# choose variable
# variable = 'wind_speed_seeing'
# variable = 'seeing_osborn'
# variable = 'SH'
# variable = 'RH'
# variable = 'T'
# variable = 'total_cloud_cover'

ls_var = ['T','RH','SH', 'TCW', 'wind_speed_seeing', 'seeing_osborn', 'total_cloud_cover']
# ls_var = ['T','RH','SH']

# masterfigure (7 different variables)
fig = plt.figure(figsize = (22, 28)) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
gs = fig.add_gridspec(7, 2) # first entry: rows, second entry: columns


for idx, variable in enumerate(ls_var):

    fig = climxa.trend_analysis_plot(variable, gs, fig, idx)
    # fig = mean_trend_analysis_plot(variable, gs, fig, idx) # defined below (commented)


# save fig
fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/figures/All_variables_Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)

# save table? --> YES!!!!


#%% define function for marker

# def trend_marker(forcing):

#     mar_collection = {'hist': 'p',
#                                 'present': 'h',
#                                 'future': 's',
#                                 'SSTfuture': 'd'}
#     trend_marker = mar_collection[forcing]
#     return trend_marker


# maybe I need that
                # if forcing=='present':
                #     my_label = 'SST present'
                # elif forcing == 'SSTfuture':
                #     my_label = 'SST future'
                # else:
                #     my_label = forcing


        
#%% second version: take average of trends

# def plot_avg(df_ls, timeframe, x_idx, color,ax):
#     #### take average
#     avg = 0.5 * (df_ls[0]['mean'][1]*120 + df_ls[1]['mean'][1]*120)
#     # print(avg)
#     # Gaussian error propagation (with 89 percentile interval --> does that make sense?)
#     # I take the absolute value between the value and the percentile, and take this as the upper/lower error
#     # then I apply gaussian error prop (0.5 * sqrt(s1**2 + s2**2))
#     yerr_0 = np.array([[abs(df_ls[0]['mean'][1]*120 - df_ls[0]['5.5%'][1]*120), abs(df_ls[0]['mean'][1]*120 - df_ls[0]['94.5%'][1]*120)]]).T
#     yerr_1 = np.array([[abs(df_ls[1]['mean'][1]*120 - df_ls[1]['5.5%'][1]*120), abs(df_ls[1]['mean'][1]*120 - df_ls[1]['94.5%'][1]*120)]]).T

#     yerr_avg = 0.5 * np.sqrt(yerr_0**2 + yerr_1**2)

#     # get marker for individual forcings (and ERA5)
#     marker = 'o' # climxa.trend_marker(timeframe)

#     # plot
#     # ax.plot(x_idx, avg, c=color, markersize = 8, marker=marker, alpha=0.85)
#     ax.errorbar(x_idx, avg, yerr=yerr_avg, 
#                         c=color, ecolor='k', markersize = 10, marker=marker, alpha=0.85 )


#     return


# def mean_trend_analysis_plot(variable, gs, fig, fig_idx):
#     # custom parameters for individual variables
#     if variable == 'wind_speed_seeing':
#         base_path =  "./Model_evaluation/seeing_nc/future_trends/wind_speed/"
#         attrib = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
#         attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level', '_single_level']
#         unit = 'arcsec'
#         title = '200hPa-wind-speed seeing'

#     elif variable == 'seeing_osborn':
#         base_path =  "./Model_evaluation/seeing_nc/future_trends/osborn/"
#         attrib = ['_200', '_200', '_200', '_200','_200', '_200', '_200', '_200']
#         attrib_ERA5 = ['_800', '_900', '_825', '_825','_975', '_950', '_850', '_850']
#         unit = 'arcsec'
#         title = 'seeing model'

#     elif variable == 'SH':
#         base_path =  "./Model_evaluation/SH/future_trends/"
#         # attrib = ['_700', '_850', '_850', '_850','_850', '_925', '_850', '_700']
#         # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
#         # attrib_ERA5 = ['_600', '_775', '_775', '_775','_800', '_950', '_850', '_750']
#         unit = '(g/kg)'
#         title = 'Specific Humidity'

#     elif variable == 'RH':
#         base_path =  "./Model_evaluation/RH/future_trends/"
#         # attrib = ['_600', '_700', '_850', '_850','_850', '_1000', '_850', '_700']
#         # attrib = ['_600', '_850', '_850', '_850','_850', '_925', '_850', '_700']
#         # attrib_ERA5 = ['_600', '_750', '_775', '_750','_850', '_950', '_875', '_750']
#         unit = '%'
#         title = 'Relative Humidity'

#     elif variable == 'T':
#         base_path =  "./Model_evaluation/T/future_trends/"
#         # attrib = ['_single_level', '_single_level', '_850', '_single_level','_850', '_925', '_single_level', '_700']
#         # new attributes due to comparison of PRIMAVERA to ERA5 for best match
#         # attrib = ['_600', '_700', '_850', '_850','_700', '_single_level', '_single_level', '_700']
#         # attrib_ERA5 = ['_600', '_750', '_750', '_775','_750', '_900', '_875', '_750']
#         unit = '°C'
#         title = 'Temperature'

#     elif variable == 'total_cloud_cover':
#         base_path =  "./Model_evaluation/total_cloud_cover/future_trends/"
#         attrib = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
#         attrib_ERA5 = ['_single_level', '_single_level', '_single_level', '_single_level','_single_level', '_single_level', '_single_level', '_single_level']
#         unit = '%'
#         title = 'Total cloud cover'

#     elif variable == 'TCW':
#         base_path = "./Model_evaluation/TCW/future_trends/"
#         attrib = ['_600', '_700', '_700', '_850','_700', '_925', '_850', '_700']
#         attrib_ERA5 = ['_600', '_750', '_775', '_775','_775', '_900', '_825', '_750']
#         unit = 'mmH2O'
#         title = 'Precipitable water vapor'

#     # for T, RH and SH, we have fixed Plevs
#     if variable == 'T' or variable == 'RH' or variable == 'SH':
#         attrib_ERA5, attrib = Prlevel_attributes_T_RH_SH()

#     ls_site_names = []
#     line_list = []

#     ls_hex = [to_hex(plt.cm.terrain(i / 8)) for i in range(8)] 

#     # fig, ax = plt.subplots(figsize=(8,4))
#     ax = fig.add_subplot(gs[int((fig_idx - (fig_idx%2))/2), fig_idx%2])

#     for idx in range(0, 8):
        
#         site_name_folder = d_site_lonlat_data['site_name_folder'][idx]
#         # "Mauna_Kea", "Cerro_Paranal", "La_Silla", "Cerro_Tololo", "La_Palma", "Siding_Spring", "Sutherland", "SPM"
#         # change site name if needed
#         if idx == 0: # mauna kea
#             site_name_folder = 'Mauna_Kea'
#         if idx == 1:
#             site_name_folder = 'Cerro_Paranal'
        
#         ls_site_names.append(site_name_folder.replace('_', ' '))



#         ls_forcing = ['hist', 'present', 'future', 'SSTfuture', 'ERA5'] # 'future', 'SSTfuture', and eventually 'ERA5'
        
#         df_ls = []

#         #### hist and present

#         for df_idx, forcing in enumerate(['hist', 'present']):
#             # read PRIMAVERA csv
#             # append to dataframe list
#             # df[0] = hist
#             # df[1] = present
#             df_ls.append(pd.read_csv(base_path + site_name_folder + '_' + forcing + attrib[idx] + variable + '_' + '_PRIMAVERA_Projections_Bayesian_model_map2stan.csv'))

#             # rename 
#             df_ls[df_idx] = df_ls[df_idx].rename(columns={'Unnamed: 0': 'parameter'})


#         # define color and x_idx
#         color = '#e69f00'#  
#         x_idx = idx + 0.5
#         timeframe = 'hist'

#         ####
#         plot_avg(df_ls, timeframe, x_idx, color, ax)
#         # for legend
#         if idx == 0:
#             marker = 'o' # climxa.trend_marker(timeframe)
#             line_list.append(Line2D([0], [0], linestyle = '', color = color, marker = marker, label = timeframe))


#         df_ls = []
#         #### future and SSTfuture

#         for df_idx, forcing in enumerate(['future', 'SSTfuture']):
#             # read PRIMAVERA csv
#             # append to dataframe list
#             # df[0] = hist
#             # df[1] = present
#             df_ls.append(pd.read_csv(base_path + site_name_folder + '_' + forcing + attrib[idx] + variable + '_' + '_PRIMAVERA_Projections_Bayesian_model_map2stan.csv'))

#             # rename 
#             df_ls[df_idx] = df_ls[df_idx].rename(columns={'Unnamed: 0': 'parameter'})


#         # define color and x_idx
#         color = '#009e73' 
#         x_idx = idx + 0.8
#         timeframe = 'future'

#         #### plot
#         plot_avg(df_ls, timeframe, x_idx, color, ax)
#         # for legend
#         if idx == 0:
#             marker = 'o' # climxa.trend_marker(timeframe)
#             line_list.append(Line2D([0], [0], linestyle = '', color = color, marker = marker, label = timeframe))


#         ########## ERA5
#         forcing = 'ERA5'

#         if variable == 'wind_speed_seeing' or variable == 'seeing_osborn':
#             ERA5_path = "./Model_evaluation/seeing_nc/ERA5_trends/"
#         else:
#             ERA5_path = "./Model_evaluation/" + variable + "/ERA5_trends/"

#         df = pd.read_csv(ERA5_path + site_name_folder + attrib_ERA5[idx] + variable + '_' + '_ERA5_Projections_Bayesian_model_map2stan.csv')

#         color = '#0072b2' # '#56b4e9'
#         x_idx = idx + 0.1    
        
#         marker = 'o' # climxa.trend_marker('ERA5')
#         # plot
#         # ax.plot(x_idx, df['mean'][1]*120, c=color, markersize = 8, marker=marker, alpha=0.85)
#         ax.errorbar(x_idx, df['mean'][1]*120, yerr=np.array([[abs(df['mean'][1]*120 - df['5.5%'][1]*120), abs(df['mean'][1]*120 - df['94.5%'][1]*120)]]).T, 
#                         c=color, ecolor='k', markersize = 10, marker=marker, alpha=0.85 )
#         # for legend
#         if idx == 0:
#             line_list.append(Line2D([0], [0], linestyle = '', color = color, marker = marker, label = 'ERA5'))
    
#         ##### ERA5 done

    
#         # individual color for every site
#         ax.axvspan(idx, idx + 1, alpha=0.3, color=ls_hex[idx])
        


#     # add line at y = 0.0 to emphasize the state zero
#     ax.axhline(y=0, xmin=0, xmax=8, color = 'red')

#     # append errorbar to legend
#     line_list.append(Line2D([0], [0], linestyle = '-', color = 'k', label = 'Gaussian error propagation of \n89% percentile interval'))

#     # legend only for last plot!
#     if fig_idx == 6:
#         ax.legend(handles=line_list, loc='lower left', bbox_to_anchor= (1.2, 0))
    
#     ax.set_xticks(np.arange(0, 8)) #, rotation = 60

#     # for seeing, set same ylimits
#     if fig_idx == 4 or fig_idx == 5:
#         ax.set_ylim(-0.03, 0.05)

#     # set labels for the x axis (site names)
#     ax.set_xticklabels(ls_site_names)

#     # shrink height to make space for xlabels
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

#     # set xlim (otherwise there is unused space left and right of the plots)
#     ax.set_xlim(0, 8)

#     # # if labels should only be displayed at the bottom of the plot
#     # if fig_idx == 5 or fig_idx == 6:
#     #     ax.set_xticklabels(ls_site_names)
#     # else:
#     #     plt.setp(ax.get_xticklabels(), visible=False)
        
#     plt.setp( ax.xaxis.get_majorticklabels(), rotation=-30, ha="left" )

#     ax.set_ylabel(unit + ' per decade')

#     # write a), b) 
#     if title != None:
#         ax.set_title(climxa.alphabet_from_idx(fig_idx) + ') Trends of ' + title)
#     else:
#         ax.set_title(climxa.alphabet_from_idx(fig_idx) + ') Trends of ' + variable.replace('_', ' '))


#     # save fig
#     # fig.savefig(base_path + 'Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)
#     # # also save it into figures folder for publication
#     # fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/figures/' + variable + '_Trend_analysis_plot_all_sites.pdf', bbox_inches = 'tight', pad_inches=0.0)

#     # plt.show()

#     return fig





#%%
