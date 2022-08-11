#%%
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

#%%
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file
import Astroclimate_function_pool
import importlib
importlib.reload(Astroclimate_function_pool)

from Astroclimate_function_pool import netcdf_to_df
from Astroclimate_function_pool import  mes_prep
from Astroclimate_function_pool import  merge_df 
from Astroclimate_function_pool import  merge_df_long
from Astroclimate_function_pool import  df_prep #(df, parameter, colname)
from Astroclimate_function_pool import  plot_cycle #(cycle_name, cycle_string,  CFHT_parameter, filename, *args)
from Astroclimate_function_pool import  plot_timeseries_merged
from Astroclimate_function_pool import plot_timeseries_long
from Astroclimate_function_pool import plot_timeseries_movav
from Astroclimate_function_pool import correlation_plot
from Astroclimate_function_pool import corr_plots_hourly_monthly_yearly

#%% define SCIDAR dates
SCIDAR_dates = []

for year in [2016, 2017, 2018]:
    if year == 2016:
        months_SCIDAR = [4,7,10,11,12]
    elif year == 2017:
        months_SCIDAR = [3,4,5,6,7,8,11,12]
    elif year == 2018:
        months_SCIDAR = [1]
    for month in months_SCIDAR:
        # 2016
        if (year == 2016) & (month == 4):
            days_SCIDAR = [26,27,28,29]
        elif (year == 2016) & (month == 7):
            days_SCIDAR = [22,23,24,25,26]
        elif (year == 2016) & (month == 10):
            days_SCIDAR = [30,31]
        elif (year == 2016) & (month == 11):
            days_SCIDAR = [1,2]
        elif (year == 2016) & (month == 12):
            days_SCIDAR = [10,11,12]
        #2017
        elif (year == 2017) & (month == 3):
            days_SCIDAR = [7,8,9]
        elif (year == 2017) & (month == 4):
            days_SCIDAR = [12,13,14,15,16,17,18]
        elif (year == 2017) & (month == 5):
            days_SCIDAR = [5,6,7,8,9]
        elif (year == 2017) & (month == 6):
            days_SCIDAR = [8,9,10]
        elif (year == 2017) & (month == 7):
            days_SCIDAR = [3,4,5,6,7,8,9]
        elif (year == 2017) & (month == 8):
            days_SCIDAR = [3,4,5,6,7,8]
        elif (year == 2017) & (month == 11):
            days_SCIDAR = [4,5,6,7,8,9, 18,19,20, 29,30]
        elif (year == 2017) & (month == 12):
            days_SCIDAR = [1,2, 5,6, 8,9,10,11,12,13,14,15,16,17,18]
        # 2018
        elif (year == 2018) & (month == 1):
            days_SCIDAR = [13,14,15, 18,19,20,21,22,23,24]

        for day in days_SCIDAR:
            SCIDAR_dates.append('{}-{:02}-{:02}'.format(year, month, day))
        del(days_SCIDAR)

df = pd.DataFrame(SCIDAR_dates)
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
df.to_csv('./sites/Paranal/Data/in-situ/SCIDAR_Osborn/SCIDAR_dates.csv', index=None, header = None)

#%% define SCIDAR dates for ERA5 data (day plus 1 in UT)
SCIDAR_dates_ERA5 = []

for year in [2016, 2017, 2018]:
    if year == 2016:
        months_SCIDAR = [4,7,10,11,12]
    elif year == 2017:
        months_SCIDAR = [3,4,5,6,7,8,11,12]
    elif year == 2018:
        months_SCIDAR = [1]
    for month in months_SCIDAR:
        # 2016
        if (year == 2016) & (month == 4):
            days_SCIDAR = [26,27,28,29]
        elif (year == 2016) & (month == 7):
            days_SCIDAR = [22,23,24,25,26]
        elif (year == 2016) & (month == 10):
            days_SCIDAR = [30,31]
        elif (year == 2016) & (month == 11):
            days_SCIDAR = [1,2]
        elif (year == 2016) & (month == 12):
            days_SCIDAR = [10,11,12]
        #2017
        elif (year == 2017) & (month == 3):
            days_SCIDAR = [7,8,9]
        elif (year == 2017) & (month == 4):
            days_SCIDAR = [12,13,14,15,16,17,18]
        elif (year == 2017) & (month == 5):
            days_SCIDAR = [5,6,7,8,9]
        elif (year == 2017) & (month == 6):
            days_SCIDAR = [8,9,10]
        elif (year == 2017) & (month == 7):
            days_SCIDAR = [3,4,5,6,7,8,9]
        elif (year == 2017) & (month == 8):
            days_SCIDAR = [3,4,5,6,7,8]
        elif (year == 2017) & (month == 11):
            days_SCIDAR = [4,5,6,7,8,9, 18,19,20, 29,30]
        elif (year == 2017) & (month == 12):
            days_SCIDAR = [1,2, 5,6, 8,9,10,11,12,13,14,15,16,17,18]
        # 2018
        elif (year == 2018) & (month == 1):
            days_SCIDAR = [13,14,15, 18,19,20,21,22,23,24]

        for day in days_SCIDAR:
            SCIDAR_dates_ERA5.append('{}'.format(dt.date(year, month, day) + dt.timedelta(days = 1))) # day plus one for balancing 'wrong' filenames (wasn't wrong for Paranal)
        del(days_SCIDAR)

#%%
# read in SCIDAR data
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

# header
# from readme file: YYYY-MM-DDTHH:MM:SS.ss,r0,seeing,coherenceTime,isoplanaticAngle,scintillationIndex,alt_0,cn2_0,ws_0,wd_0,al_t1,cn2_1,ws_1,wd_1….alt_n,cn2_n,ws_n,wd_n
Scidar_names = ['time', 'r0', 'seeing', 'coherenceTime', 'isoplanaticAngle', 'scintillationIndex']

for i in range(0,100):
    Scidar_names.extend([f'alt_{i}', f'cn2_{i}', f'ws_{i}', f'wd_{i}'])

scid_file = []
for scid_date in SCIDAR_dates:
    if '2016' in scid_date:
        scid_year = '2016' 
        
    elif '2017' in scid_date:
        scid_year = '2017'
        
    elif '2018' in scid_date:
        scid_year = '2018'
    #scid_year = scid_date[:3]
    scid_file.append(pd.read_csv('./sites/Paranal/Data/in-situ/SCIDAR_Osborn/' + str(scid_year) + '/' + str.replace(str(scid_date), '-', '') + '.txt', header=None, names = Scidar_names, index_col=0, parse_dates=True))

df_scidar = pd.concat(scid_file)

#%% Cn2 Profile
############################################## CN2 PROFILE #################################################################################
# load ERA5 data in
#ds_era5 = xr.open_dataset('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_all_SCIDAR_dates.nc')

# 800hPa
ds_seeing_ERA5 = xr.open_dataset('./sites/Paranal/Data/Era_5/seeing/ERA5_Paranal_greenpoint_2_seeing_J_1979to2020.nc')
ds_Cn2_profile_ERA5 = xr.open_dataset('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_all_SCIDAR_dates_ERA5_24hours.nc')

# # 1000hPa
# ds_seeing_ERA5 = xr.open_dataset('./sites/Paranal/Data/Era_5/seeing/ERA5_Paranal_greenpoint_to1000hPa_seeing_J_1979to2020.nc')
# ds_Cn2_profile_ERA5 = xr.open_dataset('./sites/Paranal/Data/Era_5/seeing/Cn2_Profile/ERA5_Cn2_Profile_all_SCIDAR_dates_ERA5_24hours_to1000hPa.nc')

#%% from df_scidar to xarray (for plotting Cn2 Profile)
ds_temp = []
for i in range(0,100):
    cn2_str = 'cn2_' + str(i)
    alt_str = 'alt_' + str(i)
    ds_temp.append(xr.DataArray(df_scidar[[cn2_str]],
    dims=('time', 'altitude'), 
    coords={'time': df_scidar.index ,'altitude': [df_scidar[alt_str][0]]}, name='Cn2'))
# concatenate on dimension "altitude"     
ds_cn2_scidar = xr.concat(ds_temp, 'altitude')
# IMPORTANT NOTE: The days in the dataset are in Universal Time! They match up with the ERA5 dataset perfectly!

# add unit to Cn2
ds_cn2_scidar.attrs['units'] = 'm$^{-2/3}$'

#%%
# resample to 5min bins(look at resample function, maybe there is a way to resample +/-2.5min)
ds_cn2_scidar_resampled = ds_cn2_scidar.resample(time='5Min', skipna = False, restore_coord_dims = True, keep_attrs=True, base = 57,loffset= dt.timedelta(minutes = 3), label = 'left').mean()

#%%
# only take full hour values
for hour in range(0,10):

    time_string = '2016-04-27T' + str(hour).zfill(2) + ':00:00'
    ds_cn2_scidar_resampled.loc[time_string]

# save as .nc

# I can make the same plot as James if I plot all nans as 10**(-18) !!!!!


# check with James Osborn the first altitude: 0m (should it not be 62.5m? (from 125/2)) (first bin-size = 125m only? (important for unintegrated values))
# check integral of scidar turbulence! --> better
# where is '2016-04-26'? --> changed to 2016-04-27 UT
# check how nan's are handled

# ds_cn2_scidar.loc['2016-04-27', :].plot()

# access with: ds_12.loc['2016-04-27 01:05:05', 250]

#.dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

#%% PLOT DATA
# use xarray plot
# replace date by scid_date

def plot_cn2_profile(filename, scid_date, pr_level_max):
    #g = 9.80665 #m/s^2, from Era5 documentation
    
    #ds_pl = ds_Cn2_profile_ERA5.Cn2_profile.loc[:800,scid_date + 'T00:00:00':scid_date + 'T09:00:00'] # SCIDAR data also goes from 00:00 to 09:00 UT
    

    ds_pl = ds_Cn2_profile_ERA5.rename({'Cn2_profile': 'Cn2'}).Cn2.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00']
    
    ds_pl = ds_pl.assign_coords({'altitude': (ds_Cn2_profile_ERA5.geopotential_height.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time') - 2600)/1000})
    ds_median_Cn2 = ds_pl.median(dim = 'time')
    ds_median_Cn2 = ds_median_Cn2.assign_coords({'altitude': (ds_Cn2_profile_ERA5.geopotential_height.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time') - 2600)/1000}) # add 2600m so that 0m defines the observatory level

    Cn2_value = ds_seeing_ERA5.J.loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00'].mean(dim = 'time').values

    fig, a = plt.subplots(2,2,figsize= (8,4), gridspec_kw={'width_ratios': [4, 1]}, sharey=True, sharex='col')
    fig.suptitle('Night Profile on {} (UT)'.format(scid_date))
    fig.subplots_adjust(bottom=0.2, top=0.85, hspace = 0.4)

    # subplot 00
    ds_pl.attrs['units'] = 'm$^{-2/3}$'
    #ds_pl.rename({'Cn2_profile': 'Cn2'})
    ds_pl.plot(ax = a[0][0] ,y = 'altitude', norm=colors.LogNorm(vmin=10**(-18), vmax=10**(-14)), cmap = 'magma')
    #a[0][0].invert_yaxis(), fontsize = 9
    a[0][0].set_ylabel('altitude [km]')
    a[0][0].set_title(r'ERA5: C$_{n}^2$ night profile', fontsize = 'small')
    a[0][0].set_yticks([0, 5, 10, 15, 20, 25]) 
    #a[0][0].set_yticks([-2.6, 0, 5, 10, 15, 20, 25]) # goes until -2.6km
    a[0][0].set_xlabel('hours [UT]')

    # subplot 01
    ds_median_Cn2.plot.line( '-ko',y = 'altitude',ax = a[0][1], label = 'Median Profile ERA5', markersize = 1)
    #a[0][1].invert_yaxis()
    a[0][1].set_title(r'ERA5: C$_{n}^2$ integral = ' + r'${0}$ m$^{{1/3}}$'.format(Astroclimate_function_pool.as_si(Cn2_value, 2)), fontsize = 8)
    a[0][1].set_xlabel(r'C$_{n}^2$ [m$^{-2/3}$]')
    a[0][1].set_ylabel('altitude [km]')
    a[0][1].set_xscale('log')
    a[0][1].set_xlim(10**(-18), 10**(-14))
    #a[0][1].set_ylim(-10, 25) # set the y-axis limits
    a[0][1].legend(fontsize = 'x-small', loc = 'upper left')
    
    # subplot 10
    # ds_cn2_unintegrated = ds_cn2_scidar.loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00', :].fillna(10**(-20))/250 # divide by bin-size (because data is already integrated)
    # there are nans that need to be filled with 10*3(-20) and also values equal to 0 which need to be replaced with 10**(-20)
    ds_cn2_unintegrated_filled = ds_cn2_scidar_resampled.where(ds_cn2_scidar_resampled != 0.00000000e+00, other = 10**(-20)).fillna(10**(-20)).loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00', :]/250
    ds_cn2_unintegrated_filled = ds_cn2_unintegrated_filled.assign_coords({'altitude': ds_cn2_unintegrated_filled.altitude/1000})
    ds_cn2_unintegrated_filled.attrs['units'] = 'm$^{-2/3}$'
    ds_cn2_unintegrated_filled.plot(ax = a[1][0], y = 'altitude', 
                        norm=colors.LogNorm(vmin=10**(-18), vmax=10**(-14)), cmap = 'magma') #(ax = a[1][0])  
    a[1][0].set_ylabel('altitude [km]')
    a[1][0].set_title(r'Stereo-SCIDAR: C$_{n}^2$ night profile', fontsize = 'small')
    a[1][0].set_yticks([0, 5, 10, 15, 20, 25])
    a[1][0].xaxis_date()
    #a[1][0].set_xticks([0,1,2,3,4,5,6,7,8,9])
    a[1][0].set_xticklabels(['{}:00'.format(str(tim.hour).zfill(2)) for tim in pd.to_datetime(ds_pl.time.values)], ha='right')
    a[1][0].set_xlabel('hours [UT]')
    
    # ds_mean_Cn2_scid = ds_pl_scid.mean(dim = 'time')

    # for the median, data does not need to be filtered
    ds_median_Cn2_scid = ds_cn2_scidar.loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time')/250
    ds_median_Cn2_scid = ds_median_Cn2_scid.assign_coords({'altitude': ds_median_Cn2_scid.altitude/1000})
    # subplot 11
    ds_median_Cn2_scid.plot.line( '-ko',y = 'altitude',ax = a[1][1], label = 'Median Profile SCIDAR', markersize = 1)
    # a[1][1].invert_yaxis()
    a[1][1].set_title('SCIDAR: C$_{{n}}^2$ integral = ${0}$ m$^{{1/3}}$'.format(Astroclimate_function_pool.as_si(sum(ds_median_Cn2_scid*250).values, 2)), fontsize = 8)
    a[1][1].set_xlabel(r'C$_{n}^2$ [m$^{-2/3}$]')
    a[1][1].set_ylabel('altitude [km]')
    a[1][1].set_xscale('log')
    a[1][1].legend(fontsize = 'x-small', loc = 'upper left')
    a[1][1].set_xlim(10**(-18), 10**(-14))
    a[1][1].set_xticks([10**(-18),10**(-17), 10**(-16),10**(-15), 10**(-14)])
    a[1][1].set_xticklabels([10**(-18),10**(-17), 10**(-16),10**(-15), 10**(-14)], rotation=50)
    #plt.tight_layout()

    plt.savefig(filename)
    plt.show()
    plt.close()


#%% plot CN2 profile overlay

scid_date = SCIDAR_dates_ERA5[1]

ds_pl = ds_Cn2_profile_ERA5.rename({'Cn2_profile': 'Cn2'}).Cn2.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00']

ds_pl = ds_pl.assign_coords({'altitude': (ds_Cn2_profile_ERA5.geopotential_height.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time') - 2600)/1000})
ds_median_Cn2 = ds_pl.median(dim = 'time')
ds_median_Cn2 = ds_median_Cn2.assign_coords({'altitude': (ds_Cn2_profile_ERA5.geopotential_height.loc[:pr_level_max,scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time') - 2600)/1000}) # add 2600m so that 0m defines the observatory level

Cn2_value = ds_seeing_ERA5.J.loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00'].mean(dim = 'time').values

fig, a = plt.subplots(1,1)
fig.subplots_adjust(bottom=0.2, top=0.85, hspace = 0.4)

# subplot 01
ds_median_Cn2.plot.line( '-ro',y = 'altitude',ax = a, label = 'Median Profile ERA5', markersize = 1)
#a[0][1].invert_yaxis()
a.set_title('{} Cn2 night profile'.format(scid_date))
a.set_xlabel(r'C$_{n}^2$ [m$^{-2/3}$]')
a.set_ylabel('altitude [km]')
a.set_xscale('log')
a.set_xlim(10**(-18), 10**(-14))
#a.set_ylim(-10, 25) # set the y-axis limits
a.legend(fontsize = 'x-small', loc = 'upper left')


# for the median, data does not need to be filtered
ds_median_Cn2_scid = ds_cn2_scidar.loc[scid_date + 'T00:00:00':scid_date + 'T09:00:00'].median(dim = 'time')/250
ds_median_Cn2_scid = ds_median_Cn2_scid.assign_coords({'altitude': ds_median_Cn2_scid.altitude/1000})
# subplot 11
ds_median_Cn2_scid.plot.line( '-ko',y = 'altitude',ax = a, label = 'Median Profile SCIDAR', markersize = 1)
# a[1][1].invert_yaxis()

a.set_xlabel(r'C$_{n}^2$ [m$^{-2/3}$]')
a.set_ylabel('altitude [km]')
a.set_xscale('log')
a.legend(fontsize = 'x-small', loc = 'upper left')
a.set_xlim(10**(-18), 10**(-14))
a.set_xticks([10**(-18),10**(-17), 10**(-16),10**(-15), 10**(-14)])
a.set_xticklabels([10**(-18),10**(-17), 10**(-16),10**(-15), 10**(-14)], rotation=50)
#plt.tight_layout()
filename = './sites/Paranal/Output/Plots/seeing/SCIDAR/Cn2_Profile/' + scid_date + '_ERA5_Cn2_Profile_overlay.pdf'
plt.savefig(filename)
plt.show()
plt.close()

#%%
# testing
scid_date = SCIDAR_dates_ERA5[0]
pr_level_max = 750 #950
filename_scid = './sites/Paranal/Output/Plots/seeing/SCIDAR/Cn2_Profile/' + scid_date + '_ERA5_Cn2_Profile_750hPa.pdf'
plot_cn2_profile(filename_scid, scid_date, pr_level_max)

# # 1000hPa
# for scid_date in SCIDAR_dates_ERA5:
#     filename_scid = './sites/Paranal/Output/Plots/seeing/SCIDAR/Cn2_Profile/1000hPa/' + scid_date + '_ERA5_Cn2_Profile_1000hPa.pdf'
#     plot_cn2_profile(filename_scid, scid_date)

# 800hPa
# for scid_date in SCIDAR_dates_ERA5:
#     filename_scid = './sites/Paranal/Output/Plots/seeing/SCIDAR/Cn2_Profile/' + scid_date + '_ERA5_Cn2_Profile.pdf'
#     plot_cn2_profile(filename_scid, scid_date)

#%% test fig size
fig, a = plt.subplots(2,2,figsize= (8,4), gridspec_kw={'width_ratios': [4, 1]}, sharey=True, sharex='col')
a[0][0].set_title(r'ERA5: C$_{n}^2$ night profile')
fig.suptitle('Night Profile on {} (UT)'.format(scid_date))
fig.subplots_adjust(bottom=0.1, top=0.85, hspace = 0.4)    

#a[1][0].xaxis_date()

a[1][0].set_xticks([0,1,2,3,4,5,6,7,8,9])
a[1][0].set_xticklabels(['{}:00'.format(str(tim.hour).zfill(2)) for tim in pd.to_datetime(ds_pl.time.values)])

#%% check integral of Cn2

seeing_test = df_scidar.seeing.loc['2016-04-27T00:00:00':'2016-04-27T09:00:00'].mean()
ds_median_Cn2_sum = sum(ds_cn2_scidar.loc['2016-04-27T00:00:00':'2016-04-27T09:00:00'].mean(dim = 'time'))

seeing_test = df_scidar.seeing.loc['2016-04-28T00:00:00':'2016-04-28T09:00:00'].mean()
ds_median_Cn2_sum = sum(ds_cn2_scidar.loc['2016-04-28T00:00:00':'2016-04-28T09:00:00'].mean(dim = 'time'))

# check seeing value at this day: 0.6837692307692308
# (from df_scidar.seeing.loc['2016-04-27'].mean())
def epsilon(J):
  # calculate seeing (epsilon) in arcsec
  wv = 500 * 10**(-9) # wavelength 500nm
  k_wave = 2 * np.pi /wv # wavenumber k
  epsilon = 206265 * 0.976 * wv / (0.423 * k_wave**2 * J)**(-3/5) # rad * 206265 = arcsec
  return(epsilon)

print(f'seeing sum = {epsilon(ds_median_Cn2_sum)}') # = 0.5306 arcesc
print(f'seeing SCIDAR data = {seeing_test}')
# seeing_test on 2016-04-27: 0.6639 arcsec

#%% read ERA5 data
###################################### SEEING #####################################################################

# compare 'seeing' to ERA5 seeing
# open calculated seeing ERA5 data
df_ERA5_seeing = pd.read_csv('./sites/Paranal/Data/Era_5/seeing/hourly_Seeing_integral_2000to2020.csv',index_col='time', parse_dates=True)
# df_ERA5_seeing.loc['2016-04-26']

# open 

# %%

# df_SCIDAR_seeing = test_file


# %% SEEING 1
# seeing
# show timeseries with all measurements first
# (possible for 83 nights)

fig, ax = plt.subplots()
plt.plot(df_scidar.seeing, 'bo', markersize = 1, alpha = 0.7, label = 'SCIDAR data')
for scid_date in SCIDAR_dates:
    plt.plot(df_ERA5_seeing.seeing.loc[scid_date], 'ro', markersize = 1, alpha = 0.7)
plt.ylabel('seeing Paranal [arcsec]')
plt.xlabel('time')
ax.legend(['SCIDAR data', 'ERA5 data'])
plt.savefig('./sites/Paranal/Output/Plots/seeing/SCIDAR/Seeing_comparison_ERA5_SCIDAR_all_values.pdf')

#%% SEEING 2
#show timeseries with merged dataset and resampled in-situ data (from 0 to 5 min)
df_5min_intervals = df_scidar[((pd.DatetimeIndex(df_scidar.index).minute >= 0)) 
 & (pd.DatetimeIndex(df_scidar.index).minute < 5)]

# now resample hourly
df_resampled = df_5min_intervals.resample('h').mean()

fig, ax = plt.subplots()
plt.plot(df_resampled.seeing, 'bo', markersize = 1, alpha = 0.7, label = 'SCIDAR data resampled hourly')
for scid_date in SCIDAR_dates:
    plt.plot(df_ERA5_seeing.seeing.loc[scid_date], 'ro', markersize = 1, alpha = 0.7)
plt.ylabel('seeing Paranal [arcsec]')
plt.xlabel('time')
ax.legend(['SCIDAR data', 'ERA5 data'])
plt.savefig('./sites/Paranal/Output/Plots/seeing/SCIDAR/Seeing_comparison_ERA5_SCIDAR_resampled_hourly.pdf')

#%%
# show hourly correlation plot

df_scidar_preped = mes_prep(df_resampled)

# merge df's
df_merged, seasonal_cycle, diurnal_cycle, monthly_grouped, yearly_grouped = merge_df(df_scidar_preped[['seeing', 'months', 'hours', 'YYYY_MM', 'years']], df_ERA5_seeing, dropnan = True)

filename = './sites/Paranal/Output/Plots/seeing/SCIDAR/Seeing_correlation_ERA5_SCIDAR_hourly.pdf'
title = 'SCIDAR correlation hourly'
data_x = df_merged.seeing_x
string_x = 'SCIDAR seeing [arcsec]'
data_y = df_merged.seeing_y
string_y = 'ERA5 seeing [arcsec]'
correlation_plot(filename, title,  data_x, string_x, data_y ,string_y)

# daily grouped (resample daily)
df_merged_res = df_merged.resample('d').mean()

filename = './sites/Paranal/Output/Plots/seeing/SCIDAR/Seeing_correlation_ERA5_SCIDAR_daily.pdf'
title = 'SCIDAR correlation daily'
data_x = df_merged_res.seeing_x
string_x = 'SCIDAR seeing [arcsec]'
data_y = df_merged_res.seeing_y
string_y = 'ERA5 seeing [arcsec]'
correlation_plot(filename, title,  data_x, string_x, data_y ,string_y)

#%% SEEING 3
# show timeseries with daily values

fig, ax = plt.subplots()
plt.plot(df_merged_res.seeing_x, 'bo', markersize = 1, alpha = 0.7, label = 'SCIDAR data resampled hourly')
plt.plot(df_merged_res.seeing_y, 'ro', markersize = 1, alpha = 0.7)
plt.ylabel('seeing Paranal [arcsec]')
plt.xlabel('time')
ax.legend(['SCIDAR data', 'ERA5 data'])
plt.savefig('./sites/Paranal/Output/Plots/seeing/SCIDAR/Seeing_comparison_ERA5_SCIDAR_resampled_daily.pdf')



# %%
