# 2021-05-08
# Caroline Haslebacher
# a script that reads in Tololo and Pachon in situ data and plots them
# also, we should count the number of missing dates...!

#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file

import climxa

import importlib
importlib.reload(climxa)

#%% read in in situ data from Pachon, all three possible sources

# cp: time: 32488 without nan! (2007 to 2019)
df_Pachon_cp = climxa.df_to_xarray('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_cp.csv')
df_Pachon_ct = climxa.df_to_xarray('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_ct.csv')
df_Pachon_ws = climxa.df_to_xarray('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_ws.csv')

# and in situ data from Tololo
# time: 150618 (2002 to 2019)
df_Tololo = climxa.df_to_xarray('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Data/in-situ/Specific_humidity_Cerro_Tololo_2002to2019.csv')

#%% check missing data: 
# calculate amount of hours between start date and end date
# then, drop nans and see how many dates we end up with

# start date Tololo: 2002-01-18T21:00:00
# end date Tololo: 2019-03-26T14:00:00
# 2002-01-18 to 2019-01-18: (365*17 + 4)*24 (for Schaltjahre) = 149016
# plus 2019-01-18 to 2019-03-26: 67*24 = 1608
# both together: 150624
# and written in xarray is: 150618 (Good)
# now drop nan's: 

df_Tololo['Cerro Tololo Relative Humidity'].dropna(dim='time')
# time: 146735 --> missing hours: 150618 - 146735 = 3883 hours
# relative missing hours: 3883/150618 = 2.6%

# %% plot, but first, resample daily at least!

# make 

df_Tololo = df_Tololo.resample(time = '1d').mean()
df_Pachon_cp = df_Pachon_cp.resample(time = '1d').mean()
df_Pachon_ct = df_Pachon_ct.resample(time = '1d').mean()
df_Pachon_ws = df_Pachon_ws.resample(time = '1d').mean()

# scatterplot
plt.figure(figsize=(20, 7))
plt.scatter(df_Tololo['time'], df_Tololo['Cerro Tololo Relative Humidity'], s=4, label='CTIO')
plt.scatter(df_Pachon_cp['time'], df_Pachon_cp['RH'], s=4,  label = 'Gemini cp')
plt.scatter(df_Pachon_ct['time'], df_Pachon_ct['RH'], s=4,  label = 'Gemini ct')
plt.scatter(df_Pachon_ws['time'], df_Pachon_ws['RH'], s=4,  label = 'Gemini ws')
plt.legend()

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Output/Plots/RH/Pachon_vs_Tololo_daily_scatter.pdf')

# connected through line
plt.figure(figsize=(20, 7))
plt.plot(df_Tololo['time'], df_Tololo['Cerro Tololo Relative Humidity'], alpha=0.7, label='CTIO')
plt.plot(df_Pachon_cp['time'], df_Pachon_cp['RH'], alpha=0.7, label = 'Gemini cp')
plt.plot(df_Pachon_ct['time'], df_Pachon_ct['RH'], alpha=0.7, label = 'Gemini ct')
plt.plot(df_Pachon_ws['time'], df_Pachon_ws['RH'], alpha=0.7, label = 'Gemini ws')
plt.legend()

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Output/Plots/RH/Pachon_vs_Tololo_daily_line.pdf')

#%% resample monthly
# run first cell again (load xarrays again)
df_Tololo = df_Tololo.resample(time = '1m').mean()
df_Pachon_cp = df_Pachon_cp.resample(time = '1m').mean()
df_Pachon_ct = df_Pachon_ct.resample(time = '1m').mean()
df_Pachon_ws = df_Pachon_ws.resample(time = '1m').mean()

plt.figure(figsize=(20, 7))
plt.plot(df_Tololo['time'], df_Tololo['Cerro Tololo Relative Humidity'], alpha=0.7, label='CTIO')
plt.plot(df_Pachon_cp['time'], df_Pachon_cp['RH'], alpha=0.7, label = 'Gemini cp')
plt.plot(df_Pachon_ct['time'], df_Pachon_ct['RH'], alpha=0.7, label = 'Gemini ct')
plt.plot(df_Pachon_ws['time'], df_Pachon_ws['RH'], alpha=0.7, label = 'Gemini ws')
plt.legend()

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Output/Plots/RH/Pachon_vs_Tololo_monthly.pdf')

# plot tololo alone to see missing data
plt.plot(df_Tololo['time'], df_Tololo['Cerro Tololo Relative Humidity'], label='CTIO')
plt.legend()
plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Output/Plots/RH/Tololo_monthly_to_see_missing_data.pdf')
# --> no completely missing months!


# %% and yearly 
df_Tololo = df_Tololo.resample(time = '1y').mean()
df_Pachon_cp = df_Pachon_cp.resample(time = '1y').mean()
df_Pachon_ct = df_Pachon_ct.resample(time = '1y').mean()
df_Pachon_ws = df_Pachon_ws.resample(time = '1y').mean()

plt.plot(df_Tololo['time'], df_Tololo['Cerro Tololo Relative Humidity'], label='CTIO')
plt.plot(df_Pachon_cp['time'], df_Pachon_cp['RH'], label = 'Gemini cp')
plt.plot(df_Pachon_ct['time'], df_Pachon_ct['RH'], label = 'Gemini ct')
# plt.plot(df_Pachon_ws['time'], df_Pachon_ws['RH'], label = 'Gemini ws')
plt.legend()

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Pachon/Output/Plots/RH/Pachon_vs_Tololo_yearly.pdf')


# %%



