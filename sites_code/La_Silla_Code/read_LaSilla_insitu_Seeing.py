
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
#%%
# read in data of insitu measurements on La_Silla from csv to pandas dataframe
# years 2000 - 2019
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
df_insitu_La_Silla = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/seeing_lasilla.csv',header = None)

#%% rename columns

df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={0: 'start_time'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={1: 'time_bin_s'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={2: 'time'})# midpoint_date: Midpoint (time) of the measurement in UT
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={3: 'Seeing La_Silla'}) #full width at half maximum [arcsec] = seeing
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={4: 'dimm_ra'})# The right ascension of the observed star at which the ASM-DIMM telescope measures the site seeing [deg]
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={5: 'dimm_dec'}) # The declination of the observed star at which the ASM-DIMM telescope measures the site seeing [deg]
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={6: 'dimm_airmass'}) #	Airmass at which the ASM-DIMM telescope measures the site seeing

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Silla[['time', 'Seeing La_Silla']]

#%%
# take time and convert to datetime for resampling

#time is in UT, convert to datetime
df_extracted['time'] = pd.to_datetime(df_extracted['time'])

#set time as index
df_extracted.set_index('time', inplace=True)

#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

#%%
# take INSTANTANEOUS values

df_5min_intervals = df_extracted[((pd.DatetimeIndex(df_extracted.index).minute >= 0)) 
 & (pd.DatetimeIndex(df_extracted.index).minute < 5)]

#%%
# now resample hourly
df_resampled = df_5min_intervals.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_Seeing.csv', header = True, index = True)

# %%
