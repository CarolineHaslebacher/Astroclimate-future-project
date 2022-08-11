
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
#%%
# read in data of insitu measurements on Paranal from csv to pandas dataframe
# years 2000 - 2019
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
df_insitu_Paranal = pd.read_csv('./sites/Paranal/Data/in-situ/Eso_download_Seeing/Free_atmospheres_seeing_2016to2020.csv')


#%% rename columns

df_insitu_Paranal = df_insitu_Paranal.rename(columns={'Date time': 'time'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={'Free Atmosphere Seeing ["]': 'free_atmosphere_seeing'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={'Free Atmosphere Seeing RMS': 'RMS_free_atmosphere_seeing'})# midpoint_date: Midpoint (time) of the measurement in UT
df_insitu_Paranal = df_insitu_Paranal.rename(columns={'MASS-DIMM Seeing ["]': 'MASS_DIMM_Seeing'}) #full width at half maximum [arcsec] = seeing
df_insitu_Paranal = df_insitu_Paranal.rename(columns={'Cn2 [10**(-15)m**(1/3)]': 'Cn2'})# The right ascension of the observed star at which the ASM-DIMM telescope measures the site seeing [deg]

#%%
# drop Nan's
df_extracted = df_insitu_Paranal.dropna(axis = 0, how = 'any')

#%%
# take time as index and convert to datetime for resampling

#time is in UT, convert to datetime
df_extracted['time'] = pd.to_datetime(df_extracted['time'])

#set time as index
df_extracted.set_index('time', inplace=True)

#%%
# take INSTANTANEOUS values
# delete rows if they do not have minutes (00,01,02,03,04)

# df_5min_intervals = df_extracted[((pd.DatetimeIndex(df_extracted.index).minute == 57) & (pd.DatetimeIndex(df_extracted.index).second >= 30)) 
#  | (pd.DatetimeIndex(df_extracted.index).minute > 57) | (pd.DatetimeIndex(df_extracted.index).minute < 2)
#  | ((pd.DatetimeIndex(df_extracted.index).minute == 2) & (pd.DatetimeIndex(df_extracted.index).second <= 30))]

df_5min_intervals = df_extracted[((pd.DatetimeIndex(df_extracted.index).minute >= 0)) 
 & (pd.DatetimeIndex(df_extracted.index).minute < 5)]

# now resample hourly
df_resampled = df_5min_intervals.resample('h').mean()

# %%
# resample hourly
#df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_FA_Seeing_instantaneous_MASS_DIMM_free_atmosphere_1804.csv', header = True, index = True)

# %%
