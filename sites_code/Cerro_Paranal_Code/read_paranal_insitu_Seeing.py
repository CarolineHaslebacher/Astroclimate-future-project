
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
df_insitu_Paranal = pd.read_csv('./sites/Paranal/Data/in-situ/data_from_mail/seeing_paranal.csv',header = None)

#%% rename columns

df_insitu_Paranal = df_insitu_Paranal.rename(columns={0: 'start_time'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={1: 'time_bin_s'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={2: 'time'})# midpoint_date: Midpoint (time) of the measurement in UT
df_insitu_Paranal = df_insitu_Paranal.rename(columns={3: 'Seeing Paranal'}) #full width at half maximum [arcsec] = seeing
df_insitu_Paranal = df_insitu_Paranal.rename(columns={4: 'dimm_ra'})# The right ascension of the observed star at which the ASM-DIMM telescope measures the site seeing [deg]
df_insitu_Paranal = df_insitu_Paranal.rename(columns={5: 'dimm_dec'}) # The declination of the observed star at which the ASM-DIMM telescope measures the site seeing [deg]
df_insitu_Paranal = df_insitu_Paranal.rename(columns={6: 'dimm_airmass'}) #	Airmass at which the ASM-DIMM telescope measures the site seeing

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_Paranal[['time', 'Seeing Paranal']]

#%%
# take time and convert to datetime for resampling

#time is in UT, convert to datetime
df_extracted['time'] = pd.to_datetime(df_extracted['time'])

#set time as index
df_extracted.set_index('time', inplace=True)

#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv', header = True, index = True)
 
# %%
