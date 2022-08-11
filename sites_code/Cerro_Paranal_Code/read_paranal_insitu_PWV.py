
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
df_insitu_Paranal = pd.read_csv('./sites/Paranal/Data/in-situ/data_from_mail/lhatpro_paranal.csv',header = None)

#%%
# make a copy
df_insitu_Paranal_copy = df_insitu_Paranal[:]
#%% rename columns

df_insitu_Paranal = df_insitu_Paranal.rename(columns={0: 'start_time'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={1: 'time_bin_s'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={2: 'time'})# midpoint_date: Midpoint (time) of the measurement in UT
df_insitu_Paranal = df_insitu_Paranal.rename(columns={3: 'Paranal PWV'}) #mmH2O (convert to kg/m^2)
df_insitu_Paranal = df_insitu_Paranal.rename(columns={4: 'irt0'})# Measured infrared sky brightness temperature at Zenith [Celsius]
df_insitu_Paranal = df_insitu_Paranal.rename(columns={5: 'LWP0'}) # Measured liquid water path in g/m**2 at Zenith

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_Paranal[['time', 'Paranal PWV']]

#%%
# take time and convert to datetime for resampling

#time is in UT, convert to datetime
df_extracted['time'] = pd.to_datetime(df_extracted['time'])

#set time as index
df_extracted.set_index('time', inplace=True)

#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

# %%

# do not resample hourly, because the TCW variable of ERA5 is instantaneous!
#df_resampled = df_extracted.resample('h').mean()

# # sample over a 5min intervall around every full hour
# # filter data, so that only these values remain, then you can resample hourly (but check!)
# df_extracted.loc['2015-04-28 13:30:58']
# df_extracted.loc['2016-04-28 13:57:30':'2016-04-28 14:02:30']

# # write huge mask
# mask = (df_extracted.index >pd.Timestamp(2016,1,1,12,57,29)) & (df_extracted.index < pd.Timestamp(2016,3,1))
# df_extracted[mask]

# delete rows if they do not have minutes (58,59,00,01,02)
# I could add a row with minutes, delete all rows 
df_5min_intervals = df_extracted[((pd.DatetimeIndex(df_extracted.index).minute == 57) & (pd.DatetimeIndex(df_extracted.index).second >= 30)) 
 | (pd.DatetimeIndex(df_extracted.index).minute > 57) | (pd.DatetimeIndex(df_extracted.index).minute < 2)
 | ((pd.DatetimeIndex(df_extracted.index).minute == 2) & (pd.DatetimeIndex(df_extracted.index).second <= 30))]

# now resample hourly
df_resampled = df_5min_intervals.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_instantaneous_Paranal_PWV.csv', header = True, index = True)

 # %%
