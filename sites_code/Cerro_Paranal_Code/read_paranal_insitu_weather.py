
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
df_insitu_Paranal = pd.read_csv('./sites/Paranal/Data/in-situ/data_from_mail/meteo_paranal.csv',header = None)

#%%
# make a copy
df_insitu_Paranal_copy = df_insitu_Paranal[:]
#%%
# take time and convert to datetime
# rename column [0] to time UT
df_insitu_Paranal = df_insitu_Paranal.rename(columns={0: 'time'})

#time is in UT, convert to datetime
df_insitu_Paranal['time'] = pd.to_datetime(df_insitu_Paranal['time'])

#set time as index
df_insitu_Paranal.set_index('time', inplace=True)

# %%
# rename columns which we are going to use

df_insitu_Paranal = df_insitu_Paranal.rename(columns={4: 'Paranal T 2m'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={6: 'Paranal T -20m'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={11: 'Paranal RH 2m'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={12: 'Paranal RH -20m'})
df_insitu_Paranal = df_insitu_Paranal.rename(columns={14: 'Paranal Pressure'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_Paranal[['Paranal T 2m', 'Paranal T -20m',
'Paranal RH 2m', 'Paranal RH -20m','Paranal Pressure']]

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% crop Paranal T -20m to full year

df_resampled['Paranal T -20m'][df_resampled.index < '2007'] = np.nan

#%% 
# save to csv
df_resampled.to_csv('./sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_RH_T_P.csv', header = True, index = True)
