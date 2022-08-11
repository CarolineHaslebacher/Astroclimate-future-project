
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
df_insitu_La_Silla = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/meteo_lasilla.csv',header = None)
 
#%%
# make a copy
df_insitu_La_Silla_copy = df_insitu_La_Silla[:]
#%%
# take time and convert to datetime
# rename column [0] to time UT
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={0: 'time'})

#time is in UT, convert to datetime
df_insitu_La_Silla['time'] = pd.to_datetime(df_insitu_La_Silla['time'])

#set time as index
df_insitu_La_Silla.set_index('time', inplace=True)

# %%
# rename columns which we are going to use

df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={3: 'La_Silla T 30m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={4: 'La_Silla T 2m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={5: 'La_Silla T ground'})
# df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={7: 'La_Silla RH 30m'}) # only NAN values!
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={8: 'La_Silla RH 2m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={9: 'La_Silla Pressure'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Silla[['La_Silla T 30m', 'La_Silla T 2m', 'La_Silla T ground',
 'La_Silla RH 2m','La_Silla Pressure']]

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

# crop out year 2010 (not full year)
# df_resampled[(pd.to_datetime('2010-01-01') <= df_resampled.index) & (df_resampled.index <= pd.to_datetime('2010-12-31'))]

# write mask, apply! save to new dataset!
mask_2010 = (pd.to_datetime('2010-01-01') <= df_resampled.index) & (df_resampled.index <= pd.to_datetime('2010-12-31'))
df_resampled = df_resampled[~mask_2010] # I simply inverted the mask with ~ :)

#%% 
# save to csv
df_resampled.to_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P.csv', header = True, index = True)


#%%
######################################### ESO QUERY 1993-2020 ####################################################################
# %%
# 1993-2020
df_insitu_La_Silla = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/La_Silla_Meteo_ESO_Query.csv')

#%%
# delete last rows and first  (to start from 1994)
df_insitu_La_Silla = df_insitu_La_Silla[1937:-5]

#%%
# take time and convert to datetime
# rename column [0] to time UT
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Date time': 'time'})

#time is in UT, convert to datetime
df_insitu_La_Silla['time'] = pd.to_datetime(df_insitu_La_Silla['time'])

#set time as index
df_insitu_La_Silla.set_index('time', inplace=True)

# %%
# rename columns which we are going to use

df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Ambient Temperature at 30m [C]': 'La_Silla T 30m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Ambient Temperature at 2m [C]': 'La_Silla T 2m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Ambient Temperature at ground [C]': 'La_Silla T ground'})
# df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={7: 'La_Silla RH 30m'}) # only NAN values!
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Relative Humidity at 2m [%]': 'La_Silla RH 2m'})
df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'Air Pressure at 2m [hPa]': 'La_Silla Pressure'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Silla[['La_Silla T 30m', 'La_Silla T 2m', 'La_Silla T ground',
 'La_Silla RH 2m','La_Silla Pressure']]

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P_1994to2020.csv', header = True, index = True)



# %%
