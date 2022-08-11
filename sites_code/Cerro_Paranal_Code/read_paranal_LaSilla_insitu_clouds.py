
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

#%%
df_insitu_Paranal = pd.read_csv('./sites/Paranal/Data/in-situ/clouds-Paranal.txt', 
                               names=['Paranal Clouds', 'La_Silla Clouds', 'Year_string', 'nothing'], 
                                skiprows=9, delimiter='\t')

#%% new column for datetime index

df_insitu_Paranal['month'] = [12 if x%12==0 else x%12 for x in df_insitu_Paranal.index]

#%%
df_insitu_Paranal['year'] =  ['20' + str(y)[4:] if int(str(y)[4:]) < 20 else '19' + str(y)[4:] for y in df_insitu_Paranal['Year_string']]

#%%
from datetime import datetime

df_insitu_Paranal['time'] = [datetime(int(x), int(y), 1) for x, y in zip(df_insitu_Paranal['year'], df_insitu_Paranal['month'])]


#%%
# make a copy
# df_insitu_Paranal_copy = df_insitu_Paranal[:]

# #%%
# # take time and convert to datetime
# # rename column [0] to time UT
# df_insitu_Paranal = df_insitu_Paranal.rename(columns={0: 'time'})

# #time is in UT, convert to datetime
# df_insitu_Paranal['time'] = pd.to_datetime(df_insitu_Paranal['time'])

#set time as index
df_insitu_Paranal.set_index('time', inplace=True)

#%% La_Silla to float

# filter out '*'

df_insitu_Paranal['La_Silla Clouds'] = df_insitu_Paranal['La_Silla Clouds'][df_insitu_Paranal['La_Silla Clouds'] != '*']

df_insitu_Paranal['La_Silla Clouds'] = df_insitu_Paranal['La_Silla Clouds'].astype(float)

#%%
# extract useful data
df_extracted = df_insitu_Paranal[['Paranal Clouds', 'La_Silla Clouds']]

#%% drop nan

df_extracted = df_extracted.dropna(how='any')

#%% the data displays the photometric night fraction, which is the fraction of the night that WAS PHOTOMETRIC (Photometric: No visible clouds, transparency variations under 2%, only assessable by the analysis of photometric standard stars.)
# therefore we need to take 1- value to get the fraction of non-photometric nights!!!

df_extracted['Paranal Clouds'] = 1-df_extracted['Paranal Clouds']
df_extracted['La_Silla Clouds'] = 1-df_extracted['La_Silla Clouds']

# %%
# resample monthly
df_resampled = df_extracted.resample('m').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Paranal/Data/in-situ/hourly_meteo/monthly_photometric_nights_Paranal_LaSilla.csv', header = True, index = True)

#%%