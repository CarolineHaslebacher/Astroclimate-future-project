
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

from datetime import datetime

#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#%%
# read in data of insitu measurements on La_Palma from csv to pandas dataframe
# years 2000 - 2019 (nighttime values!)

df_insitu_La_Palma = pd.read_csv('./sites/La_Palma/Data/in-situ/2003-2019_seeing.txt', 
                                names=['date', 'time1', 'Seeing La_Palma'], skiprows=2, delimiter=' ', parse_dates=True)

#%% to datetime

# df_insitu_La_Palma['date'] = pd.to_datetime(df_insitu_La_Palma['date'])
# df_insitu_La_Palma['time1'] = pd.to_datetime(df_insitu_La_Palma['time1'])

df_insitu_La_Palma['time'] = [str(x) + 'T' + str(y) for x, y in zip(df_insitu_La_Palma['date'], df_insitu_La_Palma['time1'])]

df_insitu_La_Palma['time'] = pd.to_datetime(df_insitu_La_Palma['time'])
# set as index
df_insitu_La_Palma.set_index('time', inplace=True)


#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Palma[['Seeing La_Palma']]

#%% plot to check data

plt.plot(df_extracted.index, df_extracted['Seeing La_Palma'])
# --> there is one value of 10**10: filter it!
masked1 = df_extracted['Seeing La_Palma'][df_extracted['Seeing La_Palma'] < 10**10]
plt.plot(masked1.index, masked1)

# looks ok

df_extracted = df_extracted[df_extracted['Seeing La_Palma'] < 10**10]




#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_Seeing.csv', header = True, index = True)

# %%
