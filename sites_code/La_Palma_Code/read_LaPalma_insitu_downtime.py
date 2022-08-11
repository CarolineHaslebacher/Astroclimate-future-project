
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

#%%
# read in data of insitu measurements on La_Palma from csv to pandas dataframe
# data from http://www.not.iac.es/weather/

os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

df_insitu = pd.read_csv('./sites/La_Palma/Data/in-situ/downtime_LaPalma.csv')

# %%

df_insitu['La_Palma Clouds'] = df_insitu['down_meteo']/df_insitu['night_length']

#%% plot to check

plt.plot(df_insitu['La_Palma Clouds'].index, df_insitu['La_Palma Clouds'])

#%% time index

# ATTENTION: these are night time values

df_insitu['time'] = pd.to_datetime(df_insitu['night']) 

df_insitu.set_index('time', inplace=True)

#%%

df_extracted = df_insitu[['La_Palma Clouds']]

# %%

#%% 
# save to csv

df_extracted.to_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/daily_laPalma_downtime_fraction.csv', header = True, index = True)

# %%
