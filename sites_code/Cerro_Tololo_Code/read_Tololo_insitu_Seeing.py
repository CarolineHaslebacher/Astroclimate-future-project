
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

#%%
# read in data of insitu measurements on Cerro_Tololo from csv to pandas dataframe
# years 2007 - 2019

os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

df_insitu_Cerro_Tololo_1 = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/seeing_2004-2015.csv', delimiter=';', parse_dates=True)
df_insitu_Cerro_Tololo_2 = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/DIMM2_SEEING_2015-2020.csv', delimiter=';', parse_dates=True)

# combine dataframes
# keep in mind that 2015 is not a full year (no data in between 2015-09-14 and 2015-12-22)
df_insitu_Cerro_Tololo = pd.concat([df_insitu_Cerro_Tololo_1, df_insitu_Cerro_Tololo_2])

#%%

#%% create datetime inxex from 'date' and 'time'

df_insitu_Cerro_Tololo['time'] = pd.to_datetime(df_insitu_Cerro_Tololo['ut'])

#%%
#set time as index
df_insitu_Cerro_Tololo.set_index('time', inplace=True)

#%% rename columns

df_insitu_Cerro_Tololo= df_insitu_Cerro_Tololo.rename(columns={'seeing': 'Seeing Cerro_Tololo'})

#%%
# extract useful data (Relative humidity RH, RH T, Pressure)
df_extracted = df_insitu_Cerro_Tololo[['Seeing Cerro_Tololo']]

#%%plot
plt.plot(df_extracted.index, df_extracted['Seeing Cerro_Tololo'])

# sort by date (there is one date in 1970)
df_extracted = df_extracted.sort_index()

# filter this date
df_extracted = df_extracted[df_extracted.index > pd.datetime(1970,1,1, 12,0,0)]


# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_Seeing_dimm.csv', header = True, index = True)

# %%
