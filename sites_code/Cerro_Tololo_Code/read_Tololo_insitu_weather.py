
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

df_insitu_Cerro_Tololo = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/weather.txt',header = None, 
names = ['WindSpeed (miles/hr)', 'WindDir (deg)', 'Cerro Tololo Temperature', 'Cerro Tololo Relative Humidity','Cerro Tololo Pressure (hPa)','Time (UTC)'])
 
#%% create datetime inxex from 'date' and 'time'

df_insitu_Cerro_Tololo['time'] = pd.to_datetime(df_insitu_Cerro_Tololo['Time (UTC)'])

#%%
#set time as index
df_insitu_Cerro_Tololo.set_index('time', inplace=True)

#%%
# extract useful data (Relative humidity RH, RH T, Pressure)
df_extracted = df_insitu_Cerro_Tololo[['Cerro Tololo Temperature', 'Cerro Tololo Relative Humidity', 'Cerro Tololo Pressure (hPa)']]

# drop row [940502], there is a '\\N' string
df_extracted = df_extracted[((df_extracted['Cerro Tololo Temperature'] != '\\N') & 
    (df_extracted['Cerro Tololo Relative Humidity'] != '\\N') & 
    (df_extracted['Cerro Tololo Pressure (hPa)'] != '\\N'))]

# change type object to float64 (after removing all string)
df_extracted = df_extracted.apply(pd.to_numeric)

#%%
# filter data

# Temperature values should be between -15 and 60 °C
mask = ((pd.to_numeric(df_extracted['Cerro Tololo Temperature']) > -15) & (pd.to_numeric(df_extracted['Cerro Tololo Temperature']) < 60))
df_extracted = df_extracted[mask]

# RH values should be in between 0 and 100%
mask = ((df_extracted['Cerro Tololo Relative Humidity'] <= 100) & (df_extracted['Cerro Tololo Relative Humidity'] >= 0))
df_extracted = df_extracted[mask]

# pressure should be in between 700 and 800 hPa
mask = ((df_extracted['Cerro Tololo Pressure (hPa)'] < 800) & (df_extracted['Cerro Tololo Pressure (hPa)'] > 700) )
df_extracted = df_extracted[mask]
# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/Cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_T_RH_P.csv', header = True, index = True)

# %%
