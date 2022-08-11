
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

from datetime import datetime

#%%
# read in data of insitu measurements on La_Silla from csv to pandas dataframe

# data comes from here: https://www.eso.org/gen-fac/pubs/astclim/lasilla/h2o/
# also checkout http://www.eso.org/gen-fac/pubs/astclim/forecast/meteo/ERASMUS/exp_fore.html#Humidity%20Parameters for documentation 
# for clouds as well!!

# check with mean/median if this is really for La Silla 
#  --> I checked with mean/median and can say for sure that this is data from La Silla (mean: 6.2, median 5.92)
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
df_insitu_La_Silla = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/LaSilla_PWV_2000to2007.txt', skiprows=1,  delim_whitespace=True , engine='python', index_col = False) #delim_whitespace=True,

# df_insitu_La_Silla = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/PWV_lasilla_test.txt', skiprows=1,delim_whitespace=True , engine='python') #sep = r'[ |]'


#%% rename columns

df_insitu_La_Silla = df_insitu_La_Silla.rename(columns={'PWV(mm)': 'La_Silla PWV'})

#%%
# compose time string from 'UTC' and 'hour'
time_str = df_insitu_La_Silla['UTC'] + df_insitu_La_Silla['hour']
# convert to datetime object
time_index = pd.to_datetime(time_str, format = '%Y.%m.%d%Hh')

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Silla[['La_Silla PWV']]

#%%

#set time as index
df_extracted.set_index(time_index, inplace=True)

# rename index as 'time'
df_extracted.index = df_extracted.index.rename('time')

#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

#%%
# missing data
# 1513 out of 19113

# if measurement was taken every 3 hours (from 00:00am to 12:00am until 2000.07.26, from 00:00am to 21:00pm rest) from 2000.07.10 to 2007.12.31,
# that would make 17(days) * 5(measurements a day) + (5 + 31+30+31+30+31)(days) * 8(measurements) + 365 * 7 * 8(measurements)
# = 21789
# --> 21789 - (19113 - 1513) = 4189 missing values (out of 21789) (makes 19.23%)

# df_insitu_La_Silla['La_Silla PWV'][df_insitu_La_Silla['La_Silla PWV'] == 'Pixels'] 
# this is the missing data

# find duplicate indices (1277 duplicated values!)
# --> 4189 + 1277 = 5466 missing values totally!
df_extracted = df_extracted[~df_extracted.index.duplicated()]
 
# select only valid data
# do not use the last value (duplicate index causes problems!)
df_extracted = df_extracted[df_extracted['La_Silla PWV'] != 'Pixels'].astype(float)

# time index is not properly sorted
# sort it now
df_extracted = df_extracted.sort_index()

#%%
# attention: maybe these are mean values (over 3 hours), and not instantaneous values?

#%% upsample to hourly data
# use linear interpolation

df_hourly_upsampled = df_extracted.resample('1h').interpolate("linear")

#%% 
# save to csv
df_extracted.to_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_PWV.csv', header = True, index = True)
df_hourly_upsampled.to_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_upsampled_Linear_interpolation_La_Silla_PWV.csv', header = True, index = True)

# %%
 