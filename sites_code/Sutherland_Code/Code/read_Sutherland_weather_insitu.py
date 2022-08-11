
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
import datetime as dt # here, something is not quite right

#%%

os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
df_insitu_Sutherland = pd.read_csv('./sites/Sutherland/Data/in-situ/Caroline_weather/SALT_weather_data.csv',parse_dates=True, index_col='datetime')

#%% rename columns
df_insitu_Sutherland = df_insitu_Sutherland.rename(columns={'temp': 'Sutherland Temperature', # take 'OutT' as the potentially correct temperature
                            'rel_hum': 'Sutherland Relative Humidity', 
                            'pressure': 'Sutherland Pressure'})

#%% add cloud fraction
# if dry, assign 0 clouds,
# if rain, assign 1
# this might be a very bad estimate...
# we might have to leave this data behind

df_insitu_Sutherland['Sutherland Clouds'] = [0 if x=='DRY' else 1 for x in df_insitu_Sutherland['skycon']]

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_insitu_Sutherland_extracted = df_insitu_Sutherland[['Sutherland Temperature', 'Sutherland Relative Humidity', 'Sutherland Pressure', 'Sutherland Clouds']]

#%%
# rename 'date' to 'time
df_insitu_Sutherland_extracted.index = df_insitu_Sutherland_extracted.index.rename('time')

#%% data is in UTC+2 !!! (private communication)
df_insitu_Sutherland_extracted.index = df_insitu_Sutherland_extracted.index - dt.timedelta(hours=2)

#%% filter out measurement errors

parameter = 'Sutherland Temperature'
mask_T = ((df_insitu_Sutherland_extracted[parameter] != 0) & (df_insitu_Sutherland_extracted[parameter] > -20) & (df_insitu_Sutherland_extracted[parameter] < 50))
df_insitu_Sutherland_extracted = df_insitu_Sutherland_extracted[mask_T]

parameter = 'Sutherland Relative Humidity'  
mask_RH = (df_insitu_Sutherland_extracted[parameter] <= 100) & (df_insitu_Sutherland_extracted[parameter] >= 0) & (df_insitu_Sutherland_extracted[parameter] != 'nan')
df_insitu_Sutherland_extracted = df_insitu_Sutherland_extracted[mask_RH]

parameter = 'Sutherland Pressure'
mymean = np.mean(df_insitu_Sutherland_extracted[parameter])
mask_P = (df_insitu_Sutherland_extracted[parameter] <= (mymean + 20)) & (df_insitu_Sutherland_extracted[parameter] >= (mymean - 20)) 
df_insitu_Sutherland_extracted = df_insitu_Sutherland_extracted[mask_P]

# %%
# resample hourly
df_insitu_Sutherland_extracted_resampled = df_insitu_Sutherland_extracted.resample('h').mean()

#%% 
# save to csv
df_insitu_Sutherland_extracted_resampled.to_csv('./sites/Sutherland/Data/in-situ/hourly_meteo/hourly_Sutherland_T_RH_P.csv', header = True, index = True)


# %%

