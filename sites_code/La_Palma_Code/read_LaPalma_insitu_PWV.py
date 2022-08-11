
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

#%%
# read in data of insitu measurements on La_Palma from csv to pandas dataframe
# years 2000 - 2019
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

# information about data here: https://www.iac.es/en/observatorios-de-canarias/sky-quality/sky-quality-parameters/precipitable-water-vapour
# I downloaded data from here (see diary): http://research.iac.es/proyecto/site-testing/index.php?option=com_wrapper&Itemid=122
#  

df_insitu_La_Palma = pd.DataFrame()

for year in range(2011,2018): # stop after 2017, because 2018 only contains some days of january
    file_name = 'pwv_final_'+ str(year) +'0101_'+ str(year) +'1231.dat'
    df = pd.read_csv('./sites/La_Palma/Data/in-situ/' + file_name, skiprows=7, parse_dates=True)
    df_insitu_La_Palma = df_insitu_La_Palma.append(df)

#%% rename columns

df_insitu_La_Palma = df_insitu_La_Palma.rename(columns={'PWV_final[mm]': 'La_Palma PWV'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = df_insitu_La_Palma[['La_Palma PWV']]

#%%
# take time and convert to datetime for resampling

#time is in UT, convert to datetime
df_extracted['time'] = pd.to_datetime(df_insitu_La_Palma['DATE[yyyy/mm/dd]'] + df_insitu_La_Palma['UT[HH:MM]'], format = '%Y/%m/%d%H:%M')

#set time as index
df_extracted.set_index('time', inplace=True)

#%% convert PWV from mm to kg/m^2
# nothing to do: 1mmH2O = 1 kg/m^2

# %%
# resample hourly
df_resampled = df_extracted.resample('h').mean()

#%% 
# save to csv
df_resampled.to_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_PWV.csv', header = True, index = True)

# %%
# df_lapalma = pd.read_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_PWV.csv')
# mean = 8.73 mm
# median = 7.7 mm

# %%
