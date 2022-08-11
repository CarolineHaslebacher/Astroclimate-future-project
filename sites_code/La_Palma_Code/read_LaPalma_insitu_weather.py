
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

Text_filenames_LaPalma = [f for f in os.listdir(path) if f.endswith(".txt")]
df_insitu_La_Palma = []
for filename in Text_filenames_LaPalma:
    df_insitu_La_Palma.append(pd.read_csv('./sites/La_Palma/Data/in-situ/NOT/' + filename, delimiter = '\t', parse_dates=True, index_col = 'DateTimeUT'))

LaPalma_data = pd.concat(df_insitu_La_Palma)

# %%
# rename columns which we are going to use

LaPalma_data = LaPalma_data.rename(columns={'Humidity': 'La_Palma Relative Humidity'})
LaPalma_data = LaPalma_data.rename(columns={'PressureHPA': 'La_Palma Pressure'})
LaPalma_data = LaPalma_data.rename(columns={'TempInAirDegC': 'La_Palma Temperature'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_extracted = LaPalma_data[:]

#%% filter data

# LaPalma_data['La_Palma Temperature'].min() = -10.42 C
# LaPalma_data['La_Palma Temperature'].max() = 1.63318e+16 --> filter
parameter = 'La_Palma Temperature'
mask_T = ((LaPalma_data[parameter] != 0) & (LaPalma_data[parameter] > -20) & (LaPalma_data[parameter] < 50))
LaPalma_data = LaPalma_data[mask_T]

# LaPalma_data['La_Palma Relative Humidity'].min() = 0
# LaPalma_data['La_Palma Relative Humidity'].max() = 775.2 --> filter
parameter = 'La_Palma Relative Humidity'  
mask_RH = (LaPalma_data[parameter] <= 100) & (LaPalma_data[parameter] >= 0) & (LaPalma_data[parameter] != 'nan')
LaPalma_data = LaPalma_data[mask_RH]

# LaPalma_data['La_Palma Pressure'].min() = 0 --> filter
# LaPalma_data['La_Palma Pressure'].max() = 782.49
parameter = 'La_Palma Pressure'
mymean = np.mean(LaPalma_data[parameter])
mask_P = (LaPalma_data[parameter] <= (mymean + 20)) & (LaPalma_data[parameter] >= (mymean - 20)) 
LaPalma_data = LaPalma_data[mask_P]

# %%
# resample hourly
df_resampled = LaPalma_data.resample('h').mean()

#%% 
# save to csv
df_resampled.index = df_resampled.index.rename('time')

df_resampled.to_csv('./sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_RH_T_P.csv', header = True, index = True)

# %%
