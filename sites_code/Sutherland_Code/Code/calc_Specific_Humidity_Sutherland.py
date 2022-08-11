# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import dates as d
import datetime as dt
import time

from functools import reduce
import os

# %%
def Specific_humidity(T, p, RH):
    # unit of specific humidity: kg/kg
    # pressure in hPa = mbar
    # Temperature in Celsius
    # RH = relative humidity (%)
    
    # q denotes specific humidity
    # formula: q = qs * RH = 0.622 * es/p * RH
    # with es the saturated vapor pressure, formula: Buck equation, Wikipedia
    q = RH/100 * 0.622 * 6.1121 * 1./p *np.exp((18.678 - T/234.5)*(T/(257.14 + T)))
    return(q)

# %%
#observational data from Sutherland; temperature, RH, pressure
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
Sutherland_hourly = pd.read_csv('./sites/Sutherland/Data/in-situ/hourly_meteo/hourly_Sutherland_T_RH_P.csv')

# %%
#calculate specific humidity

# initialize new column (veeery fast!)
Sutherland_hourly['specific_humidity'] = np.vectorize(Specific_humidity)(Sutherland_hourly['Sutherland Temperature'], Sutherland_hourly['Sutherland Pressure'], Sutherland_hourly['Sutherland Relative Humidity'])
#df_merged_nonan['specific_humidity'] = df_merged_nonan.apply(Specific_humidity(df_merged_nonan.T, df_merged_nonan.P, df_merged_nonan.RH), axis = 1)

#%% to csv
Sutherland_hourly.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/Sutherland/Data/in-situ/hourly_meteo/Specific_humidity_Sutherland.csv')

# %%
