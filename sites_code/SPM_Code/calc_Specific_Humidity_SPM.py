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
#observational data from SPM; temperature, RH, pressure
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
SPM_hourly = pd.read_csv('./sites/SPM/Data/in-situ/hourly_meteo/hourly_SPM_T_RH_time_in_what.csv', parse_dates = True, index_col = 'time')

# %%
#calculate specific humidity

# (no pressure archive data available on http://tango.astrosen.unam.mx/weather15/)
# I calculated the mean of the filtered pressure measured by the TMT site testing campaign betwenn 2004-2008 in another .py file as 732.76
#  Fill array of length SPM_hourly
SPM_hourly['SPM Pressure'] = 732.76 #hPa

#%%
# initialize new column (veeery fast!)
SPM_hourly['specific_humidity'] = np.vectorize(Specific_humidity)(SPM_hourly['SPM Temperature'],SPM_hourly['SPM Pressure'] , SPM_hourly['SPM Relative Humidity'])
#df_merged_nonan['specific_humidity'] = df_merged_nonan.apply(Specific_humidity(df_merged_nonan.T, df_merged_nonan.P, df_merged_nonan.RH), axis = 1)

#%% to csv
SPM_hourly.to_csv('./sites/SPM/Data/in-situ/hourly_meteo/Specific_humidity_SPM_2006to2020.csv')

# %%
