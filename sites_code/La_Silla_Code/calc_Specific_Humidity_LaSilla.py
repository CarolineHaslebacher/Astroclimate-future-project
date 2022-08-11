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
    try:
        q = RH/100 * 0.622 * 6.1121 * 1./p *np.exp((18.678 - T/234.5)*(T/(257.14 + T)))
    except ZeroDivisionError:
        q = np.nan
    return(q)



# %%
# prepare CFHT observational data for merging
def mes_prep(CFHT_hourly, parameter = None):
    CFHT_hourly = CFHT_hourly.rename(columns={'Unnamed: 0': 'time'})
    # change the format of the times column to datetime format
    CFHT_hourly['time'] = pd.to_datetime(CFHT_hourly['time']) 

    #check the format
    print(CFHT_hourly['time'].dtype)
    #print(CFHT_hourly['time'][0])

    #from HST to UTC (+10 hours)
    CFHT_hourly['time'] = CFHT_hourly['time'] + dt.timedelta(hours=10)
    
    #set index 
    CFHT_hourly.set_index('time', inplace=True)
    
    if parameter == 'temperature(C)':
        #filter out values exactly equal to 0
        mask_T = (CFHT_hourly['temperature(C)'] != 0)
        CFHT_hourly = CFHT_hourly[mask_T]
        print('masked')

    # create a new column consisting of the cycle parameter of the correspondend entry
    #for seasonal cycle (12 months), create column with "months"
    #CFHT_hourly['months'] = pd.DatetimeIndex(CFHT_hourly.index).month                                            

    #for diurnal cycle (24 hours), create column with "hours"
    #CFHT_hourly['hours'] = pd.DatetimeIndex(CFHT_hourly.index).hour
    
    return(CFHT_hourly)


# %%
#observational data from La_Silla, ESO, temperature, RH, pressure
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
La_Silla_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P.csv')

# 2020-08-12: oh wow, there is data reaching back to 1994! Why did I not take that one?
# La_Silla_hourly = pd.read_csv('./sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_RH_T_P_1994to2020.csv')
# --> data is bad. looks wrong!

# %%
#calculate specific humidity

# initialize new column (veeery fast!)
La_Silla_hourly['specific_humidity'] = np.vectorize(Specific_humidity)(La_Silla_hourly['La_Silla T 2m'], La_Silla_hourly['La_Silla Pressure'], La_Silla_hourly['La_Silla RH 2m'])
#df_merged_nonan['specific_humidity'] = df_merged_nonan.apply(Specific_humidity(df_merged_nonan.T, df_merged_nonan.P, df_merged_nonan.RH), axis = 1)

#%% to csv
La_Silla_hourly.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_La_Silla_ESO_2000to2019.csv')
# La_Silla_hourly.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_La_Silla_ESO_1994to2020.csv')

# %%

