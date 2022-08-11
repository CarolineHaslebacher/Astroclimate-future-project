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
#observational data from siding_spring; temperature, RH, pressure
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
siding_spring_hourly = pd.read_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/hourly_siding_spring_T_RH_P_time.csv')

#%% mask siding_spring_hourly

# #filter out values exactly equal to 0, smaller than -100°C and greater than 30°C (years 1991 and 1992)
# parameter = 'siding spring Temperature'
# mask_T = ((siding_spring_hourly[parameter] != 0) & (siding_spring_hourly[parameter] > -30) & (siding_spring_hourly[parameter] < 50))
# siding_spring_hourly = siding_spring_hourly[mask_T]

# parameter = 'siding spring Relative Humidity'    
# mask_RH = (siding_spring_hourly[parameter] <= 100) & (siding_spring_hourly[parameter] >= 0) & (siding_spring_hourly[parameter] != 'nan')
# siding_spring_hourly = siding_spring_hourly[mask_RH]

# parameter = 'siding spring Pressure'
# mymean = np.mean(siding_spring_hourly[parameter])
# mask_P = (siding_spring_hourly[parameter] <= (mymean + 20)) & (siding_spring_hourly[parameter] >= (mymean - 20)) 
# siding_spring_hourly = siding_spring_hourly[mask_P]

# %%
#calculate specific humidity

# initialize new column (veeery fast!)
siding_spring_hourly['specific_humidity'] = np.vectorize(Specific_humidity)(siding_spring_hourly['siding spring Temperature'], siding_spring_hourly['siding spring Pressure'], siding_spring_hourly['siding spring Relative Humidity'])
#df_merged_nonan['specific_humidity'] = df_merged_nonan.apply(Specific_humidity(df_merged_nonan.T, df_merged_nonan.P, df_merged_nonan.RH), axis = 1)

#%% to csv
siding_spring_hourly.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/siding_spring/Data/in-situ/Specific_humidity_siding_spring_2003to2020.csv')

# %%
