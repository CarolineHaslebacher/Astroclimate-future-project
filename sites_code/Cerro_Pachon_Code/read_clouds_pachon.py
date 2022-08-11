# this code reads in water vapor monitor (WVM) log files from the JCMT archive at the Canadian Astronomy Data Centre (CADC)
# PWV is converted from mmH2O to kg m-2
# it filters the data (values above 40 kg m-2 and below 0 are filtered out.)
# it resamples the data to hourly data
# it saves the data to csv


#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table, vstack, hstack
from astropy.utils.data import download_file
from astropy.time import Time
from astropy.timeseries import TimeSeries
# import sunpy.timeseries as ts  
import netCDF4
import xarray as xr

import datetime as dt

from bs4 import BeautifulSoup
from urllib.request import HTTPError

import os

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

df_insitu = pd.read_csv('./sites/Cerro_Tololo/Data/in-situ/gemini_weather_loss_2013-2017_South_Pachon.csv')

#%%
# extract dates
import calendar

# create dict with month names
month_to_num_dict = {v: k for k,v in enumerate(calendar.month_abbr)}

# df_insitu['month'] = [month_to_num_dict[str(y)[4:7]] for y in df_insitu['Night Log Date']]

from datetime import datetime
# assign value to first day (e.g. Wed Jan01, 2014 - Thu Jan02, 2014 assign to January 1). since Hawai has timedelta of -10 hours.
df_insitu['time'] = [datetime(int(str(x)[11:15]), month_to_num_dict[str(x)[4:7]], int(str(x)[7:9])) for x in df_insitu['Night Log Date']]

#set time as index
df_insitu.set_index('time', inplace=True)

# %%
# calculate weather loss fraction of the night

# note from andy adamson (gemini): "note - weather loss means "time during which nothing in the queue that matches the prevailing conditions". it doesn't mean it was raining... "

df_insitu['Cerro_Tololo Clouds'] = df_insitu['weather']/df_insitu['available']

# %%
# daily data

# extract useful data
df_extracted = df_insitu[['Cerro_Tololo Clouds']]

#%% sort
df_extracted = df_extracted.sort_index()

#%% filter inf values!!
df_extracted = df_extracted[df_extracted['Cerro_Tololo Clouds'] < 1.03]


# %%
# save to csv
df_extracted.to_csv('./sites/Cerro_Tololo/Data/in-situ/daily_weather_loss_Cerro_Pachon_Gemini.csv', header = True, index = True)

# %%
