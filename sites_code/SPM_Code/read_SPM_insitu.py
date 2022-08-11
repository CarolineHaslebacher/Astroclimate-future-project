
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
import datetime as dt

from urllib.request import HTTPError
from astropy.utils.data import download_file
from astropy.table import Table

#%% read in data from "mirror" page

base_url = "http://tango.astrosen.unam.mx/weather15/Archive/ARC-"
# example http://tango.astrosen.unam.mx/weather15/Archive/ARC-2015-05-04.txt

tables = []

# data starts in 2006-07-01, end in 2019-12-31
days_SPM = np.arange('2006-07', '2020-01', dtype='datetime64[D]')

for i in range(0, len(days_SPM)):
    try:
        path = base_url +str(days_SPM[i]) +'.txt'

        header_names = ['timestamp',	'SPM Temperature',	'Chill', 'HIndex',	'SPM Relative Humidity','Dewpt', 'Wind','HiWind', 'WindDir','Rain',	'Barom', 'Solar', 'ET', 'UV'] 
        table = pd.read_table(path, names = header_names, delimiter='\t',  skiprows=2)
        table_extracted = table[['timestamp','SPM Temperature', 'SPM Relative Humidity']]
        # print(table_extracted)
        tables.append(table_extracted)

    except HTTPError:
        print('page not available {}'.format(days_SPM[i]))
        pass
# table = pd.to_datetime(table['time'][:-1]) # 24:00 is not valid (should be 00:00:00 next day)
SPM_data_table = pd.concat(tables)

os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
SPM_data_table.to_csv('./sites/Baja/Data/in-situ/hourly_meteo/SPM_raw_df_2006to2019.csv', header = True, index = True)     

#%% takes too long! just cut out 24:00
def my_24_format_to_datetime(date_str):
    #for i, date_str in date_col.items():
        # print(date_str)
    #print(date_str[9:11])
    if date_str[9:11] != '24': # if 24 not in right part of string
        
        return pd.to_datetime(date_str, format='%Y%m%d %H:%M')
    else:
        date_str = date_str[0:9] + '00' + date_str[11:]
        return pd.to_datetime(date_str, format='%Y%m%d %H:%M') + dt.timedelta(days=1)

SPM_data_table['time'] = np.vectorize(my_24_format_to_datetime)(SPM_data_table['timestamp'])

SPM_data_table.set_index('time', inplace=True)

# # or just cut out 24:00
# SPM_data_table['timestamp'] = pd.to_datetime(SPM_data_table['timestamp'][:-1])
# # 287 * 

# %%
# quickly save before resampling
SPM_data_table.to_csv('./sites/Baja/Data/in-situ/hourly_meteo/SPM_time_index_df_2006to2019.csv', header = True, index = True)     

#%% filter out measurement errors

parameter = 'SPM Temperature'
mask_T = ((SPM_data_table[parameter] != 0) & (SPM_data_table[parameter] > -20) & (SPM_data_table[parameter] < 50))
SPM_data_table = SPM_data_table[mask_T]

# I checked for Relative Humidity: min = 0, max = 100 :)
# parameter = 'SPM Relative Humidity'  
# mask_RH = (SPM_data_table[parameter] <= 100) & (SPM_data_table[parameter] >= 0) & (SPM_data_table[parameter] != 'nan')
# SPM_data_table = SPM_data_table[mask_RH]

#%%
# resample 
SPM_data_table_resampled = SPM_data_table.resample('h').mean()

#%% 
# save to csv
SPM_data_table_resampled.to_csv('./sites/Baja/Data/in-situ/hourly_meteo/hourly_Baja_T_RH_time_in_what.csv', header = True, index = True)

# %%
