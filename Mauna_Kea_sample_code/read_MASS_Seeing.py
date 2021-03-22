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

#%%
# example https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/JCMT/20200304.wvm
# data starts in 2005-01-01
base_url = "http://mkwc.ifa.hawaii.edu/current/seeing/mass/"



# days_in_2010 = np.arange('2010-01', '2011-01', dtype='datetime64[D]')

#t = days_in_2010[0].astype(dt.datetime)ValueError
#t.strftime('%Y%m%d')

# days_2010_to_2019 = np.arange('2009-02', '2020-01', dtype='datetime64[D]')
days_2010_to_2019 = np.arange('2010-01-01', '2020-01', dtype='datetime64[D]')

tables = []


# define which columns should be removed
remove_cols = ['col1','col2', 'col3', 'col4', 'col5', 'col6', 'col7']
labels = ['col1','col2', 'col3', 'col4', 'col5', 'col6', 'Seeing MaunaKea'] # 'year', 'month', 'day', 'hour', 'min', 'sec', 
    
# time is in UTC

# implement 'try' 'except' for errors where no data is available under the composed URL
# for loop i year (= 365 days)
with open('MaunaKea_seeing_missing_dates.txt', 'w') as f:
    print('missing_dates_MK_Seeing', file=f)
    k = 0 # index for list (there are missing values, so i is not a good choice)
    for i in range(0, len(days_2010_to_2019)):
        try:

            t = days_2010_to_2019[i].astype(dt.datetime)
            t = t.strftime('%Y%m%d')
            #print(t)
            #print(base_url + '{0}.wvm'.format(t))
            with open(download_file(base_url + '{0}.mass.dat'.format(t), cache=True), 'r') as file:
                raw_table = file.read()

            table = Table.read(raw_table, format='ascii.no_header', delimiter=' ')
            
            # compose time
            table['time'] = [dt.datetime(x,y,z,u,v,w) for x,y,z,u,v,w in zip(table['col1'], table['col2'], table['col3'], table['col4'], table['col5'], table['col6'])]
            table['time'] = pd.to_datetime(table['time'])

            # # filter table so that I only are left with the hourly instantanteous values
            # table = table[((pd.DatetimeIndex(table['time']).minute == 57) & (pd.DatetimeIndex(table['time']).second >= 30)) 
            #     | (pd.DatetimeIndex(table['time']).minute > 57) | (pd.DatetimeIndex(table['time']).minute < 2)
            #     | ((pd.DatetimeIndex(table['time']).minute == 2) & (pd.DatetimeIndex(table['time']).second <= 30))]

            # --> do not filter to hourly instantaneous values, since only a few values remain (like 4 per day...)
            
            # rename columns, so that important columns do not get deleted in next for loop
            for col_ind, label in zip(remove_cols, labels):
                try:
                    if label not in table.colnames:
                        table.rename_column(col_ind, label)
                except KeyError:
                    pass
            

            for i in range(0,len(remove_cols)): # loop over columns to remove
                try:
                    if remove_cols[i] in table.colnames: # prevents problems if you run it a second time
                        table.remove_column(remove_cols[i])
                except KeyError:
                    pass
            
            # convert to dataframe to resample hourly
            df = table.to_pandas()
            
            # set time as index for resampling
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            # resample hourly
            df_resampled = df.resample('h').mean()

            tables.append(df_resampled)
            # or omit .append with the following lines
            # tables[k] = df_resampled
            # k = k + 1

        except (HTTPError, FileNotFoundError):
            print('no data for {0}'.format(t))
            # print missing dates to file
            
            print('{}'.format(t), file=f)
            pass


#%%
# merge list of dataframes into one dataframe with concat
df = pd.concat(tables)

import os
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

df.to_csv('./sites/MaunaKea/Data/in-situ/Seeing/hourly_nighttime_SeeingV_MKWC_2010to2019.csv', header = True, index = True)

#%% read in for filtering

# df = pd.read_csv('./sites/MaunaKea/Data/in-situ/PWV/hourly_PWV_MK_JCMT_instantaneously.csv', index_col = ['time'], parse_dates =True)

# #%% plot diurnal cycle
# # --> we get whole different cycle!?!?

# plt.plot(df.groupby(df.index.hour).mean())

# #%% check for values greater than 10mmH2O

# # initially 6696 rows
# # there are 263 values with '99.0' mmH2O. Filter these
# # there are 320 values above 30mmH2O
# mask1 = (df['PWV'] <= 30) & (df['PWV'] > 0)
# df[mask1] # --> got rid of ca. 300 entries

# # mask1 = (df['PWV'] > 0)
# # df[mask1] # --> got rid of ca. 300 entries

# # mask1 = (df['PWV'] >= 30)
# # df[mask1] # --> got rid of ca. 300 entries


# df_masked = df[mask1]
# plt.plot(df_masked.groupby(df_masked.index.hour).mean())
# plt.plot(df_masked.groupby(df_masked.index.hour).std())

#%%
# # make a copy of tables
# tables_copy = tables[:]


# #%%

# # select column with PWV
# # time is already in UTC

# # remove other columns than column 1 (time) and 10 (PWV in mmH2O)
# remove_cols = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8','col9','col10', 'col11', 
#     'col12', 'col13', 'col14', 'col15']
# labels = ['UTC_time','col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8','col9','PWV', 'col11', 
#     'col12', 'col13', 'col14', 'col15']
# for k in range(0, len(tables)): # loop over tables

#     # rename columns, so that important columns do not get deleted in next for loop
#     for col_ind, label in zip(remove_cols, labels):
#         if label not in tables[k].colnames:
#             tables[k].rename_column(col_ind, label)


#     for i in range(0,len(remove_cols)): # loop over columns to remove
#         if remove_cols[i] in tables[k].colnames: # prevents problems if you run it a second time
#             tables[k].remove_column(remove_cols[i])

        

# #%%
# # filter data (first 30 and last 30 entries (measurments were running when dome was opening/closing))
# # is equal to approx. 30s and 30s of time

# # mask tables (PWV data of Era 5 is never higher than 30 kg m-2)
# # -> filter values higher than 40kg m-2
# tables_filtered = []

# # for k in range(0, len(tables)):
# #     mask1 = (tables[k]['PWV'] <= 30) & (tables[k]['PWV'] > 0)
# #     tables_filtered.append(tables[k][mask1])

# for k in range(0, len(tables)): # loop over tables, cut off first 30 and last 30 values
#     tables_filtered.append(tables[k][30:-30])

# #%%
# # conversion from mmH2O to kg m-2
# tables_filtered_kgm_2 = tables_filtered[:]

# # WRONG!! 1 mmH2O = 1 kg m^(-2)
# # for k in range(0, len(tables_filtered)):
# #     tables_filtered_kgm_2[k]['PWV'] = [element * 10 for element in tables_filtered[k]['PWV']]

# # %%
# # to dataframe, with index time, then resample hourly
# df = []

# for k in range(0, len(tables_filtered)):
#     df.append(tables_filtered_kgm_2[k].to_pandas())

# #%%
# # set time as an index
# for k in range(0, len(df)):
#     df[k]['UTC_time'] = pd.to_datetime(df[k]['UTC_time'])
#     df[k].set_index('UTC_time', inplace=True)

# #%%
# # resample to hourly data
# df_resampled = []
# for k in range(0, len(df)):
#     df_resampled.append(df[k].resample('h').mean())

# # %%
# # to csv
# # concat (could have done that earlier!)
# df_concat = pd.concat(df_resampled) #, header=['time', parameter]

# # save to csv
# df_concat.to_csv('./sites/MaunaKea/Data/in-situ/PWV/hourly_PWV_MK_JCMT_30s.csv', header = True, index = True)


# %%
# plt.plot(df_concat.index, df_concat['PWV'])

# %%
