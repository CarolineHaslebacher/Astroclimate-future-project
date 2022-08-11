
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os

#%%
# read in data of insitu measurements on Cerro_Pachon from csv to pandas dataframe
# years 2007 - 2019

os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

# initialize dataframe that can be appended
#df_insitu_Cerro_Pachon = pd.DataFrame(columns=['date','time', 'index1', 'index2', 'RH'])
df_insitu_Cerro_Pachon_cp = pd.DataFrame(columns=['date','time', 'index1', 'index2', 'RH'])
df_insitu_Cerro_Pachon_ct = pd.DataFrame(columns=['date','time', 'index1', 'index2', 'RH'])
df_insitu_Cerro_Pachon_ws = pd.DataFrame(columns=['date','time', 'index1', 'index2', 'RH'])

# read RH
for year in range(2007,2020):
    print(year)
    path = './sites/Cerro_Pachon/Data/in-situ/GS_Humidity/RH_' + str(year) + '.txt'

    table = pd.read_table(path, header = None ,delimiter = ' ', names = ['date','time', 'index1', 'index2', 'RH'], 
        skipinitialspace = True, skip_blank_lines=True , comment='#')
    # split dataframe by "Channel: ws:cpHumid (averaged)" and "Channel: ws:ctHumid (averaged)" and "Channel: ws:wsHumid"
    # give back row index with "ws:cpTemp50m" = idx50
    idxcp = table.index[table['time'].str.contains('ws:cpHumid')]
    # index with "Channel: ws:ctTemp"
    idxct = table.index[table['time'].str.contains('ws:ctHumid')]
    # index with "Channel: ws:wsTemp"
    idxws = table.index[table['time'].str.contains('ws:wsHumid')]

    df_insitu_Cerro_Pachon_cp = df_insitu_Cerro_Pachon_cp.append(table.iloc[(idxcp.item() + 1) :idxct.item()])
    df_insitu_Cerro_Pachon_ct = df_insitu_Cerro_Pachon_ct.append(table.iloc[(idxct.item() + 1):idxws.item()])
    df_insitu_Cerro_Pachon_ws = df_insitu_Cerro_Pachon_ws.append(table.iloc[(idxws.item() + 1):-1])

    #df_insitu_Cerro_Pachon = df_insitu_Cerro_Pachon.append(table)

# error_bad_lines=False,

#%% create datetime inxex from 'date' and 'time'

df_insitu_Cerro_Pachon_cp['time'] = pd.to_datetime(df_insitu_Cerro_Pachon_cp.date) + pd.to_timedelta(df_insitu_Cerro_Pachon_cp['time'])
df_insitu_Cerro_Pachon_ct['time'] = pd.to_datetime(df_insitu_Cerro_Pachon_ct.date) + pd.to_timedelta(df_insitu_Cerro_Pachon_ct['time'])
df_insitu_Cerro_Pachon_ws['time'] = pd.to_datetime(df_insitu_Cerro_Pachon_ws.date) + pd.to_timedelta(df_insitu_Cerro_Pachon_ws['time'])

#%%
#set time as index
df_insitu_Cerro_Pachon_cp.set_index('time', inplace=True)
df_insitu_Cerro_Pachon_ct.set_index('time', inplace=True)
df_insitu_Cerro_Pachon_ws.set_index('time', inplace=True)

#%%
# extract useful data (Relative humidity RH, RH T, Pressure)
df_extracted_cp = df_insitu_Cerro_Pachon_cp[['RH']]
df_extracted_ct = df_insitu_Cerro_Pachon_ct[['RH']]
df_extracted_ws = df_insitu_Cerro_Pachon_ws[['RH']]

# filter data (values should not be below 0%), there are a lot in ct dataset
mask = df_extracted_ct['RH'] > 0
df_extracted_ct = df_extracted_ct[mask]

# %%
# resample hourly
df_resampled_cp = df_extracted_cp.resample('h').mean()
df_resampled_ct = df_extracted_ct.resample('h').mean()
df_resampled_ws = df_extracted_ws.resample('h').mean()

#%% 
# save to csv
df_resampled_cp.to_csv('./sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_cp.csv', header = True, index = True)
df_resampled_ct.to_csv('./sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_ct.csv', header = True, index = True)
df_resampled_ws.to_csv('./sites/Cerro_Pachon/Data/in-situ/hourly_meteo/hourly_Cerro_Pachon_RH_ws.csv', header = True, index = True)



# %%
# %% mail from John Blakeslee (quintessence: take cp!)
# but: in my answer to him, I told him that ct was the most complete... 

# Dear Caroline,

# Your Master's thesis sounds quite interesting. We don't have all the information you request, such as precipitable water vapor, but we do have a few things.  I attach two compressed tar files:

# GS_Temperature.tgz
 
# GS_Humidity.tgz

# When you un-tar, each of these is a directory (folder) containing 13 text files, each of which has the Temperature or Humidity data for roughly a year.  The columns are fairly self-evident (date, time, some sort of index, followed by temp or relative humidity).  There are also some blank and nonsense lines in there that you'll need to sort out.  The names of the individual files are not meaningful - they just represent the times at which each set of data was extracted from our engineering archive.  So, you have to look in the first column of each file to get the dates.

# For each Temperature file, there are two sensors, or "channels" – the name of the "channel" is listed on a given line, and then all the following lines are the measurements from that channel, until you get to another line with a different channel name.  For example, you will see lines like:
#      Channel: ws:cpTemp50m (averaged)
#    Channel: ws:ctTemp (averaged)

# All the data lines following "Channel: ws:cpTemp50m" are from the sensor at 50m (I believe), while all those following the "Channel: ws:ctTemp" line are from another sensor.  I don't know all the details off the top of my head.  In some files (some years) there may not be useful data from some channels, so you'll just need to discard all those.  I think using the "cpTemp50m" data might be best. 

# Similarly, there are multiple channels for the humidity, and I think using "cpHumid (averaged)" may be best, but see what you find.  You would probably be the first person ever to look at all these humidity data, so it's undiscovered territory.

# We do not have similarly sampled cloud-cover (nebulosity) data.  The only thing we could provide in that regard is information on the total number of hours lost per night as a result of bad weather, dating back about 10 years or so.  Let me know if that would be useful and we'll see what we can do.

# Good luck.

 
# Very truly yours,
# John
 
# --
# John Blakeslee
# Chief Scientist, Gemini Observatory
# Tel: +56 51-2205-628 (office)  
