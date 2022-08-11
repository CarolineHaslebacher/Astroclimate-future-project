#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
from datetime import datetime

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')

#%%

AAT_weather_loss = pd.read_csv('./sites/siding_spring/Data/in-situ/aat_time_lost_weather.csv', parse_dates=True)

#%% dates

AAT_weather_loss['time'] = pd.to_datetime(AAT_weather_loss['utdate'], format='%d/%m/%Y')
AAT_weather_loss.set_index('time', inplace=True)

# %%
# if comment not equal to np.nan, delete entire row

mask_comment = (AAT_weather_loss['comment'].map(str).map(len) < 4)
AAT_weather_loss = AAT_weather_loss[mask_comment]
# missing values: 10415 - 10370

#%%

# comments from Daniel (AAT)
# - Add the at and ct times together to get the total time available.
# - Divide the wt columns by that to get the percentage of time lost.
# - Exclude nights with comment fields.
# - The best data will be the night (n) columns, a Night Assistant uses some judgement for twilight (t) columns, and there are instrument/programme variations in the amount available.
# - There is probably insufficient data to look for long term trends in day time (d) weather losses. 

# add 'atn' and 'ctn'

AAT_weather_loss['atn+ctn'] = AAT_weather_loss['atn'] + AAT_weather_loss['ctn']

AAT_weather_loss['percentage_of_time_lost'] = AAT_weather_loss['wtn']/AAT_weather_loss['atn+ctn']

#%% check for NaN values

mask_nan_percentage = ~np.isnan(AAT_weather_loss['percentage_of_time_lost'])
AAT_weather_loss = AAT_weather_loss[mask_nan_percentage]

# missing values: 10415 - 10238

#%% write percentage into new dataset

AAT_weather_loss_extracted = AAT_weather_loss[['percentage_of_time_lost']]

#%%
# cut out years that are dawned only (not complete)
# df[datetime(2018,1,1):datetime(2018,1,10)]
AAT_weather_loss_extracted = AAT_weather_loss_extracted[datetime(1993, 1,1) : datetime(2019, 1,31)]

# %% resample monthly

AAT_weather_loss_resampled = AAT_weather_loss_extracted.resample('1m').mean()

# %% save to csv

AAT_weather_loss_resampled.to_csv('./sites/siding_spring/Data/in-situ/AAT_weather_loss_percentage.csv')


# %%
