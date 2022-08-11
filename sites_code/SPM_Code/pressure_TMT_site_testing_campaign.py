
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
SPM_Pr = pd.read_csv('./sites/SPM/Data/in-situ/SPM_TMT_pressure_2004to2008.dat', delimiter = '\t',  index_col='date', parse_dates=True)

# index to datetime
SPM_Pr.index = pd.to_datetime(SPM_Pr.index, format = '%Y %m %d %H %M %S')

# %%
PR_filtered = SPM_Pr[SPM_Pr['pressure'] != '****']

Pressure = PR_filtered['pressure'].astype('float64')

Pressure.mean() # = 732.73
Pressure.std() # = 5.08 --> resample hourly

# filter

mymean = Pressure.mean()
mask_P = (Pressure <= (mymean + 20)) & (Pressure >= (mymean - 20)) 
Pressure_masked = Pressure[mask_P]

Pressure_resampled = Pressure_masked.resample('h').mean()

Pressure_resampled.mean() # = 732.76 --> different location than SPM site data?
Pressure_resampled.std() # = 4.70

plt.plot(Pressure_resampled)


# %%
# monthly averages

mean = Pressure_resampled.resample('1m').mean().mean()
std = Pressure_resampled.resample('1m').mean().std()