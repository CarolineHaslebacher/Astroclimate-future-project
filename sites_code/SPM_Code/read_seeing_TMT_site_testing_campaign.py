
#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv

#%%
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
SPM_Pr = pd.read_csv('./sites/SPM/Data/in-situ/SPM_TMT_data_MASS_Seeing.dat', delimiter = '\t',  index_col='date', parse_dates=True)

# index to datetime
SPM_Pr.index = pd.to_datetime(SPM_Pr.index, format = '%Y %m %d %H %M %S')

SPM_Pr.index = SPM_Pr.index.rename('time')

#%% select 'fixed_seecn': "Free atmosphere seeing calculated from the fixed_cn2 values. This is the most reliable seeing measurement from the MASS."

# plt.plot(SPM_Pr.index, SPM_Pr['fixed_seecn'])
# looks fine

SPM_Pr = SPM_Pr[['fixed_seecn']]

# %%
PR_filtered = SPM_Pr[SPM_Pr['fixed_seecn'] != '****']

Seeing_SPM = PR_filtered['fixed_seecn'].astype('float64')

Seeing_SPM = Seeing_SPM.resample('h').mean()

#%% rename to 'SPM Seeing'

Seeing_SPM= Seeing_SPM.rename('Seeing SPM')

#%% plot again

plt.plot(Seeing_SPM.index, Seeing_SPM)

# I see a lot of missing data!

#%% save to csv

Seeing_SPM.to_csv('./sites/SPM/Data/in-situ/hourly_meteo/Seeing_MASS_TMT_2005to2007.csv')

# %%

# %%
