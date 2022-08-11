# This script simply generates a plot of all Patches belonging to the ERA5 pressure levels
# %%
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
 
from itertools import cycle

import seaborn as sns
sns.set()

from matplotlib import cycler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


#%%
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import climxa

import importlib
importlib.reload(climxa)

#%%

d_color = climxa.d_color_generator() # get all defined colors from climxa

#%%
Patch_list = []
for key, val in d_color.items():
    # fill Patch list wit items in defined color dictionary
    if '0' in key or '5' in key: # search for pressure levels
        Patch_list.append(Patch(facecolor=val,edgecolor=val, label = key + ' hPa'))
    else: # else, do not add 'hPa'
        # if key == 'tcw':
        #     key = 'total column water'
        # elif key == 't2m' or key == 'tas':
        #     key = 'two-metre temperature'
        # elif key == 'prw':
        #     key = 'precipitable rain water'
        if key == 'q_integrated':
            key = 'PWV from integral'
        Patch_list.append(Patch(facecolor=val,edgecolor=val, label = key))

# also include single level color
# Patch_list.append(Patch(facecolor='m', edgecolor='m', label = 'single level data'))
                
#%% plot color palette

fig, ax = plt.subplots(figsize=(2,5))
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
ax.set_axis_off()
ax.legend(handles=Patch_list , loc='upper left', bbox_to_anchor=(0, 1))

fig.savefig('/home/haslebacher/chaldene/Astroclimate_Project/Thesis/figures/Colorpalette_ERA5_Pr_levels.png')
plt.show()
# we have the problem that colors are not distinguishable for the colorblind...!?
# but: colors are often less important than symbols

# I forgot single level data!!

#%%
