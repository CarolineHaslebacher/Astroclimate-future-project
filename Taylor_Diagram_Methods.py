import math
import skill_metrics as sm

import numpy as np
import pandas as pd

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

#%% subfigure

# taylor_methods_fig = plt.figure(figsize = (10,4), constrained_layout=True) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
# gs = taylor_methods_fig.add_gridspec(1, 2) 

# ax0 = taylor_methods_fig.add_subplot(gs[0, 0])
# ax1 = taylor_methods_fig.add_subplot(gs[0, 1])

plt.figure(figsize=(15,5))

#%%

# plot sinus curves
x = np.arange(0, 2* np.pi, 0.2)

y_base = np.sin(x)
y_amplitude = 1.5*np.sin(x+0.02)
y_phase = np.sin(x+1.2)
y_mix = 1.5*np.sin(x+1.2)
# y_mix = 1.5* np.sin(x+1.2) + 3 --> Taylor diagram does not account for shift/bias!

#%%
plt.subplot(1, 2, 1)
plt.plot(x, y_base, color='k', label = 'ground truth (Observation)')
plt.plot(x, y_amplitude,color='b',marker ='o', markersize=5, label = 'higher amplitude')
plt.plot(x, y_phase,color='b',marker ='d', markersize=5, label = 'phase delay')
plt.plot(x, y_mix,color='b',marker ='^', markersize=5, label = 'higher amplitude and phase delay')
plt.xlabel('time [hours]')
plt.ylabel('variable (fictitious units)') 
# plt.yticks(visible=False)
plt.legend(loc='lower left')
plt.xticks(ticks = [0.5, 1,1.5, 2, 2.5 ,3, 3.5 ,4, 4.5 ,5, 5.5 ,6], labels=['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'])

# %%

ref_pred_ls = [y_base, y_phase, y_amplitude, y_mix]

# list comprehension for list ref_pred_ls, to get taylor statistics (compare every prediction to the reference (first entry))
taylor_stats = [sm.taylor_statistics(pred, ref_pred_ls[0],'data') for pred in ref_pred_ls[1:]]

# first entry is special
ls_sdev = []
ls_crmsd = []
ls_ccoef = []
ls_sdev.append(taylor_stats[0]['sdev'][0]/taylor_stats[0]['sdev'][0]) # is it okey, if I just divide at the end to normalize?
ls_crmsd.append(taylor_stats[0]['ccoef'][0]/taylor_stats[0]['sdev'][0] )
ls_ccoef.append(taylor_stats[0]['ccoef'][0])

# print("--- %s seconds for initializing ls_sdev,... lists ---" % (time.time() - start_time))
# start_time = time.time()

# expand
ls_sdev += [taylor_stats[int(i)]['sdev'][1]/taylor_stats[0]['sdev'][0] for i in range(0,int(len(taylor_stats)))] # first is not included; normalize
ls_crmsd += [taylor_stats[int(i)]['crmsd'][1]/taylor_stats[0]['sdev'][0]  for i in range(0,int(len(taylor_stats)))]
ls_ccoef += [taylor_stats[int(i)]['ccoef'][1] for i in range(0,int(len(taylor_stats)))]

sdev = np.array(ls_sdev) # Standard deviations
crmsd = np.array(ls_crmsd) # Centered Root Mean Square Difference 
ccoef = np.array(ls_ccoef) # Correlation

plt.subplot(1, 2, 2)
sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markercolor='b',
                colOBS = 'k', markerobs = 'o',markerLabel=['phase delay', 'higher amplitude', 'higher amplitude and phase delay'], markerLegend = 'on',stylerms ='-',colRMS='grey',
                titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                alpha=1, markerSize=8)

# box = plt.gca().get_position()
# plt.gca().set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

# %%
# save
plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/Thesis/figures/Taylor_methods_plot.pdf') #, bbox_inches='tight')
# show and close
plt.show()
plt.close()

# %%
