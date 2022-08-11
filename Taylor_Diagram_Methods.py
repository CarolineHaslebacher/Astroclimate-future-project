#%%
import math
import skill_metrics as sm
# change in /home/haslebacher/.local/lib/python3.6/site-packages/skill_metrics

import numpy as np
import pandas as pd

import seaborn as sns
# sns.set() # to turn off grid and grey background!

import matplotlib.pyplot as plt


#%% subfigure

# taylor_methods_fig = plt.figure(figsize = (10,4), constrained_layout=True) # ,constrained_layout=True) --> not needed anymore (I moved the ax4 a bit closer to the rest)
# gs = taylor_methods_fig.add_gridspec(1, 2)

# ax0 = taylor_methods_fig.add_subplot(gs[0, 0])
# ax1 = taylor_methods_fig.add_subplot(gs[0, 1])

# plt.figure(figsize=(15,5))
# revision: plot is too small, so I thought about putting two rows and only one column
# okey no, just save individual figures and assemble them in inkscape
# plt.figure(figsize=(15, 20))


#%%

# plot sinus curves
x = np.arange(0, 2* np.pi, 0.2)

y_base = np.sin(x)
y_amplitude = 1.5*np.sin(x+0.02)
y_phase = np.sin(x+1.2)
y_mix = 1.5*np.sin(x+1.2)
# revision 2: add case with offset
y_offset = np.sin(x) + 0.8
# y_mix = 1.5* np.sin(x+1.2) + 3 --> Taylor diagram does not account for shift/bias!

#%%

# plt.subplot(2, 1, 1)

# revision: turn off grid! by simply
# plt.grid(b=None)

markersize=7
# revision 2: use different colors: orange, dark orange, maroon and red 
plt.plot(x, y_base, color='k', label = 'ground truth (Observation)')
plt.plot(x, y_amplitude,color='orange',marker ='p', markersize=markersize, label = 'higher amplitude')
plt.plot(x, y_phase,color='darkorange',marker ='d', markersize=markersize, label = 'phase delay')
plt.plot(x, y_mix,color='maroon',marker ='^', markersize=markersize, label = 'higher amplitude and phase delay')
plt.plot(x, y_offset, color='r',marker ='h', markersize=markersize, label = 'offset')
plt.xlabel('time [hours]')
plt.ylabel('variable (fictitious units)')
# plt.yticks(visible=False)

# 2022-01-17: no legend in A&A!
# plt.legend(loc='lower left')
plt.xticks(ticks = [0.5, 1,1.5, 2, 2.5 ,3, 3.5 ,4, 4.5 ,5, 5.5 ,6], labels=['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24'])

# save!
# save
plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/revision2/figures/Taylor_diagram_explained/Taylor_methods_plot_row1.pdf') #, bbox_inches='tight')
# show and close
plt.show()
plt.close()

#%% corresponding Taylor diagram

ref_pred_ls = [y_base, y_phase, y_amplitude, y_mix, y_offset]

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

# plt.subplot(1, 2, 2)
# 2022-01-17: figure was too small
# plt.subplot(2, 1, 2)
# revision: increase fontsize, directly changed code in skillmetrics
# in 'get_taylor_diagram_options.py', 'plot_pattern_diagram_markers.py' and 'plot_taylor_axes.py'

# revision 2: since we need colors, we need a markerdict
marD = {'p': {'w': ['orange']}, 'd': {'w': ['darkorange']}, '^': {'w': ['maroon']}, 'h': {'w': ['r']}}
# marD = {'*': {'w': ['darkorange', 'darkgoldenrod', 'cadetblue']}

sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='off', styleOBS = '-', markercolor='r',
                colOBS = 'k', markerobs = 'o', # # 2022-01-17: no legend in A&A!
                # markerLabel=['phase delay', 'higher amplitude', 'higher amplitude and phase delay', 'offset'], 
                markerLegend = 'off',
                showlabelscor = 'off',
                stylerms ='-',colRMS='grey',
                # titleOBS = 'Observation', # revision: 'Standard deviation' should be written solely on x-axis!
                titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey',
                alpha=0, markerSize=8, fontsize=10,
                titleSTD='on', MarkerDictCH=marD) # revision: x-axis should be 'standard deviation'

# plot lines
tickval1 = [1, 0.99, 0.95, 0]
middle = np.linspace(0.9, 0.1, 9)
tickval1[3:3] = middle
tickval2 = tickval1[:]
values = np.linspace(-0.1,-0.9,9)
tickval2.extend(values)
tickval2.extend([-0.95, -0.99, -1])
option_tickcor = (tickval1, tickval2)
corr = option_tickcor[0]
th  = np.arccos(corr)
cst = np.cos(th); snt = np.sin(th);
cs = np.append(-1.0*cst, cst)
sn = np.append(-1.0*snt, snt)
for i,val in enumerate(cs):
    plt.plot([0, 2.0*cs[i]], [0, 2.0*sn[i]], 
              linestyle = '-.',
              color = 'grey', linewidth = 0.8)

# plot ground truth with 'no fill' marker
plt.plot(1.0, 0.0, marker='o', color='k', fillstyle='none', markersize=14, markeredgewidth =3)

# overplot markers (without any alpha value...)
# I could change it in the source code, but then it would mess up everything!
# from 'taylor_diagram.py'
rho   = sdev
theta = np.arccos(ccoef)
# Plot data points. Note that only rho[1:N] and theta[1:N] are 
# plotted.
X = np.multiply(rho[1:],np.cos(theta[1:]))
Y = np.multiply(rho[1:],np.sin(theta[1:]))
# marD = {'p': {'w': ['orange']}, 'd': {'w': ['darkorange']}, '^': {'w': ['maroon']}, 'h': {'w': ['r']}}
markers = ['p', 'd', '^', 'h']
colors = ['orange', 'darkorange', 'maroon', 'red']
for i in range(len(X)):
    plt.plot(X[i], Y[i], marker=markers[i], color=colors[i], markersize=11)

plt.gca().set_ylim(-0.07, 2.0)
plt.gca().get_xaxis().set_visible(False)
plt.setp(plt.gca().spines.values(), linewidth=20)

rt = 1.05*2.0
for i,cc in enumerate(corr):
    x = rt*cst[i]
    y = rt*snt[i]
    plt.text(x,y,str(round(cc,2)),
                horizontalalignment = 'center',
                color = 'grey', fontsize = 10)

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/revision2/figures/Taylor_diagram_explained/Taylor_methods_plot_row2.pdf') #, bbox_inches='tight')
# show and close
plt.show()
plt.close()

# box = plt.gca().get_position()
# plt.gca().set_position([box.x0 - 0.03, box.y0, box.width, box.height * 0.8], which='both') # which = 'both', 'original', 'active'

# %%
# save
# plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/Thesis/figures/Taylor_methods_plot.pdf') #, bbox_inches='tight')
# # show and close
# plt.show()
# plt.close()

# %% 2021-05-05

# create three different plots for powerpoint

# fig, ax =

# no time!!


