# 2021-07-05
# Pier Luigi suggested making a table for the pressure levels, and I thought a plot might do better
# author: Caroline Haslebacher


#%%

import matplotlib.pyplot as plt
import numpy as np

#%%

era5_pr =  [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
primavera_pr = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 1]

#%%


# plt.figure()

fig, ax1 = plt.subplots(figsize=(2,17))

ax1.plot(np.zeros(len(era5_pr)), era5_pr, marker='p', color='maroon', linestyle='', label='ERA5')

plt.legend()
plt.ylabel('Pressure levels ERA5 [hPa]')
plt.xticks([])
plt.yticks(era5_pr[:-5] + [1])


# new axis on the right
ax2 = ax1.twinx()
ax2.plot(np.ones(len(primavera_pr)), primavera_pr, marker='d', color='navy', linestyle='', label='PRIMAVERA')

# s2 = np.sin(2*np.pi*t)
# ax2.plot(t, s2, 'r.')
ax2.set_ylabel('Pressure levels PRIMAVERA [hPa]')
ax2.set_yticks(primavera_pr[:-2] + [1])
# ax2.set_yticklabels(primavera_pr[:-2] + [1])

ax1.legend(bbox_to_anchor=(0.5,0))
leg2 = ax2.legend(bbox_to_anchor=(1.5,0))
ax2.add_artist(leg2)

# ax0.legend(handles=line_list, loc='upper left', ncol=2, bbox_to_anchor= (0, -0.3))
# leg1 = ax3.legend(handles=forcing_line_list, loc='upper right', bbox_to_anchor = (1, -0.3)) # legend of forcings (under timeseries)
# leg2 = ax3.legend(handles=Patch_list_ensemble,loc='upper right', bbox_to_anchor=(0.7, -0.3), ncol=3)
# # because leg1 will be automatically removed, I have to add it again as an artist
# ax3.add_artist(leg1)
# ax3.add_artist(leg0) # don't forget insitu plot!

ax1.invert_yaxis()
ax2.invert_yaxis()

plt.savefig('pressure_levels_ERA5_PRIMAVERA.pdf', bbox_inches='tight')
plt.show()


# %% revised version, 2022-01-17

fig, ax1 = plt.subplots(figsize=(2,17))

# revision: for loop
for prlev in era5_pr:
    ax1.plot((0,0.45), (prlev,prlev), linestyle='-.', color='maroon')

# plt.legend()
plt.ylabel('Pressure levels [hPa]')
plt.xticks([])
plt.yticks(era5_pr[:-5] + [1])

# revision
for prlev in primavera_pr:
    ax1.plot((0.55, 1), (prlev, prlev), linestyle='--', color='navy')

ax1.invert_yaxis()

plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/publication/revision/figures/pressure_levels_ERA5_PRIMAVERA.pdf', bbox_inches='tight')
plt.show()

# %%
