
#%%
import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm

#%%

sdev = np.zeros((5))
crmsd = np.zeros((5))
ccoef = np.zeros((5))

sdev[:] = [1, 1.1, 1.2, 0.9, 0.7]
crmsd[:] = [0, 1.8, 2, 1.2, 0.5]
ccoef[:] = [1, 0.7, 0.8, 0.8, 0.75]

#%%

label = ['obs']

marD = {'d': {'w': ['darkorange', 'darkgoldenrod', 'darkgreen', 'skyblue']}# ,'p': {'w': ['darkorange', 'darkgoldenrod', 'darkgreen', 'skyblue']}
}

fig, ax = plt.subplots(1,1)
sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-',
markerLabel = label, colOBS = 'r', markerobs = 'o',
markerLegend = 'off',stylerms ='-',colRMS='grey', titleOBS = 'Obs',
titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', MarkerDictCH=marD,
alpha=0.7, markerSize= 9)

plt.tight_layout()
plt.savefig('taylor_tester.png' ,dpi=300, bbox_inches='tight')
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
import skill_metrics as sm

sdev = [1.,         0.79489189, 0.72909418, 0.67983078, 0.66597742, 0.81855761, 0.234, 0.564, 0.675]
crmsd = [0.49512275, 0.35768422, 0.4479255,  0.54590334, 0.56531812, 0.49595067, 0.7731455, 0.595637, 0.4708768]
ccoef = [1.,         0.94598721, 0.91273604, 0.85621265, 0.84382842, 0.86986516, 0.9765, 0.854, 0.914]

sdev = np.array(sdev)
crmsd = np.array(crmsd)
ccoef = np.array(ccoef)

# marD = {'*': {'w': ['red', 'maroon', 'darkorange', 'darkgoldenrod', 'm']}}

marD = {'*': {'w': ['darkorange', 'darkgoldenrod'], 'k': ['darkgoldenrod', 'cadetblue']}, 'p': {'w': ['darkorange', 'darkgreen'], 'k': ['darkorange', 'red']}}

taylor_label = ['hist', 'present']

sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                    colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                    titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                    MarkerDictCH=marD, alpha=0.2, markerSize= 9) 


#
# or another one with three labels
marD = {'d': {'w': ['darkorange', 'darkgoldenrod'], 'k': ['cadetblue'], 'blue': ['darkgoldenrod', 'cadetblue']}, 'p': {'w': ['darkorange'], 'k': ['darkorange', 'red']}}

taylor_label = ['hist', 'present', 'future']

sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = taylor_label,
                    colOBS = 'r', markerobs = 'o', markerLegend = 'off',stylerms ='-',colRMS='grey',
                    titleOBS = 'Observation', titleRMS = 'off', titleRMSDangle=20, colCOR='dimgrey', 
                    MarkerDictCH=marD, alpha=0.2, markerSize= 9) 


# %%
STDs = 0.234
STD0 = 1
CORs = 0.9765
RMSs = np.sqrt(STDs**2 + STD0**2 - 2*STDs*STD0*CORs) # 0.7731455231714144

STDs = 0.564
STD0 = 1
CORs = 0.854
RMSs = np.sqrt(STDs**2 + STD0**2 - 2*STDs*STD0*CORs) # 0.5956374736364395

STDs = 0.675
STD0 = 1
CORs = 0.914
RMSs = np.sqrt(STDs**2 + STD0**2 - 2*STDs*STD0*CORs) # 0.4708768


# %%
