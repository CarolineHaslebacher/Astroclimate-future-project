

#%% import
import os
from pathlib import Path
from shutil import copyfile

#%%

base_path = Path('/home/haslebacher/chaldene/Astroclimate_Project/HighResMIP/variables')

dest = Path('/home/haslebacher/chaldene/Astroclimate_Project/publication/wget_scripts')
wgets = sorted(base_path.glob('**/*.sh'))

for shfile in wgets:
    copyfile(shfile, dest.joinpath(shfile.name))


# %%


