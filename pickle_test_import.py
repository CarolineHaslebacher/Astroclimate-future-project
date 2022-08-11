# 2021.09.21
# test the import of a pickle file while only loading the pickle library

#%%
import pickle
with open('/home/haslebacher/chaldene/Astroclimate_Project/Astroclimate_outcome/Sutherland_T_d_obs_ERA5_and_insitu.pkl', 'rb') as myfile:
    dload = pickle.load(myfile)

    # this works! one does not even need to load xarray!

# %%
