# Astroclimate-future-project
Impact of climate change on astronomical observations, assessed with high-resolution global climate models from HighResMIP. 

In this repository, the code that we used for the following publication is stored (in 'master' branch):

C. Haslebacher, M.-E. Demory, B.-O. Demory, M. Sarazin, and P. L. Vidale, "Impact of climate change on site characteristics of eight major astronomical observatories using high-resolution global climate projections until 2050" (in revision)



## Description of code structure
All functions are defined in the file 'climxa.py'. The files 'xarray_all_sites_DSCTi_variable.py' generate a plot with subplots of all sites defined in 'sites_lon_lat_dataframe.py' (here: the eight sites evaluated in the study). The subplots contain a timeseries with yearly averages of in situ data (if available), ERA5 data (this must be available, if in situ data is not given as an input) and PRIMAVERA data (if model = True). 

other files:
- requirements.txt (not all packages used) (pip install -r requirements.txt in venv)
  - I've also used and changed code in the skill_metrics package by Peter Rocheford
  
- workflow: 
  - specify things like lon/lat of your selected site and paths for in situ data in 'sites_lon_lat_dataframe.py'
  - extract regions with 'Extract_lonlat_grid.py' to extract a small region and store it as a netCDF file for faster processing
  - ready to do skill score analysis and trend analysis for temperature, relative humidity and total cloud cover
    - import pickle files. In R, do trend analysis with adapted 'R_Bayesian_Trends.R' for PRIMAVERA and 'R_Bayesian_Trends_ERA5.R' for ERA5
  - compute specific humidity ('SH', 'q' or 'hus') for in situ dataset with adapted version of e.g. 'calc_Specific_Humidity_LaSilla.py'
  - compute precipitable water vapor (PWV) dataset for ERA5 and PRIMAVERA by integrating specific humidity with 'calc_PWV_ERA5.py' and 'Model_SH_integral.py'
  - compute seeing datasets for both methods with 'calc_model_seeing_values.py' (ERA5) and 'xarray_prepare_seeing_data_PRIMAVERA.py' (PRIMAVERA)
  - ready to do skill score analysis and trend analysis for specific humidity, PWV and the two seeing approaches
    - import pickle files. In R, do trend analysis with adapted 'R_Bayesian_Trends.R' for PRIMAVERA and 'R_Bayesian_Trends_ERA5.R' for ERA5




I still need to add the following points to the README file:
add: explain json files
add: bayesian plots
add: csv tables
add: seeing calibration factors k and A

### How to import a pickle file in python

import pickle
with open('mypklfile.pkl', 'rb') as myfile:
    dload = pickle.load(myfile)
    

