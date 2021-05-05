# Astroclimate-future-project
Impact of climate change on astronomical observations, assessed with high-resolution global climate models from HighResMIP. 

## Description of code structure
All functions are defined in the file 'climxa.py'. The files 'xarray_all_sites_DSCTi_variable.py' generate a plot with subplots of all sites defined in 'sites_lon_lat_dataframe.py' (here: the eight sites evaluated in the study). The subplots contain a timeseries with yearly averages of in situ data (if available), ERA5 data (if I remember correctly, this must be available, if in situ data is not given as an input) and PRIMAVERA data (if model = True). 

`this will be formatted as a line of code`

write about:
- requirements.txt (not all packages used)
  - plus: changed code in skill_metrics
  
- workflow: 
  - specify things like lon/lat of your selected site and paths for in situ data in 'sites_lon_lat_dataframe.py'
  - extract regions with 'Extract_lonlat_grid.py'
  - ready to do skill score analysis and trend analysis for temperature, relative humidity and total cloud cover
    - run ???? (import pickle files) and in R, do trend analysis with adapted 'R_Bayesian_Trends.R' for PRIMAVERA and 'R_Bayesian_Trends_ERA5.R' for ERA5
  - compute specific humidity ('SH', 'q' or 'hus') for in situ dataset with adapted version of 'calc_Specific_Humidity_LaSilla.py'
  - compute precipitable water vapor (PWV) dataset for ERA5 and PRIMAVERA by integrating specific humidity with 'calc_PWV_ERA5.py' and 'Model_SH_integral.py'
  - compute seeing datasets for both methods with 'calc_model_seeing_values.py' (ERA5) and 'xarray_prepare_seeing_data_PRIMAVERA.py' (PRIMAVERA)
  - ready to do skill score analysis and trend analysis for specific humidity, PWV and the two seeing approaches
    - run ???? (import pickle files) and in R, do trend analysis with adapted 'R_Bayesian_Trends.R' for PRIMAVERA and 'R_Bayesian_Trends_ERA5.R' for ERA5


I would need to change all  PATHS!!
also, change the header of the files. sometimes I copied a file and did not change the header!
