# RUN THIS SCRIPT IF YOU WANT YOUR CHANGES TO BE APPLIED!


# this script saves the site specific data such as longitude and latitude of the map and of the observatory
# pressure information, elevation, site name and observatory name in a dataframe
# the indexing is as follows:
# [0]: Mauna Kea
# [1]: Cerro Paranal
# [2]: La Silla
# [3]: Cerro Tololo
# [4]: La Palma
# [5]: Siding Spring
# [6]: Sutherland
# [7]: SPM

#%% import libraries
import pandas as pd


#%% initiate dataframe with site specific data

# initialize empty lists that can be filled in the following
data = {'site_name':  [],
        'observatory_name': [],'upper_lon': [], 'lower_lon': [],
 'lower_lat': [], 'upper_lat': [], 'lon_obs': [], 'lat_obs': [],'ele_obs': [], 'pressure [hPa]': [],
   'path_ds_SH_RH_T': [], 'path_ds_PWV': [], 'path_ds_clouds': [],'site_name_folder': [], 'ls_pr_levels_ERA5': [], 'ls_pr_levels_clim_model': [],
    'ls_pr_level_seeing': [], 'time_slice_var_seeing': [], 'path_ds_seeing': [],
    'time_slice_var_meteo': [], 'time_slice_var_PWV': [], 'time_slice_var_clouds': []} # for model plots


# %% Mauna Kea
# latitude and longitude in 0-360

lower_lon = (360-156)
upper_lon = (360-154.5)
lower_lat = 18.5
upper_lat = 20.5

# longitude and latitude of observatory

lon_obs = 360-155.46806
lat_obs = 19.82083
ele_obs = 4204

# site name
site_name = 'Mauna Kea (USA)'
observatory_name = f'Canada France Hawaii \n Telescope (CFHT): \n{ele_obs}m'

# pressure = 616

path_ds_SH_RH_T = './sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv'
path_ds_PWV = './sites/MaunaKea/Data/in-situ/PWV/hourly_PWV_MK_JCMT_instantaneously.csv'
path_ds_clouds = './sites/MaunaKea/Data/in-situ/daily_weather_loss_MaunaKea_Gemini.csv'
site_name_folder = 'MaunaKea'
ls_pr_levels_ERA5 = [600]   # code reads in automatically data for these levels
ls_pr_levels_clim_model = [600]
time_slice_var_meteo = slice('2000-01-01','2019-12-31')
time_slice_var_PWV = slice('2009-01-01','2019-12-31')
time_slice_var_clouds = slice('2014-01-01', '2017-09-30')
# seeing
ls_pr_level_seeing = [800]
time_slice_var_seeing = slice('2010-01-01', '2019-12-31')
path_ds_seeing = './sites/MaunaKea/Data/in-situ/Seeing/hourly_nighttime_SeeingV_MKWC_2010to2019.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)


data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(616) # think about error calculation



# %% Cerro Paranal

# latitude and longitude in 0-360
lower_lon = (360-71)
upper_lon = (360-69.25)
lower_lat = -25.75
upper_lat = -23.75

# longitude and latitude of observatory
lon_obs = 360-70.4
lat_obs = -24.63
ele_obs = 2635

# site name
site_name = 'Cerro Paranal (Chile)'
observatory_name = f'Paranal Observatory (ESO): \n{ele_obs}m'

path_ds_SH_RH_T = './sites/Paranal/Data/in-situ/SH_calculated/Specific_humidity_Paranal_ESO_2000to2019.csv'
path_ds_PWV = './sites/Paranal/Data/in-situ/hourly_meteo/hourly_instantaneous_Paranal_PWV.csv'
path_ds_clouds = './sites/Paranal/Data/in-situ/hourly_meteo/monthly_photometric_nights_Paranal_LaSilla.csv'
site_name_folder = 'Paranal'
ls_pr_levels_ERA5 =  [750] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [700]
time_slice_var_meteo = slice('2000-01-01','2019-12-31')
time_slice_var_PWV = slice('2016-01-01','2019-12-31') # note: data goes until 2019-12-02
time_slice_var_clouds = slice('1984-01-01', '2017-12-31')
# seeing
ls_pr_level_seeing = [900]
time_slice_var_seeing = slice('2000-01-01', '2016-12-31')
path_ds_seeing = './sites/Paranal/Data/in-situ/hourly_meteo/hourly_Paranal_Seeing.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)

data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(744)

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)


# %% La Silla

# latitude and longitude in 0-360
lower_lon = (360-71.5)
upper_lon = (360-70)
lower_lat = -29.75
upper_lat = -28.5

# longitude and latitude of observatory
lon_obs = 360-70.73
lat_obs = -29.25
ele_obs = 2400

# site name
site_name = 'La Silla (Chile)'
observatory_name = f'La Silla Observatory: \n{ele_obs}m'

# path_ds_SH_RH_T = './sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_30m_La_Silla_ESO_2000to2019.csv'
path_ds_SH_RH_T = './sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_La_Silla_ESO_2000to2019.csv'
# path_ds_SH_RH_T = './sites/La_Silla/Data/in-situ/ESO/Specific_humidity_RH_2m_T_2m_La_Silla_ESO_1994to2020.csv'
path_ds_clouds = './sites/Paranal/Data/in-situ/hourly_meteo/monthly_photometric_nights_Paranal_LaSilla.csv'
path_ds_PWV = './sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_PWV.csv'
site_name_folder = 'La_Silla'
ls_pr_levels_ERA5 = [775] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [850]
time_slice_var_meteo = slice('2000-01-01','2015-12-31') 
time_slice_var_PWV = slice('2001-01-01','2007-12-31')
time_slice_var_clouds = slice('1984-01-01', '2017-12-31')
# seeing
ls_pr_level_seeing = [825]
time_slice_var_seeing = slice('2000-01-01', '2019-12-31')
path_ds_seeing = './sites/La_Silla/Data/in-situ/ESO/hourly_meteo/hourly_La_Silla_Seeing.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)
data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(771)

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)


# %% Cerro Tololo

# latitude and longitude in 0-360
lower_lon = (360-71.5)
upper_lon = (360-70)
lower_lat = -31
upper_lat = -29.5

# longitude and latitude of observatory
lon_obs = 360-70.80
lat_obs = -30.17
ele_obs = 2207

# site name
site_name = 'Cerro Tololo (Chile)'
observatory_name = f'Astronomical Observatory Cerro \nTololo (CTIO): \n{ele_obs}m'

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)

path_ds_SH_RH_T = './sites/Cerro_Tololo/Data/in-situ/Specific_humidity_Cerro_Tololo_2002to2019.csv'
path_ds_PWV = 0
path_ds_clouds = './sites/Cerro_Tololo/Data/in-situ/daily_weather_loss_Cerro_Pachon_Gemini.csv'
site_name_folder = 'Cerro_Tololo'
ls_pr_levels_ERA5 =  [800] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [850]
time_slice_var_meteo = slice('2002-01-01','2018-12-31')
time_slice_var_PWV = 0
time_slice_var_clouds = slice('2013-01-01', '2017-09-30')
# seeing
ls_pr_level_seeing = [825]
time_slice_var_seeing = slice('2004-01-01', '2019-12-31')
path_ds_seeing = './sites/Cerro_Tololo/Data/in-situ/hourly_meteo/hourly_Cerro_Tololo_Seeing_dimm.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)
data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(781)

# %% La Palma

# latitude and longitude in 0-360
lower_lon = (360-18.5)
upper_lon = (360-17)
lower_lat = 28
upper_lat = 29.5

# longitude and latitude of observatory
lon_obs = 360-17.8851
lat_obs = 28.7572
ele_obs = 2382

# site name
site_name = 'La Palma (Spain)'
observatory_name = f'Nordic Optical Telescope (NOT): \n{ele_obs}m'

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)

path_ds_SH_RH_T = './sites/La_Palma/Data/in-situ/hourly_meteo/Specific_humidity_RH_T_La_Palma_1997to2020.csv'
path_ds_PWV = 0 # exclude la palma data due to unreliable calibration './sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_PWV.csv'
path_ds_clouds = './sites/La_Palma/Data/in-situ/hourly_meteo/daily_laPalma_downtime_fraction.csv'
site_name_folder = 'La_Palma'
ls_pr_levels_ERA5 = [800] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [850]
time_slice_var_meteo = slice('1998-01-01','2019-12-31')
time_slice_var_PWV = slice('2011-01-01','2016-12-31') # note: data ranges from 2011-03-04 to 2017-05-26
time_slice_var_clouds = slice('2012-01-01','2016-12-31')
# seeing
ls_pr_level_seeing = [975]
time_slice_var_seeing = slice('2008-01-01', '2019-12-31')
path_ds_seeing = './sites/La_Palma/Data/in-situ/hourly_meteo/hourly_La_Palma_Seeing.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)
data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)



data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(771)

# %% siding spring
# latitude and longitude in 0-360
lower_lon = 148.5
upper_lon = 150
lower_lat = -32
upper_lat = -30.5

# longitude and latitude of observatory
lon_obs = 149.067
lat_obs = -31.2754
ele_obs = 1134 # base
# ele_dome = 1184 https://www.aao.gov.au/about-us/anglo-australian-telescope

# site name
site_name = 'Siding Spring (Australia)'
observatory_name = f'Anglo-Australian Telescope (AAT): \n{ele_obs}m'

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)

path_ds_SH_RH_T = './sites/siding_spring/Data/in-situ/Specific_humidity_siding_spring_2003to2020.csv'
path_ds_PWV = 0
path_ds_clouds = './sites/siding_spring/Data/in-situ/AAT_weather_loss_percentage.csv'
time_slice_var_PWV = 0
site_name_folder = 'siding_spring'
ls_pr_levels_ERA5 = [900]
ls_pr_levels_clim_model = [925]
time_slice_var_meteo = slice('2004-01-01','2019-12-31')
time_slice_var_clouds = slice('1993-01-01', '2018-12-31', None) # adapt after ERA5 download has finished
list_of_insitu_vars = ['siding spring Specific Humidity']
# seeing
ls_pr_level_seeing = [950]
time_slice_var_seeing = slice('1993-01-01', '2019-12-31')
path_ds_seeing = './sites/siding_spring/Data/in-situ/Seeing_yearly_siding_spring.txt'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)

data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(892)



# %% Sutherland

# latitude and longitude in 0-360
lower_lon = 19.75
upper_lon = 22
lower_lat = -33.5
upper_lat = -31.75

# longitude and latitude of observatory
lon_obs = 20.8107
lat_obs = -32.376
ele_obs = 1798

# site name
site_name = 'Sutherland (South Africa)'
observatory_name = f'South African Large Telescope \n(SALT): {ele_obs}m'

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)

path_ds_SH_RH_T = './sites/Sutherland/Data/in-situ/hourly_meteo/Specific_humidity_Sutherland.csv'
path_ds_PWV = 0
time_slice_var_PWV = 0
path_ds_clouds = 0 # './sites/Sutherland/Data/in-situ/hourly_meteo/hourly_Sutherland_T_RH_P.csv'
site_name_folder = 'Sutherland'
ls_pr_levels_ERA5 =  [850] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [850]
time_slice_var_meteo = slice('2013-01-01','2019-12-31')
time_slice_var_clouds = 0 # time_slice_var_meteo
# seeing
ls_pr_level_seeing = [850]
time_slice_var_seeing = 0
path_ds_seeing = 0

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)

data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(826)




# %% SPM
# latitude and longitude in 0-360
lower_lon = 360-116.5
upper_lon = 360-114.5
lower_lat = 30
upper_lat = 32

# longitude and latitude of observatory
lon_obs = 360-115.46
lat_obs = 31.04
ele_obs = 2800

# site name
site_name = 'San Pedro Martir (Mexico)'
observatory_name = f'National Astronomical Observatory \n(OAN): {ele_obs}m'

# main_oro(upper_lon, lower_lon, lower_lat, upper_lat, lon_obs, lat_obs, site_name, observatory_name)

path_ds_SH_RH_T = './sites/SPM/Data/in-situ/hourly_meteo/Specific_humidity_SPM_2006to2020.csv'
path_ds_PWV = 0
time_slice_var_PWV = 0
path_ds_clouds = 0
site_name_folder = 'SPM'
ls_pr_levels_ERA5 =  [750] # code reads in automatically data for these levels
ls_pr_levels_clim_model = [700]
time_slice_var_meteo = slice('2007-01-01','2019-12-31')
# seeing
ls_pr_level_seeing = [850]
time_slice_var_seeing = slice('2005-01-01', '2008-12-31')
path_ds_seeing = './sites/SPM/Data/in-situ/hourly_meteo/Seeing_MASS_TMT_2005to2007.csv'

data['ls_pr_level_seeing'].append(ls_pr_level_seeing)
data['time_slice_var_seeing'].append(time_slice_var_seeing)
data['path_ds_seeing'].append(path_ds_seeing)

data['path_ds_SH_RH_T'].append(path_ds_SH_RH_T)
data['path_ds_PWV'].append(path_ds_PWV)
data['path_ds_clouds'].append(path_ds_clouds)
data['site_name_folder'].append(site_name_folder)
data['ls_pr_levels_ERA5'].append(ls_pr_levels_ERA5)
data['ls_pr_levels_clim_model'].append(ls_pr_levels_clim_model)
data['time_slice_var_meteo'].append(time_slice_var_meteo)
data['time_slice_var_PWV'].append(time_slice_var_PWV)
data['time_slice_var_clouds'].append(time_slice_var_clouds)


data['lower_lon'].append(lower_lon)
data['upper_lon'].append(upper_lon)
data['lower_lat'].append(lower_lat)
data['upper_lat'].append(upper_lat)

data['lon_obs'].append(lon_obs)
data['lat_obs'].append(lat_obs)
data['ele_obs'].append(ele_obs)

data['site_name'].append(site_name)
data['observatory_name'].append(observatory_name)

data['pressure [hPa]'].append(733) 


#%% save 'data' dict to pickle file (keep a dictionary, do not convert to dataframe)
import pickle
my_file = open("/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "wb")
pickle.dump(data, my_file)
my_file.close()

# load with: pickle.load( open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))
# %%
# save lat/lon and elevation for every observatory

# # write to dataframe
# df_site_lonlat_data = pd.DataFrame(data,columns = ['site_name', 'observatory_name','upper_lon', 'lower_lon',
#  'lower_lat', 'upper_lat', 'lon_obs', 'lat_obs','ele_obs', 'pressure [hPa]', 
#  'path_ds_SH_RH_T', 'path_ds_PWV', 'site_name_folder', 'ls_pr_levels_ERA5', 'ls_pr_levels_clim_model',
#     'time_slice_var_meteo'])

# # save as csv
# df_site_lonlat_data.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/Sites_lon_lat_ele_data.csv')


# %%
