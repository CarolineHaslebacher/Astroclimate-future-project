#%%
import os 
os.chdir('/home/haslebacher/chaldene/API')


# add these lines for making a new directory, if it not already exists:
# file.write("import os \n")
# file.write("pressure_level = '{}'\n".format(P))
# file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))


        # # skip files that are already downloaded
        # file.write("    if os.path.exists(filename_str) == True: \n")
        # file.write("        print('file already exists') \n")
        # file.write("        continue\n")

#%%
# where is single level download?

#%% write .py files for execution
# retrieve Temperature

# define pressure levels, path and area
#pressure_levels = [750, 775, 800, 825, 875, 900] # 700 and 850 are already downloaded
pressure_levels = [650]
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/T/pressure_levels/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'T' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("pressure_level = '{}'\n".format(P))
        # add these lines for making a new directory, if it not already exists:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        # # skip files that are already downloaded

        file.write("variable = 'temperature' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-22/-72.5/-32.5/-69' \n")
        file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    if os.path.exists(filename_str) == True: \n")
        file.write("        print('file already exists') \n")
        file.write("        continue\n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% Temperature: run files
import os
pressure_levels = [650]
for P in pressure_levels:
    filename = 'T' + str(P) +'.py'
    path = "/home/caroline/Master_Thesis_local/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% write .py files for execution
# retrieve Relative humidity

# define pressure levels, path and area
pressure_levels = [650] # rest is already downloaded
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/RH/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'RH' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("pressure_level = '{}'\n".format(P))
        # add these lines for making a new directory, if it not already exists:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("variable = 'relative_humidity' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-22/-72.5/-32.5/-69' \n")
        file.write("for i in range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    if os.path.exists(filename_str) == True: \n")
        file.write("        print('file already exists') \n")
        file.write("        continue\n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% RH: run files
import os
pressure_levels = [650] 
for P in pressure_levels:
    filename = 'RH' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% write .py files for execution
# retrieve specific humidity

# define pressure levels, path and area
pressure_levels = [300, 450, 500, 550, 600, 700, 850, 875]
 # rest is already downloaded

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/SH/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'SH' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("variable = 'specific_humidity' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-22/-72.5/-32.5/-69' \n")
        file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% SH: run files
for P in pressure_levels:
    filename = 'SH' + str(P) +'.py' # change filename only
    path = "/home/caroline/Master_Thesis_local/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% download for seeing parameters
# years 2000 - 2017
# 1 year, all pressure levels
# parameters: 'v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'

# write .py files for execution

# define pressure levels, path and area
pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 1000] 
# pressure_levels = [400]

path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/Paranal/Era5_data/seeing/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable =  ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'] \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-22/-72.5/-32.5/-69' \n")
        file.write("for i in range(1979, 1998): \n") # select years!
        file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_seeing_params_'+ pressure_level + 'hPa.nc' \n")
        # skip files that are already downloaded
        file.write("    if os.path.exists(filename_str) == True: \n")
        file.write("        print('file already exists') \n")
        file.write("        continue\n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")
 
#%% Seeing: run files
import os 
#pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 1000] 
pressure_levels = [400]
for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% potential temperature

# define pressure levels, path and area
pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350,400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 1000] 

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Era5_data/seeing/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'pot_Temp_' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable =  ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'] \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-22/-72.5/-32.5/-69' \n")
        file.write("for i in range(1989, 2000): \n") # select years!
        file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_potential_temperature_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")
 
#%% Seeing: run files
import os 
pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300,350,400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 1000] 

for P in pressure_levels:
    filename = 'pot_Temp_' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

# %% wind for seeing modeling
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
        ],
        'pressure_level': '200',
        'year': '2016',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        "area": '-22/-72.5/-32.5/-69',    # Subset or clip to an area, here to Europe. Specify as North/West/South/East in Geographic lat/long degrees. Southern latitudes and Western longitudes must be given as negative numbers.
        "grid": "0.25/0.25",
    },
    '/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/seeing/200hPa/Era5_wind_200hPa_2016.nc')

#%% u, v, temperature, geopotential on all pressure levels for 2017 January 10 (as in Osborn & Sarazin 2018)
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'day': '10',
        'product_type': 'reanalysis',
        'variable': [
            'v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'
        ],'cloud_base_height', 'high_cloud_cover', 'low_cloud_cover',
            'medium_cloud_cover', 'total_cloud_cover'
        'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ],
        'year': , '2015','2016', '2017',
        'month': '01',
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'format': 'netcdf',
        'area': [
            -22, -72.5, -32.5,
            -69,
        ],
    },
    '/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/seeing/Era5_seeing_test_2017-01-10.nc')

#%% cloud cover

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'cloud_base_height', 'high_cloud_cover', 'low_cloud_cover',
            'medium_cloud_cover', 'total_cloud_cover',
        ],
        'year': '2013',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            -22, -72.5, -32.5,
            -69,
        ],
    },
    'download.nc')

#%%
# cloud cover
# variables ['cloud_base_height', 'high_cloud_cover', 'low_cloud_cover','medium_cloud_cover', 'total_cloud_cover']
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/cloud_base_height/'"

filename = 'cloud_base_height.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'cloud_base_height' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # Chile grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/high_cloud_cover/'"
filename = 'high_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'high_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # Chile grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/low_cloud_cover/'"
filename = 'low_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'low_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # Chile grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/medium_cloud_cover/'"
filename = 'medium_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'medium_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # Chile grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/total_cloud_cover/'"
filename = 'total_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'total_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # Chile grid
    file.write("for i in range(1979, 2013): \n")
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")


#%% run single level files
import os

filename = 'cloud_base_height.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'high_cloud_cover.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'low_cloud_cover.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'medium_cloud_cover.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'total_cloud_cover.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)


#%% wind direction

#% u wind, single level (northward)

path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/10m_u_component_of_wind/'"

filename = '10m_u_component_of_wind.py'
with open(filename, 'w') as file:
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable =  '10m_u_component_of_wind' \n")
    file.write("path = {} \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # select area!
    file.write("for i in range(1979, 2021): \n") # select years!
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '_10m_u_component_of_wind.nc' \n")
    # skip files that are already downloaded
    file.write("    if os.path.exists(filename_str) == True: \n")
    file.write("        print('file already exists') \n")
    file.write("        continue\n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

#%%
# v wind, single level, eastward direction

path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/Paranal/Data/Era_5/10m_v_component_of_wind/'"

filename = '10m_v_component_of_wind.py'
with open(filename, 'w') as file:
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable =  '10m_v_component_of_wind' \n")
    file.write("path = {} \n".format(path))
    file.write("area = '-22/-72.5/-32.5/-69' \n") # select area!
    file.write("for i in range(1979, 2021): \n") # select years!
    file.write("    filename_str = path + 'Era5_Chile_' + str(i) + '10m_v_component_of_wind.nc' \n")
    # skip files that are already downloaded
    file.write("    if os.path.exists(filename_str) == True: \n")
    file.write("        print('file already exists') \n")
    file.write("        continue\n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

# %%
