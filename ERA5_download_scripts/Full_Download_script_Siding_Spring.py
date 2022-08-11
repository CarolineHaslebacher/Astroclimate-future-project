#%%
import os 
os.chdir('/home/haslebacher/chaldene/API/')
# add these lines for making a new directory, if it not already exists:
# file.write("import os \n")
# file.write("pressure_level = '{}'\n".format(P))
# file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))

#%% single level retrieve
 
# 2m-temperature
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/T/single_levels/'"

filename = 'T_2m.py'
with open(filename, 'w') as file:
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = '2m_temperature' \n")
    file.write("path = {} \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n")
    file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

# total column water
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/TCW/'"

filename = 'tcw.py'
with open(filename, 'w') as file:
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'total_column_water' \n")
    file.write("path = {} \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n")
    file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

# surface_pressure
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/surface_pressure/'"

filename = 'surface_pressure.py'
with open(filename, 'w') as file:
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'surface_pressure' \n")
    file.write("path = {} \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n")
    file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

#%% run single level files
import os
filename = 'T_2m.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'tcw.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

filename = 'surface_pressure.py'
path = "/home/caroline/chaldene/API/" + filename
command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
os.system(command)

#%% write .py files for execution
# retrieve Temperature

# define pressure levels, path and area
# pressure_levels = [825,850, 875, 900]
pressure_levels = [925 ,950, 975, 1000]
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/T/pressure_levels/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'T' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable = 'temperature' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-29.75/148/-32.5/150.5' \n")
        file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% Temperature: run files
import os
# pressure_levels = [825,850, 875, 900]
pressure_levels = [925 ,950, 975, 1000]
for P in pressure_levels:
    filename = 'T' + str(P) +'.py'
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% write .py files for execution
# retrieve Relative humidity

# define pressure levels, path and area
# pressure_levels = [825,850, 875, 900]
pressure_levels = [925 ,950, 975, 1000]
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/RH/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'RH' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable = 'relative_humidity' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-29.75/148/-32.5/150.5' \n")
        file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% RH: run files
import os
# pressure_levels = [825,850, 875, 900]
pressure_levels = [925 ,950, 975, 1000]
for P in pressure_levels:
    filename = 'RH' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%% write .py files for execution
# retrieve specific humidity

# define pressure levels, path and area
# pressure_levels = [825,850, 875, 900]
# pressure_levels = [925 ,950, 975, 1000]
pressure_levels = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800] # for tcw
 # rest is already downloaded
# path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/SH/' +  pressure_level + 'hPa/'"
path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/SH/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'SH' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable = 'specific_humidity' \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-29.75/148/-32.5/150.5' \n")
        file.write("for i in range(1979, 2021):#range(1979, 2021): \n")
        file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '_'+ pressure_level + 'hPa.nc' \n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")

#%% SH: run files
import os
# pressure_levels = [825,850, 875, 900]
pressure_levels = [925 ,950, 975, 1000]
for P in pressure_levels:
    filename = 'SH' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)



#%% download for seeing parameters

# 1 year, all pressure levels
# parameters: 'v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'

# write .py files for execution

# define pressure levels
#pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000] 
pressure_levels = [925, 950, 975, 1000]
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/seeing/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable =  ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'] \n")
        file.write("path = {} \n".format(path))
        file.write("area = '-29.75/148/-32.5/150.5' \n") # select area!
        file.write("for i in range(1979, 2021): \n") # select years!
        file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_seeing_params_'+ pressure_level + 'hPa.nc' \n")
        # skip files that are already downloaded
        file.write("    if os.path.exists(filename_str) == True: \n")
        file.write("        print('file already exists') \n")
        file.write("        continue\n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")
 

#%% Seeing: run files
import os 
pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000] 

for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)

#%%
# cloud cover
# variables ['cloud_base_height', 'high_cloud_cover', 'low_cloud_cover','medium_cloud_cover', 'total_cloud_cover']
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/cloud_base_height/'"

filename = 'cloud_base_height.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'cloud_base_height' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n") # siding_spring
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/high_cloud_cover/'"
filename = 'high_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'high_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n") # siding_spring grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/low_cloud_cover/'"
filename = 'low_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'low_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n") # siding_spring grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/medium_cloud_cover/'"
filename = 'medium_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'medium_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n") # siding_spring grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/siding_spring/Data/Era_5/total_cloud_cover/'"
filename = 'total_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'total_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '-29.75/148/-32.5/150.5' \n") # siding_spring grid
    file.write("for i in range(2018, 2021): \n")
    file.write("    filename_str = path + 'Era5_siding_spring_' + str(i) + '_' + variable + '.nc' \n")
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


