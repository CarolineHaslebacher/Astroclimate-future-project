from api_test import MK_retrieve_2008to2020
from api_test import MK_retrieve_1996to2007
from api_test import MK_retrieve_1984to1995
from api_test import MK_retrieve_2010
from api_test import MK_retrieve_single_level_2008to2020
from api_test import MK_retrieve_single_level_1996to2007
from api_test import MK_retrieve_single_level_1984to1995
  
# specific humidity
# 800 hPa to 250 hPa, only 1 year

MK_retrieve_2010('Era5_2010_MK_SH_800hPa.nc', 'specific_humidity',  '800')
MK_retrieve_2010('Era5_2010_MK_SH_775hPa.nc', 'specific_humidity',  '775')
MK_retrieve_2010('Era5_2010_MK_SH_750hPa.nc', 'specific_humidity',  '750')
MK_retrieve_2010('Era5_2010_MK_SH_700hPa.nc', 'specific_humidity',  '700')
MK_retrieve_2010('Era5_2010_MK_SH_650hPa.nc', 'specific_humidity',  '650')
MK_retrieve_2010('Era5_2010_MK_SH_600hPa.nc', 'specific_humidity',  '600')
MK_retrieve_2010('Era5_2010_MK_SH_550hPa.nc', 'specific_humidity',  '550')
MK_retrieve_2010('Era5_2010_MK_SH_500hPa.nc', 'specific_humidity',  '500')
MK_retrieve_2010('Era5_2010_MK_SH_450hPa.nc', 'specific_humidity',  '450')
MK_retrieve_2010('Era5_2010_MK_SH_400hPa.nc', 'specific_humidity',  '400')
MK_retrieve_2010('Era5_2010_MK_SH_350hPa.nc', 'specific_humidity',  '350')
MK_retrieve_2010('Era5_2010_MK_SH_300hPa.nc', 'specific_humidity',  '300')
MK_retrieve_2010('Era5_2010_MK_SH_250hPa.nc', 'specific_humidity',  '250')
#MK_retrieve_2010('Era5_2010_MK_SH_225hPa.nc', 'specific_humidity',  '225')
MK_retrieve_2010('Era5_2010_MK_SH_200hPa.nc', 'specific_humidity',  '200')
#MK_retrieve_2010('Era5_2010_MK_SH_175hPa.nc', 'specific_humidity',  '175')
#MK_retrieve_2010('Era5_2010_MK_SH_150hPa.nc', 'specific_humidity',  '150')
#MK_retrieve_2010('Era5_2010_MK_SH_125hPa.nc', 'specific_humidity',  '125')
MK_retrieve_2010('Era5_2010_MK_SH_100hPa.nc', 'specific_humidity',  '100')
MK_retrieve_2010('Era5_2010_MK_SH_20hPa.nc', 'specific_humidity',  '20')



# download'total_column_water'

MK_retrieve_single_level_2008to2020('Era5_2008to2020_MK_tcw.nc', 'total_column_water')
MK_retrieve_single_level_1996to2007('Era5_1996to2007_MK_tcw.nc', 'total_column_water')
MK_retrieve_single_level_1984to1995('Era5_1984to1995_MK_tcw.nc', 'total_column_water')

# download two meter temperature, '2m_temperature'

MK_retrieve_single_level_2008to2020('Era5_2008to2020_MK_t2m.nc', '2m_temperature')
MK_retrieve_single_level_1996to2007('Era5_1996to2007_MK_t2m.nc', '2m_temperature')
MK_retrieve_single_level_1984to1995('Era5_1984to1995_MK_t2m.nc', '2m_temperature')




# download temperature

# 600 hPa level
MK_retrieve_2008to2020('Era5_2008to2020_MK_T_600hPa.nc', 'temperature',  '600')
MK_retrieve_1996to2007('Era5_2008to2020_MK_T_600hPa.nc', 'temperature',  '600')
MK_retrieve_1984to1995('Era5_2008to2020_MK_T_600hPa.nc', 'temperature',  '600')

# 650 hPa level
#MK_retrieve_2008to2020('Era5_2008to2020_MK_T_650hPa.nc', 'temperature',  '650')
MK_retrieve_1996to2007('Era5_2008to2020_MK_T_650hPa.nc', 'temperature',  '650')
MK_retrieve_1984to1995('Era5_2008to2020_MK_T_650hPa.nc', 'temperature',  '650')

# 700 hPa level
MK_retrieve_2008to2020('Era5_2008to2020_MK_T_700hPa.nc', 'temperature',  '700')
MK_retrieve_1996to2007('Era5_2008to2020_MK_T_700hPa.nc', 'temperature',  '700')
MK_retrieve_1984to1995('Era5_2008to2020_MK_T_700hPa.nc', 'temperature',  '700')

# 750 hPa level
MK_retrieve_2008to2020('Era5_2008to2020_MK_T_750hPa.nc', 'temperature',  '750')
MK_retrieve_1996to2007('Era5_2008to2020_MK_T_750hPa.nc', 'temperature',  '750')
MK_retrieve_1984to1995('Era5_2008to2020_MK_T_750hPa.nc', 'temperature',  '750')

# download Surface pressure, 'surface_pressure', 
MK_retrieve_single_level_2008to2020('Era5_2008to2020_MK_sp.nc', 'surface_pressure')
MK_retrieve_single_level_1996to2007('Era5_1996to2007_MK_sp.nc', 'surface_pressure')
MK_retrieve_single_level_1984to1995('Era5_1984to1995_MK_sp.nc', 'surface_pressure')


#%% download for seeing parameters
# years 2000 - 2017
# 1 year, all pressure levels
# parameters: 'v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'

os.chdir('/home/haslebacher/chaldene/API')

# write .py files for execution

# define pressure levels, path and area
#pressure_levels = [50, 100, 125, 150, 175, 200, 225, 250, 300, 350,400,  450, 500, 550, 600, 650,  700, 750, 775, 800, 825, 850] 
pressure_levels = [400]
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/seeing/' +  pressure_level + 'hPa/'"

for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py'
    with open(filename, 'w') as file:
        file.write("import os \n")
        file.write("pressure_level = '{}'\n".format(P))
        file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
        file.write("from cds_api_era5 import Era5_pressure_level_retrieve \n")
        file.write("variable =  ['v_component_of_wind', 'u_component_of_wind', 'temperature', 'geopotential'] \n")
        file.write("path = {} \n".format(path))
        file.write("area = '20.5/-156/18.5/-154.5' \n") # select area
        file.write("for i in range(1979, 2021): \n") # select years!
        file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_seeing_params_'+ pressure_level + 'hPa.nc' \n")
        # skip files that are already downloaded
        file.write("    if os.path.exists(filename_str) == True: \n")
        file.write("        print('file already exists') \n")
        file.write("        continue\n")
        file.write("    print(filename_str) \n")
        file.write("    Era5_pressure_level_retrieve(str(i), variable, pressure_level, area , filename_str) \n")
 
#%% Seeing: run files
import os 

# pressure_levels = [50, 100, 125, 150, 225, 250, 300,350, 450,  550, 600,  700, 750, 775,  825, 850] # 650, 800, 500, 200, 175,
pressure_levels = [400]

for P in pressure_levels:
    filename = 'seeing_' + str(P) +'.py' # change filename only
    path = "/home/caroline/chaldene/API/" + filename
    command = "gnome-terminal -e 'bash -c \"python3 " + path + "; exec bash\"'"
    os.system(command)


#%%
# cloud cover
# variables ['cloud_base_height', 'high_cloud_cover', 'low_cloud_cover','medium_cloud_cover', 'total_cloud_cover']
path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/cloud_base_height/'"

filename = 'cloud_base_height.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'cloud_base_height' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '20.5/-156/18.5/-154.5' \n") # MaunaKea
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/high_cloud_cover/'"
filename = 'high_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'high_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '20.5/-156/18.5/-154.5' \n") # MaunaKea grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/low_cloud_cover/'"
filename = 'low_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'low_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '20.5/-156/18.5/-154.5' \n") # MaunaKea grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/medium_cloud_cover/'"
filename = 'medium_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'medium_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '20.5/-156/18.5/-154.5' \n") # MaunaKea grid
    file.write("for i in range(2013, 2018): \n")
    file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_' + variable + '.nc' \n")
    file.write("    print(filename_str) \n")
    file.write("    Era5_single_level_retrieve(str(i), variable, area , filename_str) \n")

path = "'/home/caroline/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/total_cloud_cover/'"
# screen ssh
# /home/haslebacher/chaldene/API/total_cloud_cover.py
path = "'/home/haslebacher/chaldene/Astroclimate_Project/sites/MaunaKea/Data/Era_5/total_cloud_cover/'"

filename = 'total_cloud_cover.py'
with open(filename, 'w') as file:
    file.write("from cds_api_era5 import Era5_single_level_retrieve \n")
    file.write("variable = 'total_cloud_cover' \n")
    file.write("path = {} \n".format(path))
    file.write("import os \n")
    file.write("os.makedirs(os.path.dirname({}), exist_ok=True) \n".format(path))
    file.write("area = '20.5/-156/18.5/-154.5' \n") # MaunaKea grid
    file.write("for i in range(1979, 2021): \n")
    file.write("    filename_str = path + 'Era5_MaunaKea_' + str(i) + '_' + variable + '.nc' \n")
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
