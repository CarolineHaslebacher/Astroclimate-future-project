# this code reads in Era-5 and in-situ measurement data and plots the diurnal and the seasonal cycle and a long timeseries

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import netCDF4
import xarray as xr

import xarray.plot as xplt
#import sunpy.timeseries as ts 

########## for cycle ##############
from matplotlib import dates as d
import datetime as dt
import time

from itertools import cycle

from functools import reduce

#%%
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file
import Astroclimate_function_pool
import importlib
importlib.reload(Astroclimate_function_pool)

from Astroclimate_function_pool import netcdf_to_df
from Astroclimate_function_pool import  mes_prep
from Astroclimate_function_pool import  merge_df 
from Astroclimate_function_pool import  merge_df_long
from Astroclimate_function_pool import  df_prep #(df, parameter, colname)
from Astroclimate_function_pool import  plot_cycle #(cycle_name, cycle_string,  CFHT_parameter, filename, *args)
from Astroclimate_function_pool import  plot_timeseries_merged
from Astroclimate_function_pool import plot_timeseries_long
from Astroclimate_function_pool import plot_timeseries_movav
from Astroclimate_function_pool import correlation_plot


#%%
# change current working directory
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
os.getcwd()

# open netcdf total column water from singel level era 5 data
ds_TCW_single_level = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/TCW/*.nc', combine = 'by_coords')

# open netcdf surface pressure from singel level era 5 data
ds_surface_pressure = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/P/*.nc', combine = 'by_coords')

# open NETCDF files on 600hPa to 750hPa
#ds_SH_600 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/600hPa/*.nc', combine = 'by_coords')
#ds_SH_650 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/650hPa/*.nc', combine = 'by_coords')
#ds_SH_700 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/700hPa/*.nc', combine = 'by_coords')
#ds_SH_750 = xr.open_mfdataset('./sites/MaunaKea/Data/Era_5/SH/750hPa/*.nc', combine = 'by_coords')

#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
#CFHT_SH_hourly = pd.read_csv('./sites/MaunaKea/Data/in-situ/SH/Specific_humidity_CFHT_masked_2000to2019.csv')


#%%
# read in SH for all pressure levels in for loop
SH_pressure_levels = []
# in hPa
pr_levels = [20, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800]
for i in range(0,16):
    path = './sites/MaunaKea/Data/Era_5/SH/2010/' + 'Era5_2010_MK_SH_' + str(pr_levels[i]) + 'hPa.nc'
    ds = xr.open_dataset(path)
    ds_sel = ds.sel(longitude=-155.5,latitude= 19.75,method='nearest')
    SH_pressure_levels.append(ds_sel)


#%% sel(longitude=lon,latitude= lat,method='nearest')
ds_TCW_single_level_MK = ds_TCW_single_level.sel(longitude=-155.5,latitude= 19.75,method='nearest')

#print(ds_TCW_single_level_MK)
#air = ds_TCW_single_level_MK.tcw
#air.plot()

#%%
ds_surface_pressure_MK = ds_surface_pressure.sel(longitude=-155.5,latitude= 19.75,method='nearest')

# select time slice from 2010-01-01T00:00:00 to 2010-12-31T23:00:00 of 
ds_surface_pressure_MK_2010 = ds_surface_pressure_MK.sel(time=slice('2010-01-01T00:00:00', '2010-12-31T23:00:00'))
# ds_surface_pressure_MK.sp.xr.max()

#%%
# multiply every value of pr_levels by 100 to convert to Pa
pr_levels_Pa = list(map(lambda x: x * 100, pr_levels))

# define function 
absolute_difference_function = lambda list_value : abs(list_value - given_value)
PWV = []
PWV_time = []
#for k in range(0,len(ds_surface_pressure_MK_2010.sp)):
for k in range(0,len(ds_surface_pressure_MK_2010.sp)): # take 10 for testing purposes only!
    # in Pa
    get_surf_press = ds_surface_pressure_MK_2010.sp[k]
    # find closest value for the surface pressure at time k
    given_value = ds_surface_pressure_MK_2010.sp[k]
    closest_value = min(pr_levels_Pa, key=absolute_difference_function)
    index_of_closest_value = pr_levels_Pa.index(closest_value)
     
    sum = 0
    # integrate specific humidity until that closest_value
    for i in range(0, (index_of_closest_value + 1)):

        # integrate over specific humidity

        #delta_p
        if i == 0:
            delta_p = pr_levels_Pa[int(i+1)]  - pr_levels_Pa[i]
        elif i == (len(pr_levels_Pa) - 1): # highest index of pr_levels_Pa is 1 smaller than length
            delta_p = pr_levels_Pa[i] - pr_levels_Pa[int(i-1)]
        else:
            delta_p = 0.5 * (pr_levels_Pa[int(i+1)] - pr_levels_Pa[int(i-1)])
            # * 100 for transformation from milibar to Pa
            #print('delta_p2 is {}'.format(delta_p2))

            #print(delta_p, i, '\n')
            sum = sum + delta_p * SH_pressure_levels[i].q[k].values # k selects the time, i selects the level
    PWV.append(sum/9.81) # fill list with integral
    PWV_time.append(SH_pressure_levels[i].q[k].time.values)
    # PWV.set_index = SH_pressure_levels[i].q[k].time # fill list with time
    #print(sum/9.81) # PWV = 1/g * sum(delta_p * q)

#%%
# put integrated values into dataframe for plotting

column_names = ["SH integral"]
df_SH_integral = pd.DataFrame(PWV, index = PWV_time, columns = column_names) 

df_SH_integral.to_csv('./sites/MaunaKea/Data/Era_5/TCW/hourly_SH_integral_2010.csv')

#%%

# read produced csv
df_SH_integral = pd.read_csv('./sites/MaunaKea/Data/Era_5/TCW/hourly_SH_integral_2010.csv')
#df_SH_integral = df_SH_integral.rename(columns={'Unnamed: 0': 'time'})
df_SH_integral['Unnamed: 0'] = pd.to_datetime(df_SH_integral['Unnamed: 0']) 
df_SH_integral['Unnamed: 0'] = df_SH_integral['Unnamed: 0'] # it is in UTC
df_SH_integral.set_index('Unnamed: 0', inplace=True)

#%%
# plot integral values and TCW single level values
TCW_single_level = ds_TCW_single_level_MK.tcw
TCW_single_level.plot()

# %% read in in-situ PWV measurements
os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
# in-situ data generated with "read_PWV.py"
#df_insitu_2010 = pd.read_csv('./sites/MaunaKea/Data/in-situ/PWV/hourly_PWV_MK_JCMT.csv')
df_insitu_2010 = pd.read_csv('./sites/MaunaKea/Data/in-situ/PWV/hourly_PWV_MK_JCMT_30s.csv')

df_insitu_2010['UTC_time'] = pd.to_datetime(df_insitu_2010['UTC_time'])


# specific file:
# df_insitu = pd.read_csv('./sites/MaunaKea/Data/in-situ/PWV/20100304_csv.csv',header = None,
#delimiter=r"\s+", engine = "python")
#time is already in UTC -> no shifting needed
#df_insitu[0] = pd.to_datetime(df_insitu[0])

# %%
# prepare datasets for correlation plot: in-situ vs tcw single level
tcw_2010 = TCW_single_level.sel(time=slice('2010-01-01T00:00:00', '2010-12-31T23:00:00'))
df_tcw_2010 = tcw_2010.to_dataframe()

# rename UTC_time of df_insitu_2010 to time, set as index

df_insitu_2010 = df_insitu_2010.rename(columns={'UTC_time': 'time'})

# change the format of the times column to datetime format
df_insitu_2010['time'] = pd.to_datetime(df_insitu_2010['time']) 

# mask df_insitu to values below 20kg/m^2
mask1 = (df_insitu_2010['PWV'] <= 20)
df_insitu_2010['PWV'] = df_insitu_2010['PWV'][mask1]

#set index 
df_insitu_2010.set_index('time', inplace=True)

df_list = [df_tcw_2010, df_insitu_2010]
df_merged = reduce(lambda left, right: pd.merge(left, right, left_on='time', right_on='time', how='outer'), df_list)



#%%

# plot TCW from single level data
TCW_single_level = ds_TCW_single_level_MK.tcw
TCW_single_level.plot(label = 'tcw single level data', color = 'g')
# plot integrated TCW
#plt.plot(df_SH_integral.index, df_SH_integral['SH integral'], 
#label = 'SH integral', color = 'b', alpha = 0.5)

# in situ data directly from web
#plt.plot(df_insitu[0], (df_insitu[9]), color = 'k', label = 'in situ', alpha = 0.75)

# in situ data generated by "read_PWV.py"
plt.plot(df_insitu_2010.index, df_insitu_2010['PWV'], color = 'k',
 label = 'in-situ')

plt.ylabel('Total Column Water [kg m-2]')
plt.xlabel('time')
plt.xlim(dt.date(2009,12,1), dt.date(2011, 1, 31))
plt.ylim(0,30)
#plt.xlim(dt.date(2010, 3, 4), dt.date(2010, 3, 5))#for in situ zoomed
#plt.xlim(dt.date(2010, 3, 1), dt.date(2010, 3, 10)) # for in situ of 2010-03-04
plt.legend()
plt.savefig('./sites/MaunaKea/Output/Plots/TCW/SH_integral_tcw_insitu_corrected_2010_timeseries_2010.pdf')
plt.close

# #%%
# # correlation plot: integral vs. tcw single level
# x = np.arange(0,30)
# plt.scatter(TCW_single_level.sel(time=slice('2010-01-01T00:00:00', '2010-12-31T23:00:00')), 
# df_insitu_2010['PWV'], s = 1)
# plt.plot(x,x, color = 'k')
# plt.xlabel('TCW single level [kg m-2]')
# plt.ylabel('in-situ PWV [kg m-2]')
# plt.title('correlation plot of Total column water (2010), Mauna Kea')
# plt.savefig('./sites/MaunaKea/Output/Plots/TCW/in-situ_tcw_2010_correlation_plot.pdf')
# plt.close


#%%
# correlation plot: integral vs. tcw single level
x = np.arange(0,30)
plt.scatter(TCW_single_level.sel(time=slice('2010-01-01T00:00:00', '2010-12-31T23:00:00')), 
df_SH_integral['SH integral'], s = 1)
plt.plot(x,x, color = 'k')
plt.xlabel('TCW single level [kg m-2]')
plt.ylabel('SH integral [kg m-2]')
plt.title('correlation plot of Total column water (2010), Mauna Kea')
plt.savefig('./sites/MaunaKea/Output/Plots/TCW/SH_integral_2010_correlation_plot.pdf')
plt.close

#df_SH_integral.resample('d')

#%%
# correlation plot: in-situ vs tcw single level
x = np.arange(0,20)
plt.scatter(df_merged['tcw'], df_merged['PWV'], s = 1)
plt.plot(x,x, color = 'k')
plt.xlabel('TCW single level [kg m-2]')
plt.ylabel('in-situ [kg m-2]')
plt.text(2.5, 2, 'hello world', bbox=dict(facecolor='white', edgecolor='red', alpha = 0.7))
plt.title('correlation plot of Total column water (2010), in-situ, Mauna Kea')
plt.savefig('./sites/MaunaKea/Output/Plots/TCW/in-situ_PWV_2010_correlation_plot_insitu_30cutoff.pdf')
plt.close

# %% use function "correlation_plot"

correlation_plot('./sites/MaunaKea/Output/Plots/TCW/PWV_2010_correlation_plot_insitu.pdf',
'correlation plot of Total column water (2010), in-situ, Mauna Kea',
df_merged['tcw'],  'TCW single level [kg m-2]',
df_merged['PWV'], 'in-situ [kg m-2]')

# %%
