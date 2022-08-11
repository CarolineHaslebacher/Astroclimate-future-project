
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import netCDF4
import xarray as xr
import os
import datetime # here, something is not quite right

#%%
# read in data of insitu measurements on Siding_Spring from csv to pandas dataframe
# years 2007 - 2019

# os.chdir('/home/haslebacher/chaldene/Astroclimate_Project')
# pd.read_table('./sites/siding_spring/Data/in-situ/aat_metdata/2018/met_20180101.dat', 
# header = None, skipinitialspace = True, nrows=8)

# pd.read_table('./sites/siding_spring/Data/in-situ/aat_metdata/2018/met_20180101.dat', 
# header = 7, skipinitialspace = True, sep='(\\#)\s+')

#%%

# # append dataframe for every year
# path = './sites/siding_spring/Data/in-situ/aat_metdata/2003/' + 'D11261.2003'
# # find header, remove # 
# with open(path,"r") as fi:
#     idx = 0
#     for ln in fi:
#         if ln.startswith("# "):
#             idx = idx + 1
#             my_header = str.split(ln[2:], ' ')
#         elif ln.startswith('"'):
#             idx = idx + 1
#             my_header = str.split(ln[1:], '\t')            
#         else:
#             break
# my_header = list(filter(None, my_header))
# print(my_header)

# df_2003 = pd.read_table(path, 
#             header = None, names = my_header, skiprows = idx, skipinitialspace = True, sep=' ')

# assign correct date (given by filename)

# append dataframe of year in path

#%% code for year 2003 to 2008 (there, a change in filenames occur)
df_2003to08 = pd.DataFrame()
# append dataframe for every year

for yr in range(2003, 2009):
    for mth in range(1,13):
        for dy in range(1,32):

            filename = 'D'+ str(mth).zfill(2) + str(dy).zfill(2) + '1.' + str(yr)
            path = './sites/siding_spring/Data/in-situ/aat_metdata/' + str(yr) + '/' + filename
             
            if os.path.exists(path) == False:
                # print('file {} does not exist'.format(filename))
                continue  
            try:      
                # find header, remove # 
                with open(path,"r") as fi:
                    idx = 0
                    for ln in fi:
                        if ln.startswith('"'):
                            idx = idx + 1
                            my_header = str.split(str.replace(str.replace(ln[1:], '"', ''), ' ', ''), '\t')            
                        else:
                            break
                my_header = list(filter(None, my_header))

                # read file, starting from row after header
                df = pd.read_table(path, 
                            header = None, names = my_header, skiprows = idx, skipinitialspace = True,
                             sep='\t', comment = '#')
                

                # IS TIME IN UT OR UT + 10 ? -> assume it is UT + 10 and find out with diurnal cycle
                                
                # create column with date, format as index
                # assume that the first column is always the time
                df['date'] = pd.to_datetime(datetime.date(yr, mth, dy))- pd.to_timedelta(dt.timedelta(hours=10)) + pd.to_timedelta(df[df.columns[0]]) 
                df.set_index('date', inplace=True)

                # append the dataframe to the 
                df_2003to08 = df_2003to08.append(df)

            except KeyError:
                print('{} caused problems'.format(filename))

#%% zwischenzelle

#%% code for years 2008 to 2020 (new filename)
df_2008to20 = pd.DataFrame()
# append dataframe for every year

for yr in range(2008, 2021):
    for mth in range(1,13):
        for dy in range(1,32):

            filename = 'met_'+ str(yr) + str(mth).zfill(2) + str(dy).zfill(2) + '.dat'
            path = './sites/siding_spring/Data/in-situ/aat_metdata/' + str(yr) + '/' + filename
            
            if os.path.exists(path) == False:
                # print('file {} does not exist'.format(filename))
                continue    
            try:    
                # find header, remove # 
                with open(path,"r") as fi:
                    idx = 0
                    for ln in fi:
                        if ln.startswith('#'):
                            idx = idx + 1
                            my_header = str.split(str.replace(ln[2:], '_O', '') , ' ')            
                        else:
                            break
                my_header = list(filter(None, my_header))

                # read file, starting from row after header
                df = pd.read_table(path, 
                            header = None, names = my_header, skiprows = idx, skipinitialspace = True, 
                            sep=' ')
                

                # IS TIME IN UT OR UT + 10 ?

                # create column with date, format as index
                # assume that the first column is always the time
                df['date'] = pd.to_datetime(datetime.date(yr, mth, dy))- pd.to_timedelta(dt.timedelta(hours=10)) + pd.to_timedelta(df[df.columns[0]]) 
                df.set_index('date', inplace=True)

                # append the dataframe to the 
                df_2008to20 = df_2008to20.append(df)

            except KeyError:
                print('{} caused problems'.format(filename))

#%% quickly save to csv and check if it looks fine

df_2003to08.to_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/Siding_spring_raw_df_2003to2008.csv', header = True, index = True)
df_2008to20.to_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/Siding_spring_raw_df_2008to2020.csv', header = True, index = True)     


#%% read csv, so you do not need to process above cells!
###################################### YOU CAN START FROM HERE #############################################################

df_2003to08 = pd.read_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/Siding_spring_raw_df_2003to2008.csv', index_col='date', parse_dates=True)
df_2008to20 = pd.read_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/Siding_spring_raw_df_2008to2020.csv', index_col='date', parse_dates=True)     

#%% rename columns
df_2003to08_renamed = df_2003to08.rename(columns={'OutT': 'siding spring Temperature', # take 'OutT' as the potentially correct temperature
                            'Hum%': 'siding spring Relative Humidity', 
                            'Bar.': 'siding spring Pressure'})
df_2008to20_renamed = df_2008to20.rename(columns={'Out_T': 'siding spring Temperature', # take 'OutT' as the potentially correct temperature
                            'Hum%': 'siding spring Relative Humidity', 
                            'Bar_mm': 'siding spring Pressure'})

#%%
# extract useful data (Relative humidity RH, Temperature T, Pressure)
df_2003to08_extracted = df_2003to08_renamed[['siding spring Temperature', 'siding spring Relative Humidity', 'siding spring Pressure']]
df_2008to20_extracted = df_2008to20_renamed[['siding spring Temperature', 'siding spring Relative Humidity', 'siding spring Pressure']]

#%% concatenate dataframes
df_2003to2020 = pd.concat([df_2003to08_extracted, df_2008to20_extracted])

#%%
# rename 'date' to 'time
df_2003to2020.index = df_2003to2020.index.rename('time')

# already in pd.to_csv
# df_2003to2020.index = pd.to_datetime(df_2003to2020.index)
#%%
# mmHg to hPa
# 1 mmHg = 1.33322 hPa

df_2003to2020['siding spring Pressure'] = 1.33322 * df_2003to2020['siding spring Pressure'] 


#%% filter out measurement errors

parameter = 'siding spring Temperature'
mask_T = ((df_2003to2020[parameter] != 0) & (df_2003to2020[parameter] > -20) & (df_2003to2020[parameter] < 50))
df_2003to2020 = df_2003to2020[mask_T]

parameter = 'siding spring Relative Humidity'  
mask_RH = (df_2003to2020[parameter] <= 100) & (df_2003to2020[parameter] >= 0) & (df_2003to2020[parameter] != 'nan')
df_2003to2020 = df_2003to2020[mask_RH]

parameter = 'siding spring Pressure'
mymean = np.mean(df_2003to2020[parameter])
mask_P = (df_2003to2020[parameter] <= (mymean + 20)) & (df_2003to2020[parameter] >= (mymean - 20)) 
df_2003to2020 = df_2003to2020[mask_P]

# %%
# resample hourly
df_2003to2020_resampled = df_2003to2020.resample('h').mean()

#%% 
# save to csv
df_2003to2020_resampled.to_csv('./sites/siding_spring/Data/in-situ/hourly_meteo/hourly_Siding_Spring_T_RH_P_time.csv', header = True, index = True)


# %%
 