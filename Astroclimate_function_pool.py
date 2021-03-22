# These functions are defined: 
# netcdf_to_df(ds, lon, lat), 
# mes_prep(insitu_hourly, parameter = None),
# merge_df(*args)
# df_prep(df, parameter, colname)
# plot_cycle(cycle_name, cycle_string,  insitu_parameter, filename, *args)
# plot_timeseries_merged(filename,merged_df, monthly_grouped_P, yearly_grouped_P, *args)

# %% 
 
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
from scipy import stats

import csv

# %%
# define function for converting dataset to time series

def netcdf_to_df(ds, lon, lat):
    #ds_MaunaKea=  ds.sel(longitude=204.53,latitude= 19.83,method='nearest')
    if 'expver' in ds.coords:
        # drop 'expver' --> YES! THAT WAS THE PROBLEM!
        # expver 1 and 5, explanation here: https://confluence.ecmwf.int/display/CUSF/ERA5+CDS+requests+which+return+a+mixture+of+ERA5+and+ERA5T+data
        # in short: 1 is for values up to 1 month in the past, 5 includes all values
        # drop 1, keep 5
        ds_sel =  ds.sel(expver = 5, longitude=lon, latitude= lat, method='nearest')
    else:
        ds_sel =  ds.sel(longitude=lon, latitude= lat, method='nearest')
    
    df = ds_sel.to_dask_dataframe()
    return(df)


# %%
# prepare insitu observational data for merging
def mes_prep(insitu_hourly, parameter = None, timezone = None):
    #insitu_hourly = insitu_hourly.rename(columns={'Unnamed: 0': 'time'})

    # change the format of the time column to datetime format
    # if not already the index
    if 'time' in insitu_hourly.columns:
        insitu_hourly['time'] = pd.to_datetime(insitu_hourly['time']) 

    #check the format
    # print(insitu_hourly['time'].dtype)
    #print(insitu_hourly['time'][0])

    #from HST to UTC (+10 hours)
    if timezone == 'HST':
        insitu_hourly['time'] = insitu_hourly['time'] + dt.timedelta(hours=10)
    
    if 'time' in insitu_hourly.columns:
        #set index 
        insitu_hourly.set_index('time', inplace=True)
    
    if ((parameter == 'temperature(C)') | (parameter == 'T')):
        #filter out values exactly equal to 0, smaller than -100°C and greater than 30°C (years 1991 and 1992)
        mask_T = (insitu_hourly[parameter] != 0) & (insitu_hourly[parameter] > -30) & (insitu_hourly[parameter] < 50)
        insitu_hourly = insitu_hourly[mask_T]
        
    elif ((parameter == 'relative_humidity(%)') | (parameter == 'RH') | ('Relative' in str(parameter))):
        mask_RH = (insitu_hourly[parameter] <= 100) & (insitu_hourly[parameter] >= 0) & (insitu_hourly[parameter] != 'nan')
        insitu_hourly = insitu_hourly[mask_RH]
    
    elif ((parameter == 'pressure (mb)') | ('Pressure' in str(parameter))):
        mymean = np.mean(insitu_hourly[parameter])
        mask_P = (insitu_hourly[parameter] <= (mymean + 20)) & (insitu_hourly[parameter] >= (mymean - 20)) 
        insitu_hourly = insitu_hourly[mask_P]
    
    else:
        print('no parameter found; nothing got masked')
    
    # create a new column consisting of the cycle parameter of the correspondend entry
    #for seasonal cycle (12 months), create column with "months"
    insitu_hourly['months'] = pd.DatetimeIndex(insitu_hourly.index).month                                            

    #for diurnal cycle (24 hours), create column with "hours"
    insitu_hourly['hours'] = pd.DatetimeIndex(insitu_hourly.index).hour

    # for monthly timeseries, create column with "YYYY-MM"
    insitu_hourly['YYYY_MM'] = insitu_hourly.index.strftime('%Y-%m')

    # for 1 year averages, create column with years
    insitu_hourly['years'] = pd.DatetimeIndex(insitu_hourly.index).year
    
    return(insitu_hourly)


# %%
# merge datasets to only compare equal rows of data
# if you want to drop nan's, set NAN to True
def merge_df(*args, dropnan = False):
    df_list = []
    for ar in args: 
        df_list.append(ar)
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_on='time', right_on='time', how='outer'), df_list)

    # delete rows containing NaN
    if dropnan == True:
        df_merged = df_merged.dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

    # reduce to seasonal cycle
    seasonal_cycle = df_merged.groupby('months').describe()

    # reduce to seasonal cycle
    diurnal_cycle = df_merged.groupby('hours').describe()
    
    # for monthly timeseries, groupby YYYY-MM
    monthly_grouped = df_merged.groupby('YYYY_MM').describe()

    # for yearly timeseries, group by 'years'
    yearly_grouped = df_merged.groupby('years').describe()
    yearly_grouped.index = yearly_grouped.index.astype(int)
    
    return df_merged, seasonal_cycle, diurnal_cycle, monthly_grouped, yearly_grouped


# %%
def df_prep(df, parameter, colname):
    # df = df.drop('longitude', axis=1)
    # df= df.drop('latitude', axis=1)
    df = df[['time', parameter]]
    df = df.set_index('time')
    df = df.rename(columns={parameter: colname})#colname = 'r_700'
    df_comp = df.compute()
    
    return(df_comp)


# %%
# plot seasonal cycle observational data and also show quartiles
from itertools import cycle


#insitu_parameter = 'temperature(C)'
# key = 't_600'
def plot_cycle(cycle_name, cycle_string,  insitu_parameter, filename, *args, insitu_parameter_2 = None, insitu_parameter_3 = None, insitu_parameter_4 = None):
    cycol = cycle('rgbcm')

    # plot model parameters first
    for ar in args: 
        my_col =next(cycol)
        plt.plot(cycle_name.index, cycle_name[ar]['mean'], '-o', c= my_col, linewidth=1.0, 
            label = 'Era5 ' + ar + r' $\pm$ std. dev.', alpha = 0.8)

        plt.fill_between(cycle_name.index, (cycle_name[ar]['mean'] - cycle_name[ar]['std']), (cycle_name[ar]['mean'] + cycle_name[ar]['std']), 
                         alpha=.25, facecolor=my_col) #, label = 'Era5 ' + ar + ' std.')
        #plt.fill_between(cycle_name.index, cycle_name[ar]['mean'], cycle_name[ar]['25%'], 
                         #alpha=.25, facecolor=my_col)

    plt.plot(cycle_name.index, cycle_name[insitu_parameter]['mean'], '-ko', linewidth=1.0, 
            label = 'in-situ ' + insitu_parameter + r' $\pm$ std. dev.', alpha = 0.8)

    plt.fill_between(cycle_name.index, (cycle_name[insitu_parameter]['mean'] - cycle_name[insitu_parameter]['std']), (cycle_name[insitu_parameter]['mean'] + cycle_name[insitu_parameter]['std']),
                     alpha=.25, facecolor='k') #, label = 'in-situ std.')
    #plt.fill_between(cycle_name.index, cycle_name[insitu_parameter]['mean'], (cycle_name[insitu_parameter]['mean'] - cycle_name[insitu_parameter]['std']), 
                     #alpha=.25, facecolor='k')
    
    # if there is a second, third or even fourth in-situ parameter which is supposed to be plotted in black
    if insitu_parameter_2 != None:
        plt.plot(cycle_name.index, cycle_name[insitu_parameter_2]['mean'], '-k*', linewidth=1.0, 
            label = 'in-situ ' + insitu_parameter_2 + r' $\pm$ std. dev.', alpha = 0.8)

        plt.fill_between(cycle_name.index, (cycle_name[insitu_parameter_2]['mean'] - cycle_name[insitu_parameter_2]['std']), (cycle_name[insitu_parameter_2]['mean'] + cycle_name[insitu_parameter_2]['std']),
                     alpha=.15, facecolor='k') #, label = 'in-situ std.')

    if insitu_parameter_3 != None:
        plt.plot(cycle_name.index, cycle_name[insitu_parameter_3]['mean'], '-kd', linewidth=1.0, 
            label = 'in-situ ' + insitu_parameter_3 + r' $\pm$ std. dev.', alpha = 0.8)
        plt.fill_between(cycle_name.index, (cycle_name[insitu_parameter_3]['mean'] - cycle_name[insitu_parameter_3]['std']), (cycle_name[insitu_parameter_3]['mean'] + cycle_name[insitu_parameter_3]['std']),
                     alpha=.15, facecolor='k') #, label = 'in-situ std.')

    if insitu_parameter_4 != None:
        plt.plot(cycle_name.index, cycle_name[insitu_parameter_4]['mean'], '-kp', linewidth=1.0, 
            label = 'in-situ ' + insitu_parameter_4 + r' $\pm$ std. dev.', alpha = 0.8)
        plt.fill_between(cycle_name.index, (cycle_name[insitu_parameter_4]['mean'] - cycle_name[insitu_parameter_4]['std']), (cycle_name[insitu_parameter_4]['mean'] + cycle_name[insitu_parameter_4]['std']),
                     alpha=.15, facecolor='k') #, label = 'in-situ std.')

    #plt.plot(diurnal_cycle.index, diurnal_cycle['r_750']['mean'], '-go', linewidth=2.0, 
    #        label = 'Era5, 750hPa')
    plt.legend()
    #plt.ylim(0, 100)
    ax = plt.subplot(111)
    

    if 'diurnal cycle' in cycle_string:
        plt.xticks(np.arange(0, 25, step=1))
        plt.setp(ax.get_xticklabels(), rotation=50, horizontalalignment='right')
        plt.xlabel('time [h]')
        plt.xlim(0, 24)

    elif 'seasonal cycle' in cycle_string:
        plt.xlabel('time [months]')
        plt.xticks(np.arange(0, 13, step=1))
        plt.xlim(1, 12)
        
        
    if '/RH/' in filename:
        plt.ylabel('Relative humidity [%]')
    
    elif '/SH/' in filename:
        plt.ylabel('Specific humidity [kg/kg]')
    #    plt.ylim(0,0.05)
    
    elif '/T/' in filename:
    #    plt.ylim(0,20)
        plt.ylabel('Temperature [°C]')
 
    elif '/P/' in filename:
        plt.ylabel('Pressure [hPa]')

    elif '/TCW/' in filename:
        plt.ylabel(r'Precipitable water vapor [$kg/m^2$]')
    
    plt.title(cycle_string)
    fig1 = plt.gcf()

    #fig, ax = plt.subplots()
    
    #ax.xaxis_date()
    # Shrink current axis by 20%
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='upper left', bbox_to_anchor= (0, -0.2))

    fig1.savefig(filename, bbox_inches='tight') #'diurnal_cycle_T_2008to2012.pdf'
    plt.show()
    plt.close()

#%%

def plot_timeseries_merged(filename,merged_df, monthly_grouped_P, yearly_grouped_P, **kwargs):
    # this function plots timeseries for variable input datasets (output from function "merge_df()")
    # it saves the file to the current path of this notebook under the chosen filename (or give the absolute path)
    # you have to name the args in a way that they serve as a label at the same time

     # define color cycle
    cycol = cycle('krgbcm')

    # it assumes that the first args is the in-situ measurement and plots it in black
    for key, ar in kwargs.items():  # e.g. args = 'sp'
        my_col =next(cycol)
        #plt.plot(cycle_name.index, cycle_name[ar]['mean'], '-o', c= my_col, linewidth=2.0, 
         #   label = 'Era5 ' + ar)
        plt.plot(merged_df.index.to_pydatetime(), merged_df[ar], c=my_col, markersize = '1', label = 'hourly ,' + key.replace('_', ' '), alpha = 0.25)
        plt.plot(pd.to_datetime(monthly_grouped_P.index) , monthly_grouped_P[ar]['mean'], '-o',c=my_col,  label = 'monthly ' + key.replace('_', ' '), markersize = '2')
        plt.plot(pd.to_datetime(yearly_grouped_P.index.astype(int), format = '%Y'), yearly_grouped_P[ar]['mean'],'-o', c=my_col, markersize = '3', label = 'yearly' + key.replace('_', ' '))

    if '/RH/' in filename:
        plt.ylabel('Relative humidity [%]')
    
    elif '/SH/' in filename:
        plt.ylabel('Specific humidity [kg/kg]')
    
    elif '/T/' in filename:
        plt.ylabel('Temperature [°C]')
 
    elif '/P/' in filename:
        plt.ylabel('Pressure [hPa]')

    elif '/TCW/' in filename:
        plt.ylabel(r'Precipitable water vapor [$kg/m^2$]')
    
    ax = plt.subplot(111)
    ax.xaxis_date() 
    plt.xlabel('time [years]')
    
    
    plt.legend()
    fig1 = plt.gcf()    
    ax = plt.subplot(111)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='upper left', bbox_to_anchor= (0, -0.2))
    fig1.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


# %%
# merge Era5 datasets to only compare equal rows of data, 
# do not drop nans in this function

def merge_df_long(*args):
    df_list = []
    for ar in args: 
        df_list.append(ar)
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_on='time', right_on='time', how='outer'), df_list)
    
    # on df_merged, create years, months and group by them
    
    # delete rows containing NaN
    #df_merged_nonan = df_merged.dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)
    
    # create a new column consisting of the cycle parameter of the correspondend entry
    #for seasonal cycle (12 months), create column with "months"
    df_merged['months'] = pd.DatetimeIndex(df_merged.index).month                                            

    #for diurnal cycle (24 hours), create column with "hours"
    df_merged['hours'] = pd.DatetimeIndex(df_merged.index).hour

    # for monthly timeseries, create column with "YYYY-MM"
    df_merged['YYYY_MM'] = df_merged.index.strftime('%Y-%m')

    # for 1 year averages, create column with years
    df_merged['years'] = pd.DatetimeIndex(df_merged.index).year

    # for monthly timeseries, groupby YYYY-MM
    monthly_grouped = df_merged.groupby('YYYY_MM').describe()

    # for yearly timeseries, group by 'years'
    yearly_grouped = df_merged.groupby('years').describe()
    
    return df_merged, monthly_grouped, yearly_grouped

#%% build function for linear regression
# (implement function into long timeseries)

def linreg(varx, vary):
    # for the coefficient of determination, R^2, simply calculate: r_value ** 2
    
    # sort out nan values
    mask = ~np.isnan(varx) & ~np.isnan(vary)

    # calculate linear regressions statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(varx[mask], vary[mask])

    # plot 
    #plt.plot(varx[mask], vary[mask], '-o')
    #plt.plot(varx[mask], slope * varx[mask] + intercept)
    return(slope, intercept, r_value, p_value, std_err, varx[mask], vary[mask])

# define function to display scientific notation with an upper index instead of e
def as_si(x, ndp):
    if 'e' in str(x):
        s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
        m, e = s.split('e')
        return r'{m:s} \cdot 10^{{{e:d}}}'.format(m=m, e=int(e))
    elif x == 'intercept':
        return(np.sign(x) * abs(x))
    else:
        return(x)

# #%% build function for displaying intercept
    
# if 'e' in str(intercept):
#     intercept = as_si(intercept, 2)
# else:
#     intercept = np.sign(intercept) * abs(intercept)
#%%  
# example: plot_timeseries_movav('./sites/Paranal/Output/Plots/SH/timeseries_movav_test.pdf', yearly_grouped_SH, In_situ = 'specific_humidity', Era5_700hPa = '700hPa')
def plot_timeseries_movav(filename, yearly_grouped, number_of_insitu_params = 1, df_merged = None, **kwargs):
    # this function plots 5-yearly moving average for the in-situ data and variable Era 5 input datasets
    # the difference to "plot_timeseries_merged" is that the Era 5 data is displayed for all years (1979 to 2020)
    # it saves the file to the current path of this notebook under the chosen filename (or give the absolute path)

    # the dataframes are plotted with their linear regression line,
    # a file with the linear regression details is produced and saved 

     # define color cycle
    # it assumes that first kwargs is the in-situ data (and therefore in black)
    # the default value for the number of insitu parameters is 1, but there can be more
    if number_of_insitu_params == 1:
        cycol = cycle('krgbcm')
        mark_cycol = cycle('oooooo')
    elif number_of_insitu_params == 2:
        cycol = cycle('kkrgbcm')
        mark_cycol = cycle('oXooooo')
    elif number_of_insitu_params == 3:
        cycol = cycle('kkkrgbcm')
        mark_cycol = cycle('oXdooooo')
    elif number_of_insitu_params == 4:
        cycol = cycle('kkkkrgbcm')
        mark_cycol = cycle('oXdpooooo')

    # calculate 5-year running average, shift value to center
    df_5_yearly = yearly_grouped.rolling(5, min_periods = 5).mean().shift(-2)
    
    #  initiate file, write header row
    # filename[:-4] cuts out .pdf, so that we can extend the filename
    filesave = filename[:-4] + '_linear_regression' +'.csv'

    import csv
    with open(filesave, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["variable", "slope", "intercept", "r_value", 'R^2', 'p_value',  'std_err'])

    for key, ar in kwargs.items():  # e.g. args = 'sp'

        my_col = next(cycol)
        my_marker = next(mark_cycol)

        # plot 5-year running average
        plt.plot(pd.to_datetime(df_5_yearly.index.astype(int), format = '%Y'), df_5_yearly[ar]['mean'], '-' + my_marker,  c=my_col,  label = '5-year running average ' + key.replace('_', ' ') , markersize = '5')

        # add error from gaussian error propagation for 5-yearly running average
        err_prop = np.sqrt(1/5 * (yearly_grouped[ar]['std'] ** 2).rolling(5, min_periods = 5).mean().shift(-2))

        plt.fill_between(pd.to_datetime(df_5_yearly.index.astype(int), format = '%Y'), (df_5_yearly[ar]['mean'] - err_prop), (df_5_yearly[ar]['mean'] + err_prop)
        ,alpha=.25, facecolor=my_col) #, label = key.replace('_', ' ') + ' error')
        
        # linear regression
        slope, intercept, r_value, p_value, std_err, varx, vary = linreg(yearly_grouped.index, df_5_yearly[ar]['mean'])
        plt.plot(pd.to_datetime(varx.astype(int), format = '%Y'), slope * varx + intercept, '-' + my_marker, c=my_col, 
        label = r'$y = {:.2} \cdot x + ({:.2f}), R^2 = {:.2}$'.format(as_si(slope,2), as_si(intercept, 2), r_value**2), markersize = '3', alpha = 0.5)
 
        # add std_err ?
        #plt.plot(pd.to_datetime(varx.astype(int), format = '%Y'), slope * varx + intercept, '-o', c=my_col, 
        # , markersize = '3', alpha = 0.5)

        # write to csv, append
        # header: rows are: slope, intercept, r_value, R^2, p_value, std_err
        import csv
        with open(filesave, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([key,'{:.3}'.format(slope), '{:.3}'.format(intercept),
            '{:.3}'.format(r_value), '{:.3}'.format(r_value**2), '{:.3}'.format(p_value), '{:.3}'.format(std_err)])

    fig = plt.gcf()
    ax = fig.add_subplot(111)

    if '/RH/' in filename:
        plt.ylabel('Relative humidity [%]')
    
    elif '/SH/' in filename:
        plt.ylabel('Specific humidity [kg/kg]')
    
    elif '/T/' in filename:
        plt.ylabel('Temperature [°C]')
 
    elif '/P/' in filename:
        plt.ylabel('Pressure [hPa]')
        
        # for the pressure, write the mean, from the whole timeseries and the std. dev.
        key_insitu, ar_insitu = list(kwargs.items())[0] # assign in-situ key and in-situ value, which are first in **kwargs
        key_model, ar_model = list(kwargs.items())[1] # assign Era5 key and value, which are second in **kwargs
        print(key_insitu, ar_insitu, key_model, ar_model)
        print(list(kwargs.items())[0])
        print(list(kwargs.items())[1])
        # use df_merged to get hourly data
        plt.text(0.05 , 0.5 , r'mean of {} = ({:.2f} $\pm$ {:.2f}) hPa' '\n' r'mean of {} pressure = ({:.2f} $\pm$ {:.2f}) hPa'.format(key_model.replace('_', ' ') , df_merged[ar_model].mean(), df_merged[ar_model].std(),
            key_insitu , df_merged[ar_insitu].mean(), df_merged[ar_insitu].std()),
            transform=ax.transAxes,  bbox=dict(facecolor='white', edgecolor='blue', alpha = 0.7))

    elif '/TCW/' in filename:
        plt.ylabel(r'Precipitable water vapor [$kg/m^2$]')

    plt.xlabel('time [years]')
    #plt.xlim(dt.date(1978, 1, 1), dt.date(2020, 1, 31))
    #ax = plt.subplot(111)
    #ax.xaxis_date()
    plt.legend()

    #plt.title(cycle_string)
    
    #ax.xaxis_date()

    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='upper left', bbox_to_anchor= (0, -0.2))

    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


#%% timeseries long
def plot_timeseries_long(filename, yearly_grouped_insitu , list_insitu, yearly_grouped_era5, moving = True,**kwargs):
    # this function plots timeseries of monthly and yearly means for the in-situ data and variable Era 5 input datasets
    # the difference to "plot_timeseries_merged" is that the Era 5 data is displayed for all years (1979 to 2020)
    # it saves the file to the current path of this notebook under the chosen filename (or give the absolute path)
    # args are columns of monthly_grouped and yearly_grouped
    # The insitu data should be inserted in a list (for the case that there are up to 4 in-situ parameters)

    # the key becomes the label
     
    # define color cycle
    cycol = cycle('rgbcm')
    
    if moving == True:
        # calculate 5-year rolling mean for comparison
        df_5_yearly_era5 = yearly_grouped_era5.rolling(5, min_periods = 5).mean().shift(-2)
        df_5_yearly_insitu = yearly_grouped_insitu.rolling(5, min_periods = 5).mean().shift(-2)

    for key, ar in kwargs.items():  # e.g. args = 'sp'
        my_col =next(cycol)

        # plot hourly means (faint)
        #plt.plot(df.index, df, c=my_col, markersize = '1', label = 'hourly ,' + key.replace('_', ' '), alpha = 0.25)
        
        # plot monthly means
        #plt.plot(df.resample('M').mean(), '-o',c=my_col,  label = 'monthly ' + key.replace('_', ' '), markersize = '1', alpha = 0.5)
        #plt.plot(pd.to_datetime(monthly_grouped.index, format = '%Y-%m'), monthly_grouped[ar]['mean'], '-o',  c=my_col,  label = 'monthly average ' + key.replace('_', ' ') , markersize = '3', alpha = 0.3)
        
        if moving == True:
            # plot 5-year running average
            plt.plot(pd.to_datetime(df_5_yearly_era5.index.astype(int), format = '%Y'), df_5_yearly_era5[ar]['mean'], '-*',  c=my_col,  label = '5-year running average ' + key.replace('_', ' ') ,linewidth = 1, markersize = '4')
            # linear regression
            slope, intercept, r_value, p_value, std_err, varx, vary = linreg(yearly_grouped_era5.index, df_5_yearly_era5[ar]['mean'])
            plt.plot(pd.to_datetime(varx.astype(int), format = '%Y'), slope * varx + intercept, '-', c=my_col, linewidth=1, alpha = 0.7, 
            label = r'$y = {:.2} \cdot x + {:.2f}, R^2 = {:.2}$'.format(as_si(slope,2),as_si(intercept,2), r_value**2))


        # plot yearly means
        #plt.plot(df.resample('Y').mean(), '-o',c=my_col,  label = 'yearly ' + key.replace('_', ' '), markersize = '4')
        #plt.plot(pd.to_datetime(yearly_grouped.index.astype(int), format = '%Y'), yearly_grouped[ar]['mean'], '-o',  c=my_col,  label = 'yearly average ' + key.replace('_', ' ') , markersize = '5')
        plt.plot(pd.to_datetime(yearly_grouped_era5.index.astype(int), format = '%Y'), yearly_grouped_era5[ar]['mean'], '-o', 
        c=my_col,  label = 'yearly average ' + key.replace('_', ' ') + r' $\pm$ std. dev.', markersize = '2', alpha = 0.5)
        
        # std. dev
        plt.fill_between(pd.to_datetime(yearly_grouped_era5.index.astype(int), format = '%Y'), (yearly_grouped_era5[ar]['mean'] - yearly_grouped_era5[ar]['std']), 
        (yearly_grouped_era5[ar]['mean'] + yearly_grouped_era5[ar]['std']) ,alpha=.25, facecolor=my_col) #, label = key.replace('_', ' ') + ' std. dev.')
        
        # plot 5-year means
        #plt.plot(df.resample('5Y').mean(), '-o',c=my_col,  label = '5-yearly ' + key.replace('_', ' '), markersize = '7')

    if '/RH/' in filename:
        plt.ylabel('Relative humidity [%]')
    
    elif '/SH/' in filename:
        plt.ylabel('Specific humidity [kg/kg]')
    
    elif '/T/' in filename:
        plt.ylabel('Temperature [°C]')
 
    elif '/P/' in filename:
        plt.ylabel('Pressure [hPa]')

    elif '/TCW/' in filename:
        plt.ylabel(r'Precipitable water vapor [$kg/m^2$]')

    #if there are more than one in-situ parameter (up to 4 parameters are valid), you need different markers, since observations are in black
    mark_cycol = cycle('oXdp')

    for insitu_parameter in list_insitu:
        mymarker = next(mark_cycol)

        if moving == True:
            # plot 5-year running average
            plt.plot(pd.to_datetime(df_5_yearly_insitu.index.astype(int), format = '%Y'), df_5_yearly_insitu[insitu_parameter]['mean'], '-' + mymarker,  c='k',  label = '5-year running average ' + insitu_parameter ,linewidth = 1, markersize = '4')
            # linear regression
            slope, intercept, r_value, p_value, std_err, varx, vary = linreg(yearly_grouped_insitu.index, df_5_yearly_insitu[insitu_parameter]['mean'])
            plt.plot(pd.to_datetime(varx.astype(int), format = '%Y'), slope * varx + intercept, '-', c='k', linewidth=1, alpha = 0.7, 
            label = r'$y = {:.2} \cdot x + ({:.2f}), R^2 = {:.2}$'.format(as_si(slope,2), as_si(intercept,2), r_value**2))
            
        # yearly means
        plt.plot(pd.to_datetime(yearly_grouped_insitu.index.astype(int), format = '%Y'), yearly_grouped_insitu[insitu_parameter]['mean'], '-' + mymarker, 
        c='k',  label = 'yearly average ' + insitu_parameter + r' $\pm$ std. dev.', markersize = '2', alpha = 0.5)
        # std. dev
        plt.fill_between(pd.to_datetime(yearly_grouped_insitu.index.astype(int), format = '%Y'), (yearly_grouped_insitu[insitu_parameter]['mean'] - yearly_grouped_insitu[insitu_parameter]['std']), 
        (yearly_grouped_insitu[insitu_parameter]['mean'] + yearly_grouped_insitu[insitu_parameter]['std']) ,alpha=.25, facecolor='k') #, label = 'in-situ std. dev.')
        
    # plot hourly means (faint)
    #plt.plot(insitu.index, insitu.iloc[:,0], c='k', markersize = '1', label = 'hourly insitu', alpha = 0.25)

    #first_col = insitu.iloc[:,0]    
    # plot monthly means
    #plt.plot(first_col.resample('M').mean(), '-o',c='k',  label = 'monthly insitu', markersize = '1', alpha = 0.5)
    
    # plot yearly means
    #plt.plot(first_col.resample('Y').mean(), '-o',c='k',  label = 'yearly insitu', markersize = '4')
    
    # plot 5-yearly means
    #plt.plot(first_col.resample('5Y').mean(), '-o',c='k',  label = '5-yearly insitu', markersize = '7')
            
    plt.xlabel('time [years]')
    plt.xlim(dt.date(1978, 1, 1), dt.date(2019, 12, 31))
    #ax = plt.subplot(111)
    #ax.xaxis_date() 
    plt.legend()

    #plt.title(cycle_string)
    
    #ax.xaxis_date()
    fig = plt.gcf()
    ax = fig.add_subplot(111)
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend below the current axis
    ax.legend(loc='upper left', bbox_to_anchor= (0, -0.2))

    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

#%% Stats 
# correlation plot and correlation information

# to understand RMSE and Bias:
# Think of a target with a bulls-eye in the middle. The mean square error represent the average squared distance from an arrow shot on the target and the center. 
# Now if your arrows scatter evenly arround the center then the shooter has no aiming bias and the mean square error is the same as the variance.
# But in general the arrows can scatter around a point away from the target. The average squared distance of the arrows from the center of the arrows is the variance. 
# This center could be looked at as the shooters aim point. The distance from this shooters center or aimpoint to the center of the target is the absolute value of the bias.

# RMSE function

# tested with
# df2 = pd.DataFrame(np.array([[2, 2], [3, 5], [4, 1]]), columns=['a', 'b'])
# RMSE = 2.0817
def myRMSE(df_P, df_O):
  # P for predictions
  # O for observations
  return(np.sqrt(np.sum((df_P - df_O)**2)/len(df_P)))

# Bias function (systematischer Fehler, p.16 in Dümbgen 2016)
# tested with
# df2 = pd.DataFrame(np.array([[2, 2], [3, 5], [4, 1]]), columns=['a', 'b'])
# Bias = 1/3
def myBias(df_P, df_O):
  # P for predictions
  # O for observations
  return((np.sum(df_P) - np.sum(df_O))/len(df_O))

# %%
# from github, https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
import numpy as np
from scipy import stats

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

#%%
#from scipy.stats import pearsonr
def correlation_plot(filename, title,  data_x, string_x, data_y ,string_y):
# for the calculation of the bias, it is important that:
# data_x = data_predictions
# data_y = data_observations

    fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])
    if 'hourly' in filename:
        markersize = 1
    elif 'monthly' in filename:
        markersize = 3
    elif 'yearly' in filename:
        markersize = 4
    else:
        markersize = 2

    # create mask to drop nan's (if not already happened)
    mask = ~np.isnan(data_x) & ~np.isnan(data_y)
    data_x = data_x[mask]
    data_y = data_y[mask]

    plt.scatter(data_x, data_y, label = 'data points', s= markersize, c='blue')
    x_eq_y = np.arange(0,100)
    plt.plot(x_eq_y, x_eq_y, color = 'k', label = 'y = x')
    plt.xlim(data_x.min() - data_x.std()/2, data_x.max() + data_x.std()/2)
    plt.ylim(data_y.min() - data_x.std()/2, data_y.max() + data_y.std())
    plt.xlabel(string_x)
    plt.ylabel(string_y)
    
    
    ax = fig.add_subplot(111)

    r, p, lo, hi = pearsonr_ci(data_x, data_y, alpha=0.05)
    slope, intercept, r_value, p_value, std_err, varx, vary = linreg(data_x, data_y)
    plt.plot(data_x, slope * data_x + intercept, '-r',
    label = r'$y = {:.2} \cdot x + ({:.2}), R^2 = {:.2}$'.format(as_si(slope,2), as_si(intercept, 2), r_value**2))
    
    plt.text( 0.45 , 0.1 , 'Bias = {:.3f} \nRMSE = {:.3f} \ncorr. coefficient = {:.2} \np-value = {:.2} \nconfidence interval = ({:.2}, {:.2})'.format(myBias(data_x, data_y) ,myRMSE(data_x, data_y), r, p, lo, hi),
     transform=ax.transAxes,  bbox=dict(facecolor='white', edgecolor='red', alpha = 0.7))

    mytitle = title

    plt.title(mytitle)
    plt.legend(loc = 'upper left')

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    fig1 = plt.gcf()
    
    fig1.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

    #corr, _ = pearsonr(data_x, data_y)
    #print('Pearsons correlation: %.3f' % corr)



# %%
def corr_plots_hourly_monthly_yearly(path, merged_df, monthly_grouped, yearly_grouped, model_parameter, insitu_param, xax, yax):
    # alpha = 0.05
    # example of a path (no .pdf ending): './sites/La_Silla/Output/Plots/T/correlation_T_30m_775hPa_La_Silla_2006to2020'

    #hourly
    # drop nans of merged df to make dataframes equal in size
    merged_df =  merged_df[[model_parameter, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

    correlation_plot(path + '_hourly.pdf',
    'hourly means',
    merged_df[model_parameter], xax,
    merged_df[insitu_param], yax)

    # monthly
    monthly_corr = monthly_grouped[[model_parameter, insitu_param]].dropna(axis='rows', how='any', thresh=None, subset=None, inplace=False)

    correlation_plot(path + '_monthly.pdf',
    'monthly means',
    monthly_corr[model_parameter]['mean'], xax,
    monthly_corr[insitu_param]['mean'], yax)

    # yearly
    correlation_plot(path + '_yearly.pdf',
    'yearly means',
    yearly_grouped[model_parameter]['mean'], xax,
    yearly_grouped[insitu_param]['mean'], yax)

    plt.close()

    return 0
# %%
