# 2021-07-29
# investigate whether or not there are scarce observations in the area close to the Puclaro Dam
# Caroline Haslebacher

#%% load libraries
from numpy.core.fromnumeric import size
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import pickle
import numpy as np 

#%%
#os.chdir('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
import sys
sys.path.append('/home/haslebacher/chaldene/Astroclimate_Project/sites/')
#%reload_ext autoreload
#%autoreload 2
# to automatically reload the .py file

import climxa

import importlib
importlib.reload(climxa)

#%% load in site specific data

# [0]: Mauna Kea
# [1]: Cerro Paranal
# [2]: La Silla
# [3]: Cerro Tololo
# [4]: La Palma
# [5]: Siding Spring
# [6]: Sutherland
# [7]: SPM

# d_site_lonlat_data = pd.read_csv('/home/haslebacher/chaldene/Astroclimate_Project/Sites_lon_lat_ele_data.csv')
d_site_lonlat_data = pickle.load( open( "/home/haslebacher/chaldene/Astroclimate_Project/d_site_lonlat_data.pkl", "rb" ))

#%%

plt.rcParams["figure.figsize"] = (20,10)

#%% read ERA5 data sent by Hans Hersbach
# 4D-var data for atmosphere
var = pd.read_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Puclaro_dam_investigation/era5_Chili/era5_Chile_4dvar_197901-200212.csv')
# ldas for land variables
ldas = pd.read_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Puclaro_dam_investigation/era5_Chili/era5_Chile_ldas_197901-200212.csv')

# it seems that data spans 1979 to 2002 !?!?

#%%
# select dataset
fdvar = ldas

#%%
# lon1 = -70
# lon2 = -71.5
# lat1 = -29.2
# lat2 = -30.5

# lat>=40 and lat<=90 and lon>=-145 and lon<=-50
# lon1 = -65
# lon2 = -74
# lat1 = -25
# lat2 = -35

lon1 = -70
lon2 = -71.5
lat1 = -29
lat2 = -30

# just make dataframe with lon and lat as index
# --> xarray
# problem: entries have a standard deviation of +- 2 degrees sometimes!!!
# take the mean first
fdvar_extent = fdvar[(lat2 < fdvar['mean(lat)']) &  (lat1 > fdvar['mean(lat)']) & (lon2 < fdvar['mean(lon)']) &  (lon1 > fdvar['mean(lon)'])  ]
# 329 entries
# plot in map

# find minimum to Dam
# fdvar_extent['mean(lon)'].max().index()
# -71.19998931884766
fdvar_extent[(-71.9<= fdvar_extent['mean(lon)']) & (fdvar_extent['mean(lon)'] >= -71.2)]
fdvar_extent[(-71.9<= fdvar_extent['mean(lon)']) & (fdvar_extent['mean(lon)'] >= -71.2) & (fdvar_extent['varno@body'] == 7)]
# show all different varno bodies
fdvar_extent['varno@body'].unique()

#%%
# filter for observations 
fdvar_extent = fdvar[(lat2 < fdvar['mean(lat)']) &  (lat1 > fdvar['mean(lat)']) & (lon2 < fdvar['mean(lon)']) &  (lon1 > fdvar['mean(lon)'])  ]

# 329 entries



# %%
# first step: plot tololo, paranal, la silla and puclaro dam in one grid
tololo_lon = d_site_lonlat_data['lon_obs'][3] - 360
tololo_lat =  d_site_lonlat_data['lat_obs'][3]

fdvar_tololo = fdvar[(tololo_lat < fdvar['mean(lat)'] + fdvar['stdev(lat)']) & 
                    (tololo_lat > fdvar['mean(lat)'] - fdvar['stdev(lat)']) & 
                    (tololo_lon < fdvar['mean(lon)'] + fdvar['stdev(lon)']) & 
                    (tololo_lon > fdvar['mean(lon)'] - fdvar['stdev(lon)'])]

print(len(fdvar_tololo))
# to csv
fdvar_tololo.to_csv('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Puclaro_dam_investigation/observations_spanning_over_tololo.csv')

# analysis:
# type 16008 is a ship, so we have three ships
# 
#%% from 'plot_oro()' of climxa

# Create a Stamen Terrain instance.
stamen_terrain = cimgt.StamenTerrain()

# Create a GeoAxes in the tile's projection.
# ax = plt.axes(projection=stamen_terrain.crs)
ax = plt.axes(projection=ccrs.PlateCarree())

projex = ccrs.PlateCarree()

# Limit the extent of the map to a small longitude/latitude range.

ax.set_extent([lon1, lon2, lat1, lat2])

MK_x_ticks = np.arange(lon1, lon2, 0.25)
ax.set_xticks(MK_x_ticks, crs=projex, minor = True)
MK_y_ticks = np.arange(lat1, lat2, 0.25)
ax.set_yticks(MK_y_ticks,crs=projex)


# Add the Stamen data at zoom level 8.
ax.add_image(stamen_terrain, 8)

# Add a marker for the Puclaro Dam
plt.plot(-70.8462, -29.9973, marker='o', color='red', markersize=9,
            alpha=0.7, transform=ccrs.Geodetic())
# Cerro Tololo, idx=3
plt.plot(d_site_lonlat_data['lon_obs'][3], d_site_lonlat_data['lat_obs'][3], marker='o', color='blue', markersize=9,
            alpha=0.7, transform=ccrs.Geodetic())
# La Silla
plt.plot(d_site_lonlat_data['lon_obs'][2], d_site_lonlat_data['lat_obs'][2], marker='d', color='blue', markersize=9,
            alpha=0.7, transform=ccrs.Geodetic())

# plot ECMWF data, one dot per observation
# but caution: data is only between 2015-2017
for idy, row in fdvar_extent.iterrows():
    # plt.plot(row['mean(lon)'], row['mean(lat)'], marker='o', color='olive', markersize=9,
    #         alpha=0.7, transform=ccrs.Geodetic())
    # # plot standard deviation
    # # if standard deviation is non-zero, this means that the wheather station is moving (ship, airplane,...)
    # plt.plot()

    # in any case, plot a dot
    plt.plot(row['mean(lon)'], row['mean(lat)'], marker='o', color='navy', markersize=9,
                alpha=0.7, transform=ccrs.Geodetic())

    if row['stdev(lon)'] != 0 and row['stdev(lat)'] != 0: # plot rectangle
        ax.add_patch(mpatches.Rectangle(xy=[row['mean(lon)'], row['mean(lat)']], width=row['stdev(lon)'], height=row['stdev(lat)'],
                                        facecolor='navy',
                                        alpha=0.3,
                                        transform=projex)
                    )


# save fig
plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Puclaro_dam_investigation/Observational_data_in_chile_ldas.png', dpi=500)

#%% make separate legend that is valid for all plots
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

line_list = []
# Puclaro dam
line_list.append(Line2D([0], [0], linestyle = '', color = 'red', marker = 'o', label = 'Puclaro Dam (-70.85, -30.00'))
# Tololo
line_list.append(Line2D([0], [0], linestyle = '', color = 'blue', marker = 'o', label = 'Cerro Tololo (-70.80, -30.17)'))
# La Silla
line_list.append(Line2D([0], [0], linestyle = '', color = 'red', marker = 'd', label = 'La Silla (-70.73, -29.25)'))
# 4d Var
line_list.append(Line2D([0], [0], linestyle = '', color = 'olive', marker = 'o', label = 'ECMWF 4D Var atmosphere observations'))
# LDAS
line_list.append(Line2D([0], [0], linestyle = '', color = 'navy', marker = 'o', label = 'ECMWF LDAS land observations'))

plt.axis('off')
plt.subplots_adjust(top=0.75, bottom=0.5,
                        left=0.4, right=0.99)
plt.legend(handles=line_list) # , loc='lower left', bbox_to_anchor= (1.2, 0)
plt.savefig('/home/haslebacher/chaldene/Astroclimate_Project/sites/Cerro_Tololo/Puclaro_dam_investigation/Observational_data_legend.png', dpi=500, facecolor='white', bbox_inches = 'tight')
plt.show()


#%%
# # define Projection
# projex = ccrs.PlateCarree()

# # define colormap
# cmap1 = mpl.cm.cividis
# rect = (-60.5,-71.5,-23,-31.5)
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent(rect)
# ax.stock_img() # elevation image
# # 

# plt.show()

#%%

#%%

# # plot model
# orography = model[param].plot(ax=ax, transform=projex, cmap=cmap1, norm=norm1,
#                     add_labels=False, add_colorbar=False)

# # add land-sea mask
# # print(model['lsm'])
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1, 1, projection=projex)
# CS = model['lsm'].plot.contour(ax=ax, transform=projex,
#                 add_labels=False, add_colorbar=False)
# # draw labels inline, but do not draw colorbar
# plt.clabel(CS, inline=1, fontsize=10)
# # plt.show()

# # orography = model[param].plot.contour(ax=ax, transform=projex,
# #              add_labels=False, add_colorbar=False)
# # ax.clabel(CS, inline=1, fontsize=10)

# # ax.set_global()
# ax.coastlines()
# ax.set_extent([lon1 , lon2, lat1, lat2])

# MK_x_ticks = np.arange(lon1, lon2, 0.25)
# ax.set_xticks(MK_x_ticks, crs=projex, minor = True)
# MK_y_ticks = np.arange(lat1, lat2, 0.25)
# ax.set_yticks(MK_y_ticks,crs=projex)

# # omit colorbar for subplot, instead, create global colorbar in the end

# #ax.xticks(rotation=90)
# # ax.gridlines(draw_labels=True)
# #observatory (google maps) Mauna Kea
# lon = lon_obs
# lat = lat_obs
# #x,y = m2(lon, lat)
# plt.plot(lon, lat, 'ko', markersize=6, label = observatory_name + ', ' + str(pressure) + 'hPa', transform=projex)

# # plot nearest gridpoint
# if 'longitude' in model['nearest'].coords:
#     lon_m = model['nearest'].longitude
#     lat_m = model['nearest'].latitude
# elif 'lon' in model['nearest'].coords:
#     lon_m = model['nearest'].lon
#     lat_m = model['nearest'].lat

# plt.plot(lon_m, lat_m,'ro', label='nearest gridpoint', transform=projex)
# # print also onto plot with textbox for overview
# if param == 'orog':
#     print('nearest: {:.1f} m'.format(model['nearest'].values))
#     plt.text( 0.1 , 0.1 , 'nearest: {:.1f} m'.format(model['nearest'].values), #(surface_ele),
#     transform=ax.transAxes,  bbox=dict(facecolor='white', alpha = 0.6))
# elif param == '_slope':
#     if model['_p_value'].values < 0.001:
#         p_value = '<0.001'
#     else:
#         p_value = '=' +  '{:.2f}'.format(model['_p_value'].values)

#     plt.text( 0.1 , 0.1 , 'nearest:\nslope={:.3f}'.format(model['nearest'].values)+ r'$\pm$' + '{:.3f}\np-value{}\n'.format(model['_std_err'].values, p_value) + r'$r^2$=' + '{:.2f}'.format((model['_r_value'].values)**2), #(surface_ele),
#     transform=ax.transAxes, bbox=dict(facecolor='white', alpha = 0.6))

# ax.set_title(alphabet_from_idx(index) + ') ' + oro_name)
