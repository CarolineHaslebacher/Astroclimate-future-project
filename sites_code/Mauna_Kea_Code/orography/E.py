

from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#%%

#fh = Dataset("/home/caroline/hulk/Astroclimate/cds_data_ERA5/Era5_orography_singleLevel.nc", mode = "r", format="NETCDF4")
#
#lons = fh.variables['longitude'][:]
#lats = fh.variables['latitude'][:]
#
#t = fh.variables['time'][:]
## hcc = fh.variables['hcc_0001']
#height = fh.variables['z_0001'][:]
#
#fh.close()

ds_oro = xr.open_dataset("/home/caroline/hulk/Astroclimate/cds_data_ERA5/Era5_orography_singleLevel.nc")
lons = ds_oro.variables['longitude'][:]
lats = ds_oro.variables['latitude'][:]
height = ds_oro.variables['z_0001'][:]

ds_RH = xr.open_dataset('/home/caroline/hulk/Astroclimate/cds_data_ERA5/RH/Era5_2009to2018_RH_600hPa.nc')
RH = ds_RH['r']

ds_orography = xr.open_dataset('/home/caroline/hulk/Astroclimate/cds_data_ERA5/Era5_orography_singleLevel.nc')
ds_MaunaKea = ds_orography.sel(longitude=204.53,latitude= 19.83,method='nearest')
MK = ds_MaunaKea['z_0001']
#grid location (lon/lat) of MaunaKea
ds_MaunaKea = ds_orography.sel(longitude=(360-155.46806),latitude= 19.82083,method='nearest')

#%%
#MaunaKea
m2 = Basemap(width=250000,height=150000,
            resolution='h',projection='lcc',\
            lat_0=19.83,lon_0=-155.47)

##some of america
#m2 = Basemap(width=12000000,height=9000000,projection='lcc',
#            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)


lon, lat = np.meshgrid(lons, lats)
xi, yi = m2(lon, lat) 

#%% 

m2.drawcoastlines()
#m2.drawparallels(np.arange(-80.,81.,0.25))
#m2.drawmeridians(np.arange(-180.,181.,0.25))

# Plot Data
g=9.80665 # divide height by g to get the height in meters
#cs = m2.pcolor(xi,yi, (height[0,:,:]/g))
z = (height[0,:-1, :-1])/g
cs = m2.pcolor(xi, yi, z)
#cs = m2.contourf(xi,yi, height[0,:,:]/g)

#cs_r = m2.contourf(xi, yi, RH[0, :, :])
#cs_r = m2.pcolormesh(xi, yi, RH[0, :, :])              

# Add Colorbar
cbar = m2.colorbar(cs, location='bottom', pad="10%") 

#plt.figure(figsize=(7,7), dpi = 300)
fig1 = plt.gcf()
#fig1.savefig('MaunaKea_basemap2.pdf')
plt.show()

#%%
ele = ds_oro.z_0001
ele[0, 279:283, 817:821]
MK = ele[0, 278:284, 816:822]
MK.plot()

#air2d = ele.isel(time = 0)



