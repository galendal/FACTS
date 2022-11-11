#%%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxess
from matplotlib.dates import date2num, num2date
from utide import solve, reconstruct
import numpy as np
import pandas as pd
from windrose import WindroseAxes
import scipy as sp
import hvplot.xarray
from eofs.xarray import Eof
import glob
import sklearn as skl
from sklearn.utils.extmath import randomized_svd
#import dask
from matplotlib import style
import EOF_funcs
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
import GA_cofine
from pyts.image import RecurrencePlot
from  importlib  import reload
import pywt
import utm
#import xesmf as xe

#%%
files=sorted(glob.glob('Data/2021/0?/NWS_ANALYSIS_FORECAST_004_013_uv*'))
#%%
amm=xr.open_dataset('Data/2021/06/NWS_ANALYSIS_FORECAST_004_013_uv_2021_06_05.nc').squeeze()
# %%
# %%
def amm_create_mask(amm, bndry_lon=[3,6], bndry_lat=[59,61] ):
    lon_min,lon_max=bndry_lon
    lat_min,lat_max=bndry_lat
    mask_lat= (amm.lat>lat_min) & (amm.lat < lat_max)
    mask_lon= (amm.lon>lon_min) & (amm.lon < lon_max)
    mask= mask_lat & mask_lon
    return mask


mask=amm_create_mask(amm)
#%%
data=[]
for file in files:
    amm=xr.open_dataset(file)
    data.append(amm.where(mask,drop=True).squeeze())
data=xr.concat(data, dim='time')
#%%
data.to_netcdf(('Data/2021/NWS_focused.nc'))
#%%


filestem='https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.'
filetail='00.nc'
yr, mnth, day=file.split('_')[6:9]
day=day.split('.')[0]
fileN800 = filestem+yr + str(mnth).zfill(1)+day+filetail

xd= xr.open_dataset(fileN800)
mask_n800=amm_create_mask(xd)

#%%
dataN800=[]
for file in files:
    yr, mnth, day=file.split('_')[6:9]
    day=day.split('.')[0]
    fileN800 = filestem+yr + str(mnth).zfill(1)+day+filetail
    xd= xr.open_dataset(fileN800)[['u_eastward','v_northward']].isel(depth=0).where(mask_n800,drop=True)
    #xd=xd.stack(z=['Y','X']).dropna(dim='z').unstack('z')
    dataN800.append(xd)

dataN800=xr.concat(dataN800, dim='time')

dataN800.to_netcdf('Data/2021/NK800_focused.nc')

# %%
xd_mask_lat=(xd.lat>lat_min) & (xd.lat < lat_max)
xd_mask_lon=(xd.lon>lon_min) & (xd.lon < lon_max)
# %%
foc=xd.u_eastward.where(xd_mask_lat & xd_mask_lon,drop=True)
# %%
foc.mean(dim='time').plot(x='lon',y='lat')
# %%
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
# %%
from scipy.interpolate import griddata
# %%
