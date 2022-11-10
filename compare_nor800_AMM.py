#%%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
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
amm=xr.open_dataset('Data/2021/06/NWS_ANALYSIS_FORECAST_004_013_uv_2021_06_05.nc').squeeze()
# %%
# %%
lon_min=3
lon_max=6
lat_min=59
lat_max=61

mask_lat= (amm.lat>lat_min) & (amm.lat < lat_max)
mask_lon= (amm.lon>lon_min) & (amm.lon < lon_max)

amm_foc=amm.where(mask_lat & mask_lon,drop=True)
#%%

xd= xr.open_dataset('https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2021060500.nc').isel(depth=0)
# %%
xd.u_eastward.stack(z=['Y','X'])
# %%
xd_mask_lat=(xd.lat>lat_min) & (xd.lat < lat_max)
xd_mask_lon=(xd.lon>lon_min) & (xd.lon < lon_max)
# %%
foc=xd.u_eastward.where(xd_mask_lat & xd_mask_lon,drop=True)
# %%
foc.mean(dim='time').plot(x='lon',y='lat')
f# %%
