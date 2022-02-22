#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:52:42 2020

@author: guttorm
"""
#%% read libraries

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import xarray as xr
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date 
from matplotlib import style
import seaborn as sns
import statsmodels.tsa.api as sm
import utide as ut
import scipy as sp
import scipy.stats as stats
import gstools as gs
from utide import solve, reconstruct
style.use('seaborn-notebook')

#%%
data=xr.open_dataset('byfjordenSubset.nc')

#%%
surface=data.isel(depth=0)
surface.h.plot.contourf()

#%%
ein=surface.isel(X=100,Y=100)
ein

coef = solve(date2num(ein.time), ein.u, ein.v, lat=60.0)#,nodal=True, rend=True, method='robust', conf_int='linear', Rayleigh_min=0.95)

tide = reconstruct(date2num(ein.time), coef)
#%%
ein.u.plot()
plt.plot(tide.t, tide.u)

#%%
x = list(range(len(ein.time)))
y=ein.u.to_dataframe().u.interpolate()

temp_fft = sp.fftpack.fft(y.values)
temp_psd = np.abs(temp_fft) ** 2
fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1. / 365)
i = fftfreq > 0

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(temp_psd[i]))
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)')
#%%
f=abs(np.fft.fft(y))

# get the list of frequencies
num=np.size(x)
freq = [i / num for i in list(range(num))]

# get the list of spectrums
spectrum=f.real*f.real+f.imag*f.imag
nspectrum=spectrum/spectrum[0]

# plot nspectrum per frequency, with a semilog scale on nspectrum
plt.semilogy(nspectrum)
#%%

# %%
JC180.rem_v.rolling(time=150).mean().plot()
# %%
pJC180=JC180.to_dataframe()
# %%
pJC180.rem_v.diff().rolling(500).mean().plot()
# %%
#tsa.graphics.qqplot((pJC180.Ua).diff(),line='q')
# %%
JC180.time
# %%
JC180.rem_u.diff(dim='time')
# %%
pJC180.vE.diff().plot()
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(pJC180['rem_u'].diff().dropna(),model='additive', period=40)
decompose.plot()
plt.show()
# %%
pd.plotting.autocorrelation_plot(pJC180.rem_u.diff()).set_xlim([1, 2])
# %%


pJC180.tide_v.diff().autocorr(lag=10)
# %%
lags=np.arange(1, 10)
# %%
lags
pJC180.tide_v.diff().autocorr(lag=lags)
# %%
acor=[]
for lag in lags:
    tmp=pJC180.rem_u.diff().autocorr(lag=lag)
    acor.append(tmp)

print(acor)
# %%
acor
# %%
pJC180.rem_u.diff().plot.hist(bins=30)
# %%
pJC180.rem_v.diff().plot.hist(bins=30)
# %%
pJC180.rem_v.plot.hist(bins=30)


# %%
x = np.arange(100)
y = np.arange(100)
model = gs.Gaussian(dim=2, var=0.01, len_scale=10,anis=1.)
srf = gs.SRF(model, mean=0, generator='VectorField', seed=19841202)
srf((x, y), mesh_type='structured')
tmp=srf((x, y),mesh_type='structured')
tmp[0,:,:]=tmp[0,:,:]-tmp[0,:,:].mean()
tmp[1,:,:]=tmp[1,:,:]-tmp[1,:,:].mean()
ptmp=xr.Dataset({'u':(['x','y'],tmp[0,:,:]),'v':(['x','y'],tmp[1,:,:])}, coords={'x':x,'y':y})
srf.plot()

# %%
help(srf)
# %%

uv=xr.open_dataset('../AdvDiff/Indata/uvt16_2Days.nc')
# %%
coef = solve(date2num(uv.time), uv.u.mean(dim=['x','y']).values, uv.v.mean(dim=['x','y']).values, lat=58.0)#,nodal=True, rend=True, method='robust', conf_int='linear', Rayleigh_min=0.95)

tide = reconstruct(date2num(uv.time), coef)
# %%
uv['tide_u']=(('time'),tide.u)
uv['tide_v']=(('time'),tide.v)

#%%
uv['tide_u']=uv.u*0
uv['tide_v']=uv.v*0
# %%
for ii in np.arange(len(uv.x)):
    print(ii)
    for jj in np.arange(len(uv.y)):
        tmpu=uv.u[:,ii,jj]
        tmpv=uv.v[:,ii,jj]
        coef = solve(date2num(uv.time), tmpu.values, tmpv.values, lat=58.0,verbose=False)
        tide = reconstruct(date2num(uv.time), coef,verbose=False)
        uv['tide_u'][:,ii,jj]=tide.u
        uv['tide_v'][:,ii,jj]=tide.v
# %%
uv.to_netcdf('uv.nc')
# %%
uv=xr.load_dataset('uv.nc')
# %%
