import numpy as np
from sklearn.utils.extmath import randomized_svd
import xarray as xr

def my_svd(data, tname='time', uname='u', vname='v', spacename='z', tresh=0.90,plot=False):
    u_mean=data[uname].mean(dim=tname)
    v_mean=data[uname].mean(dim=tname)
    uu=(data[uname]-u_mean).where(data.land_binary_mask==0).dropna(dim=spacename).data
    vv=(data[vname]-v_mean).where(data.land_binary_mask==0).dropna(dim=spacename).data
    
    C=np.dot(uu.T, vv)
    U,L,Vh = np.linalg.svd(C)
    
    V=Vh.T
    SCF=L/np.sum(L)
    indx=int(np.argwhere(np.cumsum(SCF)>tresh)[0])+1
    

    U=U[:,0:indx]
    V=V[:,0:indx]
    L=L[0:indx]
    SCF=SCF[0:indx]
    A=uu@U
    B=vv@V
    
    re_U=A@U.T
    re_V=B@V.T
    if plot:
        plt.plot(SCF)
        plt.xlim(0,10)
        plt.plot(np.cumsum(SCF))
    print(A.shape)
    out=data.copy()
    out['u_pcs']=((tname,'mode'),A)
    out['v_pcs']=((tname,'mode'),B)
    out['u_eofs']=((u_mean.dims + ('mode',)),U)
    out['v_eofs']=((v_mean.dims + ('mode',)),V)
    out['eigen']=(('mode'),L)
    out['SCF']=(('mode'),SCF)

    return out

def my_svd_randomized(data, tname='time', uname='u', vname='v', spacename=['Y','X'], N=20,plot=False,random_state=10):
    tmp=data.stack(z=spacename)
    tmp=tmp.where(tmp.land_binary_mask==0).dropna(dim='z')
    u_mean=tmp[uname].mean(dim=tname)
    v_mean=tmp[vname].mean(dim=tname)
    uu=(tmp[uname]-u_mean).data # Need a warning here. Will alter the calculations. 
    vv=(tmp[vname]-v_mean).data
    
    C=np.dot(uu.T, vv)
    
    U,L,Vh = randomized_svd(C,N,random_state=random_state)
    
    V=Vh.T

    A=uu@U
    B=vv@V
    
    #re_U=A@U.T
    #re_V=B@V.T
    #if plot:
    #    plt.plot(L)
        
    
    #out['u_mean']=u_mean
    #out['v_mean']=v_mean
    tmp=xr.Dataset(coords=tmp.coords)
    tmp[uname+'_eigen']=(('mode'),L)
    tmp[uname+'_pcs']=((tname,'mode'),A)
    tmp[vname+'_pcs']=((tname,'mode'),B)
    tmp[uname+'_eofs']=((u_mean.dims + ('mode',)),U)#,compat='override')
    tmp[vname+'_eofs']=((v_mean.dims + ('mode',)),V)#,compat='override')
    
    
    
    #out['speed']=np.sqrt(out.u**2+out.v**2)
    #angles=np.arctan2(out['v'],out['u'])
    #out['direction']=(angles + 2 * np.pi) % (2 * np.pi)*(180/np.pi)
    #out=data.copy()
    return xr.merge([data,tmp.unstack('z')])

def my_tide(data,tname='time',lat=60, uname='u',vname='v', roll=5):
    
    data['tide_u']=data[uname]*0
    data['tide_v']=data[vname]*0
    time=date2num(data.time.to_pandas().index)

    for ii in np.arange(len(data['z'])):
        tmp=data.isel(z=ii)
        uu=tmp[uname].rolling(time=roll).mean().data
        vv=tmp[vname].rolling(time=roll).mean().data
        coef = solve(time, uu, vv, lat=lat,verbose=False)
        tide = reconstruct(date2num(data.time), coef,verbose=False)
        data['tide_u'][:,ii]=tide.u
        data['tide_v'][:,ii]=tide.v

    data['res_u']=data[uname]-data['tide_u']
    data['res_v']=data[vname]-data['tide_v']

    return data
    
def eofs_GA(inxarr, variables=['u','v'], neofs=10, time_name='time', npcs=5, treshold=None):
    tmp=inxarr[variables]
    tmp=tmp.to_stacked_array('z_loc',sample_dims=[time_name]).fillna(0)
    solver = Eof(tmp)
 
    out=solver.eofs(neofs=neofs).to_unstacked_dataset('z_loc').unstack()
    #xr.Dataset()
    out['X']=inxarr['X']
    out['Y']=inxarr['Y']
    out['time']=inxarr['time']
    out['eigenvals']=solver.eigenvalues()
    out['land_binary_mask']=inxarr['land_binary_mask']
    #if treshold < lmbda.mode.max(): 
    #    np.searchsorted(lmbda.cumsum()/lmbda.sum(),[0.9,],side='right')[0]
    # eofs=solver.eofs(neofs=5).to_unstacked_dataset('z_loc').unstack()
    # out['eofs_u']=eofs['u']
    # out['eofs_v']=eofs['v']
    out['speed']=np.sqrt(out.u**2+out.v**2)
    angles=np.arctan2(out['v'],out['u'])
    out['direction']=(angles + 2 * np.pi) % (2 * np.pi)*(180/np.pi)
    out['pcs']=solver.pcs(npcs=npcs)

    return out

    
def my_mask(inarr, calcMeans=True):
    if calcMeans==True:
        inarr['umean']=inarr.u.mean(dim='time')
        inarr['vmean']=inarr.v.mean(dim='time')
        inarr['ustd']=inarr.u.std(dim='time')
        inarr['vstd']=inarr.v.std(dim='time')
    
    inarr=inarr.stack(z=['Y','X'])
    inarr=inarr.where(inarr.land_binary_mask==0,drop=True).dropna(dim='z')
    inarr['ucent']=(inarr.u-inarr.umean)
    inarr['unorm']=inarr.ucent/(1e-9+inarr.ustd)
    inarr['vcent']=(inarr.v-inarr.vmean)
    inarr['vnorm']=inarr.vcent/(1e-9+inarr.vstd)
    return inarr