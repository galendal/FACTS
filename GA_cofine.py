import numpy as np
from sklearn.utils.extmath import randomized_svd
import xarray as xr

class GA_cofine:
    # init method or constructor
    def __init__(self,data,vel_names=['u','v'],space_names=['X','Y'],timename='time',skip=6):
        self.vel_names=vel_names
        self.space_names=space_names
        self.skip=skip
        self.timename=timename
        self.uname, self.vname=vel_names
        self.xname, self.yname= space_names
        
        self.data=self.my_mask(data)
        self.coarse=self.get_coarse(data)

    def get_coarse(self, data):
        coarse=data.isel(Y=slice(0,data.dims[self.yname],self.skip),X=slice(0,data.dims[self.xname],self.skip))
        coarse=self.my_mask(coarse)
        return coarse

    def do_svd(self,data,coarse):
        focus=EOF_funcs.my_mask(data)
        u=focus.unorm#+focus.vcent.data*1j
        v=coarse.unorm#+coarse.vcent.data*1j

        Cmat=(np.transpose(u.data)@v.data)

        U,L,Vt=np.linalg.svd(Cmat)

        return U, L, Vt.T



    def my_mask(self, data):
        out=xr.Dataset(coords=data.coords)
        for name in self.vel_names:
            print(name)
            out[name+'_mean']=data[name].mean(dim=self.timename)
            out[name+'_std']=data[name].std(dim=self.timename)+1.e-9
            out[name+'_norm']=(data[name]-out[name+'_mean'])/out[name+'_std']
        
        
    
        #inarr=inarr.stack(z=self.space_names)
        #inarr=inarr.where(inarr.land_binary_mask==0,drop=True).dropna(dim='z')
        #inarr['ucent']=(inarr[self.uname]-inarr.umean)
        #inarr['unorm']=inarr.ucent/(1e-9+inarr.ustd)
        #inarr['vcent']=(inarr[self.vname]-inarr.vmean)
        #inarr['vnorm']=inarr.vcent/(1e-9+inarr.vstd)
        return out


    