import numpy as np
from sklearn.utils.extmath import randomized_svd
import xarray as xr

class GA_cofine:
    # init method or constructor
    def __init__(self,data,vel_names=['u','v'],space_names=['X','Y'],timename='time',skip=6,tresh=0.9):
        self.vel_names=vel_names
        self.space_names=space_names
        self.skip=skip
        self.timename=timename
        self.uname, self.vname=vel_names
        self.xname, self.yname= space_names
        self.tresh=tresh
        self.data=self.my_mask(data)
        self.coarse=self.get_coarse(data)

        self.U, self.L, self.V= self.do_svd(self.data,self.coarse)

    def get_coarse(self, data):
        coarse=data.isel(Y=slice(0,data.dims[self.yname],self.skip),X=slice(0,data.dims[self.xname],self.skip))
        coarse=self.my_mask(coarse)
        return coarse

    def do_svd(self,data,coarse):
        stmp=data.stack(z=['comp','X','Y'])
        u=stmp.norm.where(stmp.land_mask==0,drop=True) #data.u_norm.where(data.land_binary_mask==0).dropna(dim=spacename).data#+focus.vcent.data*1j
        stmp=coarse.stack(z=['comp','X','Y'])
        v=stmp.norm.where(stmp.land_mask==0,drop=True) 

        Cmat=(np.transpose(u.data)@v.data)

        U,L,Vt=np.linalg.svd(Cmat)
        V=Vt.T
        SCF=L/np.sum(L)
        indx=int(np.argwhere(np.cumsum(SCF)>self.tresh)[0])+1
    

        U=U[:,0:indx]
        V=V[:,0:indx]
        L=L[0:indx]
        SCF=SCF[0:indx]
        
        return U, L, V



    def my_mask(self, data):
        out=xr.Dataset(coords=data.coords)
        out['land_mask']=data.land_binary_mask
        
        out['velocity']=xr.concat([data[self.uname],data[self.vname]] ,dim='comp')
        
        out['mean']=out['velocity'].mean(dim=self.timename)
        out['std']=out['velocity'].std(dim=self.timename)+1.e-9
        out['norm']=(out['velocity']-out['mean'])/out['std']
        #out['land_mask']=data.(land_binary_mask==0)
        
    
        #inarr=inarr.stack(z=self.space_names)
        #inarr=inarr.where(inarr.land_binary_mask==0,drop=True).dropna(dim='z')
        #inarr['ucent']=(inarr[self.uname]-inarr.umean)
        #inarr['unorm']=inarr.ucent/(1e-9+inarr.ustd)
        #inarr['vcent']=(inarr[self.vname]-inarr.vmean)
        #inarr['vnorm']=inarr.vcent/(1e-9+inarr.vstd)
        return out


    