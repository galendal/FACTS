import numpy as np
from sklearn.utils.extmath import randomized_svd
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sc

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
        print('doing SVD analysis')
        self.U, self.L, self.V, self.SCF= self.do_svd(self.data,self.coarse)

    def get_coarse(self, data):
        coarse=data.isel(Y=slice(0,data.dims[self.yname],self.skip),X=slice(0,data.dims[self.xname],self.skip))
        coarse=self.my_mask(coarse)
        return coarse

    def do_svd(self,data,coarse):
        stmp=data#.stack(z=['comp','X','Y'])
        u=stmp.norm.where(stmp.land_mask==0,drop=True).dropna(dim='z') #data.u_norm.where(data.land_binary_mask==0).dropna(dim=spacename).data#+focus.vcent.data*1j
        stmp=coarse#.stack(z=['comp','X','Y'])
        v=stmp.norm.where(stmp.land_mask==0,drop=True).dropna(dim='z')

        Cmat=(np.transpose(u.data)@v.data)

        U,L,Vt=sc.linalg.svd(Cmat)
        V=Vt.T
        SCF=L/np.sum(L)
        indx=int(np.argwhere(np.cumsum(SCF)>self.tresh)[0])+1
    

        U=U[:,0:indx]
        V=V[:,0:indx]
        L=L[0:indx]
        SCF=SCF[0:indx]
        U=xr.DataArray(data=U,dims=['z','mode'],coords=u.drop(['time','depth']).coords)
        V=xr.DataArray(data=V,dims=['z','mode'],coords=v.drop(['time','depth']).coords)
        L=xr.DataArray(data=L,dims=['mode'])
        #xr.Dataset(data_vars=dict(V=(['z','mode'],V)),coords=v.coords)
        return U, L, V, SCF



    def my_mask(self, data):
        out=xr.Dataset(coords=data.coords)
        out['land_mask']=data.land_binary_mask
        
        out['velocity']=xr.concat([data[self.uname],data[self.vname]] ,dim='comp')
        
        out['mean']=out['velocity'].mean(dim=self.timename)
        out['std']=out['velocity'].std(dim=self.timename)+1.e-9
        out['norm']=(out['velocity']-out['mean'])/out['std']

        out=out.stack(z=['comp','Y','X'])
        out=out.where(out.land_mask==0,drop=True)
        #out['land_mask']=data.(land_binary_mask==0)
        
    
        #inarr=inarr.stack(z=self.space_names)
        #inarr=inarr.where(inarr.land_binary_mask==0,drop=True).dropna(dim='z')
        #inarr['ucent']=(inarr[self.uname]-inarr.umean)
        #inarr['unorm']=inarr.ucent/(1e-9+inarr.ustd)
        #inarr['vcent']=(inarr[self.vname]-inarr.vmean)
        #inarr['vnorm']=inarr.vcent/(1e-9+inarr.vstd)
        return out

    def plot_timestamp(self,itime=100,field='velocity'):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,20))
        cmap='coolwarm'
        cmap_std='BuPu'
        #print(self.coarse[field])
        
        self.coarse[field].isel(time=itime,comp=ii).where(self.coarse.land_mask==0).plot(ax=axes[0,0],cmap=cmap)
        #axes[0,0].set_title('u velocity, coarse' + time)
        self.coarse[field].isel(time=itime,comp=1).where(self.coarse.land_mask==0).plot(ax=axes[0,1],cmap=cmap)
        #axes[0,1].set_title('Mean v velocity, coarse')
        self.data[field].isel(time=itime,comp=0).where(self.data.land_mask==0).plot(ax=axes[1,0],cmap=cmap)
        #axes[1,0].set_title('Standard deviation u velocity: fine ')
        self.data[field].isel(time=itime,comp=1).where(self.data.land_mask==0).plot(ax=axes[1,1],cmap=cmap)
        #axes[1,1].set_title('Standard deviation v velocity: fine')

    def plot_timestat(self,field='velocity'):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,20))
        cmap='coolwarm'
        cmap_std='BuPu'
        #print(self.coarse[field])
        #time=self.coarse['time'].isel(time=itime).data
        for ii in np.arange(self.coarse.dims['comp']):
            self.coarse[field].isel(comp=ii).mean(dim='time').where(self.coarse.land_mask==0).plot(ax=axes[0+ii,0],cmap=cmap)
            axes[0+ii,0].set_title('Mean velocity, coarse'+ str(ii))
            self.coarse[field].isel(comp=ii).std(dim='time').where(self.coarse.land_mask==0).plot(ax=axes[0+ii,1],cmap=cmap_std)
            axes[0+ii,1].set_title('STD velocity, coarse' + str(ii))
            self.data[field].isel(comp=ii).mean(dim='time').where(self.data.land_mask==0).plot(ax=axes[2+ii,0],cmap=cmap)
            axes[2+ii,0].set_title('Mean fine '+ str(ii))
            self.data[field].isel(comp=ii).std(dim='time').where(self.data.land_mask==0).plot(ax=axes[2+ii,1],cmap=cmap_std)
            axes[2+ii,1].set_title('Standard deviation velocity: fine'+str(ii))

    def plot_EOFS(self,field='coarse',num_EOFS=4,cmap='coolwarm'):
        if field=='coarse':
            tmp=self.V.unstack('z')
        else:
            tmp=self.U.unstack('z')

        fig, axes = plt.subplots(nrows=num_EOFS, ncols=2, figsize=(20,20))
        for ii in np.arange(num_EOFS):
            tmp.isel(comp=0,mode=ii).plot(ax=axes[0+ii,0],cmap=cmap)
            axes[0+ii,0].set_title('U velocity component, mode '+ str(ii))
            tmp.isel(comp=1,mode=ii).plot(ax=axes[0+ii,1],cmap=cmap)
            axes[0+ii,1].set_title('V velocity component, mode '+ str(ii))

    def calc_PCS(self, infield):
        indim=infield.sizes['z']
        if indim == self.data.dims['z']:
            MAT=self.U.data
            z=self.data.z
        elif indim == self.coarse.dims['z']:
            MAT=self.V.data
            z=self.coarse.z
        else:
            print('Dimension of inarray is: '+ str(indim) + ' not equal to ' + str(self.data.dims['z']) +' or '+ self.coarse.dims['z']) 
            print('Leaving from calc_PCS') 
            return -1

        outtime=infield.time
        
        out=xr.DataArray(data=infield.data@MAT, dims=['time','mode']).assign_coords(time=outtime)
        #    (['time'],outtime)
        #    ))

        return out

    def est_PCS(self, infield):
        indim=infield.sizes['z']
        
        if indim != self.coarse.dims['z']:
            print('Dimension of inarray is: '+ str(indim) + ' not equal to ' + str(self.coarse.dims['z']))  
            print('Leaving from est_PCS') 
            return -1

        outtime=infield.time
        rho=self.calc_PCS(self.coarse.norm).var(dim='time')

        Bn=self.calc_PCS(infield)
        rho_n=Bn.var(dim='time')
        print(rho/rho_n)
        Anest=((np.diag(self.L.data)@np.linalg.pinv(Bn)).T)@(np.diag(rho/rho_n)) #np.linalg.pinv(B6).T@np.diag(L)
        #Anest=((infield.sizes['time']-1)/(self.coarse.dims['time']-1))**(1/2)*(np.diag(self.L.data)@np.linalg.pinv(Bn.data)).T

        out=xr.DataArray(data=Anest, dims=['time','mode']).assign_coords(time=outtime)
        #    (['time'],outtime)
        #    ))

        return out

    def calc_vel(self, infield):
        indim=infield.sizes['z']
        if indim == self.data.dims['z']:
            MAT=self.U.data
            z=self.data.z
        elif indim == self.coarse.dims['z']:
            MAT=self.V.data
            z=self.coarse.z
        else:
            print('Dimension of inarray is: '+ str(indim) + ' not equal to ' + str(self.data.dims['z']) +' or '+ self.coarse.dims['z']) 
            print('Leaving from calc_fine_vel') 
            return -1
            

        outtime=infield.time
        A=self.calc_PCS(infield)
        out=xr.DataArray(data=A.data@MAT.T, dims=['time','z']).assign_coords(time=outtime).assign_coords(z=z)
        #    (['time'],outtime)
        #    ))

        return out

    def est_vel(self, infield):
        indim=infield.sizes['z']
        if indim != self.coarse.dims['z']:
            print('Dimension of inarray is: '+ str(indim) + ' not equal to ' + str(self.coarse.dims['z']))  
            print('Leaving from est_vel') 
            return -1
        outtime=infield.time
        A=self.est_PCS(infield)
        out=xr.DataArray(data=A.data@self.U.data.T, dims=['time','z']).assign_coords(time=outtime).assign_coords(z=self.data.z)
        return out