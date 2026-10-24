"""Lossless V2 M3 mask index for repeated nominal feasibility checks.

No background RGB or depth is read. A negative mask is conclusive only under
an explicit object-only/no-external-occluder acquisition declaration. Otherwise
it is undetermined, as required by the existing dataset contract.
"""
import numpy as np


class ForegroundIndex:
    def __init__(self,dataset,config):self.dataset=dataset;self.config=config;self.cache={}

    def frame(self,k,ci):
        key=(k,ci)
        if key not in self.cache:
            f=self.dataset.observation(k,ci)
            self.cache[key]=(np.packbits(f.mask.ravel()),f.object_only_no_external_occluders)
        return self.cache[key]

    def needs_check(self,k):
        base=getattr(self.dataset,'base',None)
        if base is not None:
            return base.metadata_report['mask_visibility_premise']['object_only_no_external_occluders'] is True
        return any(self.frame(k,ci)[1] is True for ci in range(len(self.dataset.cameras)))

    def feasible(self,k,x,covariance):
        if not self.needs_check(k):return True
        for ci,camera in enumerate(self.dataset.cameras):
            bits,premise=self.frame(k,ci)
            if premise is not True:continue
            q=((camera.world_to_camera[:3,0]*x[:,0,None]+camera.world_to_camera[:3,1]*x[:,1,None])+camera.world_to_camera[:3,2]*x[:,2,None])+camera.world_to_camera[:3,3]
            h=q[:,:2]/q[:,2,None]*camera.intrinsics[:2]+camera.intrinsics[2:]
            uv=np.floor(np.nan_to_num(h)+.5).astype(np.int64)
            inside=np.isfinite(q).all(1)&(q[:,2]>camera.near)&(q[:,2]<camera.far)&(uv[:,0]>=0)&(uv[:,0]<camera.width)&(uv[:,1]>=0)&(uv[:,1]<camera.height)
            ids=np.flatnonzero(inside)
            if not len(ids):continue
            pixel=uv[ids,1]*camera.width+uv[ids,0]
            order=np.lexsort((ids,q[ids,2],pixel));_,first=np.unique(pixel[order],return_index=True);front=pixel[order[first]]
            foreground=(bits[front//8]&(128>>(front%8)))!=0
            if not foreground.all():return False
        return True
