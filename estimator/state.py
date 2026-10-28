from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True,order=True)
class ObservationRef:
    frame: int
    camera: int
    pixel: int

    def __post_init__(self):
        for name in ('frame','camera','pixel'):
            value=getattr(self,name)
            if int(value)!=value or value<0:raise ValueError('Observation identities must be nonnegative integers')
            object.__setattr__(self,name,int(value))

    def record(self,dataset):
        return {'frame':self.frame,'original_frame':dataset.frame_ids[self.frame],'camera':self.camera,'camera_id':dataset.cameras[self.camera].name,'pixel':self.pixel}


def anchor_posterior(xyz,position_covariance,velocity_covariance):
    xyz=np.asarray(xyz);shape=xyz.shape[:-1]
    state=np.zeros((*shape,6));state[...,:3]=xyz
    p=np.zeros((*shape,6,6));p[...,:3,:3]=position_covariance;p[...,3:,3:]=velocity_covariance
    return state,p
