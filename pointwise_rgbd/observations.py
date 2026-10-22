from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree
from .projection import backproject
from .state6 import anchor_posterior


@dataclass
class Samples:
    frame: object
    pixel: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    covariance: np.ndarray
    R: np.ndarray
    rgb: np.ndarray
    tree: object
    index_build_seconds: float = 0.


def samples(frame,config):
    pixel=np.flatnonzero(frame.valid.ravel())
    depth=np.asarray(frame.depth).ravel()[pixel];w=frame.camera.width
    z=np.c_[pixel%w,pixel//w,depth];r=config.measurement_covariance(depth)
    xyz,covariance=backproject(z,frame.camera,r)
    start=time.monotonic();tree=cKDTree(xyz) if len(xyz) else None
    return Samples(frame,pixel,z,xyz,covariance,r,np.asarray(frame.rgb).reshape(-1,3)[pixel],tree,time.monotonic()-start)


def initialize_all_frame_zero(dataset,config):
    states=[];covariances=[];colors=[];sources=[]
    for ci in range(len(dataset.cameras)):
        raw=samples(dataset.observation(0,ci),config)
        state,p=anchor_posterior(raw.xyz,raw.covariance,config.velocity_covariance)
        states.append(state);covariances.append(p);colors.append(raw.rgb)
        sources.append(np.c_[np.zeros(len(raw.pixel),dtype=np.int64),np.full(len(raw.pixel),ci),raw.pixel])
    if not sum(map(len,states)):raise ValueError('No valid foreground observation at frame zero')
    # No spatial downsampling or uncertain duplicate merges: every raw source
    # has its own representative. Coincident camera samples remain explicit.
    return np.concatenate(states),np.concatenate(covariances),np.concatenate(colors),np.concatenate(sources).astype(np.int64)


def initialize_seed(ref,dataset,config):
    frame=dataset.observation(ref.frame,ref.camera)
    if not frame.valid.ravel()[ref.pixel]:raise ValueError('Seed is not a valid masked real RGB-D observation')
    y,x=divmod(ref.pixel,frame.camera.width);z=np.array([x,y,float(frame.depth[y,x])])
    xyz,cov=backproject(z,frame.camera,config.measurement_covariance(z[2]))
    state,p=anchor_posterior(xyz,cov,config.velocity_covariance)
    return state,p,np.asarray(frame.rgb[y,x])
