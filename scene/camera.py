"""Discrete point transport, calibrated projection and deterministic depth bins."""
from dataclasses import dataclass
import numpy as np
import torch

PROJECTION_OPERATOR_VERSION=2


@dataclass
class Camera:
    name:str
    width:int
    height:int
    intrinsics:np.ndarray
    world_to_camera:np.ndarray
    near:float
    far:float


def rigid_transform(points,rotation,translation):
    """Fixed per-coordinate arithmetic, invariant to point batch size.

    BLAS chooses shape-dependent float32 contraction orders. At a half-pixel
    boundary that changes the discrete factor graph, breaking incremental J.
    Separate elementwise operations preserve the same order for every point.
    """
    return ((rotation[..., :,0]*points[...,0,None]+rotation[..., :,1]*points[...,1,None])+rotation[..., :,2]*points[...,2,None])+translation


def transport_and_project(xyz,group,poses,camera,mask=None):
    t=poses[group.long()];world=rigid_transform(xyz,t[:,:3,:3],t[:,:3,3])
    rc=torch.as_tensor(camera.world_to_camera[:3,:3],dtype=xyz.dtype,device=xyz.device)
    tc=torch.as_tensor(camera.world_to_camera[:3,3],dtype=xyz.dtype,device=xyz.device)
    q=rigid_transform(world,rc,tc)
    fx,fy,cx,cy=map(float,camera.intrinsics)
    safe=torch.where(q[:,2]!=0,q[:,2],torch.ones_like(q[:,2]))
    uv=q[:,:2]/safe[:,None]*xyz.new_tensor([fx,fy])+xyz.new_tensor([cx,cy])
    pixel=torch.floor(uv+.5).long()
    valid=(q[:,2]>camera.near)&(q[:,2]<camera.far)&torch.isfinite(uv).all(1)
    valid&=(pixel[:,0]>=0)&(pixel[:,0]<camera.width)&(pixel[:,1]>=0)&(pixel[:,1]<camera.height)
    if mask is not None:
        ids=torch.nonzero(valid,as_tuple=True)[0];valid[ids]&=mask[pixel[ids,1],pixel[ids,0]]
    return {'world':world,'camera':q,'uv':uv,'depth':q[:,2],'pixel':pixel,'valid':valid}


def build_pixel_bins_and_zbuffer(projected,camera,point_ids=None):
    ids=torch.nonzero(projected['valid'],as_tuple=True)[0]
    if point_ids is not None:ids=ids[torch.argsort(point_ids[ids],stable=True)]
    ids=ids[torch.argsort(projected['depth'][ids],stable=True)]
    key=projected['pixel'][ids,1]*camera.width+projected['pixel'][ids,0]
    order=torch.argsort(key,stable=True);ids=ids[order];key=key[order]
    unique,counts=torch.unique_consecutive(key,return_counts=True)
    offsets=torch.cat([torch.zeros(1,dtype=torch.long,device=ids.device),counts.cumsum(0)])
    owner=torch.full((camera.height*camera.width,),-1,dtype=torch.long,device=ids.device)
    if len(unique):owner[unique]=ids[offsets[:-1]]
    depth=torch.full((camera.height*camera.width,),torch.inf,dtype=projected['depth'].dtype,device=ids.device)
    if len(unique):depth[unique]=projected['depth'][ids[offsets[:-1]]]
    return {'pixel_keys':unique,'offsets':offsets,'candidate_ids':ids,'owner':owner,'first_index':owner.reshape(camera.height,camera.width),'first_depth':depth.reshape(camera.height,camera.width),'visible_ids':ids[offsets[:-1]]}


def inverse_transport_unmatched(world,source_pose,target_pose):
    canonical=(world-source_pose[:3,3])@source_pose[:3,:3]
    return canonical@target_pose[:3,:3].T+target_pose[:3,3]
