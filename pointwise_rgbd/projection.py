import numpy as np
import torch


def project(state,camera):
    rotation=state.new_tensor(camera.world_to_camera[:3,:3]);translation=state.new_tensor(camera.world_to_camera[:3,3])
    q=state[...,:3]@rotation.T+translation
    x,y,z=q.unbind(-1);fx,fy,cx,cy=map(float,camera.intrinsics)
    h=torch.stack((fx*x/z+cx,fy*y/z+cy,z),-1)
    j=torch.zeros((*q.shape[:-1],3,3),device=state.device,dtype=state.dtype)
    j[...,0,0]=fx/z;j[...,0,2]=-fx*x/z**2;j[...,1,1]=fy/z;j[...,1,2]=-fy*y/z**2;j[...,2,2]=1
    out=torch.zeros((*q.shape[:-1],3,6),device=state.device,dtype=state.dtype);out[...,:,:3]=j@rotation
    return h,out


def backproject(z,camera,measurement_covariance):
    z=np.asarray(z,dtype=np.float64);fx,fy,cx,cy=camera.intrinsics
    ray=np.stack(((z[...,0]-cx)/fx,(z[...,1]-cy)/fy,np.ones(z.shape[:-1])),-1)
    xyz=(ray*z[...,2,None]-camera.world_to_camera[:3,3])@camera.world_to_camera[:3,:3]
    j=np.zeros((*z.shape[:-1],3,3));j[...,0,0]=z[...,2]/fx;j[...,1,1]=z[...,2]/fy;j[...,0,2]=ray[...,0];j[...,1,2]=ray[...,1];j[...,2,2]=1
    j=camera.world_to_camera[:3,:3].T@j
    return xyz,j@measurement_covariance@j.swapaxes(-1,-2)
