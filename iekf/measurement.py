"""Analytical 6D point-output Jacobian and CUDA information reductions."""
import torch
from rgbd_common.observation_model import output,apply_precision


def skew_batch(x):
    s=torch.zeros((len(x),3,3),device=x.device,dtype=x.dtype)
    s[:,0,1]=-x[:,2];s[:,0,2]=x[:,1];s[:,1,0]=x[:,2];s[:,1,2]=-x[:,0];s[:,2,0]=-x[:,1];s[:,2,1]=x[:,0]
    return s


def h_H(state,xyz,rgb,group,camera,backend,pose_only=False):
    predicted,x,q=output(state,xyz,rgb,group,camera,backend)
    rc=backend.tensor(camera.world_to_camera[:3,:3])
    j3=rc@torch.cat([torch.eye(3,dtype=x.dtype,device=x.device).expand(len(x),3,3),-skew_batch(x)],dim=2)
    fx,fy,_,_=camera.intrinsics
    jp=torch.zeros((len(x),2,3),dtype=x.dtype,device=x.device)
    jp[:,0,0]=fx/q[:,2];jp[:,1,1]=fy/q[:,2];jp[:,0,2]=-fx*q[:,0]/q[:,2]**2;jp[:,1,2]=-fy*q[:,1]/q[:,2]**2
    h=torch.zeros((len(x),6,6 if pose_only else 12),dtype=x.dtype,device=x.device)
    h[:,:2,:6]=jp@j3;h[:,2,:6]=j3[:,2]
    return predicted,h


def reduce_information(problem,state):
    b=problem.backend;groups=len(state.poses)
    information=torch.zeros((groups,12,12),device=b.device,dtype=torch.float64)
    score=torch.zeros((groups,12),device=b.device,dtype=torch.float64)
    for match in problem.chunks():
        gid=problem.group[match.point]
        predicted,h=h_H(state,problem.xyz[match.point],problem.rgb[match.point],gid,match.camera,b,pose_only=True)
        residual=match.observed-predicted
        weighted_h=apply_precision(match.covariance,h)
        weighted_r=apply_precision(match.covariance,residual.unsqueeze(-1)).squeeze(-1)
        for g in range(groups):
            select=gid==g
            hg=h[select].reshape(-1,6);wh=weighted_h[select].reshape(-1,6)
            information[g,:6,:6]+=hg.T@wh;score[g,:6]+=hg.T@weighted_r[select].reshape(-1)
    return information,score
