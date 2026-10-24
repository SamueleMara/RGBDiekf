"""V2 M3 nominal point z-buffer; raw background is never consulted."""
import numpy as np
import torch
from rgbd_common.projection_cuda import rigid_transform,build_pixel_bins_and_zbuffer
from .association import associate


def project_bins(state,camera,backend,point_ids=None):
    x=state[:,:3];r=backend.tensor(camera.world_to_camera[:3,:3]);t=backend.tensor(camera.world_to_camera[:3,3]);q=rigid_transform(x,r,t)
    z=q[:,2];safe=torch.where(z!=0,z,torch.ones_like(z));uv=q[:,:2]/safe[:,None]*backend.tensor(camera.intrinsics[:2])+backend.tensor(camera.intrinsics[2:])
    pixel=torch.floor(uv+.5).long()
    valid=torch.isfinite(q).all(1)&torch.isfinite(uv).all(1)&(z>camera.near)&(z<camera.far)&(pixel[:,0]>=0)&(pixel[:,0]<camera.width)&(pixel[:,1]>=0)&(pixel[:,1]<camera.height)
    projected={'depth':z,'uv':uv,'pixel':pixel,'valid':valid}
    return projected,build_pixel_bins_and_zbuffer(projected,camera,point_ids)


def associate_visible(states,covariances,colors,point_ids,frame,config,backend,owners):
    if frame.valid.any() and len(states):
        _,bins=project_bins(backend.tensor(states),frame.camera,backend,backend.tensor(point_ids,torch.long));visible=bins['visible_ids'].cpu().numpy()
    else:visible=np.empty(0,np.int64)
    result,stats=associate(np.asarray(states)[visible],np.asarray(covariances)[visible],np.asarray(colors)[visible],np.asarray(point_ids)[visible],frame,config,backend,owners)
    result['point']=backend.tensor(visible,torch.long)[result['point']]
    stats.update(model_front_points=len(visible),model_nonfront_or_outside=len(states)-len(visible))
    return result,stats
