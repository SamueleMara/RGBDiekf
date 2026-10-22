"""The common six-dimensional CONDITIONAL raw matched-point observation."""
from dataclasses import dataclass
import numpy as np
import torch
from .projection_cuda import rigid_transform


def select_covariance(value,selection):
    if isinstance(selection,slice) or value.shape[0]==0:return value[selection]
    if value.ndim==3 and value.stride(0)==0:
        count=int(selection.sum()) if selection.dtype==torch.bool else len(selection)
        return value[0].expand(count,-1,-1)
    return value[selection]


def apply_precision(value,rhs):
    """Reuse one small SPD factor iff the model actually shares that matrix.

    This does not turn a depth-dependent law into constant noise. The full
    heteroscedastic/correlated block path is retained otherwise.
    """
    if rhs.shape[0]==0:return rhs.clone()
    if value.ndim==2 or value.stride(0)==0:
        small=value if value.ndim==2 else value[0]
        inverse=torch.cholesky_inverse(torch.linalg.cholesky(small))
        n,d,columns=rhs.shape
        return (rhs.transpose(1,2).reshape(-1,d)@inverse.T).reshape(n,columns,d).transpose(1,2)
    return torch.linalg.solve(value,rhs)


def gaussian_log_probability(residual,covariance):
    if len(residual)==0:return residual.new_empty((0,))
    solved=apply_precision(covariance,residual.unsqueeze(-1)).squeeze(-1)
    small=covariance if covariance.ndim==2 else (covariance[0] if covariance.stride(0)==0 else covariance)
    return -.5*((residual*solved).sum(-1)+torch.linalg.slogdet(small).logabsdet+residual.shape[-1]*np.log(2*np.pi))


def covariance(config,depth,backend):
    base=backend.tensor(config.rgbd_covariance);rate=backend.tensor(config.rgbd_depth_squared)
    order=backend.tensor([3,0,1,2],torch.long)
    if not np.any(config.rgbd_depth_squared):
        small=torch.zeros((6,6),dtype=torch.float64,device=depth.device)
        small[:2,:2]=backend.tensor(config.pixel_covariance);small[2:,2:]=base[order[:,None],order[None,:]]
        return small.expand(len(depth),-1,-1)
    rgbd=base+depth[:,None,None]**2*rate
    out=torch.zeros((len(depth),6,6),dtype=torch.float64,device=depth.device)
    out[:,:2,:2]=backend.tensor(config.pixel_covariance)
    out[:,2:,2:]=rgbd[:,order[:,None],order[None,:]]
    return out


def output(state,xyz,rgb,group,camera,backend):
    poses=backend.tensor(state.poses);t=poses[group.long()];x=rigid_transform(xyz.double(),t[:,:3,:3],t[:,:3,3])
    cw=backend.tensor(camera.world_to_camera);q=rigid_transform(x,cw[:3,:3],cw[:3,3])
    fx,fy,cx,cy=map(float,camera.intrinsics)
    uv=q[:,:2]/q[:,2,None]*q.new_tensor([fx,fy])+q.new_tensor([cx,cy])
    return torch.cat([uv,q[:,2,None],rgb.double()],dim=1),x,q


def unassigned_log_probability(frame,config,backend,pixels=None):
    if pixels is None:
        domain=np.flatnonzero(frame.valid.ravel())
        value=torch.full((frame.camera.width*frame.camera.height,),-torch.inf,dtype=torch.float64,device=backend.device)
        value[backend.tensor(domain,torch.long)]=unassigned_log_probability(frame,config,backend,domain)
        return value
    index=pixels.detach().cpu().numpy() if isinstance(pixels,torch.Tensor) else np.asarray(pixels)
    if not np.all(frame.valid.ravel()[index]):raise ValueError('Background or invalid depth cannot supply an unassigned foreground RGB-D likelihood')
    rgb=backend.tensor(frame.rgb.reshape(-1,3)[index]);depth=backend.tensor(frame.depth.reshape(-1)[index])
    value=torch.full_like(depth,np.log(config.sensor['background_no_return']))
    valid=torch.isfinite(depth)&(depth>0)
    if valid.any():
        observed=torch.cat([rgb[valid],depth[valid,None]],dim=1)
        value[valid]=np.log1p(-config.sensor['background_no_return'])+gaussian_log_probability(observed-backend.tensor(config.sensor['background_mean']),backend.tensor(config.sensor['background_covariance']))
    return value-np.log(frame.camera.width*frame.camera.height)


@dataclass
class Matches:
    point:torch.Tensor
    pixel:torch.Tensor
    observed:torch.Tensor
    covariance:torch.Tensor
    camera:object
    frame:int
    camera_index:int
    unmatched_score:torch.Tensor
    assignment_score:torch.Tensor
    association_poses:object=None
    mask_context:object=None

    def __len__(self):return len(self.point)
    def chunks(self,size):
        for start in range(0,len(self),size):
            section=slice(start,start+size)
            yield Matches(self.point[section],self.pixel[section],self.observed[section],self.covariance[section],self.camera,self.frame,self.camera_index,self.unmatched_score[section],self.assignment_score[section],self.association_poses,self.mask_context)
