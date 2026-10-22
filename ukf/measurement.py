"""Conditional raw-point sigma projection and streaming Woodbury sums."""
import numpy as np
import torch
from rgbd_common.observation_model import select_covariance
from rgbd_common.projection_cuda import rigid_transform
from .sigma_points import sigma_points


def woodbury(a,pminus,m,b):
    solved=np.linalg.solve(m,b);inverse=np.linalg.solve(m,np.eye(len(m)))
    delta=a@solved
    covariance=pminus-a@(np.eye(len(m))-inverse)@a.T
    return delta,(covariance+covariance.T)/2


def project_sigma(poses,xyz,rgb,group,camera,backend,include_rgb=True):
    """Batched nonlinear h for bounded point chunks, no measurement Jacobian."""
    selected=poses[:,group.long()]
    world=rigid_transform(xyz.double()[None],selected[:,:,:3,:3],selected[:,:,:3,3])
    cw=backend.tensor(camera.world_to_camera);q=rigid_transform(world,cw[:3,:3],cw[:3,3])
    fx,fy,cx,cy=map(float,camera.intrinsics)
    uv=q[:,:,:2]/q[:,:,2,None]*q.new_tensor([fx,fy])+q.new_tensor([cx,cy])
    value=torch.cat([uv,q[:,:,2,None]],dim=2)
    if include_rgb:value=torch.cat([value,rgb.double()[None].expand(len(poses),-1,-1)],dim=2)
    pixel=torch.floor(uv+.5).long()
    valid=(q[:,:,2]>camera.near)&(q[:,:,2]<camera.far)&torch.isfinite(uv).all(2)
    valid&=(pixel[:,:,0]>=0)&(pixel[:,:,0]<camera.width)&(pixel[:,:,1]>=0)&(pixel[:,:,1]<camera.height)
    return value,valid


def measurement_moments(problem,base,pminus,parameters=(1.,2.,0.)):
    eta,wm,wc=sigma_points(pminus,*parameters);states=[base.plus(e) for e in eta]
    deviations=np.array([s.minus(base) for s in states])
    if not np.allclose(deviations,eta,rtol=1e-9,atol=1e-10):raise ValueError('SIGMA_CHART_WRAP: sigma perturbations cross the selected logarithm chart')
    a=deviations.T*np.sqrt(wc)[None,:];backend=problem.backend;l=len(states)
    m=torch.eye(l,dtype=torch.float64,device=backend.device);b=torch.zeros(l,dtype=torch.float64,device=backend.device)
    weights=backend.tensor(wm);root=backend.tensor(np.sqrt(wc));active=0;deferred=0
    poses=backend.tensor(np.array([state.poses for state in states]))
    for match in problem.chunks(bytes_per_item=max(2048,l*6*8*4)):
        # The updated execution request fixes the conditional raw association.
        # Sigma samples evaluate h; they do not re-observe/reassign the scene.
        # Physical visibility and mask feasibility belong to outer trial poses.
        keep=torch.ones(len(match),dtype=torch.bool,device=backend.device)
        active+=len(match)
        if not len(match):continue
        ids=match.point
        h,_=project_sigma(poses,problem.xyz[ids],problem.rgb[ids],problem.group[ids],match.camera,backend,include_rgb=False)
        if not torch.isfinite(h).all() or torch.any(h[:,:,2]<=0):
            raise ValueError('UKF_CONDITIONAL_PROJECTION_UNDEFINED: sigma ensemble crosses the positive-depth projection domain; no factors or sigma weights were dropped or clipped')
        mean=torch.einsum('l,lnd->nd',weights,h)
        z=(h-mean[None])*root[:,None,None];z=z.permute(1,2,0)
        # RGB is exactly constant over this sigma ensemble. Its covariance
        # with the pose-dependent output is zero, but correlated sensor RGB
        # errors still contribute through the FULL six-dimensional precision.
        prediction=torch.cat([mean,problem.rgb[ids].double()],dim=1)
        residual=match.observed[keep]-prediction;r=select_covariance(match.covariance,keep)
        if r.ndim==2 or r.stride(0)==0:
            small=r if r.ndim==2 else r[0]
            inverse=torch.cholesky_inverse(torch.linalg.cholesky(small))
            weighted=torch.einsum('ij,njl->nil',inverse[:3,:3],z)
            weighted_residual=residual@inverse[:3,:].T
        else:
            inverse=torch.cholesky_inverse(torch.linalg.cholesky(r))
            weighted=inverse[:,:3,:3]@z
            weighted_residual=(inverse[:,:3,:]@residual[:,:,None])[:,:,0]
        m+=z.reshape(-1,l).T@weighted.reshape(-1,l)
        b+=z.reshape(-1,l).T@weighted_residual.reshape(-1)
    return a,m,b,{'sigma_count':l,'active_factors':active,'deferred_visibility_factors':deferred,'innovation_dimension':l,'visibility_qualification':'Latest execution request Section 2: fixed nominal foreground raw IDs, conditional continuous M4 quadrature; sigma raster-bin changes do not remove information. Physical mask/occlusion gates apply in outer association and accepted trial feasibility. Undefined positive-depth projection raises an explicit error.'}
