"""Nonlinear GPU sigma projection, visibility deferral, streaming Woodbury sums."""
import numpy as np
import torch
from rgbd_common.observation_model import output,apply_precision,select_covariance
from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from ukf.sigma_points import sigma_points


def woodbury(a,pminus,m,b):
    solved=np.linalg.solve(m,b);inverse=np.linalg.solve(m,np.eye(len(m)))
    delta=a@solved
    covariance=pminus-a@(np.eye(len(m))-inverse)@a.T
    return delta,(covariance+covariance.T)/2


def measurement_moments(problem,base,pminus,parameters=(1.,2.,0.)):
    eta,wm,wc=sigma_points(pminus,*parameters);states=[base.plus(e) for e in eta]
    deviations=np.array([s.minus(base) for s in states])
    if not np.allclose(deviations,eta,rtol=1e-9,atol=1e-10):raise ValueError('SIGMA_CHART_WRAP: sigma perturbations cross the selected logarithm chart')
    a=deviations.T*np.sqrt(wc)[None,:];backend=problem.backend;l=len(states)
    m=torch.eye(l,dtype=torch.float64,device=backend.device);b=torch.zeros(l,dtype=torch.float64,device=backend.device)
    weights=backend.tensor(wm);root=backend.tensor(np.sqrt(wc));active=0;deferred=0
    for match in problem.chunks(bytes_per_item=max(2048,l*6*8*4)):
        keep=torch.ones(len(match),dtype=torch.bool,device=backend.device)
        active+=int(keep.sum());deferred+=int((~keep).sum())
        if not keep.any():continue
        ids=match.point[keep];values=[]
        for state in states:values.append(output(state,problem.xyz[ids],problem.rgb[ids],problem.group[ids],match.camera,backend)[0])
        h=torch.stack(values)
        if not torch.isfinite(h).all() or torch.any(h[:,:,2]<=0):raise ValueError('UKF_CONDITIONAL_PROJECTION_UNDEFINED')
        mean=torch.einsum('l,lnd->nd',weights,h)
        z=(h-mean[None])*root[:,None,None];z=z.permute(1,2,0)
        residual=match.observed[keep]-mean;r=select_covariance(match.covariance,keep)
        weighted=apply_precision(r,z);weighted_residual=apply_precision(r,residual.unsqueeze(-1)).squeeze(-1)
        m+=z.reshape(-1,l).T@weighted.reshape(-1,l)
        b+=z.reshape(-1,l).T@weighted_residual.reshape(-1)
    return a,m,b,{'sigma_count':l,'active_factors':active,'deferred_visibility_factors':deferred,'innovation_dimension':l,'visibility_qualification':'Fixed external competing-point bins across quadrature; sigma frustum and depth-order changes defer factors. New pixel-bin overlaps are left to external matching/global acceptance, an explicitly conditional visibility-graph approximation.'}
