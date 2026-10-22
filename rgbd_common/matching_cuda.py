"""Shared deterministic local MAP matching, discrete point/pixel-cell support."""
import numpy as np
import torch
from .projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from .observation_model import Matches,covariance,gaussian_log_probability,unassigned_log_probability,select_covariance
from .foreground import check_hard_mask_feasibility,OCCLUDED


def match_rgbd(model,state,frame,config,backend,association_count=None):
    xyz=backend.tensor(model.xyz,backend.input_dtype);group=backend.tensor(model.group,torch.long)
    projected=transport_and_project(xyz,group,backend.tensor(state.poses,backend.input_dtype),frame.camera)
    bins=build_pixel_bins_and_zbuffer(projected,frame.camera,backend.tensor(model.point_id,torch.long))
    feasibility=check_hard_mask_feasibility(model,state,frame,config,backend,projected,bins)
    ids=bins['visible_ids'];pix=projected['pixel'][ids,1]*frame.camera.width+projected['pixel'][ids,0]
    host=pix.detach().cpu().numpy();yes=backend.tensor(frame.valid.ravel()[host],torch.bool)
    ids=ids[yes];pix=pix[yes];rawdepth=backend.tensor(np.asarray(frame.depth).reshape(-1)[pix.detach().cpu().numpy()])
    uv=torch.stack([pix%frame.camera.width,pix//frame.camera.width],dim=1).double()
    rgb=backend.tensor(frame.rgb.reshape(-1,3)[pix.detach().cpu().numpy()])
    observed=torch.cat([uv,rawdepth[:,None],rgb],dim=1)
    r=covariance(config,rawdepth,backend)
    from .observation_model import output
    prediction,_,_=output(state,xyz[ids],backend.tensor(model.rgb)[ids],group[ids],frame.camera,backend)
    count=model.count if association_count is None else association_count
    matched=gaussian_log_probability(observed-prediction,r)+np.log1p(-config.sensor['no_return'])+np.log(config.match_prior/max(1,count))
    unmatched=unassigned_log_probability(frame,config,backend,pix)+np.log1p(-config.match_prior)
    accepted=matched>unmatched
    if not bool(backend.tensor(frame.valid.ravel()[pix[accepted].detach().cpu().numpy()],torch.bool).all()):raise AssertionError('Nonforeground target escaped sparse matching')
    # Hidden points receive no foreground-color factor. Depth discrepancy still
    # participates in the normalized match-vs-unassigned comparison above.
    depth_difference=prediction[:,2]-observed[:,2]
    hidden=(depth_difference>0)&(~accepted)
    stats={'projected':int(projected['valid'].sum()),'visible_pixels':len(bins['visible_ids']),'matched':int(accepted.sum()),'possibly_occluded':int(hidden.sum()),'free_space_rejected':int(((depth_difference<0)&~accepted).sum()),'search':'single calibrated pixel cell; +/- 0.5 pixel quantization support; depth ties by immutable point order','local_assignment':'closest projected depth per pixel; one raw pixel owner; global correspondence optimum not claimed'}
    stats.update(mask_feasible=feasibility['feasible'],mask_constraint_violations=len(feasibility['violation_indices']),undetermined_negative_support=feasibility['undetermined_negative_support'],mask_visibility='V2 M3 foreground likelihood and model self-occlusion; background RGB/depth never used; negative support conditional on declared no-external-occluder premise')
    return Matches(ids[accepted],pix[accepted],observed[accepted],select_covariance(r,accepted),frame.camera,frame.k,frame.ci,unmatched[accepted],matched[accepted],state.poses.copy(),(frame,config)),stats,projected,bins
