"""Conditional map uncertainty and sparse likelihood coupling references.

Motion is conditioned on here. This never updates either motion filter and
never substitutes an analytical Jacobian for the UKF sigma gain.
"""
from pathlib import Path
import json
import numpy as np
import torch
from .observation_model import output,apply_precision
from .matching_cuda import match_rgbd
from .geometry import ObservationIDs
from .backends.torch_cuda import numpy
from .progress import progress


def point_output_jacobian(state,xyz,rgb,group,camera,backend):
    predicted,world,q=output(state,xyz,rgb,group,camera,backend)
    pose=backend.tensor(state.poses)[group.long()];rotation=backend.tensor(camera.world_to_camera[:3,:3])@pose[:,:3,:3]
    fx,fy,_,_=camera.intrinsics
    projection=torch.zeros((len(q),3,3),device=backend.device,dtype=torch.float64)
    projection[:,0,0]=fx/q[:,2];projection[:,1,1]=fy/q[:,2];projection[:,2,2]=1
    projection[:,0,2]=-fx*q[:,0]/q[:,2]**2;projection[:,1,2]=-fy*q[:,1]/q[:,2]**2
    h=torch.zeros((len(q),6,6),device=backend.device,dtype=torch.float64)
    h[:,:3,:3]=projection@rotation;h[:,3:,3:]=torch.eye(3,device=backend.device,dtype=torch.float64)
    return predicted,h


def pose_point_information(state,xyz,rgb,group,camera,sensor_covariance,backend):
    """Selected exact linearized factor blocks, not a UKF motion update.

    Pose columns are left/world translation-first six-dimensional errors;
    twist columns of an instantaneous RGB-D factor are identically zero.
    """
    _,point=point_output_jacobian(state,xyz,rgb,group,camera,backend)
    _,world,q=output(state,xyz,rgb,group,camera,backend)
    rotation=backend.tensor(state.poses)[group.long(),:3,:3]
    derivative=point[:,:3,:3]@rotation.transpose(1,2)
    # derivative is d(u,v,D)/d(world XYZ), including camera calibration.
    skew=torch.zeros((len(world),3,3),dtype=torch.float64,device=backend.device)
    skew[:,0,1]=-world[:,2];skew[:,0,2]=world[:,1];skew[:,1,0]=world[:,2]
    skew[:,1,2]=-world[:,0];skew[:,2,0]=-world[:,1];skew[:,2,1]=world[:,0]
    pose=torch.zeros_like(point);pose[:,:3]=derivative@torch.cat([torch.eye(3,dtype=torch.float64,device=backend.device).expand(len(world),-1,-1),-skew],dim=2)
    weighted_point=apply_precision(sensor_covariance,point);weighted_pose=apply_precision(sensor_covariance,pose)
    return point.transpose(1,2)@weighted_point,point.transpose(1,2)@weighted_pose,pose.transpose(1,2)@weighted_pose


def conditional_map_uncertainty(model,states,dataset,config,backend,directory,save_factors=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    factors=directory/'likelihood_factors'
    if save_factors:factors.mkdir(exist_ok=True)
    prior=backend.tensor(np.linalg.inv(config.map_covariance));mean=backend.tensor(np.c_[model.xyz,model.rgb])
    information=prior.expand(model.count,-1,-1).clone();rhs=-(mean-backend.tensor(config.map_mean))@prior.T
    counts=torch.zeros(model.count,dtype=torch.long,device=backend.device);raw=ObservationIDs(dataset)
    xyz=backend.tensor(model.xyz);rgb=backend.tensor(model.rgb);groups=backend.tensor(model.group,torch.long);total=0
    for k in progress(range(len(states)),desc='Final fixed-pose conditional map covariance',unit='frame'):
        for ci in range(len(dataset.cameras)):
            frame=dataset.observation(k,ci);matches,*_=match_rgbd(model,states[k],frame,config,backend)
            host=numpy(matches.point)
            if len(np.unique(host))!=len(host):raise ValueError('Map covariance requires unique raw ownership and one point sample per camera')
            pixels=numpy(matches.pixel)
            if save_factors:np.savez(factors/f'factor_{k:06d}_{ci:03d}.npz',point_index=host,point_id=model.point_id[host],raw_id=raw.encode(k,ci,pixels),group=model.group[host])
            for match in matches.chunks(backend.chunk_size(4096)):
                ids=match.point;prediction,h=point_output_jacobian(states[k],xyz[ids],rgb[ids],groups[ids],match.camera,backend)
                weighted=apply_precision(match.covariance,h)
                weighted_error=apply_precision(match.covariance,(match.observed-prediction)[...,None])
                information[ids]+=h.transpose(1,2)@weighted
                rhs[ids]+=(h.transpose(1,2)@weighted_error)[:,:,0];counts[ids]+=1;total+=len(ids)
    covariance=[];offset=[]
    size=backend.chunk_size(4096)
    for start in range(0,model.count,size):
        local=information[start:start+size];local=(local+local.transpose(1,2))/2
        root=torch.linalg.cholesky(local);inverse=torch.cholesky_inverse(root)
        covariance.append(numpy(inverse));offset.append(numpy(torch.cholesky_solve(rhs[start:start+size,:,None],root)[:,:,0]))
    covariance=np.concatenate(covariance);offset=np.concatenate(offset);accepted=np.c_[model.xyz,model.rgb]
    np.savez(directory/'conditional_map_covariance.npz',point_id=model.point_id,accepted_nominal_xyz_rgb=accepted,
             linearized_conditional_mean=accepted+offset,conditional_covariance=covariance,mean_offset_from_accepted=offset,
             error_second_moment_about_accepted=covariance+offset[:,:,None]*offset[:,None,:],owned_raw_factor_count=numpy(counts),
             semantics=np.array('Exact inverse of the chosen linearized RAW RGB-D point information plus original geometry prior, conditioning on accepted poses/calibration/groups/one-owner matches. Not nonlinear-exact or a pose-marginal covariance. The saved Gaussian mean can differ from the accepted nominal map.'))
    report={'points':model.count,'owned_raw_factors':total,'unobserved_points':int((numpy(counts)==0).sum()),
            'file':'conditional_map_covariance.npz','factor_directory':'likelihood_factors' if save_factors else None,'joint_map_motion_marginals_computed':False,
            'linearization':'Accepted solved_points and group_poses; full 6D raw sensor R and original proper map prior once, including original source pixels only when selected as actual owners',
            'coupling':'Each raw factor retains its (frame,group,point_id) keys. Use pose_point_information with these raw factors to obtain point/motion cross-information; do not replace shared motion errors with independent per-point noise.',
            'qualification':'Fixed-pose/fixed-assignment first-order Gaussian. Own IEKF/UKF motion covariance is exported separately. No posterior independence after marginalizing poses is asserted.'}
    (directory/'index.json').write_text(json.dumps(report,indent=2)+'\n');return report
