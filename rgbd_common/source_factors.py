"""Sparse source-pose handles for conditional point fusion.

These are derivatives of the ORIGINAL coordinate-fusion factors, not extra
independent observations. No marginal can be inferred from the diagonal point
blocks alone: queries require the joint source-noise/pose covariance, including
cross terms when the poses were inferred from the same raw observations.
"""
from pathlib import Path
import json
import numpy as np
from .geometry import ObservationIDs
from .fusion import backproject,canonicalize
from .lie import skew
from .progress import progress


def source_jacobians(world,pose):
    rotation=pose[:3,:3].T
    measurement=np.eye(6);measurement[:3,:3]=rotation
    motion=np.zeros((6,6));motion[:3]=rotation@np.c_[-np.eye(3),skew(world)]
    return measurement,motion


def augmentation_blocks(measurement_jacobian,pose_jacobian,joint_covariance):
    """Exact first-order pushforward of a supplied JOINT 6D raw/6D pose law."""
    from .config import spd
    joint=spd(joint_covariance,12,'joint raw/source-pose covariance',semidefinite=True)
    push=np.c_[measurement_jacobian,pose_jacobian]
    return push@joint@push.T,push@joint[:,6:]


def export_source_factors(model,states,dataset,config,backend,directory):
    """Stream CSR factors; storage scales with actual fused provenance only.

    For conditional precision fusion, dp=sum_s W_s (Jy_s dy_s+JT_s dT_s),
    W_s=C_cond C_s^-1. Shared pose keys and immutable raw keys are retained;
    neither pose covariance nor shared raw covariance is copied per point.
    The original proper geometry prior supplies the remaining precision.
    """
    directory=Path(directory);directory.mkdir(exist_ok=True)
    model.source_owners([]);raw=ObservationIDs(dataset)
    ids=model.provenance.raw_ids;count=len(ids)
    points=np.repeat(np.arange(model.count,dtype=np.int64),model.provenance.lengths)
    times,cameras,pixels=raw.decode(ids);shards=[]
    information=np.broadcast_to(np.linalg.inv(config.map_covariance),(model.count,6,6)).copy()
    rhs=np.broadcast_to(np.linalg.solve(config.map_covariance,config.map_mean),(model.count,6)).copy()
    for k in progress(np.unique(times),desc='Export shared source-pose factors',unit='time'):
        for ci in np.unique(cameras[times==k]):
            take=np.flatnonzero((times==k)&(cameras==ci));indices=points[take]
            world,color,noise,_=backproject(dataset.observation(int(k),int(ci)),config,backend,pixels[take])
            canonical=np.empty_like(world)
            labels=model.group[indices];jy=np.zeros((len(take),6,6));jt=jy.copy();conditional=jy.copy()
            for g in np.unique(labels):
                use=labels==g;pose=states[int(k)].poses[int(g)]
                canonical[use],conditional[use]=canonicalize(world[use],noise[use],pose)
                rotation=pose[:3,:3].T;jy[use]=np.eye(6);jy[use,:3,:3]=rotation
                w=world[use];hat=np.zeros((len(w),3,3));hat[:,0,1]=-w[:,2];hat[:,0,2]=w[:,1];hat[:,1,0]=w[:,2];hat[:,1,2]=-w[:,0];hat[:,2,0]=-w[:,1];hat[:,2,1]=w[:,0]
                jt[use,:3,:3]=-rotation;jt[use,:3,3:]=rotation@hat
            precision=np.linalg.inv(conditional)
            # Rebuild the source-coordinate law in the accepted pose chart.
            # The old per-point fusion covariance can predate motion updates.
            np.add.at(information,indices,precision)
            np.add.at(rhs,indices,np.einsum('nij,nj->ni',precision,np.c_[canonical,color]))
            path=directory/f'source_{int(k):06d}_{int(ci):03d}.npz'
            np.savez(path,point_index=indices,point_id=model.point_id[indices],raw_id=ids[take],source_time=times[take],source_group=labels,
                     measurement_covariance_world_xyz_rgb=noise,measurement_jacobian=jy,pose_jacobian=jt)
            shards.append({'file':path.name,'factors':len(take)})
    conditional=np.linalg.inv(information);source_mean=np.einsum('nij,nj->ni',conditional,rhs)
    accepted=np.c_[model.xyz,model.rgb];offset=source_mean-accepted
    np.savez(directory/'source_coordinate_gaussian.npz',point_id=model.point_id,conditional_covariance=conditional,
             source_coordinate_mean=source_mean,accepted_nominal_xyz_rgb=accepted,mean_offset_from_accepted=offset,
             error_second_moment_about_accepted=conditional+offset[:,:,None]*offset[:,None,:],
             semantics=np.array('Relinearized registered-source coordinate Gaussian, original proper geometry prior once, conditioning on accepted poses and frozen backprojection covariance. Its mean need not be the accepted map. This is an initialization-reference law, not the final raw-factor map posterior or a motion-marginal covariance.'))
    for shard in progress(shards,desc='Write relinearized source coupling',unit='shard'):
        path=directory/shard['file']
        with np.load(path) as source:data={key:source[key] for key in source.files}
        jy=data['measurement_jacobian'];jt=data['pose_jacobian'];noise=data['measurement_covariance_world_xyz_rgb']
        precision=np.linalg.inv(jy@noise@jy.transpose(0,2,1))
        weight=conditional[data['point_index']]@precision
        np.savez(path,**data,fusion_weight=weight,point_raw_jacobian=weight@jy,point_pose_jacobian=weight@jt)
    report={'factors':count,'points':model.count,'shards':shards,'pose_key':'(source_time, source_group), shared by all incident points',
            'raw_key':'immutable raw_id; shared, never duplicated as independent noise',
            'semantics':'Relinearized first-order registered-source coordinate-fusion sensitivities at accepted poses; the original proper geometry prior is included once. Gaussian center is saved in source_coordinate_gaussian.npz and may differ from the accepted map. The shared source-pose factors retain cross-point and point-pose dependence. They are not new likelihoods or independent landmark priors.',
            'marginals':'Not computed. A joint query must supply the source-pose covariance AND raw/pose cross moments; a pose inferred from the same observations cannot be assumed independent.',
            'units':'XYZ meters, RGB encoded [0,1]; pose left world [meters,radians]',
            'linearization_poses':'accepted group_poses.npz; frozen source covariance convention; no claim of marginal posterior or of equality between Gaussian and accepted means',
            'coordinate_gaussian':'source_coordinate_gaussian.npz','current_raw_map_covariance':'../map_uncertainty/conditional_map_covariance.npz'}
    (directory/'index.json').write_text(json.dumps(report,indent=2)+'\n');return report
