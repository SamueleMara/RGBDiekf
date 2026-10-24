#!/usr/bin/env python3
"""Audit actual bounded grouped histories, rigidity, raw factors and overlays."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.diagnostics import distribution
from pointwise_rgbd.store import json_write
from rgbd_common.array_store import ArrayStore


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();root=args.run;epoch=root/'phase1/partition_v000';manifest=json.loads((root/'manifest.json').read_text());meta=json.loads((epoch/'summary.json').read_text());data=Dataset(manifest['dataset']['input'],**manifest['dataset']['test_limits'])
    with np.load(epoch/'partition.npz') as saved:partition={key:saved[key] for key in saved.files}
    arrays={name:np.load(epoch/f'{name}.npy',mmap_mode='r') for name in ('xyz','group_poses','covariance','point_observed','group_measurements','pose_rank')}
    members=partition['membership'];order=np.argsort(members,kind='stable');keys,first=np.unique(members[order],return_index=True);last=np.r_[first[1:],len(order)];indices=[order[a:b] for a,b in zip(first,last)];canonical=[pdist(partition['xyz'][idx]) for idx in indices]
    factor_store=ArrayStore(epoch/'actual_measurement_factors.sqlite');max_pair_error=0.;max_transport_error=0.;residual=[];normalized=[];frames=[];unique_factors=0;iteration_limits=0;count_pairs=0
    snapshots={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in epoch.glob('*.npy')}
    for k in range(len(data.times)):
        xyz=arrays['xyz'][k];pose=arrays['group_poses'][k]
        for g,idx in enumerate(indices):
            if len(idx)>1:max_pair_error=max(max_pair_error,float(abs(pdist(xyz[idx])-canonical[g]).max()));count_pairs+=len(canonical[g])
            expected=partition['xyz'][idx]@pose[g,:3,:3].T+pose[g,:3,3];max_transport_error=max(max_transport_error,float(abs(xyz[idx]-expected).max()))
        if factor_store.contains(str(k)):
            factors=factor_store.read(str(k));identity=np.c_[factors['camera'],factors['pixel']]
            if len(np.unique(identity,axis=0))!=len(identity):raise AssertionError('Repeated raw observation factor')
            for ci in np.unique(factors['camera']):
                use=factors['camera']==ci
                if not data.observation(k,int(ci)).valid.ravel()[factors['pixel'][use]].all():raise AssertionError('Background or invalid raw factor')
            error=factors['residual'];residual.append(error);normalized.append(np.einsum('ni,ni->n',error,np.linalg.solve(factors['R'],error[:,:,None])[...,0]));unique_factors+=len(error)
            optimizer=json.loads((epoch/f'filter/{k:06d}.json').read_text());iteration_limits+=int(np.array(optimizer['iteration_limit']).sum())
        frames.append(json.loads((epoch/f'frames/{k:06d}.json').read_text()))
    errors=np.concatenate(residual);d2=np.concatenate(normalized);counts=np.bincount(members);rotation=arrays['group_poses'][:,:,:3,:3]
    orthogonality=float(abs(rotation.swapaxes(-1,-2)@rotation-np.eye(3)).max());evidence=arrays['group_measurements']>0
    report={'status':'GATE_A_BOUNDED_DIAGNOSTICS_ONLY','source_frames':data.frame_ids,'anchor_original_frame':meta['anchor_original_frame'],'point_count':len(members),'groups':len(counts),'group_size':distribution(counts),'group_radius_m':distribution(partition['radius']),'isolated_single_sample_groups':int((counts==1).sum()),'all_intragroup_pairs_checked_over_frames':count_pairs,'max_pair_distance_error_m':max_pair_error,'max_exact_transport_error_m':max_transport_error,'max_rotation_orthogonality_error':orthogonality,'actual_distinct_filter_factors':unique_factors,'anchor_filter_factors':0,'duplicate_raw_factors':0,'background_or_invalid_factors':0,'direct_point_updates_excluding_anchor':int(np.asarray(arrays['point_observed']).sum()-len(members)),'predicted_group_frame_count_excluding_anchor':int((~evidence).sum()-len(counts)),'observed_group_frames':int(evidence.sum()),'rank6_group_frames':int((arrays['pose_rank']==6).sum()),'inner_iteration_limit_group_frames':iteration_limits,'pixel_ray_error_norm':distribution(np.linalg.norm(errors[:,:2],axis=1)),'depth_residual_m':distribution(errors[:,2]),'absolute_depth_residual_m':distribution(abs(errors[:,2])),'conditional_sensor_normalized_residual':distribution(d2),'residual_qualification':'Selection-conditioned final residuals using sensor R; not independent innovation NIS or calibration proof','profile_seconds':{key:sum(f.get(key,0.) for f in frames) for key in ('prediction_seconds','association_seconds','filter_seconds')},'controller_runtime_seconds':json.loads((root/'result.json').read_text())['runtime_seconds'],'gpu':meta['gpu'],'physical_rigidity_validated':False,'complete_joint_pipeline':False}
    # Export all projections for selected actual views; no hidden point cap.
    directory=root/'overlays';directory.mkdir(exist_ok=True)
    for k in sorted({0,len(data.times)//2,len(data.times)-1}):
        factors=factor_store.read(str(k)) if factor_store.contains(str(k)) else None
        for ci,camera in enumerate(data.cameras):
            frame=data.observation(k,ci);image=np.uint8(np.clip(frame.rgb,0,1)*255);q=arrays['xyz'][k]@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
            good=np.isfinite(q).all(1)&(q[:,2]>0);uv=np.zeros((len(q),2),np.int64);uv[good]=np.floor(q[good,:2]/q[good,2,None]*camera.intrinsics[:2]+camera.intrinsics[2:]+.5).astype(np.int64)
            good&=(uv[:,0]>=0)&(uv[:,0]<camera.width)&(uv[:,1]>=0)&(uv[:,1]<camera.height);pixels=uv[good,1]*camera.width+uv[good,0];flat=image.reshape(-1,3);flat[pixels]=[50,130,255]
            if factors is not None:
                use=factors['camera']==ci;flat[factors['pixel'][use]]=[0,255,100]
            from scipy.ndimage import binary_erosion
            boundary=frame.mask&~binary_erosion(frame.mask);image[boundary]=[255,210,0]
            Image.fromarray(image).save(directory/f'{k:06d}_{ci:03d}.png')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(10,4));depth=np.where(frame.valid,frame.depth,np.nan)
            a=axes[0].imshow(depth);fig.colorbar(a,ax=axes[0],label='Measured metric depth (m)');axes[0].set_title('Actual valid masked depth')
            residual_map=np.full(frame.depth.size,np.nan)
            if factors is not None:
                use=factors['camera']==ci;residual_map[factors['pixel'][use]]=factors['residual'][use,2]
            b=axes[1].imshow(residual_map.reshape(frame.depth.shape),cmap='coolwarm');fig.colorbar(b,ax=axes[1],label='Actual matched depth residual (m)');axes[1].set_title('Fixed-association posterior residual')
            fig.tight_layout();fig.savefig(directory/f'{k:06d}_{ci:03d}.depth.png',dpi=130);plt.close(fig)
            json_write(directory/f'{k:06d}_{ci:03d}.json',{'original_frame':data.frame_ids[k],'camera':camera.name,'full_predicted_points':len(q),'in_image_positive_depth_projection_count':int(good.sum()),'actual_matched_pixels':0 if factors is None else int((factors['camera']==ci).sum()),'hidden_display_subsampling':False,'blue':'all in-image predicted projections, including predictions','green':'actual distinct accepted RGB-D pixels','ground_truth_overlay':False})
    if snapshots!={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in epoch.glob('*.npy')}:raise AssertionError('Diagnostics changed inferred histories')
    json_write(root/'gate_a_diagnostics.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
