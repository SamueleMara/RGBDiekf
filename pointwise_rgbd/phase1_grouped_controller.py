# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""One endpoint-anchored grouped pass. No discovery, recovery or point IEKF."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from tqdm.auto import tqdm
from rgbd_common.array_store import ArrayStore
from .store import json_write
from .surface_supernodes import build_partition
from .group_process import initialize,predict
from .group_pose import transport
from .group_iekf import associate_groups,correct
from .projection import project


def run_phase1(dataset,config,noise,backend,root,anchor='first',target_occupancy=64,resume=False,partition=None,motion_model='pose_twist_12d'):
    if motion_model=='pose_only_6d':
        from .group_pose_only import initialize as initialize_model,predict as predict_model
        dimension=6
    elif motion_model=='pose_twist_12d':initialize_model,predict_model,dimension=initialize,predict,12
    else:raise ValueError('Unknown named group motion model')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'run.json').exists():
        old=json.loads((root/'run.json').read_text())
        if old.get('motion_model','pose_twist_12d')!=motion_model:raise ValueError('Cannot reuse a checkpoint/covariance under a different motion model')
    k0=0 if anchor=='first' else len(dataset.times)-1
    order=list(range(len(dataset.times))) if anchor=='first' else list(range(len(dataset.times)-1,-1,-1))
    begin=time.monotonic()
    if partition is None:_,partition=build_partition(dataset,anchor,config,target_occupancy)
    path=root/'partition.npz'
    if path.exists():
        with np.load(path) as old:
            if str(old['partition_hash'])!=partition.digest():raise ValueError('Partition changed across resume')
    else:partition.save(path)
    n=len(partition.xyz);s=partition.count;kcount=len(dataset.times)
    schema={'group_poses':((kcount,s,4,4),np.float64),'twists':((kcount,s,6),np.float64),'covariance':((kcount,s,12,12),np.float64),'xyz':((kcount,n,3),np.float64),'point_observed':((kcount,n),bool),'group_measurements':((kcount,s),np.int64),'pose_rank':((kcount,s),np.int64)}
    schema['covariance']=((kcount,s,dimension,dimension),np.float64)
    if dimension==6:del schema['twists']
    arrays={name:np.lib.format.open_memmap(root/f'{name}.npy',mode='r+' if (root/f'{name}.npy').exists() else 'w+',dtype=dtype,shape=shape) for name,(shape,dtype) in schema.items()}
    factors=ArrayStore(root/'actual_measurement_factors.sqlite');checkpoint=root/'checkpoint.json'
    canonical=backend.tensor(partition.xyz);membership=backend.tensor(partition.membership,torch.long)
    metadata={'anchor':anchor,'anchor_frame_index':k0,'anchor_original_frame':dataset.frame_ids[k0],'processing_order':order,'chronological_original_frames':dataset.frame_ids,'physical_times_seconds':dataset.times.tolist(),'partition_version':partition.version,'partition_hash':partition.digest(),'all_raw_endpoint_points':n,'groups':s,'canonical_geometry_conditioned':True,'anchor_reassimilations':0,'target_computational_occupancy':target_occupancy,'physical_body_count_inferred_from_occupancy':False,'group_noise':noise.provenance,'baseline_independent_point_filter_called':False,'recovery_paused':True}
    json_write(root/'run.json',metadata)
    metadata.update(motion_model=motion_model,state_dimension=dimension,velocity_state_present=dimension==12)
    json_write(root/'run.json',metadata)
    def commit(k,poses,twists,p,point_observed,counts,rank,next_index,report):
        xyz=transport(canonical,membership,poses).cpu().numpy()
        values=[('group_poses',poses.cpu().numpy()),('covariance',p.cpu().numpy()),('xyz',xyz),('point_observed',point_observed),('group_measurements',counts),('pose_rank',rank)]
        if dimension==12:values.append(('twists',twists.cpu().numpy()))
        for name,value in values:arrays[name][k]=value;arrays[name].flush()
        digest=hashlib.sha256(np.ascontiguousarray(xyz).tobytes()+poses.cpu().numpy().tobytes()+p.cpu().numpy().tobytes()).hexdigest()
        json_write(root/f'frames/{k:06d}.json',{'frame_index':k,'original_frame':dataset.frame_ids[k],'time_seconds':float(dataset.times[k]),'snapshot_sha256':digest,**report})
        if hasattr(dataset,'base'):json_write(root/'input_fingerprints.json',dataset.base.observation_fingerprints)
        json_write(checkpoint,{'next_processing_index':next_index,'last_chronological_frame_index':k,'partition_hash':partition.digest()})
    if checkpoint.exists():
        if not resume:raise ValueError('Existing grouped results require --resume')
        saved=json.loads(checkpoint.read_text());start=saved['next_processing_index'];k=saved['last_chronological_frame_index']
        poses=backend.tensor(np.array(arrays['group_poses'][k]));twists=backend.tensor(np.array(arrays['twists'][k])) if dimension==12 else None;p=backend.tensor(np.array(arrays['covariance'][k]))
        if hasattr(dataset,'base'):
            previous=json.loads((root/'input_fingerprints.json').read_text())
            for completed in order[:start]:
                for ci in range(len(dataset.cameras)):
                    dataset.observation(completed,ci);key=f'{completed:06d}_{ci:03d}'
                    if previous[key]!=dataset.base.observation_fingerprints[key]:raise ValueError('Completed raw RGB-D changed on resume')
    else:
        start=1;poses,twists,p=initialize_model(s,noise,backend)
        commit(k0,poses,twists,p,np.ones(n,bool),np.zeros(s,np.int64),np.zeros(s,np.int64),1,{'stage':'CONDITIONED_ENDPOINT','actual_anchor_samples':n,'actual_filter_measurements':0,'group_rotation_evidence':False})
    for step in tqdm(range(start,kcount),desc=f'Grouped Phase 1 {anchor} endpoint',unit='frame'):
        k=order[step];previous=order[step-1];h=float(dataset.times[k]-dataset.times[previous]);clock=time.monotonic()
        poses,twists,p=predict_model(poses,twists,p,h,noise);backend.synchronize();prediction_seconds=time.monotonic()-clock
        clock=time.monotonic();matches,association=associate_groups(partition,poses,p,dataset,k,config,backend,audit_path=root/'candidates');backend.synchronize();association_seconds=time.monotonic()-clock
        clock=time.monotonic();poses,twists,p,report=correct(poses,twists,p,canonical,membership,matches,config,backend);backend.synchronize();filter_seconds=time.monotonic()-clock
        point_observed=np.zeros(n,bool);all_ids=[];cameras=[];pixels=[];zs=[];rs=[];residuals=[];pre_residuals=[];innovations=[];colors=[]
        for ci,match in enumerate(matches):
            ids=match['point'].cpu().numpy();point_observed[ids]=True;all_ids.append(ids);cameras.append(np.full(len(ids),ci,np.int64));pixels.append(match['pixel']);zs.append(match['z'].cpu().numpy());rs.append(match['R'].cpu().numpy())
            xyz=transport(canonical,membership,poses)[match['point']];prediction,_=project(xyz,match['camera']);residuals.append(match['z'].cpu().numpy()-prediction.cpu().numpy())
            colors.append(np.asarray(dataset.observation(k,ci).rgb).reshape(-1,3)[match['pixel']])
            pre_residuals.append(match['residual'] if len(ids) else np.empty((0,3)));innovations.append(match['S'] if len(ids) else np.empty((0,3,3)))
        factors.write(str(k),point_id=np.concatenate(all_ids),camera=np.concatenate(cameras),pixel=np.concatenate(pixels),z=np.concatenate(zs),R=np.concatenate(rs),residual=np.concatenate(residuals),pre_residual=np.concatenate(pre_residuals),innovation_covariance=np.concatenate(innovations),rgb=np.concatenate(colors))
        spectra=np.linalg.eigvalsh(p.cpu().numpy())
        if (spectra[:,0]<=0).any():raise ArithmeticError('Non-SPD grouped posterior; no jitter repair')
        report_serial={key:value for key,value in report.items() if key!='energy_history'}
        json_write(root/f'filter/{k:06d}.json',{**report_serial,'energy_history':report['energy_history']})
        commit(k,poses,twists,p,point_observed,report['counts'],report['pose_rank'],step+1,{'signed_dt_seconds':h,'actual_filter_measurements':int(report['counts'].sum()),'directly_observed_points':int(point_observed.sum()),'prediction_only_points':int((~point_observed).sum()),'observed_groups':int((report['counts']>0).sum()),'fully_observable_groups':int((report['pose_rank']==6).sum()),'pose_information_rank_histogram':np.bincount(report['pose_rank'],minlength=7).tolist(),'min_covariance_eigenvalue':float(spectra[:,0].min()),'max_covariance_condition':float((spectra[:,-1]/spectra[:,0]).max()),'prediction_seconds':prediction_seconds,'association_seconds':association_seconds,'filter_seconds':filter_seconds,'association_stats':association})
    for value in arrays.values():value.flush()
    factors.close();metadata.update({'status':'PHASE1_FILTER_COMPLETE_NOT_PHYSICAL_GROUP_VALIDATION','runtime_seconds':time.monotonic()-begin,'gpu':backend.memory(),'source_observations':n,'point_trajectories_shape':[kcount,n,3]});json_write(root/'summary.json',metadata)
    return partition,arrays
