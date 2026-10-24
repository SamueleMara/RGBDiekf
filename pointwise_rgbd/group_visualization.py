"""Raw RGB-D/model diagnostics. Export never updates an estimated trajectory."""
from pathlib import Path
import json,shutil,time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from .store import json_write
from .observations import samples
from .projection import backproject
from .group_evidence_audit import (read_factors,projection_numpy,group_information,
    position_jacobian,quantiles,connected_components_of_partition,replay_candidates,REASONS)


def bounds(x):return [x.min(0).tolist(),x.max(0).tolist()] if len(x) else None


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,(list,tuple)):return [serial(x) for x in value]
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def export_viewer(root,dataset,partition,arrays,factor_root=None,config=None,replay=False):
    from .config import load_config
    config=load_config() if config is None else config
    root=Path(root);view=root/'viewer';data=view/'frames';data.mkdir(parents=True,exist_ok=True)
    factor_root=Path(factor_root or root/'phase1/partition_v000')
    asset=Path(__file__).with_name('assets')
    for source,target in [('three-0.160.0.min.js','three.min.js'),('THREE_LICENSE.txt','THREE_LICENSE.txt'),('group_viewer.html','index.html')]:shutil.copyfile(asset/source,view/target)
    for name,values,dtype in [('membership',partition.membership,'<i4'),('canonical',partition.xyz,'<f4'),('source',partition.source,'<i4'),('components',connected_components_of_partition(partition),'<i4'),('adjacency',partition.adjacency,'<i4')]:
        np.asarray(values,dtype=dtype).tofile(view/(name+('.i32' if dtype=='<i4' else '.f32')))
    np.uint8(np.clip(partition.rgb,0,1)*255).tofile(view/'rgb.u8')
    with (view/'centroid_history.f32').open('wb') as stream:
        for k in range(len(dataset.times)):
            pose=np.asarray(arrays['group_poses'][k]);center=np.einsum('sij,sj->si',pose[:,:3,:3],partition.centroids)+pose[:,:3,3]
            np.asarray(center,dtype='<f4').tofile(stream)
    frames=[];prior_centers=None;start=time.monotonic();last_support=[None]*partition.count
    for k in tqdm(range(len(dataset.times)),desc='Raw RGB-D evidence export',unit='frame'):
        stem=f'{k:06d}';xyz=np.asarray(arrays['xyz'][k]);poses=np.asarray(arrays['group_poses'][k]);cov=np.asarray(arrays['covariance'][k])
        np.asarray(xyz,dtype='<f4').tofile(data/f'{stem}.xyz.f32')
        observed=np.asarray(arrays['point_observed'][k]);np.asarray(observed,np.uint8).tofile(data/f'{stem}.observed.u8')
        factors=read_factors(factor_root,k);info=group_information(partition,poses,cov,factors,dataset.cameras)
        supported=np.asarray(arrays['group_measurements'][k])>0
        for g in np.flatnonzero(supported):last_support[g]=float(dataset.times[k])
        info.update(last_direct_support_seconds=last_support.copy(),physical_class=['NOT_ESTIMATED']*partition.count,group_pose=poses,group_covariance=cov[:,:6,:6])
        filter_file=factor_root/f'filter/{stem}.json'
        if filter_file.exists():
            f=json.loads(filter_file.read_text())
            for key in ('iterations','energy_before','energy_after','iteration_limit','line_search_stops'):info[key]=f[key]
        json_write(data/f'{stem}.groups.json',serial(info))
        raw_all=[];camera_meta=[];raw_factor_count=0
        gj=position_jacobian(xyz);marginal=gj@cov[partition.membership,:6,:6]@gj.transpose(0,2,1)
        for ci,camera in enumerate(dataset.cameras):
            frame=dataset.observation(k,ci);raw=samples(frame,config);raw_all.append(raw.xyz);prefix=f'{stem}.c{ci:03d}'
            np.asarray(raw.xyz,dtype='<f4').tofile(data/(prefix+'.raw.f32'))
            np.uint8(np.clip(raw.rgb,0,1)*255).tofile(data/(prefix+'.rgb.u8'));np.asarray(raw.pixel,dtype='<i4').tofile(data/(prefix+'.pixel.i32'))
            Image.fromarray(np.uint8(np.clip(frame.rgb,0,1)*255)).save(data/(prefix+'.png'))
            Image.fromarray(np.uint8(frame.mask)*255).save(data/(prefix+'.mask.png'))
            np.asarray(frame.depth,dtype='<f4').tofile(data/(prefix+'.depth.f32'))
            h,_=projection_numpy(xyz,camera);np.asarray(h,dtype='<f4').tofile(data/(prefix+'.projection.f32'))
            if factors is None:ids=np.empty(0,np.int64);pixels=np.empty(0,np.int64);z=np.empty((0,3));r=np.empty((0,3,3))
            else:
                selected=factors['camera']==ci;ids=factors['point_id'][selected];pixels=factors['pixel'][selected];z=factors['z'][selected];r=factors['R'][selected]
            if len(np.unique(pixels))!=len(pixels) or not frame.valid.ravel()[pixels].all():raise AssertionError('Invalid/background/duplicate historical raw factor')
            if len(pixels):
                expected=np.c_[pixels%camera.width,pixels//camera.width,np.asarray(frame.depth).ravel()[pixels]]
                if not np.allclose(expected,z,rtol=0,atol=1e-12):raise ValueError('Cached factors disagree with original sensor pixels')
            measured,_=backproject(z,camera,r);residual=z-h[ids];raw_factor_count+=len(ids)
            np.asarray(measured,dtype='<f4').tofile(data/(prefix+'.matched.f32'))
            np.asarray(np.c_[ids,pixels],dtype='<i4').tofile(data/(prefix+'.factor_ids.i32'))
            np.asarray(np.c_[z,h[ids]],dtype='<f4').tofile(data/(prefix+'.uvz.f32'))
            roundtrip=projection_numpy(measured,camera)[0]-z;reason_count={'HISTORICAL_REJECTIONS_NOT_RECORDED':None}
            if replay:
                reason,candidates=replay_candidates(xyz,marginal,partition.rgb,raw,config)
                np.asarray(reason,np.uint8).tofile(data/(prefix+'.reason.u8'))
                np.savez_compressed(data/(prefix+'.candidates.npz'),**candidates,original_source=partition.source[candidates['point']],frame_id=np.array(dataset.frame_ids[k]),camera=np.array(ci),gate_3=np.array(config.gate(3)),gate_2=np.array(config.gate(2)),provenance=np.array('RETROSPECTIVE_CURRENT_POSTERIOR_REPLAY; NOT HISTORICAL DECISIONS'))
                reason_count={name:int((reason==j).sum()) for j,name in enumerate(REASONS)}
            camera_meta.append(dict(raw_points=len(raw.xyz),accepted_factors=len(ids),raw_measured_bounds=bounds(raw.xyz),valid_foreground_pixels=int(frame.valid.sum()),invalid_foreground_pixels=int((frame.mask&~frame.valid).sum()),roundtrip_max_abs_uvz=abs(roundtrip).max(0).tolist() if len(z) else [0,0,0],accepted_depth_residual_m=quantiles(abs(residual[:,2])),accepted_pixel_residual_px=quantiles(np.linalg.norm(residual[:,:2],axis=1)),retrospective_posterior_reasons=reason_count,depth_display_range_m=np.quantile(raw.z[:,2],[0,1]).tolist() if len(raw.z) else [0,1]))
        raw_cloud=np.concatenate(raw_all);tree=cKDTree(raw_cloud) if len(raw_cloud) else None
        nn=tree.query(xyz,workers=1)[0] if tree else np.full(len(xyz),np.inf)
        np.asarray(nn,dtype='<f4').tofile(data/f'{stem}.nearest.f32')
        per_group=np.zeros((partition.count,4))
        for g in range(partition.count):per_group[g]=np.quantile(nn[partition.membership==g],[.5,.95,.99,1])
        np.savez_compressed(data/f'{stem}.surface_audit.npz',nearest_raw_distance_m=nn,group_distance_quantiles_m=per_group,observed=observed,group=partition.membership,source=partition.source)
        edge=partition.adjacency;centers=info['centroid'];v=centers[edge[:,0]]-centers[edge[:,1]];length=np.linalg.norm(v,axis=1)
        direction=np.divide(v,length[:,None],out=np.zeros_like(v),where=length[:,None]>0)
        gc=position_jacobian(centers);cc=gc@cov[:,:6,:6]@gc.transpose(0,2,1)
        variance=np.einsum('ni,nij,nj->n',direction,cc[edge[:,0]]+cc[edge[:,1]],direction)
        if prior_centers is None:prior_centers=centers.copy()
        initial=np.linalg.norm(prior_centers[edge[:,0]]-prior_centers[edge[:,1]],axis=1);edge_supported=supported[edge[:,0]]&supported[edge[:,1]]
        np.asarray(np.c_[centers[edge[:,0]],centers[edge[:,1]],length-initial,variance,edge_supported],dtype='<f4').tofile(data/f'{stem}.edges.f32')
        frames.append(dict(original_frame=dataset.frame_ids[k],time_seconds=float(dataset.times[k]),direct_observed=int(observed.sum()),prediction_only=int((~observed).sum()),prediction_only_groups=int((~supported).sum()),raw_valid_foreground_points=len(raw_cloud),actual_accepted_factors=raw_factor_count,cameras=camera_meta,raw_measured_bounds=bounds(raw_cloud),current_model_bounds=bounds(xyz),nn_to_raw_m={'direct_updated':quantiles(nn[observed]),'prediction_only':quantiles(nn[~observed]),'all':quantiles(nn)},edge_length_change_m_supported=quantiles(abs(length[edge_supported]-initial[edge_supported])),reason_provenance='RETROSPECTIVE_CURRENT_POSTERIOR_REPLAY' if replay else 'HISTORICAL_REJECTIONS_NOT_RECORDED'))
    cameras=[dict(name=c.name,width=c.width,height=c.height,intrinsics=c.intrinsics.tolist(),world_to_camera=c.world_to_camera.tolist()) for c in dataset.cameras]
    manifest=dict(points=len(partition.xyz),groups=partition.count,partition_version=int(partition.version),frames=frames,cameras=cameras,canonical_focus_bounds=bounds(partition.xyz),display_encoding='float32 display only; FP64 inference/cached histories unchanged',labels='NOT_ESTIMATED: computational groups are not physical parts',hidden_point_cap=None,canonical_geometry_conditioned=True,raw_measurements_available=True,reason_names=REASONS,rgb_role='association ranking only; no RGB likelihood in IEKF',group_information_units='center translation metres, rotation radians; normalized spectra use source sigma and radius',graph_status='COMPUTATIONAL_SURFACE_ADJACENCY_NOT_PHYSICAL_CUTS',source_factor_root=str(factor_root.resolve()))
    json_write(view/'manifest.json',manifest)
    json_write(root/'sensor_evidence_audit.json',dict(frames=frames,group_count=partition.count,point_count=len(partition.xyz),actual_occupancy_quantiles=np.quantile(np.bincount(partition.membership),[0,.25,.5,.75,.95,1]),export_seconds=time.monotonic()-start,heldout_score='NOT_AVAILABLE: historical filter used these accepted factors',factor_rejection_history='NOT_RECORDED; retrospective labels are a separate diagnostic',physical_classes='NOT_ESTIMATED'))
    return manifest
