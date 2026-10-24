"""Actual raw measurements and chronological free/rigid/joint trajectories."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from .observations import samples
from .joint_rgbd import predict,poses
from .group_visualization import serial,bounds
from .store import json_write


def export(root,dataset,config,artifacts,pooled):
    root=Path(root);view=root/'viewer';frames=view/'frames';frames.mkdir(parents=True,exist_ok=True)
    assets=Path(__file__).with_name('assets')
    for source,target in [('three-0.160.0.min.js','three.min.js'),('THREE_LICENSE.txt','THREE_LICENSE.txt'),('free_viewer.html','index.html')]:shutil.copyfile(assets/source,view/target)
    export_sources=[Path(__file__),assets/'free_viewer.html']
    export_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in export_sources}
    for p in export_sources:
        target=view/'source_snapshot'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    manifest={'timestamps_seconds':dataset.times,'original_frame_ids':dataset.frame_ids,'cameras':[dict(name=c.name,width=c.width,height=c.height,intrinsics=c.intrinsics,world_to_camera=c.world_to_camera) for c in dataset.cameras],
        'export_source_hashes':export_hashes,'camera_policy':'all_available','production_camera_indices':list(range(len(dataset.cameras))),'directions':{},'frames':[],'hidden_point_cap':None,'display_precision':'float32 coordinates only for display; all estimator histories remain float64','covariance':'Grouped point Joseph marginals with closed rank-24 cross-point covariance. Rigid/joint motion-map Schur factors retain measured canonical uncertainty; fixed calibration and conditioned associations.'}
    for direction,values in artifacts.items():
        partition=values['partition'];base=view/direction;base.mkdir(exist_ok=True)
        np.asarray(partition.xyz,dtype='<f4').tofile(base/'canonical.f32');np.uint8(np.clip(partition.rgb,0,1)*255).tofile(base/'rgb.u8')
        np.asarray(partition.membership,dtype='<i4').tofile(base/'groups.i32')
        core=values.get('coherence',{}).get('point_support',np.zeros(len(partition.xyz),bool))
        np.asarray(core,dtype=np.uint8).tofile(base/'core.u8')
        label=values.get('support',{}).get('label',np.zeros(len(partition.xyz),np.uint8));label.tofile(base/'labels.u8')
        meta={'points':len(partition.xyz),'computational_groups':partition.count,'initial_endpoint':values.get('rigid',{}).get('anchor_frame',0 if direction=='first' else len(dataset.times)-1),'core_points':int(core.sum()),'support_status':'EXECUTED' if 'support' in values else 'NOT_EXECUTED_JOINT_UNRESOLVED','labels':values.get('support',{}).get('labels',{}),'frames':[]}
        rigid_canonical=values.get('rigid',{}).get('canonical_estimate',partition.xyz)
        np.asarray(rigid_canonical,dtype='<f4').tofile(base/'rigid_canonical.f32')
        if 'rigid' in values:
            meta['rigid_poses']=values['rigid']['poses']
            meta['rigid_complete_multiview_residuals']=values['rigid'].get('complete_multiview')
        selected=values.get('selected',values.get('joint',{}).get('selected'))
        joint_canonical=partition.xyz.copy()
        if selected:
            meta['joint']=selected['model'];meta['joint_poses']=poses(selected['model']);meta['joint_display_anchor']=partition.xyz[core].mean(0) if core.any() else partition.xyz.mean(0)
            state=selected.get('map_state')
            if state is not None:
                for key,p in zip(state['prior']['keys'],state['mu']):
                    if key[0]==meta['initial_endpoint']:joint_canonical[int(key[1])]=p
        np.asarray(joint_canonical,dtype='<f4').tofile(base/'joint_canonical.f32')
        meta['canonical_semantics']='Measured endpoint shown separately; rigid/joint core uses its fitted canonical MAP coordinates. Other endpoint samples retain their measured coordinates.'
        for k in range(len(dataset.times)):
            np.asarray(values['arrays']['xyz'][k],dtype='<f4').tofile(base/f'{k:06d}.free.f32')
            if 'rigid_poses' in meta:
                expected=np.array(values['arrays']['xyz'][k],copy=True);T=meta['rigid_poses'][k]
                expected[core]=rigid_canonical[core]@T[:3,:3].T+T[:3,3]
                np.asarray(expected,dtype='<f4').tofile(base/f'{k:06d}.rigid.f32')
            np.asarray(values['arrays']['point_observed'][k],dtype=np.uint8).tofile(base/f'{k:06d}.observed.u8')
            phase_root=values.get('phase_root',root/direction/'phase1');f=phase_root/f'filter/{k:06d}.json'
            import json
            report=json.loads(f.read_text()) if f.exists() else {'stage':'ENDPOINT_INITIALIZATION'}
            meta['frames'].append({key:value for key,value in report.items() if key not in ('association','energy_history','before','after','group_counts','measurement_count','structural_rank','iterations','line_search_stops','iteration_limit_reached')})
            from .group_evidence_audit import read_factors
            # All-camera raw estimator input, never a rendered target.
            from rgbd_common.array_store import ArrayStore
            store=ArrayStore(phase_root/'factors.sqlite')
            if store.contains(str(k)):
                r=store.read(str(k));ray=[]
                meta['frames'][-1]['per_camera_fitted_residuals']={str(ci):{
                    'raw_count':int((r['camera']==ci).sum()),
                    'uvz_rmse_pixels_pixels_metres':np.sqrt(np.mean(r['residual'][r['camera']==ci]**2,axis=0)).tolist() if (r['camera']==ci).any() else None}
                    for ci in range(len(dataset.cameras))}
                from .projection import backproject
                for ci in range(len(dataset.cameras)):
                    use=r['camera']==ci;xyz,_=backproject(r['z'][use],dataset.cameras[ci],r['R'][use])
                    ray.append(np.c_[np.asarray(values['arrays']['xyz'][k])[r['point_id'][use]],xyz])
                np.asarray(np.concatenate(ray),dtype='<f4').tofile(base/f'{k:06d}.rays.f32')
            store.close()
        manifest['directions'][direction]=meta
    for k in tqdm(range(len(dataset.times)),desc='Export actual RGB-D visual debug',unit='frame'):
        row=[];clouds=[]
        for ci,camera in enumerate(dataset.cameras):
            frame=dataset.observation(k,ci);raw=samples(frame,config);prefix=f'{k:06d}.c{ci:03d}'
            np.asarray(raw.xyz,dtype='<f4').tofile(frames/f'{prefix}.xyz.f32');np.uint8(np.clip(raw.rgb,0,1)*255).tofile(frames/f'{prefix}.rgb.u8')
            np.asarray(raw.pixel,dtype='<i4').tofile(frames/f'{prefix}.pixel.i32')
            Image.fromarray(np.uint8(np.clip(frame.rgb,0,1)*255)).save(frames/f'{prefix}.png');Image.fromarray(np.uint8(frame.mask)*255).save(frames/f'{prefix}.mask.png')
            low,high=np.quantile(raw.z[:,2],[0,1]) if len(raw.z) else (0,1)
            depth=np.zeros(frame.depth.shape,np.uint8)
            if high>low:depth[frame.valid]=np.uint8(255*np.clip((frame.depth[frame.valid]-low)/(high-low),0,1))
            Image.fromarray(depth).save(frames/f'{prefix}.depth.png')
            row.append({'points':len(raw.xyz),'depth_range_metres':[float(low),float(high)]});clouds.append(raw.xyz)
        manifest['frames'].append({'cameras':row,'actual_measured_bounds':bounds(np.concatenate(clouds))})
    manifest['bidirectional']=None if pooled is None else {key:value for key,value in pooled.items() if key!='selected'}
    json_write(view/'manifest.json',serial(manifest))
    plot_convergence(root,artifacts)
    shutil.copyfile(root/'reports/convergence.png',view/'convergence.png')


def plot_convergence(root,artifacts):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (root/'reports').mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for name in artifacts:
        phase=artifacts[name].get('phase_root',root/name/'phase1')
        records=[json.loads(p.read_text()) for p in sorted((phase/'filter').glob('*.json'))]
        if not records:continue
        x=[r['frame'] for r in records]
        axes[0].plot(x,[np.sum(r['before']) for r in records],label=name+' before')
        axes[0].plot(x,[np.sum(r['after']) for r in records],label=name+' after')
        axes[1].plot(x,[r['directly_observed'] for r in records],label=name+' observed points')
    axes[0].set_ylabel('Conditional fitting energy');axes[1].set_ylabel('Direct measurement support');axes[1].set_xlabel('Chronological frame index')
    for axis in axes:axis.legend();axis.grid(alpha=.3)
    fig.tight_layout();fig.savefig(root/'reports/convergence.png',dpi=160);plt.close(fig)
