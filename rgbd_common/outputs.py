"""Identical point-only products and live diagnostics for the two estimators."""
from pathlib import Path
import csv,json,time,resource,shutil
import numpy as np
from PIL import Image
import torch
from .geometry import ObservationIDs
from .projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from .backends.torch_cuda import numpy


def export_foreground_masks(directory,model,state,frame,matches,projected,bins,config,backend):
    from .foreground import projected_mask_status,VISIBLE,UNKNOWN,OCCLUDED
    directory=Path(directory);shape=(frame.camera.height,frame.camera.width)
    foreground=np.asarray(frame.mask);status=projected_mask_status(projected,bins,frame,config,backend,model,state)
    matched=np.zeros(shape,bool);matched.ravel()[numpy(matches.pixel)]=True
    projected_valid=numpy(bins['owner']).reshape(shape)>=0;visible=np.zeros(shape,bool)
    ids=status['front_indices'];pixels=numpy(status['front_pixels']);is_visible=numpy(status['status'][ids])==VISIBLE
    visible.ravel()[pixels[is_visible]]=True
    images={'fg_mask_observed':foreground,'projected_point_valid':projected_valid,'projected_visible_fg':visible&foreground,'matched_fg':matched,'unmatched_fg':frame.valid&~matched}
    for name,mask in images.items():
        target=directory/name;target.mkdir(exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(target/f'frame_{frame.k:06d}_cam_{frame.ci:03d}.png')
    path=directory/'mask_constraint_violations.csv';exists=path.exists()
    violations=numpy(status['violation_indices'])
    with path.open('a') as stream:
        writer=csv.writer(stream)
        if not exists:writer.writerow(['frame','camera','point_id','pose_stage','visibility'])
        writer.writerows((frame.k,frame.ci,int(model.point_id[i]),'accepted-final','visible') for i in violations)
    all_status=numpy(status['status'])
    return {'mask_constraint_violations':len(violations),'predicted_visible':int((all_status==VISIBLE).sum()),'predicted_occluded':int((all_status==OCCLUDED).sum()),'predicted_unknown':int((all_status==UNKNOWN).sum())}


def residual_diagnostics(result,dataset,config,backend,directory,debug=False):
    from .matching_cuda import match_rgbd
    from .observation_model import output
    from .progress import progress
    rows=[];squares=np.zeros(6);total=0;nis=0.
    for k in progress(range(len(dataset.times)),desc='Accepted-map RGB-D residual diagnostics',unit='frame'):
        for ci in range(len(dataset.cameras)):
            frame=dataset.observation(k,ci);match,stats,projected,bins=match_rgbd(result.model,result.states[k],frame,config,backend)
            ids=numpy(match.point);count=len(match)
            row={'frame':k,'camera':ci,'matched':count,'foreground_valid':int(frame.valid.sum())}
            if debug:row.update(export_foreground_masks(directory,result.model,result.states[k],frame,match,projected,bins,config,backend))
            else:row['mask_constraint_violations']=stats['mask_constraint_violations']
            if count:
                prediction,*_=output(result.states[k],backend.tensor(result.model.xyz[ids]),backend.tensor(result.model.rgb[ids]),backend.tensor(result.model.group[ids],torch.long),frame.camera,backend)
                error=match.observed-prediction;ss=numpy((error**2).sum(0));squares+=ss;total+=count
                maha=(error*torch.linalg.solve(match.covariance,error[...,None])[...,0]).sum(1);nis+=float(maha.sum())
                row.update(uv_rms_pixels=float(np.sqrt(ss[:2].sum()/count)),depth_rms_m=float(np.sqrt(ss[2]/count)),rgb_rms_encoded_01=float(np.sqrt(ss[3:].sum()/(3*count))),mean_conditional_mahalanobis6=float(maha.mean()),depth_absolute_median_m=float(torch.median(abs(error[:,2]))))
            rows.append(row)
    keys=sorted(set().union(*(row.keys() for row in rows)))
    with (Path(directory)/'reports/rgbd_residuals_by_frame.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return {'matched_factors':total,'uv_rms_pixels':float(np.sqrt(squares[:2].sum()/max(1,total))),'depth_rms_m':float(np.sqrt(squares[2]/max(1,total))),'rgb_rms_encoded_01':float(np.sqrt(squares[3:].sum()/max(1,3*total))),'mean_conditional_mahalanobis6':nis/max(1,total),
            'mask_constraint_violations':sum(row['mask_constraint_violations'] for row in rows),
            'qualification':'Accepted trajectory/map and selected one-owner foreground matches only; local Gaussian covariances do not capture hard-set truncation. Association truncation and unmatched coverage prevent interpreting these residuals as an independent calibration or unselected chi-square test.'}


def plotly_script(directory):
    from plotly.offline import get_plotlyjs
    script=Path(directory)/'plotly.min.js'
    if not script.exists():script.write_text(get_plotlyjs())
    return script.name


def display_indices(model,limit):
    # Round-robin samples retain small discovered groups in the display.
    groups=[np.flatnonzero(model.group==g) for g in np.unique(model.group)]
    quota=min(limit,model.count)
    if not groups:return np.empty(0,dtype=int)
    counts=np.zeros(len(groups),int)
    for _ in range(quota):
        available=np.array([len(ids)>counts[g] for g,ids in enumerate(groups)])
        chosen=int(np.argmin(np.where(available,counts,np.inf)));counts[chosen]+=1
    return np.concatenate([ids[np.linspace(0,len(ids)-1,n,dtype=int)] for ids,n in zip(groups,counts) if n])


def motion_html(path,model,states,limit=5000):
    path=Path(path);script=plotly_script(path.parent);ids=display_indices(model,limit)
    data={'xyz':model.xyz[ids].tolist(),'rgb':np.uint8(np.clip(model.rgb[ids],0,1)*255).tolist(),'group':model.group[ids].tolist(),'poses':np.array([s.poses for s in states]).tolist()}
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Point RGB-D SLAM</title></head><body>
<p>Canonical point display sample, with every learned rigid group represented. The saved map contains all points. Motion is estimated for every group.</p>
<button id="play">Play / pause</button><label>Frame <input id="time" type="range" min="0" value="0"></label><span id="label"></span>
<label>Color <select id="color"><option value="rgb">Measured RGB</option><option value="group">Estimated rigid group</option></select></label>
<div id="plot" style="height:85vh"></div><script src="SCRIPT"></script><script>
const data=DATA,slider=document.getElementById('time');slider.max=data.poses.length-1;let playing=false;
const colors=data.rgb.map(c=>`rgb(${c.join(',')})`),groupColors=data.group.map(g=>`hsl(${(g*137.508)%360},75%,50%)`);
function points(k){let x=[],y=[],z=[];for(let i=0;i<data.xyz.length;i++){const p=data.xyz[i],t=data.poses[k][data.group[i]],v=t.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);x.push(v[0]);y.push(v[1]);z.push(v[2]);}return{x,y,z};}
Plotly.newPlot('plot',[{...points(0),type:'scatter3d',mode:'markers',marker:{size:2,color:colors},text:data.group.map(g=>'group '+g)}],{scene:{aspectmode:'data'},uirevision:'fixed'},{responsive:true});
function show(){const k=+slider.value,p=points(k);Plotly.restyle('plot',{x:[p.x],y:[p.y],z:[p.z]});document.getElementById('label').textContent=k;}
slider.oninput=show;document.getElementById('play').onclick=()=>playing=!playing;
document.getElementById('color').onchange=e=>Plotly.restyle('plot',{'marker.color':[e.target.value==='rgb'?colors:groupColors]});
setInterval(()=>{if(playing){slider.value=(+slider.value+1)%data.poses.length;show();}},100);
</script></body></html>'''.replace('SCRIPT',script).replace('DATA',json.dumps(data,separators=(',',':')))
    pending=path.with_suffix('.pending.html');pending.write_text(html);pending.replace(path)


def live_view(reports,model,states,label,k,records=()):
    from html import escape
    directory=Path(reports).parent/'visual_debug';directory.mkdir(exist_ok=True)
    motion_html(directory/'motion.html',model,states)
    epoch=max((r.get('score_epoch',1) for r in records),default=1)
    proposals=[r for r in records if 'proposal_energy' in r and r.get('score_epoch',1)==epoch]
    accepted=[r for r in proposals if r.get('accepted')]
    traces=[{'x':[r['elapsed_seconds'] for r in rows],'y':[r['proposal_energy'] for r in rows],'name':name,'mode':mode}
            for rows,name,mode in [(proposals,'Proposed full objective','markers'),(accepted,'Accepted full objective','lines+markers')]]
    text=f'''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15"></head><body>
<h1>{escape(label)}: frame {k}</h1><p>{model.count:,} canonical points; {len(states[0].poses)} learned rigid groups. Live results are provisional.</p>
<a href="motion.html">Open interactive 3D motion and rigid-group colors</a><div id="objective" style="height:40vh"></div>
<iframe title="Estimated 3D motion" src="motion.html" style="width:98%;height:70vh;border:0"></iframe>
<p>Latest stage: <a href="../reports/live_progress.json">live_progress.json</a>. Rejected proposals remain visible in the objective chart.</p>
<script src="plotly.min.js"></script><script>Plotly.newPlot('objective',{json.dumps(traces)},{{xaxis:{{title:'Elapsed computation seconds'}},yaxis:{{title:'Same full-sequence objective'}}}},{{responsive:true}});</script></body></html>'''
    pending=directory/'index.pending.html';pending.write_text(text);pending.replace(directory/'index.html')


def write_ply(model,path):
    dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1'),('point_id','<f8'),('group','<i4')])
    if model.count and model.point_id.max()>2**53:raise ValueError('PLY double point IDs would lose integer precision; original int64 IDs remain in NPZ/CSV')
    data=np.empty(model.count,dtype=dtype)
    for j,key in enumerate(('x','y','z')):data[key]=model.xyz[:,j]
    for j,key in enumerate(('red','green','blue')):data[key]=np.uint8(np.clip(model.rgb[:,j],0,1)*255)
    data['point_id']=model.point_id;data['group']=model.group
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {model.count}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty double point_id\nproperty int group\nend_header\n'
    with Path(path).open('wb') as f:f.write(header.encode());data.tofile(f)


def sparse_projection_images(directory,model,states,dataset,backend):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    for k in sorted({0,len(states)//2,len(states)-1}):
        for ci,camera in enumerate(dataset.cameras):
            frame=dataset.observation(k,ci)
            p=transport_and_project(backend.tensor(model.xyz,backend.input_dtype),backend.tensor(model.group,torch.long),backend.tensor(states[k].poses,backend.input_dtype),camera)
            bins=build_pixel_bins_and_zbuffer(p,camera);owner=numpy(bins['owner']);valid=owner>=0
            rgb=np.zeros((camera.height*camera.width,3));depth=np.zeros(camera.height*camera.width)
            rgb[valid]=model.rgb[owner[valid]];depth[valid]=numpy(p['depth'])[owner[valid]]
            shape=(camera.height,camera.width)
            prefix=directory/f'frame_{k:06d}_cam_{ci:03d}'
            Image.fromarray(np.uint8(np.clip(rgb.reshape(*shape,3),0,1)*255)).save(str(prefix)+'_rgb.png')
            Image.fromarray(valid.reshape(shape).astype(np.uint8)*255).save(str(prefix)+'_valid.png')
            np.save(str(prefix)+'_depth.npy',depth.reshape(shape))
            scale=float(np.max(frame.depth[frame.valid])) if frame.valid.any() else 1.
            comparison=np.concatenate([frame.depth,depth.reshape(shape)],axis=1)/scale
            Image.fromarray(np.uint8(np.clip(comparison,0,1)*255)).save(str(prefix)+'_depth_comparison.png')
            Image.fromarray(np.uint8(np.clip(np.concatenate([frame.rgb,rgb.reshape(*shape,3)],axis=1),0,1)*255)).save(str(prefix)+'_rgb_comparison.png')


def export_result(result,dataset,config,backend,directory,pipeline,debug=False):
    from .outer_loop import write_json
    from .trajectory_covariance import TrajectoryMoments,error_moments_at_reference
    directory=Path(directory);model=result.model
    model.save(directory/'solved_points.npz');write_ply(model,directory/'solved_points.ply')
    with (directory/'solved_points.csv').open('w') as f:
        f.write('point_id,group,x,y,z,r,g,b,source_id\n')
        for start in range(0,model.count,100000):
            end=min(start+100000,model.count);rows=np.c_[model.point_id[start:end],model.group[start:end],model.xyz[start:end],model.rgb[start:end],model.source_id[start:end]]
            np.savetxt(f,rows,delimiter=',',fmt=['%d','%d','%.9g','%.9g','%.9g','%.9g','%.9g','%.9g','%d'])
    np.savez(directory/'group_poses.npz',poses=np.array([s.poses for s in result.states]),times_seconds=dataset.times,pipeline=np.array(pipeline),reference_frames=np.zeros(len(result.states[0].poses),dtype=int))
    np.savez(directory/'group_twists.npz',twists=np.array([s.twists for s in result.states]),times_seconds=dataset.times,units=np.array('m/s,rad/s; spatial/world translation-first'))
    offset,reference_covariance,second=error_moments_at_reference(result.smooth[0],result.smooth[1],result.states)
    np.savez(directory/'group_covariance.npz',accepted_reference_error_mean=offset,accepted_reference_covariance=reference_covariance,accepted_reference_error_second_moment=second,predicted=np.array([s.predicted_covariance for s in result.steps]),filtered=np.array([s.filtered_covariance for s in result.steps]),smoothed=np.array(result.smooth[1]),
        transitions=np.array([s.transition for s in result.steps[1:]]) if result.steps[1].transition is not None else np.empty(0),
        sigma_cross_covariances=np.array([s.cross_covariance for s in result.steps[1:]]) if result.steps[1].cross_covariance is not None else np.empty(0),
        filtered_pose_means=np.array([s.filtered.poses for s in result.steps]),filtered_twist_means=np.array([s.filtered.twists for s in result.steps]),smoothed_pose_means=np.array([s.poses for s in result.smooth[0]]),smoothed_twist_means=np.array([s.twists for s in result.smooth[0]]),accepted_mean_matches_smoothed_proposal=np.array(result.covariance is not None),
        temporal_backward_regression=np.array(TrajectoryMoments(result.steps,result.smooth).regressions),
        semantics=np.array('Conditional on final frozen points and fixed selected raw matches; UKF uses the fixed raw IDs throughout quadrature; covariances attach to these saved proposal means. Cross-time blocks equal products of temporal_backward_regression times the later smoothed block. Accepted-reference fields re-express this same Gaussian with a possibly nonzero mean error; they do not assert that the accepted nominal equals its mean. Not full map-motion marginal covariance.'),pipeline=np.array(pipeline))
    offsets=model.provenance.offsets;raw_ids=model.provenance.raw_ids
    model.source_owners([])  # fails rather than export duplicated source factors
    source_times,source_cameras,source_pixels=ObservationIDs(dataset).decode(raw_ids)
    if np.any(source_times>=len(dataset.times)):raise ValueError('Invalid source timestamp in final map provenance')
    for k in np.unique(source_times):
        for ci in np.unique(source_cameras[source_times==k]):
            use=(source_times==k)&(source_cameras==ci)
            if not np.all(dataset.observation(int(k),int(ci)).valid.ravel()[source_pixels[use]]):
                raise ValueError('BACKGROUND_SOURCE_IN_FINAL_MAP: source provenance is not valid foreground')
    np.savez(directory/'source_provenance.npz',point_id=model.point_id,offsets=offsets,raw_observation_id=raw_ids,first_source_id=model.source_id,
             semantics=np.array('Geometry initialization/fusion source IDs, not independent landmark priors. Current raw likelihood ownership is stored separately; a source pixel is counted once in the objective.'))
    from .source_factors import export_source_factors
    coupling=export_source_factors(model,result.states,dataset,config,backend,directory/'source_pose_factors') if debug else {'points':model.count,'factors':len(model.provenance.raw_ids),'semantics':'Shared source-pose and raw keys retained in source_provenance.npz with accepted trajectories/calibration. Conditional covariance summaries only; no independent cross-point or marginal claim.'}
    unresolved=[]
    if debug:
        with (directory/'unsolved_observations.csv').open('w') as f:
            f.write('frame,camera,pixel_x,pixel_y,reason\n')
            for k in range(len(dataset.times)):
                for ci,camera in enumerate(dataset.cameras):
                    pixels=result.ownership.unsolved(k,ci);unresolved.append({'frame':k,'camera':ci,'unresolved':len(pixels)})
                    for start in range(0,len(pixels),100000):
                        p=pixels[start:start+100000];rows=np.c_[np.full(len(p),k),np.full(len(p),ci),p%camera.width,p//camera.width,np.zeros(len(p),int)]
                        np.savetxt(f,rows,delimiter=',',fmt='%d')
        alias=directory/'unresolved_observations.csv'
        if alias.is_symlink():alias.unlink()
        shutil.copyfile(directory/'unsolved_observations.csv',alias)
    else:
        for k in range(len(dataset.times)):
            for ci in range(len(dataset.cameras)):
                unresolved.append({'frame':k,'camera':ci,'unresolved':len(result.ownership.unsolved(k,ci))})
        with (directory/'unsolved_counts.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['frame','camera','unresolved']);writer.writeheader();writer.writerows(unresolved)
    write_json(directory/'reports/unresolved_by_frame.json',unresolved)
    with (directory/'objective_history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result.history[0]) if result.history else ['outer','accepted_energy']);writer.writeheader();writer.writerows(result.history)
    with (directory/'stage_log.csv').open('w') as f:
        keys=sorted(set().union(*(record.keys() for record in result.log.records)));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(result.log.records)
    with (directory/'match_log.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['frame','camera','energy']);writer.writeheader()
        for row in result.cache.score['frames']:writer.writerow({key:row[key] for key in writer.fieldnames})
    records=result.log.records
    epoch=max((r.get('score_epoch',1) for r in records),default=1)
    if debug:
        sparse_projection_images(directory/'forward_rgb',model,[s.filtered for s in result.steps],dataset,backend)
        sparse_projection_images(directory/'backward_rgb',model,result.smooth[0],dataset,backend)
        # Copy semantic aliases: the requested external SSD does not support symlinks.
        for name,target in [('forward_depth','forward_rgb'),('backward_depth','backward_rgb')]:
            alias=directory/name
            if alias.is_symlink():alias.unlink()
            shutil.copytree(directory/target,alias,dirs_exist_ok=True)
        visual=directory/'visual_debug';visual.mkdir(exist_ok=True);motion_html(visual/'motion.html',model,result.states)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(2,2,figsize=(12,8));records=result.log.records
        epoch=max((r.get('score_epoch',1) for r in records),default=1)
        proposals=[row for row in records if 'proposal_energy' in row and row.get('score_epoch',1)==epoch]
        axes[0,0].plot([r['elapsed_seconds'] for r in proposals],[r['proposal_energy'] for r in proposals],'.',label='proposed')
        accepted=[r for r in proposals if r.get('accepted')];axes[0,0].plot([r['elapsed_seconds'] for r in accepted],[r['proposal_energy'] for r in accepted],'.-',label='accepted');axes[0,0].legend();axes[0,0].set_title('Same full-sequence objective')
        axes[0,1].plot([r['points'] for r in records if r['stage']=='frontier-complete']);axes[0,1].set_title('Solved point count during reverse/frontier')
        axes[1,0].plot([r['unresolved'] for r in unresolved]);axes[1,0].set_title('Final unresolved observations per frame/camera')
        axes[1,1].plot([np.linalg.norm(a.minus(b.filtered)) for a,b in zip(result.smooth[0],result.steps)]);axes[1,1].set_title('Own smoother correction to filtered states')
        for ax in axes.flat:ax.grid(alpha=.25)
        figure.tight_layout();figure.savefig(visual/'convergence.png');plt.close(figure)
        (visual/'index.html').write_text(f'<html><body><h1>{pipeline}: {result.status}</h1><a href="motion.html">Canonical points and moving rigid groups</a><img width="95%" src="convergence.png"></body></html>')
    stages=[row['stage'] for row in records if row.get('score_epoch',1)==epoch]
    current_records=[r for r in records if r.get('score_epoch',1)==epoch]
    finished=[r for r in current_records if r['stage']=='outer-complete']
    last_outer=finished[-1]['outer'] if finished else None
    complete=last_outer is not None and all(any(r['stage']==name and r.get('outer')==last_outer for r in current_records) for name in ('smoother-complete','reverse-discovery-complete','chronological-frontier-complete')) and any(r['stage']=='smoother-complete' and r.get('label')=='final-new-map' and r['elapsed_seconds']>finished[-1]['elapsed_seconds'] for r in current_records)
    covariance_min=float(min(np.linalg.eigvalsh(c).min() for c in result.smooth[1]))
    report={'projection_operator_version':epoch,'resumed_after_projection_correction':any(r['stage']=='projection-score-rebase' for r in records),'legacy_score_history_comparable':not any(r['stage']=='projection-score-rebase' for r in records),'pipeline':pipeline,'status':result.status,'complete_real_cycle':complete,'frames':len(dataset.times),'cameras':len(dataset.cameras),'initial_solved_points':result.initial_points,'final_solved_points':model.count,'unresolved_observations':sum(r['unresolved'] for r in unresolved),'groups':len(result.states[0].poses),'forward_passes':stages.count('forward-complete'),'smoother_passes':stages.count('smoother-complete'),'reverse_passes':stages.count('reverse-discovery-complete'),'frontier_passes':stages.count('chronological-frontier-complete'),'initialization_sha256':result.initial_digest,'final_energy':result.cache.score['energy'],'history':result.history,'conditional_covariance_min_eigenvalue':covariance_min,'accepted_trajectory_covariance_available':result.covariance is not None,'joint_map_marginals_computed':False,'runtime_seconds':time.monotonic()-result.log.start,'rss_peak_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,'gpu':backend.memory(),
        'limitations':['Local sparse assignment and group proposals, not unrestricted global inference','Feature initializers linearize backprojection noise and retain shared-source multi-time covariance; no feature likelihood is added as a second posterior factor','Prefix frontier correction retains earlier map linearizations; a final full new-map pass is mandatory and recorded','Point coordinate-fusion covariance is conditional on frozen poses/provenance, not a joint posterior marginal','Priors include explicitly documented parameter-free structural choices where supplied papers are symbolic','No generic UKF MAP-descent or global physical-recovery theorem','UKF quadrature conditions on nominal raw associations; sigma pixel-bin changes do not remove factors; positive-depth projection failure is explicit','Unresolved reason code 0 means unassigned valid foreground depth; invalid depth has no RGB-D factor']}
    report['source_pose_factors']=coupling
    report.update(output_mode='debug' if debug else 'compact',source_foreground_violations=0,raw_images_replicated=False)
    report['rgbd_residuals']=residual_diagnostics(result,dataset,config,backend,directory,debug)
    report.update(hard_mask_model_version=2,accepted_nominal_mask_feasible=report['rgbd_residuals']['mask_constraint_violations']==0,mask_visibility_premise=getattr(dataset,'metadata_report',{}).get('mask_visibility_premise',{}),observation_domain='M AND raw-valid AND finite positive metric depth; background and invalid depth contribute no factors',gaussian_covariance_qualification='Local conditional Gaussian; hard-set truncation is not represented exactly')
    from .map_uncertainty import conditional_map_uncertainty
    report['conditional_map_uncertainty']=conditional_map_uncertainty(model,result.states,dataset,config,backend,directory/'map_uncertainty',save_factors=debug)
    write_json(directory/'validation_report.json',report)
    return report
