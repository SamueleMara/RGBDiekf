"""Opt-in F1/B1 temporal runner. Legacy update_points is never called here."""
from pathlib import Path
import csv
import time
import resource
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.io import json_write,serial
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame,SensorNoise,REASONS
from estimator.free_projective_run import build_catalogue,_store
from estimator.point_kalman_cuda import predict,transition,process_noise
from estimator.temporal_zbuffer import first_hit_pixel_owner,compare_owner_maps,owner_at
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_rgbd_terms import build_temporal_rows,raw_values,sensor_covariance,temporal_prediction,TEMPORAL_REASONS,validate_reference_patch
from estimator.temporal_simplified_iekf import update_points_temporal_simplified,row_signature,same_domain,foreground_violations,REFERENCE_UNCERTAINTY
from estimator.temporal_full_graph import TemporalWindowGraph,FullResourceLimit


def append_csv(path,row):
    new=not path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if new:w.writeheader()
        w.writerow(row)


def batched_eigvalsh(matrix,chunk_size=256):
    """Bound cuSOLVER's batch count; independent small matrices, no graph cut."""
    return torch.cat([torch.linalg.eigvalsh(part) for part in matrix.split(chunk_size)]) if len(matrix) else matrix.new_empty((0,matrix.shape[-1]))


def full_component_preflight(rows,source,anchor,workspace,root):
    """Lower bound before allocating Python factors for a giant real surface.
    Any shared raw pixel (including a canonical source) connects the points.
    This NEVER permits a cut: the actual graph repeats a complete component check.
    """
    N=len(source);parent=list(range(N));sizes=[1]*N
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            if sizes[a]<sizes[b]:a,b=b,a
            parent[b]=a;sizes[a]+=sizes[b]
    first={(int(c),anchor,int(pix)):i for i,(_,c,pix) in enumerate(source)}
    weights=rows.weights.tolist() if hasattr(rows,'weights') else [[1.]*4 for _ in range(len(rows.point))]
    for i,c,k,pixs,ww in zip(rows.point.tolist(),rows.camera.tolist(),rows.reference_frame.tolist(),rows.reference_pixels.tolist(),weights):
        for pix,weight in zip(pixs,ww):
            if weight==0:continue # exact zero only; no uncertainty/geometry threshold
            key=(c,k,pix)
            if key in first:union(i,first[key])
            else:first[key]=i
    counts=sorted([sizes[i] for i in range(N) if parent[i]==i],reverse=True)
    # Current state alone supplies six unknowns; pixels and past states only increase this bound.
    minimum=8*8*(6*counts[0])**2 if counts else 0
    report={'points':N,'components':len(counts),'largest_point_components':counts[:20],
            'minimum_largest_component_workspace_bytes':minimum,'workspace_budget_bytes':workspace,
            'bound':'6 current scalars per connected point; excludes additional states/latents'}
    json_write(root/'component_preflight.json',report)
    if minimum>workspace:
        raise FullResourceLimit('FULL_RESOURCE_LIMIT: '+str(report)+'; exact component backend cannot fit; no simplified fallback')


def initialize_full(graph,s,source,k,frame,rig,config,noise):
    for i,(_,c,pix) in enumerate(tqdm(source,desc='Coupled source graph',unit='point')):
        c,pix=int(c),int(pix)
        value=raw_values(frame,torch.tensor(c,device=s.device),torch.tensor(pix,device=s.device)).double()
        graph.add_coupled_anchor_factors(i,k,(c,k,pix),value,sensor_covariance(value,noise),
                                        graph.tensor([pix%frame.W,pix//frame.W]),rig,c,config.pixel_covariance,config.velocity_covariance,s[i,:3])
    return s


def add_full_rows(graph,rows,k,frame,frames,rig,noise,warp):
    for a in range(len(rows)):
        i,c,rf,pix=(int(rows.point[a]),int(rows.camera[a]),int(rows.reference_frame[a]),int(rows.pixel[a]))
        current=graph.pixels.add_raw_sensor_factors_once((c,k,pix),rows.y[a].double(),sensor_covariance(rows.y[a].double(),noise))
        refkeys=[]
        for pixref in rows.reference_pixels[a].tolist():
            v=raw_values(frames[rf],torch.tensor(c,device=graph.device),torch.tensor(pixref,device=graph.device)).double()
            refkeys.append(graph.pixels.add_raw_sensor_factors_once((c,rf,pixref),v,sensor_covariance(v,noise)))
        graph.add_warp_factors(i,k,rf,c,graph.tensor([pix%frame.W,pix//frame.W]),refkeys,current,rig,warp,rows.cell[a])


def refresh_full_reference_support(graph,frames,rig,noise,policy):
    """Refresh nonlinear contributors, retaining the union of sensor factors.
    Newly touched raw variables start at their sensor means (zero extra sensor
    residual in both baseline and candidate). No old observation is deleted.
    """
    factors=[f for f in graph.factors if f.context is not None]
    if not factors:return True
    contexts=[f.context for f in factors];cache={}
    def state(i,k):
        if (i,k) not in cache:cache[i,k]=graph.state(i,k)[0]
        return cache[i,k]
    st=torch.stack([state(c['point'],c['frame']) for c in contexts]);sr=torch.stack([state(c['point'],c['reference']) for c in contexts])
    cameras=torch.tensor([c['camera'] for c in contexts],device=graph.device);uv=graph.tensor([c['pixel_uv'] for c in contexts])
    q=uv-rig.project_rows(st[:,:3],cameras)[0]+rig.project_rows(sr[:,:3],cameras)[0]
    if not bool(torch.isfinite(q).all()):return False
    cells=torch.stack([c['cell'] for c in contexts]);changed=torch.where((q.floor()!=cells).any(1))[0]
    # Within an unchanged cell the four raw keys, foreground flags, validity and
    # depth-surface test are identical. Only interpolation weights vary.
    for rf in {contexts[int(ix)]['reference'] for ix in changed}:
        ix=torch.tensor([int(i) for i in changed if contexts[int(i)]['reference']==rf],device=graph.device)
        pixs,vals,reason,newcells=validate_reference_patch(frames[rf],cameras[ix],q[ix],rig,noise,policy)
        if bool((reason!=0).any()):return False
        for j,a in enumerate(ix.tolist()):
            c=contexts[a]['camera'];keys=[]
            for pp,val in zip(pixs[j].tolist(),vals[j]):keys.append(graph.pixels.add_raw_sensor_factors_once((c,rf,pp),val,sensor_covariance(val,noise)))
            factors[a].bind_reference(keys,newcells[j])

    return True


def run_temporal_free(dataset,config,policy,direction,root,device,sub,options,debug_frames=(),debug_point_ids=None,
                      max_iters=10,tol=1e-8,save_full_covariance=False,save_state_interval=0,progress=None):
    root=Path(root);numeric=root/'numeric';temporal=root/'temporal'
    numeric.mkdir(parents=True,exist_ok=True);temporal.mkdir(exist_ok=True)
    for folder in ('traced_points','debug_frames'):(temporal/folder).mkdir(exist_ok=True)
    dev=torch.device(device);dtype=torch.float64
    rig=CameraRig(dataset.cameras,device,dtype);noise=SensorNoise.from_config(config,policy,device,dtype)
    warp=torch.as_tensor(options['warp_covariance'],device=dev,dtype=dtype)
    k0,xyz,rgb,cov,source,sel,sm=build_catalogue(dataset,config,direction,sub,policy,noise)
    if options.get('diagnostic_original_point_ids') is not None:
        from scene.canonical_subsample import ids_hash
        sel=np.asarray(sorted(set(options['diagnostic_original_point_ids'])),dtype=int)
        if not len(sel) or sel.min()<0 or sel.max()>=len(xyz):raise ValueError('Diagnostic original point IDs outside full endpoint catalogue')
        sm={'diagnostic_focal_control':True,'endpoint_selection_before_focal':sm,'requested_endpoint_factor':sub['factor'],
            'factor':None,'method':'explicit-original-ID-focal-control','N_full':len(xyz),'N_target':len(sel),'N_selected':len(sel),
            'actual_reduction':len(xyz)/len(sel),'selected_ids_sha256':ids_hash(sel),'selected_original_ids':sel.tolist(),
            'note':'Original endpoint IDs unchanged; focal selection removes other occluders; NOT full-cloud validation'}
    selected=source[sel];N=len(sel);K=len(dataset.times);C=rig.count;lag=options['lag']
    order=list(range(K)) if direction=='forward' else list(range(K-1,-1,-1))
    original_to_local={int(v):i for i,v in enumerate(sel)}
    requested=list(debug_point_ids) if debug_point_ids is not None else sel[np.linspace(0,N-1,min(6,N)).astype(int)].tolist()
    ids=[original_to_local[i] for i in requested if i in original_to_local]
    np.savez_compressed(root/'canonical.npz',canonical_frame=k0,canonical_xyz=xyz[sel],canonical_rgb=rgb[sel],original_source_ids=selected,
                        point_id=sel,local_array_index=np.arange(N),subsampled_to_full_index=sel,canonical_source_covariance=cov[sel],
                        full_endpoint_xyz=xyz,full_endpoint_rgb=rgb,full_endpoint_source=source)
    json_write(root/'subsample_manifest.json',serial(sm))
    specs={'posterior_xyz':((K,N,3),'float64'),'predicted_xyz':((K,N,3),'float32'),'velocity':((K,N,3),'float32'),
           'position_variance_diag':((K,N,3),'float32'),'velocity_variance_diag':((K,N,3),'float32'),
           'camera_reason':((K,C,N),'uint8')}
    for name,dt in [('valid_cameras','uint8'),('point_status','uint8'),('converged','bool'),('iekf_iterations','uint8'),
                    ('nis','float32'),('nis_dof','uint8'),('information_min_eig','float32'),('information_max_eig','float32'),
                    ('update_acceptance','uint8'),('posterior_cost','float32'),('posterior_cost_dof','uint8')]:specs[name]=((K,N),dt)
    mm={name:np.lib.format.open_memmap(root/f'{name}.npy',mode='w+',dtype=dt,shape=sh) for name,(sh,dt) in specs.items()}
    for arr in mm.values():arr[:]=0
    s=torch.zeros((N,6),dtype=dtype,device=dev);s[:,:3]=torch.as_tensor(xyz[sel],device=dev)
    P=torch.zeros((N,6,6),dtype=dtype,device=dev);P[:,:3,:3]=torch.as_tensor(cov[sel],device=dev);P[:,3:,3:]=torch.as_tensor(config.velocity_covariance,device=dev)
    refs=ReferenceTable(C,N,device,dtype)
    frames={k0:MeasuredFrame(dataset,k0,device,dtype)}
    owners=first_hit_pixel_owner(s[:,:3],rig,frames[k0])
    refs.commit_accepted_references(owners,frames[k0],s,0)
    refs.initialize_from_source_catalogue(selected,k0,s) # preserves known source even when cross-camera render loses
    _store(mm,k0,s,s,P,None,C)
    mm['point_status'][k0]=0
    manifest={'pass':'F1' if direction=='forward' else 'B1','direction':direction,'canonical_frame':k0,'point_count':N,
              'processing_order':order,'timestamps_seconds':dataset.times.tolist(),'original_frame_indices':list(dataset.frame_ids),
              'debug_point_ids':ids,'debug_frames':list(debug_frames),'observation_model':'temporal-rgbd','observation_model_version':1,
              'temporal_mode':options['mode'],'lag':lag,'precision':'float64','full_covariance_saved':save_full_covariance,'camera_count':C,
              'noise_and_prior':config.manifest(),'calibration_hash':rig.calibration_hash(),'reason_codes':TEMPORAL_REASONS,
              'reference_uncertainty':REFERENCE_UNCERTAINTY if options['mode']=='simplified' else 'joint latent pixels and point states; exact component marginals; FEJ lag prior',
              'anchor_initialization':'coupled source pixel/depth/velocity generative parameterization' if options['mode']=='full' else 'legacy independent source Pxx; APPROXIMATE_COVARIANCE',
              'owner_semantics':'nearest integer, round ties to even; depth then original canonical ID; source identity stored separately',
              'unexplained_semantics':'uncovered renderer pixels; not proof of new physical geometry and not exclusive tracking ownership',
              'status_codes':{0:'ANCHOR_CANONICAL',1:'DIRECT_TEMPORAL_ROWS',2:'NO_DIRECT_ROW_JOINT_MARGINAL' if options['mode']=='full' else 'PREDICTION_ONLY'},
              'downstream':'TEMPORAL_FREE_WITH_LEGACY_RIGID','pixel_reuse_assignment':'per accepted nonlinear linearization',
              'q_c_m2_s3':config.qc.tolist(),'source_identity_map':selected.tolist() if N<10 else 'canonical.npz original_source_ids',
              'objective_scope':'conservative common current pixel/owner/reference domain; refresh reference cells; retain union of raw sensor pixels',
              'cross_time_marginalization':'FEJ, full Schur retained; references outside lag unusable'}
    manifest.update({'execution_kind':'ACTUAL_TEMPORAL_IEKF_ESTIMATION','noise_assumption':options['warp_provenance'].get('assumption_class'),
                     'diagnostic_focal_control':options.get('diagnostic_original_point_ids') is not None,
                     'original_point_ids':sel.tolist() if N<1000 else 'canonical.npz point_id','debug_original_point_ids':sel[ids].tolist()})
    manifest['information_diagnostic_semantics']=('inverse joint marginal P_position, not independent-row information' if options['mode']=='full' else 'conditional independent-row position information sum')
    json_write(root/'run_manifest.json',serial(manifest));json_write(temporal/'config.json',serial(options));json_write(temporal/'calibration.json',serial(options['warp_provenance']))
    graph=None;start=time.perf_counter();traces={i:[] for i in ids}
    anchor_records={}
    for i in ids:
        entries=[]
        for c in range(C):
            u,v=owners.uv[c,i].round().long().tolist()
            if 0<=u<frames[k0].W and 0<=v<frames[k0].H:
                winner=int(owners.owner_id[c,v,u]);entries.append({'camera':c,'pixel':[u,v],'winner_local_index':winner,
                  'winner_original_id':int(sel[winner]) if winner>=0 else None,'source_original_id':int(sel[i]),
                  'point_depth_m':float(owners.z[c,i]),'winner_depth_m':float(owners.owner_depth[c,v,u]),
                  'source':selected[i].tolist(),'winner_source':selected[winner].tolist() if winner>=0 else None})
        anchor_records[int(sel[i])]=entries
    json_write(temporal/'anchor_source_ownership.json',serial(anchor_records))
    from debug.temporal_debug import temporal_frame_debug,write_temporal_dashboard
    if k0 in debug_frames:temporal_frame_debug(root,frames[k0],owners,s,s,rig,None,k0)
    write_temporal_dashboard(root,manifest,[])
    history=[]
    try:
        for ordinal,k in enumerate(tqdm(order[1:],desc=f'Temporal {options["mode"]} {direction}',unit='frame'),1):
            t0=time.perf_counter();prev=order[ordinal-1];dt=float(dataset.times[k]-dataset.times[prev])
            mean,prior=predict(s,P,dt,torch.as_tensor(config.qc,dtype=dtype,device=dev))
            tin=time.perf_counter();frames[k]=MeasuredFrame(dataset,k,device,dtype);input_seconds=time.perf_counter()-tin
            for old in list(frames):
                if old not in order[max(0,ordinal-lag+1):ordinal+1]:del frames[old]
            stage_times={'input_transfer_seconds':input_seconds,'z_sort_seconds':0.,'projection_device_seconds':0.,'ownership_device_seconds':0.,'sampling_seconds':0.,'assembly_seconds':0.,'solve_seconds':0.,'schur_seconds':0.,'source_graph_seconds':0.}
            def build(states):
                if graph is not None:
                    for c,i in zip(*torch.where((refs.frame>=0)&((ordinal-refs.ordinal)<lag))):
                        rf=int(refs.frame[c,i]);refs.state[c,i]=graph.state(int(i),rf)[0]
                tb=time.perf_counter();own=first_hit_pixel_owner(states[:,:3],rig,frames[k],dense=False)
                stage_times['projection_device_seconds']+=own.timing['projection_seconds'];stage_times['ownership_device_seconds']+=own.timing['ownership_sort_storage_seconds']
                stage_times['z_sort_seconds']+=time.perf_counter()-tb;tb=time.perf_counter()
                rows,re,li=build_temporal_rows(states,own,frames[k],refs,frames,rig,noise,policy,warp,ordinal,lag)
                stage_times['sampling_seconds']+=time.perf_counter()-tb
                return rows,own,re,li
            before_rows,before_owner,_,_=build(mean)
            if options['mode']=='simplified':
                ts=time.perf_counter()
                s,P,rows,owners,reason,life,solver=update_points_temporal_simplified(mean,prior,build,frames[k],max_iters,tol)
                assimilated_mask=solver.pop('_assimilated_mask')
                stage_times['solve_seconds']=time.perf_counter()-ts # includes relinearization; not an exclusive sum
            else:
                if graph is None:
                    tinit=time.perf_counter()
                    full_component_preflight(before_rows,selected,k0,options['workspace_bytes'],temporal)
                    graph=TemporalWindowGraph(device,options['workspace_bytes'])
                    if options.get('runtime_budget_seconds') is not None:graph.deadline=time.perf_counter()+options['runtime_budget_seconds']
                    # Use exactly the same source chart for the prediction and
                    # reference, avoiding artificial subpixel flow from NumPy vs
                    # torch backprojection roundoff at interpolation boundaries.
                    s=initialize_full(graph,s,selected,k0,frames[k0],rig,config,noise)
                    refs=ReferenceTable(C,N,device,dtype)
                    anchor_owner=first_hit_pixel_owner(s[:,:3],rig,frames[k0])
                    refs.commit_accepted_references(anchor_owner,frames[k0],s,0)
                    refs.initialize_from_source_catalogue(selected,k0,s)
                    _store(mm,k0,s,s,P,None,C)
                    mean,prior=predict(s,P,dt,torch.as_tensor(config.qc,dtype=dtype,device=dev))
                    before_rows,before_owner,_,_=build(mean)
                    stage_times['source_graph_seconds']=time.perf_counter()-tinit
                cutoff=set(order[:max(0,ordinal-lag+1)])
                expired=[key for key in graph.values if (key[2] if key[0]!='pixel' else key[2]) in cutoff]
                ts=time.perf_counter()
                if expired:
                    checks=[]
                    if options.get('diagnostic_original_point_ids') is not None and graph.posterior:
                        for i in ids:
                            _,ja=graph.state(i,prev)
                            jb=ja
                            for c in range(C):
                                key=('pixel',c,int(refs.frame[c,i]),int(refs.pixel[c,i]))
                                if key in graph.values and key not in expired:
                                    jb={key:torch.eye(4,device=dev,dtype=dtype)};break
                            checks.append((int(sel[i]),ja,jb,graph.covariance(ja,ja).clone(),graph.covariance(ja,jb).clone()))
                    graph.marginalize_expired_exactly(expired)
                    if checks:
                        graph.factor_posterior();parity=[]
                        for original,ja,jb,pbefore,crossbefore in checks:
                            pafter=graph.covariance(ja,ja);crossafter=graph.covariance(ja,jb)
                            parity.append({'original_point_id':original,'retained_frame':prev,
                                'P_before':pbefore.tolist(),'P_after':pafter.tolist(),
                                'state_covariance_max_abs':float((pafter-pbefore).abs().max()),
                                'state_pixel_cross_before':crossbefore.tolist(),'state_pixel_cross_after':crossafter.tolist(),
                                'cross_covariance_max_abs':float((crossafter-crossbefore).abs().max()),
                                'semantics':'same real accepted linearization, before/after exact constrained Schur; no new data or state update'})
                        json_write(temporal/f'real_schur_parity_{k:04d}.json',serial(parity))
                stage_times['schur_seconds']=time.perf_counter()-ts;ts=time.perf_counter()
                F=transition(dt,dtype,device);Q=process_noise(dt,torch.as_tensor(config.qc,device=dev,dtype=dtype))
                for i in tqdm(range(N),desc='Process factors',unit='point',leave=False):graph.add_process_factors(i,prev,k,F,Q,mean[i])
                rows,_,_,_=build(mean);signature=row_signature(rows);baseoutside=foreground_violations(before_owner,frames[k])
                factor_start=len(graph.factors)
                add_full_rows(graph,rows,k,frames[k],frames,rig,noise,warp)
                stage_times['assembly_seconds']=time.perf_counter()-ts
                predictive_nis=graph.correlated_warp_nis(factor_start)
                baseline_outside={}
                for fk,ff in frames.items():
                    ss=torch.stack([graph.state(i,fk)[0] for i in range(N)])
                    baseline_outside[fk]=foreground_violations(first_hit_pixel_owner(ss[:,:3],rig,ff,dense=False),ff)
                def accept(g):
                    states=torch.stack([g.state(i,k)[0] for i in range(N)])
                    rr,oo,_,_=build(states)
                    if not same_domain(signature,row_signature(rr)) or not foreground_violations(oo,frames[k]).issubset(baseoutside):return False
                    maps={k:oo}
                    for fk,ff in frames.items():
                        if fk!=k:
                            ss=torch.stack([g.state(i,fk)[0] for i in range(N)])
                            maps[fk]=first_hit_pixel_owner(ss[:,:3],rig,ff,dense=False)
                        if not foreground_violations(maps[fk],ff).issubset(baseline_outside[fk]):return False
                    for factor in g.factors:
                        if factor.context is None:continue
                        ctx=factor.context;u,v=ctx['pixel_uv'];own=maps[ctx['frame']]
                        if owner_at(own,ctx['camera'],int(v)*frames[ctx['frame']].W+int(u))!=ctx['point']:return False
                    # A changed reference cell is a new interpolation of the
                    # SAME image, not a lost observation. Register its raw sensor
                    # pixels once in BOTH comparison domains; keep old sensors.
                    return refresh_full_reference_support(g,frames,rig,noise,policy)
                active_keys={('state',int(i),k) for i in rows.point.unique().tolist()}
                graph.publication_frame=k;graph.publication_points=N
                graph.retain_keys={('state',i,k) for i in ids}
                ts=time.perf_counter();solver=graph.solve_sparse_map(max_iters,tol,accept,active_keys)
                accepted_state_keys=solver.pop('_accepted_state_keys')
                assimilated_mask=torch.tensor([('state',i,k) in accepted_state_keys for i in range(N)],device=dev,dtype=torch.bool)
                solver['assimilated_points']=int(assimilated_mask.sum())
                stage_times['solve_seconds']=time.perf_counter()-ts
                solver['correlated_predictive_innovation']=predictive_nis
                profiles=graph.profile
                json_write(temporal/f'components_{k:04d}.json',serial({'components':profiles,'global_policy':'NO_EDGES_CUT; EXACT_WITHIN_COMPONENT; FEJ linearization and finite lag are approximations'}))
                tcov=time.perf_counter();s,P=graph.extract_current_marginals(range(N),k);stage_times['covariance_publication_seconds']=time.perf_counter()-tcov
                active_components={graph.component_for_key[key] for key in active_keys}
                no_new_component=torch.tensor([graph.component_for_key[('state',i,k)] not in active_components for i in range(N)],device=dev)
                # Pure prediction is an exact branch, not a covariance recomputation
                # that adds roundoff and looks like a measurement contraction.
                s[no_new_component]=mean[no_new_component];P[no_new_component]=prior[no_new_component]
                rows,owners,reason,life=build(s)
            owners=first_hit_pixel_owner(s[:,:3],rig,frames[k]) # dense image only once for diagnostics/export
            tdiag=time.perf_counter();nrows=torch.bincount(rows.point,minlength=N);observed=nrows>0
            # Re-evaluate accepted nonlinear h using accepted clean pixels in full mode.
            if graph is not None:
                for a in range(len(rows)):
                    c,rf=int(rows.camera[a]),int(rows.reference_frame[a])
                    vals=torch.stack([graph.values[('pixel',c,rf,int(pix))] for pix in rows.reference_pixels[a]])
                    h,ht,hr,*_=temporal_prediction(rig,s[rows.point[a]:rows.point[a]+1],refs.state[c,rows.point[a]][None],rows.camera[a:a+1],
                                                 torch.stack((rows.pixel[a:a+1]%frames[k].W,rows.pixel[a:a+1]//frames[k].W),-1).double(),vals[None],rows.cell[a:a+1])
                    rows.h[a],rows.H_t[a],rows.H_ref[a]=h[0],ht[0],hr[0]
            residual=rows.y-rows.h
            info=torch.zeros((N,3,3),device=dev,dtype=dtype)
            if graph is not None:info=torch.linalg.inv(P[:,:3,:3])
            elif len(rows):info.index_add_(0,rows.point,rows.H_t[:,:,:3].transpose(-1,-2)@torch.linalg.solve(rows.R,rows.H_t[:,:,:3]))
            # Full correlated innovation is deliberately not mislabeled as independent 4m NIS.
            nis=torch.zeros(N,device=dev,dtype=dtype)
            if options['mode']=='simplified' and len(rows):
                nu=residual+torch.einsum('mij,mj->mi',rows.H_t,s[rows.point]-mean[rows.point])
                wr=torch.linalg.solve(rows.R,nu.unsqueeze(-1)).squeeze(-1)
                b=torch.zeros((N,6),device=dev,dtype=dtype);A=torch.zeros((N,6,6),device=dev,dtype=dtype)
                b.index_add_(0,rows.point,torch.einsum('mji,mj->mi',rows.H_t,wr))
                A.index_add_(0,rows.point,rows.H_t.transpose(-1,-2)@torch.linalg.solve(rows.R,rows.H_t))
                nis.index_add_(0,rows.point,(nu*wr).sum(1))
                nis-=(b*torch.linalg.solve(torch.linalg.inv(prior)+A,b.unsqueeze(-1)).squeeze(-1)).sum(1)
            rep={'nrows':nrows,'observed':observed,'reason':reason,'converged':torch.full((N,),solver['converged'],device=dev),
                 'point_iterations':torch.full((N,),len(solver['history']),device=dev),'nis':nis,
                 'nis_dof':4*nrows if options['mode']=='simplified' else torch.zeros_like(nrows),'info_eig':batched_eigvalsh(info),
                 'acceptance':assimilated_mask.to(torch.uint8),'posterior_cost':torch.zeros(N,device=dev),'posterior_cost_dof':torch.zeros_like(nrows)}
            _store(mm,k,s,mean,P,rep,C)
            stat=compare_owner_maps(before_owner,owners)
            unexplained=int((frames[k].mask & frames[k].raw_valid & (owners.owner_id<0)).sum())
            outside=len(foreground_violations(owners,frames[k]))
            unexpl_c,unexpl_y,unexpl_x=torch.where(frames[k].mask & frames[k].raw_valid & (owners.owner_id<0))
            np.savez_compressed(temporal/f'unexplained_{k:04d}.npz',frame_index=k,original_frame_id=dataset.frame_ids[k],
                                camera=unexpl_c.cpu().numpy(),pixel=(unexpl_y*frames[k].W+unexpl_x).cpu().numpy())
            summary={'frame':k,'ordinal':ordinal,'dt_seconds':dt,'rows':len(rows),'observed_points':int(observed.sum()),'prediction_only':int((~observed).sum()),
                     'nonsmooth_reference_cells':int(rows.nonsmooth.sum()),
                     'unexplained_foreground':unexplained,'rendered_outside_foreground':outside,'rgb_residual_rms':float(residual[:,:3].square().mean().sqrt()) if len(rows) else None,
                     'depth_residual_rms_m':float(residual[:,3].square().mean().sqrt()) if len(rows) else None,'solver':solver,'reference':life,**stat}
            summary['seconds']=time.perf_counter()-t0;summary['stage_times']=stage_times
            history.append(summary)
            summary['assimilated_points']=int(assimilated_mask.sum())
            summary['eligible_but_no_accepted_update']=int((observed&~assimilated_mask).sum())
            json_write(temporal/f'frame_{k:04d}.json',serial(summary))
            per=[]
            for c in range(C):
                counts={TEMPORAL_REASONS[int(v)]:int(n) for v,n in zip(*torch.unique(reason[c],return_counts=True))}
                ix=rows.camera==c
                per.append({'camera':rig.names[c],'reason_counts':counts,'rows_accepted_final':int(ix.sum()),
                            'depth_innovation_m_quantiles_prior':np.quantile(residual[ix,3].cpu(),[0,.5,1]).tolist() if bool(ix.any()) else None,
                            'information_trace_final':None})
                append_csv(temporal/'ownership_summary.csv',{'frame':k,'camera':c,'rows':int(ix.sum()),'occupied':int((owners.owner_id[c]>=0).sum()),'conflicts':int((owners.counts[c]>1).sum()),'outside_mask':int(((owners.owner_id[c]>=0)&~frames[k].mask[c]).sum())})
            eig=batched_eigvalsh(P)
            qs=[0.,.05,.25,.5,.75,.95,1.]
            pq=np.quantile(batched_eigvalsh(P[:,:3,:3]).cpu(),qs).tolist()
            correction=np.quantile((s[:,:3]-mean[:,:3]).norm(dim=1).cpu(),qs).tolist()
            speed=np.quantile(s[:,3:].norm(dim=1).cpu(),qs).tolist()
            pd_failures=int((eig<=0).any(1).sum())
            summary.update({'covariance_min_eigenvalue_mixed_units':float(eig.min()),'covariance_max_eigenvalue_mixed_units':float(eig.max()),
                            'posterior_position_variance_median_m2':pq[3],'speed_median_m_s':speed[3],'speed_p95_m_s':speed[5],
                            'posterior_pd_failures':pd_failures})
            json_write(temporal/f'frame_{k:04d}.json',serial(summary))
            numeric_record={'frame':k,'sensor':{'unique_points_observed':int(observed.sum()),'prediction_only_points':int((~observed).sum()),'rows_accepted_final':len(rows),'per_camera':per},
                            'processing_step':ordinal,'timestamp_s':float(dataset.times[k]),'dt_s':dt,
                            'filter_update':{'not_converged_points':0 if solver['converged'] else N,'posterior_position_eig_m2_quantiles':pq,
                                             'posterior_minus_prior_position_m_quantiles':correction,'speed_m_s_quantiles':speed,
                                             'posterior_positive_definite_failures':pd_failures},
                            'innovation':{'nis_per_dof_quantiles_prior_observed':None},
                            'processing':{'cuda_ms':{},'wall_seconds':summary['seconds'],'wall_ms':summary['seconds']*1000,
                                          'gpu_peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6 if dev.type=='cuda' else 0,
                                          'iekf_iterations':len(solver['history'])},
                            'acceptance':{'counts':{}},'temporal':summary}
            json_write(numeric/f'frame_{k:04d}.json',serial(numeric_record))
            final={str(i):{'s_minus':mean[i].tolist(),'P_minus':prior[i].tolist(),'s_plus':s[i].tolist(),'P_plus':P[i].tolist()} for i in ids}
            json_write(numeric/f'trace_frame_{k:04d}.json',{'points':{'final':final}})
            for i in ids:
                ri=torch.where(rows.point==i)[0]
                record={'frame':k,'source':selected[i].tolist(),**final[str(i)],'rows':[],'solver':solver,
                        'measurement_update_accepted':bool(assimilated_mask[i]),
                        'camera_reason':reason[:,i].tolist(),'reference_frames':refs.frame[:,i].tolist(),
                        'reference_processing_age':(ordinal-refs.ordinal[:,i]).tolist()}
                for a in ri.tolist():
                    c,rf=int(rows.camera[a]),int(rows.reference_frame[a])
                    row={'camera':c,'current_pixel':int(rows.pixel[a]),'reference_frame':rf,'reference_pixels':rows.reference_pixels[a].tolist(),
                         'current_time_seconds':float(dataset.times[k]),'reference_time_seconds':float(dataset.times[rf]),
                         'q':rows.q[a].tolist(),'weights':rows.weights[a].tolist(),'H_t':rows.H_t[a].tolist(),'H_ref':rows.H_ref[a].tolist(),
                         'linearization_status':'NONSMOOTH' if bool(rows.nonsmooth[a]) else 'SMOOTH_INTERIOR',
                         'H_ell':(rows.weights[a,:,None,None]*torch.eye(4,device=dev,dtype=dtype)).tolist(),
                         'reference_state':refs.state[c,i].tolist(),'reference_raw_rgbd':rows.values[a].tolist(),
                         'current_raw_rgbd':rows.y[a].tolist(),'residual':residual[a].tolist(),'prediction':rows.h[a].tolist(),
                         'R_eff_simplified_diagnostic':rows.R[a].tolist(),'reference_support_mask':[True]*4,'current_mask':True}
                    row.update({'R_current_sensor':sensor_covariance(rows.y[a],noise).tolist(),
                         'R_reference_sensor_interpolated':(rows.weights[a].square()[:,None,None]*sensor_covariance(rows.values[a],noise)).sum(0).tolist(),
                         'R_additional_warp_model':warp.tolist(),'covariance_semantics':'sensor terms diagnostic only in full; raw likelihoods once + hard/soft warp + joint states'})
                    ref_uv,ref_z,_,_=rig.project_rows(refs.state[c,i,None,:3],rows.camera[a:a+1])
                    row['reference_projected_uv']=ref_uv[0].tolist();row['reference_camera_depth_m']=float(ref_z[0])
                    row['reference_covariance_link']='joint graph state and unique pixel keys' if graph else REFERENCE_UNCERTAINTY
                    pix=int(rows.pixel[a]);competitors=torch.where((owners.pixel[c]==pix)&(owners.z[c]>rig.near[c]))[0]
                    zz=owners.z[c,competitors];orderz=torch.argsort(zz,stable=True)
                    row['collision_ids_by_depth']=competitors[orderz].tolist();row['collision_depths_m']=zz[orderz].tolist()
                    if graph is not None:
                        _,jt=graph.state(i,k);_,jr=graph.state(i,rf)
                        ell=('pixel',c,rf,int(rows.reference_pixels[a,0]))
                        row['Cov_current_reference']=graph.covariance(jt,jr).tolist()
                        row['P_reference']=graph.covariance(jr,jr).tolist()
                        row['Cov_current_first_reference_pixel']=graph.covariance(jt,{ell:torch.eye(4,device=dev,dtype=dtype)}).tolist()
                        row['reference_clean_rgbd']=[graph.values[('pixel',c,rf,int(pp))].tolist() for pp in rows.reference_pixels[a]]
                        current_key=('pixel',c,k,int(rows.pixel[a]));row['g']=(graph.values[current_key]-rows.h[a]).tolist()
                    record['rows'].append(row)
                record['original_point_id']=int(sel[i]);record['local_array_index']=i
                traces[i].append(record);json_write(temporal/'traced_points'/f'{sel[i]}.json',serial({'point_id':int(sel[i]),'local_array_index':i,'frames':traces[i]}))
            if graph is not None and k in debug_frames:
                saved=set()
                for i in ids:
                    _,jac=graph.state(i,k)
                    for key in jac:
                        ci=graph.component_for_key[key]
                        if ci in saved:continue
                        saved.add(ci);comp=graph.posterior[ci]
                        keys=comp['keys'];positions=[(key,comp['layout'][key].start,comp['layout'][key].stop) for key in keys]
                        np.savez_compressed(temporal/'traced_points'/f'frame_{k:04d}_component_{ci}.npz',
                                            information=comp['H'].cpu().numpy(),rhs=comp['b'].cpu().numpy(),constraints=comp.get('constraint_matrix',graph.tensor([])).cpu().numpy(),
                                            linearized_J=comp['linearized_J'].cpu().numpy(),linearized_residual=comp['linearized_residual'].cpu().numpy(),constraint_rows=comp['constraint_rows'].cpu().numpy(),
                                            marginal_covariance=comp['cov'].cpu().numpy(),
                                            linearization=np.concatenate([graph.values[key].cpu().numpy() for key in keys]),
                                            layout=np.array(str(positions)),marginalization_policy=np.array('FEJ'))
            append_csv(temporal/'reference_lifecycle.csv',{'frame':k,**life})
            append_csv(temporal/'innovation_summary.csv',{'frame':k,'rows':len(rows),'rgb_rms':summary['rgb_residual_rms'],'depth_rms_m':summary['depth_residual_rms_m'],
                                                        'NIS':float(nis.sum()) if graph is None else predictive_nis['NIS'],
                                                        'dof':4*len(rows) if graph is None else predictive_nis['dof'],
                                                        'NIS_status':'conditional simplified prior' if graph is None else predictive_nis['model']})
            append_csv(temporal/'covariance_summary.csv',{'frame':k,'min_eigenvalue':float(eig.min()),'max_eigenvalue':float(eig.max()),'type':manifest['reference_uncertainty'],
                                                       'schur_eliminations':len(graph.marginalization_log) if graph else 0})
            from estimator.free_projective_run import _timeline_row
            append_csv(numeric/'timeline.csv',_timeline_row(numeric_record))
            if save_full_covariance and (k in debug_frames or (save_state_interval and ordinal%save_state_interval==0)):
                (root/'states').mkdir(exist_ok=True);np.savez_compressed(root/'states'/f'frame_{k:04d}.npz',s=s.cpu().numpy(),P=P.cpu().numpy())
            stage_times['diagnostics_export_seconds']=time.perf_counter()-tdiag
            tdebug=time.perf_counter()
            if k in debug_frames:temporal_frame_debug(root,frames[k],owners,s,mean,rig,rows,k)
            stage_times['viewer_export_seconds']=time.perf_counter()-tdebug
            tref=time.perf_counter();refs.commit_accepted_references(owners,frames[k],s,ordinal);stage_times['reference_refresh_seconds']=time.perf_counter()-tref
            summary['seconds_including_export']=time.perf_counter()-t0
            append_csv(temporal/'timing_memory.csv',{'frame':k,'seconds':summary['seconds_including_export'],**stage_times,'cuda_peak_bytes':torch.cuda.max_memory_allocated() if dev.type=='cuda' else 0,'cpu_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            json_write(temporal/f'frame_{k:04d}.json',serial(summary))
            write_temporal_dashboard(root,manifest,history)
            for arr in mm.values():arr.flush()
            if progress:print(f'[{direction}] {ordinal}/{K-1} frame {k} rows {len(rows)} observed {int(observed.sum())}/{N} '
                              f'{summary["seconds"]:.2f}s | live debug: {temporal/"index.html"}',flush=True)
        if graph:
            json_write(temporal/'marginalization.json',serial(graph.marginalization_log))
            json_write(temporal/'unique_sensor_registry.json',{'total_unique_sensor_factors':graph.pixels.total_sensor_factors,
                       'active_pixel_count':len(graph.pixels.sensor_counts),'retired_frames':sorted(graph.pixels.retired_frames),
                       'maximum_sensor_factors_per_active_pixel':max(graph.pixels.sensor_counts.values(),default=0)})
        result={'status':'COMPLETE','frames':K,'point_count':N,'seconds':time.perf_counter()-start,'mode':options['mode'],'diagnostic_status':'JOINT_LOCAL_GAUSSIAN' if graph else 'APPROXIMATE_COVARIANCE'}
        json_write(root/'status.json',result)
        return result
    except Exception as err:
        failure={'status':getattr(err,'status','FAILED'),'reason':str(err),'completed_frames':len(history)+1,
                 'point_count':N,'elapsed_filter_seconds':time.perf_counter()-start,'cuda_peak_bytes':torch.cuda.max_memory_allocated() if dev.type=='cuda' else 0,
                 'cpu_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 'graph_variables':len(graph.values) if graph else None,'graph_factors':len(graph.factors) if graph else None,
                 'covariance_policy':'no graph truncation; incomplete frames not valid estimates'}
        json_write(root/'status.json',failure)
        write_temporal_dashboard(root,manifest,history,str(err))
        raise
