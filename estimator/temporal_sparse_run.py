"""Production FULL temporal F1/B1; all numeric inference is typed sparse CUDA.

The dense graph and simplified update are not imported. Only bounded selected
point diagnostics use Python loops; every original point participates in inference.
"""
import csv
from pathlib import Path
import resource
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.io import json_write, serial
from utils.temporal_output import VerifiedFrameOutput
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise
from estimator.free_projective_run import build_catalogue, _store, _timeline_row
from estimator.point_kalman_cuda import predict, transition, process_noise
from estimator.temporal_zbuffer import first_hit_pixel_owner, selected_owners, compare_owner_maps
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_rgbd_terms import build_temporal_rows, raw_values, sensor_covariance, temporal_prediction, TEMPORAL_REASONS
from estimator.temporal_sparse_gpu import TemporalSoAGraph, SparseBlockRows, sparse_coo
from estimator.temporal_sparse_nonlinear import solve_nonlinear_sparse_iekf_gpu, outside_mask
from estimator.temporal_sparse_solve import connected_columns_gpu, sparse_rows
from estimator.temporal_cudss import SparseNumericalFailure
from debug.temporal_debug import temporal_frame_debug, write_temporal_dashboard


def append_csv(path, row):
    new=not path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if new:w.writeheader()
        w.writerow(row)


def signature(rows):
    return torch.stack((rows.camera,rows.point,rows.pixel,rows.reference_frame),-1)


def batched_eigen(matrix):
    return torch.cat([torch.linalg.eigvalsh(part) for part in matrix.split(256)])


def point_cross_query(graph,factor,point,current,reference,first_pixel_key):
    point=torch.tensor([point],device=graph.device,dtype=torch.long)
    q1=graph.state_query_gpu(current,point).coalesce()
    q2=graph.state_query_gpu(reference,point).coalesce()
    pixel=graph.pixel_columns(first_pixel_key.reshape(1)).flatten()
    row=torch.arange(4,device=graph.device)
    index=torch.cat((q1.indices(),q2.indices()+torch.tensor([[6],[0]],device=graph.device),torch.stack((row+12,pixel))),1)
    value=torch.cat((q1.values(),q2.values(),torch.ones(4,device=graph.device,dtype=torch.float64)))
    Q=sparse_coo(index,value,(16,graph.scalar_dimension))
    P=factor.extract_joint_selected_marginals_gpu(Q,16)[0]
    return P[:6,6:12],P[6:12,6:12],P[:6,12:]


def save_real_sparse_oracle(root,k,factor,graph,limit=2,oracle_dimension_limit=256):
    """Archive exact bounded components, never cut a production component."""
    from estimator.temporal_sparse_marginalization import sparse_slice
    labels,_=connected_columns_gpu(factor.A,factor.C,factor.information)
    unique,counts=torch.unique(labels,return_counts=True)
    ti=factor.T.indices();offset=graph.state_offsets[k]
    state_components=torch.unique(labels[ti[1,(ti[0]>=offset)&(ti[0]<offset+6*graph.N)]])
    measurement_components=torch.unique(labels[factor.A.indices()[1]])
    eligible=(counts>=12)&(counts<=oracle_dimension_limit)&torch.isin(unique,state_components)&torch.isin(unique,measurement_components)
    small=unique[eligible][:limit]
    # These are validation queries, not inference fallback or a replacement mean.
    factor.solve_constrained_sparse_joint_gpu()
    pivots=torch.cat([p for _,p in factor.problem.pivot_groups]) if factor.problem.pivot_groups else torch.empty(0,device=labels.device,dtype=torch.long)
    free=torch.ones(factor.problem.scalar_dimension,device=labels.device,dtype=torch.bool);free[pivots]=False
    original=torch.where(free)[0]
    unit_row=torch.full((factor.A.shape[1],),-1,device=labels.device,dtype=torch.long)
    if factor.problem.prior_identity_rows is not None:
        unit_row[factor.problem.prior_identity_columns]=factor.problem.prior_identity_rows
    saved=0
    for ordinal,component in enumerate(small):
        cols=torch.where(labels==component)[0]
        ai=factor.A.indices();ci=factor.C.indices()
        ar=torch.unique(ai[0,labels[ai[1]]==component]);cr=torch.unique(ci[0,labels[ci[1]]==component])
        A=sparse_slice(factor.A,ar,cols).to_dense();C=sparse_slice(factor.C,cr,cols).to_dense()
        chosen=torch.where(unit_row[cols]>=0)[0][:6] if factor.problem.prior_transform is not None else torch.arange(min(6,len(cols)),device=cols.device)
        if not len(chosen):continue
        chosen_free=cols[chosen]
        if factor.problem.prior_transform is not None:
            Q=sparse_rows(factor.problem.A,unit_row[chosen_free])
        else:
            ri=original[chosen_free]
            lookup=sparse_slice(factor.T,ri,chosen_free).to_dense().diagonal()
            Q=sparse_coo(torch.stack((torch.arange(len(chosen),device=cols.device),ri)),lookup.reciprocal(),(len(chosen),factor.problem.scalar_dimension))
        P=factor.extract_joint_selected_marginals_gpu(Q,len(chosen))[0]
        # Gaussian prior rows are the inverse innovation chart. No inverse of
        # a physical point's conditional information is used for covariance.
        if factor.problem.prior_transform is not None:
            if bool((unit_row[cols]<0).any()):continue
            step=torch.sparse.mm(sparse_rows(factor.problem.A,unit_row[cols]),factor.step[:,None]).flatten()
        else:step=factor.step[original[cols]]/sparse_slice(factor.T,original[cols],cols).to_dense().diagonal()
        H=factor.information
        np.savez_compressed(root/f'oracle_frame_{k:04d}_component_{ordinal}.npz',A=A.cpu().numpy(),r=factor.r[ar].cpu().numpy(),
            C=C.cpu().numpy(),c=factor.c[cr].cpu().numpy(),step=step.cpu().numpy(),selected=chosen.cpu().numpy(),
            selected_covariance=P.cpu().numpy(),FEJ_information=sparse_slice(H,cols,cols).to_dense().cpu().numpy() if H is not None else np.zeros((len(cols),len(cols))),
            FEJ_gradient=factor.information_gradient[cols].cpu().numpy(),
            qualifier=np.array('Actual complete linearized sparse component; next Newton proposal, not a second accepted update'))
        saved+=1
    return {'saved':saved,'dimension_limit':oracle_dimension_limit,'complete_components_only':True,
            'selection':'nontrivial complete components intersecting actual current-state and measurement operators'}


def run_temporal_sparse_free(dataset,config,policy,direction,root,device,sub,options,debug_frames=(),debug_point_ids=None,
                            max_iters=10,tol=1e-8,save_full_covariance=False,save_state_interval=0,progress=None):
    if device!='cuda':raise ValueError('FULL_SPARSE_GPU_REQUIRES_CUDA; dense graph is test-only, not an automatic CPU fallback')
    if options.get('diagnostic_original_point_ids') is not None:raise ValueError('Full sparse production uses the whole selected catalogue; focal controls are not production inference')
    root=Path(root);numeric=root/'numeric';temporal=root/'temporal'
    for p in [numeric,temporal,temporal/'traced_points',temporal/'debug_frames']:p.mkdir(parents=True,exist_ok=True)
    rig=CameraRig(dataset.cameras,device);noise=SensorNoise.from_config(config,policy,device,torch.float64)
    k0,xyz,rgb,cov,source,sel,sm=build_catalogue(dataset,config,direction,sub,policy,noise)
    N=len(sel);K=len(dataset.times);C=rig.count;selected=source[sel];dev=rig.device
    order=list(range(K)) if direction=='forward' else list(range(K-1,-1,-1))
    requested=list(debug_point_ids) if debug_point_ids is not None else sel[np.linspace(0,N-1,min(6,N)).astype(int)].tolist()
    ids=np.flatnonzero(np.isin(sel,requested)).tolist()
    specs={'posterior_xyz':((K,N,3),'float64'),'predicted_xyz':((K,N,3),'float32'),'velocity':((K,N,3),'float32'),
        'position_variance_diag':((K,N,3),'float32'),'velocity_variance_diag':((K,N,3),'float32'),'camera_reason':((K,C,N),'uint8'),
        'posterior_covariance':((K,N,6,6),'float64')}
    for name,dtype in [('valid_cameras','uint8'),('point_status','uint8'),('converged','bool'),('iekf_iterations','uint8'),('nis','float32'),
        ('nis_dof','uint8'),('information_min_eig','float32'),('information_max_eig','float32'),('update_acceptance','uint8'),('posterior_cost','float32'),('posterior_cost_dof','uint8')]:
        specs[name]=((K,N),dtype)
    debug_count=len(set(debug_frames)&set(order))
    panel_bound=2*C*max(rig.width)*max(rig.height)*12*3*debug_count
    snapshot_bound=N*42*8*K if save_full_covariance else 0
    catalogue_bound=2*(xyz.nbytes+rgb.nbytes+cov.nbytes+source.nbytes)+1024**2
    publication=VerifiedFrameOutput(root,specs,order,options.get('output_headroom_bytes',128*1024**2),panel_bound+snapshot_bound+catalogue_bound)
    mm=publication.arrays
    np.savez_compressed(root/'canonical.npz',canonical_frame=k0,canonical_xyz=xyz[sel],canonical_rgb=rgb[sel],original_source_ids=selected,
        point_id=sel,local_array_index=np.arange(N),subsampled_to_full_index=sel,canonical_source_covariance=cov[sel],
        full_endpoint_xyz=xyz,full_endpoint_rgb=rgb,full_endpoint_source=source)
    json_write(root/'subsample_manifest.json',serial(sm))
    source_gpu=torch.as_tensor(selected,device=dev,dtype=torch.long)
    frames={k0:MeasuredFrame(dataset,k0,device,torch.float64)}
    value=raw_values(frames[k0],source_gpu[:,1],source_gpu[:,2])
    graph=TemporalSoAGraph(rig,K,N,source_gpu[:,1],source_gpu[:,2],torch.as_tensor(xyz[sel],device=dev),value,
        sensor_covariance(value,noise),torch.as_tensor(config.pixel_covariance,device=dev),torch.as_tensor(config.velocity_covariance,device=dev),k0)
    graph.layout();s=graph.state_and_blocks_gpu(k0)[0]
    P=torch.zeros((N,6,6),device=dev,dtype=torch.float64)
    P[:,:3,:3]=torch.as_tensor(cov[sel],device=dev);P[:,3:,3:]=torch.as_tensor(config.velocity_covariance,device=dev)
    warp=torch.as_tensor(options['warp_covariance'],device=dev,dtype=torch.float64)
    owners=first_hit_pixel_owner(s[:,:3],rig,frames[k0])
    refs=ReferenceTable(C,N,dev,torch.float64);refs.commit_accepted_references(owners,frames[k0],s,0);refs.initialize_from_source_catalogue(selected,k0,s)
    manifest={'pass':'F1' if direction=='forward' else 'B1','direction':direction,'canonical_frame':k0,'point_count':N,
        'processing_order':order,'timestamps_seconds':dataset.times.tolist(),'original_frame_indices':list(dataset.frame_ids),
        'debug_point_ids':ids,'debug_original_point_ids':sel[ids].tolist(),'debug_frames':list(debug_frames),'camera_count':C,
        'observation_model':'temporal-rgbd','observation_model_version':2,'temporal_mode':'full','lag':options['lag'],
        'precision':'float64','backend':'typed-SoA sparse CUDA cuDSS','source_initialization':'coupled 5D pixel/velocity chart + unique clean depth',
        'reference_uncertainty':'joint selected sparse marginal; shared sensor once; complete certified FEJ Schur',
        'execution_kind':'ACTUAL_TEMPORAL_IEKF_ESTIMATION','noise_assumption':options['warp_provenance'].get('assumption_class'),
        'diagnostic_focal_control':False,'reason_codes':TEMPORAL_REASONS,'status_codes':{0:'ANCHOR_CANONICAL',1:'DIRECT_TEMPORAL_ROWS',2:'NO_DIRECT_ROW_JOINT_MARGINAL'},
        'q_c_m2_s3':config.qc.tolist(),'noise_and_prior':config.manifest(),'calibration_hash':rig.calibration_hash(),
        'downstream':'TEMPORAL_FREE_WITH_LEGACY_RIGID','exactness':'exact chosen linearized operator, conditional on hard visibility and FEJ fixed-lag approximation',
        'NIS_status':'DEFERRED_OPTIONAL_DIAGNOSTIC; not used for acceptance; no independent-row surrogate',
        'information_diagnostic_semantics':'reciprocal eigenvalues of true joint position marginal; never used to construct posterior covariance'}
    for path,data in [(root/'run_manifest.json',manifest),(temporal/'config.json',options),(temporal/'calibration.json',options['warp_provenance'])]:json_write(path,serial(data))
    _store(mm,k0,s,s,P,None,C);mm['point_status'][k0]=0;mm['posterior_covariance'][k0]=P.cpu().numpy()
    if k0 in debug_frames:temporal_frame_debug(root,frames[k0],owners,s,s,rig,None,k0)
    anchor_records={}
    for i in ids:
        cc=[]
        for c in range(C):
            uv=owners.uv[c,i].round().long();u,v=uv.tolist()
            if 0<=u<frames[k0].W and 0<=v<frames[k0].H:
                winner=int(owners.owner_id[c,v,u]);cc.append({'camera':c,'pixel':[u,v],'winner_local_index':winner,
                    'winner_original_id':int(sel[winner]) if winner>=0 else None,'source_original_id':int(sel[i]),
                    'source':selected[i].tolist(),'point_depth_m':float(owners.z[c,i]),'winner_depth_m':float(owners.owner_depth[c,v,u])})
        anchor_records[int(sel[i])]=cc
    json_write(temporal/'anchor_source_ownership.json',serial(anchor_records))
    publication.publish_verified_frame(k0)
    history=[];traces={i:[] for i in ids};start=time.perf_counter()
    write_temporal_dashboard(root,manifest,history)
    try:
        for ordinal,k in enumerate(tqdm(order[1:],desc=f'Sparse FULL {direction}',unit='frame'),1):
            t0=time.perf_counter();times={};prev=order[ordinal-1];dt=float(dataset.times[k]-dataset.times[prev])
            mean,prior=predict(s,P,dt,torch.as_tensor(config.qc,device=dev,dtype=torch.float64))
            ts=time.perf_counter();frames[k]=MeasuredFrame(dataset,k,device,torch.float64);times['input_seconds']=time.perf_counter()-ts
            expired=order[:max(0,ordinal-options['lag']+1)]
            active_expired=[x for x in expired if x in frames]
            ts=time.perf_counter()
            graph.marginalize_lag_boundary_sparse_gpu(active_expired,options['workspace_bytes'])
            for old in active_expired:frames.pop(old)
            times['Schur_seconds']=time.perf_counter()-ts
            # Reference states are gathered by processing-time batch, never by point.
            for ref in frames:
                if ref==k:continue
                state=graph.state_and_blocks_gpu(ref)[0]
                cc,pp=torch.where(refs.frame==ref);refs.state[cc,pp]=state[pp]
            ts=time.perf_counter();before=first_hit_pixel_owner(mean[:,:3],rig,frames[k],dense=False)
            rows,reason,life=build_temporal_rows(mean,before,frames[k],refs,frames,rig,noise,policy,warp,ordinal,options['lag'])
            sig=signature(rows);times['ownership_sampling_seconds']=time.perf_counter()-ts
            ts=time.perf_counter()
            graph.pack_source_process_warp_factors_gpu(prev,k,transition(dt,device=device),process_noise(dt,torch.as_tensor(config.qc,device=dev,dtype=torch.float64)),mean,rows,frames,noise,warp)
            times['assembly_seconds']=time.perf_counter()-ts
            def same_current_domain():
                state=graph.state_and_blocks_gpu(k)[0]
                for ref in frames:
                    if ref==k:continue
                    old=graph.state_and_blocks_gpu(ref)[0]
                    cc,pp=torch.where(refs.frame==ref);refs.state[cc,pp]=old[pp]
                oo=first_hit_pixel_owner(state[:,:3],rig,frames[k],dense=False)
                rr,_,_=build_temporal_rows(state,oo,frames[k],refs,frames,rig,noise,policy,warp,ordinal,options['lag'])
                if torch.equal(signature(rr),sig):return True
                before_domain=torch.full((C,N,2),-1,device=dev,dtype=torch.long)
                after_domain=torch.full_like(before_domain,-1)
                before_domain[rows.camera,rows.point,0]=rows.pixel;before_domain[rows.camera,rows.point,1]=rows.reference_frame
                after_domain[rr.camera,rr.point,0]=rr.pixel;after_domain[rr.camera,rr.point,1]=rr.reference_frame
                graph.domain_bad_points=torch.where((before_domain!=after_domain).any(2).any(0))[0]
                return False
            ts=time.perf_counter()
            factor,solver=solve_nonlinear_sparse_iekf_gpu(graph,frames,noise,policy,max_iters,tol,same_current_domain)
            accepted_mask=solver.pop('_accepted_point_mask')
            times['nonlinear_solve_seconds']=time.perf_counter()-ts
            try:
                s=graph.state_and_blocks_gpu(k)[0]
                ts=time.perf_counter();P=factor.extract_joint_selected_marginals_gpu(graph.state_query_gpu(k));times['covariance_seconds']=time.perf_counter()-ts
                cov_profile=dict(factor.profile['selected_marginals'])
                eig=batched_eigen(P)
                if bool((eig<=0).any()):raise SparseNumericalFailure('FULL_POSTERIOR_NOT_SPD: current free point states require positive marginal covariance')
                owners=first_hit_pixel_owner(s[:,:3],rig,frames[k])
                for ref in frames:
                    if ref==k:continue
                    state=graph.state_and_blocks_gpu(ref)[0];cc,pp=torch.where(refs.frame==ref);refs.state[cc,pp]=state[pp]
                rows,reason,life=build_temporal_rows(s,owners,frames[k],refs,frames,rig,noise,policy,warp,ordinal,options['lag'])
                # Same row order as packed current warp. Prediction uses clean references.
                w=graph.warps[k]
                for ref in frames:
                    ix=torch.where(rows.reference_frame==ref)[0]
                    if not len(ix):continue
                    keys=graph.pixels.encode(rows.camera[ix,None],rows.reference_frame[ix,None],rows.reference_pixels[ix])
                    values=graph.pixels.clean[graph.pixels.lookup(keys)]
                    h,ht,hr,*_=temporal_prediction(rig,s[rows.point[ix]],refs.state[rows.camera[ix],rows.point[ix]],rows.camera[ix],
                        torch.stack((rows.pixel[ix]%frames[k].W,rows.pixel[ix]//frames[k].W),-1).double(),values,rows.cell[ix])
                    rows.h[ix]=h;rows.H_t[ix]=ht;rows.H_ref[ix]=hr
                nrows=torch.bincount(rows.point,minlength=N);observed=nrows>0;accepted=observed&accepted_mask
                ev=batched_eigen(P[:,:3,:3])
                rep={'nrows':nrows,'observed':observed,'reason':reason,'converged':torch.full((N,),solver['converged'],device=dev),
                    'point_iterations':torch.full((N,),len(solver['history']),device=dev),'nis':torch.full((N,),float('nan'),device=dev),
                    'nis_dof':torch.zeros(N,device=dev,dtype=torch.long),'info_eig':ev.flip(-1).reciprocal(),
                    'acceptance':accepted.to(torch.uint8),'posterior_cost':torch.full((N,),float('nan'),device=dev),'posterior_cost_dof':torch.zeros(N,device=dev,dtype=torch.long)}
                _store(mm,k,s,mean,P,rep,C)
                mm['posterior_covariance'][k]=P.cpu().numpy()
                residual=rows.y-rows.h
                stat=compare_owner_maps(before,owners)
                speeds=s[:,3:].norm(dim=1);posdiag=P[:,:3,:3].diagonal(dim1=1,dim2=2)
                summary={'frame':k,'ordinal':ordinal,'dt_seconds':dt,'rows':len(rows),'observed_points':int(observed.sum()),'assimilated_points':int(accepted.sum()),
                    'prediction_only':int((~observed).sum()),'eligible_but_no_accepted_update':int((observed&~accepted_mask).sum()),
                    'unexplained_foreground':int((frames[k].mask&frames[k].raw_valid&(owners.owner_id<0)).sum()),
                    'rendered_outside_foreground':int(outside_mask(owners,frames[k],N).sum()),
                    'rgb_residual_rms':float(residual[:,:3].square().mean().sqrt()) if len(rows) else None,
                    'depth_residual_rms_m':float(residual[:,3].square().mean().sqrt()) if len(rows) else None,
                    'covariance_min_eigenvalue_mixed_units':float(eig.min()),'covariance_max_eigenvalue_mixed_units':float(eig.max()),
                    'posterior_position_variance_median_m2':float(torch.median(posdiag)),'speed_median_m_s':float(torch.median(speeds)),
                    'speed_p95_m_s':float(torch.quantile(speeds,.95)),'posterior_pd_failures':0,'solver':solver,'reference':life,**stat}
                history.append(summary)
                per=[]
                for c in range(C):
                    per.append({'camera_index':c,'camera':dataset.cameras[c].name,'rows':int((rows.camera==c).sum()),
                        'rows_accepted_final':int((rows.camera==c).sum()),'information_trace_final':None,
                        'depth_innovation_m_quantiles_prior':None,'depth_innovation_m_quantiles_final':None,
                        'rgb_innovation_quantiles_prior':None,'rgb_innovation_quantiles_final':None,
                        'reason_counts':{TEMPORAL_REASONS[int(v)]:int(n) for v,n in zip(*torch.unique(reason[c],return_counts=True))}})
                for i in ids:
                    ii=torch.where(rows.point==i)[0]
                    rec={'frame':k,'source':selected[i].tolist(),'s_minus':mean[i].tolist(),'P_minus':prior[i].tolist(),
                        's_plus':s[i].tolist(),'P_plus':P[i].tolist(),'rows':[],'solver':solver,'measurement_update_accepted':bool(accepted[i]),
                        'camera_reason':reason[:,i].tolist(),'reference_frames':refs.frame[:,i].tolist(),'reference_processing_age':(ordinal-refs.ordinal[:,i]).tolist()}
                    cross={}
                    for a in ii.tolist():
                        c,rf=int(rows.camera[a]),int(rows.reference_frame[a]);reference_key=graph.pixels.encode(rows.camera[a:a+1],rows.reference_frame[a:a+1],rows.reference_pixels[a,:1])
                        first_for_reference=rf not in cross
                        if first_for_reference:cross[rf]=point_cross_query(graph,factor,i,k,rf,reference_key)
                        cr,pr,cp=cross[rf]
                        rec['rows'].append({'camera':c,'current_pixel':int(rows.pixel[a]),'reference_frame':rf,'reference_pixels':rows.reference_pixels[a].tolist(),
                            'q':rows.q[a].tolist(),'weights':rows.weights[a].tolist(),'H_t':rows.H_t[a].tolist(),'H_ref':rows.H_ref[a].tolist(),
                            'current_raw_rgbd':rows.y[a].tolist(),'reference_raw_rgbd':rows.values[a].tolist(),'prediction':rows.h[a].tolist(),'residual':residual[a].tolist(),
                            'reference_state':refs.state[c,i].tolist(),'R_current_sensor':sensor_covariance(rows.y[a],noise).tolist(),
                            'R_reference_sensor_interpolated':(rows.weights[a].square()[:,None,None]*sensor_covariance(rows.values[a],noise)).sum(0).tolist(),
                            'R_additional_warp_model':warp.tolist(),'Cov_current_reference':cr.tolist(),'P_reference':pr.tolist(),
                            'Cov_current_first_reference_pixel':cp.tolist() if first_for_reference else None,
                            'cross_pixel_scope':'first eligible camera per reference frame; other correlations retained but not exported',
                            'current_mask':True,'reference_support_mask':[True]*4,'covariance_semantics':'joint constrained marginal; raw sensors once'})
                    traces[i].append(rec);json_write(temporal/'traced_points'/f'{sel[i]}.json',serial({'point_id':int(sel[i]),'local_array_index':i,'frames':traces[i]}))
                if k in debug_frames:
                    ts=time.perf_counter();summary['real_oracle_exports']=save_real_sparse_oracle(temporal,k,factor,graph);times['oracle_export_seconds']=time.perf_counter()-ts
                sparse_profile={**factor.profile,'factor_counts':factor.problem.factor_counts,'raw_unique_sensor_keys':len(graph.pixels.keys),
                    'scalar_dimension_original':graph.scalar_dimension,'nnz_J':factor.problem.A._nnz(),'nnz_C':factor.problem.C._nnz(),
                    'current_marginal_profile':cov_profile,'FEJ':graph.marginalization_log}
                json_write(temporal/f'sparse_profile_{k:04d}.json',serial(sparse_profile))
            finally:factor.close()
            ts=time.perf_counter()
            if k in debug_frames:temporal_frame_debug(root,frames[k],owners,s,mean,rig,rows,k)
            times['viewer_seconds']=time.perf_counter()-ts
            refs.commit_accepted_references(owners,frames[k],s,ordinal)
            summary['seconds']=time.perf_counter()-t0;summary['stage_times']=times
            numeric_record={'frame':k,'processing_step':ordinal,'timestamp_s':float(dataset.times[k]),'dt_s':dt,
                'sensor':{'unique_points_observed':int(observed.sum()),'prediction_only_points':int((~observed).sum()),'rows_accepted_final':len(rows),'per_camera':per},
                'filter_update':{'not_converged_points':0 if solver['converged'] else N,'posterior_position_eig_m2_quantiles':np.quantile(ev.cpu(),[0,.01,.1,.5,.9,.95,.99,1]).tolist(),
                    'posterior_minus_prior_position_m_quantiles':np.quantile((s[:,:3]-mean[:,:3]).norm(dim=1).cpu(),[0,.01,.1,.5,.9,.95,.99,1]).tolist(),
                    'speed_m_s_quantiles':np.quantile(speeds.cpu(),[0,.01,.1,.5,.9,.95,.99,1]).tolist(),'posterior_positive_definite_failures':0},
                'innovation':{'nis_per_dof_quantiles_prior_observed':None},'processing':{'wall_ms':summary['seconds']*1000,'cuda_ms':{},'gpu_peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6,'iekf_iterations':len(solver['history'])},
                'acceptance':{'counts':{}},'temporal':summary}
            json_write(numeric/f'frame_{k:04d}.json',serial(numeric_record))
            append_csv(numeric/'timeline.csv',_timeline_row(numeric_record))
            json_write(temporal/f'frame_{k:04d}.json',serial(summary))
            if save_full_covariance and (k in debug_frames or (save_state_interval and ordinal%save_state_interval==0)):
                (root/'states').mkdir(exist_ok=True);np.savez_compressed(root/'states'/f'frame_{k:04d}.npz',s=s.cpu().numpy(),P=P.cpu().numpy())
            append_csv(temporal/'timing_memory.csv',{'frame':k,**times,'seconds':summary['seconds'],'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                'cuda_free_bytes':torch.cuda.mem_get_info()[0],'cpu_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            publication.publish_verified_frame(k,temporal/f'frame_{k:04d}.json')
            write_temporal_dashboard(root,manifest,history)
            print(f'[FULL sparse {direction}] frame={k} accepted={int(accepted.sum())}/{N} rows={len(rows)} seconds={summary["seconds"]:.3f} live={temporal/"index.html"}',flush=True)
        result={'status':'COMPLETE','frames':K,'point_count':N,'seconds':time.perf_counter()-start,'mode':'full','backend':'cuDSS sparse CUDA',
                'accepted_updates':sum(h['assimilated_points'] for h in history),'all_frames_valid':True,'scientific_acceptance':'NOT_AUTOMATICALLY_VALIDATED'}
        json_write(temporal/'marginalization.json',serial(graph.marginalization_log))
        json_write(root/'status.json',result);return result
    except BaseException as error:
        result={'status':getattr(error,'status','FAILED'),'reason':str(error),'completed_frames':publication.journal['completed_frames'],
                'point_count':N,'elapsed_filter_seconds':time.perf_counter()-start,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                'cpu_max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'backend':'cuDSS sparse CUDA',
                'sparse_failure_profile':getattr(error,'profile',None),'nonlinear_history':getattr(graph,'nonlinear_history',[])}
        json_write(root/'status.json',result);write_temporal_dashboard(root,manifest,history,str(error));raise
