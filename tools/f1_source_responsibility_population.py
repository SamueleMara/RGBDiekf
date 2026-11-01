"""Chunked frame-0/1 observational audit. No filtering and no GT in features."""
from pathlib import Path
import numpy as np
from tools.f1_definitive_evidence import csv_write,write_json


def population_features(sc,out,chunk=4096):
    import torch
    from estimator.projective_camera import CameraRig
    from estimator.projective_visibility import visibility,position_sigma_z2
    from estimator.projective_sensor import MeasuredFrame,SensorNoise
    from estimator.projective_terms import build_terms
    from tqdm.auto import tqdm
    if chunk<1:raise ValueError('Population compute chunk must be positive')
    N=len(sc.x0);C=len(sc.cams);rig=CameraRig(sc.run.data.cameras,'cpu',torch.float64)
    frames=[MeasuredFrame(sc.run.data,k,'cpu') for k in (0,1)]
    noise=SensorNoise(torch.as_tensor(sc.noise.rgb_var),sc.noise.a,sc.noise.b,sc.noise.rgb_model,sc.noise.depth_model,sc.noise.interp)
    Pvv=np.asarray(sc.run.fman['prior']['velocity_covariance_m2_s2']);qc=np.asarray(sc.run.fman['prior']['q_c_m2_s3']);dt=float(sc.t[1]-sc.t[0])
    X=torch.as_tensor(sc.x0,dtype=torch.float64);P0=torch.as_tensor(sc.P0,dtype=torch.float64);Pm=sc.P0+dt*dt*Pvv+dt**3/3*qc
    projection=rig.project(X,jacobian=False)
    reason0,front=visibility(projection,position_sigma_z2(rig.R,P0),sc.cams[0].H,sc.cams[0].W,rig.near,sc.policy['visibility_n_sigma'])
    reason1,_=visibility(projection,position_sigma_z2(rig.R,torch.as_tensor(Pm)),sc.cams[0].H,sc.cams[0].W,rig.near,sc.policy['visibility_n_sigma'])
    src=np.asarray(sc.src[:,1],int);owner_src=np.full(N,-1,np.int64)
    for c,cam in enumerate(sc.cams):
        winners=np.flatnonzero(front[c].numpy());uv=np.rint(projection.uv[c].numpy()).astype(np.int64);table=np.full(cam.W*cam.H,-1,np.int64)
        table[uv[winners,1]*cam.W+uv[winners,0]]=winners;ids=np.flatnonzero(src==c);inside=(uv[ids,0]>=0)&(uv[ids,0]<cam.W)&(uv[ids,1]>=0)&(uv[ids,1]<cam.H)
        ids=ids[inside];owner_src[ids]=table[uv[ids,1]*cam.W+uv[ids,0]]
    fields=[('point_id','i8'),('source_camera','i4'),('source_owner','i8'),('source_prior_reason','i2'),('source_admitted','?'),('source_sample_valid','?'),('source_gate_passed','?'),
            ('admitted_count','i2'),('cross_count','i2'),('cross_missing_reference','i2'),('cross_persistent_mismatch','?'),
            ('rgb_reference_chi2','f8'),('depth_reference_chi2','f8'),('rgb_temporal_delta_chi2','f8'),('depth_temporal_delta_chi2','f8'),('depth_gradient_m_per_m','f8'),('texture_gradient_per_m','f8'),
            ('initial_position_trace_m2','f8'),('first_dx_m','f8'),('first_dv_m_s','f8'),('first_accepted_frame','i4'),('first_nonzero_velocity_frame','i4'),
            ('full_hessian_next_dx_m','f8'),('non_source_removal_next_dx_effect_m','f8'),('non_source_removal_next_dv_effect_m_s','f8')]
    data=np.lib.format.open_memmap(Path(out)/f'{sc.name}_population_features.npy',mode='w+',dtype=np.dtype(fields),shape=(N,))
    status=sc.run.mm('point_status');reason=sc.run.mm('camera_reason');post=sc.run.mm('posterior_xyz');vel=sc.run.mm('velocity');measured=np.asarray(status[1])==1
    for start in tqdm(range(0,N,chunk),desc=f'{sc.name}: frame-0/1 audit',unit='chunk'):
        stop=min(start+chunk,N);ids=np.arange(start,stop);n=len(ids);source=src[ids];actual=(np.asarray(reason[1,:,start:stop])==0)&measured[ids]
        data['point_id'][ids]=ids;data['source_camera'][ids]=source;data['source_owner'][ids]=owner_src[ids]
        data['source_prior_reason'][ids]=reason1.numpy()[source,ids];data['source_admitted'][ids]=actual[source,np.arange(n)]
        data['admitted_count'][ids]=actual.sum(0);data['cross_count'][ids]=actual.sum(0)-actual[source,np.arange(n)]
        data['cross_missing_reference'][ids]=0;data['cross_persistent_mismatch'][ids]=False
        X0=X[start:stop];sigma=torch.as_tensor(Pm[start:stop]);a=torch.as_tensor(sc.run.can['canonical_rgb'][start:stop]);Xpost=torch.as_tensor(np.asarray(post[1,start:stop]),dtype=torch.float64)
        totals=np.zeros((n,3,3));gx=np.zeros((n,3));source_I=totals.copy();source_g=gx.copy();aggregates=np.zeros((n,6));valid_counts=np.zeros(n,int)
        for c in range(C):
            cameras=torch.full((n,),c,dtype=torch.long);points=torch.arange(n)
            r0=build_terms(rig,frames[0],cameras,points,X0,a,torch.as_tensor(sc.P0[start:stop]),noise,sc.policy)
            r1=build_terms(rig,frames[1],cameras,points,X0,a,sigma,noise,sc.policy)
            sensor0=np.isin(r0.reason.numpy(),[0,8,10]);sensor1=np.isin(r1.reason.numpy(),[0,8,10]);ref_ok=sensor0&(reason0[c,start:stop].numpy()!=6)
            cross=source!=c;admitted=actual[c];missing=admitted&cross&~ref_ok;data['cross_missing_reference'][ids]+=missing
            valid=ref_ok&sensor1&cross&admitted
            e0=r0.r.numpy();e1=r1.r.numpy();V=r0.var.numpy();delta=e1-e0
            aggregates[:,0]+=np.where(valid,(e0[:,:3]**2/V[:,:3]).sum(1),0);aggregates[:,1]+=np.where(valid,e0[:,3]**2/V[:,3],0)
            aggregates[:,2]+=np.where(valid,(delta[:,:3]**2/V[:,:3]).sum(1),0);aggregates[:,3]+=np.where(valid,delta[:,3]**2/V[:,3],0)
            valid_counts+=valid
            data['cross_persistent_mismatch'][ids]|=valid&np.all(delta==0,axis=1)&(r0.nis.numpy()>__import__('tools.root_cause_oracle',fromlist=['row_gate_threshold']).row_gate_threshold(sc.policy['visibility_n_sigma']))
            # Analytical gradients are for measured pixels, not inferred material normals.
            gradient_metric=torch.linalg.vector_norm(r0.Hx[:,3]+rig.R[c,2],dim=1).numpy()
            aggregates[:,4]+=np.where(valid,gradient_metric,0);aggregates[:,5]+=np.where(valid,torch.linalg.vector_norm(r0.Hx[:,:3],dim=(1,2)).numpy(),0)
            choose=source==c;data['source_sample_valid'][ids[choose]]=sensor1[choose];data['source_gate_passed'][ids[choose]]=r1.valid().numpy()[choose]
            term=build_terms(rig,frames[1],cameras,points,Xpost,a,sigma,noise,sc.policy,dx=Xpost-X0)
            # Only actual accepted rows enter this fixed-linearization score.
            ok=admitted&np.isin(term.reason.numpy(),[0,8,10]);H=term.Hx.numpy();w=1/term.var.numpy();I=np.einsum('nki,nk,nkj->nij',H,w,H);g=np.einsum('nki,nk,nk->ni',H,w,term.r.numpy())
            totals+=np.where(ok[:,None,None],I,0);gx+=np.where(ok[:,None],g,0);source_I+=np.where((ok&choose)[:,None,None],I,0);source_g+=np.where((ok&choose)[:,None],g,0)
        for j,key in enumerate(('rgb_reference_chi2','depth_reference_chi2','rgb_temporal_delta_chi2','depth_temporal_delta_chi2','depth_gradient_m_per_m','texture_gradient_per_m')):
            data[key][ids]=np.divide(aggregates[:,j],valid_counts,out=np.full(n,np.nan),where=valid_counts>0)
        P=np.zeros((n,6,6));P[:,:3,:3]=Pm[start:stop];P[:,:3,3:]=dt*Pvv+dt*dt/2*qc;P[:,3:,:3]=P[:,:3,3:].transpose(0,2,1);P[:,3:,3:]=Pvv+dt*qc
        Lam=np.linalg.inv(P);e=np.c_[np.asarray(post[1,start:stop])-sc.x0[start:stop],np.asarray(vel[1,start:stop])]
        A=Lam.copy();A[:,:3,:3]+=totals;b=-np.einsum('nij,nj->ni',Lam,e);b[:,:3]+=gx;all_delta=np.linalg.solve(A,b[...,None])[...,0]
        A=Lam.copy();A[:,:3,:3]+=source_I;b=-np.einsum('nij,nj->ni',Lam,e);b[:,:3]+=source_g;source_delta=np.linalg.solve(A,b[...,None])[...,0]
        data['initial_position_trace_m2'][ids]=np.trace(sc.P0[start:stop],axis1=1,axis2=2);data['first_dx_m'][ids]=np.linalg.norm(e[:,:3],axis=1);data['first_dv_m_s'][ids]=np.linalg.norm(e[:,3:],axis=1)
        data['full_hessian_next_dx_m'][ids]=np.linalg.norm(all_delta[:,:3],axis=1);data['non_source_removal_next_dx_effect_m'][ids]=np.linalg.norm(all_delta[:,:3]-source_delta[:,:3],axis=1);data['non_source_removal_next_dv_effect_m_s'][ids]=np.linalg.norm(all_delta[:,3:]-source_delta[:,3:],axis=1)
    first=np.full(N,-1);first_v=np.full(N,-1)
    for k in sc.order[1:]:
        accepted=np.asarray(status[k])==1;new=(first<0)&accepted;first[new]=k
        nonzero=np.any(np.asarray(vel[k])!=0,axis=1);first_v[(first_v<0)&nonzero]=k
    data['first_accepted_frame']=first;data['first_nonzero_velocity_frame']=first_v;data.flush()
    write_json(Path(out)/f'{sc.name}_population_feature_scope.json',{'frames':[0,1],'chunk_size':chunk,'GT_in_features':False,
        'force_scope':'One fixed accepted-state linearization, actual admitted rows; saved velocity is float32; not an exact traced accepted-step attribution.',
        'first_dx_dv_scope':'Frame-1 correction; first accepted time separately scanned from saved arrays. Later first-update raw rows are not manufactured.',
        'reference_variance_scope':'R0 normalization only, not estimated cross-time covariance','gradient_units':'World gradients: depth m/m and RGB/m, not pixel gradients',
        'initial_prior_scope':'Recorded measured-source Pxx, independent recorded Pvv and zero cross-blocks, exactly as standard frame-0 initialization; no filter run.'})
    return data


def score_population(sc,features,out):
    """Evaluation labels are attached only after all policy/image features exist."""
    with np.load(sc.fdir/'population_events.npz',allow_pickle=False) as z:
        if not np.array_equal(z['x0'],sc.x0):raise ValueError('Population GT catalogue differs')
        static=z['gt_static'].astype(bool)
    post=sc.run.mm('posterior_xyz');drift10=np.linalg.norm(np.asarray(post[min(10,len(sc.t)-1)])-sc.x0,axis=1);final=np.linalg.norm(np.asarray(post[sc.order[-1]])-sc.x0,axis=1)
    strata={};edges={}
    for name in ('initial_position_trace_m2','depth_gradient_m_per_m','texture_gradient_per_m'):
        values=np.asarray(features[name]);valid=np.isfinite(values);edge=np.quantile(values[valid],[1/3,2/3]) if valid.any() else np.array([])
        band=np.full(len(values),-1);band[valid]=np.searchsorted(edge,values[valid]);strata[name]=band;edges[name]=edge
    write_json(Path(out)/f'{sc.name}_population_strata.json',{'reporting_quantile_edges':edges,'scope':'Computed from all feature rows, independent of GT labels and ablation responses; not filter thresholds.'})
    keys={};rows=[]
    for pid in np.flatnonzero(static):
        r=features[pid]
        category=('source_admitted' if r['source_admitted'] else ('source_suppressed' if r['source_prior_reason'] in (6,7) else 'source_not_admitted_other_reason'))+('_cross_admitted' if r['cross_count'] else '_no_cross')
        key=(category,int(r['source_camera']),int(r['admitted_count']),bool(r['cross_missing_reference']),bool(r['cross_persistent_mismatch']),
             int(strata['initial_position_trace_m2'][pid]),int(strata['depth_gradient_m_per_m'][pid]),int(strata['texture_gradient_per_m'][pid]),bool(r['source_sample_valid']))
        keys.setdefault(key,[]).append(int(pid))
    for key,ids in sorted(keys.items()):
        ids=np.asarray(ids);rows.append({'scene':sc.name,'category':key[0],'source_camera':key[1],'accepted_camera_count':key[2],'has_missing_reference':key[3],'has_frame0_inconsistent_cross_row':key[4],
            'initial_uncertainty_quantile_stratum':key[5],'depth_gradient_quantile_stratum':key[6],'texture_gradient_quantile_stratum':key[7],'source_sample_valid':key[8],
            'N':len(ids),'GT_static_population_denominator':int(static.sum()),'median_frame1_dx_m':float(np.median(features['first_dx_m'][ids])),'median_frame1_dv_m_s':float(np.median(features['first_dv_m_s'][ids])),
            'fraction_drift_gt_1cm_frame10':float(np.mean(drift10[ids]>.01)),'fraction_drift_gt_10cm_final':float(np.mean(final[ids]>.1)),
            'median_initial_position_trace_m2':float(np.median(features['initial_position_trace_m2'][ids])),'median_depth_world_gradient':float(np.nanmedian(features['depth_gradient_m_per_m'][ids])) if np.isfinite(features['depth_gradient_m_per_m'][ids]).any() else None,
            'causal_interpretation':'OBSERVATIONAL_CLUSTERED_NOT_CAUSAL_RISK','GT_identity':'coarse nearest labelled cloud, not material identity'})
    return rows
