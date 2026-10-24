# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""Measurement-supported graph proposals and held-out one-moving-part evidence.

Graph strain only proposes candidates. Physical selection compares raw UVZ
on exactly the same global held-out identities, with explicit complexity cost.
No group-generated within-patch distances certify membership.
"""
import numpy as np
from .group_partition_validation import fit_measured_rigid
from .group_temporal_validation import fit_temporal_raw,score_temporal_raw
from .group_evidence_audit import position_jacobian,projection_numpy


def measured_centroids(partition,rows):
    K=len(rows[0]['_sequence_times']);S=partition.count
    xyz=np.full((K,S,3),np.nan);cov=np.full((K,S,3,3),np.nan);support=np.zeros((K,S),bool)
    for row in rows:
        k=row['frame'];train=partition.duplicate_unit[row['point_id']]%2==1
        for g in np.unique(partition.membership[row['point_id'][train]]):
            use=train&(partition.membership[row['point_id']]==g);ids=row['point_id'][use]
            fit=fit_measured_rigid(partition.xyz[ids],row['xyz'][use],row['covariance'][use])
            if fit['status']!='FIT':continue
            T=fit['pose'];center=T[:3,:3]@partition.centroids[g]+T[:3,3];G=position_jacobian(center[None])[0]
            xyz[k,g]=center;cov[k,g]=G@fit['covariance']@G.T;support[k,g]=True
    return {'centroid':xyz,'covariance':cov,'observed':support,
        'semantics':'training-only measured-world-XYZ rigid MLE; first-order propagated sensor noise; graph proposal only, final validation is raw UVZ'}


def edge_strain(partition,measured):
    edge=partition.adjacency;X=measured['centroid'];P=measured['covariance'];valid=measured['observed'][:,edge[:,0]]&measured['observed'][:,edge[:,1]]
    v=X[:,edge[:,0]]-X[:,edge[:,1]];length=np.linalg.norm(v,axis=2);cost=np.full(len(edge),np.inf);variance=np.full_like(length,np.nan);log_strain=np.full_like(length,np.nan)
    for e,(a,b) in enumerate(edge):
        frames=np.flatnonzero(valid[:,e]&(length[:,e]>0))
        if len(frames)<2:continue
        values=length[frames,e];order=np.argsort(values,kind='stable');reference=int(frames[order[len(frames)//2]])
        direction=v[frames,e]/values[:,None];vv=np.einsum('ni,nij,nj->n',direction,P[frames,a]+P[frames,b],direction)
        z=np.log(values/length[reference,e]);variance[frames,e]=vv/values**2+vv[np.flatnonzero(frames==reference)[0]]/length[reference,e]**2
        variance[reference,e]=0;log_strain[frames,e]=z;cost[e]=np.sqrt(np.mean(z*z))
    return {'cost':cost,'length_m':length,'log_strain':log_strain,'conditional_log_strain_variance':variance,'observed':valid,
        'covariance_semantics':'conditional independent sensor fits at distinct times; canonical/calibration and median-frame selection conditioned; correlated edges are NOT multiplied as independent likelihoods'}


def candidate_hierarchy(edges,cost,groups):
    parent=list(range(groups));members={i:{i} for i in range(groups)};candidates=[frozenset([i]) for i in range(groups)];pending={}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for index in np.argsort(cost,kind='stable'):
        if not np.isfinite(cost[index]):continue
        a,b=map(find,map(int,edges[index]))
        if a==b:continue
        key=tuple(sorted((a,b)));pending[key]=pending.get(key,0)+1
        if pending[key]<min(2,len(members[a]),len(members[b])):continue
        if len(members[a])<len(members[b]):a,b=b,a
        parent[b]=a;members[a]|=members.pop(b);candidates.append(frozenset(members[a]));remapped={}
        for (u,v),count in pending.items():
            u=a if u==b else u;v=a if v==b else v
            if u!=v:tag=tuple(sorted((u,v)));remapped[tag]=remapped.get(tag,0)+count
        pending=remapped
    return sorted(set(candidates),key=lambda x:(len(x),tuple(sorted(x))))


def static_score(partition,rows,test_masks,point_mask=None):
    total=0.;count=0
    for row,test in zip(rows,test_masks):
        if point_mask is not None:test=test&point_mask[row['point_id']]
        for ci,camera in enumerate(row['_cameras']):
            use=test&(row['camera']==ci);ids=row['point_id'][use]
            if not len(ids):continue
            h,_=projection_numpy(partition.xyz[ids],camera);e=row['z'][use]-h;R=row['R'][use]
            total+=.5*(np.einsum('ni,ni->',e,np.linalg.solve(R,e[:,:,None])[:,:,0])+np.linalg.slogdet(R)[1].sum()+3*len(ids)*np.log(2*np.pi));count+=len(ids)
    return {'nll':float(total),'raw_count':count}


def training_motion_proposal(partition,rows,point_mask):
    """Training-only shortlist; not a likelihood/physical acceptance substitute."""
    gain=0.;count=0;times=0
    for row in rows:
        use=(partition.duplicate_unit[row['point_id']]%2==1)&point_mask[row['point_id']];ids=row['point_id'][use]
        if not len(ids):continue
        fit=fit_measured_rigid(partition.xyz[ids],row['xyz'][use],row['covariance'][use])
        if fit['status']!='FIT':continue
        e=row['xyz'][use]-partition.xyz[ids];chi=float(np.einsum('ni,ni->',e,np.linalg.solve(row['covariance'][use],e[:,:,None])[:,:,0]))
        gain+=chi-fit['chi2'];count+=3*len(ids);times+=1
    return {'gain':float(gain-6*times*np.log(max(count,1))),'training_scalars':count,'observable_frames':times,
        'role':'proposal heuristic using first-order world covariance; may miss modes. Raw UVZ validation determines acceptance, not this score'}


def select_moving_component(partition,rows,calibration_conditioned=True):
    if not len(rows):return {'status':'NO_COHERENT_MOVING_COMPONENT','reason':'No independent raw factors','labels':['INCOHERENT_OR_AMBIGUOUS']*partition.count}
    measured=measured_centroids(partition,rows);strain=edge_strain(partition,measured)
    candidates=candidate_hierarchy(partition.adjacency,strain['cost'],partition.count)
    test=[partition.duplicate_unit[row['point_id']]%2==0 for row in rows];train=[~x for x in test]
    static=static_score(partition,rows,test);N=static['raw_count']
    if not N:return {'status':'NO_COHERENT_MOVING_COMPONENT','reason':'No untouched raw evaluation factors','labels':['INCOHERENT_OR_AMBIGUOUS']*partition.count}
    whole=fit_temporal_raw(partition,rows,np.zeros(len(partition.xyz),np.int64),1,train)
    global_score=score_temporal_raw(partition,rows,whole,test);freedom=6*(len(whole['order'])-1);penalty=freedom*np.log(3*N)
    accepted=[];reports=[]
    from tqdm.auto import tqdm
    for groups in tqdm(candidates,desc='Measured coherence candidates',unit='component'):
        if len(groups)==partition.count:continue
        mask=np.isin(partition.membership,list(groups));proposal=training_motion_proposal(partition,rows,mask)
        if proposal['gain']<=0:continue
        labels=mask.astype(np.int64);two=fit_temporal_raw(partition,rows,labels,2,train)
        if min(m['initial_twist_data_rank'] for m in two['models'])<6:
            reports.append({'groups':sorted(groups),'status':'REFERENCE_OR_MOTION_UNOBSERVABLE','proposal':proposal});continue
        score=score_temporal_raw(partition,rows,two,test)
        # Score the candidate while setting the complement's held-out mask to
        # false; the candidate's training and covariance model stay unchanged.
        candidate_tests=[t&mask[r['point_id']] for r,t in zip(rows,test)]
        complement_tests=[t&~mask[r['point_id']] for r,t in zip(rows,test)]
        moving=score_temporal_raw(partition,rows,two,candidate_tests);comp_free=score_temporal_raw(partition,rows,two,complement_tests)
        comp_static=static_score(partition,rows,test,~mask);composite=moving['nll']+comp_static['nll']
        model_gain=2*(min(static['nll'],global_score['nll']+penalty/2)-composite)-penalty
        relative_gain=2*(global_score['nll']-score['nll'])-penalty
        reference_gain=2*(comp_static['nll']-comp_free['nll'])-penalty
        ok=calibration_conditioned and model_gain>0 and relative_gain>0 and reference_gain<=0
        item={'groups':sorted(groups),'status':'SUPPORTED_MOVING_CORE' if ok else 'REFERENCE_UNRESOLVED' if reference_gain>0 or not calibration_conditioned else 'NOT_SUPPORTED',
            'global_same_raw_count':N,'model_gain':float(model_gain),'relative_two_vs_one_gain':float(relative_gain),'complement_free_vs_static_gain':float(reference_gain),
            'composite_nll':float(composite),'proposal':proposal,'same_raw_ids':True,'heldout_raw_overlap_with_training':0}
        reports.append(item)
        if ok:accepted.append((composite,groups,two))
    labels=['INCOHERENT_OR_AMBIGUOUS']*partition.count
    if not accepted:
        if calibration_conditioned and 2*(static['nll']-global_score['nll'])<=penalty:
            for g in range(partition.count):
                if measured['observed'][:,g].sum()>=2:labels[g]='STATIC_COMPATIBLE'
        return {'status':'NO_COHERENT_MOVING_COMPONENT','labels':labels,'candidates':reports,'selected_groups':[],
            'reason':'No measured candidate beats both static and global-drift alternatives with a static-compatible complement on same-set held-out UVZ','strain':strain,'measured':measured}
    _,selected,model=min(accepted,key=lambda x:(x[0],tuple(sorted(x[1]))))
    membership_scores=[]
    for g in range(partition.count):
        mask=partition.membership==g;local_tests=[t&mask[r['point_id']] for r,t in zip(rows,test)]
        static_local=static_score(partition,rows,local_tests)
        if not static_local['raw_count']:continue
        assignment=model['membership'].copy();assignment[mask]=1
        extrapolated={**model,'membership':assignment}
        moving_local=score_temporal_raw(partition,rows,extrapolated,local_tests)
        local_penalty=freedom*np.log(3*static_local['raw_count'])
        evidence=2*(static_local['nll']-moving_local['nll'])
        if evidence>local_penalty:labels[g]='MOVING_RIGID'
        elif evidence<-local_penalty and measured['observed'][:,g].sum()>=2:labels[g]='STATIC_COMPATIBLE'
        membership_scores.append({'group':g,'twice_predictive_gain':float(evidence),'conservative_local_complexity_cost':float(local_penalty)})
    confident=[g for g in selected if labels[g]=='MOVING_RIGID']
    if not confident:return {'status':'NO_COHERENT_MOVING_COMPONENT','labels':labels,'selected_groups':[],'candidates':reports,'reason':'No individually supported core after conservative per-group evidence checks','strain':strain,'measured':measured}
    return {'status':'SUPPORTED_MOVING_CORE','labels':labels,'selected_groups':sorted(confident),'point_support':np.isin(partition.membership,confident),'membership_evidence':membership_scores,
        'candidates':reports,'strain':strain,'measured':measured,'selected_free_model':model,
        'assumptions':'Canonical/calibration conditioned, archived independent diagnostic associations conditioned; finite training-proposed hierarchy, local Gaussian trajectory approximation. BIC is explicit model-complexity control, not calibrated posterior probability.'}
