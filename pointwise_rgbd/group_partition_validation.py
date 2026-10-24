"""Measurement-only boundary diagnostics and immutable partition refits.

Diagnostic poses are independently fitted at each measured time from training
raw samples, not from constrained Phase-1 XYZ. Thus the conditional sensor-noise
model does not invent independent member pose priors or temporal predictions.
Held-out predictive covariance retains each pose's shared rank-six effect.
"""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import heapq,json,sqlite3
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from rgbd_common.lie import exp,log,inverse,left_jacobian,skew
from rgbd_common.array_store import ArrayStore
from .projection import backproject
from .store import json_write


def weighted_kabsch(p,y,weight=None):
    weight=np.ones(len(p)) if weight is None else np.asarray(weight);weight=weight/weight.sum()
    pc=(weight[:,None]*p).sum(0);yc=(weight[:,None]*y).sum(0)
    u,_,vt=np.linalg.svd(((p-pc)*weight[:,None]).T@(y-yc));d=np.eye(3);d[-1,-1]=np.linalg.det(vt.T@u.T)
    pose=np.eye(4);pose[:3,:3]=vt.T@d@u.T;pose[:3,3]=yc-pose[:3,:3]@pc;return pose


def fit_measured_rigid(p,y,covariance):
    if len(p)<3:return {'status':'RIGID_UNIDENTIFIABLE','rank':min(3*len(p),5)}
    initial=weighted_kabsch(p,y,1/np.trace(covariance,axis1=1,axis2=2));root=np.linalg.cholesky(covariance)
    def values(d,jac=False):
        pose=exp(d)@initial;xyz=p@pose[:3,:3].T+pose[:3,3]
        if not jac:return np.linalg.solve(root,(xyz-y)[...,None])[...,0].ravel()
        g=np.stack([np.c_[np.eye(3),-skew(x)] for x in xyz]);j=g@left_jacobian(d)
        return np.linalg.solve(root,j).reshape(-1,6)
    result=least_squares(values,np.zeros(6),jac=lambda d:values(d,True),ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100)
    pose=exp(result.x)@initial;xyz=p@pose[:3,:3].T+pose[:3,3];g=np.stack([np.c_[np.eye(3),-skew(x)] for x in xyz]);j=np.linalg.solve(root,g).reshape(-1,6)
    singular=np.linalg.svd(j,compute_uv=False);tol=singular[0]*max(j.shape)*np.finfo(float).eps;rank=int((singular>tol).sum())
    if rank<6:return {'status':'RIGID_UNIDENTIFIABLE','rank':rank,'singular_values':singular.tolist()}
    return {'status':'FIT','rank':rank,'pose':pose,'covariance':np.linalg.inv(j.T@j),'chi2':float(result.fun@result.fun),'scalar_count':len(result.fun),'singular_values':singular.tolist(),'solver_success':bool(result.success)}


def predictive_score(p,y,covariance,fit):
    """Joint held-out prediction via Woodbury, without a (3N)^2 matrix."""
    pose=fit['pose'];xyz=p@pose[:3,:3].T+pose[:3,3];e=y-xyz;rinv=np.linalg.inv(covariance)
    g=np.stack([np.c_[np.eye(3),-skew(x)] for x in xyz]);precision=np.linalg.inv(fit['covariance'])
    htr=np.einsum('nji,njk->nik',g,rinv);a=precision+np.einsum('nij,njk->ik',htr,g);b=np.einsum('nij,nj->i',htr,e)
    q=float(np.einsum('ni,nij,nj->',e,rinv,e)-b@np.linalg.solve(a,b))
    logdet=float(np.linalg.slogdet(covariance)[1].sum()+np.linalg.slogdet(fit['covariance'])[1]+np.linalg.slogdet(a)[1])
    return {'nll':.5*(q+logdet+3*len(p)*np.log(2*np.pi)),'chi2':q,'scalar_count':3*len(p),'position_error_squared_m2':float((e*e).sum())}


def build_evidence_index(root,partition,frames):
    path=Path(root)/'evidence_by_group.sqlite';db=sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS factors (g INT,k INT,data BLOB,PRIMARY KEY(g,k))')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
    row=db.execute("SELECT value FROM meta WHERE key='complete_hash'").fetchone()
    if row:
        if row[0]!=partition.digest():raise ValueError('Evidence index belongs to a different partition')
        return db
    source=ArrayStore(Path(root)/'actual_measurement_factors.sqlite')
    for k in range(frames):
        if not source.contains(str(k)):continue
        values=source.read(str(k));groups=partition.membership[values['point_id']];order=np.argsort(groups,kind='stable');keys,begin=np.unique(groups[order],return_index=True);end=np.r_[begin[1:],len(order)];rows=[]
        for g,lo,hi in zip(keys,begin,end):
            ids=order[lo:hi];stream=BytesIO();np.savez_compressed(stream,**{name:values[name][ids] for name in ('point_id','camera','pixel','z','R','rgb')});rows.append((int(g),k,stream.getvalue()))
        with db:db.executemany('INSERT OR REPLACE INTO factors VALUES (?,?,?)',rows)
    with db:db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('complete_hash',partition.digest()))
    return db


def group_evidence(db,g,dataset):
    result=[]
    path=Path(db.execute('PRAGMA database_list').fetchone()[2]).parent/'run.json'
    run=json.loads(path.read_text())
    origin=db.execute("SELECT value FROM meta WHERE key='origin'").fetchone()
    for k,blob in db.execute('SELECT k,data FROM factors WHERE g=? ORDER BY k',(int(g),)):
        with np.load(BytesIO(blob)) as values:row={key:values[key] for key in values.files}
        xyz=np.empty((len(row['z']),3));cov=np.empty((len(row['z']),3,3))
        for ci in np.unique(row['camera']):
            use=row['camera']==ci;xyz[use],cov[use]=backproject(row['z'][use],dataset.cameras[ci],row['R'][use])
        row.update(frame=int(k),xyz=xyz,covariance=cov,_sequence_times=np.asarray(dataset.times),_anchor_index=run['anchor_frame_index'],_cameras=dataset.cameras,_dataset=dataset,_origin=origin[0] if origin else 'archived grouped associations');result.append(row)
    return result


def instantaneous_heldout_comparison(partition,rows,assignment,fold=0,mode='member',selection_frames=None):
    """Same raw evaluation identities for one/two hypotheses, linked folds."""
    totals={'one_nll':0.,'two_nll':0.,'one_position_error_m2':0.,'two_position_error_m2':0.,'heldout_raw_count':0,'train_raw_count':0,'frames':0,'one_parameters':0,'two_parameters':0};identities=[];train_ids=[];ranks=[]
    for row in rows:
        k=row['frame']
        if selection_frames is not None and k not in selection_frames:continue
        point=row['point_id'];units=partition.duplicate_unit[point]
        if mode=='member':test=units%2==fold
        elif mode=='camera':test=row['camera']==fold
        else:raise ValueError('Unknown independent evaluation fold')
        train=~test
        if not test.any():continue
        one=fit_measured_rigid(partition.xyz[point[train]],row['xyz'][train],row['covariance'][train])
        if one['status']!='FIT':return {'status':'UNRESOLVED_SPLIT','reason':'training single-group pose rank deficient','rank':one['rank']}
        one_score=predictive_score(partition.xyz[point[test]],row['xyz'][test],row['covariance'][test],one)
        two_nll=0.;two_error=0.
        for label in (0,1):
            a=train&(assignment[point]==label);b=test&(assignment[point]==label)
            fit=fit_measured_rigid(partition.xyz[point[a]],row['xyz'][a],row['covariance'][a]);ranks.append(fit['rank'])
            if fit['status']!='FIT':return {'status':'UNRESOLVED_SPLIT','reason':'split training pose rank deficient','rank':fit['rank']}
            if b.any():
                score=predictive_score(partition.xyz[point[b]],row['xyz'][b],row['covariance'][b],fit);two_nll+=score['nll'];two_error+=score['position_error_squared_m2']
        totals['one_nll']+=one_score['nll'];totals['two_nll']+=two_nll;totals['one_position_error_m2']+=one_score['position_error_squared_m2'];totals['two_position_error_m2']+=two_error
        totals['heldout_raw_count']+=int(test.sum());totals['train_raw_count']+=int(train.sum());totals['frames']+=1;totals['one_parameters']+=6;totals['two_parameters']+=12
        identities.extend((k,int(c),int(p)) for c,p in zip(row['camera'][test],row['pixel'][test]));train_ids.extend((k,int(c),int(p)) for c,p in zip(row['camera'][train],row['pixel'][train]))
    if not totals['heldout_raw_count']:return {'status':'UNRESOLVED_SPLIT','reason':'No independent held-out observations'}
    if set(identities)&set(train_ids):raise AssertionError('Held-out raw measurement leakage')
    penalty=(totals['two_parameters']-totals['one_parameters'])*np.log(3*totals['heldout_raw_count'])
    gain=2*(totals['one_nll']-totals['two_nll'])-penalty
    return {'status':'SPLIT_SUPPORTED' if gain>0 else 'SINGLE_PREFERRED',**totals,'bic_predictive_gain':float(gain),'extra_freedom_penalty':float(penalty),'min_training_rank':min(ranks),'identical_evaluation_refs':identities,'training_refs':train_ids,'model':'per-time free rigid MLE from training raw world XYZ; full anisotropic propagated sensor covariance; shared parameter covariance retained in held-out joint score; conditioned calibration/canonical geometry; association provenance conditioned, not globally marginalized'}


def heldout_comparison(partition,rows,assignment,fold=0,mode='member',selection_frames=None):
    from .group_temporal_validation import fit_temporal_raw,score_temporal_raw
    if not rows:return {'status':'UNRESOLVED_SPLIT','reason':'No raw observations'}
    masks=[];refs=[]
    for row in rows:
        if mode=='member':test=partition.duplicate_unit[row['point_id']]%2==fold
        elif mode=='camera':test=row['camera']==fold
        else:raise ValueError('Unknown evaluation fold')
        masks.append(test);refs.extend((row['frame'],int(c),int(p)) for c,p in zip(row['camera'][~test],row['pixel'][~test]))
    training=[~x for x in masks]
    one=fit_temporal_raw(partition,rows,np.zeros(len(assignment),np.int64),1,training)
    two=fit_temporal_raw(partition,rows,assignment,2,training)
    ranks=[x['initial_twist_data_rank'] for x in [*one['models'],*two['models']]]
    if min(ranks)<6:return {'status':'UNRESOLVED_SPLIT','reason':'Whole-window training motion geometry is unobservable','temporal_data_ranks':ranks}
    a=score_temporal_raw(partition,rows,one,masks,selection_frames);b=score_temporal_raw(partition,rows,two,masks,selection_frames)
    if set(a['evaluation_refs'])!=set(b['evaluation_refs']):raise AssertionError('Compared models do not use identical held-out raw factors')
    if set(refs)&set(a['evaluation_refs']):raise AssertionError('Training/evaluation raw factor leakage')
    if not a['raw_count']:return {'status':'UNRESOLVED_SPLIT','reason':'No held-out raw measurements'}
    count_times=len(one['order'])-1;extra=6*count_times;penalty=extra*np.log(3*a['raw_count'])
    gain=2*(a['nll']-b['nll'])-penalty
    return {'status':'SPLIT_SUPPORTED' if gain>0 else 'SINGLE_PREFERRED','one_nll':a['nll'],'two_nll':b['nll'],
        'heldout_raw_count':a['raw_count'],'train_raw_count':len(refs),'frames':count_times,'one_parameters':6*count_times,'two_parameters':12*count_times,
        'bic_predictive_gain':float(gain),'extra_freedom_penalty':float(penalty),'temporal_data_ranks':ranks,
        'identical_evaluation_refs':a['evaluation_refs'],'training_refs':refs,
        'one_joint_predictive_chi2':a['chi2'],'two_joint_predictive_chi2':b['chi2'],
        'model':'training-only raw UVZ IEKF, block-tridiagonal Gaussian smoothing and same-set joint temporal prediction; no instantaneous rank veto. Additional 6 pose degrees per time BIC cost is explicit conservative complexity assumption. Canonical geometry/calibration and archived correspondence selection conditioned.'}


def propose_split(partition,g,rows,proposal_frames,diagnostics=None):
    members=np.flatnonzero(partition.membership==g);train_members=members[partition.duplicate_unit[members]%2==1]
    if len(train_members)<6:return None,'Insufficient independent training geometry'
    feature=np.zeros((len(partition.xyz),3*len(proposal_frames)));count=np.zeros((len(partition.xyz),len(proposal_frames)))
    for j,k in enumerate(proposal_frames):
        selected=[r for r in rows if r['frame']==k]
        if not selected:continue
        row=selected[0];point=row['point_id'];use=partition.duplicate_unit[point]%2==1
        fit=fit_measured_rigid(partition.xyz[point[use]],row['xyz'][use],row['covariance'][use])
        if fit['status']!='FIT':continue
        xyz=partition.xyz[point[use]]@fit['pose'][:3,:3].T+fit['pose'][:3,3];error=row['xyz'][use]-xyz
        whitened=np.linalg.solve(np.linalg.cholesky(row['covariance'][use]),error[:,:,None])[...,0]
        np.add.at(feature[:,3*j:3*j+3],point[use],whitened);np.add.at(count[:,j],point[use],1)
    units=np.unique(partition.duplicate_unit[train_members]);features=[]
    for unit in units:
        ids=train_members[partition.duplicate_unit[train_members]==unit];denominator=np.repeat(count[ids],3,axis=1);f=np.divide(feature[ids],denominator,out=np.zeros_like(feature[ids]),where=denominator>0);features.append(f.mean(0))
    features=np.asarray(features)
    if len(features)<6:return None,'Too few linked provenance units'
    # Features are defined only by independent training sensor values. Propagate
    # them over the canonical graph; held-out values never enter this proposer.
    owner=np.full(len(partition.xyz),-1,np.int64);distance=np.full(len(partition.xyz),np.inf);queue=[]
    for unit_index,unit in enumerate(units):
        for point in members[partition.duplicate_unit[members]==unit]:heapq.heappush(queue,(0.,int(unit_index),int(point)))
    edges=partition.edges[np.isin(partition.edges[:,0],members)&np.isin(partition.edges[:,1],members)];weight=np.linalg.norm(partition.xyz[edges[:,0]]-partition.xyz[edges[:,1]],axis=1)
    graph=coo_matrix((np.r_[weight,weight],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(partition.xyz),len(partition.xyz))).tocsr()
    while queue:
        d,label,u=heapq.heappop(queue)
        if d>distance[u] or (d==distance[u] and owner[u]>=0 and owner[u]<=label):continue
        distance[u]=d;owner[u]=label
        for j in range(graph.indptr[u],graph.indptr[u+1]):
            v=graph.indices[j];candidate=d+graph.data[j]
            if candidate<distance[v]:heapq.heappush(queue,(candidate,label,int(v)))
    if (owner[members]<0).any():return None,'GROUP_UNRESOLVED: disconnected or unmeasured source component'
    # Kruskal on training feature contrasts. Removing any edge of this spanning
    # tree produces two connected patches *before* assigning physical labels.
    parent={int(i):int(i) for i in members};tree=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    contrast=((features[owner[edges[:,0]]]-features[owner[edges[:,1]]])**2).sum(1)
    for j in np.lexsort((edges[:,1],edges[:,0],contrast)):
        a,b=map(int,edges[j]);u,v=find(a),find(b)
        if u!=v:parent[v]=u;tree.append((a,b))
    if len(tree)!=len(members)-1:return None,'GROUP_UNRESOLVED: no connected canonical spanning tree'
    adjacency={int(i):[] for i in members}
    for a,b in tree:adjacency[a].append(b);adjacency[b].append(a)
    candidates=[];best=None
    for cut_a,cut_b in tree:
        side=set([cut_a]);pending=[cut_a]
        while pending:
            a=pending.pop()
            for b in adjacency[a]:
                if (a==cut_a and b==cut_b) or (a==cut_b and b==cut_a) or b in side:continue
                side.add(b);pending.append(b)
        labels=np.array([int(any(int(i) in side for i in train_members[partition.duplicate_unit[train_members]==unit])) for unit in units])
        if min(np.bincount(labels,minlength=2))<3:continue
        score=sum(float(((features[labels==label]-features[labels==label].mean(0))**2).sum()) for label in (0,1))
        candidates.append({'cut_edge':[cut_a,cut_b],'training_feature_sse':score,'sizes':[len(side),len(members)-len(side)]})
        if best is None or score<best[0]:best=(score,side,labels.copy())
    if best is None:return None,'No connected cut has two independent training supports'
    assignment=np.full(len(partition.xyz),-1,np.int64);assignment[members]=0;assignment[list(best[1])]=1
    labels=best[2];components=[1,1]
    if diagnostics is not None:
        diagnostics.update(proposed_membership=assignment.copy(),subpatch_connected_components=components,subpatch_sizes=[int((assignment[members]==label).sum()) for label in (0,1)],training_unit_ids=units.copy(),training_unit_feature_labels=labels.copy(),training_features=features.copy(),proposal_frames=proposal_frames,connected_training_candidates=candidates,selection='minimum training residual-feature SSE among graph-connected tree cuts; independent validation still required')
    if any(component!=1 for component in components) or min(int((assignment[members]==label).sum()) for label in (0,1))<3:
        return None,'Proposed subpatch is disconnected or insufficient'
    return assignment,'Training-only surface-tree cut; both sides connected by construction'


def validate_group(partition,g,rows,camera_count):
    from .group_temporal_validation import fit_temporal_raw,score_temporal_raw
    origin='archived grouped associations, conditional selection'
    if rows and '_dataset' in rows[0] and not rows[0].get('_origin','').startswith('independent'):
        from .group_sensor_tracks import independent_group_sensor_rows
        from .config import load_config
        from rgbd_common.backends.torch_cuda import Backend
        source=rows[0]
        rows=independent_group_sensor_rows(partition,g,source['_dataset'],load_config(),Backend('cpu'),source['_anchor_index'])
        origin='independent local XYZ diagnostic raw UVZ identities; grouped generated positions not used'
    elif rows:origin=rows[0].get('_origin',origin)
    frames=sorted({r['frame'] for r in rows})
    if len(frames)<2:return {'status':'UNRESOLVED_SPLIT','reason':'Insufficient observed temporal frames'}
    middle=max(1,len(frames)//2);proposal=frames[:middle];assignment,reason=propose_split(partition,g,rows,proposal)
    if assignment is None:return {'status':'UNRESOLVED_SPLIT','reason':reason}
    # Training-only candidates may leave an unsampled boundary pixel on either
    # side. Select among connected extensions on EARLY validation factors;
    # future held-out factors remain untouched for the acceptance decision.
    training=[partition.duplicate_unit[r['point_id']]%2==1 for r in rows]
    tests=[~x for x in training]
    fitted=fit_temporal_raw(partition,rows,assignment,2,training)
    selected_score=score_temporal_raw(partition,rows,fitted,tests,set(proposal))['nll']
    members=np.flatnonzero(partition.membership==g);edge=partition.edges
    graph=coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(partition.xyz),len(partition.xyz))).tocsr()
    changes=0;evaluations=0
    while True:
        best=None
        for unit in np.unique(partition.duplicate_unit[members]):
            if unit%2:continue
            ids=members[partition.duplicate_unit[members]==unit]
            if len(np.unique(assignment[ids]))!=1:continue
            trial=assignment.copy();trial[ids]=1-trial[ids]
            if any(connected_components(graph[members[trial[members]==label]][:,members[trial[members]==label]],directed=False,return_labels=False)!=1 for label in (0,1)):continue
            candidate={**fitted,'membership':np.maximum(trial,0)}
            score=score_temporal_raw(partition,rows,candidate,tests,set(proposal))['nll'];evaluations+=1
            if score<selected_score and (best is None or score<best[0]):best=(score,trial)
        if best is None:break
        selected_score,assignment=best;fitted={**fitted,'membership':np.maximum(assignment,0)};changes+=1
    member=heldout_comparison(partition,rows,assignment,fold=0)
    member['selection_conditioning']='early held-out frames selected connected extensions; future-block score below is untouched acceptance evidence'
    temporal=heldout_comparison(partition,rows,assignment,fold=0,selection_frames=set(frames[middle:]))
    camera=heldout_comparison(partition,rows,assignment,fold=camera_count-1,mode='camera') if camera_count>1 else {'status':'NOT_APPLICABLE_SINGLE_CAMERA'}
    comparisons=[member,temporal]+([camera] if camera_count>1 else [])
    accepted=all(c['status']=='SPLIT_SUPPORTED' for c in comparisons)
    status='APPROVED_SPLIT' if accepted else 'GROUP_UNRESOLVED' if any(c['status']=='UNRESOLVED_SPLIT' for c in comparisons) else 'SINGLE_GROUP_PREFERRED'
    return {'status':status,'proposal_reason':reason,'proposal_frames':proposal,'membership_labels':assignment,'member_holdout':member,'future_block_robustness':temporal,'camera_block_robustness':camera,'evidence_origin':origin,'connected_extension_search':{'evaluations':evaluations,'accepted_changes':changes,'criterion':'strictly smaller joint predictive NLL on early validation data; graph-connected single linked-unit flips; local finite candidate family, not global partition enumeration','untouched_test_frames':frames[middle:]},'conditional_association_caveat':'Raw correspondence selection remains conditioned, even when its diagnostic is independent of grouped rigidity. No claim to marginalize discrete-correspondence selection or shared calibration.'}


def split_partition(partition,g,assignment):
    membership=partition.membership.copy();membership[(membership==g)&(assignment==1)]=partition.count
    count=partition.count+1;centroids=np.empty((count,3));radius=np.empty(count)
    for group in range(count):
        xyz=partition.xyz[membership==group];centroids[group]=xyz.mean(0);radius[group]=np.linalg.norm(xyz-centroids[group],axis=1).max()
    adjacency=np.unique(np.sort(membership[partition.edges],axis=1),axis=0);adjacency=adjacency[adjacency[:,0]!=adjacency[:,1]]
    return replace(partition,membership=membership,centroids=centroids,radius=radius,adjacency=adjacency,version=partition.version+1)


def merge_partition(partition,a,b):
    if a==b or not np.any(np.all(partition.adjacency==np.sort([a,b]),axis=1)):raise ValueError('Only distinct surface-adjacent groups can merge')
    member=partition.membership.copy();member[member==b]=a
    _,member=np.unique(member,return_inverse=True);count=int(member.max())+1
    centers=np.asarray([partition.xyz[member==g].mean(0) for g in range(count)])
    radius=np.asarray([np.linalg.norm(partition.xyz[member==g]-centers[g],axis=1).max() for g in range(count)])
    edge=np.unique(np.sort(member[partition.edges],axis=1),axis=0);edge=edge[edge[:,0]!=edge[:,1]]
    return replace(partition,membership=member,centroids=centers,radius=radius,adjacency=edge,version=partition.version+1)


def validate_merge(partition,a,b,dataset,config,backend,anchor_index,rows=None):
    """Optional coarsening pays for itself on same-set unseen raw factors."""
    from .group_sensor_tracks import independent_group_sensor_rows
    if not np.any(np.all(partition.adjacency==np.sort([a,b]),axis=1)):return {'status':'MERGE_UNRESOLVED','reason':'Groups are not adjacent on the measured surface graph'}
    temp=merge_partition(partition,a,b)
    merged_id=int(temp.membership[np.flatnonzero(partition.membership==a)[0]])
    if rows is None:rows=independent_group_sensor_rows(temp,merged_id,dataset,config,backend,anchor_index)
    labels=np.zeros(len(partition.xyz),np.int64);labels[partition.membership==b]=1
    result=heldout_comparison(partition,rows,labels)
    return {'status':'APPROVED_MERGE' if result['status']=='SINGLE_PREFERRED' else 'MERGE_NOT_APPROVED','comparison':result,
        'criterion':'shared rigid trajectory beats separate trajectories after explicit complexity cost on identical held-out raw factors; not occupancy-driven'}


def run_validation_stages(root,dataset,partition,arrays,config,noise,backend,stop_after):
    # Fail closed at the authoritative acceptance boundary. This is not a
    # reconstruction completion or a successful split/segmentation result.
    result={'status':'STOPPED_AT_CORRECTIVE_GATES','complete_joint_pipeline':False,'reason':'Full-source and physical-output admission requires the corrective debug, correspondence, partition, coherence and joint acceptance gates. Passing the original boundary fixture alone does not authorize a reconstruction success claim.','unreached_stages':['validated real full-window moving support','shared rigid refinement','joint candidates and selection'],'point_recovery_started':False,'phase1_status':'NOT_RUN' if arrays is None else 'FILTER_COMPLETE_LOCAL_GROUPS_UNVALIDATED','patch_proposal':'Complete the corrective gate ledger before enabling full-source execution; preserve independent baseline and existing results.'}
    json_write(Path(root)/'gate_b_stop.json',result)
    return result
