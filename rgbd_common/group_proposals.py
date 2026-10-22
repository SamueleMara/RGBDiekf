"""Measured rigid hypotheses; every group follows the same state and prior."""
import numpy as np
from scipy.special import logsumexp
from .lie import exp,log,inverse,State,adjoint
from .fusion import backproject
from .foreground import foreground_gray
from .progress import progress


def weighted_rigid_fit(source,target,weights):
    weights=np.asarray(weights);total=weights.sum()
    if total<=np.finfo(float).eps:raise ValueError('UNOBSERVABLE_GROUP: no correspondence weight')
    x=(source*weights[:,None]).sum(0)/total;y=(target*weights[:,None]).sum(0)/total
    u,s,v=np.linalg.svd((source-x).T@((target-y)*weights[:,None]))
    if s[1]<=np.finfo(float).eps*max(1.,s[0]):raise ValueError('UNOBSERVABLE_GROUP: collinear correspondences')
    r=v.T@np.diag([1.,1.,np.linalg.det(v.T@u.T)])@u.T
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=y-r@x;return t


def _gaussian(residual,covariance):
    return -.5*(np.einsum('ni,nij,nj->n',residual,np.linalg.inv(covariance),residual)+np.linalg.slogdet(covariance)[1]+3*np.log(2*np.pi))


def track_log_probability(source,targets,group,track_noise,config,broad_component=False):
    dimension=6;times=len(targets);count=len(source)
    operators=[];residuals=[]
    for t,(pose,y) in enumerate(zip(group,targets)):
        operator=np.eye(6);operator[:3,:3]=pose[:3,:3];operators.append(operator)
        residuals.append(np.c_[y-(source@pose[:3,:3].T+pose[:3,3]),track_noise['target_rgb'][t]-track_noise['source_rgb']])
    operator=np.concatenate(operators,axis=0)
    src=track_noise['source_covariance']+(config.map_covariance if broad_component else 0)
    cov=operator@src@operator.T
    for t in range(times):
        cov[:,6*t:6*t+6,6*t:6*t+6]+=track_noise['target_covariance'][t]+(config.map_covariance if broad_component else 0)
    residual=np.stack(residuals,axis=1).reshape(count,times*dimension)
    solved=np.linalg.solve(cov,residual[...,None])[...,0]
    return -.5*((residual*solved).sum(1)+np.linalg.slogdet(cov)[1]+times*dimension*np.log(2*np.pi))

def discover_rigid_groups(source,targets,covariance,config,seed_budget=64,time_offsets=None,track_noise=None):
    """Local deterministic mixture model proposals from >=2 measured intervals.

    The seed budget limits initializer exploration, not reconstructed points.
    Membership is shared across all supplied time pairs; no index label pattern.
    """
    from .model_contract import require_model_selection
    require_model_selection(config)
    targets=np.asarray(targets);covariance=np.asarray(covariance)
    if targets.shape[0]<2 or len(source)<3:raise ValueError('Need multiframe noncollinear correspondences to discover rigid groups')
    precision_weight=1/np.trace(covariance,axis1=2,axis2=3)
    transforms=np.array([[weighted_rigid_fit(source,y,w) for y,w in zip(targets,precision_weight)]])
    rng=np.random.default_rng(0);broad=np.asarray(config.map_covariance)[:3,:3]*2
    precision=np.linalg.inv(covariance);log_normalizer=np.linalg.slogdet(covariance)[1]+3*np.log(2*np.pi)
    outlier=np.sum([_gaussian(y-source,c+broad) for y,c in zip(targets,covariance)],axis=0)+np.log(config.sensor['no_return'])
    if track_noise is not None:outlier=track_log_probability(source,targets,np.tile(np.eye(4),(len(targets),1,1)),track_noise,config,True)+np.log1p(-config.match_prior)
    def evaluate(bank):
        columns=[]
        for group in bank:
            if track_noise is not None:columns.append(track_log_probability(source,targets,group,track_noise,config))
            else:
                residual=targets-np.stack([source@t[:3,:3].T+t[:3,3] for t in group])
                columns.append(-.5*(np.einsum('tni,tnij,tnj->tn',residual,precision,residual)+log_normalizer).sum(axis=0))
        match_mass=config.match_prior if track_noise is not None else 1-config.sensor['no_return']
        count=len(bank);logits=np.c_[np.stack(columns,axis=1)+np.log(match_mass)-np.log(count),outlier]
        mixture_normal=logsumexp(logits,axis=1);quality=float(mixture_normal.sum()+config.initializer_log_prior(count))
        return quality,np.exp(logits-mixture_normal[:,None])
    quality,weights=evaluate(transforms)
    while len(transforms)<len(source)//3:
        best=None
        # Seed triples come from unexplained trajectory evidence, never labels.
        probability=np.maximum(1-weights[:,:len(transforms)].max(axis=1),np.finfo(float).eps);probability/=probability.sum()
        for _ in progress(range(seed_budget),desc=f'Rigid proposal seeds ({len(transforms)} groups)',unit='seed',leave=False):
            selected=rng.choice(len(source),3,replace=False,p=probability)
            try:new=np.array([weighted_rigid_fit(source[selected],y[selected],np.ones(3)) for y in targets])
            except ValueError:continue
            proposal=np.concatenate([transforms,new[None]])
            _,seed_weights=evaluate(proposal);hard=seed_weights[:,:len(proposal)].argmax(1)
            supported=seed_weights[:,:len(proposal)].max(1)>seed_weights[:,-1]
            # Refit the other cluster after introducing a seed. Otherwise an
            # outlier component can trap the original averaged transform with
            # exponentially tiny weight and generate duplicate rigid groups.
            for g in range(len(proposal)):
                if ((hard==g)&supported).sum()>=3:
                    try:proposal[g]=np.array([weighted_rigid_fit(source,y,((hard==g)&supported)*w) for y,w in zip(targets,precision_weight)])
                    except ValueError:pass
            value,_=evaluate(proposal)
            if best is None or value>best[0]:best=(value,proposal)
        if best is None:break
        proposal=best[1]
        for _ in range(20):
            _,weights_new=evaluate(proposal);updated=[]
            try:
                for g in range(len(proposal)):
                    updated.append([weighted_rigid_fit(source,y,weights_new[:,g]*w) for y,w in zip(targets,precision_weight)])
            except ValueError:break
            updated=np.array(updated)
            if evaluate(updated)[0]<evaluate(proposal)[0]:break
            change=np.linalg.norm(updated-proposal);proposal=updated
            if change<1e-9:break
        # Delete redundant/dead components by the SAME normalized mixture score,
        # not a hand-chosen transform-distance or component-weight cutoff.
        while len(proposal)>1:
            current,_=evaluate(proposal);alternatives=[]
            for g in range(len(proposal)):
                removed=np.delete(proposal,g,axis=0);value,_=evaluate(removed);alternatives.append((value,removed))
            value,removed=max(alternatives,key=lambda item:item[0])
            if value<=current+config.numerical['objective_tolerance']:break
            proposal=removed
        value,candidate_weights=evaluate(proposal)
        if value<=quality+config.numerical['objective_tolerance']:break
        transforms=proposal;quality=value;weights=candidate_weights
    return transforms,weights[:,:len(transforms)],{'groups':len(transforms),'initializer_score':quality,'seed_budget':seed_budget,'time_offsets_seconds':None if time_offsets is None else np.asarray(time_offsets).tolist(),'source':'multiframe measured rigid correspondence mixture; initializer only, full raw measurement AND process-prior score acceptance required. Finite differences of noisy feature transforms are not independent twist observations.'}


def world_at(frame,uv):
    finite=np.isfinite(uv).all(1);pixel=np.zeros_like(uv,dtype=int);pixel[finite]=np.floor(uv[finite]+.5).astype(int);x=pixel[:,0];y=pixel[:,1]
    valid=finite&(x>=0)&(x<frame.camera.width)&(y>=0)&(y<frame.camera.height)
    x=np.clip(x,0,frame.camera.width-1);y=np.clip(y,0,frame.camera.height-1)
    z=frame.depth[y,x];valid&=frame.valid[y,x];fx,fy,cx,cy=frame.camera.intrinsics
    result=np.full((len(uv),3),np.nan);ray=np.c_[(uv[valid,0]-cx)/fx,(uv[valid,1]-cy)/fy,np.ones(valid.sum())]
    result[valid]=(ray*z[valid,None]-frame.camera.world_to_camera[:3,3])@frame.camera.world_to_camera[:3,:3]
    return result,valid


def flow_window(frame):
    # Initializer image-processing budget, scaled for small numerical fixtures;
    # it is not an association/geometry likelihood gate.
    size=min(31,max(5,(min(frame.camera.width,frame.camera.height)//4)|1))
    return (size,size)


def optical_flow_tracks(dataset,frame_indices,config,backend,feature_budget=4000):
    import cv2
    cv2.setNumThreads(1);source=[];targets=[];sources=[];rgb_source=[];rgb_target=[];cov_source=[];cov_target=[]
    for ci in range(len(dataset.cameras)):
        first=dataset.observation(0,ci);gray=foreground_gray(first)
        corners=cv2.goodFeaturesToTrack(gray,feature_budget,.01,3,mask=np.uint8(first.valid)*255)
        if corners is None:continue
        uv=corners[:,0];x,keep=world_at(first,uv);world=[];old_gray=gray;current_uv=corners.copy();target_uv=[];target_frames=[]
        for k in progress(range(1,max(frame_indices)+1),desc=f'Bootstrap RGB-D tracks camera {ci}',unit='frame',leave=False):
            frame=dataset.observation(k,ci);new_gray=foreground_gray(frame)
            moved,status,_=cv2.calcOpticalFlowPyrLK(old_gray,new_gray,current_uv,None,winSize=flow_window(frame),maxLevel=4)
            good=status[:,0].astype(bool)&np.isfinite(moved[:,0]).all(1)
            positions,inside=world_at(frame,moved[:,0]);keep&=good&inside
            old_gray=new_gray;current_uv=moved
            if k in frame_indices:
                world.append(positions);target_uv.append(moved[:,0]);target_frames.append(frame)
        raw_coordinates=[uv,*target_uv];raw_pixels=[]
        for coords in raw_coordinates:
            p=np.floor(coords+.5).astype(int);raw_pixels.append(p[:,1]*first.camera.width+p[:,0])
        # A repeated raw pixel in two feature tracks is one noise variable.
        # Keep a deterministic disjoint track set instead of pretending independence.
        used=[set() for _ in raw_pixels];selected=[]
        for i in np.flatnonzero(keep):
            if not any(int(p[i]) in keys for p,keys in zip(raw_pixels,used)):
                selected.append(i)
                for p,keys in zip(raw_pixels,used):keys.add(int(p[i]))
        selected=np.asarray(selected,dtype=int)
        if not len(selected):continue
        source_data=backproject(first,config,backend,raw_pixels[0][selected])
        target_data=[backproject(f,config,backend,p[selected]) for f,p in zip(target_frames,raw_pixels[1:])]
        source.append(source_data[0]);targets.append(np.array([d[0] for d in target_data]));sources.append(np.c_[np.full(len(selected),ci),uv[selected]])
        rgb_source.append(source_data[1]);rgb_target.append(np.array([d[1] for d in target_data]));cov_source.append(source_data[2]);cov_target.append(np.array([d[2] for d in target_data]))
    if not source or sum(map(len,source))<3:raise ValueError('UNOBSERVABLE_GROUP_INITIALIZATION: insufficient multiframe RGB-D flow tracks')
    noise={'source_rgb':np.concatenate(rgb_source),'target_rgb':np.concatenate(rgb_target,axis=1),'source_covariance':np.concatenate(cov_source),'target_covariance':np.concatenate(cov_target,axis=1)}
    return np.concatenate(source),np.concatenate(targets,axis=1),np.concatenate(sources),noise


def bootstrap_groups(model,dataset,config,backend):
    if len(dataset.times)<3:raise ValueError('At least three times are required for multiframe rigid-group discovery')
    selected=sorted(set([max(1,len(dataset.times)//2),len(dataset.times)-1]))
    if len(selected)<2:selected=[1,2]
    source,targets,raw,noise=optical_flow_tracks(dataset,selected,config,backend)
    cov=noise['target_covariance'][:,:,:3,:3]+noise['source_covariance'][None,:,:3,:3]
    transforms,weights,report=discover_rigid_groups(source,targets,cov,config,time_offsets=dataset.times[selected]-dataset.times[0],track_noise=noise)
    groups=len(transforms)
    from scipy.spatial import cKDTree
    _,near=cKDTree(source).query(model.xyz)
    model.group=weights.argmax(axis=1)[near].astype(np.int32)
    anchor_times=np.r_[dataset.times[0],dataset.times[selected]];anchors=np.concatenate([np.tile(np.eye(4),(groups,1,1,1)),transforms],axis=1)
    states=[]
    for k,time in enumerate(dataset.times):
        j=min(np.searchsorted(anchor_times,time,side='right')-1,len(anchor_times)-2)
        alpha=(time-anchor_times[j])/(anchor_times[j+1]-anchor_times[j])
        poses=np.array([exp(alpha*log(b@inverse(a)))@a for a,b in zip(anchors[:,j],anchors[:,j+1])])
        twist=np.array([log(b@inverse(a))/(anchor_times[j+1]-anchor_times[j]) for a,b in zip(anchors[:,j],anchors[:,j+1])])
        states.append(State(poses,twist))
    report.update({'frames_used_only_for_nominal_bootstrap':[0,*selected],'tracks':len(source),'covariance_qualification':'Joint multi-time XYZ/RGB source/target noise, including shared raw-source cross blocks. Weighted rigid fits are initializers accepted against that regularized score, never posterior factors.','point_affiliation':'nearest measured trajectory initializer, revisable by common score'})
    return states,report
