"""Previously unexplained multi-frame rigid groups, from raw RGB-D tracks."""
import numpy as np
from .lie import State,exp,log,inverse
from .group_proposals import discover_rigid_groups,world_at,flow_window,track_log_probability
from .geometry import ObservationIDs,PointMap
from .fusion import backproject,canonicalize,refuse_original_factors
from .matching_cuda import match_rgbd
from .backends.torch_cuda import numpy
from .progress import progress
from .foreground import foreground_gray


def backward_feature_tracks(dataset,s,ci,corners,source):
    """Use the longest actually trackable baseline, not two noisy frame steps.

    Three noncollinear tracks are a rank requirement, not a geometry threshold.
    Only this bounded sparse INITIALIZER selects two target anchors; full
    estimation and scoring still read every timestamp and every camera.
    """
    import cv2
    old=foreground_gray(source);current=corners.copy()
    alive=np.arange(len(corners));history={}
    for k in progress(range(s-1,-1,-1),desc='Track unresolved rigid features backward',unit='frame',leave=False):
        frame=dataset.observation(k,ci);image=foreground_gray(frame)
        moved,status,_=cv2.calcOpticalFlowPyrLK(old,image,current,None,winSize=flow_window(frame),maxLevel=4)
        _,inside=world_at(frame,moved[:,0]);good=status[:,0].astype(bool)&inside&np.isfinite(moved[:,0]).all(1)
        if good.sum()<3:break
        alive=alive[good];current=moved[good];history[k]=(alive.copy(),current[:,0].copy());old=image
    if len(history)<2:return None
    far=min(history);mid=(dataset.times[s]+dataset.times[far])/2
    near=min((k for k in history if k!=far),key=lambda k:abs(dataset.times[k]-mid))
    survivors=history[far][0]
    uv=np.array([history[k][1][np.searchsorted(history[k][0],survivors)] for k in (near,far)])
    return corners[survivors,0],uv,[near,far],len(history)


def propose_new_group(model,states,dataset,s,ci,unresolved,config,backend,feature_budget=4000,feature_horizon='adjacent'):
    if s<2 or len(unresolved)<3:return None,{'attempted':False,'reason':'need three acquisition times and noncollinear correspondences'}
    import cv2
    source=dataset.observation(s,ci);mask=np.zeros(source.depth.size,np.uint8);mask[unresolved]=255;mask=mask.reshape(source.depth.shape)
    gray=foreground_gray(source)
    corners=cv2.goodFeaturesToTrack(gray,feature_budget,.01,3,mask=mask)
    if corners is None:return None,{'attempted':True,'reason':'no identifiable unresolved RGB features'}
    if feature_horizon=='longest-trackable':
        tracks=backward_feature_tracks(dataset,s,ci,corners,source)
        if tracks is None:return None,{'attempted':True,'reason':'insufficient valid multiframe depth tracks'}
        origin,uv,target_indices,tracked_frames=tracks
    else:
        origin=corners[:,0];_,valid=world_at(source,origin);uv=[];current=corners;old=gray
        target_indices=[s-1,s-2];tracked_frames=2
        for k in target_indices:
            frame=dataset.observation(k,ci);image=foreground_gray(frame)
            moved,status,_=cv2.calcOpticalFlowPyrLK(old,image,current,None,winSize=flow_window(frame),maxLevel=4)
            _,inside=world_at(frame,moved[:,0]);valid&=status[:,0].astype(bool)&inside&np.isfinite(moved[:,0]).all(1)
            uv.append(moved[:,0]);old=image;current=moved
        if valid.sum()<3:return None,{'attempted':True,'reason':'insufficient valid multiframe depth tracks'}
        origin=origin[valid];uv=np.array(uv)[:,valid]
    coords=[origin,*uv];pixels=[]
    for coordinates in coords:
        p=np.floor(coordinates+.5).astype(int);pixels.append(p[:,1]*source.camera.width+p[:,0])
    used=[set() for _ in pixels];keep=[]
    for i in range(len(origin)):
        if not any(int(p[i]) in keys for p,keys in zip(pixels,used)):
            keep.append(i)
            for p,keys in zip(pixels,used):keys.add(int(p[i]))
    keep=np.asarray(keep,dtype=int);origin=origin[keep];uv=uv[:,keep];pixels=[p[keep] for p in pixels]
    if len(keep)<3:return None,{'attempted':True,'reason':'fewer than three raw-noise-independent feature tracks'}
    sd=backproject(source,config,backend,pixels[0]);td=[backproject(dataset.observation(k,ci),config,backend,p) for k,p in zip(target_indices,pixels[1:])]
    x=sd[0];target=np.array([d[0] for d in td]);noise={'source_rgb':sd[1],'target_rgb':np.array([d[1] for d in td]),'source_covariance':sd[2],'target_covariance':np.array([d[2] for d in td])}
    covariance=noise['target_covariance'][:,:,:3,:3]+noise['source_covariance'][None,:,:3,:3]
    try:transforms,weights,report=discover_rigid_groups(x,target,covariance,config,time_offsets=dataset.times[target_indices]-dataset.times[s],track_noise=noise)
    except ValueError as error:return None,{'attempted':True,'reason':str(error)}
    existing=[]
    for g in range(len(states[0].poses)):
        bank=np.array([states[k].poses[g]@inverse(states[s].poses[g]) for k in target_indices])
        existing.append(track_log_probability(x,target,bank,noise,config))
    old=np.max(existing,axis=0);best=None
    for g,bank in enumerate(transforms):
        evidence=track_log_probability(x,target,bank,noise,config)
        use=(weights.argmax(1)==g)&(evidence>old)
        if use.sum()>=3:
            value=float((evidence[use]-old[use]).sum())
            if best is None or value>best[0]:best=(value,bank,use)
    if best is None:return None,{'attempted':True,'reason':'existing groups explain all identifiable tracks'}
    _,bank,use=best;group=len(states[0].poses);times=dataset.times
    known=[bank[1],bank[0],np.eye(4)];indices=[target_indices[1],target_indices[0],s]
    def at(k):
        j=0 if k<target_indices[0] else 1
        a,b=known[j:j+2];dt=times[indices[j+1]]-times[indices[j]]
        return exp((times[k]-times[indices[j]])/dt*log(b@inverse(a)))@a
    gauge=inverse(at(0));poses=np.array([at(k)@gauge for k in range(len(times))])
    nominal=[]
    for k,state in enumerate(states):
        j=min(k,len(times)-2);twist=log(poses[j+1]@inverse(poses[j]))/(times[j+1]-times[j])
        nominal.append(State(np.concatenate([state.poses,poses[k:k+1]]),np.concatenate([state.twists,twist[None]])))
    raw=ObservationIDs(dataset);selected=np.flatnonzero(use)
    source_pixel=np.floor(origin[selected]+.5).astype(int);source_pixel=source_pixel[:,1]*source.camera.width+source_pixel[:,0]
    all_pixels=[source_pixel]
    for target_uv in uv[:,selected]:
        pixel=np.floor(target_uv+.5).astype(int);all_pixels.append(pixel[:,1]*source.camera.width+pixel[:,0])
    provenance=np.stack([raw.encode(k,ci,pixel) for k,pixel in zip((s,*target_indices),all_pixels)],axis=1)
    provenance=[list(map(int,row)) for row in provenance]
    # Extend the NEW motion hypothesis through all earlier views. A match to an
    # original frame-0 source can reassign that existing point to the new group,
    # rather than leaving a stationary ghost and adding a duplicate moving copy.
    source_world,source_rgb,source_cov,_=backproject(source,config,backend,all_pixels[0])
    canonical,conditional=canonicalize(source_world,source_cov,poses[s])
    temporary=PointMap();temporary.append(canonical,source_rgb,np.full(len(canonical),group),conditional,raw.encode(s,ci,all_pixels[0]))
    for k in range(target_indices[1]-1,-1,-1):
        for camera_index in range(len(dataset.cameras)):
            frame=dataset.observation(k,camera_index);matches,*_=match_rgbd(temporary,nominal[k],frame,config,backend)
            for point,raw_id in zip(numpy(matches.point),raw.encode(k,camera_index,numpy(matches.pixel))):provenance[int(point)].append(int(raw_id))
    # Union tracks sharing original raw evidence or existing point identities.
    # This proposes a physical merge/reassignment, never two independent copies
    # of one source pixel. Full replay/score must accept the group edit.
    parent={}
    def root(i):
        parent.setdefault(i,i)
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a=root(a);b=root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    owners=[model.source_owners(row) for row in provenance];by_raw={}
    for i,row in enumerate(provenance):
        node=model.count+i;root(node)
        for raw_id,owner in zip(row,owners[i]):
            if owner>=0:join(node,int(owner))
            if int(raw_id) in by_raw:join(node,by_raw[int(raw_id)])
            by_raw[int(raw_id)]=node
    components={}
    for i in range(len(provenance)):components.setdefault(root(model.count+i),[]).append(i)
    if len(components)<3:return None,{'attempted':True,'reason':'fewer than three independent physical point tracks after source fusion'}
    remove=set();replacement={};born=[]
    for node,tracks in components.items():
        old_points=sorted({int(p) for track in tracks for p in owners[track] if p>=0})
        raw_ids={int(raw_id) for track in tracks for raw_id in provenance[track]}
        for point in old_points:raw_ids.update(model.provenance[point])
        if old_points:
            representative=old_points[0];remove.update(old_points[1:]);replacement[representative]=sorted(raw_ids)
        else:born.append((tracks[0],sorted(raw_ids)))
    kept=np.array([i for i in range(model.count) if i not in remove]);proposal=model.take(kept);old_to_new={int(p):j for j,p in enumerate(kept)}
    changed=[];source_rows={}
    for old_point,row in replacement.items():
        i=old_to_new[old_point];proposal.group[i]=group;source_rows[i]=row;changed.append(i)
    proposal.replace_sources(source_rows)
    if born:
        which=np.array([i for i,_ in born]);world,rgb,cov,_=backproject(source,config,backend,all_pixels[0][which])
        xyz,conditional=canonicalize(world,cov,poses[s]);start=proposal.count
        proposal.append(xyz,rgb,np.full(len(born),group),conditional,np.array([provenance[int(i)][0] for i in which]),[row for _,row in born]);changed.extend(range(start,proposal.count))
    proposal,_,_=refuse_original_factors(proposal,nominal,dataset,config,backend,changed)
    report.update({'attempted':True,'source_frame':s,'camera':ci,'new_group':group,'new_points':len(born),'reassigned_points':len(replacement),'merged_old_points':len(remove),'initializer_only':True,'requires_chosen_filter_replay':True,'feature_anchor_frames':[s,*target_indices],'actually_tracked_earlier_frames':tracked_frames})
    return (proposal,nominal,np.asarray(changed)),report


def new_group_proposals(*args):
    # Both are feature INITIALIZERS; only the common full-data score decides
    # acceptance. Trying the longer evidence baseline does not replace the
    # original local family or compare likelihoods on different image domains.
    for horizon in ('adjacent','longest-trackable'):
        proposal,report=propose_new_group(*args,feature_horizon=horizon)
        report['feature_horizon']=horizon
        yield proposal,report
