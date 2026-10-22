"""Actual backward raw-point matching and canonical fusion, shared by filters."""
import numpy as np
from .geometry import ObservationIDs
from .fusion import backproject,canonicalize,fuse_many
from .backends.torch_cuda import numpy
from .projection_cuda import inverse_transport_unmatched


def reverse_match_unsolved(model,states,dataset,source_frame,camera_index,pixels,config,backend,cache,target_frames=None):
    """One streamed source-camera batch, all requested earlier times/cameras.

    Local visibility odds rank proposals. Their acceptance always uses the exact
    common score, not this provisional per-point evidence approximation.
    """
    pixels=np.asarray(pixels,dtype=np.int64)
    if not len(pixels):return None,{'source_frame':source_frame,'camera':camera_index,'source_unresolved':0,'target_times':0}
    frame=dataset.observation(source_frame,camera_index)
    world,color,world_cov,pixels=backproject(frame,config,backend,pixels)
    raw=ObservationIDs(dataset);source_ids=raw.encode(source_frame,camera_index,pixels)
    groups=len(states[0].poses);n=len(pixels)
    best=np.full(n,-np.inf);chosen=np.full(n,-1,dtype=int)
    support_frame=np.full(n,-1,dtype=int);support_camera=np.full(n,-1,dtype=int);support_pixel=np.full(n,-1,dtype=np.int64)
    support_owner=np.full(n,-1,dtype=np.int64);target_times=list(range(source_frame-1,-1,-1)) if target_frames is None else list(target_frames)
    query_count=0;hidden_count=0;free_count=0
    world_device=backend.tensor(world);reference=backend.tensor(np.eye(4))
    for g in range(groups):
        xyz=numpy(inverse_transport_unmatched(world_device,backend.tensor(states[source_frame].poses[g]),reference))
        from .backward_cuda import backward_evidence
        evidence,strength,sf,sc,sp,so,statistics=backward_evidence(xyz,color,g,states,dataset,target_times,config,backend,cache,model)
        query_count+=statistics['camera_group_queries'];hidden_count+=statistics['hidden_tests'];free_count+=statistics['free_space_tests']
        update=(evidence>best)&(sf>=0)
        best[update]=evidence[update];chosen[update]=g;support_frame[update]=sf[update];support_camera[update]=sc[update];support_pixel[update]=sp[update];support_owner[update]=so[update]
    accepted=(chosen>=0)&(best>0)
    # Source pixels already registered in the map are never a second independent
    # point birth. Their geometry is revisited by original-factor refusion.
    selected=np.flatnonzero(accepted&(model.source_owners(source_ids)<0))
    if not len(selected):return None,{'source_frame':source_frame,'camera':camera_index,'source_unresolved':n,'supported':0,'target_times':len(target_times),'camera_group_queries':query_count,'hidden_tests':hidden_count,'free_space_tests':free_count}
    proposal=model.copy();changed=[];added=0;fused=0;new_rows=[]
    # Group/camera batches retain one raw owner and immediately fuse a source
    # into the existing physical point when its backward observation owns one.
    for g in range(groups):
        use=selected[chosen[selected]==g]
        if not len(use):continue
        p,c=canonicalize(world[use],world_cov[use],states[source_frame].poses[g])
        old=support_owner[use].copy()
        support_ids=raw.encode(support_frame[use],support_camera[use],support_pixel[use])
        registered=model.source_owners(support_ids);old[registered>=0]=registered[registered>=0]
        compatible=(old>=0)
        compatible[compatible]&=model.group[old[compatible]]==g
        if np.any(compatible):
            fused_ids=fuse_many(proposal,old[compatible],p[compatible],color[use[compatible]],c[compatible],source_ids[use[compatible]],backend)
            changed.extend(fused_ids.tolist());fused+=int(compatible.sum())
        # An already-owned target on another rigid group is ambiguous; it stays
        # unsolved until a group/reassignment proposal explains the same data.
        novel=np.flatnonzero(old<0)
        for j in novel:new_rows.append((int(use[j]),g,p[j],c[j],int(support_ids[j])))
    if new_rows:
        ids=np.array([row[0] for row in new_rows]);labels=np.array([row[1] for row in new_rows]);canonical=np.array([row[2] for row in new_rows]);conditional=np.array([row[3] for row in new_rows]);target_ids=np.array([row[4] for row in new_rows])
        # Competing groups cannot independently consume the same raw target.
        order=np.argsort(-best[ids],kind='stable');_,first=np.unique(target_ids[order],return_index=True)
        winner_group={int(target_ids[order[j]]):int(labels[order[j]]) for j in first}
        keep=np.array([winner_group[int(t)]==g for t,g in zip(target_ids,labels)])
        ids=ids[keep];labels=labels[keep];canonical=canonical[keep];conditional=conditional[keep];target_ids=target_ids[keep]
        unique,first,group=np.unique(target_ids,return_index=True,return_inverse=True);m=len(unique)
        target_xyz=np.empty((m,3));target_rgb=np.empty((m,3));target_cov=np.empty((m,6,6))
        times,cameras,rawpixels=raw.decode(unique)
        for k in np.unique(times):
            for ci in np.unique(cameras[times==k]):
                select=np.flatnonzero((times==k)&(cameras==ci));frame=dataset.observation(int(k),int(ci))
                x,a,c,_=backproject(frame,config,backend,rawpixels[select])
                for g in np.unique(labels[first[select]]):
                    take=labels[first[select]]==g;which=select[take]
                    target_xyz[which],target_cov[which]=canonicalize(x[take],c[take],states[int(k)].poses[int(g)])
                    target_rgb[which]=a[take]
        precision=np.linalg.inv(conditional);information=np.linalg.inv(target_cov)
        rhs=np.einsum('nij,nj->ni',information,np.c_[target_xyz,target_rgb])
        np.add.at(information,group,precision);np.add.at(rhs,group,np.einsum('nij,nj->ni',precision,np.c_[canonical,color[ids]]))
        variance=np.linalg.inv(information);mean=np.einsum('nij,nj->ni',variance,rhs)
        provenance=[[int(raw_id)] for raw_id in unique]
        for j,source in zip(group,source_ids[ids]):provenance[j].append(int(source))
        start=proposal.count;proposal.append(mean[:,:3],mean[:,3:],labels[first],variance,source_ids[ids[first]],provenance)
        changed.extend(range(start,proposal.count));added=m
    if not changed:return None,{'source_frame':source_frame,'camera':camera_index,'supported':len(selected),'added':0,'fused':0,'reason':'all supporting source IDs already present'}
    report={'source_frame':source_frame,'camera':camera_index,'source_unresolved':n,'supported':len(selected),'added':added,'fused':fused,'target_times':len(target_times),'camera_group_queries':query_count,'hidden_tests':hidden_count,'free_space_tests':free_count,'proposal_ranking':'conditional point matches plus Gaussian depth-order visibility odds; ranking approximation only, never the accepted global score'}
    return (proposal,np.unique(changed)),report
