"""Finite kNN candidate family, real ray tests, deterministic raw ownership."""
import time
import numpy as np
import torch
from .projection import project
from .observations import samples


def exclusive_assignment(point,pixel,cost,point_ids):
    """Deterministic local greedy matching; never two owners for a raw ray."""
    order=np.lexsort((pixel,point_ids[point],cost));remaining=order;chosen=[]
    while len(remaining):
        _,first=np.unique(point[remaining],return_index=True);best=remaining[np.sort(first)]
        _,first=np.unique(pixel[best],return_index=True);winners=best[np.sort(first)]
        chosen.extend(winners.tolist())
        remaining=remaining[~np.isin(point[remaining],point[winners])&~np.isin(pixel[remaining],pixel[winners])]
    return np.asarray(chosen,dtype=np.int64)


def associate(states,covariances,colors,point_ids,frame,config,backend,owners,raw=None):
    start=time.monotonic();raw=samples(frame,config) if raw is None else raw
    n=len(states);device=backend.device
    empty={'camera':frame.camera,'camera_index':frame.ci,'point':backend.tensor([],torch.long),'pixel':np.empty(0,np.int64),'z':backend.tensor(np.empty((0,3))),'R':backend.tensor(np.empty((0,3,3)))}
    stats={'frame':frame.k,'camera':frame.ci,'valid_raw':len(raw.pixel),'matched':0,'occluded':0,'ambiguous':0,'conflicts':0,'candidate_edges':0,'index_build_seconds':raw.index_build_seconds,'knn_seconds':0.,'projection_seconds':0.,'association_seconds':0.,'nearest_metric_quantiles_m':[],'sample_spacing_quantiles_m':[],'matched_d2':[],'unmatched_d2':[],'matched_depth_residual_m':[],'candidates':[]}
    stats.update(raw_pixels=int(frame.depth.size),foreground_pixels=int(frame.mask.sum()),invalid_foreground_pixels=int(frame.mask.sum())-len(raw.pixel))
    if not n or raw.tree is None:return empty,stats
    if len(raw.xyz)>1:
        spacing=raw.tree.query(raw.xyz,k=2,workers=1)[0][:,1]
        stats['sample_spacing_quantiles_m']=np.quantile(spacing,[.5,.9,.95,.99]).tolist()
    s=np.asarray(states);p=np.asarray(covariances)
    # Already assigned old raw observations are read-only. Points can gain at
    # most one actual geometric correspondence in this camera and timestamp.
    owned_ids=owners[owners>=0];available=~np.isin(point_ids,owned_ids)
    t=time.monotonic();h,j=project(backend.tensor(s),frame.camera);backend.synchronize();stats['projection_seconds']=time.monotonic()-t
    hh=h.detach().cpu().numpy();jj=j.detach().cpu().numpy()
    uv=np.zeros((n,2),np.int64);finite=np.isfinite(hh).all(1);uv[finite]=np.floor(hh[finite,:2]+.5).astype(np.int64)
    w,height=frame.camera.width,frame.camera.height
    inside=finite&(hh[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<height)&available
    ids=np.flatnonzero(inside)
    if not len(ids):return empty,stats
    query=uv[ids,1]*w+uv[ids,0];valid=frame.valid.ravel()[query]
    observed=np.zeros(len(ids));observed[valid]=np.asarray(frame.depth).ravel()[query[valid]]
    depth_variance=np.einsum('ni,nij,nj->n',jj[ids,2],p[ids],jj[ids,2])+config.measurement_covariance(np.where(valid,observed,0))[:,2,2]
    hidden=valid&(observed<hh[ids,2])&((observed-hh[ids,2])**2>config.gate(1)*depth_variance)
    stats['occluded']=int(hidden.sum());ids=ids[~hidden]
    if not len(ids):return empty,stats
    t=time.monotonic();distance,near=raw.tree.query(s[ids,:3],k=min(config.neighbors,len(raw.pixel)),workers=1)
    stats['knn_seconds']=time.monotonic()-t
    if near.ndim==1:near=near[:,None];distance=distance[:,None]
    stats['nearest_metric_quantiles_m']=np.quantile(distance[:,0],[.5,.9,.95,.99]).tolist()
    point=np.repeat(ids,near.shape[1]);neighbor=near.ravel();raw_pixel=raw.pixel[neighbor]
    if n==1:
        stats['candidates']=[{'frame':frame.k,'camera':frame.ci,'pixel':int(u),'point_id':int(point_ids[0]),'reason':'IRREVOCABLY_OWNED','z':raw.z[r].tolist()} for u,r in zip(raw_pixel,neighbor) if owners[u]>=0]
    free=owners[raw_pixel]<0;point=point[free];neighbor=neighbor[free];raw_pixel=raw_pixel[free]
    if not len(point):return empty,stats
    all_d2=[];all_cost=[];size=backend.chunk_size(3*3*8*20)
    for begin in range(0,len(point),size):
        pi=point[begin:begin+size];ri=neighbor[begin:begin+size]
        hq=backend.tensor(hh[pi]);hqj=backend.tensor(jj[pi]);pc=backend.tensor(p[pi]);r=backend.tensor(raw.R[ri]);error=backend.tensor(raw.z[ri])-hq
        innovation=hqj@pc@hqj.transpose(1,2)+r
        solved=torch.linalg.solve(innovation,error[...,None])[...,0];d2=(error*solved).sum(1)
        ray_error=error[:,:2];ray=torch.linalg.solve(innovation[:,:2,:2],ray_error[...,None])[...,0]
        ray_d2=(ray_error*ray).sum(1)
        rgb=backend.tensor(raw.rgb[ri]-np.asarray(colors)[pi]);color_precision=backend.tensor(np.linalg.inv(2*config.color_covariance))
        affinity=torch.einsum('ni,ij,nj->n',rgb,color_precision,rgb)
        cost=.5*(d2+torch.linalg.slogdet(innovation).logabsdet+affinity)
        good=(d2<=config.gate(3))&(ray_d2<=config.gate(2))
        all_d2.append(d2.detach().cpu().numpy());all_cost.append(torch.where(good,cost,torch.inf).detach().cpu().numpy())
    d2=np.concatenate(all_d2);cost=np.concatenate(all_cost);stats['candidate_edges']=len(cost)
    # Equal posterior scores are an unresolved discrete tie, never an inferred
    # physical group or an arbitrary distance margin.
    order=np.lexsort((raw_pixel,cost,point));sorted_points=point[order]
    first=np.r_[True,sorted_points[1:]!=sorted_points[:-1]];rows=np.flatnonzero(first)
    second=rows+1;has=second<len(order);has[has]&=sorted_points[second[has]]==sorted_points[rows[has]]
    tied=np.zeros(n,bool)
    a=order[rows[has]];b=order[second[has]];finite_pair=np.isfinite(cost[a])&np.isfinite(cost[b])
    equal=np.zeros(len(a),bool);equal[finite_pair]=abs(cost[a[finite_pair]]-cost[b[finite_pair]])<=np.finfo(float).eps*8*np.maximum(1,abs(cost[a[finite_pair]]))
    tied[point[a[equal]]]=True;stats['ambiguous']=int(tied.sum())
    eligible=np.flatnonzero(np.isfinite(cost)&~tied[point])
    selected=eligible[exclusive_assignment(point[eligible],raw_pixel[eligible],cost[eligible],np.asarray(point_ids))]
    stats['conflicts']=int(len(np.unique(point[eligible]))-len(selected));stats['matched']=len(selected)
    matched_points=point[selected]
    stats['matched_d2']=d2[selected].tolist()
    best=np.full(n,np.inf);np.minimum.at(best,point,d2)
    unmatched=np.isfinite(best)&~np.isin(np.arange(n),matched_points);stats['unmatched_d2']=best[unmatched].tolist()
    stats['association_seconds']=time.monotonic()-start
    result={'camera':frame.camera,'camera_index':frame.ci,'point':backend.tensor(matched_points,torch.long),'pixel':raw_pixel[selected],
        'z':backend.tensor(raw.z[neighbor[selected]]),'R':backend.tensor(raw.R[neighbor[selected]]),
        'd2':d2[selected],'residual':raw.z[neighbor[selected]]-hh[matched_points],
        'S':np.einsum('nij,njk,nlk->nil',jj[matched_points],p[matched_points],jj[matched_points])+raw.R[neighbor[selected]]}
    stats['matched_depth_residual_m']=result['residual'][:,2].tolist()
    if n==1:
        selected_set=set(selected.tolist())
        for i,(pi,ri,u) in enumerate(zip(point,neighbor,raw_pixel)):
            reason='REAL_MEASUREMENT' if i in selected_set else ('AMBIGUOUS' if tied[pi] else ('GEOMETRY_OR_RAY_GATE' if not np.isfinite(cost[i]) else 'EXCLUSIVE_OWNERSHIP_CONFLICT'))
            stats['candidates'].append({'frame':frame.k,'camera':frame.ci,'pixel':int(u),'point_id':int(point_ids[pi]),'z':raw.z[ri].tolist(),'residual':(raw.z[ri]-hh[pi]).tolist(),'S':(jj[pi]@p[pi]@jj[pi].T+raw.R[ri]).tolist(),'d2':float(d2[i]),'reason':reason})
    return result,stats
