"""Controlled pre-correction candidate/gate ablation, used ONLY by the audit.

Retains the old SOLVED -> observed eight-neighbor family, UVZ covariance, depth
ordering, RGB ranking and numerical-tie convention. Only ownership and exclusive
assignment are removed. Thus reclassified old-unowned pixels quantify coverage
lost through those restrictions, without changing the filter or its gates.
"""
import time
import numpy as np
import torch


def nonexclusive_legacy(surface,frame,raw,owners,config,backend):
    start=time.monotonic();m=len(raw.pixel)
    covered=np.zeros(m,bool);same_ties=np.zeros(m,bool);owned_witness=np.zeros(m,bool)
    stats={'candidate_edges':0,'compatible_edges':0,'numerical_tied_points':0}
    if not m or raw.tree is None:return covered,same_ties,owned_witness,stats
    h,j,_=surface.projection(frame,config,backend)
    finite=np.isfinite(h).all(1);uv=np.zeros((len(h),2),np.int64);uv[finite]=np.floor(h[finite,:2]+.5).astype(np.int64)
    w,height=frame.camera.width,frame.camera.height
    inside=finite&(h[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<height)
    ids=np.flatnonzero(inside);query=uv[ids,1]*w+uv[ids,0];valid=frame.valid.ravel()[query]
    depth=np.zeros(len(ids));depth[valid]=np.asarray(frame.depth).ravel()[query[valid]]
    variance=np.einsum('ni,nij,nj->n',j[ids,2],surface.p[ids],j[ids,2])+config.measurement_covariance(np.where(valid,depth,0))[:,2,2]
    hidden=valid&(depth<h[ids,2])&((depth-h[ids,2])**2>config.gate(1)*variance);ids=ids[~hidden]
    assigned=np.isin(surface.ids,owners[owners>=0]);k=min(config.neighbors,m)
    size=max(1,backend.chunk_size(3*3*8*30)//k)
    precision=backend.tensor(np.linalg.inv(2*config.color_covariance))
    for lo in range(0,len(ids),size):
        selected=ids[lo:lo+size];_,near=raw.tree.query(surface.state[selected,:3],k=k,workers=1)
        if near.ndim==1:near=near[:,None]
        pi=np.broadcast_to(selected[:,None],near.shape)
        jj=backend.tensor(j[pi]);p=backend.tensor(surface.p[pi]);error=backend.tensor(raw.z[near]-h[pi]);r=backend.tensor(raw.R[near])
        innovation=jj@p@jj.transpose(-1,-2)+r
        d2=(error*torch.linalg.solve(innovation,error[...,None])[...,0]).sum(-1)
        ray=error[...,:2];ray_d2=(ray*torch.linalg.solve(innovation[...,:2,:2],ray[...,None])[...,0]).sum(-1)
        rgb=backend.tensor(raw.rgb[near]-surface.colors[pi]);affinity=torch.einsum('...i,ij,...j->...',rgb,precision,rgb)
        cost=.5*(d2+torch.linalg.slogdet(innovation).logabsdet+affinity)
        good=((d2<=config.gate(3))&(ray_d2<=config.gate(2))).cpu().numpy()
        costs=torch.where(backend.tensor(good,torch.bool),cost,torch.inf).cpu().numpy()
        tie=np.zeros(len(selected),bool)
        if k>1:
            best=np.sort(costs,axis=1)[:,:2];finite_pair=np.isfinite(best).all(1)
            tie[finite_pair]=abs(best[finite_pair,0]-best[finite_pair,1])<=np.finfo(float).eps*8*np.maximum(1,abs(best[finite_pair,0]))
        np.logical_or.at(covered,near.ravel(),good.ravel())
        np.logical_or.at(same_ties,near.ravel(),(good&~tie[:,None]).ravel())
        np.logical_or.at(owned_witness,near.ravel(),(good&~tie[:,None]&assigned[pi]).ravel())
        stats['candidate_edges']+=near.size;stats['compatible_edges']+=int(good.sum());stats['numerical_tied_points']+=int(tie.sum())
    stats['seconds']=time.monotonic()-start
    return covered,same_ties,owned_witness,stats
