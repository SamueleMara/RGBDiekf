"""Hard nearest-integer first hit. Sort only geometry, never measured RGB-D."""
from dataclasses import dataclass
import numpy as np
import torch
import time


@dataclass
class OwnerMap:
    owner_id: torch.Tensor
    owner_depth: torch.Tensor
    counts: torch.Tensor
    ties: torch.Tensor
    uv: torch.Tensor
    z: torch.Tensor
    reason: torch.Tensor
    pixel: torch.Tensor
    selected_camera: torch.Tensor = None
    selected_point: torch.Tensor = None
    selected_pixel: torch.Tensor = None
    image_shape: tuple = None
    conflict_count: int = 0
    tie_count: int = 0
    timing: dict = None


def first_hit_pixel_owner(points, rig, frame, point_ids=None, dense=True, projection_cache=None):
    C, N = rig.count, len(points)
    H, W = frame.H, frame.W
    dev = points.device
    ids = torch.arange(N, device=dev) if point_ids is None else point_ids
    if point_ids is not None and len(torch.unique(ids)) != N: raise ValueError('point IDs must be unique')
    start=time.perf_counter();events=None
    if dev.type=='cuda':
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)];events[0].record(torch.cuda.current_stream(dev))
    pr = rig.project(points, jacobian=False) if projection_cache is None else projection_cache
    projected=time.perf_counter()
    if events:events[1].record(torch.cuda.current_stream(dev))
    uv = torch.round(torch.nan_to_num(pr.uv)).long()  # ties to even, same as numpy.rint
    inside = torch.isfinite(pr.uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    good = inside & torch.isfinite(pr.z) & (pr.z > rig.near[:, None])
    pix = uv[..., 1]*W+uv[..., 0]
    reason = torch.full((C, N), 6, dtype=torch.uint8, device=dev)
    reason[~inside] = 2
    reason[pr.z <= rig.near[:, None]] = 1
    owner = torch.full((C, H*W), -1, dtype=torch.long, device=dev) if dense else None
    depth = torch.full((C, H*W), float('inf'), dtype=points.dtype, device=dev) if dense else None
    counts = torch.zeros((C, H*W), dtype=torch.long, device=dev) if dense else None
    ties = torch.zeros_like(counts) if dense else None
    cameras=[];winners=[];pixels=[];conflicts=0;tie_count=0
    for c in range(C):
        ix = torch.where(good[c])[0]
        ix = ix[torch.argsort(ids[ix], stable=True)]
        ix = ix[torch.argsort(pr.z[c, ix], stable=True)]
        ix = ix[torch.argsort(pix[c, ix], stable=True)]
        p = pix[c, ix]
        if not len(ix): continue
        first = torch.ones(len(ix), dtype=torch.bool, device=dev)
        first[1:] = p[1:] != p[:-1]
        win = ix[first]; wp = p[first]
        cameras.append(torch.full_like(win,c));winners.append(ids[win]);pixels.append(wp)
        _,sizes=torch.unique_consecutive(p,return_counts=True)
        eq=pr.z[c,ix]==torch.repeat_interleave(pr.z[c,win],sizes)
        group=torch.repeat_interleave(torch.arange(len(win),device=dev),sizes)
        tied=torch.bincount(group[eq],minlength=len(win));conflicts+=int((sizes>1).sum());tie_count+=int((tied>1).sum())
        if dense:
            owner[c, wp], depth[c, wp] = ids[win], pr.z[c, win]
            counts[c,wp]=sizes;ties[c,wp]=tied
        reason[c, win] = 0
    empty=torch.empty(0,dtype=torch.long,device=dev)
    if events:
        events[2].record(torch.cuda.current_stream(dev));events[2].synchronize()
        timing={'projection_seconds':events[0].elapsed_time(events[1])/1000,'ownership_sort_storage_seconds':events[1].elapsed_time(events[2])/1000,'clock':'CUDA events'}
    else:timing={'projection_seconds':projected-start,'ownership_sort_storage_seconds':time.perf_counter()-projected,'clock':'CPU wall'}
    return OwnerMap(owner.reshape(C,H,W) if dense else None,depth.reshape(C,H,W) if dense else None,counts.reshape(C,H,W) if dense else None,
                    ties.reshape(C,H,W) if dense else None,pr.uv,pr.z,reason,pix,
                    torch.cat(cameras) if cameras else empty,torch.cat(winners) if winners else empty,torch.cat(pixels) if pixels else empty,(C,H,W),conflicts,tie_count,timing)


def selected_owners(owners):
    if owners.selected_camera is not None:return owners.selected_camera,owners.selected_point,owners.selected_pixel
    c,y,x=torch.where(owners.owner_id>=0)
    return c,owners.owner_id[c,y,x],y*owners.owner_id.shape[-1]+x


def incremental_first_hit_pixel_owner(points, previous, cached, rig, frame, dense=False):
    """Refresh changed projections; all competitors still participate in depth sort."""
    from estimator.projective_camera import Projection
    dirty=torch.where((points!=previous).any(1))[0]
    uv,z=cached.uv.clone(),cached.z.clone()
    if len(dirty):
        change=rig.project(points[dirty],jacobian=False)
        uv[:,dirty]=change.uv;z[:,dirty]=change.z
    projection=Projection(None,uv,z,None,rig.R[:,2,:])
    return first_hit_pixel_owner(points,rig,frame,dense=dense,projection_cache=projection)


def owners_at_gpu(owners,camera,pixel):
    c,p,pp=selected_owners(owners);_,H,W=owners.image_shape
    keys=c*(H*W)+pp;target=camera*(H*W)+pixel
    out=torch.full_like(target,-1)
    if len(keys):
        at=torch.searchsorted(keys,target);valid=at<len(keys)
        valid&=keys[at.clamp_max(len(keys)-1)]==target
        out[valid]=p[at[valid]]
    return out


def owner_at(owners,camera,pixel):
    c,p,pp=selected_owners(owners);_,H,W=owners.image_shape
    keys=c*(H*W)+pp;target=camera*(H*W)+pixel
    at=int(torch.searchsorted(keys,torch.tensor(target,device=keys.device)))
    return int(p[at]) if at<len(keys) and int(keys[at])==target else -1


def numpy_first_hit(uv, z, ids, H, W, near):
    out = np.full((len(uv), H, W), -1, np.int64)
    for c in range(len(uv)):
        p = np.rint(np.nan_to_num(uv[c])).astype(np.int64)
        ok = np.isfinite(uv[c]).all(1) & np.isfinite(z[c]) & (z[c] > near[c]) & (p[:,0]>=0) & (p[:,0]<W) & (p[:,1]>=0) & (p[:,1]<H)
        ix = np.flatnonzero(ok); flat = p[ix,1]*W+p[ix,0]
        ix = ix[np.lexsort((ids[ix], z[c,ix], flat))]
        for i in ix:
            u,v = p[i]
            if out[c,v,u] == -1: out[c,v,u] = ids[i]
    return out


def compare_owner_maps(a, b):
    ca,pa,xa=selected_owners(a);cb,pb,xb=selected_owners(b);_,H,W=a.image_shape
    ka,kb=ca*(H*W)+xa,cb*(H*W)+xb;keys=torch.unique(torch.cat((ka,kb)),sorted=True)
    def lookup(kk,pp):
        out=torch.full_like(keys,-1)
        if len(kk):
            at=torch.searchsorted(kk,keys);ok=(at<len(kk))&(kk[at.clamp_max(len(kk)-1)]==keys);out[ok]=pp[at[ok]]
        return out
    return {'switched_pixels':int((lookup(ka,pa)!=lookup(kb,pb)).sum()),'occupied_before':len(pa),'occupied_after':len(pb),
            'conflicts':b.conflict_count,'exact_depth_ties':b.tie_count}
