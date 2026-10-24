"""Endpoint-only measured-surface graph and deterministic geodesic partition.

All raw foreground samples survive, including isolated vertices. Connectivity
is a computational hypothesis subject to independent temporal validation.
"""
from dataclasses import dataclass
import hashlib,heapq
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .observations import samples


@dataclass
class Partition:
    xyz:np.ndarray
    rgb:np.ndarray
    source:np.ndarray
    source_covariance:np.ndarray
    membership:np.ndarray
    centroids:np.ndarray
    radius:np.ndarray
    edges:np.ndarray
    adjacency:np.ndarray
    duplicate_unit:np.ndarray
    version:int=0

    @property
    def count(self):return len(self.centroids)
    def digest(self):
        h=hashlib.sha256()
        for a in (self.xyz,self.rgb,self.source,self.membership,self.edges,self.duplicate_unit):h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    def save(self,path):
        np.savez_compressed(path,**vars(self),partition_hash=np.array(self.digest()),membership_status=np.array('COMPUTATIONAL_HYPOTHESIS_NOT_PHYSICAL_PART'),canonical_uncertainty=np.array('conditioned measured endpoint; source covariance retained for provenance'))


def endpoint_samples(dataset,anchor,config):
    if anchor not in ('first','last'):raise ValueError('anchor must be first or last')
    k=0 if anchor=='first' else len(dataset.times)-1
    raws=[samples(dataset.observation(k,ci),config) for ci in range(len(dataset.cameras))]
    if not sum(len(r.pixel) for r in raws):raise ValueError('No valid foreground endpoint observations')
    xyz=np.concatenate([r.xyz for r in raws]);rgb=np.concatenate([r.rgb for r in raws]);cov=np.concatenate([r.covariance for r in raws])
    source=np.concatenate([np.c_[np.full(len(r.pixel),dataset.frame_ids[k]),np.full(len(r.pixel),ci),r.pixel] for ci,r in enumerate(raws)]).astype(np.int64)
    return k,raws,xyz,rgb,cov,source


def reciprocal_depth_support(xyz,cov,raw,config):
    camera=raw.frame.camera;q=xyz@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
    positive=np.isfinite(q).all(1)&(q[:,2]>0);z=np.ones(len(q));z[positive]=q[positive,2]
    uv=q[:,:2]/z[:,None]*camera.intrinsics[:2]+camera.intrinsics[2:]
    integer=np.floor(uv+.5).astype(np.int64);w,h=camera.width,camera.height
    good=positive&(integer[:,0]>=0)&(integer[:,0]<w)&(integer[:,1]>=0)&(integer[:,1]<h)
    pixels=np.clip(integer[:,1],0,h-1)*w+np.clip(integer[:,0],0,w-1)
    good &= raw.frame.valid.ravel()[pixels]
    observed=np.asarray(raw.frame.depth).ravel()[pixels]
    direction=camera.world_to_camera[2,:3]
    variance=np.einsum('i,nij,j->n',direction,cov,direction)+config.measurement_covariance(observed)[:,2,2]
    return good&np.isfinite(variance)&(variance>0)&((q[:,2]-observed)**2<=config.gate(1)*variance)


def surface_graph(raws,xyz,cov,config,adjacency=4):
    if adjacency not in (4,8):raise ValueError('source adjacency must be 4 or 8')
    edges=[];duplicates=[];offset=np.r_[0,np.cumsum([len(r.pixel) for r in raws])]
    for ci,raw in enumerate(raws):
        n=len(raw.pixel)
        if not n:continue
        camera=raw.frame.camera;w,h=camera.width,camera.height
        lookup=np.full(w*h,-1,np.int64);lookup[raw.pixel]=np.arange(n)
        _,near=raw.tree.query(raw.xyz,k=min(n,config.neighbors+1),workers=1)
        if near.ndim==1:near=near[:,None]
        shifts=[(1,0),(0,1)]+([(1,1),(-1,1)] if adjacency==8 else [])
        for dx,dy in shifts:
            u=raw.pixel%w+dx;v=raw.pixel//w+dy;inside=(u>=0)&(u<w)&(v>=0)&(v<h)
            a=np.flatnonzero(inside);b=lookup[v[a]*w+u[a]];use=b>=0;a=a[use];b=b[use]
            local=np.any(near[a]==b[:,None],axis=1)|np.any(near[b]==a[:,None],axis=1)
            variance=raw.R[a,2,2]+raw.R[b,2,2]
            use=local&((raw.z[a,2]-raw.z[b,2])**2<=config.gate(1)*variance)
            edges.extend(np.c_[a[use]+offset[ci],b[use]+offset[ci]].tolist())
    for ci,a in enumerate(raws):
        if not len(a.pixel):continue
        for cj in range(ci+1,len(raws)):
            b=raws[cj]
            if not len(b.pixel):continue
            _,ab=b.tree.query(a.xyz,k=1,workers=1);_,ba=a.tree.query(b.xyz,k=1,workers=1)
            i=np.flatnonzero(ba[ab]==np.arange(len(a.pixel)));j=ab[i]
            error=a.xyz[i]-b.xyz[j];sigma=a.covariance[i]+b.covariance[j]
            d2=np.einsum('ni,ni->n',error,np.linalg.solve(sigma,error[...,None])[...,0])
            good=d2<=config.gate(3)
            good &= reciprocal_depth_support(a.xyz[i],a.covariance[i],b,config)
            good &= reciprocal_depth_support(b.xyz[j],b.covariance[j],a,config)
            pair=np.c_[i[good]+offset[ci],j[good]+offset[cj]]
            edges.extend(pair.tolist());duplicates.extend(pair.tolist())
    pairs=np.unique(np.sort(np.asarray(edges,dtype=np.int64).reshape(-1,2),axis=1),axis=0)
    n=len(xyz);weight=np.linalg.norm(xyz[pairs[:,0]]-xyz[pairs[:,1]],axis=1)
    graph=coo_matrix((np.r_[weight,weight],(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(n,n)).tocsr()
    dup=np.asarray(duplicates,dtype=np.int64).reshape(-1,2)
    dgraph=coo_matrix((np.ones(2*len(dup)),(np.r_[dup[:,0],dup[:,1]],np.r_[dup[:,1],dup[:,0]])),shape=(n,n)).tocsr()
    _,units=connected_components(dgraph,directed=False)
    return pairs,graph,units


def connected_geodesic_assignment(graph,target_occupancy=64):
    if target_occupancy<1:raise ValueError('Computational occupancy must be positive')
    n=graph.shape[0];number,component=connected_components(graph,directed=False)
    distance=np.full(n,np.inf);membership=np.full(n,-1,np.int64);next_group=0
    def add(seed,gid):
        queue=[(0.,int(seed))]
        while queue:
            d,u=heapq.heappop(queue)
            if d>distance[u] or (d==distance[u] and membership[u]>=0 and membership[u]<=gid):continue
            distance[u]=d;membership[u]=gid
            for index in range(graph.indptr[u],graph.indptr[u+1]):
                v=graph.indices[index];candidate=d+graph.data[index]
                if candidate<distance[v]:heapq.heappush(queue,(candidate,int(v)))
    for c in range(number):
        members=np.flatnonzero(component==c);seed=int(members[0]);add(seed,next_group);next_group+=1
        for _ in range(1,int(np.ceil(len(members)/target_occupancy))):
            nxt=int(members[np.argmax(distance[members])])
            if distance[nxt]==0:break # Coincident measured samples retained together.
            add(nxt,next_group);next_group+=1
    if (membership<0).any():raise AssertionError('Every measured sample must be assigned')
    return membership


def build_partition(dataset,anchor,config,target_occupancy=64,adjacency=4):
    k,raws,xyz,rgb,cov,source=endpoint_samples(dataset,anchor,config)
    edges,graph,units=surface_graph(raws,xyz,cov,config,adjacency)
    membership=connected_geodesic_assignment(graph,target_occupancy)
    count=int(membership.max())+1;centroids=np.empty((count,3));radius=np.empty(count)
    for g in range(count):
        points=xyz[membership==g];centroids[g]=points.mean(0);radius[g]=np.linalg.norm(points-centroids[g],axis=1).max()
        indices=np.flatnonzero(membership==g)
        if connected_components(graph[indices][:,indices],directed=False,return_labels=False)!=1:raise AssertionError('Disconnected group assignment')
    adjacent=np.unique(np.sort(membership[edges],axis=1),axis=0);adjacent=adjacent[adjacent[:,0]!=adjacent[:,1]]
    return k,Partition(xyz,rgb,source,cov,membership,centroids,radius,edges,adjacent,units)
