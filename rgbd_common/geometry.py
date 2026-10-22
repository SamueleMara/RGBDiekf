"""Canonical point samples and raw-observation ownership. No rendered geometry."""
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
from .provenance import ProvenanceCSR


@dataclass
class PointMap:
    xyz:np.ndarray=field(default_factory=lambda:np.empty((0,3),np.float32))
    rgb:np.ndarray=field(default_factory=lambda:np.empty((0,3),np.float32))
    group:np.ndarray=field(default_factory=lambda:np.empty(0,np.int32))
    covariance:np.ndarray=field(default_factory=lambda:np.empty((0,6,6)))
    source_id:np.ndarray=field(default_factory=lambda:np.empty(0,np.int64))
    point_id:np.ndarray=field(default_factory=lambda:np.empty(0,np.int64))
    provenance:ProvenanceCSR=field(default_factory=ProvenanceCSR)
    next_point_id:int=0

    def __post_init__(self):
        if not isinstance(self.provenance,ProvenanceCSR):self.provenance=ProvenanceCSR.from_rows(self.provenance)
        if len(self.provenance)!=self.count:raise ValueError("One source CSR row per canonical point is required")
        if len(self.point_id):self.next_point_id=max(self.next_point_id,int(self.point_id.max())+1)

    @property
    def count(self):return len(self.xyz)
    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            keys=('xyz','rgb','group','covariance','source_id','point_id');values=[data[key].copy() for key in keys]
            offsets=data['provenance_offsets'];ids=data['provenance_raw_ids']
            provenance=ProvenanceCSR(offsets.copy(),ids.copy())
            next_id=int(data['next_point_id']) if 'next_point_id' in data else 0
        model=cls(*values,provenance,next_id);model.source_owners([]);return model
    @property
    def nbytes(self):return sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray))+self.provenance.nbytes
    def copy(self):
        from dataclasses import fields
        return PointMap(**{f.name:getattr(self,f.name).copy() if isinstance(getattr(self,f.name),np.ndarray) else (self.provenance.copy() if f.name=='provenance' else getattr(self,f.name)) for f in fields(self)})
    def source_owners(self,raw_ids):
        if not hasattr(self,'_registry'):
            ids=self.provenance.raw_ids
            points=np.repeat(np.arange(self.count,dtype=np.int64),self.provenance.lengths)
            order=np.argsort(ids,kind='stable');self._registry=(ids[order],points[order])
            if len(ids)>1 and np.any(self._registry[0][1:]==self._registry[0][:-1]):raise ValueError('A raw source observation is assigned to multiple canonical points')
        keys,points=self._registry;query=np.asarray(raw_ids);where=np.searchsorted(keys,query);ok=where<len(keys)
        ok[ok]&=keys[where[ok]]==query[ok];result=np.full(len(query),-1,dtype=np.int64);result[ok]=points[where[ok]]
        return result
    def take(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        return PointMap(*(getattr(self,key)[indices].copy() for key in ('xyz','rgb','group','covariance','source_id','point_id')),
                        self.provenance.take(indices),self.next_point_id)
    def append(self,xyz,rgb,group,covariance,source,provenance=None):
        n=len(xyz)
        rows=ProvenanceCSR(np.arange(n+1,dtype=np.int64),np.asarray(source,dtype=np.int64)) if provenance is None else ProvenanceCSR.from_rows(provenance)
        if len(rows)!=n or len(source)!=n:raise ValueError('One original source and CSR row per new canonical point is required')
        if len(np.unique(rows.raw_ids))!=len(rows.raw_ids) or np.any(self.source_owners(rows.raw_ids)>=0):raise ValueError('New canonical points must have unique unregistered raw sources')
        if provenance is not None and any(int(s) not in row for s,row in zip(source,rows)):raise ValueError('The original source must occur in its point provenance')
        if hasattr(self,'_registry'):del self._registry
        start=self.next_point_id;self.next_point_id+=n
        for key,value in [('xyz',np.asarray(xyz,np.float32)),('rgb',np.asarray(rgb,np.float32)),('group',np.asarray(group,np.int32)),('covariance',np.asarray(covariance)),('source_id',np.asarray(source,np.int64)),('point_id',np.arange(start,start+n,dtype=np.int64))]:
            setattr(self,key,np.concatenate([getattr(self,key),value]))
        self.provenance.extend(rows)
    def add_sources(self,point,raw_ids):
        raw_ids=np.asarray(raw_ids,dtype=np.int64)
        if len(np.unique(raw_ids))!=len(raw_ids) or np.any(self.source_owners(raw_ids)>=0):raise ValueError('A raw source factor may only be registered once')
        self.provenance.add(point,raw_ids)
        if hasattr(self,'_registry'):del self._registry
    def replace_sources(self,rows):
        self.provenance.replace(rows)
        if hasattr(self,'_registry'):del self._registry
        self.source_owners([])
    def save(self,path,**extra):
        offsets=self.provenance.offsets;raw=self.provenance.raw_ids
        np.savez(path,**{key:v for key,v in self.__dict__.items() if isinstance(v,np.ndarray)},next_point_id=np.array(self.next_point_id,dtype=np.int64),provenance_offsets=offsets,provenance_raw_ids=raw,
                 covariance_semantics=np.array('Coordinate-fusion initialization blocks at their last fusion, not necessarily relinearized after later pose updates. Final current-pose raw-factor covariance is exported separately in map_uncertainty. Not independent joint-map marginals.'),**extra)


class ObservationIDs:
    def __init__(self,dataset):
        self.sizes=np.array([c.width*c.height for c in dataset.cameras],dtype=np.int64)
        self.offsets=np.r_[0,np.cumsum(self.sizes)];self.frame_size=int(self.sizes.sum())
    def encode(self,k,c,pixel):return k*self.frame_size+self.offsets[c]+np.asarray(pixel,dtype=np.int64)
    def decode(self,ids):
        ids=np.asarray(ids,dtype=np.int64);k=ids//self.frame_size;within=ids%self.frame_size
        c=np.searchsorted(self.offsets[1:],within,side='right');return k,c,within-self.offsets[c]


class RawObservationOwnership:
    """Packed per-frame explained flags on disk; no persistent observation XYZ."""
    def __init__(self,dataset,directory):
        self.dataset=dataset;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.ids=ObservationIDs(dataset)
        from .array_store import ArrayStore
        self.store=ArrayStore(self.directory/'ownership.sqlite')
    def path(self,k,c):return self.directory/f'frame_{k:06d}_camera_{c:03d}.npz'
    def write(self,k,c,pixels,points):
        pixels=np.asarray(pixels,dtype=np.int64);points=np.asarray(points,dtype=np.int64)
        if len(np.unique(pixels))!=len(pixels):raise ValueError('Duplicate raw observation ownership')
        if pixels.ndim!=1 or points.shape!=pixels.shape or np.any((pixels<0)|(pixels>=self.ids.sizes[c])):raise ValueError('Ownership requires aligned in-image raw pixel IDs')
        if len(pixels) and not np.all(self.dataset.observation(k,c).valid.ravel()[pixels]):raise ValueError('Background or invalid depth cannot own a foreground tracking observation')
        flags=np.zeros(self.ids.sizes[c],dtype=bool);flags[pixels]=True
        self.store.write(f'{k}:{c}',explained=np.packbits(flags),pixel=pixels,point_id=points)
    def read(self,k,c):
        if self.store.contains(f'{k}:{c}'):
            data=self.store.read(f'{k}:{c}');return data['pixel'],data['point_id']
        path=self.path(k,c)
        if not path.exists():return np.empty(0,np.int64),np.empty(0,np.int64)
        with np.load(path) as data:return data['pixel'],data['point_id']
    def unsolved(self,k,c,frame=None):
        frame=self.dataset.observation(k,c) if frame is None else frame
        valid=frame.valid.ravel().copy();pixels,_=self.read(k,c);valid[pixels]=False
        return np.flatnonzero(valid)
