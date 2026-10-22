"""Packed raw-source CSR with batch updates; no Python integer per factor."""
import numpy as np


class ProvenanceCSR:
    def __init__(self,offsets=None,raw_ids=None):
        offsets=np.array([0],np.int64) if offsets is None else np.array(offsets,dtype=np.int64,copy=True)
        raw_ids=np.empty(0,np.int64) if raw_ids is None else np.array(raw_ids,dtype=np.int64,copy=True)
        if offsets.ndim!=1 or raw_ids.ndim!=1 or not len(offsets) or offsets[0]!=0 or offsets[-1]!=len(raw_ids) or np.any(np.diff(offsets)<0):raise ValueError('Invalid raw-source CSR offsets')
        self.offsets=offsets;self.raw_ids=raw_ids
        self.offsets.flags.writeable=False;self.raw_ids.flags.writeable=False
    @classmethod
    def from_rows(cls,rows):
        if isinstance(rows,cls):return rows.copy()
        rows=list(rows);lengths=np.fromiter(map(len,rows),dtype=np.int64,count=len(rows))
        raw=np.concatenate(rows).astype(np.int64,copy=False) if lengths.sum() else np.empty(0,np.int64)
        return cls(np.r_[0,lengths.cumsum()],raw)
    @property
    def lengths(self):return np.diff(self.offsets)
    @property
    def nbytes(self):return self.offsets.nbytes+self.raw_ids.nbytes
    def __len__(self):return len(self.offsets)-1
    def __iter__(self):
        for i in range(len(self)):yield self[i]
    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError('Raw-source point index outside map')
        return self.raw_ids[self.offsets[index]:self.offsets[index+1]]
    def copy(self):return ProvenanceCSR(self.offsets,self.raw_ids)
    def take(self,indices):return ProvenanceCSR.from_rows(self[int(i)] for i in indices)
    def extend(self,rows):
        other=rows if isinstance(rows,ProvenanceCSR) else ProvenanceCSR.from_rows(rows)
        updated=ProvenanceCSR(np.r_[self.offsets,other.offsets[1:]+len(self.raw_ids)],np.r_[self.raw_ids,other.raw_ids])
        self.offsets,self.raw_ids=updated.offsets,updated.raw_ids
    def replace(self,rows):
        if not rows:return
        indices=np.array(sorted(rows),dtype=np.int64)
        if indices[0]<0 or indices[-1]>=len(self):raise IndexError('Raw-source replacement outside map')
        lengths=self.lengths;lengths[indices]=[len(rows[int(i)]) for i in indices]
        offsets=np.r_[0,lengths.cumsum()];raw=np.empty(offsets[-1],dtype=np.int64);begin=0
        # Unchanged adjacent rows are copied as one packed run.
        for i in indices:
            raw[offsets[begin]:offsets[i]]=self.raw_ids[self.offsets[begin]:self.offsets[i]]
            raw[offsets[i]:offsets[i+1]]=rows[int(i)];begin=i+1
        raw[offsets[begin]:]=self.raw_ids[self.offsets[begin]:]
        updated=ProvenanceCSR(offsets,raw);self.offsets,self.raw_ids=updated.offsets,updated.raw_ids
    def add(self,point,raw_ids):
        point=np.asarray(point,dtype=np.int64);raw_ids=np.asarray(raw_ids,dtype=np.int64)
        if point.shape!=raw_ids.shape or point.ndim!=1:raise ValueError('One point index per new raw source is required')
        if not len(point):return
        order=np.argsort(point,kind='stable');point=point[order];raw_ids=raw_ids[order]
        unique,start,count=np.unique(point,return_index=True,return_counts=True)
        rows={int(i):np.r_[self[int(i)],raw_ids[a:a+n]] for i,a,n in zip(unique,start,count)}
        self.replace(rows)
