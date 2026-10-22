"""Bounded RAM, immutable raw RGB/depth cache on the requested results disk."""
from pathlib import Path
import hashlib,json
import numpy as np
from .dataloader import RGBDFrame


class RGBSamples:
    """Normalize original uint8 values only AFTER sparse indexing, in float64."""
    def __init__(self,raw):self.raw=raw
    @property
    def shape(self):return self.raw.shape
    def __array__(self,dtype=None,copy=None):
        value=np.asarray(self.raw,dtype=float)/255.
        return value if dtype is None else value.astype(dtype,copy=False)
    def reshape(self,*shape):return RGBSamples(self.raw.reshape(*shape))
    def __getitem__(self,key):return np.asarray(self.raw[key],dtype=float)/255.
    def __mul__(self,value):return self.raw if np.isscalar(value) and value==255 else np.asarray(self)*value
    def __rmul__(self,value):return self.__mul__(value)


class MaskSamples:
    """Packed foreground observations, not a persistent image-sized state."""
    def __init__(self,packed,shape):self.packed=packed;self.shape=tuple(shape)
    def __array__(self,dtype=None,copy=None):
        value=np.unpackbits(self.packed)[:int(np.prod(self.shape))].reshape(self.shape).astype(bool)
        return value if dtype is None else value.astype(dtype)
    def ravel(self):return MaskSamples(self.packed,(int(np.prod(self.shape)),))
    def __getitem__(self,key):
        if len(self.shape)==1 and not isinstance(key,slice) and np.asarray(key).dtype.kind in 'iu':
            index=np.asarray(key);index=np.where(index<0,index+self.shape[0],index)
            if np.any((index<0)|(index>=self.shape[0])):raise IndexError('Mask index outside image')
        elif len(self.shape)==2 and isinstance(key,tuple) and len(key)==2 and all(not isinstance(k,slice) and np.asarray(k).dtype.kind in 'iu' for k in key):
            y,x=np.broadcast_arrays(*key);y=np.where(y<0,y+self.shape[0],y);x=np.where(x<0,x+self.shape[1],x)
            if np.any((y<0)|(y>=self.shape[0])|(x<0)|(x>=self.shape[1])):raise IndexError('Mask index outside image')
            index=y*self.shape[1]+x
        else:return np.asarray(self)[key]
        return ((self.packed[index//8]>>(7-index%8))&1).astype(bool)
    def __and__(self,value):return np.asarray(self)&value
    def __rand__(self,value):return value&np.asarray(self)
    def sum(self):return np.asarray(self).sum()


class CachedDataset:
    def __init__(self,dataset,directory):
        self.dataset=dataset;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self._background={};self._identities={}
    def __getattr__(self,key):return getattr(self.dataset,key)
    def background_total(self,k,c,config,backend):
        """Exact immutable unassigned score; independent of points/trajectories."""
        import torch
        from .observation_model import unassigned_log_probability
        frame_key=f'{k:06d}_{c:03d}'
        expected=self.dataset.observation_fingerprints.get(frame_key)
        if frame_key not in self._identities or self._identities[frame_key].get('fingerprint')!=expected:self.observation(k,c)
        parameters={key:config.sensor[key] for key in ('background_mean','background_covariance','background_no_return')}
        parameters.update(match_prior=config.match_prior,observation_domain_version=2);key=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
        if key not in self._background:
            path=self.directory/f'background_total_{key}.npy'
            if path.exists():value=np.load(path,mmap_mode='r+')
            else:
                value=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(self.times),len(self.cameras)));value[:]=np.nan
            self._background[key]=value
        values=self._background[key]
        if not np.isfinite(values[k,c]):
            frame=self.observation(k,c);bg=unassigned_log_probability(frame,config,backend)+np.log1p(-config.match_prior)
            values[k,c]=float(bg[backend.tensor(frame.valid.ravel(),torch.bool)].sum())
        return float(values[k,c])
    def observation(self,k,c):
        stem=self.directory/f'raw_{k:06d}_{c:03d}'
        rgb=stem.with_suffix('.rgb.npy');depth=stem.with_suffix('.depth.npy');mask=stem.with_suffix('.mask.npy')
        validity=stem.with_suffix('.valid.npy')
        key=f'{k:06d}_{c:03d}';original=None
        expected=self.dataset.observation_fingerprints.get(key)
        if expected is None or 'mask_sha256' not in expected:
            original=self.dataset.observation(k,c);expected=self.dataset.observation_fingerprints[key]
        identity={'scene':str(self.dataset.root),'fingerprint':expected,'observation_domain_version':2};marker=stem.with_suffix('.identity.json')
        if mask.exists():
            saved=self._identities.get(key)
            if saved is None:
                if not marker.exists():raise ValueError('UNVERIFIED_RAW_CACHE: missing source fingerprints; preserve results and use a fresh experiment directory')
                saved=json.loads(marker.read_text())
            if saved!=identity:raise ValueError(f'STALE_RAW_CACHE: RGB/depth/mask content changed at frame {k}, camera {c}; preserve the previous experiment and use a fresh output')
        if not mask.exists():
            frame=original if original is not None else self.dataset.observation(k,c)
            np.save(rgb,np.rint(frame.rgb*255).astype(np.uint8));np.save(depth,frame.depth);np.save(mask,np.packbits(frame.mask.ravel()))
            np.save(validity,np.packbits(frame.raw_valid.ravel()))
            pending=marker.with_suffix('.pending.json');pending.write_text(json.dumps(identity));pending.replace(marker)
        self._identities[key]=json.loads(json.dumps(identity))
        camera=self.cameras[c]
        return RGBDFrame(RGBSamples(np.load(rgb,mmap_mode='r')),np.load(depth,mmap_mode='r'),
                         MaskSamples(np.load(mask,mmap_mode='r'),(camera.height,camera.width)),camera,k,c,
                         MaskSamples(np.load(validity,mmap_mode='r'),(camera.height,camera.width)),
                         self.dataset.metadata_report['mask_visibility_premise']['object_only_no_external_occluders'])


class StreamingDataset:
    """Reuse one decoded observation in RAM; never replicate source images."""
    def __init__(self,dataset):
        self.dataset=dataset;self.current=None;self.frame=None;self._background={}
    def __getattr__(self,key):return getattr(self.dataset,key)
    def observation(self,k,c):
        if self.current!=(k,c):
            self.frame=self.dataset.observation(k,c);self.current=(k,c)
        return self.frame
    def background_total(self,k,c,config,backend):
        from .observation_model import unassigned_log_probability
        import torch
        parameters={key:config.sensor[key] for key in ('background_mean','background_covariance','background_no_return')}
        parameters['match_prior']=config.match_prior
        key=(k,c,json.dumps(parameters,sort_keys=True))
        if key not in self._background:
            frame=self.observation(k,c);raw=np.flatnonzero(frame.valid.ravel())
            value=unassigned_log_probability(frame,config,backend,backend.tensor(raw,torch.long))
            self._background[key]=float((value+np.log1p(-config.match_prior)).sum())
        return self._background[key]
