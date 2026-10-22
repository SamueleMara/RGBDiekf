"""File-backed initial histories and append-only new tracks; atomic raw owners."""
from pathlib import Path
from collections import OrderedDict
from io import BytesIO
import hashlib,json,sqlite3,mmap
import numpy as np
from .state6 import ObservationRef


def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');pending.replace(path)


class Ownership:
    def __init__(self,dataset,path):
        self.dataset=dataset;self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path);self.cache=OrderedDict()
        self.db.executescript('CREATE TABLE IF NOT EXISTS owners (frame INT,camera INT,data BLOB,PRIMARY KEY(frame,camera)); CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS failures (frame INT,camera INT,pixel INT,version INT,reason TEXT,PRIMARY KEY(frame,camera,pixel,version)); CREATE TABLE IF NOT EXISTS points (id INT PRIMARY KEY,path TEXT,data TEXT);')
        self.db.execute('CREATE TABLE IF NOT EXISTS reservations (frame INT,camera INT,pixel INT,point INT,PRIMARY KEY(frame,camera,pixel))')
    def reserve(self,ref,point):
        if self.get(ref.frame,ref.camera)[ref.pixel]>=0:raise ValueError('Cannot reserve an owned observation')
        with self.db:self.db.execute('INSERT INTO reservations VALUES (?,?,?,?)',(ref.frame,ref.camera,ref.pixel,int(point)))
    def release_reservations(self,point=None):
        with self.db:
            if point is None:self.db.execute('DELETE FROM reservations')
            else:self.db.execute('DELETE FROM reservations WHERE point=?',(int(point),))
    def meta(self,key,default=None):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default
    def set_meta(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value)))
    def allocate_id(self,initial_count):
        with self.db:
            point=self.meta('next_point_id',initial_count);self.set_meta('next_point_id',point+1)
        return point
    def arrays(self,k,c):
        key=(int(k),int(c))
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        row=self.db.execute('SELECT data FROM owners WHERE frame=? AND camera=?',key).fetchone()
        if row:
            with np.load(BytesIO(row[0])) as data:value=(data['owner'].copy(),data['kind'].copy())
        else:
            camera=self.dataset.cameras[c];value=(np.full(camera.width*camera.height,-1,dtype=np.int64),np.zeros(camera.width*camera.height,np.uint8))
        self.cache[key]=value
        while len(self.cache)>2*len(self.dataset.cameras):self.cache.popitem(last=False)
        return value
    def get(self,k,c):return self.arrays(k,c)[0]
    def _write(self,k,c,owner,kind):
        stream=BytesIO();np.savez_compressed(stream,owner=owner,kind=kind)
        self.db.execute('INSERT OR REPLACE INTO owners VALUES (?,?,?)',(k,c,stream.getvalue()))
        self.cache[(k,c)]=(owner,kind)
    def claim(self,claims,kind=1,point_record=None,checkpoint=None):
        """All raw claims and point registration are one SQLite transaction."""
        updates={}
        for k,c,pixels,points in claims:
            pixels=np.asarray(pixels,np.int64);points=np.asarray(points,np.int64)
            if len(np.unique(pixels))!=len(pixels):raise ValueError('Duplicate raw observation in one update')
            if not np.all(self.dataset.observation(k,c).valid.ravel()[pixels]):raise ValueError('Background/invalid-depth ownership attempt')
            key=(int(k),int(c))
            if key not in updates:updates[key]=tuple(x.copy() for x in self.arrays(k,c))
            owner,kinds=updates[key]
            if np.any((owner[pixels]>=0)&(owner[pixels]!=points)):raise ValueError('Observation already owned by a frozen point')
            owner[pixels]=points;kinds[pixels]=kind
        try:
            with self.db:
                for (k,c),(owner,kinds) in updates.items():self._write(k,c,owner,kinds)
                if point_record:
                    point,path,data=point_record
                    self.db.execute('INSERT INTO points VALUES (?,?,?)',(point,str(path),json.dumps(data)))
                    self.db.execute('DELETE FROM reservations WHERE point=?',(point,))
                    self.set_meta('model_version',self.meta('model_version',0)+1)
                if checkpoint:
                    for key,value in checkpoint.items():self.set_meta(key,value)
        except BaseException:
            for key in updates:self.cache.pop(key,None)
            raise
    def rollback_initial_after(self,last_frame):
        with self.db:self.db.execute('DELETE FROM owners WHERE frame>?',(last_frame,))
        self.cache.clear()
    def attempted(self,ref,version):
        return self.db.execute('SELECT 1 FROM failures WHERE frame=? AND camera=? AND pixel=? AND version=?',(ref.frame,ref.camera,ref.pixel,version)).fetchone() is not None
    def fail(self,ref,version,reason):
        with self.db:self.db.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?,?,?)',(ref.frame,ref.camera,ref.pixel,version,str(reason)))
    def first_unclaimed(self,version):
        for k in range(len(self.dataset.times)-1,-1,-1):
            for c in range(len(self.dataset.cameras)):
                frame=self.dataset.observation(k,c);pixels=np.flatnonzero(frame.valid.ravel()&(self.get(k,c)<0))
                for pixel in pixels:
                    ref=ObservationRef(k,c,int(pixel))
                    if not self.attempted(ref,version):return ref
        return None


class TrackStore:
    def __init__(self,directory,dataset,n0=None,initial=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.dataset=dataset
        self.new_directory=self.directory.parent.parent/'discovery/tracks';self.new_directory.mkdir(parents=True,exist_ok=True)
        metadata=self.directory/'initial_points.npz'
        if initial is not None:
            colors,sources=initial
            np.savez(metadata,colors=colors,sources=sources)
        with np.load(metadata) as data:self.colors=data['colors'];self.sources=data['sources']
        self.n0=len(self.colors);self.frozen=False
        from rgbd_common.array_store import ArrayStore
        self.measurements=ArrayStore(self.directory/'measurement_refs.sqlite')
        shapes={'state':(len(dataset.times),self.n0,6),'covariance':(len(dataset.times),self.n0,6,6),'observed':(len(dataset.times),self.n0),'nis':(len(dataset.times),self.n0),'counts':(len(dataset.times),self.n0)}
        self.arrays={}
        for name,shape in shapes.items():
            path=self.directory/f'{name}.npy';dtype=bool if name=='observed' else (np.uint8 if name=='counts' else np.float64)
            self.arrays[name]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
        self.added=[]
    def write_frame(self,k,state,p,observed,nis,counts):
        if self.frozen:raise ValueError('Frozen initial histories are immutable')
        for name,value in zip(('state','covariance','observed','nis','counts'),(state,p,observed,nis,counts)):
            self.arrays[name][k]=value;self.arrays[name].flush()
            if hasattr(self.arrays[name]._mmap,'madvise'):self.arrays[name]._mmap.madvise(mmap.MADV_DONTNEED)
        digest=hashlib.sha256(np.ascontiguousarray(state).tobytes()+np.ascontiguousarray(p).tobytes()).hexdigest()
        return digest
    def write_measurements(self,k,matches):
        if self.frozen:raise ValueError('Old measurement references are frozen')
        point=np.concatenate([np.asarray(m['point'].cpu() if hasattr(m['point'],'cpu') else m['point']) for m in matches])
        pixel=np.concatenate([np.asarray(m['pixel']) for m in matches])
        cameras=np.concatenate([np.full(len(m['pixel']),m['camera_index'],dtype=np.int64) for m in matches])
        fingerprints={}
        if hasattr(self.dataset,'base'):
            fingerprints={str(ci):self.dataset.base.observation_fingerprints[f'{k:06d}_{ci:03d}'] for ci in range(len(self.dataset.cameras))}
        self.measurements.write(str(k),point_id=point,pixel=pixel,camera=cameras,input_fingerprints=np.array(json.dumps(fingerprints)))
    def validate_completed_inputs(self,frames):
        if not hasattr(self.dataset,'base'):return
        for k in range(frames):
            prior=json.loads(str(self.measurements.read(str(k))['input_fingerprints']))
            for ci in range(len(self.dataset.cameras)):
                self.dataset.observation(k,ci)
                if prior[str(ci)]!=self.dataset.base.observation_fingerprints[f'{k:06d}_{ci:03d}']:
                    raise ValueError(f'RESUME_RAW_OBSERVATION_CHANGED: frame {k}, camera {ci}; existing results preserved')
    def freeze(self):
        self.measurements.close()
        for name,value in list(self.arrays.items()):
            value.flush();self.arrays[name]=np.load(self.directory/f'{name}.npy',mmap_mode='r')
        self.frozen=True
        self.initial_signature=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(self.directory.iterdir()) if p.is_file()]
    def frozen_signature(self):
        return tuple(self.initial_signature),tuple((r['point_id'],r['sha256']) for r in self.added)
    def assert_old_frozen(self,before):
        now=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(self.directory.iterdir()) if p.is_file()]
        if now!=list(before[0]) or any(a.flags.writeable for a in self.arrays.values()):raise AssertionError('Old initial trajectory/covariance changed')
        if tuple((r['point_id'],r['sha256']) for r in self.added[:len(before[1])])!=before[1]:raise AssertionError('Old appended trajectory index changed')
    def save_new(self,point_id,track,source,color):
        path=self.new_directory/f'point_{point_id:012d}.npz'
        if path.exists():raise FileExistsError('Point IDs are never reused')
        pending=path.with_suffix('.pending.npz')
        refs=np.array([(r['frame'],r['camera'],r['pixel']) for r in track['associations'] if r['reason'] in ('REAL_MEASUREMENT','ANCHOR_INITIALIZED_ONCE')],dtype=np.int64)
        if len(np.unique(refs,axis=0))!=len(refs):raise ValueError('Duplicate immutable raw measurement reference')
        diagnostic_keys=('associations','matching_stats','filter_reports','backward_frames','forward_frames','anchor_uses','numerical_failure')
        np.savez(pending,**{key:track[key] for key in ('state','covariance','observed','nis','counts')},source=np.array([source.frame,source.camera,source.pixel]),color=color,measurement_refs=refs,diagnostics_json=np.array(json.dumps({key:track[key] for key in diagnostic_keys})))
        pending.replace(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        return path,{'point_id':point_id,'sha256':digest,'source':[source.frame,source.camera,source.pixel],'anchor_observation_id':source.record(self.dataset),'anchor_frame':source.frame,'anchor_xyz_world':track['state'][source.frame,:3].tolist(),'color':np.asarray(color).tolist()}
    def append(self,path,record):
        if any(r['point_id']==record['point_id'] for r in self.added):raise ValueError('Duplicate persistent point identity')
        self.added.append({**record,'path':str(path)})
    def frame(self,k):
        state=[np.asarray(self.arrays['state'][k])];p=[np.asarray(self.arrays['covariance'][k])];color=[self.colors];observed=[np.asarray(self.arrays['observed'][k])];ids=[np.arange(self.n0,dtype=np.int64)]
        for record in self.added:
            with np.load(record['path']) as track:
                state.append(track['state'][k:k+1]);p.append(track['covariance'][k:k+1]);observed.append(track['observed'][k:k+1])
            color.append(np.asarray(record['color'])[None]);ids.append(np.array([record['point_id']]))
        result=tuple(np.concatenate(x) for x in (state,p,color,ids,observed))
        for value in self.arrays.values():
            if hasattr(value._mmap,'madvise'):value._mmap.madvise(mmap.MADV_DONTNEED)
        return result
    @property
    def count(self):return self.n0+len(self.added)
