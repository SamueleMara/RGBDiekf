"""Read-only cached SOLVED histories; never invoke the initial optimizer."""
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np
from .store import TrackStore
from .config import Config


class FrozenCache(TrackStore):
    def __init__(self,root,new_directory=None):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/'run_manifest.json').read_text())
        summary=json.loads((self.root/'initial_forward/summary.json').read_text())
        if summary['status']!='INITIAL_FORWARD_FROZEN':raise ValueError('Cache must contain a completed frozen initial pass')
        self.directory=Path(json.loads((self.root/'final/point_tracks.json').read_text())['initial'])
        with np.load(self.directory/'initial_points.npz') as data:self.colors=data['colors'];self.sources=data['sources']
        self.n0=len(self.colors);self.frozen=True
        self.arrays={name:np.load(self.directory/f'{name}.npy',mmap_mode='r') for name in ('state','covariance','observed','nis','counts')}
        self.initial_signature=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(self.directory.iterdir()) if p.is_file()]
        self.added=[]
        index=json.loads((self.root/'final/point_tracks.json').read_text())
        for record in index['appended']:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Cached appended track SHA256 mismatch')
            self.append(record['path'],record)
        self.new_directory=None if new_directory is None else Path(new_directory)
        if self.new_directory is not None:self.new_directory.mkdir(parents=True,exist_ok=True)
        self.owner_db=sqlite3.connect((self.root/'raw_observations/ownership_index.sqlite').as_uri()+'?mode=ro',uri=True)
        self.frame_digests=json.loads(self.owner_db.execute("SELECT value FROM meta WHERE key='initial_frame_digests'").fetchone()[0])

    def config(self):
        values=dict(self.manifest['config'])
        for name in ('qc','qx','velocity_covariance','pixel_covariance','color_covariance'):values[name]=np.array(values[name])
        result=Config(**values)
        if result.association_tail_probability!=.01:raise ValueError('This audit requires the authorized cached 1% tail; no retuning')
        return result

    def verify_frame(self,k,state,p):
        digest=hashlib.sha256(np.ascontiguousarray(state[:self.n0]).tobytes()+np.ascontiguousarray(p[:self.n0]).tobytes()).hexdigest()
        if digest!=self.frame_digests[k]:raise AssertionError(f'Cached initial means/P changed at frame {k}')

    def save_new(self,*args,**kwargs):
        if self.new_directory is None:raise ValueError('Audit cache is read only; no point optimization output directory supplied')
        return super().save_new(*args,**kwargs)
