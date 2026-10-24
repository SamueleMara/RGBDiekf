"""Local independent XYZ diagnostic for provisional-group membership evidence.

This is the existing approved 6D point model used ONLY to recover raw sensor
identities, never accepted geometry, a group prior, a new point, or a part label.
All returned positions are backprojections of actual raw UVZ observations.
"""
import numpy as np
from .state6 import anchor_posterior
from .process6 import predict
from .filter_iekf import correct
from .association import associate
from .projection import backproject


def iter_independent_group_sensor_rows(partition,g,dataset,config,backend,anchor_index=0):
    if anchor_index not in (0,len(dataset.times)-1):raise ValueError('Endpoint diagnostic required')
    members=np.arange(len(partition.xyz)) if g is None else np.flatnonzero(partition.membership==g)
    state,p=anchor_posterior(partition.xyz[members],partition.source_covariance[members],config.velocity_covariance)
    order=list(range(len(dataset.times))) if anchor_index==0 else list(range(len(dataset.times)-1,-1,-1))
    for previous,k in zip(order[:-1],order[1:]):
        dt=float(dataset.times[k]-dataset.times[previous]);pred,pp=predict(backend.tensor(state),backend.tensor(p),abs(dt),config,reverse=dt<0)
        matches=[]
        for ci,camera in enumerate(dataset.cameras):
            frame=dataset.observation(k,ci)
            match,_=associate(pred.cpu().numpy(),pp.cpu().numpy(),partition.rgb[members],members,frame,config,backend,np.full(frame.depth.size,-1,np.int64))
            matches.append(match)
        state,p,_=correct(pred,pp,matches,config);state=state.cpu().numpy();p=p.cpu().numpy()
        ids=[];cameras=[];pixels=[];zs=[];Rs=[];colors=[];xyz=[];cov=[]
        for ci,match in enumerate(matches):
            local=match['point'].cpu().numpy();z=match['z'].cpu().numpy();R=match['R'].cpu().numpy();y,cy=backproject(z,dataset.cameras[ci],R)
            ids.append(members[local]);cameras.append(np.full(len(local),ci,np.int64));pixels.append(match['pixel']);zs.append(z);Rs.append(R);xyz.append(y);cov.append(cy)
            colors.append(np.asarray(dataset.observation(k,ci).rgb).reshape(-1,3)[match['pixel']])
        row={name:np.concatenate(values) for name,values in [('point_id',ids),('camera',cameras),('pixel',pixels),('z',zs),('R',Rs),('rgb',colors),('xyz',xyz),('covariance',cov)]}
        row.update(frame=k,_sequence_times=np.asarray(dataset.times),_anchor_index=anchor_index,_cameras=dataset.cameras,
            _origin='independent local 6D XYZ diagnostic; raw UVZ only, not grouped predictions or accepted map states')
        yield row


def independent_group_sensor_rows(partition,g,dataset,config,backend,anchor_index=0):
    return sorted(iter_independent_group_sensor_rows(partition,g,dataset,config,backend,anchor_index),key=lambda x:x['frame'])


def write_sensor_pool(partition,dataset,config,backend,root,anchor_index=0):
    """One streaming diagnostic for all groups, reused across partition epochs."""
    from pathlib import Path
    from tqdm.auto import tqdm
    from rgbd_common.array_store import ArrayStore
    from .store import json_write
    import hashlib,json
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(partition.xyz.tobytes()+partition.source.tobytes()).hexdigest()
    model={'canonical_sample_hash':signature,'times':np.asarray(dataset.times).tolist(),'anchor_index':anchor_index,'sensor_config':config.manifest(),'purpose':'independent point XYZ diagnostic raw identities only; no accepted point trajectories or new geometry'}
    if (root/'complete.json').exists():
        old=json.loads((root/'complete.json').read_text())
        if old!=model:raise ValueError('Independent raw sensor pool inputs changed')
        return root/'raw_sensor_pool.sqlite'
    pool=ArrayStore(root/'raw_sensor_pool.sqlite')
    for row in tqdm(iter_independent_group_sensor_rows(partition,None,dataset,config,backend,anchor_index),total=len(dataset.times)-1,desc='Independent raw sensor evidence',unit='frame'):
        pool.write(str(row['frame']),**{key:value for key,value in row.items() if isinstance(value,np.ndarray) and not key.startswith('_')})
    pool.close();json_write(root/'complete.json',model);return root/'raw_sensor_pool.sqlite'


def index_sensor_pool(pool_path,partition,root,frames):
    from io import BytesIO
    from pathlib import Path
    import sqlite3
    from rgbd_common.array_store import ArrayStore
    root=Path(root);db=sqlite3.connect(root/'evidence_by_group.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS factors (g INT,k INT,data BLOB,PRIMARY KEY(g,k))');db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
    completed=db.execute("SELECT value FROM meta WHERE key='complete_hash'").fetchone()
    if completed:
        if completed[0]!=partition.digest():raise ValueError('Evidence index partition changed')
        return db
    pool=ArrayStore(pool_path)
    for k in range(frames):
        if not pool.contains(str(k)):continue
        row=pool.read(str(k));group=partition.membership[row['point_id']];order=np.argsort(group,kind='stable');keys,begin=np.unique(group[order],return_index=True)
        with db:
            for g,lo,hi in zip(keys,begin,np.r_[begin[1:],len(order)]):
                ids=order[lo:hi];stream=BytesIO();np.savez_compressed(stream,**{key:row[key][ids] for key in row});db.execute('INSERT OR REPLACE INTO factors VALUES (?,?,?)',(int(g),k,stream.getvalue()))
    with db:
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('complete_hash',partition.digest()))
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('origin','independent global XYZ diagnostic raw identities; not grouped generated trajectories'))
    pool.close();return db


class SensorRows:
    """Reiterable disk sequence; never load a whole scene's raw factors at once."""
    def __init__(self,path,dataset,anchor_index=0,point_mask=None):
        from rgbd_common.array_store import ArrayStore
        self.pool=ArrayStore(path);self.dataset=dataset;self.anchor_index=anchor_index;self.point_mask=point_mask
        self.frames=[k for k in range(len(dataset.times)) if k!=anchor_index and self.pool.contains(str(k))]
        self.positions={k:i for i,k in enumerate(self.frames)}
    def __len__(self):return len(self.frames)
    def __getitem__(self,i):return self.by_frame(self.frames[i])
    def __iter__(self):
        for k in self.frames:yield self.by_frame(k)
    def by_frame(self,k):
        if k not in self.positions:return None
        row=self.pool.read(str(k))
        if self.point_mask is not None:
            use=self.point_mask[row['point_id']];row={key:value[use] for key,value in row.items()}
        row.update(frame=k,_sequence_times=np.asarray(self.dataset.times),_anchor_index=self.anchor_index,_cameras=self.dataset.cameras,
            _origin='independent global XYZ diagnostic raw identities; no grouped generated positions')
        return row
