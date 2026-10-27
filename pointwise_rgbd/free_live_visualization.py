"""Read committed checkpoints for visual debugging; never modify estimation."""
from pathlib import Path
import fcntl
import hashlib
import json
import numpy as np
from .surface_supernodes import Partition
from .free_visualization import export


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def run_status(root):
    for name in ('result.json','failure.json'):
        result=read_json(root/name)
        if result:return {'status':result['status'],'complete':result.get('complete',False)}
    if (root/'run.lock').exists():
        with (root/'run.lock').open('rb') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return {'status':'RUNNING','complete':False}
            finally:
                try:fcntl.flock(stream,fcntl.LOCK_UN)
                except OSError:pass
    return {'status':'NO_ACTIVE_RUN_LOCK','complete':False,'note':'Showing saved checkpoints; no completion or failure record exists.'}


def artifacts_from_disk(root,dataset):
    root=Path(root);manifest=read_json(root/'manifest.json');args=manifest['arguments'];values={};versions=[]
    for name in ('first','last'):
        phase=(Path(args['phase1_from']) if args.get('phase1_from') else root)/name/'phase1'
        needed=[phase/'partition.npz',phase/'state.npy',phase/'point_observed.npy',phase/'run.json']
        if not all(p.exists() for p in needed):continue
        info=read_json(phase/'run.json');anchor=info['anchor_frame'];records=sorted((phase/'filter').glob('*.json'))
        available=sorted({anchor}|{int(p.stem) for p in records})
        with np.load(phase/'partition.npz') as saved:
            partition=Partition(**{key:saved[key].item() if key=='version' else saved[key].copy() for key in Partition.__dataclass_fields__})
        state=np.load(phase/'state.npy',mmap_mode='r');observed=np.load(phase/'point_observed.npy',mmap_mode='r')
        if state.shape!=(len(dataset.times),len(partition.xyz),6):raise ValueError('Visual checkpoint dimensions disagree with the selected scene')
        item={'partition':partition,'arrays':{'xyz':state[...,:3],'point_observed':observed},'phase_root':phase,'available_frames':available,'stage_diagnostics':{}}
        base=root/name;core=base/'coherence/representatives.npz'
        if core.exists():
            with np.load(core) as saved:item['coherence']={'point_support':saved['moving_support']}
        rigid=read_json(base/'rigid/result.json')
        if rigid and (base/'rigid/poses.npy').exists():
            pose=np.load(base/'rigid/poses.npy')
            if np.isfinite(pose).all():
                item['rigid']={**rigid,'poses':pose}
                canonical=base/'rigid/canonical_estimate.npy'
                if canonical.exists():item['rigid']['canonical_estimate']=np.load(canonical)
            item['stage_diagnostics']['rigid']=rigid
        for stage,path in [('rigid',base/'rigid/progress.json'),('prismatic',base/'joint/prismatic_progress.json'),('revolute',base/'joint/revolute_progress.json')]:
            record=read_json(path)
            if record:item['stage_diagnostics'][stage]=record
        joint=read_json(base/'joint/result.json')
        if joint:
            item['joint']={**joint,'selected':None}
            for kind in ('prismatic','revolute'):
                record=read_json(base/f'joint/{kind}.json')
                if record:item['stage_diagnostics'][kind]=record
        support=base/'support/endpoint_support.npz'
        if support.exists():
            with np.load(support) as saved:label=saved['label']
            item['support']={**(read_json(base/'support/result.json') or {}),'label':label}
        values[name]=item
        paths=needed+records+[p for folder in ('coherence','rigid','joint','support') for p in (base/folder).glob('*') if p.suffix in ('.json','.npy','.npz')]
        versions.extend((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths if p.exists())
    status=run_status(root);versions.append(('run_status',status))
    values['_run_status']=status;values['_revision']=hashlib.sha256(json.dumps(versions,sort_keys=True).encode()).hexdigest()
    return values


def refresh(root,dataset,config):
    root=Path(root);artifacts=artifacts_from_disk(root,dataset)
    old=read_json(root/'viewer/manifest.json')
    if old and old.get('live') is False:return False
    if old and old.get('revision')==artifacts['_revision']:return False
    if not any(not k.startswith('_') for k in artifacts):return False
    pooled=read_json(root/'bidirectional/result.json')
    export(root,dataset,config,artifacts,pooled,live=True)
    return True
