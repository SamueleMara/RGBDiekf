#!/usr/bin/env python3
"""Independently check frozen bytes and every used raw foreground observation."""
import argparse,hashlib,io,json,sqlite3,sys,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.store import json_write


def unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:return {k:data[k] for k in data.files}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.run;manifest=json.loads((root/'run_manifest.json').read_text());summary=json.loads((root/'final/summary.json').read_text())
    data=Dataset(manifest['dataset']['input'],**manifest['dataset']['test_limits'])
    db=sqlite3.connect((root/'raw_observations/ownership_index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    meta={key:json.loads(value) for key,value in db.execute('SELECT key,value FROM meta')}
    if not meta.get('initial_complete'):raise ValueError('Initial pass is not complete; do not audit it as frozen')
    directory=root/'initial_forward/trajectory_store'
    factors=sqlite3.connect((directory/'measurement_refs.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    state=np.load(directory/'state.npy',mmap_mode='r');p=np.load(directory/'covariance.npy',mmap_mode='r')
    with np.load(directory/'initial_points.npz') as source:colors=source['colors'];sources=source['sources']
    for ci in range(len(data.cameras)):
        actual=np.sort(sources[sources[:,1]==ci,2]);expected=np.flatnonzero(data.observation(0,ci).valid.ravel())
        if not np.array_equal(actual,expected):raise AssertionError('Frame-zero foreground set was changed or reduced')
    new=[]
    for point,path,record in db.execute('SELECT id,path,data FROM points ORDER BY id'):
        record=json.loads(record)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=record['sha256']:raise AssertionError('Appended track bytes changed')
        with np.load(path) as saved:new.append((point,saved['measurement_refs'].copy()))
    digest_matches=[];used=0;owned=0;unmatched=0;background=0;duplicates=0
    start=time.monotonic()
    for k in tqdm(range(len(data.times)),desc='Audit frozen bytes and every raw factor',unit='frame'):
        digest=hashlib.sha256(np.ascontiguousarray(state[k]).tobytes()+np.ascontiguousarray(p[k]).tobytes()).hexdigest()
        digest_matches.append(digest==meta['initial_frame_digests'][k])
        if not digest_matches[-1]:raise AssertionError(f'Initial state/covariance bytes changed: frame {k}')
        import mmap
        for array in (state,p):
            if hasattr(array._mmap,'madvise'):array._mmap.madvise(mmap.MADV_DONTNEED)
        raw=unpack(factors.execute('SELECT value FROM arrays WHERE key=?',(str(k),)).fetchone()[0])
        for ci in range(len(data.cameras)):
            frame=data.observation(k,ci);valid=frame.valid.ravel()
            owners=unpack(db.execute('SELECT data FROM owners WHERE frame=? AND camera=?',(k,ci)).fetchone()[0])
            own=owners['owner'];kinds=owners['kind']
            background+=int(((own>=0)&~valid).sum())
            owned+=int((own>=0).sum());unmatched+=int((valid&(own<0)).sum())
            select=raw['camera']==ci;pixels=[raw['pixel'][select].astype(np.int64)];points=[raw['point_id'][select].astype(np.int64)]
            for point,refs in new:
                select=(refs[:,0]==k)&(refs[:,1]==ci);pix=refs[select,2]
                pixels.append(pix);points.append(np.full(len(pix),point,np.int64))
            pixels=np.concatenate(pixels);points=np.concatenate(points)
            background+=int((~valid[pixels]).sum());duplicates+=len(pixels)-len(np.unique(pixels))
            if not np.array_equal(own[pixels],points):raise AssertionError('Immutable raw factor ownership changed')
            if not np.isin(kinds[pixels],[1,3]).all():raise AssertionError('A measurement became mere geometric context')
            used+=len(pixels)
    if background or duplicates:raise AssertionError(f'Invalid raw use: background={background}, duplicate={duplicates}')
    if (used,owned,unmatched)!=(summary['assimilated_raw_observations'],summary['owned_raw_observations'],summary['unmatched_raw_observations']):raise AssertionError('Summary counts disagree with immutable raw factors')
    import mmap
    for array in (state,p):
        if hasattr(array._mmap,'madvise'):array._mmap.madvise(mmap.MADV_DONTNEED)
    report={'status':'PASS','frames_checked':len(data.times),'cameras_checked':len(data.cameras),'initial_points':len(colors),'appended_tracks_checked':len(new),'background_or_invalid_contributions':background,'duplicate_raw_factors':duplicates,'raw_factor_count':used,'geometric_ownership_count':owned,'unmatched_count':unmatched,'all_initial_state_covariance_digests_unchanged':all(digest_matches),'matching_frame_digests':digest_matches,'all_appended_track_sha256_verified':True,'seconds':time.monotonic()-start,'run':str(root),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    json_write(args.output,report);print(json.dumps({k:v for k,v in report.items() if k!='matching_frame_digests'}))


if __name__=='__main__':main()
