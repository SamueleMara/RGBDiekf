#!/usr/bin/env python3
"""Compare saved incremental coverage to all-observation/new-point pair tests.

This is a reference validation, deliberately doing the work the production
spatial prefilter avoids. It never opens or fits an old point trajectory.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.frozen import FrozenCache
from pointwise_rgbd.coverage import SurfaceIndex
from pointwise_rgbd.coverage_store import CoverageIndex
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--discovery',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    import torch
    torch.set_num_threads(1);started=time.monotonic()
    audit=json.loads((args.audit/'manifest.json').read_text());source=FrozenCache(audit['cache']);before=source.frozen_signature();config=source.config()
    manifest=json.loads((args.discovery/'run_manifest.json').read_text())
    if manifest['frozen_cache']!=str(source.root):raise ValueError('Different frozen cache')
    data=Dataset(manifest['dataset']['input'],**manifest['dataset']['test_limits']);backend=Backend(args.device)
    new_index=json.loads((args.discovery/'final/point_tracks.json').read_text())
    old_ids={record['point_id'] for record in source.added};new=[]
    for record in new_index['appended']:
        if record['point_id'] in old_ids:continue
        if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('New trajectory SHA mismatch')
        with np.load(record['path']) as track:new.append((record['point_id'],track['state'],track['covariance'],np.asarray(record['color'])))
    if not new:raise ValueError('No inserted trajectory to validate')
    coverage=CoverageIndex(args.discovery/'raw_observations/surface_coverage.sqlite');version=int(coverage.horizon()['version'])
    counts={'raw_observations':0,'views':0,'mask_disagreements':0,'geometry_mask_disagreements':0,'changed_old_appearance_witnesses':0,'newly_covered':0,'newly_geometry_covered':0}
    for k in tqdm(range(len(data.times)),desc='Reference: ALL observations against NEW trajectory only',unit='frame'):
        surfaces=[SurfaceIndex(s[k:k+1],p[k:k+1],rgb[None],np.array([point])) for point,s,p,rgb in new]
        for ci in range(len(data.cameras)):
            frame=data.observation(k,ci);actual=coverage.read(k,ci,version)
            with np.load(args.audit/'raw_classification'/f'{k:06d}_{ci:03d}.npz') as base:
                expected=base['appearance_explained'].copy();expected_g=base['geometry_explained'].copy()
                old_witness=base['appearance_point_id'];old_mask=expected.copy()
                for surface in surfaces:
                    result=surface.match(frame,config,backend)
                    if not np.array_equal(result.pixel,base['pixel']):raise ValueError('Observation identity changed')
                    expected|=result.explained;expected_g|=result.geometry_explained
                counts['mask_disagreements']+=int((expected!=actual['explained']).sum())
                counts['geometry_mask_disagreements']+=int((expected_g!=actual['geometry_explained']).sum())
                counts['changed_old_appearance_witnesses']+=int((actual['point_id'][old_mask]!=old_witness[old_mask]).sum())
                counts['newly_covered']+=int((expected&~base['appearance_explained']).sum())
                counts['newly_geometry_covered']+=int((expected_g&~base['geometry_explained']).sum())
            counts['raw_observations']+=len(actual['pixel']);counts['views']+=1
    source.assert_old_frozen(before)
    passed=not any(counts[key] for key in ('mask_disagreements','geometry_mask_disagreements','changed_old_appearance_witnesses'))
    report={'status':'PASS' if passed else 'FAIL','frames':len(data.times),'cameras':len(data.cameras),'new_point_ids':[point for point,_,_,_ in new],'model_version':version,**counts,'seconds':time.monotonic()-started,'old_trajectory_frame_reads':0,'optimizer_calls':0,'reference':'audited baseline OR every new-point/all-observation compatibility, WITHOUT production spatial prefilter','source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),REPO/'pointwise_rgbd/coverage.py',REPO/'pointwise_rgbd/incremental_coverage.py')}}
    json_write(args.discovery/'discovery/incremental_reference_validation.json',report);print(json.dumps(report,indent=2));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
