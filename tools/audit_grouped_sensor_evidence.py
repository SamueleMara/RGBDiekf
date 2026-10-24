#!/usr/bin/env python3
"""Reconstruct raw sensor evidence for a cached run in a fresh output directory."""
import argparse,csv,hashlib,json,resource,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_evidence_audit import load_cached
from pointwise_rgbd.group_visualization import export_viewer
from pointwise_rgbd.store import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-candidates',action='store_true',help='Retain all posterior-pose candidate replay costs/S/IDs; NOT historical rejections')
    args=parser.parse_args();source=args.run.resolve(strict=True);out=args.output.resolve()
    if source==out or source in out.parents:parser.error('Use a fresh output outside the preserved run')
    if out.exists() and any(out.iterdir()):parser.error('Output must be fresh')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    root=source/'phase1/partition_v000';manifest=json.loads((source/'manifest.json').read_text());model=manifest['dataset']
    paths=[p for p in source.rglob('*') if p.is_file()]
    before={str(p.relative_to(source)):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    timing=model['timing'];overrides={}
    if timing['source_kind']=='explicit-user-override':
        if timing.get('fps') is None:raise ValueError('Nonuniform explicit timing requires the original full-sequence override file')
        overrides['fps']=timing['fps']
    data=Dataset(model['input'],len(model['original_frame_ids']),len(model['cameras']),**overrides)
    if data.frame_ids!=model['original_frame_ids'] or not np.array_equal(data.times,model['timestamps_seconds']):raise ValueError('Dataset timing/IDs changed')
    for c,old in zip(data.cameras,model['cameras']):
        if c.name!=old['name'] or not np.array_equal(c.world_to_camera,old['world_to_camera']) or not np.array_equal(c.intrinsics,old['intrinsics']):raise ValueError('Camera calibration changed')
    config=load_config();partition,arrays=load_cached(root)
    effective=export_viewer(out,data,partition,arrays,root,config,args.replay_candidates)
    # Scientific plots use every canonical sample and every surface edge.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9));frames=np.arange(len(data.times));allrows=[]
    for label,color in [('direct_updated','#3da5ff'),('prediction_only','#ed8c43')]:
        values=[(f['nn_to_raw_m'][label] or {}).get('p95',np.nan)*1000 for f in effective['frames']]
        axes[0,0].plot(data.times,values,'o-',label=label,color=color)
    axes[0,0].set(xlabel='Physical time (s)',ylabel='p95 nearest raw surface distance (mm)');axes[0,0].legend()
    axes[0,1].plot(data.times,[f['direct_observed']/len(partition.xyz) for f in effective['frames']],'o-')
    axes[0,1].set(xlabel='Physical time (s)',ylabel='Fraction directly updated / anchor conditioned')
    last=len(data.times)-1;info=json.loads((out/f'viewer/frames/{last:06d}.groups.json').read_text())
    with np.load(out/f'viewer/frames/{last:06d}.surface_audit.npz') as audit:distance=audit['group_distance_quantiles_m'][:,1]*1000
    center=np.asarray(info['centroid']);scatter=axes[1,0].scatter(center[:,0],center[:,1],c=distance,s=5,cmap='magma')
    axes[1,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='All groups, final-frame p95 raw NN distance');fig.colorbar(scatter,ax=axes[1,0],label='mm')
    edge=np.fromfile(out/f'viewer/frames/{last:06d}.edges.f32',dtype='<f4').reshape(-1,9)
    selected=edge[:,8]>0;axes[1,1].hist(abs(edge[selected,6])*1000,bins=50)
    axes[1,1].set(xlabel='Supported centroid edge change from anchor (mm)',ylabel='Edge count',title='Not a physical rigidity classification')
    fig.tight_layout();fig.savefig(out/'sensor_diagnostics.png',dpi=160);plt.close(fig)
    for k in frames:
        d=json.loads((out/f'viewer/frames/{k:06d}.groups.json').read_text())
        with np.load(out/f'viewer/frames/{k:06d}.surface_audit.npz') as audit:q=audit['group_distance_quantiles_m']
        for g in range(partition.count):
            ids=np.flatnonzero(partition.membership==g);allrows.append(dict(frame=int(data.frame_ids[k]),time_seconds=float(data.times[k]),group=g,size=len(ids),radius_m=partition.radius[g],distinct_points=d['distinct_points'][g],distinct_cameras=d['distinct_cameras'][g],nearest_raw_p95_m=float(q[g,1]),angular_std_max_rad=max(d['angular_std_rad'][g]),translation_std_max_m=max(d['center_translation_std_m'][g]),information_condition=d['normalized_condition'][g],status=d['status'][g],source_anchor_camera=int(partition.source[ids[0],1]),source_anchor_pixel=int(partition.source[ids[0],2])))
    with (out/'per_group_frame.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    after={str(p.relative_to(source)):(p.stat().st_size,p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
    if before!=after:raise AssertionError('Preserved input run changed during read-only audit')
    json_write(out/'audit_manifest.json',dict(source_run=str(source),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),old_results_unchanged=True,dataset=data.manifest(),model_updates=0,raw_factor_history='training/selection-conditioned; not held-out',historical_rejection_log='unavailable; posterior replay separately tagged',replay_candidates=args.replay_candidates,runtime_seconds=time.monotonic()-start,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu='not used; CPU read-only diagnosis',source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*REPO.glob('pointwise_rgbd/group*.py'),Path(__file__)]}))
    print(json.dumps({'output':str(out),'frames':len(data.times),'cameras':len(data.cameras),'old_results_unchanged':True,'runtime_seconds':time.monotonic()-start}))


if __name__=='__main__':main()
