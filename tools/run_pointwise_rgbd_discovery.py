#!/usr/bin/env python3
"""Run the isolated 22 October pointwise estimator, preserving prior outputs."""
import argparse,hashlib,json,os,sys,time,traceback,subprocess,fcntl
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from pointwise_rgbd.config import load_config
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.controller import run_exact_pointwise_discovery
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.timing import SceneFiles


class Tee:
    def __init__(self,original,log):self.original=original;self.log=log
    def write(self,value):self.original.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.original.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.original,name)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Exact scheduled R6 pointwise initial-forward / frozen / individual bidirectional discovery')
    parser.add_argument('--scene',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--estimator',choices=['iekf'],default='iekf',help='Complete 6D IEKF; optional 6D UKF is not enabled')
    parser.add_argument('--all-cameras',action='store_true',default=True);parser.add_argument('--all-frames',action='store_true',default=True)
    parser.add_argument('--debug-numeric',action='store_true',default=True);parser.add_argument('--debug-visual',action='store_true',default=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--noise-config',type=Path)
    parser.add_argument('--association-tail-probability',type=float)
    parser.add_argument('--max-events',type=int,help='Explicit TEST limit, never a full reconstruction success')
    parser.add_argument('--frame-limit',type=int,help='Explicit TEST prefix, default all frames');parser.add_argument('--camera-limit',type=int,help='Explicit TEST camera prefix, default all cameras')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cached-solved',type=Path,help='Reuse this frozen pointwise_rgbd cache; NEVER run initial_forward')
    parser.add_argument('--coverage-audit',type=Path,help='Import the completed correspondence-only audit for this cache')
    parser.add_argument('--fps',type=float);parser.add_argument('--dt-seconds',type=float);parser.add_argument('--timestamps-file',type=Path)
    args=parser.parse_args(argv)
    for key in ('frame_limit','camera_limit'):
        if getattr(args,key) is not None and getattr(args,key)<1:parser.error(f'--{key.replace("_","-")} must be positive')
    if args.max_events is not None and args.max_events<0:parser.error('--max-events must be nonnegative')
    root=args.output.expanduser().resolve()/ 'pointwise_rgbd'
    try:source=SceneFiles(args.scene).path.resolve()
    except (ValueError,OSError) as error:parser.error(str(error))
    if root==source or source in root.parents:parser.error('Output must be outside the read-only source dataset')
    if args.cached_solved is not None and (root==args.cached_solved.resolve() or args.cached_solved.resolve() in root.parents):parser.error('Cached histories are read only; select a separate output')
    if args.coverage_audit is not None and args.cached_solved is None:parser.error('--coverage-audit requires --cached-solved')
    if args.cached_solved is not None and args.coverage_audit is None:parser.error('--cached-solved requires a completed --coverage-audit before discovery')
    if root.exists() and any(root.iterdir()) and not args.resume:parser.error('Preserving nonempty output; use a fresh output path or --resume')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('This output is in use by a live worker; it has not been interrupted')
    log=(root/'run.log').open('a',buffering=1);stdout,stderr=sys.stdout,sys.stderr
    sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
    start=time.monotonic()
    try:
        import torch,numpy as np
        torch.set_num_threads(1);np.random.seed(args.seed);torch.manual_seed(args.seed)
        backend=Backend(args.device);cached=None
        if args.cached_solved is not None:
            from pointwise_rgbd.frozen import FrozenCache
            cached=FrozenCache(args.cached_solved,root/'discovery/tracks');config=cached.config()
            if args.noise_config is not None:raise ValueError('Cached correspondence validation must retain its recorded physical noise model')
            if args.association_tail_probability is not None and args.association_tail_probability!=config.association_tail_probability:raise ValueError('Association tail differs from authorized cached 1%; no automatic retuning')
        else:config=load_config(args.noise_config,args.association_tail_probability)
        data=Dataset(args.scene,args.frame_limit,args.camera_limit,fps=args.fps,dt_seconds=args.dt_seconds,timestamps_file=args.timestamps_file)
        commit=subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
        camera_hash=hashlib.sha256(json.dumps(data.manifest()['cameras'],sort_keys=True).encode()).hexdigest()
        if cached is not None:
            if camera_hash!=cached.manifest['calibration_sha256'] or data.frame_ids!=cached.manifest['dataset']['original_frame_ids'] or not np.array_equal(data.times,cached.manifest['dataset']['timestamps_seconds']):raise ValueError('Selected data do not match cached calibration/frames/timestamps')
        manifest={'specification':'CODEX_EXACT_POINTWISE_BIDIRECTIONAL_DISCOVERY_2026-10-22.md','git_commit':commit,'source_tree':str(REPO),'calibration_sha256':camera_hash,'dataset':data.manifest(),'config':config.manifest(),'estimator':'6D Euclidean iterated EKF','seed':args.seed,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'compiler':subprocess.run(['c++','--version'],capture_output=True,text=True).stdout.splitlines()[0],'gpu':backend.memory(),'max_events_test_limit':args.max_events,'state_units':'world XYZ metres; physical velocity m/s; covariance outer([m,m,m,m/s,m/s,m/s])','covariance_semantics':'Conditional per-point Gaussian; calibrated cameras conditioned fixed; association/calibration cross-correlations not marginalized','ground_truth_used_for_estimation':False,'initial_history_bytes_estimate':sum(int(data.observation(0,c).valid.sum()) for c in range(len(data.cameras)))*len(data.times)*(42*8+1+8+1)}
        manifest['source_hashes']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*(REPO/'pointwise_rgbd').glob('*.py'),Path(__file__),(REPO/'configs/pointwise_association.json')])}
        manifest['frozen_cache']=None if cached is None else str(cached.root)
        manifest['coverage_audit']=None if args.coverage_audit is None else str(args.coverage_audit.resolve())
        manifest['initial_history_bytes_allocated_this_run']=0 if cached is not None else manifest['initial_history_bytes_estimate']
        if args.resume:
            previous=json.loads((root/'run_manifest.json').read_text())
            for key in ('calibration_sha256','dataset','config','estimator','seed','source_hashes','frozen_cache'):
                if previous[key]!=manifest[key]:raise ValueError(f'RESUME_INPUT_MODEL_CHANGED: {key}; prior results preserved')
        else:json_write(root/'run_manifest.json',manifest)
        print(json.dumps({'output':str(root),'frames':len(data.times),'cameras':len(data.cameras),'state_dimension':6,'time_source':data.manifest()['timing']['time_source'],'initial_forward':'SKIPPED_FROZEN_CACHE' if cached is not None else 'one initial forward pass','initial_history_bytes_allocated_this_run':manifest['initial_history_bytes_allocated_this_run'],'association_tail_probability':config.association_tail_probability}),flush=True)
        store,ownership=run_exact_pointwise_discovery(data,config,backend,root,args.max_events,args.resume,cached,args.coverage_audit)
        summary=json.loads((root/'final/summary.json').read_text());print(json.dumps(summary,indent=2),flush=True)
        return 0 if summary['complete_discovery'] else 3
    except BaseException as error:
        (root/'failure.log').write_text(traceback.format_exc())
        json_write(root/'failure.json',{'status':'FAILED','reason':str(error),'runtime_seconds':time.monotonic()-start,'completed_reconstruction':False})
        print(f'Pointwise RGB-D failed: {error}; files preserved in {root}',file=sys.stderr)
        return 1
    finally:
        sys.stdout,sys.stderr=stdout,stderr;log.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':raise SystemExit(main())
