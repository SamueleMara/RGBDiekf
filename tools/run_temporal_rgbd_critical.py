#!/usr/bin/env python3
"""All-scene operator batch; no GT in inference. New output directory per call."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from estimator.temporal_noise import load_warp_noise
from tools.temporal_batch_runtime import execute_batch,persist_batch_summary,require_three_frame_full_cloud_gate

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/media/samuelemara/Extreme SSD/Samuele')
DEFAULT_DATA=BASE/'Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear'
DEFAULT_OUT=BASE/'results_11_10_26/RGBDekf'
IDS={'storage_45135':'8444,45588,49199,262894,205921','USB_100109':'67827,87290,85696','blade_103706':'14104,27016'}


def discover(root,names=None):
    if not root.is_dir():raise ValueError(f'Data root does not exist: {root}')
    scenes=[root/name for name in names] if names else sorted(p for p in root.iterdir() if p.is_dir() and (p/'rgb').is_dir() and (p/'metadata').is_dir())
    if not scenes:raise ValueError(f'No scenes in {root}')
    for p in scenes:
        if not (p/'rgb').is_dir() or not (p/'metadata').is_dir():raise ValueError(f'Missing scene rgb/ or metadata/: {p}')
    return scenes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=DEFAULT_DATA);ap.add_argument('--output-root',type=Path,default=DEFAULT_OUT,help='parent directory; each invocation creates a unique temporal_rgbd_* child')
    ap.add_argument('--mode',default='full,simplified',help='full, simplified, local-correlated, material-following, legacy, all (= full,simplified,legacy), or comma-separated list')
    ap.add_argument('--stop-after',choices=['free','coherence','rigid','joint','support'],default='free',
                    help='free = temporal validation only (default); support = all existing stages, TEMPORAL_FREE_WITH_LEGACY_RIGID')
    ap.add_argument('--frame-limit',type=int);ap.add_argument('--direction',choices=['forward','backward','both'],default='both')
    ap.add_argument('--scenes',nargs='+');ap.add_argument('--point-ids',help='override traced canonical IDs for each scene')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--fps',type=float)
    ap.add_argument('--temporal-lag-frames',type=int,default=3);ap.add_argument('--temporal-component-workspace-mib',type=float,default=512)
    ap.add_argument('--temporal-runtime-budget-seconds',type=float)
    ap.add_argument('--temporal-local-chunk-mib',type=float,default=512.,help='point-local GPU batching budget; forwarded only to local-correlated / material-following children')
    ap.add_argument('--temporal-material-resampling',choices=['none','second-difference'],default='none',help='forwarded only to material-following children')
    ap.add_argument('--temporal-material-pyramid-levels',type=int,default=0,help='forwarded only to material-following children (initializer only)')
    ap.add_argument('--temporal-gradient-guide',choices=['off','diagnostic','replace'],default='off',help='forwarded only to material-following children')
    ap.add_argument('--terminate-grace-seconds',type=float,default=2.,help='operational process-group termination grace')
    ap.add_argument('--continue-on-error',action='store_true',help='explicitly continue after failed scene; default is fail-fast')
    ap.add_argument('--temporal-output-headroom-mib',type=float,default=128.)
    ap.add_argument('--pcd-subsample-factor',type=int,default=1,help='same as train.py: 1 = every original point (production default)')
    ap.add_argument('--pcd-subsample-method',choices=['spatial-stratified','seeded-uniform'],default='spatial-stratified',help='same choices/default as train.py; forwarded to every child')
    ap.add_argument('--pcd-subsample-seed',type=int,default=0,help='same as train.py; forwarded to every child')
    ap.add_argument('--focal-controls',action='store_true',help='DIAGNOSTIC reduced clouds of exact original traced IDs; not full-cloud validation')
    ap.add_argument('--temporal-warp-noise',choices=['measured','config','ideal'],default='measured');ap.add_argument('--temporal-warp-noise-config',type=Path)
    ap.add_argument('--sensor-noise',choices=['measured','config'],default='measured');ap.add_argument('--debug',choices=['light','full'],default='light')
    ap.add_argument('--dry-run',action='store_true',help='print exact commands and input errors without starting inference or creating outputs')
    a=ap.parse_args();modes=['full','simplified','legacy'] if a.mode=='all' else a.mode.split(',')
    if not modes or any(m not in ('full','simplified','local-correlated','material-following','legacy') for m in modes):ap.error('invalid --mode')
    modes=list(dict.fromkeys(modes))
    if a.focal_controls and 'legacy' in modes:ap.error('focal controls are temporal diagnostics only')
    if a.focal_controls and a.stop_after!='free':ap.error('focal controls require --stop-after free; reduced occlusion catalogues are not complete reconstruction inputs')
    if a.pcd_subsample_factor<1:ap.error('pcd subsample factor must be >=1')
    if a.temporal_runtime_budget_seconds is not None and not 0<a.temporal_runtime_budget_seconds<float('inf'):ap.error('runtime budget must be positive finite')
    if not 0<=a.terminate_grace_seconds<float('inf'):ap.error('termination grace must be nonnegative finite')
    if not 0<=a.temporal_output_headroom_mib<float('inf'):ap.error('output headroom must be nonnegative finite')
    if a.frame_limit is not None and a.frame_limit<1:ap.error('--frame-limit must be positive')
    if a.fps is not None and (not math.isfinite(a.fps) or a.fps<=0):ap.error('--fps must be finite and positive')
    names=[name for item in (a.scenes or []) for name in item.split(',') if name]
    scenes=discover(a.data_root,names or None)
    if 'full' in modes and not a.dry_run and (a.frame_limit is None or a.frame_limit>3 or len(scenes)>2):
        try:require_three_frame_full_cloud_gate(REPO)
        except ValueError as error:ap.error(str(error))
    missing=[]
    if any(m!='legacy' for m in modes):
        for scene in scenes:
            try:load_warp_noise(scene,a.temporal_warp_noise,a.temporal_warp_noise_config)
            except (ValueError,KeyError,OSError) as err:missing.append({'scene':scene.name,'error':str(err)})
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if a.dry_run:out=a.output_root/f'temporal_rgbd_{stamp}_DRY_RUN'
    else:
        if missing:
            print(json.dumps({'status':'INPUT_MODEL_INCOMPLETE','errors':missing},indent=2),file=sys.stderr);return 2
        out=a.output_root/f'temporal_rgbd_{stamp}_{uuid.uuid4().hex[:8]}'
    commands=[]
    for scene in scenes:
        for mode in modes:
            target=out/f'{scene.name}_{mode}'
            cmd=[sys.executable,'-B',str(REPO/'train.py'),'--scene',str(scene),'--output',str(target),'--device',a.device,
                 '--direction',a.direction,'--stop-after',a.stop_after,'--pcd-subsample-factor',str(a.pcd_subsample_factor),
                 '--pcd-subsample-method',a.pcd_subsample_method,'--pcd-subsample-seed',str(a.pcd_subsample_seed),'--covariance-precision','float64',
                 '--sensor-noise',a.sensor_noise,'--rendered-mask-measurement','off','--debug',a.debug,
                 '--debug-frames','0,1,2,3,5,7,10,30,59','--no-evaluate','--no-qa']
            if mode!='legacy':
                cmd+=['--observation-model','temporal-rgbd','--temporal-mode',mode,'--temporal-lag-frames',str(a.temporal_lag_frames),
                      '--temporal-warp-noise',a.temporal_warp_noise,'--temporal-component-workspace-mib',str(a.temporal_component_workspace_mib)]
                if a.temporal_warp_noise_config:cmd+=['--temporal-warp-noise-config',str(a.temporal_warp_noise_config.resolve())]
                if a.temporal_runtime_budget_seconds:cmd+=['--temporal-runtime-budget-seconds',str(a.temporal_runtime_budget_seconds)]
                cmd+=['--temporal-output-headroom-mib',str(a.temporal_output_headroom_mib)]
                if mode in ('local-correlated','material-following'):cmd+=['--temporal-local-chunk-mib',str(a.temporal_local_chunk_mib)]
                if mode=='material-following':cmd+=['--temporal-material-resampling',a.temporal_material_resampling,'--temporal-material-pyramid-levels',str(a.temporal_material_pyramid_levels),
                                              '--temporal-gradient-guide',a.temporal_gradient_guide]
            else:cmd+=['--observation-model','legacy']
            if a.frame_limit:cmd+=['--frame-limit',str(a.frame_limit)]
            if a.fps:cmd+=['--fps',str(a.fps)]
            ids=a.point_ids if a.point_ids is not None else IDS.get(scene.name)
            if a.focal_controls:
                if mode=='legacy':ap.error('focal controls are specific temporal diagnostics')
                ids=ids or '0,10000,20000,30000,40000,50000'
                cmd+=['--temporal-diagnostic-original-point-ids',ids]
            if ids:cmd+=['--debug-point-ids',ids]
            print(shlex.join(cmd),flush=True)
            record={'scene':scene.name,'mode':mode,'argv':cmd,'command':shlex.join(cmd),'output':str(target),
                    'log':f'{scene.name}_{mode}.log','status':'NOT_RUN','returncode':None}
            commands.append(record)
    if a.dry_run:
        print(json.dumps({'scenes':[p.name for p in scenes],'input_errors':missing,'executed':False},indent=2))
        return 2 if missing else 0
    if a.continue_on_error:print('WARNING: explicit --continue-on-error; failed estimation is never success',file=sys.stderr)
    fallback=Path(tempfile.mkdtemp(prefix='rgbd_temporal_batch_status_'))
    print(f'Results: {out}\nIndependent failure journal: {fallback}',flush=True)
    try:out.mkdir(parents=True)
    except OSError as error:
        commands[0].update(status='OUTPUT_INITIALIZATION_FAILURE',returncode=1,reason=str(error))
        print(f'Output creation failed; summary: {persist_batch_summary(out,fallback,commands)}',file=sys.stderr)
        return 1
    return execute_batch(commands,out,fallback,REPO,a.temporal_runtime_budget_seconds,a.terminate_grace_seconds,a.continue_on_error)

if __name__=='__main__':raise SystemExit(main())
