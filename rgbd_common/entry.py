"""Shared CLI; estimator implementation is injected by its independent entry."""
import argparse,json,time,traceback,sys,os
from pathlib import Path
import numpy as np
import yaml
from .config import load_config
from .dataloader import load_dataset,scan
from .timing import SceneFiles
from .diagnostics import validate_full_dataset
from .backends.torch_cuda import Backend
from .frame_cache import StreamingDataset
from .outer_loop import run_outer,write_json
from .outputs import export_result
from .foreground import HARD_MASK_MODEL_VERSION


def run(estimator_class,argv=None):
    parser=argparse.ArgumentParser(description='Independent '+estimator_class.name+' point RGB-D SLAM; all frames/cameras')
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--noise-config',type=Path);parser.add_argument('--fps',type=float);parser.add_argument('--dt-seconds',type=float);parser.add_argument('--timestamps-file',type=Path)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu']);parser.add_argument('--max-outer',type=int)
    parser.add_argument('--inner-iterations',type=int);parser.add_argument('--time-budget-seconds',type=float)
    parser.add_argument('--no-visual-debug',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--debug-artifacts',action='store_true',help='Explicitly export detailed per-frame diagnostics; default outputs are compact')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Preserve files and restart a complete sweep from the last accepted checkpoint and original priors')
    parser.add_argument('--rebase-legacy-projection-score',action='store_true',help='Explicitly migrate an unversioned batch-dependent projection checkpoint; archive its incomparable score history')
    args=parser.parse_args(argv);directory=args.output.expanduser().resolve()
    source=SceneFiles(args.input).path
    if directory==source or source in directory.parents:parser.error('Output must be outside the read-only source scene')
    if directory.exists() and any(directory.iterdir()) and not args.resume:parser.error('Output is nonempty; choose a fresh experiment directory or --resume an accepted checkpoint. Existing results are preserved.')
    if args.rebase_legacy_projection_score and not args.resume:parser.error('--rebase-legacy-projection-score requires --resume')
    if args.resume and not (directory/'checkpoint.json').is_file():parser.error('--resume requires an existing accepted checkpoint')
    if args.max_outer is not None and args.max_outer<1:parser.error('--max-outer must be positive')
    if args.inner_iterations is not None and args.inner_iterations<1:parser.error('--inner-iterations must be positive')
    directory.mkdir(parents=True,exist_ok=True);(directory/'reports').mkdir(exist_ok=True)
    for name,relative in [('TMPDIR','runtime/tmp'),('XDG_CACHE_HOME','runtime/cache'),('MPLCONFIGDIR','runtime/matplotlib')]:
        folder=directory/relative;folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
    start=time.monotonic()
    try:
        data=load_dataset(args.input,dt_seconds=args.dt_seconds,fps=args.fps,timestamps_file=args.timestamps_file)
        if directory==data.root or data.root in directory.parents:raise ValueError('Output must be outside the read-only source scene')
        if data.masks is None or any(path is None for row in data.masks for path in row):raise ValueError('MISSING_FOREGROUND_MASK: this production contract requires actual complete dataset masks')
        config=load_config(args.noise_config)
        if args.resume:
            previous=yaml.safe_load((directory/'run_config_resolved.yaml').read_text())
            if previous['pipeline']!=estimator_class.name:raise ValueError('CHECKPOINT_PIPELINE_CHANGED: cannot resume one filter with the other')
            manifest=json.loads(json.dumps(data.manifest()))
            for key in ('root','frame_ids','times_seconds','cameras_calibration','foreground_masks','observations','raw_depth_validity_paths','metadata'):
                if previous['scene'][key]!=manifest[key]:raise ValueError(f'CHECKPOINT_DATASET_CHANGED: {key}')
            for key,value in previous['provenance'].items():
                if key in config.provenance and value['initial_value']!=config.provenance[key]['initial_value']:raise ValueError(f'CHECKPOINT_NOISE_CHANGED: {key}')
            (directory/f'run_config_before_resume_{int(time.time())}.yaml').write_text(yaml.safe_dump(previous,sort_keys=False))
        if args.inner_iterations is not None:config.numerical['inner_iterations']=args.inner_iterations
        backend=Backend(args.device);audit=validate_full_dataset(data)
        write_json(directory/'reports/memory_preflight.json',backend.scene_memory_preflight(audit,data.cameras))
        if args.resume and previous['scene'].get('observation_fingerprints')!=data.manifest()['observation_fingerprints']:
            raise ValueError('CHECKPOINT_OBSERVATIONS_CHANGED: original RGB/depth/mask content differs or the old run has no verified content fingerprints; preserve it and start a fresh experiment')
        write_json(directory/'reports/dataset_audit.json',{**audit,'frames':len(data.times),'cameras':len(data.cameras),'timing':data.timing_report,'calibration':data.metadata_report,'ground_truth_used':False})
        resolved={'pipeline':estimator_class.name,'scene':data.manifest(),'numerical':config.numerical,'overrides':{'max_outer':args.max_outer,'time_budget_seconds':args.time_budget_seconds},'noise_source':str(config.source.resolve()),'provenance':config.provenance,'foreground':{'require_foreground_mask':True,'hard_visible_mask_constraint':True,'mask_interpolation':'nearest, floor(uv+0.5)','foreground_only_observations':True,'projection_occlusion':'V2 M3 deterministic self-occlusion and foreground match-vs-unassigned likelihood','mask_visibility_premise':data.metadata_report['mask_visibility_premise'],'hard_mask_model_version':HARD_MASK_MODEL_VERSION},'compute':{'device':args.device,'input_dtype':'float32','accumulator_dtype':'float64','deterministic_reductions':True,'batching':'quarter of measured free device memory, divided by documented per-item buffer bound'},'uncertainty':'documented initial guesses, not independent calibration; Gaussian covariance is a local approximation to the hard-constrained posterior','gpu':backend.memory()}
        (directory/'run_config_resolved.yaml').write_text(yaml.safe_dump(json.loads(json.dumps(resolved)),sort_keys=False))
        write_json(directory/'reports/input_assumptions_and_provenance.json',{'timing':data.timing_report,'noise_and_prior':config.provenance,'numerical_solver':config.numerical,'masked_observation_model':data.metadata_report['mask_visibility_premise'],'raw_depth_validity':data.metadata_report['raw_depth_validity'],'warped_photometric_mode':False})
        (directory/'reports/dataset_audit.md').write_text(f"# Dataset input integrity\n\nScene: {data.root}\n\nRead every {len(data.times)} time and {len(data.cameras)} camera. FPS: {data.timing_report.get('fps')}; source: {data.timing_report.get('time_source')}.\n\nValidated timestamps (seconds): {data.times.tolist()}\n\n```json\n{json.dumps(audit,indent=2)}\n```\n\nThis validates data/export consistency, not independent physical sensor calibration. No ground-truth motion or part annotations used.\n")
        if args.validate_only:
            write_json(directory/'validation_report.json',{'status':'INPUTS_ONLY','complete_real_cycle':False,'frames':len(data.times),'cameras':len(data.cameras)});return 0
        print(json.dumps({'frames':len(data.times),'cameras':len(data.cameras),'timing':{key:data.timing_report.get(key) for key in ('time_source','time_units','fps','interval_min_max_seconds','source_kind')},'noise_source':str(config.source),'noise_class':'documented/explicit assumptions; not independently calibrated','mask_visibility_premise':data.metadata_report['mask_visibility_premise'],'full_input_report':str(directory/'reports/input_assumptions_and_provenance.json')}),flush=True)
        dataset=StreamingDataset(data);estimator=estimator_class(config,backend)
        debug=args.debug_artifacts and not args.no_visual_debug
        result=run_outer(estimator,dataset,config,backend,directory,args.max_outer,args.time_budget_seconds,debug,args.resume,args.rebase_legacy_projection_score)
        report=export_result(result,dataset,config,backend,directory,estimator.name,debug=debug)
        print(json.dumps(report,indent=2),flush=True)
        return 0 if result.status=='CONVERGED_LOCAL_SEARCH' else 3
    except Exception as error:
        from .model_contract import UnspecifiedModelSelection
        (directory/'error.log').write_text(traceback.format_exc())
        report={'pipeline':estimator_class.name,'status':'MATHEMATICAL_SPECIFICATION_GAP' if isinstance(error,UnspecifiedModelSelection) else 'FAILED','complete_real_cycle':False,'reason':str(error),'runtime_seconds':time.monotonic()-start,'final_solved_points':None,'final_unsolved_observations':None,'estimated_groups':None,'motion_estimates_available':False}
        if 'data' in locals():report.update(frames=len(data.times),cameras=len(data.cameras),timing=data.timing_report)
        if 'audit' in locals():report['input_audit']=audit
        write_json(directory/'validation_report.json',report)
        print(f'{estimator_class.name} FAILED: {error}; see {directory / "error.log"}',file=sys.stderr);return 1
