#!/usr/bin/env python3
"""Free 6D point IEKF -> reference coherence -> shared rigid RGB-D -> joint/support."""
import argparse,fcntl,hashlib,json,resource,shutil,subprocess,sys,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.store import json_write
from pointwise_rgbd.group_visualization import serial
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.timing import SceneFiles,MissingAcquisitionTiming

BASELINE=('process6.py','filter_iekf.py','controller.py','discovery.py','dataset.py','observations.py','projection.py','association.py')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene','--input',dest='scene',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchor',choices=['first','last','both'],default='both')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--fps',type=float)
    parser.add_argument('--dt-seconds',type=float);parser.add_argument('--timestamps-file',type=Path);parser.add_argument('--noise-config',type=Path)
    parser.add_argument('--target-occupancy',type=int,default=64,help='Computational correction group size, never a part count')
    parser.add_argument('--representative-occupancy',type=int,default=512,help='Separate canonical graph reduction for coherence')
    parser.add_argument('--coherence-candidates',type=int,default=48,help='Explicit reference hierarchy proposal budget, not an exhaustive partition search')
    parser.add_argument('--frame-limit',type=int,help='Explicit TEST prefix only');parser.add_argument('--camera-limit',type=int,help='Explicit TEST selection only')
    parser.add_argument('--stop-after',choices=['phase1','coherence','rigid','joint','support'],default='support')
    parser.add_argument('--no-viewer',action='store_true',help='Explicitly skip display export for numerical tests')
    parser.add_argument('--phase1-from',type=Path,help='Read-only reuse of COMPLETE corrected free-point passes; never legacy group-pose histories')
    parser.add_argument('--phase1-prefix-from',type=Path,help='Copy a verified completed prefix from an interrupted closed-covariance pass into this fresh output')
    parser.add_argument('--rigid-init-from',type=Path,help='Use compatible cached rigid poses/raw correspondences as initialization; refit with the original endpoint priors')
    args=parser.parse_args(argv)
    if any(x<1 for x in (args.target_occupancy,args.representative_occupancy,args.coherence_candidates)):parser.error('Group sizes/proposal budget must be positive')
    root=args.output.expanduser().resolve();source=SceneFiles(args.scene).path.resolve()
    if root==source or source in root.parents:parser.error('Output must be outside the input scene')
    if root.exists() and any(root.iterdir()):parser.error('Preserving existing results: choose a fresh output directory')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline={f:hashlib.sha256((REPO/'pointwise_rgbd'/f).read_bytes()).hexdigest() for f in BASELINE};start=time.monotonic()
    try:
        import numpy as np,torch
        torch.set_num_threads(1);np.random.seed(0);torch.manual_seed(0)
        data=Dataset(args.scene,args.frame_limit,args.camera_limit,fps=args.fps,dt_seconds=args.dt_seconds,timestamps_file=args.timestamps_file)
        assert args.camera_limit is not None or len(data.cameras)==len(data.base.cameras), 'Production must use every available camera'
        config=load_config(args.noise_config);backend=Backend(args.device)
        if args.phase1_prefix_from:
            old=json.loads((args.phase1_prefix_from/'manifest.json').read_text())
            if old['dataset']!=serial(data.manifest()) or old['noise']!=config.manifest():raise ValueError('Prefix dataset, timing or approved noise changed')
        manifest={'contract':'RGBDiekf all-camera correction 2026-10-24: free per-point IEKF with grouped increments, stages A-G','camera_policy':'all_available','production_camera_indices':list(range(len(data.cameras))),'arguments':vars(args),'dataset':data.manifest(),'noise':config.manifest(),'baseline_sha256':baseline,
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(REPO/'pointwise_rgbd').glob('*.py'),Path(__file__)]},
            'approximations':['restricted grouped MAP increments, not unrestricted IEKF','Joseph cross-covariance retained in a closed rank-24 representation per computational block; verified QR/Cholesky solve residual bounded by the documented numerical tolerance','fixed calibration; measured nonzero canonical-point priors and pose-map cross-covariance retained by sparse Schur elimination','finite kNN correspondences conditioned, not globally marginalized','reference finite coherence hierarchy proposal budget','joint/support inference conditional on uncalibrated approved initial noise guesses'],
            'no_point_recovery':True,'no_physical_supernode_splitting':True,'no_ground_truth_estimation':True}
        json_write(root/'manifest.json',serial(manifest));json_write(root/'reports/input_assumptions_and_provenance.json',serial({'timing':data.base.timing_report,'uncertainty':config.provenance,'algorithm_assumptions':manifest['approximations']}))
        for path in [*(REPO/'pointwise_rgbd').glob('*.py'),*(REPO/'pointwise_rgbd/assets').glob('*'),*(REPO/'configs').glob('*'),*(REPO/'tests').glob('test_free_*.py'),Path(__file__),REPO/'tools/run_grouped_rgbd_joint.py']:
            if path.is_file():
                target=root/'source_snapshot'/path.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        print(json.dumps({'frames':len(data.times),'cameras':len(data.cameras),'timing':{key:data.base.timing_report[key] for key in ('time_source','fps','source_kind')},'device':str(backend.device),'output':str(root)},default=str),flush=True)
        from pointwise_rgbd import free_phase1,free_coherence,shared_rigid_rgbd,joint_rgbd,joint_bidirectional,joint_support
        results={};artifacts={}
        anchors=['first','last'] if args.anchor=='both' else [args.anchor]
        for anchor in anchors:
            direction=root/anchor;phase_root=direction/'phase1'
            if args.phase1_from:
                phase_root=args.phase1_from/anchor/'phase1';partition,arrays=load_phase1(args.phase1_from,anchor,data,config,args.target_occupancy)
            else:partition,arrays=free_phase1.run(data,config,backend,phase_root,anchor,args.target_occupancy,None if args.phase1_prefix_from is None else args.phase1_prefix_from/anchor/'phase1')
            artifacts[anchor]={'partition':partition,'arrays':arrays,'phase_root':phase_root};report={'phase1':'COMPLETE_FREE_POINT_PASS'};results[anchor]=report
            if args.stop_after=='phase1':continue
            source_endpoint=0 if anchor=='first' else len(data.times)-1
            coherence=free_coherence.run(partition,arrays,data.times,direction/'coherence',args.representative_occupancy,args.coherence_candidates,source_endpoint)
            artifacts[anchor]['coherence']=coherence;report['coherence']=coherence['status']
            if coherence['status']!='RESOLVED' or args.stop_after=='coherence':continue
            if args.rigid_init_from:
                import numpy as np
                src=args.rigid_init_from/anchor
                with np.load(src/'coherence/representatives.npz') as saved:
                    if not np.array_equal(saved['moving_support'],coherence['point_support']):raise ValueError('Cached rigid initialization uses a different coherence support')
                original=json.loads((src/'rigid/result.json').read_text())
                if original.get('camera_policy')!='all_available' or original.get('production_camera_indices')!=list(range(len(data.cameras))):raise ValueError('Cached rigid fit used a camera subset; rerun all-camera rigid optimization')
                original['frames']=json.loads((src/'rigid/initial_rigid_fit.json').read_text())['frames']
                original.update(poses=np.load(src/'rigid/poses.npy'),covariance=np.load(src/'rigid/covariance.npy'),support=coherence['point_support'])
                rigid=shared_rigid_rgbd.refine_map(partition,arrays,original,data,config,backend,src/'rigid',direction/'rigid')
            else:rigid=shared_rigid_rgbd.run(partition,arrays,coherence['point_support'],data,config,backend,phase_root,direction/'rigid',source_endpoint)
            artifacts[anchor]['rigid']=rigid;report['rigid']=rigid['status']
            if rigid['status']!='FIT' or args.stop_after=='rigid':continue
            # Same hard mask constraints for every joint candidate, not merely
            # the correspondences accepted by the point filters.
            feasible=make_joint_constraint(data,partition,arrays,rigid['support'],config,backend,rigid['anchor_frame'])
            joint=joint_rgbd.run(partition,rigid,data,direction/'rigid',config,direction/'joint',feasible,backend)
            artifacts[anchor]['joint']=joint;report['joint']=joint['status']
        pooled=None
        if all('joint' in artifacts.get(a,{}) for a in ('first','last')) and args.stop_after in ('joint','support'):
            checks=[make_joint_constraint(data,v['partition'],v['arrays'],v['rigid']['support'],config,backend,v['rigid']['anchor_frame']) for v in artifacts.values()]
            pooled=joint_bidirectional.run(artifacts['first']['joint'],artifacts['last']['joint'],config,root/'bidirectional',lambda m:all(check(m) for check in checks),backend)
            results['bidirectional']=pooled['status']
        if args.stop_after=='support':
            for anchor,values in artifacts.items():
                selected=pooled.get('selected') if pooled else None
                if selected is None:selected=values.get('joint',{}).get('selected')
                if selected is None:continue
                a=values.get('rigid',{}).get('anchor_frame',0 if anchor=='first' else len(data.times)-1)
                support=joint_support.run(values['partition'],data,config,backend,selected,a,root/anchor/'support')
                values.update(support=support,selected=selected);results[anchor]['support']=support['labels']
        if not args.no_viewer:
            from pointwise_rgbd.free_visualization import export
            export(root,data,config,artifacts,pooled)
            print(f'Visual debug: {root / "viewer/index.html"}',flush=True)
        unchanged=baseline=={f:hashlib.sha256((REPO/'pointwise_rgbd'/f).read_bytes()).hexdigest() for f in BASELINE}
        if not unchanged:raise AssertionError('Baseline source changed')
        complete=all('support' in v for v in artifacts.values()) and (len(artifacts)==1 or pooled is not None and pooled.get('selected') is not None)
        result={'status':'COMPLETE_CONDITIONAL_JOINT_SUPPORT' if complete else 'INCOMPLETE_OR_UNRESOLVED','complete':complete,'directions':results,'runtime_seconds':time.monotonic()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'backend':backend.memory(),'baseline_unchanged':unchanged,'full_source':len(data.times)==len(data.base.times) and len(data.cameras)==len(data.base.cameras),'assumptions':manifest['approximations']}
        json_write(root/'result.json',serial(result));write_report(root,data,result)
        print(json.dumps(serial(result),indent=2),flush=True);return 0 if complete else 3
    except BaseException as error:
        (root/'failure.log').write_text(traceback.format_exc());json_write(root/'failure.json',{'status':'FAILED','reason':str(error),'complete':False,'runtime_seconds':time.monotonic()-start})
        if isinstance(error,MissingAcquisitionTiming):
            print(f'MISSING_ACQUISITION_TIMING: no FPS was assumed. Supply --fps, --dt-seconds or --timestamps-file. Search details: {root / "failure.log"}',file=sys.stderr);return 2
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def make_joint_constraint(dataset,partition,arrays,support,config,backend,anchor):
    """Lossless foreground/occluder index shared by all optimizer trials."""
    import numpy as np
    from pointwise_rgbd.joint_rgbd import predict
    from pointwise_rgbd.foreground_index import ForegroundIndex
    ids=np.flatnonzero(support);visibility=ForegroundIndex(dataset,config)
    def check(model):
        canonical=partition.xyz[ids]
        if '_map_xyz' in model:
            keys=np.asarray(model['_map_keys']);lookup={int(key[1]):i for i,key in enumerate(keys) if key[0]==anchor}
            canonical=np.stack([model['_map_xyz'][lookup[int(i)]] for i in ids])
        for k in range(len(dataset.times)):
            if k==anchor and '_map_xyz' not in model:continue
            if not visibility.needs_check(k):continue
            x=canonical if model['type']=='STATIC' else predict(model,canonical,k,np.full(len(ids),anchor))
            if not visibility.feasible(k,x,np.asarray(arrays['covariance'][k])[ids,:3,:3]):return False
        return True
    return check


def load_phase1(source,anchor,data,config,occupancy):
    import numpy as np
    from pointwise_rgbd.surface_supernodes import Partition
    manifest=json.loads((source/'manifest.json').read_text());phase=source/anchor/'phase1'
    if manifest['dataset']!=serial(data.manifest()) or manifest['noise']!=config.manifest():raise ValueError('Cached phase1 dataset/timing/noise differs from this invocation')
    run=json.loads((phase/'run.json').read_text())
    if run['state']!='individual XYZ + physical-time velocity, no group pose state' or run['anchor']!=anchor or run['occupancy']!=occupancy:raise ValueError('Cache is not the requested corrected free-point pass')
    if 'No block assumed-density projection' not in run['covariance']:raise ValueError('Cache used discarded block covariance projection, not the corrected correlation-preserving operator')
    if run.get('camera_policy')!='all_available' or run.get('production_camera_indices')!=list(range(len(data.cameras))):raise ValueError('Cached point trajectories used a camera partition; rerun filtering with all available cameras')
    if run.get('visibility_model')!='existing pointwise measured-foreground-depth association; background never read':raise ValueError('Cache used superseded temporal visibility; rerun the full pass')
    if not (phase/'summary.json').exists():raise ValueError('Cached phase1 is incomplete')
    with np.load(phase/'partition.npz') as saved:partition=Partition(**{key:saved[key].item() if key=='version' else saved[key].copy() for key in Partition.__dataclass_fields__})
    arrays={key:np.load(phase/f'{key}.npy',mmap_mode='r') for key in ('state','covariance','point_observed')};arrays['xyz']=arrays['state'][...,:3]
    return partition,arrays


def write_report(root,data,result):
    text=['# Free-point grouped RGB-D execution',f"Status: {result['status']}",f"Frames: {len(data.times)}; cameras: {len(data.cameras)}.",f"Timing: {data.base.timing_report}",'','```json',json.dumps(serial(result),indent=2),'```']
    (root/'reports/validation.md').write_text('\n\n'.join(text)+'\n')


if __name__=='__main__':raise SystemExit(main())
