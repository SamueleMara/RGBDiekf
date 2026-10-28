#!/usr/bin/env python3
"""LEGACY_RENDERED_SPLAT_BASELINE (superseded on 2026-10-28 by train.py, the four-pass projective RGB-D IEKF of
CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md). Kept for reproducibility of earlier results only.

RGBDekf: the rendered-image RGB-D IEKF articulation pipeline (RigEuler3DGS 481b2cf2 ported to coloured point clouds).

ONE command runs the whole pipeline, debug pages, ground-truth evaluation and page QA included:

    python train.py --scene <scene folder | dataset root> --output <fresh folder> --fps 59

A dataset root (a folder whose sub-folders are scenes) runs every scene sequentially into <output>/<scene>/, with one
log per scene and <output>/summary.{csv,json}; complete scenes are skipped, partial ones are never overwritten.

Stages, in order (docs/PIPELINE_CODE_MAP.md):
  phase1         grouped rendered-image IEKF of every endpoint point's [x, v], per anchor        estimator/phase1.py
  coherence      RigEuler held-out rigid coherence on 192 representative nodes                    estimator/motion_coherence_support.py
  rigid          one temporal SE(3)+twist IEKF of the moving core                                 estimator/rigid_part_iekf.py
  joint          1-DOF reduction, prequential (q, qdot) image IEKF, predictive joint decision     estimator/joint_iekf.py
  bidirectional  forward/backward joint diagnostics (needs --anchor both)                         joint_reduction/bidirectional.py
  support        static / moving / ambiguous label of every endpoint point                        joint_reduction/support.py
  debug          3D viewer + RigEuler-style debug pages                                           debug/
  evaluate       ground-truth scoring, in a SEPARATE process after inference                      tools/evaluate_image_iekf_gt.py
  qa             headless-Chrome check of every debug page                                        tools/qa_debug_pages.py

Running part of the pipeline:
  --stop-after STAGE                 run up to and including STAGE (later inference stages are not run)
  --from-stage STAGE --reuse RUN     take every stage before STAGE from the earlier run RUN (read-only: its folders are
                                     symlinked into the new output) after checking identical inputs, settings and the
                                     sources of those stages; then run STAGE and everything after it
  --skip debug evaluate qa ...       leave out individual stages after the inference (debug, evaluate, qa)

Ground truth is never read by any inference stage. measurement_model: rendered_rgbd_point_cloud_v2.
"""
import argparse, csv, fcntl, hashlib, json, os, resource, shutil, subprocess, sys, time, traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # shared GPU: limit fragmentation
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scene.dataset import Dataset
from scene.config import load_config
from scene.timing import SceneFiles, MissingAcquisitionTiming
from utils.io import json_write, serial

MEASUREMENT_MODEL = 'rendered_rgbd_point_cloud_v2'
INFERENCE = ['phase1', 'coherence', 'rigid', 'joint', 'bidirectional', 'support']
POST = ['debug', 'evaluate', 'qa']
STAGES = INFERENCE+POST
# Sources whose content defines each inference stage. Reusing a stage requires these (and those of earlier stages) unchanged.
STAGE_SOURCES = {
    'phase1': ['scene/dataset.py', 'scene/dataloader.py', 'scene/timing.py', 'scene/camera.py', 'scene/config.py', 'scene/noise.py', 'scene/endpoint.py',
               'estimator/point_renderer.py', 'estimator/observation_terms.py', 'estimator/sparse_iekf.py', 'estimator/phase1.py',
               'estimator/supernodes.py', 'estimator/small_contractions.py', 'estimator/covariance.py', 'estimator/restricted_posterior.py',
               'estimator/process.py', 'estimator/state.py', 'rigeuler_ref/trust_region.py', 'rigeuler_ref/pcg.py'],
    'coherence': ['estimator/motion_coherence_support.py', 'estimator/supported_coherence.py', 'rigeuler_ref/coherence_extraction.py',
                  'rigeuler_ref/motion_coherence_support.py'],
    'rigid': ['estimator/rigid_part_iekf.py', 'rigeuler_ref/manifold.py'],
    'joint': ['estimator/joint_iekf.py', 'joint_reduction/predictive_tests.py', 'rigeuler_ref/rigid_trajectory_reduction.py',
              'rigeuler_ref/joint_manifold.py', 'configs/image_joint_policy.json'],
    'bidirectional': ['joint_reduction/bidirectional.py'],
    'support': ['joint_reduction/support.py'],
}
DEBUG_SOURCES = ['debug/viewer.py', 'debug/rigeuler_pages.py', 'debug/assets/image_viewer.html', 'rigeuler_ref/motion_coherence_debug.py',
                 'rigeuler_ref/rigid_debug.py', 'rigeuler_ref/sequence_direction_debug.py', 'rigeuler_ref/debug_3d.py', 'tools/legacy_rendered_splat_train.py']


def source_hashes():
    files = sorted({f for v in STAGE_SOURCES.values() for f in v} | set(DEBUG_SOURCES))
    return {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files}


def is_scene(path):
    return (path/'rgb').is_dir() and (path/'metadata').is_dir()


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group('input / output')
    io.add_argument('--scene', '--input', dest='scene', type=Path, required=True, help='one scene folder, or a dataset root of scene folders')
    io.add_argument('--output', type=Path, required=True, help='fresh folder (a dataset root writes <output>/<scene>/)')
    io.add_argument('--fps', type=float); io.add_argument('--dt-seconds', type=float); io.add_argument('--timestamps-file', type=Path)
    io.add_argument('--noise-config', type=Path)
    io.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    st = ap.add_argument_group('stages')
    st.add_argument('--anchor', choices=['first', 'last', 'both'], default='both', help='endpoint catalogue(s); both = forward and backward')
    st.add_argument('--stop-after', choices=INFERENCE, default='support')
    st.add_argument('--from-stage', choices=INFERENCE[1:]+POST, help='with --reuse: load the stages before this one from the earlier run')
    st.add_argument('--reuse', type=Path, help='earlier run folder whose stages before --from-stage are reused read-only')
    st.add_argument('--skip', nargs='*', choices=POST, default=[], help='post-inference stages to leave out')
    md = ap.add_argument_group('model')
    md.add_argument('--posterior', default='rigeuler_block_information', choices=['rigeuler_block_information', 'restricted_joseph_marginals', 'restricted_joseph_structured'],
                    help='Phase-1 carried covariance (estimator/covariance.py); see docs/PIPELINE_REVIEW.md for what each one is')
    md.add_argument('--small-contractions', action='store_true', help='opt-in: tiny batched GEMMs as broadcast contractions (about 2x per frame; round-off-equal operators)')
    md.add_argument('--coherence-count', type=int, default=192); md.add_argument('--coherence-knn', type=int, default=12)
    md.add_argument('--coherence-folds', type=int, default=5)
    md.add_argument('--footprint-scale', type=float, default=1.5, help='footprint support radius in source pixels')
    bt = ap.add_argument_group('dataset root / tests')
    bt.add_argument('--only', nargs='*', help='dataset root: scene names to include'); bt.add_argument('--exclude', nargs='*', default=[])
    bt.add_argument('--frame-limit', type=int, help='TEST prefix only'); bt.add_argument('--camera-limit', type=int, help='TEST selection only')
    a = ap.parse_args(argv)
    if (a.from_stage is None) != (a.reuse is None): ap.error('--from-stage and --reuse go together')
    if a.from_stage in INFERENCE and INFERENCE.index(a.from_stage) > INFERENCE.index(a.stop_after): ap.error('--from-stage is after --stop-after')
    return ap, a


def main(argv=None):
    ap, a = arguments(argv)
    if not is_scene(a.scene):
        return run_dataset_root(a, argv if argv is not None else sys.argv[1:])
    return run_scene(ap, a)


# ----------------------------------------------------------------------------------------------------------------------
def run_dataset_root(a, argv):
    scenes = sorted(p for p in a.scene.iterdir() if p.is_dir() and is_scene(p))
    if not scenes: raise SystemExit(f'{a.scene}: neither a scene (rgb/ + metadata/) nor a folder of scenes')
    a.output.mkdir(parents=True, exist_ok=True)
    passthrough, dropping = [], False                                # forward every option except --scene/--output/--only/--exclude
    for x in argv:
        if x.startswith('--'): dropping = x.split('=')[0] in ('--scene', '--input', '--output', '--only', '--exclude')
        if not dropping: passthrough.append(x)
    rows = []
    for scene in scenes:
        if (a.only and scene.name not in a.only) or scene.name in a.exclude: continue
        out = a.output/scene.name; row = {'scene': scene.name, 'output': str(out)}
        if (out/'result.json').exists(): row['status'] = 'SKIPPED_EXISTING'
        elif out.exists() and any(out.iterdir()): row['status'] = 'SKIPPED_PARTIAL_OUTPUT'      # never overwrite: inspect or remove by hand
        else:
            cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--scene', str(scene), '--output', str(out), *passthrough]
            print('RUN', ' '.join(cmd), flush=True); start = time.time()
            with open(a.output/f'{scene.name}.log', 'w') as log:
                row['returncode'] = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            row['minutes'] = round((time.time()-start)/60, 1)
            row['status'] = json.loads((out/'result.json').read_text()).get('status') if (out/'result.json').exists() else 'FAILED'
        ev = out/'gt_evaluation.json'
        if ev.exists():
            e = json.loads(ev.read_text()); row['gt_type'] = e.get('gt_type')
            for d, r in e.get('directions', {}).items():
                j = r.get('joint', {}); row[f'{d}_joint'] = j.get('status')
                if e.get('gt_type') in j: row[f'{d}_axis_error_deg'] = j[e['gt_type']].get('axis_error_deg'); row[f'{d}_q_rmse'] = j[e['gt_type']].get('q_rmse')
                c = r.get('core') or {}; row[f'{d}_core_precision'] = c.get('precision_labelled'); row[f'{d}_core_recall'] = c.get('recall_labelled')
        rows.append(row); print(json.dumps(row), flush=True)
        (a.output/'summary.json').write_text(json.dumps(rows, indent=1))
        with open(a.output/'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    return 0


# ----------------------------------------------------------------------------------------------------------------------
def run_scene(ap, a):
    root = a.output.expanduser().resolve(); source = SceneFiles(a.scene).path.resolve()
    if root == source or source in root.parents: ap.error('Output must be outside the input scene')
    if root.exists() and any(root.iterdir()): ap.error('Preserving existing results: choose a fresh output folder')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start, timings = time.monotonic(), {}
    run_from = a.from_stage or 'phase1'
    wanted = lambda s: (STAGES.index(s) >= STAGES.index(run_from)) and (s in POST and s not in a.skip or s in INFERENCE and INFERENCE.index(s) <= INFERENCE.index(a.stop_after))
    try:
        import numpy as np, torch
        torch.set_num_threads(1); np.random.seed(0); torch.manual_seed(0)
        from utils.backend import Backend
        from estimator.sparse_iekf import IEKFConfig
        from estimator.point_renderer import RenderPolicy
        from estimator.observation_terms import ObservationPolicy
        from estimator.rigid_part_iekf import RigidConfig
        data = Dataset(a.scene, a.frame_limit, a.camera_limit, fps=a.fps, dt_seconds=a.dt_seconds, timestamps_file=a.timestamps_file)
        assert a.camera_limit is not None or len(data.cameras) == len(data.base.cameras), 'Production must use every available camera'
        config = load_config(a.noise_config); backend = Backend(a.device)
        iekf = IEKFConfig(posterior=a.posterior, small_contractions=a.small_contractions)
        policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance); obs_policy = ObservationPolicy()
        settings = {'footprint_scale': a.footprint_scale, 'renderer': policy.manifest(), 'observation': obs_policy.manifest(), 'iekf': iekf.manifest(),
                    'rigid': RigidConfig().__dict__, 'joint_policy': json.loads((REPO/'configs/image_joint_policy.json').read_text()),
                    'coherence': {'count': a.coherence_count, 'knn': a.coherence_knn, 'folds': a.coherence_folds}}
        manifest = {'measurement_model': MEASUREMENT_MODEL, 'pipeline': 'RigEuler3DGS 481b2cf2 image IEKF port to coloured RGB-D point clouds',
                    'entry': 'tools/legacy_rendered_splat_train.py', 'camera_policy': 'all_available', 'production_camera_indices': list(range(len(data.cameras))),
                    'arguments': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'dataset': data.manifest(), 'noise': config.manifest(),
                    'settings': settings, 'source_hashes': source_hashes(), 'git_commit': git_commit(), 'association_used': False,
                    'stages_requested': [s for s in STAGES if wanted(s)], 'no_ground_truth_estimation': True}
        anchors = ['first', 'last'] if a.anchor == 'both' else [a.anchor]
        reused = check_reuse(a, manifest, anchors, root) if a.reuse else []
        manifest['reused_stages'] = {'from': str(a.reuse.resolve()) if a.reuse else None, 'stages': reused}
        json_write(root/'manifest.json', serial(manifest))
        print(json.dumps({'frames': len(data.times), 'cameras': len(data.cameras), 'device': str(backend.device), 'output': str(root),
                          'stages': manifest['stages_requested'], 'reused': reused}), flush=True)

        def stage(name, anchor, fn):
            t0 = time.monotonic(); out = fn(); timings[f'{anchor}/{name}' if anchor else name] = time.monotonic()-t0; return out

        from estimator import phase1, motion_coherence_support, rigid_part_iekf, joint_iekf
        from joint_reduction import bidirectional, support
        artifacts, results = {}, {}
        for anchor in anchors:
            direction = root/anchor; endpoint = 0 if anchor == 'first' else len(data.times)-1
            art = artifacts[anchor] = {'endpoint': endpoint}; rep = results[anchor] = {}
            if 'phase1' in reused: art['partition'], art['arrays'], art['rho'] = load_phase1(direction/'phase1')
            else:
                art['partition'], art['arrays'], art['rho'] = stage('phase1', anchor, lambda: phase1.run(
                    data, config, backend, direction/'phase1', anchor, iekf, policy, obs_policy, a.footprint_scale))
            rep['phase1'] = 'COMPLETE_IMAGE_IEKF_PASS'
            if a.stop_after == 'phase1': continue
            if 'coherence' in reused: art['coherence'] = load_coherence(direction/'coherence')
            else:
                art['coherence'] = stage('coherence', anchor, lambda: motion_coherence_support.run(
                    art['partition'], art['arrays'], data.times, direction/'coherence', endpoint, a.coherence_count, a.coherence_knn, a.coherence_folds))
            rep['coherence'] = art['coherence']['status']
            if rep['coherence'] != 'RESOLVED' or a.stop_after == 'coherence': continue
            if 'rigid' in reused: art['rigid'] = load_rigid(direction/'rigid', art['partition'])
            else:
                art['rigid'] = stage('rigid', anchor, lambda: rigid_part_iekf.run(
                    data, config, backend, art['partition'], art['arrays'], art['rho'], art['coherence']['point_support'], endpoint,
                    direction/'rigid', policy, obs_policy))
            rep['rigid'] = art['rigid']['status']
            if rep['rigid'] != 'FIT' or a.stop_after == 'rigid': continue
            if 'joint' in reused: art['joint'] = load_joint(direction/'joint')
            else:
                art['joint'] = stage('joint', anchor, lambda: joint_iekf.run(
                    data, config, backend, art['partition'], art['rho'], art['rigid'], endpoint, direction/'joint', policy, obs_policy))
            rep['joint'] = art['joint']['status']
        pooled = None
        if all('joint' in artifacts.get(x, {}) for x in ('first', 'last')) and INFERENCE.index(a.stop_after) >= INFERENCE.index('bidirectional'):
            if 'bidirectional' in reused: pooled = json.loads((root/'bidirectional/result.json').read_text())
            else:
                resolution = float(np.median(np.concatenate([artifacts[x]['rho'] for x in ('first', 'last')])))
                pooled = stage('bidirectional', None, lambda: bidirectional.run(artifacts['first']['joint'], artifacts['last']['joint'], data, root/'bidirectional', resolution))
            results['bidirectional'] = pooled['status']
        if a.stop_after == 'support':
            for anchor, art in artifacts.items():                     # each direction labels its own endpoint points with its own joint
                selected = art.get('joint', {}).get('selected')
                if selected is None: continue
                if 'support' in reused: art['support'] = load_support(root/anchor/'support')
                else:
                    art['support'] = stage('support', anchor, lambda: support.run(
                        data, config, backend, art['partition'], art['rho'], art['coherence']['point_support'], selected, art['endpoint'],
                        root/anchor/'support', policy, obs_policy))
                results[anchor]['support'] = art['support']['counts']
        complete = a.stop_after == 'support' and all(('support' in x) or (x.get('joint', {}).get('selected') is None and 'joint' in x) for x in artifacts.values())
        post = {}
        if wanted('debug'):
            from debug.viewer import export
            from debug.rigeuler_pages import write as write_rigeuler_pages
            stage('debug', None, lambda: export(root, data, artifacts, pooled))
            index, _ = stage('debug_rigeuler_pages', None, lambda: write_rigeuler_pages(root, data, a.scene, 0, a.device, tuple(artifacts)))
            post['debug'] = {'viewer': str(root/'viewer/index.html'), 'rigeuler_pages': str(index)}
            print(f'Visual debug: {root/"viewer/index.html"}\nRigEuler-style debug pages: {index}', flush=True)
        result = {'status': 'COMPLETE' if complete else 'INCOMPLETE_OR_UNRESOLVED', 'complete': complete, 'directions': results,
                  'stage_seconds': timings, 'runtime_seconds': time.monotonic()-start, 'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'backend': backend.memory(), 'measurement_model': MEASUREMENT_MODEL, 'post': post,
                  'full_source': len(data.times) == len(data.base.times) and len(data.cameras) == len(data.base.cameras)}
        json_write(root/'result.json', serial(result))
        # Post-inference tools run in their own processes: ground truth never enters this one.
        if wanted('evaluate'):
            post['evaluate'] = tool('evaluate', [REPO/'tools/evaluate_image_iekf_gt.py', '--run', root, '--scene', a.scene, '--out', root/'gt_evaluation.json'], root)
        if wanted('qa'):
            post['qa'] = tool('qa', [REPO/'tools/qa_debug_pages.py', '--run', root], root) if shutil.which('google-chrome') or Path('/usr/bin/google-chrome').exists() \
                else {'status': 'SKIPPED', 'reason': 'google-chrome not found'}
        result['runtime_seconds'] = time.monotonic()-start
        json_write(root/'result.json', serial(result))
        print(json.dumps(serial(result), indent=2), flush=True)
        return 0 if complete or a.stop_after != 'support' else 3
    except BaseException as error:
        (root/'failure.log').write_text(traceback.format_exc())
        json_write(root/'failure.json', {'status': 'FAILED', 'reason': str(error), 'complete': False, 'stage_seconds': timings, 'runtime_seconds': time.monotonic()-start})
        if isinstance(error, MissingAcquisitionTiming):
            print(f'MISSING_ACQUISITION_TIMING: supply --fps, --dt-seconds or --timestamps-file. Details: {root/"failure.log"}', file=sys.stderr); return 2
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def tool(name, cmd, root):
    t0 = time.time()
    with open(root/f'{name}.log', 'w') as log:
        rc = subprocess.call([sys.executable, *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(f'{name}: exit {rc} ({root/f"{name}.log"})', flush=True)
    return {'status': 'PASS' if rc == 0 else 'FAILED', 'returncode': rc, 'seconds': time.time()-t0, 'log': str(root/f'{name}.log')}


def git_commit():
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    except Exception: return None


# ---- reuse of earlier stages ---------------------------------------------------------------------------------------
def check_reuse(a, manifest, anchors, root):
    """Stages before --from-stage come from --reuse, read-only. Identical inputs/settings and stage sources are required."""
    src = a.reuse.resolve()
    old = json.loads((src/'manifest.json').read_text())
    if old.get('measurement_model') != MEASUREMENT_MODEL: raise ValueError('Rejected reuse: different measurement model')
    if old.get('entry') != 'tools/legacy_rendered_splat_train.py': raise ValueError('Rejected reuse: the run was not produced by train.py (pre-restructure run); rerun it')
    norm = lambda v: json.loads(json.dumps(serial(v), default=str))       # compare as saved (JSON), not as in-memory types
    for key in ('dataset', 'noise', 'settings', 'production_camera_indices'):
        if norm(old.get(key)) != norm(manifest[key]): raise ValueError(f'Rejected reuse: {key} differs')
    cut = INFERENCE.index(a.from_stage) if a.from_stage in INFERENCE else len(INFERENCE)
    stages = [s for s in INFERENCE[:cut] if INFERENCE.index(s) <= INFERENCE.index(a.stop_after)]
    for s in stages:
        for f in STAGE_SOURCES[s]:
            if old['source_hashes'].get(f) != manifest['source_hashes'][f]: raise ValueError(f'Rejected reuse: {f} changed since stage {s} was run')
    for s in stages:
        folders = [src/'bidirectional'] if s == 'bidirectional' else [src/x/s for x in anchors]
        for f in folders:
            if not f.exists():
                if s in ('rigid', 'joint', 'bidirectional', 'support'): continue   # legitimately absent when an earlier stage did not resolve
                raise ValueError(f'Rejected reuse: {f} missing')
            if s == 'phase1' and not (f/'summary.json').exists(): raise ValueError(f'Rejected reuse: {f} is incomplete')
            link = root/f.relative_to(src); link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(f, target_is_directory=True)
    return stages


def load_phase1(phase):
    import numpy as np
    from scene.endpoint import Partition
    run = json.loads((phase/'run.json').read_text())
    if run.get('measurement_model') != MEASUREMENT_MODEL or run.get('association_used') is not False: raise ValueError('Rejected reuse: measurement model')
    with np.load(phase/'partition.npz') as saved:
        partition = Partition(**{k: saved[k].item() if k == 'version' else saved[k].copy() for k in Partition.__dataclass_fields__})
    arrays = {k: np.load(phase/f'{k}.npy', mmap_mode='r') for k in ('state', 'covariance', 'point_observed')}
    arrays['xyz'] = arrays['state'][..., :3]
    return partition, arrays, np.load(phase/'footprint_radius.npy')


def load_coherence(folder):
    import numpy as np
    out = json.loads((folder/'result.json').read_text())
    z = np.load(folder/'representatives.npz')
    out.update(point_support=z['moving_support'], representative_membership=z['membership'],
               moving_nodes=np.flatnonzero(z['labels'] == 'MOVING_RIGID'))
    return out


def load_rigid(folder, partition):
    import numpy as np
    out = json.loads((folder/'result.json').read_text())
    if out.get('status') != 'FIT': return out
    return {**out, 'root': str(folder), 'poses': np.load(folder/'poses.npy'), 'twists': np.load(folder/'twists.npy'), 'covariance': np.load(folder/'covariance.npy'),
            'members': np.load(folder/'moving_core_ids.npy'), 'canonical': partition.xyz, 'kabsch': np.load(folder/'kabsch_poses.npy')}


def load_joint(folder):
    out = json.loads((folder/'result.json').read_text())
    t = out.get('selected_type')
    out['selected'] = None if t is None else {**out['candidates'][t], 'anchor_frame': out['anchor_frame']}
    return out


def load_support(folder):
    import numpy as np
    out = json.loads((folder/'result.json').read_text())
    out['labels'] = np.load(folder/'support.npz')['labels']
    return out


if __name__ == '__main__':
    raise SystemExit(main())
