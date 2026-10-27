#!/usr/bin/env python3
"""Image-based RGB-D IEKF pipeline (RigEuler3DGS 481b2cf2 ported to coloured point clouds;
CLAUDE_RIGEULER_RGBD_RIGOROUS_IMPLEMENTATION_CONTRACT.md).

predict [XYZ,v],P- -> render coloured cloud into ALL cameras -> grouped trust-region IEKF (pinned solver) with the
exact restricted-Joseph covariance -> information-supported RigEuler coherence -> one temporal rigid SE(3)+twist
IEKF (pinned small-state solver) -> pinned 1-DOF reduction -> prequential (q, qdot) image IEKF and predictive
joint decision -> forward/backward diagnostics -> global-label support.

measurement_model: rendered_rgbd_point_cloud_v2. No association anywhere; older caches are rejected.
"""
import argparse, fcntl, hashlib, json, os, resource, shutil, subprocess, sys, time, traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # shared GPU: limit fragmentation
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.store import json_write
from pointwise_rgbd.group_visualization import serial
from pointwise_rgbd.image_phase1 import MEASUREMENT_MODEL
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.timing import SceneFiles, MissingAcquisitionTiming

SOURCES = ['rgbd_point_renderer.py', 'image_observation.py', 'image_iekf.py', 'image_phase1.py', 'supernodes.py', 'phase1_covariance.py',
           'restricted_posterior.py', 'image_coherence.py', 'supported_coherence.py', 'rigid_image_iekf.py', 'joint_image.py',
           'predictive_tests.py', 'image_bidirectional.py', 'image_support.py', 'image_visualization.py', 'rigeuler_debug.py',
           'process6.py', 'state6.py', 'small_contractions.py', 'surface_supernodes.py', 'reference_motion.py',
           'rigeuler_ref/manifold.py', 'rigeuler_ref/joint_manifold.py', 'rigeuler_ref/rigid_trajectory_reduction.py', 'rigeuler_ref/trust_region.py',
           'rigeuler_ref/pcg.py', 'rigeuler_ref/coherence_extraction.py']


def source_hashes():
    files = [REPO/'pointwise_rgbd'/f for f in SOURCES]+[Path(__file__)]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', '--input', dest='scene', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--anchor', choices=['first', 'last', 'both'], default='both')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--fps', type=float); ap.add_argument('--dt-seconds', type=float); ap.add_argument('--timestamps-file', type=Path)
    ap.add_argument('--noise-config', type=Path)
    ap.add_argument('--posterior', default='rigeuler_block_information', choices=['rigeuler_block_information', 'restricted_joseph_marginals', 'restricted_joseph_structured'],
                    help='Phase-1 covariance (named interface): assumed-density restricted-Joseph marginals (production); exact structured '
                         '(validated on dense fixtures, numerically unbounded at blade scale); RigEuler block information (comparison)')
    ap.add_argument('--small-contractions', action='store_true',
                    help='STEP 2B-B1 opt-in: tiny batched float64 GEMMs as broadcast contractions (round-off-equal operators; default off)')
    ap.add_argument('--coherence-count', type=int, default=192, help='RigEuler --supernode-count: fixed representative node count')
    ap.add_argument('--coherence-knn', type=int, default=12)
    ap.add_argument('--coherence-folds', type=int, default=5)
    ap.add_argument('--footprint-scale', type=float, default=1.5, help='Footprint support radius in source pixels')
    ap.add_argument('--stop-after', choices=['phase1', 'coherence', 'rigid', 'joint', 'support'], default='support')
    ap.add_argument('--phase1-from', type=Path, help='Read-only reuse of a COMPLETE rendered-image Phase 1 with identical inputs/settings')
    ap.add_argument('--frame-limit', type=int, help='Explicit TEST prefix only'); ap.add_argument('--camera-limit', type=int, help='Explicit TEST selection only')
    ap.add_argument('--no-viewer', action='store_true')
    args = ap.parse_args(argv)
    root = args.output.expanduser().resolve(); source = SceneFiles(args.scene).path.resolve()
    if root == source or source in root.parents: ap.error('Output must be outside the input scene')
    if root.exists() and any(root.iterdir()): ap.error('Preserving existing results: choose a fresh output directory')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    try:
        import numpy as np, torch
        torch.set_num_threads(1); np.random.seed(0); torch.manual_seed(0)
        data = Dataset(args.scene, args.frame_limit, args.camera_limit, fps=args.fps, dt_seconds=args.dt_seconds, timestamps_file=args.timestamps_file)
        assert args.camera_limit is not None or len(data.cameras) == len(data.base.cameras), 'Production must use every available camera'
        config = load_config(args.noise_config); backend = Backend(args.device)
        from pointwise_rgbd.image_iekf import IEKFConfig
        from pointwise_rgbd.rgbd_point_renderer import RenderPolicy
        from pointwise_rgbd.image_observation import ObservationPolicy
        iekf = IEKFConfig(posterior=args.posterior, small_contractions=args.small_contractions); policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
        obs_policy = ObservationPolicy()
        from pointwise_rgbd.rigid_image_iekf import RigidConfig
        settings = {'footprint_scale': args.footprint_scale, 'renderer': policy.manifest(), 'observation': obs_policy.manifest(),
                    'iekf': iekf.manifest(), 'rigid': RigidConfig().__dict__, 'joint_policy': json.loads((REPO/'configs/image_joint_policy.json').read_text())}
        manifest = {'measurement_model': MEASUREMENT_MODEL, 'pipeline': 'RigEuler3DGS 481b2cf2 image IEKF port to colored RGB-D point clouds',
            'camera_policy': 'all_available', 'production_camera_indices': list(range(len(data.cameras))), 'arguments': vars(args),
            'dataset': data.manifest(), 'noise': config.manifest(), 'settings': settings, 'source_hashes': source_hashes(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'association_used': False,
            'approximations': ['Phase-1 correction restricted to the fixed canonical grouped tangent (pinned build_supernodes, raw G)',
                               'IRLS weights, visibility front and contributor lists frozen per linearization (Jacobian piecewise at front ties)',
                               'Phase-1 covariance: restricted-Joseph marginals of the implemented gain under an assumed block-diagonal density between frames (exact for one frame; 19-39% below the exact error covariance after 5 fixture frames; it does NOT represent the unbounded growth of intra-group deviations, which the restricted gain cannot observe and for which the exact structured recursion breaks down numerically on the blade at frame 11)',
                               'continuous visibility (C1 taper, footprint-weighted occlusion depth) and coverage-tempered Student-t: adaptations of the point renderer that keep the fixed objective continuous',
                               'declared uncalibrated sensor noise; Student-t nu = 5; coverage kappa = tr(W_rgb)/12',
                               'Phase-2/3 posteriors: Joseph form of the linear update at the accepted state (left chart)',
                               'coherence: node-frame image-support weights (supported_coherence); the pinned extractor itself resolves a component on static noise',
                               'support: Gibbs pseudo-likelihood exp(-E(z)) with a uniform prior; labels from frame-level HAC tests with BH-FDR'],
            'no_point_insertion': True, 'no_physical_supernode_splitting': True, 'no_ground_truth_estimation': True}
        json_write(root/'manifest.json', serial(manifest))
        for f in [*manifest['source_hashes']]:
            target = root/'source_snapshot'/f; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO/f, target)
        print(json.dumps({'frames': len(data.times), 'cameras': len(data.cameras), 'device': str(backend.device), 'output': str(root)}), flush=True)
        from pointwise_rgbd import image_phase1
        anchors = ['first', 'last'] if args.anchor == 'both' else [args.anchor]
        artifacts, results = {}, {}
        for anchor in anchors:
            direction = root/anchor; endpoint = 0 if anchor == 'first' else len(data.times)-1
            if args.phase1_from:
                partition, arrays, rho = load_phase1(args.phase1_from, anchor, manifest)
            else:
                partition, arrays, rho = image_phase1.run(data, config, backend, direction/'phase1', anchor, iekf, policy, obs_policy,
                                                          args.footprint_scale)
            art = artifacts[anchor] = {'partition': partition, 'arrays': arrays, 'rho': rho, 'endpoint': endpoint}
            rep = results[anchor] = {'phase1': 'COMPLETE_IMAGE_IEKF_PASS'}
            if args.stop_after == 'phase1': continue
            from pointwise_rgbd import image_coherence
            coh = art['coherence'] = image_coherence.run(partition, arrays, data.times, direction/'coherence', endpoint,
                                                       args.coherence_count, args.coherence_knn, args.coherence_folds)
            rep['coherence'] = coh['status']
            if coh['status'] != 'RESOLVED' or args.stop_after == 'coherence': continue
            from pointwise_rgbd import rigid_image_iekf
            rigid = art['rigid'] = rigid_image_iekf.run(data, config, backend, partition, arrays, rho, coh['point_support'], endpoint,
                                                        direction/'rigid', policy, obs_policy)
            rep['rigid'] = rigid['status']
            if rigid['status'] != 'FIT' or args.stop_after == 'rigid': continue
            from pointwise_rgbd import joint_image
            joint = art['joint'] = joint_image.run(data, config, backend, partition, rho, rigid, endpoint, direction/'joint', policy, obs_policy)
            rep['joint'] = joint['status']
        pooled = None
        if all('joint' in artifacts.get(a, {}) for a in ('first', 'last')) and args.stop_after in ('joint', 'support'):
            from pointwise_rgbd import image_bidirectional
            resolution = float(np.median(np.concatenate([artifacts[a]['rho'] for a in ('first', 'last')])))
            pooled = image_bidirectional.run(artifacts['first']['joint'], artifacts['last']['joint'], data, root/'bidirectional', resolution)
            results['bidirectional'] = pooled['status']
        if args.stop_after == 'support':
            from pointwise_rgbd import image_support
            for anchor, art in artifacts.items():             # each direction labels its own endpoint points with its own joint
                selected = art.get('joint', {}).get('selected')
                if selected is None: continue
                art['support'] = image_support.run(data, config, backend, art['partition'], art['rho'], art['coherence']['point_support'],
                                                   selected, art['endpoint'], root/anchor/'support', policy, obs_policy)
                results[anchor]['support'] = art['support']['counts']
        if not args.no_viewer:
            from pointwise_rgbd.image_visualization import export
            export(root, data, artifacts, pooled)
            print(f'Visual debug: {root/"viewer/index.html"}', flush=True)
            # RigEuler3DGS debug pages (pinned writers); GT layers are evaluation-only, read after inference.
            from pointwise_rgbd.rigeuler_debug import write as write_rigeuler_debug
            index, _ = write_rigeuler_debug(root, data, args.scene, 0, args.device, tuple(artifacts))
            print(f'RigEuler-style debug pages: {index}', flush=True)
        complete = args.stop_after == 'support' and all(('support' in a) or (a.get('joint', {}).get('selected') is None and 'joint' in a) for a in artifacts.values())
        result = {'status': 'COMPLETE' if complete else 'INCOMPLETE_OR_UNRESOLVED', 'complete': complete, 'directions': results,
                  'runtime_seconds': time.monotonic()-start, 'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'backend': backend.memory(), 'measurement_model': MEASUREMENT_MODEL,
                  'full_source': len(data.times) == len(data.base.times) and len(data.cameras) == len(data.base.cameras)}
        json_write(root/'result.json', serial(result))
        print(json.dumps(serial(result), indent=2), flush=True)
        return 0 if complete or args.stop_after != 'support' else 3
    except BaseException as error:
        (root/'failure.log').write_text(traceback.format_exc())
        json_write(root/'failure.json', {'status': 'FAILED', 'reason': str(error), 'complete': False, 'runtime_seconds': time.monotonic()-start})
        if isinstance(error, MissingAcquisitionTiming):
            print(f'MISSING_ACQUISITION_TIMING: supply --fps, --dt-seconds or --timestamps-file. Details: {root/"failure.log"}', file=sys.stderr); return 2
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def load_phase1(source, anchor, manifest):
    """Reuse only a complete rendered-image Phase 1 whose inputs and settings are identical."""
    import numpy as np
    from pointwise_rgbd.surface_supernodes import Partition
    old = json.loads((source/'manifest.json').read_text())
    if old.get('measurement_model') != MEASUREMENT_MODEL: raise ValueError('Rejected cache: not the rendered-image measurement model (association-based Phase 1 is incompatible)')
    for key in ('dataset', 'noise', 'settings', 'production_camera_indices'):
        if old.get(key) != manifest[key]: raise ValueError(f'Rejected cache: {key} differs')
    phase = source/anchor/'phase1'
    run = json.loads((phase/'run.json').read_text())
    if run.get('measurement_model') != MEASUREMENT_MODEL or run.get('association_used') is not False: raise ValueError('Rejected cache: measurement model')
    if not (phase/'summary.json').exists(): raise ValueError('Cached Phase 1 is incomplete')
    for f in ('rgbd_point_renderer.py', 'image_observation.py', 'image_iekf.py', 'image_phase1.py', 'supernodes.py', 'phase1_covariance.py', 'restricted_posterior.py'):
        key = f'pointwise_rgbd/{f}'
        if old['source_hashes'].get(key) != manifest['source_hashes'].get(key): raise ValueError(f'Rejected cache: {f} changed since the cached Phase 1')
    with np.load(phase/'partition.npz') as saved:
        partition = Partition(**{k: saved[k].item() if k == 'version' else saved[k].copy() for k in Partition.__dataclass_fields__})
    arrays = {k: np.load(phase/f'{k}.npy', mmap_mode='r') for k in ('state', 'covariance', 'point_observed')}
    arrays['xyz'] = arrays['state'][..., :3]
    return partition, arrays, np.load(phase/'footprint_radius.npy')


if __name__ == '__main__':
    raise SystemExit(main())
