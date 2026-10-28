#!/usr/bin/env python3
"""RGBDekf: four independent directional RGB-D filters (CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md).

    F1 free forward    measured frame-0 cloud,   0 -> T,  independent 6D [x, v] IEKF per point
    B1 free backward   measured frame-T cloud,   T -> 0,  independent 6D [x, v] IEKF per point
    coherence          pinned RigEuler 192-node held-out coherence on each completed free pass (both weightings)
    F2 rigid forward   frame-0 moving core,      0 -> T,  one shared SE(3) pose + twist 12D IEKF
    B2 rigid backward  frame-T moving core,      T -> 0,  one shared SE(3) pose + twist 12D IEKF
    joint              fixed-axis reduction + decision per direction, optional (q, qdot) projective IEKF, F/B comparison
    support            measured projective consistency labels per direction
    debug              point-only measured RGB-D pages (debug/index.html) and numeric diagnostics
    evaluate           ground truth, separate process, after every inference output is written (evaluation_gt/)
    qa                 headless browser screenshots + numeric overlay checks of the debug pages

The ONLY observations are the actual masked calibrated RGB-D frames sampled at the projections of transported
original endpoint points; every correction is a Kalman correction. No splat, render, image energy, trust region,
group tangent, point correspondence or ground truth enters inference.

One command runs everything, debug included, for a scene folder or a dataset root of scene folders:
    python train.py --scene <scene | dataset root> --output <new folder>
Part of the pipeline:
    --stop-after free|coherence|rigid|joint|support      last inference stage
    --stage S --resume-from RUN                          take every stage before S from RUN (read-only, verified)
    --debug none|light|full,  --no-evaluate,  --no-qa
"""
import argparse, csv, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from utils.run_paths import stage_dir
RESULTS_ROOT = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf')   # results and debug live here, never in the code tree

INFERENCE = ['free', 'coherence', 'rigid', 'joint', 'support']
STAGE_SOURCES = {
    'free': ['scene/dataset.py', 'scene/dataloader.py', 'scene/timing.py', 'scene/camera.py', 'scene/config.py', 'scene/noise.py', 'scene/endpoint.py',
             'scene/canonical_subsample.py', 'estimator/projective_camera.py', 'estimator/projective_sensor.py', 'estimator/projective_visibility.py',
             'estimator/projective_terms.py', 'estimator/point_kalman_cuda.py', 'estimator/free_projective_run.py', 'configs/projective_rgbd.json'],
    'coherence': ['estimator/projective_coherence.py', 'estimator/motion_coherence_support.py', 'estimator/supported_coherence.py',
                  'rigeuler_ref/coherence_extraction.py', 'rigeuler_ref/motion_coherence_support.py'],
    'rigid': ['estimator/rigid_kalman_cuda.py', 'estimator/se3.py'],
    'joint': ['joint_reduction/joint_projective.py', 'joint_reduction/bidirectional.py', 'rigeuler_ref/rigid_trajectory_reduction.py',
              'rigeuler_ref/joint_manifold.py', 'rigeuler_ref/manifold.py', 'configs/image_joint_policy.json'],
    'support': ['joint_reduction/support_projective.py'],
}
DIRS = {'forward': 'forward', 'backward': 'backward'}


def is_scene(p):
    return (Path(p)/'rgb').is_dir() and (Path(p)/'metadata').is_dir()


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group('input / output')
    io.add_argument('--scene', '--input', dest='scene', type=Path, required=True, help='calibrated RGB-D scene folder, or a dataset root of scene folders')
    io.add_argument('--output', type=Path, default=None, help='new, non-existing (or empty) output folder outside the code tree; never overwritten '
                    '(default: RESULTS_ROOT/projective_<scene>_<YYYYmmdd_HHMM>)')
    io.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    io.add_argument('--fps', type=float); io.add_argument('--timestamps-file', type=Path); io.add_argument('--dt-seconds', type=float)
    io.add_argument('--noise-config', type=Path, help='declared RGB/depth/process noise (default configs/example_uncalibrated.json, UNCALIBRATED)')
    st = ap.add_argument_group('stages')
    st.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both')
    st.add_argument('--stage', choices=INFERENCE+['all'], default='all', help='first stage to compute (earlier ones from --resume-from)')
    st.add_argument('--stop-after', choices=INFERENCE, default='support')
    st.add_argument('--resume-from', type=Path, help='completed earlier run: stages before --stage are reused read-only after hash checks')
    st.add_argument('--no-evaluate', action='store_true', help='skip the ground-truth evaluation (separate process)')
    st.add_argument('--no-qa', action='store_true', help='skip the headless-browser debug QA')
    sb = ap.add_argument_group('endpoint subsampling (before F1/B1; images, cameras and frames are never reduced)')
    sb.add_argument('--pcd-subsample-factor', type=int, default=1, help='integer >= 1; 1 = OFF (every original point, original order)')
    sb.add_argument('--pcd-subsample-method', choices=['spatial-stratified', 'seeded-uniform'], default='spatial-stratified')
    sb.add_argument('--pcd-subsample-seed', type=int, default=0)
    sb.add_argument('--pcd-subsample-boundary-preserve', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    sb.add_argument('--pcd-subsample-report', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True)
    nm = ap.add_argument_group('numerics')
    nm.add_argument('--cuda-backend', choices=['torch', 'triton'], default='torch', help='torch reference (triton is NOT implemented)')
    nm.add_argument('--covariance-precision', choices=['float64', 'float32', 'mixed'], default='float64',
                    help='float64 everywhere; float32 everywhere; mixed = float32 projection/sampling, float64 Kalman algebra')
    nm.add_argument('--pixel-reuse-assignment', choices=['per-frame', 'per-iteration'], default='per-frame',
                    help='front-most-per-pixel reuse decided once at the prediction (default) or re-decided at every IEKF iterate')
    nm.add_argument('--point-chunk-size', type=int, default=200_000, help='points per measurement chunk (memory bound; identical per-point results)')
    nm.add_argument('--max-iekf-iters', type=int, help='default from configs/projective_rgbd.json')
    nm.add_argument('--coherence-count', type=int, default=192); nm.add_argument('--coherence-knn', type=int, default=12); nm.add_argument('--coherence-folds', type=int, default=5)
    nm.add_argument('--coherence-weighting', choices=['pinned_raw', 'measurement_supported'], default='pinned_raw',
                    help='policy whose core drives F2/B2 (both are always computed and reported)')
    nm.add_argument('--joint-q-filter', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True, help='optional constrained (q, qdot) IEKF')
    db = ap.add_argument_group('debug')
    db.add_argument('--debug', choices=['none', 'light', 'full'], default='light')
    db.add_argument('--debug-frames', type=lambda s: [int(x) for x in s.replace(',', ' ').split()], help='e.g. 0,1,2,5,10,30,59 (default: 0,1,2,5,10 and the last)')
    db.add_argument('--debug-point-ids', type=lambda s: [int(x) for x in s.replace(',', ' ').split()], help='traced canonical point ids (default: 6 evenly spaced)')
    db.add_argument('--save-state-interval', type=int, default=10, help='full 6x6 covariance snapshot interval (with --save-full-covariance or --debug full)')
    db.add_argument('--save-full-covariance', action='store_true')
    bt = ap.add_argument_group('dataset root / tests')
    bt.add_argument('--only', nargs='*'); bt.add_argument('--exclude', nargs='*', default=[])
    bt.add_argument('--frame-limit', type=int, help='TEST prefix of frames only')
    a = ap.parse_args(argv)
    if a.cuda_backend == 'triton': ap.error('--cuda-backend triton is not implemented; the verified implementation is the PyTorch CUDA reference')
    if a.pcd_subsample_factor < 1: ap.error('--pcd-subsample-factor must be >= 1')
    if a.stage != 'all' and a.stage != 'free' and a.resume_from is None: ap.error('--stage after free needs --resume-from')
    if a.stage not in ('all',) and INFERENCE.index(a.stage) > INFERENCE.index(a.stop_after): ap.error('--stage is after --stop-after')
    return ap, a


def main(argv=None):
    ap, a = parse(argv)
    if a.output is None: a.output = RESULTS_ROOT/f"projective_{a.scene.resolve().name}_{time.strftime('%Y%m%d_%H%M')}"
    if a.output.expanduser().resolve().is_relative_to(REPO): ap.error(f'{a.output} is inside the code tree {REPO}: results go to a separate output folder')
    if not is_scene(a.scene): return run_root(a, list(sys.argv[1:] if argv is None else argv))
    return run_scene(ap, a)


def run_root(a, argv):
    scenes = sorted(p for p in a.scene.iterdir() if p.is_dir() and is_scene(p))
    if not scenes: raise SystemExit(f'{a.scene}: neither a scene (rgb/ + metadata/) nor a folder of scenes')
    a.output.mkdir(parents=True, exist_ok=True)
    keep, drop = [], False
    for x in argv:
        if x.startswith('--'): drop = x.split('=')[0] in ('--scene', '--input', '--output', '--only', '--exclude')
        if not drop: keep.append(x)
    rows = []
    for sc in scenes:
        if (a.only and sc.name not in a.only) or sc.name in a.exclude: continue
        out = a.output/sc.name; row = {'scene': sc.name, 'output': str(out)}
        if (out/'result.json').exists(): row['status'] = 'SKIPPED_EXISTING'
        elif out.exists() and any(out.iterdir()): row['status'] = 'SKIPPED_PARTIAL_OUTPUT'
        else:
            t0 = time.time()
            with open(a.output/f'{sc.name}.log', 'w') as log:
                row['returncode'] = subprocess.call([sys.executable, '-B', str(REPO/'train.py'), '--scene', str(sc), '--output', str(out), *keep],
                                                    stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            row['minutes'] = round((time.time()-t0)/60, 1)
            row['status'] = json.loads((out/'result.json').read_text()).get('status') if (out/'result.json').exists() else 'FAILED'
        if (out/'result.json').exists():
            r = json.loads((out/'result.json').read_text())
            for k, v in (r.get('stages') or {}).items(): row[k] = v if isinstance(v, str) else (v or {}).get('status')
        rows.append(row); print(json.dumps(row), flush=True)
        (a.output/'summary.json').write_text(json.dumps(rows, indent=1))
        with open(a.output/'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    return 0


def source_hashes():
    return {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for v in STAGE_SOURCES.values() for f in v}


def git_commit():
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    except Exception: return None


def run_scene(ap, a):
    root = a.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()): ap.error(f'{root} exists and is not empty: results are never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t_start = time.time(); stages, timings = {}, {}
    try:
        import numpy as np, torch
        from scene.dataset import Dataset
        from scene.config import load_config
        from utils.io import json_write, serial
        policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
        alpha = float(json.loads((REPO/'configs/image_joint_policy.json').read_text())['tail_probability'])
        max_iters = a.max_iekf_iters or int(policy['max_iekf_iters'])
        data = Dataset(a.scene, a.frame_limit, None, fps=a.fps, dt_seconds=a.dt_seconds, timestamps_file=a.timestamps_file)
        config = load_config(a.noise_config)
        K = len(data.times)
        debug_frames = sorted(set(a.debug_frames if a.debug_frames is not None else [0, 1, 2, 5, 10, K-1]) & set(range(K)))
        directions = ['forward', 'backward'] if a.direction == 'both' else [a.direction]
        sub = {'factor': a.pcd_subsample_factor, 'method': a.pcd_subsample_method, 'seed': a.pcd_subsample_seed,
               'boundary_preserve': a.pcd_subsample_boundary_preserve, 'report': a.pcd_subsample_report}
        from estimator.projective_camera import CameraRig
        rig = CameraRig(data.cameras, 'cpu')
        manifest = {'pipeline': 'four-pass projective RGB-D IEKF (F1/B1 free, F2/B2 rigid)', 'entry': 'train.py',
                    'contract': 'CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md',
                    'arguments': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                    'dataset': data.manifest(), 'cameras': rig.manifest(), 'noise': config.manifest(), 'policy': policy, 'alpha': alpha,
                    'subsample': sub, 'source_hashes': source_hashes(), 'git_commit': git_commit(), 'debug_frames': debug_frames,
                    'ground_truth_used_in_inference': False, 'correspondences_used': False, 'renderer_used': False,
                    'hardware': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
        first = 'free' if a.stage == 'all' else a.stage
        reused, folders = check_resume(a, manifest, first, directions) if a.resume_from else ([], [])
        manifest['reused_stages'] = {'from': str(a.resume_from.resolve()) if a.resume_from else None, 'stages': reused, 'folders': folders}
        json_write(root/'run_manifest.json', serial(manifest))
        want = lambda s: INFERENCE.index(s) >= INFERENCE.index(first) and INFERENCE.index(s) <= INFERENCE.index(a.stop_after)
        report = progress_line
        # ---- F1 / B1
        from estimator.free_projective_run import run_free
        for d in directions:
            if want('free'):
                t0 = time.time()
                stages[f'free_{d}'] = run_free(data, config, policy, d, root/f'free_{d}', a.device, sub, a.covariance_precision, max_iters,
                                               float(policy['iekf_step_tolerance']), a.point_chunk_size, debug_frames, a.debug_point_ids,
                                               a.save_full_covariance or a.debug == 'full', a.save_state_interval, report, a.pixel_reuse_assignment)
                timings[f'free_{d}'] = time.time()-t0
            elif 'free' in reused: stages[f'free_{d}'] = json.loads((stage_dir(root, f'free_{d}')/'status.json').read_text())
        # ---- coherence
        if INFERENCE.index(a.stop_after) >= 1:
            from estimator.projective_coherence import run_coherence
            for d in directions:
                if want('coherence'):
                    t0 = time.time()
                    stages[f'coherence_{d}'] = run_coherence(stage_dir(root, f'free_{d}'), root/f'coherence_{d}', a.coherence_count, a.coherence_knn, a.coherence_folds, a.coherence_weighting)
                    timings[f'coherence_{d}'] = time.time()-t0
                else: stages[f'coherence_{d}'] = json.loads((stage_dir(root, f'coherence_{d}')/'coherence_summary.json').read_text())
                print(f'[coherence {d}] {stages[f"coherence_{d}"]["status"]}', flush=True)
        # ---- F2 / B2
        if INFERENCE.index(a.stop_after) >= 2:
            from estimator.rigid_kalman_cuda import run_rigid
            for d in directions:
                if want('rigid'):
                    t0 = time.time()
                    stages[f'rigid_{d}'] = run_rigid(data, config, policy, d, stage_dir(root, f'free_{d}'), stage_dir(root, f'coherence_{d}'), root/f'rigid_{d}', a.device,
                                                     max_iters, 1e-10, debug_frames, a.covariance_precision, report, a.noise_config, a.pixel_reuse_assignment)
                    timings[f'rigid_{d}'] = time.time()-t0
                else: stages[f'rigid_{d}'] = json.loads((stage_dir(root, f'rigid_{d}')/'status.json').read_text())
                print(f'[rigid {d}] {stages[f"rigid_{d}"]["status"]}', flush=True)
        # ---- joint (+ forward/backward comparison)
        joints = {}
        if INFERENCE.index(a.stop_after) >= 3:
            from joint_reduction.joint_projective import run_joint, compare_directions
            for d in directions:
                if want('joint'):
                    t0 = time.time()
                    joints[d] = run_joint(data, config, policy, d, stage_dir(root, f'free_{d}'), stage_dir(root, f'coherence_{d}'), stage_dir(root, f'rigid_{d}'), root/'joint'/d, alpha,
                                          a.device, a.joint_q_filter, a.noise_config)
                    timings[f'joint_{d}'] = time.time()-t0
                else: joints[d] = json.loads((stage_dir(root, f'joint/{d}')/'result.json').read_text())
                stages[f'joint_{d}'] = {'status': joints[d]['status']}
                print(f'[joint {d}] {joints[d]["status"]}', flush=True)
            if len(joints) == 2 and all('candidates' in j for j in joints.values()):
                (root/'bidirectional').mkdir(exist_ok=True)
                cmp = compare_directions(joints['forward'], joints['backward'])
                json_write(root/'bidirectional'/'comparison.json', serial(cmp)); stages['bidirectional'] = {'status': 'COMPARED'}
        # ---- support
        if INFERENCE.index(a.stop_after) >= 4:
            from joint_reduction.support_projective import run_support
            for d in directions:
                t0 = time.time()
                stages[f'support_{d}'] = run_support(data, config, policy, stage_dir(root, f'free_{d}'), stage_dir(root, f'rigid_{d}'), joints.get(d, {}), root/'joint'/d/'support', alpha, a.device)
                timings[f'support_{d}'] = time.time()-t0
                print(f'[support {d}] {stages[f"support_{d}"]["status"]}', flush=True)
        # ---- performance summary, debug, evaluation, QA
        write_performance(root, directions, timings)
        complete = all(str(v.get('status') if isinstance(v, dict) else v).startswith(('COMPLETE', 'RESOLVED', 'COMPARED', 'PRISMATIC', 'REVOLUTE', 'STATIC'))
                       for v in stages.values())
        result = {'status': 'COMPLETE' if complete else 'INCOMPLETE_OR_UNRESOLVED', 'stages': stages, 'stage_seconds': timings,
                  'runtime_seconds': time.time()-t_start}
        json_write(root/'result.json', serial(result))
        if a.debug != 'none':
            t0 = time.time()
            from debug.projective_debug import write_debug
            result['debug'] = write_debug(root, data, config, policy, directions, debug_frames, a.debug, a.device)
            timings['debug'] = time.time()-t0
            print(f'Debug index: {root/"debug"/"index.html"}', flush=True)
        if not a.no_evaluate:
            result['evaluation'] = tool('evaluate', [REPO/'tools/evaluate_projective_gt.py', '--run', root, '--scene', a.scene], root)
        if not a.no_qa and a.debug != 'none':
            result['qa'] = tool('qa', [REPO/'tools/qa_projective_debug.py', '--run', root, '--scene', a.scene], root) \
                if (shutil.which('google-chrome') or Path('/usr/bin/google-chrome').exists()) else {'status': 'SKIPPED', 'reason': 'google-chrome not found'}
        result['runtime_seconds'] = time.time()-t_start; result['stage_seconds'] = timings
        json_write(root/'result.json', serial(result))
        print(json.dumps({'status': result['status'], 'stages': {k: (v.get('status') if isinstance(v, dict) else v) for k, v in stages.items()},
                          'runtime_s': round(result['runtime_seconds'], 1)}, indent=1), flush=True)
        return 0 if complete else 3
    except BaseException as err:
        (root/'failure.log').write_text(traceback.format_exc())
        (root/'failure.json').write_text(json.dumps({'status': 'FAILED', 'reason': str(err), 'stages': {k: (v.get('status') if isinstance(v, dict) else v) for k, v in stages.items()}}, indent=1, default=str))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close(); (root/'run.lock').unlink(missing_ok=True)


def progress_line(direction, i, n, info):
    if 'sensor' in info:            # free pass frame record
        p, f = info['processing'], info['filter_update']
        msg = (f"frame {info['frame']:3d} {p['wall_ms']:7.0f} ms  observed {info['sensor']['unique_points_observed']}/{info['sensor']['points']}"
               f"  not-converged {f['not_converged_points']}  iters {p['iekf_iterations']}")
    else:                           # rigid pass timeline row
        msg = (f"frame {info['frame']:3d} {info['wall_ms']:7.0f} ms  rows {info['rows']}  |t| {info['translation_m']:.4f} m  "
               f"|rot| {info['rotation_rad']:.4f} rad  converged {info['converged']}")
    print(f'[{direction}] {i}/{n} {msg}', flush=True)


def tool(name, cmd, root):
    t0 = time.time()
    out = root/{'evaluate': 'evaluation_gt', 'qa': 'debug'}[name]; out.mkdir(exist_ok=True)
    with open(out/f'{name}.log', 'w') as log:
        rc = subprocess.call([sys.executable, *map(str, cmd)], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(f'{name}: exit {rc} ({out/(name+".log")})', flush=True)
    return {'status': 'PASS' if rc == 0 else 'FAILED', 'returncode': rc, 'seconds': time.time()-t0, 'log': str(out/f'{name}.log')}


def write_performance(root, directions, timings):
    import numpy as np
    from utils.io import json_write
    (root/'performance').mkdir(exist_ok=True)
    perf = {'stage_wall_seconds': timings}
    for d in directions:
        f = stage_dir(root, f'free_{d}')/'numeric'/'timeline.csv'
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            col = lambda k: np.array([float(r[k]) for r in rows if r[k] not in ('', 'None')])
            perf[f'free_{d}'] = {'frames': len(rows), 'wall_ms_median': float(np.median(col('wall_ms'))), 'wall_ms_p95': float(np.quantile(col('wall_ms'), .95)),
                                 'cuda_ms_median': float(np.median(col('cuda_ms_total'))) if len(col('cuda_ms_total')) else None,
                                 'gpu_peak_allocated_MB_max': float(col('gpu_peak_allocated_MB').max()) if len(col('gpu_peak_allocated_MB')) else None}
    json_write(root/'performance'/'summary.json', perf)


def check_resume(a, manifest, first, directions):
    """Stages before `first` come from --resume-from, read in place (utils.run_paths.stage_dir), after exact checks of point-id subsample
    mapping, calibration, timestamps, cameras, noise, policy and the implementation hashes of every reused stage."""
    src = a.resume_from.resolve()
    old = json.loads((src/'run_manifest.json').read_text())
    if old.get('entry') != 'train.py' or 'four-pass' not in old.get('pipeline', ''): raise ValueError('Rejected resume: not a four-pass projective run')
    norm = lambda v: json.loads(json.dumps(v, default=str))
    for key in ('dataset', 'cameras', 'noise', 'policy', 'subsample'):
        if norm(old.get(key)) != norm(manifest[key]): raise ValueError(f'Rejected resume: {key} differs')
    stages = INFERENCE[:INFERENCE.index(first)]
    for s in stages:
        for f in STAGE_SOURCES[s]:
            if old['source_hashes'].get(f) != manifest['source_hashes'][f]: raise ValueError(f'Rejected resume: {f} changed since stage {s} ran')
    reused = []
    for s in stages:
        folders = {'free': [f'free_{d}' for d in directions], 'coherence': [f'coherence_{d}' for d in directions],
                   'rigid': [f'rigid_{d}' for d in directions], 'joint': [f'joint/{d}' for d in directions], 'support': []}[s]
        for f in folders:
            if not (src/f).exists(): raise ValueError(f'Rejected resume: {src/f} missing')
            reused.append(f)
    return stages, reused


if __name__ == '__main__':
    raise SystemExit(main())
