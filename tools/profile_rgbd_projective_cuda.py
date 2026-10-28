#!/usr/bin/env python3
"""Gate D profiling of the free projective IEKF (contract 7.3, 8.4): per-stage CUDA-event milliseconds, wall time, peak
allocated/reserved GPU memory and counters for the first --frames frames of the FORWARD free pass, for every requested
endpoint subsample factor and precision; float32 / mixed results compared with float64 at the same factor; one
torch.profiler kernel table (factor 1, float64). Writes <out>/profile.json, profile.md, kernels.txt and scratch runs
under <out>/runs/ (never into an existing result folder)."""
import argparse, json, shutil, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from scene.dataset import Dataset
from scene.config import load_config
from estimator.free_projective_run import run_free


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=4); ap.add_argument('--factors', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--precisions', nargs='+', default=['float64', 'float32', 'mixed']); ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--memory-fraction', type=float, help='cap this process on a shared GPU')
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit(f'{a.out} exists and is not empty')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.memory_fraction: torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
    data = Dataset(a.scene, a.frames, None); config = load_config()
    rows, finals = [], {}
    for f in a.factors:
        for prec in a.precisions:
            run = a.out/'runs'/f'f{f}_{prec}'
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                st = run_free(data, config, policy, 'forward', run, 'cuda', {'factor': f, 'method': 'spatial-stratified', 'seed': 0, 'boundary_preserve': True, 'report': True},
                              prec, int(policy['max_iekf_iters']), float(policy['iekf_step_tolerance']), a.chunk)
            except torch.cuda.OutOfMemoryError as e:
                rows.append({'factor': f, 'precision': prec, 'status': 'OUT_OF_MEMORY', 'detail': str(e).split('GiB')[0]+'GiB'}); continue
            total = time.perf_counter()-t0
            nums = [json.loads(p.read_text()) for p in sorted((run/'numeric').glob('frame_*.json'))]
            steady = nums[1:] if len(nums) > 1 else nums                    # first update includes CUDA warm-up
            stages = sorted({k for n in steady for k in n['processing']['cuda_ms']})
            sub = json.loads((run/'subsample_manifest.json').read_text())
            r = {'factor': f, 'precision': prec, 'status': 'OK', 'N_full': sub['N_full'], 'N_selected': sub['N_selected'], 'actual_reduction': sub['actual_reduction'],
                 'subsample_seconds': sub.get('seconds'), 'run_seconds_total': total,
                 'frame_wall_ms_median': float(np.median([n['processing']['wall_ms'] for n in steady])),
                 'frame_cuda_ms_median': float(np.median([sum(n['processing']['cuda_ms'].values()) for n in steady])),
                 'stage_cuda_ms_median': {s: float(np.median([n['processing']['cuda_ms'].get(s, 0.) for n in steady])) for s in stages},
                 'gpu_peak_allocated_MB_max': max(n['processing']['gpu_peak_allocated_MB'] for n in nums),
                 'gpu_peak_reserved_MB_max': max(n['processing']['gpu_peak_reserved_MB'] for n in nums),
                 'h2d_bytes_per_frame': nums[0]['host_to_gpu_bytes'], 'rows_accepted_median': float(np.median([n['sensor']['rows_accepted_final'] for n in steady])),
                 'observed_points_median': float(np.median([n['sensor']['unique_points_observed'] for n in steady])),
                 'iekf_iterations': [n['processing']['iekf_iterations'] for n in nums], 'not_converged_median': float(np.median([n['filter_update']['not_converged_points'] for n in steady]))}
            X = np.load(run/'posterior_xyz.npy')[a.frames-1]
            finals[(f, prec)] = X
            if prec != 'float64' and (f, 'float64') in finals:
                d = np.linalg.norm(X-finals[(f, 'float64')], axis=1)
                r['vs_float64_position_diff_m'] = {'median': float(np.median(d)), 'p99': float(np.quantile(d, .99)), 'max': float(d.max())}
            rows.append(r); print(json.dumps({k: v for k, v in r.items() if k != 'stage_cuda_ms_median'}), flush=True)
            if not (f == a.factors[0] and prec == 'float64'): shutil.rmtree(run/'covariance', ignore_errors=True)
    # kernel table: one update, factor 1 float64
    from torch.profiler import profile, ProfilerActivity
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise
    from estimator.point_kalman_cuda import update_points, predict
    first = a.out/'runs'/f'f{a.factors[0]}_float64'
    if (first/'canonical.npz').exists():
        can = np.load(first/'canonical.npz'); X = np.load(first/'posterior_xyz.npy'); V = np.load(first/'velocity.npy')
        dev = 'cuda'; rig = CameraRig(data.cameras, dev); noise = SensorNoise.from_config(config, policy, dev)
        n = len(X[1]); s = torch.zeros((n, 6), dtype=torch.float64, device=dev); s[:, :3] = torch.as_tensor(X[1], device=dev); s[:, 3:] = torch.as_tensor(V[1], dtype=torch.float64, device=dev)
        P = torch.zeros((n, 6, 6), dtype=torch.float64, device=dev); P[:, :3, :3] = 1e-4*torch.eye(3, dtype=torch.float64, device=dev); P[:, 3:, 3:] = torch.eye(3, dtype=torch.float64, device=dev)
        frame = MeasuredFrame(data, 2, dev)
        sm, Pm = predict(s, P, float(data.times[2]-data.times[1]), torch.as_tensor(np.asarray(config.qc), dtype=torch.float64, device=dev))
        update_points(sm, Pm, frame, rig, torch.as_tensor(can['canonical_rgb'], dtype=torch.float64, device=dev), noise, policy, int(policy['max_iekf_iters']), float(policy['iekf_step_tolerance']), a.chunk)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            update_points(sm, Pm, frame, rig, torch.as_tensor(can['canonical_rgb'], dtype=torch.float64, device=dev), noise, policy, int(policy['max_iekf_iters']), float(policy['iekf_step_tolerance']), a.chunk)
        (a.out/'kernels.txt').write_text(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))
    env = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'scene': str(a.scene), 'frames': a.frames,
           'memory_fraction': a.memory_fraction, 'note': 'steady-state medians exclude the first (warm-up) update; CUDA ms are per-stage event times with one sync per frame'}
    (a.out/'profile.json').write_text(json.dumps({'environment': env, 'rows': rows}, indent=1))
    md = ['# Free projective IEKF profile', '', f"scene `{a.scene.name}`, first {a.frames} frames, forward, {env['gpu']}", '',
          '| factor | precision | N selected | frame wall ms | frame CUDA ms | peak alloc MB | rows | observed | not conv. | vs float64 position diff (median / max, m) |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['status'] != 'OK': md.append(f"| {r['factor']} | {r['precision']} | — | {r['status']} |"); continue
        v = r.get('vs_float64_position_diff_m')
        dtext = '—' if v is None else '%.2e / %.2e' % (v['median'], v['max'])
        md.append(f"| {r['factor']} | {r['precision']} | {r['N_selected']} | {r['frame_wall_ms_median']:.0f} | {r['frame_cuda_ms_median']:.0f} | {r['gpu_peak_allocated_MB_max']:.0f} | "
                  f"{r['rows_accepted_median']:.0f} | {r['observed_points_median']:.0f} | {r['not_converged_median']:.0f} | {dtext} |")
    ok = [r for r in rows if r['status'] == 'OK']
    if ok:
        md += ['', 'stage CUDA ms (median), first row:', '', '| stage | ms |', '|---|---|'] + [f'| {k} | {v:.1f} |' for k, v in sorted(ok[0]['stage_cuda_ms_median'].items(), key=lambda kv: -kv[1])]
    (a.out/'profile.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
