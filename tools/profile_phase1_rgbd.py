#!/usr/bin/env python3
"""STEP 2A-P: per-stage CUDA profile of one Phase-1 frame update on fixed saved inputs (read-only).

Every instrumented call is bracketed by torch.cuda.Event records with a synchronize at its end (this adds one host
sync per call, so absolute totals are slightly pessimistic; the split between stages is what matters). Frames are run
from the SAVED prior (process6.predict of the saved posterior of the previous frame), after one warm-up frame.
A larger scene can be profiled for one frame from its endpoint catalogue (anchor prior predicted to frame 1).
Outputs: reports/phase1_gpu_profile.csv, reports/phase1_gpu_profile.json (+ torch.profiler kernel table).
"""
import argparse, collections, functools, json, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, torch

TIMES = collections.defaultdict(list); COUNTS = collections.Counter()


def timed(name, fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        if not torch.cuda.is_available(): t0 = time.perf_counter(); r = fn(*a, **k); TIMES[name].append(1e3*(time.perf_counter()-t0)); return r
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        w0 = time.perf_counter(); s.record(); r = fn(*a, **k); e.record(); torch.cuda.synchronize()
        TIMES[name].append(s.elapsed_time(e)); TIMES[name+' [wall]'].append(1e3*(time.perf_counter()-w0)); COUNTS[name] += 1
        return r
    return wrap


def instrument():
    import estimator.point_renderer as R, estimator.observation_terms as O, estimator.supernodes as SN
    import estimator.sparse_iekf as I, estimator.restricted_posterior as RP, estimator.covariance as PC
    import rigeuler_ref.trust_region as TR
    R._candidates = timed('render: projection + footprint candidates', R._candidates)
    R.render_view = timed('render: one camera total (candidates, front, splat, Jacobian)', R.render_view)
    R.render = timed('render: all cameras', R.render)
    for m in ('apply_H', 'apply_HT', 'block_fisher'):
        setattr(R.Linearization, m, timed(f'H: {m}', getattr(R.Linearization, m)))
    O.FrameObservation.terms = timed('observation: robust terms + IRLS weights', O.FrameObservation.terms)
    O.FrameObservation.apply_W = timed('observation: apply_W', O.FrameObservation.apply_W)
    for m in ('expand', 'reduce', 'blocks'):
        setattr(SN.GroupBasis, m, timed(f'G: {m}', getattr(SN.GroupBasis, m)))
    I.group_precond = timed('solver: group preconditioner (G^T M G + block Cholesky)', I.group_precond)
    I.steihaug_cg = timed('solver: Steihaug-CG (incl. A_g applications)', I.steihaug_cg)
    I.linear_system = timed('solver: linear system assembly (b, Fisher, preconditioner)', I.linear_system)
    I.render = R.render                                    # image_iekf imported render by name
    PC.rigeuler_block_information = timed('posterior: rigeuler_block_information', PC.rigeuler_block_information)
    PC.restricted_joseph_marginals = timed('posterior: restricted_joseph_marginals', PC.restricted_joseph_marginals)
    I.frame_step = timed('FRAME: frame_step total', I.frame_step)
    return I


def gpu_state():
    q = lambda *a: subprocess.run(['nvidia-smi', *a], capture_output=True, text=True).stdout.strip()
    return {'gpu': q('--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'),
            'other_processes': q('--query-compute-apps=pid,used_memory', '--format=csv,noheader')}


def contributions(lin):
    return {'contributions': sum(int(len(v.point)) for v in lin.views), 'per_camera': [int(len(v.point)) for v in lin.views],
            'candidates': sum(v.stats['candidate_contributors'] for v in lin.views), 'occluded': sum(v.stats['occluded_contributors'] for v in lin.views),
            'covered_pixels': sum(v.stats['covered_pixels'] for v in lin.views), 'projected_points': [v.stats['projected_points'] for v in lin.views],
            'max_contributors_per_pixel': max(int(v.pix_seg[2].max()) if len(v.pix_seg[2]) else 0 for v in lin.views),
            'mean_contributors_per_covered_pixel': float(np.mean([float(v.pix_seg[2].double().mean()) for v in lin.views if len(v.pix_seg[2])]))}


def profile_frame(I, sm, Pm, frames, cfg, rgb, rho, policy, G, dev, tag, config):
    from estimator.observation_terms import levels, ObservationPolicy
    TIMES.clear(); COUNTS.clear(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); lv = levels(frames, config, ObservationPolicy(), dev); torch.cuda.synchronize(); t_obs = 1e3*(time.perf_counter()-t0)
    s, P, obs_, lin, terms, rep = I.frame_step(sm, Pm.clone(), lv, rgb, rho, policy, G, cfg)
    rows = []
    total = float(np.sum(TIMES['FRAME: frame_step total']))
    for name, v in TIMES.items():
        if name.endswith('[wall]'): continue
        v = np.asarray(v); rows.append({'case': tag, 'stage': name, 'calls': len(v), 'total_ms': float(v.sum()), 'median_ms': float(np.median(v)),
                                        'p95_ms': float(np.quantile(v, .95)), 'percent_of_frame': 100*float(v.sum())/total if total else None,
                                        'wall_total_ms': float(np.sum(TIMES.get(name+' [wall]', [np.nan])))})
    rows.append({'case': tag, 'stage': 'observation: CPU image decode + GPU upload (levels)', 'calls': 1, 'total_ms': t_obs, 'median_ms': t_obs, 'p95_ms': t_obs,
                 'percent_of_frame': None, 'wall_total_ms': t_obs})
    meta = {'case': tag, 'points': int(len(sm)), 'groups': G.count, 'cameras': len(frames), 'resolution': [frames[0].camera.width, frames[0].camera.height],
            'iterations': len(rep['history']), 'accepted': rep['accepted_steps'], 'rejected': rep['rejected_trials'],
            'pcg_iterations': [h['pcg_iterations'] for h in rep['history']], 'peak_allocated_MB': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MB': torch.cuda.max_memory_reserved()/2**20, **contributions(lin)}
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--frames', nargs='*', type=int, default=[1, 10, 30]); ap.add_argument('--fps', type=float, default=59)
    ap.add_argument('--large-scene', type=Path); ap.add_argument('--kernel-table', action='store_true')
    ap.add_argument('--memory-fraction', type=float, help='cap this process (shared GPU): torch.cuda.set_per_process_memory_fraction')
    a = ap.parse_args(); dev = 'cuda'
    if a.memory_fraction: torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    from tools.audit_phase1_operators import blade_setup, saved_prior
    from estimator.sparse_iekf import IEKFConfig
    I = instrument()
    env = {'memory_fraction': a.memory_fraction, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
           'threads': torch.get_num_threads(), **gpu_state()}
    B = blade_setup(a.run, a.scene, dev, a.fps); cfg = IEKFConfig()
    rows, metas = [], []
    frames_of = lambda k: [B['data'].observation(k, c) for c in range(len(B['cams']))]
    sm, Pm = saved_prior(B, a.frames[0]); profile_frame(I, sm, Pm, frames_of(a.frames[0]), cfg, B['rgb'], B['rho'], B['policy'], B['G'], dev, 'warmup', B['config'])
    for k in a.frames:
        sm, Pm = saved_prior(B, k)
        r, m = profile_frame(I, sm, Pm, frames_of(k), cfg, B['rgb'], B['rho'], B['policy'], B['G'], dev, f'blade_frame{k}', B['config'])
        rows += r; metas.append(m); print(json.dumps(m), flush=True)
    if a.kernel_table:
        from torch.profiler import profile, ProfilerActivity
        sm, Pm = saved_prior(B, a.frames[-1])
        from estimator.observation_terms import levels, ObservationPolicy
        lv = levels(frames_of(a.frames[-1]), B['config'], ObservationPolicy(), dev)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            I.frame_step(sm, Pm.clone(), lv, B['rgb'], B['rho'], B['policy'], B['G'], cfg)
        (REPO/'reports/phase1_gpu_kernels.txt').write_text(prof.key_averages().table(sort_by='cuda_time_total', row_limit=40))
    if a.large_scene:
        from scene.dataset import Dataset
        from estimator.phase1 import endpoint_catalogue
        from estimator.point_renderer import footprints, CameraTensors
        from estimator.supernodes import GroupBasis
        from estimator.state import anchor_posterior
        from estimator.process import predict
        data = Dataset(a.large_scene, 2, None, fps=a.fps); config = B['config']
        k0, part, _ = endpoint_catalogue(data, 'first', config, cfg)
        t = B['t']; rho = t(footprints(part, data, 1.5)); G = GroupBasis(t(part.xyz), t(part.membership, torch.long), t(part.centroids))
        st, p = anchor_posterior(part.xyz, part.source_covariance, config.velocity_covariance)
        sm, Pm = predict(t(st), t(p), float(data.times[1]-data.times[0]), config)
        try:
            r, m = profile_frame(I, sm, Pm, [data.observation(1, c) for c in range(len(data.cameras))], cfg, t(part.rgb), rho, B['policy'], G, dev,
                                 f'{a.large_scene.name}_frame1_from_anchor', config)
            rows += r; metas.append(m); print(json.dumps(m), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            metas.append({'case': f'{a.large_scene.name}_frame1_from_anchor', 'points': int(len(sm)), 'groups': G.count, 'out_of_memory': str(e).split('.')[0],
                          'memory_fraction': a.memory_fraction}); print('OOM on the large scene', len(sm), flush=True)
    import csv
    with open(REPO/'reports/phase1_gpu_profile.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (REPO/'reports/phase1_gpu_profile.json').write_text(json.dumps({'environment': env, 'cases': metas, 'rows': rows,
                                                                  'method': 'torch.cuda.Event per instrumented call with synchronize; warmed; saved priors'}, indent=1, default=float))
    print('written reports/phase1_gpu_profile.{csv,json}')


if __name__ == '__main__':
    main()
