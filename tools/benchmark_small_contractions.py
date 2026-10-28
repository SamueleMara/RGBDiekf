#!/usr/bin/env python3
"""STEP 2B-B1 equal-input benchmark: IEKFConfig(small_contractions=False) vs True on the SAME completed frame update.

Fixed inputs per frame k: the saved prior (process6.predict of the saved posterior k-1) and the same 8 observed RGB-D
images. No per-call instrumentation (wall time of the whole update, synchronized). Repeats alternate the order of the
two variants. Also reports operator discrepancies (H, H^T, Fisher, G^T M G, A_g, b_g) at one shared linearization, the
filtered-state / posterior discrepancy and the accept/reject sequences. Read-only: writes only --out.
Optional --large-scene: one frame of a larger scene under --memory-fraction, both variants (OOM recorded, not raised).
"""
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from estimator import small_contractions as sc
from estimator.sparse_iekf import IEKFConfig, frame_step, linear_system
from estimator.observation_terms import levels, ObservationPolicy


def timed_update(sm, Pm, lv, rgb, rho, policy, G, flag):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter(); out = frame_step(sm, Pm.clone(), lv, rgb, rho, policy, G, IEKFConfig(small_contractions=flag))
    torch.cuda.synchronize(); wall = time.perf_counter()-t0
    return out, wall, (torch.cuda.max_memory_allocated()-base)/2**20, torch.cuda.max_memory_reserved()/2**20


def rel(a, b): return float((a-b).norm()/b.norm().clamp_min(1e-300))


def operators(s, sm, Pm, lin, terms, obs, G):
    Lam = torch.linalg.inv(Pm); Lam = (Lam+Lam.transpose(1, 2))/2
    g = torch.Generator(device='cpu').manual_seed(0); dev = s.device
    v = torch.randn(lin.n, 3, generator=g, dtype=torch.float64).to(dev); z = torch.randn(G.count, 12, generator=g, dtype=torch.float64).to(dev)
    a = [torch.randn(len(r_), 5, generator=g, dtype=torch.float64).to(dev) for r_ in terms['residual_adjoint']]
    out = {}
    for flag in (False, True):
        with sc.enabled(flag):
            b, A, M = linear_system(s, sm, Lam, lin, terms, obs, G, 1e-3)
            out[flag] = {'H v': torch.cat(lin.apply_H(v)), 'H^T a': lin.apply_HT(a), 'block_fisher': lin.block_fisher(terms['weights']),
                         'G^T M G blocks': G.blocks(Pm), 'b_g': b, 'A_g z': A(z), 'preconditioner z': M(z)}
    return {k: rel(out[True][k], out[False][k]) for k in out[False]}


def compare(res0, res1, sm):
    (s0, P0, *_, r0), (s1, P1, *_, r1) = res0, res1
    return {'state_correction_rel': rel(s1-sm, s0-sm), 'state_max_abs_m': float((s1-s0).abs().max()), 'posterior_blocks_rel': rel(P1, P0),
            'posterior_max_abs': float((P1-P0).abs().max()), 'energy_final': [r0['energy_final'], r1['energy_final']],
            'accept_sequence_equal': [h['accepted'] for h in r0['history']] == [h['accepted'] for h in r1['history']],
            'pcg_iterations': [[h['pcg_iterations'] for h in r0['history']], [h['pcg_iterations'] for h in r1['history']]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--frames', nargs='*', type=int, default=[1, 10, 30]); ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--fps', type=float, default=59)
    ap.add_argument('--large-scene', type=Path); ap.add_argument('--memory-fraction', type=float); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); dev = 'cuda'
    if a.memory_fraction: torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    from tools.audit_phase1_operators import blade_setup, saved_prior, observation
    from tools.profile_phase1_rgbd import gpu_state
    report = {'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'memory_fraction': a.memory_fraction, **gpu_state()}, 'frames': []}
    B = blade_setup(a.run, a.scene, dev, a.fps)
    obs_of = lambda k: [B['data'].observation(k, c) for c in range(len(B['cams']))]
    sm, Pm = saved_prior(B, a.frames[0]); lv = levels(obs_of(a.frames[0]), B['config'], ObservationPolicy(), dev)
    for flag in (False, True): timed_update(sm, Pm, lv, B['rgb'], B['rho'], B['policy'], B['G'], flag)          # warm-up
    for k in a.frames:
        sm, Pm = saved_prior(B, k); lv = levels(obs_of(k), B['config'], ObservationPolicy(), dev)
        walls, mem, res = {False: [], True: []}, {}, {}
        for r in range(a.repeats):
            for flag in ((False, True) if r % 2 == 0 else (True, False)):
                res[flag], w, pa, pr = timed_update(sm, Pm, lv, B['rgb'], B['rho'], B['policy'], B['G'], flag)
                walls[flag].append(w); mem[flag] = {'peak_allocated_above_inputs_MB': pa, 'peak_reserved_MB': pr}
        s0, _, _, lin0, terms0, _ = res[False]
        row = {'frame': k, 'wall_s_off': walls[False], 'wall_s_on': walls[True], 'median_wall_s_off': float(np.median(walls[False])),
               'median_wall_s_on': float(np.median(walls[True])), 'speedup_median': float(np.median(walls[False])/np.median(walls[True])),
               'memory_off': mem[False], 'memory_on': mem[True], **compare(res[False], res[True], sm),
               'operator_rel_discrepancy_at_off_final_linearization': operators(s0, sm, Pm, lin0, terms0, observation(B, k, dev), B['G'])}
        report['frames'].append(row); print(json.dumps(row), flush=True)
        del res, lin0, terms0; torch.cuda.empty_cache()
    if a.large_scene:
        from scene.dataset import Dataset
        from estimator.phase1 import endpoint_catalogue
        from estimator.point_renderer import footprints
        from estimator.supernodes import GroupBasis
        from estimator.state import anchor_posterior
        from estimator.process import predict
        del B; torch.cuda.empty_cache()
        from scene.config import load_config
        from estimator.point_renderer import CameraTensors, RenderPolicy
        config = load_config(); data = Dataset(a.large_scene, 2, None, fps=a.fps); t = lambda x, dt=torch.float64: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        _, part, _ = endpoint_catalogue(data, 'first', config, IEKFConfig())
        rho = t(footprints(part, data, 1.5)); G = GroupBasis(t(part.xyz), t(part.membership, torch.long), t(part.centroids))
        policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
        st, p = anchor_posterior(part.xyz, part.source_covariance, config.velocity_covariance)
        sm, Pm = predict(t(st), t(p), float(data.times[1]-data.times[0]), config)
        lv = levels([data.observation(1, c) for c in range(len(data.cameras))], config, ObservationPolicy(), dev)
        large = {'scene': a.large_scene.name, 'points': len(sm), 'groups': G.count}
        for flag in (True, False):
            try:
                res, w, pa, pr = timed_update(sm, Pm, lv, t(part.rgb), rho, policy, G, flag)
                large['on' if flag else 'off'] = {'wall_s': w, 'peak_allocated_above_inputs_MB': pa, 'peak_reserved_MB': pr, 'energy_final': res[-1]['energy_final'],
                                                  'accepted': [h['accepted'] for h in res[-1]['history']]}
                del res
            except torch.cuda.OutOfMemoryError as e:
                large['on' if flag else 'off'] = {'out_of_memory': str(e).split('GiB')[0]+'GiB'}
            torch.cuda.empty_cache(); print(json.dumps(large), flush=True)
        report['large_scene'] = large
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
