#!/usr/bin/env python3
"""Which observation variable (R, G, B, depth) drives the static-point drift of a --free-solver temporal-ieks run.
Offline diagnostic: reads the run's outputs and the scene's frames; ground truth / the static proxy only SELECTS the
static points (tools/audit_static_equilibrium.py basis), nothing goes back into any estimator.

Three per-channel measurements on every static point (whitened rows of the solver's own measurement model, exactly as
estimator/ieks_point_trajectory.py::evaluate_frame builds them; the channel sums are checked against it):

  A  ONSET PUSH AT REST. The static hypothesis S0 (x_k = x_ref, v_k = 0, colour = source colour) with the solver's
     production admission (visibility with every other point at its final trajectory, row gate with the warm start's
     prior-predictive P-_k rebuilt from filter_cov / F / Q, frame-0 source camera excluded). At S0 the prior and
     dynamics gradients are exactly zero, so the full-trajectory Gauss-Newton step is A^-1 (sum of the channels' data
     gradients) and splits EXACTLY into one step per channel (same A: information of all channels). Share of channel c
     = <step_c, step> / |step|^2 over the positions of frames 0..W (shares sum to 1).
  B  OBJECTIVE PREFERENCE. On the rows of the final solution (camera_reason == 0, frozen, R frozen at the final
     trajectory, final latent colour for both): dJ_c = J_c(S0) - J_c(S_final) per channel, and the prior + dynamics
     change. dJ_c > 0: channel c prefers the drifted trajectory to rest.
  C  FORCE ALONG THE DRIFT at the final trajectory (same rows): f_c = sum_k <g_c,k, x_k - x_ref>, g = H^T R^-1 r the
     descent direction of the data term; f_c > 0 pushes further away from rest.

    python tools/attribute_static_drift_channels.py --run RUN --basis-from AUDIT_BASELINE --out OUT_DIR [--window 50]
                                                    [--scene SCENE] [--max-points N] [--verify-frames 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHANNELS = ('R', 'G', 'B', 'depth')


def channel_frame(M, frame, X, a, Sigma, pts, support=None, src=None, X_lin=None):
    """evaluate_frame (estimator/ieks_point_trajectory.py) with the data gradient / objective kept per channel."""
    from estimator.projective_sensor import CODE
    from estimator.projective_terms import build_terms
    from estimator.projective_visibility import visibility, position_sigma_z2
    N, C = len(X), M.rig.count
    dev, dt = X.device, M.dtype
    z = lambda *s: torch.zeros(s, dtype=dt, device=dev)
    out = {'Ixx': z(N, 3, 3), 'Ixa': z(N, 3, 3), 'Iaa': z(N, 3), 'gx': z(4, N, 3), 'q': z(4, N), 'ga': z(4, N, 3),
           'nrows': torch.zeros(N, dtype=torch.long, device=dev), 'valid': torch.ones(N, dtype=torch.bool, device=dev),
           'support': torch.zeros((C, N), dtype=torch.bool, device=dev)}
    if support is None:
        proj = M.rig.project(X, jacobian=False)
        vreason, _ = visibility(proj, position_sigma_z2(M.rig.R.to(dt), Sigma), frame.H, frame.W, M.rig.near, M.nsig)
        cand = vreason.T == CODE['VALID']
        if src is not None: cand[torch.arange(N, device=dev), src] = False
        policy = M.policy
    else:
        cand = support.T.clone(); policy = M.policy_ungated
    cand &= pts[:, None]
    rows = torch.nonzero(cand)
    if not len(rows): return out
    pr = rows[:, 0]
    bounds = torch.searchsorted(pr, torch.arange(0, N+M.chunk, M.chunk, device=dev).clamp_max(N)).tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo: continue
        p, c = pr[lo:hi], rows[lo:hi, 1]
        t = build_terms(M.rig, frame, c, p, X[p], a[p], Sigma[p], M.noise, policy, dt, None)
        ok = t.valid()
        if support is not None and X_lin is not None:
            tl = build_terms(M.rig, frame, c, p, X_lin[p], a[p], Sigma[p], M.noise, policy, dt, None)
            t.var = torch.where(tl.valid()[:, None], tl.var, t.var)
        if support is None: out['support'][c[ok], p[ok]] = True
        else: out['valid'][p[~ok]] = False
        w = torch.where(ok[:, None], 1/t.var, torch.zeros_like(t.var)); r = torch.where(ok[:, None], t.r, torch.zeros_like(t.r))
        Hx = torch.where(ok[:, None, None], t.Hx, torch.zeros_like(t.Hx))
        up, lens = torch.unique_consecutive(p, return_counts=True)
        seg = lambda v: torch.segment_reduce(v, 'sum', lengths=lens)
        out['Ixx'][up] += seg((Hx[:, :, :, None]*w[:, :, None, None]*Hx[:, :, None, :]).sum(1))
        for ch in range(4):
            out['gx'][ch, up] += seg(Hx[:, ch]*(w*r)[:, ch, None]); out['q'][ch, up] += seg(w[:, ch]*r[:, ch]**2)
        for ch in range(3):
            out['ga'][ch, up, ch] += seg(-w[:, ch]*r[:, ch])
        out['Ixa'][up] += seg(-(Hx[:, :3, :]*w[:, :3, None]).transpose(1, 2)); out['Iaa'][up] += seg(w[:, :3])
        out['nrows'][up] += seg(ok.to(dt)).round().long()
    return out


def verify(M, frame, X, a, Sigma, pts, support, src, X_lin, got, tag):
    """The channel sums must reproduce the solver's evaluate_frame (rows, information, gradient, objective)."""
    from estimator.ieks_point_trajectory import evaluate_frame
    ev = evaluate_frame(M, frame, X, a, Sigma, pts=pts, support=support, src=src, X_lin=X_lin)
    chk = {'nrows': (ev['nrows'], got['nrows']), 'Ixx': (ev['Ixx'], got['Ixx']), 'gx': (ev['gx'], got['gx'].sum(0)), 'q': (ev['q'], got['q'].sum(0))}
    if M.latent: chk.update(Ixa=(ev['Ixa'], got['Ixa']), Iaa=(ev['Iaa'], got['Iaa']), ga=(ev['ga'], got['ga'].sum(0)))
    if support is None: chk['support'] = (ev['support'], got['support'])
    for k, (e, g) in chk.items():
        if not torch.allclose(e.to(torch.float64), g.to(torch.float64), rtol=1e-9, atol=1e-12*max(float(e.abs().max()) if e.numel() else 0., 1.)):
            raise AssertionError(f'{tag}: channel split of {k} differs from evaluate_frame')


def rebuild(run, scene, device):
    """Dataset, TemporalModel and catalogue exactly as train.py / run_temporal_stage built them (R from the manifest)."""
    from scene.dataset import Dataset
    from scene.config import load_config
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import SensorNoise
    from estimator.ieks_point_trajectory import TemporalModel, FrameStore
    man = json.loads((run/'run_manifest.json').read_text()); tm = json.loads((run/'free_forward_temporal'/'run_manifest.json').read_text())
    a = man['arguments']
    data = Dataset(Path(scene or a['scene']), a.get('frame_limit'), None, fps=a.get('fps'), dt_seconds=a.get('dt_seconds'), timestamps_file=a.get('timestamps_file'))
    config = load_config(a.get('noise_config'))
    nz = man['noise']                                                     # the configuration AFTER the measured-R calibration
    config.color_covariance = np.asarray(nz['color_covariance'], float)
    config.depth_variance, config.depth_squared_variance = float(nz['depth_variance']), float(nz['depth_squared_variance'])
    policy = man['policy']
    noise = SensorNoise.from_config(config, policy, device)
    R = tm['R']
    same = (np.allclose(noise.rgb_variance.cpu().numpy(), R['rgb_sensor_variance'], rtol=0, atol=0) and noise.depth_variance == R['depth_variance_m2'] and
            noise.depth_squared_variance == R['depth_squared_variance'] and float(policy['rgb_model_variance']) == R['rgb_model_variance'] and
            float(policy['depth_model_variance']) == R['depth_model_variance_m2'])
    if not same: raise SystemExit('rebuilt R differs from the run manifest: refusing to attribute with a different measurement model')
    tcfg = tm['temporal_config']
    if isinstance(tcfg, str): tcfg = json.loads(tcfg)
    rig = CameraRig(data.cameras, device, torch.float64)
    M = TemporalModel(rig, noise, policy, tcfg, device)
    K = len(data.times)
    if K != tm['frames']: raise SystemExit(f'scene has {K} frames, the run {tm["frames"]}')
    return data, M, FrameStore(data, list(range(K)), device), tm


def summarise(per, m):
    if not m.any(): return {'points': 0}
    out = {'points': int(m.sum())}
    sh = per['onset_share'][m]
    okA = np.isfinite(sh).all(1)
    out['A_onset_push_at_rest'] = {'points_with_rows_at_rest': int(okA.sum()),
                                   'step_norm_at_W_mm_median': float(np.median(per['onset_step_norm_mm'][m][okA])) if okA.any() else None,
                                   **{c: {'share_median': float(np.median(sh[okA, i])), 'share_mean': float(np.mean(sh[okA, i])),
                                          'dominant_fraction': float(np.mean(np.argmax(sh[okA], 1) == i)),
                                          'own_step_norm_at_W_mm_median': float(np.median(per['onset_channel_step_norm_mm'][m][okA, i]))}
                                      for i, c in enumerate(CHANNELS)}} if okA.any() else None
    dJ = per['dJ'][m]; okB = per['B_valid'][m]; pref = okB & (dJ.max(1) > 0)        # largest-of: points where some channel prefers drift
    out['B_objective_preference_final_rows'] = {'points_valid_at_rest': int(okB.sum()), 'points_some_channel_prefers_drift': int(pref.sum()),
                                                'prior_dynamics_dJ_sum': float(per['dJ_prior_dyn'][m][okB].sum()),
                                                **{c: {'dJ_sum': float(dJ[okB, i].sum()), 'dJ_median': float(np.median(dJ[okB, i])),
                                                       'prefers_drift_fraction': float(np.mean(dJ[okB, i] > 0)),
                                                       'largest_preference_fraction': float(np.mean(np.argmax(dJ[pref], 1) == i)) if pref.any() else None}
                                                   for i, c in enumerate(CHANNELS)}} if okB.any() else None
    f = per['force'][m]; push = f.max(1) > 0
    out['C_force_along_drift_final'] = {'points_some_channel_pushes_away': int(push.sum()),
                                        **{c: {'pushes_away_fraction': float(np.mean(f[:, i] > 0)), 'force_sum': float(f[:, i].sum()),
                                               'largest_push_fraction': float(np.mean(np.argmax(f[push], 1) == i)) if push.any() else None}
                                           for i, c in enumerate(CHANNELS)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='train.py run folder holding free_forward_temporal/')
    ap.add_argument('--basis-from', type=Path, required=True, help='audit baseline run holding static_audit/static_basis.npz + selection.json')
    ap.add_argument('--out', type=Path, required=True, help='new folder (never overwritten)')
    ap.add_argument('--scene', type=Path, help='scene folder (default: the run manifest\'s)')
    ap.add_argument('--window', type=int, help='frames 0..W (real scenes: the static-proxy window); default all')
    ap.add_argument('--max-points', type=int, help='evenly spaced subset of the static points (default all)')
    ap.add_argument('--verify-frames', type=int, default=2, help='frames whose channel split is checked against evaluate_frame')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.out.exists(): raise SystemExit(f'{a.out} exists: never overwritten')
    if not (a.run/'free_forward_temporal'/'status.json').exists():
        raise SystemExit(f'{a.run} holds no finished --free-solver temporal-ieks run (free_forward_temporal/status.json missing): run train.py first')
    t0 = time.perf_counter()
    from estimator.ieks_point_trajectory import TrajectoryProblem, solve_chunk, point_chunks
    d = a.run/'free_forward_temporal'
    sel = json.loads((a.basis_from/'static_audit'/'selection.json').read_text())
    data, M, frames, tm = rebuild(a.run, a.scene, a.device)
    if tm['selected_ids_sha256'] != sel['selected_ids_sha256']: raise SystemExit('point ids differ from the static basis run')
    dev, dt = M.device, M.dtype
    static = np.load(a.basis_from/'static_audit'/'static_basis.npz')['static']
    si = np.flatnonzero(static)
    if a.max_points and len(si) > a.max_points: si = si[np.linspace(0, len(si)-1, a.max_points).round().astype(int)]
    S = torch.as_tensor(np.load(d/'smooth_mean.npy'), dtype=dt)               # [K,N,6] final trajectory
    K, N = S.shape[:2]; C = M.rig.count
    W = K-1 if a.window is None else min(a.window, K-1)
    ap_ = np.load(d/'anchor_prior.npz'); can = np.load(d/'canonical.npz')
    x_ref = torch.as_tensor(ap_['x_ref'], dtype=dt); P_x0 = torch.as_tensor(ap_['P_x0'], dtype=dt); P_v0 = torch.as_tensor(ap_['P_v0'], dtype=dt)
    src = torch.as_tensor(ap_['source_frame_camera_pixel'][:, 1], dtype=torch.long)
    rgb0 = torch.as_tensor(can['canonical_rgb'], dtype=dt)
    a_fin = torch.as_tensor(np.load(d/'latent_colour.npy'), dtype=dt) if M.latent else rgb0
    F, Q = torch.as_tensor(np.load(d/'transition_F.npy')), torch.as_tensor(np.load(d/'process_Q.npy'))
    Pf = np.load(d/'filter_cov.npy', mmap_mode='r')
    reason = np.load(d/'camera_reason.npy', mmap_mode='r')                   # [K,C,N] final linearization, 0 = row used
    n = len(si); idx_d = torch.as_tensor(si, device=dev)
    pts = torch.zeros(N, dtype=torch.bool, device=dev); pts[idx_d] = True
    S0 = torch.zeros((K, n, 6), dtype=dt); S0[..., :3] = x_ref[si][None]       # rest hypothesis of the analysed points
    acc = {k: torch.zeros((K, n, *s), dtype=dt) for k, s in (('Ixx', (3, 3)), ('Ixa', (3, 3)), ('Iaa', (3,)))}
    acc.update(gx=torch.zeros((4, K, n, 3), dtype=dt), ga=torch.zeros((4, K, n, 3), dtype=dt))
    rows_rest = torch.zeros((K, n), dtype=torch.int16)
    qB0, qB1 = torch.zeros((4, n), dtype=dt), torch.zeros((4, n), dtype=dt)
    validB = torch.ones(n, dtype=torch.bool); force = torch.zeros((4, n), dtype=dt)
    Xfin_all = S[..., :3]
    for k in range(K):
        frame = frames.get(k)
        if k == 0: Sig = P_x0.to(dev)
        else:
            P = torch.as_tensor(np.asarray(Pf[k-1]), dtype=dt).to(dev); Fk, Qk = F[k-1].to(dev), Q[k-1].to(dev)
            Sig = (Fk@P@Fk.T+Qk)[:, :3, :3]; Sig = (Sig+Sig.transpose(1, 2))/2
        # A: production admission at rest (others at their final trajectory)
        X = Xfin_all[k].clone().to(dev); X[idx_d] = x_ref[si].to(dev)
        a_rest = a_fin.clone().to(dev); a_rest[idx_d] = rgb0[si].to(dev)
        s = src.to(dev) if k == 0 else None
        o = channel_frame(M, frame, X, a_rest, Sig, pts, src=s)
        if k < a.verify_frames: verify(M, frame, X, a_rest, Sig, pts, None, s, None, o, f'A frame {k}')
        for key in ('Ixx', 'Ixa', 'Iaa'): acc[key][k] = o[key][idx_d].cpu()
        acc['gx'][:, k] = o['gx'][:, idx_d].cpu(); acc['ga'][:, k] = o['ga'][:, idx_d].cpu(); rows_rest[k] = o['nrows'][idx_d].cpu().to(torch.int16)
        if k <= W:
            sup = torch.as_tensor(np.asarray(reason[k]) == 0, device=dev)
            Xf = Xfin_all[k].to(dev); X0 = Xf.clone(); X0[idx_d] = x_ref[si].to(dev); af = a_fin.to(dev)
            b0 = channel_frame(M, frame, X0, af, Sig, pts, support=sup, X_lin=Xf)
            b1 = channel_frame(M, frame, Xf, af, Sig, pts, support=sup, X_lin=Xf)
            if k < a.verify_frames:
                verify(M, frame, X0, af, Sig, pts, sup, None, Xf, b0, f'B rest frame {k}'); verify(M, frame, Xf, af, Sig, pts, sup, None, Xf, b1, f'B final frame {k}')
            qB0 += b0['q'][:, idx_d].cpu(); qB1 += b1['q'][:, idx_d].cpu(); validB &= b0['valid'][idx_d].cpu()
            dx = (Xf[idx_d]-x_ref[si].to(dev))
            force += (b1['gx'][:, idx_d]*dx[None]).sum(-1).cpu()
        del frame
        if k % 10 == 0 or k == K-1: print(f'[attribution] frame {k+1}/{K} ({time.perf_counter()-t0:.0f} s)', flush=True)
    # A: the Gauss-Newton step at rest, one per channel (same information, the channel's gradient only)
    mean0 = torch.cat([x_ref[si], torch.zeros((n, 3), dtype=dt)], 1)
    cov0 = torch.zeros((n, 6, 6), dtype=dt); cov0[:, :3, :3] = P_x0[si]; cov0[:, 3:, 3:] = P_v0
    Pr = TrajectoryProblem(mean0, cov0, F, Q, rgb0[si], M.sigma_a2.cpu(), M.latent)
    steps = torch.zeros((5, K, n, 6), dtype=dt)                              # 4 channels + all
    for sl in point_chunks(n, K):
        ii = torch.arange(sl.start, sl.stop)
        for j in range(5):
            fac = {'Ixx': acc['Ixx'], 'Ixa': acc['Ixa'], 'Iaa': acc['Iaa'], 'q': torch.zeros((K, n), dtype=dt),
                   'gx': acc['gx'].sum(0) if j == 4 else acc['gx'][j], 'ga': acc['ga'].sum(0) if j == 4 else acc['ga'][j]}
            r = solve_chunk(Pr, S0[:, sl].to(dev), rgb0[si][sl].to(dev), fac, ii, dev)
            steps[j, :, sl] = r['dT'].cpu()
    add_err = float((steps[:4].sum(0)-steps[4]).abs().max()/max(float(steps[4].abs().max()), 1e-300))
    if add_err > 1e-8: raise AssertionError(f'channel steps do not add up to the full step (relative {add_err:.2e})')
    px = steps[..., :W+1, :, :3].permute(0, 2, 1, 3).reshape(5, n, -1)        # positions of frames 0..W
    den = (px[4]*px[4]).sum(-1)
    share = ((px[:4]*px[4][None]).sum(-1)/torch.where(den > 0, den, torch.ones_like(den))).T.numpy()
    share[(den <= 0).numpy()] = np.nan
    # B: prior + dynamics change over frames 0..W (windowed objective) and per-channel data change
    pd = lambda T: Pr.prior_dyn(T[:W+1].to(dev), torch.arange(n), dev)[0].cpu() if W+1 == K else _windowed_prior_dyn(Pr, T, W, dev)
    dJ_pd = pd(S0)-pd(S[:, si])
    dJ = (.5*(qB0-qB1)).T.numpy()
    err = np.linalg.norm(S[W, si, :3].numpy()-x_ref[si].numpy(), axis=-1)
    per = {'point_id': si, 'final_error_m': err, 'onset_share': share, 'onset_step_norm_mm': 1e3*steps[4, W, :, :3].norm(dim=-1).numpy(),
           'onset_channel_step_norm_mm': 1e3*steps[:4, W, :, :3].norm(dim=-1).T.numpy(), 'onset_rows_at_rest': rows_rest.numpy(),
           'dJ': dJ, 'dJ_prior_dyn': dJ_pd.numpy(), 'B_valid': validB.numpy(), 'force': force.T.numpy()}
    groups = {'all_static': np.ones(n, bool), 'drift_over_1cm': err > .01, 'drift_over_10cm': err > .10, 'stable_under_1mm': err < 1e-3}
    rep = {'run': str(a.run), 'solver': 'temporal-ieks', 'channels': list(CHANNELS), 'frames': K, 'window_frame_W': W, 'static_points_analysed': n,
           'static_basis': sel['static_basis']['rule'], 'kind': sel['kind'],
           'notes': {'A': 'Gauss-Newton step from rest (x = x_ref, v = 0, source colour), production admission, others at their final trajectory; '
                          'gate covariance P-_k rebuilt from filter_cov (stored float32, computed float64 in the run)',
                     'B': 'final rows frozen, R frozen at the final trajectory, final latent colour in both states (colour re-estimation at rest not included)',
                     'C': 'descent direction of each channel at the final trajectory projected on x_k - x_ref, frames 0..W',
                     'verification': f'channel sums equal evaluate_frame on the first {a.verify_frames} frames of every pass; channel steps add to the full step (rel {add_err:.1e})'},
           'groups': {g: summarise(per, m) for g, m in groups.items()}, 'seconds': time.perf_counter()-t0}
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'per_point.npz', **per)
    (a.out/'attribution.json').write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({g: rep['groups'][g] for g in ('all_static', 'drift_over_10cm')}, default=float)[:4000])


def _windowed_prior_dyn(Pr, T, W, dev):
    """Prior + dynamics objective of frames 0..W only (the window's own posterior terms)."""
    T = T[:W+1].to(dev); n = T.shape[1]
    mean0, P0inv = Pr.mean0.to(dev), Pr.P0inv.to(dev)
    e0 = T[0]-mean0; J = .5*(e0*(P0inv@e0[..., None])[..., 0]).sum(-1)
    if W > 0 and not Pr.reduced:
        F, Qi = Pr.F[:W].to(dev), Pr.Qi[:W].to(dev)
        e = T[1:]-(F[:, None]@T[:-1][..., None])[..., 0]
        J = J+.5*(e*(Qi[:, None]@e[..., None])[..., 0]).sum((0, 2))
    return J.cpu()


if __name__ == '__main__':
    main()
