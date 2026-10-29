"""F1 / B1: one free directional filter of one measured endpoint cloud (contract sections 1, 3, 7.2, 9.2).

forward : canonical P_0 (measured at frame 0), processing order 0 -> T
backward: canonical P_T (measured at frame T), processing order T -> 0 (signed dt < 0)
The two directions share no point ids, no state and no covariance. Disk arrays are indexed by ORIGINAL chronological
frame: free_backward/posterior_xyz.npy[0] holds the frame-0 estimate of the frame-T canonical points.

Per frame: load the 8 actual masked RGB-D images to the GPU, predict (point_kalman_cuda.predict), fixed-prior IEKF
(point_kalman_cuda.update_points), store. Only the current state lives on the GPU; trajectories go to memory maps.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scene.endpoint import endpoint_samples
from scene.canonical_subsample import subsample, boundary_priority, ids_hash
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise, REASONS
from estimator.point_kalman_cuda import predict, update_points, Timers, eigvalsh3, spd_extreme_eigs, ACCEPTANCE, transition, process_noise
from utils.io import json_write, serial
from estimator.rendered_mask_measurement import footprints, mask_rows, hard_pseudomask, frame_statistics
from estimator.rendered_mask_iekf import update_points_rendered_mask, MASK_STATUS, MCODE

QUANT = (0., .05, .25, .5, .75, .95, 1.)
STATUS = {0: 'ANCHOR_CANONICAL', 1: 'MEASURED', 2: 'PREDICTION_ONLY'}


def quantiles(x):
    x = x.reshape(-1)
    x = x[torch.isfinite(x)] if torch.is_floating_point(x) else x
    if x.numel() == 0: return None
    return torch.quantile(x.double()[torch.randperm(x.numel(), device=x.device, generator=torch.Generator(device=x.device).manual_seed(0))[:5_000_000]] if x.numel() > 5_000_000 else x.double(),
                          torch.tensor(QUANT, dtype=torch.float64, device=x.device)).tolist()


def build_catalogue(dataset, config, direction, sub, policy, noise):
    anchor = 'first' if direction == 'forward' else 'last'
    k0, raws, xyz, rgb, cov, source = endpoint_samples(dataset, anchor, config)
    t0 = time.perf_counter()
    bflag = grad = None
    if sub['factor'] > 1 and sub['method'] == 'spatial-stratified' and sub['boundary_preserve']:
        bflag, grad = boundary_priority(dataset, source, k0, policy, noise)
    sel, man = subsample(xyz, source, sub['factor'], sub['method'], sub['seed'], bflag, grad, sub['report'], [c.width for c in dataset.cameras])
    man['seconds'] = time.perf_counter()-t0
    man['selected_ids_sha256'] = ids_hash(sel)
    return k0, xyz, rgb, cov, source, sel, man


def run_free(dataset, config, policy, direction, root, device, sub, precision='float64', max_iters=10, tol=1e-8, chunk=200_000,
             debug_frames=(), debug_point_ids=None, save_full_covariance=False, save_state_interval=0, progress=None, assignment='per-frame',
             mask_noise=None):
    root = Path(root); (root/'numeric').mkdir(parents=True, exist_ok=True)
    kal_dtype = torch.float32 if precision == 'float32' else torch.float64
    rig = CameraRig(dataset.cameras, device, torch.float64 if precision == 'float64' else torch.float32)
    noise = SensorNoise.from_config(config, policy, device)
    k0, xyz, rgb, cov, source, sel, sub_man = build_catalogue(dataset, config, direction, sub, policy, noise)
    K, n, C = len(dataset.times), len(sel), rig.count
    order = list(range(K)) if direction == 'forward' else list(range(K-1, -1, -1))
    if order[0] != k0: raise AssertionError('processing must start at the canonical endpoint')
    np.savez_compressed(root/'canonical.npz', canonical_frame=k0, canonical_xyz=xyz[sel], canonical_rgb=rgb[sel],
                        original_source_ids=source[sel], point_id=np.arange(n), subsampled_to_full_index=sel,
                        canonical_source_covariance=cov[sel], full_endpoint_xyz=xyz, full_endpoint_rgb=rgb, full_endpoint_source=source)
    json_write(root/'subsample_manifest.json', serial(sub_man))
    if debug_point_ids is None:
        debug_point_ids = sorted({int(i) for i in np.linspace(0, n-1, 6).round()})
    debug_point_ids = [int(i) for i in debug_point_ids if 0 <= int(i) < n]
    # chronological memory maps
    mm = {}
    for name, shape, dt in [('posterior_xyz', (K, n, 3), 'float64'), ('predicted_xyz', (K, n, 3), 'float32'), ('velocity', (K, n, 3), 'float32'),
                            ('position_variance_diag', (K, n, 3), 'float32'), ('velocity_variance_diag', (K, n, 3), 'float32'),
                            ('valid_cameras', (K, n), 'uint8'), ('point_status', (K, n), 'uint8'), ('camera_reason', (K, C, n), 'uint8'),
                            ('converged', (K, n), 'bool'), ('iekf_iterations', (K, n), 'uint8'), ('nis', (K, n), 'float32'), ('nis_dof', (K, n), 'uint8'),
                            ('information_min_eig', (K, n), 'float32'), ('information_max_eig', (K, n), 'float32'),
                            ('update_acceptance', (K, n), 'uint8'), ('posterior_cost', (K, n), 'float32'), ('posterior_cost_dof', (K, n), 'uint8')]:
        mm[name] = np.lib.format.open_memmap(root/f'{name}.npy', mode='w+', dtype=dt, shape=shape)
    mcfg = policy.get('rendered_mask_measurement', {'mode': 'off'})
    mmode = mcfg['mode']
    if mmode not in ('off', 'diagnostic', 'on'): raise ValueError(f'rendered_mask_measurement.mode {mmode!r}')
    mvar = None
    if mask_noise is not None:
        mvar = np.broadcast_to(np.asarray(mask_noise['mask_image_variance'], float), (C,)).copy()
    if mmode == 'on' and mvar is None: raise ValueError('rendered_mask_measurement on needs a declared mask_image_variance (--mask-image-noise-config)')
    if mmode != 'off':                                   # new arrays only when the mask channel is evaluated: 'off' writes exactly the old outputs
        for name, shape, dt in [('rendered_mask_status', (K, n), 'uint8'), ('rendered_mask_rows', (K, n), 'uint16'),
                                ('rendered_mask_gradient_norm', (K, n), 'float32'), ('rendered_mask_outside_cameras_prior', (K, n), 'uint8'),
                                ('rendered_mask_outside_cameras_post', (K, n), 'uint8'), ('rendered_mask_rgbd_consistent', (K, n), 'bool')]:
            mm[name] = np.lib.format.open_memmap(root/f'{name}.npy', mode='w+', dtype=dt, shape=shape)
        mm['rendered_mask_status'][k0] = MCODE['NOT_EVALUATED']
    dev = torch.device(device)
    s = torch.zeros((n, 6), dtype=kal_dtype, device=dev); s[:, :3] = torch.as_tensor(xyz[sel], dtype=kal_dtype, device=dev)
    P = torch.zeros((n, 6, 6), dtype=kal_dtype, device=dev)
    P[:, :3, :3] = torch.as_tensor(cov[sel], dtype=kal_dtype, device=dev)
    P[:, 3:, 3:] = torch.as_tensor(np.asarray(config.velocity_covariance), dtype=kal_dtype, device=dev)
    rgb0 = torch.as_tensor(rgb[sel], dtype=kal_dtype, device=dev)
    qc = torch.as_tensor(np.asarray(config.qc), dtype=kal_dtype, device=dev)
    _store(mm, k0, s, s, P, None, C)
    mm['point_status'][k0] = 0
    manifest = {'pass': 'F1' if direction == 'forward' else 'B1', 'direction': direction, 'canonical_frame': k0,
                'canonical_cloud': f'measured frame-{k0} endpoint (all cameras, hard foreground, valid metric depth)',
                'processing_order': order, 'original_frame_indices': list(dataset.frame_ids), 'timestamps_seconds': np.asarray(dataset.times).tolist(),
                'points_selected': n, 'points_full_endpoint': int(len(xyz)), 'cameras': C, 'camera_geometry_hash': rig.calibration_hash(),
                'subsample_manifest': 'subsample_manifest.json', 'selected_ids_sha256': sub_man['selected_ids_sha256'],
                'state': 'independent [x (m, world), v (m/s, physical time)] per point, 6x6 covariance per point; no group, no shared increment',
                'prior': {'position': 'measured endpoint back-projection covariance (scene/endpoint.py)', 'velocity_mean': 0.,
                          'velocity_covariance_m2_s2': np.asarray(config.velocity_covariance).tolist(), 'q_c_m2_s3': np.asarray(config.qc).tolist(),
                          'noise_provenance': {k: config.provenance.get(k) for k in ('qc', 'velocity_covariance', 'color_covariance', 'depth_variance')}},
                'measurement': 'projective sampling of the ACTUAL masked RGB-D frames, r = [a - I(u,v); Z - D(u,v)], bilinear, exact interpolant gradient',
                'precision': precision, 'pixel_reuse_assignment': assignment, 'max_iekf_iters': max_iters, 'step_tolerance': tol, 'point_chunk_size': chunk,
                'debug_point_ids': debug_point_ids, 'debug_frames': list(debug_frames), 'arrays_indexing': 'original chronological frame index',
                'full_covariance_saved': bool(save_full_covariance), 'save_state_interval': save_state_interval, 'reason_codes': REASONS, 'status_codes': STATUS,
                'acceptance_codes': ACCEPTANCE, 'update_acceptance': policy.get('update_acceptance', 'final_iterate'),
                'row_innovation_gate': policy.get('row_innovation_gate', 'depth_front_only'), 'final_visibility': policy.get('final_visibility', 'frozen'),
                'rendered_mask_measurement': {**mcfg, 'status_codes': MASK_STATUS, 'mask_noise': mask_noise,
                                              'observation': 'y_obs = the measured binary mask of the frame; y_hat = the rendered white-point pseudomask of ALL '
                                                             'transported points; r = y_obs - y_hat on measured-background pixels only',
                                              'acceptance': 'rendered_mask_joint_monotone_line_search' if mmode == 'on' else policy.get('update_acceptance', 'final_iterate')}}
    json_write(root/'run_manifest.json', serial(manifest))
    timeline = []; gtot = {}
    total_t0 = time.perf_counter()
    for i, (prev, k) in enumerate(zip(order, order[1:])):
        t_wall = time.perf_counter()
        timers = Timers(device)
        if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        with timers('transfer_images'):
            frame = MeasuredFrame(dataset, k, device)
        dt = float(dataset.times[k]-dataset.times[prev])
        with timers('predict'):
            s_m, P_m = predict(s, P, dt, qc)
        trace_ids = debug_point_ids if k in debug_frames else None
        pre = None if trace_ids is None else {'previous_frame': prev, 't_previous_s': float(dataset.times[prev]), 't_s': float(dataset.times[k]), 'dt_s': dt,
                                              'F': transition(dt).tolist(), 'Q': process_noise(dt, qc.double().cpu()).tolist(), 'q_c': qc.double().cpu().tolist(),
                                              'points': {pid: {'s_previous': s[pid].tolist(), 'P_previous': P[pid].tolist(), 's_minus': s_m[pid].tolist(),
                                                               'P_minus': P_m[pid].tolist()} for pid in trace_ids}}
        if mmode == 'on':
            s, P, rep, trace = update_points_rendered_mask(s_m, P_m, frame, rig, rgb0, noise, policy, max_iters, tol, chunk, kal_dtype, trace_ids, timers,
                                                           assignment, mvar)
        else:
            s, P, rep, trace = update_points(s_m, P_m, frame, rig, rgb0, noise, policy, max_iters, tol, chunk, kal_dtype, trace_ids, timers, assignment)
        if trace is not None: trace['prediction'] = pre
        msum = None
        if mmode != 'off':
            with timers('mask_diagnostics'):
                msum = _store_mask(mm, k, s_m, s, rep, frame, rig, mvar, mmode, kal_dtype, root if k in debug_frames else None)
        with timers('store'):
            _store(mm, k, s, s_m, P, rep, C)
            if save_full_covariance and (k in debug_frames or (save_state_interval and i % save_state_interval == 0) or k == order[-1]):
                (root/'covariance').mkdir(exist_ok=True); np.save(root/'covariance'/f'frame_{k:04d}.npy', P.double().cpu().numpy())
        times = timers.resolve()
        num = _frame_numeric(k, i+1, dt, frame, rep, s, s_m, P, P_m, times, time.perf_counter()-t_wall, dev, dataset)
        if msum is not None:
            num['rendered_mask'] = msum
            for kk, v in msum['status_counts'].items(): gtot[kk] = gtot.get(kk, 0)+v
            for kk in ('hard_white_outside_measured_mask_prior', 'hard_white_outside_measured_mask_post'):
                gtot[kk] = gtot.get(kk, 0)+msum[kk]
        json_write(root/'numeric'/f'frame_{k:04d}.json', serial(num))
        if trace is not None: json_write(root/'numeric'/f'trace_frame_{k:04d}.json', serial({'frame': k, 'points': trace}))
        timeline.append(_timeline_row(num))
        if progress: progress(direction, i+1, K-1, num)
        del frame
    for a in mm.values(): a.flush()
    json_write(root/'final_state.json', {'note': 'final GPU state is also saved as final_state.npz'})
    np.savez_compressed(root/'final_state.npz', state=s.double().cpu().numpy(), covariance=P.double().cpu().numpy())
    _write_timeline(root/'numeric'/'timeline.csv', timeline)
    status = {'pass': manifest['pass'], 'status': 'COMPLETE', 'frames_processed': K-1, 'points': n, 'seconds': time.perf_counter()-total_t0,
              'not_converged_point_frames': int(sum(r['not_converged_points'] for r in timeline)),
              'rejected_inconsistent_point_frames': int(sum(r['rejected_inconsistent'] for r in timeline)),
              'accepted_earlier_iterate_point_frames': int(sum(r['accepted_earlier_iterate'] for r in timeline)),
              'prediction_only_point_frames': int(sum(r['prediction_only_points'] for r in timeline)),
              'rendered_mask': None if mmode == 'off' else {'mode': mmode, 'point_frame_totals': gtot}}
    json_write(root/'status.json', status)
    return status


def _store_mask(mm, k, s_m, s, rep, frame, rig, mvar, mmode, dtype, debug_root):
    """Mask-channel arrays and summary of frame k: the hard pseudomask outside the measured mask at s- and at the
    published s+ (all points, white once), per-point status / rows / gradient (on) and debug images on debug frames.
    The direct RGB-D fields (observed, nrows, camera_reason) are untouched."""
    C, H, W, N = rig.count, frame.H, frame.W, len(s)
    background = ~frame.mask
    var = torch.as_tensor(mvar if mvar is not None else np.ones(C), dtype=dtype, device=s.device)
    out = {'mode': mmode}
    per = {}
    for tag, X in (('prior', s_m[:, :3].to(dtype)), ('post', s[:, :3].to(dtype))):
        if tag == 'post' and mmode == 'on':                          # the IEKF's own final rendering of the published state
            fp, rows, B = rep['mask']['footprints'], rep['mask']['rows'], rep['mask']['hard_pseudomask']
        else:
            fp = footprints(X, rig, H, W, background); rows = mask_rows(fp, background, var, H, W, N); B = hard_pseudomask(fp, C, H, W)
        st = frame_statistics(fp, rows, background, B, H, W)
        if mvar is None: st['mask_objective'] = None
        cam_bg = background.reshape(C, -1).gather(1, fp.nearest) & fp.in_image          # [C,N] nearest pixel white in B and black in M_obs
        mm[f'rendered_mask_outside_cameras_{tag}'][k] = cam_bg.sum(0).clamp_max(255).to(torch.uint8).cpu().numpy()
        out[f'hard_white_outside_measured_mask_{tag}'] = st['hard_white_outside_measured_mask']
        per[tag] = (fp, rows, B, st)
        out[f'statistics_{tag}'] = st
    if mmode == 'on':
        mk = rep['mask']
        status = mk['status']
        mm['rendered_mask_rows'][k] = mk['rows_per_point'].clamp_max(65535).to(torch.int32).cpu().numpy().astype(np.uint16)
        mm['rendered_mask_gradient_norm'][k] = mk['gradient_norm'].float().cpu().numpy()
        mm['rendered_mask_rgbd_consistent'][k] = mk['rgbd_consistent'].cpu().numpy()
        out['iekf'] = {kk: v for kk, v in mk['statistics'].items()}
        out['history'] = rep['history']
    else:
        fp, rows, B, st = per['post']
        rp = torch.zeros(N, dtype=torch.long, device=s.device)
        if rows.n_rows: rp[rows.point_ids] = rows.point_len
        status = torch.where(rp > 0, MCODE['DIAGNOSTIC_ONLY'], MCODE['NO_MASK_ROWS'])
        mm['rendered_mask_rows'][k] = rp.clamp_max(65535).to(torch.int32).cpu().numpy().astype(np.uint16)
        mm['rendered_mask_gradient_norm'][k] = (rows.jtw(rows.r/rows.var, N).norm(dim=1) if rows.n_rows else torch.zeros(N, dtype=dtype, device=s.device)).float().cpu().numpy()
        mm['rendered_mask_rgbd_consistent'][k] = False
    mm['rendered_mask_status'][k] = status.to(torch.uint8).cpu().numpy()
    counts = torch.bincount(status.long(), minlength=len(MASK_STATUS)).cpu().numpy()
    out['status_counts'] = {MASK_STATUS[i]: int(counts[i]) for i in range(len(MASK_STATUS)) if counts[i]}
    if debug_root is not None:
        out['debug_images'] = _mask_debug_images(debug_root, k, frame, per)
    return out


def _mask_debug_images(root, k, frame, per):
    """mask_debug/frame_KKKK_camC.png per camera, panels left to right: measured mask M_obs | hard pseudomask B of ALL
    points (prior s-) | red B & ~M_obs on gray M_obs (prior) | the same at the published s+ | anti-aliased m_hat (s+) |
    selected residual -(1 - M_obs) m_hat (s+, blue = more negative) | contributor count per pixel (s+, all pixels)."""
    from PIL import Image
    out = root/'mask_debug'; out.mkdir(exist_ok=True)
    C, H, W = frame.mask.shape
    M = frame.mask.cpu().numpy()
    fp0, rows0, B0, _ = per['prior']; fp1, rows1, B1, _ = per['post']
    B0, B1 = B0.cpu().numpy(), B1.cpu().numpy()
    mh = np.zeros(C*H*W); cnt = np.zeros(C*H*W)
    if rows1.n_rows: mh[(rows1.row_cam*H*W+rows1.row_pix).cpu().numpy()] = rows1.m_hat.cpu().numpy()
    c_idx = torch.arange(C, device=fp1.nearest.device)[:, None].expand_as(fp1.nearest)
    np.add.at(cnt, (c_idx*H*W+fp1.nearest)[fp1.in_image].cpu().numpy(), 1)          # points per pixel (nearest centre, all points)
    mh, cnt = mh.reshape(C, H, W), cnt.reshape(C, H, W)
    names = []
    for c in range(C):
        gray = lambda a: np.repeat((a*255).astype(np.uint8)[..., None], 3, -1)
        def over(B):
            img = gray(M[c]*.5); img[B[c] & ~M[c]] = (255, 0, 0); return img
        res = np.zeros((H, W, 3), np.uint8); res[..., 2] = (np.clip(mh[c]*(~M[c]), 0, 1)*255).astype(np.uint8)
        cn = gray(np.clip(cnt[c]/max(cnt[c].max(), 1), 0, 1))
        panel = np.concatenate([gray(M[c]), gray(B0[c]), over(B0), over(B1), gray(np.clip(mh[c], 0, 1)), res, cn], 1)
        name = f'frame_{k:04d}_cam{c}.png'; Image.fromarray(panel).save(out/name); names.append(f'mask_debug/{name}')
    return {'files': names, 'panels': ['M_obs', 'B(s-) all points white', 'B(s-) & ~M_obs red', 'B(s+) & ~M_obs red', 'm_hat(s+)',
                                       '-(1-M_obs) m_hat(s+)', 'points per pixel (nearest centre, s+)']}


def _store(mm, k, s, s_m, P, rep, C):
    mm['posterior_xyz'][k] = s[:, :3].double().cpu().numpy()
    mm['predicted_xyz'][k] = s_m[:, :3].float().cpu().numpy()
    mm['velocity'][k] = s[:, 3:].float().cpu().numpy()
    mm['position_variance_diag'][k] = P[:, :3, :3].diagonal(dim1=1, dim2=2).float().cpu().numpy()
    mm['velocity_variance_diag'][k] = P[:, 3:, 3:].diagonal(dim1=1, dim2=2).float().cpu().numpy()
    if rep is None:
        mm['camera_reason'][k] = 9                # anchor: no update at the canonical frame (NOT_PROCESSED)
        return
    mm['valid_cameras'][k] = rep['nrows'].clamp_max(255).to(torch.uint8).cpu().numpy()
    mm['point_status'][k] = np.where(rep['observed'].cpu().numpy(), 1, 2)
    mm['camera_reason'][k] = rep['reason'].cpu().numpy()
    mm['converged'][k] = rep['converged'].cpu().numpy()
    mm['iekf_iterations'][k] = rep['point_iterations'].clamp_max(255).to(torch.uint8).cpu().numpy()
    mm['nis'][k] = rep['nis'].float().cpu().numpy(); mm['nis_dof'][k] = rep['nis_dof'].clamp_max(255).to(torch.uint8).cpu().numpy()
    mm['information_min_eig'][k] = rep['info_eig'][:, 0].float().cpu().numpy(); mm['information_max_eig'][k] = rep['info_eig'][:, 2].float().cpu().numpy()
    mm['update_acceptance'][k] = rep['acceptance'].cpu().numpy()
    mm['posterior_cost'][k] = rep['posterior_cost'].float().cpu().numpy(); mm['posterior_cost_dof'][k] = rep['posterior_cost_dof'].clamp_max(255).to(torch.uint8).cpu().numpy()


def _frame_numeric(k, step, dt, frame, rep, s, s_m, P, P_m, times, wall, dev, dataset):
    C = frame.mask.shape[0]
    reason = rep['reason']
    counts = torch.stack([torch.bincount(reason[c].long(), minlength=len(REASONS)) for c in range(C)]).cpu().numpy()
    z_pos = (rep['proj'].z > 0).sum(1).cpu().numpy()
    obs = rep['observed']
    nisn = rep['nis']/rep['nis_dof'].clamp_min(1)
    disp = (s[:, :3]-s_m[:, :3]).norm(dim=1)
    dv = (s[:, 3:]-s_m[:, 3:]).norm(dim=1)
    acc = rep['acceptance']; acc_counts = torch.bincount(acc.long(), minlength=len(ACCEPTANCE)).cpu().numpy()
    cost_dof = rep['posterior_cost']/rep['posterior_cost_dof'].clamp_min(1)
    top = torch.argsort(torch.where(obs, disp, torch.zeros_like(disp)), descending=True)[:20]
    ev_pos_post, ev_vel_post = eigvalsh3(P[:, :3, :3].double()), eigvalsh3(P[:, 3:, 3:].double())
    ev_pos_prior = eigvalsh3(P_m[:, :3, :3].double())
    pd, lmin, lmax = spd_extreme_eigs(P.double())
    r0, c0, _ = rep['prior_innovation']; r1, c1 = rep['final_innovation']
    per_cam = []
    for c in range(C):
        q0 = r0[c0 == c]; q1 = r1[c1 == c]
        per_cam.append({'camera_index': c, 'camera': dataset.cameras[c].name, 'foreground_pixels': int(frame.mask[c].sum()),
                        'valid_foreground_pixels': int(frame.valid[c].sum()), 'projected_positive_z': int(z_pos[c]),
                        'reason_counts': {REASONS[i]: int(counts[c, i]) for i in range(len(REASONS)) if counts[c, i]},
                        'rows_accepted_final': int(counts[c, 0]),
                        'information_trace_final': float(rep['cam_information_trace'][c]),
                        'depth_innovation_m_quantiles_prior': quantiles(q0[:, 3]) if len(q0) else None,
                        'rgb_innovation_quantiles_prior': quantiles(q0[:, :3]) if len(q0) else None,
                        'depth_innovation_m_quantiles_final': quantiles(q1[:, 3]) if len(q1) else None,
                        'rgb_innovation_quantiles_final': quantiles(q1[:, :3]) if len(q1) else None})
    return {'frame': k, 'original_frame': dataset.frame_ids[k], 'processing_step': step, 'timestamp_s': float(dataset.times[k]), 'dt_s': dt,
            'sources': frame.sources, 'source_fingerprints': frame.fingerprints, 'host_to_gpu_bytes': frame.bytes,
            'processing': {'wall_ms': 1e3*wall, **times, 'gpu_peak_allocated_MB': torch.cuda.max_memory_allocated()/2**20 if dev.type == 'cuda' else None,
                           'gpu_peak_reserved_MB': torch.cuda.max_memory_reserved()/2**20 if dev.type == 'cuda' else None,
                           'iekf_iterations': rep['iterations'], 'termination': 'all_points_met_step_criterion' if bool(rep['converged'].all()) else 'max_iekf_iters',
                           'float64_retries': rep['float64_retry'],
                           'history': [{kk: (float(v) if torch.is_tensor(v) else v) for kk, v in h.items()} for h in rep['history']]},
            'sensor': {'points': int(len(s)), 'unique_points_observed': int(obs.sum()), 'prediction_only_points': int((~obs).sum()),
                       'rows_accepted_final': int(counts[:, 0].sum()), 'per_camera': per_cam},
            'acceptance': {'rule': rep['acceptance_rule'], 'counts': {ACCEPTANCE[i]: int(acc_counts[i]) for i in range(len(ACCEPTANCE)) if acc_counts[i]},
                           'posterior_cost_per_dof_quantiles_accepted': quantiles(cost_dof[obs]) if bool(obs.any()) else None,
                           'posterior_cost_per_dof_quantiles_rejected': quantiles(cost_dof[acc == 3]) if bool((acc == 3).any()) else None,
                           'accepted_not_converged_points': int((obs & ~rep['converged']).sum()),
                           'largest_accepted_corrections': [{'point_id': int(i), 'correction_m': float(disp[i]), 'velocity_change_m_s': float(dv[i]),
                                                             'acceptance': ACCEPTANCE[int(acc[i])], 'rows': int(rep['nrows'][i]), 'converged': bool(rep['converged'][i]),
                                                             'posterior_cost_per_dof': float(cost_dof[i])} for i in top.tolist() if bool(obs[i])],
                           'note': 'trace any of these ids with --debug-point-ids ID,.. --debug-frames FRAME (numeric/trace_frame_*.json: rows, S_zz, row NIS, gain, cost, acceptance)'},
            'innovation': {'nis_per_dof_quantiles_prior_observed': quantiles(nisn[rep['nis_dof'] > 0]),
                           'nis_note': 'NIS = r^T (H P- H^T + R)^-1 r at the prior, per point over its valid rows, divided by 4 x rows (expected 1 if consistent)'},
            'filter_update': {'not_converged_points': int((~rep['converged']).sum()),
                              'posterior_minus_prior_position_m_quantiles': quantiles(disp[obs]) if bool(obs.any()) else None,
                              'speed_m_s_quantiles': quantiles(s[:, 3:].norm(dim=1)),
                              'velocity_correction_m_s_quantiles': quantiles(dv[obs]) if bool(obs.any()) else None,
                              'prior_position_eig_m2_quantiles': quantiles(ev_pos_prior), 'posterior_position_eig_m2_quantiles': quantiles(ev_pos_post),
                              'posterior_velocity_eig_m2s2_quantiles': quantiles(ev_vel_post),
                              'posterior_positive_definite_failures': int((~pd).sum()), 'posterior_positive_definite_failure_ids': torch.nonzero(~pd).flatten()[:50].tolist(),
                              'posterior_min_eigenvalue_sampled': float(lmin.min()), 'posterior_condition_number_quantiles_sampled': quantiles(lmax/lmin),
                              'sampled_note': '6x6 eigenvalues on a deterministic evenly spaced sample of <= 16384 points; positive definiteness checked on ALL points',
                              'information_min_eig_quantiles_observed': quantiles(rep['info_eig'][obs, 0]) if bool(obs.any()) else None}}


def _timeline_row(n):
    p = n['processing']; f = n['filter_update']; sen = n['sensor']
    q = lambda v, i: None if v is None else v[i]
    return {'frame': n['frame'], 'processing_step': n['processing_step'], 'timestamp_s': n['timestamp_s'], 'dt_s': n['dt_s'], 'wall_ms': p['wall_ms'],
            'cuda_ms_total': sum(p['cuda_ms'].values()) if p['cuda_ms'] else None, 'gpu_peak_allocated_MB': p['gpu_peak_allocated_MB'],
            'iekf_iterations': p['iekf_iterations'], 'not_converged_points': f['not_converged_points'],
            'accepted_earlier_iterate': n['acceptance']['counts'].get('ACCEPTED_EARLIER_CONSISTENT_ITERATE', 0),
            'rejected_inconsistent': n['acceptance']['counts'].get('REJECTED_INCONSISTENT', 0),
            'observed_points': sen['unique_points_observed'], 'prediction_only_points': sen['prediction_only_points'], 'rows_accepted': sen['rows_accepted_final'],
            'nis_per_dof_median': q(n['innovation']['nis_per_dof_quantiles_prior_observed'], 3),
            'correction_m_median': q(f['posterior_minus_prior_position_m_quantiles'], 3), 'speed_m_s_median': q(f['speed_m_s_quantiles'], 3),
            'speed_m_s_p95': q(f['speed_m_s_quantiles'], 5), 'posterior_position_eig_median_m2': q(f['posterior_position_eig_m2_quantiles'], 3),
            'posterior_pd_failures': f['posterior_positive_definite_failures']}


def _write_timeline(path, rows):
    import csv
    if not rows: return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
