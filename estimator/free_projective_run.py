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
from estimator.point_kalman_cuda import predict, update_points, Timers, eigvalsh3, spd_extreme_eigs, ACCEPTANCE
from utils.io import json_write, serial

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
             debug_frames=(), debug_point_ids=None, save_full_covariance=False, save_state_interval=0, progress=None, assignment='per-frame'):
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
                'row_innovation_gate': policy.get('row_innovation_gate', 'depth_front_only'), 'final_visibility': policy.get('final_visibility', 'frozen')}
    json_write(root/'run_manifest.json', serial(manifest))
    timeline = []
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
        s, P, rep, trace = update_points(s_m, P_m, frame, rig, rgb0, noise, policy, max_iters, tol, chunk, kal_dtype, trace_ids, timers, assignment)
        with timers('store'):
            _store(mm, k, s, s_m, P, rep, C)
            if save_full_covariance and (k in debug_frames or (save_state_interval and i % save_state_interval == 0) or k == order[-1]):
                (root/'covariance').mkdir(exist_ok=True); np.save(root/'covariance'/f'frame_{k:04d}.npy', P.double().cpu().numpy())
        times = timers.resolve()
        num = _frame_numeric(k, i+1, dt, frame, rep, s, s_m, P, P_m, times, time.perf_counter()-t_wall, dev, dataset)
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
              'prediction_only_point_frames': int(sum(r['prediction_only_points'] for r in timeline))}
    json_write(root/'status.json', status)
    return status


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
