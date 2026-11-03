"""F1/B1 runner of the LOCAL-CORRELATED temporal RGB-D IEKF (train.py --temporal-mode local-correlated).

Each point keeps a small joint Gaussian (current state, retained reference states or the source chart, and the raw
sensor errors it has itself used). Cross-point covariance is zero by the declared block-diagonal assumed-density
approximation. This runner never constructs a global sparse system: no cuDSS factor, no selected inverse, no FEJ Schur.
Catalogue, camera-local references, hard first-hit ownership, hard masks, verified-frame journal and the live viewer
are the existing ones.
"""
from pathlib import Path
import csv
import resource
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.io import json_write, serial
from utils.temporal_output import VerifiedFrameOutput
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise
from estimator.free_projective_run import build_catalogue, _store, _timeline_row
from estimator.point_kalman_cuda import transition, process_noise
from estimator.temporal_zbuffer import first_hit_pixel_owner, compare_owner_maps
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_rgbd_terms import build_temporal_rows, raw_values, sensor_covariance, TEMPORAL_REASONS
from estimator.temporal_simplified_iekf import outside_points
from estimator.temporal_local_covariance import (Codec, SourceChart, initial_local_gaussian, predict_local, retain,
    retention_mask, record_expiry, current_jacobian, memory_profile, chunk_plan, batched_eigvalsh, KEY, STATE, CHART,
    CROSS_POINT_COVARIANCE, QUALIFIER, REFERENCE_UNCERTAINTY, LocalCovarianceFailure)
from estimator.temporal_local_iekf import LocalFrameUpdate, current_marginals
from debug.temporal_debug import temporal_frame_debug, write_temporal_dashboard

QS = [0., .01, .05, .5, .95, .99, 1.]
APPROXIMATION = ('LOCAL point Gaussian linearization, conditional on hard first-hit owners, camera-local reference policy '
                 'and active supports; cross-point covariance zero by assumption (points sharing a raw pixel carry '
                 'independent local copies of its sensor error). NOT the globally correlated full posterior, NOT globally '
                 'exact nonlinear Bayesian inference, NOT sequence smoothing.')


def append_csv(path, row):
    new = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def quantiles(x):
    x = x[torch.isfinite(x)] if x.is_floating_point() else x.double()
    return torch.quantile(x.double(), torch.tensor(QS, dtype=torch.float64, device=x.device)).tolist() if len(x) else None


def initial_marginal(g, chart, codec, budget):
    """6x6 current covariance at the anchor via the chart Jacobian, and the anchor state."""
    N, dev = g.N, g.mean.device
    s = torch.zeros((N, 6), dtype=torch.float64, device=dev)
    P = torch.zeros((N, 6, 6), dtype=torch.float64, device=dev)
    for pts in chunk_plan(torch.arange(N, device=dev), g.layout.dims, budget):
        mean, cov = g.gather(pts)
        st, J = current_jacobian(g, chart, codec, pts, mean)
        s[pts], P[pts] = st, J@cov@J.transpose(1, 2)
    return s, (P+P.transpose(1, 2))/2


class LocalFilter:
    """Point-local filter state: packed local Gaussians, source chart, camera-local references, expiry ledger."""
    def __init__(self, rig, noise, policy, warp, qc, lag, budget, K, frame0, source, base):
        self.rig, self.noise, self.policy, self.lag, self.budget = rig, noise, policy, int(lag), int(budget)
        dev = rig.device
        self.warp = torch.as_tensor(warp, dtype=torch.float64, device=dev)
        self.qc = torch.as_tensor(qc, dtype=torch.float64, device=dev)
        C, N = rig.count, len(source)
        self.codec = Codec(C, K, frame0.H*frame0.W)
        src = torch.as_tensor(source, device=dev, dtype=torch.long)
        value = raw_values(frame0, src[:, 1], src[:, 2]).double()
        self.chart = SourceChart(rig, frame0.k, src[:, 1], src[:, 2], frame0.W, torch.as_tensor(base, device=dev, dtype=torch.float64), value[:, 3])
        self.source_variance = sensor_covariance(value, noise).diagonal(dim1=-2, dim2=-1)
        self.source = source
        self.frame_ordinal = torch.full((K,), -1, dtype=torch.long, device=dev)
        self.frame_ordinal[frame0.k] = 0
        self.expired_upto = torch.full((C, N), -1, dtype=torch.long, device=dev)
        self.refs = ReferenceTable(C, N, dev, torch.float64)

    def initialize(self, frame0, pixel_covariance, velocity_covariance):
        self.bank = initial_local_gaussian(self.chart, self.source_variance, pixel_covariance, velocity_covariance)
        s, P = initial_marginal(self.bank, self.chart, self.codec, self.budget)
        owners = first_hit_pixel_owner(s[:, :3], self.rig, frame0)
        self.refs.commit_accepted_references(owners, frame0, s, 0)
        self.refs.initialize_from_source_catalogue(self.source, frame0.k, s)
        return s, P, owners

    def _peak(self, times, stage):
        if self.rig.device.type == 'cuda':
            times[stage+'_cuda_peak_bytes'] = torch.cuda.max_memory_allocated(self.rig.device)
            torch.cuda.reset_peak_memory_stats(self.rig.device)

    def step(self, frames, k, ordinal, dt, max_iters=10, tol=1e-8, traced=()):
        """predict -> expire unreferenced previous state -> fixed-prior local IEKF -> commit references -> ONE
        Joseph covariance -> exact local marginalization. `frames` holds the current and in-lag reference frames."""
        dev, times = self.rig.device, {}
        self._peak(times, 'before_step')
        self.frame_ordinal[k] = ordinal
        ts = time.perf_counter()
        prior = predict_local(self.bank, self.chart, self.codec, k, transition(dt, torch.float64, dev), process_noise(dt, self.qc), self.budget)
        keep = retention_mask(prior.layout, self.refs, self.chart, k, ordinal-1, self.lag)
        record_expiry(prior.layout, keep, self.expired_upto, self.frame_ordinal)
        prior = retain(prior, keep, self.budget)
        mean, prior6 = current_marginals(prior)
        times['predict_seconds'] = time.perf_counter()-ts
        self._peak(times, 'predict')
        ownership = {'seconds': 0., 'calls': 0}

        def build(states):
            tb = time.perf_counter()
            own = first_hit_pixel_owner(states[:, :3], self.rig, frames[k], dense=False)
            rows, reason, life = build_temporal_rows(states, own, frames[k], self.refs, frames, self.rig, self.noise, self.policy, self.warp, ordinal, self.lag)
            ownership['seconds'] += time.perf_counter()-tb
            ownership['calls'] += 1
            return rows, own, reason, life
        ts = time.perf_counter()
        update = LocalFrameUpdate(prior, self.chart, self.codec, self.refs, build, frames[k], k, ordinal, self.lag, self.rig, self.noise,
                                  self.warp, self.budget, self.expired_upto, self.frame_ordinal)
        solver = update.solve(max_iters, tol)
        times['nonlinear_solve_seconds'] = time.perf_counter()-ts
        times['ownership_and_row_sampling_seconds'] = ownership['seconds']
        self._peak(times, 'nonlinear_solve')
        s_acc = update.current_states('it')
        owners = first_hit_pixel_owner(s_acc[:, :3], self.rig, frames[k])
        before_owner = first_hit_pixel_owner(mean[:, :3], self.rig, frames[k], dense=False)
        ts = time.perf_counter()
        self.refs.commit_accepted_references(owners, frames[k], s_acc, ordinal)
        times['reference_commit_seconds'] = time.perf_counter()-ts
        ts = time.perf_counter()
        self.bank, s, P, post = update.posterior(self.refs, traced)
        times['covariance_and_marginalization_seconds'] = time.perf_counter()-ts
        self._peak(times, 'covariance_and_marginalization')
        if not torch.equal(s, s_acc):
            raise LocalCovarianceFailure('published state differs from the accepted iterate')
        return {'s': s, 'P': P, 'mean': mean, 'prior6': prior6, 'solver': solver, 'update': update, 'lr': update.final,
                'owners': owners, 'before_owner': before_owner, 'post': post, 'times': times, 'ownership_calls': ownership['calls']}


class LocalVariant:
    """Runner hooks; the defaults are exactly the local-correlated mode (the material-following mode overrides them)."""
    mode, label, bar = 'local-correlated', 'LOCAL-CORRELATED', 'Local-correlated'

    def filter(self, *args):
        return LocalFilter(*args)

    def configure(self, filt, options):
        pass

    def specs(self, specs, K, C, N):
        return specs

    def manifest(self, manifest):
        return manifest

    def frame_outputs(self, mm, k, step, summary):
        pass

    def row_extra(self, rows, lr, a):
        return {}


def run_temporal_local_free(dataset, config, policy, direction, root, device, sub, options, debug_frames=(), debug_point_ids=None,
                            max_iters=10, tol=1e-8, save_full_covariance=False, save_state_interval=0, progress=None, variant=None):
    variant = variant or LocalVariant()
    root = Path(root)
    numeric, temporal = root/'numeric', root/'temporal'
    for p in (numeric, temporal, temporal/'traced_points', temporal/'debug_frames'):
        p.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    budget = int(options.get('local_chunk_bytes', 512*1024**2))
    rig = CameraRig(dataset.cameras, device, torch.float64)
    noise = SensorNoise.from_config(config, policy, device, torch.float64)
    warp = torch.as_tensor(options['warp_covariance'], dtype=torch.float64, device=dev)
    lag = int(options['lag'])
    k0, xyz, rgb, cov, source, sel, sm = build_catalogue(dataset, config, direction, sub, policy, noise)
    if options.get('diagnostic_original_point_ids') is not None:
        from scene.canonical_subsample import ids_hash
        sel = np.asarray(sorted(set(options['diagnostic_original_point_ids'])), dtype=int)
        if not len(sel) or sel.min() < 0 or sel.max() >= len(xyz):
            raise ValueError('Diagnostic original point IDs outside full endpoint catalogue')
        sm = {'diagnostic_focal_control': True, 'endpoint_selection_before_focal': sm, 'factor': None, 'method': 'explicit-original-ID-focal-control',
              'N_full': len(xyz), 'N_selected': len(sel), 'selected_ids_sha256': ids_hash(sel), 'selected_original_ids': sel.tolist(),
              'note': 'Original endpoint IDs unchanged; focal selection removes other occluders; NOT full-cloud validation'}
    selected = source[sel]
    N, K, C = len(sel), len(dataset.times), rig.count
    order = list(range(K)) if direction == 'forward' else list(range(K-1, -1, -1))
    requested = list(debug_point_ids) if debug_point_ids is not None else sel[np.linspace(0, N-1, min(6, N)).astype(int)].tolist()
    ids = np.flatnonzero(np.isin(sel, requested)).tolist()
    specs = {'posterior_xyz': ((K, N, 3), 'float64'), 'predicted_xyz': ((K, N, 3), 'float32'), 'velocity': ((K, N, 3), 'float32'),
             'position_variance_diag': ((K, N, 3), 'float32'), 'velocity_variance_diag': ((K, N, 3), 'float32'),
             'camera_reason': ((K, C, N), 'uint8'), 'posterior_covariance': ((K, N, 6, 6), 'float64')}
    for name, dtype in [('valid_cameras', 'uint8'), ('point_status', 'uint8'), ('converged', 'bool'), ('iekf_iterations', 'uint8'),
                        ('nis', 'float32'), ('nis_dof', 'uint8'), ('information_min_eig', 'float32'), ('information_max_eig', 'float32'),
                        ('update_acceptance', 'uint8'), ('posterior_cost', 'float32'), ('posterior_cost_dof', 'uint8'),
                        ('local_dimension', 'uint16'), ('local_raw_keys', 'uint16'), ('local_reference_blocks', 'uint8'),
                        ('innovation_min_eig', 'float32')]:
        specs[name] = ((K, N), dtype)
    specs = variant.specs(specs, K, C, N)
    debug_count = len(set(debug_frames) & set(order))
    panel_bound = 2*C*max(rig.width)*max(rig.height)*12*3*debug_count
    catalogue_bound = 2*(xyz.nbytes+rgb.nbytes+cov.nbytes+source.nbytes)+1024**2
    publication = VerifiedFrameOutput(root, specs, order, options.get('output_headroom_bytes', 128*1024**2), panel_bound+catalogue_bound)
    mm = publication.arrays
    np.savez_compressed(root/'canonical.npz', canonical_frame=k0, canonical_xyz=xyz[sel], canonical_rgb=rgb[sel], original_source_ids=selected,
                        point_id=sel, local_array_index=np.arange(N), subsampled_to_full_index=sel, canonical_source_covariance=cov[sel],
                        full_endpoint_xyz=xyz, full_endpoint_rgb=rgb, full_endpoint_source=source)
    json_write(root/'subsample_manifest.json', serial(sm))
    frames = {k0: MeasuredFrame(dataset, k0, device, torch.float64)}
    filt = variant.filter(rig, noise, policy, warp, config.qc, lag, budget, K, frames[k0], selected, xyz[sel])
    variant.configure(filt, options)
    s, P, owners = filt.initialize(frames[k0], config.pixel_covariance, config.velocity_covariance)
    bank, refs = filt.bank, filt.refs
    manifest = {'pass': 'F1' if direction == 'forward' else 'B1', 'direction': direction, 'canonical_frame': k0, 'point_count': N,
                'processing_order': order, 'timestamps_seconds': dataset.times.tolist(), 'original_frame_indices': list(dataset.frame_ids),
                'debug_point_ids': ids, 'debug_original_point_ids': sel[ids].tolist(), 'debug_frames': list(debug_frames), 'camera_count': C,
                'observation_model': 'temporal-rgbd', 'observation_model_version': 3, 'temporal_mode': variant.mode, 'lag': lag,
                'precision': 'float64', 'backend': 'batched point-local dense GPU algebra; size-sorted bounded chunks',
                'cross_point_covariance': CROSS_POINT_COVARIANCE, 'scientific_qualifier': QUALIFIER, 'approximation': APPROXIMATION,
                'reference_uncertainty': REFERENCE_UNCERTAINTY,
                'local_history_policy': {'states': 'current + every state a camera can still reference (processing-order age < lag)',
                                         'raw_errors': 'every (camera,frame,pixel) key of a camera-usable reference frame; source error while its chart is kept',
                                         'expiry': 'exact Gaussian marginal = covariance submatrix of retained blocks; expired keys never recreated',
                                         'reference_selection': 'existing camera-local latest accepted valid reference (ReferenceTable)'},
                'source_initialization': 'minimal 5D chart (du,dv,v) + ONE shared source raw-depth error; no independent canonical Pxx',
                'warp_covariance_semantics': 'R_warp is additional conditional warp discrepancy only; sensor errors are local variables',
                'execution_kind': 'ACTUAL_TEMPORAL_IEKF_ESTIMATION', 'noise_assumption': options['warp_provenance'].get('assumption_class'),
                'diagnostic_focal_control': options.get('diagnostic_original_point_ids') is not None, 'reason_codes': TEMPORAL_REASONS,
                'status_codes': {0: 'ANCHOR_CANONICAL', 1: 'DIRECT_TEMPORAL_ROWS', 2: 'PREDICTION_ONLY'},
                'q_c_m2_s3': config.qc.tolist(), 'noise_and_prior': config.manifest(), 'calibration_hash': rig.calibration_hash(),
                'downstream': 'TEMPORAL_FREE_WITH_LEGACY_RIGID', 'global_sparse_solver_used': False,
                'NIS_semantics': 'predictive NIS of the stacked local innovation at the prior linearization; local model with cross-point independence assumed',
                'information_diagnostic_semantics': 'reciprocal eigenvalues of the local posterior position marginal',
                'original_point_ids': sel.tolist() if N < 1000 else 'canonical.npz point_id'}
    manifest = variant.manifest(manifest)
    for path, data in [(root/'run_manifest.json', manifest), (temporal/'config.json', options), (temporal/'calibration.json', options['warp_provenance'])]:
        json_write(path, serial(data))
    _store(mm, k0, s, s, P, None, C)
    mm['point_status'][k0] = 0
    mm['posterior_covariance'][k0] = P.cpu().numpy()
    mm['local_dimension'][k0] = bank.layout.dims.cpu().numpy()
    mm['local_raw_keys'][k0] = torch.bincount(bank.layout.point[bank.layout.kind == KEY], minlength=N).cpu().numpy()
    if k0 in debug_frames:
        temporal_frame_debug(root, frames[k0], owners, s, s, rig, None, k0)
    publication.publish_verified_frame(k0)
    history, traces, start = [], {i: [] for i in ids}, time.perf_counter()
    write_temporal_dashboard(root, manifest, history)
    try:
        for ordinal, k in enumerate(tqdm(order[1:], desc=f'{variant.bar} {direction}', unit='frame'), 1):
            t0 = time.perf_counter()
            times = {}
            if dev.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(dev)
            prev = order[ordinal-1]
            dt = float(dataset.times[k]-dataset.times[prev])
            ts = time.perf_counter()
            frames[k] = MeasuredFrame(dataset, k, device, torch.float64)
            for old in list(frames):
                if old not in order[max(0, ordinal-lag+1):ordinal+1]:
                    del frames[old]
            times['input_seconds'] = time.perf_counter()-ts
            step = filt.step(frames, k, ordinal, dt, max_iters, tol, ids)
            times.update(step['times'])
            s, P, mean, prior6, solver, lr, owners, before_owner, post = (step[x] for x in
                ('s', 'P', 'mean', 'prior6', 'solver', 'lr', 'owners', 'before_owner', 'post'))
            update, bank = step['update'], filt.bank
            ownership = {'calls': step['ownership_calls']}
            rows = lr.rows
            nrows = torch.bincount(rows.point, minlength=N)
            observed = nrows > 0
            assimilated = update.assimilated
            residual = rows.y-rows.h
            ev = batched_eigvalsh(P[:, :3, :3])
            rep = {'nrows': nrows, 'observed': observed, 'reason': lr.reason, 'converged': solver['point_converged'],
                   'point_iterations': torch.full((N,), len(solver['history']), device=dev), 'nis': solver['predictive_nis'],
                   'nis_dof': 4*nrows, 'info_eig': ev.flip(-1).reciprocal(), 'acceptance': assimilated.to(torch.uint8),
                   'posterior_cost': solver['final_point_objective'], 'posterior_cost_dof': 4*nrows}
            _store(mm, k, s, mean, P, rep, C)
            mm['posterior_covariance'][k] = P.cpu().numpy()
            dims = bank.layout.dims
            mm['local_dimension'][k] = dims.clamp_max(65535).cpu().numpy()
            mm['local_raw_keys'][k] = torch.bincount(bank.layout.point[bank.layout.kind == KEY], minlength=N).clamp_max(65535).cpu().numpy()
            mm['local_reference_blocks'][k] = torch.bincount(bank.layout.point[((bank.layout.kind == STATE) & (bank.layout.start > 0)) | (bank.layout.kind == CHART)],
                                                             minlength=N).clamp_max(255).cpu().numpy()
            mm['innovation_min_eig'][k] = post['innovation_min_eigenvalue'].float().cpu().numpy()
            stat = compare_owner_maps(before_owner, owners)
            speeds = s[:, 3:].norm(dim=1)
            posdiag = P[:, :3, :3].diagonal(dim1=1, dim2=2)
            per_camera = []
            for c in range(C):
                ix = rows.camera == c
                per_camera.append({'camera_index': c, 'camera': rig.names[c], 'rows': int(ix.sum()), 'rows_accepted_final': int(ix.sum()),
                                   'depth_innovation_m_quantiles_prior': None, 'information_trace_final': None,
                                   'reason_counts': {TEMPORAL_REASONS[int(v)]: int(n) for v, n in zip(*torch.unique(lr.reason[c], return_counts=True))},
                                   'reference_age_quantiles': quantiles((ordinal-refs.ordinal[c][refs.frame[c] >= 0]).double()),
                                   'depth_residual_m_quantiles': quantiles(residual[ix, 3]) if bool(ix.any()) else None,
                                   'rgb_residual_rms': float(residual[ix, :3].square().mean().sqrt()) if bool(ix.any()) else None})
            norms = {k2: quantiles(v[observed]) for k2, v in post['cross_norms'].items()}
            keys = torch.bincount(bank.layout.point[bank.layout.kind == KEY], minlength=N)
            summary = {'frame': k, 'ordinal': ordinal, 'dt_seconds': dt, 'frames_completed': len(publication.journal['completed_frames'])+1,
                       'processed_points': N, 'rows': len(rows), 'observed_points': int(observed.sum()), 'assimilated_points': int(assimilated.sum()),
                       'accepted_points': int(assimilated.sum()), 'no_row': int((~observed).sum()), 'prediction_only': int((~observed).sum()),
                       'rejected': int((observed & ~assimilated).sum()), 'eligible_but_no_accepted_update': int((observed & ~assimilated).sum()),
                       'nonconverged': solver['nonconverged_points'],
                       'unexplained_foreground': int((frames[k].mask & frames[k].raw_valid & (owners.owner_id < 0)).sum()),
                       'rendered_outside_foreground': int(outside_points(owners, frames[k], N).sum()),
                       'rgb_residual_rms': float(residual[:, :3].square().mean().sqrt()) if len(rows) else None,
                       'depth_residual_rms_m': float(residual[:, 3].square().mean().sqrt()) if len(rows) else None,
                       'residual_semantics': 'y_current - clean temporal prediction (reference raw minus local posterior eta) at the accepted iterate',
                       'prior_objective': solver['prior_objective'], 'accepted_objective': solver['accepted_objective'],
                       'accepted_step_m_quantiles': quantiles((s[:, :3]-mean[:, :3]).norm(dim=1)[assimilated]),
                       'nonlinear_iterations': len(solver['history']),
                       'innovation_min_eigenvalue_quantiles': quantiles(post['innovation_min_eigenvalue']),
                       'local_posterior_correlation_min_eigenvalue': float(post['local_correlation_min_eigenvalue'].min()) if N else None,
                       'local_posterior_relative_asymmetry_max': post['relative_asymmetry_max'],
                       'joseph_vs_algebraic_max_correlation_units': post['joseph_vs_algebraic_max_correlation_units'],
                       'minimum_posterior_position_variance_m2': float(posdiag.min()),
                       'posterior_position_variance_median_m2': float(torch.median(posdiag)),
                       'cross_covariance_norms_observed_quantiles': norms,
                       'raw_keys_per_point_quantiles': quantiles(keys.double()),
                       'repeated_key_reuse_rows': lr.reused_history_keys, 'source_key_reuse_rows': lr.reused_source_keys,
                       'expired_raw_keys': post['expired_raw_keys'], 'forbidden_key_recreations': 0,
                       'fresh_raw_keys_registered': solver['counters']['fresh_raw_keys_registered'],
                       'linearizations': solver['counters']['linearizations'], 'ownership_calls': ownership['calls'],
                       'same_camera_bootstraps': lr.life.get('bootstrap'), 'reference_outside_lag': lr.life.get('reference_outside_lag'),
                       'invalid_reference_support': lr.life.get('invalid_reference'), 'occluded_no_row': lr.life.get('occluded_no_row'),
                       'reference_commits_this_frame': int((refs.ordinal == ordinal).sum()), 'per_camera': per_camera,
                       'local_memory': memory_profile(bank), 'posterior_blocks_before_expiry': post['posterior_blocks'],
                       'retained_blocks_after_expiry': post['retained_blocks'],
                       'speed_median_m_s': float(torch.median(speeds)), 'speed_p95_m_s': float(torch.quantile(speeds, .95)),
                       'covariance_min_eigenvalue_mixed_units': float(batched_eigvalsh(P).min()),
                       'posterior_pd_failures': 0, 'scientific_qualifier': QUALIFIER,
                       'solver': {kk: v for kk, v in solver.items() if kk not in ('predictive_nis', 'final_point_objective', 'point_converged')},
                       'reference': lr.life, **stat}
            for i in ids:
                ii = torch.where(rows.point == i)[0]
                tr = post['traces'].get(i, {})
                record = {'frame': k, 'source': selected[i].tolist(), 's_minus': mean[i].tolist(), 'P_minus': prior6[i].tolist(),
                          's_plus': s[i].tolist(), 'P_plus': P[i].tolist(), 'solver': {'converged': bool(solver['point_converged'][i]),
                          'iterations': len(solver['history'])}, 'measurement_update_accepted': bool(assimilated[i]),
                          'status': 'ACCEPTED_UPDATE' if bool(assimilated[i]) else ('REJECTED_PREDICTION_ONLY' if bool(observed[i]) else 'NO_ROW_PREDICTION_ONLY'),
                          'camera_reason': lr.reason[:, i].tolist(), 'reference_frames': refs.frame[:, i].tolist(),
                          'reference_processing_age': (ordinal-refs.ordinal[:, i]).tolist(), 'local_posterior': local_blocks(tr), 'rows': []}
                for a in ii.tolist():
                    c, rf, pix = int(rows.camera[a]), int(rows.reference_frame[a]), int(rows.pixel[a])
                    competitors = torch.where((owners.pixel[c] == pix) & (owners.z[c] > rig.near[c]))[0]
                    zz = owners.z[c, competitors]
                    oz = torch.argsort(zz, stable=True)
                    record['rows'].append({'camera': c, 'current_pixel': pix, 'reference_frame': rf, 'reference_pixels': rows.reference_pixels[a].tolist(),
                        'current_time_seconds': float(dataset.times[k]), 'reference_time_seconds': float(dataset.times[rf]),
                        'q': rows.q[a].tolist(), 'weights': rows.weights[a].tolist(), 'H_t': rows.H_t[a].tolist(), 'H_ref': rows.H_ref[a].tolist(),
                        'H_eta_reference': [(-float(w)*np.eye(4)).tolist() for w in rows.weights[a]], 'H_eta_current': np.eye(4).tolist(),
                        'current_raw_rgbd': rows.y[a].tolist(), 'reference_raw_rgbd': rows.values[a].tolist(),
                        'reference_clean_rgbd': lr.clean_reference[a].tolist(), 'prediction': rows.h[a].tolist(), 'residual': residual[a].tolist(),
                        'reference_state': rows.reference_state[a].tolist(), 'R_current_sensor': sensor_covariance(rows.y[a], noise).tolist(),
                        'R_additional_warp_model': warp.tolist(), 'collision_ids_by_depth': competitors[oz].tolist(),
                        'collision_original_ids_by_depth': sel[competitors[oz].cpu().numpy()].tolist(), 'collision_depths_m': zz[oz].tolist(),
                        'current_mask': True, 'reference_support_mask': [True]*4,
                        'covariance_semantics': 'local joint Gaussian; reference/current raw errors are local variables (no R_eff sum)',
                        **variant.row_extra(rows, lr, a)})
                record['original_point_id'] = int(sel[i])
                record['local_array_index'] = i
                traces[i].append(record)
                json_write(temporal/'traced_points'/f'{sel[i]}.json', serial({'point_id': int(sel[i]), 'local_array_index': i,
                           'covariance_scope': QUALIFIER, 'frames': traces[i]}))
            variant.frame_outputs(mm, k, step, summary)
            ts = time.perf_counter()
            if k in debug_frames:
                temporal_frame_debug(root, frames[k], owners, s, mean, rig, rows, k)
            times['viewer_seconds'] = time.perf_counter()-ts
            times['cuda_peak_allocated_bytes'] = max([torch.cuda.max_memory_allocated(dev)]+[v for kk, v in times.items() if kk.endswith('_cuda_peak_bytes')]) if dev.type == 'cuda' else 0
            summary['seconds'] = time.perf_counter()-t0
            summary['stage_times'] = times
            summary['chunk_budget_bytes'] = budget
            history.append(summary)
            numeric_record = {'frame': k, 'processing_step': ordinal, 'timestamp_s': float(dataset.times[k]), 'dt_s': dt,
                'sensor': {'unique_points_observed': int(observed.sum()), 'prediction_only_points': int((~observed).sum()), 'rows_accepted_final': len(rows), 'per_camera': per_camera},
                'filter_update': {'not_converged_points': solver['nonconverged_points'], 'posterior_position_eig_m2_quantiles': quantiles(ev.reshape(-1)),
                                  'posterior_minus_prior_position_m_quantiles': quantiles((s[:, :3]-mean[:, :3]).norm(dim=1)),
                                  'speed_m_s_quantiles': quantiles(speeds), 'posterior_positive_definite_failures': 0},
                'innovation': {'nis_per_dof_quantiles_prior_observed': quantiles((solver['predictive_nis']/(4*nrows).clamp_min(1))[observed]),
                               'semantics': manifest['NIS_semantics']},
                'processing': {'wall_ms': summary['seconds']*1000, 'cuda_ms': {}, 'gpu_peak_allocated_MB': times['cuda_peak_allocated_bytes']/1e6,
                               'iekf_iterations': len(solver['history'])},
                'acceptance': {'counts': {}}, 'temporal': summary}
            json_write(numeric/f'frame_{k:04d}.json', serial(numeric_record))
            append_csv(numeric/'timeline.csv', _timeline_row(numeric_record))
            json_write(temporal/f'frame_{k:04d}.json', serial(summary))
            append_csv(temporal/'timing_memory.csv', {'frame': k, **times, 'seconds': summary['seconds'], 'local_packed_bytes': bank.nbytes,
                       'cuda_free_bytes': torch.cuda.mem_get_info()[0] if dev.type == 'cuda' else 0,
                       'cpu_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            append_csv(temporal/'reference_lifecycle.csv', {'frame': k, **lr.life, 'expired_raw_keys': post['expired_raw_keys']})
            if save_full_covariance and (k in debug_frames or (save_state_interval and ordinal % save_state_interval == 0)):
                (root/'states').mkdir(exist_ok=True)
                np.savez_compressed(root/'states'/f'frame_{k:04d}.npz', s=s.cpu().numpy(), P=P.cpu().numpy())
            publication.publish_verified_frame(k, temporal/f'frame_{k:04d}.json')
            write_temporal_dashboard(root, manifest, history)
            print(f'[{variant.label} {direction}] frame={k} accepted={int(assimilated.sum())}/{N} rows={len(rows)} '
                  f'seconds={summary["seconds"]:.2f} live={temporal/"index.html"}', flush=True)
        result = {'status': 'COMPLETE', 'frames': K, 'point_count': N, 'seconds': time.perf_counter()-start, 'mode': variant.mode,
                  'accepted_updates': sum(h['assimilated_points'] for h in history), 'all_frames_valid': True,
                  'cross_point_covariance': CROSS_POINT_COVARIANCE, 'scientific_acceptance': 'NOT_AUTOMATICALLY_VALIDATED'}
        json_write(root/'status.json', result)
        return result
    except BaseException as error:
        result = {'status': getattr(error, 'status', 'FAILED'), 'reason': str(error), 'completed_frames': publication.journal['completed_frames'],
                  'point_count': N, 'elapsed_filter_seconds': time.perf_counter()-start,
                  'cuda_peak_bytes': torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else 0,
                  'cpu_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, 'mode': variant.mode,
                  'covariance_policy': 'incomplete frames are NaN and absent from completed_frames.json'}
        json_write(root/'status.json', result)
        write_temporal_dashboard(root, manifest, history, str(error))
        raise


def local_blocks(trace):
    """Cov(s_t, block) for every block of a traced point's posterior, with key provenance."""
    if not trace:
        return None
    S = np.asarray(trace['Sigma_plus'])
    out = {'posterior_layout': trace['posterior_layout'], 'retained_layout': trace['retained_layout'], 'cross_with_current_state': []}
    for blk in trace['posterior_layout']:
        a, b = blk['columns']
        if a == 0:
            continue
        out['cross_with_current_state'].append({**{kk: blk[kk] for kk in ('kind', 'camera', 'frame', 'pixel')},
                                                'Cov_current_block': S[:6, a:b].tolist(), 'block_mean': trace['mean_plus'][a:b],
                                                'block_covariance': S[a:b, a:b].tolist()})
    return out
