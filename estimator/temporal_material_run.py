"""F1/B1 runner of the MATERIAL-FOLLOWING temporal RGB-D IEKF (train.py --temporal-mode material-following).

Same point-local joint Gaussian, catalogue, source chart, verified-frame journal and viewer as local-correlated
(temporal_local_run); different observation domain (temporal_material_terms), acceptance (temporal_material_iekf),
reference commit (first-hit winner on its own measured surface, only after an accepted update or a frame without
temporal rows; otherwise the previous reference is preserved inside the lag window) and raw-key retention (only the
reference raw pixel of each camera). No global sparse solver, no motion graph, no point coupling.
"""
import time
from pathlib import Path
import torch
from estimator.point_kalman_cuda import transition, process_noise
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_local_covariance import predict_local, retain, LocalCovarianceFailure, batched_eigvalsh
from estimator.temporal_local_iekf import current_marginals
from estimator.temporal_local_run import LocalFilter, LocalVariant, run_temporal_local_free, initial_marginal, initial_local_gaussian, quantiles
from estimator.temporal_material_terms import (build_material_rows, commit_material_references, material_retention_mask,
                                               material_record_expiry, MATERIAL_REASONS, OBSERVATION_DOMAIN)
from estimator.temporal_material_iekf import MaterialFrameUpdate
from estimator.temporal_material_pyramid import build_level, CoarseEvaluator, LEVEL_NOTE


class MaterialFilter(LocalFilter):
    resampling = 'none'
    pyramid_levels = 0
    gradient_guide = 'off'

    def _pyramid(self, frames, k):
        if not self.pyramid_levels:
            return None
        levels = list(range(self.pyramid_levels, 0, -1))
        cache = self.__dict__.setdefault('_level_cache', {})
        for key in [key for key in cache if key not in frames]:
            del cache[key]
        for key, fr in frames.items():
            if key not in cache:
                cache[key] = {L: build_level(fr, L, self.rig, self.noise, self.policy) for L in levels}
        ev = CoarseEvaluator(self.rig, self.noise, self.policy, self.refs, cache[k], cache, self.warp, frames[k].W)
        return ev, levels

    def initialize(self, frame0, pixel_covariance, velocity_covariance):
        self.bank = initial_local_gaussian(self.chart, self.source_variance, pixel_covariance, velocity_covariance)
        s, P = initial_marginal(self.bank, self.chart, self.codec, self.budget)
        owners = first_hit_pixel_owner(s[:, :3], self.rig, frame0)
        allowed = torch.ones(len(s), dtype=torch.bool, device=s.device)
        self.last_commit = commit_material_references(self.refs, owners, frame0, s, 0, allowed, self.rig, self.noise, self.policy)
        self.refs.initialize_from_source_catalogue(self.source, frame0.k, s)
        return s, P, owners

    def step(self, frames, k, ordinal, dt, max_iters=10, tol=1e-8, traced=()):
        dev, times = self.rig.device, {}
        self._peak(times, 'before_step')
        self.frame_ordinal[k] = ordinal
        ts = time.perf_counter()
        prior = predict_local(self.bank, self.chart, self.codec, k, transition(dt, torch.float64, dev), process_noise(dt, self.qc), self.budget)
        keep = material_retention_mask(prior.layout, self.refs, self.chart, k, ordinal-1, self.lag)
        material_record_expiry(prior.layout, keep, self.refs, self.expired_upto, self.frame_ordinal)
        prior = retain(prior, keep, self.budget)
        mean, prior6 = current_marginals(prior)
        times['predict_seconds'] = time.perf_counter()-ts
        self._peak(times, 'predict')
        ownership = {'seconds': 0., 'calls': 0}

        def build(states):
            tb = time.perf_counter()
            own = first_hit_pixel_owner(states[:, :3], self.rig, frames[k], dense=False)
            rows, reason, life = build_material_rows(states, own, frames[k], self.refs, frames, self.rig, self.noise, self.policy, ordinal, self.lag, self.resampling)
            ownership['seconds'] += time.perf_counter()-tb
            ownership['calls'] += 1
            return rows, own, reason, life
        ts = time.perf_counter()
        if self.gradient_guide == 'replace':
            from estimator.temporal_gradient_iekf import GradientMaterialFrameUpdate as Update
        elif self.gradient_guide in ('off', 'diagnostic'):
            Update = MaterialFrameUpdate
        else:
            raise ValueError(f'unknown gradient guide {self.gradient_guide!r}')
        update = Update(prior, self.chart, self.codec, self.refs, build, frames[k], k, ordinal, self.lag, self.rig, self.noise,
                        self.warp, self.budget, self.expired_upto, self.frame_ordinal)
        update.pyramid = self._pyramid(frames, k)
        solver = update.solve(max_iters, tol)
        times['nonlinear_solve_seconds'] = time.perf_counter()-ts
        times['ownership_and_row_sampling_seconds'] = ownership['seconds']
        self._peak(times, 'nonlinear_solve')
        s_acc = update.current_states('it')
        owners = first_hit_pixel_owner(s_acc[:, :3], self.rig, frames[k])
        before_owner = first_hit_pixel_owner(mean[:, :3], self.rig, frames[k], dense=False)
        observed = torch.zeros(len(s_acc), dtype=torch.bool, device=dev)
        observed[update.final.rows.point] = True
        ts = time.perf_counter()
        commit = commit_material_references(self.refs, owners, frames[k], s_acc, ordinal, update.assimilated | ~observed,
                                            self.rig, self.noise, self.policy)
        times['reference_commit_seconds'] = time.perf_counter()-ts
        ts = time.perf_counter()
        self.bank, s, P, post = update.posterior(self.refs, traced)
        times['covariance_and_marginalization_seconds'] = time.perf_counter()-ts
        self._peak(times, 'covariance_and_marginalization')
        if not torch.equal(s, s_acc):
            raise LocalCovarianceFailure('published state differs from the accepted iterate')
        return {'s': s, 'P': P, 'mean': mean, 'prior6': prior6, 'solver': solver, 'update': update, 'lr': update.final,
                'owners': owners, 'before_owner': before_owner, 'post': post, 'times': times, 'ownership_calls': ownership['calls'],
                'commit': commit, 'dt': dt, 'traced': list(traced), 'ordinal': ordinal}


def position_information(lr, warp, N):
    """(N,3,3) sum over final rows of H_x^T (R_eff + R_warp)^-1 H_x at the accepted iterate (observability diagnostic)."""
    rows = lr.rows
    info = torch.zeros((N, 3, 3), dtype=torch.float64, device=lr.e0.device)
    if len(lr):
        R = torch.diag_embed(lr.r_eff+rows.resampling_variance)+warp
        Hx = rows.H_t[:, :, :3]
        info.index_add_(0, rows.point, Hx.transpose(1, 2)@torch.linalg.solve(R, Hx))
    return info


class MaterialVariant(LocalVariant):
    mode, label, bar = 'material-following', 'MATERIAL-FOLLOWING', 'Material-following'

    def filter(self, *args):
        return MaterialFilter(*args)

    def configure(self, filt, options):
        filt.resampling = options.get('material_resampling', 'none')
        filt.pyramid_levels = int(options.get('material_pyramid_levels', 0))
        filt.gradient_guide = options.get('gradient_guide', 'off')
        self.filt = filt
        self.options = {'resampling_discrepancy': filt.resampling, 'pyramid_levels': filt.pyramid_levels, 'gradient_guide': filt.gradient_guide}

    def specs(self, specs, K, C, N):
        specs.update({'material_rows': ((K, N), 'uint8'), 'material_pixel_shift_max_px': ((K, N), 'float32'),
                      'material_target_pixel_changed': ((K, N), 'bool'), 'material_info_eig_min': ((K, N), 'float32'),
                      'material_info_eig_max': ((K, N), 'float32')})
        return specs

    def manifest(self, manifest):
        manifest.update({'observation_domain': OBSERVATION_DOMAIN, 'reason_codes': MATERIAL_REASONS,
                         'material_identity': '(point, camera, reference frame, measured integer source pixel p_ref); NOT a fixed current pixel',
                         'acceptance': 'stable material identities valid at base and trial; reject identity loss to invalid support/mask/image '
                                       'bounds, new background projection, no common identity, or no MAP descent on the common set; '
                                       'occlusion losses allowed and reported; gained identities excluded from the comparison',
                         'visibility': 'first-hit with all tracked points at every candidate state; hidden when another projected point or the '
                                       'measured depth is in front by more than the documented single-surface allowance',
                         'reference_commit': 'first-hit winner on a valid foreground pixel whose measured depth is within the single-surface '
                                             'allowance of the point; only after an accepted update or a frame without temporal rows',
                         'local_history_policy': {**manifest['local_history_policy'],
                                                  'raw_errors': 'only the reference raw pixel (camera, frame, p_ref) of each still-usable camera '
                                                                'reference; the source error while its chart is kept'},
                         'pyramid': {'levels': self.options['pyramid_levels'], 'semantics': LEVEL_NOTE} if self.options['pyramid_levels'] else 'not used (original resolution)',
                         'resampling_discrepancy': self.options['resampling_discrepancy'],
                         'gradient_guide': self.options['gradient_guide'] if self.options['gradient_guide'] == 'off' else
                             {'mode': self.options['gradient_guide'], 'semantics': 'diagnostic: velocity information of the SAME measurement, never assimilated'
                              if self.options['gradient_guide'] == 'diagnostic' else
                              'replace: material-following residual/Jacobian re-assembled from image-gradient velocity rows (equivalent; no new information)'},
                         'information_diagnostic_semantics': 'material_info_eig_*: eigenvalues of sum H_x^T (R_eff+R_warp)^-1 H_x at the accepted iterate (1/m^2)'})
        return manifest

    def frame_outputs(self, mm, k, step, summary):
        lr, update, N = step['lr'], step['update'], len(step['s'])
        rows = lr.rows
        dev = step['s'].device
        shift = rows.pixel_shift.norm(dim=1) if len(rows) else torch.zeros(0, dtype=torch.float64, device=dev)
        smax = torch.full((N,), float('nan'), dtype=torch.float64, device=dev)
        if len(rows):
            smax = smax.scatter_reduce(0, rows.point, shift, reduce='amax', include_self=False)
        nrows = torch.bincount(rows.point, minlength=N)
        info = position_information(lr, update.warp, N)
        ev = batched_eigvalsh(info)
        mm['material_rows'][k] = nrows.clamp_max(255).cpu().numpy()
        mm['material_pixel_shift_max_px'][k] = smax.float().cpu().numpy()
        mm['material_target_pixel_changed'][k] = update.target_changed.cpu().numpy()
        mm['material_info_eig_min'][k] = ev[:, 0].float().cpu().numpy()
        mm['material_info_eig_max'][k] = ev[:, 2].float().cpu().numpy()
        assim = update.assimilated
        acc_rows = assim[rows.point] if len(rows) else torch.zeros(0, dtype=torch.bool, device=dev)
        observed = nrows > 0
        summary['material'] = {
            'observation_domain': OBSERVATION_DOMAIN,
            'pixel_shift_px_quantiles_all_rows': quantiles(shift),
            'pixel_shift_px_quantiles_assimilated_rows': quantiles(shift[acc_rows]),
            'assimilated_rows_with_shift_above_1px_fraction': float((shift[acc_rows] > 1).double().mean()) if bool(acc_rows.any()) else None,
            'target_pixel_changed_assimilated_points': int(update.target_changed.sum()),
            'identity_counters': step['solver']['counters'],
            'reference_commit': step['commit'],
            'position_information_eig_quantiles_observed': [quantiles(ev[observed, j]) for j in range(3)],
            'weakest_direction_std_m_quantiles_observed': quantiles(ev[observed, 0].clamp_min(1e-300).rsqrt()),
            'rank_lt3_relative_1e-12_observed_points': int((ev[observed, 0] <= 1e-12*ev[observed, 2]).sum()),
            'lifecycle': lr.life,
            'coarse_to_fine': step['solver'].get('coarse_to_fine')}
        if self.options['gradient_guide'] != 'off':
            summary['gradient_guidance'] = gradient_guidance_summary(self.filt, step)

    def row_extra(self, rows, lr, a):
        return {'material_identity': {'reference_frame': int(rows.reference_frame[a]), 'reference_pixel': int(rows.reference_pixel[a])},
                'target_pixel': int(rows.target_pixel[a]), 'current_support_pixels': rows.current_pixels[a].tolist(),
                'current_raw_support_rgbd': rows.current_values[a].tolist(), 'current_raw_rgbd': rows.y[a].tolist(),
                'q': rows.q[a].tolist(), 'pixel_shift_px': rows.pixel_shift[a].tolist(), 'u_t': rows.u_t[a].tolist(), 'u_ref': rows.u_ref[a].tolist(),
                'reference_raw_rgbd': rows.reference_values[a].tolist(), 'reference_clean_rgbd': lr.clean_reference[a].tolist(),
                'prediction': rows.h[a].tolist(), 'residual_profiled': lr.e0[a].tolist(), 'f_at_iterate': rows.f[a].tolist(),
                'H_eta_reference': torch.eye(4).tolist(), 'H_eta_current': [(-float(w)*torch.eye(4)).tolist() for w in rows.weights[a]],
                'R_current_effective_diag': lr.r_eff[a].tolist(),
                'covariance_semantics': 'local joint Gaussian; reference raw error and current support raw errors are local variables'}


def gradient_guidance_summary(filt, step):
    """Diagnostic velocity information at the prediction and at the accepted iterate, visibility stage matrix and traced
    per-camera rows. Reads the solver's linearizations; never modifies any state."""
    from estimator.temporal_motion_observability import velocity_guidance, stage_matrix, summarize
    update, dt = step['update'], step['dt']
    out = {}
    for tag, lr in (('at_prediction', update.prior_rows), ('at_accepted_iterate', step['lr'])):
        g = velocity_guidance(lr, filt.noise, update.warp, dt, step['prior6'], update.frame, filt.rig, filt.policy)
        out[tag] = summarize(g, update, step, dt)
        if tag == 'at_prediction':
            gp = g
    import json as _json
    from estimator.temporal_motion_observability import photometric_consistency
    alpha = float(_json.loads((Path(__file__).resolve().parents[1]/'configs/image_joint_policy.json').read_text())['tail_probability'])
    out['photometric_consistency_at_accepted_iterate'] = photometric_consistency(step['lr'], filt.noise, update.warp, alpha)
    per_cam, total, _ = stage_matrix(step['lr'], update.assimilated, update.frame, filt.refs, step['ordinal'], filt.lag)
    out['stage_counts_total'], out['stage_counts_per_camera'] = total, per_cam
    traced, rows = {}, update.prior_rows.rows
    for i in step['traced']:
        ix = torch.where(rows.point == i)[0]
        rec = {'s_minus': step['mean'][i].tolist(), 's_plus': step['s'][i].tolist(), 'velocity_guidance_dv': gp['dv'][i].tolist(),
               'Lambda_v': gp['Lambda_v'][i].tolist(), 'b_v': gp['b_v'][i].tolist(),
               'camera_reason_final': step['lr'].reason[:, i].tolist(), 'cameras': []}
        for a in ix.tolist():
            rec['cameras'].append({'camera': int(rows.camera[a]), 'reference_frame': int(rows.reference_frame[a]), 'reference_pixel': int(rows.reference_pixel[a]),
                                   'q': rows.q[a].tolist(), 'G_rows_H_t_xyz': rows.H_t[a, :, :3].tolist(), 'residual_f': rows.f[a].tolist(),
                                   'row_info_v': gp['row_info'][a].tolist(), 'row_b_v': gp['row_b'][a].tolist(),
                                   'basis_t1_t2_n': gp['row_basis'][a].tolist(), 'depth_row_normal_abs_cos': float(gp['row_normal_alignment'][a])})
        traced[str(i)] = rec
    out['traced_points_local_index'] = traced
    return out


def run_temporal_material_free(*args, **kwargs):
    return run_temporal_local_free(*args, variant=MaterialVariant(), **kwargs)
