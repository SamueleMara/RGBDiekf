"""Point-local fixed-prior IEKF of the MATERIAL-FOLLOWING temporal observation (temporal_material_terms).

Same local joint Gaussian a_i = [s_t | retained reference states or source chart | raw sensor errors] and the same
proposal / Joseph covariance / exact local marginalization as the local-correlated mode (LocalFrameUpdate); only the
observation and the acceptance rule differ:

  * per row the raw variables are the ONE reference raw error eta_{c,tau,p_ref} (reused if this point already used that
    pixel, e.g. its source pixel or a pixel it observed as a current sample) and the four CURRENT bilinear raw errors at
    q_t (fresh N(0, R_sensor)); H_eta_ref = +I4, H_eta_cur,a = -w_a I4, and R_warp is only the additional discrepancy;
  * the acceptance objective integrates out the fresh current errors exactly:
    J = 1/2 delta^T z + sum_rows [1/2 e0^T R^-1 e0 + 1/2 log det R],  R = sum_a w_a^2 R_a (+ R_resampling) + R_warp,
    e0 = sum_a w_a y_a - (y_ref - eta_ref) - e4 dZ (R depends on q_t, so its log-determinant is part of the objective);
  * NO common current-pixel/owner domain: a step may change the target pixel and first-hit owners. A point's trial is
    compared with its base on the STABLE set of material identities valid at both states; a trial is rejected when an
    identity is lost to invalid support/mask/image bounds (identity validity destroyed), when the point's own projection
    newly lands on background in any camera, when no common identity remains, or without MAP descent on the common set.
    Identities lost to genuine occlusion (hidden behind another surface) are allowed and reported; identities gained are
    reported and enter the next proposal, never the comparison. Deleting a row can therefore never create descent.
"""
from dataclasses import dataclass
import torch
from estimator.temporal_rgbd_terms import sensor_covariance
from estimator.temporal_local_iekf import LocalFrameUpdate, _add_blocks
from estimator.temporal_local_covariance import LocalCovarianceFailure, check_no_recreation, QUALIFIER
from estimator.temporal_material_terms import (material_prediction, material_retention_mask, material_record_expiry,
                                               HIDDEN_CODES, INVALIDATING_CODES, OBSERVATION_DOMAIN)


@dataclass
class MaterialLocalRows:
    rows: object
    owners: object
    reason: torch.Tensor
    life: dict
    ref_code: torch.Tensor      # (M,) reference state / chart block
    src_code: torch.Tensor      # (M,) source raw key of a chart reference, -1 otherwise
    ja: torch.Tensor
    jl: torch.Tensor
    key_code: torch.Tensor      # (M,) reference raw key (camera, tau, p_ref)
    cur_code: torch.Tensor      # (M,4) current support raw keys
    weights: torch.Tensor       # (M,4)
    residual: torch.Tensor      # (M,4) = 0 - f(a^j) (current errors at their iterate values)
    e0: torch.Tensor            # (M,4) profiled residual (current errors at 0)
    r_eff: torch.Tensor         # (M,4) sum_a w_a^2 R_a (diagonal)
    clean_reference: torch.Tensor
    reused_history_keys: int
    reused_source_keys: int

    def __len__(self):
        return len(self.rows)


class MaterialFrameUpdate(LocalFrameUpdate):
    pyramid = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_report = None
        self.coarse_started = None
        self.counters.update({'trial_rejections': {'identity_support_lost': 0, 'outside_mask': 0, 'no_common_identity': 0,
                                                   'no_common_map_descent': 0},
                              'identities_lost_to_occlusion_accepted': 0, 'identities_gained_accepted': 0,
                              'target_pixel_changes_accepted': 0})

    def _ledger_check(self, pp, cc, ff):
        check_no_recreation(pp, cc, ff, self.expired_upto, self.frame_ordinal)

    def linearize(self, name='it', register=True):
        self.counters['linearizations'] += 1
        refs, k = self.refs, self.k
        s = self.current_states(name)
        age = self.ordinal-refs.ordinal
        c, p = torch.where((refs.frame >= 0) & (age < self.lag) & (age > 0))
        if len(p):
            refs.state[c, p] = self._reference_states(name, p, refs.frame[c, p])[0]
        rows, owners, reason, life = self.build(s)
        p, c, rf = rows.point, rows.camera, rows.reference_frame
        M = len(p)
        _, ref_code, ja, jl = self._reference_states(name, p, rf)
        ch = rf == self.chart.anchor_frame
        src_code = torch.full_like(p, -1)
        src_code[ch] = self.chart.source_codes(self.codec, p[ch])
        key_code = self.codec.key(p, c, rf, rows.reference_pixel)
        e4 = lambda x: x[:, None].expand(-1, 4)
        cur_code = self.codec.key(e4(p), e4(c), torch.full_like(e4(p), k), rows.current_pixels)
        ref_var = sensor_covariance(rows.reference_values, self.noise).diagonal(dim1=-2, dim2=-1)
        cur_var = sensor_covariance(rows.current_values, self.noise).diagonal(dim1=-2, dim2=-1)
        codes = torch.cat((key_code, cur_code.reshape(-1)))
        missing = (self.key_index.lookup(codes) < 0) & (self.ex.lookup(codes) < 0)
        if bool(missing.any()):
            pp = torch.cat((p, e4(p).reshape(-1)))[missing]
            cc = torch.cat((c, e4(c).reshape(-1)))[missing]
            ff = torch.cat((rf, torch.full_like(e4(p).reshape(-1), k)))[missing]
            xx = torch.cat((rows.reference_pixel, rows.current_pixels.reshape(-1)))[missing]
            self._ledger_check(pp, cc, ff)
            if register:
                self.counters['fresh_raw_keys_registered'] += self.ex.add(codes[missing], pp, cc, ff, xx,
                                                                          torch.cat((ref_var, cur_var.reshape(-1, 4)))[missing])
        eta_ref, pb, _ = self._key_values(name, key_code)
        eta_cur, _, _ = self._key_values(name, cur_code)
        clean_ref = rows.reference_values-eta_ref
        f, Ht, Hr, w = material_prediction(rows, rows.current_values-eta_cur, clean_ref)
        raw = (w[:, :, None]*rows.current_values).sum(1)
        e0 = raw-clean_ref
        e0[:, 3] -= rows.z_t-rows.z_ref
        rows.f, rows.h, rows.H_t, rows.H_ref, rows.weights, rows.y = f, raw-e0, Ht, Hr, w, raw
        reused = int((pb >= 0).sum())
        src = int(((pb >= 0) & (key_code == self.chart.source_codes(self.codec, p)) & ch).sum()) if M else 0
        return MaterialLocalRows(rows, owners, reason, life, ref_code, src_code, ja, jl, key_code, cur_code, w, -f, e0,
                                 ((w*w)[:, :, None]*cur_var).sum(1), clean_ref, reused, src)

    def _system(self, pts, lr, loc, index, n):
        B, C, dev = len(pts), self.C, self.dev
        rix = torch.where(loc[lr.rows.point] >= 0)[0]
        b, slot = loc[lr.rows.point[rix]], lr.rows.camera[rix]
        look = lambda codes: index.layout.start[index.lookup(codes)]
        src = lr.src_code[rix]
        for codes in (lr.ref_code[rix], lr.key_code[rix], lr.cur_code[rix].reshape(-1), src[src >= 0]):
            if bool((index.lookup(codes) < 0).any()):
                raise LocalCovarianceFailure('material row variable missing from the local Gaussian')
        H = torch.zeros((B, 4*C, n), dtype=torch.float64, device=dev)
        _add_blocks(H, b, slot, torch.zeros_like(b), lr.rows.H_t[rix])
        rc = look(lr.ref_code[rix])
        ch = src >= 0
        st = ~ch
        if bool(st.any()):
            _add_blocks(H, b[st], slot[st], rc[st], lr.rows.H_ref[rix][st])
        if bool(ch.any()):
            hr = lr.rows.H_ref[rix][ch]
            _add_blocks(H, b[ch], slot[ch], rc[ch], hr@lr.ja[rix][ch])
            _add_blocks(H, b[ch], slot[ch], look(src[ch]), hr@lr.jl[rix][ch])
        I4 = torch.eye(4, dtype=torch.float64, device=dev)
        _add_blocks(H, b, slot, look(lr.key_code[rix]), I4.expand(len(rix), -1, -1))
        kc = look(lr.cur_code[rix].reshape(-1)).reshape(-1, 4)
        for a in range(4):
            _add_blocks(H, b, slot, kc[:, a], -lr.weights[rix, a, None, None]*I4)
        valid = torch.zeros((B, C), dtype=torch.bool, device=dev)
        valid[b, slot] = True
        valid = valid.repeat_interleave(4, 1)
        r = torch.zeros((B, 4*C), dtype=torch.float64, device=dev)
        r[b[:, None], 4*slot[:, None]+torch.arange(4, device=dev)] = lr.residual[rix]
        Rw = torch.block_diag(*[self.warp]*C).expand(B, -1, -1)*(valid[:, :, None] & valid[:, None, :])
        if bool(lr.rows.resampling_variance.any()):
            Rw = Rw.clone()
            d = 4*slot[:, None]+torch.arange(4, device=dev)
            Rw[b[:, None], d, d] += lr.rows.resampling_variance[rix]
        return H, r, Rw, valid, b

    def _row_codes(self, lr, rows):
        return torch.cat((lr.key_code[rows], lr.cur_code[rows].reshape(-1)))

    def _retention(self, layout, refs):
        return material_retention_mask(layout, refs, self.chart, self.k, self.ordinal, self.lag)

    def _expire(self, layout, keep, refs):
        return material_record_expiry(layout, keep, refs, self.expired_upto, self.frame_ordinal)

    # ---------- profiled MAP objective, split into prior and per-identity terms
    def prior_cost(self, name='it', zname='z'):
        cost = torch.zeros(self.N, dtype=torch.float64, device=self.dev)
        cost.index_add_(0, self.entry_point, (self.P[name]-self.mu)*self.P[zname])
        old = self.ex.frame != self.k
        if bool(old.any()):
            cost.index_add_(0, self.ex.point[old], (self.ex.v[name][old]*self.ex.v[zname][old]).sum(1))
        return cost/2

    def identity_costs(self, lr):
        """(C,N) negative log marginal likelihood of each valid identity with its fresh current raw errors integrated out:
        1/2 e0^T R^-1 e0 + 1/2 log det R,  R = sum_a w_a^2 R_a + R_resampling + R_warp. R depends on q_t through the
        bilinear weights (and the optional resampling term), so the log-determinant is kept: without it the comparison
        would prefer integer (sensor) or half-pixel (resampling) positions for no physical reason."""
        cost = torch.zeros((self.C, self.N), dtype=torch.float64, device=self.dev)
        valid = torch.zeros((self.C, self.N), dtype=torch.bool, device=self.dev)
        if len(lr):
            R = torch.diag_embed(lr.r_eff+lr.rows.resampling_variance)+self.warp
            e = lr.e0
            L = torch.linalg.cholesky(R)
            w = torch.cholesky_solve(e.unsqueeze(-1), L).squeeze(-1)
            logdet = 2*torch.log(L.diagonal(dim1=1, dim2=2)).sum(1)
            cost[lr.rows.camera, lr.rows.point] = ((e*w).sum(1)+logdet)/2
            valid[lr.rows.camera, lr.rows.point] = True
        return cost, valid

    def objective(self, lr, name='it', zname='z'):
        return self.prior_cost(name, zname)+self.identity_costs(lr)[0].sum(0)

    def _on_foreground(self, owners):
        c = torch.arange(self.C, device=self.dev)[:, None].expand(-1, self.N)
        inside = (owners.reason != 1) & (owners.reason != 2)
        pix = owners.pixel.clamp(0, self.frame.H*self.frame.W-1)
        fg = self.frame.mask[c, pix//self.frame.W, pix % self.frame.W]
        return inside & fg, inside

    def compare(self, base, trial, trial_prior):
        """Per-point acceptance of `trial` against `base` on stable material identities. Returns (good, reasons)."""
        cb, vb, fgb = base['cost'], base['valid'], base['fg']
        ct, vt = trial['cost'], trial['valid']
        lost = vb & ~vt
        code = trial['reason']
        bad_loss = lost & torch.isin(code, torch.tensor(INVALIDATING_CODES, dtype=code.dtype, device=self.dev))
        hidden_loss = lost & ~bad_loss
        identity_ok = ~bad_loss.any(0)
        fg_t, inside_t = trial['fg'], trial['inside']
        mask_ok = ~(fgb & inside_t & ~fg_t).any(0)
        common = vb & vt
        supported = common.any(0) | ~vb.any(0)
        jb = base['prior']+(cb*common).sum(0)
        jt = trial_prior+(ct*common).sum(0)
        descent = jt <= jb+1e-12*jb.abs().clamp_min(1.)
        good = identity_ok & mask_ok & supported & descent
        reasons = {'identity_support_lost': ~identity_ok, 'outside_mask': identity_ok & ~mask_ok,
                   'no_common_identity': identity_ok & mask_ok & ~supported,
                   'no_common_map_descent': identity_ok & mask_ok & supported & ~descent}
        return good, reasons, hidden_loss, vt & ~vb

    def _state(self, lr, name, zname):
        cost, valid = self.identity_costs(lr)
        fg, inside = self._on_foreground(lr.owners)
        return {'cost': cost, 'valid': valid, 'fg': fg, 'inside': inside, 'reason': self._reason(lr),
                'prior': self.prior_cost(name, zname)}

    def _reason(self, lr):
        return lr.reason

    def _condition_current(self, points, s6):
        """Iterate = conditional mean of the local prior given the current 6D state s6 for `points`:
        a = mu + Sigma[:, :6] Sigma_66^-1 (s6 - mu_6),  z = Sigma^-1 (a - mu) = E_6 Sigma_66^-1 (s6 - mu_6)."""
        from estimator.temporal_local_covariance import chunk_plan
        L = self.prior.layout
        for pts in chunk_plan(torch.where(points)[0], L.dims, self.budget):
            mu, Sig = self.prior.gather(pts)
            n = mu.shape[1]
            w = torch.linalg.solve(Sig[:, :6, :6], (s6[pts]-mu[:, :6]).unsqueeze(-1))
            a = mu+(Sig[:, :, :6]@w).squeeze(-1)
            z = torch.zeros_like(mu)
            z[:, :6] = w.squeeze(-1)
            vm = torch.arange(n, device=self.dev)[None, :] < L.dims[pts][:, None]
            idx = (self.prior.moff[pts][:, None]+torch.arange(n, device=self.dev))[vm]
            self.P['it'][idx] = a[vm]
            self.P['z'][idx] = z[vm]
        x = points[self.ex.point]
        self.ex.v['it'][x] = 0
        self.ex.v['z'][x] = 0

    def _pyramid_start(self, lr):
        """Coarse-to-fine starting iterate (temporal_material_pyramid). Kept per point only when, at ORIGINAL resolution,
        it passes the same identity/mask/common-descent comparison against the prediction; otherwise the point restarts
        from the prediction. Returns the linearization at the chosen start."""
        from estimator.temporal_local_iekf import current_marginals
        from estimator.temporal_material_pyramid import coarse_to_fine
        evaluator, levels = self.pyramid
        mean6, P6 = current_marginals(self.prior)
        s0, report = coarse_to_fine(evaluator, self.build, mean6, P6, levels, self.C)
        moved = (s0 != mean6).any(1)
        self.coarse_report = {'levels': report, 'coarse_moved_points': int(moved.sum())}
        self.coarse_started = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        if not bool(moved.any()):
            return lr
        base = self._state(lr, 'it', 'z')
        self._condition_current(moved, s0)
        lr1 = self.linearize('it')
        joint = self._state(lr1, 'it', 'z')
        good, why, _, _ = self.compare(base, joint, joint['prior'])
        revert = moved & ~good
        if bool(revert.any()):
            self._reset_to_prior(revert)
            lr1 = self.linearize('it')
        self.coarse_started = moved & good
        self.coarse_report.update({'coarse_start_kept_points': int(self.coarse_started.sum()),
                                   'coarse_start_rejected_at_original_resolution': {k: int((v & moved).sum()) for k, v in why.items()}})
        return lr1

    def _reset_to_prior(self, points):
        e = points[self.entry_point]
        self.P['it'][e] = self.mu[e]
        self.P['z'][e] = 0
        x = points[self.ex.point]
        self.ex.v['it'][x] = 0
        self.ex.v['z'][x] = 0

    def solve(self, max_iters=10, tol=1e-8):
        N, dev = self.N, self.dev
        lr = self.linearize('it')
        self.prior_rows = lr          # read-only handle for diagnostics (linearization at the prediction)
        ever_accepted = torch.zeros(N, dtype=torch.bool, device=dev)
        point_converged = torch.zeros(N, dtype=torch.bool, device=dev)
        observed0 = torch.zeros(N, dtype=torch.bool, device=dev)
        observed0[lr.rows.point] = True
        target0 = torch.where(lr.owners.reason <= 6, lr.owners.pixel, -1)
        stationary = torch.zeros(N, dtype=torch.bool, device=dev)
        prior_cost = self.objective(lr)
        if self.pyramid is not None:
            lr = self._pyramid_start(lr)
            ever_accepted |= self.coarse_started
        history, converged, nis = [], not len(lr), None
        for it in range(max_iters if len(lr) else 0):
            observed = torch.zeros(N, dtype=torch.bool, device=dev)
            observed[lr.rows.point] = True
            step_nis = self.proposal(lr, observed, first=it == 0)
            if it == 0:
                nis = step_nis
            base = self._state(lr, 'it', 'z')
            old = float((base['prior']+base['cost'].sum(0)).sum())
            self._set('base', 'it')
            self._set('zbase', 'z')
            proposal_metric = self._metric('it', 'z', 'prop', 'zprop')
            point_converged |= observed & (proposal_metric < tol)
            pending = observed.clone()
            accepted_points = torch.zeros_like(pending)
            trial_cost, full_step, reasons = None, None, None
            for ls in range(25):
                rate = 2.**-ls
                self._set('trial', 'it')
                self._set('ztrial', 'z')
                self._set('trial', 'base', 'prop', rate, pending)
                self._set('ztrial', 'zbase', 'zprop', rate, pending)
                tr = self.linearize('trial', register=False)
                good, why, _, _ = self.compare(base, self._state(tr, 'trial', 'ztrial'), self.prior_cost('trial', 'ztrial'))
                good &= pending
                reasons = {key: int((pending & v).sum()) for key, v in why.items()}
                if ls == 0:
                    full_step = reasons
                self._set('it', 'trial', points=good)
                self._set('z', 'ztrial', points=good)
                accepted_points |= good
                pending &= ~good
                trial_cost = float(self.objective(tr, 'trial', 'ztrial').sum())
                if not bool(pending.any()):
                    break
            if bool(pending.any()) and ls == 24:
                # the smallest backtracked step still fails ONLY on MAP descent: a local minimum along the GN direction
                stationary |= pending & why['no_common_map_descent']
            for key, v in reasons.items():
                self.counters['trial_rejections'][key] += v
            # accepted points interact through first-hit visibility: verify the joint state, roll back offenders
            for verification in range(25):
                vr = self.linearize('it')
                joint = self._state(vr, 'it', 'z')
                good, _, hidden, gained = self.compare(base, joint, joint['prior'])
                bad = accepted_points & ~good
                if not bool(bad.any()):
                    break
                self._set('it', 'base', points=bad)
                self._set('z', 'zbase', points=bad)
                accepted_points &= ~bad
            else:
                self._set('it', 'base')
                self._set('z', 'zbase')
                accepted_points[:] = False
                vr = self.linearize('it')
                joint = self._state(vr, 'it', 'z')
                _, _, hidden, gained = self.compare(base, joint, joint['prior'])
            self.counters['identities_lost_to_occlusion_accepted'] += int((hidden & accepted_points[None]).sum())
            self.counters['identities_gained_accepted'] += int((gained & accepted_points[None]).sum())
            lr = vr
            accepted = bool(accepted_points.any())
            ever_accepted |= accepted_points
            actual = self._metric('base', 'zbase', 'it', 'z')
            metric = float(actual.max()) if N else 0.
            history.append({'iteration': it, 'objective': old, 'proposal_objective': trial_cost, 'accepted': accepted,
                            'step_prior_metric_max': metric, 'backtracks': ls, 'accepted_points': int(accepted_points.sum()),
                            'rejected_points': int((pending & ~accepted_points).sum()), 'rollback_verifications': verification,
                            'full_step_rejections': full_step, 'smallest_step_rejections': reasons,
                            'identities_lost_to_occlusion': int((hidden & accepted_points[None]).sum()),
                            'identities_gained': int((gained & accepted_points[None]).sum()),
                            'reason': 'ACCEPTED' if accepted else 'IDENTITY_OR_MASK_OR_COMMON_DESCENT_REJECTED'})
            if not accepted:
                break
            unresolved = (~accepted_points) & observed & (proposal_metric >= tol)
            if metric < tol and not bool(unresolved.any()):
                converged = True
                break
        observed = torch.zeros(N, dtype=torch.bool, device=dev)
        observed[lr.rows.point] = True
        # a point whose accepted step is no longer supported by any valid identity at the final joint state (others moved
        # in front of it) has no observation to condition on: exact prediction-only, never an unsupported moved mean
        orphan = ever_accepted & ~observed
        if bool(orphan.any()):
            self._reset_to_prior(orphan)
        self.counters['orphaned_accepted_points_reset_to_prediction'] = int(orphan.sum())
        self.final = lr
        # observed points are conditioned when a step was accepted, when the proposal already sits at the iterate, or when
        # the iterate is a local minimum along the GN direction (e.g. an exact constant-velocity prediction); points
        # rejected for identity/mask validity stay prediction-only
        stationary = observed & (point_converged | stationary) & ~ever_accepted
        self.counters['stationary_observed_points_conditioned'] = int(stationary.sum())
        self.assimilated = observed & (ever_accepted | stationary)
        target = torch.where(lr.owners.reason <= 6, lr.owners.pixel, -1)
        changed = ((target != target0) & (target0 >= 0) & (target >= 0)).any(0) & self.assimilated
        self.counters['target_pixel_changes_accepted'] = int(changed.sum())
        self.target_changed = changed
        final_cost = self.objective(lr)
        return {'history': history, 'converged': converged, 'point_converged': point_converged | ~observed,
                'assimilated_points': int(self.assimilated.sum()), 'eligible_but_rejected_points': int((observed & ~self.assimilated).sum()),
                'observed_points': int(observed.sum()), 'observed_at_prior': int(observed0.sum()),
                'nonconverged_points': int((observed & ~point_converged).sum()),
                'predictive_nis': nis if nis is not None else torch.zeros(N, dtype=torch.float64, device=dev),
                'prior_objective': float(prior_cost.sum()), 'accepted_objective': float(final_cost.sum()),
                'final_point_objective': final_cost, 'counters': {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.counters.items()},
                'coarse_to_fine': self.coarse_report,
                'reference_uncertainty': QUALIFIER, 'fixed_prior': True, 'observation_domain': OBSERVATION_DOMAIN,
                'objective_scope': 'material identities (point, camera, reference frame, source pixel) valid at both base and trial; '
                                   'current target pixel and first-hit owners free; current sensor errors profiled; R_warp only additional discrepancy'}
