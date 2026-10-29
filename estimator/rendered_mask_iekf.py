"""Fixed-prior IEKF with the rendered white-point mask as a second image measurement (RGBDiekf_Codex_Rendered_Binary_
Mask_IEKF_Replacement.md, sections 4-5). Used when configs/projective_rgbd.json rendered_mask_measurement.mode = on.

Exact 3D reduction (measurements act on positions only). With the predicted blocks S = P-_xx, B = P-_vx:
    L = B S^-1,  C = P-_vv - B S^-1 B^T,   v(x) = v- + L (x - x-)        (the prior-conditional velocity, exact)
so the fixed-prior IEKF is a 3N-position problem with prior S^-1 and every 6D block is reconstructed afterwards.

At iterate X (all points), with the UNCHANGED direct RGB-D terms of point_kalman_cuda.PointMeasurement (I_rgbd, g_rgbd at
X, existing visibility / row admission) and the mask rows of rendered_mask_measurement (measured-background pixels touched
by a white-point footprint, r = 0 - m_hat, R = declared mask variance):
    A dX = b,   A = blockdiag(S^-1 + I_rgbd) + J^T R^-1 J,   b = -S^-1 (X - X-) + g_rgbd + J^T R^-1 r
J couples every point sharing a background pixel (its off-diagonal blocks J_i^T R^-1 J_j are KEPT). Points are grouped
into connected components through shared rows (label propagation); a point with no background row is its own
component and its 3x3 system is solved exactly; coupled components are solved together by a matrix-free conjugate
gradient with a 3x3 block-Jacobi preconditioner (diagonal blocks including the point's own mask information), with
per-component step lengths (independent CG runs, batched). CG stops when every component's preconditioned residual
energy r^T M^-1 r (~ the A-norm error of the step) is below 1e-2 times the IEKF step tolerance (it cannot change the
convergence decision), or at the floating-point floor (64 eps)^2 of the initial energy; failures are counted.

Globalization (the actual combined objective with the same fixed prior and measured frame, never an external loss):
    Phi(X) = 0.5 |X - X-|^2_{S^-1} + 0.5 sum_rgbd r^2/R + 0.5 sum_background_pixels m_hat^2/R
per component, by backtracking on the step (alpha = 1, 1/2, ..., 2^-10); a component accepts the first alpha that does
not increase its objective, else does not move (stalled, reported). The RGB-D term of a trial is evaluated on the rows
admitted at the linearization, re-sampled at the trial position (a row that cannot be sampled there rejects the trial:
objectives over different row sets are not comparable); the mask term of a component is its rows at the trial with the
other components held at their current positions (components are disjoint at the linearization). The per-point RGB-D
chi-square acceptance of update_points (last_consistent_iterate) is NOT used here: a coupled image solve cannot be
reverted point by point; RGB-D consistency is reported separately (posterior_cost, rgbd_consistent).

Covariance (named approximation 'pixel_marginal_adf', NOT the exact marginal of the joint posterior A^-1, which is
dense when pixels are shared): each point's mask information uses an effective noise that absorbs the prior position
uncertainty of the OTHER contributors of the pixel,
    R_eff[p,i] = R[p] + sum_{j != i} J[p,j] S_j J[p,j]^T,
    S_i+ = (S_i^-1 + I_rgbd,i + sum_p J[p,i]^T R_eff[p,i]^-1 J[p,i])^-1,
    P_i+ = [[S_i+, S_i+ L_i^T], [L_i S_i+, C_i + L_i S_i+ L_i^T]].
A point with no RGB-D row and no mask row at the final linearization that did not move keeps (s-, P-) exactly.
"""
import math
import torch
from scipy.stats import chi2, norm
from estimator.point_kalman_cuda import PointMeasurement, Timers, spd_factor, eigvalsh3, _row_dict
from estimator.projective_visibility import position_sigma_z2
from estimator.projective_terms import build_terms
from estimator.projective_sensor import CODE, REASONS
from estimator.rendered_mask_measurement import footprints, mask_rows, hard_pseudomask, frame_statistics

MASK_STATUS = {0: 'NOT_EVALUATED', 1: 'NO_MASK_ROWS', 2: 'MASK_AND_RGBD', 3: 'MASK_ONLY', 4: 'MASK_ROWS_UNIFORM_BACKGROUND', 5: 'DIAGNOSTIC_ONLY'}
MCODE = {v: k for k, v in MASK_STATUS.items()}
ACCEPTANCE_ON = {1: 'DIRECT_RGBD_ROWS_AT_FINAL_LINEARIZATION', 4: 'NO_DIRECT_RGBD_ROWS'}
LINE_SEARCH_HALVINGS = 10
SAMPLEABLE = (CODE['VALID'], CODE['MEASURED_SURFACE_IN_FRONT'], CODE['INNOVATION_INCONSISTENT'])


def components(rows, N, device):
    """Connected components of points through shared background rows (label = smallest point id of the component)."""
    label = torch.arange(N, device=device)
    if rows.n_rows == 0: return label
    for _ in range(N):
        rmin = torch.full((rows.n_rows,), N, dtype=label.dtype, device=device).scatter_reduce_(0, rows.pair_row, label[rows.pair_point], 'amin')
        new = label.clone(); new.scatter_reduce_(0, rows.pair_point, rmin[rows.pair_row], 'amin')
        if torch.equal(new, label): return label
        label = new
    return label


def seg_sum(values, cid_lengths):
    return torch.segment_reduce(values, 'sum', lengths=cid_lengths)


class MaskSystem:
    """The coupled part of A on a set of points (sorted by component), on local indices (O(pairs), not O(N))."""
    def __init__(self, rows, idx, comp_len, Ablk, N):
        self.rows = rows.restrict(idx, N)
        self.idx, self.comp_len, self.n = idx, comp_len, len(idx)
        self.Ablk = Ablk
        self.winv = 1/self.rows.var
        D = self.rows.block_information(self.n, self.winv)
        self.Mchol = torch.linalg.cholesky(Ablk+D)
        self.cid = torch.repeat_interleave(torch.arange(len(comp_len), device=idx.device), comp_len)

    def matvec(self, v):
        y = self.rows.jtw(self.rows.jv(v)*self.winv, self.n)
        return (self.Ablk@v[..., None])[..., 0]+y

    def precond(self, r):
        return torch.cholesky_solve(r[..., None], self.Mchol)[..., 0]


PCG_FORCING = 1e-3     # inexact-Newton forcing term: relative error of the step in the A-norm


def pcg(sysm, b, tol_energy, max_iter, forcing=PCG_FORCING):
    """Batched per-component preconditioned CG (independent scalars per component). A component stops when its
    preconditioned residual energy r^T M^-1 r (~ the A-norm error of the step) is below max(tol_energy, PCG_FORCING^2 x
    the initial energy): an inexact Newton step (relative error 1e-3, still a descent direction; the line search keeps
    the iteration monotone), exact to the IEKF tolerance near convergence (tol_energy = 1e-2 x the step tolerance)."""
    x = torch.zeros_like(b); r = b.clone(); z = sysm.precond(r); p = z.clone()
    rz = seg_sum((r*z).sum(-1), sysm.comp_len)
    floor = torch.maximum((64*torch.finfo(b.dtype).eps)**2*rz, forcing**2*rz)
    done = rz <= torch.maximum(floor, torch.full_like(rz, tol_energy))
    it = 0
    for it in range(1, max_iter+1):
        if bool(done.all()): it -= 1; break
        Ap = sysm.matvec(p)
        pAp = seg_sum((p*Ap).sum(-1), sysm.comp_len)
        alpha = torch.where(done | (pAp <= 0), torch.zeros_like(rz), rz/torch.where(pAp > 0, pAp, torch.ones_like(pAp)))
        x = x+alpha[sysm.cid, None]*p; r = r-alpha[sysm.cid, None]*Ap
        z = sysm.precond(r)
        rz_new = seg_sum((r*z).sum(-1), sysm.comp_len)
        beta = torch.where(done, torch.zeros_like(rz), rz_new/torch.where(rz > 0, rz, torch.ones_like(rz)))
        p = z+beta[sysm.cid, None]*p
        rz = torch.where(done, rz, rz_new)
        done = done | (rz <= torch.maximum(floor, torch.full_like(rz, tol_energy)))
    return x, {'iterations': it, 'components': int(len(rz)), 'not_converged_components': int((~done).sum()),
               'max_residual_energy': float(rz.max()) if len(rz) else 0., 'tolerance_energy': tol_energy}


def update_points_rendered_mask(s_minus, P_minus, frame, rig, rgb0, noise, policy, max_iters, tol, chunk=200_000, dtype=torch.float64,
                                trace_ids=None, timers=None, assignment='per-frame', mask_var=None):
    """Returns (s+, P+, report, trace) like point_kalman_cuda.update_points; report['mask'] holds the image-channel results."""
    timers = timers or Timers(s_minus.device)
    N, dev = len(s_minus), s_minus.device
    s_minus, P_minus = s_minus.to(dtype), P_minus.to(dtype)
    C, H, W = rig.count, frame.H, frame.W
    background = ~frame.mask
    var_c = torch.as_tensor(mask_var, dtype=dtype, device=dev).expand(C).clone()
    with timers('prior_information'):
        S = P_minus[:, :3, :3]; Bvx = P_minus[:, 3:, :3]
        Ls, retry = spd_factor(S, 'prior position covariance P-_xx')
        Sinv = torch.cholesky_inverse(Ls).to(dtype); Sinv = (Sinv+Sinv.transpose(1, 2))/2
        L = torch.cholesky_solve(Bvx.transpose(1, 2), Ls).transpose(1, 2)        # B S^-1
        Cc = P_minus[:, 3:, 3:]-L@Bvx.transpose(1, 2)
        sigma_z2 = position_sigma_z2(rig.R.to(dtype), S)
    meas = PointMeasurement(rig, frame, noise, policy, rgb0.to(dtype), S, sigma_z2, chunk, dtype, timers=timers, assignment=assignment)
    recheck = policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion'
    keep = None if trace_ids is None else torch.as_tensor(trace_ids, device=dev)
    x_minus = s_minus[:, :3]; X = x_minus.clone()
    prior_obj = lambda Xs, idx: 0.5*(((Xs-x_minus[idx])[:, None, :]@Sinv[idx]@(Xs-x_minus[idx])[:, :, None])[:, 0, 0])
    converged = torch.zeros(N, dtype=torch.bool, device=dev); stalled = torch.zeros_like(converged)
    iters = torch.zeros(N, dtype=torch.long, device=dev); had_mask = torch.zeros_like(converged)
    rule = policy.get('update_acceptance', 'final_iterate')
    if rule not in ('last_consistent_iterate', 'final_iterate'): raise ValueError(f'update_acceptance {rule!r}')
    alpha_t = 2*norm.sf(policy['visibility_n_sigma'])
    thr = torch.as_tensor([float('inf')]+[float(chi2.isf(alpha_t, 4*k)) for k in range(1, C+1)], dtype=dtype, device=dev)
    allp = torch.arange(N, device=dev)
    acc_state, acc_iter = x_minus.clone(), torch.full((N,), -1, dtype=torch.long, device=dev)
    had_rows = torch.zeros_like(converged)
    history, trace = [], ({'iterations': []} if keep is not None else None)
    nis = nis_dof = prior_innov = None; pcg_log = []
    for j in range(max_iters):
        act_pts = ~converged
        with timers('mask_render'):
            fp = footprints(X, rig, H, W, background)
            rows = mask_rows(fp, background, var_c, H, W, N)
            has_rows = torch.zeros(N, dtype=torch.bool, device=dev); has_rows[rows.point_ids] = True
            had_mask |= has_rows
            comp = components(rows, N, dev)
            comp_active = torch.zeros(N, dtype=torch.int32, device=dev).scatter_reduce_(0, comp, act_pts.int(), 'amax') > 0   # (no bool reductions on CUDA)
            act = comp_active[comp]                                    # a converged point re-joins when its component moves
        with timers('rgbd_terms'):
            m = meas(X, keep, None if j == 0 else act, X-x_minus)
        with timers('mask_solve'):
            Ablk = Sinv.clone(); Ablk += m['Ixx']
            b = -(Sinv@(X-x_minus)[..., None])[..., 0]+m['gx']+rows.jtw(rows.r/rows.var, N)
            delta = torch.zeros_like(X)
            unc = torch.nonzero(act & ~has_rows).flatten()
            if len(unc):
                Lu, _ = spd_factor(Ablk[unc], f'information block (iteration {j})')
                delta[unc] = torch.cholesky_solve(b[unc][..., None], Lu)[..., 0]
            cpl = torch.nonzero(act & has_rows).flatten()
            info = {'iterations': 0, 'components': 0, 'not_converged_components': 0}
            if len(cpl):
                cpl = cpl[torch.argsort(comp[cpl], stable=True)]
                _, comp_len = torch.unique_consecutive(comp[cpl], return_counts=True)
                sysm = MaskSystem(rows, cpl, comp_len, Ablk[cpl], N)
                d_c, info = pcg(sysm, b[cpl], 1e-2*tol, 3*int(comp_len.max())+10)
                delta[cpl] = d_c
            pcg_log.append(info)
        with timers('acceptance'):                                    # RGB-D consistency of the iterate (the existing rule, never-coupled points)
            nr = m['nrows']; had_rows |= act & (nr > 0)
            J = 2*prior_obj(X, allp)+m['rTWr']
            if j >= 1:
                okc = act & (nr > 0) & (J <= thr[nr.clamp_max(C)])
                acc_state = torch.where(okc[:, None], X, acc_state); acc_iter = torch.where(okc, j, acc_iter)
        if j == 0:                                                     # RGB-D NIS at the prior (unchanged definition)
            Lb, _ = spd_factor(Sinv+m['Ixx'], 'prior + RGB-D information')
            gq = torch.cholesky_solve(m['gx'][..., None], Lb)[..., 0]
            nis = (m['rTWr']-(m['gx']*gq).sum(-1)).clamp_min(0); nis_dof = 4*m['nrows']
            prior_innov = (m['r_valid'], m['cam_valid'], m['reason'])
        # a step whose energy delta^T A delta (= delta^T b for the solved system) is below the IEKF step tolerance is the
        # original filter's convergence: taken as is (its objective change is at rounding level). Points with no mask row
        # take the plain IEKF step (the existing RGB-D path); mask-coupled components are line-searched.
        e_step = (delta*b).sum(-1)
        comp_e = torch.zeros(N, dtype=dtype, device=dev).index_add_(0, comp, torch.where(act, e_step, torch.zeros_like(e_step)))
        comp_size = torch.bincount(comp, minlength=N).to(dtype)
        small = act & (comp_e[comp] <= tol*comp_size[comp].clamp_min(1))
        ls = act & has_rows & ~small
        with timers('line_search'):
            alpha, phi0, phi1 = line_search(X, delta, ls, comp, rows, m, x_minus, Sinv, S, rig, frame, rgb0.to(dtype), noise, policy, dtype,
                                            background, var_c, H, W, N, prior_obj)
        alpha = torch.where(act & ~ls, torch.ones_like(alpha), alpha)
        step_x = alpha[:, None]*delta
        moved = act & (alpha > 0)
        newly = act & (small | (ls & (alpha == 0)))                        # converged step, or no non-increasing joint step from here
        stalled |= ls & (alpha == 0)
        if trace is not None:
            trace['iterations'].append(_trace_iteration(j, X, delta, alpha, rows, m, keep, comp, phi0, phi1, x_minus, act, W))
        X = torch.where(moved[:, None], X+step_x, X)
        iters += act.long()
        converged = converged | newly
        history.append({'iteration': j, 'rows': m['rows_considered'], 'mask_rows': rows.n_rows, 'active_points': int(act.sum()),
                        'coupled_points': int(len(cpl)), 'line_searched_points': int(ls.sum()), 'pcg': info,
                        'alpha_backtracked': int((ls & (alpha > 0) & (alpha < 1)).sum()), 'alpha_zero': int((ls & (alpha == 0)).sum()),
                        'objective_change_line_searched': float(phi1[torch.unique(comp[ls])].sum()) if bool(ls.any()) else 0.,
                        'not_converged': int((~converged).sum())})
        if bool(converged.all()): break
    # acceptance: mask-coupled points (any iteration) keep the final iterate of the monotone joint iteration; points the mask
    # never touched follow the existing per-point rule exactly (update_points): their footprints never met a background
    # pixel, so reverting them cannot change the rendered image of any other point
    with timers('rgbd_terms'):
        final = meas(X, keep, None, X-x_minus, recheck)
    with timers('acceptance'):
        nf = final['nrows']; had_rows |= nf > 0
        Jf = 2*prior_obj(X, allp)+final['rTWr']
        Ixx_f, nrows_f, reason_f = final['Ixx'].clone(), nf.clone(), final['reason'].clone()
        plain = ~had_mask
        if rule == 'final_iterate':
            ok_f = nf > 0; ok_e = torch.zeros_like(ok_f)
        else:
            ok_f = (nf > 0) & (Jf <= thr[nf.clamp_max(C)])
            fb = plain & ~ok_f & (acc_iter >= 1)
            ok_e = torch.zeros_like(ok_f)
            if bool(fb.any()):
                X = torch.where(fb[:, None], acc_state, X)
                m2 = meas(X, keep, fb, X-x_minus, recheck)
                J2 = 2*prior_obj(X, allp)+m2['rTWr']
                ok_e = fb & (m2['nrows'] > 0) & (J2 <= thr[m2['nrows'].clamp_max(C)])
                Ixx_f[fb] = m2['Ixx'][fb]; nrows_f[fb] = m2['nrows'][fb]; reason_f[:, fb] = m2['reason'][:, fb]
        observed_plain = ok_f | ok_e
        revert = plain & ~observed_plain                                  # prediction only (s-, P-) as in update_points
        X = torch.where(revert[:, None], x_minus, X)
        Ixx_f = torch.where(revert[:, None, None], torch.zeros_like(Ixx_f), Ixx_f); nrows_f = torch.where(revert, torch.zeros_like(nrows_f), nrows_f)
        observed = torch.where(plain, observed_plain, nrows_f > 0)
        acceptance = torch.where(plain, torch.where(ok_f, 1, torch.where(ok_e, 2, torch.where(had_rows, 3, 4))), torch.where(nrows_f > 0, 1, 4)).to(torch.uint8)
    with timers('mask_render'):
        fp = footprints(X, rig, H, W, background); rows = mask_rows(fp, background, var_c, H, W, N)
        Bh = hard_pseudomask(fp, C, H, W)
    with timers('covariance'):
        S_plus, P_plus, s_plus, mask_info = posterior(X, s_minus, P_minus, S, Sinv, Ixx_f, rows, L, Cc, nrows_f, dtype)
    has_rows_f = torch.zeros(N, dtype=torch.bool, device=dev); has_rows_f[rows.point_ids] = True
    grad = rows.jtw(rows.r/rows.var, N).norm(dim=1)
    status = torch.full((N,), MCODE['NO_MASK_ROWS'], dtype=torch.long, device=dev)
    status = torch.where(has_rows_f & observed, MCODE['MASK_AND_RGBD'], status)
    status = torch.where(has_rows_f & ~observed, MCODE['MASK_ONLY'], status)
    status = torch.where(has_rows_f & ~fp.straddle, MCODE['MASK_ROWS_UNIFORM_BACKGROUND'], status)   # no camera's footprint meets the boundary
    J_rgbd = 2*prior_obj(X, allp)+torch.where(observed, final['rTWr'], torch.zeros_like(final['rTWr']))
    rgbd_consistent = observed & (J_rgbd <= thr[nrows_f.clamp_max(C)])
    stats = frame_statistics(fp, rows, background, Bh, H, W)
    stats.update(mask_only_points=int((status == MCODE['MASK_ONLY']).sum()), uniform_background_points=int((status == MCODE['MASK_ROWS_UNIFORM_BACKGROUND']).sum()),
                 points_with_mask_rows_any_iteration=int(had_mask.sum()), stalled_points=int(stalled.sum()),
                 mask_gradient_norm_total=float(grad.norm()), pcg=pcg_log, covariance_model='pixel_marginal_adf_approximation',
                 effective_noise_inflation_quantiles=mask_info,
                 acceptance='mask-coupled points: final iterate of the monotone joint iteration; never-coupled points: ' + rule)
    report = {'s_iter_final': torch.cat([X, s_plus[:, 3:]], 1), 'observed': observed, 'nrows': torch.where(observed, nrows_f, torch.zeros_like(nrows_f)),
              'reason': reason_f, 'front': final['front'],
              'converged': converged, 'point_iterations': iters, 'step': torch.zeros(N, dtype=dtype, device=dev), 'nis': nis, 'nis_dof': nis_dof,
              'iterations': len(history), 'history': history, 'info_eig': eigvalsh3(Ixx_f), 'proj': final['proj'], 'prior_innovation': prior_innov,
              'final_innovation': (final['r_valid'], final['cam_valid']), 'cam_information_trace': final['cam_information_trace'],
              'acceptance': acceptance, 'posterior_cost': J_rgbd, 'posterior_cost_dof': 4*nrows_f,
              'accepted_iteration': torch.where(observed, iters, -1), 'acceptance_rule': 'rendered_mask: joint monotone (coupled) / '+rule+' (never coupled)',
              'float64_retry': [r for r in (retry,) if r],
              'mask': {'status': status, 'rows_per_point': _rows_per_point(rows, N), 'gradient_norm': grad, 'rgbd_consistent': rgbd_consistent,
                       'stalled': stalled, 'had_rows_any_iteration': had_mask, 'statistics': stats, 'hard_pseudomask': Bh, 'rows': rows, 'footprints': fp}}
    if trace is not None:
        trace['final'] = _trace_final(X, s_minus, s_plus, P_minus, P_plus, rows, final, keep, status, rgbd_consistent, J_rgbd, thr, C)
    return s_plus, P_plus, report, trace


def _rows_per_point(rows, N):
    out = torch.zeros(N, dtype=torch.long, device=rows.r.device if rows.n_rows else rows.point_ids.device)
    if rows.n_rows: out[rows.point_ids] = rows.point_len
    return out


def line_search(X, delta, act, comp, rows, m, x_minus, Sinv, S, rig, frame, rgb0, noise, policy, dtype, background, var_c, H, W, N, prior_obj):
    """Per-component backtracking on the actual objective (module docstring). Returns alpha [N], phi before / after per point
    component attribution (for the history)."""
    dev = X.device
    alpha = torch.zeros(N, dtype=dtype, device=dev)
    nonzero = act & (delta.abs().amax(1) > 0)
    if not bool(nonzero.any()):
        z = torch.zeros(N, dtype=dtype, device=dev); return alpha, z, z
    # Comparisons are over the SAME measurement set at both ends of a trial (objectives over different sets are not
    # comparable): RGB-D rows admitted at the linearization and still sampleable at the trial (a 2x2 support flipping to
    # a masked cell at a pixel-centre knife edge drops that row from BOTH sums); mask pixels touched by the component at
    # the linearization OR at the trial, each with its m_hat at both states (a pixel lit by another component is not
    # counted as a new cost). The decision is on the change dPhi_c <= 0.
    rr = torch.nonzero((m['reason'].T == CODE['VALID']) & act[:, None])           # (point, camera), ordered by point
    allp = torch.arange(N, device=dev)
    cost0 = None
    if len(rr):
        t0 = build_terms(rig, frame, rr[:, 1], rr[:, 0], X[rr[:, 0]], rgb0[rr[:, 0]], S[rr[:, 0]], noise, policy, dtype, X[rr[:, 0]]-x_minus[rr[:, 0]])
        cost0 = 0.5*(t0.r*t0.r/t0.var).sum(1)
    prior0 = prior_obj(X, allp)
    key0 = rows.row_cam*(H*W)+rows.row_pix                                            # sorted (rows are built in key order)
    pending = torch.zeros(N, dtype=torch.int32, device=dev).scatter_reduce_(0, comp, nonzero.int(), 'amax') > 0   # components with a step
    a = 1.
    d_final = torch.zeros(N, dtype=dtype, device=dev)
    for _ in range(LINE_SEARCH_HALVINGS+1):
        if not bool(pending.any()): break
        mv = pending[comp] & act
        Xt = torch.where(mv[:, None], X+a*delta, X)
        d_pt = torch.where(mv, prior_obj(Xt, allp)-prior0, torch.zeros_like(prior0))
        if len(rr):
            sel = mv[rr[:, 0]]
            p, c = rr[sel, 0], rr[sel, 1]
            if len(p):
                t = build_terms(rig, frame, c, p, Xt[p], rgb0[p], S[p], noise, policy, dtype, Xt[p]-x_minus[p])
                okrow = torch.isin(t.reason, torch.as_tensor(SAMPLEABLE, device=dev, dtype=t.reason.dtype))
                dr = torch.where(okrow, 0.5*(t.r*t.r/t.var).sum(1)-cost0[sel], torch.zeros_like(cost0[sel]))
                d_pt = d_pt.index_add(0, p, dr)
        d_c = torch.zeros(N, dtype=dtype, device=dev).index_add_(0, comp, d_pt)+_mask_change(rows, key0, Xt, rig, background, comp, mv, H, W, dtype, var_c)
        ok = pending & (d_c <= 0)
        alpha = torch.where(ok[comp] & act, torch.full_like(alpha, a), alpha)
        d_final = torch.where(ok, d_c, d_final)
        pending = pending & ~ok
        a *= .5
    return alpha, torch.zeros_like(d_final), d_final


def _mask_change(rows0, key0, Xt, rig, background, comp, mv, H, W, dtype, var_c):
    """Per component: sum over the background pixels its moving points touch at the linearization or at the trial of
    0.5 (m_hat_trial^2 - m_hat_0^2) / R. Only the moving points are re-rendered: every other contributor of such a pixel is
    already a pair of the linearization's rows (a background pixel any point touches is a row), so the trial union is
    1 - prod_static (1 - a) prod_moving_trial (1 - a), with the same zero-factor handling as mask_rows."""
    N, dev = len(comp), comp.device
    out = torch.zeros(N, dtype=dtype, device=dev)
    idx = torch.nonzero(mv).flatten()
    if len(idx) == 0: return out
    HW = H*W
    # static (non-moving) factors per linearization row
    if rows0.n_rows:
        a0 = rows0.a; one0 = a0 >= 1; stat = ~mv[rows0.pair_point]
        l0 = torch.where(one0 | ~stat, torch.zeros_like(a0), torch.log1p(-torch.where(one0, torch.zeros_like(a0), a0)))
        lnm = torch.zeros(rows0.n_rows, dtype=dtype, device=dev).index_add_(0, rows0.pair_row, l0)
        znm = torch.zeros(rows0.n_rows, dtype=dtype, device=dev).index_add_(0, rows0.pair_row, (one0 & stat).to(dtype))
    # moving points at the trial
    fpt = footprints(Xt[idx], rig, H, W, background)
    kt = fpt.cam*HW+fpt.pix; pt = idx[fpt.point]; at = fpt.a; onet = at >= 1
    lt = torch.where(onet, torch.zeros_like(at), torch.log1p(-torch.where(onet, torch.zeros_like(at), at)))
    # key universe per component: moving pairs at the linearization and at the trial
    cc_list, kk_list = [comp[pt]], [kt]
    if rows0.n_rows:
        s0 = mv[rows0.pair_point]; cc_list.append(comp[rows0.pair_point[s0]]); kk_list.append(key0[rows0.pair_row[s0]])
    cc = torch.cat(cc_list); kk = torch.cat(kk_list)
    if len(kk) == 0: return out
    span = int(kk.max())+1
    uniq = torch.unique(cc*span+kk); cc, kk = uniq//span, uniq % span
    keys_t, inv_t = torch.unique(kt, return_inverse=True)
    lmt = torch.zeros(len(keys_t), dtype=dtype, device=dev).index_add_(0, inv_t, lt)
    zmt = torch.zeros(len(keys_t), dtype=dtype, device=dev).index_add_(0, inv_t, onet.to(dtype))
    def find(sorted_keys, k):
        if len(sorted_keys) == 0: return torch.zeros(len(k), dtype=torch.long, device=dev), torch.zeros(len(k), dtype=torch.bool, device=dev)
        i = torch.searchsorted(sorted_keys, k).clamp_max(len(sorted_keys)-1)
        return i, sorted_keys[i] == k
    zero = torch.zeros(len(kk), dtype=dtype, device=dev)
    if rows0.n_rows:
        i0, h0 = find(key0, kk)
        m0 = torch.where(h0, rows0.m_hat[i0], zero); L0 = torch.where(h0, lnm[i0], zero); Z0 = torch.where(h0, znm[i0], zero)
    else:
        m0 = L0 = Z0 = zero
    if len(keys_t):                                                   # no moving point on a background pixel at the trial: static part only
        it, ht = find(keys_t, kk)
        Lt = L0+torch.where(ht, lmt[it], zero); Zt = Z0+torch.where(ht, zmt[it], zero)
    else:
        Lt, Zt = L0, Z0
    mt = torch.where(Zt > 0, torch.ones_like(Lt), -torch.expm1(Lt))
    return out.index_add_(0, cc, 0.5*(mt*mt-m0*m0)/var_c[kk//HW])


def posterior(X, s_minus, P_minus, S, Sinv, Ixx, rows, L, Cc, nrows, dtype):
    """pixel_marginal_adf covariance (module docstring) and the 6D reconstruction."""
    N, dev = len(X), X.device
    info = Sinv+Ixx
    infl = None
    if rows.n_rows:
        gS = (rows.g[:, None, :]@S[rows.pair_point]@rows.g[:, :, None])[:, 0, 0]     # J[p,i] S_i J[p,i]^T
        T = torch.segment_reduce(gS, 'sum', lengths=rows.row_len)
        Reff = rows.var[rows.pair_row]+T[rows.pair_row]-gS
        Reff = torch.maximum(Reff, rows.var[rows.pair_row])                         # = R + (others' terms >= 0); guards rounding only
        gg = (rows.g[:, :, None]*rows.g[:, None, :]/Reff[:, None, None])[rows.by_point]
        info[rows.point_ids] += torch.segment_reduce(gg, 'sum', lengths=rows.point_len, axis=0)
        r_ = Reff/rows.var[rows.pair_row]
        infl = torch.quantile(r_, torch.tensor([0., .5, .9, .99, 1.], dtype=dtype, device=dev)).tolist() if len(r_) else None
    Li, _ = spd_factor(info, 'posterior position information')
    Sp = torch.cholesky_inverse(Li); Sp = (Sp+Sp.transpose(1, 2))/2
    v = s_minus[:, 3:]+(L@(X-s_minus[:, :3])[..., None])[..., 0]
    s_plus = torch.cat([X, v], 1)
    P = torch.empty_like(P_minus)
    P[:, :3, :3] = Sp; P[:, 3:, :3] = L@Sp; P[:, :3, 3:] = P[:, 3:, :3].transpose(1, 2); P[:, 3:, 3:] = Cc+L@Sp@L.transpose(1, 2)
    P = (P+P.transpose(1, 2))/2
    has = torch.zeros(N, dtype=torch.bool, device=dev)
    if rows.n_rows: has[rows.point_ids] = True
    untouched = (nrows == 0) & ~has & (X == s_minus[:, :3]).all(1)           # no information and no motion: prediction exactly
    s_plus = torch.where(untouched[:, None], s_minus, s_plus); P = torch.where(untouched[:, None, None], P_minus, P)
    return Sp, P, s_plus, infl


def _trace_iteration(j, X, delta, alpha, rows, m, keep, comp, phi0, phi1, x_minus, act, W):
    out = {'iteration': j, 'points': {}}
    for pid in keep.tolist():
        rows_rgbd = [t.select(t.point == pid) for t in m['kept']]
        rows_rgbd = [r for r in rows_rgbd if len(r.point)]
        sel = (rows.pair_point == pid) if rows.n_rows else torch.zeros(0, dtype=torch.bool, device=X.device)
        pairs = []
        for q in torch.nonzero(sel).flatten().tolist():
            rw = int(rows.pair_row[q]); others = rows.pair_point[rows.pair_row == rw]
            pairs.append({'camera': int(rows.row_cam[rw]), 'pixel_u_v': [int(rows.row_pix[rw]) % W, int(rows.row_pix[rw])//W],
                          'coverage_a': float(rows.a[q]), 'm_hat': float(rows.m_hat[rw]), 'observed_mask': 0, 'r': float(rows.r[rw]),
                          'R_mask': float(rows.var[rw]), 'J_x': rows.g[q].tolist(), 'contribution_JtRinv_r': (rows.g[q]*rows.r[rw]/rows.var[rw]).tolist(),
                          'shared_with_points': [int(o) for o in others.tolist() if int(o) != pid][:16]})
        out['points'][pid] = {'x': X[pid].tolist(), 'x_minus_prior': (X[pid]-x_minus[pid]).tolist(), 'delta_x': delta[pid].tolist(), 'alpha': float(alpha[pid]),
                              'active': bool(act[pid]), 'component': int(comp[pid]),
                              'component_objective_change': float(phi1[comp[pid]]), 'rgbd_rows': [_row_dict(r, i) for r in rows_rgbd for i in range(len(r.point))],
                              'mask_rows': pairs}
    return out


def _trace_final(X, s_minus, s_plus, P_minus, P_plus, rows, final, keep, status, rgbd_consistent, J_rgbd, thr, C):
    out = {}
    for pid in keep.tolist():
        out[pid] = {'s_minus': s_minus[pid].tolist(), 'P_minus': P_minus[pid].tolist(), 's_plus': s_plus[pid].tolist(), 'P_plus': P_plus[pid].tolist(),
                    'P_plus_eigenvalues': torch.linalg.eigvalsh(P_plus[pid]).tolist(), 'mask_status': MASK_STATUS[int(status[pid])],
                    'direct_rgbd_rows': int(final['nrows'][pid]), 'rgbd_posterior_cost': float(J_rgbd[pid]),
                    'rgbd_threshold': float(thr[min(int(final['nrows'][pid]), C)]), 'rgbd_consistent': bool(rgbd_consistent[pid]),
                    'mask_rows_final': int((rows.pair_point == pid).sum()) if rows.n_rows else 0,
                    'covariance_model': 'pixel_marginal_adf_approximation (not the exact joint marginal)'}
    return out
