"""DIAGNOSTIC-ONLY velocity information of the material-following observation (never updates any state).

Rows are the material-following linearization at a given iterate (H_t[:, :3] = [G_I ; -a], residual f). Re-expressed
in a PREVIOUS-TIME velocity, X_t = X_{t-1} + dt v_{t-1}, the same measurement has G^v = dt H (spec 3.2): this is a
coordinate change of one measurement, not an extra pseudo-measurement. Per point (cameras have distinct raw pixels):
    Lambda_v = sum_c dt^2 H_c^T R_c^-1 H_c ,   b_v = sum_c dt H_c^T R_c^-1 (-f_c)
    R_c = sum_a w_a^2 R_cur,a + R_ref(p_ref) + R_resampling + R_warp     (marginal row noise; reference STATE uncertainty
                                                                           is a state and enters only through the IEKF)
A direction is reported MOTION_UNIDENTIFIABLE when its data information does not exceed the declared prior velocity
precision of the same point (eigenvalue of Lambda_v <= eigenvalue scale of P_vv^-1): the declared prior, not a new
threshold. The velocity guidance compared with the actual 6D update is dv = (Lambda_v + P_vv^-1)^-1 b_v.
"""
import torch
from estimator.temporal_rgbd_terms import sensor_covariance
from estimator.temporal_image_gradients import gradients_at
from estimator.temporal_motion_geometry import depth_basis, normal_alignment
from estimator.temporal_local_covariance import batched_eigvalsh

STAGES = ('projected_in_image', 'foreground_target', 'first_depth_winner_or_same_surface', 'has_usable_temporal_reference',
          'material_support_valid', 'admitted_row', 'accepted_update')


def row_noise(lr, noise, warp):
    ref = sensor_covariance(lr.rows.reference_values, noise).diagonal(dim1=-2, dim2=-1)
    return torch.diag_embed(lr.r_eff+lr.rows.resampling_variance+ref)+warp


def velocity_guidance(lr, noise, warp, dt, prior6, frame, rig, policy):
    rows = lr.rows
    N, dev = len(prior6), prior6.device
    H = rows.H_t[:, :, :3]
    M = len(rows)
    out = {'Lambda_v': torch.zeros((N, 3, 3), dtype=torch.float64, device=dev), 'b_v': torch.zeros((N, 3), dtype=torch.float64, device=dev)}
    if M:
        R = row_noise(lr, noise, warp)
        L = torch.linalg.cholesky(R)
        info = H.transpose(1, 2)@torch.cholesky_solve(H, L)*dt*dt
        b = (H.transpose(1, 2)@torch.cholesky_solve(-rows.f[..., None], L))[..., 0]*dt
        out['Lambda_v'].index_add_(0, rows.point, info)
        out['b_v'].index_add_(0, rows.point, b)
        out['row_info'], out['row_b'] = info, b
        g = gradients_at(frame, rows.camera, rows.q, noise, policy, rig)
        B, okB = depth_basis(rig, rows.camera, rows.q, g.depth, g.grad_depth, g.valid)
        out['row_basis'], out['row_basis_valid'] = B, okB
        out['row_normal_alignment'] = torch.where(okB, normal_alignment(-H[:, 3], B), torch.full((M,), float('nan'), dtype=torch.float64, device=dev))
    Pvv = prior6[:, 3:, 3:]
    Pinv = torch.linalg.inv(Pvv)
    out['dv'] = torch.linalg.solve(out['Lambda_v']+Pinv, out['b_v'][..., None])[..., 0]
    ev = batched_eigvalsh(out['Lambda_v'])
    prior_prec = batched_eigvalsh(Pinv)
    out['eig'] = ev
    out['identifiable_directions'] = (ev > prior_prec[:, :1]).sum(1)     # data information above the declared prior
    return out


def stage_matrix(lr, assimilated, frame, refs, ordinal, lag):
    """Per camera, count of (camera, point) pairs surviving each successive stage (ownership at the final iterate)."""
    own = lr.owners
    C, N = own.reason.shape
    dev = own.reason.device
    inside = (own.reason != 1) & (own.reason != 2)
    pix = own.pixel.clamp(0, frame.H*frame.W-1)
    c = torch.arange(C, device=dev)[:, None].expand(-1, N)
    fg = inside & frame.mask[c, pix//frame.W, pix % frame.W] & frame.raw_valid[c, pix//frame.W, pix % frame.W]
    code = lr.reason
    visible = fg & (code != 6) & (code != 8)
    # readiness as it was DURING this frame (the table is already committed when diagnostics run): reason codes
    # 11 BOOTSTRAP_PENDING / 12 REFERENCE_OUTSIDE_LAG mark identities without a usable reference
    ready = (code != 11) & (code != 12)
    support = (code == 0)
    admitted = torch.zeros((C, N), dtype=torch.bool, device=dev)
    admitted[lr.rows.camera, lr.rows.point] = True
    accepted = admitted & assimilated[None]
    chain = [inside, fg, visible, visible & ready, visible & ready & support, admitted, accepted]
    per_camera = [{s: int(m[cc].sum()) for s, m in zip(STAGES, chain)} for cc in range(C)]
    return per_camera, {s: int(m.sum()) for s, m in zip(STAGES, chain)}, chain


def summarize(g, update, step, dt):
    """Frame-level numbers for the run summary; arrays stay on the update for traced points."""
    s, mean = step['s'], step['mean']
    obs = torch.bincount(step['lr'].rows.point, minlength=len(s)) > 0
    actual_dv = (s[:, 3:]-mean[:, 3:])
    pred = g['dv']
    cosang = (actual_dv*pred).sum(1)/(actual_dv.norm(dim=1)*pred.norm(dim=1)).clamp_min(1e-300)
    q = lambda x: torch.quantile(x.double(), torch.tensor([.5, .9, .99], dtype=torch.float64, device=x.device)).tolist() if len(x) else None
    ident = g['identifiable_directions'][obs]
    rec = {'semantics': 'diagnostic velocity information of the SAME material-following measurement (G^v = dt H); not assimilated',
           'observed_points': int(obs.sum()),
           'identifiable_velocity_directions_hist': {str(i): int((ident == i).sum()) for i in range(4)},
           'velocity_guidance_norm_m_s_quantiles': q(pred[obs].norm(dim=1)),
           'actual_velocity_update_norm_m_s_quantiles': q(actual_dv[obs].norm(dim=1)),
           'guidance_vs_actual_update_cos_quantiles': q(cosang[obs & (actual_dv.norm(dim=1) > 0)]),
           'data_velocity_std_m_s_weak_mid_strong_median': [float(torch.median(g['eig'][obs, j].clamp_min(1e-300).rsqrt())) for j in range(3)] if bool(obs.any()) else None}
    if 'row_normal_alignment' in g:
        na = g['row_normal_alignment']
        rec['depth_row_vs_image_normal_abs_cos_quantiles'] = q(na[torch.isfinite(na)])
    return rec


def photometric_consistency(lr, noise, warp, alpha):
    """RGB part of the profiled material residual e0 at an iterate, normalized by its declared row noise (3 dof). Under
    brightness constancy the exceedance rate of chi2_3(1 - alpha) is alpha; a binomial upper-tail p-value below the same
    DECLARED alpha (configs/image_joint_policy.json tail_probability) reports PHOTOMETRIC_MODEL_INVALID. Diagnostic only:
    it never changes rows, weights or state."""
    from scipy.stats import chi2, binom
    if not len(lr):
        return {'rows': 0, 'status': 'NO_ROWS'}
    R = row_noise(lr, noise, warp)[:, :3, :3]
    e = lr.e0[:, :3]
    nis = (e*torch.linalg.solve(R, e[..., None])[..., 0]).sum(1)
    thr = float(chi2.isf(alpha, 3))
    n, k = len(nis), int((nis > thr).sum())
    p = float(binom.sf(k-1, n, alpha))
    return {'rows': n, 'exceed_fraction': k/n, 'expected_fraction': alpha, 'binomial_p_value': p,
            'nis_median': float(nis.median()), 'status': 'PHOTOMETRIC_MODEL_INVALID' if p < alpha else 'PHOTOMETRIC_CONSISTENT'}
