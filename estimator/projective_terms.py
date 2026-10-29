"""Direct projective RGB-D measurement terms (contract section 2): zero-target implicit observation, its innovation,
analytic Jacobian and sensor covariance, row by row. No image is rendered and no point is matched.

For a point x (world) seen by camera c at frame k, with fixed canonical colour a and projection (u, v, Z):

    h(x) = [ I_k,c(u,v) - a ;  D_k,c(u,v) - Z ]  in R^4,   target z = 0,   r = z - h = [ a - I ; Z - D ]
    dh/dx = [ dI/d(u,v) J_u ;  dD/d(u,v) J_u - J_Z ]                       (H_x, 4 x 3; velocity columns are zero)
    R = diag(var_rgb (3), var_depth)                                        (projective_sensor variances)

Row admission (configs/projective_rgbd.json visibility_n_sigma and row_innovation_gate). With Sigma_x the PRIOR
covariance of x and dx = x_j - x- the iterate's offset from the prior mean, the innovation of the row linearized back to
the prior (the quantity a Kalman gate tests; identical to r at the prior iterate dx = 0) and its predicted covariance are
    nu = r + H_x dx,      S = H_x Sigma_x H_x^T + R          (4 x 4, RGB and depth jointly, cross terms kept)
    MEASURED_SURFACE_IN_FRONT  nu_D > n_sigma sqrt(S_DD): the measured surface lies in front of the prediction (external
                               occlusion OR misprediction; reported, not used, not deleted)
    INNOVATION_INCONSISTENT    otherwise nu^T S^-1 nu > chi2_4(1 - alpha_row), alpha_row = 2 (1 - Phi(n_sigma)), the
                               two-sided tail of the same n-sigma rule: the sampled pixel is not the transported point's
                               surface within the prior uncertainty (a surface far behind the point, a different colour),
                               which the 2x2 interpolant-validity test cannot tell (it only checks one surface per cell).
row_innovation_gate selects the rule: 'rgbd_chi2' (above, default), 'depth_two_sided' (|nu_D| > n_sigma sqrt(S_DD) only,
sign -> reason) or 'depth_front_only' (the earlier rule exactly: only r_D > n_sigma sqrt(S_DD) on the ITERATE's own
residual, not the prior-linearized nu_D; the other sign kept as motion).
A row moving within its prior uncertainty passes every rule; the prior covariance (velocity prior and q_c included)
is what decides how far a legitimate motion may go in one frame.
"""
from dataclasses import dataclass
import torch
from scipy.stats import chi2, norm
from estimator.projective_sensor import sample, CODE

GATES = ('rgbd_chi2', 'depth_two_sided', 'depth_front_only')


def row_gate_threshold(policy, dof=4):
    """chi2_dof(1 - alpha_row) with alpha_row = 2 (1 - Phi(visibility_n_sigma)): one declared n-sigma, no new number."""
    return float(chi2.isf(2*norm.sf(policy['visibility_n_sigma']), dof))


@dataclass
class Terms:
    cam: torch.Tensor      # [M]
    point: torch.Tensor    # [M] row -> point index (caller's indexing)
    uv: torch.Tensor       # [M,2]
    z: torch.Tensor        # [M]
    r: torch.Tensor        # [M,4] innovation r = -h
    Hx: torch.Tensor       # [M,4,3] dh/dx
    var: torch.Tensor      # [M,4] diagonal R
    reason: torch.Tensor   # [M] uint8, 0 = used
    meas_rgb: torch.Tensor  # [M,3]
    meas_depth: torch.Tensor  # [M]
    pixel00: torch.Tensor  # [M,2]
    S_zz: torch.Tensor     # [M]
    nis: torch.Tensor      # [M] nu^T S^-1 nu of the prior-linearized innovation (4 dof)

    def valid(self):
        return self.reason == CODE['VALID']

    def select(self, m):
        return Terms(*[getattr(self, f)[m] for f in self.__dataclass_fields__])


def build_terms(rig, frame, cam, point, x, rgb0, Sigma_x, noise, policy, dtype=torch.float64, dx=None):
    """Rows (cam [M], point [M]) at world positions x [M,3] with canonical colours rgb0 [M,3], prior position
    covariance Sigma_x [M,3,3] and offset dx [M,3] of x from the prior mean (None = 0, x is the prior).
    Returns Terms (invalid rows carry their reason; their r, H are not used)."""
    uv, Z, Ju, Jz = rig.project_rows(x, cam)
    s = sample(frame, cam, uv, noise, policy, rig.K, rig.dtype)          # sampling precision = rig precision (mixed mode)
    if rig.dtype != dtype:
        s = type(s)(*[v.to(dtype) if torch.is_floating_point(v) else v for v in (getattr(s, f) for f in s.__dataclass_fields__)])
    reason = s.reason.clone()
    reason[(Z <= rig.near[cam]) | ~torch.isfinite(Z)] = CODE['BEHIND_CAMERA']
    Ju, Jz, Z = Ju.to(dtype), Jz.to(dtype), Z.to(dtype)
    Hx = torch.empty((len(cam), 4, 3), dtype=dtype, device=x.device)
    Hx[:, :3] = (s.grad_rgb[:, :, :, None]*Ju[:, None, :, :]).sum(2)          # [M,3,2] x [M,2,3]; broadcast sums, not tiny batched GEMMs
    Hx[:, 3] = (s.grad_depth[:, :, None]*Ju).sum(1)-Jz
    r = torch.cat([rgb0.to(dtype)-s.rgb, (Z-s.depth)[:, None]], 1)
    var = torch.cat([s.var_rgb, s.var_depth[:, None]], 1)
    Sx = Sigma_x.to(dtype)
    S = Hx@Sx@Hx.transpose(1, 2)+torch.diag_embed(var)                                  # [M,4,4] predicted innovation covariance
    nu = r if dx is None else r+(Hx@dx.to(dtype)[:, :, None])[..., 0]
    S_zz = S[:, 3, 3]
    gate = policy.get('row_innovation_gate', 'depth_front_only')
    if gate not in GATES: raise ValueError(f'row_innovation_gate {gate!r} not in {GATES}')
    ok = reason == 0
    n_sig = policy['visibility_n_sigma']
    # 'depth_front_only' is the audited rule exactly (29a23a7): the iterate's own depth residual r_D, not nu_D
    front = ok & ((r[:, 3] if gate == 'depth_front_only' else nu[:, 3]) > n_sig*torch.sqrt(S_zz.clamp_min(0)))
    reason[front] = CODE['MEASURED_SURFACE_IN_FRONT']
    L, info = torch.linalg.cholesky_ex(torch.where(ok[:, None, None], S, torch.eye(4, dtype=dtype, device=S.device)))
    nis = (torch.linalg.solve_triangular(L, nu[:, :, None], upper=False)[..., 0]**2).sum(1)
    nis = torch.where(ok & (info == 0), nis, torch.full_like(nis, float('nan')))
    if gate == 'depth_two_sided':
        reason[ok & ~front & (nu[:, 3] < -n_sig*torch.sqrt(S_zz.clamp_min(0)))] = CODE['INNOVATION_INCONSISTENT']
    elif gate == 'rgbd_chi2':
        reason[ok & ~front & ~(nis <= row_gate_threshold(policy))] = CODE['INNOVATION_INCONSISTENT']     # NaN (S not PD) fails too
    return Terms(cam, point, uv, Z, r, Hx, var, reason, s.rgb, s.depth, s.pixel00, S_zz, nis)


def h_value(rig, frame, cam, x, rgb0, noise, policy, dtype=torch.float64):
    """h(x) itself (finite-difference oracle in tests and debug traces): [M,4] and validity."""
    uv, Z, _, _ = rig.project_rows(x, cam)
    s = sample(frame, cam, uv, noise, policy, rig.K, dtype)
    return torch.cat([s.rgb-rgb0.to(dtype), (s.depth-Z.to(dtype))[:, None]], 1), s.reason == 0
