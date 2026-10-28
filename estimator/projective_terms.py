"""Direct projective RGB-D measurement terms (contract section 2): zero-target implicit observation, its innovation,
analytic Jacobian and sensor covariance, row by row. No image is rendered and no point is matched.

For a point x (world) seen by camera c at frame k, with fixed canonical colour a and projection (u, v, Z):

    h(x) = [ I_k,c(u,v) - a ;  D_k,c(u,v) - Z ]  in R^4,   target z = 0,   r = z - h = [ a - I ; Z - D ]
    dh/dx = [ dI/d(u,v) J_u ;  dD/d(u,v) J_u - J_Z ]                       (H_x, 4 x 3; velocity columns are zero)
    R = diag(var_rgb (3), var_depth)                                        (projective_sensor variances)

External-occlusion rule (configs/projective_rgbd.json, visibility_n_sigma): with Sigma_x the PRIOR covariance of x,
the predicted depth-innovation variance is S_zz = H_D Sigma_x H_D^T + R_D. A row whose measured surface lies in front
of the prediction by more than n_sigma sqrt(S_zz), i.e. (Z - D) > n_sigma sqrt(S_zz), is MEASURED_SURFACE_IN_FRONT
(occluded OR mispredicted: it is reported, not used, not deleted). The opposite sign cannot be an occlusion; it is the
motion signal and is kept.
"""
from dataclasses import dataclass
import torch
from estimator.projective_sensor import sample, CODE


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

    def valid(self):
        return self.reason == CODE['VALID']

    def select(self, m):
        return Terms(*[getattr(self, f)[m] for f in self.__dataclass_fields__])


def build_terms(rig, frame, cam, point, x, rgb0, Sigma_x, noise, policy, dtype=torch.float64):
    """Rows (cam [M], point [M]) at world positions x [M,3] with canonical colours rgb0 [M,3] and prior position
    covariance Sigma_x [M,3,3]. Returns Terms (invalid rows carry their reason; their r, H are not used)."""
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
    hd = Hx[:, 3]
    S_zz = torch.einsum('mi,mij,mj->m', hd, Sigma_x.to(dtype), hd)+s.var_depth
    front = (r[:, 3] > policy['visibility_n_sigma']*torch.sqrt(S_zz.clamp_min(0)))
    reason[(reason == 0) & front] = CODE['MEASURED_SURFACE_IN_FRONT']
    return Terms(cam, point, uv, Z, r, Hx, var, reason, s.rgb, s.depth, s.pixel00, S_zz)


def h_value(rig, frame, cam, x, rgb0, noise, policy, dtype=torch.float64):
    """h(x) itself (finite-difference oracle in tests and debug traces): [M,4] and validity."""
    uv, Z, _, _ = rig.project_rows(x, cam)
    s = sample(frame, cam, uv, noise, policy, rig.K, dtype)
    return torch.cat([s.rgb-rgb0.to(dtype), (s.depth-Z.to(dtype))[:, None]], 1), s.reason == 0
