"""World-frame velocity rows of the RGB-D temporal physics and a DIAGNOSTIC local tangent/normal basis.

For a fixed calibrated camera c, world point X with u = pi_c(X) and optical depth Z_c(X):
    RGB constancy of the same material:  dI/dt + grad_u I J_pi v = 0          ->  G_I v = -dI/dt,  G_I = grad_u I J_pi (3x3)
    depth identity D(pi(X), t) = Z(X):   dD/dt + (grad_u D J_pi - J_Z) v = 0  ->  a v = dD/dt,     a = J_Z - grad_u D J_pi
(J_pi, J_Z from CameraRig.project_rows, world frame). The material-following Jacobian H_t[:, :3] is [G_I ; -a]: the
same linearization, never an additional observation.

The basis B = [t1, t2, n] is derived from the CURRENT depth image only (X(u) = D(u) K^-1 [u,v,1], n ~ X_u x X_v, rotated
to world); it is a diagnostic coordinate system, never stored in any state. Where the depth sample is not a valid
single-surface support no basis exists (NaN) -- no tangent basis is manufactured.
"""
import torch


def velocity_rows(rig, cam, X, grad_rgb, grad_depth):
    """G_I (M,3,3), a (M,3), J_pi (M,2,3), J_Z (M,3) at world points X observed by cameras `cam`."""
    _, _, Jpi, Jz = rig.project_rows(X, cam)
    G = grad_rgb@Jpi
    a = Jz-(grad_depth[:, None, :]@Jpi)[:, 0]
    return G, a, Jpi, Jz


def depth_basis(rig, cam, uv, depth, grad_depth, valid):
    """World unit normal n of the measured depth surface at uv and an orthonormal tangent pair (diagnostic only)."""
    K = rig.K[cam]
    fx, fy, cx, cy = K.unbind(-1)
    r = torch.stack(((uv[:, 0]-cx)/fx, (uv[:, 1]-cy)/fy, torch.ones_like(fx)), -1)
    Xu = grad_depth[:, :1]*r
    Xu[:, 0] += depth/fx
    Xv = grad_depth[:, 1:]*r
    Xv[:, 1] += depth/fy
    n_cam = torch.cross(Xu, Xv, dim=1)
    n = torch.einsum('mji,mj->mi', rig.R[cam], n_cam)               # R_wc = R_cw^T
    norm = n.norm(dim=1, keepdim=True)
    ok = valid & (norm[:, 0] > 0) & torch.isfinite(norm[:, 0])
    n = n/norm.clamp_min(torch.finfo(n.dtype).tiny)
    e = torch.eye(3, dtype=n.dtype, device=n.device)[n.abs().argmin(1)]
    t1 = e-(e*n).sum(1, keepdim=True)*n
    t1 = t1/t1.norm(dim=1, keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)
    t2 = torch.cross(n, t1, dim=1)
    B = torch.stack((t1, t2, n), -1)                                 # columns [t1, t2, n]
    B[~ok] = float('nan')
    return B, ok


def basis_components(G, a, B):
    """G_I B (M,3,3) and a B (M,3); for a smooth surface a B ~ [0, 0, +-|a|] (reported, never forced)."""
    return G@B, (a[:, None, :]@B)[:, 0]


def normal_alignment(a, B):
    """|cos| between the depth row a and the image-derived normal (sign of the equation retained elsewhere)."""
    n = B[:, :, 2]
    return ((a*n).sum(1).abs()/a.norm(dim=1).clamp_min(torch.finfo(a.dtype).tiny))


def row_information(H, R, r):
    """Per-row 3x3 information H^T R^-1 H and signed guidance H^T R^-1 (-r) for rows H (M,4,3), R (M,4,4), residual r."""
    L = torch.linalg.cholesky(R)
    RH = torch.cholesky_solve(H, L)
    Rr = torch.cholesky_solve(-r[..., None], L)[..., 0]
    return H.transpose(1, 2)@RH, (H.transpose(1, 2)@Rr[..., None])[..., 0]
