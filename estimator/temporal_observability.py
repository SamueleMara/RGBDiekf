"""Observability diagnostics of the Phase-1 temporal solver (RGBDiekf_Phase1 spec P1.5, G2.4, G2.5, G4.3).

Data-only information is reported separately from the regularized posterior:
  * per frame: eigenvalues of the 3x3 data information sum_c H_x^T R^-1 H_x of the accepted rows (rank 0..3);
  * geometry correction b = x_0 - x_ref: generalized eigenvalues lambda_j of the prior covariance P_x0 against the posterior
    marginal covariance of x_0 (P_x0 u = lambda Sigma_x0 u). lambda_j - 1 is the information the DATA (all frames, through
    the dynamics) added along u_j relative to the prior's own information. A direction is reported IDENTIFIED only when
    lambda_j - 1 >= 1 (the data carry at least as much information as the prior there); otherwise the posterior is
    prior-dominated in that direction and the correction is marked not identifiable (it is kept, never zeroed).
  * structured camera parameters (G4.3): joint Jacobian rank of per-point geometry plus per-camera depth offsets, and
    the gauge of a free per-point-per-camera bias.
"""
import numpy as np
import torch


def frame_information_eigs(Ixx):
    """Ixx [..., 3, 3] -> ascending eigenvalues [..., 3] (host float64)."""
    return torch.linalg.eigvalsh(((Ixx+Ixx.transpose(-1, -2))/2).double().cpu())


def geometry_identifiability(P_x0, Sigma_x0):
    """P_x0, Sigma_x0 [n,3,3] -> lambda [n,3] ascending (prior / posterior variance ratios along the generalized
    eigenvectors), identified [n,3] bool (lambda - 1 >= 1), count of non-identified directions [n]."""
    P, S = P_x0.double().cpu(), Sigma_x0.double().cpu()
    L = torch.linalg.cholesky((S+S.transpose(-1, -2))/2)
    Li = torch.linalg.inv(L)
    M = Li@P@Li.transpose(-1, -2)
    lam = torch.linalg.eigvalsh((M+M.transpose(-1, -2))/2)
    ident = (lam-1) >= 1
    return lam, ident, (~ident).sum(-1)


def joint_rank(J, rtol=None):
    """Numerical rank of a dense Jacobian with the standard tolerance max(m, n) eps sigma_max (or rtol)."""
    s = np.linalg.svd(np.asarray(J, float), compute_uv=False)
    tol = (rtol if rtol is not None else max(J.shape)*np.finfo(float).eps)*s[0]
    return int((s > tol).sum()), s


def camera_offset_system(H_points, cams, n_cams, free_per_point_camera=False):
    """Linear(ized) depth-row system for G4.3. H_points: list over rows of (point index, 1x3 depth Jacobian H_D) ;
    cams: camera of each row. Unknowns: 3 per point + one depth offset per camera (h = D(pi(x)) + beta_c - Z), or one
    offset per (point, camera) pair when free_per_point_camera. Returns the dense Jacobian and the unknown layout."""
    pts = sorted({p for p, _ in H_points})
    pidx = {p: i for i, p in enumerate(pts)}
    pairs = sorted({(p, c) for (p, _), c in zip(H_points, cams)})
    nb = len(pairs) if free_per_point_camera else n_cams
    J = np.zeros((len(H_points), 3*len(pts)+nb))
    for r, ((p, H), c) in enumerate(zip(H_points, cams)):
        J[r, 3*pidx[p]:3*pidx[p]+3] = H
        J[r, 3*len(pts)+(pairs.index((p, c)) if free_per_point_camera else c)] = 1.
    return J, {'points': pts, 'bias_unknowns': nb, 'free_per_point_camera': free_per_point_camera}
