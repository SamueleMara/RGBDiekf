"""K = 1 joint pipeline, shared reference uncertainty integrated ONCE (spec section 3.3).

    log L_i(T) = log int N6(u; mu_i, Sigma_i) exp[a_i f_i(u; T)] du,   f_i(u; T) = sum_{t=1..T} log ell_t(R_t r + t_t, a),  u = (r, a)
The SAME latent u enters every frame (a per-frame convolution multiplied afterwards would count the reference uncertainty T times).
Production approximation (declared, error-tested): second-order expansion of a f at the PRIOR MEAN mu_i (Laplace at the prior mean),
whose Gaussian integral is exact for quadratic f:
    log L ~ a f(mu) + 1/2 a^2 (L^T g)^T M^-1 (L^T g) - 1/2 log det M,   M = I - a L^T H L,   Sigma = L L^T,
    g = sum_t J_t^T g_t, H = sum_t J_t^T H_t J_t, J_t = blkdiag(R_t, I3)     (g_t, H_t: analytic derivatives of log ell_t)
valid iff lambda_min(M) >= LAPLACE_MIN_EIG (checked by Cholesky per point). Points where it is not use common-random-number scrambled-Sobol QMC
    log L = logmeanexp_s a f(mu + L z_s),   z_s = Phi^-1(sobol_s), fixed seed (same nodes for every hypothesis: differentiable, low variance).
`qmc_log_evidence` with many nodes is the independent oracle. Everything is differentiable w.r.t. the pose tensors.
"""
import math
import torch
from estimator.svgd_k1_observation import log_potential

_SOBOL = {}


def sobol_normal(S, seed, dtype, device):
    key = (S, seed, str(device))
    if key not in _SOBOL:
        eng = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed); u = eng.draw(S, dtype=torch.float64).clamp(1e-12, 1-1e-12)
        _SOBOL[key] = (torch.erfinv(2*u-1)*math.sqrt(2)).to(device, dtype)
    return _SOBOL[key]


def f_g_H(frames, poses, mu, need_hessian=True):
    """Sum over frames t = 1..T of log ell_t at the transformed prior mean, with g [n,6], H [n,6,6] w.r.t. u. frames: list of
    (kde, bg, rho); poses [T+1,4,4] (poses[0] = I is the reference and is not an observation)."""
    f = torch.zeros(len(mu), dtype=mu.dtype, device=mu.device); g = torch.zeros(len(mu), 6, dtype=mu.dtype, device=mu.device)
    H = torch.zeros(len(mu), 6, 6, dtype=mu.dtype, device=mu.device) if need_hessian else None; certs = []
    for t, (kde, bg, rho) in enumerate(frames, start=1):
        R, tr = poses[t, :3, :3], poses[t, :3, 3]
        x = mu[:, :3]@R.T+tr
        ll, gt, Ht, cert = log_potential(kde, bg, rho, x, mu[:, 3:], need_hessian=need_hessian)
        f = f+ll
        J = torch.zeros(6, 6, dtype=mu.dtype, device=mu.device); J[:3, :3] = R; J[3:, 3:] = torch.eye(3, dtype=mu.dtype, device=mu.device)
        g = g+gt@J
        if need_hessian: H = H+J.T@Ht@J
        certs.append(cert)
    return f, g, H, torch.stack(certs, 1) if certs else None


LAPLACE_MIN_EIG = .5     # numerical validity margin (declared): Laplace only if lambda_min(M) >= 0.5, i.e. the integrand Gaussian is at most
                         # sqrt(2) wider than the reference prior along every direction; otherwise the CRN-QMC route (measured: near-singular
                         # M gave corrections up to +590 nats on Blade). Not a physical gate.


def laplace_at_mean(f, g, H, Lchol, a, min_eig=LAPLACE_MIN_EIG):
    """(log L [n], ok [n]) for the second-order expansion; ok = M SPD with lambda_min(M) >= min_eig (Cholesky of M - min_eig I)."""
    eye = torch.eye(6, dtype=f.dtype, device=f.device)
    M = eye-a[:, None, None]*(Lchol.transpose(-1, -2)@H@Lchol); M = (M+M.transpose(-1, -2))/2
    with torch.no_grad(): ok = torch.linalg.cholesky_ex(M.detach()-min_eig*eye)[1] == 0      # validity decided without a graph
    Cs = torch.linalg.cholesky(torch.where(ok[:, None, None], M, eye.expand_as(M)))      # invalid rows factor I: no NaN in the backward
    b = (a[:, None]*(Lchol.transpose(-1, -2)@g[:, :, None])[..., 0])
    y = torch.linalg.solve_triangular(Cs, b[:, :, None], upper=False)[..., 0]
    logdet = 2*torch.log(torch.diagonal(Cs, dim1=-2, dim2=-1)).sum(-1)
    return a*f+.5*(y*y).sum(-1)-.5*logdet, ok


def qmc_log_evidence(frames, poses, mu, Lchol, a, S=128, seed=0):
    """logmeanexp over S common scrambled-Sobol normal nodes of a f(mu + L z) (value-only evaluation of every frame)."""
    z = sobol_normal(S, seed, mu.dtype, mu.device)                                       # [S,6]
    n = len(mu); U = (mu[:, None, :]+torch.einsum('nij,sj->nsi', Lchol, z)).reshape(n*S, 6)
    f = torch.zeros(n*S, dtype=mu.dtype, device=mu.device)
    for t, (kde, bg, rho) in enumerate(frames, start=1):
        R, tr = poses[t, :3, :3], poses[t, :3, 3]
        f = f+log_potential(kde, bg, rho, U[:, :3]@R.T+tr, U[:, 3:], need_hessian=False)[0]
    v = a[:, None]*f.reshape(n, S)
    return torch.logsumexp(v, 1)-math.log(S)


def log_evidence(frames, poses, mu, Lchol, a, qmc_nodes=128, qmc_seed=0):
    """Production per-point evidence: Laplace at the prior mean where valid, CRN-QMC where M is not SPD.
    Returns (log L [n], diagnostics dict)."""
    f, g, H, cert = f_g_H(frames, poses, mu)
    lap, ok = laplace_at_mean(f, g, H, Lchol, a)
    out = lap
    if not bool(ok.all()):
        bad = torch.nonzero(~ok)[:, 0]
        q = qmc_log_evidence(frames, poses, mu[bad], Lchol[bad], a[bad], qmc_nodes, qmc_seed)
        out = lap.clone(); out[bad] = q
    return out, {'n': len(mu), 'qmc_fallback': int((~ok).sum()), 'cert_max': float(cert.max()) if cert is not None and cert.numel() else 0.,
                 'cert_sum_max': float(cert.sum(1).max()) if cert is not None and cert.numel() else 0.}
