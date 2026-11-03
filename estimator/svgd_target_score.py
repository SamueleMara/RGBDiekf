"""Operator A: exact appearance-guided 3D target density and XYZ score (PDF (8)-(12), spec A1-A2; contract E01-E04).

    p_t(x,c) = sum_j w_j N3(x; y_j, LX_j) N3(c; d_j, LC_j),   w_j = 1/M_t (PDF selection; log_weights = explicit policy)
    log a_j  = log w_j - 1/2 dx^T LX_j^-1 dx - 1/2 log det(2 pi LX_j) - 1/2 dc^T LC_j^-1 dc - 1/2 log det(2 pi LC_j)
    r_j = a_j / sum_m a_m,   g = grad_x log p = sum_j r_j LX_j^-1 (y_j - x),   grad_c log p = sum_j r_j LC_j^-1 (d_j - c)

EVERY component enters: queries x components are streamed in tiles with an online log-sum-exp (running max, rescaled
sums) and the exponents in quadratic-expansion matmul form centred per tile, a mathematically equal rearrangement of the dense sum (TF.5 row 1), so candidate_count = M and the omitted-tail
bound is exactly 0. No kNN, radius or top-k. Reference colour uncertainty (THEORY_CONFLICT TC-0001): query_color_cov =
Sigma_a replaces LC_j by LC_j + Sigma_a (exact Gaussian convolution of the colour factor over the uncertain reference);
None is the PDF's fixed-colour form. Covariances must be SPD: a failed Cholesky raises NUMERICAL_FAILURE, no jitter.
An empty frame returns NO_OBSERVATION with NaN density/score (never a fake zero score); callers keep the prior.
"""
from dataclasses import dataclass
import math
import torch

SCORE_STATUS = {0: 'VALID', 1: 'NO_OBSERVATION'}
LOG2PI = math.log(2*math.pi)


@dataclass
class ScoreResult:
    log_density: torch.Tensor        # [N] log p_t(x_i, c_i)
    grad_xyz: torch.Tensor           # [N,3] 1/m
    grad_rgb: torch.Tensor           # [N,3] diagnostic only
    support_status: torch.Tensor     # [N] int8, SCORE_STATUS
    candidate_count: torch.Tensor    # [N] components summed (= M: exact)
    tail_bound: torch.Tensor         # [N] omitted-mass bound (0: exact)
    effective_components: torch.Tensor   # [N] 1 / sum_j r_j^2
    camera_mass: torch.Tensor | None     # [N,C] sum of r_j per source camera (provenance)


def _precision(L, what):
    L = torch.as_tensor(L)
    try: C = torch.linalg.cholesky(L)
    except RuntimeError as e: raise FloatingPointError(f'NUMERICAL_FAILURE: {what} is not SPD ({e})') from None
    eye = torch.eye(3, dtype=L.dtype, device=L.device).expand_as(L)
    P = torch.cholesky_solve(eye, C)
    return (P+P.transpose(-1, -2))/2, 2*torch.log(torch.diagonal(C, dim1=-2, dim2=-1)).sum(-1)+3*LOG2PI


def _quad(d, P):
    """d [n,m,3], P [3,3] or [m,3,3] -> (d^T P d [n,m], P d [n,m,3])."""
    Pd = d@P if P.dim() == 2 else torch.einsum('mij,nmj->nmi', P, d)
    return (d*Pd).sum(-1), Pd


def _tiles(n, m, workspace_bytes, per_pair=6*8):
    pairs = max(1, int(workspace_bytes//per_pair))
    ni = max(1, min(n, int(math.sqrt(pairs)), 4096)); mj = max(1, min(m, pairs//ni))
    return ni, mj


IU = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])          # upper-triangle entries of a symmetric 3x3


def quad_coefficients(P, mu):
    """Expansion of (x - mu_j)^T P_j (x - mu_j) = phi(x) . a_j + x . b_j + k_j, phi = [x1^2, x2^2, x3^2, 2x1x2, 2x1x3, 2x2x3]:
    returns A [m,9] = [a_j, b_j] and k [m]. Exact algebra; callers centre x and mu on the query tile (cancellation)."""
    Pm = P.expand(len(mu), 3, 3) if P.dim() == 2 else P
    Pmu = torch.einsum('mij,mj->mi', Pm, mu)
    return torch.cat([Pm[:, IU[0], IU[1]], -2*Pmu], 1), (mu*Pmu).sum(1), Pm, Pmu


def phi(x):
    return torch.cat([x*x, 2*x[:, [0]]*x[:, [1]], 2*x[:, [0]]*x[:, [2]], 2*x[:, [1]]*x[:, [2]], x], 1)


def sym(v):
    """[n,6] upper entries -> [n,3,3]."""
    S = torch.empty(len(v), 3, 3, dtype=v.dtype, device=v.device); S[:, IU[0], IU[1]] = v; S[:, IU[1], IU[0]] = v
    return S


def target_score(qx, qc, y, d, lam_x, lam_c, log_weights=None, query_color_cov=None, obs_camera=None, cameras=0,
                 workspace_bytes=512*2**20):
    """Tiled exact evaluation in quadratic-expansion (matmul) form, a verified equivalent rearrangement (OPT-01):
    per (query tile, component tile) the exponents are phi(x') A^T + k with x', mu' centred on the tile's query mean, and
    sum_j e_ij LX_j^-1 (y_j - x) = e (P_j y'_j) - (e P_j) x'. Running max / rescaled sums give the log-sum-exp over ALL M."""
    N, M, dev, dt = len(qx), len(y), qx.device, qx.dtype
    if M == 0:
        nan = torch.full((N,), float('nan'), dtype=dt, device=dev)
        return ScoreResult(nan, torch.full((N, 3), float('nan'), dtype=dt, device=dev), torch.full((N, 3), float('nan'), dtype=dt, device=dev),
                           torch.ones(N, dtype=torch.int8, device=dev), torch.zeros(N, dtype=torch.int64, device=dev), torch.zeros(N, dtype=dt, device=dev),
                           torch.zeros(N, dtype=dt, device=dev), None)
    lam_c = torch.as_tensor(lam_c, dtype=dt, device=dev)
    if query_color_cov is not None: lam_c = lam_c+torch.as_tensor(query_color_cov, dtype=dt, device=dev)
    Px, ldx = _precision(torch.as_tensor(lam_x, dtype=dt, device=dev), 'Lambda_X')
    Pc, ldc = _precision(lam_c, 'Lambda_C')
    lw = torch.full((M,), -math.log(M), dtype=dt, device=dev) if log_weights is None else torch.as_tensor(log_weights, dtype=dt, device=dev)
    const = lw-.5*(ldx if ldx.dim() else ldx.expand(M))-.5*(ldc if ldc.dim() else ldc.expand(M))      # [M]
    ni, mj = _tiles(N, M, workspace_bytes)
    logp = torch.empty(N, dtype=dt, device=dev); gx = torch.empty(N, 3, dtype=dt, device=dev); gc = torch.empty(N, 3, dtype=dt, device=dev)
    neff = torch.empty(N, dtype=dt, device=dev); cmass = torch.zeros(N, cameras, dtype=dt, device=dev) if obs_camera is not None else None
    for i0 in range(0, N, ni):
        x, c = qx[i0:i0+ni], qc[i0:i0+ni]; n = len(x); ox, oc = x.mean(0), c.mean(0); xp, cp = x-ox, c-oc; fx, fc = phi(xp), phi(cp)
        m = torch.full((n,), -float('inf'), dtype=dt, device=dev); s = torch.zeros(n, dtype=dt, device=dev); s2 = torch.zeros_like(s)
        gy = torch.zeros(n, 3, dtype=dt, device=dev); gP = torch.zeros(n, 6, dtype=dt, device=dev)
        cy = torch.zeros(n, 3, dtype=dt, device=dev); cP = torch.zeros(n, 6, dtype=dt, device=dev)
        cm = torch.zeros(n, cameras, dtype=dt, device=dev) if cmass is not None else None
        for j0 in range(0, M, mj):
            sl = slice(j0, j0+mj)
            Ax, kx, Pxm, Pyx = quad_coefficients(Px if Px.dim() == 2 else Px[sl], y[sl]-ox)
            Ac, kc, Pcm, Pyc = quad_coefficients(Pc if Pc.dim() == 2 else Pc[sl], d[sl]-oc)
            l = (const[sl]-.5*(kx+kc))[None]-.5*(fx@Ax.T+fc@Ac.T)
            mn = torch.maximum(m, l.max(1).values); a = torch.exp(m-mn); e = torch.exp(l-mn[:, None])
            s = s*a+e.sum(1); s2 = s2*a*a+(e*e).sum(1)
            gy = gy*a[:, None]+e@Pyx; gP = gP*a[:, None]+e@Pxm[:, IU[0], IU[1]]
            cy = cy*a[:, None]+e@Pyc; cP = cP*a[:, None]+e@Pcm[:, IU[0], IU[1]]
            if cm is not None: cm = cm*a[:, None]; cm.index_add_(1, obs_camera[sl], e)
            m = mn
        logp[i0:i0+n] = m+torch.log(s)
        gx[i0:i0+n] = (gy-torch.einsum('nij,nj->ni', sym(gP), xp))/s[:, None]; gc[i0:i0+n] = (cy-torch.einsum('nij,nj->ni', sym(cP), cp))/s[:, None]
        neff[i0:i0+n] = s*s/s2
        if cm is not None: cmass[i0:i0+n] = cm/s[:, None]
    return ScoreResult(logp, gx, gc, torch.zeros(N, dtype=torch.int8, device=dev), torch.full((N,), M, dtype=torch.int64, device=dev),
                       torch.zeros(N, dtype=dt, device=dev), neff, cmass)


def responsibilities(qx, qc, y, d, lam_x, lam_c, log_weights=None, query_color_cov=None):
    """Dense [n,M] responsibilities r_j(x_i, c_i) (small n only: trace points and oracles)."""
    dt, dev, M = qx.dtype, qx.device, len(y)
    lam_c = torch.as_tensor(lam_c, dtype=dt, device=dev)
    if query_color_cov is not None: lam_c = lam_c+torch.as_tensor(query_color_cov, dtype=dt, device=dev)
    Px, ldx = _precision(torch.as_tensor(lam_x, dtype=dt, device=dev), 'Lambda_X'); Pc, ldc = _precision(lam_c, 'Lambda_C')
    lw = torch.full((M,), -math.log(M), dtype=dt, device=dev) if log_weights is None else torch.as_tensor(log_weights, dtype=dt, device=dev)
    l = lw-.5*ldx-.5*ldc-.5*_quad(y[None]-qx[:, None], Px)[0]-.5*_quad(d[None]-qc[:, None], Pc)[0]
    return torch.softmax(l, 1)
