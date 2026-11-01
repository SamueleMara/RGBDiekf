"""Batched symmetric positive-definite block-tridiagonal systems with 6x6 blocks (RGBDiekf_Phase1 spec section 4, G1.2).

One system per point over its whole trajectory s_0 .. s_{K-1}: diagonal blocks A_kk [n,K,6,6], upper blocks
B_k = A_{k,k+1} [n,K-1,6,6] (A_{k+1,k} = B_k^T). Block LDL^T (block Cholesky / block Thomas), forward from k = 0:

    D_0 = A_00,   D_{k+1} = A_{k+1,k+1} - B_k^T D_k^-1 B_k                       (every D_k SPD, Cholesky-factored)

Solve A x = r:  y_0 = r_0, y_{k+1} = r_{k+1} - B_k^T D_k^-1 y_k;   x_{K-1} = D_{K-1}^-1 y_{K-1}, x_k = D_k^-1 (y_k - B_k x_{k+1}).
Selected inverse (posterior covariance blocks of the Laplace approximation), backward:
    S_{K-1,K-1} = D_{K-1}^-1,   G_k = -D_k^-1 B_k,   S_{k,k+1} = G_k S_{k+1,k+1},   S_kk = D_k^-1 + G_k S_{k+1,k+1} G_k^T
and every farther block S_{k,j} = G_k S_{k+1,j} (k < j), so S_{0,t} = G_0 ... G_{t-1} S_tt (frame-0 / frame-t queries).
No dense (6K x 6K) inverse is ever formed; tests/test_temporal_phase1_math.py compares every block with one.

Q_a = 0 is NOT a block-tridiagonal precision (Q^-1 does not exist): estimator/ieks_point_trajectory.py then solves the exact
reduced problem in s_0 only (s_k = Phi_k s_0) with reduced_solve below.
"""
import torch


class BlockSolveFailure(RuntimeError):
    pass


def _chol(A, what):
    L, info = torch.linalg.cholesky_ex((A+A.transpose(-1, -2))/2)
    if bool((info != 0).any()):
        raise BlockSolveFailure(f'{what}: pivot block not SPD for {int((info != 0).sum())} systems')
    return L, info


def factor(A, B):
    """A [n,K,6,6], B [n,K-1,6,6] -> dict(L [n,K,6,6] Cholesky of D_k, B)."""
    n, K = A.shape[:2]
    L = torch.empty_like(A)
    D = A[:, 0]
    for k in range(K):
        if k > 0:
            Bk = B[:, k-1]
            X = torch.cholesky_solve(Bk, L[:, k-1])                    # D_{k-1}^-1 B_{k-1}
            D = A[:, k]-Bk.transpose(-1, -2)@X
        L[:, k], _ = _chol(D, f'block {k}')
    return {'L': L, 'B': B}


def solve(fac, r):
    """r [n,K,6] or [n,K,6,m] -> x of the same shape."""
    L, B = fac['L'], fac['B']
    vec = r.dim() == 3
    if vec: r = r[..., None]
    K = L.shape[1]
    y = torch.empty_like(r)
    y[:, 0] = r[:, 0]
    for k in range(1, K):
        y[:, k] = r[:, k]-B[:, k-1].transpose(-1, -2)@torch.cholesky_solve(y[:, k-1], L[:, k-1])
    x = torch.empty_like(r)
    x[:, K-1] = torch.cholesky_solve(y[:, K-1], L[:, K-1])
    for k in range(K-2, -1, -1):
        x[:, k] = torch.cholesky_solve(y[:, k]-B[:, k]@x[:, k+1], L[:, k])
    return x[..., 0] if vec else x


def selected_inverse(fac, want_cross0=True):
    """Diagonal blocks S_kk [n,K,6,6], lag-one blocks S_{k+1,k} [n,K-1,6,6] (= Cov(s_{k+1}, s_k)), gains G_k and
    (optionally) S_{0,t} [n,K,6,6] of the inverse."""
    L, B = fac['L'], fac['B']
    n, K = L.shape[:2]
    I6 = torch.eye(6, dtype=L.dtype, device=L.device).expand(n, 6, 6)
    S = torch.empty_like(L); lag = torch.empty((n, max(K-1, 0), 6, 6), dtype=L.dtype, device=L.device); G = torch.empty_like(lag)
    S[:, K-1] = torch.cholesky_inverse(L[:, K-1])
    for k in range(K-2, -1, -1):
        Gk = -torch.cholesky_solve(B[:, k], L[:, k])
        Dinv = torch.cholesky_inverse(L[:, k])
        Skk = Dinv+Gk@S[:, k+1]@Gk.transpose(-1, -2)
        S[:, k] = (Skk+Skk.transpose(-1, -2))/2
        lag[:, k] = (Gk@S[:, k+1]).transpose(-1, -2)                    # S_{k+1,k} = (S_{k,k+1})^T
        G[:, k] = Gk
    out = {'diag': S, 'lag1': lag, 'gains': G}
    if want_cross0:
        C = torch.empty_like(S); C[:, 0] = S[:, 0]; M = I6.clone()
        for t in range(1, K):
            M = M@G[:, t-1]; C[:, t] = M@S[:, t]
        out['cross0'] = C
    del I6
    return out


def reduced_solve(Phi, A0, g):
    """Deterministic dynamics (Q_a = 0): s_k = Phi_k s_0. A0 [n,6,6] = P0^-1 + sum_k Phi_k^T A_obs,k Phi_k, g [n,6] the
    reduced right-hand side. Returns (delta_0, covariance of s_0)."""
    L, _ = _chol(A0, 'reduced (Q_a = 0) information')
    return torch.cholesky_solve(g[..., None], L)[..., 0], torch.cholesky_inverse(L)
