"""Batched exact Rauch-Tung-Striebel smoother for independent 6D [x, v] point states (RGBDiekf_Phase1 spec P1.3, Gate 1).

Given ONE internally consistent forward pass (filtered means / covariances s_{k|k}, P_{k|k} of every point and the
transition F_k, Q_k of every interval), for k = K-2 .. 0:

    P_{k+1|k} = F_k P_{k|k} F_k^T + Q_k                      (recomputed exactly; the predicted history is never stored)
    G_k       = P_{k|k} F_k^T P_{k+1|k}^-1                    (Cholesky solve, no explicit inverse)
    s_{k|T}   = s_{k|k} + G_k (s_{k+1|T} - F_k s_{k|k})
    P_{k|T}   = P_{k|k} + G_k (P_{k+1|T} - P_{k+1|k}) G_k^T

Cross-time covariances follow from the backward Markov structure s_k = G_k s_{k+1} + w_k (w_k independent of s_{k+1:T}
given all data): Cov(s_k, s_j | T) = G_k Cov(s_{k+1}, s_j | T) for k < j. Hence the lag-one block
P_{k,k+1|T} = G_k P_{k+1|T} and the frame-0 / frame-t block Cov(s_0, s_t | T) = G_0 G_1 ... G_{t-1} P_{t|T}, which gives
the displacement covariance Cov(d_t) = P_xx,t + P_xx,0 - C_t - C_t^T with C_t = Cov(x_t, x_0) (spec section 1: never the
sum of marginals).

Q_a = 0 (deterministic dynamics) needs no special case here: P_{k+1|k} = F P F^T is SPD whenever P_{k|k} is, and no Q^-1
is ever formed. A non-SPD predicted covariance raises (no silent jitter).

Also provides a batched linear Kalman filter (information-free covariance form) for the Gate-1 oracles and the
identification E-step (estimator/temporal_model_identification.py).
"""
import math
import torch


class SmootherFailure(RuntimeError):
    pass


def transition(dt, dtype=torch.float64, device='cpu'):
    F = torch.eye(6, dtype=dtype, device=device); F[:3, 3:] = dt*torch.eye(3, dtype=dtype, device=device)
    return F


def process_noise(dt, qa):
    """Exact white-acceleration discretization for spectral density qa [3,3] (m^2/s^3), signed dt (B1 runs backward)."""
    qa = torch.as_tensor(qa)
    tau, sg = abs(dt), math.copysign(1., dt)
    if not (math.isfinite(dt) and tau > 0): raise ValueError('process noise needs a non-zero finite physical interval')
    return torch.cat([torch.cat([tau**3/3*qa, sg*tau**2/2*qa], 1), torch.cat([sg*tau**2/2*qa, tau*qa], 1)], 0)


def transitions(times, qa, dtype=torch.float64, device='cpu'):
    """F [K-1,6,6], Q [K-1,6,6] for the actual (possibly irregular, signed) intervals of `times` (processing order)."""
    qa = torch.as_tensor(qa, dtype=dtype, device=device)
    F, Q = [], []
    for a, b in zip(times[:-1], times[1:]):
        dt = float(b-a); F.append(transition(dt, dtype, device)); Q.append(process_noise(dt, qa))
    return torch.stack(F), torch.stack(Q)


def _chol(A, what):
    L, info = torch.linalg.cholesky_ex((A+A.transpose(-1, -2))/2)
    if bool((info != 0).any()):
        raise SmootherFailure(f'{what}: not SPD for {int((info != 0).sum())} points (first {torch.nonzero(info).flatten()[:10].tolist()})')
    return L


def kalman_filter_linear(m0, P0, F, Q, H, R, y, present):
    """Linear-Gaussian batched Kalman filter (oracle / E-step). m0 [n,6], P0 [n,6,6]; F,Q [K-1,6,6];
    H [K,n,m,6], R [K,n,m,m], y [K,n,m]; present [K,n] bool (False: no measurement at that frame: prediction only,
    s_plus = s_minus, P_plus = P_minus exactly). Returns filtered means/covs [K,n,6], [K,n,6,6] and the predictive
    log-likelihood of the measurements per point [n] (prediction-error decomposition)."""
    K = H.shape[0]
    mf, Pf = [], []
    m, P = m0.clone(), P0.clone()
    ll = torch.zeros(len(m0), dtype=m0.dtype, device=m0.device)
    for k in range(K):
        if k > 0:
            m = m@F[k-1].T; P = F[k-1]@P@F[k-1].T+Q[k-1]; P = (P+P.transpose(-1, -2))/2
        pr = present[k]
        if bool(pr.any()):
            Hk, Rk, yk = H[k][pr], R[k][pr], y[k][pr]
            S = Hk@P[pr]@Hk.transpose(-1, -2)+Rk
            L = _chol(S, f'innovation covariance frame {k}')
            nu = yk-(Hk@m[pr][..., None])[..., 0]
            Kt = torch.cholesky_solve(Hk@P[pr], L)                         # S^-1 H P = K^T
            Kg = Kt.transpose(-1, -2)
            mm = m[pr]+(Kg@nu[..., None])[..., 0]
            IKH = torch.eye(6, dtype=m.dtype, device=m.device)-Kg@Hk
            PP = IKH@P[pr]@IKH.transpose(-1, -2)+Kg@Rk@Kg.transpose(-1, -2)    # Joseph form
            z = torch.linalg.solve_triangular(L, nu[..., None], upper=False)[..., 0]
            ll[pr] += -.5*((z*z).sum(-1)+2*torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)+nu.shape[-1]*math.log(2*math.pi))
            m = m.clone(); P = P.clone(); m[pr] = mm; P[pr] = (PP+PP.transpose(-1, -2))/2
        mf.append(m); Pf.append(P)
    return torch.stack(mf), torch.stack(Pf), ll


def rts_smooth(mf, Pf, F, Q, want_lag1=True, want_cross0=True):
    """mf [K,n,6], Pf [K,n,6,6] filtered (processing order); F, Q [K-1,6,6] (or [K-1,n,6,6]).
    Returns dict smooth_mean [K,n,6], smooth_cov [K,n,6,6], lag1 [K-1,n,6,6] = Cov(s_{k+1}, s_k | T), gains G [K-1,n,6,6],
    cross0 [K,n,6,6] = Cov(s_0, s_t | T) (cross0[0] = P_{0|T})."""
    K = mf.shape[0]
    ms, Ps = mf.clone(), Pf.clone()
    G = torch.zeros((max(K-1, 0), *Pf.shape[1:]), dtype=Pf.dtype, device=Pf.device)
    lag1 = torch.zeros_like(G) if want_lag1 else None
    for k in range(K-2, -1, -1):
        Fk, Qk = F[k], Q[k]
        Pp = Fk@Pf[k]@Fk.transpose(-1, -2)+Qk
        L = _chol(Pp, f'predicted covariance P_{{{k+1}|{k}}}')
        Gk = torch.cholesky_solve(Fk@Pf[k], L).transpose(-1, -2)          # P_{k+1|k}^-1 F P_{k|k} = G^T
        mp = mf[k]@Fk.transpose(-1, -2) if Fk.dim() == 2 else (Fk@mf[k][..., None])[..., 0]
        ms[k] = mf[k]+(Gk@(ms[k+1]-mp)[..., None])[..., 0]
        Pk = Pf[k]+Gk@(Ps[k+1]-Pp)@Gk.transpose(-1, -2)
        Ps[k] = (Pk+Pk.transpose(-1, -2))/2
        G[k] = Gk
        if want_lag1: lag1[k] = Ps[k+1]@Gk.transpose(-1, -2)               # Cov(s_{k+1}, s_k) = P_{k+1|T} G_k^T
    out = {'smooth_mean': ms, 'smooth_cov': Ps, 'lag1': lag1, 'gains': G}
    if want_cross0: out['cross0'] = cross_from_gains(G, Ps)
    return out


def cross_from_gains(G, Ps):
    """Cov(s_0, s_t | T) = G_0 ... G_{t-1} P_{t|T} for every t (G [K-1,n,6,6], Ps [K,n,6,6])."""
    K = Ps.shape[0]
    C = torch.empty_like(Ps); C[0] = Ps[0]
    M = torch.eye(6, dtype=Ps.dtype, device=Ps.device).expand_as(Ps[0]).clone()
    for t in range(1, K):
        M = M@G[t-1]; C[t] = M@Ps[t]
    return C


def displacement_cov(Ps, cross0):
    """Cov(d_t) = P_xx,t + P_xx,0 - Cov(x_0, x_t) - Cov(x_t, x_0), exactly (spec section 1)."""
    C = cross0[..., :3, :3]
    D = Ps[..., :3, :3]+Ps[0, ..., :3, :3][None]-C-C.transpose(-1, -2)
    return (D+D.transpose(-1, -2))/2
