"""Structured stochastic identification for the Phase-1 temporal model (RGBDiekf_Phase1 spec P1.7, Gate 5). DISABLED in the
pipeline (configs/temporal_phase1.json identification.enabled = false); exercised by tests/test_temporal_phase1_identification.py.

Parameters theta (shared by a population of independent tracks, never per point and never from ground-truth speeds):
    Q_a = q I            white-acceleration spectral density (m^2/s^3); Q_k = Q(dt_k; Q_a) exact, never a free 6x6 per time
    P_v0 = sv2 I         population initial-velocity variance (m^2/s^2); mu_v0 = 0
    R = sr2 I            measurement variance of the linear(ized) rows
Approximate EM with the smoother E-step (Shumway & Stoffer 1982). For each transition e_k = s_{k+1} - F_k s_k:
    E[e_k e_k^T] = e^ e^T + P_{k+1|T} + F P_{k|T} F^T - P_{k+1,k|T} F^T - F P_{k,k+1|T}           (lag-one terms essential)
M-step (closed form, normalized Gaussian likelihoods, log|Q| / log|R| included):
    q   = sum_k tr(Qunit_k^-1 E[e_k e_k^T]) / (6 * #transitions),  Qunit_k = Q(dt_k; I)
    sv2 = sum_tracks tr(E[(v_0 - 0)(v_0 - 0)^T]) / (3 * #tracks)
    sr2 = sum_rows (|y - H s^|^2 + tr(H P H^T)) / (#scalar measurements)
Marginal log-likelihood by the prediction-error decomposition. Identifiability: observed information of the log-likelihood in
log(theta) (central differences); a parameter is IDENTIFIED only when its 95 % interval is narrower than a factor 2 either
way (1.96 se(log theta) < ln 2); otherwise the estimate is returned with status NOT_IDENTIFIED and must not be used as a
confident value (spec G5.5).
"""
import math
import numpy as np
import torch
from estimator.rts_point_smoother import kalman_filter_linear, rts_smooth, process_noise, transition

NAMES = ('q', 'sv2', 'sr2')


def _system(times, theta, dtype=torch.float64):
    q, _, _ = theta
    F, Q, Qu = [], [], []
    for a, b in zip(times[:-1], times[1:]):
        dt = float(b-a)
        F.append(transition(dt, dtype)); Qu.append(process_noise(dt, torch.eye(3, dtype=dtype))); Q.append(q*Qu[-1])
    return torch.stack(F), torch.stack(Q), torch.stack(Qu)


def _prior(tracks, theta):
    n = tracks['m0x'].shape[0]
    m0 = torch.zeros((n, 6), dtype=torch.float64); m0[:, :3] = tracks['m0x']
    P0 = torch.zeros((n, 6, 6), dtype=torch.float64); P0[:, :3, :3] = tracks['P0x']; P0[:, 3:, 3:] = theta[1]*torch.eye(3, dtype=torch.float64)
    return m0, P0


def loglik(tracks, theta):
    """Marginal log-likelihood of every track's measurements under theta (sum over tracks)."""
    F, Q, _ = _system(tracks['times'], theta)
    m0, P0 = _prior(tracks, theta)
    K, n, m = tracks['y'].shape
    R = theta[2]*torch.eye(m, dtype=torch.float64).expand(K, n, m, m)
    _, _, ll = kalman_filter_linear(m0, P0, F, Q, tracks['H'], R, tracks['y'], tracks['present'])
    return float(ll.sum())


def e_step(tracks, theta):
    F, Q, Qu = _system(tracks['times'], theta)
    m0, P0 = _prior(tracks, theta)
    K, n, m = tracks['y'].shape
    R = theta[2]*torch.eye(m, dtype=torch.float64).expand(K, n, m, m)
    mf, Pf, ll = kalman_filter_linear(m0, P0, F, Q, tracks['H'], R, tracks['y'], tracks['present'])
    sm = rts_smooth(mf, Pf, F, Q, want_lag1=True, want_cross0=False)
    return sm, F, Qu, float(ll.sum())


def transition_second_moments(ms, Ps, lag1, F):
    """E[e_k e_k^T] for every transition and track: ms [K,n,6], Ps [K,n,6,6], lag1 [K-1,n,6,6] = Cov(s_{k+1}, s_k)."""
    e = ms[1:]-(F[:, None]@ms[:-1][..., None])[..., 0]
    Ft = F.transpose(1, 2)[:, None]
    return (e[..., None]*e[..., None, :]+Ps[1:]+F[:, None]@Ps[:-1]@Ft-lag1@Ft-F[:, None]@lag1.transpose(-1, -2))


def m_step(tracks, sm, F, Qu):
    ms, Ps, lag1 = sm['smooth_mean'], sm['smooth_cov'], sm['lag1']
    E = transition_second_moments(ms, Ps, lag1, F)
    Qi = torch.linalg.inv(Qu)[:, None]
    q = float(torch.diagonal(Qi@E, dim1=-2, dim2=-1).sum()/(6*E.shape[0]*E.shape[1]))
    v0 = ms[0, :, 3:]
    sv2 = float(((v0*v0).sum(-1)+torch.diagonal(Ps[0, :, 3:, 3:], dim1=-2, dim2=-1).sum(-1)).sum()/(3*v0.shape[0]))
    H, y, pr = tracks['H'], tracks['y'], tracks['present']
    res = y-(H@ms[..., None])[..., 0]
    tr = torch.diagonal(H@Ps@H.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1)
    num = ((res*res).sum(-1)+tr)[pr].sum()
    sr2 = float(num/(pr.sum()*y.shape[-1]))
    return (q, sv2, sr2)


def em(tracks, theta0, iters=200, rtol=1e-9):
    """EM from theta0; returns dict(theta, loglik history (non-decreasing up to round-off), iterations)."""
    theta, hist = tuple(float(t) for t in theta0), []
    for it in range(iters):
        sm, F, Qu, ll = e_step(tracks, theta)
        hist.append(ll)
        new = m_step(tracks, sm, F, Qu)
        if max(abs(math.log(a/b)) for a, b in zip(new, theta)) < rtol: theta = new; break
        theta = new
    hist.append(loglik(tracks, theta))
    return {'theta': dict(zip(NAMES, theta)), 'loglik_history': hist, 'iterations': it+1}


def identifiability(tracks, theta, h=1e-3):
    """Observed information of the log-likelihood in log(theta) by central differences -> se(log theta) and status."""
    th = np.log(np.asarray(theta, float)); d = len(th)
    f = lambda x: loglik(tracks, tuple(np.exp(x)))
    Hm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i]*h, np.eye(d)[j]*h
            Hm[i, j] = Hm[j, i] = (f(th+ei+ej)-f(th+ei-ej)-f(th-ei+ej)+f(th-ei-ej))/(4*h*h)
    info = -Hm
    w = np.linalg.eigvalsh(info)
    out = {'observed_information_eigenvalues': w.tolist()}
    if w.min() <= 0:
        out.update(se_log=[float('inf')]*d, status={n: 'NOT_IDENTIFIED' for n in NAMES}, reason='observed information not positive definite')
        return out
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    out.update(se_log=se.tolist(), status={n: ('IDENTIFIED' if 1.96*s < math.log(2) else 'NOT_IDENTIFIED') for n, s in zip(NAMES, se)},
               rule='identified when 1.96 se(log theta) < ln 2 (95 % interval within a factor 2)')
    return out


def heldout_nis(tracks, theta):
    """Normalized innovation squared per scalar measurement (expected 1) and the signed mean innovation on held-out tracks."""
    F, Q, _ = _system(tracks['times'], theta)
    m0, P0 = _prior(tracks, theta)
    K, n, m = tracks['y'].shape
    R = theta[2]*torch.eye(m, dtype=torch.float64)
    s, P = m0, P0; nis, cnt, sig = 0., 0, torch.zeros(m, dtype=torch.float64)
    for k in range(K):
        if k > 0: s = s@F[k-1].T; P = F[k-1]@P@F[k-1].T+Q[k-1]
        pr = tracks['present'][k]
        if not bool(pr.any()): continue
        Hk = tracks['H'][k][pr]; nu = tracks['y'][k][pr]-(Hk@s[pr][..., None])[..., 0]
        S = Hk@P[pr]@Hk.transpose(-1, -2)+R
        nis += float((nu*torch.linalg.solve(S, nu[..., None])[..., 0]).sum()); cnt += int(pr.sum())*m; sig += nu.sum(0)
        Kg = torch.linalg.solve(S, Hk@P[pr]).transpose(-1, -2)
        s = s.clone(); P = P.clone(); s[pr] = s[pr]+(Kg@nu[..., None])[..., 0]
        IKH = torch.eye(6, dtype=torch.float64)-Kg@Hk; P[pr] = IKH@P[pr]@IKH.transpose(-1, -2)+Kg@R@Kg.transpose(-1, -2)
    return {'nis_per_dof': nis/cnt, 'scalar_measurements': cnt, 'signed_mean_innovation': (sig/max(cnt//m, 1)).tolist()}


def finite_difference_q(positions, times):
    """The NAIVE estimate (G5.3 counterexample): q from second differences of noisy positions, ignoring the measurement
    noise and the posterior. For white acceleration the exact second difference over a uniform dt has per-axis variance
    q int K^2 = (2/3) q dt^3 (triangular kernel), so q = mean |x_{k+1} - 2 x_k + x_{k-1}|^2 / (2 dt^3); measurement noise
    adds 6 sr2 per axis and inflates it."""
    dt = float(np.mean(np.diff(times)))
    d2 = positions[2:]-2*positions[1:-1]+positions[:-2]
    return float((d2**2).sum(-1).mean()/(2*dt**3))
