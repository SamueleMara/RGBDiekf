"""Whole-sequence temporal prior over provisional trajectories (PDF (16), spec C4; contract E18) and matrix-free PCG.

State s_it = [x_it; v_it] (row vectors), actual signed intervals dt_t = tau_t - tau_t-1, transition/process noise of
estimator/point_kalman_cuda (white acceleration q_c, F = [[I, dt I], [0, I]], Q = [[tau^3/3, sgn tau^2/2],[sgn tau^2/2, tau]] q_c):
    E_time(S) = 1/2 sum_i |s_i0 - s0bar_i|^2_{P_i0^-1} + 1/2 sum_t sum_i |s_it - s_i,t-1 F_t^T|^2_{Q_t^-1}
    s0bar_i = [x_i^0; 0],  P_i0 = blockdiag(Sigma_i^0 (the source back-projection covariance, counted ONCE here), P_vv)
neg-log prior density = E_time + log_normalizer (all Gaussian constants kept for evidence use). E_time is quadratic:
its Hessian L (block tridiagonal per point in time, 6x6 blocks) is applied exactly, and L z = r is solved exactly by a
batched block-Thomas elimination, point-chunked (no N x T x 6 x 6 factor stored). q_c must be SPD: a zero/improper
process covariance needs a constraint (null-space) method, which is not implemented -> BLOCKED_MODEL_UNSPECIFIED.
"""
import math
import torch
from estimator.point_kalman_cuda import transition, process_noise


class TemporalPrior:
    def __init__(self, times, qc, x0, P0x, Pvv, chunk=16384):
        dt, dev = x0.dtype, x0.device
        self.times = [float(t) for t in times]; self.T = len(self.times)-1; self.N = len(x0); self.chunk = chunk
        qc = torch.as_tensor(qc, dtype=dt, device=dev)
        if float(torch.linalg.eigvalsh(qc).min()) <= 0:
            raise ValueError('BLOCKED_MODEL_UNSPECIFIED: q_c is not SPD; an improper process covariance needs a null-space constraint method (not implemented)')
        self.F, self.Qi, ldq = [], [], 0.
        for t in range(1, self.T+1):
            d = self.times[t]-self.times[t-1]
            F = transition(d, dt, dev); Q = process_noise(d, qc)
            self.F.append(F); self.Qi.append(torch.cholesky_inverse(torch.linalg.cholesky(Q)))
            ldq += float(torch.logdet(2*math.pi*Q))
        P0 = torch.zeros(self.N, 6, 6, dtype=dt, device=dev); P0[:, :3, :3] = torch.as_tensor(P0x, dtype=dt, device=dev)
        P0[:, 3:, 3:] = torch.as_tensor(Pvv, dtype=dt, device=dev)
        C0 = torch.linalg.cholesky(P0)
        self.P0i = torch.cholesky_inverse(C0)
        self.s0bar = torch.cat([x0, torch.zeros_like(x0)], 1)
        self.log_normalizer = .5*float((2*torch.log(torch.diagonal(C0, dim1=-2, dim2=-1)).sum()+self.N*6*math.log(2*math.pi)))+.5*self.N*ldq

    def _lin(self, S, offset):
        """(E, grad) of the quadratic with s0bar replaced by `offset` (None: 0, i.e. the Hessian action)."""
        r0 = S[:, 0]-(offset if offset is not None else 0.)
        g = torch.zeros_like(S); P0r = torch.einsum('nij,nj->ni', self.P0i, r0); g[:, 0] = P0r; E = .5*float((r0*P0r).sum())
        for t in range(1, self.T+1):
            w = S[:, t]-S[:, t-1]@self.F[t-1].T; wq = w@self.Qi[t-1]
            g[:, t] += wq; g[:, t-1] -= wq@self.F[t-1]; E += .5*float((w*wq).sum())
        return E, g

    def energy(self, S): return self._lin(S, self.s0bar)[0]

    def grad(self, S): return self._lin(S, self.s0bar)[1]

    def hess_apply(self, U): return self._lin(U, None)[1]

    def solve(self, R):
        """Exact L^-1 R by block-Thomas per point (L = Hessian of E_time)."""
        out = torch.empty_like(R)
        Up = [-(F.T@Qi) for F, Qi in zip(self.F, self.Qi)]           # L[t-1, t]
        Lo = [-(Qi@F) for F, Qi in zip(self.F, self.Qi)]             # L[t, t-1]
        FQF = [F.T@Qi@F for F, Qi in zip(self.F, self.Qi)]
        for i0 in range(0, self.N, self.chunk):
            r = R[i0:i0+self.chunk, :, :, None]; n = len(r)
            Cp, dp = [], []
            for t in range(self.T+1):
                Dt = (self.P0i[i0:i0+n] if t == 0 else self.Qi[t-1].expand(n, 6, 6)) + (FQF[t] if t < self.T else 0.)
                rt = r[:, t]
                if t > 0: Dt = Dt-Lo[t-1]@Cp[t-1]; rt = rt-Lo[t-1]@dp[t-1]
                ch = torch.linalg.cholesky(Dt)
                dp.append(torch.cholesky_solve(rt, ch)); Cp.append(torch.cholesky_solve(Up[t].expand(n, 6, 6), ch) if t < self.T else None)
            z = dp[self.T]; out[i0:i0+n, self.T] = z[..., 0]
            for t in range(self.T-1, -1, -1):
                z = dp[t]-Cp[t]@z; out[i0:i0+n, t] = z[..., 0]
        return out


def pcg(apply_A, b, apply_Minv, rtol, maxit):
    """Preconditioned CG on flattened tensors; returns (x, certificate). Hitting maxit is ITERATION_CAP, not convergence."""
    dot = lambda u, v: float((u*v).sum())
    x = torch.zeros_like(b); r = b.clone(); z = apply_Minv(r); p = z.clone(); rz = dot(r, z); bn = math.sqrt(dot(b, b))
    if bn == 0: return x, {'iterations': 0, 'relative_residual': 0., 'status': 'ZERO_RHS'}
    hist = []
    for k in range(1, maxit+1):
        Ap = apply_A(p); pAp = dot(p, Ap)
        if not pAp > 0: return x, {'iterations': k, 'relative_residual': math.sqrt(dot(r, r))/bn, 'status': 'NONPOSITIVE_CURVATURE', 'history': hist}
        a = rz/pAp; x = x+a*p; r = r-a*Ap
        rel = math.sqrt(dot(r, r))/bn; hist.append(rel)
        if rel <= rtol: return x, {'iterations': k, 'relative_residual': rel, 'status': 'CONVERGED', 'history': hist}
        z = apply_Minv(r); rz2 = dot(r, z); p = z+(rz2/rz)*p; rz = rz2
    return x, {'iterations': maxit, 'relative_residual': hist[-1] if hist else 1., 'status': 'ITERATION_CAP', 'history': hist}
