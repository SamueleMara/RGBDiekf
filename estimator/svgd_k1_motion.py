"""K = 1 joint pipeline, shared rigid motion (spec section 2): ONE body-twist sequence for the one moving support.

    T_0 = I,   T_t = T_{t-1} Exp_SE(3)(dt_t nu_t),   nu_t = [v_t (m/s, body frame); w_t (rad/s)]
    nu_1 ~ N(0, P_nu0),   nu_t | nu_{t-1} ~ N(nu_{t-1}, dt_t Q_nu)   (t >= 2)      -- paid ONCE per moving body, never per point
Exp: R = I + A W + B W^2, V = I + B W + C W^2, A = sin th/th, B = (1 - cos th)/th^2, C = (th - sin th)/th^3, with Taylor series in th^2
below 1e-6 rad^2 so values AND autograd gradients stay finite at nu = 0. The stacked prior over (nu_1..nu_T) is Gaussian with
Cov(nu_t, nu_s) = P_nu0 + sum_{r=2}^{min(t,s)} dt_r Q_nu; its Cholesky factor L whitens (nu = L v). SVGD works on this body-twist
coordinate density (no intrinsic Haar-measure claim).
"""
import math
import torch

LOG2PI = math.log(2*math.pi)


def hat(w):
    O = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    O[..., 0, 1], O[..., 0, 2], O[..., 1, 0] = -w[..., 2], w[..., 1], w[..., 2]
    O[..., 1, 2], O[..., 2, 0], O[..., 2, 1] = -w[..., 0], -w[..., 1], w[..., 0]
    return O


def _coeffs(th2):
    small = th2 < 1e-6; ts = torch.where(small, torch.ones_like(th2), th2); th = torch.sqrt(ts)
    A = torch.where(small, 1-th2/6+th2*th2/120, torch.sin(th)/th)
    B = torch.where(small, .5-th2/24+th2*th2/720, (1-torch.cos(th))/ts)
    C = torch.where(small, 1/6-th2/120+th2*th2/5040, (th-torch.sin(th))/(ts*th))
    return A, B, C


def se3_exp(xi):
    """xi [...,6] = (rho, phi) -> 4x4."""
    rho, phi = xi[..., :3], xi[..., 3:]; W = hat(phi); W2 = W@W; A, B, C = _coeffs((phi*phi).sum(-1))
    I = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(W.shape)
    R = I+A[..., None, None]*W+B[..., None, None]*W2; V = I+B[..., None, None]*W+C[..., None, None]*W2
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device); T[..., :3, :3] = R; T[..., :3, 3] = (V@rho[..., None])[..., 0]; T[..., 3, 3] = 1
    return T


def so3_log(R):
    c = ((R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]-1)/2).clamp(-1, 1); th = torch.acos(c)
    v = torch.stack([R[..., 2, 1]-R[..., 1, 2], R[..., 0, 2]-R[..., 2, 0], R[..., 1, 0]-R[..., 0, 1]], -1)
    s = torch.where(th < 1e-6, torch.ones_like(th), torch.sin(th))
    return torch.where((th < 1e-6)[..., None], v/2, v*(th/(2*s))[..., None])


def se3_log(T):
    """Inverse of se3_exp (offline / oracle use; |rotation| < pi)."""
    phi = so3_log(T[..., :3, :3]); W = hat(phi); W2 = W@W; A, B, C = _coeffs((phi*phi).sum(-1))
    I = torch.eye(3, dtype=T.dtype, device=T.device).expand(W.shape)
    V = I+B[..., None, None]*W+C[..., None, None]*W2
    return torch.cat([torch.linalg.solve(V, T[..., :3, 3:])[..., 0], phi], -1)


def compose(nu, dts):
    """nu [T,6] (frames 1..T), dts [T] -> poses [T+1,4,4] with poses[0] = I exactly."""
    out = [torch.eye(4, dtype=nu.dtype, device=nu.device)]
    for t in range(len(dts)): out.append(out[-1]@se3_exp(dts[t]*nu[t]))
    return torch.stack(out)


class MotionPrior:
    def __init__(self, dts, P0, Q, device='cpu'):
        self.dts = torch.as_tensor(dts, dtype=torch.float64, device=device); T = len(dts); self.T = T
        P0 = torch.as_tensor(P0, dtype=torch.float64, device=device); Q = torch.as_tensor(Q, dtype=torch.float64, device=device)
        for nm, A in (('P_nu0', P0), ('Q_nu', Q)):
            if float(torch.linalg.eigvalsh(A).min()) <= 0: raise ValueError(f'BLOCKED_MODEL_UNSPECIFIED: {nm} not SPD')
        self.P0, self.Q = P0, Q
        self.P0i, self.Qi = torch.linalg.inv(P0), torch.linalg.inv(Q)
        self.ld0 = float(torch.logdet(P0)); self.ldq = float(torch.logdet(Q))
        C = torch.zeros(6*T, 6*T, dtype=torch.float64, device=device); acc = [P0]
        for t in range(1, T): acc.append(acc[-1]+self.dts[t]*Q)
        for t in range(T):
            for s in range(T): C[6*t:6*t+6, 6*s:6*s+6] = acc[min(t, s)]
        self.cov = C; self.L = torch.linalg.cholesky(C)

    def log_prob(self, nu):
        """Exact normalized log density of the stacked twists (random walk), evaluated once per hypothesis."""
        lp = -.5*(nu[0]@self.P0i@nu[0])-.5*self.ld0-3*LOG2PI
        for t in range(1, self.T):
            d = nu[t]-nu[t-1]; dt = self.dts[t]
            lp = lp-.5*(d@self.Qi@d)/dt-.5*(self.ldq+6*torch.log(dt))-3*LOG2PI
        return lp

    def from_white(self, v): return (self.L@v.reshape(-1)).reshape(self.T, 6)

    def to_white(self, nu): return torch.linalg.solve_triangular(self.L, nu.reshape(-1, 1), upper=False)[:, 0]
