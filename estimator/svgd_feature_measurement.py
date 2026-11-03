"""Operator C: XYZ-RGB feature observation (PDF (17)-(22), spec C1-C3/C5; contract E09-E14).

Declared finite feature model (TF.5 exception: the finite map IS the observation model psi, not the infinite kernel):
    psi_r(x,c) = sqrt(2/D) cos(wx_r^T x + wc_r^T c + b_r),  wx_r ~ N(0, BX^-1), wc_r ~ N(0, BC^-1), b_r ~ U(0, 2 pi)
drawn ONCE from a fixed seed (CPU generator, float64) and hashed. E[psi(z)^T psi(z')] = exp(-1/2 dx^T BX^-1 dx - 1/2 dc^T BC^-1 dc).
    y^F_t = sum_j w_j psi(y_j, d_j) (w_j = 1/M_t),   h_t(X) = sum_i alpha_i psi(x_i, c_i) (alpha_i = 1/|V_t| on the declared V_t)
    e_t = y^F_t - h_t,  E_F,t = 1/2 e^T Gamma^-1 e + 1/2 log det(2 pi Gamma)          (the ONE feature likelihood per frame)
    d h_t / d x_i = alpha_i J_psi(x_i,c_i),  J_psi = -sqrt(2/D) sin(phase) wx^T     (H_t,i = -alpha_i [J_psi, 0] for e)
JVP  J u = sum_i alpha_i J_psi,i u_i  [D];  VJP  J^T w = alpha_i J_psi,i^T w  [N,3]: tiled over points, never an N x D Jacobian
held across frames. alpha frozen inside a conditional step (V_t fixed): the Jacobian is then the CONDITIONAL one (spec C3).
Exact-kernel MMD (exact_mmd2) is the CPU/NumPy oracle only.
"""
import hashlib
import math
import numpy as np
import torch


class RFFFeatures:
    def __init__(self, B_x, B_c, D, seed, dtype=torch.float64, device='cpu', chunk=8192):
        g = torch.Generator('cpu').manual_seed(int(seed))
        xi = torch.randn(D, 6, generator=g, dtype=torch.float64); b = torch.rand(D, generator=g, dtype=torch.float64)*2*math.pi
        Lx = torch.linalg.cholesky(torch.linalg.inv(torch.as_tensor(B_x, dtype=torch.float64)))
        Lc = torch.linalg.cholesky(torch.linalg.inv(torch.as_tensor(B_c, dtype=torch.float64)))
        self.Wx, self.Wc = (xi[:, :3]@Lx.T).to(device, dtype), (xi[:, 3:]@Lc.T).to(device, dtype)
        self.b, self.D, self.scale, self.chunk, self.seed = b.to(device, dtype), int(D), math.sqrt(2/D), chunk, int(seed)
        self.B_x, self.B_c = np.asarray(B_x, float), np.asarray(B_c, float)

    def digest(self):
        h = hashlib.sha256()
        for a in (self.Wx, self.Wc, self.b): h.update(a.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def phase(self, x, c): return x@self.Wx.T+c@self.Wc.T+self.b

    def psi(self, x, c): return self.scale*torch.cos(self.phase(x, c))

    def _w(self, n, weights, like):
        return torch.full((n,), 1./n, dtype=like.dtype, device=like.device) if weights is None else weights

    def mean(self, x, c, weights=None):
        w = self._w(len(x), weights, x); out = torch.zeros(self.D, dtype=x.dtype, device=x.device)
        for i in range(0, len(x), self.chunk): out += w[i:i+self.chunk]@self.psi(x[i:i+self.chunk], c[i:i+self.chunk])
        return out

    def jvp(self, x, c, u, weights=None):
        """sum_i alpha_i J_psi(x_i,c_i) u_i  [D]  (u [N,3] metres)."""
        w = self._w(len(x), weights, x); out = torch.zeros(self.D, dtype=x.dtype, device=x.device)
        for i in range(0, len(x), self.chunk):
            s = torch.sin(self.phase(x[i:i+self.chunk], c[i:i+self.chunk]))
            out -= self.scale*(w[i:i+self.chunk, None]*s*(u[i:i+self.chunk]@self.Wx.T)).sum(0)
        return out

    def vjp(self, x, c, v, weights=None):
        """alpha_i J_psi(x_i,c_i)^T v  [N,3]."""
        w = self._w(len(x), weights, x); out = torch.empty(len(x), 3, dtype=x.dtype, device=x.device)
        for i in range(0, len(x), self.chunk):
            s = torch.sin(self.phase(x[i:i+self.chunk], c[i:i+self.chunk]))
            out[i:i+self.chunk] = -self.scale*w[i:i+self.chunk, None]*((s*v[None])@self.Wx)
        return out

    def covariance(self, x, c, weights=None):
        """sum_i w_i (psi_i - mu)(psi_i - mu)^T  [D,D] (two-pass, centred)."""
        w = self._w(len(x), weights, x); mu = self.mean(x, c, w); C = torch.zeros(self.D, self.D, dtype=x.dtype, device=x.device)
        for i in range(0, len(x), self.chunk):
            p = self.psi(x[i:i+self.chunk], c[i:i+self.chunk])-mu
            C += (p*w[i:i+self.chunk, None]).T@p
        return (C+C.T)/2


def exact_kernel(x1, c1, x2, c2, B_x, B_c):
    Px, Pc = np.linalg.inv(B_x), np.linalg.inv(B_c)
    dx, dc = x1[:, None]-x2[None], c1[:, None]-c2[None]
    return np.exp(-.5*np.einsum('nmi,ij,nmj->nm', dx, Px, dx)-.5*np.einsum('nmi,ij,nmj->nm', dc, Pc, dc))


def exact_mmd2(xp, cp, xq, cq, B_x, B_c, wp=None, wq=None):
    """Squared MMD of the weighted empirical measures under the exact Gaussian product kernel (PDF (22)); NumPy oracle."""
    wp = np.full(len(xp), 1/len(xp)) if wp is None else wp; wq = np.full(len(xq), 1/len(xq)) if wq is None else wq
    return float(wp@exact_kernel(xp, cp, xp, cp, B_x, B_c)@wp+wq@exact_kernel(xq, cq, xq, cq, B_x, B_c)@wq-2*wp@exact_kernel(xp, cp, xq, cq, B_x, B_c)@wq)


class FeatureGaussian:
    """N_D(0, Gamma) with its exact normalizer; a non-SPD Gamma is a NUMERICAL_FAILURE (no added diagonal)."""
    def __init__(self, Gamma):
        G = torch.as_tensor(Gamma); G = (G+G.T)/2
        try: self.L = torch.linalg.cholesky(G)
        except RuntimeError as e: raise FloatingPointError(f'NUMERICAL_FAILURE: Gamma is not SPD ({e})') from None
        self.Gamma, self.D = G, len(G)
        self.logdet = float(2*torch.log(torch.diagonal(self.L)).sum())
        ev = torch.linalg.eigvalsh(G); self.condition = float(ev[-1]/ev[0])

    def solve(self, e): return torch.cholesky_solve(e[:, None], self.L)[:, 0]

    def quad(self, e): return float(e@self.solve(e))

    def nll(self, e): return .5*self.quad(e)+.5*self.logdet+.5*self.D*math.log(2*math.pi)


def iid_sampling_gamma(features, y, d, x0, c0):
    """Experimental Gamma_t (DEV-0002, PROPOSED_NOT_AUTHORIZED): covariance of the difference of two INDEPENDENT
    i.i.d.-sample feature means, C_Q/M_t + C_P/N, frozen at the stationary initialization. Ignores spatial correlation
    of neighbouring rays, cross-camera and cross-time dependence: not a calibrated feature covariance."""
    return features.covariance(y, d)/len(y)+features.covariance(x0, c0)/len(x0)


class FrameFeatureTerm:
    """E_F,t(X_t) for one frame with frozen comparable support (alpha = 1/N on V_t = all; DEV-0001 when used)."""
    def __init__(self, features, y, d, gamma, c_ref):
        self.f, self.c, self.G = features, c_ref, gamma
        self.mu_q = features.mean(y, d)

    def residual(self, x): return self.mu_q-self.f.mean(x, self.c)

    def nll(self, x): return self.G.nll(self.residual(x))

    def grad(self, x, e=None):
        """grad_x E_F,t = -alpha J_psi^T Gamma^-1 e  [N,3]."""
        e = self.residual(x) if e is None else e
        return -self.f.vjp(x, self.c, self.G.solve(e))

    def gn_apply(self, x, u):
        """Gauss-Newton block J^T Gamma^-1 J u  [N,3] (exact for the frozen support)."""
        return self.f.vjp(x, self.c, self.G.solve(self.f.jvp(x, self.c, u)))
