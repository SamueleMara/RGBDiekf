"""SE(3) for the rigid passes, ONE convention throughout (contract 5.2): twist xi = [v, omega] (translation first),
spatial/world LEFT perturbation T' = Exp(delta^) T.

    hat(xi) = [[omega^, v], [0, 0]],    Exp = matrix exponential (exact, branch free),
    ad(xi)  = [[omega^, v^], [0, omega^]],   Ad(T) = [[R, t^ R], [0, R]]     (Ad(T) xi^ = T xi^ T^-1 in [v, omega])
    J_l(xi) = sum_n ad(xi)^n / (n+1)!        (left Jacobian: Exp(xi + d) = Exp(J_l(xi) d) Exp(xi) to first order)
    Log     : omega from the SO(3) logarithm, v = V(omega)^-1 t with V = sum (omega^)^n/(n+1)!
All functions take/return float64 torch tensors; tests/test_rigid_kalman_cuda.py checks each against finite
differences and scipy's matrix exponential/logarithm in both time directions.
"""
import math
import torch


def skew(w):
    W = torch.zeros((*w.shape[:-1], 3, 3), dtype=w.dtype, device=w.device)
    W[..., 0, 1], W[..., 0, 2], W[..., 1, 0] = -w[..., 2], w[..., 1], w[..., 2]
    W[..., 1, 2], W[..., 2, 0], W[..., 2, 1] = -w[..., 0], -w[..., 1], w[..., 0]
    return W


def hat(xi):
    X = torch.zeros((*xi.shape[:-1], 4, 4), dtype=xi.dtype, device=xi.device)
    X[..., :3, :3] = skew(xi[..., 3:]); X[..., :3, 3] = xi[..., :3]
    return X


def exp(xi):
    return torch.linalg.matrix_exp(hat(xi))


def ad(xi):
    A = torch.zeros((*xi.shape[:-1], 6, 6), dtype=xi.dtype, device=xi.device)
    W = skew(xi[..., 3:])
    A[..., :3, :3] = W; A[..., 3:, 3:] = W; A[..., :3, 3:] = skew(xi[..., :3])
    return A


def Ad(T):
    A = torch.zeros((*T.shape[:-2], 6, 6), dtype=T.dtype, device=T.device)
    R, t = T[..., :3, :3], T[..., :3, 3]
    A[..., :3, :3] = R; A[..., 3:, 3:] = R; A[..., :3, 3:] = skew(t)@R
    return A


def _series(X, terms=30):
    """sum_n X^n/(n+1)!  (converges for every X; terms suffice for |X| <= pi in float64)."""
    out = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device).expand_as(X).clone()
    term = out.clone()
    for n in range(1, terms):
        term = term@X/(n+1)
        out = out+term
    return out


def left_jacobian(xi):
    return _series(ad(xi))


def left_jacobian_inv(xi):
    return torch.linalg.inv(left_jacobian(xi))


def so3_log(R):
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr-1)/2).clamp(-1, 1)
    th = torch.acos(c)
    v = torch.stack((R[..., 2, 1]-R[..., 1, 2], R[..., 0, 2]-R[..., 2, 0], R[..., 1, 0]-R[..., 0, 1]), -1)
    s = torch.sin(th)
    small = th < 1e-6
    near_pi = (math.pi-th) < 1e-4
    w = torch.where(small[..., None], v/2*(1+th[..., None]**2/6), v*(th/torch.where(small | near_pi, torch.ones_like(s), 2*s))[..., None])
    if bool(near_pi.any()):                       # R ~ I + 2 a a^T - ... : axis from the largest diagonal entry
        B = (R+R.transpose(-1, -2))/2-c[..., None, None]*torch.eye(3, dtype=R.dtype, device=R.device)
        i = torch.argmax(B.diagonal(dim1=-2, dim2=-1), -1)
        col = torch.gather(B, -1, i[..., None, None].expand(*B.shape[:-1], 1))[..., 0]
        a = col/col.norm(dim=-1, keepdim=True).clamp_min(1e-300)
        a = torch.where(((a*v).sum(-1) < 0)[..., None], -a, a)
        w = torch.where(near_pi[..., None], a*th[..., None], w)
    return w


def log(T):
    w = so3_log(T[..., :3, :3])
    V = _series(skew(w))
    v = torch.linalg.solve(V, T[..., :3, 3:])[..., 0]
    return torch.cat([v, w], -1)


def inv(T):
    out = torch.zeros_like(T); R, t = T[..., :3, :3], T[..., :3, 3]
    out[..., :3, :3] = R.transpose(-1, -2); out[..., :3, 3] = -(R.transpose(-1, -2)@t[..., None])[..., 0]; out[..., 3, 3] = 1
    return out


def point_jacobian(x):
    """d(Exp(delta^) x)/d delta at delta = 0 for world points x [M,3]: B = [I, -x^]  ([M,3,6])."""
    B = torch.zeros((len(x), 3, 6), dtype=x.dtype, device=x.device)
    B[:, :, :3] = torch.eye(3, dtype=x.dtype, device=x.device); B[:, :, 3:] = -skew(x)
    return B
