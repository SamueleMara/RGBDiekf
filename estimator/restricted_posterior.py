"""Phase-1 posterior covariance behind a named interface (contract section 5).

The implemented estimator updates s+ = s- + K_G (r + H e) with the RESTRICTED gain

    K_G = G B^-1 G^T H^T W,     B = G^T (Lambda- + H^T W H) G,     H = [dh/dx, 0]

(W: the frozen IRLS weights of the final linearization; no trust-region damping). Its error
covariance under measurement noise R is the restricted Joseph form

    P+ = (I - K_G H) P- (I - K_G H)^T + K_G R K_G^T .

P+ = (G^T A G)^-1 is NOT this (it is not even 6N x 6N).

'restricted_joseph_marginals' (production): the EXACT 6x6 marginal blocks of that P+ given a
block-diagonal (assumed-density) P-. With U = G^T C~ (C~ = embedded H^T W H, nonzero only on
position columns), V = G^T H^T W R W H G and M_i = G_i B^-1:

    P+_ii = P-_ii - M_i U_:,i P-_ii - (.)^T + M_i (U P- U^T + V) M_i^T .

U, V and U P- U^T are accumulated from co-contributing (point, point) pairs of every pixel of every
camera (exact, no sampling); B is dense 12S x 12S and is inverted on its range: null(B) = null(G)
(Lambda- is PD), so (B + Pi_null)^-1 = B^+ + Pi_null and G (B + Pi_null)^-1 G^T = G B^+ G^T.
The only approximation is the assumed density: cross-point covariance of P- is dropped between
frames. tests/test_image_iekf.py quantifies it against the dense 6N recursion over >= 5 frames.

'rigeuler_block_information' (comparison only): the pinned covariance.approximate_posterior_information
contract Lambda+_i = Lambda-_i + blockdiag(F_i, 0); ignores the restriction and cross-point terms.
"""
import time
import numpy as np
import torch
from estimator.point_renderer import _segments

METHODS = ('restricted_joseph_marginals', 'rigeuler_block_information')


def segment_pairs(lengths, budget):
    """All ordered (i, j) positions within each segment of a segment-sorted array, chunked to ~budget pairs."""
    if len(lengths) == 0: return
    offs = torch.cumsum(lengths, 0)-lengths; sq = lengths*lengths; csq = torch.cumsum(sq, 0)
    s0, n = 0, len(lengths)
    while s0 < n:
        base = int(csq[s0-1]) if s0 else 0
        s1 = max(int(torch.searchsorted(csq, torch.tensor(base+budget, device=csq.device), right=True)), s0+1)
        L, o, q = lengths[s0:s1], offs[s0:s1], sq[s0:s1]
        seg = torch.repeat_interleave(torch.arange(len(L), device=L.device), q)
        local = torch.arange(int(q.sum()), device=L.device)-torch.repeat_interleave(torch.cumsum(q, 0)-q, q)
        yield o[seg]+torch.div(local, L[seg], rounding_mode='floor'), o[seg]+local % L[seg]
        s0 = s1


def _reduce(keys, vals):
    order, unique, lengths = _segments(keys)
    return unique, torch.segment_reduce(vals[order], 'sum', lengths=lengths)


def channel_weights(obs, terms):
    """Per camera diagonal [H*W,5] IRLS weights W and W R W (R: obs.noise_variance)."""
    out = []
    for (scale, rgb, dprec, cprec), r in zip(terms['weights'], obs.noise_variance()):
        w = torch.cat([scale[:, None]*rgb, dprec[:, None], cprec[:, None]], 1)
        out.append((w, w*w*r))
    return out


@torch.no_grad()
def group_point_blocks(lin, weights, basis, budget=400_000):
    """Sparse U_W, U_D: keys g*N+b -> (G_a^T J_pa^T W_p J_pb summed over pixels p and contributors a of group g) [K,6,3]."""
    n = lin.n; keys, vals = [], []
    for v, (w, wrw) in zip(lin.views, weights):
        order, _, lengths = v.pix_seg
        pt, pix, jac = v.point[order], v.pixel[order], v.jac[order]
        for i, j in segment_pairs(lengths, budget):
            a, b = pt[i], pt[j]
            jg = jac[i]@basis.pos[a]                                                    # [P,5,6]
            ww = torch.stack([w[pix[i]], wrw[pix[i]]], 1)                               # [P,2,5]
            prod = torch.einsum('pcg,pkc,pcx->pkgx', jg, ww, jac[j])                     # [P,2,6,3]
            k, val = _reduce(basis.member[a]*n+b, prod.reshape(len(a), -1))
            keys.append(k); vals.append(val)
    if not keys:
        z = basis.pos.new_zeros((0, 2, 6, 3)); return torch.zeros(0, dtype=torch.long, device=basis.pos.device), z
    k, val = _reduce(torch.cat(keys), torch.cat(vals))
    return k, val.reshape(-1, 2, 6, 3)


def _dense_pose(S, rows, cols, blocks):
    """Sum [K,6,6] blocks into a dense [6S,6S] (deterministic segment sums)."""
    k, val = _reduce(rows*S+cols, blocks.reshape(len(blocks), -1))
    out = blocks.new_zeros((S*S, 36)); out[k] = val
    return out.reshape(S, S, 6, 6).permute(0, 2, 1, 3).reshape(6*S, 6*S)


@torch.no_grad()
def restricted_system(P_minus, lin, obs, terms, basis, threads=8, budget=600_000):
    """Shared pieces of the restricted update at the final linearization (gain from the block prior information).

    Returns g, b (group, point) of every sparse U block, Uw = G_g^T C~_{:,b} [K,6,3], Ud (same with W R W),
    GCG = G^T H^T W H G and V = G^T H^T W R W H G on pose coordinates [6S,6S], Binv [12S,12S] (CPU), pose index."""
    N, S = lin.n, basis.count
    clock = time.monotonic()
    keys, U = group_point_blocks(lin, channel_weights(obs, terms), basis, budget)
    t_pairs = time.monotonic()-clock
    g, b = torch.div(keys, N, rounding_mode='floor'), keys % N
    Uw, Ud = U[:, 0], U[:, 1]
    pos_b = basis.pos[b]
    GCG = _dense_pose(S, g, basis.member[b], Uw@pos_b)
    V = _dense_pose(S, g, basis.member[b], Ud@pos_b)
    prior = basis.blocks(torch.linalg.inv(P_minus))                                    # [S,12,12]
    null, n_null = basis.null_projector()
    prev = torch.get_num_threads(); torch.set_num_threads(threads)
    try:
        cpu = lambda t: t.detach().to('cpu', torch.float64)
        B = torch.block_diag(*cpu(prior+null))
        pose = (torch.arange(S)[:, None]*12+torch.arange(6)[None]).reshape(-1)
        B[pose[:, None], pose[None, :]] += cpu(GCG)
        B = (B+B.T)/2
        # Symmetric Jacobi equilibration (pure numerics; K_G unchanged): B = S Bt S, S = diag(B)^(1/2). Measured on a blade
        # crop: unscaled cond(B) ~ 1e12-5e13 (position precisions ~1e7 next to velocity precisions ~1e-3), and the expanded
        # Joseph marginals left their dense float64 oracle (1e-8 -> 2e-3) exactly when cond(B) reached 5e13.
        sc = B.diagonal().clamp_min(1e-300).sqrt()
        Bt = B/sc[:, None]/sc[None, :]
        L, info = torch.linalg.cholesky_ex(Bt)
        if int(info): raise np.linalg.LinAlgError('restricted information G^T(Lambda- + H^T W H)G + Pi_null is not positive definite')
        Binv = torch.cholesky_inverse(L)/sc[:, None]/sc[None, :]
        cond_eq = None
    finally:
        torch.set_num_threads(prev)
    return {'g': g, 'b': b, 'Uw': Uw, 'Ud': Ud, 'GCG': GCG, 'V': V, 'Binv': Binv, 'pose': pose, 'null_directions': n_null,
            'seconds': {'pixel_pairs': t_pairs, 'system_total': time.monotonic()-clock}}


def _point_pairs_sum(sys, S, D, budget=600_000):
    """sum_b U_b D_b^{xx} U_b^T on pose coordinates [6S,6S] (pairs of U blocks sharing a point b); chunked, deterministic."""
    g, b, Uw = sys['g'], sys['b'], sys['Uw']
    order, _, lengths = _segments(b)
    ob = b[order]
    out = torch.zeros((6*S, 6*S), dtype=D.dtype, device=D.device)
    for i, j in segment_pairs(lengths, budget):
        e1, e2 = order[i], order[j]
        out += _dense_pose(S, g[e1], g[e2], Uw[e1]@D[ob[i], :3, :3]@Uw[e2].transpose(1, 2))
    return out


@torch.no_grad()
def restricted_joseph_marginals(P_minus, lin, obs, terms, basis, threads=8, budget=400_000):
    clock = time.monotonic()
    N, S = lin.n, basis.count
    sys = restricted_system(P_minus, lin, obs, terms, basis, threads, budget)
    g, b, Uw, Binv, pose = sys['g'], sys['b'], sys['Uw'], sys['Binv'], sys['pose']
    Y = sys['V']+_point_pairs_sum(sys, S, P_minus, budget)
    prev = torch.get_num_threads(); torch.set_num_threads(threads)
    try:
        cpu = lambda t: t.detach().to('cpu', torch.float64)
        Bp = Binv[:, pose]                                                              # [12S,6S]
        T = Bp@cpu(Y)                                                                   # [12S,6S]
        Z = torch.bmm(T.reshape(S, 12, 6*S), Bp.reshape(S, 12, 6*S).transpose(1, 2))   # [S,12,12] diag blocks of B^-1 Y B^-1
        Bsub = Bp.reshape(S, 12, S, 6)[cpu(basis.member[b]).long(), :, cpu(g).long()]    # [K,12,6]
        Q = Bsub@cpu(Uw)                                                                # [K,12,3]
    finally:
        torch.set_num_threads(prev)
    dev = P_minus.device
    Qb = P_minus.new_zeros((N, 12, 3))
    if len(b):
        kb, qv = _reduce(b, Q.to(dev).reshape(len(Q), -1)); Qb[kb] = qv.reshape(-1, 12, 3)
    Gi = basis.matrix()
    X = Gi@Qb@P_minus[:, :3, :]
    post = P_minus-X-X.transpose(1, 2)+Gi@Z.to(dev)[basis.member]@Gi.transpose(1, 2)
    post = (post+post.transpose(1, 2))/2
    post, floor = project_psd(post)
    measured = torch.zeros(N, dtype=torch.bool, device=dev); measured[b] = True
    return post, {'method': 'restricted_joseph_marginals', 'group_point_blocks': int(len(b)), 'restricted_dimension': 12*S,
                  'null_directions_of_G': sys['null_directions'], 'psd_repair': floor, 'measured_points': int(measured.sum()),
                  'seconds': time.monotonic()-clock}


@torch.no_grad()
def rigeuler_block_information(P_minus, lin, obs, terms, basis=None):
    F = lin.block_fisher(terms['weights'])
    observed = F.abs().sum((1, 2)) > 0
    info = torch.linalg.inv(P_minus); info[:, :3, :3] += F
    post = P_minus.clone()
    if bool(observed.any()): post[observed] = torch.linalg.inv(info[observed])
    post, floor = project_psd((post+post.transpose(1, 2))/2)
    return post, {'method': METHODS[1], 'psd_repair': floor}


def posterior(method, *args, **kw):
    if method == METHODS[0]: return restricted_joseph_marginals(*args, **kw)
    if method == METHODS[1]: return rigeuler_block_information(*args, **{k: v for k, v in kw.items() if k == 'basis'})
    raise ValueError(f'unknown posterior method {method!r}; choose one of {METHODS}')


def project_psd(P, rel=1e-12):
    """Symmetric eigenvalue floor at rel * max eigenvalue per block; every repair is counted and returned."""
    vals, vecs = torch.linalg.eigh(P.cpu())
    floor = vals[:, -1:].clamp_min(0)*rel
    bad = (vals < floor).any(1)
    report = {'blocks_floored': int(bad.sum()), 'min_eigenvalue_before': float(vals[:, 0].min()) if len(vals) else None,
              'most_negative_relative': float((vals[:, 0]/vals[:, -1].clamp_min(1e-300)).min()) if len(vals) else None}
    if bool(bad.any()):
        fixed = (vecs[bad]*torch.maximum(vals[bad], floor[bad])[:, None, :])@vecs[bad].transpose(1, 2)
        P = P.clone(); P[bad.to(P.device)] = fixed.to(P.device, P.dtype)
    return P, report


# ---------------------------------------------------------------- dense oracles (tests, small scenes only)

def dense_operators(lin, obs, terms, basis):
    """Dense H [m,3N] (by unit-vector JVPs), diagonal W, R [m] and G [6N,12S]."""
    n = lin.n
    cols = []
    for k in range(3*n):
        e = torch.zeros(n*3, dtype=torch.float64, device=basis.pos.device); e[k] = 1
        cols.append(torch.cat([h.reshape(-1) for h in lin.apply_H(e.reshape(n, 3))]))
    H = torch.stack(cols, 1)
    w = torch.cat([a.reshape(-1) for a, _ in channel_weights(obs, terms)])
    r = torch.cat([x.reshape(-1) for x in obs.noise_variance()])
    Gm = basis.matrix(); G = H.new_zeros((6*n, 12*basis.count))
    for i in range(n):
        gi = int(basis.member[i]); G[6*i:6*i+6, 12*gi:12*gi+12] = Gm[i]
    return H, w, r, G


def dense_restricted_joseph(P, H, w, r, G, P_gain=None):
    """Full (6N x 6N) restricted Joseph update of a dense prior P (pseudo-inverse of B on range(G^T)).
    P_gain: the prior the GAIN is built from (default P); Joseph is the error covariance of that gain under P."""
    n3 = H.shape[1]
    Hf = torch.cat([H.reshape(H.shape[0], -1, 3), torch.zeros_like(H.reshape(H.shape[0], -1, 3))], 2).reshape(H.shape[0], 2*n3)
    lam = torch.linalg.inv(P if P_gain is None else P_gain)
    B = G.T@(lam+Hf.T@(w[:, None]*Hf))@G
    K = G@torch.linalg.pinv(B, hermitian=True, rtol=1e-12)@G.T@Hf.T*w[None, :]
    I = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    A = I-K@Hf
    out = A@P@A.T+(K*r[None, :])@K.T
    return (out+out.T)/2, K, Hf


def blocks_of(P, n):
    return torch.stack([P[6*i:6*i+6, 6*i:6*i+6] for i in range(n)])
