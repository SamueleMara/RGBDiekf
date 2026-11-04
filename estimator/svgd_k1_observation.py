"""K = 1 joint pipeline, observation potential (spec section 3.1-3.2). Point sets only.

    p_t(x,c) = sum_j w_j N3(x; y_j, LX_j) N3(c; d_j, LC),   LX_j = Sigma_obs,j + s_t^2 I,   LC = R_obs + Sigma_repr^C      (normalized)
    b_0(x,c) = N3(x; m_x, Cov_P0(x) + mean Sigma_0) N3(c; m_c, Cov_P0(c) + R_0)    (frozen, from the REFERENCE cloud only; never refit)
    ell_t    = rho_t + (1 - rho_t) p_t / b_0,       log ell_t = logaddexp(log rho, log(1 - rho) + log p - log b)
Analytic value, 6D gradient and 6x6 Hessian w.r.t. u = (x, c) (exact; differentiable w.r.t. x, c for the pose gradient):
    s_j = [LX_j^-1 (y_j - x); LC^-1 (d_j - c)],  grad log p = <s>,  hess log p = <s s^T - blkdiag(LX_j^-1, LC^-1)> - <s><s>^T   (<.> = responsibility mean)
    L = log p - log b,  sig = sigmoid(log(1-rho) + L - log rho):  grad log ell = sig grad L,  hess log ell = sig hess L + sig (1-sig) grad L grad L^T
Spatial acceleration (spec 8.4) with a CERTIFIED bound, not a fixed kNN/radius: candidates = all samples in the 27 grid cells of edge
R = kappa sqrt(lambda_max(LX)) around the query, so every omitted sample is farther than R and
    p_true - p_trunc <= dp = c_max exp(-R^2 / (2 lambda_max)),  c_max = max_j N_x-normalizer_j x N_c-normalizer   (sum w <= 1)
    0 <= log ell_true - log ell_trunc <= cert = (1 - rho) dp / (rho b + (1 - rho) p_trunc)      (returned per query)
kappa is chosen from the declared tolerance tau on the domain {b >= b_ref}; queries with cert > tau are counted, never hidden.
"""
import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import chi2

LOG2PI = math.log(2*math.pi)
IU = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
TRI = torch.triu_indices(6, 6).tolist()


def spd_precision(L, what):
    try: C = torch.linalg.cholesky(L)
    except RuntimeError as e: raise FloatingPointError(f'NUMERICAL_FAILURE: {what} not SPD ({e})') from None
    eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device).expand_as(L)
    P = torch.cholesky_solve(eye, C)
    return (P+P.transpose(-1, -2))/2, 2*torch.log(torch.diagonal(C, dim1=-2, dim2=-1)).sum(-1)


def representation_scales(xyz, rgb, cov_rgb):
    """Point-cloud statistics only: s^2 = median squared exact nearest-OTHER-sample distance (all samples, KD-tree);
    Sigma_repr^C = diag(1/2 median (d_j - d_nn(j))^2). Returns (s2, LC = R_obs + Sigma_repr^C)."""
    d, j = cKDTree(xyz).query(xyz, k=2); d, j = d[:, 1], j[:, 1]
    return float(np.median(d)**2), np.asarray(cov_rgb, float)+np.diag(.5*np.median((rgb-rgb[j])**2, 0))


class Background:
    """b_0: frozen reference-cloud Gaussian over (x, c) (spec 3.2). Positive everywhere; never refit."""
    def __init__(self, xyz, rgb, cov_xyz, cov_rgb, weights=None, device='cpu'):
        w = np.full(len(xyz), 1/len(xyz)) if weights is None else np.asarray(weights)/np.sum(weights)
        mx, mc = w@xyz, w@rgb
        Cx = ((xyz-mx)*w[:, None]).T@(xyz-mx)+np.einsum('n,nij->ij', w, cov_xyz); Cc = ((rgb-mc)*w[:, None]).T@(rgb-mc)+np.asarray(cov_rgb, float)
        B = np.zeros((6, 6)); B[:3, :3], B[3:, 3:] = Cx, Cc
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        self.mean, self.cov = t(np.r_[mx, mc]), t(B)
        self.P, ld = spd_precision(self.cov, 'background covariance'); self.logdet = float(ld)

    def log(self, u):
        r = u-self.mean
        return -.5*((r@self.P)*r).sum(-1)-.5*self.logdet-3*LOG2PI

    def log_at_mahalanobis2(self, m2): return -.5*m2-.5*self.logdet-3*LOG2PI


class Grid:
    """Sorted voxel hash of the observed samples (cell edge R) on the device; 27-cell candidate lists, padded per tile."""
    def __init__(self, y, R):
        self.R, dev = float(R), y.device
        self.origin = y.min(0).values-R
        c = torch.floor((y-self.origin)/R).long(); self.dims = (c.max(0).values+2).tolist()
        key = self.key(c); order = torch.argsort(key); self.order = order
        self.ukey, cnt = torch.unique_consecutive(key[order], return_counts=True)
        self.start = torch.cumsum(cnt, 0)-cnt; self.count = cnt
        self.off = torch.tensor([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], device=dev)

    def key(self, c): return (c[..., 0]*self.dims[1]+c[..., 1])*self.dims[2]+c[..., 2]

    def cells(self, x):
        """(start [n,27], count [n,27]) of the 27 neighbour cells of each query (count 0 where empty/outside)."""
        c = torch.floor((x.detach()-self.origin)/self.R).long()[:, None, :]+self.off[None]
        inside = ((c >= 0) & (c < torch.tensor(self.dims, device=x.device))).all(-1)
        k = self.key(c.clamp_min(0)); pos = torch.searchsorted(self.ukey, k.contiguous()).clamp_max(len(self.ukey)-1)
        hit = inside & (self.ukey[pos] == k)
        return self.start[pos], torch.where(hit, self.count[pos], torch.zeros_like(pos))

    def pairs(self, st, cnt):
        """Exact CSR candidate list: (query index [P], sample index [P]) with no padding."""
        cf = cnt.reshape(-1); sf = st.reshape(-1); tot = int(cf.sum())
        qc = torch.repeat_interleave(torch.arange(len(cf), device=cf.device), cf)
        off = torch.arange(tot, device=cf.device)-torch.repeat_interleave(torch.cumsum(cf, 0)-cf, cf)
        return qc//27, self.order[sf[qc]+off]

    def candidates(self, x):
        c = torch.floor((x.detach()-self.origin)/self.R).long()[:, None, :]+self.off[None]          # [n,27,3]
        inside = ((c >= 0) & (c < torch.tensor(self.dims, device=x.device))).all(-1)
        k = self.key(c.clamp_min(0)); pos = torch.searchsorted(self.ukey, k).clamp_max(len(self.ukey)-1)
        hit = inside & (self.ukey[pos] == k)
        cnt = torch.where(hit, self.count[pos], torch.zeros_like(pos)); st = self.start[pos]
        cum = torch.cumsum(cnt, 1); tot = cum[:, -1]; Kmax = int(tot.max()) if len(tot) else 0
        if Kmax == 0: return torch.zeros(len(x), 0, dtype=torch.long, device=x.device), torch.zeros(len(x), 0, dtype=torch.bool, device=x.device)
        p = torch.arange(Kmax, device=x.device)[None].expand(len(x), Kmax)
        cell = torch.searchsorted(cum, p, right=True).clamp_max(26)
        prev = torch.gather(cum-cnt, 1, cell); idx = torch.gather(st, 1, cell)+(p-prev)
        valid = p < tot[:, None]
        return self.order[torch.where(valid, idx, torch.zeros_like(idx))], valid


class FrameKDE:
    """Normalized observed XYZ-RGB mixture of one frame with the certified candidate grid."""
    def __init__(self, xyz, rgb, cov_xyz, s2, LC, log_w=None, device='cpu'):
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=device)
        self.y, self.d, self.M = t(xyz), t(rgb), len(xyz)
        LX = t(cov_xyz)+s2*torch.eye(3, dtype=torch.float64, device=device)
        self.Px, ldx = spd_precision(LX, 'Lambda_X'); self.Pc, ldc = spd_precision(t(LC), 'Lambda_C')
        lw = torch.full((self.M,), -math.log(self.M), dtype=torch.float64, device=device) if log_w is None else t(log_w)
        self.const = lw-.5*ldx-.5*float(ldc)-3*LOG2PI
        self.lam_max = float(np.linalg.eigvalsh(np.asarray(cov_xyz, float)).max()+s2)
        self.log_cmax = float((-.5*ldx-.5*float(ldc)-3*LOG2PI).max())
        self.s2, self.grid, self.kappa = s2, None, None

    def build_grid(self, kappa, tau=None, kappa_small=None):
        """Full grid (radius kappa sqrt(lambda_max), certified on the whole domain) and an optional small first-pass grid: rows whose
        small-grid certificate exceeds tau are recomputed on the full grid, so every returned value satisfies the same certificate."""
        self.kappa = float(kappa); self.R = kappa*math.sqrt(self.lam_max); self.grid = Grid(self.y, self.R)
        self.log_dp = self.log_cmax-.5*kappa*kappa          # log of the omitted-density bound
        self.tau, self.small = tau, None
        if tau is not None and kappa_small is not None and kappa_small < kappa:
            self.small = (Grid(self.y, kappa_small*math.sqrt(self.lam_max)), self.log_cmax-.5*kappa_small*kappa_small, float(kappa_small))

    def stats_csr(self, x, c, q, j):
        """Same quantities as `stats` from an exact pair list (query q, sample j): segment reductions, no padding."""
        n = len(x); dx, dc = self.y[j]-x[q], self.d[j]-c[q]; Pk = self.Px[j]
        sx = (Pk@dx[:, :, None])[:, :, 0]; sc = dc@self.Pc
        l = self.const[j]-.5*(dx*sx).sum(-1)-.5*(dc*sc).sum(-1)
        with torch.no_grad():
            m = torch.full((n,), -float('inf'), dtype=x.dtype, device=x.device).scatter_reduce(0, q, l.detach(), 'amax')
        fin = torch.isfinite(m); m0 = torch.where(fin, m, torch.zeros_like(m))
        e = torch.exp(l-m0[q]); S = torch.zeros(n, dtype=x.dtype, device=x.device).index_add(0, q, e); Ss = torch.where(fin, S, torch.ones_like(S))
        s = torch.cat([sx, sc], -1)
        Es = torch.zeros(n, 6, dtype=x.dtype, device=x.device).index_add(0, q, e[:, None]*s)/Ss[:, None]
        Ess = torch.zeros(n, 21, dtype=x.dtype, device=x.device).index_add(0, q, e[:, None]*(s[:, TRI[0]]*s[:, TRI[1]]))/Ss[:, None]
        Pm = torch.zeros(n, 9, dtype=x.dtype, device=x.device).index_add(0, q, e[:, None]*Pk.reshape(-1, 9))/Ss[:, None]
        M2 = torch.zeros(n, 6, 6, dtype=x.dtype, device=x.device); M2[:, TRI[0], TRI[1]] = Ess; M2[:, TRI[1], TRI[0]] = Ess
        blk = torch.zeros(n, 6, 6, dtype=x.dtype, device=x.device); blk[:, :3, :3] = Pm.reshape(-1, 3, 3); blk[:, 3:, 3:] = self.Pc
        logp = torch.where(fin, m0+torch.log(Ss), torch.full_like(m0, -float('inf')))
        return logp, Es, M2-blk, fin

    def stats(self, x, c, idx, valid):
        """(log p [n], E[s] [n,6], E[s s^T - blkdiag(P)] [n,6,6]) over the candidates (differentiable in x, c)."""
        yk, dk, Pk = self.y[idx], self.d[idx], self.Px[idx]
        dx, dc = yk-x[:, None], dk-c[:, None]
        sx = torch.einsum('nkij,nkj->nki', Pk, dx); sc = dc@self.Pc
        l = self.const[idx]-.5*(dx*sx).sum(-1)-.5*(dc*sc).sum(-1)
        l = torch.where(valid, l, torch.full_like(l, -float('inf')))
        m = l.max(1).values; fin = torch.isfinite(m); m0 = torch.where(fin, m, torch.zeros_like(m))
        e = torch.where(valid, torch.exp(l-m0[:, None]), torch.zeros_like(l)); S = e.sum(1); Ss = torch.where(fin, S, torch.ones_like(S))
        r = e/Ss[:, None]
        s = torch.cat([sx, sc], -1)
        Es = torch.bmm(r[:, None, :], s)[:, 0]                                            # batched products: no [n,K,6,6] intermediate
        Ess = torch.bmm((r[:, :, None]*s).transpose(1, 2), s)
        blk = torch.zeros(len(x), 6, 6, dtype=x.dtype, device=x.device)
        blk[:, :3, :3] = torch.bmm(r[:, None, :], Pk.reshape(len(x), -1, 9))[:, 0].reshape(-1, 3, 3); blk[:, 3:, 3:] = self.Pc
        logp = torch.where(fin, m0+torch.log(Ss), torch.full_like(m0, -float('inf')))
        return logp, Es, Ess-blk, fin


PAIR_BUDGET = {'grad': 1_500_000, 'nograd': 4_000_000}      # hardware budget: query x candidate pairs per chunk (memory only; sums unchanged)


def log_potential(kde, bg, rho, x, c, tile=4096, need_hessian=True):
    """log ell_t(x, c) with gradient [n,6], Hessian [n,6,6] and the per-query truncation certificate (all w.r.t. u = (x, c)).
    Exact CSR candidate pairs, grouped so that each group holds <= PAIR_BUDGET pairs (memory only; sums unchanged)."""
    out_l, out_g, out_H, out_c = [], [], [], []
    budget = PAIR_BUDGET['grad' if torch.is_grad_enabled() and (x.requires_grad or c.requires_grad) else 'nograd']
    for i in range(0, len(x), tile):
        xi, ci = x[i:i+tile], c[i:i+tile]
        st, cnt = kde.grid.cells(xi); per = cnt.sum(1); cum = torch.cumsum(per, 0)
        bounds = [0]+torch.searchsorted(cum, torch.arange(budget, int(cum[-1])+budget, budget, device=x.device, dtype=cum.dtype)).add(1).clamp_max(len(xi)).tolist()
        bounds = sorted(set(bounds+[len(xi)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a: continue
            r = _potential_rows(kde, bg, rho, xi[a:b], ci[a:b], st[a:b], cnt[a:b], need_hessian, kde.log_dp)
            out_l.append(r[0]); out_g.append(r[1]); out_c.append(r[3])
            if need_hessian: out_H.append(r[2])
    cat = lambda v: torch.cat(v) if v else None
    return cat(out_l), cat(out_g), cat(out_H), cat(out_c)


def _potential_rows(kde, bg, rho, xi, ci, st, cnt, need_hessian, log_dp):
    lr, l1r = math.log(rho), math.log1p(-rho)
    if int(cnt.sum()) == 0:
        n = len(xi); lp = torch.full((n,), -float('inf'), dtype=xi.dtype, device=xi.device); Es = torch.zeros(n, 6, dtype=xi.dtype, device=xi.device)
        Hp = torch.zeros(n, 6, 6, dtype=xi.dtype, device=xi.device); fin = torch.zeros(n, dtype=torch.bool, device=xi.device)
    else:
        q, j = kde.grid.pairs(st, cnt); lp, Es, M2, fin = kde.stats_csr(xi, ci, q, j); Hp = M2-Es[:, :, None]*Es[:, None, :]
    u = torch.cat([xi, ci], 1); lb = bg.log(u); gb = -(u-bg.mean)@bg.P
    L = lp-lb; z = l1r+L-lr
    sig = torch.where(fin, torch.sigmoid(torch.where(fin, z, torch.zeros_like(z))), torch.zeros_like(z))
    ll = torch.where(fin, torch.logaddexp(torch.full_like(z, lr), torch.where(fin, l1r+L, torch.zeros_like(z))), torch.full_like(z, lr))
    gL = torch.where(fin[:, None], Es-gb, torch.zeros_like(Es))
    H = None
    if need_hessian:
        HL = torch.where(fin[:, None, None], Hp+bg.P, torch.zeros_like(Hp))
        H = sig[:, None, None]*HL+(sig*(1-sig))[:, None, None]*gL[:, :, None]*gL[:, None, :]
    with torch.no_grad():                                                       # certificate: (1-rho) dp / (rho b + (1-rho) p_trunc)
        lpt = torch.where(fin, lp.detach(), torch.full_like(lp, -float('inf')))
        cert = torch.exp(l1r+log_dp-torch.logaddexp(lr+lb.detach(), l1r+lpt))
    return ll, sig[:, None]*gL, H, cert


def choose_kappa(kde, bg, rho, tau, mahal2_domain):
    """kappa with cert <= tau for every query in {b >= b(mahalanobis^2 = mahal2_domain)} even when p_trunc = 0."""
    lb = bg.log_at_mahalanobis2(mahal2_domain)
    k2 = 2*(math.log1p(-rho)+kde.log_cmax-math.log(tau)-math.log(rho)-lb)
    return math.sqrt(max(k2, 1.))


def kernel_sum(xyz, rgb, H_X, H_C, kappa, device='cpu', tile=8192):
    """rho_i = sum_j exp(-1/2 |x_i - x_j|^2_{H_X^-1} - 1/2 |c_i - c_j|^2_{H_C^-1}) with the same certified grid (omitted terms each
    <= exp(-kappa^2/2), at most N of them): returns (rho [N], absolute omitted-sum bound per point)."""
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=device)
    x, c = t(xyz), t(rgb); Px = torch.linalg.inv(t(H_X)); Pc = torch.linalg.inv(t(H_C))
    lam = float(np.linalg.eigvalsh(np.asarray(H_X, float)).max()); g = Grid(x, kappa*math.sqrt(lam)); out = []
    for i in range(0, len(x), tile):
        idx, valid = g.candidates(x[i:i+tile]); dx = x[idx]-x[i:i+tile, None]; dc = c[idx]-c[i:i+tile, None]
        q = ((dx@Px)*dx).sum(-1)+((dc@Pc)*dc).sum(-1)
        out.append(torch.where(valid, torch.exp(-.5*q), torch.zeros_like(q)).sum(1))
    return torch.cat(out), len(x)*math.exp(-.5*kappa*kappa)


def spatial_block_folds(xyz, edge, n_folds=4):
    """Reproducible spatial blocks: voxel of edge `edge`, fold = hash(voxel) mod n_folds (deterministic integer hash)."""
    v = np.floor((xyz-xyz.min(0))/edge).astype(np.int64)
    h = (v[:, 0]*73856093) ^ (v[:, 1]*19349663) ^ (v[:, 2]*83492791)
    return np.mod(h, n_folds)


def calibrate_rho(frame, s2, LC, bg, tau, mahal2_domain, prior_ab=(1., 1.), edge=None, n_folds=4, device='cpu'):
    """Held-out within-frame spatial-block CV (spec 3.2): for each fold fit the KDE on the other folds, then maximize
    sum_q log[(1 - rho) p_train(q) + rho b_0(q)] + (a-1) log rho + (b-1) log(1 - rho) over rho (golden section on logit rho).
    Returns rho (median over folds), per-fold values and the dispersion. NOT a physical visibility label."""
    folds = spatial_block_folds(frame.xyz, edge, n_folds); res = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0: continue
        kde = FrameKDE(frame.xyz[tr], frame.rgb[tr], frame.cov_xyz[tr], s2, LC, device=device)
        kde.build_grid(choose_kappa(kde, bg, .5, tau, mahal2_domain))
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=device)
        lp = []
        for i in range(0, int(te.sum()), 4096):
            xq, cq = t(frame.xyz[te][i:i+4096]), t(frame.rgb[te][i:i+4096]); idx, valid = kde.grid.candidates(xq)
            lp.append(kde.stats(xq, cq, idx, valid)[0] if idx.shape[1] else torch.full((len(xq),), -float('inf'), dtype=torch.float64, device=device))
        lp = torch.cat(lp).cpu().numpy(); lb = bg.log(t(np.c_[frame.xyz[te], frame.rgb[te]])).cpu().numpy()
        a, b = prior_ab
        obj = lambda z: -(np.logaddexp(np.log1p(-1/(1+np.exp(-z)))+lp, -np.log1p(np.exp(-z))+lb).sum()+(a-1)*(-np.log1p(np.exp(-z)))+(b-1)*np.log1p(-1/(1+np.exp(-z))))
        lo, hi = -30., 10.; g = (math.sqrt(5)-1)/2
        for _ in range(100):
            m1, m2 = hi-g*(hi-lo), lo+g*(hi-lo)
            (lo, hi) = (lo, m2) if obj(m1) < obj(m2) else (m1, hi)
        z = (lo+hi)/2; res.append({'fold': f, 'held_out': int(te.sum()), 'rho': float(1/(1+math.exp(-z))), 'at_lower_bound': z < -29})
    r = np.array([x['rho'] for x in res])
    return {'rho': float(np.median(r)), 'folds': res, 'dispersion_iqr': float(np.subtract(*np.percentile(r, [75, 25]))), 'block_edge_m': edge,
            'prior_beta': list(prior_ab), 'semantics': 'held-out sampling/model-mismatch weight, NOT a visibility label'}
