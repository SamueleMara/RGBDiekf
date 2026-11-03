"""Coarse-to-fine basin enlargement (spec B1) for the material-following likelihood. INITIALIZER ONLY.

Level L (m = 2^L) of one camera image is the box average of m x m raw pixels that are all valid foreground with finite
positive depth and lie on one surface (max - min depth within the documented single-surface allowance taken over the
block extent, n_sigma sqrt(2) sigma_D(z) + z tan(theta_max) sqrt(2) m / f). Metric depth stays metric; pixel
coordinates map as u_L = (u + 1/2)/m - 1/2 and d u_L / d x = J_pi / m (camera scaling preserved). Raw sensor errors of
the m^2 averaged pixels are independent, so a coarse pixel carries R/m^2; the coarse sample and the coarse reference
sample (bilinear at the subpixel coarse anchor of p_ref) are independent raw sets of different frames.

The SAME material-following residual is evaluated at every level,
    r_L = [I_t,L(q_L) - I_tau,L(p_ref,L),  D_t,L(q_L) - D_tau,L(p_ref,L) - dZ],  q_L = p_ref,L + (pi(x_t) - pi(x_ref))/m,
and a per-point Gauss-Newton IEKF on the 6x6 predicted marginal of the current state (reference states fixed at their
predicted means, coarse errors marginal) moves the iterate from coarse to fine with the same acceptance rule as level 0
(stable visible identities, valid coarse supports, foreground projection, descent of the marginal NLL). No point is
coupled to another point, no smoothness or motion prior is added, no GT is read. The coarse result is only a starting
iterate: the original-resolution local-correlated IEKF then evaluates the exact material likelihood, keeps the coarse
start for a point only if it does not increase that point's original-resolution objective, and alone defines the
posterior mean and Joseph covariance.
"""
from types import SimpleNamespace
import math
import torch

LEVEL_NOTE = 'coarse levels initialize the original-resolution IEKF only; never part of the posterior'


def build_level(frame, L, rig, noise, policy):
    m = 2**L
    C, H, W = frame.depth.shape
    Hc, Wc = H//m, W//m
    blk = lambda a: a[:, :Hc*m, :Wc*m].reshape(C, Hc, m, Wc, m, *a.shape[3:])
    D = blk(frame.depth.double())
    I = blk(frame.rgb.double())
    ok = blk(frame.mask & frame.raw_valid).all(4).all(2) & torch.isfinite(D).all(4).all(2) & (D > 0).all(4).all(2) & torch.isfinite(I).all(5).all(4).all(2)
    Dm = torch.where(torch.isfinite(D) & (D > 0), D, torch.zeros_like(D))
    z = Dm.mean((2, 4))
    f = rig.K[:, :2].min(1).values.double()[:, None, None]
    allow = (policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(z).clamp_min(0).sqrt()
             + z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)*m/f)
    span = Dm.amax((2, 4))-Dm.amin((2, 4))
    ok &= span <= allow
    rgb = torch.where(ok[..., None], torch.nan_to_num(I).mean((2, 4)), torch.zeros((C, Hc, Wc, 3), dtype=torch.float64, device=D.device))
    depth = torch.where(ok, z, torch.ones_like(z))
    var_depth = noise.depth_sigma2(Dm).mean((2, 4))/(m*m)
    return SimpleNamespace(k=frame.k, L=L, m=m, H=Hc, W=Wc, rgb=rgb, depth=depth, valid=ok, var_depth=var_depth,
                           var_rgb=noise.rgb_variance.double()/(m*m), f=f[:, 0, 0])


def coarse_sample(level, cam, q, noise, policy):
    """Bilinear coarse sample, exact interpolant gradient, sum w^2 var; valid only on a complete one-surface 2x2 support."""
    base = torch.floor(torch.nan_to_num(q))
    inside = torch.isfinite(q).all(-1) & (base[:, 0] >= 0) & (base[:, 1] >= 0) & (base[:, 0]+1 < level.W) & (base[:, 1]+1 < level.H)
    x0 = base[:, 0].long().clamp(0, level.W-2)
    y0 = base[:, 1].long().clamp(0, level.H-2)
    idx = [(y0, x0), (y0, x0+1), (y0+1, x0), (y0+1, x0+1)]
    V = torch.stack([torch.cat((level.rgb[cam, y, x], level.depth[cam, y, x, None]), -1) for y, x in idx], 1)
    ok = inside & torch.stack([level.valid[cam, y, x] for y, x in idx], 1).all(1)
    vd = torch.stack([level.var_depth[cam, y, x] for y, x in idx], 1)
    z = V[:, :, 3]
    zm = z.mean(1)
    allow = (policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(zm).clamp_min(0).sqrt()/level.m
             + zm*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)*level.m/level.f[cam])
    ok &= (z.amax(1)-z.amin(1)) <= allow
    u, v = (q-base).unbind(-1)
    w = torch.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), -1)
    dw = torch.stack((torch.stack((-(1-v), 1-v, -v, v), -1), torch.stack((-(1-u), -u, 1-u, u), -1)), -1)
    val = torch.einsum('ma,mad->md', w, V)
    grad = torch.einsum('mab,mad->mdb', dw, V)
    w2 = w*w
    var = torch.cat((w2.sum(1)[:, None]*level.var_rgb[None], (w2*vd).sum(1)[:, None]), 1)
    return val, grad, var, ok


class CoarseEvaluator:
    def __init__(self, rig, noise, policy, refs, current_levels, reference_levels, warp, full_width):
        self.rig, self.noise, self.policy, self.refs = rig, noise, policy, refs
        self.cur, self.ref, self.warp, self.full_width = current_levels, reference_levels, warp, int(full_width)

    def rows(self, s, identities, L):
        c, p, vis = identities
        c, p = c[vis], p[vis]
        m = 2**L
        ut, zt, Jt, Jzt = self.rig.project_rows(s[p, :3], c)
        rs = self.refs.state[c, p]
        ur, zr, _, _ = self.rig.project_rows(rs[:, :3], c)
        rpix = self.refs.pixel[c, p]
        full_w = self.full_width
        pref = torch.stack((rpix % full_w, rpix//full_w), -1).double()
        pref_L = (pref+.5)/m-.5
        qL = pref_L+(ut-ur)/m
        cur, grad, vcur, okc = coarse_sample(self.cur[L], c, qL, self.noise, self.policy)
        ref = torch.zeros_like(cur)
        vref = torch.zeros_like(vcur)
        okr = torch.zeros_like(okc)
        rf = self.refs.frame[c, p]
        for k in rf.unique().tolist():
            ix = torch.where(rf == k)[0]
            ref[ix], _, vref[ix], okr[ix] = coarse_sample(self.ref[int(k)][L], c[ix], pref_L[ix], self.noise, self.policy)
        e = cur-ref
        e[:, 3] -= zt-zr
        Hx = grad@(Jt/m)
        Hx[:, 3] -= Jzt
        R = torch.diag_embed(vcur+vref)+self.warp
        return SimpleNamespace(c=c, p=p, e=e, H=Hx, R=R, valid=okc & okr, shift=qL-pref_L)

    def costs(self, rows, C, N):
        cost = torch.zeros((C, N), dtype=torch.float64, device=rows.e.device)
        valid = torch.zeros((C, N), dtype=torch.bool, device=rows.e.device)
        g = torch.where(rows.valid)[0]
        if len(g):
            L = torch.linalg.cholesky(rows.R[g])
            e = rows.e[g]
            w = torch.cholesky_solve(e.unsqueeze(-1), L).squeeze(-1)
            cost[rows.c[g], rows.p[g]] = ((e*w).sum(1)+2*torch.log(L.diagonal(dim1=1, dim2=2)).sum(1))/2
            valid[rows.c[g], rows.p[g]] = True
        return cost, valid


def coarse_to_fine(evaluator, build, mean, P, levels, C, iters=5, backtracks=8):
    """Per-point fixed-prior Gauss-Newton IEKF on the 6x6 predicted marginal, coarse to fine (levels descending, > 0).
    Returns the starting iterate for the original-resolution IEKF and a level-by-level report."""
    N, dev = len(mean), mean.device
    Pinv = torch.linalg.inv(P)
    s = mean.clone()
    report = []
    Ex = torch.zeros((3, 6), dtype=torch.float64, device=dev)
    Ex[:, :3] = torch.eye(3, dtype=torch.float64, device=dev)

    def prior(x):
        d = x-mean
        return torch.einsum('ni,nij,nj->n', d, Pinv, d)/2

    def state(x, L):
        rows, owners, reason, life = build(x)
        ident = rows.identities
        cr = evaluator.rows(x, ident, L)
        cost, valid = evaluator.costs(cr, C, N)
        # a visible identity lost at the trial (background projection, occlusion or invalid coarse support) rejects the
        # coarse step: the initializer never moves a point by deleting its own observations
        return {'rows': cr, 'cost': cost, 'valid': valid, 'prior': prior(x)}
    for L in levels:
        moved = torch.zeros(N, dtype=torch.bool, device=dev)
        level_report = {'level': L, 'scale': 2**L, 'iterations': []}
        for it in range(iters):
            base = state(s, L)
            cr = base['rows']
            g = torch.where(cr.valid)[0]
            A = Pinv.clone()
            b = torch.zeros((N, 6), dtype=torch.float64, device=dev)
            if len(g):
                Hg = cr.H[g]
                RinvH = torch.linalg.solve(cr.R[g], Hg)
                info = Hg.transpose(1, 2)@RinvH
                A[:, :3, :3].index_add_(0, cr.p[g], info)
                rhs = -cr.e[g]+(Hg@(s[cr.p[g], :3]-mean[cr.p[g], :3]).unsqueeze(-1)).squeeze(-1)
                b[:, :3].index_add_(0, cr.p[g], (RinvH.transpose(1, 2)@rhs.unsqueeze(-1)).squeeze(-1))
            observed = torch.zeros(N, dtype=torch.bool, device=dev)
            observed[cr.p[g]] = True
            if not bool(observed.any()):
                break
            prop = mean+torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
            pending = observed.clone()
            accepted = torch.zeros_like(pending)
            base_s = s.clone()
            for ls in range(backtracks):
                rate = 2.**-ls
                trial = s.clone()
                trial[pending] = base_s[pending]+rate*(prop[pending]-base_s[pending])
                tr = state(trial, L)
                common = base['valid'] & tr['valid']
                lost = base['valid'] & ~tr['valid']
                jb = base['prior']+(base['cost']*common).sum(0)
                jt = tr['prior']+(tr['cost']*common).sum(0)
                good = pending & ~lost.any(0) & common.any(0) & (jt <= jb+1e-12*jb.abs().clamp_min(1.))
                s[good] = trial[good]
                accepted |= good
                pending &= ~good
                if not bool(pending.any()):
                    break
            moved |= accepted
            step = (s-base_s)[:, :3].norm(dim=1)
            level_report['iterations'].append({'iteration': it, 'observed_points': int(observed.sum()), 'accepted_points': int(accepted.sum()),
                                               'coarse_rows': int(len(g)), 'accepted_step_m_median': float(step[accepted].median()) if bool(accepted.any()) else None})
            if not bool(accepted.any()):
                break
        cr = state(s, L)['rows']
        g = torch.where(cr.valid)[0]
        level_report.update({'moved_points': int(moved.sum()), 'coarse_rows_final': int(len(g)),
                             'coarse_pixel_shift_level_px_quantiles': torch.quantile(cr.shift[g].norm(dim=1), torch.tensor([.5, .9, .99], dtype=torch.float64, device=dev)).tolist() if len(g) else None,
                             'displacement_from_prediction_m_quantiles': torch.quantile((s-mean)[:, :3].norm(dim=1), torch.tensor([.5, .9, .99], dtype=torch.float64, device=dev)).tolist()})
        report.append(level_report)
    return s, report
