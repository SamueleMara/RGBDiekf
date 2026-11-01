"""Phase-1 temporal solver for independent free 6D point trajectories (RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md
P1.1-P1.5): forward inner IEKF warm start with a frozen-support line search, RTS warm start, outer full-trajectory iterated
extended Kalman smoother (Gauss-Newton on the ORIGINAL posterior), Laplace covariance with cross-time blocks.

Posterior of point i (fixed covariances, fixed visibility / measurement support O):
    J(S) = 1/2 |s_0 - mu_0|^2_{P_0^-1} + 1/2 sum_k |s_{k+1} - F_k s_k|^2_{Q_k^-1} + 1/2 sum_{k, c in O_k} |h_{c,k}(x_k)|^2_{R^-1}
      [+ 1/2 |a - a_obs|^2_{Sigma_a^-1} with the canonical colour a a latent variable shared by every RGB row of the point]
h = [I(pi(x)) - a ; D(pi(x)) - Z(x)] (estimator/projective_terms.py, r = -h), H = [H_x, 0].
mu_0 = [x_ref, 0], P_0 = blockdiag(P_x0 back-projection covariance, P_v0 declared). The frame-0 SOURCE camera is not a
measurement (its depth / colour already define x_ref, P_x0 and a: spec P1.1.4 / G1.5); other frame-0 cameras are.

Support (O): admitted rows at a state = global projected-cloud visibility over ALL points (projective_visibility, never per
chunk) + valid 2x2 foreground / depth / same-surface sample (projective_sensor) + the production row gate
(projective_terms, policy row_innovation_gate) with the gate covariance stated in the manifest. A support is FROZEN for
every objective comparison (line search); it is refreshed only between accepted steps / outer iterations, and support
changes are counted, never used to make a step look better. A candidate whose frozen rows are not all sampleable is not
evaluable and is rejected (no objective "improvement" by losing rows).

Globalization (spec P1.2.6, P1.4.4): GN direction from the local linearization; backtracking alpha = 1, 1/2, ... on the TRUE
nonlinear objective (actual interpolated RGB-D at the candidate) with the frozen support; a candidate is accepted only if
J(candidate) < J(current) - 64 eps |J(current)| (floating-point precision, no chi-square acceptance). At a pixel-cell border
(a used row at fractional pixel < border_px from an integer) the bilinear gradient is one-sided: if no step decreases J,
the direction is recomputed from the linearization a distance eta along the step (the other side's interpolant) and the
backtracking is repeated - the actual residual is tested on both sides, no Jacobian is treated as globally valid. If both
fail the point keeps its state and is reported (LINE_SEARCH_STALLED / NUMERICALLY_FAILED), never as converged.
Stationarity: Newton decrement delta^T A delta < tol (policy iekf_step_tolerance) under the CURRENT support.
"""
import math
import time
import numpy as np
import torch
from estimator.projective_sensor import CODE, REASONS, MeasuredFrame
from estimator.projective_terms import build_terms
from estimator.projective_visibility import visibility, position_sigma_z2
from estimator import block_tridiagonal as BT
from estimator.rts_point_smoother import rts_smooth

SOURCE_EXCLUDED = 11
TEMPORAL_REASONS = {**REASONS, SOURCE_EXCLUDED: 'FRAME0_SOURCE_DUPLICATE_EXCLUDED'}
SUPPORT_LOSS = (CODE['OUTSIDE_IMAGE_FOOTPRINT'], CODE['MASK_EXCLUDED'], CODE['INVALID_DEPTH'], CODE['DEPTH_DISCONTINUITY'], CODE['BEHIND_CAMERA'])
FORWARD_STATUS = {0: 'NOT_PROCESSED', 1: 'OBSERVED_CONVERGED', 2: 'OBSERVED_NOT_CONVERGED', 3: 'PREDICTION_ONLY', 4: 'MEASUREMENT_LOST',
                  5: 'NUMERICALLY_FAILED'}
FRAME_STATE = {0: 'NOT_PROCESSED', 1: 'OBSERVED', 2: 'PREDICTION_ONLY', 3: 'MEASUREMENT_LOST', 4: 'NUMERICALLY_FAILED'}
TRAJ_STATUS = {0: 'CONVERGED', 1: 'LINE_SEARCH_STALLED', 2: 'ITERATION_CAP', 3: 'NUMERICALLY_FAILED_WARM_START_KEPT', 4: 'NO_OBSERVATIONS',
               5: 'SUPPORT_OSCILLATION'}
EPS = float(np.finfo(np.float64).eps)


class SourceDuplicateError(AssertionError):
    pass


class FrameStore:
    """Measured frames loaded once (MeasuredFrame). The IEKS revisits every frame once per factor pass and once per
    line-search trial, so the frames stay resident on the device when they fit `device_budget_bytes`; otherwise they stay in
    pinned host memory and move per use. Identical tensors either way (no effect on results)."""
    def __init__(self, dataset, frames, device, device_budget_bytes=3.0e9):
        self.device = torch.device(device)
        self.cpu = {int(k): MeasuredFrame(dataset, int(k), 'cpu') for k in frames}
        self.bytes = sum(f.bytes for f in self.cpu.values())
        self.resident = self.device.type == 'cuda' and self.bytes <= device_budget_bytes
        self.dev = {}
        if self.device.type == 'cuda':
            for k, f in self.cpu.items():
                g = object.__new__(MeasuredFrame); g.__dict__.update(f.__dict__)
                for a in ('rgb', 'depth', 'mask', 'raw_valid', 'valid'):
                    t = getattr(f, a)
                    setattr(g, a, t.to(self.device) if self.resident else t.pin_memory())
                (self.dev if self.resident else self.cpu)[k] = g

    def get(self, k):
        k = int(k)
        if self.resident: return self.dev[k]
        f = self.cpu[k]
        if self.device.type == 'cpu': return f
        g = object.__new__(MeasuredFrame); g.__dict__.update(f.__dict__)
        for a in ('rgb', 'depth', 'mask', 'raw_valid', 'valid'): setattr(g, a, getattr(f, a).to(self.device, non_blocking=True))
        return g


class TemporalModel:
    """Everything that defines the measurement and the numerics (fixed during one solve)."""
    def __init__(self, rig, noise, policy, tcfg, device, dtype=torch.float64, chunk=200_000):
        self.rig, self.noise, self.device, self.dtype, self.chunk = rig, noise, torch.device(device), dtype, chunk
        self.policy = {**policy, 'depth_interpolation': tcfg['depth_interpolation']}
        self.policy_ungated = {**self.policy, 'row_innovation_gate': 'none'}
        self.nsig = float(policy['visibility_n_sigma'])
        self.tol = float(tcfg.get('newton_decrement_tolerance', policy['iekf_step_tolerance']))
        self.max_inner, self.max_outer = int(tcfg['max_inner_iterations']), int(tcfg['max_outer_iterations'])
        self.halvings = int(tcfg['line_search_halvings'])
        self.precision_factor = float(tcfg['objective_decrease_precision_factor'])
        self.eta, self.border_px = float(tcfg['knife_edge_shift_m']), float(tcfg['cell_border_px'])
        self.latent = bool(tcfg['latent_colour'])
        self.sigma_a2 = noise.rgb_variance.to(dtype)                     # source-pixel colour noise (one draw per point)

    def improves(self, Jnew, Jold):
        return Jnew < Jold-self.precision_factor*EPS*(Jold.abs()+1)


def _zeros(N, dev, dt, *shape):
    return torch.zeros((N, *shape), dtype=dt, device=dev)


def evaluate_frame(M, frame, X, a, Sigma, pts=None, support=None, src=None, audit=None, X_lin=None):
    """Rows of one frame. X [N,3] positions of ALL points (visibility is global), a [N,3] canonical colours, Sigma [N,3,3]
    position covariance for visibility and the row gate. support None: admission at X (src [N] excludes the frame-0 source
    camera); support [C,N] bool: FROZEN rows, ungated, no visibility, valid[p] False if any frozen row of p is not
    sampleable at X. X_lin [N,3] (frozen mode): the linearization state whose row covariances R are used - the objective
    is defined for FIXED covariance and support (spec section 4); the interpolation variance sum_j w_j^2 sigma_j^2 depends
    on the sub-pixel position and is therefore frozen with the support, never re-evaluated at a candidate.
    pts [N] bool restricts the evaluated points. Returns per-point sums (H^T W H, H^T W r, r^T W r, rows)."""
    N, C = len(X), M.rig.count
    dev, dt = X.device, M.dtype
    out = {'Ixx': _zeros(N, dev, dt, 3, 3), 'gx': _zeros(N, dev, dt, 3), 'q': _zeros(N, dev, dt), 'nrows': torch.zeros(N, dtype=torch.long, device=dev),
           'valid': torch.ones(N, dtype=torch.bool, device=dev), 'border': torch.zeros(N, dtype=torch.bool, device=dev)}
    if M.latent: out.update(Ixa=_zeros(N, dev, dt, 3, 3), Iaa=_zeros(N, dev, dt, 3), ga=_zeros(N, dev, dt, 3))
    ar = torch.arange(N, device=dev)
    if support is None:
        proj = M.rig.project(X, jacobian=False)
        vreason, _ = visibility(proj, position_sigma_z2(M.rig.R.to(dt), Sigma), frame.H, frame.W, M.rig.near, M.nsig)
        reason = vreason.clone()
        cand = vreason.T == CODE['VALID']
        if src is not None:
            cand[ar, src] = False; reason[src, ar] = SOURCE_EXCLUDED
        policy = M.policy
        out['proj'] = proj
    else:
        cand = support.T.clone(); reason = None; policy = M.policy_ungated
    if pts is not None: cand &= pts[:, None]
    rows = torch.nonzero(cand)
    supp = torch.zeros((C, N), dtype=torch.bool, device=dev)
    if len(rows):
        pr = rows[:, 0]
        bounds = torch.searchsorted(pr, torch.arange(0, N+M.chunk, M.chunk, device=dev).clamp_max(N)).tolist()
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo: continue
            p, c = pr[lo:hi], rows[lo:hi, 1]
            t = build_terms(M.rig, frame, c, p, X[p], a[p], Sigma[p], M.noise, policy, dt, None)
            ok = t.valid()
            if support is not None and X_lin is not None:                 # R of the linearization state (fixed covariance)
                tl = build_terms(M.rig, frame, c, p, X_lin[p], a[p], Sigma[p], M.noise, policy, dt, None)
                t.var = torch.where(tl.valid()[:, None], tl.var, t.var)
            if support is None:
                reason[c, p] = t.reason; supp[c[ok], p[ok]] = True
            else:
                out['valid'][p[~ok]] = False
            w = torch.where(ok[:, None], 1/t.var, torch.zeros_like(t.var)); r = torch.where(ok[:, None], t.r, torch.zeros_like(t.r))
            Hx = torch.where(ok[:, None, None], t.Hx, torch.zeros_like(t.Hx))
            up, lens = torch.unique_consecutive(p, return_counts=True)
            seg = lambda v: torch.segment_reduce(v, 'sum', lengths=lens)
            out['Ixx'][up] += seg((Hx[:, :, :, None]*w[:, :, None, None]*Hx[:, :, None, :]).sum(1))
            out['gx'][up] += seg((Hx*(w*r)[:, :, None]).sum(1)); out['q'][up] += seg((w*r*r).sum(1))
            out['nrows'][up] += seg(ok.to(dt)).round().long()
            fr = t.uv-torch.floor(t.uv)
            bd = ok & (torch.minimum(fr, 1-fr).min(1).values < M.border_px)
            out['border'][up] |= seg(bd.to(dt)) > 0
            if M.latent:                                                # dh_rgb/da = -I
                out['Ixa'][up] += seg(-(Hx[:, :3, :]*w[:, :3, None]).transpose(1, 2))
                out['Iaa'][up] += seg(w[:, :3]); out['ga'][up] += seg(-w[:, :3]*r[:, :3])
            if audit is not None: audit(t, ok, p, c)
    out['support'] = supp; out['reason'] = reason
    return out


def assert_single_counted(support0, src):
    """Spec G1.5: the frame-0 source camera must never be a measurement row of its own point."""
    N = support0.shape[1]
    if bool(support0[src, torch.arange(N, device=support0.device)].any()):
        raise SourceDuplicateError('frame-0 source camera row present: the source depth / colour would be counted twice')


# ----------------------------------------------------------------------------------------------- inner forward IEKF
def _info_step(Lam, ev, e):
    A = Lam.clone(); A[:, :3, :3] += ev['Ixx']
    b = -(Lam@e[..., None])[..., 0]; b[:, :3] += ev['gx']
    L, info = torch.linalg.cholesky_ex((A+A.transpose(1, 2))/2)
    if bool((info != 0).any()): raise BT.BlockSolveFailure('frame information matrix not SPD')
    d = torch.cholesky_solve(b[..., None], L)[..., 0]
    return d, (d*b).sum(-1), L


def _frame_objective(Lam, e, q):
    return .5*(e*(Lam@e[..., None])[..., 0]).sum(-1)+.5*q


def _frame_line_search(M, frame, x, d, go, support, s_m, Lam, Jc, a, Sig):
    alpha = torch.ones(len(x), dtype=x.dtype, device=x.device)
    pending, acc, newx = go.clone(), torch.zeros_like(go), x.clone()
    halv = torch.zeros(len(x), dtype=torch.long, device=x.device)
    for _ in range(M.halvings+1):
        if not bool(pending.any()): break
        cand = torch.where(pending[:, None], x+alpha[:, None]*d, x)
        ev = evaluate_frame(M, frame, cand[:, :3], a, Sig, pts=pending, support=support, X_lin=x[:, :3])
        J = _frame_objective(Lam, cand-s_m, ev['q'])
        ok = pending & ev['valid'] & M.improves(J, Jc)
        newx = torch.where(ok[:, None], cand, newx); acc |= ok; pending &= ~ok
        alpha = torch.where(pending, alpha/2, alpha); halv += pending.long()
    return acc, newx, halv


def forward_frame(M, frame, s_m, P_m, a, src=None, prev_rows=None, trace_ids=None):
    """Fixed-prior IEKF of one frame for every point (spec P1.2). Returns s+, P+, status, final support, diagnostics."""
    N = len(s_m)
    L0, info = torch.linalg.cholesky_ex(P_m)
    if bool((info != 0).any()): raise BT.BlockSolveFailure('prior covariance not SPD')
    Lam = torch.cholesky_inverse(L0); Lam = (Lam+Lam.transpose(1, 2))/2
    Sig = P_m[:, :3, :3]
    x = s_m.clone()
    ev = evaluate_frame(M, frame, x[:, :3], a, Sig, src=src)
    accepted_any = torch.zeros(N, dtype=torch.bool, device=x.device)
    stats = {'iterations': 0, 'accepted_steps': 0, 'line_search_halvings': 0, 'border_retries': 0, 'border_retry_accepted': 0, 'support_refreshes': 0}
    for j in range(M.max_inner):
        has = ev['nrows'] > 0
        d, dec, _ = _info_step(Lam, ev, x-s_m)
        go = has & (dec >= M.tol)
        if not bool(go.any()): break
        stats['iterations'] = j+1
        Jc = _frame_objective(Lam, x-s_m, ev['q'])
        acc, newx, halv = _frame_line_search(M, frame, x, d, go, ev['support'], s_m, Lam, Jc, a, Sig)
        stats['line_search_halvings'] += int(halv.sum())
        retry = go & ~acc & ev['border']
        if bool(retry.any()):
            stats['border_retries'] += int(retry.sum())
            nd = d[:, :3].norm(dim=1, keepdim=True).clamp_min(1e-300)
            xs = torch.where(retry[:, None], x+M.eta*d/nd, x)
            evs = evaluate_frame(M, frame, xs[:, :3], a, Sig, pts=retry, support=ev['support'])
            retry &= evs['valid']
            d2, _, _ = _info_step(Lam, evs, x-s_m)                     # other side's interpolant, same prior gradient at x
            acc2, newx2, _ = _frame_line_search(M, frame, x, d2, retry, ev['support'], s_m, Lam, Jc, a, Sig)
            newx = torch.where(acc2[:, None], newx2, newx); acc |= acc2
            stats['border_retry_accepted'] += int(acc2.sum())
        if not bool(acc.any()): break
        stats['accepted_steps'] += int(acc.sum())
        accepted_any |= acc
        x = torch.where(acc[:, None], newx, x)
        ev = evaluate_frame(M, frame, x[:, :3], a, Sig, src=src); stats['support_refreshes'] += 1
    has = ev['nrows'] > 0
    d, dec, _ = _info_step(Lam, ev, x-s_m)
    A = Lam.clone(); A[:, :3, :3] += ev['Ixx']
    La, _ = torch.linalg.cholesky_ex((A+A.transpose(1, 2))/2)
    P_post = torch.cholesky_inverse(La); P_post = (P_post+P_post.transpose(1, 2))/2
    conv = has & (dec < M.tol)
    failed = has & ~accepted_any & ~conv
    lost_moved = ~has & accepted_any                                   # moved off its support: no information at x
    keep = has & ~failed
    s_p = torch.where(keep[:, None], x, s_m); P_p = torch.where(keep[:, None, None], P_post, P_m)
    status = torch.full((N,), 3, dtype=torch.uint8, device=x.device)
    status[conv] = 1; status[keep & ~conv] = 2; status[failed] = 5
    loss = (ev['reason'] == torch.as_tensor(SUPPORT_LOSS, device=x.device)[:, None, None]).any(0).any(0)
    if prev_rows is not None: status[~has & (prev_rows > 0) & loss] = 4
    status[lost_moved] = 4
    return s_p, P_p, status, ev, {**stats, 'converged': int(conv.sum()), 'not_converged': int((keep & ~conv).sum()), 'numerically_failed': int(failed.sum()),
                                  'lost_after_moving': int(lost_moved.sum()), 'points_with_rows': int(has.sum())}


# ------------------------------------------------------------------------------------------ outer trajectory IEKS
class TrajectoryProblem:
    """Fixed parts of the full-trajectory posterior: prior, dynamics blocks, processing order."""
    def __init__(self, mean0, cov0, F, Q, a_obs, sigma_a2, latent):
        self.mean0, self.cov0 = mean0, cov0                              # [N,6], [N,6,6] (host, float64)
        L, info = torch.linalg.cholesky_ex(cov0)
        if bool((info != 0).any()): raise BT.BlockSolveFailure('P0 not SPD')
        self.P0inv = torch.cholesky_inverse(L); self.P0inv = (self.P0inv+self.P0inv.transpose(1, 2))/2
        self.F, self.Q = F, Q                                            # [K-1,6,6] host float64
        self.K = len(F)+1
        self.reduced = bool((Q == 0).all())                              # Q_a = 0: deterministic dynamics (exact reduced problem)
        if not self.reduced:
            Lq, info = torch.linalg.cholesky_ex(Q)
            if bool((info != 0).any()): raise BT.BlockSolveFailure('Q_k not SPD (Q_a must be SPD or exactly 0)')
            Qi = torch.cholesky_inverse(Lq); self.Qi = (Qi+Qi.transpose(1, 2))/2
            self.DFF = F.transpose(1, 2)@self.Qi@F; self.Bdyn = -F.transpose(1, 2)@self.Qi
        Phi = [torch.eye(6, dtype=F.dtype)]
        for k in range(self.K-1): Phi.append(F[k]@Phi[-1])
        self.Phi = torch.stack(Phi)
        self.a_obs, self.sigma_a2, self.latent = a_obs, sigma_a2, latent

    def prior_dyn(self, S, idx, dev):
        """Objective and gradient of the prior + dynamics terms for points idx: S [K,n,6] (device)."""
        mean0, P0inv = self.mean0[idx].to(dev), self.P0inv[idx].to(dev)
        e0 = S[0]-mean0
        J = .5*(e0*(P0inv@e0[..., None])[..., 0]).sum(-1)
        G = torch.zeros_like(S); G[0] = (P0inv@e0[..., None])[..., 0]
        if not self.reduced and self.K > 1:
            F, Qi = self.F.to(dev), self.Qi.to(dev)
            e = S[1:]-(F[:, None]@S[:-1][..., None])[..., 0]               # [K-1,n,6]
            w = (Qi[:, None]@e[..., None])[..., 0]
            J = J+.5*(e*w).sum((0, 2))
            G[1:] += w; G[:-1] -= (F.transpose(1, 2)[:, None]@w[..., None])[..., 0]
        return J, G

    def latent_prior(self, a, idx, dev):
        da = a-self.a_obs[idx].to(dev); s2 = self.sigma_a2.to(dev)
        return .5*(da*da/s2).sum(-1), da/s2


def solve_chunk(P, S, a, fac, idx, dev, covariance=False):
    """Gauss-Newton step of the original posterior for points idx. S [K,n,6], a [n,3] on dev; fac: host factor arrays.
    Returns dict(dT [K,n,6], da, dec, J, grad_norm, cov blocks if covariance)."""
    K, n = S.shape[:2]
    t = lambda name: fac[name][:, idx].to(dev)
    Ixx, gx, q = t('Ixx'), t('gx'), t('q')
    J, G = P.prior_dyn(S, idx, dev)
    J = J+.5*q.sum(0)
    G[..., :3] -= gx
    out = {}
    if P.latent:
        Ja, Ga = P.latent_prior(a, idx, dev); J = J+Ja
        Ga = Ga-t('ga').sum(0)
        Aaa = torch.diag_embed(1/P.sigma_a2.to(dev)+t('Iaa').sum(0))
        Cx = t('Ixa').permute(1, 0, 2, 3)                                 # [n,K,3,3]  (x rows, a cols)
    if P.reduced:                                                         # Q_a = 0: s_k = Phi_k s_0, exact
        Phi = P.Phi.to(dev)
        Hs = torch.zeros((K, n, 6, 6), dtype=S.dtype, device=dev); Hs[..., :3, :3] = Ixx
        A0 = P.P0inv[idx].to(dev)+(Phi.transpose(1, 2)[:, None]@Hs@Phi[:, None]).sum(0)
        g0 = (Phi.transpose(1, 2)[:, None]@G[..., None])[..., 0].sum(0)
        if P.latent:
            C0 = (Phi.transpose(1, 2)[:, None, :, :3]@t('Ixa')).sum(0)    # [n,6,3]
            Afull = torch.cat([torch.cat([A0, C0], 2), torch.cat([C0.transpose(1, 2), Aaa], 2)], 1)
            gfull = torch.cat([g0, Ga], 1)
            Lf, _ = BT._chol(Afull, 'reduced latent information')
            dfull = -torch.cholesky_solve(gfull[..., None], Lf)[..., 0]
            d0, da = dfull[:, :6], dfull[:, 6:]; dec = -(gfull*dfull).sum(-1)
            Sfull = torch.cholesky_inverse(Lf) if covariance else None
            S00 = Sfull[:, :6, :6] if covariance else None
        else:
            d0, S00 = BT.reduced_solve(None, A0, -g0); da = None; dec = -(g0*d0).sum(-1)
            if not covariance: S00 = None
        dT = (Phi[:, None]@d0[None, ..., None])[..., 0]
        out.update(dT=dT, da=da, dec=dec, J=J, grad_norm=G.norm(dim=(0, 2)))
        if covariance:
            diag = Phi[:, None]@S00[None]@Phi.transpose(1, 2)[:, None]
            out.update(diag=diag, lag1=Phi[1:, None]@S00[None]@Phi[:-1].transpose(1, 2)[:, None],
                       cross0=(S00[None]@Phi.transpose(1, 2)[:, None]), cov_a=Sfull[:, 6:, 6:] if P.latent else None)
        return out
    A = torch.zeros((n, K, 6, 6), dtype=S.dtype, device=dev)
    A[:, :, :3, :3] = Ixx.permute(1, 0, 2, 3)
    A[:, 0] += P.P0inv[idx].to(dev)
    if K > 1:
        A[:, :-1] += P.DFF.to(dev)[None]; A[:, 1:] += P.Qi.to(dev)[None]
        B = P.Bdyn.to(dev)[None].expand(n, K-1, 6, 6).contiguous()
    else:
        B = torch.zeros((n, 0, 6, 6), dtype=S.dtype, device=dev)
    fac_ = BT.factor(A, B)
    Gp = G.permute(1, 0, 2)                                               # [n,K,6]
    if P.latent:
        C = torch.zeros((n, K, 6, 3), dtype=S.dtype, device=dev); C[:, :, :3, :] = Cx
        X = BT.solve(fac_, torch.cat([-Gp[..., None], C], -1))
        X0, XC = X[..., 0], X[..., 1:]
        Sch = Aaa-(C.transpose(-1, -2)@XC).sum(1)
        Ls, _ = BT._chol(Sch, 'latent-colour Schur complement')
        rhs = -Ga-(C.transpose(-1, -2)@X0[..., None])[..., 0].sum(1)
        da = torch.cholesky_solve(rhs[..., None], Ls)[..., 0]
        dTp = X0-(XC@da[:, None, :, None])[..., 0]
        dec = -(Gp*dTp).sum((1, 2))-(Ga*da).sum(-1)
    else:
        dTp = BT.solve(fac_, -Gp); da = None; dec = -(Gp*dTp).sum((1, 2))
    out.update(dT=dTp.permute(1, 0, 2), da=da, dec=dec, J=J, grad_norm=G.norm(dim=(0, 2)))
    if covariance:
        si = BT.selected_inverse(fac_, want_cross0=True)
        diag, lag, cross = si['diag'], si['lag1'], si['cross0']
        cov_a = None
        if P.latent:                                                      # + U S^-1 U^T, U = A_TT^-1 C (rank-3 correction)
            Sinv = torch.cholesky_inverse(Ls); U = XC
            diag = diag+U@Sinv[:, None]@U.transpose(-1, -2)
            lag = lag+U[:, 1:]@Sinv[:, None]@U[:, :-1].transpose(-1, -2)
            cross = cross+U[:, :1]@Sinv[:, None]@U.transpose(-1, -2)
            cov_a = Sinv
        out.update(diag=diag.permute(1, 0, 2, 3), lag1=lag.permute(1, 0, 2, 3), cross0=cross.permute(1, 0, 2, 3), cov_a=cov_a)
    return out


def point_chunks(N, K, budget_bytes=1.2e9):
    n = max(256, int(budget_bytes/(K*36*8*12)))
    return [slice(i, min(i+n, N)) for i in range(0, N, n)]
