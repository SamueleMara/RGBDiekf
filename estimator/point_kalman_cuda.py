"""Free Phase 1: independent 6D [x, v] IEKF per original endpoint point (contract section 3), batched on the GPU.

Prediction (signed physical interval dt, tau = |dt|, white acceleration density q_c [3x3], m^2/s^3):
    F = [[I, dt I], [0, I]],   Q = [[tau^3/3 q_c, sgn(dt) tau^2/2 q_c], [sgn(dt) tau^2/2 q_c, tau q_c]]
    s- = F s+,  P- = F P+ F^T + Q
(no position random walk: the contract's Q is white acceleration only.)

Update, fixed prior (s-, P-) through all iterations; at iterate s_j every valid camera row (projective_terms) gives
r_c = -h_c(s_j), H_c = [H_x,c, 0], R_c = diag(var). Per point (independently, 6x6):
    A_j = Lambda- + sum_c H_c^T R_c^-1 H_c,      Lambda- = (P-)^-1
    b_j = sum_c H_c^T R_c^-1 r_c - Lambda- (s_j - s-)
    delta_j = A_j^-1 b_j,   s_{j+1} = s_j + delta_j
which equals s_{j+1} = s- + K_j [r_j + H_j (s_j - s-)], K_j = P- H^T (H P- H^T + R)^-1 (tests/test_point_kalman_cuda.py).
Per-point stopping: a point stops once delta^T A delta < tol (squared step in the information metric) and keeps that
iterate; the loop ends when every point stopped or at max_iters. Points that never met it are counted and reported
(piecewise-bilinear image fields make the plain IEKF oscillate between pixel cells near the solution; no line search
or damping is added: that would be an optimizer, not the Kalman update).
Posterior ONCE at the final iterate's own linearization: P+ = (Lambda- + H^T R^-1 H)^-1, the Joseph-equivalent form
(verified against (I-KH)P-(I-KH)^T + K R K^T in float64). A point with no valid row at the final linearization keeps
s+ = s-, P+ = P- (no unsupported shifted state).
Pixel-reuse assignment (contract 2.2 item 5: front-most point per pixel per camera): 'per-frame' (default) decides it
once at the prior prediction s- and keeps it through the iterations, while samples, validity and Jacobians are re-
evaluated at every iterate; 'per-iteration' re-decides it at every iterate (the contract's pseudocode placement).
Measured on the ray-cast fixture (9300 textured points, 2.5 px prior offset): per-iteration 6.8 mm median error, 3815
points not converged; per-frame 0.68 mm, 1136 (reports/projective_four_pass_cuda_implementation.md).
Reductions over a point's rows are deterministic segment sums (rows ordered by point, then camera); no atomics.
Supernodes / groups do not exist here: --point-chunk-size only bounds memory, every point keeps its own update.
"""
import math
import time
import numpy as np
import torch
from estimator.projective_sensor import CODE, REASONS
from estimator.projective_terms import build_terms, h_value
from estimator.projective_visibility import visibility, position_sigma_z2, apply_frozen_assignment


def transition(dt, dtype=torch.float64, device='cpu'):
    F = torch.eye(6, dtype=dtype, device=device); F[:3, 3:] = dt*torch.eye(3, dtype=dtype, device=device)
    return F


def process_noise(dt, qc):
    qc = torch.as_tensor(qc)
    tau, sg = abs(dt), math.copysign(1., dt)
    if not (math.isfinite(dt) and tau > 0): raise ValueError('process noise needs a non-zero finite physical interval')
    return torch.cat([torch.cat([tau**3/3*qc, sg*tau**2/2*qc], 1), torch.cat([sg*tau**2/2*qc, tau*qc], 1)], 0)


def predict(s, P, dt, qc):
    F = transition(dt, s.dtype, s.device); Q = process_noise(dt, torch.as_tensor(qc, dtype=s.dtype, device=s.device))
    Pm = F@P@F.T+Q
    return s@F.T, (Pm+Pm.transpose(-1, -2))/2


class CholeskyFailure(RuntimeError):
    pass


def spd_factor(A, what):
    L, info = torch.linalg.cholesky_ex(A)
    if bool((info != 0).any()):
        if A.dtype != torch.float64:
            L64, info64 = torch.linalg.cholesky_ex(A.double())
            if not bool((info64 != 0).any()): return L64, 'float64_retry'
        bad = torch.nonzero(info != 0).flatten()[:20]
        ev = torch.linalg.eigvalsh(A[bad].double()).cpu().numpy()
        raise CholeskyFailure(f'{what}: Cholesky failed for {int((info != 0).sum())} rows, first ids {bad.tolist()}, eigenvalues {ev.tolist()}')
    return L, None


def jacobi_eigvalsh(A, sweeps=None):
    """Ascending eigenvalues of symmetric [N,d,d] batches (small d) by cyclic Jacobi rotations, elementwise on the
    device. cuSOLVER's batched syev needs ~0.5 MB workspace per matrix (measured; unusable for 1e5 points) and the
    closed-form 3x3 formula loses sqrt(eps)*lambda_max on rank-deficient information; Jacobi keeps small eigenvalues
    to ~eps*lambda_max."""
    A = ((A+A.transpose(-1, -2))/2).clone()
    d = A.shape[-1]
    sweeps = sweeps or (8 if d <= 3 else 12)
    tiny = torch.finfo(A.dtype).tiny
    for _ in range(sweeps):
        for p in range(d-1):
            for q in range(p+1, d):
                apq = A[:, p, q]; app = A[:, p, p]; aqq = A[:, q, q]
                rot = apq.abs() > tiny
                theta = (aqq-app)/(2*torch.where(rot, apq, torch.ones_like(apq)))
                t = torch.sign(theta)/(theta.abs()+torch.sqrt(theta*theta+1)); t = torch.where(theta == 0, torch.ones_like(t), t)
                t = torch.where(rot, t, torch.zeros_like(t))
                c = 1/torch.sqrt(t*t+1); sn = t*c
                Ap, Aq = A[:, :, p].clone(), A[:, :, q].clone()
                A[:, :, p] = c[:, None]*Ap-sn[:, None]*Aq; A[:, :, q] = sn[:, None]*Ap+c[:, None]*Aq
                Ap, Aq = A[:, p, :].clone(), A[:, q, :].clone()
                A[:, p, :] = c[:, None]*Ap-sn[:, None]*Aq; A[:, q, :] = sn[:, None]*Ap+c[:, None]*Aq
    return torch.sort(A.diagonal(dim1=-2, dim2=-1), dim=-1).values


def eigvalsh3(A):
    return jacobi_eigvalsh(A)


def spd_extreme_eigs(P, sample=16384):
    """Positive definiteness of EVERY matrix (Cholesky) and (lambda_min, lambda_max) of a deterministic evenly spaced
    sample of at most `sample` matrices (diagnostic quantiles; Jacobi on 6x6 costs ~1.4 s per 378k matrices)."""
    _, info = torch.linalg.cholesky_ex(P)
    idx = torch.linspace(0, len(P)-1, min(sample, len(P)), device=P.device).round().long().unique()
    ev = jacobi_eigvalsh(P[idx])
    return info == 0, ev[:, 0], ev[:, -1]


def segment_sum(values, lengths):
    return torch.segment_reduce(values, 'sum', lengths=lengths)


class Timers:
    """CUDA events per named stage, resolved once per frame (one host sync at the diagnostic boundary)."""
    def __init__(self, device):
        self.cuda = torch.device(device).type == 'cuda'; self.ev = []; self.wall = {}

    def __call__(self, name):
        timers = self
        class _T:
            def __enter__(s):
                s.t0 = time.perf_counter()
                if timers.cuda: s.a = torch.cuda.Event(enable_timing=True); s.a.record()
            def __exit__(s, *e):
                if timers.cuda:
                    b = torch.cuda.Event(enable_timing=True); b.record(); timers.ev.append((name, s.a, b))
                timers.wall[name] = timers.wall.get(name, 0.)+1e3*(time.perf_counter()-s.t0)
        return _T()

    def resolve(self):
        out = {}
        if self.cuda:
            torch.cuda.synchronize()
            for n, a, b in self.ev: out[n] = out.get(n, 0.)+a.elapsed_time(b)
        return {'cuda_ms': out, 'host_ms': self.wall}


class PointMeasurement:
    """One linearization of all points: visibility, sampled terms and per-point information sums."""
    def __init__(self, rig, frame, noise, policy, rgb0, Sigma_prior, sigma_z2, chunk, dtype, measured=None, timers=None, assignment='per-frame'):
        self.rig, self.frame, self.noise, self.policy, self.rgb0 = rig, frame, noise, policy, rgb0
        self.Sigma_prior, self.sigma_z2, self.chunk, self.dtype, self.measured = Sigma_prior, sigma_z2, chunk, dtype, measured
        self.timers = timers or Timers('cpu')
        if assignment not in ('per-frame', 'per-iteration'): raise ValueError(assignment)
        self.assignment, self.frozen = assignment, None

    def __call__(self, x, keep_terms=None, active=None):
        """x [N,3] -> dict(Ixx [N,3,3], gx [N,3], rTWr [N], nrows [N], reason [C,N], terms of keep_terms points).
        active [N] bool: only these points' rows are sampled (stopped points need no new terms); visibility always uses
        every point."""
        N, C = len(x), self.rig.count
        T = self.timers
        with T('projection'):
            proj = self.rig.project(x, jacobian=False)
        with T('visibility'):
            vreason, front = visibility(proj, self.sigma_z2, self.frame.H, self.frame.W, self.rig.near, self.policy['visibility_n_sigma'], self.measured)
            if self.assignment == 'per-frame':
                if self.frozen is None: self.frozen = vreason.clone()          # first call = the prior prediction s-
                vreason = apply_frozen_assignment(vreason, self.frozen)
            cand = vreason.T == CODE['VALID']
            if active is not None: cand &= active[:, None]
            rows = torch.nonzero(cand)                                   # [M,2] (point, cam), ordered by point then camera
        reason = vreason.clone()
        Ixx = torch.zeros((N, 3, 3), dtype=self.dtype, device=x.device); gx = torch.zeros((N, 3), dtype=self.dtype, device=x.device)
        rTWr = torch.zeros(N, dtype=self.dtype, device=x.device); nrows = torch.zeros(N, dtype=torch.long, device=x.device)
        kept, rv, cv = [], [], []
        cam_info = torch.zeros(C, dtype=self.dtype, device=x.device)
        M = len(rows)
        # point-range chunks: every point's rows are inside one chunk (exact per-point sums for any chunk size)
        pts = rows[:, 0]
        bounds = torch.searchsorted(pts, torch.arange(0, N+self.chunk, self.chunk, device=x.device).clamp_max(N)).tolist() if M else [0]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a: continue
            p, c = pts[a:b], rows[a:b, 1]
            with T('sampling_terms'):
                t = build_terms(self.rig, self.frame, c, p, x[p], self.rgb0[p], self.Sigma_prior[p], self.noise, self.policy, self.dtype)
            reason[c, p] = t.reason
            with T('information_accumulation'):
                ok = t.valid()
                w = torch.where(ok[:, None], 1/t.var, torch.zeros_like(t.var))
                wr = w*t.r
                I = (t.Hx[:, :, :, None]*w[:, :, None, None]*t.Hx[:, :, None, :]).sum(1)      # [m,3,3] H^T R^-1 H
                g = (t.Hx*wr[:, :, None]).sum(1)                                               # [m,3]   H^T R^-1 r
                q = (wr*t.r).sum(1)
                up, lengths = torch.unique_consecutive(p, return_counts=True)
                Ixx[up] = segment_sum(I, lengths); gx[up] = segment_sum(g, lengths); rTWr[up] = segment_sum(q, lengths)
                nrows[up] = segment_sum(ok.to(self.dtype), lengths).round().long()
                cam_info.index_add_(0, c, I.diagonal(dim1=1, dim2=2).sum(-1))     # per-camera information trace (diagnostic)
                rv.append(t.r[ok]); cv.append(c[ok])
            if keep_terms is not None:
                sel = torch.isin(p, keep_terms)
                if bool(sel.any()): kept.append(t.select(sel))
        e = torch.zeros((0, 4), dtype=self.dtype, device=x.device)
        return {'Ixx': Ixx, 'gx': gx, 'rTWr': rTWr, 'nrows': nrows, 'reason': reason, 'front': front, 'proj': proj, 'rows_considered': M,
                'kept': kept, 'r_valid': torch.cat(rv) if rv else e, 'cam_valid': torch.cat(cv) if cv else torch.zeros(0, dtype=torch.long, device=x.device),
                'cam_information_trace': cam_info}


def update_points(s_minus, P_minus, frame, rig, rgb0, noise, policy, max_iters, tol, chunk=200_000, dtype=torch.float64,
                  trace_ids=None, timers=None, assignment='per-frame'):
    """Fixed-prior iterated update of every point (contract 3.2-3.3). Returns (s+, P+, report tensors, trace)."""
    timers = timers or Timers(s_minus.device)
    N = len(s_minus)
    s_minus, P_minus = s_minus.to(dtype), P_minus.to(dtype)
    with timers('prior_information'):
        Lp, retry = spd_factor(P_minus, 'prior covariance P-')
        Lam = torch.cholesky_inverse(Lp).to(dtype); Lam = (Lam+Lam.transpose(1, 2))/2
        Sigma_prior = P_minus[:, :3, :3]
        sigma_z2 = position_sigma_z2(rig.R.to(dtype), Sigma_prior)
    meas = PointMeasurement(rig, frame, noise, policy, rgb0.to(dtype), Sigma_prior, sigma_z2, chunk, dtype, timers=timers, assignment=assignment)
    keep = None if trace_ids is None else torch.as_tensor(trace_ids, device=s_minus.device)
    sj = s_minus.clone()
    trace = {'iterations': []} if keep is not None else None
    history, converged, step = [], torch.zeros(N, dtype=torch.bool, device=sj.device), None
    iters = torch.zeros(N, dtype=torch.long, device=sj.device)
    nis = nis_dof = prior_innov = None
    for j in range(max_iters):
        act = ~converged                                               # stopped points are neither re-sampled nor re-solved
        m = meas(sj[:, :3], keep, None if j == 0 else act)
        with timers('solve'):
            ia = torch.nonzero(act).flatten()
            A = Lam[ia].clone(); A[:, :3, :3] += m['Ixx'][ia]
            e = sj[ia]-s_minus[ia]
            b_a = -(Lam[ia]*e[:, None, :]).sum(-1); b_a[:, :3] += m['gx'][ia]
            L, _ = spd_factor(A, f'information matrix A (iteration {j})')
            d_a = torch.cholesky_solve(b_a[..., None], L)[..., 0]
            delta = torch.zeros_like(sj); delta[ia] = d_a
            b = torch.zeros_like(sj); b[ia] = b_a
            step = torch.zeros(N, dtype=sj.dtype, device=sj.device); step[ia] = (d_a*b_a).sum(-1)     # delta^T A delta
        if j == 0:                                                    # NIS at the prior (s_0 = s-): r^T S^-1 r = r^T W r - g^T A^-1 g
            nis = (m['rTWr']-step).clamp_min(0); nis_dof = 4*m['nrows']
            prior_innov = (m['r_valid'], m['cam_valid'], m['reason'])
        if trace is not None:
            Afull = Lam.clone(); Afull[:, :3, :3] += m['Ixx']
            trace['iterations'].append(_trace_iteration(j, sj, m, Afull, b, delta, keep))
        # per-point stopping: a point whose step met the criterion keeps its iterate and is not iterated again, so its
        # result never depends on how long OTHER points iterate (independent filters)
        sj = torch.where(converged[:, None], sj, sj+delta)
        newly = (~converged) & (step < tol)
        iters = iters+(~converged).long()
        converged = converged | newly
        history.append({'iteration': j, 'rows': m['rows_considered'], 'not_converged': (~converged).sum(), 'step_max': torch.where(converged & ~newly, torch.zeros_like(step), step).max(),
                        'points_with_rows': (m['nrows'] > 0).sum()})
        if bool(converged.all()): break
    final = meas(sj[:, :3], keep)
    with timers('covariance'):
        A = Lam.clone(); A[:, :3, :3] += final['Ixx']
        L, retry2 = spd_factor(A, 'final information matrix')
        P_plus = torch.cholesky_inverse(L); P_plus = (P_plus+P_plus.transpose(1, 2))/2
        observed = final['nrows'] > 0
        s_plus = torch.where(observed[:, None], sj, s_minus)
        P_plus = torch.where(observed[:, None, None], P_plus, P_minus)
    if trace is not None:
        trace['final'] = _trace_final(sj, s_minus, P_minus, P_plus, final, Lam, keep, rig, frame, rgb0, noise, policy, dtype)
    report = {'s_iter_final': sj, 'observed': observed, 'nrows': final['nrows'], 'reason': final['reason'], 'front': final['front'],
              'converged': converged, 'point_iterations': iters, 'step': step, 'nis': nis, 'nis_dof': nis_dof, 'iterations': len(history), 'history': history,
              'info_eig': eigvalsh3(final['Ixx']), 'proj': final['proj'], 'prior_innovation': prior_innov,
              'final_innovation': (final['r_valid'], final['cam_valid']), 'cam_information_trace': final['cam_information_trace'],
              'float64_retry': [r for r in (retry, retry2) if r]}
    return s_plus, P_plus, report, trace


def _trace_iteration(j, sj, m, A, b, delta, keep):
    out = {'iteration': j, 'points': {}}
    for pid in keep.tolist():
        rows = [t.select(t.point == pid) for t in m['kept']]
        rows = [r for r in rows if len(r.point)]
        out['points'][pid] = {'state': sj[pid].tolist(), 'A': A[pid].tolist(), 'b': b[pid].tolist(), 'delta': delta[pid].tolist(),
                              'rows': [_row_dict(r, i) for r in rows for i in range(len(r.point))]}
    return out


def _row_dict(t, i):
    return {'camera': int(t.cam[i]), 'u': float(t.uv[i, 0]), 'v': float(t.uv[i, 1]), 'z_pred': float(t.z[i]),
            'sampled_rgb': t.meas_rgb[i].tolist(), 'sampled_depth_m': float(t.meas_depth[i]), 'pixel00_u_v': t.pixel00[i].tolist(),
            'reason': REASONS[int(t.reason[i])], 'r': t.r[i].tolist(), 'H_x': t.Hx[i].tolist(), 'R_diag': t.var[i].tolist(), 'S_zz': float(t.S_zz[i])}


def _trace_final(sj, s_minus, P_minus, P_plus, final, Lam, keep, rig, frame, rgb0, noise, policy, dtype):
    """Final linearization of traced points: explicit gain, Joseph cross-check and finite-difference H."""
    out = {}
    for pid in keep.tolist():
        rows = [t.select((t.point == pid) & t.valid()) for t in final['kept']]
        rows = [r for r in rows if len(r.point)]
        d = {'s_minus': s_minus[pid].tolist(), 's_plus': sj[pid].tolist() if bool(final['nrows'][pid] > 0) else s_minus[pid].tolist(),
             'P_minus': P_minus[pid].tolist(), 'P_plus': P_plus[pid].tolist(), 'valid_rows': int(final['nrows'][pid])}
        if rows:
            Hx = torch.cat([r.Hx for r in rows]).reshape(-1, 3).double(); Rv = torch.cat([r.var for r in rows]).reshape(-1).double()
            H = torch.zeros((len(Hx), 6), dtype=torch.float64, device=Hx.device); H[:, :3] = Hx
            Pm = P_minus[pid].double()
            S = H@Pm@H.T+torch.diag(Rv); K = Pm@H.T@torch.linalg.inv(S)
            IKH = torch.eye(6, dtype=torch.float64, device=Hx.device)-K@H
            joseph = IKH@Pm@IKH.T+K@torch.diag(Rv)@K.T
            d.update(K=K.tolist(), joseph_P_plus=joseph.tolist(), joseph_vs_information_max_abs=float((joseph-P_plus[pid].double()).abs().max()),
                     rows=[_row_dict(r, i) for r in rows for i in range(len(r.point))])
            # finite differences of h at the final iterate, same rows
            cam = torch.cat([r.cam for r in rows]); x = sj[pid, :3].double()
            fd = torch.zeros((len(cam), 4, 3), dtype=torch.float64, device=x.device); eps = 1e-6
            for a in range(3):
                dx = torch.zeros(3, dtype=torch.float64, device=x.device); dx[a] = eps
                hp, okp = h_value(rig, frame, cam, (x+dx).expand(len(cam), 3), rgb0[pid].expand(len(cam), 3), noise, policy)
                hm, okm = h_value(rig, frame, cam, (x-dx).expand(len(cam), 3), rgb0[pid].expand(len(cam), 3), noise, policy)
                fd[:, :, a] = (hp-hm)/(2*eps)
            an = torch.cat([r.Hx for r in rows]).double()
            d['H_finite_difference'] = fd.tolist()
            d['H_fd_max_abs_error'] = float((fd-an).abs().max())
            d['H_fd_rel_error'] = float((fd-an).norm()/an.norm().clamp_min(1e-300))
        out[pid] = d
    return out
