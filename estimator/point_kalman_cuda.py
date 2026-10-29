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
Acceptance (configs/projective_rgbd.json update_acceptance = 'last_consistent_iterate'). Convergence of the step is
not evidence that the state explains the measurements, and a large step can cross many pixels and surfaces. Every
iterate s_j is therefore tested against the ACTUAL RGB-D samples at its own projection, with the posterior cost
    J(s_j) = (s_j - s-)^T Lambda- (s_j - s-) + r_j^T R^-1 r_j,      J ~ chi2(4 m_j) for a consistent update,
m_j valid rows at s_j (the IEKF analogue of the NIS: at a linear-Gaussian optimum J equals r_0^T S^-1 r_0). The
accepted state is the final iterate if J <= chi2_{4m}(1 - alpha), else the LAST earlier iterate (j >= 1) that passed,
re-linearized at that state; if none passed, the point is prediction-only for this frame (s+ = s-, P+ = P-). alpha is
the row-gate tail 2 (1 - Phi(visibility_n_sigma)) (projective_terms.row_gate_threshold): no new number. Non-converged
iterates are accepted when they are consistent (bilinear oscillation near the solution), so the ablation that
rejected every non-converged point (static drift down, moving parts lost) is NOT what this does. The prior term is
what rejects a 0.2 m correction of a point whose prior allows centimetres; a legitimate fast motion passes whenever the
prior covariance (velocity prior, q_c) allows it. 'final_iterate' is the earlier rule (accept the final iterate
whenever it has a valid row), kept for the one-variable ablation.
Posterior ONCE at the accepted state's own linearization: P+ = (Lambda- + H^T R^-1 H)^-1, the Joseph-equivalent form
(verified against (I-KH)P-(I-KH)^T + K R K^T in float64). The acceptance linearization re-checks self-occlusion at
the current states (final_visibility = 'recheck_self_occlusion', projective_visibility.apply_frozen_assignment).
Invalid rows enter the information sums through torch.where (never 0 * value: IEEE 0 * NaN = NaN).
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
from estimator.projective_sensor import CODE, REASONS, sample
from scipy.stats import chi2, norm
from estimator.projective_terms import build_terms, h_value
from estimator.projective_visibility import visibility, position_sigma_z2, apply_frozen_assignment


ACCEPTANCE = {0: 'ANCHOR_OR_NOT_PROCESSED', 1: 'ACCEPTED_FINAL_ITERATE', 2: 'ACCEPTED_EARLIER_CONSISTENT_ITERATE',
              3: 'REJECTED_INCONSISTENT', 4: 'NO_VALID_ROWS'}


def cost_thresholds(policy, cameras, dtype, device):
    """chi2_{4m}(1 - alpha) for m = 0..cameras valid rows (index 0: no rows, +inf), alpha = 2 (1 - Phi(n_sigma))."""
    alpha = 2*norm.sf(policy['visibility_n_sigma'])
    return torch.tensor([float('inf')]+[float(chi2.isf(alpha, 4*m)) for m in range(1, cameras+1)], dtype=dtype, device=device)


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

    def __call__(self, x, keep_terms=None, active=None, dx=None, recheck=False):
        """x [N,3] -> dict(Ixx [N,3,3], gx [N,3], rTWr [N], nrows [N], reason [C,N], terms of keep_terms points).
        active [N] bool: only these points' rows are sampled (stopped points need no new terms); visibility always uses
        every point. dx [N,3] = x - x- (row gate on the prior-linearized innovation); recheck: acceptance linearization
        (fresh self-occlusion added to the frozen assignment)."""
        N, C = len(x), self.rig.count
        T = self.timers
        with T('projection'):
            proj = self.rig.project(x, jacobian=False)
        with T('visibility'):
            vreason, front = visibility(proj, self.sigma_z2, self.frame.H, self.frame.W, self.rig.near, self.policy['visibility_n_sigma'], self.measured)
            fresh = vreason
            if self.assignment == 'per-frame':
                if self.frozen is None: self.frozen = vreason.clone()          # first call = the prior prediction s-
                vreason = apply_frozen_assignment(vreason, self.frozen, recheck)
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
                t = build_terms(self.rig, self.frame, c, p, x[p], self.rgb0[p], self.Sigma_prior[p], self.noise, self.policy, self.dtype,
                                None if dx is None else dx[p])
            reason[c, p] = t.reason
            with T('information_accumulation'):
                ok = t.valid()
                w = torch.where(ok[:, None], 1/t.var, torch.zeros_like(t.var))
                tr_ = torch.where(ok[:, None], t.r, torch.zeros_like(t.r)); Hx = torch.where(ok[:, None, None], t.Hx, torch.zeros_like(t.Hx))
                wr = w*tr_
                I = (Hx[:, :, :, None]*w[:, :, None, None]*Hx[:, :, None, :]).sum(1)          # [m,3,3] H^T R^-1 H
                g = (Hx*wr[:, :, None]).sum(1)                                                 # [m,3]   H^T R^-1 r
                q = (wr*tr_).sum(1)
                up, lengths = torch.unique_consecutive(p, return_counts=True)
                Ixx[up] = segment_sum(I, lengths); gx[up] = segment_sum(g, lengths); rTWr[up] = segment_sum(q, lengths)
                nrows[up] = segment_sum(ok.to(self.dtype), lengths).round().long()
                cam_info.index_add_(0, c, I.diagonal(dim1=1, dim2=2).sum(-1))     # per-camera information trace (diagnostic)
                rv.append(t.r[ok]); cv.append(c[ok])
            if keep_terms is not None:
                sel = torch.isin(p, keep_terms)
                if bool(sel.any()): kept.append(t.select(sel))
        e = torch.zeros((0, 4), dtype=self.dtype, device=x.device)
        probe = None if keep_terms is None else {pid: self.probe(x, pid, proj, fresh, reason, dx) for pid in keep_terms.tolist()}
        return {'Ixx': Ixx, 'gx': gx, 'rTWr': rTWr, 'nrows': nrows, 'reason': reason, 'front': front, 'proj': proj, 'rows_considered': M,
                'kept': kept, 'r_valid': torch.cat(rv) if rv else e, 'cam_valid': torch.cat(cv) if cv else torch.zeros(0, dtype=torch.long, device=x.device),
                'cam_information_trace': cam_info, 'probe': probe}

    def probe(self, x, pid, proj, fresh, reason, dx, occupants=8):
        """Trace only (debug point ids; nothing here reaches the solver): every camera of point pid at this linearization.
        Visibility before / after the frozen assignment, the owner and occupants of its nearest pixel, and the sampled row
        evaluated for EVERY camera (also cameras the visibility excluded) with its raw 2x2 neighbours, bilinear weights,
        interpolant gradients, projection Jacobians, r, H_x, R and the row-gate statistics."""
        C, dev = self.rig.count, x.device
        cam = torch.arange(C, device=dev); p = torch.full((C,), pid, dtype=torch.long, device=dev)
        dxp = None if dx is None else dx[p]
        t = build_terms(self.rig, self.frame, cam, p, x[p], self.rgb0[p], self.Sigma_prior[p], self.noise, self.policy, self.dtype, dxp)
        uv, Z, Ju, Jz = self.rig.project_rows(x[p], cam)
        s = sample(self.frame, cam, uv, self.noise, self.policy, self.rig.K, self.rig.dtype)
        u0, v0 = s.pixel00[:, 0], s.pixel00[:, 1]
        nb = [(v0, u0), (v0, u0+1), (v0+1, u0), (v0+1, u0+1)]
        du, dv = (uv[:, 0]-torch.floor(uv[:, 0])).clamp(0, 1), (uv[:, 1]-torch.floor(uv[:, 1])).clamp(0, 1)
        out = []
        finite = torch.isfinite(proj.uv).all(-1) & torch.isfinite(proj.z)
        pu, pv = torch.round(torch.where(finite, proj.uv[..., 0], torch.zeros_like(proj.z))), torch.round(torch.where(finite, proj.uv[..., 1], torch.zeros_like(proj.z)))
        for c in range(C):
            same = finite[c] & (proj.z[c] > self.rig.near[c]) & (pu[c] == pu[c, pid]) & (pv[c] == pv[c, pid])
            ids = torch.nonzero(same).flatten()
            occ = ids[torch.argsort(ids, stable=True)]; occ = occ[torch.argsort(proj.z[c, occ], stable=True)][:occupants]
            out.append({'camera': c, 'u': float(uv[c, 0]), 'v': float(uv[c, 1]), 'Z': float(Z[c]), 'nearest_pixel_u_v': [int(pu[c, pid]), int(pv[c, pid])],
                        'visibility_fresh': REASONS[int(fresh[c, pid])], 'visibility_frozen': None if self.frozen is None else REASONS[int(self.frozen[c, pid])],
                        'reason_used': REASONS[int(reason[c, pid])], 'reason_if_sampled': REASONS[int(t.reason[c])],
                        'sigma_z2_prior': float(self.sigma_z2[c, pid]),
                        'pixel_occupants_front_to_back': [{'point': int(i), 'z': float(proj.z[c, i]), 'sigma_z2_prior': float(self.sigma_z2[c, i])} for i in occ],
                        'pixel_occupants_total': int(len(ids)),
                        'pixel00_u_v': [int(u0[c]), int(v0[c])], 'du_dv': [float(du[c]), float(dv[c])],
                        'neighbours_order': '00 (u0,v0), 10 (u0+1,v0), 01 (u0,v0+1), 11 (u0+1,v0+1)',
                        'neighbour_rgb': [self.frame.rgb[c, y[c], xx[c]].double().tolist() for y, xx in nb],
                        'neighbour_depth_m': [float(self.frame.depth[c, y[c], xx[c]]) for y, xx in nb],
                        'neighbour_mask': [bool(self.frame.mask[c, y[c], xx[c]]) for y, xx in nb],
                        'neighbour_raw_valid': [bool(self.frame.raw_valid[c, y[c], xx[c]]) for y, xx in nb],
                        'sampled_rgb': t.meas_rgb[c].tolist(), 'sampled_depth_m': float(t.meas_depth[c]),
                        'grad_rgb_uv': s.grad_rgb[c].double().tolist(), 'grad_depth_uv': s.grad_depth[c].double().tolist(),
                        'J_uv_x': Ju[c].double().tolist(), 'J_Z_x': Jz[c].double().tolist(),
                        'r': t.r[c].tolist(), 'H_x': t.Hx[c].tolist(), 'R_diag': t.var[c].tolist(), 'S_zz': float(t.S_zz[c]),
                        'row_nis_prior_linearized': None if not math.isfinite(float(t.nis[c])) else float(t.nis[c])})
        return out


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
    rule = policy.get('update_acceptance', 'final_iterate')
    if rule not in ('last_consistent_iterate', 'final_iterate'): raise ValueError(f'update_acceptance {rule!r}')
    recheck = policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion'
    thr = cost_thresholds(policy, rig.count, dtype, s_minus.device)
    prior_cost = lambda idx, st: (lambda e: (e*(Lam[idx]@e[..., None])[..., 0]).sum(-1))(st[idx]-s_minus[idx])
    keep = None if trace_ids is None else torch.as_tensor(trace_ids, device=s_minus.device)
    sj = s_minus.clone()
    trace = {'iterations': []} if keep is not None else None
    history, converged, step = [], torch.zeros(N, dtype=torch.bool, device=sj.device), None
    iters = torch.zeros(N, dtype=torch.long, device=sj.device)
    acc_state, acc_iter = s_minus.clone(), torch.full((N,), -1, dtype=torch.long, device=sj.device)
    had_rows = torch.zeros(N, dtype=torch.bool, device=sj.device)
    nis = nis_dof = prior_innov = None
    for j in range(max_iters):
        act = ~converged                                               # stopped points are neither re-sampled nor re-solved
        m = meas(sj[:, :3], keep, None if j == 0 else act, sj[:, :3]-s_minus[:, :3])
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
        with timers('acceptance'):                                    # J(s_j) of the iterate just measured (s_0 = s- is no update)
            nr = m['nrows'][ia]; had_rows[ia] |= nr > 0
            J = prior_cost(ia, sj)+m['rTWr'][ia]
            if j >= 1:
                ok = (nr > 0) & (J <= thr[nr.clamp_max(rig.count)])
                acc_state[ia[ok]] = sj[ia[ok]]; acc_iter[ia[ok]] = j
        if j == 0:                                                    # NIS at the prior (s_0 = s-): r^T S^-1 r = r^T W r - g^T A^-1 g
            nis = (m['rTWr']-step).clamp_min(0); nis_dof = 4*m['nrows']
            prior_innov = (m['r_valid'], m['cam_valid'], m['reason'])
        if trace is not None:
            Afull = Lam.clone(); Afull[:, :3, :3] += m['Ixx']
            Jfull = torch.full((N,), float('nan'), dtype=sj.dtype, device=sj.device); Jfull[ia] = J
            trace['iterations'].append(_trace_iteration(j, sj, m, Afull, b, delta, keep, Jfull, thr, rig.count, step, act))
        # per-point stopping: a point whose step met the criterion keeps its iterate and is not iterated again, so its
        # result never depends on how long OTHER points iterate (independent filters)
        sj = torch.where(converged[:, None], sj, sj+delta)
        newly = (~converged) & (step < tol)
        iters = iters+(~converged).long()
        converged = converged | newly
        history.append({'iteration': j, 'rows': m['rows_considered'], 'not_converged': (~converged).sum(), 'step_max': torch.where(converged & ~newly, torch.zeros_like(step), step).max(),
                        'points_with_rows': (m['nrows'] > 0).sum()})
        if bool(converged.all()): break
    final = meas(sj[:, :3], keep, None, sj[:, :3]-s_minus[:, :3], recheck)
    with timers('acceptance'):
        allp = torch.arange(N, device=sj.device)
        nf = final['nrows']; had_rows |= nf > 0
        Jf = prior_cost(allp, sj)+final['rTWr']
        Ixx, nrows, reason, cost = final['Ixx'].clone(), nf.clone(), final['reason'].clone(), Jf.clone()
        if rule == 'final_iterate':
            ok_f = nf > 0; ok_e = torch.zeros_like(ok_f); state = sj
        else:
            ok_f = (nf > 0) & (Jf <= thr[nf.clamp_max(rig.count)])
            fb = ~ok_f & (acc_iter >= 1)                            # fall back to the last consistent iterate
            state = torch.where(ok_f[:, None], sj, torch.where(fb[:, None], acc_state, s_minus))
            ok_e = torch.zeros_like(ok_f)
            if bool(fb.any()):
                m2 = meas(state[:, :3], keep, fb, state[:, :3]-s_minus[:, :3], recheck)
                J2 = prior_cost(allp, state)+m2['rTWr']
                ok_e = fb & (m2['nrows'] > 0) & (J2 <= thr[m2['nrows'].clamp_max(rig.count)])
                Ixx[fb] = m2['Ixx'][fb]; nrows[fb] = m2['nrows'][fb]; reason[:, fb] = m2['reason'][:, fb]; cost[fb] = J2[fb]
                if keep is not None: final = {**final, 'kept_fallback': m2['kept'], 'fallback': fb, 'probe_fallback': m2['probe']}
        observed = ok_f | ok_e
        acceptance = torch.where(ok_f, 1, torch.where(ok_e, 2, torch.where(had_rows, 3, 4))).to(torch.uint8)
    with timers('covariance'):
        A = Lam.clone(); A[:, :3, :3] += Ixx
        L, retry2 = spd_factor(A, 'final information matrix')
        P_plus = torch.cholesky_inverse(L); P_plus = (P_plus+P_plus.transpose(1, 2))/2
        s_plus = torch.where(observed[:, None], state, s_minus)
        P_plus = torch.where(observed[:, None, None], P_plus, P_minus)
    if trace is not None:
        trace['final'] = _trace_final(state, s_plus, s_minus, P_minus, P_plus, final, Lam, keep, rig, frame, rgb0, noise, policy, dtype,
                                      {'acceptance': acceptance, 'cost': cost, 'nrows': nrows, 'thr': thr, 'converged': converged, 'acc_iter': acc_iter})
    report = {'s_iter_final': sj, 'observed': observed, 'nrows': torch.where(observed, nrows, torch.zeros_like(nrows)) if rule != 'final_iterate' else nrows,
              'reason': reason, 'front': final['front'],
              'converged': converged, 'point_iterations': iters, 'step': step, 'nis': nis, 'nis_dof': nis_dof, 'iterations': len(history), 'history': history,
              'info_eig': eigvalsh3(Ixx), 'proj': final['proj'], 'prior_innovation': prior_innov,
              'final_innovation': (final['r_valid'], final['cam_valid']), 'cam_information_trace': final['cam_information_trace'],
              'acceptance': acceptance, 'posterior_cost': cost, 'posterior_cost_dof': 4*nrows, 'accepted_iteration': torch.where(ok_f, iters, torch.where(ok_e, acc_iter, -1)),
              'acceptance_rule': rule, 'float64_retry': [r for r in (retry, retry2) if r]}
    return s_plus, P_plus, report, trace


def _trace_iteration(j, sj, m, A, b, delta, keep, J, thr, C, step, active):
    out = {'iteration': j, 'points': {}}
    for pid in keep.tolist():
        rows = [t.select(t.point == pid) for t in m['kept']]
        rows = [r for r in rows if len(r.point)]
        n = int(m['nrows'][pid])
        out['points'][pid] = {'state': sj[pid].tolist(), 'A': A[pid].tolist(), 'b': b[pid].tolist(), 'delta': delta[pid].tolist(),
                              'posterior_cost': float(J[pid]), 'posterior_cost_dof': 4*n, 'cost_threshold': float(thr[min(n, C)]),
                              'rows': [_row_dict(r, i) for r in rows for i in range(len(r.point))],
                              'active': bool(active[pid]), 'step_delta_A_delta': float(step[pid]), 'Ixx': m['Ixx'][pid].tolist(), 'gx': m['gx'][pid].tolist(),
                              'rTWr': float(m['rTWr'][pid]), 'nrows': n, 'cameras': m['probe'][pid]}
    return out


def _row_dict(t, i):
    return {'camera': int(t.cam[i]), 'u': float(t.uv[i, 0]), 'v': float(t.uv[i, 1]), 'z_pred': float(t.z[i]),
            'sampled_rgb': t.meas_rgb[i].tolist(), 'sampled_depth_m': float(t.meas_depth[i]), 'pixel00_u_v': t.pixel00[i].tolist(),
            'reason': REASONS[int(t.reason[i])], 'r': t.r[i].tolist(), 'H_x': t.Hx[i].tolist(), 'R_diag': t.var[i].tolist(), 'S_zz': float(t.S_zz[i]),
            'row_nis_prior_linearized': float(t.nis[i])}


def _trace_final(state, s_plus, s_minus, P_minus, P_plus, final, Lam, keep, rig, frame, rgb0, noise, policy, dtype, acc):
    """Accepted linearization of traced points: acceptance decision, explicit gain, Joseph cross-check and
    finite-difference H (at the state whose rows give the covariance)."""
    out = {}
    fb = final.get('fallback')
    for pid in keep.tolist():
        src = final['kept_fallback'] if fb is not None and bool(fb[pid]) else final['kept']
        rows = [t.select((t.point == pid) & t.valid()) for t in src]
        rows = [r for r in rows if len(r.point)]
        n = int(acc['nrows'][pid])
        d = {'s_minus': s_minus[pid].tolist(), 's_plus': s_plus[pid].tolist(), 'linearization_state': state[pid].tolist(),
             'P_minus': P_minus[pid].tolist(), 'P_plus': P_plus[pid].tolist(), 'valid_rows': n,
             'acceptance': ACCEPTANCE[int(acc['acceptance'][pid])], 'posterior_cost': float(acc['cost'][pid]), 'posterior_cost_dof': 4*n,
             'cost_threshold': float(acc['thr'][min(n, rig.count)]), 'converged': bool(acc['converged'][pid]),
             'last_consistent_iteration': int(acc['acc_iter'][pid]), 'Lambda_minus': Lam[pid].tolist(),
             'cameras_final_iterate': final['probe'][pid], 'final_iterate_state_sampled': bool(fb is None or not bool(fb[pid]))}
        if fb is not None and bool(fb[pid]): d['cameras_fallback_state'] = final['probe_fallback'][pid]
        if rows:
            Hx = torch.cat([r.Hx for r in rows]).reshape(-1, 3).double(); Rv = torch.cat([r.var for r in rows]).reshape(-1).double()
            H = torch.zeros((len(Hx), 6), dtype=torch.float64, device=Hx.device); H[:, :3] = Hx
            Pm = P_minus[pid].double()
            S = H@Pm@H.T+torch.diag(Rv); K = Pm@H.T@torch.linalg.inv(S)
            IKH = torch.eye(6, dtype=torch.float64, device=Hx.device)-K@H
            joseph = IKH@Pm@IKH.T+K@torch.diag(Rv)@K.T
            d.update(K=K.tolist(), joseph_P_plus=joseph.tolist(), joseph_vs_information_max_abs=float((joseph-P_plus[pid].double()).abs().max()),
                     rows=[_row_dict(r, i) for r in rows for i in range(len(r.point))])
            # finite differences of h at the linearization state, same rows
            cam = torch.cat([r.cam for r in rows]); x = state[pid, :3].double()
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
