"""F2 / B2: one shared rigid SE(3) pose + spatial twist IEKF for the coherent moving core (contract section 5).

State (T_k, xi_k), xi = [v, omega] (m/s, rad/s, physical time), 12x12 covariance in the LEFT tangent
[delta_tau; delta_xi], T = Exp(delta_tau^) T_hat (estimator/se3.py). Canonical core point p_i (its own endpoint):
x_i(k) = R_k p_i + t_k, T_anchor = I.

Prediction, signed dt, tau = |dt|, E = Exp(dt xi^):
    T- = E T+,  xi- = xi+,   F = [[Ad(E), dt J_l(dt xi)], [0, I6]],   P- = F P+ F^T + Q
    Q = [[tau^3/3 Qc, sgn(dt) tau^2/2 Qc], [sgn(dt) tau^2/2 Qc, tau Qc]]   (integrated white twist acceleration, density
    Qc = noise-config process_covariance_rate[6:12, 6:12]; first-order in the pose block, stated approximation)
Measurement: the free filter's projective terms (projective_terms.build_terms: same sampler, r, H_x, R, visibility)
at x_i = T p_i, chained H_rigid = [H_x B_i, 0], B_i = [I, -[x_i]x]. Static-compatible endpoint points (coherence
labels) stay at their canonical XYZ only as OCCLUDERS in the per-pixel point table; ambiguous points are not used.
Iterated update with the fixed prior (T-, xi-, P-), left-chart prior error and its Jacobian:
    e_j = [Log(T_j T-^-1); xi_j - xi-],   L_j = blkdiag(J_l(e_pose)^-1, I6)
    A_j = L_j^T (P-)^-1 L_j + sum H^T R^-1 H,   b_j = sum H^T R^-1 r - L_j^T (P-)^-1 e_j,   delta = A_j^-1 b_j
    T_{j+1} = Exp(delta_tau^) T_j,  xi_{j+1} = xi_j + delta_xi
Covariance chart policy (ONE policy, contract 5.3): FINAL-CHART. At the accepted iterate P = (L^T (P-)^-1 L +
H^T R^-1 H)^-1 is computed directly in the final pose's tangent; no reset is applied afterwards. Cross-check: the
local-chart Joseph form with P_loc = L^-1 P- L^-T (debug trace). No valid row at the final iterate: (T, xi, P) = prior.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from estimator import se3
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise, REASONS, CODE
from estimator.projective_terms import build_terms
from estimator.projective_visibility import visibility, apply_frozen_assignment
from estimator.point_kalman_cuda import Timers, spd_factor
from utils.io import json_write, serial


def rigid_predict(T, xi, P, dt, Qc):
    tau, sg = abs(dt), math.copysign(1., dt)
    E = se3.exp(dt*xi)
    F = torch.zeros((12, 12), dtype=T.dtype, device=T.device)
    F[:6, :6] = se3.Ad(E); F[:6, 6:] = dt*se3.left_jacobian(dt*xi); F[6:, 6:] = torch.eye(6, dtype=T.dtype, device=T.device)
    Q = torch.cat([torch.cat([tau**3/3*Qc, sg*tau**2/2*Qc], 1), torch.cat([sg*tau**2/2*Qc, tau*Qc], 1)], 0)
    Pm = F@P@F.T+Q
    return E@T, xi.clone(), (Pm+Pm.T)/2, F, Q


def chain_measurement(rig, frame, noise, policy, x, G, P_block, rgb, occ_p, occ_cov, dtype=torch.float64, chunk=400_000, keep_rows=False, memo=None, dx=None):
    """Projective RGB-D information of measured points x [m,3] whose positions depend on a small state through
    G = dx/dstate [m,3,d]: H = H_x G per valid row (projective_terms, identical to the free filter), prior position
    covariance Sigma_x = G P_block G^T, occluders occ_p (point table only), dx = x - x(prior) for the row gate on the
    prior-linearized innovation. Returns I = sum H^T R^-1 H [d,d], g = sum H^T R^-1 r [d] and diagnostics. Invalid
    rows enter through torch.where, never as 0 * value (IEEE 0 * NaN = NaN)."""
    m, C, d = len(x), rig.count, G.shape[-1]
    Sx = (G[:, :, :, None]*(P_block@G.transpose(1, 2))[:, None, :, :]).sum(2)
    allx = torch.cat([x, occ_p]); allS = torch.cat([Sx, occ_cov])
    proj = rig.project(allx, jacobian=False)
    r3 = rig.R[:, 2, :].to(allS.dtype)
    sz2 = torch.einsum('ci,nij,cj->cn', r3, allS, r3).clamp_min(0)
    measured = torch.zeros(len(allx), dtype=torch.bool, device=x.device); measured[:m] = True
    vreason, _ = visibility(proj, sz2, frame.H, frame.W, rig.near, policy['visibility_n_sigma'], measured)
    if memo is not None:                     # per-frame pixel-reuse assignment, decided at the first call (the prior)
        if 'frozen' not in memo: memo['frozen'] = vreason.clone()
        vreason = apply_frozen_assignment(vreason, memo['frozen'])
    reason = vreason[:, :m].clone()
    rows = torch.nonzero(reason.T == CODE['VALID'])
    I = torch.zeros((d, d), dtype=dtype, device=x.device); g = torch.zeros(d, dtype=dtype, device=x.device); q = torch.zeros((), dtype=dtype, device=x.device)
    rv, cv, kept, nvalid = [], [], [], torch.zeros(m, dtype=torch.long, device=x.device)
    for a in range(0, len(rows), chunk):
        p, c = rows[a:a+chunk, 0], rows[a:a+chunk, 1]
        t = build_terms(rig, frame, c, p, x[p], rgb[p], Sx[p], noise, policy, dtype, None if dx is None else dx[p])
        reason[c, p] = t.reason
        ok = t.valid()
        w = torch.where(ok[:, None], 1/t.var, torch.zeros_like(t.var)); tr_ = torch.where(ok[:, None], t.r, torch.zeros_like(t.r))
        H = (torch.where(ok[:, None, None], t.Hx, torch.zeros_like(t.Hx))[:, :, :, None]*G[p][:, None, :, :]).sum(2)   # [k,4,d] = H_x G
        I = I+(H[:, :, :, None]*w[:, :, None, None]*H[:, :, None, :]).sum((0, 1))
        g = g+(H*(w*tr_)[:, :, None]).sum((0, 1)); q = q+(w*tr_*tr_).sum()
        nvalid.index_add_(0, p[ok], torch.ones(int(ok.sum()), dtype=torch.long, device=x.device))
        rv.append(t.r[ok]); cv.append(c[ok])
        if keep_rows: kept.append((t.select(ok), H[ok]))
    e = torch.zeros((0, 4), dtype=dtype, device=x.device)
    return {'I': I, 'g': g, 'rTWr': q, 'reason': reason, 'nvalid_points': nvalid, 'rows': int(sum(len(r) for r in rv)),
            'r_valid': torch.cat(rv) if rv else e, 'cam_valid': torch.cat(cv) if cv else torch.zeros(0, dtype=torch.long, device=x.device),
            'x': x, 'kept': kept}


class RigidMeasurement:
    def __init__(self, rig, frame, noise, policy, core_p, core_rgb, occ_p, occ_cov, dtype=torch.float64, chunk=400_000, assignment='per-frame'):
        self.rig, self.frame, self.noise, self.policy = rig, frame, noise, policy
        self.core_p, self.core_rgb, self.occ_p, self.occ_cov, self.dtype, self.chunk = core_p, core_rgb, occ_p, occ_cov, dtype, chunk
        self.memo = {} if assignment == 'per-frame' else None
        self.x_prior = None

    def __call__(self, T, P_pose, keep_rows=False):
        x = self.core_p@T[:3, :3].T+T[:3, 3]
        if self.x_prior is None: self.x_prior = x          # first call = the prior pose T- (rigid_update)
        return chain_measurement(self.rig, self.frame, self.noise, self.policy, x, se3.point_jacobian(x), P_pose, self.core_rgb,
                                 self.occ_p, self.occ_cov, self.dtype, self.chunk, keep_rows, self.memo, x-self.x_prior)


def rigid_update(T_m, xi_m, P_m, meas, max_iters, tol, trace=False):
    dt = P_m.dtype
    Lp, _ = spd_factor(P_m[None], 'rigid prior covariance'); Lam = torch.cholesky_inverse(Lp)[0]; Lam = (Lam+Lam.T)/2
    T, xi = T_m.clone(), xi_m.clone()
    hist, tr = [], ([] if trace else None)
    converged = False
    nis = None
    for j in range(max_iters):
        m = meas(T, P_m[:6, :6], keep_rows=False)
        e = torch.cat([se3.log(T@se3.inv(T_m)), xi-xi_m])
        L = torch.eye(12, dtype=dt, device=T.device); L[:6, :6] = se3.left_jacobian_inv(e[:6])
        A = L.T@Lam@L; A[:6, :6] += m['I']
        b = -(L.T@Lam@e); b[:6] += m['g']
        Lc, _ = spd_factor(A[None], f'rigid information (iteration {j})')
        delta = torch.cholesky_solve(b[None, :, None], Lc)[0, :, 0]
        step = float(delta@b)
        if j == 0: nis = {'value': float(m['rTWr']-step), 'dof': 4*m['rows']}
        hist.append({'iteration': j, 'rows': m['rows'], 'step': step, 'prior_error_norm': float(e.norm())})
        if tr is not None: tr.append({'iteration': j, 'T': T.tolist(), 'xi': xi.tolist(), 'e': e.tolist(), 'L': L.tolist(), 'A': A.tolist(), 'b': b.tolist(), 'delta': delta.tolist(), 'rows': m['rows']})
        T = se3.exp(delta[:6])@T; xi = xi+delta[6:]
        if step < tol: converged = True; break
    final = meas(T, P_m[:6, :6], keep_rows=trace)
    e = torch.cat([se3.log(T@se3.inv(T_m)), xi-xi_m])
    L = torch.eye(12, dtype=dt, device=T.device); L[:6, :6] = se3.left_jacobian_inv(e[:6])
    A = L.T@Lam@L; A[:6, :6] += final['I']
    Lc, _ = spd_factor(A[None], 'rigid final information')
    P = torch.cholesky_inverse(Lc)[0]; P = (P+P.T)/2
    observed = final['rows'] > 0
    out = {'T': T if observed else T_m, 'xi': xi if observed else xi_m, 'P': P if observed else P_m, 'observed': observed,
           'converged': converged, 'iterations': len(hist), 'history': hist, 'nis_prior': nis, 'final': final}
    if tr is not None:
        Linv = torch.linalg.inv(L); P_loc = Linv@P_m@Linv.T
        Im = torch.zeros((12, 12), dtype=dt, device=T.device); Im[:6, :6] = final['I']
        Ainv = torch.linalg.inv(torch.linalg.inv(P_loc)+Im)
        IKH = torch.eye(12, dtype=dt, device=T.device)-Ainv@Im
        joseph = IKH@P_loc@IKH.T+Ainv@Im@Ainv                 # (I-KH) P_loc (I-KH)^T + K R K^T with K = A^-1 H^T R^-1
        out['trace'] = {'iterations': tr, 'final_T': T.tolist(), 'final_xi': xi.tolist(), 'P_final_chart': P.tolist(),
                        'P_local_joseph': joseph.tolist(), 'final_chart_vs_local_joseph_max_abs': float((joseph-P).abs().max()),
                        'covariance_chart_policy': 'final-chart (no reset)'}
    return out


def rigidity_check(core_p, T, exact_max=4096):
    """max_ij | ||x_i - x_j|| - ||p_i - p_j|| |: exact over EVERY pair of a deterministic evenly spaced subset of
    <= exact_max core points, plus a bound for all pairs: | ||R d|| - ||d|| | <= ||R^T R - I||_2 ||d|| <= ||R^T R - I||_2 span."""
    idx = torch.linspace(0, len(core_p)-1, min(exact_max, len(core_p)), device=core_p.device).round().long().unique()
    p = core_p[idx]; x = p@T[:3, :3].T+T[:3, 3]
    err = 0.
    for a in range(0, len(p), 256):          # direct differences (torch.cdist's matmul form cancels at ~1e-9)
        dx = (x[a:a+256, None, :]-x[None, :, :]).norm(dim=-1); dp = (p[a:a+256, None, :]-p[None, :, :]).norm(dim=-1)
        err = max(err, float((dx-dp).abs().max()))
    R = T[:3, :3]; dev = float(torch.linalg.matrix_norm(R.T@R-torch.eye(3, dtype=R.dtype, device=R.device), ord=2))
    span = float((core_p.max(0).values-core_p.min(0).values).norm())
    return {'pairs_checked_points': int(len(p)), 'max_pair_distance_error_m': err,
            'all_pairs_bound_m': dev*span, 'orthonormality_error': dev, 'core_span_m': span}


def run_rigid(dataset, config, policy, direction, free_dir, coherence_dir, root, device, max_iters=10, tol=1e-10,
              debug_frames=(), precision='float64', progress=None, noise_path=None, assignment='per-frame'):
    root, free_dir, coherence_dir = Path(root), Path(free_dir), Path(coherence_dir)
    (root/'numeric').mkdir(parents=True, exist_ok=True)
    summ = json.loads((coherence_dir/'coherence_summary.json').read_text())
    if summ['status'] != 'RESOLVED':
        st = {'pass': 'F2' if direction == 'forward' else 'B2', 'status': 'SKIPPED_UNRESOLVED_COHERENCE', 'coherence_status': summ['status']}
        json_write(root/'status.json', st); return st
    pol = summ['driving_policy']
    can = np.load(free_dir/'canonical.npz')
    core = np.load(coherence_dir/pol/'moving_core_point_ids.npy')
    rep = np.load(coherence_dir/pol/'representatives.npz')
    node_label = dict(zip(rep['node_ids'].tolist(), rep['labels'].tolist()))
    point_label = np.array([node_label.get(int(m), 'INCOHERENT_OR_AMBIGUOUS') for m in rep['membership']])
    static = np.flatnonzero((point_label == 'STATIC_COMPATIBLE') & ~np.isin(np.arange(len(point_label)), core))
    dtype = torch.float64
    dev = torch.device(device)
    rig = CameraRig(dataset.cameras, device, torch.float64 if precision == 'float64' else torch.float32)
    noise = SensorNoise.from_config(config, policy, device)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)
    core_p, core_rgb = t(can['canonical_xyz'][core]), t(can['canonical_rgb'][core])
    occ_p, occ_cov = t(can['canonical_xyz'][static]), t(can['canonical_source_covariance'][static])
    from scene.noise import load_config as load_noise
    raw = load_noise(noise_path)
    Qc = t(raw.rate[6:12, 6:12]); P_xi0 = t(raw.twist_covariance)
    K = len(dataset.times); k0 = int(can['canonical_frame'])
    order = list(range(K)) if direction == 'forward' else list(range(K-1, -1, -1))
    T = torch.eye(4, dtype=dtype, device=dev); xi = torch.zeros(6, dtype=dtype, device=dev)
    P = torch.zeros((12, 12), dtype=dtype, device=dev); P[6:, 6:] = P_xi0           # pose fixed at the anchor (gauge), broad twist prior
    poses = np.zeros((K, 4, 4)); twists = np.zeros((K, 6)); covs = np.zeros((K, 12, 12)); observed = np.zeros(K, bool)
    poses[k0], twists[k0], covs[k0] = T.cpu().numpy(), xi.cpu().numpy(), P.cpu().numpy()
    manifest = {'pass': 'F2' if direction == 'forward' else 'B2', 'direction': direction, 'canonical_frame': k0, 'processing_order': order,
                'coherence_policy': pol, 'core_points': int(len(core)), 'static_occluder_points': int(len(static)),
                'core_point_ids_file': str(coherence_dir/pol/'moving_core_point_ids.npy'),
                'convention': 'T = Exp(delta^) T_hat (spatial left), xi = [v (m/s), omega (rad/s)], x_i = R p_i + t, T_anchor = I',
                'process_density_Qc': raw.rate[6:12, 6:12].tolist(), 'initial_twist_covariance': raw.twist_covariance.tolist(),
                'initial_twist_mean': [0.]*6, 'covariance_chart_policy': 'final-chart (no reset)', 'precision': precision, 'pixel_reuse_assignment': assignment,
                'free_trajectories_used_for': 'core membership only (no free state enters the rigid measurement)'}
    json_write(root/'run_manifest.json', serial(manifest))
    timeline, t_all = [], time.perf_counter()
    for i, (prev, k) in enumerate(zip(order, order[1:])):
        t0 = time.perf_counter(); timers = Timers(device)
        with timers('transfer_images'):
            frame = MeasuredFrame(dataset, k, device)
        dt = float(dataset.times[k]-dataset.times[prev])
        T_m, xi_m, P_m, F, Q = rigid_predict(T, xi, P, dt, Qc)
        meas = RigidMeasurement(rig, frame, noise, policy, core_p, core_rgb, occ_p, occ_cov, dtype, assignment=assignment)
        with timers('rigid_iekf'):
            out = rigid_update(T_m, xi_m, P_m, meas, max_iters, tol, trace=k in debug_frames)
        T, xi, P = out['T'], out['xi'], out['P']
        poses[k], twists[k], covs[k], observed[k] = T.cpu().numpy(), xi.cpu().numpy(), P.cpu().numpy(), out['observed']
        fin = out['final']
        counts = torch.stack([torch.bincount(fin['reason'][c].long(), minlength=len(REASONS)) for c in range(rig.count)]).cpu().numpy()
        rig_chk = rigidity_check(core_p, T)
        from estimator.free_projective_run import quantiles
        num = {'frame': k, 'processing_step': i+1, 'timestamp_s': float(dataset.times[k]), 'dt_s': dt, 'sources': frame.sources,
               'T': T.tolist(), 'xi_v_m_s_omega_rad_s': xi.tolist(), 'P': P.tolist(), 'pose_sigma_sqrt_diag': P[:6, :6].diagonal().sqrt().tolist(),
               'observed': out['observed'], 'converged': out['converged'], 'iterations': out['iterations'], 'history': out['history'],
               'nis_prior': out['nis_prior'], 'rows_accepted': fin['rows'], 'core_points_measured': int((fin['nvalid_points'] > 0).sum()),
               'per_camera_reason_counts': [{REASONS[j]: int(c[j]) for j in range(len(REASONS)) if c[j]} for c in counts],
               'depth_innovation_m_quantiles_final': quantiles(fin['r_valid'][:, 3]) if len(fin['r_valid']) else None,
               'rgb_innovation_quantiles_final': quantiles(fin['r_valid'][:, :3]) if len(fin['r_valid']) else None,
               'rigidity': rig_chk, 'F': F.tolist(), 'Q': Q.tolist()}
        times = timers.resolve(); num['processing'] = {'wall_ms': 1e3*(time.perf_counter()-t0), **times}
        json_write(root/'numeric'/f'frame_{k:04d}.json', serial(num))
        if 'trace' in out: json_write(root/'numeric'/f'trace_frame_{k:04d}.json', serial(out['trace']))
        timeline.append({'frame': k, 'wall_ms': num['processing']['wall_ms'], 'rows': fin['rows'], 'observed': out['observed'], 'converged': out['converged'],
                         'iterations': out['iterations'], 'translation_m': float(T[:3, 3].norm()), 'rotation_rad': float(se3.so3_log(T[:3, :3]).norm()),
                         'speed_m_s': float(xi[:3].norm()), 'omega_rad_s': float(xi[3:].norm()), 'pose_sigma_max': max(num['pose_sigma_sqrt_diag']),
                         'rigidity_error_m': rig_chk['max_pair_distance_error_m'], 'nis_per_dof': (out['nis_prior']['value']/max(out['nis_prior']['dof'], 1)) if out['nis_prior'] else None})
        if progress: progress(direction, i+1, K-1, timeline[-1])
    np.save(root/'poses.npy', poses); np.save(root/'twists.npy', twists); np.save(root/'covariance.npy', covs); np.save(root/'observed.npy', observed)
    np.save(root/'core_point_ids.npy', core); np.save(root/'static_occluder_point_ids.npy', static)
    import csv
    with open(root/'numeric'/'timeline.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(timeline[0])); w.writeheader(); w.writerows(timeline)
    st = {'pass': manifest['pass'], 'status': 'COMPLETE', 'frames': K-1, 'unobserved_frames': int((~observed).sum()-1), 'core_points': int(len(core)),
          'not_converged_frames': int(sum(not r['converged'] for r in timeline)), 'seconds': time.perf_counter()-t_all,
          'max_rigidity_error_m': max(r['rigidity_error_m'] for r in timeline)}
    json_write(root/'status.json', st)
    return st
