"""Downstream of F2 / B2 (contract section 6): fixed joint geometry from ONE directional rigid trajectory, then an
optional constrained (q, qdot) projective IEKF. Neither step changes a free or rigid result.

1. Reduction (no images): rigeuler_ref.rigid_trajectory_reduction.reduce_trajectory (pinned, vendored verbatim) on the
   chronological filtered poses T_k with their 6x6 left-chart pose covariances: prismatic and revolute fits, the
   gamma_1dof structural diagnostic, far-axis ratio (reported, not thresholded).
2. Decision from motion and its uncertainty (one declared alpha, configs/image_joint_policy.json):
     static      S0 = sum_{k != anchor} Log(T_k)^T P_k^-1 Log(T_k)  ~ chi2(6 (K-1))            [no motion hypothesis]
     prismatic   E_P = sum_{k != anchor} r_k^T P_k^-1 r_k, r_k the pinned fit's left residual, ~ chi2(6 (K-1) - 2 - (K-1))
     revolute    E_R likewise ~ chi2(6 (K-1) - 4 - (K-1))
     nested      E_P - E_R ~ chi2(2) (prismatic = infinite-radius limit of revolute)
   STATIC if S0 is not rejected; PRISMATIC if E_P is adequate and the revolute generalization is not significantly
   better; REVOLUTE if E_R is adequate and prismatic is rejected or significantly worse; otherwise UNIDENTIFIABLE.
   The anchor frame (pose I by definition, zero covariance) is excluded. Approximation, stated: the filtered pose
   marginals are correlated across frames and the residuals are treated as independent.
3. Axis constancy over time: every increment Delta T_k = T_{k+1} T_k^-1 gives zeta_k = Log(Delta T_k)/dt; for a revolute
   joint the instantaneous screw axis (direction omega/|omega|, point omega x v/|omega|^2) is compared with the fitted
   line; for a prismatic joint the translation direction with the fitted direction. Reported per increment.
4. Optional (q, qdot) IEKF (--joint-q-filter) for an accepted type: process F = [[1, dt], [0, 1]], Q = q_d [[tau^3/3,
   sgn tau^2/2], [sgn tau^2/2, tau]]; core points x_i(q) = T_J(q) p_i (T_J(0) = I at the anchor, so no constant offset),
   T_prismatic(q) = [I, q e], T_revolute(q) = [R_e(q), (I - R_e(q)) a]; measurement = the same projective terms, chained
   by dx/dq = e (prismatic) or e x (x - a) (revolute); fixed-prior iterated update, posterior (Lambda- + H^T R^-1 H)^-1.
   The joint geometry (a, e) is held fixed: its uncertainty is not propagated (stated approximation).
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from rigeuler_ref.rigid_trajectory_reduction import reduce_trajectory
from estimator import se3
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise
from estimator.rigid_kalman_cuda import chain_measurement
from estimator.point_kalman_cuda import spd_factor
from utils.io import json_write, serial


def joint_transform(kind, axis, point, q):
    e = np.asarray(axis, float); e = e/np.linalg.norm(e)
    T = np.eye(4)
    if kind == 'PRISMATIC':
        T[:3, 3] = q*e
    else:
        R = se3.exp(torch.tensor(np.r_[0, 0, 0, q*e], dtype=torch.float64)).numpy()[:3, :3]
        T[:3, :3] = R; T[:3, 3] = (np.eye(3)-R)@np.asarray(point, float)
    return T


def reduce_and_decide(poses, covs, times, canonical_core, anchor, alpha):
    K = len(poses)
    # the pinned reduction takes index 0 as its gauge (T_0 = I, theta_0 = 0): feed the trajectory in PROCESSING order
    # (anchor first) with elapsed physical time from the anchor, then map every per-frame output back to chronology
    order = list(range(K)) if anchor == 0 else list(range(K-1, -1, -1))
    inv = np.argsort(order)
    T = torch.as_tensor(poses[order]); P6 = torch.as_tensor(covs[order, :6, :6]).clone()
    tt = np.abs(np.asarray(times, float)[order]-float(times[anchor]))
    red = reduce_trajectory(T[:, :3, :3], T[:, :3, 3], torch.as_tensor(canonical_core), torch.arange(len(canonical_core)), P_tau=P6,
                            times=torch.as_tensor(tt))
    for key in ('prismatic', 'revolute'):
        red[key]['pose_residual']['per_frame_left'] = np.asarray(red[key]['pose_residual']['per_frame_left'])[inv].tolist()
    red['prismatic']['s'] = np.asarray(red['prismatic']['s'])[inv].tolist(); red['revolute']['theta'] = np.asarray(red['revolute']['theta'])[inv].tolist()
    red['processing_order_used'] = order
    m = K-1
    S0 = 0.
    for k in range(K):
        if k == anchor: continue
        e = se3.log(T[k]).numpy()
        S0 += float(e@np.linalg.solve(covs[k, :6, :6], e))
    out = {'static': {'statistic': S0, 'dof': 6*m, 'p_value': float(stats.chi2.sf(S0, 6*m))}}
    for kind, key, npar in (('PRISMATIC', 'prismatic', 2), ('REVOLUTE', 'revolute', 4)):
        res = np.asarray(red[key]['pose_residual']['per_frame_left'], float)      # left-chart residual Log(T_k T_model^-1)
        # the anchor pose is I by definition (gauge, zero covariance): it carries no information and is excluded;
        # the model's residual there is the reference-pose offset, reported
        E = float(sum(res[k]@np.linalg.solve(covs[k, :6, :6], res[k]) for k in range(K) if k != anchor))
        dof = 6*m-npar-m
        out[kind] = {'energy_excluding_anchor': E, 'dof': dof, 'p_value': float(stats.chi2.sf(E, dof)),
                     'anchor_offset_residual': res[anchor].tolist(),
                     'rms_translation_m': red[key]['pose_residual']['rms_translation'], 'rms_rotation_rad': red[key]['pose_residual']['rms_rotation']}
    # nested comparison: a prismatic joint is the infinite-radius limit of a revolute one with 2 fewer parameters
    lr = out['PRISMATIC']['energy_excluding_anchor']-out['REVOLUTE']['energy_excluding_anchor']
    out['revolute_vs_prismatic_likelihood_ratio'] = {'statistic': lr, 'dof': 2, 'p_value': float(stats.chi2.sf(max(lr, 0.), 2))}
    if out['static']['p_value'] > alpha:
        status, reason = 'STATIC', 'no motion detected against the pose uncertainty (identity not rejected)'
    else:
        okP, okR = out['PRISMATIC']['p_value'] > alpha, out['REVOLUTE']['p_value'] > alpha
        rev_better = out['revolute_vs_prismatic_likelihood_ratio']['p_value'] <= alpha
        if okP and not rev_better: status, reason = 'PRISMATIC', 'prismatic adequate; the revolute generalization is not significantly better (nested LR test)'
        elif okR and (not okP or rev_better): status, reason = 'REVOLUTE', 'revolute adequate and prismatic rejected or significantly worse'
        else: status, reason = 'UNIDENTIFIABLE', 'neither one-DOF model adequate against the rigid pose uncertainty'
    out['degeneracy'] = {'revolute_axis_radius_over_part_scale': red['revolute'].get('axis_radius_over_part_scale'),
                         'note': 'a large ratio means the revolute fit approximates a translation with a far axis (reported, not thresholded)'}
    s = np.asarray(red['prismatic']['s'], float); th = np.asarray(red['revolute']['theta'], float)
    cand = {'PRISMATIC': {'axis': list(map(float, red['prismatic']['direction'])), 'axis_point': None, 'q': (s-s[anchor]).tolist()},
            'REVOLUTE': {'axis': list(map(float, red['revolute']['axis'])), 'axis_point': list(map(float, red['revolute']['axis_point'])),
                         'q': (th-th[anchor]).tolist()}}
    for kind, c in cand.items():
        q = np.asarray(c['q']); tt = np.asarray(times, float)
        c['qdot'] = np.gradient(q, tt).tolist()
        c['reference_pose_offset'] = 'identity: T_J(0) = I at the anchor frame (canonical gauge); residual of the fitted model at the anchor reported in the pinned pose residuals'
    return red, out, status, reason, cand


def axis_constancy(poses, times, cand, anchor):
    out = {'PRISMATIC': [], 'REVOLUTE': []}
    e_p = np.asarray(cand['PRISMATIC']['axis']); e_r = np.asarray(cand['REVOLUTE']['axis']); a_r = np.asarray(cand['REVOLUTE']['axis_point'])
    for k in range(len(poses)-1):
        dt = float(times[k+1]-times[k])
        z = se3.log(torch.as_tensor(poses[k+1]@np.linalg.inv(poses[k]))).numpy()/dt
        v, w = z[:3], z[3:]
        rp = {'k': k, 'speed_m_s': float(np.linalg.norm(v)), 'omega_rad_s': float(np.linalg.norm(w))}
        out['PRISMATIC'].append({**rp, 'direction_angle_deg': float(np.degrees(np.arccos(min(1., abs(v@e_p)/max(np.linalg.norm(v), 1e-300)))))})
        if np.linalg.norm(w) > 0:
            pt = np.cross(w, v)/(w@w)
            d = pt-a_r; dist = float(np.linalg.norm(d-(d@e_r)*e_r))
            ang = float(np.degrees(np.arccos(min(1., abs(w@e_r)/np.linalg.norm(w)))))
        else:
            dist = ang = None
        out['REVOLUTE'].append({**rp, 'axis_angle_deg': ang, 'axis_line_distance_m': dist})
    summ = {}
    for kind, key in (('PRISMATIC', 'direction_angle_deg'), ('REVOLUTE', 'axis_angle_deg')):
        vals = np.array([r[key] for r in out[kind] if r[key] is not None], float)
        summ[kind] = {'median_deg': float(np.median(vals)) if len(vals) else None,
                      'max_deg': float(vals.max()) if len(vals) else None, 'increments': int(len(vals))}
    return out, summ


def q_filter(dataset, config, policy, rig, noise, kind, axis, point, core_p, core_rgb, occ_p, occ_cov, anchor, q_var0, qd, device,
             max_iters=10, tol=1e-10):
    e = torch.as_tensor(np.asarray(axis, float)/np.linalg.norm(axis), dtype=torch.float64, device=device)
    a = torch.as_tensor(np.zeros(3) if point is None else np.asarray(point, float), dtype=torch.float64, device=device)
    K = len(dataset.times)
    order = list(range(K)) if anchor == 0 else list(range(K-1, -1, -1))
    s = torch.zeros(2, dtype=torch.float64, device=device); P = torch.diag(torch.tensor([0., q_var0], dtype=torch.float64, device=device))
    out = {anchor: {'q': 0., 'qdot': 0., 'P': P.tolist(), 'rows': 0}}
    def positions(q):
        T = torch.as_tensor(joint_transform(kind, axis, point, float(q)), dtype=torch.float64, device=device)
        x = core_p@T[:3, :3].T+T[:3, 3]
        dxdq = e.expand(len(x), 3) if kind == 'PRISMATIC' else torch.linalg.cross(e.expand(len(x), 3), x-a)
        return x, dxdq[:, :, None]
    for prev, k in zip(order, order[1:]):
        dt = float(dataset.times[k]-dataset.times[prev]); tau, sg = abs(dt), math.copysign(1., dt)
        F = torch.tensor([[1., dt], [0., 1.]], dtype=torch.float64, device=device)
        Q = qd*torch.tensor([[tau**3/3, sg*tau**2/2], [sg*tau**2/2, tau]], dtype=torch.float64, device=device)
        s_m, P_m = F@s, F@P@F.T+Q
        Lam = torch.linalg.inv(P_m); Lam = (Lam+Lam.T)/2
        frame = MeasuredFrame(dataset, k, device)
        sj = s_m.clone(); conv = False
        x_prior = positions(s_m[0])[0]                          # row gate on the prior-linearized innovation
        for j in range(max_iters):
            x, G = positions(sj[0])
            m = chain_measurement(rig, frame, noise, policy, x, G, P_m[:1, :1], core_rgb, occ_p, occ_cov, dx=x-x_prior)
            A = Lam.clone(); A[0, 0] += m['I'][0, 0]
            b = -(Lam@(sj-s_m)); b[0] += m['g'][0]
            delta = torch.linalg.solve(A, b); step = float(delta@b)
            sj = sj+delta
            if step < tol: conv = True; break
        x, G = positions(sj[0])
        m = chain_measurement(rig, frame, noise, policy, x, G, P_m[:1, :1], core_rgb, occ_p, occ_cov, dx=x-x_prior)
        A = Lam.clone(); A[0, 0] += m['I'][0, 0]
        if m['rows'] > 0: s, P = sj, torch.linalg.inv(A)
        else: s, P = s_m, P_m
        P = (P+P.T)/2
        out[k] = {'q': float(s[0]), 'qdot': float(s[1]), 'P': P.tolist(), 'rows': m['rows'], 'converged': conv, 'iterations': j+1}
    return [out[k] for k in range(K)]


def run_joint(dataset, config, policy, direction, free_dir, coherence_dir, rigid_dir, root, alpha, device, q_filter_on=True, noise_path=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    st_r = json.loads((Path(rigid_dir)/'status.json').read_text())
    if st_r['status'] != 'COMPLETE':
        st = {'status': 'SKIPPED_NO_RIGID_TRAJECTORY', 'rigid_status': st_r['status']}; json_write(root/'result.json', st); return st
    can = np.load(Path(free_dir)/'canonical.npz')
    poses, covs = np.load(Path(rigid_dir)/'poses.npy'), np.load(Path(rigid_dir)/'covariance.npy')
    core = np.load(Path(rigid_dir)/'core_point_ids.npy'); static = np.load(Path(rigid_dir)/'static_occluder_point_ids.npy')
    anchor = int(can['canonical_frame'])
    t0 = time.perf_counter()
    red, tests, status, reason, cand = reduce_and_decide(poses, covs, dataset.times, can['canonical_xyz'][core], anchor, alpha)
    incr, const = axis_constancy(poses, dataset.times, cand, anchor)
    result = {'direction': direction, 'anchor_frame': anchor, 'status': status, 'reason': reason, 'alpha': alpha, 'tests': tests,
              'candidates': cand, 'gamma_1dof': red['one_dof']['gamma_1dof'], 'one_dof_note': red['one_dof']['note'],
              'revolute_axis_radius_over_part_scale': red['revolute'].get('axis_radius_over_part_scale'), 'part_scale_L': red['part_scale_L'],
              'axis_constancy_summary': const, 'decision_note': 'chi-square adequacy with the filtered pose marginals treated as independent across frames (approximation)',
              'renderer_called': False, 'ground_truth_used': False}
    json_write(root/'reduction_increments.json', serial(incr))
    if q_filter_on and status in ('PRISMATIC', 'REVOLUTE'):
        from scene.noise import load_config as load_noise
        raw = load_noise(noise_path)
        blk = slice(6, 9) if status == 'PRISMATIC' else slice(9, 12)
        qd = float(np.mean(np.diag(raw.rate)[blk])); q_var0 = float(np.mean(np.diag(raw.twist_covariance)[blk.start-6:blk.stop-6]))
        rig = CameraRig(dataset.cameras, device); noise = SensorNoise.from_config(config, policy, device)
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
        qf = q_filter(dataset, config, policy, rig, noise, status, cand[status]['axis'], cand[status]['axis_point'], tt(can['canonical_xyz'][core]),
                      tt(can['canonical_rgb'][core]), tt(can['canonical_xyz'][static]), tt(can['canonical_source_covariance'][static]),
                      anchor, q_var0, qd, device)
        result['q_filter'] = {'type': status, 'q': [r['q'] for r in qf], 'qdot': [r['qdot'] for r in qf], 'q_sigma': [math.sqrt(max(r['P'][0][0], 0)) for r in qf],
                              'rows': [r['rows'] for r in qf], 'not_converged_frames': int(sum(not r.get('converged', True) for r in qf)),
                              'process_density': qd, 'initial_qdot_variance': q_var0,
                              'approximation': 'joint geometry (a, e) fixed; its uncertainty is not propagated into the q filter'}
    result['seconds'] = time.perf_counter()-t0
    json_write(root/'result.json', serial(result))
    return result


def compare_directions(jf, jb, scale_hint=None):
    """Forward vs backward in the common calibrated world frame: no point ids, no fused posterior."""
    from joint_reduction.bidirectional import line_distance
    out = {'forward_status': jf.get('status'), 'backward_status': jb.get('status'), 'per_type': {}}
    for kind in ('PRISMATIC', 'REVOLUTE'):
        cf, cb = (jf.get('candidates') or {}).get(kind), (jb.get('candidates') or {}).get(kind)
        if not cf or not cb: continue
        ef, eb = np.asarray(cf['axis']), np.asarray(cb['axis']); sgn = 1. if ef@eb >= 0 else -1.
        qf, qb = np.asarray(cf['q']), np.asarray(cb['q'])
        qb_rel = sgn*(qb-qb[0])                      # chronological displacement relative to frame 0
        d = {'axis_angle_deg_up_to_sign': float(np.degrees(np.arccos(min(1., abs(ef@eb))))), 'axis_sign': sgn,
             'q_forward_minus_backward_rmse': float(np.sqrt(np.mean((qf-qb_rel)**2))), 'q_max_abs_difference': float(np.abs(qf-qb_rel).max())}
        if kind == 'REVOLUTE': d['axis_line_distance_m'] = line_distance(np.asarray(cf['axis_point']), ef, np.asarray(cb['axis_point']), eb)
        out['per_type'][kind] = d
    out['note'] = 'the two directions use different endpoint clouds and the same images: compared, never fused'
    return out
