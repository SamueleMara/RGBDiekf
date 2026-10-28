"""Phase 2: ONE shared rigid body, temporal SE(3)+twist IEKF on the SAME rendered RGB-D images.

Port of RigEuler3DGS 481b2cf2 rigid_part_runtime (RigidState, kabsch_initialize,
process_matrix, process_noise, run_rigid_window) and joint_iekf.frame_step /
_FramePrior, with the Gaussian renderer replaced by rgbd_point_renderer.

State per frame: (T_k in SE(3), xi_k in R^6), xi = [v; omega] spatial twist in
PHYSICAL (chronological) time. Error [delta_tau; delta_xi]: left pose increment
T = Exp(delta_tau^) T_hat, xi = xi_hat + delta_xi. Process with signed dt
(negative when processing backward):

    E = Exp(dt xi^),  T- = E T+,  xi- = xi+
    F = [[Ad_E, dt J_l(dt xi)], [0, I]],  P- = F P+ F^T + Q(dt)
    Q = s^2 [[|dt|^3/3, dt|dt|/2], [dt|dt|/2, |dt|]] per channel (white acceleration)

Measurement: the WHOLE endpoint catalog is rendered into every camera: moving-core
points at T_k p_i (T_anchor = I), every other point at identity (static
hypothesis, for visibility and scene pixels, not a background). For moving points
delta x_i = [I, -[x_i]x] delta_tau; the twist has no direct image derivative.
Inner iteration (fixed x-, Lambda-):

    e = [Log(T T-^-1); xi - xi-],  J_e = d e(x [+] d)/dd = blockdiag(J_l(e_tau)^-1, I)
    A = J_e^T Lambda- J_e + H^T W H,  b = H^T W r - J_e^T Lambda- e

solved by the pinned joint_iekf.frame_step (Steihaug-CG, prior-information metric, dense probe preconditioner,
adapt_radius + true decrease on a fresh render, 8 iterations). The observation model is Phase 1's (no model-error
map in estimation). Posterior once per frame at the accepted state (left chart): Joseph form of the linear update.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from rigeuler_ref.joint_manifold import se3_exp, se3_log, se3_adjoint, se3_left_jacobian
from rigeuler_ref.trust_region import ScalingMetric, TrustRegionConfig, adapt_radius, steihaug_cg
from estimator.point_renderer import CameraTensors, render
from estimator.observation_terms import FrameObservation, estimate_model_error
from estimator.phase1 import save_images
from utils.io import serial
from utils.io import json_write


@dataclass
class RigidConfig:
    """Pinned rigid_part_runtime.RigidIEKFConfig defaults."""
    sigma_vdot: float = 1.
    sigma_wdot: float = 1.
    sigma_tau0: float = 1e-2
    sigma_xi0: float = 1.
    max_iters: int = 8
    max_cg_iters: int = 40
    radius_init: float = 1.
    psd_floor: float = 1e-12


def hat(x):
    return torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=torch.float64)


def matrix(R, t):
    T = torch.eye(4, dtype=torch.float64); T[:3, :3] = R; T[:3, 3] = t
    return T


def exp4(eta):
    R, t = se3_exp(eta.reshape(1, 6)); return matrix(R[0], t[0])


def log4(T):
    return se3_log(T[:3, :3].reshape(1, 3, 3), T[:3, 3].reshape(1, 3))[0]


def inv4(T):
    out = torch.eye(4, dtype=torch.float64); out[:3, :3] = T[:3, :3].T; out[:3, 3] = -T[:3, :3].T@T[:3, 3]
    return out


def weighted_kabsch(X0, Xt, w=None):
    """Pinned rigid_part_runtime.weighted_kabsch (reflection-guarded)."""
    X0, Xt = torch.as_tensor(X0).double(), torch.as_tensor(Xt).double()
    w = torch.ones(len(X0), dtype=torch.float64) if w is None else torch.as_tensor(w).double()
    W = w.sum().clamp_min(1e-300); c0 = (w[:, None]*X0).sum(0)/W; ct = (w[:, None]*Xt).sum(0)/W
    M = (w[:, None]*(Xt-ct)).T@(X0-c0)
    U, S, Vh = torch.linalg.svd(M)
    D = torch.diag(torch.tensor([1., 1., float(torch.det(U@Vh))], dtype=torch.float64))
    R = U@D@Vh
    return R, ct-R@c0, S


def process(xi, dt, cfg):
    E = exp4(dt*xi)
    F = torch.eye(12, dtype=torch.float64)
    F[:6, :6] = se3_adjoint(E[:3, :3].reshape(1, 3, 3), E[:3, 3].reshape(1, 3))[0]
    F[:6, 6:] = dt*se3_left_jacobian((dt*xi).reshape(1, 6))[0]
    s2 = torch.tensor([cfg.sigma_vdot**2]*3+[cfg.sigma_wdot**2]*3, dtype=torch.float64)
    a = abs(dt)
    Q = torch.zeros(12, 12, dtype=torch.float64)
    Q[:6, :6] = torch.diag(s2*a**3/3); Q[:6, 6:] = Q[6:, :6] = torch.diag(s2*dt*a/2); Q[6:, 6:] = torch.diag(s2*a)
    return E, F, Q


def small_state_step(x_minus, Lam, scene, retract, prior_error, prior_jacobian, motion, obs, cams, rgb, rho, policy, cfg):
    """Pinned joint_iekf.frame_step for a small state (rigid pose/twist, or joint q/qdot), one objective

        E(x) = E_meas(x) + 1/2 e(x)^T Lambda- e(x),   e = prior_error(x),  J_e = prior_jacobian(x)

    per iteration: b = J_meas^T W r - J_e^T Lambda- e, A = J_meas^T W J_meas + J_e^T Lambda- J_e (matrix-free,
    probed densely: its inverse is the CG preconditioner), Steihaug-CG in the prior-information-diagonal RMS metric
    (_metric), predicted reduction b^T s - s^T A s/2, acceptance by adapt_radius AND a true decrease of E on a fresh
    render, stop at max_iters or the radius floor. scene(x) -> all catalogue xyz; motion(x) -> (moving ids, M [m,3,d])
    with d the image-observable leading tangent block.

    Posterior at the accepted state: the reference information inverse A^-1 and the Joseph form of the same linear
    update, A^-1 (N + J^T W R W J) A^-1 (N = J_e^T Lambda- J_e), which is the error covariance when the IRLS weights
    W differ from R^-1 (Student-t); Joseph is carried."""
    dev = rgb.device
    n = Lam.shape[0]
    d_all = torch.diagonal(Lam).clamp_min(1e-12)
    metric = ScalingMetric(weight=(d_all/n).to(torch.float64))
    tr = TrustRegionConfig(radius_init=cfg.radius_init, max_pcg_iters=cfg.max_cg_iters)

    def objective(x):
        lin = render(scene(x), rgb, rho, cams, policy)
        terms = obs.terms(lin)
        e = prior_error(x); prior = .5*float(e@Lam@e)
        return prior+terms['energy'], prior, lin, terms

    def meas_normal(x, lin, weights):
        """v -> J^T W J v on the observable block, W given per camera as residual-shaped multipliers."""
        ids, M = motion(x); Md = M.to(dev)

        def op(v):
            dx = torch.zeros(lin.n, 3, dtype=torch.float64, device=dev); dx[ids] = Md@v[:M.shape[2]].to(dev)
            out = torch.zeros(n, dtype=torch.float64)
            out[:M.shape[2]] = torch.einsum('mij,mi->j', Md, lin.apply_HT(weights(lin.apply_H(dx)))[ids]).cpu()
            return out
        return op, ids, Md

    def system(x, lin, terms):
        op, ids, Md = meas_normal(x, lin, lambda hv: obs.apply_W(terms, hv))
        g = torch.zeros(n, dtype=torch.float64)
        g[:Md.shape[2]] = torch.einsum('mij,mi->j', Md, lin.apply_HT(terms['residual_adjoint'])[ids]).cpu()
        Je, e = prior_jacobian(x), prior_error(x)
        N = Je.T@Lam@Je; N = (N+N.T)/2
        return (lambda v: op(v)+N@v), g-Je.T@Lam@e, N, op

    E, prior_e, lin, terms = objective(x_minus)
    E0, x, radius, history, accepted, stop = E, x_minus, cfg.radius_init, [], 0, 'iteration_budget'
    eye = torch.eye(n, dtype=torch.float64)
    for it in range(1, cfg.max_iters+1):
        A_mv, b, N, _ = system(x, lin, terms)
        A_dense = torch.stack([A_mv(eye[:, k]) for k in range(n)], 1); A_dense = (A_dense+A_dense.T)/2
        ev = torch.linalg.eigvalsh(A_dense)
        A_pc = A_dense if float(ev.min()) > cfg.psd_floor else A_dense+(cfg.psd_floor-float(ev.min()))*eye
        A_inv = torch.linalg.inv(A_pc)
        res = steihaug_cg(A_mv, b, radius, metric, M_inv=lambda r: A_inv@r, max_iters=cfg.max_cg_iters)
        step = res.step
        if not bool(torch.isfinite(step).all()) or float(step.norm()) == 0.:
            stop = 'zero_or_nonfinite_step'; break
        pred = float(b@step-.5*step@A_mv(step))
        trial = retract(x, step)
        Et, pt, lint, termst = objective(trial)
        actual = E-Et
        rho_ = actual/pred if pred > 0 else float('-inf')
        before = radius
        radius, ok = adapt_radius(radius, rho_, res.boundary_hit, actual, tr)
        took = bool(ok and np.isfinite(Et) and Et < E)
        if took: x, E, prior_e, lin, terms = trial, Et, pt, lint, termst; accepted += 1
        history.append({'iteration': it, 'accepted': took, 'objective_trial': Et, 'predicted_reduction': pred, 'actual_reduction': actual,
                        'rho': rho_, 'radius_before': before, 'radius': radius, 'pcg_iterations': res.iterations,
                        'pcg_stop_reason': res.stop_reason, 'step_norm_scaled': res.step_norm_scaled})
        if radius <= tr.radius_min: stop = 'trust_radius_floor'; break
    A_mv, b, N, op = system(x, lin, terms)
    H_info = torch.stack([op(eye[:, k]) for k in range(n)], 1); H_info = (H_info+H_info.T)/2
    info = N+H_info; info = (info+info.T)/2
    ev = torch.linalg.eigvalsh(info)
    floored = float(ev.min()) <= cfg.psd_floor
    if floored: info = info+(cfg.psd_floor-float(ev.min()))*eye
    A_inv = torch.linalg.inv(info); A_inv = (A_inv+A_inv.T)/2
    from estimator.restricted_posterior import channel_weights
    wrw = [b_ for _, b_ in channel_weights(obs, terms)]
    op_r, _, _ = meas_normal(x, lin, lambda hv: [h*w_ for h, w_ in zip(hv, wrw)])
    R_info = torch.stack([op_r(eye[:, k]) for k in range(n)], 1); R_info = (R_info+R_info.T)/2
    P = A_inv@(N+R_info)@A_inv; P = (P+P.T)/2
    report = {'energy_initial': E0, 'energy_final': E, 'prior_energy_final': prior_e, 'image_energy_final': terms['energy'],
              'accepted_steps': accepted, 'rejected_trials': len(history)-accepted, 'stop': stop, 'history': history,
              'measurement_information_eigenvalues': torch.linalg.eigvalsh(H_info).tolist(),
              'posterior_information_eigenvalues': ev.tolist(), 'psd_floor_applied': floored,
              'posterior': 'Joseph A^-1 (N + J^T W R W J) A^-1 at the accepted state (carried); reference information inverse reported',
              'reference_information_inverse_trace_ratio': float(torch.trace(A_inv)/torch.trace(P)), 'cameras': terms['cameras']}
    return x, P, lin, terms, report


def run(dataset, config, backend, partition, arrays, rho, support, endpoint, root, policy, obs_policy, cfg=None):
    cfg = cfg or RigidConfig()
    root = Path(root); (root/'frames').mkdir(parents=True, exist_ok=True); (root/'images').mkdir(exist_ok=True)
    K = len(dataset.times); ids = np.flatnonzero(support)
    if len(ids) < 3: return {'status': 'RIGID_UNIDENTIFIABLE', 'reason': 'fewer than three moving-core points'}
    order = list(range(K)) if endpoint == 0 else list(range(K-1, -1, -1))
    times = torch.as_tensor(dataset.times, dtype=torch.float64)
    # Kabsch of the coherent core only (uniform weights), for the initial twist and the comparison.
    X = np.asarray(arrays['xyz'])
    kabsch = torch.stack([torch.eye(4, dtype=torch.float64) if k == endpoint else matrix(*weighted_kabsch(X[endpoint, ids], X[k, ids])[:2]) for k in range(K)])
    nxt = order[1]; dt0 = float(times[nxt]-times[endpoint])
    xi0 = log4(kabsch[nxt]@inv4(kabsch[endpoint]))/dt0
    dev = backend.device; t = backend.tensor
    canonical = t(partition.xyz); rgb = t(partition.rgb); rho_t = t(rho)
    idt = torch.as_tensor(ids, device=dev)
    cams = [CameraTensors(c, dev) for c in dataset.cameras]

    def scene(x):
        T = x[0].to(dev); xyz = canonical.clone(); xyz[idt] = canonical[idt]@T[:3, :3].T+T[:3, 3]
        return xyz

    def motion(x):
        T = x[0]; xs = canonical[idt].cpu()@T[:3, :3].T+T[:3, 3]
        M = torch.zeros(len(ids), 3, 6, dtype=torch.float64); M[:, :, :3] = torch.eye(3, dtype=torch.float64)
        M[:, 0, 4], M[:, 0, 5] = xs[:, 2], -xs[:, 1]          # -[x]x
        M[:, 1, 3], M[:, 1, 5] = -xs[:, 2], xs[:, 0]
        M[:, 2, 3], M[:, 2, 4] = xs[:, 1], -xs[:, 0]
        return idt, M

    def retract(x, d):
        return (exp4(d[:6])@x[0], x[1]+d[6:])

    poses = torch.zeros(K, 4, 4, dtype=torch.float64); twists = torch.zeros(K, 6, dtype=torch.float64)
    cov = torch.zeros(K, 12, 12, dtype=torch.float64); reports = []
    x = (torch.eye(4, dtype=torch.float64), xi0.clone())
    P = torch.diag(torch.tensor([cfg.sigma_tau0**2]*6+[cfg.sigma_xi0**2]*6, dtype=torch.float64))
    poses[endpoint], twists[endpoint], cov[endpoint] = x[0], x[1], P
    frames = [dataset.observation(endpoint, c) for c in range(len(cams))]
    lin = render(scene(x), rgb, rho_t, cams, policy)
    # Model mismatch DIAGNOSTIC (pinned estimate_model_error of the canonical catalogue at its own endpoint); never
    # used in estimation (contract section 3: one sensor model in every stage).
    model_error_stats = estimate_model_error(lin, frames)[1]
    obs = FrameObservation(frames, config, obs_policy, dev)
    save_images(root/'images'/f'{endpoint:06d}.npz', frames, lin)
    json_write(root/'frames'/f'{endpoint:06d}.json', serial({'frame': endpoint, 'role': 'CANONICAL_GAUGE_T_ANCHOR_I', 'image_energy': obs.terms(lin)['energy'],
                                                              'model_error': model_error_stats}))
    start = time.monotonic()
    for prev, k in tqdm(list(zip(order, order[1:])), desc='Rigid SE(3) image IEKF', unit='frame'):
        clock = time.monotonic()
        dt = float(times[k]-times[prev])                           # signed
        E, F, Q = process(x[1], dt, cfg)
        x_minus = (E@x[0], x[1].clone())
        P_minus = F@P@F.T+Q; P_minus = (P_minus+P_minus.T)/2
        Lam = torch.linalg.inv(P_minus); Lam = (Lam+Lam.T)/2
        Tm_inv = inv4(x_minus[0])

        def prior_error(xx):
            return torch.cat([log4(xx[0]@Tm_inv), xx[1]-x_minus[1]])

        def prior_jacobian(xx):
            J = torch.eye(12, dtype=torch.float64)
            J[:6, :6] = torch.linalg.inv(se3_left_jacobian(log4(xx[0]@Tm_inv).reshape(1, 6))[0])
            return J
        frames = [dataset.observation(k, c) for c in range(len(cams))]
        x, P, lin, terms, rep = small_state_step(x_minus, Lam, scene, retract, prior_error, prior_jacobian, motion,
                                                 FrameObservation(frames, config, obs_policy, dev), cams, rgb, rho_t, policy, cfg)
        poses[k], twists[k], cov[k] = x[0], x[1], P
        save_images(root/'images'/f'{k:06d}.npz', frames, lin)
        rep.update(frame=k, original_frame=dataset.frame_ids[k], signed_dt_seconds=dt, seconds=time.monotonic()-clock,
                   pose=x[0].tolist(), twist=x[1].tolist(), kabsch_pose=kabsch[k].tolist(),
                   pose_sigma=torch.sqrt(torch.diagonal(P)[:6]).tolist())
        json_write(root/'frames'/f'{k:06d}.json', serial(rep))
        reports.append({key: rep[key] for key in ('frame', 'energy_initial', 'energy_final', 'accepted_steps', 'rejected_trials', 'stop', 'seconds')})
    np.save(root/'poses.npy', poses.numpy()); np.save(root/'twists.npy', twists.numpy())
    np.save(root/'covariance.npy', cov.numpy()); np.save(root/'pose_covariance.npy', cov[:, :6, :6].numpy())
    np.save(root/'kabsch_poses.npy', kabsch.numpy()); np.save(root/'moving_core_ids.npy', ids)
    rot = lambda T: float(torch.linalg.norm(log4(T)[3:]))
    rel = [rot(poses[k]@inv4(poses[k-1])) for k in range(1, K)]
    result = {'status': 'FIT', 'anchor_frame': endpoint, 'core_points': int(len(ids)), 'frames': reports, 'runtime_seconds': time.monotonic()-start,
              'config': asdict(cfg), 'initial_twist_from_kabsch': xi0.tolist(), 'camera_policy': 'all_available', 'model_error': model_error_stats,
              'trajectory_diagnostics': {'max_rotation_rad': max(rot(T) for T in poses), 'max_frame_to_frame_rotation_rad': max(rel),
                                         'translation_range_m': float((poses[:, :3, 3]-poses[endpoint, :3, 3]).norm(dim=1).max()),
                                         'kabsch_max_rotation_rad': max(rot(T) for T in kabsch)},
              'semantics': 'ONE temporal SE(3)+twist Bayesian IEKF (left chart), whole-scene rendered RGB-D, all cameras; canonical geometry fixed (not co-optimized)'}
    json_write(root/'result.json', serial(result))
    return {**result, 'root': str(root), 'poses': poses.numpy(), 'twists': twists.numpy(), 'covariance': cov.numpy(), 'members': ids,
            'canonical': partition.xyz, 'kabsch': kabsch.numpy()}
