"""Joint stage: pinned pose-space reduction, fixed-geometry (q, qdot) image IEKF, prequential predictive decision.

1. RigEuler3DGS 481b2cf2 rigid_trajectory_reduction.reduce_trajectory on the chronological Phase-2 poses
   (endpoint gauge T_anchor = I) with their pose covariances: relative twists, gamma_1dof, P and R fits.
   Reported as the joint geometry of the direction.
2. Prequential refinement per candidate type (contract section 9: "only then image refinement"): the pinned
   Phase-3 (q, qdot) IEKF (process [[1,h],[0,1]], sigma_qddot^2 [h^3/3, h^2/2; h^2/2, h]; the pinned small-state
   frame_step) on the SAME whole-scene rendered RGB-D objective and images, with the joint geometry at frame k
   fitted by reduce_trajectory on the Phase-2 poses PROCESSED BEFORE k only (causal), so that a model is scored on
   images its geometry has not seen. The first `burn_in` processed frames (geometry from the burn-in prefix) are
   not scored.
3. Decision (configs/image_joint_policy.json): one-step predictive image NLLs per scored frame of the null, the
   free rigid body (Phase 2), the prismatic and the revolute joint are compared with Diebold-Mariano tests across
   frames (predictive_tests.py; one loss per frame, pixels are never treated as independent):
     - no 1-DOF model predicts better than the PERSISTENCE null  -> NO_IDENTIFIABLE_JOINT
     - P vs R significant                                        -> the better type
     - otherwise                                                 -> UNRESOLVED_INDISTINGUISHABLE
   The null is persistence: frame k predicted with the moving core at its previous Phase-2 pose. It absorbs any
   constant displacement of the core (model error of the canonical catalogue), so only systematic motion can
   out-predict it. Measured on the static control: against the bare static scene, a noise-selected core displaced
   by a constant pose wins every frame (~145 nats/frame, p ~ 1e-28) without any motion. The static scene comparison
   is reported as a diagnostic.
   The free rigid body significantly out-predicting the selected type is reported as ONE_DOF_INADEQUATE (also
   NO_IDENTIFIABLE_JOINT). The far-axis ratio of the revolute fit is reported, not thresholded; there is no
   displacement/excitation threshold. No ground truth is read.
"""
import json
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .rigeuler_ref.rigid_trajectory_reduction import reduce_trajectory
from .rigid_image_iekf import small_state_step, RigidConfig, matrix
from .rigeuler_ref.manifold import so3_exp
from .rgbd_point_renderer import CameraTensors, render
from .image_observation import FrameObservation
from .image_phase1 import save_images
from .predictive_tests import diebold_mariano
from .group_visualization import serial
from .store import json_write

POLICY = Path(__file__).resolve().parents[1]/'configs/image_joint_policy.json'
TYPES = ('PRISMATIC', 'REVOLUTE')


def joint_transform(model, q):
    e = torch.as_tensor(model['axis'], dtype=torch.float64)
    if model['type'] == 'PRISMATIC': return matrix(torch.eye(3, dtype=torch.float64), q*e)
    R = so3_exp((q*e).reshape(1, 3))[0]; a = torch.as_tensor(model['axis_point'], dtype=torch.float64)
    return matrix(R, a-R@a)


def reduce(poses, cov6, canonical, members, times, frames, endpoint):
    """reduce_trajectory on the chronological subset `frames` (contains the endpoint); q gauge q(endpoint) = 0."""
    idx = sorted(frames)
    T = torch.as_tensor(poses[idx]); P6 = torch.as_tensor(cov6[idx]).clone()
    red = reduce_trajectory(T[:, :3, :3], T[:, :3, 3], torch.as_tensor(canonical), torch.as_tensor(members), P_tau=P6,
                            times=torch.as_tensor(np.asarray(times)[idx]))
    e = idx.index(endpoint)
    s, th = np.asarray(red['prismatic']['s'], float), np.asarray(red['revolute']['theta'], float)
    return red, {'PRISMATIC': {'type': 'PRISMATIC', 'axis': list(map(float, red['prismatic']['direction'])), 'q': dict(zip(idx, s-s[e]))},
                 'REVOLUTE': {'type': 'REVOLUTE', 'axis': list(map(float, red['revolute']['axis'])),
                              'axis_point': list(map(float, red['revolute']['axis_point'])), 'q': dict(zip(idx, th-th[e]))}}


def refine(dataset, config, obs_policy, geometry, q0, order, canonical, rgb, rho, idt, cams, policy, root, sigma_qddot=1., cfg=None):
    """Fixed type; per-frame (q, qdot) IEKF on the rendered images with the geometry `geometry[k]` used at frame k."""
    cfg = cfg or RigidConfig()
    root = Path(root); (root/'frames').mkdir(parents=True, exist_ok=True); (root/'images').mkdir(exist_ok=True)
    times = np.asarray(dataset.times); dev = rgb.device; K = len(times)
    q = np.zeros(K); qd = np.zeros(K); predictive = {}; posterior_nll = {}
    endpoint, nxt = order[0], order[1]
    x = (torch.tensor([0.], dtype=torch.float64), torch.tensor([(q0[nxt]-q0[endpoint])/(times[nxt]-times[endpoint])], dtype=torch.float64))
    P = torch.diag(torch.tensor([1e-4, 1.], dtype=torch.float64))
    for prev, k in tqdm(list(zip(order, order[1:])), desc=f'{geometry[order[1]]["type"]} fixed-joint image IEKF', unit='frame'):
        model = geometry[k]
        e = torch.as_tensor(model['axis'], dtype=torch.float64); a = torch.as_tensor(model.get('axis_point', [0, 0, 0]), dtype=torch.float64)

        def scene(xx, model=model):
            T = joint_transform(model, float(xx[0][0])).to(dev); xyz = canonical.clone()
            xyz[idt] = canonical[idt]@T[:3, :3].T+T[:3, 3]; return xyz

        def motion(xx, model=model, e=e, a=a):
            T = joint_transform(model, float(xx[0][0])); xs = canonical[idt].cpu()@T[:3, :3].T+T[:3, 3]
            d = e.expand(len(xs), 3) if model['type'] == 'PRISMATIC' else torch.linalg.cross(e.expand(len(xs), 3), xs-a)
            return idt, d[:, :, None].clone()
        dt = float(times[k]-times[prev]); ad = abs(dt)
        F = torch.tensor([[1., dt], [0., 1.]], dtype=torch.float64)
        Q = sigma_qddot**2*torch.tensor([[ad**3/3, dt*ad/2], [dt*ad/2, ad]], dtype=torch.float64)
        xm = (x[0]+dt*x[1], x[1].clone()); Pm = F@P@F.T+Q; Lam = torch.linalg.inv((Pm+Pm.T)/2)
        pe = lambda xx, xm=xm: torch.cat([xx[0]-xm[0], xx[1]-xm[1]])
        frames = [dataset.observation(k, c) for c in range(len(cams))]
        x, P, lin, terms, rep = small_state_step(xm, Lam, scene, lambda xx, d: (xx[0]+d[:1], xx[1]+d[1:]), pe,
                                                 lambda xx: torch.eye(2, dtype=torch.float64), motion,
                                                 FrameObservation(frames, config, obs_policy, dev), cams, rgb, rho, policy, cfg)
        q[k], qd[k] = float(x[0][0]), float(x[1][0])
        predictive[k] = rep['energy_initial']; posterior_nll[k] = terms['energy']
        if k % 5 == 0 or k == order[-1]: save_images(root/'images'/f'{k:06d}.npz', frames, lin)
        rep.update(frame=k, q=q[k], qdot=qd[k], geometry={kk: v for kk, v in model.items() if kk != 'q'}, predictive_image_nll=predictive[k])
        json_write(root/'frames'/f'{k:06d}.json', serial(rep))
    np.save(root/'q.npy', q); np.save(root/'qdot.npy', qd)
    return {'q': q, 'qdot': qd, 'predictive_image_nll': predictive, 'posterior_image_nll': posterior_nll, 'sigma_qddot': sigma_qddot}


def run(dataset, config, backend, partition, rho, rigid, endpoint, root, policy, obs_policy, burn_in=3):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    pol = json.loads(POLICY.read_text()); alpha = pol['tail_probability']
    K = len(dataset.times)
    order = list(range(K)) if endpoint == 0 else list(range(K-1, -1, -1))
    poses, cov6 = np.asarray(rigid['poses']), np.asarray(rigid['covariance'])[:, :6, :6]
    members, canonical_np = np.asarray(rigid['members']), np.asarray(partition.xyz)
    red, full = reduce(poses, cov6, canonical_np, members, dataset.times, order, endpoint)
    # causal geometry: frame order[j] uses the poses of order[:j]; the burn-in frames share the burn-in prefix fit
    geometry = {t_: {} for t_ in TYPES}; q0 = {t_: {} for t_ in TYPES}
    _, burn = reduce(poses, cov6, canonical_np, members, dataset.times, order[:burn_in+1], endpoint)
    for j, k in enumerate(order[1:], start=1):
        if j <= burn_in: g = burn
        else: _, g = reduce(poses, cov6, canonical_np, members, dataset.times, order[:j], endpoint)
        for t_ in TYPES: geometry[t_][k] = g[t_]
    for t_ in TYPES: q0[t_] = burn[t_]['q']
    dev = backend.device; t = backend.tensor
    canonical, rgb, rho_t = t(canonical_np), t(partition.rgb), t(rho)
    idt = torch.as_tensor(members, device=dev)
    cams = [CameraTensors(c, dev) for c in dataset.cameras]
    scored = order[burn_in+1:]
    rigid_root = Path(rigid['root'])
    L = {'RIGID': [json.loads((rigid_root/'frames'/f'{k:06d}.json').read_text())['energy_initial'] for k in scored]}
    lin = render(canonical, rgb, rho_t, cams, policy)
    obs_of = lambda k: FrameObservation([dataset.observation(k, c) for c in range(len(cams))], config, obs_policy, dev)
    L['STATIC'] = [obs_of(k).terms(lin)['energy'] for k in scored]
    prev = {k: order[j-1] for j, k in enumerate(order) if j}

    def at_pose(T):
        T = torch.as_tensor(T, device=dev); xyz = canonical.clone(); xyz[idt] = canonical[idt]@T[:3, :3].T+T[:3, 3]; return xyz
    L['PERSISTENCE'] = [obs_of(k).terms(render(at_pose(poses[prev[k]]), rgb, rho_t, cams, policy))['energy'] for k in scored]
    refined = {}
    for t_ in TYPES:
        refined[t_] = refine(dataset, config, obs_policy, geometry[t_], q0[t_], order, canonical, rgb, rho_t, idt, cams, policy, root/t_.lower())
        L[t_] = [refined[t_]['predictive_image_nll'][k] for k in scored]
    tests = {'P_vs_R': diebold_mariano(L['PRISMATIC'], L['REVOLUTE'], 'two-sided'),
             **{f'PERSISTENCE_vs_{t_}': diebold_mariano(L['PERSISTENCE'], L[t_], 'greater') for t_ in TYPES},
             'PERSISTENCE_vs_RIGID': diebold_mariano(L['PERSISTENCE'], L['RIGID'], 'greater'),
             **{f'STATIC_vs_{t_}': diebold_mariano(L['STATIC'], L[t_], 'greater') for t_ in (*TYPES, 'RIGID')}}
    pr = tests['P_vs_R']
    best = 'PRISMATIC' if pr['mean_differential'] < 0 else 'REVOLUTE'
    moving = {t_: tests[f'PERSISTENCE_vs_{t_}']['p_value'] < alpha for t_ in TYPES}
    if not any(moving.values()):
        status, selected, reason = 'NO_IDENTIFIABLE_JOINT', None, 'no 1-DOF joint predicts the images better than persistence (no systematic motion)'
    elif pr['p_value'] < alpha and moving[best]:
        status, selected, reason = best, best, f'{best} predicts better (DM p = {pr["p_value"]:.3g} < {alpha})'
    elif sum(moving.values()) == 1:
        only = next(t_ for t_ in TYPES if moving[t_])
        status, selected, reason = 'UNRESOLVED_INDISTINGUISHABLE', None, (f'P and R not predictively distinguishable (p = {pr["p_value"]:.3g}); '
                                                                          f'only {only} beats persistence')
    else:
        status, selected, reason = 'UNRESOLVED_INDISTINGUISHABLE', None, f'P and R not predictively distinguishable (p = {pr["p_value"]:.3g})'
    adequacy = None
    if selected is not None:
        adequacy = diebold_mariano(L[selected], L['RIGID'], 'greater')
        if adequacy['p_value'] < alpha:
            status, reason = 'NO_IDENTIFIABLE_JOINT', f'ONE_DOF_INADEQUATE: the free rigid body predicts better than {selected} (p = {adequacy["p_value"]:.3g})'
            selected = None
    X = canonical_np[members]
    cands = {}
    for t_ in TYPES:
        c = {kk: v for kk, v in full[t_].items() if kk != 'q'}
        c.update(q_pose=np.array([full[t_]['q'][k] for k in range(K)]), q=refined[t_]['q'], qdot=refined[t_]['qdot'],
                 predictive_image_nll_sum=float(np.sum(L[t_])))
        for key, qq in (('point_discrepancy_pose_q_m', c['q_pose']), ('point_discrepancy_image_q_m', c['q'])):
            d = [np.linalg.norm(X@poses[k][:3, :3].T+poses[k][:3, 3]-(X@joint_transform(c, qq[k])[:3, :3].numpy().T+joint_transform(c, qq[k])[:3, 3].numpy()), axis=1) for k in range(K)]
            c[key] = {'rms': float(np.sqrt(np.mean(np.concatenate(d)**2))), 'max': float(np.max(np.concatenate(d)))}
        cands[t_] = c
    out = {'status': status, 'selected_type': selected, 'reason': reason, 'anchor_frame': endpoint, 'policy': pol, 'burn_in_frames': burn_in,
           'scored_frames': scored, 'predictive_image_nll': L, 'tests': tests, 'adequacy_vs_free_rigid': adequacy,
           'revolute_axis_over_part_scale': red['revolute'].get('axis_radius_over_part_scale'), 'gamma_1dof': red['one_dof']['gamma_1dof'],
           'one_dof': red['one_dof'],
           'pose_space': {m: {k: v for k, v in red[m.lower()].items() if k in ('pose_residual', 'pose_residual_initial', 'translation_rank1_fraction', 'rank1_fraction', 'finite_axis_solver', 'radius_from_centroid', 'axis_radius_over_part_scale', 'max_abs_s', 'max_rotation_angle', 'residual_rotation_angle_max')} for m in ('PRISMATIC', 'REVOLUTE')},
           'candidates': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in cands.items()},
           'semantics': 'pinned reduction (reported geometry); causal-prefix geometry for the prequential (q, qdot) image IEKF; '
                        'Diebold-Mariano/Newey-West tests on one-step predictive image NLLs across frames'}
    json_write(root/'result.json', serial(out))
    out['selected'] = None if selected is None else {**out['candidates'][selected], 'anchor_frame': endpoint}
    return out
