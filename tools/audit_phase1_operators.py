#!/usr/bin/env python3
"""STEP 2A (CLAUDE_STEP2_PHASE1_SUPERNODES_GPU_PERFORMANCE_AUDIT.md): read-only Phase-1 grouped-operator parity and
one-step IEKF audit on saved v2 forward arrays. Nothing under the run folder is written; all outputs go to --out and
reports/. Ground truth is read ONLY by the post-hoc A3 lag evaluation (never by the estimator).

A1  group basis G on the saved 537-group partition vs an independent per-group np.cross implementation; adjoint on CPU
    and CUDA; finite differences of the rigid retraction; shuffle invariance; per-group rank and condition.
A2  at saved blade frames (prior = predict(saved posterior of the previous frame)): H finite differences, adjoint,
    A_g symmetry, b_g vs the finite-difference derivative of the TRUE objective, production Steihaug vs accurate CG,
    damped vs undamped; dense oracles (A_dense = G^T(Lambda + H^T W H)G) on synthetic geometries.
A3  one-step IEKF from identical saved priors at frames 1, 2, 5, 10, 20: production vs accurate linear solve vs larger
    iteration budgets (offline counterfactual, not production), per-iteration logs and post-hoc moving-part recovery.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, torch

SOURCES = ['pointwise_rgbd/supernodes.py', 'pointwise_rgbd/image_iekf.py', 'pointwise_rgbd/image_phase1.py', 'pointwise_rgbd/rgbd_point_renderer.py',
           'pointwise_rgbd/image_observation.py', 'pointwise_rgbd/phase1_covariance.py', 'pointwise_rgbd/restricted_posterior.py', 'pointwise_rgbd/process6.py',
           'pointwise_rgbd/rigeuler_ref/trust_region.py', 'pointwise_rgbd/rigeuler_ref/pcg.py']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while (b := f.read(1 << 24)): h.update(b)
    return h.hexdigest()


def cross_np(a, b): return np.cross(a, b)


# ------------------------------------------------------------------------------------------------ A1
def audit_basis(xyz, member, centroids, devices):
    from pointwise_rgbd.supernodes import GroupBasis
    out = {}
    S = int(member.max())+1
    sizes = np.bincount(member, minlength=S)
    recomputed = np.stack([xyz[member == g].mean(0) if sizes[g] else np.full(3, np.nan) for g in range(S)])
    out['groups'] = S; out['empty_groups'] = int((sizes == 0).sum()); out['singleton_groups'] = int((sizes == 1).sum())
    out['size_quantiles'] = np.quantile(sizes, [0, .01, .5, .99, 1]).tolist()
    out['centroid_minus_member_mean_max_m'] = float(np.nanmax(np.abs(recomputed-centroids)))
    r = xyz-centroids[member]
    # per-group pose Gram sum_i P_i^T P_i, P_i = [I, -[r_i]x]
    ev = []
    for g in range(S):
        R = r[member == g]; M = np.zeros((6, 6))
        for ri in R:
            P = np.zeros((3, 6)); P[:, :3] = np.eye(3); P[:, 3:] = -np.array([[0, -ri[2], ri[1]], [ri[2], 0, -ri[0]], [-ri[1], ri[0], 0]])
            M += P.T@P
        ev.append(np.linalg.eigvalsh(M))
    ev = np.array(ev); cond = ev[:, -1]/np.maximum(ev[:, 0], 1e-300)
    out['pose_gram_rank_lt_6_groups'] = int((ev[:, 0] <= ev[:, -1]*1e-10).sum())
    out['pose_gram_condition_quantiles'] = np.quantile(cond[ev[:, 0] > ev[:, -1]*1e-10], [0, .5, .9, .99, 1]).tolist()
    g0 = np.random.default_rng(0)
    for dev in devices:
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
        G = GroupBasis(t(xyz), t(member).long(), t(centroids))
        z = g0.normal(size=(S, 12)); f = g0.normal(size=(len(xyz), 6))
        Gz = G.expand(t(z)).cpu().numpy()
        ind = np.concatenate([z[member, :3]+cross_np(z[member, 3:6], r), z[member, 6:9]+cross_np(z[member, 9:12], r)], 1)   # independent formula
        lhs = float((Gz*f).sum()); rhs = float((z*G.reduce(t(f)).cpu().numpy()).sum())
        out[f'expand_vs_independent_max_abs_{dev}'] = float(np.abs(Gz-ind).max())
        out[f'expand_vs_independent_max_rel_{dev}'] = float(np.abs(Gz-ind).max()/np.abs(ind).max())
        out[f'adjoint_rel_{dev}'] = abs(lhs-rhs)/max(abs(lhs), abs(rhs))
    # rigid-retraction finite difference (pose block) and linear velocity block
    from scipy.spatial.transform import Rotation
    G = GroupBasis(torch.as_tensor(xyz), torch.as_tensor(member).long(), torch.as_tensor(centroids))
    worst = 0.
    for comp in range(6):
        e = np.zeros(6); h = 1e-7; e[comp] = h
        moved = r@Rotation.from_rotvec(e[3:]).as_matrix().T+e[:3]
        z = torch.zeros(S, 12, dtype=torch.float64); z[:, comp] = 1
        worst = max(worst, float(np.abs((moved-r)/h-G.expand(z)[:, :3].numpy()).max()))
    out['retraction_fd_max_abs'] = worst
    # shuffle invariance: permute points and relabel groups consistently
    perm = g0.permutation(len(xyz)); gperm = g0.permutation(S); inv = np.argsort(gperm)
    Gs = GroupBasis(torch.as_tensor(xyz[perm]), torch.as_tensor(inv[member[perm]]).long(), torch.as_tensor(centroids[gperm]))
    z = g0.normal(size=(S, 12))
    a = G.expand(torch.as_tensor(z)).numpy()[perm]; b = Gs.expand(torch.as_tensor(z[gperm])).numpy()
    out['shuffle_invariance_max_abs'] = float(np.abs(a-b).max())
    out['ordering'] = 'group [dt(3), dtheta(3), du(3), domega(3)]; point [dx(3), dv(3)]; dx = dt + dtheta x r, dv = du + domega x r, r = x_can - c_can'
    out['reference_mapping'] = ('pinned supernodes.GroupedObservation.expand: dmu = dt + dtheta x (mu0 - c0), dphi = dtheta; sparse_iekf._G applies the '
                                'same map to the velocity block (dv = dv_c + domega_c x r). The point port drops dphi (points have no orientation).')
    return out


# ------------------------------------------------------------------------------------------------ helpers on the blade
def blade_setup(run, scene, dev, fps):
    from pointwise_rgbd.dataset import Dataset
    from pointwise_rgbd.config import load_config
    from pointwise_rgbd.supernodes import GroupBasis
    from pointwise_rgbd.rgbd_point_renderer import CameraTensors, RenderPolicy
    d = Path(run)/'first/phase1'
    part = np.load(d/'partition.npz'); st = np.load(d/'state.npy', mmap_mode='r'); cov = np.load(d/'covariance.npy', mmap_mode='r')
    rho = np.load(d/'footprint_radius.npy'); config = load_config(); data = Dataset(scene, None, None, fps=fps)
    t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    G = GroupBasis(t(part['xyz']), t(part['membership'], torch.long), t(part['centroids']))
    policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
    return {'part': part, 'state': st, 'cov': cov, 'rho': t(rho), 'rgb': t(part['rgb']), 'config': config, 'data': data, 'G': G, 't': t,
            'policy': policy, 'cams': [CameraTensors(c, torch.device(dev)) for c in data.cameras]}


def saved_prior(B, k):
    """Fixed prior of frame k: process6.predict of the SAVED posterior of frame k-1 (exactly the production prediction)."""
    from pointwise_rgbd.process6 import predict
    dt = float(B['data'].times[k]-B['data'].times[k-1])
    return predict(B['t'](B['state'][k-1]), B['t'](B['cov'][k-1]), dt, B['config'])


def observation(B, k, dev):
    from pointwise_rgbd.image_observation import FrameObservation, ObservationPolicy
    return FrameObservation([B['data'].observation(k, c) for c in range(len(B['cams']))], B['config'], ObservationPolicy(), dev)


def energy(B, obs, s, sm, Lam):
    from pointwise_rgbd.rgbd_point_renderer import render
    lin = render(s[:, :3], B['rgb'], B['rho'], B['cams'], B['policy']); terms = obs.terms(lin); e = s-sm
    return .5*float(torch.einsum('ni,nij,nj->', e, Lam, e))+terms['energy'], lin, terms


def audit_operator(B, k, dev):
    from pointwise_rgbd.rgbd_point_renderer import render
    from pointwise_rgbd.image_iekf import linear_system
    from pointwise_rgbd.rigeuler_ref.trust_region import ScalingMetric, steihaug_cg
    sm, Pm = saved_prior(B, k); obs = observation(B, k, dev); G = B['G']
    Lam = torch.linalg.inv(Pm); Lam = (Lam+Lam.transpose(1, 2))/2
    out = {'frame': k}
    s = B['t'](B['state'][k])                                # linearize at the SAVED final state of frame k (moved, realistic)
    E0, lin, terms = energy(B, obs, s, sm, Lam)
    g = torch.Generator(device='cpu').manual_seed(k)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64).to(dev)
    # H finite differences (central; pixels whose contributor sets are frozen at both sides)
    dx = rnd(lin.n, 3)*1e-4; eps = 1e-3
    p, m = render(s[:, :3]+eps*dx, B['rgb'], B['rho'], B['cams'], B['policy']), render(s[:, :3]-eps*dx, B['rgb'], B['rho'], B['cams'], B['policy'])
    errs = []
    for v, vp, vm, h in zip(lin.views, p.views, m.views, lin.apply_H(dx)):
        fd = torch.cat(((vp.rgb-vm.rgb)/(2*eps), ((vp.depth-vm.depth)/(2*eps))[:, None], ((vp.coverage-vm.coverage)/(2*eps))[:, None]), 1)
        use = v.total > 0
        errs.append(float((fd-h)[use].norm()/fd[use].norm()))
    out['H_fd_rel_error_per_camera'] = errs
    a = [rnd(len(t_), 5) for t_ in terms['residual_adjoint']]; v = rnd(lin.n, 3)
    lhs = float(sum((h*ai).sum() for h, ai in zip(lin.apply_H(v), a))); rhs = float((v*lin.apply_HT(a)).sum())
    out['H_adjoint_rel'] = abs(lhs-rhs)/max(abs(lhs), abs(rhs))
    for lm in (1e-3, 0.):
        b, A, precond = linear_system(s, sm, Lam, lin, terms, obs, G, lm)
        u, w = rnd(G.count, 12), rnd(G.count, 12)
        out[f'A_symmetry_rel_lm{lm:g}'] = abs(float((u*A(w)).sum()-(A(u)*w).sum()))/abs(float((u*A(w)).sum()))
        out[f'A_min_rayleigh_random_lm{lm:g}'] = min(float((x*A(x)).sum()/(x*x).sum()) for x in (u, w))
    b, A, precond = linear_system(s, sm, Lam, lin, terms, obs, G, 1e-3)
    # b = -grad E in group coordinates: check against the finite-difference derivative of the TRUE objective along G z
    z = rnd(G.count, 12); z = z/z.norm()*1e-2
    for h in (1e-4, 1e-5):
        Ep = energy(B, obs, s+h*G.expand(z), sm, Lam)[0]; Em = energy(B, obs, s-h*G.expand(z), sm, Lam)[0]
        out[f'b_vs_true_objective_fd_rel_h{h:g}'] = abs((Ep-Em)/(2*h)+float((b*z).sum()))/abs(float((b*z).sum()))
    metric = ScalingMetric.from_physical_scales((G.count, 12), s.device, s.dtype)
    prod = steihaug_cg(A, b, 1e9, metric, M_inv=precond, max_iters=30, eta=1e-2)
    acc = steihaug_cg(A, b, 1e9, metric, M_inv=precond, max_iters=2000, eta=1e-12)
    out['solve_production_pcg30_eta1e-2'] = {'iterations': prod.iterations, 'rel_residual': prod.rel_residual, 'stop': prod.stop_reason}
    out['solve_accurate'] = {'iterations': acc.iterations, 'rel_residual': acc.rel_residual, 'stop': acc.stop_reason}
    out['production_vs_accurate_step_rel'] = float((prod.step-acc.step).norm()/acc.step.norm())
    out['energy_at_linearization'] = E0
    out['counters'] = {'points': lin.n, 'groups': G.count, 'cameras': len(lin.views), 'contributions_per_camera': [int(len(v.point)) for v in lin.views]}
    return out


def dense_oracles():
    """A_dense vs matrix-free A_g and b_g on synthetic geometries (overlap/taper, disocclusion, multi-camera)."""
    from tests.image_fixtures import two_plate_scene, catalog
    from pointwise_rgbd.config import load_config
    from pointwise_rgbd.supernodes import GroupBasis, build_supernodes
    from pointwise_rgbd.rgbd_point_renderer import CameraTensors, RenderPolicy, render
    from pointwise_rgbd.image_observation import FrameObservation, ObservationPolicy
    from pointwise_rgbd.process6 import predict
    from pointwise_rgbd.image_iekf import linear_system
    from pointwise_rgbd import restricted_posterior as rp
    config = load_config(); out = {}
    for name, kw, frame in (('overlap_taper_offset3cm', dict(motion='prismatic', offset=.03), 1), ('revolute_disocclusion', dict(motion='revolute'), 2),
                            ('prismatic_4cams', dict(motion='prismatic', cameras=4), 1)):
        ds = two_plate_scene(frames=3, width=40, height=30, **{'cameras': 2, **kw})
        xyz, rgb, rho, _ = catalog(ds, 0)
        keep = np.flatnonzero(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1) < .3)[:90]
        xyz, rgb, rho = xyz[keep], rgb[keep], rho[keep]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
        sn = build_supernodes(t(xyz), num_supernodes=6); G = GroupBasis(t(xyz), sn.supernode_id, sn.centroids)
        n = len(xyz); s0 = torch.zeros(n, 6, dtype=torch.float64); s0[:, :3] = t(xyz)
        P0 = torch.zeros(n, 6, 6, dtype=torch.float64); P0[:, :3, :3] = 4e-6*torch.eye(3); P0[:, 3:, 3:] = torch.eye(3)
        sm, Pm = predict(s0, P0, 1/30, config); Lam = torch.linalg.inv(Pm)
        obs = FrameObservation([ds.observation(frame, c) for c in range(len(ds.cameras))], config, ObservationPolicy(), 'cpu')
        cams = [CameraTensors(c, torch.device('cpu')) for c in ds.cameras]
        s = sm+1e-3*torch.randn(sm.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        lin = render(s[:, :3], t(rgb), t(rho), cams, RenderPolicy()); terms = obs.terms(lin)
        b, A, _ = linear_system(s, sm, Lam, lin, terms, obs, G, 0.)
        H, w, _, Gd = rp.dense_operators(lin, obs, terms, G)
        Hf = torch.cat([H.reshape(len(H), n, 3), torch.zeros(len(H), n, 3, dtype=torch.float64)], 2).reshape(len(H), 6*n)
        adj = torch.cat([x.reshape(-1) for x in terms['residual_adjoint']])
        Ad = Gd.T@(torch.block_diag(*Lam)+Hf.T@(w[:, None]*Hf))@Gd; bd = Gd.T@(Hf.T@adj-torch.block_diag(*Lam)@(s-sm).reshape(-1))
        Z = torch.eye(Ad.shape[0], dtype=torch.float64); Am = torch.stack([A(Z[k].reshape(G.count, 12)).reshape(-1) for k in range(len(Z))], 1)
        ev = torch.linalg.eigvalsh((Ad+Ad.T)/2)
        out[name] = {'points': n, 'groups': G.count, 'cameras': len(cams), 'A_rel_max': float((Am-Ad).abs().max()/Ad.abs().max()),
                     'b_rel': float((b.reshape(-1)-bd).norm()/bd.norm()), 'A_dense_eig_min': float(ev[0]), 'A_dense_eig_max': float(ev[-1]),
                     'cross_group_coupling_fraction': float(Ad.reshape(G.count, 12, G.count, 12).abs().sum((1, 3)).fill_diagonal_(0).sum()/Ad.abs().sum())}
    return out


# ------------------------------------------------------------------------------------------------ A3
def one_step(B, k, dev, variants, gt_slide=None):
    from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
    from pointwise_rgbd.image_observation import levels, ObservationPolicy
    sm, Pm = saved_prior(B, k)
    frames = [B['data'].observation(k, c) for c in range(len(B['cams']))]
    lv = levels(frames, B['config'], ObservationPolicy(), dev)
    res = {'frame': k}
    for name, cfg in variants.items():
        torch.cuda.synchronize() if dev == 'cuda' else None; t0 = time.time()
        s, P, observed, lin, terms, rep = frame_step(sm, Pm.clone(), lv, B['rgb'], B['rho'], B['policy'], B['G'], cfg)
        torch.cuda.synchronize() if dev == 'cuda' else None
        row = {'seconds': time.time()-t0, 'energy_initial': rep['energy_initial'], 'energy_final': rep['energy_final'], 'stop': rep['stop'],
               'accepted': rep['accepted_steps'], 'rejected': rep['rejected_trials'],
               'iterations': [{kk: h[kk] for kk in ('accepted', 'rho', 'trust_radius_before', 'trust_radius_after', 'predicted_reduction', 'actual_reduction',
                                                     'pcg_iterations', 'pcg_rel_residual', 'pcg_stop_reason', 'step_norm_scaled', 'max_point_step_m', 'gradient_norm')}
                              for h in rep['history']],
               'per_camera': [{kk: c.get(kk) for kk in ('camera', 'energy', 'energy_rgb', 'energy_depth', 'energy_coverage', 'unexplained_domain_pixels',
                                                         'predicted_outside_domain_pixels', 'rgb_rmse', 'depth_rmse_m', 'mean_rgb_irls_weight')} for c in rep['cameras']],
               'prior_velocity_sigma_median': float(Pm[:, 3:, 3:].diagonal(dim1=1, dim2=2).mean(1).sqrt().median()),
               'prior_position_sigma_median_m': float(Pm[:, :3, :3].diagonal(dim1=1, dim2=2).mean(1).sqrt().median())}
        if gt_slide is not None:          # post-hoc evaluation only
            lab, axis, truth = gt_slide
            ds_ = (s[:, :3]-sm[:, :3]).cpu().numpy()@axis
            row['posthoc_moving_correction_median_m'] = float(np.median(ds_[lab == 1])); row['posthoc_static_correction_median_m'] = float(np.median(ds_[lab == 0]))
            row['posthoc_prior_moving_slide_m'] = float(np.median((sm[:, :3].cpu().numpy()-B['part']['xyz'])[lab == 1]@axis))
            row['posthoc_true_slide_m'] = truth
            row['posthoc_posterior_moving_slide_m'] = float(np.median((s[:, :3].cpu().numpy()-B['part']['xyz'])[lab == 1]@axis))
        res[name] = row
        print(k, name, {kk: row[kk] for kk in ('energy_final', 'stop', 'accepted', 'seconds')}, row.get('posthoc_posterior_moving_slide_m'), row.get('posthoc_true_slide_m'), flush=True)
    return res


def gt_labels(B, scene):
    """Post-hoc only: moving label per catalogue point from the sparse GT part cloud and the GT slide per frame."""
    from scipy.spatial import cKDTree
    from tools.evaluate_image_iekf_gt import ply
    scene = Path(scene)
    x0, l0 = ply(scene/'pointcloud/points3D_parts.ply'); xf, lf = ply(scene/'pointcloud/points3D_parts_final.ply')
    mov = max(set(l0) & set(lf), key=lambda l: np.linalg.norm(x0[l0 == l].mean(0)-xf[lf == l].mean(0)))
    dist, nn = cKDTree(x0).query(B['part']['xyz']); lab = np.where(dist < .01, (l0[nn] == mov).astype(int), -1)
    meta = json.loads((scene/'pointcloud/joint_metadata.json').read_text()); axis = np.asarray(meta['axis_vector'], float)
    vals = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True); q = np.asarray(vals[vals.dtype.names[1]], float)
    return lab, axis, q-q[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True); ap.add_argument('--fps', type=float, default=59)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--parts', nargs='*', default=['A1', 'A2', 'A3'])
    ap.add_argument('--frames', nargs='*', type=int, default=[1, 2, 5, 10, 20]); ap.add_argument('--device', default='cuda')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    watched = {str(p): sha(p) for p in sorted((a.run/'first/phase1').glob('*.np[yz]'))} | {str(REPO/f): sha(REPO/f) for f in SOURCES}
    report_path = REPO/'reports/phase1_grouped_operator_parity.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report['sources_sha256'] = {f: watched[str(REPO/f)] for f in SOURCES}
    report['run'] = str(a.run); report['torch'] = torch.__version__; report['cuda'] = torch.version.cuda
    part = np.load(a.run/'first/phase1/partition.npz')
    if 'A1' in a.parts:
        report['A1_group_basis'] = audit_basis(part['xyz'], part['membership'], part['centroids'], ['cpu']+(['cuda'] if torch.cuda.is_available() else []))
        print(json.dumps(report['A1_group_basis'], indent=1), flush=True)
    if 'A2' in a.parts or 'A3' in a.parts:
        B = blade_setup(a.run, a.scene, a.device, a.fps)
    if 'A2' in a.parts:
        report['A2_dense_oracles_synthetic'] = dense_oracles(); print(json.dumps(report['A2_dense_oracles_synthetic'], indent=1), flush=True)
        report['A2_blade_operator'] = [audit_operator(B, k, a.device) for k in (1, 10)]; print(json.dumps(report['A2_blade_operator'], indent=1, default=float), flush=True)
    if 'A3' in a.parts:
        from pointwise_rgbd.image_iekf import IEKFConfig
        variants = {'production': IEKFConfig(), 'accurate_linear_solve': IEKFConfig(forcing_max=1e-10, forcing_min=1e-12, pcg_max_iters=500),
                    'budget30': IEKFConfig(max_iters=30), 'budget100_accurate': IEKFConfig(max_iters=100, forcing_max=1e-10, forcing_min=1e-12, pcg_max_iters=500)}
        lab, axis, qrel = gt_labels(B, a.scene)
        report['A3_one_step'] = [one_step(B, k, a.device, variants, (lab, axis, float(qrel[k]))) for k in a.frames]
        report['A3_note'] = ('priors are process6.predict of the SAVED production posterior of frame k-1 (identical for all variants); ground truth is '
                             'used only for the posthoc_* fields')
    report['unchanged_after_audit'] = all(sha(p) == h for p, h in watched.items())
    report_path.write_text(json.dumps(report, indent=1, default=float))
    print('unchanged', report['unchanged_after_audit'])


if __name__ == '__main__':
    main()
