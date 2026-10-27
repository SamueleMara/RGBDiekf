"""Contract section 13 Kalman gates for the Phase-1 rendered-image IEKF (no correspondences are given)."""
import json, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, catalog, source_parts, true_positions
from pointwise_rgbd.config import load_config
from pointwise_rgbd.rgbd_point_renderer import CameraTensors, RenderPolicy, render
from pointwise_rgbd.image_observation import FrameObservation, ObservationPolicy
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.supernodes import GroupBasis, build_supernodes
from pointwise_rgbd.process6 import predict, transition, process_covariance
from pointwise_rgbd import restricted_posterior as rp
from pointwise_rgbd import image_phase1
from rgbd_common.backends.torch_cuda import Backend

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REPORT = Path(__file__).resolve().parents[1]/'reports/image_iekf_dense_covariance_comparison.json'
t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)


def small_problem(n_max=70, groups=6, width=40, height=30, cameras=2, radius=.25, frames=2, seed_point=0, k0=0):
    ds = two_plate_scene('prismatic', frames=frames, cameras=cameras, width=width, height=height)
    xyz, rgb, rho, _ = catalog(ds, k0)
    keep = np.flatnonzero(np.linalg.norm(xyz[:, :2]-xyz[seed_point, :2], axis=1) < radius)[:n_max]
    xyz, rgb, rho = xyz[keep], rgb[keep], rho[keep]
    sn = build_supernodes(t(xyz), num_supernodes=groups)
    G = GroupBasis(t(xyz), sn.supernode_id, sn.centroids)
    cams = [CameraTensors(c, torch.device('cpu')) for c in ds.cameras]
    n = len(xyz)
    s0 = torch.zeros(n, 6, dtype=torch.float64); s0[:, :3] = t(xyz)
    P0 = torch.zeros(n, 6, 6, dtype=torch.float64); P0[:, :3, :3] = 4e-6*torch.eye(3); P0[:, 3:, 3:] = torch.eye(3)
    return ds, xyz, t(rgb), t(rho), G, cams, s0, P0


def observation(ds, k, config):
    return FrameObservation([ds.observation(k, c) for c in range(len(ds.cameras))], config, ObservationPolicy(), 'cpu')


class ProcessModelTest(unittest.TestCase):
    def test_dense_F_Q_white_acceleration(self):
        config = load_config(); dt = .02
        F, Q = transition(dt), process_covariance(dt, config.qc, config.qx)
        self.assertTrue(np.array_equal(F, np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])))
        expect = np.block([[dt**3/3*config.qc+dt*config.qx, dt**2/2*config.qc], [dt**2/2*config.qc, dt*config.qc]])
        self.assertTrue(np.allclose(Q, expect, rtol=0, atol=1e-18))
        self.assertGreaterEqual(np.linalg.eigvalsh(Q).min(), -1e-15)
        # Exact discretization: Monte-Carlo-free check by integrating the white-acceleration SDE covariance numerically.
        m = 4000; tau = (np.arange(m)+.5)*dt/m
        Phi = lambda s: np.block([[np.eye(3), s*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        L = np.vstack([np.zeros((3, 3)), np.eye(3)])
        Qn = sum(Phi(dt-s)@L@config.qc@L.T@Phi(dt-s).T for s in tau)*dt/m
        self.assertTrue(np.allclose(Q-np.block([[dt*config.qx, np.zeros((3, 3))], [np.zeros((3, 3)), np.zeros((3, 3))]]), Qn, rtol=1e-6, atol=1e-14))

    def test_prediction_matches_F_by_finite_difference(self):
        config = load_config(); dt = 1/59
        s = torch.randn(5, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        P = torch.eye(6, dtype=torch.float64).expand(5, 6, 6).clone()
        for reverse in (False, True):
            base, _ = predict(s, P, dt, config, reverse)
            F = np.stack([((predict(s+1e-6*torch.eye(6, dtype=torch.float64)[j], P, dt, config, reverse)[0]-base)/1e-6)[0].numpy() for j in range(6)], 1)
            self.assertTrue(np.allclose(F, transition(dt, reverse), atol=1e-8))

    def test_reverse_time_equals_reference_processing_time_convention(self):
        """Reference: positive processing dt, processing-time velocity v_p = -v. Ours: physical v, F(-dt)."""
        config = load_config(); dt = 1/59
        s = torch.tensor([[.1, -.2, .3, .5, -1., .25]], dtype=torch.float64)
        P = torch.tensor(np.cov(np.random.default_rng(1).normal(size=(6, 40))), dtype=torch.float64)[None]
        ours, Po = predict(s, P, dt, config, reverse=True)
        J = torch.diag(torch.tensor([1., 1, 1, -1, -1, -1], dtype=torch.float64))
        ref, Pr = predict(s@J, J@P@J, dt, config, reverse=False)
        self.assertTrue(torch.allclose(ours, ref@J, atol=1e-15))
        self.assertTrue(torch.allclose(Po, J@Pr@J, atol=1e-15))
        self.assertTrue(np.allclose(process_covariance(dt, config.qc, config.qx, True)[:3, 3:], -(dt**2)/2*config.qc))


class GroupBasisTest(unittest.TestCase):
    def test_raw_reference_tangent_adjoint_and_rigid_fd(self):
        _, xyz, _, _, G, *_ = small_problem()
        g = torch.Generator().manual_seed(2)
        z = torch.randn(G.count, 12, generator=g, dtype=torch.float64); f = torch.randn(len(xyz), 6, generator=g, dtype=torch.float64)
        self.assertLess(abs(float((G.expand(z)*f).sum()-(z*G.reduce(f)).sum())), 1e-11*float(G.expand(z).norm()*f.norm()))
        dense = G.matrix()
        self.assertTrue(torch.allclose(G.expand(z), (dense@z[G.member, :, None])[..., 0]))
        # FD: G pose block is the derivative of a rigid motion about the group centroid.
        from scipy.spatial.transform import Rotation
        lever = G.lever; eps = 1e-7
        for gi in range(G.count):
            for comp in range(6):
                e = torch.zeros(6, dtype=torch.float64); e[comp] = eps
                R = torch.as_tensor(Rotation.from_rotvec(e[3:].numpy()).as_matrix())
                moved = lever@R.T+e[:3]
                zz = torch.zeros(G.count, 12, dtype=torch.float64); zz[gi, comp] = 1
                fd = torch.where((G.member == gi)[:, None], (moved-lever)/eps, torch.zeros_like(lever))
                self.assertLess(float((fd-G.expand(zz)[:, :3]).abs().max()), 1e-6)
                zz[gi] = 0; zz[gi, 6+comp] = 1
                self.assertTrue(torch.allclose(G.expand(zz)[:, 3:], torch.where((G.member == gi)[:, None], G.pos[:, :, comp], torch.zeros_like(lever))))

    def test_build_supernodes_matches_pinned_semantics(self):
        x = torch.randn(1000, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        a = build_supernodes(x, target_size=64); b = build_supernodes(x, target_size=64)
        self.assertEqual(a.num_supernodes, 1000//64); self.assertTrue(torch.equal(a.supernode_id, b.supernode_id))
        for k in range(a.num_supernodes):
            self.assertTrue(torch.allclose(a.centroids[k], x[a.supernode_id == k].mean(0)))


class KalmanFormsTest(unittest.TestCase):
    """Dense linear-Gaussian identities of the update (no renderer): information form == K form, Joseph, R scaling."""
    def setUp(self):
        g = torch.Generator().manual_seed(5); n = 8; m = 40
        A = torch.randn(6*n, 6*n, generator=g, dtype=torch.float64)
        self.P = A@A.T/(6*n)+.1*torch.eye(6*n, dtype=torch.float64)
        self.H = torch.randn(m, 3*n, generator=g, dtype=torch.float64)
        self.w = torch.rand(m, generator=g, dtype=torch.float64)+.5
        self.n, self.m = n, m
        self.Hf = torch.cat([self.H.reshape(m, n, 3), torch.zeros(m, n, 3, dtype=torch.float64)], 2).reshape(m, 6*n)

    def test_information_equals_gain_form_and_joseph(self):
        P, Hf, w = self.P, self.Hf, self.w
        R = torch.diag(1/w); r = torch.randn(self.m, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        K = P@Hf.T@torch.linalg.inv(Hf@P@Hf.T+R)
        A = torch.linalg.inv(P)+Hf.T@(w[:, None]*Hf)
        self.assertLess(float((K@r-torch.linalg.solve(A, Hf.T@(w*r))).norm()), 1e-10)
        I = torch.eye(len(P), dtype=torch.float64)
        joseph = (I-K@Hf)@P@(I-K@Hf).T+K@R@K.T
        self.assertLess(float((joseph-torch.linalg.inv(A)).abs().max()), 1e-10)
        # restricted function with G = I reproduces the unrestricted Joseph
        Pj, _, _ = rp.dense_restricted_joseph(P, self.H, w, 1/w, torch.eye(len(P), dtype=torch.float64))
        self.assertLess(float((Pj-joseph).abs().max()), 1e-10)

    def test_noise_scaling(self):
        G = torch.eye(len(self.P), dtype=torch.float64)[:, ::2]
        P1, K1, _ = rp.dense_restricted_joseph(self.P, self.H, self.w, 1/self.w, G)
        P2, K2, _ = rp.dense_restricted_joseph(7*self.P, self.H, self.w/7, 7/self.w, G)
        self.assertLess(float((K1-K2).abs().max()), 1e-10)
        self.assertLess(float((7*P1-P2).abs().max()), 1e-10)

    def test_velocity_updated_only_through_cross_covariance(self):
        P = self.P.clone().reshape(self.n, 6, self.n, 6)
        P[:, 3:, :, :3] = 0; P[:, :3, :, 3:] = 0
        P = P.reshape(6*self.n, 6*self.n)
        Pj, K, _ = rp.dense_restricted_joseph(P, self.H, self.w, 1/self.w, torch.eye(len(P), dtype=torch.float64))
        vel = (torch.arange(len(P)) % 6) >= 3
        self.assertLess(float(K[vel].abs().max()), 1e-12)
        self.assertLess(float((Pj[vel][:, vel]-P[vel][:, vel]).abs().max()), 1e-12)


class RestrictedJosephTest(unittest.TestCase):
    def test_marginal_blocks_equal_dense_restricted_joseph(self):
        config = load_config()
        for groups, cams in ((6, 2), (3, 3), (12, 2)):
            ds, xyz, rgb, rho, G, C, s0, P0 = small_problem(groups=groups, cameras=cams)
            P0[:, :3, 3:] = P0[:, 3:, :3] = .3e-3*torch.eye(3)
            sm, Pm = predict(s0, P0, 1/30, config)
            obs = observation(ds, 1, config)
            s, P, observed, lin, terms, rep = frame_step(sm, Pm, [(obs, C, 1.)], rgb, rho, RenderPolicy(), G, IEKFConfig(posterior='restricted_joseph_marginals'))
            H, w, r, Gd = rp.dense_operators(lin, obs, terms, G)
            Pd, _, _ = rp.dense_restricted_joseph(torch.block_diag(*Pm), H, w, r, Gd)
            B = rp.blocks_of(Pd, len(xyz))
            self.assertLess(float((B-P).norm()/B.norm()), 1e-8, f'groups={groups}')
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(Pd).min()), -1e-12)
            self.assertEqual(rep['posterior']['psd_repair']['blocks_floored'], 0)

    def test_unmeasured_point_keeps_prediction_but_inherits_group_correction_noise(self):
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        sm, Pm = predict(s0, P0, 1/30, config)
        far = sm.clone(); far[0, :3] += torch.tensor([0, 0, 50.], dtype=torch.float64)     # never rendered into any camera
        obs = observation(ds, 1, config)
        s, P, observed, lin, terms, rep = frame_step(far, Pm, [(obs, C, 1.)], rgb, rho, RenderPolicy(), G, IEKFConfig(posterior='restricted_joseph_marginals'))
        self.assertFalse(bool(observed[0]))
        H, w, r, Gd = rp.dense_operators(lin, obs, terms, G)
        Pd, _, _ = rp.dense_restricted_joseph(torch.block_diag(*Pm), H, w, r, Gd)
        self.assertLess(float((rp.blocks_of(Pd, len(xyz))[0]-P[0]).norm()/P[0].norm()), 1e-8)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(P[0]-Pm[0]).min()), -1e-15)


class DenseMultiFrameCovarianceTest(unittest.TestCase):
    """>= 5 frames on the SAME linearizations (those of the production filter), growing fixtures, both directions:
      actual       dense 6N Joseph of the production gain (block prior information from the carried marginals)
                   applied to the full propagated covariance of its errors: the TRUE error covariance;
      structured   production StructuredCovariance marginals: must equal 'actual' to round-off;
      optimal      dense 6N restricted Kalman with gain and Joseph from the full propagated prior;
      assumed      restricted-Joseph marginals carried as a block-diagonal density (ablation);
      rigeuler     pinned block information (comparison)."""
    def test_multi_frame_forward_and_reverse(self):
        from pointwise_rgbd.phase1_covariance import StructuredCovariance, BlockCovariance
        config = load_config(); out = []
        tr = lambda M: float(M[:, :3, :3].diagonal(dim1=1, dim2=2).sum())
        trv = lambda M: float(M[:, 3:, 3:].diagonal(dim1=1, dim2=2).sum())
        for n_max, groups in ((40, 4), (80, 8), (160, 12)):
            for reverse in (False, True):
                ds, xyz, rgb, rho, G, C, s0, P0 = small_problem(n_max=n_max, groups=groups, frames=6, radius=.4, k0=5 if reverse else 0)
                order = list(range(6)) if not reverse else list(range(5, -1, -1))
                n = len(xyz); dt = float(ds.times[1]-ds.times[0])
                cov = StructuredCovariance(P0.clone(), G, chunk=17)
                assumed, rig = BlockCovariance(P0.clone(), 'restricted_joseph_marginals'), BlockCovariance(P0.clone(), 'rigeuler_block_information')
                s = s0; Pact = Popt = torch.block_diag(*P0)
                F = torch.as_tensor(transition(dt, reverse)); Q = torch.as_tensor(process_covariance(dt, config.qc, config.qx, reverse))
                Fn, Qn = torch.block_diag(*[F]*n), torch.block_diag(*[Q]*n)
                rows = []
                for k in order[1:]:
                    sm = s@F.T
                    for m in (cov, assumed, rig): m.predict(dt, reverse, config)
                    Pact = Fn@Pact@Fn.T+Qn; Popt = Fn@Popt@Fn.T+Qn
                    self.assertLess(float((rp.blocks_of(Pact, n)-cov.marginals()).norm()/cov.marginals().norm()), 1e-9, 'structured prediction')
                    obs = observation(ds, k, config)
                    s, P, observed, lin, terms, rep = frame_step(sm, cov, [(obs, C, 1.)], rgb, rho, RenderPolicy(), G, IEKFConfig())
                    for m in (assumed, rig): m.update(lin, obs, terms, G)
                    H, w, r, Gd = rp.dense_operators(lin, obs, terms, G)
                    Pact, _, _ = rp.dense_restricted_joseph(Pact, H, w, r, Gd, P_gain=torch.block_diag(*rp.blocks_of(Pact, n)))
                    Popt, _, _ = rp.dense_restricted_joseph(Popt, H, w, r, Gd)
                    A, O = rp.blocks_of(Pact, n), rp.blocks_of(Popt, n)
                    self.assertGreaterEqual(float(torch.linalg.eigvalsh(Pact).min()), -1e-10)
                    err = float((P-A).norm()/A.norm())
                    self.assertLess(err, 1e-8, f'structured marginals vs dense actual, frame {k}')
                    Pa, Pr = assumed.marginals(), rig.marginals()
                    rows.append({'frame': k, 'structured_vs_actual_rel': err,
                                 'actual_over_optimal_position_trace': tr(A)/tr(O),
                                 'assumed_over_actual_position_trace': tr(Pa)/tr(A), 'assumed_over_actual_velocity_trace': trv(Pa)/trv(A),
                                 'assumed_vs_actual_rel': float((Pa-A).norm()/A.norm()),
                                 'rigeuler_over_actual_position_trace': tr(Pr)/tr(A), 'rigeuler_vs_actual_rel': float((Pr-A).norm()/A.norm())})
                out.append({'points': n, 'groups': G.count, 'direction': 'reverse' if reverse else 'forward', 'frames': rows})
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps({'semantics': self.__doc__, 'fixtures': out}, indent=1))


class IteratedSolverTest(unittest.TestCase):
    def test_fixed_prior_one_posterior_and_velocity_through_covariance_only(self):
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        sm, Pm = predict(s0, P0, 1/30, config)
        Pm[:, :3, 3:] = 0; Pm[:, 3:, :3] = 0
        P_copy = Pm.clone()
        s, P, observed, lin, terms, rep = frame_step(sm, Pm, [(observation(ds, 1, config), C, 1.)], rgb, rho, RenderPolicy(), G, IEKFConfig())
        self.assertTrue(torch.equal(Pm, P_copy), 'prior covariance must stay fixed')
        self.assertGreater(rep['accepted_steps'], 0)
        self.assertLess(rep['energy_final'], rep['energy_initial'])
        self.assertLess(float((s[:, 3:]-sm[:, 3:]).abs().max()), 1e-12)
        self.assertLess(rep['jvp_vjp_adjoint_check']['relative_error'], 1e-12)

    def test_matrix_free_system_and_steihaug_equal_dense(self):
        """At one linearization: b, A = G^T(Lambda + H^T W H + lambda I)G and the Steihaug solve (huge radius) vs dense algebra."""
        from pointwise_rgbd.image_iekf import linear_system
        from pointwise_rgbd.rigeuler_ref.trust_region import ScalingMetric, steihaug_cg
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        sm, Pm = predict(s0, P0, 1/30, config)
        obs = observation(ds, 1, config); Lam = torch.linalg.inv(Pm)
        s = sm+1e-3*torch.randn(sm.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        lin = render(s[:, :3], rgb, rho, C, RenderPolicy()); terms = obs.terms(lin)
        b, A_solve, precond = linear_system(s, sm, Lam, lin, terms, obs, G, 1e-3)
        H, w, _, Gd = rp.dense_operators(lin, obs, terms, G)
        n = len(s); Hf = torch.cat([H.reshape(len(H), n, 3), torch.zeros(len(H), n, 3, dtype=torch.float64)], 2).reshape(len(H), 6*n)
        adj = torch.cat([a.reshape(-1) for a in terms['residual_adjoint']])
        Ld = torch.block_diag(*Lam)
        bd = Gd.T@(Hf.T@adj-Ld@(s-sm).reshape(-1))
        Ad = Gd.T@(Ld+Hf.T@(w[:, None]*Hf)+1e-3*torch.eye(6*n, dtype=torch.float64))@Gd
        self.assertLess(float((b.reshape(-1)-bd).norm()/bd.norm()), 1e-12)
        Z = torch.eye(Ad.shape[0], dtype=torch.float64)
        Am = torch.stack([A_solve(Z[k].reshape(G.count, 12)).reshape(-1) for k in range(len(Z))], 1)
        self.assertLess(float((Am-Ad).abs().max()/Ad.abs().max()), 1e-12)
        metric = ScalingMetric.from_physical_scales((G.count, 12), s.device, s.dtype)
        res = steihaug_cg(A_solve, b, 1e9, metric, M_inv=precond, max_iters=500, eta=1e-12)
        self.assertLess(float((res.step.reshape(-1)-torch.linalg.solve(Ad, bd)).norm()/torch.linalg.solve(Ad, bd).norm()), 1e-8)

    def test_trust_region_reaches_the_dense_gauss_newton_stationary_region(self):
        """Nonlinear check on the same fixed objective: dense restricted Gauss-Newton with backtracking vs the vendored TR.
        Both are local methods on a continuous but piecewise-smooth objective (front ties); measured: the TR stops 0.15%
        above dense GN with its gradient reduced 640x. Its steps are accepted with rho ~ 2 (IRLS curvature ~2x the true
        curvature along the step) and the reference radius grows only at the boundary, so convergence is slow."""
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        sm, Pm = predict(s0, P0, 1/30, config)
        obs = observation(ds, 1, config); policy = RenderPolicy()
        Lam = torch.linalg.inv(Pm)

        def energy(s):
            lin = render(s[:, :3], rgb, rho, C, policy); terms = obs.terms(lin); e = s-sm
            return .5*float(torch.einsum('ni,nij,nj->', e, Lam, e))+terms['energy'], lin, terms
        s = sm.clone()
        for _ in range(200):
            E, lin, terms = energy(s)
            H, w, _, Gd = rp.dense_operators(lin, obs, terms, G)
            n = len(s); Hf = torch.cat([H.reshape(len(H), n, 3), torch.zeros(len(H), n, 3, dtype=torch.float64)], 2).reshape(len(H), 6*n)
            adj = torch.cat([a.reshape(-1) for a in terms['residual_adjoint']])
            b = Gd.T@(Hf.T@adj-torch.block_diag(*Lam)@(s-sm).reshape(-1))
            A = Gd.T@(torch.block_diag(*Lam)+Hf.T@(w[:, None]*Hf))@Gd
            step = (Gd@torch.linalg.solve(A, b)).reshape(n, 6); a = 1.
            while energy(s+a*step)[0] >= E and a > 1e-6: a /= 2
            if a <= 1e-6: break
            s = s+a*step
        E_dense = energy(s)[0]
        s_tr, *_, rep = frame_step(sm, Pm, [(obs, C, 1.)], rgb, rho, policy, G, IEKFConfig(max_iters=200))
        print('TR', rep['energy_final'], 'dense GN', E_dense, 'gradient', rep['history'][0]['gradient_norm'], '->', rep['history'][-1]['gradient_norm'])
        self.assertLess(abs(rep['energy_final']-E_dense), 5e-3*abs(E_dense))
        self.assertLess(rep['history'][-1]['gradient_norm'], 1e-2*rep['history'][0]['gradient_norm'])


class Phase1TrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(); cls.backend = Backend(DEVICE)

    def run_phase(self, motion, anchor, frames=5):
        ds = two_plate_scene(motion, frames=frames, cameras=3)
        root = tempfile.mkdtemp()
        with mock.patch('pointwise_rgbd.association.associate', side_effect=AssertionError('associate() called')):
            part, arr, rho = image_phase1.run(ds, self.config, self.backend, root, anchor)
        return ds, root, part, arr

    def check(self, ds, root, part, arr, k0):
        parts = source_parts(ds, part.source); errs = {}
        for k in range(len(ds.times)):
            err = np.linalg.norm(np.asarray(arr['xyz'][k])-true_positions(ds, part.xyz, parts, k, k0), axis=1)
            errs[k] = (float(np.median(err[parts == 1])), float(np.median(err[parts == 0])))
            if k != k0:
                rep = json.loads(Path(root, 'frames', f'{k:06d}.json').read_text())
                self.assertLessEqual(rep['energy_final'], rep['energy_initial'])
                self.assertEqual(len(rep['cameras']), len(ds.cameras))
        run = json.loads(Path(root, 'run.json').read_text())
        self.assertEqual(run['measurement_model'], 'rendered_rgbd_point_cloud_v2'); self.assertFalse(run['association_used'])
        self.assertEqual(sorted(run['order'][1:]), sorted(set(range(len(ds.times)))-{k0}), 'each non-anchor frame assimilated exactly once')
        return errs

    def test_forward_and_backward_prismatic(self):
        for anchor, k0 in (('first', 0), ('last', 4)):
            ds, root, part, arr = self.run_phase('prismatic', anchor)
            errs = self.check(ds, root, part, arr, k0)
            step = float(np.linalg.norm(ds.poses(1)[1][:3, 3]-ds.poses(0)[1][:3, 3]))
            for k, (moving, static) in errs.items():   # tracked: well under one inter-frame displacement (~21 mm); not re-tuned
                self.assertLess(moving, .25*step, f'{anchor} moving frame {k}: {errs}')
                self.assertLess(static, .05*step, f'{anchor} static frame {k}: {errs}')
            print(anchor, 'median errors (moving, static) m:', errs)


class ObservationModelTest(unittest.TestCase):
    def test_background_never_read_and_coverage_loss_costs(self):
        config = load_config()
        ds = two_plate_scene('prismatic', frames=2, cameras=2, width=48, height=36)
        xyz, rgb, rho, _ = catalog(ds, 0)
        cams = [CameraTensors(c, torch.device('cpu')) for c in ds.cameras]
        frames = [ds.observation(0, c) for c in range(2)]
        lin = render(t(xyz), t(rgb), t(rho), cams, RenderPolicy())
        e1 = FrameObservation(frames, config, ObservationPolicy(), 'cpu').terms(lin)['energy']
        noisy = []
        for f in frames:
            g = type(f)(f.rgb.copy(), f.depth.copy(), f.mask, f.camera, f.k, f.ci, None, True)
            g.rgb[~f.mask] = np.random.default_rng(0).random(g.rgb[~f.mask].shape); g.depth[~f.mask] = 3.
            noisy.append(g)
        self.assertEqual(e1, FrameObservation(noisy, config, ObservationPolicy(), 'cpu').terms(lin)['energy'])
        far = render(t(xyz)+t([0, 0, 5.]), t(rgb), t(rho), cams, RenderPolicy())
        terms = FrameObservation(frames, config, ObservationPolicy(), 'cpu').terms(far)
        self.assertGreater(terms['energy'], e1)
        for cam in terms['cameras']:   # nothing predicted on the domain: no RGB/depth innovation, the mismatch is coverage only
            self.assertEqual(cam['energy_rgb'], 0.); self.assertEqual(cam['energy_depth'], 0.)
            self.assertEqual(cam['unexplained_domain_pixels'], cam['domain_pixels'])

    def test_residual_adjoint_is_exact_gradient(self):
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        obs = observation(ds, 1, config); policy = RenderPolicy()
        x = s0[:, :3]+.003
        lin = render(x, rgb, rho, C, policy); terms = obs.terms(lin)
        grad = -lin.apply_HT(terms['residual_adjoint'])
        d = torch.randn(x.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        h = 1e-7
        fd = (obs.terms(render(x+h*d, rgb, rho, C, policy))['energy']-obs.terms(render(x-h*d, rgb, rho, C, policy))['energy'])/(2*h)
        self.assertLess(abs(fd-float((grad*d).sum())), 1e-4*abs(fd))

    def test_student_t_irls_weight_and_variance(self):
        from pointwise_rgbd.image_observation import student_t
        r = torch.linspace(-.3, .3, 41, dtype=torch.float64); sigma, nu = .05, 5.
        e, w = student_t(r, sigma, nu)
        de = (student_t(r+1e-7, sigma, nu)[0]-student_t(r-1e-7, sigma, nu)[0])/2e-7
        self.assertTrue(torch.allclose(de, w*r, atol=1e-6))
        self.assertTrue(torch.allclose(w, (nu+1)/(nu+(r/sigma)**2)/sigma**2))
        obs = observation(two_plate_scene('static', frames=1, cameras=1, width=24, height=18), 0, load_config())
        var = obs.noise_variance()[0]
        dom = obs.cameras[0]['domain']
        self.assertTrue(torch.allclose(var[dom, 0], torch.full_like(var[dom, 0], nu/(nu-2)*.0025)))
        self.assertTrue(torch.allclose(var[dom, 4], torch.full_like(var[dom, 4], 1/obs.kappa)))
        self.assertAlmostEqual(obs.kappa, 3/.0025/12)


if __name__ == '__main__':
    unittest.main()
