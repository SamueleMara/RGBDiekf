"""Contract section 13 rigid/joint gates on image-estimated trajectories (no correspondences, no GT inside the estimators)."""
import tempfile, unittest
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, source_parts, true_positions
from scene.config import load_config
from estimator import phase1 as image_phase1
from estimator import motion_coherence_support as image_coherence
from estimator import rigid_part_iekf as rigid_image_iekf
from estimator import joint_iekf as joint_image
from estimator.phase1 import endpoint_catalogue
from estimator.sparse_iekf import IEKFConfig
from estimator.point_renderer import RenderPolicy, footprints
from estimator.observation_terms import ObservationPolicy
from estimator.rigid_part_iekf import process, exp4, log4, inv4, RigidConfig
from rigeuler_ref.joint_manifold import se3_left_jacobian
from utils.backend import Backend
from utils.lie import log as se3log

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pipeline(motion, anchor='first', frames=20, cameras=4, noise=None):
    """20 frames (16 scored after the joint burn-in); 1.2 cm/frame slide or 0.06 rad/frame hinge."""
    ds = two_plate_scene(motion, frames=frames, cameras=cameras, noise=noise, speed=.012, rate=.06)
    config, backend, root = load_config(), Backend(DEVICE), tempfile.mkdtemp()
    pol, op = RenderPolicy(depth_variance=config.depth_variance), ObservationPolicy()
    part, arr, rho = image_phase1.run(ds, config, backend, root+'/p1', anchor)
    k0 = 0 if anchor == 'first' else frames-1
    coh = image_coherence.run(part, arr, ds.times, root+'/coh', k0)
    out = {'ds': ds, 'part': part, 'coh': coh, 'k0': k0, 'parts': source_parts(ds, part.source)}
    if coh['status'] == 'RESOLVED':
        out['rigid'] = rigid_image_iekf.run(ds, config, backend, part, arr, rho, coh['point_support'], k0, root+'/rigid', pol, op)
        out['joint'] = joint_image.run(ds, config, backend, part, rho, out['rigid'], k0, root+'/joint', pol, op)
    return out


def pose_errors(ds, poses, k0):
    e = [se3log(np.linalg.inv(ds.poses(k)[1]@np.linalg.inv(ds.poses(k0)[1]))@poses[k]) for k in range(len(ds.times))]
    return max(np.linalg.norm(x[:3]) for x in e), max(np.linalg.norm(x[3:]) for x in e)


class PipelineTest(unittest.TestCase):
    def check_core(self, r, precision=.95, recall=.75):
        sup, parts = r['coh']['point_support'], r['parts']
        self.assertEqual(r['coh']['status'], 'RESOLVED')
        self.assertGreaterEqual((parts[sup] == 1).mean(), precision)
        self.assertGreaterEqual(sup[parts == 1].mean(), recall)

    def test_prismatic_forward(self):
        r = pipeline('prismatic')
        self.check_core(r)
        t, a = pose_errors(r['ds'], r['rigid']['poses'], 0)
        self.assertLess(t, 2e-3); self.assertLess(a, np.radians(.5))
        self.assertEqual(r['joint']['status'], 'PRISMATIC')
        print('prismatic forward joint tests:', {k: (round(v['p_value'], 5), round(v['mean_differential'], 1)) for k, v in r['joint']['tests'].items()})
        e = np.asarray(r['joint']['candidates']['PRISMATIC']['axis']); truth = np.array([1, .3, 0])/np.linalg.norm([1, .3, 0])
        self.assertLess(np.degrees(np.arccos(abs(e@truth))), 1.)

    def test_prismatic_backward_chronological(self):
        r = pipeline('prismatic', 'last')
        self.check_core(r)
        t, a = pose_errors(r['ds'], r['rigid']['poses'], r['k0'])
        self.assertLess(t, 2e-3)
        self.assertEqual(r['joint']['status'], 'PRISMATIC')
        q = np.asarray(r['joint']['candidates']['PRISMATIC']['q'])
        self.assertAlmostEqual(q[-1], 0., places=12)          # endpoint gauge at the final frame
        self.assertGreater(abs(q[0]), .1)                       # chronological output, full travel

    def test_revolute_forward(self):
        r = pipeline('revolute')
        self.check_core(r, recall=.6)
        t, a = pose_errors(r['ds'], r['rigid']['poses'], 0)
        self.assertLess(t, 5e-3); self.assertLess(a, np.radians(.6))
        self.assertEqual(r['joint']['status'], 'REVOLUTE')
        c = r['joint']['candidates']['REVOLUTE']
        self.assertLess(np.degrees(np.arccos(abs(np.asarray(c['axis'])@[0, 1, 0]))), 1.)
        a = np.asarray(c['axis_point']); e = np.asarray(c['axis'])
        self.assertLess(np.linalg.norm(np.cross(a-np.array([-.2, 0, .06]), e)), .01)

    def test_static_scene_has_no_identifiable_joint(self):
        r = pipeline('static')
        if r['coh']['status'] == 'RESOLVED':       # the pinned extractor can resolve noise (documented); the joint stage must not
            print('static joint tests:', {k: (v['p_value'], v['mean_differential']) for k, v in r['joint']['tests'].items()})
            self.assertEqual(r['joint']['status'], 'NO_IDENTIFIABLE_JOINT')
            self.assertIsNone(r['joint']['selected'])

    def test_sparse_noisy_prismatic_never_picks_the_wrong_type(self):
        r = pipeline('prismatic', cameras=3, noise=(.03, .003))
        print('noisy prismatic joint:', r['joint']['status'] if 'joint' in r else r['coh']['status'])
        if 'joint' in r: self.assertIn(r['joint']['status'], ('PRISMATIC', 'UNRESOLVED_INDISTINGUISHABLE', 'NO_IDENTIFIABLE_JOINT'))


class RigidFromZeroInitTest(unittest.TestCase):
    def test_image_iekf_recovers_hinge_without_phase1_motion(self):
        ds = two_plate_scene('revolute', frames=6, cameras=4)
        config = load_config()
        _, part, _ = endpoint_catalogue(ds, 'first', config, IEKFConfig()); rho = footprints(part, ds)
        arr = {'xyz': np.repeat(part.xyz[None], 6, 0)}
        parts = source_parts(ds, part.source)
        res = rigid_image_iekf.run(ds, config, Backend(DEVICE), part, arr, rho, parts == 1, 0, tempfile.mkdtemp(),
                                   RenderPolicy(depth_variance=config.depth_variance), ObservationPolicy())
        t, a = pose_errors(ds, res['poses'], 0)
        self.assertLess(t, 5e-3); self.assertLess(a, np.radians(.6))


class PredictiveTestCalibration(unittest.TestCase):
    """Diebold-Mariano / Newey-West: size under H0 with serially dependent loss differentials, and power."""
    def test_size_and_power(self):
        from joint_reduction.predictive_tests import diebold_mariano
        g = np.random.default_rng(0); n, reps = 56, 4000
        def ar1(phi, shift):
            e = g.normal(size=(reps, n)); d = np.zeros_like(e)
            for t_ in range(n): d[:, t_] = (phi*d[:, t_-1] if t_ else 0)+e[:, t_]
            return d+shift
        for phi in (0., .5, .7):
            d = ar1(phi, 0.)
            rej = np.mean([diebold_mariano(x, np.zeros(n), 'two-sided')['p_value'] < .05 for x in d[:2000]])
            self.assertLess(rej, .08, f'phi={phi}: size {rej}')     # nominal .05
        d = ar1(.5, 1.2)        # mean shift 0.6 long-run SDs: noncentrality ~4.5 at n = 56
        power = np.mean([diebold_mariano(x, np.zeros(n), 'greater')['p_value'] < .01 for x in d[:500]])
        self.assertGreater(power, .8)
        d = ar1(.5, 0.)+.05*np.arange(n)       # trending evidence (a hypothesis falling further behind): must keep power
        power = np.mean([diebold_mariano(x, np.zeros(n), 'greater')['p_value'] < .01 for x in d[:500]])
        self.assertGreater(power, .8)


class RigidConventionTest(unittest.TestCase):
    def test_constant_twist_log_and_process_conventions(self):
        cfg = RigidConfig()
        xi = torch.tensor([.05, -.02, .01, .1, -1.8, .3], dtype=torch.float64)   # spatial twist [v; omega], rad/s
        for dt in (1/30, -1/30):
            T = [exp4(k*dt*xi) for k in range(4)]
            self.assertLess(float((log4(T[1]@inv4(T[0]))/dt-xi).abs().max()), 1e-12)
            for k in range(3):
                E, F, Q = process(xi, dt, cfg)
                self.assertLess(float((E@T[k]-T[k+1]).abs().max()), 1e-12)

    def test_phase2_tracks_fast_hinge_from_exact_trajectories(self):
        """Isolates Phase 2: exact point trajectories (Phase 1 replaced by ground truth) at 0.12 rad/frame."""
        ds = two_plate_scene('revolute', frames=8, cameras=4)
        config = load_config()
        _, part, _ = endpoint_catalogue(ds, 'first', config, IEKFConfig()); rho = footprints(part, ds)
        parts = source_parts(ds, part.source)
        arr = {'xyz': np.stack([true_positions(ds, part.xyz, parts, k, 0) for k in range(8)])}
        res = rigid_image_iekf.run(ds, config, Backend(DEVICE), part, arr, rho, parts == 1, 0, tempfile.mkdtemp(),
                                   RenderPolicy(depth_variance=config.depth_variance), ObservationPolicy())
        t, a = pose_errors(ds, res['poses'], 0)
        print('exact-trajectory fast hinge: max translation', t, 'max rotation deg', np.degrees(a))
        # measured 1.30 deg / 1.8 mm; the gate separates tracking from losing the hinge (24.6 deg measured when initialised
        # from a poorly tracked Phase 1), it is not a precision target
        self.assertLess(t, 5e-3); self.assertLess(a, np.radians(5.))


class LieJacobianTest(unittest.TestCase):
    def test_process_jacobian_matches_finite_difference(self):
        cfg = RigidConfig()
        T = exp4(torch.tensor([.1, -.2, .3, .2, -.1, .4], dtype=torch.float64)); xi = torch.tensor([.3, .1, -.2, .5, .2, -.3], dtype=torch.float64)
        for dt in (1/59, -1/59):
            E, F, Q = process(xi, dt, cfg)
            pred = lambda T, xi: (exp4(dt*xi)@T, xi)
            Tm, _ = pred(T, xi); h = 1e-7; J = torch.zeros(12, 12, dtype=torch.float64)
            for k in range(12):
                d = torch.zeros(12, dtype=torch.float64); d[k] = h
                Tp, xp = pred(exp4(d[:6])@T, xi+d[6:])
                d[k] = -h; Tn, xn = pred(exp4(d[:6])@T, xi+d[6:])
                J[:6, k] = (log4(Tp@inv4(Tm))-log4(Tn@inv4(Tm)))/(2*h); J[6:, k] = (xp-xn)/(2*h)
            self.assertLess((J-F).abs().max().item(), 1e-6)
            self.assertGreaterEqual(torch.linalg.eigvalsh(Q).min().item(), -1e-15)
            self.assertTrue(torch.allclose(Q[:6, 6:], torch.diag(torch.tensor([1.]*6, dtype=torch.float64)*dt*abs(dt)/2)))

    def test_prior_error_jacobian_is_not_identity_and_matches_fd(self):
        Tm = exp4(torch.tensor([.2, .1, 0, .3, .1, -.2], dtype=torch.float64)); Tm_inv = inv4(Tm)
        T = exp4(torch.tensor([.05, -.1, .02, .4, -.3, .2], dtype=torch.float64))@Tm
        e = lambda T: log4(T@Tm_inv)
        J = torch.linalg.inv(se3_left_jacobian(e(T).reshape(1, 6))[0])
        h = 1e-7; Jfd = torch.zeros(6, 6, dtype=torch.float64)
        for k in range(6):
            d = torch.zeros(6, dtype=torch.float64); d[k] = h
            Jfd[:, k] = (e(exp4(d)@T)-e(exp4(-d)@T))/(2*h)
        self.assertLess((J-Jfd).abs().max().item(), 1e-6)
        self.assertGreater((J-torch.eye(6, dtype=torch.float64)).abs().max().item(), 1e-2)


if __name__ == '__main__':
    unittest.main()
