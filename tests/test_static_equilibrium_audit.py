"""Section 7 of RGBDiekf_Codex_Static_Point_Equilibrium_Root_Cause_Audit.md: the oracles the static-point audit relies on.
(1) a static point with exactly zero residual keeps its mean through predict and update; (2) a point with no valid row
keeps its prior exactly and moves only by dt v in the prediction; (3) an occluder in front of a static point in ONE
camera removes only that camera's row; (4) Delta_v = P_vx P_xx^-1 Delta_x for a position-only Kalman update and for
accepted IEKF states, and the exact trajectory attribution closes on non-uniform times; (5) the audit's linearization
check: no cell change and a vanishing error for a small in-cell step, a reported cell crossing (not a Jacobian failure)
for a multi-pixel step; (6) the audit's comparable-row objective and the oracle's acceptance codes agree with the
production update; (7) tracing on/off parity through train.py. Diagnostic only: no inference code is exercised beyond
its public functions."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tests.projective_fixtures import camera, smooth_field, noise, POLICY, FieldDataset, write_scene, hires
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, CODE
from estimator.point_kalman_cuda import predict, update_points, process_noise, transition
from tools import root_cause_oracle as O
from tools import audit_static_equilibrium as A

DT = 1/30.
QC = torch.eye(3, dtype=torch.float64)*.01


def prior(x, sig=.002, coupling=0.):
    n = len(x)
    s = torch.zeros((n, 6), dtype=torch.float64); s[:, :3] = torch.as_tensor(x, dtype=torch.float64)
    P = torch.zeros((n, 6, 6), dtype=torch.float64); P[:, :3, :3] = sig**2*torch.eye(3, dtype=torch.float64); P[:, 3:, 3:] = torch.eye(3, dtype=torch.float64)
    P[:, :3, 3:] = P[:, 3:, :3] = coupling*torch.eye(3, dtype=torch.float64)
    return s, P


def flat_scene(masks=(True, True), cams=None):
    """Fronto-parallel plane z = 2 m, constant colour 0.5 and constant depth 2 m in every camera (cameras look along +z,
    translated in x): every point of the plane has r = 0 exactly in colour and to round-off in depth."""
    W, H = 96, 72
    cams = cams or [camera('a', np.eye(3), np.zeros(3)), camera('b', np.eye(3), np.array([-.2, 0., 0.]))]
    fr = []
    for c, m in zip(cams, masks):
        fr.append((np.full((H, W, 3), .5), np.full((H, W), 2.), np.full((H, W), m)))
    ds = FieldDataset(cams, [fr])
    return ds, CameraRig(cams, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)


class ZeroResidualOracleTest(unittest.TestCase):
    def test_static_point_with_zero_residual_keeps_its_mean(self):
        self.assertEqual(POLICY['rendered_mask_measurement']['mode'], 'off')              # no mask-induced update exists by default
        ds, rig, frame = flat_scene()
        x = np.array([[.05, .02, 2.], [.09, .04, 2.], [.11, -.05, 2.]])             # inside both cameras
        s0, P0 = prior(x)
        sm, Pm = predict(s0, P0, DT, QC)
        self.assertTrue(torch.equal(sm, s0))                                                 # v = 0: prediction keeps the mean exactly
        sp, Pp, rep, _ = update_points(sm, Pm, frame, rig, torch.full((3, 3), .5, dtype=torch.float64), noise(), POLICY, 10, 1e-8)
        self.assertTrue(bool(rep['observed'].all()))
        self.assertTrue(bool((rep['nrows'] == 2).all()))
        # r_rgb = 0 exactly; r_D = Z - D is round-off of the bilinear weights (sum of weights = 1 to 1 ulp)
        self.assertLess(float((sp[:, :3]-sm[:, :3]).abs().max()), 1e-12)
        self.assertLess(float(sp[:, 3:].abs().max()), 1e-9)


class NoObservationOracleTest(unittest.TestCase):
    def test_no_valid_row_keeps_the_prior_and_prediction_moves_by_dt_v(self):
        ds, rig, frame = flat_scene(masks=(False, False))
        x = np.array([[.05, .02, 2.], [.09, .04, 2.]])
        s0, P0 = prior(x, coupling=1e-4)
        s0[:, 3:] = torch.tensor([.3, -.1, .05], dtype=torch.float64)
        sm, Pm = predict(s0, P0, DT, QC)
        np.testing.assert_allclose(sm[:, :3].numpy(), (s0[:, :3]+DT*s0[:, 3:]).numpy(), rtol=0, atol=1e-15)
        F, Q = transition(DT), process_noise(DT, QC)
        np.testing.assert_allclose(Pm.numpy(), (F@P0@F.T+Q).numpy(), rtol=1e-14, atol=1e-18)
        sp, Pp, rep, _ = update_points(sm, Pm, frame, rig, torch.full((2, 3), .5, dtype=torch.float64), noise(), POLICY, 10, 1e-8)
        self.assertTrue(torch.equal(sp, sm)); self.assertTrue(torch.equal(Pp, Pm))
        self.assertTrue(bool((rep['acceptance'] == 4).all()))                               # NO_VALID_ROWS
        self.assertTrue(bool((rep['reason'] == CODE['MASK_EXCLUDED']).all()))


class OneCameraOcclusionOracleTest(unittest.TestCase):
    def test_occluder_in_one_camera_removes_only_that_row(self):
        ds, rig, frame = flat_scene()
        a = np.array([.05, .02, 2.])                                    # static point on the measured plane
        b = a/2.                                                        # transported occluder on a's ray of camera 0, 1 m in front
        s, P = prior(np.stack([a, b]))
        sp, Pp, rep, _ = update_points(s, P, frame, rig, torch.full((2, 3), .5, dtype=torch.float64), noise(), POLICY, 10, 1e-8)
        self.assertEqual(int(rep['reason'][0, 0]), CODE['SELF_OCCLUDED'])                   # hidden in camera 0 only
        self.assertEqual(int(rep['reason'][1, 0]), CODE['VALID'])                           # camera 1 still measures it
        self.assertEqual(int(rep['reason'][1, 1]), CODE['OUTSIDE_IMAGE_FOOTPRINT'])         # the occluder is not in camera 1
        self.assertTrue(bool(rep['observed'][0])); self.assertEqual(int(rep['nrows'][0]), 1)
        self.assertLess(float((sp[0, :3]-s[0, :3]).abs().max()), 1e-12)                     # its one row has r = 0


class CrossCovarianceIdentityTest(unittest.TestCase):
    def test_position_only_kalman_update(self):
        g = np.random.default_rng(1)
        for _ in range(20):
            B = g.normal(size=(6, 6)); Pm = B@B.T+.1*np.eye(6)
            m = int(g.integers(1, 9))*4
            H = np.zeros((m, 6)); H[:, :3] = g.normal(size=(m, 3)); R = np.diag(g.uniform(.01, 1, m)); r = g.normal(size=m)
            d = Pm@H.T@np.linalg.solve(H@Pm@H.T+R, r)
            np.testing.assert_allclose(d[3:], Pm[3:, :3]@np.linalg.solve(Pm[:3, :3], d[:3]), rtol=1e-9, atol=1e-12)

    def test_accepted_iekf_states(self):
        from tests.test_point_kalman_cuda import plane_problem
        rig, frame, x, rgb0 = plane_problem('cpu')
        s0, P0 = prior(x.numpy(), .004, 0.)
        sm, Pm = predict(s0, P0, DT, QC)                                   # P_xv from the prediction, as in the filter
        sm[:, :3] += torch.tensor([.0015, -.001, .003], dtype=torch.float64)
        sp, Pp, rep, tr = update_points(sm, Pm, frame, rig, rgb0, noise(), POLICY, 10, 1e-8, trace_ids=[0, 3])
        obs = rep['observed']
        self.assertGreater(int(obs.sum()), 10)
        dx, dv = (sp-sm)[obs, :3], (sp-sm)[obs, 3:]
        L = Pm[obs, 3:, :3]@torch.linalg.inv(Pm[obs, :3, :3])
        np.testing.assert_allclose(dv.numpy(), (L@dx[..., None])[..., 0].numpy(), rtol=1e-9, atol=1e-12)
        pre = {'P_minus': Pm[3].tolist(), 'P_previous': P0[3].tolist(), 'Q': process_noise(DT, QC).tolist(), 'dt_s': DT}
        c = A.coupling(pre, sm[3].numpy(), sp[3].numpy(), DT, x[3].numpy())
        self.assertLess(c['identity_rel_error'], 1e-9)
        tr_ = c['P_vx_minus_terms_trace']
        self.assertAlmostEqual(tr_['P_vx_previous']+tr_['dt_P_vv_previous']+tr_['Q_vx'], tr_['total'], places=12)

    def test_trajectory_attribution_closes_on_non_uniform_times(self):
        g = np.random.default_rng(2)
        K, n = 9, 5
        t = np.cumsum(np.r_[0, g.uniform(.02, .06, K-1)])
        X, V = np.zeros((K, n, 3)), np.zeros((K, n, 3), np.float32)
        X[0] = g.normal(size=(n, 3)); dxi, dvi = g.normal(scale=1e-3, size=(K, n, 3)), g.normal(scale=.1, size=(K, n, 3)).astype(np.float32)
        st = np.ones((K, n), np.int64)
        for k in range(1, K):
            V[k] = V[k-1]+dvi[k]
            X[k] = X[k-1]+(t[k]-t[k-1])*V[k-1].astype(float)+dxi[k]
        T = {'X': X, 'V': V.astype(float), 'V32': V, 't': t, 'st': st, 'nrows': st, 'acc': st}
        ev, att, aux = A.events(T, None)
        self.assertLess(float(att['closure_max_abs_m'].max()), 1e-12)
        np.testing.assert_allclose(att['position_corrections_vector'], dxi[1:].sum(0), atol=1e-9)
        tT = t[-1]
        np.testing.assert_allclose(att['velocity_injections_vector'], ((tT-t[1:])[:, None, None]*(V[1:]-V[:-1]).astype(float)).sum(0), atol=1e-12)
        self.assertTrue(bool((ev['first_velocity_change'] == 1).all()))


def field_problem(uv=(40.3, 30.4)):
    field = smooth_field()
    cam = camera('a', np.eye(3), np.zeros(3))
    ds = FieldDataset([cam], [[field]])
    from tests.test_projective_measurement import on_surface
    rig = CameraRig([cam], 'cpu')
    x = on_surface(rig, [uv], field)[0].numpy()
    a = np.asarray(O.sample(O.Image(ds.observation(0, 0), np.float64), O.Cam(cam), *uv, O.Noise([.0025]*3, 1e-4, 0.), POLICY)['rgb'])
    s, P = prior(x[None], .01)
    nz = noise()
    on = O.Noise(nz.rgb_variance.numpy(), nz.depth_variance, nz.depth_squared_variance, nz.rgb_model_variance, nz.depth_model_variance, nz.interpolation_variance)
    return O.Problem([O.Cam(cam)], [O.Image(ds.observation(0, 0), np.float64)], a, s[0].numpy(), P[0].numpy(), [O.VALID], on, POLICY, 10, 1e-8), x


class LinearizationOracleTest(unittest.TestCase):
    def test_in_cell_step_is_linear_and_cell_crossing_is_reported(self):
        prob, x = field_problem()
        s = np.r_[x, 0, 0, 0]
        cam = prob.cams[0]
        ray = cam.R.T@np.array([0., 0., 1.])
        small = np.r_[1e-7*np.array([1., -.5, .3]), 0, 0, 0]
        out = A.step_linearization(prob, s, small, [0])
        self.assertFalse(out['classification']['any_cell_crossing'])
        e = out['cameras'][0]['h_linearization_error_whitened']
        self.assertLess(float(np.abs(e).max()), 1e-6)
        fd = A.fd_check(prob, x, [0])
        self.assertFalse(fd[0]['crosses_pixel_cell']); self.assertLess(fd[0]['rel_error'], 1e-5)
        big = np.r_[np.array([.012, .008, 0.]), 0, 0, 0]                    # ~3 px lateral: several cells
        out = A.step_linearization(prob, s, big, [0])
        self.assertTrue(out['classification']['any_cell_crossing'])
        self.assertGreater(out['classification']['max_pixel_displacement_px'], 2.)
        self.assertIsNotNone(out['alphas'][0]['gain_ratio_rho'])
        del ray


class AcceptanceAuditTest(unittest.TestCase):
    def test_comparable_objective_and_oracle_acceptance_match_production(self):
        from tests.test_root_cause_oracle import scene
        ds, x, rgb = scene()
        rig, frame = CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        n = len(x)
        offs = np.zeros((n, 3)); offs[::3] = [.05, 0., 0.]; offs[1::3] = [.0015, -.001, .004]
        s, P = prior(x+offs, .004, 0.)
        sm, Pm = predict(s, P, DT, QC)
        ids = list(range(n))
        sp, Pp, rep, tr = update_points(sm, Pm, frame, rig, torch.as_tensor(rgb), noise(), POLICY, 10, 1e-8, trace_ids=ids)
        cams = [O.Cam(c) for c in ds.cameras]; imgs = [O.Image(ds.observation(0, c), np.float64) for c in range(len(cams))]
        nz = noise()
        on = O.Noise(nz.rgb_variance.numpy(), nz.depth_variance, nz.depth_squared_variance, nz.rgb_model_variance, nz.depth_model_variance, nz.interpolation_variance)
        codes = set()
        for pid in ids:
            fin = tr['final'][pid]
            frozen = [O.CODE[c['visibility_frozen']] for c in tr['iterations'][0]['points'][pid]['cameras']]
            recheck = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']]
            prob = O.Problem(cams, imgs, rgb[pid], np.array(fin['s_minus']), np.array(fin['P_minus']), frozen, on, POLICY, 10, 1e-8, recheck)
            res = prob.run(POLICY['update_acceptance'])
            self.assertEqual(A.O_ACC[res['acceptance_code']], fin['acceptance'], pid)
            codes.add(fin['acceptance'])
            used = [o['camera'] for o in prob.used(prob.rows(prob.s_minus[:3]))]
            truth = np.r_[x[pid], prob.s_minus[3:]]
            cmp_ = A.comparable_objective(prob, truth, prob.s_minus, used, used)
            self.assertEqual(cmp_['common_cameras'], used)
            if pid % 3 == 1:            # a few mm off a textured surface: the comparable data cost is above the truth's
                self.assertTrue(used)
                self.assertTrue(cmp_['data_only_b_worse_than_a'], pid)
                self.assertFalse(A.comparable_objective(prob, prob.s_minus, truth, used, used)['data_only_b_worse_than_a'], pid)
            elif pid % 3 == 2:          # predicted exactly at the truth: identical states, identical costs
                self.assertEqual(cmp_['a']['total'], cmp_['b']['total'])
        self.assertGreater(len(codes), 1, codes)                                            # accepted and rejected cases both exercised


class TraceParityTest(unittest.TestCase):
    def test_tracing_does_not_change_any_published_array(self):
        import train
        from tests.image_fixtures import two_plate_scene
        root = Path(tempfile.mkdtemp())
        scene = write_scene(hires(two_plate_scene('static', frames=3, cameras=4)), root/'scene')
        base = ['--scene', str(scene), '--fps', '30', '--direction', 'forward', '--stop-after', 'free', '--debug', 'none', '--no-evaluate', '--no-qa']
        train.main([*base, '--output', str(root/'off'), '--debug-frames', ''])
        train.main([*base, '--output', str(root/'on'), '--debug-frames', '1,2', '--debug-point-ids', '0,7,500,1200'])
        self.assertFalse(list((root/'off'/'free_forward'/'numeric').glob('trace_frame_*.json')))
        self.assertEqual(len(list((root/'on'/'free_forward'/'numeric').glob('trace_frame_*.json'))), 2)
        par = A.parity(root/'on', root/'off')
        self.assertTrue(par)
        self.assertEqual(set(par.values()), {'IDENTICAL'}, json.dumps(par))



class CovarianceCalibrationRuleTest(unittest.TestCase):
    def test_second_moment_match(self):
        g = np.random.default_rng(3)
        v = g.uniform(.5, 2., 20000); r2 = (v+.7)*g.chisquare(1, 20000)               # true additive variance 0.7
        m = A._mean_match(r2, v)
        self.assertAlmostEqual(float(np.mean(r2/(v+m))), 1., places=9)
        self.assertLess(abs(m-.7), .05)
        self.assertEqual(A._mean_match(.5*v*g.chisquare(1, 20000), v), 0.)              # already explained: no inflation


if __name__ == '__main__':
    unittest.main()
