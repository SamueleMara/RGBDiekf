"""Independent minimal oracles of RGBDiekf_Static_Drift_INNER_IEKF_Only_Codex_2026-10-30.md section 9 (no scene run).

1 exact slanted plane: bilinear METRIC depth is biased (behind the surface, sign of the convexity of Z in (u, v)), bilinear
  INVERSE depth is exact; analytic interpolant gradients vs central differences; delta-method variance vs Monte Carlo.
2 depth discontinuity: a 2x2 cell across a large step is rejected by the same-surface rule, while a second surface within
  the declared allowance is admitted as "same surface" and biases the sample: the local spread test cannot establish
  identity (characterised, not asserted as correct).
3 6D Kalman step: the production update_points vs the independent float64 oracle on a ray-cast frame with a prior that
  has non-zero P_vx; one depth row vs the dense textbook K; a velocity correction only through P_vx.
4 prediction and loss: an erroneous accepted velocity integrates exactly (x_N = x_0 + sum dt v), the covariance grows
  monotonically with no information gain, production predict == oracle predict.
5 acceptance: the production rule is a chi-square adequacy test of the iterate, not a fixed-support decrease test; the
  oracle reproduces every decision and the cases where an accepted state RAISES the fixed-support cost are counted.
Thresholds: float64 tolerances only (no new decision threshold)."""
import math
import unittest
import numpy as np
import torch
from tools import root_cause_oracle as O
from tests.projective_fixtures import POLICY, noise as prod_noise
from estimator.point_kalman_cuda import predict, update_points


class Img:
    def __init__(self, depth, rgb=None):
        H, W = depth.shape
        self.depth = depth; self.rgb = np.zeros((H, W, 3)) if rgb is None else rgb
        self.mask = np.ones((H, W), bool); self.raw_valid = np.isfinite(depth) & (depth > 0)


class PinCam:
    def __init__(self, f=500., cx=31.5, cy=23.5, W=64, H=48):
        self.fx = self.fy = f; self.cx, self.cy, self.W, self.H, self.near = cx, cy, W, H, .01


def plane_depth(cam, n, d):
    """Z(u, v) of the plane n . X = d (camera frame): Z = d / (n . [(u-cx)/fx, (v-cy)/fy, 1])."""
    u, v = np.meshgrid(np.arange(cam.W, dtype=float), np.arange(cam.H, dtype=float))
    return d/(n[0]*(u-cam.cx)/cam.fx+n[1]*(v-cam.cy)/cam.fy+n[2])


def oracle_noise(sigma_d=1e-3):
    return O.Noise([1e-4]*3, sigma_d**2, 0., 0., 0., 'sum_w2')


class SlantedPlaneTest(unittest.TestCase):
    def test_metric_biased_inverse_exact_jacobian_and_variance(self):
        cam = PinCam(); n = np.array([.0, .55, .835]); n /= np.linalg.norm(n); d = 1.2
        D = plane_depth(cam, n, d); img = Img(D)
        pol = {**POLICY, 'same_surface_n_sigma': 1e9}                     # no rejection: the interpolants themselves
        g = np.random.default_rng(0); errs_m, errs_i = [], []
        for _ in range(200):
            u, v = g.uniform(5, cam.W-6), g.uniform(5, cam.H-6)
            Z = d/(n[0]*(u-cam.cx)/cam.fx+n[1]*(v-cam.cy)/cam.fy+n[2])
            sm = O.sample(img, cam, u, v, oracle_noise(), pol); si = O.sample(img, cam, u, v, oracle_noise(), {**pol, 'depth_interpolation': 'inverse'})
            errs_m.append(sm['depth']-Z); errs_i.append(si['depth']-Z)
            self.assertAlmostEqual(sm['depth_inverse'], si['depth'], places=15)
            for s_, pol_ in ((sm, pol), (si, {**pol, 'depth_interpolation': 'inverse'})):      # analytic interpolant gradient vs central differences (in-cell)
                e = 1e-6
                if min(u-math.floor(u), math.ceil(u)-u, v-math.floor(v), math.ceil(v)-v) < 10*e: continue
                fd = [(O.sample(img, cam, u+e, v, oracle_noise(), pol_)['depth']-O.sample(img, cam, u-e, v, oracle_noise(), pol_)['depth'])/(2*e),
                      (O.sample(img, cam, u, v+e, oracle_noise(), pol_)['depth']-O.sample(img, cam, u, v-e, oracle_noise(), pol_)['depth'])/(2*e)]
                np.testing.assert_allclose(s_['grad_depth'], fd, rtol=1e-6, atol=8*np.finfo(float).eps*s_['depth']/e)   # central-difference rounding bound
        errs_m, errs_i = np.array(errs_m), np.array(errs_i)
        self.assertLess(np.abs(errs_i).max(), 1e-12)                          # inverse depth exact on one plane
        self.assertGreater(errs_m.min(), -1e-12)                              # metric bilinear: never in front ...
        self.assertGreater(errs_m.max(), 1e-7)                                # ... and biased behind the surface
        print(f'\n[oracle 1] slanted plane: metric bilinear bias max {errs_m.max()*1e6:.3f} um (mean {errs_m.mean()*1e6:.3f} um), inverse max |err| {np.abs(errs_i).max():.1e} m')
        # delta-method variance of the inverse interpolant vs Monte Carlo of independent pixel noise
        u, v = 20.37, 17.81; s = O.sample(img, cam, u, v, oracle_noise(1e-3), {**pol, 'depth_interpolation': 'inverse'})
        u0, v0 = s['pixel00']; Dn = s['neighbour_depth']; w = s['w']
        noise_ = g.normal(0, 1e-3, (200000, 4))
        mc = 1/((w/(Dn+noise_)).sum(1))
        self.assertLess(abs(mc.var()/s['var_depth']-1), .02)


class DiscontinuityTest(unittest.TestCase):
    def test_large_step_rejected_small_step_admitted_as_same_surface(self):
        cam = PinCam(); D = np.full((cam.H, cam.W), 1.0)
        nz = oracle_noise(5.7e-6)
        allow = None
        for step, expect in ((.30, O.DISCONT), (.004, O.VALID)):
            Dd = D.copy(); Dd[:, 32:] = 1.0+step                               # a second surface behind, from column 32
            s = O.sample(Img(Dd), cam, 31.4, 20.5, nz, POLICY)
            self.assertEqual(s['reason'], expect)
            if expect == O.VALID:
                allow = s['same_surface_allowance_m']
                bias = s['depth']-1.0                                          # the point is on the FRONT surface at u = 31.4
                self.assertGreater(bias, .4*step-1e-12)                        # 2x2 mixes the two surfaces: wrong-surface depth
                print(f'\n[oracle 2] step {step*1e3:.1f} mm admitted (spread {s["same_surface_spread_m"]*1e3:.1f} mm <= allowance {allow*1e3:.2f} mm): '
                      f'sample {bias*1e3:.2f} mm behind the true surface = {bias/math.sqrt(s["var_depth"]):.0f} sigma_R')


class KalmanStepTest(unittest.TestCase):
    def test_production_step_equals_oracle_and_velocity_moves_only_through_Pvx(self):
        from tests.image_fixtures import two_plate_scene, catalog, look_at
        from estimator.projective_camera import CameraRig
        from estimator.projective_sensor import MeasuredFrame
        ds = two_plate_scene('static', frames=1, cameras=4)
        ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
        xyz, rgb, _, _ = catalog(ds, 0)
        ok = np.flatnonzero((np.abs(xyz[:, 0]) < .3) & (np.abs(xyz[:, 1]) < .2) & (np.abs(xyz[:, 2]) < 1e-9))
        rig, frame = CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        nz = prod_noise(); onz = O.Noise(nz.rgb_variance.numpy(), nz.depth_variance, nz.depth_squared_variance, nz.rgb_model_variance, nz.depth_model_variance, nz.interpolation_variance)
        cams = [O.Cam(c) for c in ds.cameras]; imgs = [O.Image(ds.observation(0, c), np.float64) for c in range(len(ds.cameras))]
        g = np.random.default_rng(1); worst = 0.
        for p in ok[::max(1, len(ok)//6)][:6]:
            A = g.normal(size=(6, 6)); P = A@A.T*1e-6+np.diag([1e-5]*3+[1e-2]*3)
            P[:3, 3:] = 3e-4*np.eye(3); P[3:, :3] = 3e-4*np.eye(3)          # prior position-velocity correlation
            s = np.r_[xyz[p]+g.normal(0, 2e-3, 3), 0, 0, 0]
            sp, Pp, rep, _ = update_points(torch.tensor(s[None]), torch.tensor(P[None]), frame, rig, torch.tensor(rgb[p:p+1], dtype=torch.float64), nz, POLICY, 10, 1e-8)
            prob = O.Problem(cams, imgs, rgb[p], s, P, [O.VALID]*4, onz, POLICY, 10, 1e-8)
            res = prob.run(POLICY['update_acceptance'])
            self.assertEqual(bool(rep['observed'][0]), res['observed'])
            worst = max(worst, float(np.abs(sp[0].numpy()-res['s_plus']).max()))
            np.testing.assert_allclose(sp[0].numpy(), res['s_plus'], atol=1e-10)
            np.testing.assert_allclose(Pp[0].numpy(), res['P_plus'], rtol=1e-8, atol=1e-16)
            # one depth row alone (dense textbook K) and the velocity correction is P_vx (P_xx)^-1 dx for position-only rows
            rows = prob.rows(s[:3]); used = prob.used(rows)
            H = np.zeros((1, 6)); H[0, :3] = used[0]['H'][3]; Rr = np.array([[used[0]['var'][3]]])
            K = P@H.T@np.linalg.inv(H@P@H.T+Rr); d = (K@np.array([used[0]['r'][3]]))
            np.testing.assert_allclose(d[3:], P[3:, :3]@np.linalg.solve(P[:3, :3], d[:3]), rtol=1e-9, atol=1e-15)
        print(f'\n[oracle 3] production vs float64 oracle accepted state max |diff| {worst:.1e} m')


class PredictionLossTest(unittest.TestCase):
    def test_false_velocity_integrates_exactly_without_information_gain(self):
        qc = np.diag([.01, .01, .01]); dt = 1/59.
        s = np.r_[.1, .2, 1.5, .05, -.02, .01]; P = np.diag([1e-6]*3+[1e-3]*3); P[:3, 3:] = P[3:, :3] = 2e-5*np.eye(3)
        st, Pt = torch.tensor(s[None]), torch.tensor(P[None])
        so, Po = s.copy(), P.copy(); prev = P[:3, :3]
        for k in range(30):
            st, Pt = predict(st, Pt, dt, torch.tensor(qc)); so, Po, _, _ = O.predict(so, Po, dt, qc)
            np.testing.assert_allclose(st[0].numpy(), so, atol=1e-15); np.testing.assert_allclose(Pt[0].numpy(), Po, rtol=1e-12, atol=1e-20)
            self.assertGreaterEqual(np.linalg.eigvalsh(Po[:3, :3]-prev).min(), -1e-18)       # P_xx never shrinks without data
            prev = Po[:3, :3]
        np.testing.assert_allclose(so[:3], s[:3]+30*dt*s[3:], atol=1e-14)
        np.testing.assert_allclose(so[3:], s[3:], atol=0)


class AcceptanceTest(unittest.TestCase):
    def test_chi2_adequacy_not_decrease_oracle_reproduces_decisions(self):
        from tests.image_fixtures import two_plate_scene, catalog, look_at
        from estimator.projective_camera import CameraRig
        from estimator.projective_sensor import MeasuredFrame
        ds = two_plate_scene('static', frames=1, cameras=4)
        ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
        xyz, rgb, _, _ = catalog(ds, 0)
        ok = np.flatnonzero((np.abs(xyz[:, 0]) < .3) & (np.abs(xyz[:, 1]) < .2) & (np.abs(xyz[:, 2]) < 1e-9))[::40][:25]
        rig, frame = CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        nz = prod_noise(); onz = O.Noise(nz.rgb_variance.numpy(), nz.depth_variance, nz.depth_squared_variance, nz.rgb_model_variance, nz.depth_model_variance, nz.interpolation_variance)
        cams = [O.Cam(c) for c in ds.cameras]; imgs = [O.Image(ds.observation(0, c), np.float64) for c in range(len(ds.cameras))]
        g = np.random.default_rng(2); raised = accepted = 0
        for p in ok:
            for off in (.002, .008, .02):
                P = np.diag([off**2]*3+[1.]*3); s = np.r_[xyz[p]+g.normal(0, off, 3), 0, 0, 0]
                sp, Pp, rep, _ = update_points(torch.tensor(s[None]), torch.tensor(P[None]), frame, rig, torch.tensor(rgb[p:p+1], dtype=torch.float64), nz, POLICY, 10, 1e-8)
                prob = O.Problem(cams, imgs, rgb[p], s, P, [O.VALID]*4, onz, POLICY, 10, 1e-8)
                res = prob.run(POLICY['update_acceptance'])
                self.assertEqual(int(rep['acceptance'][0]), res['acceptance_code'])
                if res['observed']:
                    accepted += 1
                    cams_acc = [o['camera'] for o in prob.used(res['rows_accepted'])]
                    a, b = prob.cost_on(s, cams_acc), prob.cost_on(res['s_plus'], cams_acc)
                    if not a['lost_rows'] and prob.prior_cost(res['s_plus'])+b['rgb']+b['depth'] > a['rgb']+a['depth']: raised += 1
        print(f'\n[oracle 5] {accepted} accepted updates; {raised} of them RAISE the fixed-support posterior cost relative to the prediction '
              '(allowed by the chi-square adequacy rule; reported, not asserted)')
        self.assertGreater(accepted, 0)


if __name__ == '__main__':
    unittest.main()
