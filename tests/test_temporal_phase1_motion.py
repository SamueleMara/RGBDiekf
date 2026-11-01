"""Gate 3 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: real physical motion is preserved (analytic RGB-D,
tests/temporal_fixtures.py; ground truth used by the TEST only; no static/moving label reaches the estimator).

G3.1 correct frame-0 geometry with known translations: stationary, constant velocity along the image axis and along camera
     Z, mixed 3D acceleration, smooth reversal: b consistent with 0, d_t with the true displacement (NEES against the exact
     Cov(d_t)), small absolute error.
G3.2 geometry perturbation delta AND motion together: b ~ -delta and d ~ d_GT; the production solver equals an independent
     dense NumPy batch Gauss-Newton; compared with the ORIGINAL forward IEKF on the same data.
G3.3 occluder passing in front of static background points: no RGB-D row from a camera that truly cannot see the point,
     other cameras still measure it, identity kept, later evidence corrects the smoothed earlier states.
G3.4 motion of several pixels per frame on a fine texture: every accepted step decreased the objective on its frozen
     support (also when the linearized prediction is wrong); lost support and basin failures are counted, not hidden."""
import unittest
import numpy as np
from scipy.stats import chi2
from tests.temporal_fixtures import (fronto_cameras, AffinePlate, AnalyticDataset, catalogue, run_solver, dense_oracle_map, nees, same_map,
                                     run_original_filter)
from tests.test_temporal_phase1_static import RING, interior_pixels

K = 20
TIMES = np.arange(K)/30.
# Known synthetic generative prior per motion (spec P2 / Stage A): white-acceleration density q_a [m^2/s^3]. A constant
# velocity has zero acceleration (any q_a); the accelerating / reversing fixtures (|a| <= 3.3 m/s^2) need a velocity
# change per interval sqrt(q_a dt) >= |a| dt, i.e. q_a >= |a|^2 dt ~ 0.4: declared 1.
QA = {'stationary': .01, 'constant_image_axis': .01, 'constant_camera_z': .01, 'acceleration_mixed_3d': 1., 'reversal': 1.}
MOTIONS = {'stationary': lambda t: np.zeros(3),
           'constant_image_axis': lambda t: np.array([.3, -.1, 0.])*t,
           'constant_camera_z': lambda t: np.array([0., 0., .3])*t,
           'acceleration_mixed_3d': lambda t: .5*np.array([2., -1., 1.])*t*t+np.array([.05, .02, 0])*t,
           'reversal': lambda t: np.array([.03*np.sin(2*np.pi*t/.6), .01*np.sin(2*np.pi*t/.6), 0.])}


def true_traj(motion, x0, times):
    return np.stack([x0+motion(t)-motion(times[0]) for t in times])                 # [K,N,3]


class MotionPreservedTest(unittest.TestCase):
    def check_motion(self, name, motion, delta=np.zeros(3), qa=.01):
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6), motion=motion)], TIMES)
        cat = catalogue(ds, 6, interior_pixels(20, seed=5)); x0 = cat['x_ref'].copy(); cat['x_ref'] = x0+delta
        res = run_solver(ds, cat, qa=qa)
        ok = res['trajectory_status'].numpy() != 4
        gt = true_traj(motion, x0, TIMES)
        d, dc = res['displacement'].numpy(), res['displacement_cov'].numpy().astype(float)
        b, bc = res['b'].numpy(), res['b_cov'].numpy()
        X = res['smooth_mean'].numpy()[..., :3]
        for i in np.flatnonzero(ok):
            self.assertLess(nees(b[i]+delta, bc[i]), chi2.ppf(.999, 3), (name, i))
            for t in (5, 12, K-1):
                self.assertLess(nees(d[t, i]-(gt[t, i]-gt[0, i]), dc[t, i]), chi2.ppf(.999, 3), (name, t, i))
        err = np.linalg.norm(X[:, ok]-gt[:, ok], axis=-1)
        mot = np.linalg.norm(gt[-1, ok]-gt[0, ok], axis=-1)
        return ds, cat, res, err, mot, gt

    def test_G3_1_known_translations(self):
        for name, motion in MOTIONS.items():
            ds, cat, res, err, mot, gt = self.check_motion(name, motion, qa=QA[name])
            self.assertLess(float(np.median(err)), 1e-3, name)                          # sub-millimetre trajectories
            if mot.max() > 0: self.assertLess(float(np.median(err[-1])), .01*float(np.median(mot)), name)
            self.assertTrue((res['trajectory_status'].numpy() != 3).all(), name)        # no numerically failed trajectory

    def test_G3_2_geometry_error_and_motion_together(self):
        for name in ('constant_image_axis', 'acceleration_mixed_3d'):
            self._g32(name)

    def _g32(self, name):
        motion = MOTIONS[name]
        sd = np.sqrt(np.diag(catalogue(AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], TIMES[:2]), 6, [(40, 30)])['P_x0'][0]))
        delta = .8*sd*np.array([1., -1., 1.])/np.sqrt(3)
        ds, cat, res, err, mot, gt = self.check_motion('delta+'+name, motion, delta, qa=QA[name])
        ok = res['trajectory_status'].numpy() != 4
        b = res['b'].numpy()
        self.assertLess(float(np.median(np.linalg.norm(b[ok]+delta, axis=1)/np.linalg.norm(delta))), .6)
        # independent dense NumPy batch solver on the production support
        for i in (0, 1):
            support = [[c for c in range(len(ds.cameras)) if res['camera_reason'][k, c, i] == 0] for k in range(K)]
            So, Co = dense_oracle_map(ds, cat, i, support, res['warm_start_smooth_mean'].numpy()[:, i], qa=QA[name])
            self.assertLess(same_map(res['smooth_mean'].numpy()[:, i], So, Co), 1e-6)
        # the ORIGINAL forward IEKF on the same data, prior and R pins frame 0 to the perturbed catalogue
        old = run_original_filter(ds, cat, qa=QA[name])
        err_old = np.linalg.norm(old[:, ok, :3]-gt[:, ok], axis=-1)
        self.assertLess(float(np.median(err)), float(np.median(err_old)))
        self.assertLess(float(np.median(err[0])), .5*float(np.median(err_old[0])))     # the frame-0 geometry is corrected, not pinned

    def test_G3_3_occlusion_then_reappearance(self):
        front_motion = lambda t: np.array([-.45+.9*t/TIMES[-1]*1., 0., 0.])           # slides across in front of the background
        T = np.arange(40)/30.
        front_motion = lambda t, T=T: np.array([-.4+.8*t/T[-1], 0., 0.])
        plates = [AffinePlate(1.8, (-.9, .9), (-.7, .7)), AffinePlate(1.2, (-.08, .08), (-.08, .08), c0=(.2, .7, .3), Gx=(-1, .5, .3), Gy=(.4, -.6, .9), motion=front_motion)]
        ds = AnalyticDataset(fronto_cameras(RING), plates, T)
        cam = ds.cameras[6]; fx, fy, cx, cy = cam.intrinsics
        rgb, depth, mask = ds.render(0, 6)
        front = [(u, v) for v in range(72) for u in range(96) if mask[v, u] and abs(depth[v, u]-1.2) < 1e-6]
        g = np.random.default_rng(2)
        back = []                                                                        # background pixels on the occluder's path (y ~ 0)
        while len(back) < 30:
            u, v = int(g.integers(10, 86)), int(g.integers(31, 41))
            if abs(depth[v, u]-1.8) < 1e-6 and (u, v) not in back: back.append((u, v))
        cat = catalogue(ds, 6, front+back)
        nb = len(back); bi = np.arange(len(front), len(front)+nb)
        truth = cat['x_ref'][bi].copy()
        sd = np.sqrt(np.diag(cat['P_x0'][bi[0]])); delta = .8*sd*np.array([1., 1., -1.])/np.sqrt(3)
        cat['x_ref'][bi] += delta
        res = run_solver(ds, cat)
        reason = res['camera_reason'].numpy()
        occluded_rows = visible_rows = 0
        for j, i in enumerate(bi):
            for k in range(len(T)):
                for c, camc in enumerate(ds.cameras):
                    o = -camc.world_to_camera[:3, 3]; p = truth[j]
                    lam = (1.2-o[2])/(p[2]-o[2]); q = o+lam*(p-o)-front_motion(T[k])      # ray hits the front plate's plane
                    occ = abs(q[0]) <= .08 and abs(q[1]) <= .08
                    if occ:
                        occluded_rows += 1
                        self.assertNotEqual(reason[k, c, i], 0, (j, k, c))                # zero rows from the occluding camera
                    elif reason[k, c, i] == 0: visible_rows += 1
        self.assertGreater(occluded_rows, 50); self.assertGreater(visible_rows, 50)
        # identity kept and earlier states corrected by later evidence (filtered vs smoothed error of the background)
        X = res['smooth_mean'].numpy()[:, bi, :3]; Xf = res['filter_mean'].numpy()[:, bi, :3]
        es, ef = np.linalg.norm(X-truth[None], axis=-1), np.linalg.norm(Xf-truth[None], axis=-1)
        self.assertLess(float(np.median(es)), 1e-3)
        self.assertLess(float(np.median(es[:10])), float(np.median(ef[:10])))

    def test_G3_4_large_motion_never_accepts_a_worse_objective(self):
        fine = lambda s, t: np.stack((.5+.4*np.sin(2*np.pi*s/.02), .5+.4*np.sin(2*np.pi*(s+t)/.025), .5+.4*np.cos(2*np.pi*t/.018)), -1)
        speed = np.array([1.2, .4, 0.])                                                  # ~3.2 px / frame at 1.5 m (beyond one cell)
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.9, .9), (-.6, .6), texture=fine, motion=lambda t: speed*t)], np.arange(12)/30.)
        cat = catalogue(ds, 6, interior_pixels(30, seed=8, box=(30, 66, 22, 50)))
        res = run_solver(ds, cat)
        for h in res['history']:
            if h.get('accepted'):
                self.assertLess(h['objective_after_step_same_support'], h['objective_total_on_this_support'])
                self.assertGreater(h['accepted_min_actual_decrease'], 0)
        gt = true_traj(lambda t: speed*t, cat['x_ref'], ds.times)
        err = np.linalg.norm(res['smooth_mean'].numpy()[..., :3]-gt, axis=-1)
        self.recovered = float((err[-1] < 1e-3).mean())                                   # reported (basin of attraction), not asserted
        print(f'\nG3.4: tracked within 1 mm at the last frame: {self.recovered:.2f}; frame states {np.bincount(res["frame_state"].numpy().ravel(), minlength=5).tolist()}; '
              f'trajectory status {np.bincount(res["trajectory_status"].numpy(), minlength=6).tolist()}')


if __name__ == '__main__':
    unittest.main()
