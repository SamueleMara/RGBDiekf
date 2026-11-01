"""Gate 4 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: projective measurement-model diagnostics and bias
separation (oracles independent of the estimator; ground truth used by the TEST only).

G4.1 analytic bilinear h, H vs central finite differences inside a cell, and exactly on both sides of an integer pixel
     boundary (one-sided derivatives differ there; the solver flags the row as a border row).
G4.2 analytic tilted plane: bilinear METRIC depth is biased behind the surface, bilinear INVERSE depth is exact (float32
     storage precision); its analytic Jacobian and delta-method variance; no interpolation across a depth discontinuity.
G4.3 known static geometry error + shared per-camera depth offset: identifiable and recovered with the structured camera
     model; a free per-point-per-camera bias is a gauge (rank deficiency detected, no unique solution reported).
G4.4 one noisy source colour reused by every RGB row: the naive (fixed-colour) posterior is overconfident (NEES coverage
     fails); the latent canonical colour eliminated exactly restores calibration.
G4.5 the second-moment R (mean z^2 = 1 at GT): heterogeneous (tiny median), and a fit on one frame set is not held-out
     evidence on another."""
import unittest
import numpy as np
import torch
from scipy.stats import chi2
from tests.projective_fixtures import camera, POLICY, FieldDataset
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise, sample, CODE
from estimator.projective_terms import build_terms
from estimator.temporal_observability import camera_offset_system, joint_rank
from tests.temporal_fixtures import fronto_cameras, AffinePlate, AnalyticDataset, catalogue, run_solver, nees
from tests.test_temporal_phase1_static import RING, interior_pixels

W, H = 96, 72
NOISE = lambda: SensorNoise(torch.tensor([1e-4]*3, dtype=torch.float64), 1e-6, 0., 0., 0., 'sum_w2')


def tilted_plane(alpha=.002, beta=-.0015, gamma=.55, cam=None):
    """1/Z affine in pixels: the depth image of a single tilted plane seen by a pinhole camera; colour smooth."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    depth = 1/(gamma+alpha*(u-W/2)+beta*(v-H/2))
    rgb = np.stack((.5+.3*np.sin(u/7.)*np.cos(v/9.), .5+.25*np.sin((u+v)/11.), .45+.3*np.cos((u-.6*v)/6.)), -1)
    cam = cam or camera('a', np.eye(3), np.zeros(3), W=W, H=H)
    ds = FieldDataset([cam], [[(rgb, depth, np.ones((H, W), bool))]])
    return ds, CameraRig([cam], 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64), (alpha, beta, gamma)


def plane_point(rig, uv, abg):
    a, b, g = abg
    Z = 1/(g+a*(uv[0]-W/2)+b*(uv[1]-H/2))
    return rig.backproject(torch.zeros(1, dtype=torch.long), torch.tensor([uv], dtype=torch.float64), torch.tensor([Z], dtype=torch.float64))[0], Z


class BilinearJacobianTest(unittest.TestCase):
    def test_G4_1_finite_differences_inside_and_on_both_sides_of_a_boundary(self):
        ds, rig, frame, abg = tilted_plane()
        for mode in ('metric', 'inverse'):
            pol = {**POLICY, 'row_innovation_gate': 'none', 'depth_interpolation': mode}
            t = lambda x: build_terms(rig, frame, torch.zeros(len(x), dtype=torch.long), torch.arange(len(x)), x, torch.full((len(x), 3), .5, dtype=torch.float64),
                                      torch.eye(3, dtype=torch.float64).expand(len(x), 3, 3)*1e-6, NOISE(), pol)
            for uv in ((30.37, 22.61), (51.73, 40.18)):                                  # interior of a cell
                x, _ = plane_point(rig, uv, abg); x = x[None]
                tt = t(x); eps = 1e-7
                fd = torch.stack([(-t(x+eps*e).r+t(x-eps*e).r)/(2*eps) for e in torch.eye(3, dtype=torch.float64)[:, None]], -1)[0]
                np.testing.assert_allclose(fd.numpy(), tt.Hx[0].numpy(), rtol=1e-5, atol=1e-6)
            # exactly on the integer column u = 40: floor picks the right cell; the right one-sided FD matches it, the left
            # one-sided FD matches the left cell's interpolant, and they differ (not differentiable)
            x, _ = plane_point(rig, (40.0, 30.3), abg); x = x[None]
            ex = torch.tensor([[1., 0., 0.]], dtype=torch.float64); eps = 1e-6
            right = (-t(x+eps*ex).r+t(x).r)/eps; left = (-t(x).r+t(x-eps*ex).r)/eps
            np.testing.assert_allclose(right[0].numpy(), t(x).Hx[0, :, 0].numpy(), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(left[0].numpy(), t(x-1e-9*ex).Hx[0, :, 0].numpy(), rtol=1e-4, atol=1e-6)
            self.assertGreater(float((left-right).abs().max()), 1e-3)
            from estimator.ieks_point_trajectory import TemporalModel, evaluate_frame
            import json; from pathlib import Path
            tcfg = {**json.loads((Path(__file__).resolve().parents[1]/'configs/temporal_phase1.json').read_text()), 'depth_interpolation': mode}
            M = TemporalModel(rig, NOISE(), POLICY, tcfg, 'cpu')
            a0 = sample(frame, torch.zeros(1, dtype=torch.long), torch.tensor([[40.0, 30.3]], dtype=torch.float64), NOISE(), pol, rig.K).rgb   # its true colour
            ev = evaluate_frame(M, frame, x, a0, torch.eye(3, dtype=torch.float64)[None]*1e-2)
            self.assertEqual(int(ev['nrows'][0]), 1)
            self.assertTrue(bool(ev['border'][0]))                                         # the solver knows this row is one-sided


class InverseDepthTest(unittest.TestCase):
    def test_G4_2_inverse_depth_exact_metric_depth_biased(self):
        ds, rig, frame, abg = tilted_plane()
        g = np.random.default_rng(0)
        uv = torch.as_tensor(np.c_[g.uniform(5, W-6, 200), g.uniform(5, H-6, 200)])
        cam = torch.zeros(200, dtype=torch.long)
        a, b, c = abg
        Z = 1/(c+a*(uv[:, 0]-W/2)+b*(uv[:, 1]-H/2))
        sm = sample(frame, cam, uv, NOISE(), {**POLICY, 'depth_interpolation': 'metric'}, rig.K)
        si = sample(frame, cam, uv, NOISE(), {**POLICY, 'depth_interpolation': 'inverse'}, rig.K)
        self.assertTrue(bool((sm.reason == 0).all()) and bool((si.reason == 0).all()))
        err_m, err_i = (sm.depth-Z).numpy(), (si.depth-Z).numpy()
        self.assertTrue((err_m > -1e-6).all())                                        # metric bilinear depth lies behind the plane
        self.assertGreater(float(np.abs(err_m).max()), 1e-5)
        self.assertLess(float(np.abs(err_i/Z.numpy()).max()), 2e-7)                    # inverse depth exact to float32 storage
        # analytic Jacobian of the inverse-depth interpolant (inside cells)
        eps = 1e-6
        for d in range(2):
            e = torch.zeros(2, dtype=torch.float64); e[d] = eps
            fd = (sample(frame, cam, uv+e, NOISE(), {**POLICY, 'depth_interpolation': 'inverse'}, rig.K).depth-
                  sample(frame, cam, uv-e, NOISE(), {**POLICY, 'depth_interpolation': 'inverse'}, rig.K).depth)/(2*eps)
            np.testing.assert_allclose(fd.numpy(), si.grad_depth[:, d].numpy(), rtol=1e-4, atol=1e-7)
        # transformed noise: Monte Carlo of the interpolated inverse depth vs the delta-method variance
        sig = 2e-4; noise = SensorNoise(torch.tensor([1e-4]*3, dtype=torch.float64), sig**2, 0., 0., 0., 'sum_w2')
        vals = []
        for r in range(400):
            dn = np.asarray(ds.frames[0][0][1])+np.random.default_rng(r).normal(0, sig, (H, W))
            dsn = FieldDataset(ds.cameras, [[(ds.frames[0][0][0], dn, np.ones((H, W), bool))]])
            vals.append(sample(MeasuredFrame(dsn, 0, 'cpu', torch.float64), cam[:20], uv[:20], noise, {**POLICY, 'depth_interpolation': 'inverse'}, rig.K).depth.numpy())
        var_mc = np.var(np.array(vals), 0)
        var_model = sample(frame, cam[:20], uv[:20], noise, {**POLICY, 'depth_interpolation': 'inverse'}, rig.K).var_depth.numpy()
        np.testing.assert_allclose(var_mc/var_model, 1., atol=.2)
        # no interpolation across a depth discontinuity (same rule as the metric sampler)
        d2 = np.asarray(ds.frames[0][0][1]).copy(); d2[:, 48:] += .5
        ds2 = FieldDataset(ds.cameras, [[(ds.frames[0][0][0], d2, np.ones((H, W), bool))]])
        s2 = sample(MeasuredFrame(ds2, 0, 'cpu', torch.float64), cam[:1], torch.tensor([[47.5, 30.3]], dtype=torch.float64), NOISE(),
                    {**POLICY, 'depth_interpolation': 'inverse'}, rig.K)
        self.assertEqual(int(s2.reason[0]), CODE['DEPTH_DISCONTINUITY'])


class StructuredBiasTest(unittest.TestCase):
    def test_G4_3_geometry_plus_camera_depth_offset_identifiability(self):
        # Linearized depth rows of an UNTEXTURED fronto-parallel plane (depth is the only geometry information): each point i
        # seen by camera c gives h = (D - Z) + beta_c, H = [-e3 (geometry z_i) | +1 (offset)]. Lateral geometry is not observed
        # by depth on a plane (G2.4) and is not an unknown here.
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6), Gx=(0, 0, 0), Gy=(0, 0, 0))], np.arange(1)/30.)
        cat = catalogue(ds, 6, interior_pixels(25, seed=4))
        rig = CameraRig(ds.cameras, 'cpu'); frame = MeasuredFrame(ds, 0, 'cpu', torch.float64)
        C, n = len(ds.cameras), len(cat['x_ref'])
        rows = []
        for i, x in enumerate(cat['x_ref']):
            t = build_terms(rig, frame, torch.arange(C), torch.full((C,), i), torch.as_tensor(x).expand(C, 3), torch.as_tensor(cat['rgb0'][i]).expand(C, 3),
                            torch.eye(3, dtype=torch.float64).expand(C, 3, 3)*1e-6, NOISE(), {**POLICY, 'row_innovation_gate': 'none'})
            rows += [(i, c, float(t.Hx[c, 3, 2])) for c in range(C) if int(t.reason[c]) == 0]
        ref = int(cat['src_cam'][0])                                                    # metric reference: the catalogue's own camera
        def system(free):
            cols_b = sorted({(i, c) for i, c, _ in rows}) if free else [c for c in range(C) if c != ref]
            J = np.zeros((len(rows), n+len(cols_b)))
            for r, (i, c, hz) in enumerate(rows):
                J[r, i] = hz
                key = (i, c) if free else c
                if key in cols_b: J[r, n+cols_b.index(key)] = 1.
            return J
        Js = system(False)
        rank_s, _ = joint_rank(Js)
        self.assertEqual(rank_s, Js.shape[1])                                           # geometry + structured offsets identified
        g = np.random.default_rng(1)
        theta = np.r_[g.normal(0, 2e-3, n), g.normal(0, 3e-3, C-1)]
        np.testing.assert_allclose(np.linalg.lstsq(Js, Js@theta, rcond=None)[0], theta, rtol=0, atol=1e-12)   # recovered
        Jn = np.zeros((len(rows), n+C))                                                 # without the reference camera: one global gauge
        for r, (i, c, hz) in enumerate(rows): Jn[r, i] = hz; Jn[r, n+c] = 1.
        self.assertEqual(joint_rank(Jn)[0], Jn.shape[1]-1)
        Jf = system(True)
        rank_f, _ = joint_rank(Jf)
        self.assertLessEqual(rank_f, Jf.shape[1]-n)                                     # free per-point-camera bias: every point's z is a gauge
        null = np.linalg.svd(Jf)[2][rank_f:]
        self.assertGreater(float(np.abs(null[:, :n]).max()), .1)                        # the gauge moves the geometry: no unique solution
        # the textured case: colour parallax identifies the geometry, so even free depth biases are determined (reported)
        dst = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(1)/30.)
        catt = catalogue(dst, 6, interior_pixels(25, seed=4)); frt = MeasuredFrame(dst, 0, 'cpu', torch.float64)
        Hrows, cams_ = [], []
        for i, x in enumerate(catt['x_ref']):
            t = build_terms(rig, frt, torch.arange(C), torch.full((C,), i), torch.as_tensor(x).expand(C, 3), torch.as_tensor(catt['rgb0'][i]).expand(C, 3),
                            torch.eye(3, dtype=torch.float64).expand(C, 3, 3)*1e-6, NOISE(), {**POLICY, 'row_innovation_gate': 'none'})
            for c in range(C):
                if int(t.reason[c]) == 0: Hrows.append((i, t.Hx[c].numpy())); cams_.append(c)
        Jd, _ = camera_offset_system([(i, H[3]) for i, H in Hrows], cams_, C, True)
        Jr = np.zeros((3*len(Hrows), Jd.shape[1]))
        for r, (i, H) in enumerate(Hrows): Jr[3*r:3*r+3, 3*i:3*i+3] = H[:3]
        Jt = np.vstack([Jd/1e-3, Jr/1e-2])
        self.assertEqual(joint_rank(Jt)[0], Jt.shape[1])


class LatentColourTest(unittest.TestCase):
    def test_G4_4_reused_source_colour_naive_overconfident_latent_calibrated(self):
        # controlled generator where the linearized likelihood holds (texture gradient 30 per pixel against pixel noise 0.01):
        # measured with the same generator, latent-colour mean NEES is 27.9 / 7.0 / 3.8 at gradient 1 / 3 / 10 per pixel
        # (noisy per-camera image gradients break the colour-position gauge spuriously): an OPEN limitation, reported.
        sig = (.01, 1e-3); K = 10; GX = 3000.
        plate = lambda: [AffinePlate(1.5, (-.8, .8), (-.6, .6), c0=(.5, .5, .5), Gx=(GX*.8, -GX*.3, GX*.2), Gy=(GX*.1, GX*.6, -GX*.5))]
        pix = interior_pixels(20, seed=9)
        clean = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(K)/30.)
        truth = catalogue(clean, 6, pix)['x_ref']
        out = {False: [], True: []}; total = excluded = 0
        for run in range(8):
            ds = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(K)/30., noise=sig, seed=run)
            cat = catalogue(ds, 6, pix, depth_var=sig[1]**2)                            # source colour AND depth from the noisy frame 0
            for latent in (False, True):
                res = run_solver(ds, cat, noise_sig=sig, latent_colour=latent)
                ok = res['trajectory_status'].numpy() == 0                               # coverage is claimed for CONVERGED trajectories only
                total += len(ok); excluded += int((~ok).sum())
                b, bc = res['b'].numpy(), res['b_cov'].numpy()
                out[latent] += [nees(cat['x_ref'][i]+b[i]-truth[i], bc[i]) for i in np.flatnonzero(ok)]
        self.assertLess(excluded/total, .1)                                              # non-converged (e.g. support oscillation) stay few
        n = len(out[True]); lo, hi = chi2.ppf(.005, 3*n)/n, chi2.ppf(.995, 3*n)/n
        self.assertGreater(np.mean(out[False]), hi)                                    # naive: reused colour counted as independent
        self.assertTrue(lo < np.mean(out[True]) < hi, (np.mean(out[True]), lo, hi))     # latent colour: calibrated
        self.naive, self.latent = np.mean(out[False]), np.mean(out[True])
        print(f'\nG4.4 mean NEES naive {self.naive:.2f}, latent {self.latent:.2f}, interval [{lo:.2f}, {hi:.2f}]')


class SecondMomentRTest(unittest.TestCase):
    def test_G4_5_second_moment_R_is_not_held_out_calibration(self):
        from tools.audit_static_equilibrium import _mean_match
        g = np.random.default_rng(0)
        v = np.full(20000, 1e-10)                                                       # declared sensor variance (noiseless-render floor)
        # heterogeneous population: most rows tiny, 3% edge-like rows with large errors (training frames)
        r_train = np.where(g.random(20000) < .03, g.normal(0, 3e-3, 20000), g.normal(0, 2e-5, 20000))
        m = _mean_match(r_train**2, v)
        z = r_train**2/(v+m)
        self.assertAlmostEqual(float(z.mean()), 1., places=6)                           # fitted: mean standardized square 1 ...
        self.assertLess(float(np.median(z)), .01)                                        # ... median tiny: heterogeneity, not calibration
        r_held = np.where(g.random(20000) < .10, g.normal(0, 3e-3, 20000), g.normal(0, 2e-5, 20000))   # other frames: more edge rows
        self.assertGreater(float((r_held**2/(v+m)).mean()), 2.)                           # fit-on-training is not held-out evidence
        self.assertGreater(float((r_held**2/(v+m) > chi2.ppf(.99, 1)).mean()), .05)        # tails far above the nominal 1 %


if __name__ == '__main__':
    unittest.main()
