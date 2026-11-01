"""Gate 2 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: static equilibrium and fixed-geometry recovery with the
production temporal solver on analytic RGB-D (tests/temporal_fixtures.py; ground truth used by the TEST only).

G2.1 noiseless static, fully visible, exact frame-0 points: zero first-step information gradient, trajectory stays at the
     truth with zero velocity for all 60 frames.
G2.2 noiseless static with a constant delta injected into the frame-0 catalogue only (images unchanged): b recovers -delta
     up to the prior shrinkage of the posterior (NEES of b + delta, recovery ratio), d and v consistent with zero; equals an
     independent dense NumPy MAP. Sweep of perturbation direction, source camera and point incidence.
G2.3 noisy static with known covariance, independent noise realizations, catalogue drawn from its own prior (correct
     generative assumptions): unbiased b, NEES coverage of b and of d_t against the exact Cov(d_t), residual whiteness.
G2.4 untextured plane: the tangent directions are reported unidentifiable; the posterior stays prior-dominated there.
G2.5 frame-0 information comes only from independent (non-source) cameras; more cameras -> more identified dimensions;
     the source depth is never counted twice (G1.5 is the companion: a duplicated source row raises)."""
import unittest
import numpy as np
from scipy.stats import chi2
from tests.temporal_fixtures import fronto_cameras, AffinePlate, AnalyticDataset, catalogue, run_solver, dense_oracle_map, nees, same_map

RING = [(x, y, 0) for x in (-.2, 0, .2) for y in (-.15, .15)]+[(0, 0, 0), (.1, .05, 0)]


def interior_pixels(n, seed=0, box=(25, 70, 20, 52)):
    g = np.random.default_rng(seed)
    return [(int(u), int(v)) for u, v in zip(g.integers(box[0], box[1], n), g.integers(box[2], box[3], n))]


class StaticEquilibriumTest(unittest.TestCase):
    def test_G2_1_noiseless_static_stays_exactly(self):
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(60)/30.)
        cat = catalogue(ds, 6, interior_pixels(30))
        for production in (False, True):                                                # Stage A and the shipped defaults
            res = run_solver(ds, cat, production=production)
            # first-step information gradient: the forward frame-1 update takes no step (Newton decrement below tolerance)
            self.assertEqual(res['forward_stats'][1]['accepted_steps'], 0)
            self.assertEqual(res['forward_stats'][1]['converged'], 30)
            S = res['smooth_mean'].numpy()
            self.assertLess(float(np.abs(S[..., :3]-cat['x_ref'][None]).max()), 1e-9)   # within float precision of the truth
            self.assertLess(float(np.abs(S[..., 3:]).max()), 1e-9)
            self.assertTrue((res['trajectory_status'].numpy() == 0).all())
            self.assertLess(float(res['objective_final'].max()), 1e-6)                  # residuals are float32 rounding only

    def run_delta(self, delta, src, pix, K=20, cams=RING):
        ds = AnalyticDataset(fronto_cameras(cams), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(K)/30.)
        cat = catalogue(ds, src, pix); truth = cat['x_ref'].copy(); cat['x_ref'] = truth+delta
        return ds, cat, truth, run_solver(ds, cat)

    def test_G2_2_injected_geometry_error_is_a_constant_correction(self):
        sd = None
        for src, box in ((6, (25, 70, 20, 52)), (0, (10, 40, 10, 30))):            # central / oblique source camera and incidence
            for direction in np.eye(3).tolist()+[[1, -1, 1]]:
                pix = interior_pixels(12, seed=src, box=box)
                ds, cat, truth, res = self.run_delta(np.zeros(3), src, pix, K=4)
                if sd is None: sd = np.sqrt(np.diag(cat['P_x0'][0]))
                delta = .8*sd*np.asarray(direction, float)/np.linalg.norm(direction)   # within the prior-supported size
                ds, cat, truth, res = self.run_delta(delta, src, pix)
                ok = res['trajectory_status'].numpy() != 4
                b, bc = res['b'].numpy(), res['b_cov'].numpy()
                for i in np.flatnonzero(ok):
                    self.assertLess(nees(b[i]+delta, bc[i]), chi2.ppf(.999, 3), (src, direction, i))
                    self.assertLess(np.linalg.norm(b[i]+delta), np.linalg.norm(delta), (src, direction, i))      # moved toward -delta
                    d, dc = res['displacement'].numpy()[:, i], res['displacement_cov'].numpy()[:, i]
                    for t in (1, 10, 19): self.assertLess(nees(d[t], dc[t].astype(float)), chi2.ppf(.999, 3))
                    v = res['velocity'].numpy()[:, i]; vs = np.sqrt(np.diagonal(res['smooth_cov'].numpy()[:, i, 3:, 3:], axis1=1, axis2=2))
                    self.assertTrue((np.abs(v) < 4*vs).all())
                ratio = np.median(np.linalg.norm(b[ok]+delta, axis=1)/np.linalg.norm(delta))
                self.assertLess(ratio, .6, (src, direction, ratio))                    # most of delta recovered; the rest is prior shrinkage
        # the production MAP equals an independent dense NumPy batch Gauss-Newton on the same support
        support = [[c for c in range(len(ds.cameras)) if res['camera_reason'][k, c, 0] == 0] for k in range(len(ds.times))]
        So, Co = dense_oracle_map(ds, cat, 0, support, res['warm_start_smooth_mean'].numpy()[:, 0])
        self.assertLess(same_map(res['smooth_mean'].numpy()[:, 0], So, Co), 1e-6)
        np.testing.assert_allclose(res['b_cov'].numpy()[0], Co[:3, :3], rtol=1e-3, atol=1e-12)

    def test_G2_3_noisy_static_monte_carlo_coverage(self):
        # Correct generative assumptions: independent Gaussian pixel noise with the declared R, catalogue drawn from its own
        # prior, noiseless canonical colour, and a texture whose image gradient per pixel (3.0) is large against the pixel
        # noise (0.01) so the linearized Gaussian likelihood is valid. At gradient / noise ~ 1 the bilinear Jacobian is built
        # from noisy pixels (errors-in-variables) and the posterior is overconfident (measured mean NEES 73 at ratio 1, 5.7 at
        # 10, 4.5 at 30, 3.4 at 100, 3.0 at 300 with depth noise 1 mm): an OPEN limitation (reports/temporal_phase1_ieks.md), not hidden here.
        sig = (.01, 1e-3); K = 12; pix = interior_pixels(25, seed=3); GX = 300.
        plate = lambda: [AffinePlate(1.5, (-.8, .8), (-.6, .6), c0=(.5, .5, .5), Gx=(GX*.8, -GX*.3, GX*.2), Gy=(GX*.1, GX*.6, -GX*.5))]
        clean = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(K)/30.)
        base = catalogue(clean, 6, pix, depth_var=sig[1]**2); truth = base['x_ref']
        nb, nd, errs, lag1, dmc, dcov = [], [], [], [], [], []
        for run in range(12):
            ds = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(K)/30., noise=sig, seed=run)
            g = np.random.default_rng(100+run)
            cat = dict(base)
            cat['x_ref'] = truth+np.stack([g.multivariate_normal(np.zeros(3), P) for P in base['P_x0']])   # catalogue drawn from its prior
            res = run_solver(ds, cat, noise_sig=sig)
            ok = res['trajectory_status'].numpy() == 0                                   # coverage is claimed for CONVERGED trajectories only
            self.assertLess(float((~ok).mean()), .1)
            b, bc = res['b'].numpy(), res['b_cov'].numpy()
            for i in np.flatnonzero(ok):
                e = cat['x_ref'][i]+b[i]-truth[i]; nb.append(nees(e, bc[i])); errs.append(e)
                d, dc = res['displacement'].numpy()[:, i], res['displacement_cov'].numpy()[:, i].astype(float)
                nd.append(nees(d[K-1], dc[K-1])); dmc.append(d[K-1]); dcov.append(dc[K-1])
            lag1.append(res['residual_audit']['lag1_temporal_autocorrelation_z_RGBD'])
            if run == 0:                                            # Cov(d_t) against the dense oracle's inverse Hessian blocks
                support = [[c for c in range(len(ds.cameras)) if res['camera_reason'][k, c, 0] == 0] for k in range(K)]
                So, Co = dense_oracle_map(ds, cat, 0, support, res['warm_start_smooth_mean'].numpy()[:, 0], noise_sig=sig)
                for t in (3, K-1):
                    Dt = Co[6*t:6*t+3, 6*t:6*t+3]+Co[:3, :3]-Co[6*t:6*t+3, :3]-Co[:3, 6*t:6*t+3]
                    np.testing.assert_allclose(res['displacement_cov'].numpy()[t, 0], Dt, rtol=2e-3, atol=1e-13)
        nb, nd, errs = np.array(nb), np.array(nd), np.array(errs)
        n = len(nb)
        lo, hi = chi2.ppf(.005, 3*n)/n, chi2.ppf(.995, 3*n)/n                # aggregate interval of the mean NEES (3 dof)
        self.assertTrue(lo < nb.mean() < hi, (nb.mean(), lo, hi))
        self.assertTrue(lo < nd.mean() < hi, (nd.mean(), lo, hi))
        se = errs.std(0)/np.sqrt(n)
        self.assertTrue((np.abs(errs.mean(0)) < 4*se).all(), (errs.mean(0), se))      # unbiased
        l1 = np.array([x for x in lag1 if None not in x], float)
        self.assertTrue((np.abs(l1.mean(0)) < .1).all(), l1.mean(0))                   # no persistent residual correlation
        # Monte Carlo covariance of d_T against the reported Cov(d_T) (trace ratio, 3n samples)
        ratio = np.trace(np.cov(np.array(dmc).T))/np.trace(np.mean(dcov, 0))
        self.assertTrue(.6 < ratio < 1.6, ratio)

    def test_G2_4_untextured_plane_tangent_unidentifiable(self):
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6), Gx=(0, 0, 0), Gy=(0, 0, 0))], np.arange(10)/30.)
        cat = catalogue(ds, 6, interior_pixels(15))
        res = run_solver(ds, cat)
        ok = res['trajectory_status'].numpy() != 4
        self.assertTrue((res['b_unidentified_directions'].numpy()[ok] == 2).all())      # two tangent directions, z identified
        eig = res['data_information_eig'].numpy()[:, ok]
        self.assertTrue((eig[..., :2] <= 1e-9*eig[..., 2:]).all())                     # data rank 1 (depth normal) in every frame
        post = res['b_cov'].numpy()[ok][:, :2, :2]; prior = cat['P_x0'][ok][:, :2, :2]
        self.assertTrue((np.diagonal(post, axis1=1, axis2=2) > .9*np.diagonal(prior, axis1=1, axis2=2)).all())   # no false certainty

    def test_G2_5_frame0_independent_cameras_and_single_counting(self):
        pix = interior_pixels(10)
        one = AnalyticDataset(fronto_cameras([(0, 0, 0)]), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(3)/30.)
        r1 = run_solver(one, catalogue(one, 0, pix))
        self.assertTrue((r1['frame0_independent_cameras'].numpy() == 0).all())         # the source camera is not a frame-0 measurement
        self.assertTrue((r1['frame0_information_eig'].numpy() == 0).all())
        many = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(3)/30.)
        rm = run_solver(many, catalogue(many, 6, pix))
        self.assertTrue((rm['frame0_independent_cameras'].numpy() >= 5).all())
        rank = (rm['frame0_information_eig'].numpy() > 1e-9*rm['frame0_information_eig'].numpy()[:, 2:]).sum(1)
        self.assertTrue((rank == 3).all())
        self.assertTrue((rm['camera_reason'][0][6] == 11).all())                       # FRAME0_SOURCE_DUPLICATE_EXCLUDED


class SingleCountingTest(unittest.TestCase):
    def test_G1_5_backprojection_prior_equals_one_raw_factor_and_duplicate_raises(self):
        import torch
        from estimator.ieks_point_trajectory import assert_single_counted, SourceDuplicateError
        # a raw pixel/depth factor (u, v, D) with covariance Sigma under a flat prior has posterior covariance
        # (J_h^T Sigma^-1 J_h)^-1 = J_bp Sigma J_bp^T: the back-projection prior IS that one factor
        cam = fronto_cameras([(0, 0, 0)])[0]; fx, fy, cx, cy = cam.intrinsics
        D, u, v = 1.7, 30., 22.; b = np.array([(u-cx)/fx, (v-cy)/fy, 1.]); x = D*b
        Jbp = np.array([[D/fx, 0, b[0]], [0, D/fy, b[1]], [0, 0, 1.]]); Sig = np.diag([1/12, 1/12, 1e-6])
        Jh = np.array([[fx/x[2], 0, -fx*x[0]/x[2]**2], [0, fy/x[2], -fy*x[1]/x[2]**2], [0, 0, 1.]])
        np.testing.assert_allclose(np.linalg.inv(Jh.T@np.linalg.inv(Sig)@Jh), Jbp@Sig@Jbp.T, rtol=1e-12)
        # counting the same factor again would halve it: the solver refuses such a support
        sup = torch.zeros((3, 2), dtype=torch.bool); sup[1, 0] = True
        assert_single_counted(sup, torch.tensor([0, 2]))
        sup[0, 0] = True
        self.assertRaises(SourceDuplicateError, assert_single_counted, sup, torch.tensor([0, 2]))


if __name__ == '__main__':
    unittest.main()
