"""Gate 5 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: structured covariance identification
(estimator/temporal_model_identification.py, disabled in the pipeline). Linear-Gaussian tracks generated under known P0,
Q_a = q I, R = sr2 I with irregular intervals; the estimator never sees the generating parameters or any speed.

G5.1 EM recovers (q, sv2, sr2) within its own observed-information intervals; log-likelihood non-decreasing.
G5.2 the M-step expectation E[e e^T] equals Monte Carlo sampling of the exact dense joint posterior; dropping the lag-one
     terms does not.
G5.3 q from finite differences of noisy positions is inflated; the smoother expectation is not. P_v0 is a population
     parameter estimated from the tracks' own posterior, never from a GT speed percentile.
G5.4 held-out tracks: normalized innovation squared per dof consistent with 1, no signed innovation trend.
G5.5 identifiability counterexample (one short track): the estimator reports NOT_IDENTIFIED instead of a confident value."""
import math
import unittest
import numpy as np
import torch
from scipy.stats import chi2
from estimator import temporal_model_identification as TI
from estimator.rts_point_smoother import kalman_filter_linear, rts_smooth
from tests.test_temporal_phase1_math import dense_oracle

TRUE = (.05, .09, 4e-6)                       # q [m^2/s^3], sv2 [m^2/s^2], sr2 [m^2]


def simulate(n, K, seed, theta=TRUE, missing=.1, times=None):
    g = np.random.default_rng(seed)
    times = np.cumsum(np.r_[0., g.uniform(.02, .05, K-1)]) if times is None else times
    F, Q, _ = TI._system(torch.as_tensor(times), theta)
    F, Q = F.numpy(), Q.numpy()
    P0x = np.stack([np.diag(g.uniform(1e-6, 4e-6, 3)) for _ in range(n)])
    x0 = g.normal(0, .3, (n, 3))
    s = np.zeros((K, n, 6))
    s[0, :, :3] = x0+np.stack([g.multivariate_normal(np.zeros(3), P) for P in P0x]); s[0, :, 3:] = g.normal(0, math.sqrt(theta[1]), (n, 3))
    for k in range(K-1):
        s[k+1] = s[k]@F[k].T+g.multivariate_normal(np.zeros(6), Q[k], n)
    H = np.zeros((K, n, 3, 6)); H[..., :3] = np.eye(3)
    y = s[..., :3]+g.normal(0, math.sqrt(theta[2]), (K, n, 3))
    present = g.random((K, n)) > missing
    present[0] = True
    t = lambda a: torch.as_tensor(a)
    return {'times': t(times), 'm0x': t(x0), 'P0x': t(P0x), 'H': t(H), 'y': t(y), 'present': t(present)}, s


class IdentificationTest(unittest.TestCase):
    def test_G5_1_em_recovers_parameters(self):
        tracks, _ = simulate(300, 25, 0)
        out = TI.em(tracks, (1., 1., 1e-4), iters=400)
        ll = np.array(out['loglik_history'])
        self.assertTrue((np.diff(ll) > -1e-7*np.abs(ll[:-1])).all())                     # EM never decreases the likelihood
        th = tuple(out['theta'][n] for n in TI.NAMES)
        idf = TI.identifiability(tracks, th)
        for name, est, true, se in zip(TI.NAMES, th, TRUE, idf['se_log']):
            self.assertEqual(idf['status'][name], 'IDENTIFIED', name)
            self.assertLess(abs(math.log(est/true)), 3*se, (name, est, true, se))

    def test_G5_2_m_step_expectation_vs_posterior_sampling(self):
        tracks, _ = simulate(1, 5, 3, missing=0.)
        th = TRUE
        sm, F, Qu, _ = TI.e_step(tracks, th)
        E = TI.transition_second_moments(sm['smooth_mean'], sm['smooth_cov'], sm['lag1'], F)[:, 0].numpy()
        E_nolag = TI.transition_second_moments(sm['smooth_mean'], sm['smooth_cov'], torch.zeros_like(sm['lag1']), F)[:, 0].numpy()
        # exact dense joint posterior of the 5-state trajectory (independent oracle), Monte Carlo of e e^T
        Fn, Q, _ = TI._system(tracks['times'], th)
        m0, P0 = TI._prior(tracks, th)
        R = th[2]*np.eye(3)[None].repeat(5, 0)
        mo, So = dense_oracle(m0[0].numpy(), P0[0].numpy(), Fn.numpy(), Q.numpy(), tracks['H'][:, 0].numpy(), R, tracks['y'][:, 0].numpy(), tracks['present'][:, 0].numpy())
        S = np.random.default_rng(0).multivariate_normal(mo, So, 400_000).reshape(-1, 5, 6)
        for k in range(4):
            e = S[:, k+1]-S[:, k]@Fn[k].numpy().T
            mc = (e[:, :, None]*e[:, None, :]).mean(0)
            se = (e[:, :, None]*e[:, None, :]).std(0)/math.sqrt(len(e))
            self.assertTrue((np.abs(mc-E[k]) < 5*se+1e-15).all(), k)
            self.assertTrue((np.abs(mc-E_nolag[k]) > 5*se).any(), k)                    # without lag-one terms: wrong

    def test_G5_3_finite_difference_q_is_inflated(self):
        times = np.arange(25)*.033
        tracks, s = simulate(300, 25, 1, missing=0., times=times)
        y = tracks['y'].numpy()
        naive = np.mean([TI.finite_difference_q(y[:, i], times) for i in range(y.shape[1])])
        exact = np.mean([TI.finite_difference_q(s[:, i, :3], times) for i in range(y.shape[1])])   # noiseless positions (oracle only)
        self.assertGreater(naive, 5*TRUE[0])                                             # measurement noise inflates the acceleration
        self.assertLess(abs(math.log(exact/TRUE[0])), .2)                                # the derivation itself is right
        out = TI.em(tracks, (1., 1., 1e-4), iters=400)
        self.assertLess(abs(math.log(out['theta']['q']/TRUE[0])), .3)                    # the smoother expectation is not inflated
        self.assertNotIn('speed', ' '.join(TI.em.__code__.co_varnames))                  # P_v0 from the tracks' posterior, no speed input

    def test_G5_4_heldout_innovations(self):
        train, _ = simulate(200, 25, 4)
        held, _ = simulate(200, 25, 5, times=train['times'].numpy())
        th = TI.em(train, (1., 1., 1e-4), iters=400)['theta']
        h = TI.heldout_nis(held, tuple(th[n] for n in TI.NAMES))
        n = h['scalar_measurements']
        lo, hi = chi2.ppf(.005, n)/n, chi2.ppf(.995, n)/n
        self.assertTrue(lo < h['nis_per_dof'] < hi, (h['nis_per_dof'], lo, hi))
        self.assertTrue((np.abs(h['signed_mean_innovation']) < 4*math.sqrt(TRUE[2]+1e-5)/math.sqrt(n/3)).all())

    def test_G5_5_refuses_unidentifiable_parameters(self):
        tracks, _ = simulate(1, 3, 6, missing=0.)
        out = TI.em(tracks, (1., 1., 1e-4), iters=200)
        idf = TI.identifiability(tracks, tuple(out['theta'][n] for n in TI.NAMES))
        self.assertEqual(idf['status']['q'], 'NOT_IDENTIFIED')
        self.assertEqual(idf['status']['sv2'], 'NOT_IDENTIFIED')


if __name__ == '__main__':
    unittest.main()
