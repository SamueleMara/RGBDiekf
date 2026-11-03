"""Gate 6 / TF.6 item 4: model evidence and point marginals on tiny exactly enumerable models (no GT in inference).

Toy declared model (1D per point, evaluation-only truth): y_it = x_i0 + m_i a_i b_t + eps, eps ~ N(0, r^2), t = 0..T;
x_i0 ~ N(xbar_i, p0^2); for every MOVING support a shared latent b with b_0 = 0, b_t = b_t-1 + w, w ~ N(0, q^2) (proper,
anchored to identity at frame 0); a_i = lever arm (1: translation; small: near-hinge point). Static supports: b = 0."""
import itertools
import math
import unittest
import numpy as np
from estimator.svgd_support_posterior import (linear_gaussian_log_evidence, laplace_log_evidence, binary_posterior, coherent_hypotheses,
                                              enumerate_marginals, edge_marginals, select_seeds, support_posterior)
from tests.svgd_fixtures import record


def design(n, T, blocks, labels, a):
    """theta = (x_10..x_n0, b^(1)_1..T, b^(2)_1..T, ...); returns A [(T+1) n, dim], prior cov builder of the b part."""
    movers = [b for b, l in zip(blocks, labels) if l]
    A = np.zeros(((T+1)*n, n+T*len(movers)))
    for t in range(T+1):
        for i in range(n): A[t*n+i, i] = 1
    for k, blk in enumerate(movers):
        for t in range(1, T+1):
            for i in blk: A[t*n+i, n+k*T+t-1] = a[i]
    return A, len(movers)


def evidence(y, xbar, p0, r, q, a, blocks, labels, T):
    n = len(xbar); A, nm = design(n, T, blocks, labels, a)
    Lw = np.tril(np.ones((T, T)))*q; Pb = Lw@Lw.T                         # random walk covariance of (b_1..b_T)
    Sigma = np.zeros((A.shape[1],)*2); Sigma[:n, :n] = np.eye(n)*p0**2
    for k in range(nm): Sigma[n+k*T:n+(k+1)*T, n+k*T:n+(k+1)*T] = Pb
    mu = np.r_[xbar, np.zeros(nm*T)]
    return linear_gaussian_log_evidence(y, A, mu, Sigma, np.eye(len(y))*r**2)


def simulate(rng, n, T, truth_blocks, truth_labels, a, r, q, p0=1e-3):
    xbar = rng.normal(0, .05, n); x0 = xbar+rng.normal(0, p0, n); y = np.zeros(((T+1), n))
    for blk, l in zip(truth_blocks, truth_labels):
        b = np.r_[0, np.cumsum(rng.normal(0, q, T))]*l
        for i in blk: y[:, i] = x0[i]+a[i]*b
    return (y+rng.normal(0, r, y.shape)).ravel(), xbar


class PosteriorTests(unittest.TestCase):
    def test_linear_gaussian_evidence_vs_quadrature(self):
        rng = np.random.default_rng(0); T, r, q, p0 = 2, 2e-3, 5e-3, 1e-3
        y, xbar = simulate(rng, 1, T, [[0]], [1], [1.], r, q, p0)
        closed = evidence(y, xbar, p0, r, q, [1.], [[0]], [1], T)
        z, w = np.polynomial.hermite_e.hermegauss(120); Lw = np.tril(np.ones((T, T)))*q
        Z = np.stack(np.meshgrid(z, z, z, indexing='ij'), -1).reshape(-1, 3); lw = np.log(np.einsum('i,j,k->ijk', w, w, w).ravel())
        pred = (xbar[0]+p0*Z[:, :1])+np.c_[np.zeros(len(Z)), Z[:, 1:]@Lw.T]
        acc = lw-.5*((y[None]-pred)**2).sum(1)/r**2
        quad = float(np.logaddexp.reduce(acc))-1.5*math.log(2*math.pi)-.5*len(y)*math.log(2*math.pi*r*r)
        self.assertLess(abs(quad-closed), 1e-8)
        record('E29', test='gauss_hermite_120^3', abs_error=abs(quad-closed), tolerance=1e-8)

    def test_laplace_error_nonlinear_rotation(self):
        """2D points rotated by a latent random-walk angle (T = 2): Laplace vs 2D quadrature; exact for linear-Gaussian."""
        rng = np.random.default_rng(1); x0 = rng.normal(0, .05, (6, 2)); r, q = 1e-3, .05
        th_true = np.cumsum(rng.normal(0, q, 2))
        R = lambda a: np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        y = np.stack([x0@R(a).T for a in th_true])+rng.normal(0, r, (2, 6, 2))
        Lw = np.tril(np.ones((2, 2)))*q; Pi = np.linalg.inv(Lw@Lw.T)
        def E(th):                                                         # -log p(y | th) p(th), all constants kept
            res = y-np.stack([x0@R(a).T for a in th])
            return .5*(res**2).sum()/r**2+.5*y.size*math.log(2*math.pi*r*r)+.5*th@Pi@th+.5*math.log(np.linalg.det(2*math.pi*Lw@Lw.T))
        th = th_true.copy(); h = 1e-5
        for _ in range(30):                                                # Newton with FD derivatives (oracle side)
            g = np.array([(E(th+h*e)-E(th-h*e))/(2*h) for e in np.eye(2)])
            H = np.array([[(E(th+h*ei+h*ej)-E(th+h*ei-h*ej)-E(th-h*ei+h*ej)+E(th-h*ei-h*ej))/(4*h*h) for ej in np.eye(2)] for ei in np.eye(2)])
            th = th-np.linalg.solve(H, g)
        lap = laplace_log_evidence(E(th), H)
        sd = np.sqrt(np.diag(np.linalg.inv(H))); g1 = np.linspace(-8, 8, 161)
        vals = [-E(th+np.array([u, v])*sd) for u in g1 for v in g1]
        quad = float(np.logaddexp.reduce(vals))+math.log((g1[1]-g1[0])**2*sd.prod())
        err = abs(lap-quad); self.assertLess(err, .05)
        A = np.random.default_rng(2).normal(size=(5, 3)); S = np.eye(3); Rn = np.eye(5)*.1; yy = A@np.ones(3)
        P = np.linalg.inv(A.T@np.linalg.inv(Rn)@A+np.linalg.inv(S)); m = P@A.T@np.linalg.inv(Rn)@yy
        El = .5*(yy-A@m)@np.linalg.inv(Rn)@(yy-A@m)+.5*m@m+.5*math.log(np.linalg.det(2*math.pi*Rn))+.5*math.log(np.linalg.det(2*math.pi*S))
        self.assertAlmostEqual(laplace_log_evidence(El, np.linalg.inv(P)), linear_gaussian_log_evidence(yy, A, np.zeros(3), S, Rn), places=10)
        record('E31', test='laplace_vs_quadrature_rotation', abs_error_nats=err, tolerance=.05, linear_case='exact to 1e-10')

    def test_moving_fits_jitter_but_loses_evidence(self):
        rng = np.random.default_rng(3); n, T, r, q = 5, 6, 2e-3, 5e-3
        y, xbar = simulate(rng, n, T, [list(range(n))], [0], np.ones(n), r, q)
        A1, _ = design(n, T, [list(range(n))], [1], np.ones(n)); A0, _ = design(n, T, [list(range(n))], [0], np.ones(n))
        rss = lambda A: float(((y-A@np.linalg.lstsq(A, y, rcond=None)[0])**2).sum())
        self.assertLess(rss(A1), rss(A0))                                                    # flexible model fits jitter better
        L1 = evidence(y, xbar, 1e-3, r, q, np.ones(n), [list(range(n))], [1], T); L0 = evidence(y, xbar, 1e-3, r, q, np.ones(n), [list(range(n))], [0], T)
        self.assertGreater(L0, L1)                                                           # but has lower integrated evidence
        record('E28', test='occam', rss_static=rss(A0), rss_moving=rss(A1), log_bayes_factor_moving=L1-L0)

    def test_no_observation_returns_prior(self):
        y = np.zeros(0); L = linear_gaussian_log_evidence(y, np.zeros((0, 2)), np.zeros(2), np.eye(2), np.zeros((0, 0)))
        p1, p0, lbf = binary_posterior(L, L, .3)
        self.assertAlmostEqual(p1, .3, places=15); self.assertAlmostEqual(p0+p1, 1., places=15); self.assertEqual(lbf, 0.)

    def test_point_marginals_exact_enumeration(self):
        """Coherent hypotheses (partitions x labels) enumerated exactly; 2 points move together, 1 static, plus a near-hinge
        point whose lever arm (displacement) is below the noise: it must stay uncertain, not be forced static."""
        rng = np.random.default_rng(4); n, T, r, q = 4, 8, 2e-3, 8e-3; a = np.array([1., 1., 1., .02])
        y, xbar = simulate(rng, n, T, [[0, 1, 3], [2]], [1, 0], a, r, q)
        hyps = coherent_hypotheses(n)
        logL = np.array([evidence(y, xbar, 1e-3, r, q, a, b, l, T) for b, l, _ in hyps]); logp = np.full(len(hyps), -math.log(len(hyps)))
        gamma, post, logZ = enumerate_marginals(np.array([m for _, _, m in hyps]), logL, logp)
        self.assertAlmostEqual(post.sum(), 1., places=12); self.assertTrue(((gamma >= 0) & (gamma <= 1)).all())
        naive = sum(math.exp(lL+lp-logZ)*m for (_, _, m), lL, lp in zip(hyps, logL, logp))
        np.testing.assert_allclose(gamma, naive, atol=1e-12)
        self.assertGreater(gamma[0], .99); self.assertGreater(gamma[1], .99); self.assertLess(gamma[2], .05)
        prior_gamma = np.array([m for _, _, m in hyps]).mean(0)[3]
        self.assertGreater(gamma[3], .05); self.assertLess(gamma[3], .95)                     # near-hinge: unresolved
        p = [2, 0, 3, 1]                                                                       # permutation invariance
        hp = coherent_hypotheses(n); yp = y.reshape(T+1, n)[:, p].ravel()
        lp2 = np.array([evidence(yp, xbar[p], 1e-3, r, q, a[p], b, l, T) for b, l, _ in hp])
        gp, _, _ = enumerate_marginals(np.array([m for _, _, m in hp]), lp2, np.full(len(hp), -math.log(len(hp))))
        np.testing.assert_allclose(gp, gamma[p], atol=1e-12)
        record('E27', test='exact_enumeration', gamma=gamma.tolist(), prior_gamma_near_hinge=prior_gamma, hypotheses=len(hyps))

    def test_partition_enumeration_coherent(self):
        hyps = coherent_hypotheses(4); w = np.random.default_rng(5).dirichlet(np.ones(len(hyps)))
        W = edge_marginals(hyps, w, 4)
        np.testing.assert_allclose(W, W.T); np.testing.assert_allclose(np.diag(W), 1)
        for i, j, l in itertools.permutations(range(4), 3): self.assertGreaterEqual(W[i, l]+1e-12, W[i, j]+W[j, l]-1)   # transitivity bound
        self.assertEqual(len(hyps), 1*2+7*4+6*8+1*16)                                          # sum_k S(4,k) 2^k: every partition x labelling

    def test_seed_risk_rule(self):
        g = np.array([.999, .99, .2, .97, np.nan, .5]); ids = np.array([10, 11, 12, 13, 14, 15])
        s, fdr = select_seeds(g, ids, .02, 'EXACT_TINY_MODEL_POSTERIOR')
        self.assertEqual(s.tolist(), [10, 11, 13]); self.assertLessEqual(fdr, .02)
        with self.assertRaisesRegex(ValueError, 'PROBABILITY_NOT_YET_ESTABLISHED'): select_seeds(g, ids, .02, 'UNCALIBRATED_SCORE')

    def test_production_blocked_without_priors(self):
        import json
        from estimator.svgd_contract import model_contract
        cfg = json.loads(open('configs/svgd_support_experimental.json').read())
        out = support_posterior(model_contract(cfg, 'experimental'))
        self.assertEqual(out['status'], 'BLOCKED_MODEL_UNSPECIFIED'); self.assertIsNone(out['gamma'])


if __name__ == '__main__':
    unittest.main()
