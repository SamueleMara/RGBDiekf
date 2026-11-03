"""Gate 5 / TF.6 item 6: operator D (distance histories, Jacobians, shared-reference covariance) vs FD, dense and Monte Carlo."""
import math
import unittest
import numpy as np
import torch
from estimator.svgd_rigidity import (knn_graph, edge_residuals, edge_jacobian, dense_edge_covariance, edge_covariance_structured,
                                     structured_quad_logdet, rigid_edge_loglik, differential_rigidity)
from tests.svgd_fixtures import record


def rot(axis, a):
    axis = np.asarray(axis, float)/np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3)+math.sin(a)*K+(1-math.cos(a))*K@K


class RigidityTests(unittest.TestCase):
    def test_edge_jacobian_finite_difference(self):
        rng = np.random.default_rng(0); X = torch.tensor(rng.normal(0, .05, (4, 6, 3))); edges = torch.tensor([[0, 1], [2, 5], [1, 4]])
        n = edge_jacobian(X, edges).numpy(); h = 1e-7; worst = 0.
        for e, (i, l) in enumerate(edges.tolist()):
            for t in range(1, 4):
                for p, s, tt in ((i, 1, t), (l, -1, t), (i, -1, 0), (l, 1, 0)):
                    for k in range(3):
                        Xp, Xm = X.clone(), X.clone(); Xp[tt, p, k] += h; Xm[tt, p, k] -= h
                        fd = (edge_residuals(Xp, edges)[0][e, t-1]-edge_residuals(Xm, edges)[0][e, t-1]).item()/(2*h)
                        want = s*n[e, tt, k]
                        worst = max(worst, abs(fd-want)); self.assertLess(abs(fd-want), 1e-7)
        record('E22', test='fd', max_abs=worst, tolerance=1e-7)

    def test_rigid_motion_preserves_and_cross_support_changes(self):
        rng = np.random.default_rng(1); x0 = rng.normal(0, .05, (20, 3)); T = 6
        X = np.zeros((T, 20, 3)); V = np.zeros_like(X); w, b = np.array([.3, -.2, .5]), np.array([.01, 0, .02])
        for t in range(T):
            R = rot(w, np.linalg.norm(w)*t*.1); X[t, :10] = x0[:10]@R.T+b*t*.1; X[t, 10:] = x0[10:]                   # 10 moving, 10 static
            V[t, :10] = np.cross(w, X[t, :10]-b*t*.1)+b; V[t, 10:] = 0
        Xt, Vt = torch.tensor(X), torch.tensor(V)
        within = torch.tensor([[0, 1], [2, 7], [3, 9]]); static = torch.tensor([[10, 11], [12, 19]]); cross = torch.tensor([[0, 10], [5, 15]])
        self.assertLess(float(edge_residuals(Xt, within)[0].abs().max()), 1e-14); self.assertLess(float(edge_residuals(Xt, static)[0].abs().max()), 1e-15)
        self.assertGreater(float(edge_residuals(Xt, cross)[0].abs().max()), 1e-3)
        self.assertLess(float(differential_rigidity(Xt, Vt, within).abs().max()), 1e-15)
        # a static point ON the rotation axis keeps its distance to every moving point: low strain is not membership
        x0a = np.r_[x0[:3], [[0, 0, 0]]]; Xa = np.stack([np.r_[x0a[:3]@rot([0, 0, 1], .2*t).T, [[0, 0, 0]]] for t in range(T)])
        self.assertLess(float(edge_residuals(torch.tensor(Xa), torch.tensor([[0, 3], [1, 3]]))[0].abs().max()), 1e-15)

    def test_covariance_monte_carlo_and_structured_exact(self):
        """Omega = J P_zeta J^T: (a) vs Monte Carlo of the nonlinear e with jointly Gaussian states, (b) the structured
        production form equals the dense first-order form whenever P_zeta has the assumed structure."""
        rng = np.random.default_rng(2); T = 4
        X = np.zeros((T+1, 2, 3)); X[:, 0] = [0, 0, 0]; X[:, 1] = [.05, .01, 0]; X[1:, 1] += rng.normal(0, .002, (T, 3))
        A = rng.normal(size=(6*(T+1), 6*(T+1))); Pz = A@A.T*2e-9+np.eye(6*(T+1))*1e-9           # full joint (cross-time, cross-point)
        Xt = torch.tensor(X); e = torch.tensor([[0, 1]])
        Om = dense_edge_covariance(Xt, e[0], torch.tensor(Pz)).numpy()
        z = rng.multivariate_normal(np.zeros(6*(T+1)), Pz, 200000).reshape(-1, T+1, 2, 3)
        d = np.linalg.norm(X[None, :, 0]+z[:, :, 0]-X[None, :, 1]-z[:, :, 1], axis=-1); ee = d[:, 1:]-d[:, :1]
        mc = np.cov(ee.T); rel = float(np.abs(mc-Om).max()/np.abs(Om).max())
        self.assertLess(rel, .03)                                                                  # MC sd ~ sqrt(2/2e5) ~ 0.3%, + 2nd order
        P_blocks = np.zeros((T+1, 2, 3, 3)); Pd = np.zeros_like(Pz)
        for t in range(T+1):
            for p in range(2):
                B = rng.normal(size=(3, 3)); P_blocks[t, p] = B@B.T*1e-8+np.eye(3)*1e-9; s = 6*t+3*p; Pd[s:s+3, s:s+3] = P_blocks[t, p]
        c, delta = edge_covariance_structured(Xt, e, torch.tensor(P_blocks))
        Ostruct = c[0].item()*np.ones((T, T))+np.diag(delta[0].numpy())
        np.testing.assert_allclose(Ostruct, dense_edge_covariance(Xt, e[0], torch.tensor(Pd)).numpy(), rtol=1e-12)
        self.assertGreater(c[0].item(), 0)                                                         # shared frame-0 reference -> cross-time covariance
        record('E23', test='monte_carlo', max_rel=rel, samples=200000, tolerance=.03)

    def test_structured_inverse_equals_dense(self):
        rng = np.random.default_rng(3); E, T = 5, 7
        e = torch.tensor(rng.normal(size=(E, T))); c = torch.tensor(rng.uniform(.1, 2, E)); delta = torch.tensor(rng.uniform(.5, 3, (E, T)))
        q, ld = structured_quad_logdet(e, c, delta); ll, _ = rigid_edge_loglik(e, c, delta)
        for k in range(E):
            O = c[k].item()*np.ones((T, T))+np.diag(delta[k].numpy())
            self.assertAlmostEqual(q[k].item(), float(e[k].numpy()@np.linalg.solve(O, e[k].numpy())), places=11)
            self.assertAlmostEqual(ld[k].item(), float(np.linalg.slogdet(O)[1]), places=11)
            from scipy.stats import multivariate_normal
            self.assertAlmostEqual(ll[k].item(), multivariate_normal(np.zeros(T), O).logpdf(e[k].numpy()), places=10)
        diag_only = (e*e/(delta+c[:, None])).sum(1)                                                # treating times as independent
        self.assertFalse(torch.allclose(diag_only, q))
        record('E24', test='sherman_morrison_vs_dense', tolerance=1e-11)

    def test_hinge_displacement_identity(self):
        for r, deg in ((.01, 1.), (.2, 1.), (.05, 30.)):
            a = math.radians(deg); p = np.array([r, 0, 0]); q = rot([0, 0, 1], a)@p
            self.assertAlmostEqual(np.linalg.norm(q-p), 2*r*abs(math.sin(a/2)), places=15)
        self.assertAlmostEqual(2*.01*math.sin(math.radians(1)/2)*1e3, .1745, places=4)              # PDF 9.5: 0.175 mm at 1 cm

    def test_knn_graph_exact(self):
        rng = np.random.default_rng(4); x = torch.tensor(rng.normal(size=(50, 3))); x[7] = x[3]
        e, info = knn_graph(x, 4, chunk=7)
        self.assertEqual(info['coincident_pairs_dropped'], 2); self.assertFalse(bool((e[:, 0] == e[:, 1]).any()))
        D = torch.tensor(np.linalg.norm(x.numpy()[:, None]-x.numpy()[None], axis=-1)); D.fill_diagonal_(float('inf'))
        want = set()
        for i in range(50):
            for j in torch.topk(D[i], 4, largest=False).indices.tolist():
                if D[i, j] > 0: want.add(tuple(sorted((i, j))))
        self.assertEqual(set(map(tuple, e.tolist())), want)


if __name__ == '__main__':
    unittest.main()
