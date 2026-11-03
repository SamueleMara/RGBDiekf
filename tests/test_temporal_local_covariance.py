"""Local-correlated covariance bookkeeping against independent NumPy witnesses (spec tests 1, 4, 9, 10)."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_temporal_warp import rig
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_local_covariance import (STATE, CHART, KEY, Codec, Layout, LocalGaussian, SourceChart, initial_local_gaussian,
    predict_local, retain, retention_mask, record_expiry, check_no_recreation, current_jacobian, chunk_plan,
    ForbiddenKeyRecreation, LocalResourceLimit)

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def numpy_backprojection(K, R, t, uv, depth):
    """Direct (uncentred) pinhole backprojection: x = R^T (D K^{-1}[u,v,1] - t)."""
    fx, fy, cx, cy = K
    ray = np.array([(uv[0]-cx)/fx, (uv[1]-cy)/fy, 1.])
    return R.T@(depth*ray-t)


def random_spd(n, rng, scale=1.):
    A = rng.standard_normal((n, n))
    return scale*(A@A.T+n*np.eye(n))


def random_bank(N, rng, device, kinds_per_point):
    rows = []
    for p, kinds in enumerate(kinds_per_point):
        for j, kind in enumerate(kinds):
            rows.append((p, kind, (j % 2) if kind == KEY else 0, j, 10+j if kind == KEY else 0))
    t = lambda i: torch.tensor([r[i] for r in rows], device=device)
    layout = Layout(N, t(0), t(1), t(2), t(3), t(4))
    g = LocalGaussian.empty(layout)
    dense = []
    for p in range(N):
        n = int(layout.dims[p])
        m, S = rng.standard_normal(n), random_spd(n, rng, 1e-2)
        dense.append((m, S))
        g.scatter(torch.tensor([p], device=device), torch.tensor(m[None], device=device), torch.tensor(S[None], device=device))
    return g, dense


class SourceChartTests(unittest.TestCase):
    def test_chart_equals_direct_backprojection_and_finite_differences(self):
        r = rig()
        K, R, t = r.K.numpy(), r.R.numpy(), r.t.numpy()
        cam, pix, W, D0 = 1, 27, 8, 2.013
        uv0 = np.array([pix % W, pix//W], float)
        base = numpy_backprojection(K[cam], R[cam], t[cam], uv0, D0)
        chart = SourceChart(r, 0, torch.tensor([cam]), torch.tensor([pix]), W, torch.tensor(base[None]), torch.tensor([D0], dtype=torch.float64))
        rng = np.random.default_rng(3)
        for _ in range(5):
            theta, eta = rng.normal(0, [.3, .3, 1, 1, 1]), rng.normal(0, .01)
            s, ja, jl = chart.evaluate(torch.tensor([0]), torch.tensor(theta[None]), torch.tensor([eta], dtype=torch.float64))
            direct = numpy_backprojection(K[cam], R[cam], t[cam], uv0+theta[:2], D0-eta)
            np.testing.assert_allclose(s[0, :3].numpy(), direct, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(s[0, 3:].numpy(), theta[2:])
            f = lambda th, e: np.r_[numpy_backprojection(K[cam], R[cam], t[cam], uv0+th[:2], D0-e), th[2:]]
            h = 1e-6
            fd_theta = np.stack([(f(theta+h*np.eye(5)[j], eta)-f(theta-h*np.eye(5)[j], eta))/(2*h) for j in range(5)], 1)
            fd_eta = (f(theta, eta+h)-f(theta, eta-h))/(2*h)
            np.testing.assert_allclose(ja[0].numpy(), fd_theta, rtol=0, atol=1e-8)
            np.testing.assert_allclose(jl[0, :, 3].numpy(), fd_eta, rtol=0, atol=1e-8)
            np.testing.assert_array_equal(jl[0, :, :3].numpy(), 0)

    def test_source_depth_correlation_matches_monte_carlo_and_no_redundant_state(self):
        r = rig()
        K, R, t = r.K.numpy(), r.R.numpy(), r.t.numpy()
        cam, pix, W, D0 = 0, 19, 8, 1.7
        uv0 = np.array([pix % W, pix//W], float)
        base = numpy_backprojection(K[cam], R[cam], t[cam], uv0, D0)
        chart = SourceChart(r, 0, torch.tensor([cam]), torch.tensor([pix]), W, torch.tensor(base[None]), torch.tensor([D0], dtype=torch.float64))
        Rs = np.array([1e-4, 2e-4, 3e-4, 4e-4])
        pixel_cov, vel_cov = np.eye(2)/12, np.eye(3)*.5
        g = initial_local_gaussian(chart, torch.tensor(Rs[None]), pixel_cov, vel_cov)
        kinds = g.layout.kind.tolist()
        self.assertEqual(kinds, [CHART, KEY])
        self.assertEqual(int(g.layout.dims[0]), 9)
        self.assertNotIn(STATE, kinds)
        codec = Codec(2, 4, 64)
        mean, cov = g.gather(torch.tensor([0]))
        s, J = current_jacobian(g, chart, codec, torch.tensor([0]), mean)
        analytic = (J@cov)[0, :3, 5+3].numpy()                   # Cov(x0, eta_source.depth)
        rng = np.random.default_rng(11)
        M = 200000
        th = rng.multivariate_normal(np.zeros(2), pixel_cov, M)
        eta = rng.normal(0, np.sqrt(Rs[3]), M)
        fx, fy, cx, cy = K[cam]
        rays = np.stack(((uv0[0]+th[:, 0]-cx)/fx, (uv0[1]+th[:, 1]-cy)/fy, np.ones(M)), 1)
        x = ((D0-eta)[:, None]*rays-t[cam])@R[cam]
        sample = ((x-x.mean(0))*(eta-eta.mean())[:, None]).mean(0)
        se = ((x-x.mean(0))*(eta-eta.mean())[:, None]).std(0)/np.sqrt(M)
        self.assertTrue(np.all(np.abs(sample-analytic) < 5*se+1e-12), (sample, analytic, se))
        delta = -R[cam].T@np.array([(uv0[0]-cx)/fx, (uv0[1]-cy)/fy, 1.])*Rs[3]
        np.testing.assert_allclose(analytic, delta, rtol=1e-12, atol=0)


class PackingTests(unittest.TestCase):
    def test_packed_roundtrip_and_slice_marginal(self):
        for device in DEVICES:
            rng = np.random.default_rng(5)
            kinds = [[STATE, KEY], [STATE, STATE, KEY, KEY, KEY], [STATE], [CHART, KEY, KEY]]
            kinds = [[STATE]+k[1:] if k[0] == STATE else k for k in kinds]
            g, dense = random_bank(len(kinds), rng, device, kinds)
            for budget in (10**9, 200000):
                for pts in chunk_plan(torch.arange(len(kinds), device=device), g.layout.dims, budget):
                    m, S = g.gather(pts)
                    for b, p in enumerate(pts.tolist()):
                        n = len(dense[p][0])
                        np.testing.assert_array_equal(m[b, :n].cpu().numpy(), dense[p][0])
                        np.testing.assert_array_equal(S[b, :n, :n].cpu().numpy(), dense[p][1])
                        self.assertEqual(float(S[b, n:].abs().sum()+S[b, :, n:].abs().sum()), 0.)
            keep = torch.tensor(rng.random(len(g.layout)) < .6, device=device)
            keep[g.layout.boff[:-1]] = True
            new = retain(g, keep, 200000)
            for p in range(len(kinds)):
                cols = []
                for j in range(int(g.layout.boff[p]), int(g.layout.boff[p+1])):
                    if bool(keep[j]):
                        a = int(g.layout.start[j])
                        cols += list(range(a, a+int(g.layout.size[j])))
                m, S = new.gather(torch.tensor([p], device=device))
                np.testing.assert_array_equal(m[0].cpu().numpy(), dense[p][0][cols])
                np.testing.assert_array_equal(S[0].cpu().numpy(), dense[p][1][np.ix_(cols, cols)])

    def test_chunk_budget_fails_explicitly(self):
        with self.assertRaises(LocalResourceLimit):
            chunk_plan(torch.arange(2), torch.tensor([10, 400]), 1000)


class PredictionTests(unittest.TestCase):
    def test_affine_prediction_matches_dense_transform(self):
        for device in DEVICES:
            rng = np.random.default_rng(8)
            kinds = [[STATE, STATE, KEY], [STATE, KEY, KEY]]
            g, dense = random_bank(2, rng, device, kinds)
            dt = -.07                                            # signed backward interval
            F = np.eye(6)
            F[:3, 3:] = dt*np.eye(3)
            qc = np.diag([.01, .02, .03])
            tau = abs(dt)
            Q = np.block([[tau**3/3*qc, np.sign(dt)*tau**2/2*qc], [np.sign(dt)*tau**2/2*qc, tau*qc]])
            chart = SimpleNamespace(anchor_frame=99)
            new = predict_local(g, chart, Codec(2, 100, 64), 7, torch.tensor(F, device=device), torch.tensor(Q, device=device), 10**8)
            for p, (m, S) in enumerate(dense):
                n = len(m)
                T = np.zeros((n+6, n))
                T[:6, :6] = F
                T[6:] = np.eye(n)
                Qa = np.zeros((n+6, n+6))
                Qa[:6, :6] = Q
                mm, SS = new.gather(torch.tensor([p], device=device))
                np.testing.assert_allclose(mm[0].cpu().numpy(), T@m, rtol=0, atol=1e-15)
                np.testing.assert_allclose(SS[0].cpu().numpy(), T@S@T.T+Qa, rtol=1e-13, atol=1e-17)
                self.assertEqual(int(new.layout.frame[new.layout.boff[p]]), 7)

    def test_chart_prediction_uses_chain_rule(self):
        r = rig()
        chart = SourceChart(r, 0, torch.tensor([0]), torch.tensor([27]), 8, torch.tensor([[.05, -.02, 2.]], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64))
        g = initial_local_gaussian(chart, torch.tensor([[1e-4, 1e-4, 1e-4, 4e-6]], dtype=torch.float64), np.eye(2)/12, np.eye(3))
        dt = .04
        F = torch.eye(6, dtype=torch.float64)
        F[:3, 3:] = dt*torch.eye(3)
        Q = torch.eye(6, dtype=torch.float64)*1e-9
        codec = Codec(2, 4, 64)
        new = predict_local(g, chart, codec, 1, F, Q, 10**8)
        m0, S0 = g.gather(torch.tensor([0]))
        s, J = current_jacobian(g, chart, codec, torch.tensor([0]), m0)
        m1, S1 = new.gather(torch.tensor([0]))
        np.testing.assert_allclose(m1[0, :6].numpy(), (F@s[0]).numpy(), atol=1e-15, rtol=0)
        np.testing.assert_allclose(S1[0, :6, 6:].numpy(), (F@J[0]@S0[0]).numpy(), rtol=1e-14, atol=1e-20)
        np.testing.assert_allclose(S1[0, :6, :6].numpy(), (F@J[0]@S0[0]@J[0].T@F.T+Q).numpy(), rtol=1e-13, atol=1e-20)


class MarginalizationTests(unittest.TestCase):
    def test_slicing_equals_square_root_schur_of_correlated_posterior(self):
        rng = np.random.default_rng(21)
        n = 6+6+4*3
        prior = random_spd(n, rng, 1e-3)
        H = rng.standard_normal((8, n))
        Rm = np.diag(rng.uniform(1e-4, 1e-3, 8))
        S = H@prior@H.T+Rm
        Kg = prior@H.T@np.linalg.inv(S)
        post = (np.eye(n)-Kg@H)@prior@(np.eye(n)-Kg@H).T+Kg@Rm@Kg.T
        mean = rng.standard_normal(n)
        layout = Layout(1, torch.zeros(5, dtype=torch.long), torch.tensor([STATE, STATE, KEY, KEY, KEY]), torch.tensor([0, 0, 0, 1, 1]),
                        torch.tensor([2, 1, 1, 1, 2]), torch.tensor([0, 0, 5, 6, 7]))
        g = LocalGaussian.empty(layout)
        g.scatter(torch.tensor([0]), torch.tensor(mean[None]), torch.tensor(post[None]))
        for drop in ([1, 2], [1, 2, 3], [3]):
            keep = torch.ones(5, dtype=torch.bool)
            keep[drop] = False
            new = retain(g, keep, 10**8)
            R = [c for j in range(5) if bool(keep[j]) for c in range(int(layout.start[j]), int(layout.start[j]+layout.size[j]))]
            E = [c for c in range(n) if c not in R]
            # independent square-root Schur: information L L^T, eliminate E (last) by Cholesky of the reordered information
            Lam = np.linalg.inv(post)
            Lr = np.linalg.cholesky(Lam[np.ix_(R+E, R+E)][::-1, ::-1])[::-1, ::-1]   # upper-left block of R is the Schur root
            schur_root = Lr[:len(R), :len(R)]
            expected = np.linalg.inv(schur_root@schur_root.T)
            m, Snew = new.gather(torch.tensor([0]))
            np.testing.assert_allclose(Snew[0].numpy(), expected, rtol=0, atol=1e-9*np.abs(expected).max())  # oracle inverts twice
            np.testing.assert_array_equal(m[0].numpy(), mean[R])
            self.assertGreater(np.abs(post[np.ix_(R[:6], R[6:])]).max() if len(R) > 6 else 1., 0.)


class RetentionTests(unittest.TestCase):
    def references(self):
        refs = ReferenceTable(2, 1, 'cpu', torch.float64)
        refs.frame[:, 0] = torch.tensor([3, 2])
        refs.ordinal[:, 0] = torch.tensor([3, 2])
        return refs

    def layout(self):
        # current state 3, states 2 and 1, keys of camera0@frame3, camera1@frame2, camera1@frame1, camera0@frame2
        return Layout(1, torch.zeros(7, dtype=torch.long), torch.tensor([STATE, STATE, STATE, KEY, KEY, KEY, KEY]),
                      torch.tensor([0, 0, 0, 0, 1, 1, 0]), torch.tensor([3, 2, 1, 3, 2, 1, 2]), torch.tensor([0, 0, 0, 5, 6, 7, 8]))

    def test_lag_two_and_three_retention(self):
        chart = SimpleNamespace(camera=torch.tensor([0]), pixel=torch.tensor([99]), anchor_frame=0)
        refs, L = self.references(), self.layout()
        keep3 = retention_mask(L, refs, chart, 3, 3, 3)
        self.assertEqual(keep3.tolist(), [True, True, False, True, True, False, False])
        keep2 = retention_mask(L, refs, chart, 3, 3, 2)
        self.assertEqual(keep2.tolist(), [True, False, False, True, False, False, False])

    def test_expired_keys_are_never_recreated(self):
        chart = SimpleNamespace(camera=torch.tensor([0]), pixel=torch.tensor([99]), anchor_frame=0)
        refs, L = self.references(), self.layout()
        keep = retention_mask(L, refs, chart, 3, 3, 3)
        frame_ordinal = torch.arange(5)
        expired = torch.full((2, 1), -1, dtype=torch.long)
        self.assertEqual(record_expiry(L, keep, expired, frame_ordinal), 2)
        self.assertEqual(expired[:, 0].tolist(), [2, 1])
        with self.assertRaises(ForbiddenKeyRecreation):
            check_no_recreation(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]), expired, frame_ordinal)
        check_no_recreation(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([3, 2]), expired, frame_ordinal)

    def test_source_error_lives_with_its_chart(self):
        chart = SimpleNamespace(camera=torch.tensor([1]), pixel=torch.tensor([33]), anchor_frame=0)
        refs = ReferenceTable(2, 1, 'cpu', torch.float64)
        refs.frame[:, 0] = torch.tensor([0, 1])
        refs.ordinal[:, 0] = torch.tensor([0, 1])
        L = Layout(1, torch.zeros(3, dtype=torch.long), torch.tensor([STATE, CHART, KEY]), torch.tensor([0, 0, 1]),
                   torch.tensor([1, 0, 0]), torch.tensor([0, 0, 33]))
        self.assertEqual(retention_mask(L, refs, chart, 1, 1, 3).tolist(), [True, True, True])
        self.assertEqual(retention_mask(L, refs, chart, 1, 1, 2).tolist(), [True, False, False])
