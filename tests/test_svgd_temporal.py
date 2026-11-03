"""Gate 4 (operator level): whole-sequence temporal prior (C4) vs dense autograd, an independent RTS smoother and SciPy."""
import unittest
import numpy as np
import torch
from scipy.stats import multivariate_normal
from estimator.svgd_temporal import TemporalPrior, pcg
from estimator.point_kalman_cuda import transition, process_noise
from tests.svgd_fixtures import record

TIMES = [0., .03, .07, .08, .12]                            # non-uniform actual timestamps


def prior(N=3, seed=0):
    rng = np.random.default_rng(seed); x0 = torch.tensor(rng.normal(0, .05, (N, 3)))
    A = rng.normal(size=(N, 3, 3)); P0x = torch.tensor(np.einsum('nij,nkj->nik', A, A)*1e-6+np.eye(3)*1e-7)
    return TemporalPrior(TIMES, np.diag([.2, .1, .3]), x0, P0x, np.eye(3)*.01, chunk=2), x0, rng


class TemporalTests(unittest.TestCase):
    def test_energy_gradient_and_hessian_dense(self):
        tp, x0, rng = prior()
        S = torch.tensor(rng.normal(0, .05, (3, 5, 6)), requires_grad=True)
        E = tp.energy(S)
        Et = .5*((S[:, 0]-tp.s0bar)[:, None]@tp.P0i@(S[:, 0]-tp.s0bar)[:, :, None]).sum()
        for t in range(1, 5):
            w = S[:, t]-S[:, t-1]@tp.F[t-1].T; Et = Et+.5*(w@tp.Qi[t-1]*w).sum()
        self.assertAlmostEqual(E, float(Et), places=9)
        g, = torch.autograd.grad(Et, S, create_graph=True)
        np.testing.assert_allclose(tp.grad(S.detach()).numpy(), g.detach().numpy(), rtol=1e-11, atol=1e-9)
        H = torch.stack([torch.autograd.grad(g.reshape(-1)[k], S, retain_graph=True)[0].reshape(-1) for k in range(S.numel())])
        U = torch.tensor(rng.normal(size=(3, 5, 6)))
        np.testing.assert_allclose(tp.hess_apply(U).numpy().reshape(-1), (H@U.reshape(-1)).numpy(), rtol=1e-10, atol=1e-8*float(H.abs().max()))
        Z = tp.solve(U); np.testing.assert_allclose((H@Z.reshape(-1)).numpy(), U.reshape(-1).numpy(), rtol=1e-8, atol=1e-8)
        record('E18', test='dense_autograd', solve_residual=float((H@Z.reshape(-1)-U.reshape(-1)).abs().max()), tolerance=1e-8,
               basis='block-Thomas vs dense Hessian, condition of L ~1e10')

    def test_log_normalizer_matches_markov_density(self):
        tp, x0, rng = prior(N=2)
        S = torch.tensor(rng.normal(0, .02, (2, 5, 6)))
        lp = 0.
        for i in range(2):
            P0 = np.linalg.inv(tp.P0i[i].numpy()); lp += multivariate_normal(tp.s0bar[i].numpy(), P0).logpdf(S[i, 0].numpy())
            for t in range(1, 5):
                Q = np.linalg.inv(tp.Qi[t-1].numpy()); lp += multivariate_normal(tp.F[t-1].numpy()@S[i, t-1].numpy(), Q).logpdf(S[i, t].numpy())
        self.assertAlmostEqual(-(tp.energy(S)+tp.log_normalizer)/lp, 1., places=10)

    def test_gauss_newton_equals_rts_smoother(self):
        """min E_time + 1/2 sum |y_t - x_t|^2_R (linear Gaussian) by PCG == independent Kalman filter + RTS smoother mean."""
        tp, x0, rng = prior(N=2, seed=3); R = np.diag([4e-6, 1e-6, 9e-6]); Ri = torch.tensor(np.linalg.inv(R))
        y = torch.tensor(x0.numpy()[:, None]+np.array([.0, .002, .005, .006, .01])[None, :, None]*np.array([1, -.5, .3])+rng.normal(0, 1e-3, (2, 5, 3)))
        def A(U):
            out = tp.hess_apply(U); out[:, :, :3] += U[:, :, :3]@Ri; return out
        S0 = torch.zeros(2, 5, 6, dtype=torch.float64); S0[:, :, :3] = x0[:, None]
        b = -tp.grad(S0); b[:, :, :3] += (y-S0[:, :, :3])@Ri
        d, cert = pcg(A, b, tp.solve, 1e-14, 200); S = S0+d
        self.assertEqual(cert['status'], 'CONVERGED')
        for i in range(2):                                             # independent RTS
            m = np.r_[x0[i].numpy(), 0, 0, 0]; P = np.linalg.inv(tp.P0i[i].numpy()); H = np.c_[np.eye(3), np.zeros((3, 3))]
            ms, Ps, mp, Pp = [], [], [], []
            for t in range(5):
                if t:
                    dt = TIMES[t]-TIMES[t-1]; F = transition(dt).numpy(); Q = process_noise(dt, torch.tensor(np.diag([.2, .1, .3]))).numpy()
                    m, P = F@m, F@P@F.T+Q
                mp.append(m); Pp.append(P)
                K = P@H.T@np.linalg.inv(H@P@H.T+R); m = m+K@(y[i, t].numpy()-H@m); P = (np.eye(6)-K@H)@P; ms.append(m); Ps.append(P)
            sm = [ms[-1]]
            for t in range(3, -1, -1):
                dt = TIMES[t+1]-TIMES[t]; F = transition(dt).numpy(); G = Ps[t]@F.T@np.linalg.inv(Pp[t+1]); sm.insert(0, ms[t]+G@(sm[0]-mp[t+1]))
            err = float(np.abs(np.array(sm)-S[i].numpy()).max())
            self.assertLess(err, 1e-9)
        record('E18', test='rts_oracle', max_abs_m=err, tolerance=1e-9)

    def test_improper_process_covariance_blocked(self):
        with self.assertRaisesRegex(ValueError, 'BLOCKED_MODEL_UNSPECIFIED'):
            TemporalPrior(TIMES, np.zeros((3, 3)), torch.zeros(1, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64)[None]*1e-6, np.eye(3))

    def test_pcg_cap_is_reported(self):
        tp, x0, rng = prior()
        b = torch.tensor(rng.normal(size=(3, 5, 6))); _, cert = pcg(lambda U: tp.hess_apply(U)+U, b, lambda r: r, 1e-14, 2)
        self.assertEqual(cert['status'], 'ITERATION_CAP')


if __name__ == '__main__':
    unittest.main()
