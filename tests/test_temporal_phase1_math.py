"""Gate 1 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: the linear-Gaussian mathematics of the temporal solver
against an independent dense NumPy oracle (full-trajectory Gaussian conditioning, nothing imported from the estimator).

G1.1 batched Kalman + RTS means and ALL covariance blocks (marginal, lag-one, frame-0 / frame-t) vs dense conditioning,
     random SPD priors, random Q_a, R, irregular actual dt, T = 3 .. 8.
G1.2 block-tridiagonal normal-equation solve and selected inverse vs the dense solve / dense inverse; equal to RTS.
G1.3 missing frames: s+ = s-, P+ = P- exactly; later measurements change earlier smoothed states exactly as the oracle.
G1.4 Q(dt) dimensions / symmetry / PSD for irregular and signed intervals; Q_a = 0 handled exactly (filter / smoother
     need no Q^-1; the batch solver uses the exact reduced problem) instead of a silent singular inversion.
Tolerances: float64, relative to the oracle's scale times its condition number bound (stated per assertion)."""
import unittest
import numpy as np
import torch
from estimator.rts_point_smoother import kalman_filter_linear, rts_smooth, displacement_cov, process_noise, transitions
from estimator import block_tridiagonal as BT


def spd(g, n, scale=1.):
    A = g.normal(size=(n, n)); return scale*(A@A.T+.5*np.eye(n))


def dense_oracle(mu0, P0, F, Q, H, R, y, present):
    """Joint Gaussian of the stacked trajectory S = [s_0..s_{K-1}] and the present measurements; posterior by covariance-
    form conditioning (works for singular Q, i.e. Q_a = 0)."""
    K = len(F)+1
    T = np.zeros((6*K, 6*K))                      # S = T [s0, w_0 .. w_{K-2}]
    Phi = np.eye(6)
    for k in range(K):
        if k > 0: Phi = F[k-1]@Phi
        T[6*k:6*k+6, :6] = Phi
        for j in range(1, k+1):                   # w_{j-1} enters s_k through F_{k-1} .. F_j
            M = np.eye(6)
            for i in range(j, k): M = F[i]@M
            T[6*k:6*k+6, 6*j:6*j+6] = M
    C0 = np.zeros((6*K, 6*K)); C0[:6, :6] = P0
    for j in range(1, K): C0[6*j:6*j+6, 6*j:6*j+6] = Q[j-1]
    m = T@np.r_[mu0, np.zeros(6*(K-1))]; S = T@C0@T.T
    rows, ys, Rs = [], [], []
    for k in range(K):
        if present[k]:
            Hb = np.zeros((H.shape[1], 6*K)); Hb[:, 6*k:6*k+6] = H[k]; rows.append(Hb); ys.append(y[k]); Rs.append(R[k])
    if not rows: return m, S
    Hb = np.vstack(rows); yb = np.concatenate(ys)
    Rb = np.zeros((len(yb), len(yb))); o = 0
    for Rk in Rs: Rb[o:o+len(Rk), o:o+len(Rk)] = Rk; o += len(Rk)
    Sy = Hb@S@Hb.T+Rb; Kg = np.linalg.solve(Sy, Hb@S).T
    return m+Kg@(yb-Hb@m), S-Kg@Hb@S


def problem(g, K, n, m=3, qa_zero=False, missing=()):
    times = np.cumsum(np.r_[0., g.uniform(.01, .09, K-1)])          # irregular actual intervals
    qa = np.zeros((3, 3)) if qa_zero else spd(g, 3, .05)
    F, Q = transitions(times, torch.as_tensor(qa))
    F, Q = F.numpy(), Q.numpy()
    mu0 = g.normal(size=(n, 6)); P0 = np.stack([spd(g, 6, .1) for _ in range(n)])
    H = np.zeros((K, n, m, 6)); H[..., :3] = g.normal(size=(K, n, m, 3))
    R = np.stack([[spd(g, m, .01) for _ in range(n)] for _ in range(K)])
    y = g.normal(size=(K, n, m))
    present = np.ones((K, n), bool)
    for k in missing: present[k] = False
    return times, qa, F, Q, mu0, P0, H, R, y, present


def rel(a, b):
    return float(np.abs(a-b).max()/max(np.abs(b).max(), 1e-300))


class KalmanRTSOracleTest(unittest.TestCase):
    def check(self, K, n=4, seed=0, qa_zero=False, missing=()):
        g = np.random.default_rng(seed)
        times, qa, F, Q, mu0, P0, H, R, y, present = problem(g, K, n, qa_zero=qa_zero, missing=missing)
        t = lambda a: torch.as_tensor(a)
        mf, Pf, _ = kalman_filter_linear(t(mu0), t(P0), t(F), t(Q), t(H), t(R), t(y), t(present))
        sm = rts_smooth(mf, Pf, t(F), t(Q))
        dcov = displacement_cov(sm['smooth_cov'], sm['cross0']).numpy()
        for i in range(n):
            mo, So = dense_oracle(mu0[i], P0[i], F, Q, H[:, i], R[:, i], y[:, i], present[:, i])
            scale = np.linalg.cond(So+1e-300*np.eye(len(So))) if not qa_zero else 1e8
            tol = 1e-13*max(1., scale)
            np.testing.assert_allclose(sm['smooth_mean'][:, i].numpy().ravel(), mo, rtol=0, atol=tol*max(1, np.abs(mo).max()))
            for k in range(K):
                self.assertLess(rel(sm['smooth_cov'][k, i].numpy(), So[6*k:6*k+6, 6*k:6*k+6]), tol, (k, i))
                self.assertLess(rel(sm['cross0'][k, i].numpy(), So[:6, 6*k:6*k+6]), tol, ('cross0', k, i))
                Dk = So[6*k:6*k+3, 6*k:6*k+3]+So[:3, :3]-So[6*k:6*k+3, :3]-So[:3, 6*k:6*k+3]
                if k > 0: self.assertLess(rel(dcov[k, i], Dk), tol, ('Cov(d)', k, i))
                if k < K-1: self.assertLess(rel(sm['lag1'][k, i].numpy(), So[6*k+6:6*k+12, 6*k:6*k+6]), tol, ('lag1', k, i))
        return mf, Pf, present

    def test_G1_1_random_spd_irregular_dt(self):
        for K in range(3, 9):
            self.check(K, seed=K)

    def test_G1_3_missing_frames_exact_prediction_and_smoothing(self):
        mf, Pf, present = self.check(7, seed=11, missing=(2, 3, 5))
        g = np.random.default_rng(11)
        times, qa, F, Q, *_ = problem(g, 7, 4, missing=(2, 3, 5))
        for k in (2, 3, 5):                                  # no fake information: P+ = P- = F P+_{k-1} F^T + Q exactly
            Pm = torch.as_tensor(F[k-1])@Pf[k-1]@torch.as_tensor(F[k-1]).T+torch.as_tensor(Q[k-1])
            np.testing.assert_allclose(Pf[k].numpy(), ((Pm+Pm.transpose(-1, -2))/2).numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(mf[k].numpy(), (mf[k-1]@torch.as_tensor(F[k-1]).T).numpy(), rtol=0, atol=0)

    def test_G1_4_process_noise_and_deterministic_dynamics(self):
        g = np.random.default_rng(5)
        qa = torch.as_tensor(spd(g, 3, .02))
        for dt in (1/59, .0371, -1/59, -.25, 2.):
            Qd = process_noise(dt, qa).numpy()
            self.assertEqual(Qd.shape, (6, 6)); np.testing.assert_array_equal(Qd, Qd.T)
            self.assertGreaterEqual(np.linalg.eigvalsh(Qd).min(), -1e-15*np.abs(Qd).max())
            # cross block sign follows the signed interval (backward pass B1): E[(x_{k+1}-x_k-dt v_k) dv^T] = sgn(dt) tau^2/2 q
            np.testing.assert_allclose(Qd[:3, 3:], np.sign(dt)*dt**2/2*qa.numpy())
        self.assertRaises(ValueError, process_noise, 0., qa)
        self.check(6, seed=3, qa_zero=True)                     # filter + RTS with Q = 0: no Q^-1, exact vs the oracle


class BlockTridiagonalOracleTest(unittest.TestCase):
    def test_G1_2_normal_equations_and_selected_inverse(self):
        g = np.random.default_rng(21)
        for K in (3, 5, 8):
            n = 3
            times, qa, F, Q, mu0, P0, H, R, y, present = problem(g, K, n)
            for i in range(n):
                # information form of the same linear problem: J = |s0-mu0|^2_P0^-1 + sum |s_{k+1}-F s_k|^2_Q^-1 + sum |y-Hs|^2_R^-1
                N = 6*K; A = np.zeros((N, N)); b = np.zeros(N)
                P0i = np.linalg.inv(P0[i]); A[:6, :6] += P0i; b[:6] += P0i@mu0[i]
                for k in range(K-1):
                    Qi = np.linalg.inv(Q[k]); E = np.zeros((6, N)); E[:, 6*k:6*k+6] = -F[k]; E[:, 6*k+6:6*k+12] = np.eye(6)
                    A += E.T@Qi@E
                for k in range(K):
                    Ri = np.linalg.inv(R[k, i]); Hk = H[k, i]
                    A[6*k:6*k+6, 6*k:6*k+6] += Hk.T@Ri@Hk; b[6*k:6*k+6] += Hk.T@Ri@y[k, i]
                Ad = torch.as_tensor(np.stack([A[6*k:6*k+6, 6*k:6*k+6] for k in range(K)]))[None]
                Bd = torch.as_tensor(np.stack([A[6*k:6*k+6, 6*k+6:6*k+12] for k in range(K-1)]))[None]
                fac = BT.factor(Ad, Bd)
                x = BT.solve(fac, torch.as_tensor(b.reshape(K, 6))[None])[0].numpy().ravel()
                xd = np.linalg.solve(A, b); Sd = np.linalg.inv(A)
                tol = 1e-12*np.linalg.cond(A)
                self.assertLess(rel(x, xd), tol)
                si = BT.selected_inverse(fac)
                for k in range(K):
                    self.assertLess(rel(si['diag'][0, k].numpy(), Sd[6*k:6*k+6, 6*k:6*k+6]), tol)
                    self.assertLess(rel(si['cross0'][0, k].numpy(), Sd[:6, 6*k:6*k+6]), tol)        # Cov(x_t, x_0) query
                    if k < K-1: self.assertLess(rel(si['lag1'][0, k].numpy(), Sd[6*k+6:6*k+12, 6*k:6*k+6]), tol)
                # linear Gaussian: the batch MAP equals the RTS smoother (spec section 4)
                mo, So = dense_oracle(mu0[i], P0[i], F, Q, H[:, i], R[:, i], y[:, i], present[:, i])
                self.assertLess(rel(x, mo), tol); self.assertLess(rel(Sd, So), tol)

    def test_G1_4_reduced_solve_for_deterministic_dynamics(self):
        g = np.random.default_rng(8)
        times, qa, F, Q, mu0, P0, H, R, y, present = problem(g, 5, 2, qa_zero=True)
        for i in range(2):
            Phi = [np.eye(6)]
            for k in range(4): Phi.append(F[k]@Phi[-1])
            P0i = np.linalg.inv(P0[i]); A0 = P0i.copy(); gg = P0i@mu0[i]
            for k in range(5):
                Ri = np.linalg.inv(R[k, i]); A0 += Phi[k].T@H[k, i].T@Ri@H[k, i]@Phi[k]; gg += Phi[k].T@H[k, i].T@Ri@y[k, i]
            x0, S0 = BT.reduced_solve(None, torch.as_tensor(A0)[None], torch.as_tensor(gg)[None])
            mo, So = dense_oracle(mu0[i], P0[i], F, Q, H[:, i], R[:, i], y[:, i], present[:, i])
            self.assertLess(rel(x0[0].numpy(), mo[:6]), 1e-9); self.assertLess(rel(S0[0].numpy(), So[:6, :6]), 1e-9)
            for k in range(5):                                       # deterministic trajectory s_k = Phi_k s_0 exactly
                self.assertLess(rel(Phi[k]@x0[0].numpy(), mo[6*k:6*k+6]), 1e-9)


if __name__ == '__main__':
    unittest.main()
