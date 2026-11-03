"""Gate 2 / TF.6 items 1, 9: operator B (restricted physical-particle Stein field) against dense NumPy and Liu & Wang SVGD."""
import unittest
import numpy as np
import torch
from estimator.svgd_particle_operator import stein_field
from tests.svgd_fixtures import record

T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
TOL = 1e-12


def oracle(x, c, g, Hx, Hc, color=True):
    N = len(x); Px, Pc = np.linalg.inv(Hx), np.linalg.inv(Hc)
    K = np.zeros((N, N)); att = np.zeros((N, 3)); rep = np.zeros((N, 3))
    for i in range(N):
        for l in range(N):
            dx, dc = x[l]-x[i], c[l]-c[i]
            K[l, i] = np.exp(-.5*dx@Px@dx-(.5*dc@Pc@dc if color else 0.))
            att[i] += K[l, i]*g[l]; rep[i] += K[l, i]*Px@(x[i]-x[l])
    return att/N, rep/N, K


def liu_wang_svgd(theta, dlnprob, h):
    """Liu & Wang reference (python/svgd.py svgd_kernel + update direction) with a FIXED bandwidth h:
    k = exp(-|d|^2/h); phi = (k @ dlnprob + dxkxy)/n, dxkxy = (x sum_j k - k @ x) * 2/h."""
    sq = ((theta[:, None]-theta[None])**2).sum(-1); K = np.exp(-sq/h)
    dxkxy = (theta*K.sum(1)[:, None]-K@theta)*2/h
    return (K@dlnprob+dxkxy)/len(theta)


def fixture(seed, N):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)); Hx = A@A.T*1e-4+np.eye(3)*2e-5; B = rng.normal(size=(3, 3)); Hc = B@B.T*1e-2+np.eye(3)*1e-3
    x = rng.normal(0, .01, (N, 3))
    if N > 1: x[1] = x[0]+1e-7                                              # near-coincident pair
    c = rng.uniform(0, 1, (N, 3))
    if N > 3: c[2] = c[3]                                                    # equal colours
    return x, c, rng.normal(0, 100, (N, 3)), Hx, Hc


class SteinTests(unittest.TestCase):
    def test_dense_numpy_oracle_all_terms(self):
        worst = 0.
        for N in (1, 2, 13, 40):
            x, c, g, Hx, Hc = fixture(N, N)
            for color in (True, False):
                att, rep, K = oracle(x, c, g, Hx, Hc, color)
                for ws in (80, 1 << 26):
                    r = stein_field(T(x), T(c), T(g), T(Hx), T(Hc), kernel_color=color, workspace_bytes=ws)
                    sa = np.abs(att).max()+1e-300
                    # matmul form (OPT-02): repulsion = H^-1 (x_i' sum K - K x') loses below eps |x'| sum_l K ||H^-1|| / N (summand
                    # scale), not below eps |result|: the error bound is stated relative to that scale (rounding characterization).
                    sr = max(np.abs(rep).max(), np.abs(x-x.mean(0)).max()*K.sum(0).max()*np.linalg.norm(np.linalg.inv(Hx), 2)/N*1e-3)+1e-300
                    np.testing.assert_allclose(r.attraction.numpy()/sa, att/sa, atol=TOL); np.testing.assert_allclose(r.repulsion.numpy()/sr, rep/sr, atol=TOL)
                    np.testing.assert_allclose(r.total.numpy(), att+rep, rtol=1e-10, atol=1e-10*(sa+sr))
                    worst = max(worst, float(np.abs(r.attraction.numpy()-att).max()/sa), float(np.abs(r.repulsion.numpy()-rep).max()/sr))
                    self.assertEqual(r.pair_count, N); self.assertEqual(r.truncation_error_bound, 0.)
        r1 = stein_field(T(x[:1]), T(c[:1]), T(g[:1]), T(Hx), T(Hc))
        np.testing.assert_allclose(r1.attraction.numpy(), g[:1]); self.assertEqual(float(r1.repulsion.abs().max()), 0.)     # N = 1
        record('E07', test='dense_numpy_oracle', max_rel=worst, tolerance=TOL, basis='float64; repulsion relative to max(|rep|, 1e-3 x summand scale eps|x\'| sum K ||H^-1||/N)')

    def test_kernel_gradient_finite_difference(self):
        x, c, g, Hx, Hc = fixture(3, 6); Px = np.linalg.inv(Hx); Pc = np.linalg.inv(Hc)
        Kf = lambda xl, i: np.exp(-.5*(xl-x[i])@Px@(xl-x[i])-.5*(c[4]-c[i])@Pc@(c[4]-c[i]))
        h = 1e-6; want = np.array([Kf(x[4], i)*Px@(x[i]-x[4]) for i in range(6)])
        fd = np.array([[(-Kf(x[4]+2*h*e, i)+8*Kf(x[4]+h*e, i)-8*Kf(x[4]-h*e, i)+Kf(x[4]-2*h*e, i))/(12*h) for e in np.eye(3)] for i in range(6)])
        worst = float(np.abs(fd-want).max()/np.abs(want).max()); self.assertLess(worst, 1e-6)
        record('E06', test='kernel_gradient_fd', max_rel=worst, tolerance=1e-6)

    def test_repulsion_sign_two_particles(self):
        x = np.array([[0., 0, 0], [.004, .001, 0]]); Hx = np.eye(3)*1e-5
        r = stein_field(T(x), T(np.zeros((2, 3))), T(np.zeros((2, 3))), T(Hx), T(np.eye(3)), workspace_bytes=1 << 20)
        rep = r.repulsion.numpy(); K = np.exp(-.5*(x[1]-x[0])@(x[1]-x[0])/1e-5)
        self.assertGreater(rep[0]@(x[0]-x[1]), 0); self.assertGreater(rep[1]@(x[1]-x[0]), 0)      # pushes apart
        np.testing.assert_allclose(rep[0], K*(x[0]-x[1])/1e-5/2, rtol=1e-13)                     # global 1/N = 1/2

    def test_liu_wang_reference_gaussian(self):
        """Colour off, isotropic H = h/2 I: identical to the reference SVGD direction. Then 300 accepted SVGD steps on a
        Gaussian target improve the distribution (MMD to target samples falls) without collapse; attraction-only collapses."""
        rng = np.random.default_rng(0); mu = np.array([.01, -.02, .03]); S = np.diag([4e-4, 1e-4, 2.25e-4]); P = np.linalg.inv(S)
        th = rng.normal(0, .01, (60, 3)); h = 2e-4
        dl = (mu-th)@P
        ref = liu_wang_svgd(th, dl, h)
        ours = stein_field(T(th), T(np.zeros((60, 3))), T(dl), T(np.eye(3)*h/2), kernel_color=False).total.numpy()
        np.testing.assert_allclose(ours, ref, rtol=1e-11, atol=1e-11*np.abs(ref).max())
        record('E08', test='liu_wang_reference', max_abs=float(np.abs(ours-ref).max()), tolerance=1e-11)
        def evolve(rep):
            x = th.copy()
            for _ in range(300):
                x = x+2e-5*stein_field(T(x), T(np.zeros_like(x)), T((mu-x)@P), T(np.eye(3)*h/2), kernel_color=False, repulsion=rep).total.numpy()
            return x
        tgt = rng.multivariate_normal(mu, S, 400)
        mmd = lambda a: float(np.exp(-((a[:, None]-a[None])**2).sum(-1)/1e-3).mean()+np.exp(-((tgt[:, None]-tgt[None])**2).sum(-1)/1e-3).mean()
                              -2*np.exp(-((a[:, None]-tgt[None])**2).sum(-1)/1e-3).mean())
        full, att = evolve(True), evolve(False)
        self.assertLess(mmd(full), mmd(th)); self.assertLess(mmd(full), mmd(att))
        self.assertGreater(np.linalg.det(np.cov(full.T)), 20*np.linalg.det(np.cov(att.T)))          # attraction-only collapses
        record('E07', test='gaussian_target_distributional_improvement', mmd_init=mmd(th), mmd_full=mmd(full), mmd_attraction_only=mmd(att))

    def test_colour_kernel_and_fixed_colours(self):
        x, c, g, Hx, Hc = fixture(5, 12)
        a = stein_field(T(x), T(c), T(g), T(Hx), T(Hc)); b = stein_field(T(x), T(c*0+.5), T(g), T(Hx), T(Hc))
        off = stein_field(T(x), T(c), T(g), T(Hx), T(Hc), kernel_color=False)
        self.assertGreater(float((a.total-b.total).abs().max()), 0)                                   # colour changes the kernel
        np.testing.assert_allclose(b.total.numpy(), off.total.numpy(), rtol=1e-13)                    # equal colours = colour off
        self.assertEqual(len(a.total[0]), 3)                                                           # XYZ only: colours never transported

    def test_mutations_detected(self):
        """TF.6 item 9: degree normalization, removed repulsion and neighbour truncation all fail the oracle check."""
        x, c, g, Hx, Hc = fixture(8, 30); att, rep, K = oracle(x, c, g, Hx, Hc)
        want = att+rep; good = stein_field(T(x), T(c), T(g), T(Hx), T(Hc))
        check = lambda total: float(np.abs(total-want).max()/np.abs(want).max()) < 1e-10
        self.assertTrue(check(good.total.numpy()))
        self.assertFalse(check(good.total.numpy()*30/good.kernel_mass.numpy()[:, None]))              # 1/N -> 1/degree
        self.assertFalse(check(good.attraction.numpy()))                                               # repulsion removed
        Kt = K.copy()
        for i in range(30): Kt[np.argsort(-K[:, i])[5:], i] = 0                                         # 5-neighbour truncation
        Px = np.linalg.inv(Hx)
        trunc = (Kt.T@g+np.stack([sum(Kt[l, i]*Px@(x[i]-x[l]) for l in range(30)) for i in range(30)]))/30
        self.assertFalse(check(trunc))
        record('E07', test='mutations', detected=['degree_normalization', 'repulsion_removed', 'knn_truncation'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_bounded_memory_no_dense_allocation(self):
        N = 6000; x = torch.randn(N, 3, dtype=torch.float64, device='cuda')*.01
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        stein_field(x, x*0+.5, x*0, torch.eye(3, dtype=torch.float64)*1e-5, torch.eye(3, dtype=torch.float64), workspace_bytes=8*2**20)
        torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated()-base
        self.assertLess(peak, N*N*8/4)                                                                 # far below one dense N x N float64
        record('E07', test='bounded_memory', N=N, peak_bytes=peak, dense_bytes=N*N*8)


if __name__ == '__main__':
    unittest.main()
