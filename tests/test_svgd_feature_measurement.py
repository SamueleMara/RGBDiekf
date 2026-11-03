"""Gate 3: operator C (declared RFF feature map, JVP/VJP, normalized feature likelihood) against exact-kernel and FD oracles."""
import unittest
import numpy as np
import torch
from scipy.stats import multivariate_normal
from estimator.svgd_feature_measurement import RFFFeatures, exact_kernel, exact_mmd2, FeatureGaussian, FrameFeatureTerm, iid_sampling_gamma
from tests.svgd_fixtures import record

T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
BX, BC = np.diag([4e-4, 2.5e-4, 9e-4]), np.eye(3)*.02


def cloud(seed, n, shift=0.):
    rng = np.random.default_rng(seed); x = rng.normal(0, .02, (n, 3))+shift; return x, np.clip(.5+5*x[:, [1, 2, 0]], 0, 1)


class FeatureTests(unittest.TestCase):
    def test_rff_kernel_and_mmd(self):
        D = 40000; f = RFFFeatures(BX, BC, D, 7, chunk=4096)
        x, c = cloud(0, 40); kx = exact_kernel(x, c, x, c, BX, BC)
        P = f.psi(T(x), T(c)).numpy(); approx = P@P.T
        err = float(np.abs(approx-kx).max())
        self.assertLess(err, 6/np.sqrt(D))                                   # Monte Carlo: sd(psi^T psi') <= 1/sqrt(D)
        xq, cq = cloud(1, 50, shift=.01)
        mm_exact = exact_mmd2(x, c, xq, cq, BX, BC); mm_rff = float(((f.mean(T(x), T(c))-f.mean(T(xq), T(cq)))**2).sum())
        self.assertLess(abs(mm_rff-mm_exact), 6/np.sqrt(D))
        p = np.random.default_rng(2).permutation(40)                       # identical DISTRIBUTION, different order/IDs
        np.testing.assert_allclose(f.mean(T(x[p]), T(c[p])).numpy(), f.mean(T(x), T(c)).numpy(), atol=1e-15)
        self.assertNotEqual(f.digest(), RFFFeatures(BX, BC, D, 8).digest()); self.assertEqual(f.digest(), RFFFeatures(BX, BC, D, 7).digest())
        record('E14', test='rff_vs_exact_kernel', D=D, kernel_max_abs=err, mmd_exact=mm_exact, mmd_rff=mm_rff, bound=6/np.sqrt(D),
               basis='finite map is the declared model; agreement with the infinite kernel only to MC accuracy')

    def test_jvp_vjp_dot_product_and_fd(self):
        f = RFFFeatures(BX, BC, 256, 3, chunk=7); x, c = cloud(3, 25)
        rng = np.random.default_rng(4); u, w = rng.normal(size=(25, 3)), rng.normal(size=256)
        Ju = f.jvp(T(x), T(c), T(u)).numpy(); JTw = f.vjp(T(x), T(c), T(w)).numpy()
        dot = abs(Ju@w-(u*JTw).sum())/abs(Ju@w)
        self.assertLess(dot, 1e-13)
        h = 1e-6; fd = (f.mean(T(x+h*u), T(c)).numpy()-f.mean(T(x-h*u), T(c)).numpy())/(2*h)
        fderr = float(np.abs(fd-Ju).max()/np.abs(Ju).max()); self.assertLess(fderr, 1e-7)
        record('E13', test='jvp_vjp', dot_product_rel=dot, fd_rel=fderr, tolerance='1e-13 / 1e-7 (central FD O(h^2), h=1e-6 m vs B^1/2 >= 1.6e-2 m)')

    def test_normalized_nll_and_logdet(self):
        rng = np.random.default_rng(5); A = rng.normal(size=(12, 12)); G = A@A.T+np.eye(12)*.1; e = rng.normal(size=12)
        fg = FeatureGaussian(T(G))
        self.assertAlmostEqual(fg.nll(T(e)), -multivariate_normal(np.zeros(12), G).logpdf(e), places=10)
        with self.assertRaises(FloatingPointError): FeatureGaussian(T(np.diag([1., 1., -1e-9])))        # no silent jitter
        record('E12', test='nll_vs_scipy', abs_error=abs(fg.nll(T(e))+multivariate_normal(np.zeros(12), G).logpdf(e)), tolerance=1e-10)

    def test_frame_term_gradient_fd_and_gn(self):
        f = RFFFeatures(BX, BC, 32, 9); x, c = cloud(6, 30); y, d = cloud(7, 40, shift=.003)
        G = FeatureGaussian(iid_sampling_gamma(f, T(y), T(d), T(x), T(c)))      # M+N >= D: SPD (D > M+N is NUMERICAL_FAILURE)
        term = FrameFeatureTerm(f, T(y), T(d), G, T(c)); g = term.grad(T(x)).numpy()
        h = 1e-7; worst = 0.
        for i in (0, 7, 29):
            for k in range(3):
                e = np.zeros_like(x); e[i, k] = h
                fd = (term.nll(T(x+e))-term.nll(T(x-e)))/(2*h); worst = max(worst, abs(fd-g[i, k])/(np.abs(g).max()))
        self.assertLess(worst, 1e-5)
        u, v = np.random.default_rng(8).normal(size=(2, 30, 3))
        Au, Av = term.gn_apply(T(x), T(u)).numpy(), term.gn_apply(T(x), T(v)).numpy()
        self.assertAlmostEqual((Au*v).sum()/abs((Au*u).sum()), (Av*u).sum()/abs((Au*u).sum()), places=10); self.assertGreater((Au*u).sum(), 0)
        record('E13', test='objective_gradient_fd', max_rel=worst, tolerance=1e-5)

    def test_textured_vs_textureless_tangential_information(self):
        """Rigid in-plane translation of a uniform-colour plane is observable ONLY through its boundary: its information
        relative to the normal translation is small and shrinks with plane size; texture adds interior information.
        Information u^T J^T J u of the rigid direction u (Gamma = I); no velocity is fabricated."""
        f = RFFFeatures(np.eye(3)*1e-4, np.eye(3)*.01, 1024, 1); ratios = {}
        for L, n in ((.05, 30), (.1, 60)):
            g = np.linspace(-L, L, n); X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2); x = np.c_[X, np.zeros(len(X))]
            info = lambda c, u: float((f.jvp(T(x), T(c), T(np.tile(u, (len(x), 1)))).numpy()**2).sum())
            flat, tex = np.full((len(x), 3), .5), np.clip(.5+.4*np.sin(60*x[:, [0, 1, 0]]+np.array([0, 1, 2])), 0, 1)
            ratios[L] = (info(flat, [1, 0, 0])/info(flat, [0, 0, 1]), info(tex, [1, 0, 0])/info(tex, [0, 0, 1]))
        self.assertLess(ratios[.1][0], ratios[.05][0]); self.assertGreater(ratios[.1][1], 5*ratios[.1][0])
        record('E13', test='textured_vs_flat_tangential_information', tangential_over_normal={str(k): v for k, v in ratios.items()})

    def test_stepping_decreases_common_data_objective(self):
        f = RFFFeatures(np.eye(3)*4e-4, np.eye(3)*.02, 256, 2); x, c = cloud(9, 300); y = x+[.004, 0, 0]
        G = FeatureGaussian(iid_sampling_gamma(f, T(y), T(c), T(x), T(c)))
        term = FrameFeatureTerm(f, T(y), T(c), G, T(c)); E0 = term.nll(T(x))
        u = -term.grad(T(x)); Hu = term.gn_apply(T(x), u); step = float((u*u).sum()/(u*Hu).sum())
        self.assertLess(term.nll(T(x)+step*u), E0)

    def test_camera_density_duplication_changes_mean(self):
        """Uniform 1/M_t is biased under camera overlap (PDF 8.4): duplicating one camera's coverage moves y^F."""
        f = RFFFeatures(BX, BC, 256, 4); x, c = cloud(10, 200)
        a = f.mean(T(x), T(c)).numpy(); b = f.mean(T(np.r_[x, x[:50]]), T(np.r_[c, c[:50]])).numpy()
        self.assertGreater(np.abs(a-b).max(), 1e-3)
        record('E17', test='camera_duplication_bias', max_feature_shift=float(np.abs(a-b).max()))


if __name__ == '__main__':
    unittest.main()
