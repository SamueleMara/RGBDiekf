"""Gate 0/1 oracles of the joint K = 1 pipeline (spec section 9): SE(3), KDE potential vs a dense NumPy oracle with its truncation
certificate, the shared-anchor integral vs an exact Gaussian model and QMC, the once-vs-per-frame reference distinction, exact 2^N label
enumeration, the F gradient vs central finite differences, T = I => gamma = pi, a_eff duplication invariance, SVGD direction, boundaries."""
import ast
import math
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.stats import multivariate_normal
from estimator.svgd_k1_motion import se3_exp, se3_log, compose, MotionPrior
from estimator.svgd_k1_observation import Background, FrameKDE, log_potential, choose_kappa, kernel_sum
from estimator.svgd_k1_reference import laplace_at_mean, qmc_log_evidence, f_g_H
from estimator.svgd_k1_joint_posterior import JointPosterior, enumerate_exact, log_prior_h
from estimator.svgd_k1_stein import svgd_direction
from tests.svgd_fixtures import record

REPO = Path(__file__).resolve().parents[1]
T64 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)


def scene(seed=0, M=60, N=8, K=3, shift=(.004, 0, 0)):
    g = np.random.default_rng(seed)
    y0 = g.normal(0, .02, (M, 3)); d0 = np.clip(.5+4*y0[:, [1, 2, 0]], 0, 1)
    frames = []
    for t in range(1, K):
        y = y0+np.asarray(shift)*t*(np.arange(M) < M//2)[:, None]; frames.append((y+g.normal(0, 3e-4, y.shape), d0, np.tile(np.eye(3)*1e-7, (M, 1, 1))))
    x0 = y0[:N]; c0 = d0[:N]
    return y0, d0, frames, x0, c0


def build(frames, ref_xyz, ref_rgb, rho=.3, s2=4e-6, LC=np.eye(3)*2e-3, tau=1e-9):
    bg = Background(ref_xyz, ref_rgb, np.tile(np.eye(3)*1e-7, (len(ref_xyz), 1, 1)), np.eye(3)*1e-4)
    out = []
    for y, d, S in frames:
        k = FrameKDE(y, d, S, s2, LC); k.build_grid(choose_kappa(k, bg, rho, tau, 40.)); out.append((k, bg, rho))
    return out, bg


class K1MathTests(unittest.TestCase):
    def test_se3_exp_log_and_finite_differences(self):
        for xi in ([.01, -.02, .03, 0, 0, 0], [.1, .2, -.1, .4, -.3, .7], [0, 0, 0, 1e-9, 0, 0], [.0, .0, .0, .9, 0, 0]):
            x = T64(xi); T = se3_exp(x)
            np.testing.assert_allclose(se3_log(T).numpy(), xi, atol=1e-10)
            R = T[:3, :3].numpy(); np.testing.assert_allclose(R@R.T, np.eye(3), atol=1e-14); self.assertAlmostEqual(np.linalg.det(R), 1, places=13)
            from scipy.linalg import expm                                            # independent Lie-group oracle: matrix exponential
            Xi = np.zeros((4, 4)); w = xi[3:]; Xi[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]; Xi[:3, 3] = xi[:3]
            np.testing.assert_allclose(T.numpy(), expm(Xi), atol=1e-12)
        x = T64([.1, .2, -.1, 1e-8, 2e-8, 0]).requires_grad_(); g, = torch.autograd.grad(se3_exp(x)[:3, 3].sum(), x)
        self.assertTrue(bool(torch.isfinite(g).all()))
        P = compose(torch.zeros(3, 6, dtype=torch.float64), T64([.1, .1, .1])); self.assertTrue(torch.equal(P[0], torch.eye(4, dtype=torch.float64)))
        self.assertTrue(torch.equal(P, torch.eye(4, dtype=torch.float64).expand(4, 4, 4)))           # zero twist -> identity bitwise

    def test_motion_prior_normalized_and_paid_once(self):
        dts = [.02, .03, .025]; pr = MotionPrior(dts, np.diag([1, 1, 1, 4, 4, 4.]), np.diag([.01]*3+[1]*3))
        nu = T64(np.random.default_rng(1).normal(size=(3, 6)))
        np.testing.assert_allclose(float(pr.log_prob(nu)), multivariate_normal(np.zeros(18), pr.cov.numpy()).logpdf(nu.numpy().ravel()), rtol=1e-10)   # 18-D dense logpdf roundoff ~1e-12
        np.testing.assert_allclose(pr.from_white(pr.to_white(nu)).numpy(), nu.numpy(), atol=1e-12)

    def test_kde_potential_dense_oracle_and_certificate(self):
        y0, d0, frames, x0, c0 = scene(); fr, bg = build(frames[:1], y0, d0)
        kde, _, rho = fr[0]; y, d, S = frames[0]
        q = np.r_[x0, [[1., 1., 1.]]]; c = np.r_[c0, [[.5, .5, .5]]]
        ll, g, H, cert = log_potential(kde, bg, rho, T64(q), T64(c))
        LX = S+4e-6*np.eye(3); LC = np.eye(3)*2e-3
        from scipy.special import logsumexp
        for i in range(len(q)):
            lp = logsumexp([multivariate_normal(y[j], LX[j]).logpdf(q[i])+multivariate_normal(d[j], LC).logpdf(c[i]) for j in range(len(y))])-math.log(len(y))
            lb = float(bg.log(T64(np.r_[q[i], c[i]])[None]))
            want = np.logaddexp(math.log(rho), math.log1p(-rho)+lp-lb)
            if np.isfinite(float(cert[i])) and float(cert[i]) < 1: self.assertLessEqual(abs(float(ll[i])-want), float(cert[i])+1e-10)
            else: self.assertEqual(float(ll[i]), math.log(rho))                                  # uncertified (outside the domain): neutral, counted
        h = 1e-6; fun = lambda u: float(log_potential(kde, bg, rho, T64(u[None, :3]), T64(u[None, 3:]))[0])
        u0 = np.r_[q[2], c[2]]
        fd = np.array([(fun(u0+h*e)-fun(u0-h*e))/(2*h) for e in np.eye(6)]); np.testing.assert_allclose(g[2].numpy(), fd, rtol=1e-5, atol=1e-6)
        Hf = np.array([[(fun(u0+h*ei+h*ej)-fun(u0+h*ei-h*ej)-fun(u0-h*ei+h*ej)+fun(u0-h*ei-h*ej))/(4*h*h) for ej in np.eye(6)] for ei in np.eye(6)])
        np.testing.assert_allclose(H[2].numpy(), Hf, rtol=2e-3, atol=2e-3*np.abs(Hf).max())
        self.assertAlmostEqual(float(ll[-1]), math.log(rho), places=9)                              # no support -> neutral log rho
        record('K1', test='kde_potential_oracle', max_certificate=float(cert.max()))

    def test_shared_anchor_integral_once_vs_per_frame(self):
        """Gaussian toy where every frame factor is Gaussian in u: the joint integral is exact; Laplace-at-mean is exact for it;
        per-frame convolution multiplied afterwards (reference counted T times) is NOT equal."""
        g = np.random.default_rng(3); S = np.diag([1e-6, 2e-6, 1.5e-6, 1e-4, 1e-4, 2e-4]); mu = g.normal(size=6)*.01; T = 4
        A = [np.diag(g.uniform(1e2, 1e4, 6)) for _ in range(T)]; m = [mu+g.normal(size=6)*.002 for _ in range(T)]
        # f(u) = sum_t -1/2 (u - m_t)^T A_t (u - m_t) (+ const 0): exact log int N(u;mu,S) exp(f) du
        Asum = sum(A); bsum = sum(At@mt for At, mt in zip(A, m)); c = sum(-.5*mt@At@mt for At, mt in zip(A, m))
        P = np.linalg.inv(S)+Asum; mpost = np.linalg.solve(P, np.linalg.solve(S, mu)+bsum)
        exact = c-.5*mu@np.linalg.solve(S, mu)+.5*mpost@P@mpost-.5*np.linalg.slogdet(S@P)[1]
        fmu = sum(-.5*(mu-mt)@At@(mu-mt) for At, mt in zip(A, m)); gmu = sum(-At@(mu-mt) for At, mt in zip(A, m))
        lap, ok = laplace_at_mean(T64([fmu]), T64(gmu[None]), T64(-Asum[None]), T64(np.linalg.cholesky(S)[None]), T64([1.]))
        self.assertTrue(bool(ok[0])); self.assertAlmostEqual(float(lap[0]), exact, places=8)
        per_frame = 0.
        for At, mt in zip(A, m):
            Pt = np.linalg.inv(S)+At; mp = np.linalg.solve(Pt, np.linalg.solve(S, mu)+At@mt)
            per_frame += -.5*mt@At@mt-.5*mu@np.linalg.solve(S, mu)+.5*mp@Pt@mp-.5*np.linalg.slogdet(S@Pt)[1]
        self.assertGreater(abs(per_frame-exact), 1e-3)
        record('K1', test='shared_anchor_once', exact=exact, laplace=float(lap[0]), wrong_per_frame=per_frame)

    def test_laplace_vs_qmc_on_kde_scene(self):
        y0, d0, frames, x0, c0 = scene(4); fr, bg = build(frames, y0, d0)
        mu = T64(np.c_[x0, c0]); Lc = T64(np.tile(np.linalg.cholesky(np.diag([1e-6]*3+[1e-5]*3)), (len(x0), 1, 1))); a = T64(np.full(len(x0), .5))
        P = torch.eye(4, dtype=torch.float64).expand(len(frames)+1, 4, 4).clone(); P[1:, 0, 3] = .002
        f, g, H, _ = f_g_H(fr, P, mu); lap, ok = laplace_at_mean(f, g, H, Lc, a)
        q = qmc_log_evidence(fr, P, mu, Lc, a, S=8192, seed=1)
        err = (lap-q).abs()
        self.assertTrue(bool(ok.all())); self.assertLess(float(err.max()), 1.)                       # MEASURED approximation (declared), not exactness
        self.assertGreater(float(np.corrcoef(lap.numpy(), q.numpy())[0, 1]), .95)
        record('K1', test='laplace_at_mean_vs_qmc8192', max_abs_nats=float(err.max()), median_abs_nats=float(err.median()),
               semantics='NUMERICAL_APPROX: B error ~0.1 nat median, up to ~0.7 nat when reference sigma ~ KDE bandwidth (measured)')

    def test_enumeration_gradients_identity_and_no_evidence(self):
        y0, d0, frames, x0, c0 = scene(5, N=5); fr, bg = build(frames, y0, d0)
        mu = T64(np.c_[x0, c0]); Lc = T64(np.tile(np.linalg.cholesky(np.diag([1e-6]*3+[1e-5]*3)), (5, 1, 1))); a = T64(np.full(5, .7))
        pr = MotionPrior([.03, .03], np.diag([1, 1, 1, 4, 4, 4.]), np.diag([.01]*3+[1]*3))
        post = JointPosterior(fr, mu, Lc, a, pr, (2., 3.), tile=3); post.set_static()
        nu = T64([[.1, -.05, .02, .3, -.1, .2], [.12, -.04, .02, .25, -.1, .1]]); h = T64(-.3)
        r = post.evaluate(nu, h)
        Fe, gam = enumerate_exact(post.lS.tolist(), r['logLM'].tolist(), float(h), 2., 3.)
        self.assertAlmostEqual(r['F'], Fe+float(pr.log_prob(nu)), places=8); np.testing.assert_allclose(r['gamma'].numpy(), gam, atol=1e-12)
        eps = 1e-6; fd = np.zeros((2, 6))
        for t in range(2):
            for k in range(6):
                e = torch.zeros(2, 6, dtype=torch.float64); e[t, k] = eps
                fd[t, k] = (post.evaluate(nu+e, h, grad=False)['F']-post.evaluate(nu-e, h, grad=False)['F'])/(2*eps)
        rel = np.abs(fd-r['grad_nu'].numpy()).max()/np.abs(fd).max(); self.assertLess(rel, 1e-5)
        fdh = (post.evaluate(nu, h+eps, grad=False)['F']-post.evaluate(nu, h-eps, grad=False)['F'])/(2*eps)
        pi = 1/(1+math.exp(-float(h))); want_h = float((r['gamma']-pi).sum())+2*(1-pi)-3*pi
        self.assertAlmostEqual(float(r['grad_h']), fdh, places=5); self.assertAlmostEqual(float(r['grad_h']), want_h, places=9)
        r0 = post.evaluate(torch.zeros(2, 6, dtype=torch.float64), h, grad=False)
        self.assertTrue(torch.equal(r0['logLM'], post.lS)); np.testing.assert_allclose(r0['gamma'].numpy(), pi, atol=1e-15)   # T = I
        record('K1', test='F_gradient_fd', max_rel=float(rel))

    def test_a_eff_duplication_invariance(self):
        g = np.random.default_rng(6); x = g.normal(0, .01, (40, 3)); c = g.uniform(0, 1, (40, 3)); H = np.eye(3)*1e-5; Hc = np.eye(3)*1e-2
        def aeff(x, c):
            r, _ = kernel_sum(x, c, H, Hc, 8.); N = len(x); neff = N*N/float(r.sum()); inv = 1/r; return neff, (neff*inv/inv.sum()).numpy()
        n1, a1 = aeff(x, c); n2, a2 = aeff(np.r_[x, x], np.r_[c, c])
        self.assertAlmostEqual(n1, n2, places=9); np.testing.assert_allclose(a2[:40]+a2[40:], a1, rtol=1e-12)

    def test_svgd_direction_uses_score_and_repulsion_sign(self):
        V = T64([[0., 0.], [1., 0.]]); G = T64([[1., 2.], [3., 4.]]); phi, d = svgd_direction(V, G)
        K = np.exp(-np.array([[0, 1], [1, 0.]])/d['kernel_bandwidth'])
        want = (K@G.numpy()+(2/d['kernel_bandwidth'])*np.array([[-K[0, 1], 0], [K[0, 1], 0]]))/2
        np.testing.assert_allclose(phi.numpy(), want, atol=1e-14)

    def test_point_only_boundary_and_gt_leak(self):
        from scene.svgd_k1_cloud_adapter import PointOnlyFrame
        f = PointOnlyFrame(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3, 3)), np.eye(3), 0.)
        for k in ('camera', 'pixel', 'depth', 'source', 'ray_origin'): self.assertRaises(AttributeError, getattr, f, k)
        banned = ('gt_joint', 'gt_transform', 'frame_values', 'ground_truth', 'tools.audit', 'evaluate_projective_gt', 'svgd_falsification')
        for p in list((REPO/'estimator').glob('svgd_k1_*.py'))+[REPO/'scene/svgd_k1_cloud_adapter.py', REPO/'tools/run_svgd_joint_k1.py']:
            src = p.read_text(); tree = ast.parse(src)
            names = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
            self.assertFalse([x for x in names if any(b in x for b in banned)], p.name); self.assertNotIn('frame_values.csv', src)
        for p in (REPO/'estimator').glob('svgd_k1_*.py'):
            src = p.read_text()
            for forbidden in ('.camera', '.pixel', 'depth_map', 'intrinsics', 'world_to_camera'): self.assertNotIn(forbidden, src, f'{p.name}: {forbidden}')


if __name__ == '__main__':
    unittest.main()
