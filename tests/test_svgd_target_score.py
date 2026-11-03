"""Gate 1 / TF.6 item 1: operator A (target density + XYZ score) against an independent dense NumPy/SciPy oracle."""
import itertools
import unittest
import numpy as np
import torch
from scipy.stats import multivariate_normal
from estimator.svgd_target_score import target_score, responsibilities
from tests.svgd_fixtures import record

TOL_REL = 1e-11      # float64: log-sum-exp over <= 10^2 terms, condition <= 1e4 covariances -> ~1e-13 roundoff; 100x margin


def spd(rng, n, scale):
    A = rng.normal(size=(n, 3, 3)); return np.einsum('nij,nkj->nik', A, A)*scale+np.eye(3)*scale*.2


def oracle(qx, qc, y, d, LX, LC, w=None, Sa=None):
    """Independent: scipy multivariate_normal per component, explicit softmax, explicit score sum."""
    M = len(y); w = np.full(M, 1/M) if w is None else w
    LC = np.broadcast_to(LC, (M, 3, 3)) if LC.ndim == 2 else LC; LX = np.broadcast_to(LX, (M, 3, 3)) if LX.ndim == 2 else LX
    if Sa is not None: LC = LC+Sa
    la = np.array([[np.log(w[j])+multivariate_normal(y[j], LX[j]).logpdf(qx[i])+multivariate_normal(d[j], LC[j]).logpdf(qc[i]) for j in range(M)] for i in range(len(qx))])
    m = la.max(1, keepdims=True); r = np.exp(la-m); s = r.sum(1, keepdims=True); r /= s
    g = np.einsum('nm,mnk->nk', r, np.stack([np.linalg.solve(LX[j], (y[j]-qx).T).T for j in range(M)]))
    gc = np.einsum('nm,mnk->nk', r, np.stack([np.linalg.solve(LC[j], (d[j]-qc).T).T for j in range(M)]))
    return (m+np.log(s))[:, 0], g, gc, r


def fixture(seed, N=7, M=23, hetero=True, equal_colours=False):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, .01, (M, 3)); d = rng.uniform(.2, .8, (M, 3))
    if equal_colours: d[:] = .5
    if M >= 6:
        d[M//2:M//2+3] = d[M//2]+rng.normal(0, 1e-3, (3, 3))                 # coloured near-overlap
        y[M//2:M//2+3] = y[M//2]+rng.normal(0, 1e-4, (3, 3))
    qx = np.r_[rng.normal(0, .01, (N-2, 3)), [[.3, -.2, .1]], [[1., 1., 1.]]]   # two far queries
    qc = rng.uniform(.2, .8, (N, 3))
    LX = spd(rng, M, 1e-5) if hetero else spd(rng, 1, 1e-5)[0]
    LC = spd(rng, M, 1e-3) if hetero else spd(rng, 1, 1e-3)[0]
    return qx, qc, y, d, LX, LC


def run_t(qx, qc, y, d, LX, LC, ws=4096, **kw):
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
    return target_score(t(qx), t(qc), t(y), t(d), t(LX), t(LC), workspace_bytes=ws, **{k: (t(v) if isinstance(v, np.ndarray) else v) for k, v in kw.items()})


class TargetScoreTests(unittest.TestCase):
    def test_log_density_and_score_match_dense_numpy(self):
        worst = {'logp_abs': 0., 'grad_rel': 0., 'resp_abs': 0.}
        for seed, (N, M), hetero, eq in itertools.product((0, 1), ((7, 23), (1, 5), (9, 1)), (True, False), (False, True)):
            qx, qc, y, d, LX, LC = fixture(seed, max(N, 2), M, hetero, eq)
            qx, qc = qx[:N], qc[:N]
            lp, g, gc, r = oracle(qx, qc, y, d, LX, LC)
            for ws in (200, 1 << 26):                                   # forces 1x1 tiles vs one tile: same result
                out = run_t(qx, qc, y, d, LX, LC, ws=ws)
                np.testing.assert_allclose(out.log_density.numpy(), lp, rtol=TOL_REL, atol=1e-9)
                scale = np.abs(g).max(1, keepdims=True)+1e-300
                np.testing.assert_allclose(out.grad_xyz.numpy()/scale, g/scale, atol=TOL_REL)
                np.testing.assert_allclose(out.grad_rgb.numpy()/(np.abs(gc).max(1, keepdims=True)+1e-300), gc/(np.abs(gc).max(1, keepdims=True)+1e-300), atol=TOL_REL)
                worst['logp_abs'] = max(worst['logp_abs'], float(np.abs(out.log_density.numpy()-lp).max()))
                worst['grad_rel'] = max(worst['grad_rel'], float((np.abs(out.grad_xyz.numpy()-g)/scale).max()))
                self.assertTrue(bool((out.candidate_count == M).all())); self.assertEqual(float(out.tail_bound.abs().max()), 0.)
            t = lambda a: torch.as_tensor(a, dtype=torch.float64)
            rr = responsibilities(t(qx), t(qc), t(y), t(d), t(LX), t(LC)).numpy()
            np.testing.assert_allclose(rr, r, atol=1e-12); np.testing.assert_allclose(rr.sum(1), 1, atol=1e-13)
            worst['resp_abs'] = max(worst['resp_abs'], float(np.abs(rr-r).max()))
        record('E01', test='dense_numpy_oracle', **worst, tolerance=TOL_REL, basis='float64 roundoff of <=23-term log-sum-exp, 100x margin')
        record('E02', test='responsibilities', max_abs=worst['resp_abs'], tolerance=1e-12)
        record('E03', test='score_vs_oracle', max_rel=worst['grad_rel'], tolerance=TOL_REL)

    def test_score_five_point_finite_difference(self):
        qx, qc, y, d, LX, LC = fixture(3)
        out = run_t(qx[:5], qc[:5], y, d, LX, LC)
        worst = 0.
        for i, k in itertools.product(range(5), range(3)):
            h = 2e-5; f = lambda s: run_t(qx[i:i+1]+s*np.eye(3)[k], qc[i:i+1], y, d, LX, LC).log_density.item()
            fd = (-f(2*h)+8*f(h)-8*f(-h)+f(-2*h))/(12*h)
            err = abs(fd-out.grad_xyz[i, k].item())/(abs(out.grad_xyz[i]).max().item()+1)
            worst = max(worst, err); self.assertLess(err, 1e-6)
            g = lambda s: run_t(qx[i:i+1], qc[i:i+1]+s*np.eye(3)[k], y, d, LX, LC).log_density.item()
            fdc = (-g(2*h)+8*g(h)-8*g(-h)+g(-2*h))/(12*h)
            self.assertLess(abs(fdc-out.grad_rgb[i, k].item())/(abs(out.grad_rgb[i]).max().item()+1), 1e-6)
        record('E03', test='five_point_fd', max_rel=worst, tolerance=1e-6, basis='O(h^4) stencil, h=2e-5 m vs Lambda^1/2 ~ 3e-3 m')

    def test_normalization_integral(self):
        """int p(x, c) dx over a 3D grid = sum_j w_j N3(c; d_j, LC_j) (the x-factors are normalized densities)."""
        rng = np.random.default_rng(5)
        y = rng.normal(0, .004, (3, 3)); d = rng.uniform(.3, .7, (3, 3)); LX = np.stack([np.diag([4e-6, 9e-6, 6e-6])]*3); LC = np.eye(3)*1e-3
        c = d[1]+.01
        g = np.linspace(-.03, .03, 81); X = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
        out = run_t(X, np.broadcast_to(c, X.shape), y, d, LX, LC, ws=1 << 24)
        integral = np.exp(out.log_density.numpy()).sum()*(g[1]-g[0])**3
        want = np.mean([multivariate_normal(d[j], LC).pdf(c) for j in range(3)])
        self.assertLess(abs(integral/want-1), 1e-4)
        record('E01', test='normalization_integral', rel_error=abs(integral/want-1), tolerance=1e-4, basis='81^3 trapezoid grid, +-10 sigma box')

    def test_reference_colour_marginalization(self):
        """TC-0001: int p(x, c) N(c; a, Sa) dc = mixture with LC + Sa (Gauss-Hermite quadrature in colour)."""
        qx, qc, y, d, LX, LC = fixture(7, N=3, M=6, hetero=False)
        Sa = np.diag([2e-4, 3e-4, 1e-4])
        marg = run_t(qx, qc, y, d, LX, LC, query_color_cov=Sa).log_density.numpy()
        z, w = np.polynomial.hermite_e.hermegauss(30); Ls = np.linalg.cholesky(Sa)
        Z = np.stack(np.meshgrid(z, z, z, indexing='ij'), -1).reshape(-1, 3); Wq = np.einsum('i,j,k->ijk', w, w, w).ravel()/(2*np.pi)**1.5
        for i in range(3):
            cols = qc[i]+Z@Ls.T
            from scipy.special import logsumexp
            acc = logsumexp(np.log(Wq)+run_t(np.repeat(qx[i:i+1], len(cols), 0), cols, y, d, LX, LC, ws=1 << 24).log_density.numpy())
            self.assertLess(abs(acc-marg[i]), 1e-8*max(1, abs(marg[i])))
        record('E04', test='gauss_hermite_30', tolerance=1e-8, basis='30-point Gauss-Hermite per colour axis')

    def test_permutation_invariance(self):
        qx, qc, y, d, LX, LC = fixture(11)
        base = run_t(qx, qc, y, d, LX, LC)
        p = np.random.default_rng(0).permutation(len(y)); q = np.random.default_rng(1).permutation(len(qx))
        perm = run_t(qx[q], qc[q], y[p], d[p], LX[p], LC[p])
        np.testing.assert_allclose(perm.log_density.numpy(), base.log_density.numpy()[q], rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(perm.grad_xyz.numpy(), base.grad_xyz.numpy()[q], rtol=1e-10, atol=1e-8)

    def test_duplicate_identifier_vs_new_exposure(self):
        """A bookkeeping duplicate of the same raw sample is removed by source ID (invariant); a genuinely different camera
        exposure of the same location is a new sample under the declared 1/M_t law and DOES change the density."""
        from scene.pointcloud_sequence import PointCloudFrame, deduplicate_sources
        qx, qc, y, d, LX, LC = fixture(12, hetero=False)
        src = np.c_[np.zeros(len(y)), np.arange(len(y)) % 2, np.arange(len(y))].astype(np.int64)
        cl = PointCloudFrame(0, 0, 0., y, d, np.zeros((len(y), 3, 3)), np.ones(len(y)), src[:, 1], src, np.zeros((2, 3)))
        dup = PointCloudFrame(0, 0, 0., np.r_[y, y[:4]], np.r_[d, d[:4]], np.zeros((len(y)+4, 3, 3)), np.ones(len(y)+4), np.r_[src[:4, 1], src[:, 1]][:len(y)+4],
                              np.r_[src, src[:4]], np.zeros((2, 3)))
        dd = deduplicate_sources(dup)
        np.testing.assert_array_equal(dd.source, cl.source)
        a = run_t(qx, qc, cl.xyz, cl.rgb, LX, LC).log_density.numpy(); b = run_t(qx, qc, dd.xyz, dd.rgb, LX, LC).log_density.numpy()
        np.testing.assert_array_equal(a, b)
        new = src[:4].copy(); new[:, 1] = 7                                        # other camera, same XYZ/RGB
        c = run_t(qx, qc, np.r_[y, y[:4]], np.r_[d, d[:4]], LX, LC).log_density.numpy()
        self.assertGreater(np.abs(c-a).max(), 1e-3)

    def test_empty_frame_no_observation(self):
        out = run_t(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.eye(3), np.eye(3))
        self.assertTrue(bool((out.support_status == 1).all())); self.assertTrue(bool(torch.isnan(out.grad_xyz).all()))

    def test_constant_rgb_and_colour_distinct_surfaces(self):
        rng = np.random.default_rng(2)
        y = np.r_[rng.normal(0, .002, (20, 3)), rng.normal(0, .002, (20, 3))+[.01, 0, 0]]
        LX = np.eye(3)*4e-6
        spatial = run_t([[.005, 0, 0]], [[.5, .5, .5]], y, np.full((40, 3), .5), LX, np.eye(3)*1e9).grad_xyz.numpy()
        d = np.r_[np.full((20, 3), .2), np.full((20, 3), .8)]
        wide = run_t([[.005, 0, 0]], [[.2, .2, .2]], y, d, LX, np.eye(3)*1e9).grad_xyz.numpy()
        np.testing.assert_allclose(wide, spatial, rtol=1e-6)                       # huge LC: colour has no effect
        sharp = run_t([[.005, 0, 0]], [[.2, .2, .2]], y, d, LX, np.eye(3)*1e-3).grad_xyz.numpy()
        self.assertLess(sharp[0, 0], 0)                                             # pulled toward its own colour's surface
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
        r_sharp = responsibilities(t([[.005, 0, 0]]), t([[.2, .2, .2]]), t(y), t(d), t(LX), t(np.eye(3)*1e-3)).numpy()[0]
        r_spatial = responsibilities(t([[.005, 0, 0]]), t([[.2, .2, .2]]), t(y), t(d), t(LX), t(np.eye(3)*1e9)).numpy()[0]
        self.assertGreater(r_sharp[:20].sum(), .999); self.assertTrue(.01 < r_spatial[:20].sum() < .99)

    def test_smaller_lambda_sharpens_and_narrows(self):
        rng = np.random.default_rng(9); y = rng.normal(0, .003, (50, 3)); d = np.full((50, 3), .5)
        res = [run_t(y[:5], d[:5], y, d, np.eye(3)*s, np.eye(3)) for s in (1e-4, 1e-5, 1e-6)]
        lp = [r.log_density.mean().item() for r in res]; ne = [r.effective_components.mean().item() for r in res]
        self.assertTrue(lp[0] < lp[1] < lp[2]); self.assertTrue(ne[0] > ne[1] > ne[2])


if __name__ == '__main__':
    unittest.main()
