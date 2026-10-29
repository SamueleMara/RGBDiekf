"""Contract section 10 C-E: the joint rendered-mask IEKF.
C: matrix-free Jv / J^T w / A(v) / PCG step against a dense image-by-position oracle (overlapping pixels: the cross-point
blocks are nonzero and kept), and the 3D-reduced step with its velocity v = v- + L dx against a full dense 6N IEKF.
D: the pixel_marginal_adf covariance against the exact diagonal blocks of the dense joint inverse (exact without shared
pixels, quantified with them), SPD 6D reconstruction, the exact cross-covariance of two points sharing one pixel.
E: mask-only correction with zero direct RGB-D rows, an interior point without rows kept exactly, the uniform-background
limitation reported (no invented attraction), monotone objective."""
import copy
import unittest
import numpy as np
import torch
from tests.test_rendered_mask_measurement import rig, at_pixel, H, W
from estimator.rendered_mask_measurement import footprints, mask_rows
from estimator.rendered_mask_iekf import MaskSystem, pcg, components, posterior, MASK_STATUS

T = lambda a: torch.as_tensor(np.asarray(a, float))


def problem(n=6, ncam=2, spread=.002, seed=0, overlap=True, var=.25):
    g = np.random.default_rng(seed); r = rig(ncam)
    base = at_pixel(r, 0, 40.37, 30.61)
    if overlap: X = torch.stack([base+T(g.normal(scale=spread, size=3)) for _ in range(n)])
    else: X = torch.stack([at_pixel(r, 0, 10.4+12*i, 30.6) for i in range(n)])
    mask = torch.zeros((r.count, H, W), dtype=torch.bool)
    rows = mask_rows(footprints(X, r, H, W), ~mask, torch.full((r.count,), var, dtype=torch.float64), H, W, n)
    P = []
    for i in range(n):
        A0 = g.normal(size=(6, 6)); Pi = A0@A0.T*1e-3+np.diag([1e-5]*3+[1e-2]*3); P.append(Pi)
    P = T(np.stack(P)); s_minus = torch.cat([X+T(g.normal(scale=.001, size=(n, 3))), T(g.normal(scale=.1, size=(n, 3)))], 1)
    Ixx = T(np.stack([(lambda B: B@B.T*1e3)(g.normal(size=(3, 2))) for _ in range(n)]))           # RGB-D blocks (rank 2)
    gx = T(g.normal(size=(n, 3)))
    return r, X, rows, P, s_minus, Ixx, gx


def dense_J(rows, n):
    J = torch.zeros((rows.n_rows, 3*n), dtype=torch.float64)
    for q in range(len(rows.pair_point)):
        J[rows.pair_row[q], 3*rows.pair_point[q]:3*rows.pair_point[q]+3] = rows.g[q]
    return J


class SparseOracleTest(unittest.TestCase):
    def test_matrix_free_operators_and_pcg_match_dense(self):
        for seed in range(4):
            r, X, rows, P, s_minus, Ixx, gx = problem(n=2+2*seed, ncam=1+seed % 3, seed=seed)
            n = len(X); J = dense_J(rows, n); Rinv = torch.diag(1/rows.var)
            v = T(np.random.default_rng(seed).normal(size=(n, 3))); w = T(np.random.default_rng(seed+9).normal(size=rows.n_rows))
            np.testing.assert_allclose(rows.jv(v).numpy(), (J@v.reshape(-1)).numpy(), atol=1e-12)
            np.testing.assert_allclose(rows.jtw(w, n).reshape(-1).numpy(), (J.T@w).numpy(), atol=1e-12)
            S = P[:, :3, :3]; Sinv = torch.linalg.inv(S); Ablk = Sinv+Ixx
            Ad = torch.block_diag(*Ablk)+J.T@Rinv@J
            off = Ad.clone(); off[torch.block_diag(*torch.ones(n, 3, 3, dtype=torch.bool))] = 0
            self.assertGreater(float(off.abs().max()), 0)                                   # cross-point terms exist ...
            comp = components(rows, n, 'cpu'); idx = torch.argsort(comp, stable=True)
            _, cl = torch.unique_consecutive(comp[idx], return_counts=True)
            sysm = MaskSystem(rows, idx, cl, Ablk[idx], n)
            np.testing.assert_allclose(sysm.matvec(v[idx]).numpy(), (Ad@v.reshape(-1)).reshape(n, 3)[idx].numpy(), rtol=1e-11, atol=1e-6)   # ... and kept
            b = -(Sinv@(X-s_minus[:, :3])[..., None])[..., 0]+gx+rows.jtw(rows.r/rows.var, n)
            d, info = pcg(sysm, b[idx], 0., 3*n+10, forcing=0.)                     # the solver itself, to exactness
            dd = torch.linalg.solve(Ad, b.reshape(-1)).reshape(n, 3)
            np.testing.assert_allclose(d.numpy(), dd[idx].numpy(), rtol=1e-8, atol=1e-10)
            self.assertEqual(info['not_converged_components'], 0)

    def test_reduced_step_and_velocity_equal_the_dense_6n_iekf(self):
        r, X, rows, P, s_minus, Ixx, gx = problem(n=5, ncam=2, seed=3)
        n = len(X); J = dense_J(rows, n); Rinv = torch.diag(1/rows.var)
        S = P[:, :3, :3]; Sinv = torch.linalg.inv(S); L = P[:, 3:, :3]@Sinv
        Ad = torch.block_diag(*(Sinv+Ixx))+J.T@Rinv@J
        b = -(Sinv@(X-s_minus[:, :3])[..., None])[..., 0]+gx+rows.jtw(rows.r/rows.var, n)
        dX = torch.linalg.solve(Ad, b.reshape(-1)).reshape(n, 3)
        # dense 6N: s on the prior-conditional velocity manifold, H6 = [J_x, 0]
        s = torch.cat([X, s_minus[:, 3:]+(L@(X-s_minus[:, :3])[..., None])[..., 0]], 1)
        Lam = torch.linalg.inv(P); A6 = torch.block_diag(*Lam)
        H6 = torch.zeros((rows.n_rows, 6*n), dtype=torch.float64); I6 = torch.zeros((6*n, 6*n), dtype=torch.float64); g6 = torch.zeros(6*n, dtype=torch.float64)
        for i in range(n):
            H6[:, 6*i:6*i+3] = J[:, 3*i:3*i+3]; I6[6*i:6*i+3, 6*i:6*i+3] = Ixx[i]; g6[6*i:6*i+3] = gx[i]
        A6 = A6+I6+H6.T@Rinv@H6
        b6 = -(torch.block_diag(*Lam)@(s-s_minus).reshape(-1))+g6+H6.T@Rinv@rows.r
        d6 = torch.linalg.solve(A6, b6).reshape(n, 6)
        np.testing.assert_allclose(d6[:, :3].numpy(), dX.numpy(), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(d6[:, 3:].numpy(), (L@dX[..., None])[..., 0].numpy(), rtol=1e-8, atol=1e-10)   # dv = P_vx S^-1 dx


    def test_fast_trial_mask_change_equals_full_rerender(self):
        from estimator.rendered_mask_iekf import _mask_change
        g = np.random.default_rng(11)
        for seed in range(4):
            r, X, rows, P, s_minus, Ixx, gx = problem(n=8, ncam=2, seed=seed)
            n = len(X); bg = torch.ones((r.count, H, W), dtype=torch.bool); var = torch.full((r.count,), .25, dtype=torch.float64)
            comp = components(rows, n, 'cpu')
            mv = torch.as_tensor(g.random(n) < .5)
            Xt = torch.where(mv[:, None], X+T(g.normal(scale=.002, size=(n, 3))), X)
            if seed == 3: Xt = torch.where(mv[:, None], X+T([0., 0., -50.]), X)        # moving points leave every camera (no trial pair)
            key0 = rows.row_cam*(H*W)+rows.row_pix
            fast = _mask_change(rows, key0, Xt, r, bg, comp, mv, H, W, torch.float64, var)
            # brute force: full re-render, per component sum over the union of pixels its moving points touch
            rt = mask_rows(footprints(Xt, r, H, W), bg, var, H, W, n); kt = rt.row_cam*(H*W)+rt.row_pix
            ref = torch.zeros(n, dtype=torch.float64)
            for c in torch.unique(comp[mv]).tolist():
                ks = set(key0[rows.pair_row[(comp[rows.pair_point] == c) & mv[rows.pair_point]]].tolist()) | set(kt[rt.pair_row[(comp[rt.pair_point] == c) & mv[rt.pair_point]]].tolist())
                m0 = {int(k): float(v) for k, v in zip(key0, rows.m_hat)}; m1 = {int(k): float(v) for k, v in zip(kt, rt.m_hat)}
                ref[c] = sum(.5*(m1.get(k, 0.)**2-m0.get(k, 0.)**2)/.25 for k in ks)
            np.testing.assert_allclose(fast.numpy(), ref.numpy(), atol=1e-10)


class CovarianceTest(unittest.TestCase):
    def joint_vs_adf(self, overlap):
        r, X, rows, P, s_minus, Ixx, gx = problem(n=6, ncam=2, seed=5, overlap=overlap)
        n = len(X); J = dense_J(rows, n); Rinv = torch.diag(1/rows.var)
        S = P[:, :3, :3]; Sinv = torch.linalg.inv(S); L = P[:, 3:, :3]@Sinv; Cc = P[:, 3:, 3:]-L@P[:, :3, 3:]
        Pj = torch.linalg.inv(torch.block_diag(*(Sinv+Ixx))+J.T@Rinv@J)
        Sp, P6, sp, _ = posterior(X, s_minus, P, S, Sinv, Ixx, rows, L, Cc, torch.ones(n, dtype=torch.long), torch.float64)
        exact = torch.stack([Pj[3*i:3*i+3, 3*i:3*i+3] for i in range(n)])
        self.assertTrue(bool((torch.linalg.eigvalsh(P6) > 0).all()))
        np.testing.assert_allclose(P6[:, :3, :3].numpy(), Sp.numpy(), atol=0)
        np.testing.assert_allclose(P6[:, 3:, :3].numpy(), (L@Sp).numpy(), atol=1e-14)            # cross-covariance via L
        return exact, Sp, Pj

    def test_adf_is_exact_without_shared_pixels(self):
        exact, Sp, _ = self.joint_vs_adf(overlap=False)
        np.testing.assert_allclose(Sp.numpy(), exact.numpy(), rtol=1e-9, atol=1e-16)

    def test_adf_with_shared_pixels_is_quantified(self):
        exact, Sp, Pj = self.joint_vs_adf(overlap=True)
        ratio = (Sp.diagonal(dim1=1, dim2=2)/exact.diagonal(dim1=1, dim2=2)).numpy()
        print(f'\n[pixel_marginal_adf vs exact joint marginal, shared pixels] variance ratio min {ratio.min():.3f} median {np.median(ratio):.3f} max {ratio.max():.3f}')
        self.assertTrue(np.isfinite(ratio).all())
        off = Pj.clone()
        for i in range(len(Sp)): off[3*i:3*i+3, 3*i:3*i+3] = 0
        self.assertGreater(float(off.abs().max()), 0)                                              # the joint posterior is not block diagonal

    def test_two_points_one_pixel_have_nonzero_cross_covariance(self):
        r = rig(); n = 2
        X = torch.stack([at_pixel(r, 0, 40.3, 30.4), at_pixel(r, 0, 40.6, 30.5, Z=1.3)])
        rows = mask_rows(footprints(X, r, H, W), torch.ones((1, H, W), dtype=torch.bool), torch.full((1,), .25, dtype=torch.float64), H, W, n)
        J = dense_J(rows, n); Sinv = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)*1e4
        Pj = torch.linalg.inv(torch.block_diag(*Sinv)+J.T@torch.diag(1/rows.var)@J)
        self.assertGreater(float(Pj[:3, 3:].abs().max()), 0)


class IEKFBehaviourTest(unittest.TestCase):
    """update_points_rendered_mask on the ray-cast plate fixture (tests/image_fixtures.py)."""
    @classmethod
    def setUpClass(cls):
        from tests.image_fixtures import two_plate_scene, catalog, look_at
        from estimator.projective_camera import CameraRig
        from estimator.projective_sensor import MeasuredFrame
        from tests.projective_fixtures import POLICY
        ds = two_plate_scene('static', frames=1, cameras=4)
        ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
        cls.ds, cls.rig, cls.frame = ds, CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        cls.xyz, cls.rgb, _, _ = catalog(ds, 0)
        cls.policy = copy.deepcopy(POLICY)

    def run_points(self, x0, offset, pvv=.01):
        from estimator.rendered_mask_iekf import update_points_rendered_mask
        from tests.projective_fixtures import noise
        n = len(x0); dt = 1/30; I = torch.eye(3, dtype=torch.float64)
        s = torch.zeros((n, 6), dtype=torch.float64); s[:, :3] = torch.as_tensor(x0)+torch.as_tensor(offset)
        P = torch.zeros((n, 6, 6), dtype=torch.float64); P[:, :3, :3] = (1e-5+dt**2*pvv)*I; P[:, 3:, 3:] = pvv*I; P[:, :3, 3:] = P[:, 3:, :3] = dt*pvv*I
        rgb0 = torch.as_tensor(self.rgb[self.pick(x0)]) if False else torch.full((n, 3), .5, dtype=torch.float64)
        return s, P, update_points_rendered_mask(s, P, self.frame, self.rig, rgb0, noise(), self.policy, 10, 1e-8, mask_var=.25)

    def edge_points(self, k=8):
        x = self.xyz; sel = np.flatnonzero((np.abs(x[:, 2]) < 1e-9) & (x[:, 0] > .43) & (np.abs(x[:, 1]) < .2))     # base plate, +x edge
        return x[sel[np.linspace(0, len(sel)-1, k).round().astype(int)]]

    def test_mask_only_correction_without_direct_rows(self):
        x0 = self.edge_points()
        s, P, (sp, Pp, rep, _) = self.run_points(x0, [.012, 0, 0])                    # 1.2 cm beyond the silhouette edge (~2.5 px)
        st = rep['mask']['status']
        hid = (rep['nrows'] == 0)
        self.assertGreater(int(hid.sum()), 0)
        moved = (sp[:, 0] < s[:, 0]-1e-4) & hid                                       # pulled back toward the measured silhouette
        self.assertGreater(int(moved.sum()), 0)
        self.assertFalse(bool(rep['observed'][hid].any()))                            # still not a direct RGB-D observation
        for p in torch.nonzero(moved).flatten().tolist():
            dv = sp[p, 3:]-s[p, 3:]; dx = sp[p, :3]-s[p, :3]
            np.testing.assert_allclose(dv.numpy(), (P[p, 3:, :3]@torch.linalg.solve(P[p, :3, :3], dx)).numpy(), atol=1e-10)
        self.assertTrue(bool((torch.linalg.eigvalsh(Pp) > 0).all()))
        self.assertTrue(all(h['objective_change_line_searched'] <= 0 for h in rep['history']))   # every accepted joint step is non-increasing
        self.assertIn(MASK_STATUS[int(st[torch.nonzero(moved).flatten()[0]])], ('MASK_ONLY', 'MASK_AND_RGBD', 'NO_MASK_ROWS'))

    def test_interior_point_without_rows_is_kept_exactly(self):
        x = self.xyz; sel = np.flatnonzero((np.abs(x[:, 2]) < 1e-9) & (np.abs(x[:, 0]) < .3) & (np.abs(x[:, 1]) < .15) & ((np.abs(x[:, 0]) > .26) | (np.abs(x[:, 1]) > .14)))[:6]
        s, P, (sp, Pp, rep, _) = self.run_points(x[sel], [0, 0, -.05])                 # 5 cm behind the plate: no RGB-D row, inside the mask
        keep = (rep['nrows'] == 0) & (rep['mask']['rows_per_point'] == 0)
        self.assertGreater(int(keep.sum()), 0)
        self.assertTrue(torch.equal(sp[keep], s[keep])); self.assertTrue(torch.equal(Pp[keep], P[keep]))

    def test_far_outside_in_uniform_black_is_reported_not_attracted(self):
        x0 = self.edge_points(4)
        s, P, (sp, Pp, rep, _) = self.run_points(x0, [.30, 0, 0])                     # 30 cm outside: uniform black in every view
        st = rep['mask']['status']
        far = st == 4
        self.assertTrue(bool(far.all()), [MASK_STATUS[int(v)] for v in st])          # MASK_ROWS_UNIFORM_BACKGROUND
        px = 1.9/400                                                                  # one pixel at the fixture's depth (m)
        print(f"\n[uniform background] |J^T R^-1 r| {rep['mask']['gradient_norm'].tolist()}, |dx| {(sp[:, :3]-s[:, :3]).norm(dim=1).tolist()}")
        self.assertTrue(bool(((sp[:, :3]-s[:, :3]).norm(dim=1) < px).all()))          # at most a sub-pixel move: no invented attraction


if __name__ == '__main__':
    unittest.main()
