"""Contract section 10 A-B: the rendered white-point pseudomask and its one-sided image innovation.
A: one point in a background pixel gives y_obs = 0, y_hat = 1, r = -1; foreground pixels carry no row; two points in
one pixel render white once (union in [0,1]); an occluded point still renders; a missing white pixel inside the mask
gives no correction; behind / outside a camera renders nothing there while other cameras still render.
B: the analytic d m_hat / d x against central finite differences of the full renderer, with overlapping contributors,
saturation (a = 1) and coincident points (finite, deterministic)."""
import unittest
import numpy as np
import torch
from tests.projective_fixtures import camera, rot
from estimator.projective_camera import CameraRig
from estimator.rendered_mask_measurement import footprints, hard_pseudomask, mask_rows, frame_statistics

W, H = 96, 72


def rig(n=1):
    cams = [camera('c0')]
    if n > 1: cams.append(camera('c1', rot([0, 1, 0], .3), [-.3, 0., .05]))
    if n > 2: cams.append(camera('c2', rot([1, 0, 0], -.25), [0., .25, .02]))
    return CameraRig(cams, 'cpu')


def at_pixel(r, c, u, v, Z=1.):
    """World point projecting to (u, v) at depth Z in camera c (inverse of the pinhole + extrinsics)."""
    fx, fy, cx, cy = r.K[c].tolist()
    Xc = torch.tensor([(u-cx)/fx*Z, (v-cy)/fy*Z, Z], dtype=torch.float64)
    return (r.R[c].T@(Xc-r.t[c]))


def rows_for(x, r, mask, var=.25):
    fp = footprints(x, r, H, W)
    return fp, mask_rows(fp, ~mask, torch.full((r.count,), var, dtype=torch.float64), H, W, len(x))


class SemanticsTest(unittest.TestCase):
    def test_point_in_background_pixel_is_minus_one(self):
        r = rig(); mask = torch.zeros((1, H, W), dtype=torch.bool)                  # all black
        x = at_pixel(r, 0, 40., 30.)[None]                                          # exactly on a pixel centre
        fp, rows = rows_for(x, r, mask)
        B = hard_pseudomask(fp, 1, H, W)
        self.assertTrue(bool(B[0, 30, 40])); self.assertEqual(int(B.sum()), 1)     # y_hat = 1 there, y_obs = 0
        i = int(torch.nonzero((rows.row_pix == 30*W+40)).flatten()[0])
        self.assertEqual(float(rows.m_hat[i]), 1.); self.assertEqual(float(rows.r[i]), -1.)   # r = y_obs - y_hat = -1 exactly

    def test_foreground_rows_excluded_and_black_black_zero(self):
        r = rig(); mask = torch.zeros((1, H, W), dtype=torch.bool); mask[0, 20:40, 30:50] = True
        x = at_pixel(r, 0, 40.3, 30.6)[None]                                        # footprint fully inside the white mask
        fp, rows = rows_for(x, r, mask)
        self.assertEqual(rows.n_rows, 0)                                            # white/white: no measurement row
        st = frame_statistics(fp, rows, ~mask, hard_pseudomask(fp, 1, H, W), H, W)
        self.assertEqual(st['hard_white_outside_measured_mask'], 0)
        # black/black: background pixels no point touches carry zero prediction and are not rows (exact omission)
        self.assertEqual(st['background_pixels_touched'], 0)

    def test_two_points_one_pixel_white_once_and_union_bounded(self):
        r = rig(); mask = torch.zeros((1, H, W), dtype=torch.bool)
        x = torch.stack([at_pixel(r, 0, 40., 30.), at_pixel(r, 0, 40., 30., Z=1.7)])  # same pixel, different depths
        fp, rows = rows_for(x, r, mask)
        self.assertEqual(int(hard_pseudomask(fp, 1, H, W).sum()), 1)                # white once, not intensity two
        x2 = torch.stack([at_pixel(r, 0, 40.3, 30.2), at_pixel(r, 0, 40.6, 30.7)])
        fp2, rows2 = rows_for(x2, r, mask)
        self.assertTrue(bool((rows2.m_hat >= 0).all() and (rows2.m_hat <= 1).all()))
        self.assertGreater(int((rows2.contributors > 1).sum()), 0)

    def test_occluded_point_still_renders(self):
        from estimator.projective_visibility import visibility, position_sigma_z2
        r = rig(); mask = torch.zeros((1, H, W), dtype=torch.bool)
        x = torch.stack([at_pixel(r, 0, 40.2, 30.3), at_pixel(r, 0, 40.2, 30.3, Z=2.)])
        vis, _ = visibility(r.project(x, jacobian=False), position_sigma_z2(r.R, torch.eye(3, dtype=torch.float64).expand(2, 3, 3)*1e-8), H, W, r.near, 3.)
        self.assertEqual(int(vis[0, 1]), 6)                                        # SELF_OCCLUDED for direct RGB-D ...
        fp, rows = rows_for(x, r, mask)
        self.assertTrue(bool((rows.pair_point == 1).any()))                        # ... and still a white contributor

    def test_missing_foreground_white_gives_no_correction(self):
        r = rig(); mask = torch.zeros((1, H, W), dtype=torch.bool); mask[0, 10:60, 10:80] = True
        x = at_pixel(r, 0, 30.4, 20.6)[None]                                        # a lone point inside a large white region
        fp, rows = rows_for(x, r, mask)
        self.assertEqual(rows.n_rows, 0); self.assertEqual(float(rows.jtw(rows.r, 1).abs().sum()), 0.)

    def test_behind_or_outside_camera_is_not_black_padding(self):
        r = rig(2); mask = torch.zeros((2, H, W), dtype=torch.bool)
        x = at_pixel(r, 1, 50.3, 40.4)                                              # visible in camera 1
        u0 = r.project(x[None], jacobian=False).uv[0, 0]
        fp, rows = rows_for(x[None], r, mask)
        in0 = bool(fp.in_image[0, 0])
        self.assertTrue(bool(fp.in_image[1, 0]))
        self.assertTrue(set(rows.row_cam.tolist()) <= ({0, 1} if in0 else {1}))     # no row from a camera that cannot see it
        far = torch.tensor([[0., 0., -5.]], dtype=torch.float64)                    # behind camera 0
        fpf, rowsf = rows_for(far, r, mask)
        self.assertFalse(bool(fpf.projected[0, 0])); self.assertFalse(bool((rowsf.row_cam == 0).any()))


class JacobianTest(unittest.TestCase):
    def dense_image(self, x, r):
        fp, rows = rows_for(x, r, torch.zeros((r.count, H, W), dtype=torch.bool))  # all background: every touched pixel is a row
        img = torch.zeros(r.count*H*W, dtype=torch.float64); img[rows.row_cam*H*W+rows.row_pix] = rows.m_hat
        return img, rows

    def test_analytic_matches_finite_differences_with_overlap(self):
        g = np.random.default_rng(0)
        for ncam in (1, 2, 3):
            r = rig(ncam)
            base = at_pixel(r, 0, 40.37, 30.61)
            x = torch.stack([base+torch.as_tensor(g.normal(scale=.002, size=3)) for _ in range(6)])   # several points share pixels
            img, rows = self.dense_image(x, r)
            self.assertGreater(int((rows.contributors > 1).sum()), 0)
            dense = torch.zeros((r.count*H*W, len(x), 3), dtype=torch.float64)
            dense[(rows.row_cam*H*W+rows.row_pix)[rows.pair_row], rows.pair_point] = rows.g
            e = 1e-7
            for i in range(len(x)):
                for k in range(3):
                    d = torch.zeros_like(x); d[i, k] = e
                    fd = (self.dense_image(x+d, r)[0]-self.dense_image(x-d, r)[0])/(2*e)
                    np.testing.assert_allclose(fd.numpy(), dense[:, i, k].numpy(), atol=1e-5, rtol=1e-5)

    def test_saturation_and_coincident_points_are_finite_and_deterministic(self):
        r = rig()
        p = at_pixel(r, 0, 40., 30.)                                                 # exact pixel centre: a = 1
        for x in (p[None], torch.stack([p, p])):
            img, rows = self.dense_image(x, r)
            self.assertTrue(bool(torch.isfinite(rows.g).all() and torch.isfinite(rows.m_hat).all()))
            i = int(torch.nonzero(rows.row_pix == 30*W+40).flatten()[0])
            self.assertEqual(float(rows.m_hat[i]), 1.)
            img2, rows2 = self.dense_image(x, r)
            self.assertTrue(torch.equal(rows.g, rows2.g))
        # two coincident points at a = 1: each one's removal leaves the pixel white -> zero derivative on that pixel
        img, rows = self.dense_image(torch.stack([p, p]), r)
        on = rows.pair_row == int(torch.nonzero(rows.row_pix == 30*W+40).flatten()[0])
        self.assertEqual(float(rows.g[on].abs().sum()), 0.)

    def test_jv_jtw_are_adjoint(self):
        g = np.random.default_rng(1); r = rig(3)
        x = torch.stack([at_pixel(r, 0, 40.37, 30.61)+torch.as_tensor(g.normal(scale=.003, size=3)) for _ in range(8)])
        _, rows = self.dense_image(x, r)
        v = torch.as_tensor(g.normal(size=(len(x), 3))); w = torch.as_tensor(g.normal(size=rows.n_rows))
        self.assertAlmostEqual(float((rows.jv(v)*w).sum()), float((rows.jtw(w, len(x))*v).sum()), places=12)


if __name__ == '__main__':
    unittest.main()
