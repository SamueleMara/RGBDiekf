"""Gate A (measurement): measured-image sampling against an independent oracle, the complete 4x6 H against finite
differences, the sign r = -h, and strict validity (mask, invalid depth, FOV, behind camera, depth breaks, duplicate
pixels, self-occlusion, external occlusion, a missing camera), non-finite neighbours, and the row innovation gate on
the prior-linearized innovation (audit 2026-10-28 items 4, 5, 7)."""
import unittest
import numpy as np
import torch
from tests.projective_fixtures import camera, rot, smooth_field, noise, POLICY, DEVICES, FieldDataset
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, sample, sample_cpu_oracle, CODE
from estimator.projective_terms import build_terms, h_value, row_gate_threshold
from estimator.projective_visibility import visibility, position_sigma_z2


def on_surface(rig, uv, field, cam=0):
    """World points ON the measured surface of camera `cam` at pixel coordinates uv (bilinear depth of the field)."""
    rgb, depth, mask = field
    z = [sample_cpu_oracle(rgb, depth, mask, u, v)['depth'] for u, v in uv]
    return rig.backproject(torch.full((len(uv),), cam, dtype=torch.long, device=rig.device), torch.as_tensor(np.asarray(uv, float), device=rig.device),
                           torch.as_tensor(np.asarray(z), device=rig.device))


def setup(dev, field=None, cams=None):
    cams = cams or [camera('a', np.eye(3), np.zeros(3))]
    field = field or smooth_field()
    ds = FieldDataset(cams, [[field]*len(cams)])
    return ds, CameraRig(cams, dev), MeasuredFrame(ds, 0, dev, torch.float64)


class SamplingTest(unittest.TestCase):
    def test_values_and_gradients_equal_cpu_oracle(self):
        rgb, depth, mask = smooth_field()
        g = np.random.default_rng(1)
        uv = np.c_[g.uniform(0, 94.99, 300), g.uniform(0, 70.99, 300)]
        for dev in DEVICES:
            _, rig, frame = setup(dev)
            s = sample(frame, torch.zeros(300, dtype=torch.long, device=dev), torch.as_tensor(uv, device=dev), noise(dev), POLICY, rig.K)
            for i in range(300):
                o = sample_cpu_oracle(rgb, depth, mask, *uv[i])
                self.assertEqual(int(s.reason[i]), 0)
                np.testing.assert_allclose(s.rgb[i].cpu().numpy(), o['rgb'], atol=1e-12)
                np.testing.assert_allclose(float(s.depth[i]), o['depth'], atol=1e-12)
                np.testing.assert_allclose(s.grad_rgb[i, :, 0].cpu().numpy(), o['rgb_du'], atol=1e-12)
                np.testing.assert_allclose(s.grad_rgb[i, :, 1].cpu().numpy(), o['rgb_dv'], atol=1e-12)
                np.testing.assert_allclose(s.grad_depth[i].cpu().numpy(), [o['depth_du'], o['depth_dv']], atol=1e-12)

    def test_invalid_neighbourhoods_never_sampled(self):
        rgb, depth, mask = smooth_field()
        depth = depth.copy(); mask = mask.copy()
        depth[30, 40] = 0.                      # a no-return hole
        mask[10:20, 10:20] = False              # outside the hard foreground
        depth[50:, 60:] += .5                   # a 50 cm depth break
        _, rig, frame = setup('cpu', (rgb, depth, mask))
        uv = torch.tensor([[40.5, 29.5], [12.3, 12.7], [59.5, 52.2], [95.5, 3.], [-.2, 4.], [30.2, 40.6]], dtype=torch.float64)
        s = sample(frame, torch.zeros(6, dtype=torch.long), uv, noise(), POLICY, rig.K)
        self.assertEqual([int(r) for r in s.reason], [CODE['INVALID_DEPTH'], CODE['MASK_EXCLUDED'], CODE['DEPTH_DISCONTINUITY'],
                                                      CODE['OUTSIDE_IMAGE_FOOTPRINT'], CODE['OUTSIDE_IMAGE_FOOTPRINT'], CODE['VALID']])


class JacobianTest(unittest.TestCase):
    def test_complete_H_against_finite_differences_and_sign(self):
        cams = [camera('a', rot([.1, .2, .3], .2), np.array([.05, -.02, .1])), camera('b', rot([.12, .2, .3], .215), np.array([.07, -.01, .11]), (505., 492., 46.2, 37.9))]
        for dev in DEVICES:
            ds, rig, frame = setup(dev, cams=cams)
            g = np.random.default_rng(2)
            # points on the observed surface of camera a, slightly displaced
            uv = np.c_[g.uniform(25, 70, 40), g.uniform(20, 50, 40)]
            x0 = on_surface(rig, uv, smooth_field())
            x0 = x0+torch.as_tensor(g.normal(0, .003, (40, 3)), device=dev)
            rgb0 = torch.as_tensor(g.uniform(.2, .8, (40, 3)), device=dev)
            # arbitrary colours give non-zero residuals for r = -h; they are inconsistent with the image by construction,
            # so this test of H (not of row admission) runs with the occlusion-only rule
            pol = {**POLICY, 'row_innovation_gate': 'depth_front_only'}
            for c in range(2):
                cam = torch.full((40,), c, device=dev)
                t = build_terms(rig, frame, cam, torch.arange(40, device=dev), x0, rgb0, torch.eye(3, dtype=torch.float64, device=dev).expand(40, 3, 3),
                                noise(dev), pol)
                h, ok = h_value(rig, frame, cam, x0, rgb0, noise(dev), POLICY)
                use = t.valid() & ok
                self.assertGreater(int(use.sum()), 20)
                np.testing.assert_allclose(t.r[use].cpu().numpy(), -h[use].cpu().numpy(), atol=1e-14)    # r = -h
                eps = 1e-7
                for a in range(3):
                    d = torch.zeros(3, dtype=torch.float64, device=dev); d[a] = eps
                    hp, okp = h_value(rig, frame, cam, x0+d, rgb0, noise(dev), POLICY); hm, okm = h_value(rig, frame, cam, x0-d, rgb0, noise(dev), POLICY)
                    fd = ((hp-hm)/(2*eps))[use]
                    # exact inside a bilinear cell; rows whose +-eps step crosses a cell border are excluded
                    same = (torch.floor(rig.project_rows(x0+d, cam)[0]) == torch.floor(rig.project_rows(x0-d, cam)[0])).all(1)[use]
                    err = (fd-t.Hx[use][:, :, a]).abs()[same]
                    self.assertLess(float(err.max()), 1e-5*max(1., float(t.Hx[use][:, :, a].abs().max())))

    def test_textureless_plane_gives_rank_one_depth_information(self):
        field = smooth_field(texture=False)
        _, rig, frame = setup('cpu', field)
        x = on_surface(rig, [[30.3, 30.2], [50.1, 20.7], [60.6, 40.4], [20.2, 50.5], [70.7, 33.3]], field)
        t = build_terms(rig, frame, torch.zeros(5, dtype=torch.long), torch.arange(5), x, torch.full((5, 3), .5, dtype=torch.float64),
                        torch.eye(3, dtype=torch.float64).expand(5, 3, 3)*1e-4, noise(), POLICY)
        self.assertTrue(bool(t.valid().all()))
        self.assertEqual(float(t.Hx[:, :3].abs().max()), 0.)                    # no RGB gradient -> no RGB information
        I = torch.einsum('mi,mj->mij', t.Hx[:, 3], t.Hx[:, 3])
        ev = torch.linalg.eigvalsh(I)
        self.assertTrue(bool((ev[:, :2].abs() < 1e-12*ev[:, 2:3]).all()))         # rank one: two lateral directions unconstrained


class VisibilityTest(unittest.TestCase):
    def test_duplicates_occlusion_and_external_occluder(self):
        _, rig, frame = setup('cpu')
        b = lambda u, v, z: rig.backproject(torch.zeros(1, dtype=torch.long), torch.tensor([[u, v]], dtype=torch.float64), torch.tensor([z]))[0]
        x = torch.stack([b(40.2, 30.1, 2.0), b(40.1, 29.9, 2.0), b(40.0, 30.0, 2.6), b(20.4, 20.3, 1.0), b(70.2, 50.4, 9.0)])
        Sx = torch.eye(3, dtype=torch.float64).expand(5, 3, 3)*1e-6
        proj = rig.project(x)
        reason, front = visibility(proj, position_sigma_z2(rig.R, Sx), frame.H, frame.W, rig.near, POLICY['visibility_n_sigma'])
        self.assertEqual([int(r) for r in reason[0]], [CODE['VALID'], CODE['DUPLICATE_PIXEL_SUPPRESSED'], CODE['SELF_OCCLUDED'], CODE['VALID'], CODE['VALID']])
        t = build_terms(rig, frame, torch.zeros(5, dtype=torch.long), torch.arange(5), x, torch.full((5, 3), .5, dtype=torch.float64), Sx, noise(), POLICY)
        # point 4 is 7 m behind the measured surface (~2 m): the measured surface is in front -> no update, reported
        self.assertEqual(int(t.reason[4]), CODE['MEASURED_SURFACE_IN_FRONT'])
        # point 3 floats 1 m in front of the surface: not an occlusion, and 1 m is ~1000 sigma of its prior: the sampled
        # pixel is not its surface (audit item 4: the earlier one-sided rule admitted it as "motion")
        self.assertEqual(int(t.reason[3]), CODE['INNOVATION_INCONSISTENT'])
        legacy = build_terms(rig, frame, torch.zeros(5, dtype=torch.long), torch.arange(5), x, torch.full((5, 3), .5, dtype=torch.float64), Sx, noise(),
                             {**POLICY, 'row_innovation_gate': 'depth_front_only'})
        self.assertEqual(int(legacy.reason[3]), CODE['VALID'])
        self.assertEqual(int(legacy.reason[4]), CODE['MEASURED_SURFACE_IN_FRONT'])

    def test_missing_camera_gives_no_rows(self):
        rgb, depth, mask = smooth_field()
        cams = [camera('a', np.eye(3), np.zeros(3)), camera('b', np.eye(3), np.zeros(3))]
        ds = FieldDataset(cams, [[(rgb, depth, mask), (rgb, np.zeros_like(depth), np.zeros_like(mask))]])
        rig, frame = CameraRig(cams, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        uv = [[30.2, 30.3], [50.4, 40.1], [60.3, 20.6]]
        x = on_surface(rig, uv, (rgb, depth, mask))
        rgb0 = torch.as_tensor(np.array([sample_cpu_oracle(rgb, depth, mask, u, v)['rgb'] for u, v in uv]))     # the measured colour of the point
        for c, expect in ((0, CODE['VALID']), (1, CODE['MASK_EXCLUDED'])):
            t = build_terms(rig, frame, torch.full((3,), c), torch.arange(3), x, rgb0,
                            torch.eye(3, dtype=torch.float64).expand(3, 3, 3)*1e-6, noise(), POLICY)
            self.assertTrue(bool((t.reason == expect).all()))


class RowGateTest(unittest.TestCase):
    """Row admission on nu = r + H dx, S = H Sigma H^T + R (projective_terms): both depth signs, colour, the iterate."""
    def rows(self, dz, drgb=0., sig=1e-3, dx=None, gate='rgbd_chi2'):
        field = smooth_field()
        _, rig, frame = setup('cpu', field)
        uv = [[30.3, 30.2], [50.1, 20.7], [60.6, 40.4]]
        x = on_surface(rig, uv, field)
        ray = x/x.norm(dim=1, keepdim=True)                     # camera at the origin: moving along the ray changes Z only
        rgb0 = torch.as_tensor(np.array([sample_cpu_oracle(*field, u, v)['rgb'] for u, v in uv]))+drgb
        Sx = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)*sig**2
        return build_terms(rig, frame, torch.zeros(3, dtype=torch.long), torch.arange(3), x+dz*ray, rgb0, Sx, noise(), {**POLICY, 'row_innovation_gate': gate},
                           dx=None if dx is None else dx*ray)

    def test_consistent_rows_pass_and_both_depth_signs_are_gated(self):
        self.assertTrue(bool(self.rows(0.).valid().all()))
        self.assertTrue(bool((self.rows(-.05).reason == CODE['INNOVATION_INCONSISTENT']).all()))      # floats 5 cm in front of its surface
        self.assertTrue(bool((self.rows(+.05).reason == CODE['MEASURED_SURFACE_IN_FRONT']).all()))    # behind the measured surface
        self.assertTrue(bool((self.rows(-.05, gate='depth_two_sided').reason == CODE['INNOVATION_INCONSISTENT']).all()))
        self.assertTrue(bool(self.rows(-.05, gate='depth_front_only').valid().all()))                    # the earlier rule

    def test_colour_identity_and_prior_uncertainty(self):
        self.assertTrue(bool((self.rows(0., drgb=.3).reason == CODE['INNOVATION_INCONSISTENT']).all()))   # same depth, other colour
        self.assertTrue(bool(self.rows(0., drgb=.3, gate='depth_two_sided').valid().all()))
        self.assertTrue(bool(self.rows(-.05, sig=.05).valid().all()))      # 5 cm is 1 sigma of a 5 cm prior: a legitimate motion passes
        t = self.rows(0.)
        self.assertTrue(bool((t.nis < row_gate_threshold(POLICY)).all()))

    def test_gate_uses_the_prior_linearized_innovation(self):
        # an iterate 5 cm off the prior along the ray whose sample matches: r = 0 there, but nu = r + H dx is the 5 cm the
        # update had to explain, which a 1 mm prior does not allow
        self.assertTrue(bool(self.rows(0.).valid().all()))
        self.assertTrue(bool((self.rows(0., dx=.05).reason != CODE['VALID']).all()))
        self.assertTrue(bool(self.rows(0., dx=.05, sig=.05).valid().all()))


class NonFiniteTest(unittest.TestCase):
    """Audit item 7: NaN / inf pixels (raw invalid or not) never reach a VALID row and never poison the sums."""
    def test_nan_neighbours_are_invalid_and_rows_are_finite(self):
        rgb, depth, mask = smooth_field()
        depth = depth.copy(); rgb = rgb.copy()
        depth[30, 30] = np.nan; depth[40, 50] = np.inf; rgb[20, 60, 1] = np.nan
        ds = FieldDataset([camera('a', np.eye(3), np.zeros(3))], [[(rgb, depth, mask)]])
        rig, frame = CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        frame.raw_valid[:] = True                                   # a dataset whose validity map does not flag them
        uv = torch.tensor([[29.6, 29.7], [49.5, 39.4], [59.5, 19.5], [70.2, 50.3]], dtype=torch.float64)
        s = sample(frame, torch.zeros(4, dtype=torch.long), uv, noise(), POLICY, rig.K)
        self.assertEqual([int(r) for r in s.reason[:3]], [CODE['INVALID_DEPTH']]*3)
        self.assertEqual(int(s.reason[3]), CODE['VALID'])
        for v in (s.rgb, s.depth, s.grad_rgb, s.grad_depth, s.var_rgb, s.var_depth):
            self.assertTrue(bool(torch.isfinite(v).all()))


if __name__ == '__main__':
    unittest.main()
