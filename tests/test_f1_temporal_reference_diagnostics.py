"""T0-T8 of RGBDiekf_F1_Temporal_Reference_Depth_Surface_Reacquisition_Debug_Codex.md section 10 (CPU, deterministic, no writes).
The reference-differenced residual r^Delta = r - b_c0 is an OFFLINE diagnostic (tools/f1_temporal_reference_diagnostics.py)."""
import math
import types
import unittest
from pathlib import Path
import numpy as np
from tools import root_cause_oracle as O
from tools.f1_temporal_reference_diagnostics import ref_row, references, RefProblem, CASES
from tools.f1_second_pass import ShiftCam
from tests.projective_fixtures import camera, POLICY

FOR = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030')


class Img:
    def __init__(self, depth, rgb):
        self.depth, self.rgb = depth, rgb
        self.mask = np.ones(depth.shape, bool); self.raw_valid = np.ones(depth.shape, bool)


def noise():
    return O.Noise([1e-5]*3, 1e-8, 0., 0., 0., 'sum_w2')


def textured(offset=np.zeros(3), shift_px=0.):
    D = np.full((72, 96), 1.5)+1e-3*np.arange(96)[None]
    uu, vv = np.meshgrid(np.arange(96)-shift_px, np.arange(72))
    I = np.stack([.5+.2*np.sin(uu/7.+vv/5.+c) for c in range(3)], -1)+offset
    return Img(D, I)


def cam():
    return O.Cam(camera('c0', K=(400., 400., 47.5, 35.5), W=96, H=72))


class T0Cancellation(unittest.TestCase):
    def test_constant_camera_offset_cancels_and_jacobian_is_unchanged(self):
        c = cam(); x = np.array([0.0131, -0.0077, 1.55]); a = np.array([.4, .5, .6]); off = np.array([.03, -.02, .01])
        im0, imt = textured(off), textured(off)                                 # static scene, constant per-camera colour offset
        r0 = O.row(im0, c, x, a, noise(), {**POLICY, 'visibility_n_sigma': float('inf')}, np.eye(3)*1e-4, np.zeros(3))
        ref = {'valid': True, 'b': r0['r'].copy(), 'var0': r0['var'].copy()}
        o = ref_row(imt, c, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), ref, 'RGBD')
        np.testing.assert_allclose(o['r_original'], r0['r'], atol=0)          # r_orig = b_c0
        np.testing.assert_allclose(o['r'], 0, atol=1e-15)                       # r^Delta = 0
        np.testing.assert_allclose(o['H'], r0['H'], atol=0)                     # the reference does not change the Jacobian
        e = 1e-7; fd = np.zeros((4, 3))                                         # production stores H = dh/dx with h = -r
        for i in range(3):
            d = np.zeros(3); d[i] = e
            rp = ref_row(imt, c, x+d, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), ref, 'RGBD')['r']
            rm = ref_row(imt, c, x-d, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), ref, 'RGBD')['r']
            fd[:, i] = -(rp-rm)/(2*e)
        u, v = o['u'], o['v']; self.assertGreater(min(u-math.floor(u), math.ceil(u)-u, v-math.floor(v), math.ceil(v)-v), 1e-3)   # away from cell borders
        np.testing.assert_allclose(fd, o['H'], rtol=1e-5, atol=1e-6)


class T1MovingSignal(unittest.TestCase):
    def test_textured_motion_is_seen_textureless_tangent_is_not(self):
        c = cam(); a = np.array([.4, .5, .6]); x_ref = np.array([0.0131, -0.0077, 1.55])
        z = 1.55; px = 2.0; dx_world = px*z/400.                                 # image shift of 2 px = tangential motion of the surface
        x_true = x_ref+np.array([dx_world, 0, 0])
        im0 = textured(); imt = textured(shift_px=px)                            # material texture moved with the surface (in-plane, depth plane unchanged in u)
        imt.depth = np.full((72, 96), 1.5)+1e-3*(np.arange(96)[None]-px)        # depth field moves with the surface
        r0 = O.row(im0, c, x_ref, a, noise(), {**POLICY, 'visibility_n_sigma': float('inf')}, np.eye(3)*1e-4, np.zeros(3))
        ref = {'valid': True, 'b': r0['r'].copy(), 'var0': r0['var'].copy()}
        at = lambda x, im: ref_row(im, c, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), ref, 'RGBD', gated=False)
        chi = lambda o: float((o['r']**2/o['var']).sum())
        moving, static = chi(at(x_true, imt)), chi(at(x_ref, imt))
        self.assertLess(moving, .01*static)                                     # textured: the moving position explains the differenced data
        flat0, flatt = Img(im0.depth.copy(), np.full((72, 96, 3), .5)), Img(imt.depth.copy(), np.full((72, 96, 3), .5))
        flat0.depth = np.full((72, 96), 1.5); flatt.depth = np.full((72, 96), 1.5)  # textureless fronto plane, tangential motion
        r0f = O.row(flat0, c, x_ref, a, noise(), {**POLICY, 'visibility_n_sigma': float('inf')}, np.eye(3)*1e-4, np.zeros(3))
        reff = {'valid': True, 'b': r0f['r'].copy(), 'var0': r0f['var'].copy()}
        atf = lambda x: ref_row(flatt, c, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), reff, 'RGBD', gated=False)
        self.assertAlmostEqual(chi(atf(x_true)), chi(atf(x_ref)), places=12)    # unobservable: reported, not required
        print(f'\n[T1] textured: chi2_delta(x_true) {moving:.3g} vs chi2_delta(x_ref) {static:.3g}; textureless tangent: equal (unobservable)')


class T2Covariance(unittest.TestCase):
    def test_differential_covariance_and_shared_reference(self):
        g = np.random.default_rng(0)
        for corr in (0., .6):
            d = 2; A = g.normal(size=(3*d, 3*d)); S = A@A.T+.5*np.eye(3*d)
            if corr == 0: S[:d, d:] = S[d:, :d] = 0; S[d:2*d, 2*d:] = S[2*d:, d:2*d] = 0
            R0, Rt, Rs = S[:d, :d], S[d:2*d, d:2*d], S[2*d:, 2*d:]; Ct0, Cs0, Cts = S[d:2*d, :d], S[2*d:, :d], S[d:2*d, 2*d:]
            I, Z = np.eye(d), np.zeros((d, d))
            Lt = np.hstack([-I, I, Z]); Ls = np.hstack([-I, Z, I])
            np.testing.assert_allclose(Lt@S@Lt.T, Rt+R0-Ct0-Ct0.T, atol=1e-12)               # R_Delta = R_t + R_0 - C_t0 - C_0t
            np.testing.assert_allclose(Lt@S@Ls.T, Cts-Ct0-Cs0.T+R0, atol=1e-12)              # shared-reference cross-time term
            L = np.vstack([Lt, Ls]); self.assertGreater(np.linalg.eigvalsh(L@S@L.T).min(), -1e-12)


def stub_scene(img0, src_cam=0):
    c = cam(); x = np.array([0.0131, -0.0077, 1.55]); a = np.array([.4, .5, .6])
    run = types.SimpleNamespace(can={'canonical_rgb': {7: a}})
    src = np.zeros((8, 3), int); src[7] = [0, src_cam, 0]                       # catalogue layout: (frame, camera, pixel) per point
    sc = types.SimpleNamespace(x0={7: x}, src=src, cams=[c], noise=noise(), policy=POLICY, run=run,
                               images=lambda t: [img0], cov0=lambda pid: np.diag([1e-4]*3+[1.]*3))
    return sc


class T3NoDoubleCounting(unittest.TestCase):
    def test_source_reference_is_flagged_and_adds_no_measurement(self):
        c = cam(); x = np.array([0.0131, -0.0077, 1.55])
        im = textured(); u, v, Z, _, _ = c.project(x)
        a = O.sample(im, c, u, v, noise(), POLICY)['rgb']                       # a defined by the source sample at x_ref
        sc = stub_scene(im); sc.run.can['canonical_rgb'] = {7: a}
        refs = references(sc, 7, [O.VALID])
        self.assertTrue(refs[0]['is_source_camera']); self.assertIn('SOURCE_REFERENCE_IS_PRIOR_DEFINITION', refs[0]['source_note'])
        np.testing.assert_allclose(refs[0]['b'][:3], 0, atol=1e-15)
        p = RefProblem([c], [im], a, np.r_[x, 0, 0, 0], np.diag([1e-4]*3+[1.]*3), [O.VALID], noise(), POLICY, 10, 1e-8, refs=refs)
        self.assertEqual(len(p.rows(x)), 1)                                     # the reference is subtracted, never added as a frame-0 row


class T4InvalidReference(unittest.TestCase):
    def test_masked_reference_is_not_zeroed_or_substituted(self):
        c = cam(); x = np.array([0.0131, -0.0077, 1.55]); im = textured()
        u, v, *_ = c.project(x); im.mask[int(v), int(u)] = False
        sc = stub_scene(im, src_cam=1)
        refs = references(sc, 7, [O.VALID])
        self.assertFalse(refs[0]['valid']); self.assertEqual(refs[0]['status'], 'NO_VALID_REFERENCE'); self.assertIn('mask_excluded', refs[0]['invalid_reasons'])
        self.assertIsNone(refs[0]['b'])
        good = textured()
        o = ref_row(good, c, x, sc.run.can['canonical_rgb'][7], noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), refs[0], 'RGBD')
        np.testing.assert_allclose(o['r'], o['r_original'], atol=0)             # original residual kept, flagged
        self.assertIn('NO_VALID_REFERENCE', o['reference_status'])


@unittest.skipUnless((FOR/'storage_45135'/'traced').exists(), 'first-pass traced runs not mounted')
class T5T6Baseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.f1_second_pass import Scene
        cls.sc = {s: Scene(FOR, s) for s in ('storage_45135', 'USB_100109')}

    def test_T5_baseline_replay_reproduces_production(self):
        from tools.f1_temporal_reference_diagnostics import make
        from tools.f1_second_pass import run_prob
        for case in ('storage_8444', 'usb_67827'):
            scene, pid, k, _ = CASES[case]; sc = self.sc[scene]
            p = make(sc, k, pid); r = run_prob(p, sc); fin = sc.trace(k)['final'][str(pid)]
            self.assertLess(float(np.abs(r['s_plus']-np.array(fin['s_plus'])).max()), 1e-9, case)
            self.assertEqual(sorted(o['camera'] for o in p.used(r['rows_accepted'])), sorted({x['camera'] for x in fin['rows']}), case)

    def test_T6_frozen_out_cameras_contribute_nothing(self):
        from tools.f1_temporal_reference_diagnostics import make
        for scene, pid, k, out in (('USB_100109', 67827, 1, {4}), ('storage_45135', 8444, 1, {0, 7}), ('storage_45135', 262894, 1, {3, 4})):
            sc = self.sc[scene]
            for kw in ({}, {'refs': references(sc, pid, sc.frame0_ownership([pid])[pid])}):
                p = make(sc, k, pid, **kw)
                used = {o['camera'] for o in p.used(p.rows(p.s_minus[:3]))}
                self.assertFalse(used & out, (scene, pid, used))


class T7PixelBoundary(unittest.TestCase):
    def test_one_sided_derivatives_differ_and_shifts_select_different_cells(self):
        c = cam(); a = np.array([.4, .5, .6]); im = textured()
        x = c.ray_world(40., 30.)*1.5/(c.R@c.ray_world(40., 30.))[2]+c.centre()   # projects exactly onto the integer pixel (40, 30)
        cells, H = [], []
        for e in (-1e-12, 0., 1e-12):
            o = ref_row(im, ShiftCam(c, e, e), x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), gated=False)
            cells.append(tuple(o['sample']['pixel00'])); H.append(o['H'])
        self.assertNotEqual(cells[0], cells[2])                                 # rounding selects the cell
        self.assertGreater(float(np.abs(H[0]-H[2]).max()), 1e-6)                # one-sided Jacobians differ at the boundary


class T8GateOracle(unittest.TestCase):
    def test_prior_centred_rejection_vs_recentred_oracle_without_state_change(self):
        from tools.f1_second_pass import all_reasons
        c = cam(); a = np.array([.4, .5, .6]); im = textured()
        x_true = np.array([0.0131, -0.0077, 1.55]); u, v, *_ = c.project(x_true)
        a = O.sample(im, c, u, v, noise(), POLICY)['rgb']                       # the visible surface IS the point
        x_pred = x_true+np.array([.03, .02, 0.]); Sigma = np.eye(3)*(4e-3)**2
        snap = x_pred.copy()
        prior_centred = all_reasons(im, c, x_true, a, noise(), POLICY, Sigma, x_true-x_pred)
        recentred = all_reasons(im, c, x_true, a, noise(), POLICY, Sigma, np.zeros(3))
        self.assertTrue(prior_centred['chi2_inconsistent'] or prior_centred['front'])
        self.assertTrue(recentred['valid_production_rules'])
        np.testing.assert_array_equal(x_pred, snap)                             # diagnostic evaluations never alter the state


if __name__ == '__main__':
    unittest.main()
