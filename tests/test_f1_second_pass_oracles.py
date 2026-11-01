"""Deterministic checks of the second-pass diagnostics (tools/f1_second_pass.py; RGBDiekf_Inner_F1_Second_Pass_Causal_Debug_Codex.md):
frame-0 vs frame-1 residual algebra at a fixed point, the 6D solve / Schur / amplifier identities used in the decomposition,
support bookkeeping that reports every rejection reason (no priority collapse), and the frozen-support objective comparison."""
import unittest
import numpy as np
from tools import root_cause_oracle as O
from tools.f1_second_pass import all_reasons, FrozenProblem
from tests.projective_fixtures import camera, POLICY


class Img:
    def __init__(self, depth, rgb):
        self.depth, self.rgb = depth, rgb
        self.mask = np.ones(depth.shape, bool); self.raw_valid = np.ones(depth.shape, bool)


def scene(seed=0):
    g = np.random.default_rng(seed)
    cam = O.Cam(camera('c0', K=(400., 400., 47.5, 35.5), W=96, H=72))
    D = np.full((72, 96), 1.5)+1e-3*np.arange(96)[None]                      # gently slanted surface
    I = .5+.2*np.sin(np.arange(96)[None, :, None]/7.+np.arange(72)[:, None, None]/5.+np.arange(3))
    return cam, Img(D, I), g


def noise():
    return O.Noise([1e-5]*3, 1e-8, 0., 0., 0., 'sum_w2')


class Frame0Frame1Test(unittest.TestCase):
    def test_delta_residual_is_the_image_change_and_source_colour_cancels(self):
        cam, im0, g = scene()
        im1 = Img(im0.depth.copy(), im0.rgb.copy())
        im1.rgb[30:40, 40:50] += .01; im1.depth[30:40, 40:50] += 2e-4                 # a temporal change of the imaged surface
        x = np.array([0.0, 0.0, 1.55]); a = np.array([.3, .4, .5])
        r0 = all_reasons(im0, cam, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3))
        r1 = all_reasons(im1, cam, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3))
        d = np.array(r1['r'])-np.array(r0['r'])
        w = np.array([(1-r0['du'])*(1-r0['dv']), r0['du']*(1-r0['dv']), (1-r0['du'])*r0['dv'], r0['du']*r0['dv']])
        I0, I1 = np.array(r0['rgb4']), np.array(r1['rgb4']); D0, D1 = np.array(r0['depth4']), np.array(r1['depth4'])
        np.testing.assert_allclose(d[:3], w@(I0-I1), atol=1e-15)                   # Delta r_RGB = I_0 - I_1 (a cancels)
        np.testing.assert_allclose(d[3], w@(D0-D1), atol=1e-15)                    # Delta r_D = D_0 - D_1 (Z cancels)
        r1b = all_reasons(im1, cam, x, a+.1, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3))
        r0b = all_reasons(im0, cam, x, a+.1, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3))
        np.testing.assert_allclose(np.array(r1b['r'])-np.array(r0b['r']), d, atol=1e-15)


class SolveIdentityTest(unittest.TestCase):
    def test_schur_equality_and_amplifier(self):
        g = np.random.default_rng(1)
        for _ in range(20):
            M = g.normal(size=(6, 6)); P = M@M.T+.1*np.eye(6)
            H = g.normal(size=(8, 3)); W = np.diag(g.uniform(.5, 2, 8)); r = g.normal(size=8)
            Fm = H.T@W@H; gx = H.T@W@r
            sm = g.normal(size=6); s = sm+g.normal(scale=.2, size=6)
            A = np.linalg.inv(P); A[:3, :3] += Fm
            b = -np.linalg.solve(P, s-sm); b[:3] += gx
            delta = np.linalg.solve(A, b); e = s-sm
            geff = H.T@W@(r+H@e[:3])
            Eff = np.linalg.inv(P[:3, :3])+Fm
            np.testing.assert_allclose(np.linalg.inv(A)[:3, :3], np.linalg.inv(Eff), rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(np.linalg.solve(Eff, geff)-delta[:3], e[:3], rtol=1e-9, atol=1e-11)
            # linear update from the prior: dv = P_vx P_xx^-1 dx exactly for position-only rows
            A0 = np.linalg.inv(P); A0[:3, :3] += Fm; b0 = np.zeros(6); b0[:3] = gx
            d0 = np.linalg.solve(A0, b0)
            np.testing.assert_allclose(d0[3:], P[3:, :3]@np.linalg.solve(P[:3, :3], d0[:3]), rtol=1e-8, atol=1e-12)


class ReasonBookkeepingTest(unittest.TestCase):
    def test_every_reason_is_reported_without_priority_collapse(self):
        cam, im, g = scene()
        x = np.array([0.0, 0.0, 1.55]); a = np.array([.3, .4, .5])
        u, v, Z, _, _ = cam.project(x); u0, v0 = int(np.floor(u)), int(np.floor(v))
        im.mask[v0, u0] = False; im.depth[v0+1, u0+1] += .5                          # masked pixel AND a depth break in the same cell
        r = all_reasons(im, cam, x, a, noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), frozen_code=O.DUP)
        self.assertTrue(r['mask_excluded']); self.assertTrue(r['discontinuity']); self.assertTrue(r['frozen_duplicate'])
        self.assertFalse(r['invalid_depth'])


class FrozenSupportTest(unittest.TestCase):
    def test_frozen_rows_signal_invalid_support_and_common_cost_lists_lost_rows(self):
        cam, im, g = scene()
        im.mask[:, 60:] = False                                                      # the frozen camera loses the point beyond u = 60
        x = np.array([0.0, 0.0, 1.55]); a = np.array([.3, .4, .5])
        s = np.r_[x, 0, 0, 0]; P = np.diag([1e-4]*3+[1.]*3)
        far = s.copy(); far[0] += .05                                                 # projects beyond the mask edge
        fp = FrozenProblem([cam], [im], a, s, P, [O.VALID], noise(), POLICY, 10, 1e-8, None, frozen_rows=[0])
        fp.rows(far[:3]); self.assertTrue(fp.invalid_events)                          # signalled, not silently dropped
        prob = O.Problem([cam], [im], a, s, P, [O.VALID], noise(), POLICY, 10, 1e-8)
        c = prob.cost_on(far, [0]); self.assertTrue(c['lost_rows'])                   # 'not evaluable', never a zero cost
        c0 = prob.cost_on(s, [0]); self.assertFalse(c0['lost_rows'])


if __name__ == '__main__':
    unittest.main()
