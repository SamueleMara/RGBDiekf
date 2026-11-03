"""Material-following residual and analytical Jacobians vs an independent NumPy oracle (spec section 6.A).

Production rows come from build_material_rows/material_prediction; the oracle re-implements the residual
f = sum_a w_a (y_a - eta_a) - (y_ref - eta_ref) - e4 (Z_t - Z_ref) at q = p_ref + pi(x_t) - pi(x_ref) with NumPy
projection and bilinear sampling, and differentiates it by 5-point finite differences in the SAME bilinear cell.
"""
import unittest
import numpy as np
import torch
from tests.material_fixtures import (Scene, rig, noise, POLICY, DEVICES, S, FOC, np_material_f, rgbd, fd)
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_material_terms import build_material_rows, material_prediction

MATERIAL_WARP_WITNESSES = []


def rows_for(device, scene, x_ref, x_t, p_ref, cams=(0, 1)):
    r = rig(device)
    f0, f1 = scene.render(0, device), scene.render(1, device)
    N = len(x_t)
    refs = ReferenceTable(2, N, device, torch.float64)
    for c in cams:
        refs.frame[c] = 0
        refs.ordinal[c] = 0
        refs.pixel[c] = torch.as_tensor(p_ref[c], device=device)
        refs.state[c, :, :3] = torch.as_tensor(x_ref, device=device)
    s = torch.zeros((N, 6), dtype=torch.float64, device=device)
    s[:, :3] = torch.as_tensor(x_t, device=device)
    own = first_hit_pixel_owner(s[:, :3], r, f1, dense=False)
    rows, reason, life = build_material_rows(s, own, f1, refs, {0: f0, 1: f1}, r, noise(device), POLICY, 1, 3)
    return rows, reason, f0, f1, r


class MaterialWarpTests(unittest.TestCase):
    def cases(self):
        """(true per-frame material motion, state offset) incl. +-1.3 and +-3 px, depth motion, source projection offset."""
        px = 2./FOC   # one pixel at Z=2
        return [((1.3*px, 0., 0.), (0., 0., 0.)), ((-1.3*px, .4*px, 0.), (0., 0., 0.)), ((3*px, -1.1*px, 0.), (.2*px, 0., 0.)),
                ((-3*px, 0., .02), (0., .3*px, .01)), ((.7*px, .7*px, -.03), (0., 0., 0.))]

    def test_residual_and_state_jacobians_match_oracle(self):
        for device in DEVICES:
            for d, off in self.cases():
                with self.subTest(device=device, d=d):
                    sc = Scene(d=d)
                    # two points with different subpixel source offsets: p_ref is the camera's nearest pixel of x_ref
                    pix0 = [15*S+14, 17*S+18]
                    x_ref = np.array([sc.point(0, p) for p in pix0])+np.array([[.004, -.003, 0.], [-.002, .005, .001]])*.25
                    x_t = x_ref+np.asarray(d)+np.asarray(off)
                    p_ref = {}
                    for c in (0, 1):
                        uv = [np.rint(np.array([FOC*(x[0]+sc.t[c][0])/(x[2]+sc.t[c][2])+15.5, FOC*(x[1]+sc.t[c][1])/(x[2]+sc.t[c][2])+15.5])) for x in x_ref]
                        p_ref[c] = [int(u[1])*S+int(u[0]) for u in uv]
                    rows, reason, f0, f1, r = rows_for(device, sc, x_ref, x_t, p_ref)
                    self.assertGreater(len(rows), 0)
                    clean_ref = rows.reference_values.clone()
                    f, Ht, Hr, w = material_prediction(rows, rows.current_values, clean_ref)
                    for a in range(len(rows)):
                        i, c = int(rows.point[a]), int(rows.camera[a])
                        cell = rows.cell[a].cpu().numpy()
                        cur, ref = rgbd(f1, c), rgbd(f0, c)
                        pr = int(rows.reference_pixel[a])
                        fo, q, wo = np_material_f(x_t[i], x_ref[i], sc.t[c], pr, cur, ref, cell)
                        np.testing.assert_allclose(f[a].cpu().numpy(), fo, atol=1e-12)
                        np.testing.assert_allclose(rows.q[a].cpu().numpy(), q, atol=1e-12)
                        np.testing.assert_allclose(w[a].cpu().numpy(), wo, atol=1e-12)
                        Jt = fd(lambda x: np_material_f(x, x_ref[i], sc.t[c], pr, cur, ref, cell)[0], x_t[i])
                        Jr = fd(lambda x: np_material_f(x_t[i], x, sc.t[c], pr, cur, ref, cell)[0], x_ref[i])
                        np.testing.assert_allclose(Ht[a, :, :3].cpu().numpy(), Jt, rtol=1e-6, atol=1e-7)
                        np.testing.assert_allclose(Hr[a, :, :3].cpu().numpy(), Jr, rtol=1e-6, atol=1e-7)
                        self.assertEqual(float(Ht[a, :, 3:].abs().max()), 0.)    # instantaneous residual: no velocity
                        self.assertEqual(float(Hr[a, :, 3:].abs().max()), 0.)
                        MATERIAL_WARP_WITNESSES.append({'device': device, 'd': list(d), 'camera': c, 'H_t_fd_max_abs': float(np.abs(Ht[a, :, :3].cpu().numpy()-Jt).max()),
                                                        'H_ref_fd_max_abs': float(np.abs(Hr[a, :, :3].cpu().numpy()-Jr).max()),
                                                        'pixel_shift': (rows.q[a].cpu().numpy()-np.array([pr % S, pr//S])).tolist()})

    def test_raw_error_jacobians(self):
        sc = Scene(d=(1.3*2/FOC, 0., .01))
        x_ref = np.array([sc.point(0, 16*S+16)])
        x_t = x_ref+sc.d
        rows, _, f0, f1, r = rows_for('cpu', sc, x_ref, x_t, {0: [16*S+16], 1: [16*S+24]})
        a = 0
        cell, c, pr = rows.cell[a].numpy(), int(rows.camera[a]), int(rows.reference_pixel[a])
        cur, ref = rgbd(f1, c), rgbd(f0, c)
        Jref = fd(lambda e: np_material_f(x_t[0], x_ref[0], sc.t[c], pr, cur, ref, cell, eta_ref=e)[0], np.zeros(4))
        np.testing.assert_allclose(Jref, np.eye(4), atol=1e-9)
        w = rows_w = material_prediction(rows, rows.current_values, rows.reference_values)[3][a].numpy()
        for j in range(4):
            def fun(e, j=j):
                eta = np.zeros((4, 4))
                eta[j] = e
                return np_material_f(x_t[0], x_ref[0], sc.t[c], pr, cur, ref, cell, eta_cur=eta)[0]
            np.testing.assert_allclose(fd(fun, np.zeros(4)), -rows_w[j]*np.eye(4), atol=1e-9)
        self.assertAlmostEqual(float(w.sum()), 1., places=12)

    def test_cell_transition_is_continuous_and_jacobian_piecewise(self):
        """Crossing a bilinear cell boundary: the residual is continuous, the cell (and raw support) changes, the
        Jacobian is that of the new piece, and the production builder always uses floor(q)."""
        px = 2./FOC
        sc = Scene(d=(0., 0., 0.))
        x_ref = np.array([sc.point(0, 16*S+16)])
        out = []
        for eps in (-1e-7, 1e-7):
            x_t = x_ref+np.array([[x_ref[0, 2]/FOC*(1+eps), 0., 0.]])     # q_u = 17 -/+ 1e-7: integer cell boundary
            rows, _, f0, f1, r = rows_for('cpu', sc, x_ref, x_t, {0: [16*S+16], 1: [16*S+20]}, cams=(0,))
            f, Ht, _, _ = material_prediction(rows, rows.current_values, rows.reference_values)
            out.append((rows.cell[0].numpy().copy(), f[0].numpy(), Ht[0, :, :3].numpy(), rows.q[0].numpy()))
        (c1, f1_, H1, q1), (c2, f2_, H2, q2) = out
        self.assertTrue(np.all(np.floor(q1) == c1) and np.all(np.floor(q2) == c2))
        self.assertEqual(int(c1[0]), 16)
        self.assertEqual(int(c2[0]), 17)
        np.testing.assert_allclose(f1_, f2_, atol=1e-6)
        MATERIAL_WARP_WITNESSES.append({'cell_transition': [c1.tolist(), c2.tolist()], 'residual_jump': float(np.abs(f1_-f2_).max()),
                                        'jacobian_jump': float(np.abs(H1-H2).max())})


if __name__ == '__main__':
    unittest.main()
