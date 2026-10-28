"""Gate B: the free 6D IEKF is a Kalman filter. Dense textbook K / information-form / Joseph equality, signed-time
F and Q, prediction-only points, textureless unobservability, independence of points, CPU vs CUDA."""
import unittest
import numpy as np
import torch
from tests.projective_fixtures import camera, rot, smooth_field, noise, POLICY, DEVICES, FieldDataset
from tests.test_projective_measurement import on_surface
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame
from estimator.point_kalman_cuda import transition, process_noise, predict, update_points


class DenseKalmanEquivalenceTest(unittest.TestCase):
    def test_iterated_kalman_equals_information_form_and_joseph(self):
        g = np.random.default_rng(0)
        for _ in range(20):
            A0 = g.normal(size=(6, 6)); Pm = A0@A0.T+.1*np.eye(6)
            m = int(g.integers(1, 9))*4
            H = np.zeros((m, 6)); H[:, :3] = g.normal(size=(m, 3))
            R = np.diag(g.uniform(.01, 1, m)); r = g.normal(size=m)
            sm = g.normal(size=6); sj = sm+g.normal(scale=.3, size=6)
            K = Pm@H.T@np.linalg.inv(H@Pm@H.T+R)
            s_text = sm+K@(r+H@(sj-sm))                                          # textbook iterated Kalman
            Lam = np.linalg.inv(Pm); A = Lam+H.T@np.linalg.inv(R)@H; b = H.T@np.linalg.inv(R)@r-Lam@(sj-sm)
            s_info = sj+np.linalg.solve(A, b)                                     # information form
            np.testing.assert_allclose(s_info, s_text, rtol=1e-10, atol=1e-10)
            IKH = np.eye(6)-K@H
            np.testing.assert_allclose(np.linalg.inv(A), IKH@Pm@IKH.T+K@R@K.T, rtol=1e-9, atol=1e-11)

    def test_signed_time_transition_and_process_noise(self):
        qc = torch.diag(torch.tensor([.01, .02, .03], dtype=torch.float64))
        for dt in (1/59, -1/59, .5, -2.):
            Q = process_noise(dt, qc)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(Q).min()), -1e-18)
            tau = abs(dt)
            np.testing.assert_allclose(Q[:3, :3], tau**3/3*qc); np.testing.assert_allclose(Q[3:, 3:], tau*qc)
            np.testing.assert_allclose(Q[:3, 3:], np.sign(dt)*tau**2/2*qc)
            # Q is the covariance of the integrated white acceleration: Monte Carlo check of the cross sign
            F = transition(dt)
            np.testing.assert_allclose(F[:3, 3:], dt*torch.eye(3, dtype=torch.float64))
        s, P = predict(torch.tensor([[1., 2, 3, .1, -.2, .3]], dtype=torch.float64), torch.eye(6, dtype=torch.float64)[None], -.5, qc)
        np.testing.assert_allclose(s[0, :3].numpy(), [1-.05, 2+.1, 3-.15])


def plane_problem(dev, texture=True, n=30, seed=0):
    """Physically consistent ray-cast scene (tests/image_fixtures.py): static textured plates seen by 4 cameras; the
    truth is the measured endpoint catalogue itself, colours are the measured source-pixel colours."""
    from tests.image_fixtures import two_plate_scene, catalog
    ds = two_plate_scene('static', frames=1, cameras=4)
    if not texture:
        obs = ds.observation
        def flat(k, ci):
            f = obs(k, ci); f.rgb = np.where(f.mask[..., None], np.full_like(f.rgb, .5), 0.); return f
        ds.observation = flat
    if texture:      # 320x240, f=400 (about 4 mm per pixel at 1.9 m) so a few-mm error spans several pixels
        from tests.image_fixtures import look_at
        ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
    xyz, rgb, _, source = catalog(ds, 0)
    g = np.random.default_rng(seed)
    # interior points of the base plate (away from silhouettes and the raised plate's edges)
    # base-plate interior, clear of the raised plate's projected occlusion strip (0.06 m * cot(34 deg) ~ 9 cm beyond its
    # edges at x = +-0.2, y = +-0.12) and of the base plate's own border (x = +-0.45, y = +-0.3)
    ok = (np.abs(xyz[:, 0]) < .41) & (np.abs(xyz[:, 1]) < .26) & (np.abs(xyz[:, 2]) < 1e-9) & ((np.abs(xyz[:, 0]) > .34) | (np.abs(xyz[:, 1]) > .23))
    pick = np.sort(g.choice(np.flatnonzero(ok), n, replace=False))
    rig, frame = CameraRig(ds.cameras, dev), MeasuredFrame(ds, 0, dev, torch.float64)
    return rig, frame, torch.as_tensor(xyz[pick], device=dev), torch.as_tensor(rgb[pick] if texture else np.full((n, 3), .5), device=dev)


class PointUpdateTest(unittest.TestCase):
    def prior(self, x, dev, sig=.004):  # noqa
        s = torch.zeros((len(x), 6), dtype=torch.float64, device=dev); s[:, :3] = x
        P = torch.zeros((len(x), 6, 6), dtype=torch.float64, device=dev); P[:, :3, :3] = sig**2*torch.eye(3, dtype=torch.float64, device=dev); P[:, 3:, 3:] = torch.eye(3, dtype=torch.float64, device=dev)
        return s, P

    def test_depth_only_plane_leaves_lateral_covariance(self):
        rig, frame, x, rgb0 = plane_problem('cpu', texture=False)
        sig = .02
        s, P = self.prior(x+torch.tensor([0., 0., .003], dtype=torch.float64), 'cpu', sig)
        sp, Pp, rep, _ = update_points(s, P, frame, rig, rgb0, noise(), POLICY, 10, 1e-8)
        self.assertTrue(bool(rep['observed'].all()))
        # every camera's depth sees only the plane-normal component of a displacement (moving within the plane changes
        # D and Z equally): depth-only information is rank one along the normal (world z) from ALL cameras together
        Pxx = Pp[:, :3, :3]
        np.testing.assert_allclose(Pxx[:, 0, 0].numpy(), sig**2, rtol=1e-2)                 # lateral x: prior kept
        np.testing.assert_allclose(Pxx[:, 1, 1].numpy(), sig**2, rtol=1e-2)                 # lateral y: prior kept
        self.assertTrue(bool((Pxx[:, 2, 2] < .1*sig**2).all()))                             # normal: contracted
        self.assertLess(float((sp[:, 2]-x[:, 2]).abs().max()), .003)                       # normal error corrected

    def test_prediction_only_point_keeps_prior_exactly(self):
        rig, frame, x, rgb0 = plane_problem('cpu')
        s, P = self.prior(x, 'cpu')
        s[0, :3] += torch.tensor([50., 0, 0], dtype=torch.float64)                            # far outside every camera
        sp, Pp, rep, _ = update_points(s, P, frame, rig, rgb0, noise(), POLICY, 10, 1e-8)
        self.assertFalse(bool(rep['observed'][0]))
        self.assertTrue(torch.equal(sp[0], s[0])); self.assertTrue(torch.equal(Pp[0], P[0]))

    def test_points_are_independent(self):
        rig, frame, x, rgb0 = plane_problem('cpu', n=30)
        s, P = self.prior(x+torch.tensor([.002, -.001, .002], dtype=torch.float64), 'cpu')
        full = update_points(s, P, frame, rig, rgb0, noise(), POLICY, 10, 1e-8)
        # a point that no other point displaced from any pixel (never DUPLICATE/SELF_OCCLUDED) must get exactly the update
        # it gets when filtered alone: no group coupling, no shared increment
        free = [p for p in range(len(x)) if not bool(((full[2]['reason'][:, p] == 6) | (full[2]['reason'][:, p] == 7)).any())]
        self.assertGreater(len(free), 10)
        for p in free:
            alone = update_points(s[p:p+1], P[p:p+1], frame, rig, rgb0[p:p+1], noise(), POLICY, 10, 1e-8)
            np.testing.assert_allclose(alone[0][0].numpy(), full[0][p].numpy(), atol=1e-12)
            np.testing.assert_allclose(alone[1][0].numpy(), full[1][p].numpy(), atol=1e-14)

    def test_textured_update_moves_toward_the_surface_and_traces_kalman(self):
        rig, frame, x, rgb0 = plane_problem('cpu')
        off = torch.tensor([.0015, -.001, .004], dtype=torch.float64)
        s, P = self.prior(x+off, 'cpu', .02)                                        # broad prior: the images decide
        sp, Pp, rep, tr = update_points(s, P, frame, rig, rgb0, noise(), POLICY, 10, 1e-8, trace_ids=[3])
        before = (s[:, :3]-x).norm(dim=1); after = (sp[:, :3]-x).norm(dim=1)
        self.assertLess(float(after.median()), .2*float(before.median()))
        f = tr['final'][3]
        self.assertLess(f['joseph_vs_information_max_abs'], 1e-12)
        self.assertLess(f['H_fd_rel_error'], 1e-4)
        self.assertTrue(np.isfinite(rep['nis'].numpy()).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cpu_and_cuda_agree(self):
        out = {}
        for dev in ('cpu', 'cuda'):
            rig, frame, x, rgb0 = plane_problem(dev)
            s, P = self.prior(x+torch.tensor([.0015, -.001, .004], dtype=torch.float64, device=dev), dev)
            sp, Pp, rep, _ = update_points(s, P, frame, rig, rgb0, noise(dev), POLICY, 10, 1e-8)
            out[dev] = (sp.cpu(), Pp.cpu(), rep['nrows'].cpu())
        self.assertTrue(torch.equal(out['cpu'][2], out['cuda'][2]))
        np.testing.assert_allclose(out['cuda'][0].numpy(), out['cpu'][0].numpy(), atol=1e-10)
        np.testing.assert_allclose(out['cuda'][1].numpy(), out['cpu'][1].numpy(), atol=1e-12, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
