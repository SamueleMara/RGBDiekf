"""Gate A: projection derivatives, RGB/depth temporal physics and the diagnostic tangent/normal basis (spec 7.1-7.4, 7.6).

Independent references: NumPy projection with explicit rotation/translation, 5-point finite differences, and exactly
rendered analytic images of an inclined textured plane (tests/material_fixtures.py). Nothing here compares two
functions of the same implementation.
"""
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tests.material_fixtures import Scene, S, FOC, CX, CY, POLICY, DEVICES, rig, noise, fd, ring_scene, ring_filter, run_frames
from estimator.projective_camera import CameraRig
from estimator.temporal_image_gradients import gradients_at, eulerian_difference
from estimator.temporal_motion_geometry import velocity_rows, depth_basis, basis_components, normal_alignment

GEOMETRY_WITNESSES = []
DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')


def np_project(R, t, K, X):
    xc = R@X+t
    return np.array([K[0]*xc[0]/xc[2]+K[2], K[1]*xc[1]/xc[2]+K[3]]), xc[2]


class GradientGeometryTests(unittest.TestCase):
    def cameras(self):
        cams = []
        rng = np.random.default_rng(3)
        for j in range(5):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(rng.normal(0, .5, 3)).as_matrix()
            T[:3, 3] = rng.normal(0, .3, 3)+np.array([0, 0, 2.5])
            cams.append(SimpleNamespace(name=f'r{j}', width=640, height=480, intrinsics=np.array([579.4, 579.4, 319.5, 239.5]), near=.01, world_to_camera=T))
        if (DATA/'blade_103706').is_dir():
            from scene.dataset import Dataset
            cams += list(Dataset(DATA/'blade_103706', 1).cameras)                 # actual off-axis Blade calibration
        return cams

    def test_projection_jacobians_match_independent_numpy(self):
        cams = self.cameras()
        for device in DEVICES:
            r = CameraRig(cams, device, torch.float64)
            rng = np.random.default_rng(0)
            worst = 0.
            for c, cam in enumerate(cams):
                T = np.asarray(cam.world_to_camera, float)
                R, t, K = T[:3, :3], T[:3, 3], np.asarray(cam.intrinsics, float)
                for depth in (.4, 1.2, 3.):
                    # world points in front of the camera at the given optical depth, off-centre
                    xc = np.array([rng.uniform(-.3, .3)*depth, rng.uniform(-.2, .2)*depth, depth])
                    X = R.T@(xc-t)
                    uu, zz, Jp, Jzz = r.project_rows(torch.tensor(X[None], device=device), torch.tensor([c], device=device))
                    u_np, z_np = np_project(R, t, K, X)
                    np.testing.assert_allclose(uu[0].cpu().numpy(), u_np, rtol=0, atol=1e-9)    # row/col: u = column, v = row
                    np.testing.assert_allclose(float(zz[0]), z_np, atol=1e-12)
                    Jfd = fd(lambda x: np_project(R, t, K, x)[0], X, 1e-6)
                    Jzfd = fd(lambda x: np.array([np_project(R, t, K, x)[1]]), X, 1e-6)
                    np.testing.assert_allclose(Jp[0].cpu().numpy(), Jfd, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(Jzz[0].cpu().numpy(), Jzfd[0], atol=1e-9)
                    np.testing.assert_allclose(Jzz[0].cpu().numpy(), R[2], atol=1e-12)        # J_Z = e3^T R_cw (fixed camera)
                    worst = max(worst, float(np.abs(Jp[0].cpu().numpy()-Jfd).max()))
            GEOMETRY_WITNESSES.append({'device': device, 'cameras': len(cams), 'J_pi_fd_max_abs': worst})

    def test_temporal_rgb_and_depth_physics_on_inclined_moving_plane(self):
        """Eulerian d/dt of the image at a fixed subpixel location vs -grad I J_pi v and (J_Z - grad D J_pi) v, with v
        the true material velocity: exact analytic renders at frame times k = -eps, +eps (central difference)."""
        dt = 1/59.
        for v_true in (np.array([.6, -.3, .2]), np.array([-.4, .5, -.25])):
            for device in DEVICES:
                sc = Scene(d=v_true*dt, slope=(.25, -.15))
                eps = 1e-4
                fm, fp, f0 = sc.render(-eps, device), sc.render(eps, device), sc.render(0, device)
                r = rig(device)
                uv = torch.tensor([[13.37, 15.61], [18.2, 12.4], [10.7, 19.3]], dtype=torch.float64, device=device)
                cam = torch.zeros(3, dtype=torch.long, device=device)
                g = gradients_at(f0, cam, uv, noise(device), POLICY, r)
                self.assertTrue(bool(g.valid.all()))
                dI, ok, _ = eulerian_difference(fm, fp, cam, uv, noise(device), POLICY, r)
                dIdt = dI/(2*eps*dt)
                X = r.backproject(cam, uv, g.depth)
                G, a, _, _ = velocity_rows(r, cam, X, g.grad_rgb, g.grad_depth)
                v = torch.tensor(v_true, device=device)
                lhs_rgb = dIdt[:, :3]+(G@v)
                lhs_d = dIdt[:, 3]-(a@v)
                # normalize by the magnitudes of the individual terms (they cancel: a near-tangential velocity has a
                # tiny normal component, so the difference itself is not a scale)
                _, _, Jpi, Jz = r.project_rows(X, cam)
                scale_rgb = (dIdt[:, :3].abs()+(G@v).abs()).max()
                scale_d = (dIdt[:, 3].abs()+((g.grad_depth[:, None, :]@Jpi)[:, 0]@v).abs()+(Jz@v).abs()).max()
                # bilinear interpolant vs continuous field: residual is second order in the pixel size
                self.assertLess(float(lhs_rgb.abs().max()/scale_rgb), .05)
                self.assertLess(float(lhs_d.abs().max()/scale_d), 1e-3)
                GEOMETRY_WITNESSES.append({'device': device, 'v_true': v_true.tolist(), 'rgb_relative_residual': float(lhs_rgb.abs().max()/scale_rgb),
                                           'depth_relative_residual': float(lhs_d.abs().max()/scale_d)})

    def test_depth_row_is_the_surface_normal_and_tangents_are_orthogonal(self):
        sc = Scene(slope=(.3, -.2))
        f0 = sc.render(0, 'cpu')
        r = rig('cpu')
        uv = torch.tensor([[12.3, 14.8], [17.6, 16.2]], dtype=torch.float64)
        cam = torch.zeros(2, dtype=torch.long)
        g = gradients_at(f0, cam, uv, noise('cpu'), POLICY, r)
        X = r.backproject(cam, uv, g.depth)
        G, a, _, _ = velocity_rows(r, cam, X, g.grad_rgb, g.grad_depth)
        B, ok = depth_basis(r, cam, uv, g.depth, g.grad_depth, g.valid)
        self.assertTrue(bool(ok.all()))
        n_true = np.array([-.3, .2, 1.])/np.linalg.norm([-.3, .2, 1.])        # plane Z = Z0 + aX + bY
        for i in range(2):
            Bi = B[i].numpy()
            np.testing.assert_allclose(Bi.T@Bi, np.eye(3), atol=1e-12)
            self.assertGreater(abs(float(Bi[:, 2]@n_true)), 1-1e-6)
            self.assertGreater(abs(float(a[i].numpy()@n_true))/np.linalg.norm(a[i].numpy()), 1-1e-4)
        Gb, ab = basis_components(G, a, B)
        self.assertLess(float(ab[:, :2].abs().max()/ab[:, 2].abs().min()), 1e-4)     # a T ~ 0, reported not forced
        self.assertGreater(float(normal_alignment(a, B).min()), 1-1e-4)
        # no derivative row across a depth break or a mask edge
        f1 = sc.render(0, 'cpu')
        f1.depth[0, :, 16:] += .5
        f1.mask[0, 20:, :] = False
        f1.raw_valid = f1.mask.clone()
        g2 = gradients_at(f1, torch.zeros(2, dtype=torch.long), torch.tensor([[15.5, 10.], [10., 19.5]], dtype=torch.float64), noise('cpu'), POLICY, r)
        self.assertEqual(g2.reason.tolist(), [5, 3])
        B2, ok2 = depth_basis(r, torch.zeros(2, dtype=torch.long), torch.tensor([[15.5, 10.], [10., 19.5]], dtype=torch.float64), g2.depth, g2.grad_depth, g2.valid)
        self.assertFalse(bool(ok2.any()))
        self.assertTrue(bool(torch.isnan(B2).all()))

    def test_tangent_rgb_rank_and_no_motion_from_texture_alone(self):
        r = rig('cpu')
        uv = torch.tensor([[14.3, 15.7]], dtype=torch.float64)
        cam = torch.zeros(1, dtype=torch.long)
        ranks = {}
        for name, tex in (('stripes_x', lambda X, Y: np.stack((.5+.2*np.sin(9*X),)*3, -1)),
                          ('two_directions', lambda X, Y: np.stack((.5+.2*np.sin(9*X), .5+.2*np.cos(7*Y), .4+.1*np.sin(5*X+4*Y)), -1)),
                          ('constant', lambda X, Y: np.full(X.shape+(3,), .5))):
            sc = Scene(slope=(0., 0.))
            import tests.material_fixtures as MF
            orig = MF.texture
            MF.texture = lambda X, Y, kind='textured', tex=tex: tex(X, Y)
            try:
                f0 = sc.render(0, 'cpu')
            finally:
                MF.texture = orig
            g = gradients_at(f0, cam, uv, noise('cpu'), POLICY, r)
            X = r.backproject(cam, uv, g.depth)
            G, a, _, _ = velocity_rows(r, cam, X, g.grad_rgb, g.grad_depth)
            B, ok = depth_basis(r, cam, uv, g.depth, g.grad_depth, g.valid)
            GT = (G@B)[0, :, :2].numpy()                                       # RGB response to tangent velocity
            sv = np.linalg.svd(GT, compute_uv=False)
            ranks[name] = int((sv > 1e-9*max(sv.max(), 1e-300)).sum()) if sv.max() > 1e-12 else 0
            b = G[0].T@torch.tensor([.01, .01, .01], dtype=torch.float64)       # a temporal change with no spatial structure
            if name == 'constant':
                self.assertEqual(float(b.abs().max()), 0.)
        self.assertEqual(ranks, {'stripes_x': 1, 'two_directions': 2, 'constant': 0})
        GEOMETRY_WITNESSES.append({'tangent_rgb_rank': ranks})

    def test_static_textured_surface_gives_zero_velocity_guidance_one_and_eight_cameras(self):
        from estimator.temporal_material_run import gradient_guidance_summary
        from tests.material_fixtures import make_filter
        from tests.test_temporal_material_association import PIXELS
        for label, sc, mk in (('2 cameras', Scene(slope=(.2, -.1)), lambda s: make_filter(s, 'cpu', PIXELS)),
                              ('8 cameras', ring_scene(slope=(.2, -.1)), lambda s: ring_filter(s, 'cpu', PIXELS))):
            frames = [sc.render(k, 'cpu') for k in range(2)]
            filt, _, base = mk(sc)
            filt.gradient_guide = 'diagnostic'
            st = run_frames(filt, frames, 1, traced=[0])[0]
            out = gradient_guidance_summary(filt, st)
            tr = out['traced_points_local_index']['0']
            # residual 0 up to float round-off (amplified by the 1e-8 depth variance): no data-driven velocity
            self.assertLess(float(np.abs(tr['velocity_guidance_dv']).max()), 1e-9)
            self.assertLess(float(np.abs(st['s'][:, :3].numpy()-base).max()), 1e-12)
            GEOMETRY_WITNESSES.append({'static': label, 'cameras_with_rows_point0': len(tr['cameras']),
                                       'photometric': out['photometric_consistency_at_accepted_iterate']['status']})

    def test_illumination_change_is_reported_photometric_model_invalid(self):
        """Global 8% gain on a static scene: the diagnostic reports PHOTOMETRIC_MODEL_INVALID. The material-following
        estimator itself still moves static points (measured and recorded; brightness constancy is violated)."""
        from estimator.temporal_material_run import gradient_guidance_summary
        from tests.material_fixtures import make_filter
        from tests.test_temporal_material_association import PIXELS
        sc = Scene(slope=(.2, -.1), gain={1: 1.08})
        frames = [sc.render(k, 'cpu') for k in range(2)]
        filt, _, base = make_filter(sc, 'cpu', PIXELS)
        filt.gradient_guide = 'diagnostic'
        st = run_frames(filt, frames, 1)[0]
        out = gradient_guidance_summary(filt, st)
        self.assertEqual(out['photometric_consistency_at_accepted_iterate']['status'], 'PHOTOMETRIC_MODEL_INVALID')
        ok = Scene(slope=(.2, -.1))
        frames = [ok.render(k, 'cpu') for k in range(2)]
        filt, _, _ = make_filter(ok, 'cpu', PIXELS)
        filt.gradient_guide = 'diagnostic'
        st2 = run_frames(filt, frames, 1)[0]
        self.assertNotEqual(gradient_guidance_summary(filt, st2)['photometric_consistency_at_accepted_iterate']['status'], 'PHOTOMETRIC_MODEL_INVALID')
        GEOMETRY_WITNESSES.append({'gain_1.08_status': 'PHOTOMETRIC_MODEL_INVALID',
                                   'static_motion_under_gain_m': float(np.abs(st['s'][:, :3].numpy()-base).max())})


if __name__ == '__main__':
    unittest.main()
