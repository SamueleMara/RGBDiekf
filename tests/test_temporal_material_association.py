"""Material identity, visibility and acceptance on exactly rendered moving scenes (spec section 6.B / 6.C).

The actual MaterialFilter / LocalFilter estimators run on analytic RGB-D frames (tests/material_fixtures.py). Truth is
known only to the assertions. Pixel scale: one pixel = Z/128 m (about 1.6 cm at Z = 2).
"""
import unittest
import numpy as np
import torch
from tests.material_fixtures import Scene, S, FOC, DEVICES, make_filter, run_frames
from estimator.temporal_local_run import LocalFilter
from estimator.temporal_material_iekf import MaterialFrameUpdate

PX = 2./FOC
PIXELS = [12*S+12, 12*S+19, 16*S+16, 19*S+12, 19*S+19]
MATERIAL_ASSOCIATION_WITNESSES = []


def displacement_error(step, base, d, k):
    s = step['s'][:, :3].cpu().numpy()
    return np.linalg.norm(s-(base+k*np.asarray(d)), axis=1)


class MaterialAssociationTests(unittest.TestCase):
    def test_translations_recovered_where_fixed_pixel_rejects(self):
        for device in DEVICES:
            for d in [(1.3*PX, 0., 0.), (-1.3*PX, .5*PX, 0.), (3*PX, 0., 0.), (-3*PX, -1*PX, 0.), (1.3*PX, 0., .03), (0., -1.3*PX, -.03)]:
                with self.subTest(device=device, d=d):
                    sc = Scene(d=d)
                    frames = [sc.render(k, device) for k in range(3)]
                    filt, _, base = make_filter(sc, device, PIXELS)
                    steps = run_frames(filt, frames, 2)
                    for k, st in enumerate(steps, 1):
                        err = displacement_error(st, base, d, k)
                        self.assertTrue(bool(st['update'].assimilated.all()))
                        self.assertLess(float(err.max()), .05*PX, f'frame {k} errors {err}')
                    v = steps[-1]['s'][:, 3:].cpu().numpy()
                    np.testing.assert_allclose(v, np.tile(np.asarray(d)/.04, (len(PIXELS), 1)), atol=.05*PX/.04)
                    shifts = steps[0]['lr'].rows.pixel_shift.norm(dim=1)
                    MATERIAL_ASSOCIATION_WITNESSES.append({'device': device, 'd_m': list(d), 'max_error_px_frame2': float(err.max()/PX),
                                                           'accepted_pixel_shift_px_frame1': [float(shifts.min()), float(shifts.max())],
                                                           'target_pixel_changes': steps[0]['solver']['counters']['target_pixel_changes_accepted']})
                    if abs(d[0]) == 1.3*PX and d[2] == 0:
                        # acceptance regression: the frozen current-pixel Eulerian domain cannot follow a > 0.5 px move
                        loc, _, lbase = make_filter(sc, device, PIXELS, cls=LocalFilter)
                        lerr = displacement_error(run_frames(loc, frames, 1)[0], lbase, d, 1)
                        self.assertGreater(float(np.median(lerr)), .3*PX)
                        MATERIAL_ASSOCIATION_WITNESSES.append({'fixed_pixel_local_correlated_median_error_px': float(np.median(lerr)/PX), 'd_m': list(d)})

    def test_static_scene_stays_exactly_static(self):
        for device in DEVICES:
            sc = Scene()
            frames = [sc.render(k, device) for k in range(4)]
            filt, _, base = make_filter(sc, device, PIXELS)
            steps = run_frames(filt, frames, 3)
            for st in steps:
                self.assertLess(float(np.abs(st['s'][:, :3].cpu().numpy()-base).max()), 1e-12)
                self.assertLess(float(st['s'][:, 3:].abs().max()), 1e-10)
                self.assertLess(float(st['lr'].rows.pixel_shift.abs().max()), 1e-10)

    def test_no_background_update_when_material_leaves_the_mask(self):
        """Material moves +3 px right but the hard mask (both cameras) ends one pixel to the right of the source: no
        identity may be sampled on background and no accepted state may project onto background."""
        sc = Scene(d=(3*PX, 0., 0.))
        mask = np.zeros((2, S, S), bool)
        mask[:, 8:24, 8:18] = True
        frames = [sc.render(k, 'cpu', mask) for k in range(2)]
        filt, _, base = make_filter(sc, 'cpu', [16*S+16], frame0=frames[0])
        st = run_frames(filt, frames, 1)[0]
        own = st['owners']
        for c in range(2):
            if int(own.reason[c, 0]) in (0, 6):
                pix = int(own.pixel[c, 0])
                self.assertTrue(bool(frames[1].mask[c, pix//S, pix % S]), 'accepted state projects onto background')
        rows = st['lr'].rows
        for a in range(len(rows)):
            c = int(rows.camera[a])
            for pix in rows.current_pixels[a].tolist():
                self.assertTrue(bool(frames[1].mask[c, pix//S, pix % S]))
        MATERIAL_ASSOCIATION_WITNESSES.append({'leaving_mask_rows': len(rows), 'reasons': st['lr'].reason[:, 0].tolist(),
                                               'state_shift_px': float((st['s'][0, 0]-base[0, 0])/PX)})

    def test_deleting_a_measurement_never_creates_descent(self):
        """compare(): a trial that loses an identity to invalid support is rejected even with a lower total cost; a trial
        that loses an identity to occlusion is judged ONLY on the identities valid at both states."""
        upd = MaterialFrameUpdate.__new__(MaterialFrameUpdate)
        upd.dev, upd.C, upd.N = torch.device('cpu'), 2, 3
        z = lambda *v: torch.tensor(v, dtype=torch.float64)
        base = {'cost': torch.stack((z(1., 1., 1.), z(50., 50., 50.))), 'valid': torch.ones((2, 3), dtype=torch.bool),
                'fg': torch.ones((2, 3), dtype=torch.bool), 'inside': torch.ones((2, 3), dtype=torch.bool), 'prior': z(0., 0., 0.)}
        valid = torch.tensor([[True, True, True], [False, False, True]])
        reason = torch.tensor([[0, 0, 0], [5, 8, 0]], dtype=torch.uint8)       # point 0: support break; point 1: occluded
        trial = {'cost': torch.stack((z(.5, 3., .5), z(0., 0., 49.))), 'valid': valid, 'fg': torch.ones((2, 3), dtype=torch.bool),
                 'inside': torch.ones((2, 3), dtype=torch.bool), 'reason': reason}
        good, why, hidden, gained = upd.compare(base, trial, z(.1, .1, .1))
        self.assertFalse(bool(good[0]))            # lower total cost but an identity destroyed: rejected
        self.assertTrue(bool(why['identity_support_lost'][0]))
        self.assertFalse(bool(good[1]))            # occlusion loss allowed, but the remaining identity got worse (3 > 1)
        self.assertTrue(bool(why['no_common_map_descent'][1]))
        self.assertTrue(bool(hidden[1, 1]))
        self.assertTrue(bool(good[2]))             # common identities improved
        trial['fg'] = torch.tensor([[True, True, True], [True, True, False]])
        good, why, _, _ = upd.compare(base, trial, z(.1, .1, .1))
        self.assertFalse(bool(good[2]))            # new background projection: rejected
        self.assertTrue(bool(why['outside_mask'][2]))

    def test_occlusion_preserves_reference_and_reappears(self):
        """A nearer measured surface covers the point in camera 0 at frame 1 only: camera 0 is prediction-only for that
        point at frame 1 (MEASURED_SURFACE_IN_FRONT), its frame-0 reference is preserved, and the identity reappears at
        frame 2 with reference age 2 (< lag 3) while camera 1 keeps tracking."""
        d = (1.3*PX, 0., 0.)
        sc = Scene(d=d)
        frames = [sc.render(k, 'cpu') for k in range(3)]
        frames[1].depth[0, 10:24, 10:26] -= .4      # 40 cm nearer: beyond the single-surface allowance (12.5 cm at Z=2)
        filt, _, base = make_filter(sc, 'cpu', [16*S+16])
        st1, st2 = run_frames(filt, frames, 2)
        self.assertEqual(int(st1['lr'].reason[0, 0]), 8)
        self.assertIn(1, st1['lr'].rows.camera.tolist())
        rows2 = st2['lr'].rows
        cam0 = torch.where(rows2.camera == 0)[0]
        self.assertEqual(len(cam0), 1)
        self.assertEqual(int(rows2.reference_frame[cam0[0]]), 0)
        self.assertLess(float(displacement_error(st2, base, d, 2).max()), .05*PX)
        MATERIAL_ASSOCIATION_WITNESSES.append({'occlusion_reason_frame1_cam0': int(st1['lr'].reason[0, 0]),
                                               'reference_frame_used_frame2_cam0': int(rows2.reference_frame[cam0[0]])})

    def test_same_surface_collision_keeps_material_identity(self):
        """Two catalogue points on the same surface collapse onto one current pixel: the loser of the first-hit sort is
        not occluded (depth gap inside the single-surface allowance) and keeps its identity and row."""
        sc = Scene(d=(1.3*PX, 0., 0.))
        frames = [sc.render(k, 'cpu') for k in range(2)]
        filt, _, base = make_filter(sc, 'cpu', [16*S+16, 16*S+17])
        filt.bank.mean[filt.bank.moff[1]:filt.bank.moff[1]+2] = torch.tensor([-1., 0.], dtype=torch.float64)  # theta_u: half-pixel toward point 0
        st = run_frames(filt, frames, 1)[0]
        rows = st['lr'].rows
        self.assertEqual(sorted(set(rows.point.tolist())), [0, 1])
        self.assertGreaterEqual(st['lr'].life['same_surface_collision_kept'], 0)

    def test_two_camera_handover_bootstraps_without_a_temporal_row(self):
        """Camera 1 is masked at frame 0: no camera-1 identity at frame 1 (bootstrap, reason BOOTSTRAP_PENDING), a camera-1
        reference is committed at frame 1, and frame 2 has a camera-1 row against reference frame 1."""
        sc = Scene(d=(1.3*PX, 0., 0.))
        m0 = np.ones((2, S, S), bool)
        m0[1] = False
        frames = [sc.render(0, 'cpu', m0)]+[sc.render(k, 'cpu') for k in (1, 2)]
        filt, _, base = make_filter(sc, 'cpu', [16*S+16], frame0=frames[0])
        st1, st2 = run_frames(filt, frames, 2)
        self.assertNotIn(1, st1['lr'].rows.camera.tolist())
        self.assertEqual(int(st1['lr'].reason[1, 0]), 11)
        r2 = st2['lr'].rows
        cam1 = torch.where(r2.camera == 1)[0]
        self.assertEqual(len(cam1), 1)
        self.assertEqual(int(r2.reference_frame[cam1[0]]), 1)
        self.assertLess(float(displacement_error(st2, base, sc.d, 2).max()), .05*PX)


if __name__ == '__main__':
    unittest.main()
