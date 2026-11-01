"""Gate 6 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md: foreground and visibility invariants (stage 1: masks are a
hard admission condition of every RGB-D sample plus full-frame feasibility DIAGNOSTICS; no mask term in the objective).

G6.1 a static occluder between a camera and background points: no row from the occluded camera, other cameras update the
     point, no switch to the occluder's surface.
G6.2 feasibility is evaluated for every point in every in-view camera, occluded or not; a mask change away from every
     point's footprint changes nothing (no correction by mask alone).
G6.3 contradictory masks (object erased in one camera): infeasibility detected and recorded per point-frame, no fake
     displacement.
G6.4 one-pixel mask shift at the silhouette: boundary ambiguity reported; interior points unaffected.
G6.5 the rendered pseudomask cannot be enabled with the temporal solver."""
import unittest
import numpy as np
from tests.temporal_fixtures import fronto_cameras, AffinePlate, AnalyticDataset, catalogue, run_solver
from tests.test_temporal_phase1_static import RING, interior_pixels

T = np.arange(12)/30.


def scene(mask_edit=None):
    plates = [AffinePlate(1.8, (-.9, .9), (-.7, .7)), AffinePlate(1.2, (-.1, .1), (-.06, .06), c0=(.2, .7, .3), Gx=(-1, .5, .3), Gy=(.4, -.6, .9))]
    return AnalyticDataset(fronto_cameras(RING), plates, T, mask_edit=mask_edit)


class MaskInvariantTest(unittest.TestCase):
    def setUp(self):
        ds = scene()
        rgb, depth, mask = ds.render(0, 6)
        self.front = [(u, v) for v in range(72) for u in range(96) if mask[v, u] and abs(depth[v, u]-1.2) < 1e-6]
        g = np.random.default_rng(3); back = []
        while len(back) < 25:
            u, v = int(g.integers(8, 88)), int(g.integers(8, 64))
            if abs(depth[v, u]-1.8) < 1e-6 and (u, v) not in back: back.append((u, v))
        self.back = back

    def run_scene(self, ds):
        cat = catalogue(ds, 6, self.front+self.back)
        return cat, run_solver(ds, cat)

    def test_G6_1_occluded_camera_gives_no_row_and_no_surface_switch(self):
        ds = scene(); cat, res = self.run_scene(ds)
        bi = np.arange(len(self.front), len(self.front)+len(self.back))
        reason = res['camera_reason'].numpy(); hidden = updated = 0
        for i in bi:
            p = cat['x_ref'][i]
            for c, cam in enumerate(ds.cameras):
                o = -cam.world_to_camera[:3, 3]; q = o+(1.2-o[2])/(p[2]-o[2])*(p-o)
                if abs(q[0]) <= .1 and abs(q[1]) <= .06:
                    hidden += 1; self.assertTrue((reason[:, c, i] != 0).all(), (i, c))
                    others = [cc for cc in range(len(ds.cameras)) if cc != c]
                    updated += int((reason[1:, others, i] == 0).any())
        self.assertGreater(hidden, 3); self.assertGreater(updated, 0)
        X = res['smooth_mean'].numpy()[:, bi, :3]
        self.assertLess(float(np.abs(X[..., 2]-1.8).max()), 1e-3)                        # stays on its own (background) surface

    def test_G6_2_feasibility_for_all_points_and_no_correction_by_mask_alone(self):
        ds = scene(); cat, res = self.run_scene(ds)
        inview, inside = res['mask_cameras_in_view'].numpy(), res['mask_cameras_nearest_inside'].numpy()
        X = res['smooth_mean'].numpy()[..., :3]                                          # independent geometric count, occlusion ignored
        expect = np.zeros_like(inview)
        for cam in ds.cameras:
            fx, fy, cx, cy = cam.intrinsics; Xc = X-(-cam.world_to_camera[:3, 3])
            u, v = fx*Xc[..., 0]/Xc[..., 2]+cx, fy*Xc[..., 1]/Xc[..., 2]+cy
            expect += ((Xc[..., 2] > cam.near) & (u >= -.5) & (u <= cam.width-.5) & (v >= -.5) & (v <= cam.height-.5)).astype(np.uint8)
        np.testing.assert_array_equal(inview, expect)                                    # every in-view camera is evaluated, occluded or not
        self.assertGreater(int((res['camera_reason'].numpy() == 6).sum()), 0)            # ... including cameras where the point is occluded
        np.testing.assert_array_equal(inside, inview)                                    # consistent object masks: always inside
        self.assertFalse(res['mask_infeasible'].numpy().any())
        def grow(k, ci, m):                                                               # change far from every footprint: image corner
            m[:3, :3] = True; return m
        cat2, res2 = self.run_scene(scene(grow))
        np.testing.assert_array_equal(res['smooth_mean'].numpy(), res2['smooth_mean'].numpy())

    def test_G6_3_contradictory_mask_detected_without_fake_motion(self):
        def erase(k, ci, m):
            if ci == 2: m[20:52, 30:66] = False                                          # the object vanishes in camera 2 only
            return m
        ds = scene(erase); cat, res = self.run_scene(ds)
        inf = res['mask_infeasible'].numpy()
        self.assertGreater(int(inf.any(0).sum()), 5)                                      # point-frames recorded as mask-infeasible
        d = res['displacement'].numpy()
        self.assertLess(float(np.linalg.norm(d, axis=-1).max()), 1e-3)                   # no fake displacement

    def test_G6_4_off_by_one_mask_at_the_silhouette(self):
        def shift(k, ci, m):
            if ci == 0: m = np.roll(m, 1, axis=1)
            return m
        ds0, ds1 = scene(), scene(shift)
        cat, r0 = self.run_scene(ds0); _, r1 = self.run_scene(ds1)
        amb = r1['mask_cameras_boundary_ambiguous'].numpy()
        self.assertEqual(amb.shape, r0['mask_cameras_boundary_ambiguous'].numpy().shape)
        # the plates fill the frame edges of camera 0; points whose cam-0 support never touches column 0 are unaffected
        reason0, reason1 = r0['camera_reason'].numpy(), r1['camera_reason'].numpy()
        same = (reason0 == reason1).all((0, 1))
        self.assertGreater(int(same.sum()), len(same)//2)
        np.testing.assert_allclose(r1['smooth_mean'].numpy()[:, same], r0['smooth_mean'].numpy()[:, same], rtol=0, atol=1e-9)

    def test_G6_5_rendered_pseudomask_refused(self):
        import train
        with self.assertRaises(SystemExit):
            train.main(['--scene', '/nonexistent', '--output', '/tmp/never_written_temporal', '--free-solver', 'temporal-ieks', '--stop-after', 'free',
                        '--direction', 'forward', '--rendered-mask-measurement', 'diagnostic'])
        from estimator.temporal_phase1 import run_temporal_stage
        from tests.projective_fixtures import POLICY
        with self.assertRaises(ValueError):
            run_temporal_stage(None, None, {**POLICY, 'rendered_mask_measurement': {'mode': 'on'}}, {}, 'forward', '/tmp/never_written_temporal2', 'cpu', None)


if __name__ == '__main__':
    unittest.main()
