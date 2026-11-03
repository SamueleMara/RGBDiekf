"""Gate 0 / Gate 7: strict masked point-only clouds (NumPy back-projection oracle), source IDs, timing, and the point-only
3D visibility diagnostic (static handle hidden by a moving blade)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scene.config import load_config
from scene.pointcloud_sequence import build_masked_cloud, canonical_from_frame, increment_noise_study
from estimator.svgd_visibility import ray_visibility
from estimator.svgd_target_score import target_score
from tests.svgd_fixtures import SyntheticDataset, record


class VisibilityTests(unittest.TestCase):
    def test_masked_cloud_matches_numpy_backprojection(self):
        d = SyntheticDataset(K=2, n_cams=3, depth_noise=1e-3, seed=1); cfg = load_config()
        cl = build_masked_cloud(d, 1, cfg); worst = 0.
        for c, cam in enumerate(d.cameras):
            fr = d.observation(1, c); fx, fy, cx, cy = cam.intrinsics; W = cam.world_to_camera
            v, u = np.nonzero(fr.mask & fr.raw_valid); D = fr.depth[v, u].astype(float)
            Xc = np.stack([(u-cx)/fx*D, (v-cy)/fy*D, D], 1); Xw = (Xc-W[:3, 3])@W[:3, :3]          # R_cw^T (X_c - t_cw)
            sel = cl.camera == c
            np.testing.assert_array_equal(cl.source[sel, 2], v*cam.width+u)
            worst = max(worst, float(np.abs(cl.xyz[sel]-Xw).max())); np.testing.assert_allclose(cl.xyz[sel], Xw, atol=1e-12)
            np.testing.assert_array_equal(cl.rgb[sel], fr.rgb[v, u])
            self.assertEqual(cl.counts[c]['foreground_valid'], int(sel.sum()))
        self.assertEqual(len(np.unique(cl.source, axis=0)), cl.M)
        can = canonical_from_frame(build_masked_cloud(d, 0, cfg)); np.testing.assert_array_equal(can.original_id, np.arange(can.N))
        self.assertEqual(can.ids_sha256(), canonical_from_frame(build_masked_cloud(d, 0, cfg)).ids_sha256())
        record('E01', test='backprojection_oracle', max_abs_m=worst)

    def test_background_never_enters(self):
        d = SyntheticDataset(K=1, n_cams=2); cfg = load_config(); cl = build_masked_cloud(d, 0, cfg)
        for c in range(2):
            fr = d.observation(0, c); sel = cl.camera == c
            self.assertTrue(fr.mask.ravel()[cl.source[sel, 2]].all()); self.assertEqual(int(sel.sum()), int((fr.mask & fr.raw_valid).sum()))

    def test_nonbinary_mask_rejected(self):
        from scene.dataloader import exact_binary_mask
        with self.assertRaises(ValueError): exact_binary_mask(np.array([[0, 128, 255]], np.uint8))

    def test_missing_acquisition_timing_fails(self):
        from scene.timing import resolve_frame_times
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'metadata').mkdir(); (Path(t)/'metadata/cameras.json').write_text('{"cameras": [{"name": "cam_000"}]}')
            with self.assertRaisesRegex(Exception, 'MISSING_ACQUISITION_TIMING'): resolve_frame_times(Path(t), frame_ids=[0, 1, 2])

    def test_occluded_static_behind_blade(self):
        """Camera 0 sees the moving plate IN FRONT of part of the static base. Particles of the base behind the plate are
        OCCLUDED (never SUPPORTED by the plate's samples); with V_t = all (DEV-0001) the KDE score at such a particle IS
        pulled toward the occluder -- exactly the failure the visibility law must prevent, made detectable here."""
        d = SyntheticDataset(K=1, n_cams=1, motion='none'); cfg = load_config(); cl = build_masked_cloud(d, 0, cfg)
        fr = d.observation(0, 0); cam = d.cameras[0]
        # base-plane points (z = 0) hidden by the plate along camera-0 rays: intersect those rays with z = 0
        W = cam.world_to_camera; eye = -W[:3, :3].T@W[:3, 3]; fx, fy, cx, cy = cam.intrinsics
        v, u = np.nonzero(fr.gt_moving)
        dirs = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones(len(u))], 1)@W[:3, :3]
        s = -eye[2]/dirs[:, 2]; hidden = eye+s[:, None]*dirs
        hidden = hidden[(np.abs(hidden[:, 0]) < .08) & (np.abs(hidden[:, 1]) < .06)]
        x = torch.tensor(np.r_[hidden, cl.xyz[:200], [[5., 5., 5.]]])
        dv = lambda z: torch.full_like(z, 1e-6)      # DECLARED 1 mm depth sigma (the approved config's 1 cm cannot separate 3 cm surfaces)
        codes, point, _ = ray_visibility(x, cl, d.cameras, dv, 3.)
        nh = len(hidden)
        self.assertTrue(bool((codes[:nh, 0] == 1).all()))                       # OCCLUDED_BY_OBSERVED_SURFACE
        self.assertTrue(bool((codes[nh:nh+200, 0] == 0).all()))                 # observed samples themselves: SUPPORTED
        self.assertEqual(int(codes[-1, 0]), 4)                                   # outside the frustum
        lx = torch.tensor(cl.cov)+torch.eye(3, dtype=torch.float64)*4e-6
        r = target_score(x[:nh], torch.full((nh, 3), .5, dtype=torch.float64), torch.tensor(cl.xyz), torch.tensor(cl.rgb), lx, torch.eye(3, dtype=torch.float64)*.05)
        toward_camera = ((r.grad_xyz.numpy())*(eye-hidden)).sum(1) > 0
        record('E15', test='occluded_static_behind_blade', hidden=nh, flagged_occluded=int((codes[:nh, 0] == 1).sum()),
               score_pulled_toward_occluder_fraction=float(toward_camera.mean()), note='DEV-0001 failure mode exposed, not fixed')

    def test_increment_anticorrelation_identity(self):
        """PDF (4): i.i.d. stationary depth noise gives lag-1 increment correlation -1/2 (noise study, not a residual)."""
        d = SyntheticDataset(K=12, n_cams=2, motion='none', depth_noise=2e-3, rgb_noise=.01, seed=3)
        st = increment_noise_study(d, triples=10)
        self.assertAlmostEqual(st['depth_increment_lag1_correlation'], -.5, delta=.03)
        for c in st['rgb_increment_lag1_correlation']: self.assertAlmostEqual(c, -.5, delta=.04)
        self.assertAlmostEqual(st['depth_increment_variance_m2'], 2*(2e-3)**2, delta=.1*2*(2e-3)**2)
        record('E35', test='iid_increments', lag1=st['depth_increment_lag1_correlation'], expected=-.5)


if __name__ == '__main__':
    unittest.main()
