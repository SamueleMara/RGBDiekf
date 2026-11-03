"""Gate B: diagnostic velocity information, visibility stages, no double count, off/diagnostic bitwise identity, lag
expiry and process-only propagation (spec 7.7-7.10, 7.13)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tests.material_fixtures import Scene, S, FOC, DEVICES, make_filter, run_frames
from tests.test_temporal_material_association import PIXELS
from estimator.temporal_material_run import gradient_guidance_summary
from estimator.temporal_motion_observability import velocity_guidance, row_noise
from estimator.point_kalman_cuda import transition, process_noise

REPO = Path(__file__).resolve().parents[1]
PX = 2./FOC
OBSERVABILITY_WITNESSES = []


class GradientObservabilityTests(unittest.TestCase):
    def test_camera_fusion_equals_independent_row_sum(self):
        sc = Scene(d=(1.3*PX, 0., .01))
        frames = [sc.render(k, 'cpu') for k in range(2)]
        filt, _, _ = make_filter(sc, 'cpu', PIXELS)
        st = run_frames(filt, frames, 1)[0]
        lr = st['update'].prior_rows
        g = velocity_guidance(lr, filt.noise, st['update'].warp, .04, st['prior6'], frames[1], filt.rig, filt.policy)
        H = lr.rows.H_t[:, :, :3].numpy()
        R = row_noise(lr, filt.noise, st['update'].warp).numpy()
        f = lr.rows.f.numpy()
        for i in range(len(PIXELS)):
            ix = np.flatnonzero(lr.rows.point.numpy() == i)
            Lam = sum(.04**2*H[a].T@np.linalg.solve(R[a], H[a]) for a in ix)
            b = sum(.04*H[a].T@np.linalg.solve(R[a], -f[a]) for a in ix)
            np.testing.assert_allclose(g['Lambda_v'][i].numpy(), Lam, rtol=1e-10)
            np.testing.assert_allclose(g['b_v'][i].numpy(), b, rtol=1e-8, atol=1e-12)
            self.assertEqual(len(ix), 2)                                       # both cameras admitted, one row each
        OBSERVABILITY_WITNESSES.append({'fusion_rows_per_point': 2})

    def test_identifiable_directions_textured_vs_textureless_aperture(self):
        out = {}
        for name, sc in (('textured', Scene(d=(1.3*PX, -.4*PX, .01), slope=(.2, -.1))),
                         ('textureless_fronto_parallel', Scene(d=(1.3*PX, 0., 0.), slope=(0., 0.), kind='textureless'))):
            frames = [sc.render(k, 'cpu') for k in range(2)]
            filt, _, _ = make_filter(sc, 'cpu', PIXELS)
            filt.gradient_guide = 'diagnostic'
            st = run_frames(filt, frames, 1)[0]
            g = velocity_guidance(st['update'].prior_rows, filt.noise, st['update'].warp, .04, st['prior6'], frames[1], filt.rig, filt.policy)
            out[name] = g['identifiable_directions'].tolist()
            if name.startswith('textureless'):
                self.assertTrue(all(x < 3 for x in out[name]))
                self.assertLess(float(g['dv'][:, :2].abs().max()), 1e-9)       # no invented in-plane velocity
            else:
                self.assertTrue(all(x == 3 for x in out[name]))
        OBSERVABILITY_WITNESSES.append({'identifiable_velocity_directions': out})

    def test_rear_surface_points_never_get_rows_and_pixel_crossing_stays_eligible(self):
        """Catalogue = front plane points + points 0.4 m BEHIND it (not rendered: a hidden rear surface). Only the frontmost
        material yields rows; a 3 px motion crosses pixels and stays admitted."""
        for d in ((0., 0., 0.), (3*PX, 0., 0.)):
            sc = Scene(d=d)
            frames = [sc.render(k, 'cpu') for k in range(2)]
            filt, _, base = make_filter(sc, 'cpu', PIXELS)
            rear = torch.tensor(base*np.array([1., 1., 1.2]))                 # same rays, 20% farther
            from estimator.temporal_material_run import MaterialFilter
            from tests.material_fixtures import rig, noise, POLICY
            src = np.array([[0, 0, p] for p in PIXELS]*2)
            filt = MaterialFilter(rig('cpu'), noise('cpu'), POLICY, np.zeros((4, 4)), np.eye(3)*1e-4, 3, 10**8, 8, frames[0], src,
                                  np.concatenate((base, rear.numpy())))
            filt.gradient_guide = 'diagnostic'
            filt.initialize(frames[0], np.eye(2)/12, np.eye(3))
            st = run_frames(filt, frames, 1)[0]
            pts = st['lr'].rows.point.numpy()
            self.assertTrue((pts < len(PIXELS)).all())                        # rear (hidden) points: no rows
            out = gradient_guidance_summary(filt, st)
            if any(d):
                self.assertGreater(st['solver']['counters']['target_pixel_changes_accepted'], 0)
                self.assertEqual(out['stage_counts_total']['admitted_row'], 2*len(PIXELS))
            OBSERVABILITY_WITNESSES.append({'d_px': [x/PX for x in d], 'stage_counts_total': out['stage_counts_total'],
                                            'rear_reasons_cam0': st['lr'].reason[0, len(PIXELS):].tolist()})

    def test_diagnostic_is_bitwise_identical_to_off_in_the_runner(self):
        from tests.test_temporal_rgbd_pipeline import TinyDataset
        from tests.test_temporal_local_pipeline import options
        from scene.config import load_config
        from estimator.temporal_material_run import run_temporal_material_free
        cfg = load_config()
        policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
        sub = {'factor': 1, 'method': 'spatial-stratified', 'seed': 0, 'boundary_preserve': False, 'report': False}
        roots = {}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('off', 'diagnostic'):
                root = Path(tmp)/mode
                opt = {**options(), 'mode': 'material-following', 'gradient_guide': mode}
                run_temporal_material_free(TinyDataset(), cfg, policy, 'forward', root, 'cpu', sub, opt, [1, 2], [4], max_iters=5)
                roots[mode] = root
            arrays = sorted(p.name for p in roots['off'].glob('*.npy'))
            self.assertTrue(arrays)
            for name in arrays:
                a, b = np.load(roots['off']/name), np.load(roots['diagnostic']/name)
                self.assertTrue(np.array_equal(a, b, equal_nan=True), name)
            jo, jd = (json.loads((roots[m]/'completed_frames.json').read_text()) for m in ('off', 'diagnostic'))
            self.assertEqual(jo['completed_frames'], jd['completed_frames'])   # (the journal also records free disk bytes)
            for k in (1, 2, 3):
                fo = json.loads((roots['off']/f'temporal/frame_{k:04d}.json').read_text())
                fd_ = json.loads((roots['diagnostic']/f'temporal/frame_{k:04d}.json').read_text())
                self.assertNotIn('gradient_guidance', fo)
                self.assertIn('gradient_guidance', fd_)
                for key in ('rows', 'assimilated_points', 'accepted_objective', 'prior_objective', 'solver'):
                    self.assertEqual(fo[key], fd_[key], key)
            OBSERVABILITY_WITNESSES.append({'bitwise_identical_arrays': len(arrays)})

    def test_lag_expiry_process_only_propagation_and_reacquisition(self):
        """The point is masked in every camera for frames 2-5 (lag 3): no rows, exact process-only 6x6 propagation
        P = F P F^T + Q, no zero-velocity measurement; at frame 6 the old reference is outside the lag (no row, new
        reference committed because the point had no rows) and frame 7 has a row against reference frame 6."""
        sc = Scene(d=(1.3*PX, 0., 0.))
        mask = np.ones((2, S, S), bool)
        hidden = np.zeros((2, S, S), bool)
        frames = [sc.render(k, 'cpu', hidden if 2 <= k <= 5 else mask) for k in range(8)]
        filt, _, base = make_filter(sc, 'cpu', [16*S+16])
        qc = torch.eye(3, dtype=torch.float64)*1e-4
        prev = None
        for k in range(1, 8):
            st = filt.step({j: frames[j] for j in range(max(0, k-2), k+1)}, k, k, .04, 10, 1e-10)
            if 2 <= k <= 5:
                self.assertEqual(len(st['lr'].rows), 0)
                F = transition(.04, torch.float64)
                Q = process_noise(.04, qc)
                np.testing.assert_allclose(st['s'][0].numpy(), (F@prev['s'][0]).numpy(), atol=1e-15)
                np.testing.assert_allclose(st['P'][0].numpy(), (F@prev['P'][0]@F.T+Q).numpy(), rtol=1e-9, atol=1e-18)
            if k == 6:
                self.assertEqual(len(st['lr'].rows), 0)
                self.assertTrue(set(st['lr'].reason[:, 0].tolist()) <= {11, 12})
                self.assertTrue((filt.refs.frame[:, 0] == 6).all())
            if k == 7:
                self.assertEqual(sorted(st['lr'].rows.reference_frame.tolist()), [6, 6])
            prev = st
        OBSERVABILITY_WITNESSES.append({'lag_expiry_reacquired_reference_frame': 6})


if __name__ == '__main__':
    unittest.main()
