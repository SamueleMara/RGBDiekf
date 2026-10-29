"""Contract section 10 E-F end to end (train.py on the ray-cast fixtures; ground truth used by the TEST only, after the
run): --rendered-mask-measurement off writes exactly the old outputs (no new file), diagnostic changes no state or
covariance, on keeps a static scene static (no joint, the prior-derived bounds of test_four_passes) and keeps the moving
part tracked in both processing directions (signed dt); the removed --geometry-* flags fail instead of aliasing."""
import json
import unittest
from pathlib import Path
import numpy as np
from tests import test_four_passes as fp
from tests.image_fixtures import true_positions

ARRAYS = ('posterior_xyz', 'velocity', 'point_status', 'valid_cameras', 'camera_reason', 'position_variance_diag', 'velocity_variance_diag',
          'update_acceptance', 'posterior_cost', 'nis')
NOISE = str(Path(__file__).resolve().parents[1]/'configs/example_mask_image_noise.json')


def run(name):
    """name = <fixture>_<mode>, fixture static | prismatic, mode off | diagnostic | on."""
    if name in fp.RUNS: return fp.RUNS[name]
    fixture, mode = name.split('_')
    extra = ['--rendered-mask-measurement', mode]+(['--mask-image-noise-config', NOISE] if mode == 'on' else [])
    orig = fp.train.main
    fp.train.main = lambda argv: orig([*argv, *extra])
    try:
        fp.RUNS.pop(fixture, None)
        ds, out = fp.run(fixture)
    finally:
        fp.train.main = orig
        fp.RUNS.pop(fixture, None)
    fp.RUNS[name] = (ds, out)
    return ds, out


class RenderedMaskFourPassTest(unittest.TestCase):
    def test_off_writes_old_outputs_and_diagnostic_changes_nothing(self):
        for fixture in ('static', 'prismatic'):
            _, off = run(f'{fixture}_off'); _, dia = run(f'{fixture}_diagnostic')
            for d in ('forward', 'backward'):
                self.assertFalse(list((off/f'free_{d}').glob('rendered_mask_*.npy')), 'off must write exactly the old outputs')
                self.assertTrue((dia/f'free_{d}'/'rendered_mask_status.npy').exists())
                for a in ARRAYS:
                    np.testing.assert_array_equal(np.load(off/f'free_{d}'/f'{a}.npy'), np.load(dia/f'free_{d}'/f'{a}.npy'), err_msg=f'{fixture} {d} {a}')
                frames = sorted((dia/f'free_{d}'/'numeric').glob('frame_*.json'))
                rm = json.loads(frames[0].read_text())['rendered_mask']
                self.assertIn('hard_white_outside_measured_mask_prior', rm)

    def test_on_keeps_a_static_scene_static(self):
        from scipy.stats import chi2
        v99 = float(np.sqrt(chi2.ppf(.99, 3)))
        ds, out = run('static_on')
        T = float(ds.times[-1]-ds.times[0])
        for d in ('forward', 'backward'):
            j = out/'joint'/d/'result.json'
            status = json.loads(j.read_text())['status'] if j.exists() else 'NO_JOINT_STAGE'
            self.assertNotIn(status, ('PRISMATIC', 'REVOLUTE'), d)
            X = np.load(out/f'free_{d}'/'posterior_xyz.npy'); V = np.load(out/f'free_{d}'/'velocity.npy')
            self.assertLess(float(np.linalg.norm(V, axis=2).max()), v99, d)
            self.assertLess(float(np.linalg.norm(X-X[0 if d == 'forward' else -1], axis=2).max()), v99*T, d)
            self.assertEqual(json.loads((out/f'free_{d}'/'status.json').read_text())['rendered_mask']['mode'], 'on')

    def test_on_keeps_the_moving_part_tracked_both_directions(self):
        ds, out = run('prismatic_on')
        K = len(ds.times)
        for d, kl in (('forward', K-1), ('backward', 0)):
            can, X, reason, parts = fp.free(ds, out, d)
            tp = true_positions(ds, can['canonical_xyz'], parts, kl, int(can['canonical_frame']))
            err = np.linalg.norm(X[kl]-tp, axis=1); mot = np.linalg.norm(tp-can['canonical_xyz'], axis=1)
            self.assertLess(np.median(err[parts == 1]), .25*np.median(mot[parts == 1]), d)       # the unconstrained test's criterion
            self.assertLess(np.median(err[parts == 0]), 1e-3, d)
            tr = sorted((out/f'free_{d}'/'numeric').glob('trace_frame_*.json'))
            self.assertTrue(tr)
            dt = json.loads(tr[0].read_text())['points']['prediction']['dt_s']
            self.assertEqual(np.sign(dt), 1 if d == 'forward' else -1)                           # signed time preserved

    def test_removed_hard_geometry_flags_fail(self):
        import train
        for flag in (['--geometry-constraints', 'hard'], ['--geometry-posterior-mean', 'map'], ['--geometry-depth-mode', 'hard_ideal']):
            with self.assertRaises(SystemExit):
                train.main(['--scene', '/nonexistent', '--output', '/tmp/never_written_rendered_mask_test', *flag])
        with self.assertRaises(SystemExit):                                                        # on without a declared mask variance
            train.main(['--scene', '/nonexistent', '--output', '/tmp/never_written_rendered_mask_test', '--rendered-mask-measurement', 'on'])


if __name__ == '__main__':
    unittest.main()
