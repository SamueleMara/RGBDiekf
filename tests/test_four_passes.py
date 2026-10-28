"""Gates C/E/F/G end to end through train.py on ray-cast RGB-D scenes written to disk (the observations are simulated
sensor fields, not the estimator's own predictions; ground truth is used by the TEST only, after the run):
independent endpoint gauges and processing orders, chronological storage, free tracking within the projective
capture range, temporarily occluded static points, coherence -> rigid -> joint -> support, motion reversal, and the
static scene (no joint may be identified).
The fixtures are noiseless ray-cast renders: the default --sensor-noise measured would measure only quantization
floors there (scene/noise_calibration.py), so the pipeline checks run with the declared --noise-config R they were
written for, and the *_measured runs check the default path (no false joint, free tracking kept)."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
from tests.image_fixtures import two_plate_scene, source_parts, true_positions
from tests.projective_fixtures import write_scene, hires
from utils.lie import exp as se3_exp
import train

RUNS = {}
V99 = .5        # declared speed for the *_v99 runs (the fixtures move at 0 and 0.12 m/s)


def run(name):
    if name in RUNS: return RUNS[name]
    extra = ['--sensor-noise', 'measured'] if name.endswith('_measured') else ['--sensor-noise', 'config']
    if name == 'prismatic_measured': extra += ['--stop-after', 'free']
    if name.endswith('_v99'): extra += ['--velocity-prior-speed99', str(V99)]
    name_ = name.replace('_measured', '').replace('_v99', '')
    if name_ == 'reversal':
        ds = hires(two_plate_scene('prismatic', frames=10, cameras=4, speed=.004))
        ds.motion[1] = lambda t: se3_exp(np.r_[.004*(t*30 if t*30 <= 5 else 10-t*30)*np.array([1., .3, 0.]), 0, 0, 0])
    elif name_ == 'static':
        ds = hires(two_plate_scene('static', frames=6, cameras=4))
    else:   # ~0.8 px per frame: inside the capture range of a local projective IEKF
        ds = hires(two_plate_scene('prismatic', frames=10, cameras=4, speed=.004))
    root = Path(tempfile.mkdtemp())
    scene = write_scene(ds, root/'scene'); out = root/'run'
    train.main(['--scene', str(scene), '--output', str(out), '--fps', '30', '--debug', 'none', '--no-evaluate', '--no-qa', *extra])
    RUNS[name] = (ds, out)
    return RUNS[name]


def free(ds, out, d):
    can = np.load(out/f'free_{d}'/'canonical.npz')
    return can, np.load(out/f'free_{d}'/'posterior_xyz.npy'), np.load(out/f'free_{d}'/'camera_reason.npy'), source_parts(ds, can['original_source_ids'])


class FourPassTest(unittest.TestCase):
    def test_independent_gauges_orders_and_storage(self):
        ds, out = run('prismatic')
        K = len(ds.times)
        mf = json.loads((out/'free_forward'/'run_manifest.json').read_text()); mb = json.loads((out/'free_backward'/'run_manifest.json').read_text())
        self.assertEqual(mf['processing_order'], list(range(K))); self.assertEqual(mb['processing_order'], list(range(K-1, -1, -1)))
        cf, Xf, _, _ = free(ds, out, 'forward'); cb, Xb, _, _ = free(ds, out, 'backward')
        self.assertEqual(int(cf['canonical_frame']), 0); self.assertEqual(int(cb['canonical_frame']), K-1)
        np.testing.assert_array_equal(Xf[0], cf['canonical_xyz']); np.testing.assert_array_equal(Xb[K-1], cb['canonical_xyz'])
        self.assertTrue((cb['original_source_ids'][:, 0] == K-1).all() and (cf['original_source_ids'][:, 0] == 0).all())
        # two different measured populations: the static base is measured identically at both endpoints, the moving plate is not
        self.assertFalse(len(cf['canonical_xyz']) == len(cb['canonical_xyz']) and np.array_equal(cf['canonical_xyz'], cb['canonical_xyz']))

    def test_free_tracking_within_capture_range_and_occluded_static_points(self):
        for name in ('prismatic', 'prismatic_measured'):
            with self.subTest(name):
                self._free_tracking(*run(name))

    def _free_tracking(self, ds, out):
        K = len(ds.times)
        for d, kl in (('forward', K-1), ('backward', 0)):
            can, X, reason, parts = free(ds, out, d)
            tp = true_positions(ds, can['canonical_xyz'], parts, kl, int(can['canonical_frame']))
            err = np.linalg.norm(X[kl]-tp, axis=1); mot = np.linalg.norm(tp-can['canonical_xyz'], axis=1)
            self.assertLess(np.median(err[parts == 1]), .25*np.median(mot[parts == 1]), d)
            self.assertLess(np.median(err[parts == 0]), 1e-3, d)
            occl = (parts == 0) & ((reason == 6) | (reason == 8)).any(axis=(0, 1))      # static points hidden at some frame
            self.assertGreater(occl.sum(), 0)
            self.assertLess(np.median(err[occl]), 2e-3, d)

    def test_coherence_rigid_joint_support(self):
        ds, out = run('prismatic')
        K = len(ds.times)
        for d, kl in (('forward', K-1), ('backward', 0)):
            cs = json.loads((out/f'coherence_{d}'/'coherence_summary.json').read_text())
            self.assertEqual(cs['status'], 'RESOLVED')
            self.assertIn('seed_core_nodes', cs['policies']['pinned_raw']); self.assertIn('recruited_nodes', cs['policies']['pinned_raw'])
            can, X, _, parts = free(ds, out, d)
            core = np.load(cs['moving_core_point_ids_file'])
            self.assertGreaterEqual((parts[core] == 1).mean(), .95)
            P = np.load(out/f'rigid_{d}'/'poses.npy'); k0 = int(can['canonical_frame'])
            true = ds.poses(kl)[1]@np.linalg.inv(ds.poses(k0)[1])
            self.assertLess(np.linalg.norm(P[kl][:3, 3]-true[:3, 3]), 1e-3)
            st = json.loads((out/f'rigid_{d}'/'status.json').read_text())
            self.assertLess(st['max_rigidity_error_m'], 1e-9)
            j = json.loads((out/'joint'/d/'result.json').read_text())
            self.assertEqual(j['status'], 'PRISMATIC', j['reason'])
            e = np.asarray(j['candidates']['PRISMATIC']['axis']); g = np.array([1., .3, 0.])/np.linalg.norm([1., .3, 0.])
            self.assertLess(np.degrees(np.arccos(min(1, abs(e@g)))), 2.)
            sup = np.load(out/'joint'/d/'support'/'support.npz', allow_pickle=True)
            lab = sup['labels']; mv = lab == 'MOVING_JOINT_COMPATIBLE'; stc = lab == 'STATIC_COMPATIBLE'
            if mv.any(): self.assertGreaterEqual((parts[mv] == 1).mean(), .9)
            if stc.any(): self.assertGreaterEqual((parts[stc] == 0).mean(), .9)
        b = json.loads((out/'bidirectional'/'comparison.json').read_text())
        self.assertLess(b['per_type']['PRISMATIC']['axis_angle_deg_up_to_sign'], 2.)

    def test_motion_reversal_rigid(self):
        ds, out = run('reversal')
        K = len(ds.times)
        for d, kl in (('forward', K-1), ('backward', 0)):
            can = np.load(out/f'free_{d}'/'canonical.npz'); k0 = int(can['canonical_frame'])
            if not (out/f'rigid_{d}'/'poses.npy').exists(): self.fail(f'{d}: no rigid pass')
            P = np.load(out/f'rigid_{d}'/'poses.npy')
            for k in range(K):
                true = ds.poses(k)[1]@np.linalg.inv(ds.poses(k0)[1])
                self.assertLess(np.linalg.norm(P[k][:3, 3]-true[:3, 3]), 1e-3, f'{d} frame {k}')

    def test_static_scene_identifies_no_joint(self):
        for name in ('static', 'static_measured', 'static_v99'):
            ds, out = run(name)
            for d in ('forward', 'backward'):
                j = out/'joint'/d/'result.json'
                status = json.loads(j.read_text())['status'] if j.exists() else 'NO_JOINT_STAGE'
                self.assertNotIn(status, ('PRISMATIC', 'REVOLUTE'), f'{name} {d}: a joint was identified in a static scene')

    def test_static_scene_has_no_runaway_points(self):
        # audit 2026-10-28: accepted inconsistent corrections + constant-velocity extrapolation gave 0.74 m and 4.4 m/s
        # on this exactly static scene. Bounds come from each run's own velocity prior, not from the result: no speed
        # beyond v99 (the 99% speed of the prior: 3.37 m/s for the uncalibrated 1 (m/s)^2, V99 when declared) and no
        # displacement beyond v99 times the sequence duration. The audited code broke both.
        from scipy.stats import chi2
        for name, v99 in (('static', float(np.sqrt(chi2.ppf(.99, 3)))), ('static_measured', float(np.sqrt(chi2.ppf(.99, 3)))), ('static_v99', V99)):
            ds, out = run(name)
            T = float(ds.times[-1]-ds.times[0])
            for d in ('forward', 'backward'):
                X = np.load(out/f'free_{d}'/'posterior_xyz.npy'); V = np.load(out/f'free_{d}'/'velocity.npy')
                disp = np.linalg.norm(X-X[0 if d == 'forward' else -1], axis=2)
                self.assertLess(float(np.linalg.norm(V, axis=2).max()), v99, f'{name} {d}')
                self.assertLess(float(disp.max()), v99*T, f'{name} {d}')

if __name__ == '__main__':
    unittest.main()
