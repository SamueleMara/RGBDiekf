"""Gate 0 of RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md through train.py on a written scene (tests/temporal_fixtures):
G0.2 targeted tracing on/off gives bit-identical temporal outputs; G0.4 removing / adding ground-truth files in the scene
folder changes nothing (no GT on the inference call graph); the CLI guards (opt-in only, F1 only, no speed percentile, no
downstream without an adapter). G0.1 (original F1 bitwise) and G0.3 (existing suites) are checked on the real runs and the
full test suite (reports/temporal_phase1_ieks.md)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import train
from tests.projective_fixtures import write_scene
from tests.temporal_fixtures import fronto_cameras, AffinePlate, AnalyticDataset
from tests.test_temporal_phase1_static import RING

BASE = ['--fps', '30', '--direction', 'forward', '--stop-after', 'free', '--free-solver', 'temporal-ieks', '--debug', 'none', '--no-evaluate', '--no-qa',
        '--sensor-noise', 'config', '--debug-frames', '']


def arrays(run):
    return {f.name: np.load(f) for f in sorted((Path(run)/'free_forward_temporal').glob('*.npy'))}


class TemporalPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6), motion=lambda t: np.array([.2, -.1, .05])*t)], np.arange(6)/30.)
        cls.scene = write_scene(ds, cls.root/'scene')

    def run_train(self, name, scene=None, extra=()):
        out = self.root/name
        train.main(['--scene', str(scene or self.scene), '--output', str(out), *BASE, '--pcd-subsample-factor', '20', *extra])
        return out

    def test_G0_2_trace_on_off_identical(self):
        a = self.run_train('trace_off'); b = self.run_train('trace_on', extra=('--debug-point-ids', '0,3,17'))
        self.assertTrue((b/'free_forward_temporal'/'trace_points.json').exists())
        self.assertFalse((a/'free_forward_temporal'/'trace_points.json').exists())
        A, B = arrays(a), arrays(b)
        self.assertEqual(set(A), set(B))
        for k in A: np.testing.assert_array_equal(A[k], B[k], err_msg=k)
        st = json.loads((a/'free_forward_temporal'/'status.json').read_text())
        self.assertEqual(st['status'], 'COMPLETE')
        self.assertFalse((a/'free_forward').exists())                                    # the old free output is not written

    def test_G0_4_ground_truth_files_never_read(self):
        with_gt = self.root/'scene_with_gt'
        shutil.copytree(self.scene, with_gt)
        (with_gt/'pointcloud').mkdir(); (with_gt/'pointcloud'/'joint_metadata.json').write_text('{"type": "prismatic", "axis_vector": [1, 0, 0]}')
        (with_gt/'metadata'/'frame_values.csv').write_text('frame_index,joint\n'+''.join(f'{k},{.1*k}\n' for k in range(6)))
        a = self.run_train('no_gt'); b = self.run_train('with_gt', scene=with_gt)
        A, B = arrays(a), arrays(b)
        for k in A: np.testing.assert_array_equal(A[k], B[k], err_msg=k)

    def test_cli_guards(self):
        bad = [['--stop-after', 'coherence'], ['--direction', 'both'], ['--velocity-prior-speed99', '0.5']]
        for extra in bad:
            args = [x for x in BASE]
            for flag in extra[::2]:
                i = args.index(flag) if flag in args else None
                if i is not None: del args[i:i+2]
            with self.assertRaises(SystemExit, msg=str(extra)):
                train.main(['--scene', str(self.scene), '--output', str(self.root/'never'), *args, *extra])


if __name__ == '__main__':
    unittest.main()
