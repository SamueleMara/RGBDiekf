"""Local-correlated runner: publication, journals, CLI, isolation from the global solvers (spec tests 16-20)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from tests.test_temporal_rgbd_pipeline import TinyDataset
from estimator.temporal_noise import load_warp_noise
from scene.config import load_config
from scene.canonical_subsample import subsample
from estimator.temporal_local_run import run_temporal_local_free
from utils.temporal_output import require_valid_frame

REPO = Path(__file__).resolve().parents[1]
LOCAL_PIPELINE_WITNESSES = []


def options(lag=3, warp='ideal'):
    cov, doc = load_warp_noise('/not-read', 'ideal') if warp == 'ideal' else (np.diag([1e-4, 1e-4, 1e-4, 1e-6]), {'assumption_class': 'unit-test'})
    return {'mode': 'local-correlated', 'lag': lag, 'warp_covariance': cov.tolist(), 'warp_provenance': doc, 'workspace_bytes': 128*1024**2,
            'local_chunk_bytes': 2*1024**2}


def run(tmp, device='cpu', direction='forward', lag=3, ids=(4, 9), warp='ideal'):  # ID 4 has temporal rows; 9 never does
    cfg = load_config()
    policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
    sub = {'factor': 1, 'method': 'spatial-stratified', 'seed': 0, 'boundary_preserve': False, 'report': False}
    return run_temporal_local_free(TinyDataset(), cfg, policy, direction, tmp, device, sub, options(lag, warp), [1, 2], list(ids), max_iters=5)


class LocalPipelineTests(unittest.TestCase):
    def test_end_to_end_publication_and_qualifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run(tmp)
            root = Path(tmp)
            self.assertEqual(result['status'], 'COMPLETE')
            man = json.loads((root/'run_manifest.json').read_text())
            self.assertEqual(man['temporal_mode'], 'local-correlated')
            self.assertEqual(man['cross_point_covariance'], 'block_diagonal_assumed_density')
            self.assertIn('BLOCK_DIAGONAL_ACROSS_POINTS', man['scientific_qualifier'])
            self.assertFalse(man['global_sparse_solver_used'])
            journal = json.loads((root/'completed_frames.json').read_text())
            self.assertEqual(journal['completed_frames'], [0, 1, 2, 3])
            for k in range(4):
                require_valid_frame(root, k)
            P = np.load(root/'posterior_covariance.npy')
            self.assertTrue(np.isfinite(P).all())
            self.assertTrue((np.linalg.eigvalsh(P) > 0).all())
            can = np.load(root/'canonical.npz')
            np.testing.assert_array_equal(can['point_id'], np.arange(18))
            trace = json.loads((root/'temporal/traced_points/4.json').read_text())
            observed = [f for f in trace['frames'] if f['rows']]
            self.assertTrue(observed, 'traced point must have measured frames in this fixture')
            blocks = observed[0]['local_posterior']['cross_with_current_state']
            self.assertTrue(any(b['kind'] == 'raw_sensor_error' and b['camera'] is not None for b in blocks))
            frame = json.loads((root/'temporal/frame_0002.json').read_text())
            self.assertEqual(frame['forbidden_key_recreations'], 0)
            self.assertGreater(frame['repeated_key_reuse_rows'], 0)
            self.assertLess(frame['joseph_vs_algebraic_max_correlation_units'], 1e-9)
            self.assertTrue((root/'temporal/index.html').exists())
            LOCAL_PIPELINE_WITNESSES.append({'test': 'tiny_end_to_end', 'frames': journal['completed_frames'],
                                             'accepted_points': [json.loads((root/f'temporal/frame_{k:04d}.json').read_text())['assimilated_points'] for k in (1, 2, 3)]})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_cpu(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            run(a, 'cpu')
            run(b, 'cuda')
            for name in ('posterior_xyz', 'velocity', 'posterior_covariance'):
                x, y = np.load(Path(a)/f'{name}.npy'), np.load(Path(b)/f'{name}.npy')
                np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-12)
            gap = float(np.abs(np.load(Path(a)/'posterior_covariance.npy')-np.load(Path(b)/'posterior_covariance.npy')).max())
            LOCAL_PIPELINE_WITNESSES.append({'test': 'cpu_cuda_parity', 'posterior_covariance_max_abs': gap})

    def test_never_enters_global_solvers_or_simplified_update(self):
        forbid = mock.Mock(side_effect=AssertionError('local-correlated must not use a global solver'))
        with mock.patch('estimator.temporal_cudss.CudaSparseFactor.__init__', forbid), \
             mock.patch('estimator.temporal_sparse_gpu.TemporalSoAGraph.__init__', forbid), \
             mock.patch('estimator.temporal_sparse_gpu.TemporalSoAGraph.marginalize_lag_boundary_sparse_gpu', forbid), \
             mock.patch('estimator.temporal_full_graph.TemporalWindowGraph.__init__', forbid), \
             mock.patch('estimator.temporal_simplified_iekf.update_points_temporal_simplified', forbid), \
             tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(tmp)['status'], 'COMPLETE')
        self.assertEqual(forbid.call_count, 0)
        source = (REPO/'estimator/temporal_local_run.py').read_text()+(REPO/'estimator/temporal_local_iekf.py').read_text()+(REPO/'estimator/temporal_local_covariance.py').read_text()
        for name in ('temporal_cudss', 'temporal_sparse_gpu', 'temporal_sparse_solve', 'temporal_sparse_marginalization', 'temporal_selected_inverse_gpu', 'temporal_full_graph'):
            self.assertNotIn(name, source)

    def test_lag_two_backward_and_nonideal_warp(self):
        for direction, lag, warp in (('forward', 2, 'ideal'), ('backward', 3, 'ideal'), ('forward', 3, 'config')):
            with self.subTest(direction=direction, lag=lag, warp=warp), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(run(tmp, direction=direction, lag=lag, warp=warp)['status'], 'COMPLETE')
                man = json.loads((Path(tmp)/'run_manifest.json').read_text())
                order = man['processing_order']
                if direction == 'backward':
                    self.assertEqual(order, [3, 2, 1, 0])
                expired = [json.loads((Path(tmp)/f'temporal/frame_{k:04d}.json').read_text())['expired_raw_keys'] for k in order[1:]]
                self.assertGreater(sum(expired), 0)
                dims = np.load(Path(tmp)/'local_dimension.npy')
                self.assertTrue((dims[order] >= 6).all())

    def test_failure_leaves_nan_tail_and_unpublished_frame(self):
        from estimator import temporal_local_iekf
        original = temporal_local_iekf.LocalFrameUpdate.posterior

        def fail_on_two(self, *a, **k):
            if self.k == 2:
                raise temporal_local_iekf.LocalCovarianceFailure('injected unit-test covariance failure')
            return original(self, *a, **k)
        with mock.patch.object(temporal_local_iekf.LocalFrameUpdate, 'posterior', fail_on_two), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(temporal_local_iekf.LocalCovarianceFailure):
                run(tmp)
            root = Path(tmp)
            self.assertEqual(json.loads((root/'completed_frames.json').read_text())['completed_frames'], [0, 1])
            self.assertTrue(np.isnan(np.load(root/'posterior_xyz.npy')[2:]).all())
            self.assertTrue(np.isnan(np.load(root/'posterior_covariance.npy')[2:]).all())
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'LOCAL_COVARIANCE_FAILURE')
            with self.assertRaises(Exception):
                require_valid_frame(root, 2)

    def test_cli_accepts_mode_and_forwards_batching_and_subsampling(self):
        import train
        ap, a = train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd', '--temporal-mode', 'local-correlated', '--temporal-local-chunk-mib', '64'])
        self.assertEqual(a.temporal_mode, 'local-correlated')
        with self.assertRaises(SystemExit):
            train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd', '--temporal-mode', 'simplified', '--temporal-local-chunk-mib', '64'])
        self.assertIn('estimator/temporal_local_iekf.py', train.STAGE_SOURCES['free_temporal_rgbd'])
        data = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
        if not (data/'USB_100109').is_dir():
            self.skipTest('dataset root unavailable for dry-run forwarding check')
        out = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_temporal_rgbd_critical.py'), '--dry-run', '--mode', 'local-correlated,simplified',
                              '--scenes', 'USB_100109', '--direction', 'forward', '--frame-limit', '3', '--pcd-subsample-factor', '8',
                              '--pcd-subsample-method', 'seeded-uniform', '--pcd-subsample-seed', '5', '--temporal-warp-noise', 'ideal',
                              '--temporal-local-chunk-mib', '96'], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = [line for line in out.stdout.splitlines() if 'train.py' in line]
        local = [line for line in lines if '--temporal-mode local-correlated' in line]
        simple = [line for line in lines if '--temporal-mode simplified' in line]
        self.assertEqual(len(local), 1)
        self.assertEqual(len(simple), 1)
        for line in local+simple:
            for flag in ('--pcd-subsample-factor 8', '--pcd-subsample-method seeded-uniform', '--pcd-subsample-seed 5', '--frame-limit 3',
                         '--temporal-lag-frames 3', '--temporal-warp-noise ideal'):
                self.assertIn(flag, line)
        self.assertIn('--temporal-local-chunk-mib 96.0', local[0])
        self.assertNotIn('--temporal-local-chunk-mib', simple[0])

    def test_seeded_uniform_selection_is_deterministic_identity_at_factor_one(self):
        rng = np.random.default_rng(2)
        xyz = rng.uniform(-1, 1, (4001, 3))
        src = np.c_[np.zeros(4001, int), rng.integers(0, 8, 4001), rng.integers(0, 640*480, 4001)]
        sel1, _ = subsample(xyz, src, 1, 'seeded-uniform', 0)
        np.testing.assert_array_equal(sel1, np.arange(4001))
        for f in (2, 4, 8):
            a, _ = subsample(xyz, src, f, 'seeded-uniform', 0)
            b, _ = subsample(xyz, src, f, 'seeded-uniform', 0)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(a), -(-4001//f))
            self.assertTrue(np.all(np.diff(a) > 0))
