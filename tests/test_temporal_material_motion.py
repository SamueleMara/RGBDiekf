"""Motion observability, diagnostics, optional terms and the runner/CLI of the material-following mode (6.E, 6.G).

Every number asserted here specifies the analytic fixture only; real-scene motion is evaluated post hoc against GT.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tests.material_fixtures import Scene, S, FOC, DEVICES, make_filter, run_frames
from estimator.temporal_material_run import position_information
from estimator.temporal_material_terms import resampling_discrepancy_variance

REPO = Path(__file__).resolve().parents[1]
PX = 2./FOC
PIXELS = [12*S+12, 12*S+19, 16*S+16, 19*S+12, 19*S+19]
MATERIAL_MOTION_WITNESSES = []


class MaterialMotionTests(unittest.TestCase):
    def test_textured_motion_full_rank_and_recovered(self):
        sc = Scene(d=(1.3*PX, -.6*PX, .01))
        frames = [sc.render(k, 'cpu') for k in range(2)]
        filt, _, base = make_filter(sc, 'cpu', PIXELS)
        st = run_frames(filt, frames, 1)[0]
        ev = torch.linalg.eigvalsh(position_information(st['lr'], st['update'].warp, len(PIXELS)))
        self.assertTrue(bool((ev[:, 0] > 1e-9*ev[:, 2]).all()))
        err = np.linalg.norm(st['s'][:, :3].numpy()-(base+sc.d), axis=1)
        self.assertLess(float(err.max()), .05*PX)
        MATERIAL_MOTION_WITNESSES.append({'textured_info_eig_min_over_max': float((ev[:, 0]/ev[:, 2]).min()), 'max_error_px': float(err.max()/PX)})

    def test_textureless_aperture_is_rank_deficient_and_invents_no_motion(self):
        """Uniform colour on a fronto-parallel plane translating in its own plane: the RGB-D data at every material sample
        is unchanged, the in-plane information is exactly zero, the estimate stays at the prediction (no invented
        velocity) and the posterior keeps the prior uncertainty in-plane while depth is still measured."""
        sc = Scene(d=(1.3*PX, 0., 0.), slope=(0., 0.), kind='textureless')
        frames = [sc.render(k, 'cpu') for k in range(3)]
        filt, _, base = make_filter(sc, 'cpu', PIXELS)
        steps = run_frames(filt, frames, 2)
        for k, st in enumerate(steps, 1):
            info = position_information(st['lr'], st['update'].warp, len(PIXELS))
            ev = torch.linalg.eigvalsh(info)
            self.assertTrue(bool((ev[:, 1] <= 1e-12*ev[:, 2]).all()))
            np.testing.assert_allclose(st['s'][:, :2].numpy(), base[:, :2], atol=1e-12)
            self.assertLess(float(st['s'][:, 3:].abs().max()), 1e-9)
            P = st['P'].numpy()
            prior = st['prior6'].numpy()
            # in-plane variance unchanged up to the source-ray x-z prior correlation (x = z * ray) conditioned by depth
            np.testing.assert_allclose(P[:, 0, 0], prior[:, 0, 0], rtol=1e-6)
            self.assertTrue(bool((P[:, 2, 2] < (1e-3 if k == 1 else 1.)*prior[:, 2, 2]).all()))   # depth still measured
        MATERIAL_MOTION_WITNESSES.append({'aperture_in_plane_info_eig_over_max': float((ev[:, 1]/ev[:, 2]).max()),
                                          'aperture_prior_std_kept_m': float(np.sqrt(P[0, 0, 0])), 'status': 'IMAGE_MOTION_UNOBSERVABLE (in-plane)'})

    def test_photometric_mismatch_is_diagnosed_by_predictive_nis(self):
        sc = Scene(d=(0., 0., 0.), gain={1: 1.08})
        frames = [sc.render(k, 'cpu') for k in range(2)]
        filt, _, base = make_filter(sc, 'cpu', PIXELS)
        st = run_frames(filt, frames, 1)[0]
        nrows = torch.bincount(st['lr'].rows.point, minlength=len(PIXELS))
        nis = st['solver']['predictive_nis']/(4*nrows).clamp_min(1)
        self.assertTrue(bool((nis > 100).all()))
        MATERIAL_MOTION_WITNESSES.append({'photometric_gain_1.08_predictive_NIS_per_dof_min': float(nis.min()),
                                          'static_drift_m_under_mismatch': float(np.abs(st['s'][:, :3].numpy()-base).max())})

    def test_pyramid_initializer_recovers_and_keeps_static(self):
        for d in ((3*PX, -1*PX, 0.), (0., 0., 0.)):
            sc = Scene(d=d)
            frames = [sc.render(k, 'cpu') for k in range(2)]
            filt, _, base = make_filter(sc, 'cpu', PIXELS, pyramid=2)
            st = run_frames(filt, frames, 1)[0]
            rep = st['solver']['coarse_to_fine']
            self.assertEqual([lv['level'] for lv in rep['levels']], [2, 1])
            err = np.linalg.norm(st['s'][:, :3].numpy()-(base+np.asarray(d)), axis=1)
            self.assertLess(float(err.max()), .05*PX if any(d) else 1e-12)
            MATERIAL_MOTION_WITNESSES.append({'pyramid_d_px': [x/PX for x in d], 'coarse_start_kept': rep.get('coarse_start_kept_points'),
                                              'level_reports': [{k: v for k, v in lv.items() if k != 'iterations'} for lv in rep['levels']]})

    def test_resampling_discrepancy_is_the_bilinear_remainder(self):
        """On F(u,v) = u^2 (F_uu = 2, F_vv = 0) the bilinear error at fraction du is du(1-du); the data-derived variance is
        exactly its square, and it vanishes at integer coordinates."""
        yy, xx = torch.meshgrid(torch.arange(S, dtype=torch.float64), torch.arange(S, dtype=torch.float64), indexing='ij')
        F = (xx/S)**2
        fr = SimpleNamespace(H=S, W=S, rgb=torch.stack((F, F, F), -1)[None], depth=(1+F)[None], mask=torch.ones((1, S, S), dtype=torch.bool))
        fr.raw_valid = fr.mask
        for du in (0., .25, .5):
            q = torch.tensor([[10+du, 12.]], dtype=torch.float64)
            var, ok = resampling_discrepancy_variance(fr, torch.tensor([0]), q, torch.floor(q).long())
            self.assertTrue(bool(ok.all()))
            want = (du*(1-du)*2/S**2/2)**2
            np.testing.assert_allclose(var[0].numpy(), [want]*4, rtol=1e-12, atol=1e-30)

    def test_resampling_term_keeps_static_exact_and_tracks(self):
        for d in ((1.3*PX, 0., 0.), (0., 0., 0.)):
            sc = Scene(d=d)
            frames = [sc.render(k, 'cpu') for k in range(2)]
            filt, _, base = make_filter(sc, 'cpu', PIXELS, resampling='second-difference')
            st = run_frames(filt, frames, 1)[0]
            err = np.linalg.norm(st['s'][:, :3].numpy()-(base+np.asarray(d)), axis=1)
            self.assertLess(float(err.max()), .05*PX if any(d) else 1e-12)


class MaterialPipelineTests(unittest.TestCase):
    def test_end_to_end_runner_journal_and_manifest(self):
        from tests.test_temporal_rgbd_pipeline import TinyDataset
        from tests.test_temporal_local_pipeline import options
        from scene.config import load_config
        from estimator.temporal_material_run import run_temporal_material_free
        from utils.temporal_output import require_valid_frame
        cfg = load_config()
        policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
        sub = {'factor': 1, 'method': 'spatial-stratified', 'seed': 0, 'boundary_preserve': False, 'report': False}
        for device in DEVICES:
            with tempfile.TemporaryDirectory() as tmp:
                opt = {**options(), 'mode': 'material-following', 'material_resampling': 'none', 'material_pyramid_levels': 1}
                result = run_temporal_material_free(TinyDataset(), cfg, policy, 'forward', tmp, device, sub, opt, [1, 2], [4], max_iters=5)
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertEqual(result['mode'], 'material-following')
                root = Path(tmp)
                man = json.loads((root/'run_manifest.json').read_text())
                self.assertEqual(man['temporal_mode'], 'material-following')
                self.assertIn('PROPOSED_CHANGE_OF_TEMPORAL_OBSERVATION_DOMAIN', man['observation_domain'])
                self.assertFalse(man['global_sparse_solver_used'])
                self.assertEqual(json.loads((root/'completed_frames.json').read_text())['completed_frames'], [0, 1, 2, 3])
                for k in range(4):
                    require_valid_frame(root, k)
                P = np.load(root/'posterior_covariance.npy')
                self.assertTrue((np.linalg.eigvalsh(P) > 0).all())
                for name in ('material_rows', 'material_pixel_shift_max_px', 'material_target_pixel_changed', 'material_info_eig_min'):
                    self.assertTrue((root/f'{name}.npy').exists())
                fr = json.loads((root/'temporal/frame_0002.json').read_text())
                self.assertIn('material', fr)
                self.assertIn('identity_counters', fr['material'])

    def test_cli_mode_flags_and_forwarding(self):
        import train
        _, a = train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd', '--temporal-mode', 'material-following',
                            '--temporal-material-pyramid-levels', '2', '--temporal-material-resampling', 'second-difference', '--temporal-local-chunk-mib', '64'])
        self.assertEqual((a.temporal_mode, a.temporal_material_pyramid_levels, a.temporal_material_resampling), ('material-following', 2, 'second-difference'))
        for bad in (['--temporal-mode', 'local-correlated', '--temporal-material-pyramid-levels', '2'],
                    ['--temporal-mode', 'material-following', '--temporal-material-pyramid-levels', '9']):
            with self.assertRaises(SystemExit):
                train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd']+bad)
        self.assertIn('estimator/temporal_material_iekf.py', train.STAGE_SOURCES['free_temporal_rgbd'])
        data = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
        if not (data/'USB_100109').is_dir():
            self.skipTest('dataset root unavailable for dry-run forwarding check')
        out = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_temporal_rgbd_critical.py'), '--dry-run', '--mode', 'material-following,local-correlated',
                              '--scenes', 'USB_100109', '--direction', 'forward', '--frame-limit', '3', '--pcd-subsample-factor', '8',
                              '--pcd-subsample-method', 'seeded-uniform', '--temporal-warp-noise', 'ideal', '--temporal-material-pyramid-levels', '2'],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = [l for l in out.stdout.splitlines() if 'train.py' in l]
        mat = [l for l in lines if '--temporal-mode material-following' in l]
        loc = [l for l in lines if '--temporal-mode local-correlated' in l]
        self.assertEqual((len(mat), len(loc)), (1, 1))
        self.assertIn('--temporal-material-pyramid-levels 2', mat[0])
        self.assertIn('--temporal-material-resampling none', mat[0])
        self.assertNotIn('--temporal-material', loc[0])

    def test_never_enters_global_solvers(self):
        from unittest import mock
        sc = Scene(d=(1.3*PX, 0., 0.))
        frames = [sc.render(k, 'cpu') for k in range(2)]
        with mock.patch('estimator.temporal_sparse_nonlinear.solve_nonlinear_sparse_iekf_gpu', side_effect=AssertionError('global solver')), \
             mock.patch('estimator.temporal_simplified_iekf.update_points_temporal_simplified', side_effect=AssertionError('simplified')):
            filt, _, base = make_filter(sc, 'cpu', PIXELS)
            run_frames(filt, frames, 1)


if __name__ == '__main__':
    unittest.main()
