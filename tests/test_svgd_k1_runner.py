"""Gate 0/2 + runner contract of the joint K = 1 pipeline (spec 7, 9): output contract, K != 1 rejection, camera/point-order invariance,
bounded confidence under uniform reference duplication, calibration reproducibility, no silent covariance floors. CPU, tiny fixtures."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tests.svgd_fixtures import SyntheticDataset, record

REPO = Path(__file__).resolve().parents[1]


def tiny_cfg(path):
    c = json.loads((REPO/'configs/svgd_joint_k1_overnight.json').read_text())
    c['svgd']['value'].update(particles=3, iterations=2); c['map']['value'].update(iterations=2); Path(path).write_text(json.dumps(c)); return path


def args(out, cfg, **kw):
    a = SimpleNamespace(scene='/synthetic', output=out, config=cfg, num_moving_parts=1, device='cpu', frame_limit=None, fps=None, dt_seconds=None,
                        timestamps_file=None, noise_config=None, sensor_noise='measured')
    for k, v in kw.items(): setattr(a, k, v)
    return a


class Perm:
    def __init__(s, base, p): s.b, s.p = base, p; s.cameras = [base.cameras[i] for i in p]
    def __getattr__(s, k): return getattr(s.b, k)
    def observation(s, k, c): return s.b.observation(k, s.p[c])


class K1RunnerTests(unittest.TestCase):
    def test_output_contract_and_identity_frame(self):
        from estimator.svgd_k1_run import run
        with tempfile.TemporaryDirectory() as t:
            cfg = tiny_cfg(Path(t)/'c.json'); out = Path(t)/'run'
            self.assertEqual(run(args(out, cfg), data=SyntheticDataset(K=3, motion='translate', speed=.15, depth_noise=5e-4, n_cams=2)), 0)
            sv = out/'svgd_joint_k1'
            for f in ('gamma_provisional.npy', 'evidence_static.npy', 'evidence_moving_conditional.npy', 'log_evidence_ratio.npy', 'identifiability_status.npy',
                      'pose_particles.npy', 'moving_seed_indices.npy', 'map_pose.npy', 'calibration_provenance.json', 'optimizer_diagnostics.json', 'status.json'):
                self.assertTrue((sv/f).is_file(), f)
            P = np.load(sv/'pose_particles.npy'); self.assertTrue(np.array_equal(P[:, 0], np.broadcast_to(np.eye(4), P[:, 0].shape)))
            st = json.loads((sv/'status.json').read_text()); self.assertEqual(st['probability_semantics'], 'UNCALIBRATED_APPROXIMATION'); self.assertEqual(st['stage'], 'DONE')
            self.assertFalse((sv/'gamma_moving.npy').exists())                                   # never published as calibrated
            g = np.load(sv/'gamma_provisional.npy'); self.assertTrue(np.isfinite(g).all() and ((g >= 0) & (g <= 1)).all())

    def test_k_not_one_rejected(self):
        r = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_svgd_joint_k1.py'), '--scene', '/x', '--num-moving-parts', '2'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0); self.assertIn('NOT_IMPLEMENTED_K_GT_1', r.stderr+r.stdout)

    def test_camera_and_point_order_invariance(self):
        """Permuting the camera loading order permutes every cloud and the reference catalogue; F at a fixed motion must not change."""
        from estimator.svgd_k1_run import build_model
        cfg = json.loads((REPO/'configs/svgd_joint_k1_overnight.json').read_text()); a = SimpleNamespace(noise_config=None, sensor_noise='measured')
        Fs = []
        for data in (SyntheticDataset(K=3, motion='translate', n_cams=3), Perm(SyntheticDataset(K=3, motion='translate', n_cams=3), [2, 0, 1])):
            m = build_model(cfg, data, a, torch.device('cpu'), verbose=False)
            nu = torch.zeros(2, 6, dtype=torch.float64); nu[:, 0] = .15
            Fs.append(m['post'].evaluate(nu, torch.tensor(-.5, dtype=torch.float64), grad=False)['F'])
        # reduction order over ~1e3 terms with 6x6 solves of condition up to ~1e9 (nm-scale depth sigma): measured 6.5e-9 relative
        self.assertLess(abs(Fs[0]-Fs[1]), 1e-7*abs(Fs[0]))
        record('K1', test='camera_order_invariance', rel=abs(Fs[0]-Fs[1])/abs(Fs[0]))

    def test_uniform_duplication_does_not_inflate_confidence(self):
        """Gate 2: duplicating every reference point (no new physical information) keeps n_eff and splits a_eff; the evidence gain
        F(T) - F(I) must stay bounded (not double)."""
        from estimator.svgd_k1_observation import Background, FrameKDE, representation_scales, calibrate_rho, choose_kappa, kernel_sum
        from estimator.svgd_k1_motion import MotionPrior
        from estimator.svgd_k1_joint_posterior import JointPosterior
        from scene.svgd_k1_cloud_adapter import from_dataset
        from scene.config import load_config
        seq, _ = from_dataset(SyntheticDataset(K=3, motion='translate', depth_noise=5e-4, n_cams=2), load_config())
        gains = []
        for rep in (1, 2):
            P0 = seq.reference; xyz, rgb, cov = (np.concatenate([a]*rep) for a in (P0.xyz, P0.rgb, P0.cov_xyz))
            bg = Background(xyz, rgb, cov, P0.cov_rgb); frames = []
            for fr in seq.frames[1:]:
                s2, LC = representation_scales(fr.xyz, fr.rgb, fr.cov_rgb); k = FrameKDE(fr.xyz, fr.rgb, fr.cov_xyz, s2, LC); k.build_grid(choose_kappa(k, bg, .3, 1e-6, 40.)); frames.append((k, bg, .3))
            s2, LC = representation_scales(P0.xyz, P0.rgb, P0.cov_rgb)
            r0, _ = kernel_sum(xyz, rgb, s2*np.eye(3), LC, 6.); N = len(xyz); neff = N*N/float(r0.sum()); a_eff = neff*(1/r0)/(1/r0).sum()
            t = lambda x: torch.as_tensor(x, dtype=torch.float64)
            mu = torch.cat([t(xyz), t(rgb)], 1); S = torch.zeros(N, 6, 6, dtype=torch.float64); S[:, :3, :3] = t(cov); S[:, 3:, 3:] = t(P0.cov_rgb)
            post = JointPosterior(frames, mu, torch.linalg.cholesky(S), a_eff, MotionPrior(np.diff(seq.times), np.diag([1, 1, 1, 4, 4, 4.]), np.diag([.01]*3+[1]*3)), (1., 1.))
            post.set_static(); nu = torch.zeros(2, 6, dtype=torch.float64); nu[:, 0] = .15; h = torch.tensor(0., dtype=torch.float64)
            gains.append(post.evaluate(nu, h, grad=False)['F']-post.evaluate(torch.zeros_like(nu), h, grad=False)['F'])
        self.assertLess(abs(gains[1]), 1.5*abs(gains[0])+1.)
        record('K1', test='duplication_bounded', gain_original=gains[0], gain_duplicated=gains[1])

    def test_calibration_reproducible_and_no_silent_floor(self):
        from estimator.svgd_k1_observation import Background, FrameKDE, representation_scales, calibrate_rho
        from scene.svgd_k1_cloud_adapter import from_dataset
        from scene.config import load_config
        seq, _ = from_dataset(SyntheticDataset(K=2, motion='none', depth_noise=5e-4, n_cams=2), load_config()); fr = seq.frames[1]; P0 = seq.reference
        bg = Background(P0.xyz, P0.rgb, P0.cov_xyz, P0.cov_rgb); s2, LC = representation_scales(fr.xyz, fr.rgb, fr.cov_rgb)
        r1 = calibrate_rho(fr, s2, LC, bg, 1e-6, 40., edge=8*math.sqrt(s2)); r2 = calibrate_rho(fr, s2, LC, bg, 1e-6, 40., edge=8*math.sqrt(s2))
        self.assertEqual(r1['rho'], r2['rho']); self.assertTrue(0 < r1['rho'] < 1)
        self.assertTrue(math.isfinite(float(bg.log(torch.tensor([[10., 10, 10, 1, 1, 1]], dtype=torch.float64)))))   # b_0 > 0 everywhere
        with self.assertRaises(FloatingPointError): FrameKDE(fr.xyz[:5], fr.rgb[:5], -np.tile(np.eye(3), (5, 1, 1)), -1., LC)   # no jitter
        record('K1', test='rho_cv', rho=r1['rho'], dispersion=r1['dispersion_iqr'])


if __name__ == '__main__':
    unittest.main()
