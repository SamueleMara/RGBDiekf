"""`--temporal-gradient-guide replace` (spec 7.8-7.10): residual/Jacobian parity with material-following on fixed support,
filter equivalence, first-depth occlusion, no extra raw variables, CLI."""
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from tests.material_fixtures import Scene, S, FOC, DEVICES, make_filter, run_frames
from tests.test_temporal_material_association import PIXELS
from tests.test_temporal_material_warp import rows_for
from estimator.temporal_material_terms import material_prediction
from estimator.temporal_gradient_iekf import gradient_prediction, clean_bilinear

REPO = Path(__file__).resolve().parents[1]
PX = 2./FOC
GRADIENT_IEKF_WITNESSES = []


class GradientIEKFTests(unittest.TestCase):
    def test_residual_and_jacobian_parity_with_material_following(self):
        rng = np.random.default_rng(4)
        worst = 0.
        for device in DEVICES:
            for d in ((1.3*PX, 0., 0.), (-3*PX, 1.*PX, .02), (.4*PX, -.9*PX, -.01)):
                sc = Scene(d=d, slope=(.2, -.1))
                pix0 = [15*S+14, 17*S+18]
                x_ref = np.array([sc.point(0, p) for p in pix0])
                x_t = x_ref+np.asarray(d)+rng.normal(0, .002, x_ref.shape)
                p_ref = {0: pix0, 1: [16*S+9, 18*S+13]}
                rows, _, _, _, r = rows_for(device, sc, x_ref, x_t, p_ref)
                eta = torch.tensor(rng.normal(0, 1e-3, (len(rows), 4, 4)), device=device)    # nonzero latent raw errors
                clean_cur = rows.current_values-eta
                clean_ref = rows.reference_values-torch.tensor(rng.normal(0, 1e-3, (len(rows), 4)), device=device)
                f0, Ht0, Hr0, w0 = material_prediction(rows, clean_cur, clean_ref)
                rows.x_t = torch.tensor(x_t, device=device)[rows.point]
                f1, Ht1, Hr1, w1 = gradient_prediction(r, rows, clean_cur, clean_ref)
                for a, b in ((f0, f1), (Ht0, Ht1), (Hr0, Hr1), (w0, w1)):
                    np.testing.assert_allclose(b.cpu().numpy(), a.cpu().numpy(), rtol=1e-12, atol=1e-12)
                    worst = max(worst, float((a-b).abs().max()))
        GRADIENT_IEKF_WITNESSES.append({'replace_vs_material_max_abs': worst})

    def test_clean_bilinear_matches_numpy(self):
        vals = torch.tensor(np.random.default_rng(5).normal(size=(3, 4, 4)))
        cell = torch.tensor([[3, 4], [0, 0], [7, 2]])
        q = cell.double()+torch.tensor([[.3, .8], [.0, .5], [.99, .01]])
        h, g, w = clean_bilinear(vals, q, cell)
        for m in range(3):
            u, v = float(q[m, 0]-cell[m, 0]), float(q[m, 1]-cell[m, 1])
            F = vals[m].numpy()
            want = (1-u)*(1-v)*F[0]+u*(1-v)*F[1]+(1-u)*v*F[2]+u*v*F[3]
            np.testing.assert_allclose(h[m].numpy(), want, atol=1e-14)
            np.testing.assert_allclose(g[m, :, 0].numpy(), (1-v)*(F[1]-F[0])+v*(F[3]-F[2]), atol=1e-14)
            np.testing.assert_allclose(g[m, :, 1].numpy(), (1-u)*(F[2]-F[0])+u*(F[3]-F[1]), atol=1e-14)

    def test_replace_filter_equals_material_following(self):
        for device in DEVICES:
            for d in ((1.3*PX, -.5*PX, .01), (3*PX, 0., 0.), (0., 0., 0.)):
                sc = Scene(d=d)
                frames = [sc.render(k, device) for k in range(3)]
                out = {}
                for mode in ('off', 'replace'):
                    filt, _, base = make_filter(sc, device, PIXELS)
                    filt.gradient_guide = mode
                    out[mode] = (run_frames(filt, frames, 2), filt)
                for a, b in zip(out['off'][0], out['replace'][0]):
                    np.testing.assert_allclose(b['s'].cpu().numpy(), a['s'].cpu().numpy(), atol=1e-12)
                    np.testing.assert_allclose(b['P'].cpu().numpy(), a['P'].cpu().numpy(), rtol=1e-9, atol=1e-18)
                    self.assertTrue(torch.equal(a['update'].assimilated, b['update'].assimilated))
                    self.assertEqual(len(a['lr'].rows), len(b['lr'].rows))
                    self.assertTrue(bool((torch.linalg.eigvalsh(b['P']) > 0).all()))
                Lo, Lr = out['off'][1].bank.layout, out['replace'][1].bank.layout
                self.assertTrue(torch.equal(Lo.codes(out['off'][1].codec), Lr.codes(out['replace'][1].codec)))   # same raw variables
                GRADIENT_IEKF_WITNESSES.append({'device': device, 'd_px': [x/PX for x in d],
                                                'max_state_diff': max(float((a['s']-b['s']).abs().max()) for a, b in zip(out['off'][0], out['replace'][0]))})

    def test_replace_keeps_first_depth_occlusion(self):
        sc = Scene(d=(1.3*PX, 0., 0.))
        frames = [sc.render(k, 'cpu') for k in range(2)]
        frames[1].depth[0, 10:24, 10:26] -= .4
        filt, _, _ = make_filter(sc, 'cpu', [16*S+16])
        filt.gradient_guide = 'replace'
        st = run_frames(filt, frames, 1)[0]
        self.assertEqual(int(st['lr'].reason[0, 0]), 8)
        self.assertNotIn(0, st['lr'].rows.camera.tolist())

    def test_cli(self):
        import train
        _, a = train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd', '--temporal-mode', 'material-following', '--temporal-gradient-guide', 'replace'])
        self.assertEqual(a.temporal_gradient_guide, 'replace')
        with self.assertRaises(SystemExit):
            train.parse(['--scene', '/x', '--observation-model', 'temporal-rgbd', '--temporal-mode', 'local-correlated', '--temporal-gradient-guide', 'diagnostic'])
        for f in ('estimator/temporal_gradient_iekf.py', 'estimator/temporal_motion_observability.py', 'estimator/temporal_part_twist.py'):
            self.assertIn(f, train.STAGE_SOURCES['free_temporal_rgbd'])
        opts = train.parse(['--scene', '/x'])[0]._option_string_actions
        self.assertFalse(any('gt' in o.lower() or 'oracle' in o.lower() for o in opts))       # no GT / oracle option in train.py
        data = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
        if not (data/'USB_100109').is_dir():
            self.skipTest('dataset root unavailable for dry-run forwarding check')
        out = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_temporal_rgbd_critical.py'), '--dry-run', '--mode', 'material-following,simplified',
                              '--scenes', 'USB_100109', '--direction', 'forward', '--frame-limit', '3', '--temporal-warp-noise', 'ideal',
                              '--temporal-gradient-guide', 'diagnostic'], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = [l for l in out.stdout.splitlines() if 'train.py' in l]
        self.assertIn('--temporal-gradient-guide diagnostic', [l for l in lines if 'material-following' in l][0])
        self.assertNotIn('--temporal-gradient-guide', [l for l in lines if '--temporal-mode simplified' in l][0])


if __name__ == '__main__':
    unittest.main()
