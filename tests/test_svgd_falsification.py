"""Falsification-audit contract (CODEX_RGBDiekf_SVGD_Falsification_And_Root_Cause_2026-11-03.md section 11).

What these tests establish: the post-hoc audit evaluates EXACTLY the production objective (rebuild parity, archived line-search parity),
its decompositions close, its finite differences close, the counterfactual arms change only what they claim, GT never reaches inference,
and deliberately injected bugs are detected. They do NOT establish that the model discriminates motion: that is the audit's measured result."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from tests.svgd_fixtures import SyntheticDataset, record

REPO = Path(__file__).resolve().parents[1]


def make_run(root, perm=None, frames=3):
    from tools.svgd_synthetic_run import main
    out = Path(root)/('run' if perm is None else 'perm')
    if perm is None: main(['--motion', 'translate', '--frames', str(frames), '--n-cams', '3', '--output', str(out), '--device', 'cpu'])
    else:
        from tests.svgd_fixtures import runner_args
        from estimator.svgd_support_run import run
        class Perm:
            def __init__(s, base, p): s.b, s.p = base, p; s.cameras = [base.cameras[i] for i in p]
            def __getattr__(s, k): return getattr(s.b, k)
            def observation(s, k, c): return s.b.observation(k, s.p[c])
        spec = json.loads((Path(root)/'run'/'synthetic_fixture.json').read_text())
        run(runner_args(out, str(REPO/'configs/svgd_support_experimental.json'), device='cpu', sensor_noise='measured', debug='none', stop_after_step='STEIN_PROPOSAL',
                        save_stein_diagnostics=True), data=Perm(SyntheticDataset(**spec), perm))
    return out


class FalsificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.run_dir = make_run(cls.tmp.name)
        spec = json.loads((cls.run_dir/'synthetic_fixture.json').read_text()); cls.data = SyntheticDataset(**spec)
        from tools.audit_svgd_falsification import Model, gt_oracle
        cls.m = Model(cls.run_dir, cls.data, 'cpu'); cls.Xgt, cls.mv, cls.gtv = gt_oracle(cls.m, synthetic=cls.data)
        cls.saved = dict(np.load(cls.run_dir/'svgd_support/stein_diagnostics.npz')); cls.saved['eta0'] = cls.m.status['gates']['STEIN_DIAGNOSTICS'][0]['eta0_m2']

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_rebuild_and_archived_line_search_parity(self):
        from tools.audit_svgd_falsification import gate_f4
        self.assertTrue(self.m.valid, [c for c in self.m.checks if not c['ok']])
        rows, f4 = gate_f4(self.m, self.Xgt, self.mv, self.saved)
        self.assertTrue(f4['audit_valid']); self.assertLess(f4['A_B_reconstruction_max_rel_error'], 1e-9)
        record('FALS', test='archived_line_search_parity', max_rel=f4['A_B_reconstruction_max_rel_error'])

    def test_frame0_identity_and_gt_verification(self):
        from tools.audit_svgd_falsification import gate_f1_f2
        self.assertTrue(np.array_equal(self.Xgt[0], self.m.canon.xyz0)); self.assertEqual(self.gtv['status'], 'GT_TRAJECTORY_VERIFIED')
        self.assertEqual(self.gtv['moving_members'], int(self.mv.sum())); self.assertGreater(self.mv.sum(), 0)
        rows, f12 = gate_f1_f2(self.m, self.Xgt, self.mv)
        self.assertEqual(f12['dE_obs_GT_minus_S_t'][0], 0.)                                   # frame 0 identical candidate -> exactly 0
        self.assertEqual(rows[2]['alpha'], 0); self.assertEqual(rows[2]['E_obs'], rows[0]['E_obs'])   # I(0) == S exactly

    def test_time_decomposition_closure_and_kkt(self):
        from tools.audit_svgd_falsification import time_parts, lift_positions
        g = torch.Generator().manual_seed(0); S = self.m.S0.clone(); S += torch.randn(S.shape, generator=g, dtype=torch.float64)*1e-3
        tp = time_parts(self.m.temporal, S)
        self.assertAlmostEqual(tp['total']/self.m.temporal.energy(S), 1., places=12)
        self.assertAlmostEqual(sum(tp['anchor_by_source_eigendirection_small_to_large'])/tp['anchor_xx'], 1., places=9)
        self.assertNotEqual(sum(i['xv'] for i in tp['intervals']), 0.)                          # cross terms present and counted
        X = torch.as_tensor(self.Xgt, dtype=torch.float64).transpose(0, 1)
        Sl, kkt = lift_positions(self.m, X); self.assertLess(kkt, 1e-9)

    def test_finite_difference_closure(self):
        from tools.audit_svgd_falsification import fd_closure
        Phi = torch.as_tensor(self.saved['Phi']); P = torch.zeros_like(self.m.S0); P[:, :, :3] = Phi
        an = float((self.m.obj.grad(self.m.S0)*P).sum())
        fd = fd_closure(lambda h: self.m.obj.value(self.m.S0+h*P)['total'], an, float(self.saved['eta0']))
        self.assertLess(fd['best_rel_error'], 1e-6)
        record('FALS', test='fd_closure_E_total', best_rel_error=fd['best_rel_error'])

    def test_arms_C_D_leave_X0_and_change_only_the_anchor_displacement(self):
        Phi = torch.as_tensor(self.saved['Phi']); eta = float(self.saved['eta0'])
        from tools.audit_svgd_falsification import time_parts
        dX = eta*Phi; dX[:, 0] = 0
        C = self.m.S0.clone(); C[:, :, :3] += dX
        D = C.clone(); D[:, :, 3:] += self.m.temporal.velocity_lift(self.m.S0, dX)
        for S in (C, D): self.assertTrue(torch.equal(S[:, 0, :3], self.m.S0[:, 0, :3]))
        self.assertEqual(time_parts(self.m.temporal, C)['anchor_xx'], 0.)
        self.assertEqual(time_parts(self.m.temporal, D)['anchor_xx'], 0.)                     # lift moves velocities only

    def test_camera_order_invariance_of_candidate_energies(self):
        from tools.audit_svgd_falsification import Model, gt_oracle, gate_f1_f2
        run_b = make_run(self.tmp.name, perm=[2, 0, 1])
        spec = json.loads((self.run_dir/'synthetic_fixture.json').read_text())
        class Perm:
            def __init__(s, base, p): s.b, s.p = base, p; s.cameras = [base.cameras[i] for i in p]
            def __getattr__(s, k): return getattr(s.b, k)
            def observation(s, k, c): return s.b.observation(k, s.p[c])
        mb = Model(run_b, Perm(SyntheticDataset(**spec), [2, 0, 1]), 'cpu')
        Xb, mvb, _ = gt_oracle(mb, synthetic=Perm(SyntheticDataset(**spec), [2, 0, 1]))
        _, fa = gate_f1_f2(self.m, self.Xgt, self.mv); _, fb = gate_f1_f2(mb, Xb, mvb)
        # tolerance relative to the ENERGY scale (|E_obs| ~ 1e4-1e5): a GT-minus-S difference is a cancellation of two large sums whose
        # Gamma is accumulated in a different camera order (ill-conditioned Cholesky), so 1e-12 x |E_obs| is the reduction-order budget
        scale = abs(self.m.E0['E_obs'])
        self.assertLess(abs(fa['dE_obs_GT_minus_S']-fb['dE_obs_GT_minus_S']), 1e-11*scale*1e3)
        self.assertLess(abs(fa['dE_total_star_GT_minus_S']-fb['dE_total_star_GT_minus_S']), 1e-11*scale*1e3)
        record('FALS', test='camera_order_invariance', abs_diff=abs(fa['dE_obs_GT_minus_S']-fb['dE_obs_GT_minus_S']), energy_scale=scale)

    def test_gt_leak_sentinel_and_artifact_immutability(self):
        banned = ('tools.audit', 'audit_temporal_dynamic_correspondence', 'evaluate_projective_gt', 'ground_truth', 'svgd_falsification')
        for f in list((REPO/'estimator').glob('svgd_*.py'))+[REPO/'tools/run_svgd_support.py', REPO/'scene/pointcloud_sequence.py']:
            tree = ast.parse(f.read_text())
            mods = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            names = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
            self.assertFalse([x for x in mods+names if any(b in x for b in banned)], f.name)
        sv = self.run_dir/'svgd_support'; before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sv.iterdir() if p.is_file()}
        import subprocess, sys
        r = subprocess.run([sys.executable, '-B', str(REPO/'tools/audit_svgd_falsification.py'), '--run', str(self.run_dir), '--synthetic', '--device', 'cpu', '--skip-sensitivity'],
                           capture_output=True, text=True)
        self.assertIn(r.returncode, (0, 2), r.stderr[-2000:])
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sv.iterdir() if p.is_file()}
        self.assertEqual(before, after); self.assertFalse(list(self.run_dir.rglob('p_dynamic.npy')))
        self.assertTrue((self.run_dir/'posthoc/svgd_falsification/energy_oracle.csv').is_file())

    def test_injected_bugs_are_detected(self):
        """Mutations: Gamma normalizer dropped, feature seed changed, velocity-lift sign flipped -> rebuild / KKT checks fail."""
        from tools.audit_svgd_falsification import Model, lift_positions
        from estimator import svgd_feature_measurement as F
        orig = F.FeatureGaussian.nll
        with mock.patch.object(F.FeatureGaussian, 'nll', lambda s, e: .5*s.quad(e)):
            self.assertFalse(Model(self.run_dir, self.data, 'cpu').valid)
        cfg = json.loads(Path(self.m.man['arguments']['config']).read_text()); cfg['features']['fixed_seed']['value'] += 1
        with mock.patch('json.loads', side_effect=lambda b, _j=json.loads: (cfg if isinstance(b, (bytes, bytearray)) and b == Path(self.m.man['arguments']['config']).read_bytes() else _j(b))):
            self.assertFalse(Model(self.run_dir, self.data, 'cpu').valid)
        self.assertIs(F.FeatureGaussian.nll, orig)
        from estimator.svgd_temporal import TemporalPrior
        lift = TemporalPrior.velocity_lift
        with mock.patch.object(TemporalPrior, 'velocity_lift', lambda s, S, dX: -lift(s, S, dX)):
            X = torch.as_tensor(self.Xgt, dtype=torch.float64).transpose(0, 1)
            _, kkt = lift_positions(self.m, X); self.assertGreater(kkt, 1e-3)


if __name__ == '__main__':
    unittest.main()
