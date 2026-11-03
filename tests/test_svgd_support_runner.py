"""Gate 4 + TF.6 items 5, 7, 10: the opt-in runner end to end on synthetic masked RGB-D sequences.

Asserted: strict blocking without probability files, objective monotone on accepted steps, the tested operator functions
are the ones the runner calls, camera-order invariance, no legacy stage outputs, immutable output dirs. Physical motion
recovery (static drift, moving-point recovery) is MEASURED with evaluation-only GT after the run and recorded, never
asserted as a pass: the gate report states it."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import estimator.svgd_support_run as R
from estimator import svgd_target_score, svgd_particle_operator, svgd_temporal, svgd_feature_measurement
from tests.svgd_fixtures import SyntheticDataset, runner_args, small_config, record

REPO = Path(__file__).resolve().parents[1]
EXP, STRICT = REPO/'configs/svgd_support_experimental.json', REPO/'configs/svgd_support.json'


def run_case(tmp, data, base=EXP, mode='experimental', **kw):
    Path(tmp).mkdir(parents=True, exist_ok=True)
    cfg = small_config(str(Path(tmp)/'cfg.json'), base, **{'features.D': 64, 'convergence': {'max_outer': 3, 'pcg_max_iter': 25}})
    out = Path(tmp)/'run'; code = R.run(runner_args(out, cfg, mode=mode, **kw), data=data)
    return code, out


def evaluate(out, data):
    """Evaluation-only GT (after the inference artifacts are written)."""
    sv = out/'svgd_support'; can = np.load(sv/'canonical.npz'); X = np.load(sv/'transport_provisional.npy')
    mov = np.array([data.observation(0, c).gt_moving.ravel()[p] for _, c, p in can['source']])
    gt = np.stack([can['xyz0']@data.pose(k)[0].T+data.pose(k)[1] for k in range(len(X))]); gt[:, ~mov] = can['xyz0'][~mov]
    disp_true = np.linalg.norm(gt[-1]-gt[0], axis=1); err = np.linalg.norm(X[-1]-gt[-1], axis=1); drift = np.linalg.norm(X[-1]-X[0], axis=1)
    return {'static_points': int((~mov).sum()), 'moving_points': int(mov.sum()),
            'static_drift_median_m': float(np.median(drift[~mov])), 'static_drift_p90_m': float(np.quantile(drift[~mov], .9)),
            'moving_true_displacement_median_m': float(np.median(disp_true[mov])) if mov.any() else 0.,
            'moving_endpoint_error_median_m': float(np.median(err[mov])) if mov.any() else None,
            'moving_recovered_fraction_median': float(np.median(1-err[mov]/np.maximum(disp_true[mov], 1e-12))) if mov.any() and disp_true[mov].max() > 0 else None}


def accepted_objective(out):
    import csv
    rows = list(csv.DictReader(open(out/'svgd_support/objective_timeline.csv')))
    return [float(r['E_total']) for r in rows if r.get('accepted') == 'True' and r.get('E_total')]


class RunnerTests(unittest.TestCase):
    def test_strict_blocks_without_probability_files(self):
        with tempfile.TemporaryDirectory() as t:
            code, out = run_case(t, SyntheticDataset(K=3), base=STRICT, mode='strict')
            self.assertEqual(code, R.EXIT['BLOCKED'])
            st = json.loads((out/'svgd_support/status.json').read_text())
            self.assertEqual(st['blocked']['stage'], 'TARGET'); self.assertIn('BLOCKED', st['blocked']['status'])
            for f in ('p_dynamic.npy', 'p_static.npy', 'reliable_moving_seed_ids.npy'): self.assertFalse((out/'svgd_support'/f).exists())
            self.assertTrue((out/'svgd_support/canonical.npz').exists())
            self.assertEqual(json.loads((out/'result.json').read_text())['status'], 'UNRESOLVED')
            # experimental config under STRICT fidelity: unauthorized deviations are refused
            code2, out2 = run_case(Path(t)/'b', SyntheticDataset(K=3), base=EXP, mode='strict')
            st2 = json.loads((out2/'svgd_support/status.json').read_text())
            self.assertTrue(any('UNAUTHORIZED_DEVIATION' in r['reason'] for r in st2['blocked']['reasons']))

    def test_objective_monotone_on_accepted_steps(self):
        with tempfile.TemporaryDirectory() as t:
            d = SyntheticDataset(K=4, motion='translate', speed=.15, depth_noise=5e-4)
            code, out = run_case(t, d)
            self.assertEqual(code, R.EXIT['BLOCKED'])                                          # stops at MODEL_EVIDENCE, by design
            st = json.loads((out/'svgd_support/status.json').read_text()); self.assertEqual(st['blocked']['stage'], 'MODEL_EVIDENCE')
            E = accepted_objective(out); self.assertGreater(len(E), 1)
            self.assertTrue(all(b <= a for a, b in zip(E, E[1:])))
            X = np.load(out/'svgd_support/transport_provisional.npy'); self.assertEqual(X.shape[0], 4)
            self.assertFalse((out/'svgd_support/p_dynamic.npy').exists())
            record('E32', test='objective_monotone', accepted_values=E)

    def test_production_path_calls_tested_operators(self):
        """TF.6 item 10: the runner's operator names ARE the tested functions, and they are called in a run."""
        self.assertIs(R.target_score, svgd_target_score.target_score); self.assertIs(R.stein_field, svgd_particle_operator.stein_field)
        self.assertIs(R.pcg, svgd_temporal.pcg); self.assertIs(R.FrameFeatureTerm, svgd_feature_measurement.FrameFeatureTerm)
        calls = {}
        def spy(name, fn):
            def w(*a, **k): calls[name] = calls.get(name, 0)+1; return fn(*a, **k)
            return w
        with tempfile.TemporaryDirectory() as t, mock.patch.object(R, 'target_score', spy('target_score', R.target_score)), \
                mock.patch.object(R, 'stein_field', spy('stein_field', R.stein_field)), mock.patch.object(R, 'pcg', spy('pcg', R.pcg)):
            run_case(t, SyntheticDataset(K=3))
        for k in ('target_score', 'stein_field', 'pcg'): self.assertGreater(calls.get(k, 0), 0, k)
        record('E07', test='runtime_call_audit', calls=calls)

    def test_camera_order_invariance(self):
        """Permuting the camera loading order permutes canonical IDs only: per-source results agree to reduction roundoff."""
        class Perm:
            def __init__(s, base, perm): s.b, s.perm = base, perm; s.cameras = [base.cameras[p] for p in perm]
            def __getattr__(s, k): return getattr(s.b, k)
            def observation(s, k, c): return s.b.observation(k, s.perm[c])
        with tempfile.TemporaryDirectory() as t:
            a = SyntheticDataset(K=3, n_cams=3); _, oa = run_case(Path(t)/'a', a)
            b = Perm(SyntheticDataset(K=3, n_cams=3), [2, 0, 1]); _, ob = run_case(Path(t)/'b', b)
            ca, cb = np.load(oa/'svgd_support/canonical.npz'), np.load(ob/'svgd_support/canonical.npz')
            Xa, Xb = np.load(oa/'svgd_support/transport_provisional.npy'), np.load(ob/'svgd_support/transport_provisional.npy')
            ia = {(a.cameras[s[1]].name, s[2]): i for i, s in enumerate(ca['source'])}
            perm = np.array([ia[(b.cameras[s[1]].name, s[2])] for s in cb['source']])
            # Stein + target + features are exact to ~1e-18 m under the permutation; the capped PCG (cond(L) ~ 1e10) amplifies
            # 1e-16 reduction-order roundoff, so the tolerance is relative to the transported displacement.
            scale = float(np.abs(Xa-Xa[:1]).max())
            np.testing.assert_allclose(Xb, Xa[:, perm], atol=1e-4*scale)
            record('E07', test='camera_order_invariance', max_abs_m=float(np.abs(Xb-Xa[:, perm]).max()), displacement_scale_m=scale,
                   tolerance='1e-4 x displacement (Krylov-amplified roundoff; Stein step alone agrees to 1e-18 m)')

    def test_no_legacy_stage_outputs_and_immutable_dir(self):
        with tempfile.TemporaryDirectory() as t:
            _, out = run_case(t, SyntheticDataset(K=2))
            for legacy in ('free_forward', 'coherence_forward', 'rigid_forward', 'joint_forward', 'support_forward'): self.assertFalse((out/legacy).exists())
            with self.assertRaises(SystemExit): R.run(runner_args(out, str(Path(t)/'cfg.json')), data=SyntheticDataset(K=2))
            with self.assertRaises(SystemExit): R.run(runner_args(Path(t)/'x', str(Path(t)/'cfg.json'), svgd_kernel_backend='triton'), data=SyntheticDataset(K=2))

    def test_gate4_scenarios_measured(self):
        """(a) static + depth noise, (b) RGB flicker, (c) translation, (d) rotation about a hinge; GT only post hoc."""
        cases = {'a_static_depth_noise': dict(motion='none', depth_noise=1e-3), 'b_static_rgb_flicker': dict(motion='none', rgb_noise=.03),
                 'c_translation': dict(motion='translate', speed=.15, depth_noise=5e-4), 'd_rotation_hinge': dict(motion='rotate', speed=.15, depth_noise=5e-4)}
        for name, kw in cases.items():
            with tempfile.TemporaryDirectory() as t:
                d = SyntheticDataset(K=4, seed=5, **kw); code, out = run_case(t, d)
                E = accepted_objective(out); self.assertTrue(all(b <= a for a, b in zip(E, E[1:])))
                st = json.loads((out/'svgd_support/status.json').read_text())
                record('GATE4', scenario=name, **evaluate(out, d), convergence=st['gates']['TEMPORAL_REFINEMENT']['convergence'])
        record('GATE4', scenario='e_two_moving_parts / f_intermittent / g_variable_cameras', status='NOT_RUN (fixture has one moving plate)')


if __name__ == '__main__':
    unittest.main()
