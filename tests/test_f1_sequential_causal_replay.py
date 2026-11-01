"""U2-U12 of RGBDiekf_F1_Fourth_Pass_Sequential_Replay_and_Diagnostic_Bugfix_Codex.md (CPU, deterministic, no writes to inference
paths). U2 / U8 / U10 / U11-real use the first-pass traced runs and are skipped when they are not mounted."""
import hashlib
import unittest
from pathlib import Path
import numpy as np
from tools import root_cause_oracle as O
from tools.f1_temporal_reference_diagnostics import ref_row, RefProblem, references, model_cost
from tools.f1_sequential_causal_replay import covariance_oracle, POINTS, ARMS
from tests.projective_fixtures import camera, POLICY
from tests.test_f1_temporal_reference_diagnostics import textured, cam, noise, stub_scene, Img

FOR = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030')
HAVE = (FOR/'storage_45135'/'traced').exists()


class U3PredictionOnly(unittest.TestCase):
    def test_no_rows_keeps_propagated_mean_and_full_covariance(self):
        c = cam(); im = textured(); a = np.array([.4, .5, .6])
        g = np.random.default_rng(0); M = g.normal(size=(6, 6)); P = M@M.T*1e-4+np.eye(6)*1e-6
        s, P_, F, Q = O.predict(np.r_[0.0131, -0.0077, 1.55, .1, -.2, .05], P, 1/59., np.eye(3)*.01)
        p = RefProblem([c], [im], a, s, P_, [O.DUP], noise(), POLICY, 10, 1e-8)       # the only camera is frozen DUPLICATE: no row
        r = p.run(POLICY['update_acceptance'])
        self.assertFalse(r['observed'])
        np.testing.assert_array_equal(r['s_plus'], s); np.testing.assert_array_equal(r['P_plus'], P_)   # incl. P_xv cross blocks
        from estimator.point_kalman_cuda import process_noise
        np.testing.assert_allclose(Q, process_noise(1/59., np.eye(3)*.01).numpy(), rtol=0, atol=0)        # Q_k = production process_noise


class U4U5U7Reference(unittest.TestCase):
    def test_U4_static_reference_vanishes_jacobian_unchanged_source_once(self):
        c = cam(); x = np.array([0.0131, -0.0077, 1.55]); im = textured(np.array([.02, 0, -.01]))
        r0 = O.row(im, c, x, np.array([.4, .5, .6]), noise(), {**POLICY, 'visibility_n_sigma': float('inf')}, np.eye(3)*1e-4, np.zeros(3))
        o = ref_row(im, c, x, np.array([.4, .5, .6]), noise(), POLICY, np.eye(3)*1e-4, np.zeros(3), {'valid': True, 'b': r0['r'], 'var0': r0['var']}, 'RGBD')
        np.testing.assert_allclose(o['r'], 0, atol=1e-15); np.testing.assert_array_equal(o['H'], r0['H'])
        u, v, *_ = c.project(x); a = O.sample(im, c, u, v, noise(), POLICY)['rgb']
        sc = stub_scene(im); sc.run.can['canonical_rgb'] = {7: a}
        refs = references(sc, 7, [O.VALID]); self.assertIn('SOURCE_REFERENCE_IS_PRIOR_DEFINITION', refs[0]['source_note'])

    def test_U5_textured_motion_detectable_textureless_tangent_ambiguous(self):
        from tests.test_f1_temporal_reference_diagnostics import T1MovingSignal
        T1MovingSignal('test_textured_motion_is_seen_textureless_tangent_is_not').test_textured_motion_is_seen_textureless_tangent_is_not()

    def test_U7_invalid_reference_never_zero_filled(self):
        from tests.test_f1_temporal_reference_diagnostics import T4InvalidReference
        T4InvalidReference('test_masked_reference_is_not_zeroed_or_substituted').test_masked_reference_is_not_zeroed_or_substituted()


class U6SharedReferenceCovariance(unittest.TestCase):
    def test_dense_gls_equals_latent_marginalization_naive_differs(self):
        for T in (3, 4, 5):
            r = covariance_oracle(T, 2, seed=T, correlated=False)
            self.assertLess(r['latent_vs_dense_mean_rel'], 1e-9); self.assertLess(r['latent_vs_dense_cov_rel'], 1e-9); self.assertLess(r['latent_vs_dense_loglik_abs'], 1e-8)
            self.assertGreater(r['Sigma_eigs_min_max'][0], 0)
            self.assertGreater(abs(r['naive_vs_exact_cov_trace_ratio']-1), 1e-3)   # independent R_t + R_0 is a different posterior
        rc = covariance_oracle(4, 2, seed=9, correlated=True)
        self.assertGreater(rc['Sigma_eigs_min_max'][0], 0)


class U12UndefinedSupport(unittest.TestCase):
    def test_fixed_camera_invalid_after_move_is_undefined_not_zero(self):
        c = cam(); im = textured(); im.mask[:, 60:] = False; a = np.array([.4, .5, .6])
        s = np.r_[0.0131, -0.0077, 1.55, 0, 0, 0]; P = np.diag([1e-4]*3+[1.]*3)
        p = RefProblem([c], [im], a, s, P, [O.VALID], noise(), POLICY, 10, 1e-8)
        far = s.copy(); far[0] += .05
        J, lost = model_cost(p, far, [0]); self.assertIsNone(J); self.assertTrue(lost)
        J0, lost0 = model_cost(p, s, [0]); self.assertIsNotNone(J0); self.assertFalse(lost0)


@unittest.skipUnless(HAVE, 'first-pass traced runs not mounted')
class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.f1_second_pass import Scene
        from tools.f1_sequential_causal_replay import CloudOwnership
        cls.sc = {s: Scene(FOR, s) for s in ('storage_45135', 'USB_100109')}
        cls.own = {s: CloudOwnership(v) for s, v in cls.sc.items()}

    def replay(self, case, arm, mode, N):
        from tools.f1_sequential_causal_replay import replay
        scene, pid, _, _ = POINTS[case]; sc = self.sc[scene]
        refs = references(sc, pid, sc.frame0_ownership([pid])[pid])
        return sc, pid, replay(sc, pid, arm, mode, N, refs, own=self.own[scene])

    def test_U2_continuous_baseline_parity_frames_1_to_3(self):
        from tools.f1_diagnostic_units import compare_state6
        for case in ('storage_8444', 'usb_67827'):
            sc, pid, fr = self.replay(case, 'A0_ORIGINAL_F1', 'P', 3)
            tr1 = sc.trace(1)['prediction']['points'][str(pid)]
            np.testing.assert_array_equal(np.array(tr1['s_previous']), np.r_[sc.x0[pid], 0, 0, 0])   # s0+ = [x_ref, 0]
            for f in fr:
                tr = sc.trace(f['frame']); fin = tr['final'][str(pid)]; pre = tr['prediction']['points'][str(pid)]
                cm, cp = compare_state6(f['s_minus'], pre['s_minus']), compare_state6(f['s_plus'], fin['s_plus'])
                self.assertLessEqual(cm['position_linf_m'], 1e-9, (case, f['frame'])); self.assertLessEqual(cm['velocity_linf_m_s'], 1e-7, (case, f['frame']))
                self.assertLessEqual(cp['position_linf_m'], 1e-9, (case, f['frame'])); self.assertLessEqual(cp['velocity_linf_m_s'], 1e-7, (case, f['frame']))
                self.assertEqual(f['acceptance'], fin['acceptance'], (case, f['frame']))
                prod_rows = sorted({r['camera'] for r in fin.get('rows', [])}) if fin['acceptance'].startswith('ACCEPTED') else []
                self.assertEqual(f['rows_accepted'], prod_rows, (case, f['frame']))

    def test_U8_ownership_rows_frame1(self):
        for case, rows in (('storage_8444', [1]), ('usb_67827', [5, 6])):
            for mode in ('P', 'C'):
                sc, pid, fr = self.replay(case, 'A0_ORIGINAL_F1', mode, 1)
                self.assertEqual(fr[0]['rows_initial'], rows, (case, mode))

    def test_U9_replay_never_touches_ground_truth(self):
        import tools.forensic_f1_drift as F
        import tools.audit_static_equilibrium as A
        orig = (F.GroundTruth, A.gt_basis)
        def boom(*a, **k): raise AssertionError('GT used inside replay')
        F.GroundTruth, A.gt_basis = boom, boom
        try: self.replay('usb_87290', 'A3_REFERENCE_RGBD', 'C', 2)
        finally: F.GroundTruth, A.gt_basis = orig

    def test_U10_deterministic_and_nonmutating(self):
        f = FOR/'USB_100109'/'traced'/'free_forward'/'numeric'/'trace_frame_0001.json'
        h0 = hashlib.sha256(f.read_bytes()).hexdigest()
        _, _, a = self.replay('usb_67827', 'A3_REFERENCE_RGBD', 'P', 2); _, _, b = self.replay('usb_67827', 'A3_REFERENCE_RGBD', 'P', 2)
        for x, y in zip(a, b): np.testing.assert_array_equal(x['s_plus'], y['s_plus']); np.testing.assert_array_equal(x['P_plus'], y['P_plus'])
        self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), h0)
        import torch; self.assertFalse(torch.cuda.is_initialized())

    def test_U11_85696_frame3_no_reacquisition_invented(self):
        sc, pid, fr = self.replay('usb_85696', 'A3_REFERENCE_RGBD', 'P', 3)
        f3 = fr[2]
        frozen_out = [c for c in f3['cameras'] if c['frozen_or_recomputed_code'] in ('DUPLICATE_PIXEL_SUPPRESSED', 'SELF_OCCLUDED')]
        if len(frozen_out) == len(f3['cameras']):
            self.assertFalse(f3['observed']); np.testing.assert_array_equal(f3['s_plus'], f3['s_minus']); np.testing.assert_array_equal(f3['P_plus'], f3['P_minus'])


if __name__ == '__main__':
    unittest.main()
