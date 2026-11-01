"""U1 (RGBDiekf_F1_Fourth_Pass_Sequential_Replay_and_Diagnostic_Bugfix_Codex.md): unit-safe 6-state diagnostics. A scalar over the
mixed [x (m), v (m/s)] difference is never a distance; the historical mixed fields must no longer feed charts / reports."""
import ast
import unittest
from pathlib import Path
import numpy as np
from tools.f1_diagnostic_units import compare_state6, compare_iterates, compare_cov6, dominance, LEGACY_MIXED_FIELDS

REPO = Path(__file__).resolve().parents[1]


class StateUnits(unittest.TestCase):
    def test_identical_position_velocity_differs(self):
        a = np.array([1., 2., 3., 0., 0., 0.]); b = np.array([1., 2., 3., 2., 0., 0.])
        c = compare_state6(a, b)
        self.assertEqual(c['position_linf_m'], 0.); self.assertEqual(c['position_l2_m'], 0.)
        self.assertEqual(c['velocity_linf_m_s'], 2.); self.assertEqual(dominance(c), 'velocity-dominated')

    def test_position_only(self):
        a = np.array([1., 2., 3.004, .5, .5, .5]); b = np.array([1., 2., 3., .5, .5, .5])
        c = compare_state6(a, b)
        self.assertAlmostEqual(c['position_linf_m'], .004, places=15); self.assertEqual(c['velocity_linf_m_s'], 0.)

    def test_mixed_changes_keep_distinct_units(self):
        a = np.zeros(6); b = np.array([.003, 0, 0, 0, .7, 0])
        c = compare_state6(a, b)
        self.assertAlmostEqual(c['position_linf_m'], .003); self.assertAlmostEqual(c['velocity_linf_m_s'], .7)
        self.assertNotIn('max_abs_m', ' '.join(c))                               # no unit-less max labelled metres
        it = compare_iterates([a, a, b], [a, a, a])
        self.assertEqual(it['first_iteration_position_diverging_over_1e-9_m'], 2); self.assertEqual(it['first_iteration_velocity_diverging_over_1e-9_m_s'], 2)
        it2 = compare_iterates([a, np.r_[0, 0, 0, 1e-3, 0, 0]], [a, a])
        self.assertIsNone(it2['first_iteration_position_diverging_over_1e-9_m']); self.assertEqual(it2['first_iteration_velocity_diverging_over_1e-9_m_s'], 1)

    def test_covariance_blockwise(self):
        A = np.eye(6); B = np.eye(6); B[3, 3] = 2.; B[0, 4] = B[4, 0] = .1
        c = compare_cov6(A, B)
        self.assertEqual(c['P_xx_max_abs_m2'], 0.); self.assertAlmostEqual(c['P_xv_max_abs_m2_per_s'], .1); self.assertAlmostEqual(c['P_vv_max_abs_m2_per_s2'], 1.)


class LegacyFieldAudit(unittest.TestCase):
    def test_historical_mixed_fields_no_longer_produced_or_plotted(self):
        """AST audit: none of the historical mixed-state field names appears as a string constant in the diagnostic tools any more
        (they may only survive as 'legacy_mixed_state_max_UNITLESS_INVALID')."""
        for f, names in LEGACY_MIXED_FIELDS.items():
            tree = ast.parse((REPO/f).read_text())
            consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            fstr = ' '.join(c for c in consts)
            for nm in names:
                self.assertNotIn(nm, consts, f'{f}: {nm}')
                self.assertNotIn(f"['{nm}']", fstr, f'{f}: {nm}')

    def test_no_unitless_state_max_labelled_metres(self):
        """grep-style audit: '.max()' over a full s_plus difference never sits in a field whose name ends in '_m'."""
        import re
        pat = re.compile(r"'([a-z_]+_m)':\s*float\(np\.abs\([^)]*s_plus[^)]*\)\.max\(\)\)")
        for f in LEGACY_MIXED_FIELDS:
            self.assertFalse(pat.findall((REPO/f).read_text()), f)


if __name__ == '__main__':
    unittest.main()
