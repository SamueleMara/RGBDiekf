"""Unit-safe comparison of the F1 6D state s = [x (m), v (m/s)] and its 6x6 covariance (diagnostics only).

A scalar over all six state components (e.g. np.abs(s_a - s_b).max()) has NO physical unit; it must never be labelled metres
nor decide a positional parity / stability verdict (RGBDiekf_F1_Fourth_Pass_Sequential_Replay_and_Diagnostic_Bugfix_Codex.md
section 2). Position and velocity are compared separately; covariances blockwise: P_xx (m^2), P_xv (m^2/s), P_vv (m^2/s^2).
The 1e-9 divergence cutoffs are DIAGNOSTIC RESOLUTION, not physical acceptance thresholds: the actual differences are reported.
"""
import numpy as np
from numpy.linalg import norm

LEGACY_MIXED_FIELDS = {   # historical fields that took one max over the mixed [x, v] difference and were labelled / used as metres
    'tools/f1_temporal_reference_diagnostics.py': ['production_s_plus_max_abs_m', 'accepted_state_diff_vs_unshifted_m', 'iterate_diff_vs_unshifted_m',
                                                   'first_iteration_diverging_over_1e-9_m'],
    'tools/f1_second_pass.py': ['production_s_plus_max_abs_m', 'accepted_state_diff_vs_A_m', 'iterate_diff_vs_A_m'],
    'tools/forensic_f1_drift.py': ['s_plus_max_abs_m', 'oracle_s_plus_max_abs_m', 'delta_abs_diff_m', 'oracle_iteration_delta_max_abs_m'],
    'tools/root_cause_trace.py': ['replay_s_plus_max_abs_m', 'accepted_linearization_state_max_abs_m'],
}
RESOLUTION_POSITION_M, RESOLUTION_VELOCITY_M_S = 1e-9, 1e-9


def compare_state6(a, b):
    """Position and velocity differences of two 6-states, each in its own unit."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape == (6,), (a.shape, b.shape)
    dx, dv = a[:3]-b[:3], a[3:]-b[3:]
    return {'position_l2_m': float(norm(dx)), 'position_linf_m': float(np.abs(dx).max()),
            'velocity_l2_m_s': float(norm(dv)), 'velocity_linf_m_s': float(np.abs(dv).max())}


def compare_iterates(A, B):
    """Per-iterate comparison of two iterate sequences (common length) and the two INDEPENDENT first-divergence indices."""
    n = min(len(A), len(B))
    per = [compare_state6(A[i], B[i]) for i in range(n)]
    fp = next((i for i, c in enumerate(per) if c['position_linf_m'] > RESOLUTION_POSITION_M), None)
    fv = next((i for i, c in enumerate(per) if c['velocity_linf_m_s'] > RESOLUTION_VELOCITY_M_S), None)
    return {'per_iterate': per, 'position_linf_m_per_iterate': [c['position_linf_m'] for c in per],
            'velocity_linf_m_s_per_iterate': [c['velocity_linf_m_s'] for c in per],
            'first_iteration_position_diverging_over_1e-9_m': fp, 'first_iteration_velocity_diverging_over_1e-9_m_s': fv,
            'compared_iterates': n, 'iterate_counts': [len(A), len(B)]}


def compare_cov6(A, B):
    """Blockwise covariance comparison (never one m^2 number for the 6x6)."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    assert A.shape == B.shape == (6, 6)
    out = {}
    for nm, sl, unit in (('P_xx', (slice(0, 3), slice(0, 3)), 'm2'), ('P_xv', (slice(0, 3), slice(3, 6)), 'm2_per_s'), ('P_vv', (slice(3, 6), slice(3, 6)), 'm2_per_s2')):
        d = A[sl]-B[sl]; ref = max(float(np.abs(B[sl]).max()), 1e-300)
        out[f'{nm}_max_abs_{unit}'] = float(np.abs(d).max()); out[f'{nm}_frobenius_{unit}'] = float(norm(d)); out[f'{nm}_rel_max'] = float(np.abs(d).max()/ref)
    return out


def dominance(c):
    """Which component produced a historical mixed max: position (m) or velocity (m/s) - their raw magnitudes."""
    p, v = c['position_linf_m'], c['velocity_linf_m_s']
    return 'velocity-dominated' if v > p else ('position-dominated' if p > v else 'equal')
