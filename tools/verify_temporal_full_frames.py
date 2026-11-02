#!/usr/bin/env python3
"""Read-only G3 verification of a finished full temporal sparse run (no GT, no inference).

For every journal-committed frame after the anchor: array digests recomputed against the commit journal; finite
states; symmetric positive-definite 6x6 joint marginals for EVERY point; physical prediction x-_k = x+_{k-1} +
dt v+_{k-1} at the stored precision; genuine nonzero accepted corrections; per-frame rows, reference lifecycle and
raw residual statistics; the covariance certificate, KKT inertia and final-linearization provenance recorded by
the run. Uncommitted frames must remain NaN (never estimates).

    python -B tools/verify_temporal_full_frames.py RUN/USB_100109_full/free_forward [more roots] --output verification.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.temporal_output import atomic_json


def eig_min(P, chunk=8192):
    return np.concatenate([np.linalg.eigvalsh(P[i:i+chunk]).min(1) for i in range(0, len(P), chunk)])


def verify(root):
    root = Path(root); journal = json.loads((root/'completed_frames.json').read_text())
    manifest = json.loads((root/'run_manifest.json').read_text()); order = journal['processing_order']
    arrays = {name: np.load(root/f'{name}.npy', mmap_mode='r') for name in journal['frame_sha256'][str(order[0])]}
    out = {'root': str(root), 'journal_status': journal['status'], 'completed_frames': journal['completed_frames'],
           'point_count': manifest['point_count'], 'camera_count': manifest['camera_count'], 'frames': {}}
    times = np.asarray(manifest['timestamps_seconds'], float); failures = []
    for frame in journal['completed_frames']:
        rec = {'array_digests_match_journal': all(hashlib.sha256(np.ascontiguousarray(arrays[n][frame]).tobytes()).hexdigest() == d
                                                  for n, d in journal['frame_sha256'][str(frame)].items())}
        x = np.asarray(arrays['posterior_xyz'][frame]); v = np.asarray(arrays['velocity'][frame]); P = np.asarray(arrays['posterior_covariance'][frame])
        rec['finite_state_and_covariance'] = bool(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(P).all())
        rec['covariance_max_asymmetry'] = float(np.abs(P-P.transpose(0, 2, 1)).max())
        e = eig_min(P); rec['points_with_nonpositive_marginal'] = int((e <= 0).sum()); rec['marginal_min_eigenvalue_mixed_units'] = float(e.min())
        rec['position_variance_median_m2'] = float(np.median(np.diagonal(P[:, :3, :3], axis1=1, axis2=2)))
        if frame != order[0]:
            index = order.index(frame); prev = order[index-1]; dt = float(times[frame]-times[prev])
            pred = np.asarray(arrays['predicted_xyz'][frame], np.float64)
            expected = np.asarray(arrays['posterior_xyz'][prev], np.float64)+dt*np.asarray(arrays['velocity'][prev], np.float64)
            rec['prediction_vs_constant_velocity_max_abs_m'] = float(np.abs(pred-expected).max())   # float32 storage of predicted/velocity
            correction = np.linalg.norm(x-pred, axis=1); accepted = np.asarray(arrays['update_acceptance'][frame]).astype(bool)
            rec.update(assimilated_points=int(accepted.sum()), points_with_nonzero_correction=int((correction > 0).sum()),
                       correction_m_quantiles_assimilated=np.quantile(correction[accepted], [.5, .9, .99, 1]).tolist() if accepted.any() else None,
                       correction_m_max_not_assimilated=float(correction[~accepted].max()) if (~accepted).any() else None,
                       speed_m_s_quantiles=np.quantile(np.linalg.norm(v, axis=1), [.5, .9, .99, 1]).tolist())
            summary = json.loads((root/'temporal'/f'frame_{frame:04d}.json').read_text())
            profile = json.loads((root/'temporal'/f'sparse_profile_{frame:04d}.json').read_text())
            solver = summary['solver']; marg = profile['current_marginal_profile']
            rec.update(rows=summary['rows'], observed_points=summary['observed_points'], reference_lifecycle=summary['reference'],
                       rgb_residual_rms=summary['rgb_residual_rms'], depth_residual_rms_m=summary['depth_residual_rms_m'],
                       unexplained_foreground_pixels=summary['unexplained_foreground'], rendered_outside_foreground=summary['rendered_outside_foreground'],
                       nonlinear_iterations=len(solver['history']), accepted_iterations=solver['accepted_iterations'], converged_flag=solver['converged'],
                       objective_history=[h['objective'] for h in solver['history']]+[solver['objective']],
                       frozen_components_last_iteration=solver['history'][-1]['frozen_components'],
                       covariance_from_final_linearization=solver.get('covariance_linearization_sha256') is not None and solver.get('covariance_linearization_sha256') == solver.get('final_linearization_sha256'),
                       final_KKT_inertia=solver.get('final_factor_inertia'), final_KKT_expected_inertia=solver.get('final_factor_expected_inertia'),
                       covariance_method=marg.get('method'), covariance_certificate=marg.get('certificate'), covariance_LDL_inertia=marg.get('LDL_inertia'),
                       covariance_seconds=summary['stage_times'].get('covariance_seconds'), frame_seconds=summary['seconds'],
                       nnz_KKT=profile.get('nnz_KKT'), selected_inverse_nnz_L=marg.get('nnz_L'), peak_cuda_allocated_bytes=marg.get('peak_cuda_allocated_bytes'),
                       raw_unique_sensor_keys=profile.get('raw_unique_sensor_keys'), factor_counts=profile.get('factor_counts'))
            cert = rec['covariance_certificate'] or {}
            checks = {'digests': rec['array_digests_match_journal'], 'finite': rec['finite_state_and_covariance'],
                      'spd': rec['points_with_nonpositive_marginal'] == 0, 'rows': rec['rows'] > 0, 'assimilated': rec['assimilated_points'] > 0,
                      'nonzero_corrections': rec['points_with_nonzero_correction'] > 0,
                      'certificate': bool(cert) and cert['max_error']['correlation_units'] <= cert['tolerance_correlation_units'],
                      'inertia': rec['covariance_LDL_inertia'] is not None and rec['covariance_LDL_inertia'] == rec['final_KKT_expected_inertia'] == rec['final_KKT_inertia'],
                      'final_linearization': rec['covariance_from_final_linearization']}
            rec['checks'] = checks; failures += [f'{frame}:{k}' for k, ok in checks.items() if not ok]
        out['frames'][str(frame)] = rec
    pending = [k for k in order if k not in journal['completed_frames']]
    out['uncommitted_frames_all_NaN'] = all(np.isnan(np.asarray(arrays['posterior_xyz'][k])).all() for k in pending)
    if not out['uncommitted_frames_all_NaN']:failures.append('uncommitted_frames_not_NaN')
    out['failures'] = failures; out['verified'] = not failures and journal['status'] == 'COMPLETE'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+', type=Path); ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    if a.output is not None and a.output.exists():ap.error('Output exists; never overwrite verification evidence')
    report = {'runs': [verify(r) for r in a.roots]}; report['all_verified'] = all(r['verified'] for r in report['runs'])
    if a.output is not None:atomic_json(a.output, report)
    print(json.dumps(report, indent=1, default=float)[:20000])
    return 0 if report['all_verified'] else 1


if __name__ == '__main__':raise SystemExit(main())
