#!/usr/bin/env python3
"""POST-INFERENCE ONLY: matched-point GT comparison of temporal modes, and the GT-motion temporal-warp check.

Ground truth (joint metadata, labelled endpoint clouds, frame_values.csv) is read HERE ONLY, after inference, from
journal-verified frames. Nothing produced here is read by any estimator. Modes are compared on the IDENTICAL selected
original point IDs (checked), with the same sensor data and the same 6x6 frame-0 marginal prior.

    python -B tools/evaluate_temporal_local_pilot.py --batch RESULTS/temporal_rgbd_<stamp> --data-root DATA --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_projective_gt import gt_joint, labelled_endpoints, gt_transform
from utils.io import json_write, serial
from utils.temporal_output import require_valid_frame

QUALIFIER = ('local-correlated: BLOCK_DIAGONAL_ACROSS_POINTS; TEMPORAL_AND_SENSOR_CROSS_CORRELATION_RETAINED_WITHIN_POINT; '
             'simplified: fixed reference state + independent marginal sensor noise. NEES/NIS are conditional on each '
             "mode's own covariance approximation; temporal samples are correlated, no pooled chi-square claim.")


def truth_for(run_root, scene):
    """Post-hoc GT position/velocity of every selected point at every processed frame (same rule as evaluate_temporal_points_gt)."""
    man = json.loads((run_root/'run_manifest.json').read_text())
    can = np.load(run_root/'canonical.npz')
    k0 = int(can['canonical_frame'])
    kind, axis, origin, _ = gt_joint(scene)
    values = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True)
    allq = np.asarray(values[values.dtype.names[1]], float)
    frames = man['original_frame_indices']
    q, times = allq[frames], np.asarray(man['timestamps_seconds'])
    clouds = labelled_endpoints(scene, kind, axis, origin, allq[-1]-allq[0])
    cloud = (clouds[0][0].copy(), clouds[0][1])
    if k0 != 0:
        T0 = gt_transform(kind, axis, origin, q[k0]-allq[0])
        mv = cloud[1] == clouds[2]
        cloud[0][mv] = cloud[0][mv]@T0[:3, :3].T+T0[:3, 3]
    distance, nn = cKDTree(cloud[0]).query(can['canonical_xyz'])
    moving = cloud[1][nn] == clouds[2]
    x0 = can['canonical_xyz']
    K = len(times)
    X = np.repeat(x0[None], K, 0)
    V = np.zeros_like(X)
    dq = np.gradient(q, times) if K > 1 else np.zeros(K)
    for k in range(K):
        T = gt_transform(kind, axis, origin, q[k]-q[k0])
        X[k, moving] = x0[moving]@T[:3, :3].T+T[:3, 3]
        V[k, moving] = axis*dq[k] if kind == 'PRISMATIC' else np.cross(axis, X[k, moving]-origin)*dq[k]
    return {'X': X, 'V': V, 'moving': moving, 'label_distance': distance, 'point_id': can['point_id'], 'k0': k0, 'kind': kind,
            'true_speed_moving_m_s': [float(np.linalg.norm(V[k, moving], axis=1).mean()) if moving.any() else 0. for k in range(K)]}


def true_pixel_motion(scene, truth, run_root):
    """GT image displacement of moving points between consecutive processed frames, per camera (pixels)."""
    from scene.dataset import Dataset
    man = json.loads((run_root/'run_manifest.json').read_text())
    cams = Dataset(scene, len(man['original_frame_indices'])).cameras
    order, out = man['processing_order'], []
    mv = truth['moving']
    for a, b in zip(order[:-1], order[1:]):
        shifts = []
        for c in cams:
            T = np.asarray(c.world_to_camera, float)
            fx, fy, cx, cy = c.intrinsics
            pa, pb = truth['X'][a, mv]@T[:3, :3].T+T[:3, 3], truth['X'][b, mv]@T[:3, :3].T+T[:3, 3]
            ok = (pa[:, 2] > 0) & (pb[:, 2] > 0)
            ua = np.c_[fx*pa[ok, 0]/pa[ok, 2]+cx, fy*pa[ok, 1]/pa[ok, 2]+cy]
            ub = np.c_[fx*pb[ok, 0]/pb[ok, 2]+cx, fy*pb[ok, 1]/pb[ok, 2]+cy]
            inside = (ua[:, 0] >= 0) & (ua[:, 0] < c.width) & (ua[:, 1] >= 0) & (ua[:, 1] < c.height)
            shifts.append(np.linalg.norm(ub-ua, axis=1)[inside])
        s = np.concatenate(shifts) if shifts else np.zeros(0)
        out.append({'from_frame': int(a), 'to_frame': int(b), 'moving_point_camera_pairs': int(len(s)),
                    'pixels_median': float(np.median(s)) if len(s) else None, 'pixels_p10': float(np.quantile(s, .1)) if len(s) else None,
                    'pixels_p90': float(np.quantile(s, .9)) if len(s) else None,
                    'fraction_at_most_one_pixel': float((s <= 1).mean()) if len(s) else None})
    return out


def stats(x):
    return {'median': float(np.median(x)), 'p90': float(np.quantile(x, .9)), 'p99': float(np.quantile(x, .99)), 'count': int(len(x))} if len(x) else None


def mode_metrics(root, truth):
    completed = json.loads((root/'completed_frames.json').read_text())['completed_frames']
    X = np.load(root/'posterior_xyz.npy', mmap_mode='r')
    V = np.load(root/'velocity.npy', mmap_mode='r')
    acc = np.load(root/'update_acceptance.npy', mmap_mode='r')
    status = np.load(root/'point_status.npy', mmap_mode='r')
    conv = np.load(root/'converged.npy', mmap_mode='r')
    nis, dof = np.load(root/'nis.npy', mmap_mode='r'), np.load(root/'nis_dof.npy', mmap_mode='r')
    full = (root/'posterior_covariance.npy').exists()
    P = np.load(root/'posterior_covariance.npy', mmap_mode='r') if full else None
    D = np.load(root/'position_variance_diag.npy', mmap_mode='r')
    out = []
    for k in completed:
        require_valid_frame(root, k)
        e = np.asarray(X[k])-truth['X'][k]
        err = np.linalg.norm(e, axis=1)
        ve = np.linalg.norm(np.asarray(V[k], float)-truth['V'][k], axis=1)
        rec = {'frame': int(k), 'anchor': int(k) == truth['k0']}
        for name, group in (('static', ~truth['moving']), ('moving', truth['moving'])):
            rec[f'{name}_position_error_mm'] = stats(1e3*err[group])
            rec[f'{name}_velocity_error_m_s'] = stats(ve[group])
        if int(k) != truth['k0']:
            observed = status[k] == 1
            rec['observed_fraction'] = float(observed.mean())
            rec['prediction_only_fraction'] = float((status[k] == 2).mean())
            rec['accepted_update_fraction'] = float((acc[k] > 0).mean())
            rec['accepted_among_observed'] = float((acc[k][observed] > 0).mean()) if observed.any() else None
            rec['converged_among_observed'] = float(conv[k][observed].mean()) if observed.any() else None
            d = np.asarray(dof[k], float)
            ok = (d > 0) & np.isfinite(nis[k])
            rec['NIS_per_dof_median_observed'] = float(np.median(nis[k][ok]/d[ok])) if ok.any() else None
        nees_diag = (e*e/np.maximum(np.asarray(D[k], float), 1e-300)).sum(1)
        rec['NEES_diagonal_position'] = {'median': float(np.median(nees_diag)), 'inside_99pct': float((nees_diag <= chi2.isf(.01, 3)).mean()),
                                         'semantics': 'diagonal position variances only (both modes)'}
        if full:
            Pk = np.asarray(P[k][:, :3, :3], float)
            nees = np.einsum('ni,ni->n', e, np.linalg.solve(Pk, e[..., None])[..., 0])
            rec['NEES_full_position'] = {'median': float(np.median(nees)), 'inside_99pct': float((nees <= chi2.isf(.01, 3)).mean()),
                                         'moving_median': float(np.median(nees[truth['moving']])) if truth['moving'].any() else None,
                                         'static_median': float(np.median(nees[~truth['moving']])) if (~truth['moving']).any() else None}
        out.append(rec)
    return out


def traced(root, truth):
    rows = []
    ids = list(truth['point_id'])
    for path in sorted((root/'temporal/traced_points').glob('*.json')):
        tr = json.loads(path.read_text())
        i = ids.index(tr['point_id'])
        for f in tr['frames']:
            k = f['frame']
            s = np.asarray(f['s_plus'])
            rows.append({'original_point_id': tr['point_id'], 'frame': k, 'GT_moving': bool(truth['moving'][i]),
                         'position_error_mm': float(1e3*np.linalg.norm(s[:3]-truth['X'][k, i])),
                         'speed_m_s': float(np.linalg.norm(s[3:])), 'true_speed_m_s': float(np.linalg.norm(truth['V'][k, i])),
                         'accepted_camera_rows': len(f['rows']), 'measurement_update_accepted': f.get('measurement_update_accepted'),
                         'status': f.get('status'), 'reference_processing_age': f.get('reference_processing_age'),
                         'position_std_mm': (1e3*np.sqrt(np.diag(np.asarray(f['P_plus'])[:3, :3]))).tolist(),
                         'local_blocks_retained': len(f['local_posterior']['retained_layout']) if f.get('local_posterior') else None})
    return rows


def warp_check(run_root, scene, truth, device, frames_to_check, estimated=None):
    """Spec test 21: raw temporal RGB-D warp residual at GT motion (and at the estimated states), static vs moving.
    Reference = previous processed frame where the point is the GT first-hit owner; same masks/support rules as inference.
    Normalized by the independent sensor-only innovation R_cur + sum w^2 R_ref (no warp term): chi2_4 if the warp is exact."""
    from scene.dataset import Dataset
    from scene.config import load_config
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise
    from estimator.temporal_zbuffer import first_hit_pixel_owner
    from estimator.temporal_reference import ReferenceTable
    from estimator.temporal_rgbd_terms import build_temporal_rows
    man = json.loads((run_root/'run_manifest.json').read_text())
    top = json.loads((run_root.parent/'run_manifest.json').read_text())
    calib = json.loads((run_root.parent/'noise_calibration.json').read_text())['result']
    policy = top['policy']
    data = Dataset(scene, len(man['original_frame_indices']))
    cfg = load_config()
    cfg.color_covariance = np.diag(calib['rgb_variance'])
    cfg.depth_variance, cfg.depth_squared_variance = calib['depth_variance_m2'], calib['depth_squared_variance']
    rig = CameraRig(data.cameras, device, torch.float64)
    noise = SensorNoise.from_config(cfg, policy, device, torch.float64)
    order = man['processing_order']
    N, C = len(truth['point_id']), rig.count
    report = []
    for tag, states in (('GT', truth['X']), ('estimated', estimated)):
        if states is None:
            continue
        for ordinal in frames_to_check:
            k, kr = order[ordinal], order[ordinal-1]
            fr, fc = MeasuredFrame(data, kr, device, torch.float64), MeasuredFrame(data, k, device, torch.float64)
            sr = torch.zeros((N, 6), dtype=torch.float64, device=device)
            sc = torch.zeros_like(sr)
            sr[:, :3] = torch.as_tensor(np.asarray(states[kr]), device=device)
            sc[:, :3] = torch.as_tensor(np.asarray(states[k]), device=device)
            refs = ReferenceTable(C, N, device, torch.float64)
            refs.commit_accepted_references(first_hit_pixel_owner(sr[:, :3], rig, fr), fr, sr, 0)
            own = first_hit_pixel_owner(sc[:, :3], rig, fc, dense=False)
            rows, reason, life = build_temporal_rows(sc, own, fc, refs, {kr: fr, k: fc}, rig, noise, policy, torch.zeros((4, 4), dtype=torch.float64, device=device), 1, 2)
            r = (rows.y-rows.h)
            z = torch.linalg.solve(rows.R, r.unsqueeze(-1)).squeeze(-1)
            nis = (r*z).sum(1).cpu().numpy()
            std = (r/rows.R.diagonal(dim1=1, dim2=2).sqrt()).cpu().numpy()
            mv = truth['moving'][rows.point.cpu().numpy()]
            rec = {'states': tag, 'frame': int(k), 'reference_frame': int(kr), 'rows': len(rows), 'lifecycle': life,
                   'reason_counts': {int(v): int(n) for v, n in zip(*np.unique(reason.cpu().numpy(), return_counts=True))},
                   'owner_conflicts': int(own.conflict_count), 'chi2_4_median_expected': float(chi2.median(4))}
            for name, g in (('static', ~mv), ('moving', mv)):
                rec[name] = {'rows': int(g.sum()), 'normalized_NIS_median': float(np.median(nis[g])) if g.any() else None,
                             'normalized_NIS_p90': float(np.quantile(nis[g], .9)) if g.any() else None,
                             'abs_standardized_residual_median_rgb_depth': np.median(np.abs(std[g]), 0).tolist() if g.any() else None,
                             'depth_residual_rms_m': float(np.sqrt(np.mean(r[torch.as_tensor(g, device=device), 3].cpu().numpy()**2))) if g.any() else None}
            report.append(rec)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=Path, required=True, action='append', help='temporal_rgbd_* batch folder (repeatable)')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--warp-check-ordinals', default='1,2')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('output exists; never overwrite evaluations')
    report = {'ground_truth_namespace': 'POSTHOC_ONLY', 'inference_readback': False, 'qualification': QUALIFIER, 'scenes': {}}
    runs = {}
    for batch in a.batch:
        for d in sorted(p for p in batch.iterdir() if (p/'free_forward/run_manifest.json').exists() or (p/'free_backward/run_manifest.json').exists()):
            mode = json.loads((d/'run_manifest.json').read_text()).get('temporal', {}).get('mode', 'legacy')
            scene = d.name[:-len(mode)-1] if d.name.endswith('_'+mode) else d.name
            runs.setdefault(scene, {})[mode] = d
    for scene, modes in runs.items():
        entry = {'modes': {}, 'matched_point_ids': None}
        for direction in ('forward', 'backward'):
            roots = {m: d/f'free_{direction}' for m, d in modes.items() if (d/f'free_{direction}/completed_frames.json').exists()}
            if not roots:
                continue
            ids = {m: np.load(r/'canonical.npz')['point_id'] for m, r in roots.items()}
            first = next(iter(ids.values()))
            if not all(np.array_equal(first, v) for v in ids.values()):
                raise ValueError(f'{scene}: modes do not share the identical selected original point IDs; not a controlled comparison')
            truth = truth_for(next(iter(roots.values())), a.data_root/scene)
            entry['matched_point_ids'] = {'count': int(len(first)), 'GT_moving_points': int(truth['moving'].sum()),
                                          'true_mean_speed_moving_m_s': truth['true_speed_moving_m_s']}
            entry.setdefault('true_pixel_motion', {})[direction] = true_pixel_motion(a.data_root/scene, truth, next(iter(roots.values())))
            for m, r in roots.items():
                status = json.loads((r/'status.json').read_text())
                entry['modes'].setdefault(m, {})[direction] = {'status': status.get('status'), 'seconds': status.get('seconds'),
                                                               'frames': mode_metrics(r, truth), 'traced_points': traced(r, truth)}
            if 'local-correlated' in roots:
                ords = [int(x) for x in a.warp_check_ordinals.split(',') if int(x) < len(json.loads((roots['local-correlated']/'completed_frames.json').read_text())['completed_frames'])]
                est = np.load(roots['local-correlated']/'posterior_xyz.npy', mmap_mode='r')
                entry.setdefault('warp_check', {})[direction] = warp_check(roots['local-correlated'], a.data_root/scene, truth, a.device, ords, est)
        report['scenes'][scene] = entry
    json_write(a.output, serial(report))
    print(a.output)


if __name__ == '__main__':
    raise SystemExit(main())
