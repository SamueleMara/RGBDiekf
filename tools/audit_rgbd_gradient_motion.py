#!/usr/bin/env python3
"""READ-ONLY offline audit of a completed material-following run: camera-by-camera velocity guidance, first false
velocity impulse, visibility/reference stages, raw gradients. GT (labels, true motion) is used ONLY here, after inference;
the run directory is never written.

Per traced point and frame (temporal/traced_points/<id>.json, rows at the accepted iterate):
  correction dx = x+ - x-, dv = v+ - v-; per camera row G = H_t[:, :3], residual f, row noise R = R_cur,eff + R_ref;
  information Lambda_c = G^T R^-1 G and the share of the correction direction carried by each camera,
  share_c = d^T Lambda_c d / d^T Lambda d (d = unit correction).
Events = frames whose correction or speed is the largest of the traced trajectory (top-k), so no threshold is invented;
the spec's historical 80 mm / 6 m/s values are only REPORTED (flags), never used for selection.
For requested frame pairs (a, b) every camera is re-measured at the published states: projection, first-hit owner at the
published cloud, foreground, bilinear support validity, image gradients I_u, I_v, D_u, D_v, Eulerian temporal differences
at the fixed subpixel location, J_pi, J_Z, depth-derived normal/tangents and the |cos| of the depth row to that normal.

    python -B tools/audit_rgbd_gradient_motion.py --scene blade_103706 --run-root RUN/blade_103706_material-following \
        --point-ids 14104,27016 --frames 0,1,2,3,4,5,6,7,23,24,25,26,30,59 --output-root OUT
"""
import argparse
import datetime
import json
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io import json_write, serial

DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
REASONS = {0: 'ROW', 1: 'BEHIND', 2: 'OUTSIDE', 3: 'BACKGROUND_OR_SUPPORT_MASK', 4: 'INVALID_DEPTH', 5: 'DEPTH_BREAK', 6: 'HIDDEN_BY_POINT',
           8: 'MEASURED_SURFACE_IN_FRONT', 11: 'NO_REFERENCE_BOOTSTRAP', 12: 'REFERENCE_OUTSIDE_LAG', 14: 'INTERP_SUPPORT_INCOMPLETE'}


def row_decomposition(frame_rec, rgb_var, depth_var, depth_sq):
    s_minus, s_plus = np.asarray(frame_rec['s_minus']), np.asarray(frame_rec['s_plus'])
    dx, dv = s_plus[:3]-s_minus[:3], s_plus[3:]-s_minus[3:]
    cams, Lam = [], np.zeros((3, 3))
    for r in frame_rec['rows']:
        G = np.asarray(r['H_t'])[:, :3]
        ref = np.asarray(r['reference_raw_rgbd'])
        Rd = np.asarray(r['R_current_effective_diag'])+np.r_[[rgb_var]*3, depth_var+depth_sq*ref[3]**2]
        Lc = G.T@(G/Rd[:, None])
        Lam += Lc
        cams.append({'camera': r['camera'], 'reference_frame': r['reference_frame'], 'reference_pixel': r['material_identity']['reference_pixel'],
                     'pixel_shift_px': r['pixel_shift_px'], 'q': r['q'], 'G_xyz': G.tolist(), 'residual_profiled': r['residual_profiled'],
                     'f_at_iterate': r['f_at_iterate'], 'R_diag': Rd.tolist(), 'Lambda': Lc.tolist(),
                     'depth_row_a': (-G[3]).tolist(), 'rgb_gradient_norm': float(np.linalg.norm(G[:3]))})
    n = np.linalg.norm(dx)
    d = dx/n if n > 0 else None
    tot = float(d@Lam@d) if d is not None else 0.
    for c in cams:
        c['share_of_correction_information'] = float(d@np.asarray(c['Lambda'])@d/tot) if d is not None and tot > 0 else None
    ev = np.linalg.eigvalsh(Lam) if cams else np.zeros(3)
    return {'frame': frame_rec['frame'], 'status': frame_rec['status'], 'rows': len(cams), 'correction_mm': 1e3*n,
            'velocity_change_m_s': float(np.linalg.norm(dv)), 'speed_m_s': float(np.linalg.norm(s_plus[3:])),
            'correction_direction': d.tolist() if d is not None else None, 'camera_reason': frame_rec['camera_reason'],
            'reference_frames': frame_rec['reference_frames'], 'reference_age': frame_rec['reference_processing_age'],
            'information_eigenvalues': ev.tolist(), 'position_std_mm': (1e3*np.sqrt(np.diag(np.asarray(frame_rec['P_plus'])[:3, :3]))).tolist(),
            'velocity_std_m_s': np.sqrt(np.diag(np.asarray(frame_rec['P_plus'])[3:, 3:])).tolist(), 'cameras': cams}


def remeasure(run, data, frames_pair, local_ids, original_ids, device, policy, noise):
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame
    from estimator.temporal_zbuffer import first_hit_pixel_owner
    from estimator.temporal_image_gradients import gradients_at, eulerian_difference
    from estimator.temporal_motion_geometry import velocity_rows, depth_basis, normal_alignment
    rig = CameraRig(data.cameras, device, torch.float64)
    X = np.load(run/'posterior_xyz.npy', mmap_mode='r')
    times = np.asarray(json.loads((run/'run_manifest.json').read_text())['timestamps_seconds'])
    out = []
    cache = {}
    get = lambda k: cache.setdefault(k, MeasuredFrame(data, k, device, torch.float64))
    for a, b in frames_pair:
        Fa, Fb = get(a), get(b)
        xb_all = torch.as_tensor(np.asarray(X[b]), device=device)
        own = first_hit_pixel_owner(xb_all, rig, Fb, dense=False)
        dt = float(times[b]-times[a])
        for li, oi in zip(local_ids, original_ids):
            x = xb_all[li][None].expand(rig.count, 3)
            cam = torch.arange(rig.count, device=device)
            u, z, _, _ = rig.project_rows(x, cam)
            g = gradients_at(Fb, cam, u, noise, policy, rig)
            de, ok, _ = eulerian_difference(Fa, Fb, cam, u, noise, policy, rig)
            G, av, Jpi, Jz = velocity_rows(rig, cam, x, g.grad_rgb, g.grad_depth)
            B, okB = depth_basis(rig, cam, u, g.depth, g.grad_depth, g.valid)
            na = normal_alignment(av, B)
            rec = {'from_frame': a, 'to_frame': b, 'dt_s': dt, 'original_id': int(oi), 'local_index': int(li), 'cameras': []}
            for c in range(rig.count):
                rec['cameras'].append({'camera': c, 'u': u[c].tolist(), 'Z': float(z[c]), 'first_hit_reason': int(own.reason[c, li]),
                                       'support_reason': int(g.reason[c]), 'I_u_I_v': g.grad_rgb[c].tolist(), 'D_u_D_v': g.grad_depth[c].tolist(),
                                       'I_t_D_t_eulerian_per_s': (de[c]/dt).tolist() if bool(ok[c]) else None,
                                       'eulerian_valid': bool(ok[c]), 'J_pi': Jpi[c].tolist(), 'J_Z': Jz[c].tolist(),
                                       'G_I': G[c].tolist(), 'a_depth_row': av[c].tolist(), 'basis_t1_t2_n': B[c].tolist() if bool(okB[c]) else None,
                                       'depth_row_normal_abs_cos': float(na[c]) if bool(okB[c]) else None})
            out.append(rec)
    return out


def stage_histograms(run, truth_moving):
    R = np.load(run/'camera_reason.npy', mmap_mode='r')
    done = json.loads((run/'completed_frames.json').read_text())['completed_frames']
    out = []
    for k in done:
        r = np.asarray(R[k])
        rec = {'frame': int(k)}
        for name, g in (('moving', truth_moving), ('static', ~truth_moving)):
            codes, counts = np.unique(r[:, g], return_counts=True)
            rec[name] = {REASONS.get(int(c), str(int(c))): int(n) for c, n in zip(codes, counts)}
            rec[name+'_points_with_any_row'] = int((r[:, g] == 0).any(0).sum())
        out.append(rec)
    return out


def drift(run, truth, frames):
    X = np.load(run/'posterior_xyz.npy', mmap_mode='r')
    V = np.load(run/'velocity.npy', mmap_mode='r')
    out = []
    q = lambda x: {'p50': float(np.quantile(x, .5)), 'p90': float(np.quantile(x, .9)), 'p99': float(np.quantile(x, .99)), 'max': float(x.max())} if len(x) else None
    for k in frames:
        if k >= len(truth['X']):
            continue
        e = np.linalg.norm(np.asarray(X[k])-truth['X'][k], axis=1)
        sp = np.linalg.norm(np.asarray(V[k], float), axis=1)
        ve = np.linalg.norm(np.asarray(V[k], float)-truth['V'][k], axis=1)
        out.append({'frame': int(k), 'static_error_mm': q(1e3*e[~truth['moving']]), 'static_speed_m_s': q(sp[~truth['moving']]),
                    'moving_error_mm': q(1e3*e[truth['moving']]), 'moving_velocity_error_m_s': q(ve[truth['moving']])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--run-root', type=Path, required=True, help='<batch>/<scene>_material-following (read only)')
    ap.add_argument('--point-ids', required=True, help='original catalogue IDs with traced records')
    ap.add_argument('--frames', default='0,1,2,3,4,5,6,7,23,24,25,26,30,59')
    ap.add_argument('--pairs', default='0:1,1:2,4:7,23:26', help='frame pairs re-measured in every camera')
    ap.add_argument('--data-root', type=Path, default=DATA)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--top-events', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    from tools.evaluate_temporal_local_pilot import truth_for
    from scene.dataset import Dataset
    from scene.config import load_config
    from estimator.projective_sensor import SensorNoise
    out = a.output_root/f'gradient_audit_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    ids = [int(x) for x in a.point_ids.split(',')]
    frames = [int(x) for x in a.frames.split(',')]
    pairs = [tuple(int(y) for y in p.split(':')) for p in a.pairs.split(',')]
    calib = json.loads((a.run_root/'noise_calibration.json').read_text())['result']
    top = json.loads((a.run_root/'run_manifest.json').read_text())
    cfg = load_config()
    cfg.color_covariance = np.diag(calib['rgb_variance'])
    cfg.depth_variance, cfg.depth_squared_variance = calib['depth_variance_m2'], calib['depth_squared_variance']
    noise = SensorNoise.from_config(cfg, top['policy'], a.device, torch.float64)
    report = {'ground_truth_namespace': 'POSTHOC_GT_AUDIT_ONLY', 'run_root': str(a.run_root), 'run_written': False, 'directions': {}}
    for direction in ('forward', 'backward'):
        run = a.run_root/f'free_{direction}'
        if not (run/'completed_frames.json').exists():
            continue
        truth = truth_for(run, a.data_root/a.scene)
        can = np.load(run/'canonical.npz')
        pid = list(can['point_id'])
        local = [pid.index(i) for i in ids if i in pid]
        rec = {'labels': 'NN part-cloud labels (truth_for); see temporal_dynamic_gt_warp_audit for label-verification caveats', 'points': {}}
        for i, li in zip([i for i in ids if i in pid], local):
            tr = json.loads((run/f'temporal/traced_points/{i}.json').read_text())
            dec = [row_decomposition(f, calib['rgb_variance'][0], calib['depth_variance_m2'], calib['depth_squared_variance']) for f in tr['frames']]
            for dd in dec:
                k = dd['frame']
                dd['GT_velocity_m_s'] = truth['V'][k, li].tolist()
                dd['GT_position_error_mm_after'] = float(1e3*np.linalg.norm(np.asarray(tr['frames'][[f['frame'] for f in tr['frames']].index(k)]['s_plus'][:3])-truth['X'][k, li]))
                dd['flag_historical_80mm_or_6m_s'] = bool(dd['correction_mm'] > 80 or dd['speed_m_s'] > 6)
            by_corr = sorted(dec, key=lambda r: -r['correction_mm'])[:a.top_events]
            first = next((r for r in dec if r['correction_mm'] > 0 and r['velocity_change_m_s'] > 0), None)
            rec['points'][str(i)] = {'GT_moving': bool(truth['moving'][li]), 'local_index': li, 'frames': dec,
                                     'largest_corrections': [{k: r[k] for k in ('frame', 'correction_mm', 'speed_m_s', 'rows', 'status', 'camera_reason')} |
                                                             {'cameras': [{k: c[k] for k in ('camera', 'reference_frame', 'pixel_shift_px', 'share_of_correction_information',
                                                                                              'residual_profiled', 'depth_row_a', 'rgb_gradient_norm')} for c in r['cameras']]}
                                                             for r in by_corr],
                                     'first_nonzero_velocity_impulse': first and {k: first[k] for k in ('frame', 'correction_mm', 'velocity_change_m_s', 'speed_m_s', 'rows', 'cameras')}}
        data = Dataset(a.data_root/a.scene)
        order = json.loads((run/'run_manifest.json').read_text())['processing_order']
        dpairs = pairs if direction == 'forward' else [(b, a_) for a_, b in pairs]
        rec['remeasured_all_cameras'] = remeasure(run, data, [p for p in dpairs if max(p) < len(order)], local, [i for i in ids if i in pid], a.device, top['policy'], noise)
        rec['stage_histograms'] = stage_histograms(run, truth['moving'])
        rec['drift'] = drift(run, truth, frames)
        report['directions'][direction] = rec
        print(f'[audit] {direction}: traced {list(rec["points"])}', flush=True)
    json_write(out/'gradient_motion_audit.json', serial(report))
    print(out)


if __name__ == '__main__':
    raise SystemExit(main())
