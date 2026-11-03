#!/usr/bin/env python3
"""POST-INFERENCE ONLY: material-following vs legacy / simplified / local-correlated on identical original point IDs.

Ground truth (joint metadata, labelled part clouds, frame values and, for label verification, distant frames) is read
HERE ONLY, after inference, from journal-verified frames. Two moving/static labelings are reported:
  * NN: nearest labelled-part-cloud point (the rule of evaluate_temporal_local_pilot.truth_for);
  * VERIFIED: NN label AND the multi-frame on-surface vote of tools/audit_temporal_dynamic_correspondence agree
    (points where they disagree are excluded from VERIFIED statistics and counted).

    python -B tools/evaluate_temporal_material_pilot.py --batch RESULTS/temporal_rgbd_A --batch RESULTS/temporal_rgbd_B \
        --data-root DATA --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_temporal_local_pilot import truth_for, mode_metrics, true_pixel_motion, stats
from utils.io import json_write, serial
from utils.temporal_output import require_valid_frame

DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')


def verified_moving(run_root, scene, truth, device):
    from scene.dataset import Dataset
    from scene.config import load_config
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import SensorNoise
    from tools.audit_temporal_dynamic_correspondence import verified_labels
    top = json.loads((run_root.parent/'run_manifest.json').read_text())
    calib = json.loads((run_root.parent/'noise_calibration.json').read_text())['result']
    cfg = load_config()
    cfg.color_covariance = np.diag(calib['rgb_variance'])
    cfg.depth_variance, cfg.depth_squared_variance = calib['depth_variance_m2'], calib['depth_squared_variance']
    full = Dataset(scene)
    rig = CameraRig(full.cameras, device, torch.float64)
    noise = SensorNoise.from_config(cfg, top['policy'], device, torch.float64)
    can = np.load(run_root/'canonical.npz')
    K = len(full.times)
    v, votes, rep = verified_labels(scene, rig, noise, top['policy'], can['canonical_xyz'], truth['moving'], device, sorted({K//2, K-1}))
    return v, rep


def completed(root):
    """Verified-frame journal of temporal modes; the legacy free filter has no journal, so a COMPLETE legacy status means
    every frame of its arrays was written (labelled LEGACY_NO_JOURNAL in the report)."""
    if (root/'completed_frames.json').exists():
        return json.loads((root/'completed_frames.json').read_text())['completed_frames'], True
    st = json.loads((root/'status.json').read_text())
    if st.get('status', 'COMPLETE') != 'COMPLETE' and 'frames' not in st:
        return [], False
    return list(range(np.load(root/'posterior_xyz.npy', mmap_mode='r').shape[0])), False


def decomposition(root, truth, groups):
    X = np.load(root/'posterior_xyz.npy', mmap_mode='r')
    done, journal = completed(root)
    out = []
    for k in done:
        if k == truth['k0']:
            continue
        kp = truth['k0']             # cumulative from the anchor: fraction of the TOTAL true displacement recovered
        if journal:
            require_valid_frame(root, k)
        d = truth['X'][k]-truth['X'][kp]
        e = np.asarray(X[k])-truth['X'][kp]
        dn = np.linalg.norm(d, axis=1)
        dh = d/np.maximum(dn, 1e-12)[:, None]
        along = (e*dh).sum(1)
        perp = np.linalg.norm(e-along[:, None]*dh, axis=1)
        err = np.linalg.norm(np.asarray(X[k])-truth['X'][k], axis=1)
        rec = {'frame': int(k)}
        for name, g in groups.items():
            m = g & (dn > 0) if 'moving' in name else g
            rec[name] = {'points': int(g.sum()), 'position_error_mm': stats(1e3*err[g]),
                         'cumulative_estimated_displacement_along_true_ratio_quantiles_10_25_50_75_90': np.quantile(along[m]/dn[m], [.1, .25, .5, .75, .9]).tolist() if m.any() and 'moving' in name else None,
                         'cumulative_perpendicular_displacement_mm': stats(1e3*perp[m]) if 'moving' in name and m.any() else None,
                         'moving_points_within_3mm_fraction': float((err[g] < 3e-3).mean()) if 'moving' in name and g.any() else None}
        out.append(rec)
    return out


def material_extras(root, truth, groups):
    done = json.loads((root/'completed_frames.json').read_text())['completed_frames']
    out = []
    arr = {name: np.load(root/f'{name}.npy', mmap_mode='r') for name in ('material_rows', 'material_pixel_shift_max_px', 'material_target_pixel_changed',
                                                                         'material_info_eig_min', 'material_info_eig_max', 'update_acceptance')}
    for k in done:
        if k == truth['k0']:
            continue
        fr = json.loads((root/f'temporal/frame_{k:04d}.json').read_text())
        rec = {'frame': int(k), 'identity_counters': fr['material']['identity_counters'], 'lifecycle': fr['material']['lifecycle'],
               'reference_commit': fr['material']['reference_commit'], 'coarse_to_fine': fr['material'].get('coarse_to_fine'),
               'seconds': fr['seconds'], 'cuda_peak_allocated_bytes': fr['stage_times'].get('cuda_peak_allocated_bytes'),
               'solver_history': [{k2: h[k2] for k2 in ('iteration', 'accepted_points', 'backtracks', 'full_step_rejections', 'identities_lost_to_occlusion',
                                                       'identities_gained')} for h in fr['solver']['history']]}
        for name, g in groups.items():
            rows = np.asarray(arr['material_rows'][k])[g]
            acc = np.asarray(arr['update_acceptance'][k])[g] > 0
            sh = np.asarray(arr['material_pixel_shift_max_px'][k])[g]
            emin, emax = np.asarray(arr['material_info_eig_min'][k], float)[g], np.asarray(arr['material_info_eig_max'][k], float)[g]
            obs = rows > 0
            rec[name] = {'points': int(g.sum()), 'with_material_rows_fraction': float(obs.mean()) if len(obs) else None,
                         'assimilated_fraction': float(acc.mean()) if len(acc) else None,
                         'assimilated_max_pixel_shift_px': stats(sh[acc & np.isfinite(sh)]),
                         'assimilated_with_shift_above_1px_fraction': float((sh[acc & np.isfinite(sh)] > 1).mean()) if (acc & np.isfinite(sh)).any() else None,
                         'target_pixel_changed_fraction': float(np.asarray(arr['material_target_pixel_changed'][k])[g].mean()) if len(rows) else None,
                         'rank_deficient_relative_1e-12_fraction_of_observed': float((emin[obs] <= 1e-12*emax[obs]).mean()) if obs.any() else None,
                         'weakest_direction_std_mm_observed': stats(1e3/np.sqrt(np.maximum(emin[obs], 1e-300))) if obs.any() else None}
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=Path, required=True, action='append')
    ap.add_argument('--data-root', type=Path, default=DATA)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('output exists; never overwrite evaluations')
    runs = {}
    for batch in a.batch:
        for d in sorted(p for p in batch.iterdir() if (p/'free_forward/run_manifest.json').exists()):
            man = json.loads((d/'run_manifest.json').read_text())
            mode = man.get('temporal', {}).get('mode', 'legacy') if man.get('observation_model') == 'temporal-rgbd' else 'legacy'
            scene = d.name[:-len(mode)-1] if d.name.endswith('_'+mode) else d.name
            runs.setdefault(scene, {})
            if mode in runs[scene]:
                raise ValueError(f'{scene}: two {mode} runs given; pass one batch per mode')
            runs[scene][mode] = d/'free_forward'
    report = {'ground_truth_namespace': 'POSTHOC_ONLY', 'inference_readback': False, 'scenes': {}}
    for scene, roots in runs.items():
        roots = {m: r for m, r in roots.items() if completed(r)[0]}
        ids = {m: np.load(r/'canonical.npz')['point_id'] for m, r in roots.items()}
        first = next(iter(ids.values()))
        same = {m: bool(np.array_equal(first, v)) for m, v in ids.items()}
        if not all(same.values()):
            raise ValueError(f'{scene}: modes do not share identical original point IDs: {same}')
        ref = next(iter(roots.values()))
        truth = truth_for(ref, a.data_root/scene)
        v, label_report = verified_moving(ref, a.data_root/scene, truth, a.device)
        groups = {'moving_NN': truth['moving'], 'static_NN': ~truth['moving'], 'moving_VERIFIED': v == 1, 'static_VERIFIED': v == 0}
        entry = {'matched_point_ids': int(len(first)), 'modes': sorted(roots), 'label_verification': label_report,
                 'true_pixel_motion': true_pixel_motion(a.data_root/scene, truth, ref), 'per_mode': {}}
        for m, r in roots.items():
            st = json.loads((r/'status.json').read_text())
            journal = completed(r)[1]
            entry['per_mode'][m] = {'status': st.get('status', 'COMPLETE'), 'seconds': st.get('seconds'), 'run_root': str(r),
                                    'frame_journal': 'VERIFIED_FRAME_JOURNAL' if journal else 'LEGACY_NO_JOURNAL',
                                    'metrics_NN_labels': mode_metrics(r, truth) if journal else None, 'decomposition': decomposition(r, truth, groups)}
            if m == 'material-following':
                entry['per_mode'][m]['material'] = material_extras(r, truth, groups)
        report['scenes'][scene] = entry
    json_write(a.output, serial(report))
    print(a.output)


if __name__ == '__main__':
    raise SystemExit(main())
