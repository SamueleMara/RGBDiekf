#!/usr/bin/env python3
"""GROUND-TRUTH motion summary of simulated (Sapien) scenes: reporting only, never read by an estimator.

Per scene: joint name / type / limits / axis / point of application (pointcloud/joint_metadata.json,
metadata/joint_info.json), the joint value per frame (metadata/frame_values.csv) and its increments, the moving part
(the labelled part of points3D_parts.ply whose centroid moves most to points3D_parts_final.ply, as
tools/evaluate_projective_gt.py), and the moving points' true motion: displacement from frame 0, speed (|dq/dt| for a
prismatic joint, |dq/dt| times the distance to the axis for a revolute one) and the per-frame image motion in pixels in
every camera (the free IEKF linearizes inside about one pixel cell).

    python tools/gt_motion_summary.py DATASET_ROOT_OR_SCENE [--out FOLDER]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.evaluate_projective_gt import gt_transform, gt_joint, labelled_endpoints    # noqa: E402


def scene_summary(sc):
    from scene.dataset import Dataset
    meta = json.loads((sc/'pointcloud/joint_metadata.json').read_text())
    info = json.loads((sc/'metadata/joint_info.json').read_text()).get('selected_joint', {})
    kind, axis, point, declared = gt_joint(sc)
    vals = np.genfromtxt(sc/'metadata/frame_values.csv', delimiter=',', names=True); q = np.asarray(vals[vals.dtype.names[1]], float)
    ds = Dataset(sc); t = np.asarray(ds.times, float); K = len(t); q = q[:K]
    dq = np.diff(q); dt = np.diff(t); qdot = dq/dt
    (x0, l0), (xf, lf), moving, label_note = labelled_endpoints(sc, kind, axis, point, q[-1]-q[0])
    P = x0[l0 == moving]
    unit = 'm' if kind == 'PRISMATIC' else 'rad'
    if kind == 'PRISMATIC': r = np.ones(len(P))
    else:
        dv = P-point; r = np.linalg.norm(dv-(dv@axis)[:, None]*axis, axis=1)                  # distance to the axis
    speed = np.abs(qdot)[:, None]*r[None]                                                       # [K-1, n] m/s
    traj = np.stack([P@gt_transform(kind, axis, point, qk-q[0])[:3, :3].T+gt_transform(kind, axis, point, qk-q[0])[:3, 3] for qk in q])   # [K,n,3]
    disp = np.linalg.norm(traj-traj[0], axis=2)
    px = []
    for c in ds.cameras:
        T = c.world_to_camera; fx, fy, cx, cy = c.intrinsics
        Xc = traj@T[:3, :3].T+T[:3, 3]; uv = np.stack([fx*Xc[..., 0]/Xc[..., 2]+cx, fy*Xc[..., 1]/Xc[..., 2]+cy], -1)
        inside = (Xc[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < c.width) & (uv[..., 1] >= 0) & (uv[..., 1] < c.height)
        step = np.linalg.norm(np.diff(uv, axis=0), axis=2); ok = inside[1:] & inside[:-1]
        px.append(np.where(ok, step, np.nan))
    px = np.stack(px)                                                                           # [C, K-1, n]
    q_ = lambda a, p: float(np.nanquantile(a, p)) if np.isfinite(a).any() else None
    return {'scene': sc.name, 'joint_name': meta.get('name'), 'type': kind, 'declared_type': declared, 'label_note': label_note, 'axis': axis.tolist(), 'point_of_application': point.tolist(),
            'limits': info.get('limits'), 'unit': unit, 'frames': K, 'fps': float(1/np.median(dt)), 'duration_s': float(t[-1]-t[0]),
            'q_first': float(q[0]), 'q_last': float(q[-1]), 'q_range': float(np.ptp(q)), 'q_total_change': float(q[-1]-q[0]),
            'dq_per_frame_min_median_max': [float(np.abs(dq).min()), float(np.median(np.abs(dq))), float(np.abs(dq).max())],
            'qdot_per_s_min_median_max': [float(np.abs(qdot).min()), float(np.median(np.abs(qdot))), float(np.abs(qdot).max())],
            'motion_profile': 'constant rate' if np.ptp(np.abs(dq)) <= 1e-6*max(np.abs(dq).max(), 1e-12) else 'varying rate',
            'monotonic': bool((np.sign(dq[dq != 0]) == np.sign(dq[dq != 0][0])).all()) if (dq != 0).any() else True,
            'moving_label': int(moving), 'labelled_points_total': int(len(l0)), 'moving_labelled_points': int(len(P)), 'moving_fraction_of_labelled': float(len(P)/len(l0)),
            'moving_distance_to_axis_m_min_max': None if kind == 'PRISMATIC' else [float(r.min()), float(r.max())],
            'moving_point_speed_m_s_median_max': [float(np.median(speed)), float(speed.max())],
            'moving_point_final_displacement_m_median_max': [float(np.median(disp[-1])), float(disp[-1].max())],
            'moving_point_max_displacement_any_frame_m': float(disp.max()),
            'image_motion_px_per_frame_median_p95_max': [q_(px, .5), q_(px, .95), float(np.nanmax(px))],
            'image_motion_px_per_frame_max_per_camera': [float(np.nanmax(p)) if np.isfinite(p).any() else None for p in px],
            'q_per_frame': q.tolist(),
            'note': 'GROUND TRUTH: reporting only; displacement / speed / pixels from the labelled endpoint part cloud moved by the GT joint'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path); ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    scenes = [a.root] if (a.root/'pointcloud/joint_metadata.json').exists() else sorted(p for p in a.root.iterdir() if (p/'pointcloud/joint_metadata.json').exists())
    rows = [scene_summary(sc) for sc in scenes]
    L = ['# Ground-truth motion of the simulated scenes (reporting only)', '',
         '| scene | joint | q first → last (range) | Δq per frame | q̇ median (max) | moving part | moving point speed median / max m/s | final displacement median / max m | image motion px/frame median / p95 / max |',
         '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        u = r['unit']; dqm = r['dq_per_frame_min_median_max']; qd = r['qdot_per_s_min_median_max']
        L.append(f"| {r['scene']} | {r['type']}{' (declared ' + r['declared_type'].lower() + ')' if r['declared_type'] != r['type'] else ''}{' *' if r['label_note'] else ''} | {r['q_first']:.4g} → {r['q_last']:.4g} {u} ({r['q_range']:.4g}) | {dqm[1]:.4g} {u} ({r['motion_profile']}) | {qd[1]:.4g} ({qd[2]:.4g}) {u}/s | "
                 f"label {r['moving_label']}: {r['moving_labelled_points']}/{r['labelled_points_total']} ({100*r['moving_fraction_of_labelled']:.0f}%) | "
                 f"{r['moving_point_speed_m_s_median_max'][0]:.3g} / {r['moving_point_speed_m_s_median_max'][1]:.3g} | "
                 f"{r['moving_point_final_displacement_m_median_max'][0]:.3g} / {r['moving_point_final_displacement_m_median_max'][1]:.3g} | "
                 f"{'/'.join(f'{x:.3g}' for x in r['image_motion_px_per_frame_median_p95_max'])} |")
    L += ['', '* the frame-0 labelled part cloud has no moving label; it is rebuilt from the final cloud by the inverse GT joint transform.',
          '', f"{len(rows)} scenes, {rows[0]['frames']} frames at {rows[0]['fps']:.4g} fps ({rows[0]['duration_s']:.3g} s) for the first; per-scene values, axis, limits and the joint value of every frame in gt_motion_summary.json."]
    text = '\n'.join(L)+'\n'
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out/'gt_motion_summary.json').write_text(json.dumps(rows, indent=1)); (a.out/'gt_motion_summary.md').write_text(text)


if __name__ == '__main__':
    main()
