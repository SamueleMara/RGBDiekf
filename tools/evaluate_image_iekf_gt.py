#!/usr/bin/env python3
"""Evaluation ONLY, after inference: score an image-IEKF run against the dataset GT.

Nothing here is read by any estimator stage. Part labels come from the sparse GT
part cloud (nearest GT point within 1 cm), so core precision/recall are approximate.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def ply(path):
    lines = Path(path).read_text().split('end_header\n')[1].split('\n')
    a = np.array([list(map(float, l.split())) for l in lines if l.strip()])
    return a[:, :3], a[:, 6].astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    meta = json.loads((a.scene/'pointcloud/joint_metadata.json').read_text())
    axis = np.asarray(meta['axis_vector'], float); jtype = meta['type'].upper()
    vals = np.genfromtxt(a.scene/'metadata/frame_values.csv', delimiter=',', names=True)
    q_gt = np.asarray(vals[vals.dtype.names[1]], float)
    K = len(json.loads((a.run/'manifest.json').read_text())['dataset']['original_frame_ids'])
    report = {'gt_type': jtype, 'gt_axis': axis.tolist(), 'gt_q_range': float(np.ptp(q_gt)), 'directions': {}, 'run_frames': K, 'gt_frames': len(q_gt)}
    if K < len(q_gt):   # --frame-limit TEST prefix: score the prefix; the 'last' part labels come from the true final frame, so they are approximate
        report['frame_limited_note'] = 'test prefix: q truncated to the run frames; last-anchor GT part cloud is the true final frame, not the prefix end'
        q_gt = q_gt[:K]
    (xi, li), (xf, lf) = ply(a.scene/'pointcloud/points3D_parts.ply'), ply(a.scene/'pointcloud/points3D_parts_final.ply')
    moving_label = max(set(li) & set(lf), key=lambda l: np.linalg.norm(xi[li == l].mean(0)-xf[lf == l].mean(0)))
    for anchor, cloud in (('first', 'points3D_parts.ply'), ('last', 'points3D_parts_final.ply')):
        d = a.run/anchor
        p1 = d/'phase1'
        if not (p1/'partition.npz').exists(): continue  # direction not run (reused stages are symlinked folders, so they resolve here)
        k0 = 0 if anchor == 'first' else K-1
        part = np.load(p1/'partition.npz'); xyz = part['xyz']
        x0, l0 = ply(a.scene/'pointcloud'/cloud)
        dist, nn = cKDTree(x0).query(xyz); lab = np.where(dist < .01, l0[nn], -1)
        X = np.load(p1/'state.npy', mmap_mode='r')
        r = {'anchor_frame': k0, 'labelled_points': int((lab >= 0).sum())}
        if jtype == 'PRISMATIC':
            slide = lambda k: (X[k][:, :3]-xyz)@axis
            gt = (q_gt-q_gt[k0])
            r['phase1_slide'] = {int(k): {'gt_m': float(gt[k]), 'moving_p25_p50_p90_m': np.percentile(slide(k)[lab == moving_label], [25, 50, 90]).round(4).tolist(),
                                          'static_median_m': float(np.median(slide(k)[lab == 1-moving_label]))} for k in range(0, K, 10)}
        if (d/'coherence/representatives.npz').exists():
            sup = np.load(d/'coherence/representatives.npz')['moving_support']
            known = lab >= 0
            r['core'] = {'points': int(sup.sum()), 'precision_labelled': float((lab[sup & known] == moving_label).mean()) if (sup & known).any() else None,
                         'recall_labelled': float(sup[lab == moving_label].mean())}
        for stage, sub in (('rigid', 'rigid/poses.npy'), ('kabsch', 'rigid/kabsch_poses.npy')):
            if not (d/sub).exists(): continue
            T = np.load(d/sub)
            if jtype == 'PRISMATIC':
                t_err = [np.linalg.norm(T[k][:3, 3]-(q_gt[k]-q_gt[k0])*axis) for k in range(K)]
                r_err = [np.degrees(np.linalg.norm(Rotation.from_matrix(T[k][:3, :3]).as_rotvec())) for k in range(K)]
                r[stage] = {'translation_error_mean_max_m': [float(np.mean(t_err)), float(np.max(t_err))], 'rotation_error_mean_max_deg': [float(np.mean(r_err)), float(np.max(r_err))]}
            else:   # revolute about the GT axis line; the GT angle sign/unit is resolved by the best match (evaluation only)
                pa = np.asarray(meta['point_of_application'], float)
                best = None
                for scale in (1., -1., np.pi/180, -np.pi/180):
                    errs = []
                    for k in range(K):
                        Rg = Rotation.from_rotvec(axis/np.linalg.norm(axis)*scale*(q_gt[k]-q_gt[k0])).as_matrix()
                        dR = Rotation.from_matrix(Rg.T@T[k][:3, :3]).as_rotvec()
                        # translation error at the moving part's centroid scale: compare the transforms on the axis point
                        errs.append((np.degrees(np.linalg.norm(dR)), np.linalg.norm(T[k][:3, :3]@pa+T[k][:3, 3]-pa)))
                    errs = np.array(errs)
                    if best is None or errs[:, 0].mean() < best[1][:, 0].mean(): best = (scale, errs)
                r[stage] = {'rotation_error_mean_max_deg': [float(best[1][:, 0].mean()), float(best[1][:, 0].max())],
                            'axis_point_displacement_mean_max_m': [float(best[1][:, 1].mean()), float(best[1][:, 1].max())], 'gt_angle_scale': best[0]}
        if (d/'joint/result.json').exists():
            j = json.loads((d/'joint/result.json').read_text())
            r['joint'] = {'status': j['status'], 'reason': j['reason']}
            for t, c in j['candidates'].items():
                ang = float(np.degrees(np.arccos(min(1., abs(np.asarray(c['axis'])@axis)))))
                q = np.asarray(c['q']); s = np.sign(np.asarray(c['axis'])@axis) or 1.
                if t == jtype and jtype == 'PRISMATIC': qe = float(np.sqrt(np.mean((s*q-(q_gt-q_gt[k0]))**2)))
                elif t == jtype: qe = min(float(np.sqrt(np.mean((s*q-sc*(q_gt-q_gt[k0]))**2))) for sc in (1., np.pi/180))
                else: qe = None
                r['joint'][t] = {'axis_error_deg': ang, 'q_rmse': qe}
                if t == 'REVOLUTE' and jtype == 'REVOLUTE':
                    pa, e = np.asarray(meta['point_of_application'], float), np.asarray(c['axis'], float)
                    r['joint'][t]['axis_line_distance_m'] = float(np.linalg.norm(np.cross(np.asarray(c['axis_point'])-pa, e)))
        if (a.run/anchor/'support/support.npz').exists():
            s = np.load(a.run/anchor/'support/support.npz', allow_pickle=True); labels = s['labels']
            known = lab >= 0
            r['support'] = {name: {'count': int((labels == name).sum()), 'moving_fraction_labelled': float((lab[(labels == name) & known] == moving_label).mean()) if ((labels == name) & known).any() else None}
                            for name in ('MOVING_CORE', 'MOVING_JOINT_COMPATIBLE', 'STATIC_COMPATIBLE', 'AMBIGUOUS')}
        report['directions'][anchor] = r
    text = json.dumps(report, indent=1)
    if a.out: a.out.write_text(text)
    print(text)


if __name__ == '__main__':
    raise SystemExit(main())
