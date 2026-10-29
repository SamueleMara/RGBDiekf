#!/usr/bin/env python3
"""POST-HOC ground-truth evaluation of a four-pass run (contract 9.2: a separate namespace, after every inference output
is written). Writes <run>/evaluation_gt/evaluation.json and evaluation.md; reads nothing back into any estimator.

Ground truth: pointcloud/joint_metadata.json (type, axis, point of application), metadata/frame_values.csv (joint value
per frame), sparse labelled part clouds points3D_parts(.ply / _final.ply) for the two endpoints. A canonical point gets
the label of the nearest labelled GT point (also reported: the fraction within 1 cm, as a labelling-quality figure).
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.run_paths import stage_dir


def ply(path):
    lines = Path(path).read_text().split('end_header\n')[1].split('\n')
    a = np.array([list(map(float, l.split())) for l in lines if l.strip()])
    return a[:, :3], a[:, 6].astype(int)


def gt_joint(scene):
    """(kind, axis, point, declared_type): a 'continuous' joint is a revolute joint without limits (REVOLUTE here, the
    declared type is reported)."""
    meta = json.loads((scene/'pointcloud/joint_metadata.json').read_text())
    declared = meta['type'].upper(); kind = 'REVOLUTE' if declared == 'CONTINUOUS' else declared
    axis = np.asarray(meta['axis_vector'], float); axis /= np.linalg.norm(axis)
    return kind, axis, np.asarray(meta['point_of_application'], float), declared


def labelled_endpoints(scene, kind, axis, point, dq_total):
    """Labelled part clouds of both endpoints and the moving label. An endpoint cloud that lacks the moving label
    (washer_103776: its frame-0 cloud is a single label) is rebuilt from the other endpoint with the GT joint
    transform (static labels kept in place, moving points mapped by the GT transform), and the fact is reported."""
    (x0, l0), (xf, lf) = ply(scene/'pointcloud/points3D_parts.ply'), ply(scene/'pointcloud/points3D_parts_final.ply')
    common = sorted(set(l0) & set(lf))
    shift = {l: np.linalg.norm(x0[l0 == l].mean(0)-xf[lf == l].mean(0)) for l in common}
    candidates = sorted(set(l0) | set(lf))
    note = None
    if len(common) >= 2: moving = max(common, key=shift.get)
    else:
        only = [l for l in candidates if l not in common]
        if not only: raise ValueError(f'{scene}: one part label at both endpoints, no moving part identifiable')
        moving = only[0]
        T = gt_transform(kind, axis, point, dq_total)
        if moving in set(lf):                                                 # rebuild frame 0 from the final cloud
            Ti = np.linalg.inv(T); mv = xf[lf == moving]@Ti[:3, :3].T+Ti[:3, 3]
            x0, l0 = np.r_[xf[lf != moving], mv], np.r_[lf[lf != moving], np.full(len(mv), moving)]
            note = 'frame-0 part cloud has no moving label: rebuilt from the final cloud by the inverse GT joint transform'
        else:
            mv = x0[l0 == moving]@T[:3, :3].T+T[:3, 3]
            xf, lf = np.r_[x0[l0 != moving], mv], np.r_[l0[l0 != moving], np.full(len(mv), moving)]
            note = 'final part cloud has no moving label: rebuilt from the frame-0 cloud by the GT joint transform'
    return (x0, l0), (xf, lf), int(moving), note


def gt_transform(kind, axis, point, dq):
    T = np.eye(4)
    if kind == 'PRISMATIC': T[:3, 3] = dq*axis
    else:
        R = Rotation.from_rotvec(dq*axis).as_matrix(); T[:3, :3] = R; T[:3, 3] = (np.eye(3)-R)@point
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--out', type=Path, help='output folder (default RUN/evaluation_gt); a post-hoc re-evaluation goes elsewhere, never over an existing one')
    a = ap.parse_args(); run = a.run; out_dir = a.out or run/'evaluation_gt'
    if a.out is not None and out_dir.exists() and any(out_dir.iterdir()): raise SystemExit(f'{out_dir} exists: never overwritten')
    out_dir.mkdir(parents=True, exist_ok=True)
    kind, axis, point, declared = gt_joint(a.scene)
    vals = np.genfromtxt(a.scene/'metadata/frame_values.csv', delimiter=',', names=True); q = np.asarray(vals[vals.dtype.names[1]], float)
    man = json.loads((run/'run_manifest.json').read_text()); frames = man['dataset']['original_frame_ids']; q = q[frames] if len(q) > max(frames) else q
    K = len(frames)
    (x0, l0), (xf, lf), moving, label_note = labelled_endpoints(a.scene, kind, axis, point, q[-1]-q[0])
    report = {'gt': {'type': kind, 'declared_type': declared, 'axis': axis.tolist(), 'point': point.tolist(), 'q_range': float(np.ptp(q)), 'moving_label': int(moving),
                     'label_note': label_note},
              'frames_evaluated': K, 'note': 'ground truth used ONLY here, after inference', 'directions': {}}
    for d, cloud in (('forward', (x0, l0)), ('backward', (xf, lf))):
        fd = stage_dir(run, f'free_{d}')
        if not (fd/'canonical.npz').exists(): continue
        can = np.load(fd/'canonical.npz'); k0 = int(can['canonical_frame']); P0 = can['canonical_xyz']
        dist, nn = cKDTree(cloud[0]).query(P0); lab = cloud[1][nn] == moving
        X = np.load(fd/'posterior_xyz.npy', mmap_mode='r'); S = np.load(fd/'point_status.npy', mmap_mode='r')
        r = {'canonical_frame': k0, 'points': int(len(P0)), 'gt_moving_points': int(lab.sum()), 'labelled_within_1cm_fraction': float((dist < .01).mean())}
        per = []
        for k in range(K):
            T = gt_transform(kind, axis, point, q[k]-q[k0]); tp = P0.copy(); tp[lab] = P0[lab]@T[:3, :3].T+T[:3, 3]     # static part stays
            e = np.linalg.norm(np.asarray(X[k], float)-tp, axis=1); mot = np.linalg.norm(tp-P0, axis=1)
            per.append({'frame': k, 'true_motion_m_median_moving': float(np.median(mot[lab])) if lab.any() else None,
                        'error_m_median_moving': float(np.median(e[lab])) if lab.any() else None,
                        'error_m_median_moving_measured': float(np.median(e[lab & (np.asarray(S[k]) == 1)])) if (lab & (np.asarray(S[k]) == 1)).any() else None,
                        'within_5mm_fraction_moving': float((e[lab] < .005).mean()) if lab.any() else None,
                        'error_m_median_static': float(np.median(e[~lab])) if (~lab).any() else None,
                        # tails (run-away points), static and moving populations separately
                        **{f'error_m_{q}_{nm}': (float(np.quantile(e[m], .99)) if q == 'p99' else float(e[m].max())) if m.any() else None
                           for nm, m in (('static', ~lab), ('moving', lab)) for q in ('p99', 'max')},
                        **{f'fraction_over_{t}_{nm}': float((e[m] > v).mean()) if m.any() else None
                           for nm, m in (('static', ~lab), ('moving', lab)) for t, v in (('1cm', .01), ('10cm', .1))}})
        r['free_per_frame'] = per
        last = 0 if d == 'backward' else K-1
        r['free_final'] = per[last]
        cs = stage_dir(run, f'coherence_{d}')/'coherence_summary.json'
        if cs.exists():
            summ = json.loads(cs.read_text()); r['coherence'] = {}
            for pol, v in summ['policies'].items():
                core = np.load(v['moving_core_point_ids_file'])
                r['coherence'][pol] = {'verdict': v['verdict'], 'core_points': int(len(core)),
                                       'precision': float(lab[core].mean()) if len(core) else None, 'recall': float(np.isin(np.flatnonzero(lab), core).mean()) if lab.any() else None}
        rp = stage_dir(run, f'rigid_{d}')/'poses.npy'
        if rp.exists():
            Pz = np.load(rp); errs = []
            for k in range(K):
                T = gt_transform(kind, axis, point, q[k]-q[k0]); dT = Pz[k]@np.linalg.inv(T)
                errs.append({'frame': k, 'translation_error_m': float(np.linalg.norm(dT[:3, 3])), 'rotation_error_deg': float(np.degrees(np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec()))),
                             'true_translation_m': float(np.linalg.norm(T[:3, 3])), 'true_rotation_deg': float(np.degrees(abs(q[k]-q[k0]))) if kind == 'REVOLUTE' else 0.})
            r['rigid'] = {'per_frame': errs, 'final': errs[last], 'max_translation_error_m': max(e['translation_error_m'] for e in errs)}
        jp = stage_dir(run, f'joint/{d}')/'result.json'
        if jp.exists():
            j = json.loads(jp.read_text()); jr = {'status': j.get('status'), 'gt_type': kind, 'type_correct': j.get('status') == kind}
            c = (j.get('candidates') or {}).get(kind)
            if c:
                e = np.asarray(c['axis']); jr['axis_angle_error_deg_up_to_sign'] = float(np.degrees(np.arccos(min(1., abs(e@axis)))))
                if kind == 'REVOLUTE' and c.get('axis_point') is not None:
                    dv = np.asarray(c['axis_point'])-point; jr['axis_line_distance_m'] = float(np.linalg.norm(dv-(dv@axis)*axis))
                qq = np.asarray(j['q_filter']['q'] if 'q_filter' in j and j['q_filter']['type'] == kind else c['q'])
                g = q-q[k0]; s = 1. if (qq@g) >= 0 else -1.
                jr['q_rmse_sign_aligned'] = float(np.sqrt(np.mean((s*qq-g)**2))); jr['q_source'] = 'q_filter' if 'q_filter' in j else 'reduction'
            r['joint'] = jr
        sp = stage_dir(run, f'joint/{d}/support')/'support.npz'
        if sp.exists():
            lb = np.load(sp, allow_pickle=True)['labels'].astype(str)
            mv = lb == 'MOVING_JOINT_COMPATIBLE'; st = lb == 'STATIC_COMPATIBLE'
            r['support'] = {'counts': {k: int(v) for k, v in zip(*np.unique(lb, return_counts=True))},
                            'moving_label_precision': float(lab[mv].mean()) if mv.any() else None, 'static_label_precision': float((~lab[st]).mean()) if st.any() else None}
        report['directions'][d] = r
    (out_dir/'evaluation.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Post-hoc GT evaluation: {run}", '', f"GT joint {kind}, q range {np.ptp(q):.4f}", '']
    for d, r in report['directions'].items():
        f = r['free_final']; lines.append(f"## {d} (canonical frame {r['canonical_frame']}, {r['points']} points, {r['gt_moving_points']} GT-moving)")
        lines.append(f"- free, final frame: moving true motion {f['true_motion_m_median_moving']}, moving error {f['error_m_median_moving']}, measured-moving error {f['error_m_median_moving_measured']}, static error {f['error_m_median_static']}")
        for pol, c in (r.get('coherence') or {}).items(): lines.append(f"- coherence {pol}: {c}")
        if 'rigid' in r: lines.append(f"- rigid final: {r['rigid']['final']}")
        if 'joint' in r: lines.append(f"- joint: {r['joint']}")
        if 'support' in r: lines.append(f"- support: {r['support']}")
        lines.append('')
    (out_dir/'evaluation.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
