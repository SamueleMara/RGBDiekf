#!/usr/bin/env python3
"""GROUND-TRUTH / proxy evaluation of a free F1 run, the SAME protocol for the original directional filter (free_forward/) and
the Phase-1 temporal solver (free_forward_temporal/) (RGBDiekf_Phase1 spec G7.2-G7.4). Separate from inference: reads the
run's outputs after they are written; nothing goes back into any estimator.

Static identity (offline only): the static basis of tools/audit_static_equilibrium.py (synthetic: GT part labels with the
purity rule; real: the DEPTH-CONSTANT PROXY over --window frames, not GT). Moving points (synthetic): nearest GT part label,
true trajectory by the GT joint transform (tools/evaluate_projective_gt.py). For a static point x_GT = x_ref.

    python tools/evaluate_temporal_phase1.py --run RUN --scene SCENE --basis-from AUDIT_BASELINE --out OUT.json [--window 50]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def q(x, qs=(.5, .9, .99)):
    x = np.asarray(x, float)
    return {f'p{int(100*a)}': float(np.quantile(x, a)) for a in qs} if x.size else None


def load(run):
    run = Path(run)
    if (run/'free_forward_temporal').exists():
        d = run/'free_forward_temporal'
        S = np.load(d/'smooth_mean.npy', mmap_mode='r')
        out = {'solver': 'temporal-ieks', 'X': S[..., :3], 'V': S[..., 3:], 'rows': np.load(d/'valid_rows.npy', mmap_mode='r'), 'dir': d}
        if (d/'forward_valid_rows.npy').exists(): out['rows_warm'] = np.load(d/'forward_valid_rows.npy', mmap_mode='r')
        out['b'] = np.load(d/'geometry_correction_b.npy'); out['traj_status'] = np.load(d/'trajectory_status.npy')
        out['dcov'] = np.load(d/'displacement_cov.npy', mmap_mode='r'); out['displacement'] = np.load(d/'displacement.npy', mmap_mode='r')
    else:
        d = run/'free_forward'
        out = {'solver': 'directional', 'X': np.load(d/'posterior_xyz.npy', mmap_mode='r'), 'V': np.load(d/'velocity.npy', mmap_mode='r'),
               'rows': np.load(d/'valid_cameras.npy', mmap_mode='r'), 'dir': d}
    out['can'] = dict(np.load(d/'canonical.npz')); out['times'] = np.asarray(json.loads((d/'run_manifest.json').read_text()).get('frame_times_s') or
                                                                          json.loads((d/'run_manifest.json').read_text())['timestamps_seconds'], float)
    return out


def decomposition(X, V, t):
    """x_T - x_0 = sum_k dt_k v_{k-1} (velocity integration) + sum_k (x_k - x_{k-1} - dt_k v_{k-1}) (everything else), projected
    on the net displacement direction (shares sum to 1; either may exceed 1 by cancellation)."""
    dt = np.diff(t)[:, None, None]
    vel = (dt*V[:-1]).sum(0); rest = (X[1:]-X[:-1]-dt*V[:-1]).sum(0); net = X[-1]-X[0]
    nn = np.linalg.norm(net, axis=-1); dh = net/np.where(nn > 0, nn, 1)[:, None]
    return (vel*dh).sum(-1)/np.where(nn > 0, nn, 1), (rest*dh).sum(-1)/np.where(nn > 0, nn, 1), nn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--basis-from', type=Path, required=True, help='audit baseline run holding static_audit/static_basis.npz + selection.json (same point ids)')
    ap.add_argument('--window', type=int, help='real scenes: evaluate frames 0..window (the proxy window)'); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists(): raise SystemExit(f'{a.out} exists: never overwritten')
    from tools.audit_static_equilibrium import gt_basis
    from tools.evaluate_projective_gt import gt_transform
    sel = json.loads((a.basis_from/'static_audit'/'selection.json').read_text())
    R = load(a.run)
    man = json.loads((R['dir']/'run_manifest.json').read_text())
    if man['selected_ids_sha256'] != sel['selected_ids_sha256']: raise SystemExit('point ids differ from the static basis run')
    static = np.load(a.basis_from/'static_audit'/'static_basis.npz')['static']
    K = R['X'].shape[0] if a.window is None else a.window+1
    X, V, t = np.asarray(R['X'][:K], float), np.asarray(R['V'][:K], float), R['times'][:K]
    x0 = R['can']['canonical_xyz']
    rep = {'run': str(a.run), 'solver': R['solver'], 'frames_evaluated': K, 'static_basis': sel['static_basis']['rule'], 'kind': sel['kind']}
    si = np.flatnonzero(static)
    err = np.linalg.norm(X[:, si]-x0[si][None], axis=-1)
    fin = err[-1]
    vs, rs, nn = decomposition(X[:, si], V[:, si], t)
    dep = fin > .10
    rep['static'] = {'points': int(len(si)), 'final_error_m' if sel['kind'] != 'real' else 'final_drift_from_x_ref_m': {**q(fin), 'over_10cm_fraction': float(dep.mean())},
                     'frame1_speed_m_s': q(np.linalg.norm(V[1, si], axis=-1)), 'final_speed_m_s': q(np.linalg.norm(V[-1, si], axis=-1)),
                     'velocity_integration_share_of_departures': q(vs[dep], (.1, .5, .9)) if dep.any() else None,
                     'other_share_of_departures': q(rs[dep], (.1, .5, .9)) if dep.any() else None}
    rows = np.asarray(R['rows'][:K], float)[:, si]
    had = np.maximum.accumulate(rows > 0, 0); lost = (rows == 0) & np.vstack([np.zeros((1, len(si)), bool), had[:-1]])
    first_loss = np.where(lost.any(0), lost.argmax(0), -1)
    rep['static']['time_to_first_loss_of_rows_s'] = q(t[first_loss[first_loss >= 0]]-t[0]) if (first_loss >= 0).any() else None
    rep['static']['never_lose_rows_fraction'] = float((first_loss < 0).mean())
    if 'rows_warm' in R:
        rw = np.asarray(R['rows_warm'][:K], float)[:, si]
        rep['relinearization'] = {'static_point_frames_with_rows_warm_start': int((rw > 0).sum()), 'static_point_frames_with_rows_final': int((rows > 0).sum()),
                                  'static_point_frames_recovered': int(((rw == 0) & (rows > 0)).sum()), 'static_point_frames_lost': int(((rw > 0) & (rows == 0)).sum())}
        b = R['b'][si]; rep['static']['geometry_correction_b_m'] = q(np.linalg.norm(b, axis=-1))
        rep['static']['trajectory_status_counts'] = {str(k): int(v) for k, v in zip(*np.unique(R['traj_status'][si], return_counts=True))}
        dc = np.asarray(R['dcov'][K-1, si], float); dd = np.asarray(R['displacement'][K-1, si], float) if sel['kind'] != 'real' else None
        if dd is not None:
            ne = np.einsum('ni,ni->n', dd, np.linalg.solve(dc+1e-30*np.eye(3), dd[..., None])[..., 0])
            rep['static']['displacement_nees_final'] = {**q(ne, (.5, .9, .99)), 'mean': float(ne.mean()), 'note': 'GT displacement 0; chi2_3 has mean 3, median 2.37'}
        ra = json.loads((R['dir']/'residual_audit.json').read_text())
        g = ra['groups'].get('all', {})
        rep['likelihood'] = {'mean_square_z_RGBD': g.get('mean_square_z_RGBD'), 'median_square_z_RGBD': ra.get('subsample_median_square_z_RGBD'),
                             'mean_z_RGBD': g.get('mean_z_RGBD'), 'lag1_autocorrelation_RGBD': ra.get('lag1_temporal_autocorrelation_z_RGBD')}
        rep['status'] = json.loads((R['dir']/'status.json').read_text())
    if sel['kind'] == 'synthetic':
        g = gt_basis(Path(a.scene), x0)
        fids = man.get('original_frame_ids') or json.loads((Path(a.run)/'run_manifest.json').read_text())['dataset']['original_frame_ids']
        qv = g['q'][fids] if len(g['q']) > max(fids) else g['q']
        mv = np.flatnonzero(g['nearest_label_moving'])
        T = [gt_transform(g['joint'], g['axis'], g['point'], qv[k]-qv[0]) for k in range(K)]
        gt = np.stack([x0[mv]@Tk[:3, :3].T+Tk[:3, 3] for Tk in T])
        e = np.linalg.norm(X[:, mv]-gt, axis=-1); mot = np.linalg.norm(gt[-1]-gt[0], axis=-1)
        rep['moving'] = {'points': int(len(mv)), 'true_motion_m': q(mot), 'final_error_m': q(e[-1]),
                         **{f'final_within_{tag}': float((e[-1] < v).mean()) for tag, v in (('5mm', .005), ('1cm', .01), ('5cm', .05))},
                         'median_error_over_time_m': q(np.median(e, 0)), 'final_error_over_true_motion_median': float(np.median(e[-1]/np.maximum(mot, 1e-9)))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in rep if k in ('solver', 'static', 'moving', 'relinearization')}, default=float)[:3000])


if __name__ == '__main__':
    main()
