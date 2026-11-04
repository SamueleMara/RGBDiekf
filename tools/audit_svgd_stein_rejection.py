#!/usr/bin/env python3
"""POST-HOC audit of a Stein-proposal rejection (audit b62c7cee, F2/F3/F6). Separate process, AFTER the run's inference
artifacts are committed; GT membership/trajectories are used here only, never by the runner.

    python -B tools/run_svgd_support.py ... --stop-after-step STEIN_PROPOSAL --save-stein-diagnostics --output RUN
    python -B tools/audit_svgd_stein_rejection.py --run RUN --scene SCENE            # Sapien GT (metadata/frame_values.csv + joint)
    python -B tools/audit_svgd_stein_rejection.py --run RUN --membership moving.npy [--gt-trajectory X.npy]   # synthetic oracle

Reports (RUN/posthoc/stein_rejection_audit.{json,md}):
  * energy decomposition per backtrack for both proposal families (positions only = original B3 map; conditional temporal
    lift = DEV-0011): dE_obs, dE_time, and the directional derivatives <grad E_obs, P>, <grad E_time, P> (from the run);
  * cross-support kernel mass: for static receivers, the fraction of sum_l K(l, i) coming from GT-moving sources (and vice
    versa), with the exact split of the Stein field into moving-source and static-source parts (source_mask; parts add up);
  * fraction of static points whose Stein displacement is dominated by moving sources; |Phi| on static vs moving points;
  * alignment: cos(Phi, -grad E_obs) per GT group and cos(Phi_i, true displacement_i) for moving points (GT post hoc).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimator.svgd_particle_operator import stein_field


def q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return {'n': int(len(v)), 'median': float(np.median(v)), 'p10': float(np.quantile(v, .1)), 'p90': float(np.quantile(v, .9))} if len(v) else {'n': 0}


def cosine(a, b):
    num = (a*b).sum(-1); den = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.where(den > 0, num/np.maximum(den, 1e-300), np.nan)


def gt_from_scene(scene, run):
    from scene.dataset import Dataset
    from tools.audit_temporal_dynamic_correspondence import ground_truth
    man = json.loads((run/'run_manifest.json').read_text()); K = len(man['dataset']['original_frame_ids'])
    can = np.load(run/'svgd_support/canonical.npz')
    X, moving, dist, ev = ground_truth(Path(scene), Dataset(scene, K), can['xyz0'])
    return moving, X, ev


def audit(run, moving, Xgt=None, device='cuda'):
    run = Path(run); d = np.load(run/'svgd_support/stein_diagnostics.npz'); st = json.loads((run/'svgd_support/status.json').read_text())
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu'); t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
    X, Phi, g, gobs = d['X'], d['Phi'], d['score'], d['grad_Eobs_x']; N, K = X.shape[:2]; mv = np.asarray(moving, bool)
    if len(mv) != N: raise ValueError(f'membership has {len(mv)} entries, run has N = {N}')
    kw = dict(attraction=bool(d['attraction_enabled']), repulsion=bool(d['repulsion_enabled']), kernel_color=bool(d['kernel_color']))
    frames = []
    for k in range(K):
        parts = {name: stein_field(t(X[:, k]), t(d['rgb0']), t(g[:, k]), t(d['H_x']), t(d['H_c']), source_mask=t(mask), **kw)
                 for name, mask in (('from_moving', mv.astype(float)), ('from_static', (~mv).astype(float)))}
        tot = (parts['from_moving'].total+parts['from_static'].total).cpu().numpy()
        recon = float(np.abs(tot-Phi[:, k]).max()/max(np.abs(Phi[:, k]).max(), 1e-300))
        mass_m, mass_s = parts['from_moving'].kernel_mass.cpu().numpy(), parts['from_static'].kernel_mass.cpu().numpy()
        fm, fs = parts['from_moving'].total.cpu().numpy(), parts['from_static'].total.cpu().numpy()
        row = {'frame': k, 'split_reconstruction_rel_error': recon,
               'static_receivers_mass_fraction_from_moving': q((mass_m/(mass_m+mass_s))[~mv]), 'moving_receivers_mass_fraction_from_static': q((mass_s/(mass_m+mass_s))[mv]),
               'static_points_dominated_by_moving_sources_fraction': float((np.linalg.norm(fm, axis=1) > np.linalg.norm(fs, axis=1))[~mv].mean()) if (~mv).any() else None,
               'phi_norm_static': q(np.linalg.norm(Phi[~mv, k], axis=1)), 'phi_norm_moving': q(np.linalg.norm(Phi[mv, k], axis=1)),
               'cos_phi_vs_neg_grad_Eobs_static': q(cosine(Phi[~mv, k], -gobs[~mv, k])), 'cos_phi_vs_neg_grad_Eobs_moving': q(cosine(Phi[mv, k], -gobs[mv, k]))}
        if Xgt is not None and k > 0 and mv.any():
            disp = Xgt[k, mv]-Xgt[0, mv]; ok = np.linalg.norm(disp, axis=1) > 0
            row['cos_phi_vs_true_displacement_moving'] = q(cosine(Phi[mv, k][ok], disp[ok]))
            row['true_displacement_moving_median_m'] = float(np.median(np.linalg.norm(disp, axis=1)))
        frames.append(row)
    return {'run': str(run), 'N': N, 'frames_audited': K, 'gt_moving_points': int(mv.sum()), 'gt_usage': 'POST_HOC_ONLY (separate process)',
            'stein_diagnostics_from_run': st['gates'].get('STEIN_DIAGNOSTICS'), 'per_frame': frames,
            'reading': 'cross-support mass fractions > 0 mean the all-pairs kernel transfers motion guidance across GT supports (F6); '
                       'dE_time > 0 with dE_obs < 0 on positions-only proposals and not on lifted ones points to F2; negative cos(Phi, -grad E_obs) to F3'}


def to_md(rep):
    L = [f"# Stein-proposal rejection audit (post hoc)", '', f"Run `{rep['run']}`, N = {rep['N']}, GT-moving {rep['gt_moving_points']} ({rep['gt_usage']}).", '',
         '## Energy decomposition (from the run, no GT)', '', '| outer | <gradE_obs,P> | <gradE_time,P> | cos(Phi,-gradE_obs) | cos(P_lift,-gradE) | first backtrack dE_obs / dE_time (positions) | (lift) |', '|---|---|---|---|---|---|---|']
    for s in rep['stein_diagnostics_from_run'] or []:
        b = s['backtracks'][0] if s['backtracks'] else {'positions_only': {}, 'temporal_lift': {}}
        L.append(f"| {s['outer']} | {s['dEobs_dP']:.3e} | {s['dEtime_dP']:.3e} | {s['cos_phi_vs_neg_grad_Eobs']:.3f} | {s['cos_P_lift_vs_neg_grad_Etotal']:.3f} | "
                 f"{b['positions_only'].get('dE_obs', float('nan')):.3e} / {b['positions_only'].get('dE_time', float('nan')):.3e} | "
                 f"{b['temporal_lift'].get('dE_obs', float('nan')):.3e} / {b['temporal_lift'].get('dE_time', float('nan')):.3e} |")
    L += ['', '## Cross-support kernel coupling (GT post hoc)', '', '| frame | static mass from moving (median) | moving mass from static (median) | static dominated by moving | cos(Phi,-gradE_obs) static / moving | cos(Phi, true disp) moving |', '|---|---|---|---|---|---|']
    for r in rep['per_frame']:
        cm = r.get('cos_phi_vs_true_displacement_moving', {}).get('median', float('nan'))
        L.append(f"| {r['frame']} | {r['static_receivers_mass_fraction_from_moving'].get('median', float('nan')):.3f} | {r['moving_receivers_mass_fraction_from_static'].get('median', float('nan')):.3f} | "
                 f"{r['static_points_dominated_by_moving_sources_fraction']} | {r['cos_phi_vs_neg_grad_Eobs_static'].get('median', float('nan')):.3f} / "
                 f"{r['cos_phi_vs_neg_grad_Eobs_moving'].get('median', float('nan')):.3f} | {cm:.3f} |")
    return '\n'.join(L+['', rep['reading'], ''])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path); ap.add_argument('--membership', type=Path)
    ap.add_argument('--gt-trajectory', type=Path); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if (a.scene is None) == (a.membership is None): ap.error('give exactly one of --scene (Sapien GT) or --membership')
    if a.scene: moving, Xgt, ev = gt_from_scene(a.scene, a.run)
    else: moving, Xgt, ev = np.load(a.membership), (np.load(a.gt_trajectory) if a.gt_trajectory else None), {'source': str(a.membership)}
    rep = audit(a.run, moving, Xgt, a.device); rep['gt_evidence'] = ev
    out = a.run/'posthoc'; out.mkdir(exist_ok=True)
    (out/'stein_rejection_audit.json').write_text(json.dumps(rep, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    (out/'stein_rejection_audit.md').write_text(to_md(rep))
    print(to_md(rep)); return 0


if __name__ == '__main__':
    sys.exit(main())
