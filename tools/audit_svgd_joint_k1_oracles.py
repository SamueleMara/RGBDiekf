#!/usr/bin/env python3
"""POST-HOC, separate-process GT evaluation of a joint K = 1 run (spec sections 1.7, 9 Gate 3, 10). Writes RUN/posthoc/k1_oracle[_N]/.

    python -B tools/audit_svgd_joint_k1_oracles.py --run RUN --scene SCENE      # Sapien (frame_values.csv + joint)
    python -B tools/audit_svgd_joint_k1_oracles.py --run RUN --synthetic        # tools/run_svgd_joint_k1_synthetic.py run

The posterior is rebuilt by estimator.svgd_k1_run.build_model from the run's manifest (config SHA-256 and source hashes checked) and F is
evaluated at: identity, the saved SVGD particles, the saved MAP pose, and the GT pose sequence (twists nu_t = Log(T_{t-1}^-1 T_t)/dt from the
GT rigid transforms of the moving members; pi maximized on the SAME F by golden section). Reports seed precision/recall vs GT membership,
gamma by GT class, pose errors, and one of PASS_PHYSICAL_GATE / MODEL_MISMATCH / OPTIMIZATION_FAILURE / UNRESOLVED_IDENTIFIABILITY /
FALSE_MOVING_SEEDS / NO_GT. GT never reaches the run directory's inference artifacts (they are only read).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def kabsch(A, B):
    """R, t with B ~ R A + t (exact for noise-free GT member positions)."""
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A-ca).T@(B-cb)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T@U.T))])
    R = Vt.T@D@U.T; return R, cb-R@ca


def gt_for(run, scene=None, synthetic=None, data=None, xyz0=None):
    if synthetic is not None:
        mv = np.array([data.observation(0, c).gt_moving.ravel() for c in range(len(data.cameras))], dtype=object)
        from scene.pointcloud_sequence import build_masked_cloud
        from scene.config import load_config
        cl = build_masked_cloud(data, 0, load_config()); mv = np.array([data.observation(0, c).gt_moving.ravel()[p] for _, c, p in cl.source])
        poses = [np.r_[np.c_[data.pose(k)[0], data.pose(k)[1]], [[0, 0, 0, 1]]] for k in range(len(data.times))]
        return mv, poses, {'source': 'synthetic fixture pose'}
    from tools.audit_temporal_dynamic_correspondence import ground_truth
    X, mv, dist, ev = ground_truth(Path(scene), data, xyz0)
    poses = [np.eye(4)]
    for k in range(1, len(X)):
        if mv.any(): R, t = kabsch(xyz0[mv], X[k, mv])
        else: R, t = np.eye(3), np.zeros(3)
        poses.append(np.r_[np.c_[R, t], [[0, 0, 0, 1]]])
    return mv, poses, ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path); ap.add_argument('--synthetic', action='store_true'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from estimator.svgd_k1_run import build_model, SOURCES
    from estimator.svgd_k1_motion import se3_log, compose
    man = json.loads((a.run/'run_manifest.json').read_text()); A = man['arguments']; sv = a.run/'svgd_joint_k1'
    out = a.run/'posthoc'/'k1_oracle'; n = 1
    while out.exists(): out = a.run/'posthoc'/f'k1_oracle_{n}'; n += 1
    out.mkdir(parents=True)
    cfg_path = Path(A['config']); ok_cfg = hashlib.sha256(cfg_path.read_bytes()).hexdigest() == man['config_sha256']
    changed = [s for s, h in man['source_sha256'].items() if (REPO/s).is_file() and hashlib.sha256((REPO/s).read_bytes()).hexdigest() != h]
    cfg = json.loads(cfg_path.read_text()); dev = torch.device(a.device)
    if a.synthetic:
        from tests.svgd_fixtures import SyntheticDataset
        data = SyntheticDataset(**json.loads((a.run/'synthetic_fixture.json').read_text()))
    else:
        from scene.dataset import Dataset
        fl = None if A.get('frame_limit') in (None, 'None') else int(A['frame_limit'])
        data = Dataset(a.scene, fl, None)
    args = SimpleNamespace(noise_config=None if A.get('noise_config') in (None, 'None') else A['noise_config'], sensor_noise=A.get('sensor_noise', 'measured'))
    model = build_model(cfg, data, args, dev, verbose=False); post, prior, seq = model['post'], model['prior'], model['seq']
    status = json.loads((sv/'status.json').read_text())
    try: mv, poses, ev = gt_for(a.run, a.scene, a.synthetic or None, data, seq.reference.xyz)
    except Exception as e:
        rep = {'status': 'NO_GT', 'reason': f'{type(e).__name__}: {e}'}; (out/'oracle.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep)); return 0
    dt = torch.float64; t = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dt, device=dev)
    P = [t(p) for p in poses]; dts = np.diff(np.asarray(seq.times, float))
    nu_gt = torch.stack([se3_log(torch.linalg.inv(P[k-1])@P[k])/float(dts[k-1]) for k in range(1, len(P))])
    from estimator.svgd_k1_joint_posterior import log_prior_h
    def best_h(nu):
        """ONE posterior evaluation at the pose; at a fixed pose F(h) = sum_i logaddexp(log(1-pi) + lS_i, log pi + lM_i) + log p(nu) + log p(h)
        depends on h only through the cached per-point evidences, so pi is maximized exactly on the same F without re-evaluating frames."""
        r0 = post.evaluate(nu, torch.tensor(0., dtype=dt, device=dev), grad=False); lS, lM = post.lS, r0['logLM']
        def h_terms(h):                                          # data term + log p(h): the only h-dependent part of F
            h = torch.tensor(h, dtype=dt, device=dev)
            return float(torch.logaddexp(torch.nn.functional.logsigmoid(-h)+lS, torch.nn.functional.logsigmoid(h)+lM).sum()+log_prior_h(h, *post.ab))
        rest = r0['F']-h_terms(0.); Fh = lambda h: rest+h_terms(h)
        lo, hi = -15., 15.; g = (math.sqrt(5)-1)/2
        for _ in range(60):
            m1, m2 = hi-g*(hi-lo), lo+g*(hi-lo); lo, hi = (m1, hi) if Fh(m1) < Fh(m2) else (lo, m2)
        h = (lo+hi)/2; B = lM-lS
        return h, {'F': Fh(h), 'B': B, 'gamma': torch.sigmoid(h+B), 'logLM': lM}
    h_gt, r_gt = best_h(nu_gt); h_id, r_id = best_h(torch.zeros_like(nu_gt))
    pi_check = abs(post.evaluate(nu_gt, torch.tensor(h_gt, dtype=dt, device=dev), grad=False)['F']-r_gt['F'])   # exact-pi shortcut vs full F
    seeds = np.load(sv/'moving_seed_indices.npy'); gam = np.load(sv/'gamma_provisional.npy'); F_svgd = float(np.load(sv/'particle_F.npy').max())
    F_map = status.get('F_map'); best = max(F_svgd, F_map if F_map is not None else -np.inf)
    map_pose = np.load(sv/'map_pose.npy') if (sv/'map_pose.npy').is_file() else None
    pose_err = None
    if map_pose is not None:
        Pe = np.linalg.inv(poses[-1])@map_pose[-1]; ang = math.degrees(math.acos(max(-1, min(1, (np.trace(Pe[:3, :3])-1)/2))))
        pose_err = {'final_frame_rotation_error_deg': ang, 'final_frame_translation_error_m': float(np.linalg.norm(Pe[:3, 3])),
                    'gt_final_rotation_deg': math.degrees(math.acos(max(-1, min(1, (np.trace(poses[-1][:3, :3])-1)/2)))), 'gt_final_translation_m': float(np.linalg.norm(poses[-1][:3, 3]))}
    prec = float(mv[seeds].mean()) if len(seeds) else None; rec = float(mv[seeds].sum()/max(mv.sum(), 1)) if len(seeds) else 0.
    alpha = status.get('seed_alpha', .05); tol = 1e-6*abs(r_id['F'])
    gt_moves = bool(mv.any()) and any(not np.allclose(p, np.eye(4), atol=1e-12) for p in poses)
    if not gt_moves:                                             # static control: the only failure is a hallucinated confident moving subset
        cls = 'FALSE_MOVING_SEEDS' if len(seeds) else 'PASS_PHYSICAL_GATE (static control: no hallucinated seeds)'
    elif r_gt['F'] <= r_id['F']+tol: cls = 'MODEL_MISMATCH'
    elif best < r_gt['F']-tol: cls = 'OPTIMIZATION_FAILURE' if not len(seeds) or (prec is not None and prec < 1-alpha) else 'OPTIMIZATION_FAILURE (seeds found, GT pose not reached)'
    elif len(seeds) and prec >= 1-alpha: cls = 'PASS_PHYSICAL_GATE'
    elif not len(seeds): cls = 'UNRESOLVED_IDENTIFIABILITY'
    else: cls = 'FALSE_MOVING_SEEDS'
    rep = {'status': cls, 'gt_motion_present': gt_moves, 'run': str(a.run), 'config_sha_ok': ok_cfg, 'sources_changed_since_run': changed, 'gt_evidence': ev, 'gt_moving': int(mv.sum()), 'N': int(len(mv)),
           'F': {'identity_pi_opt': r_id['F'], 'gt_pi_opt': r_gt['F'], 'svgd_best': F_svgd, 'map': F_map, 'gt_minus_identity': r_gt['F']-r_id['F'], 'best_found_minus_gt': best-r_gt['F']},
           'pi_opt': {'gt': 1/(1+math.exp(-h_gt)), 'identity': 1/(1+math.exp(-h_id)), 'gt_fraction': float(mv.mean()), 'shortcut_vs_full_F_abs_nats': pi_check},
           'gamma_at_gt_pose': {'moving_median': float(np.median(r_gt['gamma'].cpu().numpy()[mv])) if mv.any() else None, 'static_median': float(np.median(r_gt['gamma'].cpu().numpy()[~mv]))},
           'gamma_provisional': {'moving_median': float(np.median(gam[mv])) if mv.any() else None, 'static_median': float(np.median(gam[~mv])), 'static_p99': float(np.quantile(gam[~mv], .99))},
           'B_at_gt_pose': {'moving_median': float(np.median(r_gt['B'].cpu().numpy()[mv])) if mv.any() else None, 'static_median': float(np.median(r_gt['B'].cpu().numpy()[~mv]))},
           'seeds': {'count': int(len(seeds)), 'precision': prec, 'recall': rec, 'alpha': alpha}, 'pose_error_map': pose_err,
           'map_seeds_control': (lambda ms: {'count': int(len(ms)), 'precision': float(mv[ms].mean()) if len(ms) else None, 'recall': float(mv[ms].sum()/max(mv.sum(), 1))})(
               np.load(sv/'map_moving_seed_indices.npy')) if (sv/'map_moving_seed_indices.npy').is_file() else None,
           'semantics': 'GT used post hoc only; F is a conditional Gibbs log-posterior; gamma UNCALIBRATED'}
    (out/'oracle.json').write_text(json.dumps(rep, indent=1, default=float)); print(json.dumps({k: rep[k] for k in ('status', 'F', 'seeds', 'pose_error_map')}, default=float, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
