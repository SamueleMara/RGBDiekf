#!/usr/bin/env python3
"""POST-HOC objective probe (evaluation only; the estimator never sees ground truth).

At frame k, with the production fixed prior (process6.predict of the saved posterior of frame k-1), evaluates the SAME
Phase-1 objective E = 1/2 e^T Lambda- e + image terms along s(alpha) = (1-alpha) s_production + alpha s_probe, where
s_probe moves the points labelled moving by the sparse GT part cloud to canonical + q_k * axis (velocity q_rate * axis),
everything else at the production posterior. Reports prior, RGB, depth and coverage energies and per-camera totals.
If E(s_probe) < E(s_production) the trust region stops in a worse basin; if E increases towards the probe, the objective
itself prefers the lagged state.
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from tools.audit_phase1_operators import blade_setup, saved_prior, observation, gt_labels
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.image_observation import levels, ObservationPolicy
from pointwise_rgbd.rgbd_point_renderer import render

ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path, required=True)
ap.add_argument('--frames', nargs='*', type=int, default=[1, 10]); ap.add_argument('--fps', type=float, default=59); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args(); dev = 'cuda'
B = blade_setup(a.run, a.scene, dev, a.fps); _, axis, qrel = gt_labels(B, a.scene); times = np.asarray(B['data'].times)
# Probe labelling: nearest sparse-GT part label for EVERY catalogue point (the 1 cm labelling covers only ~1/5 of the moving
# surface, and moving a fifth of the part splits its surface, which is not the true scene).
from scipy.spatial import cKDTree
from tools.evaluate_image_iekf_gt import ply
x0, l0 = ply(a.scene/'pointcloud/points3D_parts.ply'); xf, lf = ply(a.scene/'pointcloud/points3D_parts_final.ply')
mov = max(set(l0) & set(lf), key=lambda l: np.linalg.norm(x0[l0 == l].mean(0)-xf[lf == l].mean(0)))
lab = (l0[cKDTree(x0).query(B['part']['xyz'])[1]] == mov).astype(int)
out = []
for k in a.frames:
    sm, Pm = saved_prior(B, k); Lam = torch.linalg.inv(Pm); Lam = (Lam+Lam.transpose(1, 2))/2
    obs = observation(B, k, dev)
    lv = levels([B['data'].observation(k, c) for c in range(len(B['cams']))], B['config'], ObservationPolicy(), dev)
    s_prod = frame_step(sm, Pm.clone(), lv, B['rgb'], B['rho'], B['policy'], B['G'], IEKFConfig())[0]
    s_probe = s_prod.clone(); mv = torch.as_tensor(lab == 1, device=dev); ax = torch.as_tensor(axis, dtype=torch.float64, device=dev)
    rate = float((qrel[k]-qrel[k-1])/(times[k]-times[k-1]))
    s_probe[mv, :3] = B['t'](B['part']['xyz'])[mv]+float(qrel[k])*ax; s_probe[mv, 3:] = rate*ax
    rows = []
    for al in (0., .25, .5, .75, 1.):
        s = (1-al)*s_prod+al*s_probe; e = s-sm
        lin = render(s[:, :3], B['rgb'], B['rho'], B['cams'], B['policy']); t = obs.terms(lin)
        prior = .5*float(torch.einsum('ni,nij,nj->', e, Lam, e))
        prior_moving = .5*float(torch.einsum('ni,nij,nj->', e[mv], Lam[mv], e[mv]))
        c = t['cameras']
        rows.append({'alpha': al, 'total': prior+t['energy'], 'prior': prior, 'prior_moving_points': prior_moving, 'image': t['energy'],
                     'rgb': sum(x['energy_rgb'] for x in c), 'depth': sum(x['energy_depth'] for x in c), 'coverage': sum(x['energy_coverage'] for x in c),
                     'per_camera': [x['energy'] for x in c]})
        print(k, json.dumps({kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in rows[-1].items() if kk != 'per_camera'}), flush=True)
    out.append({'frame': k, 'true_slide_m': float(qrel[k]), 'moving_points_labelled': int(mv.sum()), 'path': rows})
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
