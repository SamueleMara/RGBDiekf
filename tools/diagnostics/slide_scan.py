#!/usr/bin/env python3
"""POST-HOC: image-only Phase-1 energy at frame k when the GT-labelled moving set (nearest sparse-GT label) is slid
rigidly by d along the GT axis and everything else stays canonical. Locates the image-preferred slide per frame."""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from scipy.spatial import cKDTree
from tools.audit_phase1_operators import blade_setup, observation, gt_labels
from tools.evaluate_image_iekf_gt import ply
from pointwise_rgbd.rgbd_point_renderer import render
run, scene = Path(sys.argv[1]), Path(sys.argv[2]); frames = [int(x) for x in sys.argv[3].split(',')]
B = blade_setup(run, scene, 'cuda', 59); _, axis, qrel = gt_labels(B, scene)
x0, l0 = ply(scene/'pointcloud/points3D_parts.ply'); xf, lf = ply(scene/'pointcloud/points3D_parts_final.ply')
mov = max(set(l0) & set(lf), key=lambda l: np.linalg.norm(x0[l0 == l].mean(0)-xf[lf == l].mean(0)))
mv = torch.as_tensor(l0[cKDTree(x0).query(B['part']['xyz'])[1]] == mov, device='cuda')
can = B['t'](B['part']['xyz']); ax = torch.as_tensor(axis, dtype=torch.float64, device='cuda')
out = {}
for k in frames:
    obs = observation(B, k, 'cuda'); q = float(qrel[k]); rows = []
    for d in np.linspace(0, 2*max(q, .01), 21):
        x = can.clone(); x[mv] += float(d)*ax
        t = obs.terms(render(x, B['rgb'], B['rho'], B['cams'], B['policy']))
        rows.append((float(d), t['energy']))
    best = min(rows, key=lambda r: r[1])
    out[k] = {'gt_slide_m': q, 'image_preferred_slide_m': best[0], 'scan': rows}
    print(k, 'gt slide %.4f  image-preferred slide %.4f' % (q, best[0]), ' '.join('%.3f:%.0f' % r for r in rows[::2]), flush=True)
(REPO/'reports/diagnostics/phase1_slide_scan.json').write_text(json.dumps(out, indent=1))
