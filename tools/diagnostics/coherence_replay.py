#!/usr/bin/env python3
"""Offline coherence replay (no Phase-1 rerun): per-node GT composition of the selected seed, complement and refinement,
for one or more saved runs. GT part labels are used for EVALUATION ONLY."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from tools.evaluate_image_iekf_gt import ply
scene = Path(sys.argv[1])
x0, l0 = ply(scene/'pointcloud/points3D_parts.ply')
xf, lf = ply(scene/'pointcloud/points3D_parts_final.ply')
mov = max(set(l0) & set(lf), key=lambda l: np.linalg.norm(x0[l0 == l].mean(0)-xf[lf == l].mean(0)))
for run in sys.argv[2:]:
    d = Path(run)/'first'
    part = np.load(d/'phase1/partition.npz'); xyz = part['xyz']
    dist, nn = cKDTree(x0).query(xyz); lab = np.where(dist < .01, l0[nn], -1)
    z = np.load(d/'coherence/representatives.npz', allow_pickle=True); c = json.loads((d/'coherence/result.json').read_text())
    sid, ids = z['membership'], z['node_ids']
    frac = np.array([((lab[sid == s] == mov).sum()/max((lab[sid == s] >= 0).sum(), 1)) for s in ids])   # moving fraction per node
    labelled = np.array([(lab[sid == s] >= 0).sum() for s in ids])
    print('==', run); print('points', len(xyz), 'nodes', len(ids), 'GT-moving nodes (frac>.5)', int((frac > .5).sum()), 'labelled nodes', int((labelled > 0).sum()))
    if 'seed_core_nodes' in c:
        seed = np.array(c['seed_core_nodes'], int); rest = np.setdiff1d(np.arange(len(ids)), seed)
        print('seed', len(seed), 'GT-moving among seed', int((frac[seed] > .5).sum()), '| complement', len(rest), 'GT-moving among complement', int((frac[rest] > .5).sum()))
        rec = np.array(c['refinement']['recruited_nodes'], int) if c.get('refinement') else np.zeros(0, int)
        print('recruited', len(rec), 'GT-moving among recruited', int((frac[rec] > .5).sum()) if len(rec) else 0)
    lab_nodes = np.array(z['labels'])
    print('MOVING_RIGID nodes', int((lab_nodes == 'MOVING_RIGID').sum()), 'of which GT-moving', int(((lab_nodes == 'MOVING_RIGID') & (frac > .5)).sum()))
