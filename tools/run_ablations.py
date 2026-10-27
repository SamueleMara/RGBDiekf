#!/usr/bin/env python3
"""Contract section 13 ablations on synthetic ray-cast scenes (ground truth used ONLY for evaluation here).

  stale_jacobian      Phase 1 with the renderer Jacobian of the first render in every inner system (fresh residual)
  single_camera       Phase 1 with one camera instead of all (test-only)
  covariance          dense 6N recursion vs structured exact / assumed-density / RigEuler marginals (from the dense test)
  occupancy_coherence coherence on the Phase-1 correction groups as nodes vs the pinned fixed-count 192-node graph
  independent_poses   Phase 2 with a negligible motion prior (per-frame independent poses) vs the shared rigid IEKF
None of these is a production setting. Output: reports/ablations.json.
"""
import json, sys, tempfile
from dataclasses import replace
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, source_parts, true_positions
from pointwise_rgbd.config import load_config
from pointwise_rgbd import image_phase1, image_coherence, rigid_image_iekf
from pointwise_rgbd.image_iekf import IEKFConfig
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy
from pointwise_rgbd.image_observation import ObservationPolicy
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.lie import log as se3log

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def phase1_errors(ds, part, arr):
    parts = source_parts(ds, part.source); out = {}
    for k in (len(ds.times)//2, len(ds.times)-1):
        e = np.linalg.norm(np.asarray(arr['xyz'][k])-true_positions(ds, part.xyz, parts, k, 0), axis=1)
        out[f'frame_{k}'] = {'moving_median_m': float(np.median(e[parts == 1])), 'static_median_m': float(np.median(e[parts == 0]))}
    return out


def pose_errors(ds, poses):
    e = [se3log(np.linalg.inv(ds.poses(k)[1]@np.linalg.inv(ds.poses(0)[1]))@poses[k]) for k in range(len(ds.times))]
    return {'max_translation_m': float(max(np.linalg.norm(x[:3]) for x in e)), 'max_rotation_deg': float(np.degrees(max(np.linalg.norm(x[3:]) for x in e)))}


def main():
    config, backend = load_config(), Backend(DEVICE)
    pol, op = RenderPolicy(depth_variance=config.depth_variance), ObservationPolicy()
    out = {}
    for motion in ('prismatic', 'revolute'):
        ds = two_plate_scene(motion, frames=12, cameras=4)
        res = {}
        runs = {}
        for name, iekf, cams in (('production', IEKFConfig(), None), ('stale_jacobian', IEKFConfig(stale_jacobian=True), None),
                                 ('single_camera', IEKFConfig(), [0])):
            d = ds if cams is None else two_plate_scene(motion, frames=12, cameras=4)
            if cams is not None: d.cameras = [d.cameras[c] for c in cams]
            part, arr, rho = image_phase1.run(d, config, backend, tempfile.mkdtemp(), 'first', iekf, pol, op)
            runs[name] = (part, arr, rho)
            res[f'phase1_{name}'] = phase1_errors(ds, part, arr)
        part, arr, rho = runs['production']
        parts = source_parts(ds, part.source)
        coh = image_coherence.run(part, arr, ds.times, tempfile.mkdtemp(), 0)
        iou = lambda s: float((s & (parts == 1)).sum()/max((s | (parts == 1)).sum(), 1))
        res['coherence_fixed_count_192'] = {'status': coh['status'], 'iou_with_true_moving': iou(coh['point_support'])}
        from types import SimpleNamespace
        occ = image_coherence.run(SimpleNamespace(xyz=part.xyz, count=part.count, membership=part.membership), arr, ds.times, tempfile.mkdtemp(), 0,
                                  count=part.count)   # node count = Phase-1 group count; nodes from the pinned builder on the same catalogue
        res['coherence_occupancy_group_count'] = {'status': occ['status'], 'nodes': part.count, 'iou_with_true_moving': iou(occ['point_support'])}
        if coh['status'] == 'RESOLVED':
            core = coh['point_support']
            rig = rigid_image_iekf.run(ds, config, backend, part, arr, rho, core, 0, tempfile.mkdtemp(), pol, op)
            ind = rigid_image_iekf.run(ds, config, backend, part, arr, rho, core, 0, tempfile.mkdtemp(), pol, op,
                                       rigid_image_iekf.RigidConfig(sigma_vdot=1e4, sigma_wdot=1e4))
            res['rigid_shared_iekf'] = pose_errors(ds, rig['poses']); res['rigid_independent_poses'] = pose_errors(ds, ind['poses'])
        out[motion] = res
        print(motion, json.dumps(res, indent=1), flush=True)
    dense = REPO/'reports/image_iekf_dense_covariance_comparison.json'
    if dense.exists():
        d = json.loads(dense.read_text())
        out['covariance'] = {f"{f['points']}pts_{f['groups']}groups_{f['direction']}": f['frames'][-1] for f in d['fixtures']}
    (REPO/'reports/ablations.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
