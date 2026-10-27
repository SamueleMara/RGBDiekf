#!/usr/bin/env python3
"""Stage I synthetic runs: the production stage sequence on analytic ray-cast scenes (tests/image_fixtures.py).

Scenes: a revolute hinge (distinct from the prismatic blade), a static control and a prismatic slide, each forward
and backward, with bidirectional diagnostics, global-label support and the viewer export. Ground truth (exact part
labels and part poses of the ray-cast scene) is read ONLY after inference, by evaluate() below.
"""
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, source_parts
from pointwise_rgbd.config import load_config
from pointwise_rgbd import image_phase1, image_coherence, rigid_image_iekf, joint_image, image_bidirectional, image_support
from pointwise_rgbd.image_iekf import IEKFConfig
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy
from pointwise_rgbd.image_observation import ObservationPolicy
from pointwise_rgbd.group_visualization import serial
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.lie import log as se3log

SCENES = {'revolute': dict(motion='revolute', rate=.06), 'static': dict(motion='static'), 'prismatic': dict(motion='prismatic', speed=.012)}


def pipeline(ds, root, backend, config, anchors=('first', 'last')):
    pol, op = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance), ObservationPolicy()
    arts, res = {}, {}
    for anchor in anchors:
        d = root/anchor; k0 = 0 if anchor == 'first' else len(ds.times)-1
        part, arr, rho = image_phase1.run(ds, config, backend, d/'phase1', anchor, IEKFConfig(), pol, op)
        a = arts[anchor] = {'partition': part, 'arrays': arr, 'rho': rho, 'endpoint': k0}; r = res[anchor] = {'phase1': 'COMPLETE'}
        coh = a['coherence'] = image_coherence.run(part, arr, ds.times, d/'coherence', k0); r['coherence'] = coh['status']
        if coh['status'] != 'RESOLVED': continue
        rig = a['rigid'] = rigid_image_iekf.run(ds, config, backend, part, arr, rho, coh['point_support'], k0, d/'rigid', pol, op); r['rigid'] = rig['status']
        jo = a['joint'] = joint_image.run(ds, config, backend, part, rho, rig, k0, d/'joint', pol, op); r['joint'] = jo['status']
        if jo.get('selected') is not None:
            a['support'] = image_support.run(ds, config, backend, part, rho, coh['point_support'], jo['selected'], k0, d/'support', pol, op)
            r['support'] = a['support']['counts']
    if all('joint' in arts.get(x, {}) for x in ('first', 'last')):
        res['bidirectional'] = image_bidirectional.run(arts['first']['joint'], arts['last']['joint'], ds, root/'bidirectional',
                                                       float(np.median(np.concatenate([arts[x]['rho'] for x in ('first', 'last')]))))['status']
    return arts, res


def evaluate(ds, arts, motion):
    """Post-hoc: core, rigid, joint and support against the analytic ground truth."""
    out = {}
    for anchor, a in arts.items():
        parts = source_parts(ds, a['partition'].source); k0 = a['endpoint']; r = out[anchor] = {}
        if 'coherence' in a:
            sup = np.asarray(a['coherence']['point_support'], bool)
            r['core'] = {'points': int(sup.sum()), 'precision': float((parts[sup] == 1).mean()) if sup.any() else None, 'recall': float(sup[parts == 1].mean())}
        if 'rigid' in a and motion != 'static':
            P = np.asarray(a['rigid']['poses'])
            e = [se3log(np.linalg.inv(ds.poses(k)[1]@np.linalg.inv(ds.poses(k0)[1]))@P[k]) for k in range(len(ds.times))]
            r['rigid'] = {'max_translation_m': float(max(np.linalg.norm(x[:3]) for x in e)), 'max_rotation_deg': float(np.degrees(max(np.linalg.norm(x[3:]) for x in e)))}
        if 'joint' in a:
            j = a['joint']; r['joint'] = {'status': j['status'], 'reason': j['reason']}
            truth = {'prismatic': ('PRISMATIC', np.array([1., .3, 0.])/np.linalg.norm([1., .3, 0.])), 'revolute': ('REVOLUTE', np.array([0., 1., 0.]))}.get(motion)
            if truth:
                c = j['candidates'][truth[0]]
                r['joint']['axis_error_deg'] = float(np.degrees(np.arccos(min(1., abs(np.asarray(c['axis'])@truth[1])))))
                if truth[0] == 'REVOLUTE':
                    r['joint']['axis_line_distance_m'] = float(np.linalg.norm(np.cross(np.asarray(c['axis_point'])-np.array([-.2, 0, .06]), np.asarray(c['axis']))))
        if 'support' in a:
            lab = a['support']['labels']
            r['support'] = {name: {'count': int((lab == name).sum()), 'true_moving_fraction': float((parts[lab == name] == 1).mean()) if (lab == name).any() else None}
                            for name in ('MOVING_CORE', 'MOVING_JOINT_COMPATIBLE', 'STATIC_COMPATIBLE', 'AMBIGUOUS')}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--scenes', nargs='*', default=list(SCENES))
    ap.add_argument('--frames', type=int, default=20); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()): ap.error('choose a fresh output directory')
    config, backend = load_config(), Backend(a.device); summary = {}
    for name in a.scenes:
        start = time.monotonic()
        ds = two_plate_scene(frames=a.frames, cameras=4, **SCENES[name])
        root = a.output/name; root.mkdir(parents=True, exist_ok=True)
        arts, res = pipeline(ds, root, backend, config)
        ev = evaluate(ds, arts, SCENES[name]['motion'])
        json_write(root/'result.json', serial({'scene': name, 'frames': a.frames, 'directions': res, 'runtime_seconds': time.monotonic()-start}))
        json_write(root/'gt_evaluation.json', serial(ev))
        try:
            from pointwise_rgbd.image_visualization import export
            export(root, ds, arts, None)
        except Exception as error:                        # the viewer is optional for synthetic scenes
            json_write(root/'viewer_failure.json', {'error': repr(error)})
        summary[name] = {'directions': res, 'evaluation': ev}
        print(name, json.dumps(serial(summary[name]), indent=1), flush=True)
    json_write(a.output/'summary.json', serial(summary))


if __name__ == '__main__':
    main()
