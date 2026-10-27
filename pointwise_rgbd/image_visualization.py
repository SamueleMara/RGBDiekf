"""Viewer export for the rendered-image pipeline: the ACTUAL image measurements and predictions.

Per direction, stage (phase1, rigid, joint_prismatic, joint_revolute), frame and camera, one
full-resolution mosaic of the saved crops (no resampling of values):
    row 1: observed RGB | rendered RGB | |RGB residual| (0..0.5) | domain/coverage map
    row 2: observed metric depth | rendered depth | depth residual (+-0.05 m) | contributor count
Coverage map: green = observed domain and covered, red = observed but unexplained,
orange = predicted outside the domain (not scored), black = neither.
Statistics in manifest.json are computed from the same arrays the PNG shows, so a
browser QA can check pixels against numbers. 3D: full point buffers (no cap).
"""
import json, shutil, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .group_visualization import serial
from .store import json_write

STAGES = {'phase1': 'phase1', 'rigid': 'rigid', 'joint_prismatic': 'joint/prismatic', 'joint_revolute': 'joint/revolute'}
DEPTH_RES_M = .05
SUPPORT_CODES = {'AMBIGUOUS': 0, 'MOVING_CORE': 1, 'MOVING_JOINT_COMPATIBLE': 2, 'STATIC_COMPATIBLE': 3}


def _gray(a, lo, hi, valid):
    g = np.clip((a-lo)/max(hi-lo, 1e-9), 0, 1)
    out = np.stack([g, g, g], -1)
    out[~valid] = [0, 0, .25]
    return (out*255).astype(np.uint8)


def _diverging(r, scale, valid):
    x = np.clip(r/scale, -1, 1)
    out = np.ones(r.shape+(3,))
    out[..., 0] = np.where(x < 0, 1+x, 1); out[..., 1] = 1-np.abs(x); out[..., 2] = np.where(x > 0, 1-x, 1)
    out[~valid] = 0
    return (out*255).astype(np.uint8)


def _heat(a, scale, valid):
    x = np.clip(a/scale, 0, 1)
    out = np.stack([x, x**2, .2*(1-x)], -1); out[~valid] = 0
    return (out*255).astype(np.uint8)


def panel(z, ci):
    dom, cov = z[f'c{ci}_domain'], z[f'c{ci}_covered']
    orgb, prgb = z[f'c{ci}_obs_rgb'], z[f'c{ci}_pred_rgb']
    od, pd, cnt = z[f'c{ci}_obs_depth'].astype(float), z[f'c{ci}_pred_depth'].astype(float), z[f'c{ci}_contributors']
    both = dom & cov
    rr = np.linalg.norm(orgb.astype(float)/255-prgb.astype(float)/255, axis=-1)
    dr = od-pd
    lo, hi = (float(od[dom].min()), float(od[dom].max())) if dom.any() else (0., 1.)
    covmap = np.zeros(dom.shape+(3,), np.uint8)
    covmap[both] = [40, 200, 70]; covmap[dom & ~cov] = [220, 40, 40]; covmap[~dom & cov] = [240, 160, 30]
    prgb_show = prgb.copy(); prgb_show[~cov] = [0, 0, 64]
    rows = [np.concatenate([orgb*dom[..., None], prgb_show, _heat(rr, .5, both), covmap], 1),
            np.concatenate([_gray(od, lo, hi, dom), _gray(pd, lo, hi, cov), _diverging(dr, DEPTH_RES_M, both), _heat(cnt.astype(float), max(1, cnt.max()), cnt > 0)], 1)]
    image = np.concatenate(rows, 0).astype(np.uint8)
    stats = {'box_y0_y1_x0_x1': z[f'c{ci}_box'].tolist(), 'tile_hw': list(dom.shape), 'domain_pixels': int(dom.sum()),
             'covered_domain_pixels': int(both.sum()), 'unexplained_domain_pixels': int((dom & ~cov).sum()),
             'predicted_outside_domain_pixels': int((~dom & cov).sum()),
             'rgb_rmse': float(np.sqrt((rr[both]**2).mean()/3)) if both.any() else None,
             'depth_rmse_m': float(np.sqrt((dr[both]**2).mean())) if both.any() else None,
             'depth_residual_p95_abs_m': float(np.percentile(np.abs(dr[both]), 95)) if both.any() else None,
             'observed_depth_range_m': [lo, hi], 'depth_residual_scale_m': DEPTH_RES_M, 'rgb_residual_scale': .5,
             'max_contributors': int(cnt.max())}
    return image, stats


def export(root, dataset, artifacts, pooled):
    root = Path(root); view = root/'viewer'; view.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name('assets')
    shutil.copyfile(assets/'three-0.160.0.min.js', view/'three.min.js'); shutil.copyfile(assets/'THREE_LICENSE.txt', view/'THREE_LICENSE.txt')
    shutil.copyfile(assets/'image_viewer.html', view/'index.html')
    K = len(dataset.times)
    manifest = {'measurement_model': 'rendered_rgbd_point_cloud_v2', 'timestamps_seconds': np.asarray(dataset.times).tolist(),
                'original_frame_ids': list(dataset.frame_ids), 'cameras': [dict(name=c.name, width=c.width, height=c.height, intrinsics=c.intrinsics,
                world_to_camera=c.world_to_camera) for c in dataset.cameras], 'world_frame': 'calibrated world (metadata/cameras.json)',
                'hidden_point_cap': None, 'panels_legend': __doc__, 'directions': {}, 'bidirectional': None if pooled is None else
                {k: v for k, v in pooled.items()}, 'export_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), assets/'image_viewer.html')}}
    for anchor, art in artifacts.items():
        d = {'anchor_frame': art['endpoint'], 'stages': {}, 'panels': {}}
        base = root/anchor
        for stage, sub in STAGES.items():
            folder = base/sub/'images'
            if not folder.exists(): continue
            d['stages'][stage] = []; d['panels'][stage] = {}
            for f in sorted(folder.glob('*.npz')):
                k = int(f.stem); z = np.load(f)
                out = view/'panels'/anchor/stage; out.mkdir(parents=True, exist_ok=True)
                d['panels'][stage][k] = {}
                for ci in range(len(dataset.cameras)):
                    if f'c{ci}_box' not in z: continue
                    image, stats = panel(z, ci)
                    Image.fromarray(image).save(out/f'{k:06d}_c{ci}.png', optimize=True)
                    d['panels'][stage][k][ci] = stats
                d['stages'][stage].append(k)
            reports = base/sub/'frames'
            if reports.exists():
                d.setdefault('reports', {})[stage] = {int(p.stem): {key: v for key, v in json.loads(p.read_text()).items() if key in
                    ('energy_initial', 'energy_final', 'image_energy', 'image_energy_final', 'accepted_steps', 'rejected_trials', 'stop', 'seconds', 'q', 'pose', 'kabsch_pose', 'pose_sigma', 'twist', 'directly_observed_points')}
                    for p in sorted(reports.glob('*.json'))}
        # 3D buffers: full catalog every frame, float32 for display only.
        pts = view/'points'/anchor; pts.mkdir(parents=True, exist_ok=True)
        part, arrays = art['partition'], art['arrays']
        n = len(part.xyz)
        for k in range(K):
            np.asarray(arrays['xyz'][k], np.float32).tofile(pts/f'xyz_{k:06d}.bin')
            np.asarray(arrays['point_observed'][k], np.uint8).tofile(pts/f'observed_{k:06d}.bin')
            np.linalg.norm(np.asarray(arrays['state'][k])[:, 3:], axis=1).astype(np.float32).tofile(pts/f'speed_{k:06d}.bin')
            np.sqrt(np.trace(np.asarray(arrays['covariance'][k])[:, :3, :3], axis1=1, axis2=2)).astype(np.float32).tofile(pts/f'sigma_{k:06d}.bin')
        np.round(np.asarray(part.rgb)*255).astype(np.uint8).tofile(pts/'rgb.bin')
        np.asarray(part.membership, np.int32).tofile(pts/'group.bin')
        coh = art.get('coherence') or {}
        if 'point_support' in coh: np.asarray(coh['point_support'], np.uint8).tofile(pts/'core.bin')
        if 'representative_membership' in coh: np.asarray(coh['representative_membership'], np.int32).tofile(pts/'node.bin')
        sup = art.get('support')
        if sup is not None: np.array([SUPPORT_CODES[str(x)] for x in sup['labels']], np.uint8).tofile(pts/'support.bin')
        rep = base/'coherence/representatives.npz'
        graph = None
        if rep.exists():
            z = np.load(rep)
            graph = {'trajectory': z['trajectory'].astype(np.float32).round(5), 'edges': z['edges'], 'labels': z['labels'],
                     'strain_C': z['C'].astype(np.float32).round(5)}
        rig = art.get('rigid') or {}
        d.update(points=n, representative_graph=graph, coherence={k: v for k, v in coh.items() if k in ('status', 'core_points', 'core_nodes', 'representative_nodes', 'node_frame_support', 'edges_supported', 'edges_total', 'adaptation')},
                 rigid={'status': rig.get('status'), 'poses': rig.get('poses'), 'kabsch': rig.get('kabsch'), 'twists': rig.get('twists'),
                        'diagnostics': rig.get('trajectory_diagnostics'), 'core_points': rig.get('core_points')},
                 joint=None if 'joint' not in art else {k: v for k, v in art['joint'].items() if k != 'pose_space'},
                 support=None if sup is None else {k: v for k, v in sup.items() if k in ('counts', 'alpha_fdr', 'map', 'uninfluenced_candidates', 'map_moving_but_ambiguous')},
                 buffers={'xyz': 'points/%s/xyz_{k}.bin float32 [N,3]' % anchor, 'observed': 'uint8 [N]', 'speed': 'float32 [N] |v| m/s', 'sigma': 'float32 [N] sqrt(tr P_xx) m', 'rgb': 'uint8 [N,3]', 'group': 'int32', 'core': 'uint8', 'node': 'int32', 'support': 'uint8 '+json.dumps(SUPPORT_CODES)})
        manifest['directions'][anchor] = d
    json_write(view/'manifest.json', serial(manifest))
    rows = [(a, st, k, ci, manifest['cameras'][ci]['name'], s) for a, d in manifest['directions'].items() for st, frames in d['panels'].items()
            for k, cams in frames.items() for ci, s in cams.items()]
    keys = ['domain_pixels', 'covered_domain_pixels', 'unexplained_domain_pixels', 'predicted_outside_domain_pixels', 'rgb_rmse', 'depth_rmse_m', 'depth_residual_p95_abs_m', 'max_contributors']
    with open(view/'image_residuals.csv', 'w') as f:
        f.write(','.join(['direction', 'stage', 'frame_index', 'original_frame', 'timestamp_s', 'camera_index', 'camera']+keys)+'\n')
        for a, st, k, ci, name, s in rows:
            f.write(','.join(map(str, [a, st, k, dataset.frame_ids[k], manifest['timestamps_seconds'][k], ci, name]+[s.get(x) for x in keys]))+'\n')
    return view
