"""Point-only measured RGB-D debug pages for the four-pass run (contract section 9). Nothing here is rendered from the
cloud: every image is a recorded RGB / metric depth / mask frame (RGB and masks copied byte-for-byte from the source
files, sha256 recorded; depth colour-mapped from the metric depth_npy with a labelled colorbar), and every 3D element is
an existing canonical XYZ point, its own propagated state, an estimated rigid transform or a joint axis.

Output: <run>/debug/index.html + standalone pages, data/*.js (window.PD_DATA[...] = {...}: works from file:// without a
server), img/*.png (debug frames only). Display decimation (deterministic, evenly spaced ids, traced ids always kept)
bounds page size; every count shown is the exact count from the run, not from the displayed subset.
"""
import base64
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise, sample, REASONS
from utils.io import serial

ASSETS = Path(__file__).with_name('assets')
CAPS = {'light': {'points3d': 15000, 'markers': 3000}, 'full': {'points3d': 40000, 'markers': 12000}}


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype)).tobytes()).decode()


def write_js(path, key, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'window.PD_DATA=window.PD_DATA||{{}};window.PD_DATA[{json.dumps(key)}]=' + json.dumps(serial(obj), allow_nan=False, default=str) + ';\n')


def display_ids(n, cap, keep=()):
    ids = np.unique(np.linspace(0, n-1, min(cap, n)).round().astype(int)) if n else np.zeros(0, int)
    return np.union1d(ids, np.asarray([k for k in keep if 0 <= k < n], int))


def cameras_js(rig):
    return [{'name': n, 'T_wc': rig.T_wc[i].tolist(), 'K': rig.K[i].tolist(), 'width': rig.width[i], 'height': rig.height[i]} for i, n in enumerate(rig.names)]


def source_images(data, k, c, out_dir):
    """Copy the recorded RGB and mask PNGs of (k, c) (byte identical) and write a metric depth colour map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = getattr(data, 'base', data)
    root = Path(getattr(base, 'root', '.'))
    entries, masks = getattr(base, 'entries', None), getattr(base, 'masks', None)
    f = data.observation(k, c)
    rec = {'frame': k, 'camera': c}
    names = {'rgb': f'rgb_f{k:04d}_c{c}.png', 'mask': f'mask_f{k:04d}_c{c}.png', 'depth': f'depth_f{k:04d}_c{c}.png'}
    if entries:
        src = root/entries[k][c][0]; shutil.copyfile(src, out_dir/names['rgb'])
        rec['rgb_source'] = str(src); rec['rgb_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
    else:
        Image.fromarray(np.round(np.clip(f.rgb, 0, 1)*255).astype(np.uint8)).save(out_dir/names['rgb']); rec['rgb_source'] = 'in-memory dataset'
    if masks and masks[k][c]:
        src = root/masks[k][c]; shutil.copyfile(src, out_dir/names['mask'])
        rec['mask_source'] = str(src); rec['mask_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
    else:
        Image.fromarray(np.where(f.mask, 255, 0).astype(np.uint8)).save(out_dir/names['mask']); rec['mask_source'] = 'in-memory dataset'
    d = np.asarray(f.depth, float); v = np.isfinite(d) & (d > 0)
    lo, hi = (np.quantile(d[v], [.01, .99]) if v.any() else (0., 1.))
    t = np.clip((d-lo)/max(hi-lo, 1e-9), 0, 1)
    rgb = (np.stack([_turbo(t[..., None])[..., i] for i in range(3)], -1)*255).astype(np.uint8)
    rgb[~v] = 0
    Image.fromarray(rgb).save(out_dir/names['depth'])
    rec.update(depth_source=str(root/entries[k][c][1]) if entries else 'in-memory dataset', depth_colormap='turbo, black = no valid depth',
               depth_range_m=[float(lo), float(hi)], images={kk: f'img/{v_}' for kk, v_ in names.items()}, width=f.depth.shape[1], height=f.depth.shape[0])
    return rec


def _turbo(t):
    """Google Turbo colour map polynomial approximation, t in [0,1] -> rgb [0,1]."""
    r = .1357+t*(4.5974-t*(42.3277-t*(130.5887-t*(150.5666-t*58.1375))))
    g = .0914+t*(2.1856+t*(4.8052-t*(14.0195-t*(4.2109+t*2.7747))))
    b = .1067+t*(12.5925-t*(60.1097-t*(109.0745-t*(88.5066-t*26.8183))))
    return np.clip(np.concatenate([r, g, b], -1), 0, 1)


# ---------------------------------------------------------------------------------------------------------------------
def export_free(root, dbg, d, data, rig, noise, policy, debug_frames, caps, device):
    fd = root/f'free_{d}'
    if not (fd/'canonical.npz').exists(): return None
    can = np.load(fd/'canonical.npz'); man = json.loads((fd/'run_manifest.json').read_text())
    X = np.load(fd/'posterior_xyz.npy', mmap_mode='r'); Xp = np.load(fd/'predicted_xyz.npy', mmap_mode='r')
    V = np.load(fd/'velocity.npy', mmap_mode='r'); S = np.load(fd/'point_status.npy', mmap_mode='r'); NV = np.load(fd/'valid_cameras.npy', mmap_mode='r')
    CR = np.load(fd/'camera_reason.npy', mmap_mode='r'); IM = np.load(fd/'information_min_eig.npy', mmap_mode='r'); PV = np.load(fd/'position_variance_diag.npy', mmap_mode='r')
    K, N = X.shape[:2]
    traced = man['debug_point_ids']
    ids = display_ids(N, caps['points3d'], traced)
    k0 = int(can['canonical_frame'])
    traces = {}
    for k in debug_frames:
        p = fd/'numeric'/f'trace_frame_{k:04d}.json'
        if p.exists(): traces[k] = json.loads(p.read_text())['points']['final']
    counts = []
    for k in range(K):
        nj = fd/'numeric'/f'frame_{k:04d}.json'
        if nj.exists():
            n = json.loads(nj.read_text())
            counts.append({'frame': k, 'observed': n['sensor']['unique_points_observed'], 'prediction_only': n['sensor']['prediction_only_points'],
                           'rows': n['sensor']['rows_accepted_final'], 'not_converged': n['filter_update']['not_converged_points'],
                           'per_camera': [c['reason_counts'] for c in n['sensor']['per_camera']]})
        else:
            counts.append({'frame': k, 'anchor': True})
    meta = {'direction': d, 'pass': man['pass'], 'canonical_frame': k0, 'K': K, 'N': N, 'processing_order': man['processing_order'],
            'timestamps': man['timestamps_seconds'], 'original_frames': man['original_frame_indices'], 'display_ids': ids.tolist(),
            'canonical_xyz': b64(can['canonical_xyz'][ids], np.float32), 'canonical_rgb': b64(np.round(can['canonical_rgb'][ids]*255), np.uint8),
            'source': can['original_source_ids'][ids].tolist() if len(ids) <= 5000 else None,
            'traced_ids': traced, 'trajectories': {int(i): {'posterior': np.asarray(X[:, i]).tolist(), 'predicted': np.asarray(Xp[:, i]).tolist(),
                                                            'velocity': np.asarray(V[:, i]).tolist(), 'var_diag': np.asarray(PV[:, i]).tolist(),
                                                            'status': np.asarray(S[:, i]).tolist(), 'source': can['original_source_ids'][i].tolist()} for i in traced},
            'glyph_covariance': {int(k): {int(i): (np.asarray(v['P_plus'])[:3, :3].tolist()) for i, v in t.items()} for k, t in traces.items()},
            'cameras': cameras_js(rig), 'reasons': REASONS, 'status_codes': {0: 'ANCHOR_CANONICAL', 1: 'MEASURED', 2: 'PREDICTION_ONLY'},
            'counts': counts, 'debug_frames': debug_frames,
            'note': f'canonical cloud = measured frame-{k0} endpoint ({"P_0" if k0 == 0 else "P_T"}); arrays indexed by ORIGINAL chronological frame'}
    write_js(dbg/'data'/f'free_{d}_meta.js', f'free_{d}_meta', meta)
    for k in range(K):
        cr = np.asarray(CR[k][:, ids])
        anyvalid = (cr == 0).any(0)
        cnt = np.stack([(cr == code).sum(0) for code in range(1, 9)])                       # most frequent non-valid reason
        dom = np.where(anyvalid, 0, np.where(cnt.sum(0) > 0, cnt.argmax(0)+1, 9))
        write_js(dbg/'data'/f'free_{d}_f{k:04d}.js', f'free_{d}_f{k:04d}', {
            'frame': k, 'posterior': b64(X[k][ids], np.float32), 'predicted': b64(Xp[k][ids], np.float32), 'velocity': b64(V[k][ids], np.float32),
            'status': b64(S[k][ids], np.uint8), 'valid_cameras': b64(NV[k][ids], np.uint8), 'reason': b64(dom, np.uint8),
            'info_min_log10': b64(np.log10(np.maximum(np.asarray(IM[k][ids], float), 1e-12)), np.float32),
            'sigma_mm': b64(1e3*np.sqrt(np.asarray(PV[k][ids], float).sum(1)), np.float32)})
    # measured RGB-D overlays at the debug frames
    frames_rec = []
    dev = torch.device(device)
    for k in debug_frames:
        if k >= K: continue
        frame = MeasuredFrame(data, k, device)
        rec = {'frame': k, 'cameras': []}
        for c in range(rig.count):
            img = source_images(data, k, c, dbg/'img')
            cr = np.asarray(CR[k][c]); vis = (cr != 1) & (cr != 2) & (cr != 9)                # in front of the camera and inside the image
            if k == k0: vis = np.ones(N, bool)
            cand = np.flatnonzero(vis)
            pick = cand[np.unique(np.linspace(0, len(cand)-1, min(caps['markers'], len(cand))).round().astype(int))] if len(cand) else cand
            pick = np.union1d(pick, np.asarray([i for i in traced if vis[i]], int))
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)
            cam = torch.full((len(pick),), c, dtype=torch.long, device=dev)
            uv0, _, _, _ = rig.project_rows(t(can['canonical_xyz'][pick]), cam)
            uvp, zp, _, _ = rig.project_rows(t(np.asarray(Xp[k][pick], float)), cam)
            uvq, zq, _, _ = rig.project_rows(t(np.asarray(X[k][pick], float)), cam)
            s = sample(frame, cam, uvq, noise, policy, rig.K, torch.float64)
            ok = s.reason == 0
            rec['cameras'].append({**img, 'markers': len(pick), 'exact_reason_counts': {REASONS[int(r)]: int(n) for r, n in zip(*np.unique(cr, return_counts=True))},
                                   'ids': pick.tolist(), 'uv_initial': b64(uv0.cpu(), np.float32), 'uv_predicted': b64(uvp.cpu(), np.float32),
                                   'uv_posterior': b64(uvq.cpu(), np.float32), 'reason': b64(cr[pick], np.uint8),
                                   'sampled_rgb': b64(torch.where(ok[:, None], s.rgb, torch.full_like(s.rgb, -1)).cpu(), np.float32),
                                   'sampled_depth': b64(torch.where(ok, s.depth, torch.full_like(s.depth, float('nan'))).cpu(), np.float32),
                                   'z_posterior': b64(zq.cpu(), np.float32), 'canonical_rgb': b64(can['canonical_rgb'][pick], np.float32)})
        write_js(dbg/'data'/f'free_{d}_rgbd_f{k:04d}.js', f'free_{d}_rgbd_f{k:04d}', rec)
        frames_rec.append(k)
    return {'meta': f'free_{d}_meta', 'rgbd_frames': frames_rec}


def export_coherence(root, dbg, d):
    cd = root/f'coherence_{d}'
    if not (cd/'coherence_summary.json').exists(): return None
    summ = json.loads((cd/'coherence_summary.json').read_text())
    can = np.load(root/f'free_{d}'/'canonical.npz')
    out = {'direction': d, 'summary': summ, 'policies': {}}
    for pol in ('pinned_raw', 'measurement_supported'):
        z = np.load(cd/pol/'representatives.npz', allow_pickle=True); res = json.loads((cd/pol/'result.json').read_text())
        k0 = int(can['canonical_frame'])
        seed = set(summ['policies'][pol]['seed_core_nodes']); rec = set(summ['policies'][pol]['recruited_nodes'])
        role = ['SEED_CORE' if i in seed else 'RECRUITED' if i in rec else str(l) for i, l in enumerate(z['labels'])]
        ids = display_ids(len(z['membership']), 12000)
        out['policies'][pol] = {'node_xyz_canonical': z['trajectory'][k0].tolist(), 'trajectory': b64(z['trajectory'], np.float32),
                                'K': int(z['trajectory'].shape[0]), 'n': int(z['trajectory'].shape[1]), 'node_support': b64(z['node_support'], np.float32),
                                'edges': z['edges'].tolist(), 'edge_supported': z['edge_supported'].tolist(), 'C': z['C'].tolist(),
                                'log_strain': b64(z['log_strain'], np.float32), 'role': role, 'labels': z['labels'].astype(str).tolist(),
                                'point_ids': ids.tolist(), 'point_xyz': b64(can['canonical_xyz'][ids], np.float32),
                                'point_node': b64(z['membership'][ids], np.int32), 'node_ids': z['node_ids'].tolist(),
                                'candidates': res.get('candidates', [])[:60], 'selection': res.get('selection'), 'refinement': res.get('refinement'),
                                'status': res.get('status'), 'verdict': summ['policies'][pol]['verdict']}
    write_js(dbg/'data'/f'coherence_{d}.js', f'coherence_{d}', out)
    return f'coherence_{d}'


def export_rigid(root, dbg, d, data, rig, noise, policy, debug_frames, caps, device):
    rd = root/f'rigid_{d}'
    st = json.loads((rd/'status.json').read_text()) if (rd/'status.json').exists() else None
    if st is None: return None
    out = {'direction': d, 'status': st}
    if st['status'] == 'COMPLETE':
        can = np.load(root/f'free_{d}'/'canonical.npz')
        poses, tw, cov = np.load(rd/'poses.npy'), np.load(rd/'twists.npy'), np.load(rd/'covariance.npy')
        core, static = np.load(rd/'core_point_ids.npy'), np.load(rd/'static_occluder_point_ids.npy')
        ci, si = core[display_ids(len(core), caps['points3d'])], static[display_ids(len(static), caps['points3d'])]
        tl = list(__import__('csv').DictReader(open(rd/'numeric'/'timeline.csv')))
        out.update(poses=poses.tolist(), twists=tw.tolist(), pose_sigma=np.sqrt(np.maximum(np.diagonal(cov[:, :6, :6], axis1=1, axis2=2), 0)).tolist(),
                   twist_sigma=np.sqrt(np.maximum(np.diagonal(cov[:, 6:, 6:], axis1=1, axis2=2), 0)).tolist(),
                   core_xyz=b64(can['canonical_xyz'][ci], np.float32), core_rgb=b64(np.round(can['canonical_rgb'][ci]*255), np.uint8),
                   static_xyz=b64(can['canonical_xyz'][si], np.float32), core_count=int(len(core)), static_count=int(len(static)),
                   timeline=tl, cameras=cameras_js(rig), canonical_frame=int(can['canonical_frame']),
                   timestamps=np.asarray(data.times).tolist())
        # overlays of the rigidly moved core on measured RGB-D at the debug frames
        out['rgbd'] = {}
        dev = torch.device(device)
        for k in debug_frames:
            if k >= len(poses): continue
            frame = MeasuredFrame(data, k, device); T = torch.as_tensor(poses[k], device=dev)
            pts = torch.as_tensor(can['canonical_xyz'][core], device=dev); x = pts@T[:3, :3].T+T[:3, 3]
            cams = []
            for c in range(rig.count):
                cam = torch.full((len(x),), c, dtype=torch.long, device=dev)
                uv, z, _, _ = rig.project_rows(x, cam)
                s = sample(frame, cam, uv, noise, policy, rig.K, torch.float64)
                inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < rig.width[c]) & (uv[:, 1] >= 0) & (uv[:, 1] < rig.height[c])
                cand = torch.nonzero(inside).flatten().cpu().numpy()
                pick = cand[np.unique(np.linspace(0, len(cand)-1, min(caps['markers'], len(cand))).round().astype(int))] if len(cand) else cand
                pk = torch.as_tensor(pick, device=dev, dtype=torch.long)
                img = source_images(data, k, c, dbg/'img')
                ok = s.reason[pk] == 0
                cams.append({**img, 'uv': b64(uv[pk].cpu(), np.float32), 'reason': b64(s.reason[pk].cpu(), np.uint8),
                             'depth_residual_m': b64(torch.where(ok, z[pk]-s.depth[pk], torch.full_like(z[pk], float('nan'))).cpu(), np.float32),
                             'markers': int(len(pick)), 'projected_in_image': int(len(cand))})
            out['rgbd'][k] = cams
    write_js(dbg/'data'/f'rigid_{d}.js', f'rigid_{d}', out)
    return f'rigid_{d}'


def export_joint(root, dbg, d, caps):
    jp = root/'joint'/d/'result.json'
    if not jp.exists(): return None
    j = json.loads(jp.read_text()); out = {'direction': d, 'joint': j}
    ip = root/'joint'/d/'reduction_increments.json'
    if ip.exists(): out['increments'] = json.loads(ip.read_text())
    fd = root/f'free_{d}'
    if (fd/'canonical.npz').exists():
        can = np.load(fd/'canonical.npz'); ids = display_ids(len(can['canonical_xyz']), caps['points3d'])
        out.update(xyz=b64(can['canonical_xyz'][ids], np.float32), rgb=b64(np.round(can['canonical_rgb'][ids]*255), np.uint8),
                   canonical_frame=int(can['canonical_frame']))
        cp = root/f'rigid_{d}'/'core_point_ids.npy'
        if cp.exists(): out['core_mask'] = b64(np.isin(ids, np.load(cp)), np.uint8)
        sp = root/'joint'/d/'support'/'support.npz'
        if sp.exists():
            lab = np.load(sp, allow_pickle=True)['labels'].astype(str)
            names = ['MOVING_CORE', 'MOVING_JOINT_COMPATIBLE', 'STATIC_COMPATIBLE', 'AMBIGUOUS', 'OCCLUDED_OR_UNOBSERVED']
            out['support_codes'] = names; out['support'] = b64([names.index(l) if l in names else 3 for l in lab[ids]], np.uint8)
            out['support_counts'] = json.loads((root/'joint'/d/'support'/'result.json').read_text()).get('counts')
    rp = root/f'rigid_{d}'/'poses.npy'
    if rp.exists(): out['poses'] = np.load(rp).tolist()
    write_js(dbg/'data'/f'joint_{d}.js', f'joint_{d}', out)
    return f'joint_{d}'


def export_numeric(root, dbg, directions):
    import csv
    out = {}
    for d in directions:
        for kind in ('free', 'rigid'):
            p = root/f'{kind}_{d}'/'numeric'/'timeline.csv'
            if p.exists(): out[f'{kind}_{d}'] = list(csv.DictReader(open(p)))
        per = []
        fd = root/f'free_{d}'/'numeric'
        for f in sorted(fd.glob('frame_*.json')) if fd.exists() else []:
            n = json.loads(f.read_text())
            per.append({'frame': n['frame'], 'cameras': [{'camera': c['camera'], 'reason_counts': c['reason_counts'], 'rows': c['rows_accepted_final'],
                                                          'depth_innovation_m_quantiles_prior': c['depth_innovation_m_quantiles_prior'],
                                                          'information_trace_final': c['information_trace_final']} for c in n['sensor']['per_camera']],
                        'nis_quantiles': n['innovation']['nis_per_dof_quantiles_prior_observed'], 'cuda_ms': n['processing']['cuda_ms'],
                        'posterior_position_eig_m2_quantiles': n['filter_update']['posterior_position_eig_m2_quantiles']})
        out[f'free_{d}_frames'] = per
    write_js(dbg/'data'/'numeric.js', 'numeric', out)


def write_debug(root, data, config, policy, directions, debug_frames, level, device):
    root = Path(root); dbg = root/'debug'; dbg.mkdir(exist_ok=True)
    caps = CAPS[level]
    rig = CameraRig(data.cameras, device); noise = SensorNoise.from_config(config, policy, device)
    shutil.copyfile(ASSETS/'three-0.160.0.min.js', dbg/'three.min.js'); shutil.copyfile(ASSETS/'THREE_LICENSE.txt', dbg/'THREE_LICENSE.txt')
    shutil.copyfile(ASSETS/'projective_common.js', dbg/'projective_common.js')
    pages = {}
    with torch.no_grad():
        for d in directions:
            pages[f'free_{d}'] = export_free(root, dbg, d, data, rig, noise, policy, debug_frames, caps, device)
            pages[f'coherence_{d}'] = export_coherence(root, dbg, d)
            pages[f'rigid_{d}'] = export_rigid(root, dbg, d, data, rig, noise, policy, debug_frames, caps, device)
            pages[f'joint_{d}'] = export_joint(root, dbg, d, caps)
        export_numeric(root, dbg, directions)
    bid = root/'bidirectional'/'comparison.json'
    write_js(dbg/'data'/'bidirectional.js', 'bidirectional', {'comparison': json.loads(bid.read_text()) if bid.exists() else None, 'directions': directions})
    written = []
    for name, tpl in (('free_3d', 'projective_free_3d.html'), ('free_rgbd', 'projective_free_rgbd.html'), ('coherence', 'projective_coherence.html'),
                      ('rigid_3d', 'projective_rigid_3d.html'), ('joint', 'projective_joint.html')):
        for d in directions:
            html = (ASSETS/tpl).read_text().replace('__DIRECTION__', d)
            target = {'free_3d': f'free_{d}_3d.html', 'free_rgbd': f'free_{d}_rgbd.html', 'coherence': f'coherence_{d}.html',
                      'rigid_3d': f'rigid_{d}_3d.html', 'joint': f'joint_{d}.html'}[name]
            (dbg/target).write_text(html); written.append(target)
    for tpl, target in (('projective_rigid_rgbd.html', 'rigid_rgbd.html'), ('projective_bidirectional.html', 'bidirectional.html'),
                        ('projective_numeric.html', 'numerical_diagnostics.html'), ('projective_index.html', 'index.html')):
        (dbg/target).write_text((ASSETS/tpl).read_text().replace('__DIRECTIONS__', json.dumps(directions))); written.append(target)
    res = json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
    write_js(dbg/'data'/'index.js', 'index', {'pages': written, 'directions': directions, 'debug_frames': debug_frames, 'result': res,
                                              'run_manifest': json.loads((root/'run_manifest.json').read_text()) if (root/'run_manifest.json').exists() else None})
    return {'index': str(dbg/'index.html'), 'pages': written}
