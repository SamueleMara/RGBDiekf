"""RigEuler3DGS debug pages for the rendered-image RGB-D pipeline.

The page writers are the pinned RigEuler3DGS 481b2cf2 modules (rigeuler_ref/*_debug.py),
called exactly as tools/run_motion_coherence_support.py and tools/run_rigid_se3_pipeline.py
call them, but fed with THIS pipeline's data:

    <run>/<dir>/phase1_direction/frame_order.html        frame order and endpoint checks
    <run>/<dir>/debug/motion_coherence_support_3d.html   representative graph, classes, frame slider
    <run>/<dir>/debug/finite_strain_heatmap.html         z_ij(t), edges sorted by C
    <run>/<dir>/debug/differential_strain_heatmap.html   eps_ij(t)
    <run>/<dir>/debug/representative_edges.html          the four strain regimes
    <run>/<dir>/debug/coherence_component_tree.html      every candidate, held-out evidence
    <run>/<dir>/debug/static_dynamic_comparison_3d.html  GT / Otsu baseline / coherence, slider
    <run>/<dir>/rigid/html/rigid_phase2_debug.html       Phase 1, Kabsch, rigid IEKF layers, slider
    <run>/<dir>/rigid/html/rigid_to_joint_debug.html     rigid trajectory and both joint models
    <run>/<dir>/rigid/html/rigid_timelines.html          pose, twist, residual, posterior timelines
    <run>/<dir>/rigid/html/rigid_renders.html            one camera, every frame: observed / Phase 1 /
                                                         Kabsch / rigid IEKF renders (RGB, depth, residual)

Conventions follow RigEuler: arrays in PROCESSING order (the backward slider starts at its
anchor, the final frame) with the original frame ids as labels. The coherence diagnostics
(z, eps, C, J, components) come from the pinned run_motion_coherence on the SAME
image-supported, visibility x precision weighted node trajectories the extractor used; node
classes and the candidate tree are this pipeline's extractor output. GT (sparse labelled
cloud, joint metadata) is read only here, after inference, exactly as RigEuler's runners do
after writing their prediction. three.js is copied next to the pages (no network needed).
"""
from pathlib import Path
import json, shutil
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from rigeuler_ref import motion_coherence_debug as mcd, rigid_debug as rd, sequence_direction_debug as sdd
from rigeuler_ref.motion_coherence_support import run_motion_coherence, supernode_trajectories
from rigeuler_ref.moving_support import select_moving_support
from rigeuler_ref.rigid_trajectory_reduction import reduce_trajectory
from rigeuler_ref.gt_evaluation import load_joint_gt, evaluate
from rigeuler_ref.joint_manifold import se3_log
from scene.endpoint import Partition
from estimator.point_renderer import CameraTensors, RenderPolicy, render
from utils.io import json_write

THREE = Path(__file__).with_name('assets')/'three-0.160.0.min.js'
CDN = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.min.js'


def _local_three(page):
    """Point a written page at a local three.js copy so it works offline."""
    page = Path(page)
    s = page.read_text()
    if CDN in s:
        shutil.copyfile(THREE, page.parent/'three.min.js')
        page.write_text(s.replace(CDN, 'three.min.js'))
    return page


class DebugCamera:
    """RigEuler CameraSpec view (viewmatrix = world_to_camera^T, tan half-FOV) for frustums."""
    def __init__(self, cam):
        self.viewmatrix = torch.as_tensor(np.asarray(cam.world_to_camera).T)
        fx, fy = cam.intrinsics[:2]
        self.tanfovx, self.tanfovy = cam.width/(2*fx), cam.height/(2*fy)


def _ply(path):
    lines = Path(path).read_text().split('end_header\n')[1].split('\n')
    a = np.array([list(map(float, l.split())) for l in lines if l.strip()])
    return a[:, :3], a[:, 6].astype(int)


def gt_point_labels(scene, xyz, anchor):
    """Evaluation only: sparse labelled cloud of the direction's endpoint, nearest point within 1 cm.
    Returns 1 moving part, 0 static, -1 unknown (as RigEuler's sparse-NN labels)."""
    scene = Path(scene)
    pi, li = _ply(scene/'pointcloud/points3D_parts.ply'); pf, lf = _ply(scene/'pointcloud/points3D_parts_final.ply')
    moving = max(set(li) & set(lf), key=lambda l: np.linalg.norm(pi[li == l].mean(0)-pf[lf == l].mean(0)))
    x0, l0 = (pi, li) if anchor == 'first' else (pf, lf)
    d, nn = cKDTree(x0).query(xyz)
    return np.where(d < .01, (l0[nn] == moving).astype(int), -1)


def frame_record(data, order, anchor):
    """RigEuler frame_ids record (utils/sequence_direction): processing order, original frames,
    |dt| actually used (the backward transition is a reverse-time flag on a positive interval),
    and the per-camera RGB / depth files and valid-depth counts actually read."""
    frames = []
    for i, k in enumerate(order):
        dt = None if i == 0 else abs(float(data.times[k]-data.times[order[i-1]]))
        cams = []
        for ci in range(len(data.cameras)):
            rgb_path, depth_path = data.base.entries[k][ci]
            cams.append({'camera': data.cameras[ci].name, 'rgb_frame_id': Path(rgb_path).name, 'depth_frame_id': Path(depth_path).name,
                         'depth_valid_pixels': int(np.asarray(data.observation(k, ci).valid).sum())})
        frames.append({'processing_index': i, 'original_frame_index': int(data.frame_ids[k]), 'dt_used_s': dt, 'cameras': cams})
    return {'sequence_direction': 'forward' if anchor == 'first' else 'backward', 'frames': frames,
            'canonical_endpoint': int(data.frame_ids[order[0]]), 'processing_order_original_indices': [int(data.frame_ids[k]) for k in order]}


def _held(xyz, observed, endpoint):
    held = np.array(xyz)
    K = len(held)
    for k in (range(1, K) if endpoint == 0 else range(K-2, -1, -1)):
        prev = k-1 if endpoint == 0 else k+1
        held[k] = np.where(observed[k][:, None], held[k], held[prev])
    return held


def _colour_depth(d, lo, hi, valid):
    g = np.clip((d-lo)/max(hi-lo, 1e-9), 0, 1)
    rgb = np.stack([g, g*.85+.15*(1-g), 1-g], -1)
    rgb[~valid] = 0
    return (rgb*255).astype(np.uint8)


def _write_renders(out, data, cams, ci, rgb, rho, policy, layers, order):
    """RigEuler render rows for ONE camera: gt / phase1 / kabsch / phase2, RGB, depth, |residual|."""
    out.mkdir(parents=True, exist_ok=True)
    cam = data.cameras[ci]; H, W = cam.height, cam.width
    for f, k in enumerate(order):
        obs = data.observation(k, ci)
        dom = np.asarray(obs.valid)
        lo, hi = float(obs.depth[dom].min()), float(obs.depth[dom].max())
        Image.fromarray((np.asarray(obs.rgb)*255*dom[..., None]).astype(np.uint8)).save(out/f'gt_rgb_{f:04d}.png')
        Image.fromarray(_colour_depth(np.asarray(obs.depth), lo, hi, dom)).save(out/f'gt_depth_{f:04d}.png')
        for name, xyz in layers(k):
            v = render(xyz, rgb, rho, [cams[ci]], policy).views[0]
            cov = v.covered.reshape(H, W).cpu().numpy()
            prgb = v.rgb.reshape(H, W, 3).cpu().numpy(); pd = v.depth.reshape(H, W).cpu().numpy()
            Image.fromarray((prgb*255*cov[..., None]).clip(0, 255).astype(np.uint8)).save(out/f'{name}_rgb_{f:04d}.png')
            Image.fromarray(_colour_depth(pd, lo, hi, cov)).save(out/f'{name}_depth_{f:04d}.png')
            # |residual|: RGB residual norm where both observed and predicted; unexplained
            # observed pixels red, predicted-outside-observation orange (no background values read).
            res = np.zeros((H, W, 3)); both = dom & cov
            e = np.clip(np.linalg.norm(np.asarray(obs.rgb)-prgb, axis=-1)/.5, 0, 1)
            res[both] = np.stack([e, e, e], -1)[both]; res[dom & ~cov] = [.86, .16, .16]; res[cov & ~dom] = [.94, .63, .12]
            Image.fromarray((res*255).astype(np.uint8)).save(out/f'{name}_residual_{f:04d}.png')
    return {'frames': list(range(len(order))), 'camera': cam.name}


def write(run, data, scene=None, camera=0, device='cuda', anchors=('first', 'last')):
    run = Path(run); written = {}
    manifest = json.loads((run/'manifest.json').read_text())
    policy = RenderPolicy(depth_variance=manifest['noise']['depth_variance'], depth_squared_variance=manifest['noise']['depth_squared_variance'])
    K = len(data.times); dev = torch.device(device)
    for anchor in anchors:
        d = run/anchor
        if not (d/'phase1/partition.npz').exists(): continue
        endpoint = 0 if anchor == 'first' else K-1
        order = list(range(K)) if endpoint == 0 else list(range(K-1, -1, -1))
        fids = [int(data.frame_ids[k]) for k in order]
        # Positive elapsed PROCESSING time (the reference convention): backward runs process decreasing timestamps,
        # which relative_increments would clamp to dt = 1e-300.
        proc_times = torch.as_tensor(np.abs(np.asarray(data.times)[order]-data.times[order[0]]))
        with np.load(d/'phase1/partition.npz') as s:
            part = Partition(**{k: s[k].item() if k == 'version' else s[k].copy() for k in Partition.__dataclass_fields__})
        xyz = np.load(d/'phase1/state.npy', mmap_mode='r')[..., :3]; obs = np.load(d/'phase1/point_observed.npy')
        rho = np.load(d/'phase1/footprint_radius.npy')
        mu0 = torch.as_tensor(part.xyz)
        positions = [torch.as_tensor(np.asarray(xyz[k])) for k in order]          # processing order
        pages = written[anchor] = []
        # ---- Phase-1 direction ------------------------------------------------------------
        sdd.write_phase_debug(d/'phase1_direction', frame_record(data, order, anchor), scene=f'{data.root.name} {anchor}')
        pages.append(d/'phase1_direction/frame_order.html')
        gt = gt_point_labels(scene, part.xyz, anchor) if scene else np.full(len(part.xyz), -1)
        # ---- coherence ------------------------------------------------------------------
        if (d/'coherence/representatives.npz').exists():
            z = np.load(d/'coherence/representatives.npz'); coh = json.loads((d/'coherence/result.json').read_text())
            sid = torch.as_tensor(z['membership']).long()
            # Per-point positions = the pipeline's information-supported node trajectory of the point's node, so the pinned
            # page code reproduces exactly the node trajectories the extractor used; strains and labels are then replaced by
            # the pipeline's support-weighted values (edges matched by node pair).
            node_traj = torch.as_tensor(z['trajectory'])                                    # chronological [K, n, 3]
            slot = torch.full((int(sid.max())+1,), -1, dtype=torch.long); slot[torch.as_tensor(z['node_ids'])] = torch.arange(len(z['node_ids']))
            pos_of = lambda k: node_traj[k][slot[sid].clamp_min(0)]
            res = run_motion_coherence([pos_of(k) for k in order], sid, knn=coh.get('knn', 12), times=proc_times, verbose=False)
            if not np.array_equal(res['node_ids'].numpy(), z['node_ids']): raise AssertionError('representative node order differs from the extractor')
            ours = {(int(a), int(b)): i for i, (a, b) in enumerate(z['edges'])}
            idx = torch.as_tensor([ours.get((int(a), int(b)), ours.get((int(b), int(a)), -1)) for a, b in zip(res['edge_i'], res['edge_j'])])
            if bool((idx < 0).any()): raise AssertionError('canonical kNN graph differs from the extractor')
            res['C'] = torch.as_tensor(z['C'])[idx]; res['z'] = torch.as_tensor(z['log_strain'])[:, idx]
            labels = [str(x) for x in z['labels']]
            res['node_class'] = labels; res['node_class_counts'] = coh.get('counts')
            res['labels'] = torch.as_tensor([1 if l == 'MOVING_RIGID' else 0 for l in labels])   # page edge colouring = this pipeline's decision
            res['identification'] = {'status': coh['status'], 'dynamic_component': 1}; res['status'] = coh['status']
            for c in coh['candidates']:
                for key in ('death_merge_edge_cost', 'persistence'):
                    if c.get(key) is None: c[key] = float('inf')
            res['extraction'] = coh
            dbg = d/'debug'
            mcd.write_finite_strain_heatmap(dbg/'finite_strain_heatmap.html', res)
            mcd.write_differential_strain_heatmap(dbg/'differential_strain_heatmap.html', res)
            mcd.write_representative_edges(dbg/'representative_edges.html', res)
            mcd.write_candidate_tree(dbg/'coherence_component_tree.html', coh, f'{data.root.name} {anchor} - coherence candidate tree')
            base = select_moving_support(positions)
            pred_base = torch.zeros(len(mu0), dtype=torch.bool); pred_base[base['members']] = True
            pred_coh = torch.as_tensor(z['moving_support'])
            mcd.write_selector_comparison_3d(dbg/'static_dynamic_comparison_3d.html', mu0, positions, torch.as_tensor(gt), pred_base, pred_coh,
                                             title=f'{data.root.name} {anchor} - static/dynamic: motion baseline vs coherence (GT = sparse labelled cloud, evaluation only)')
            ids = res['node_ids']
            gt_node = torch.full((len(ids),), -1, dtype=torch.long)
            for i, s in enumerate(ids.tolist()):
                l = gt[z['membership'] == s]; l = l[l >= 0]
                if len(l): gt_node[i] = int(round(l.mean()))
            node_xyz = torch.stack([mu0[sid == int(s)].mean(0) for s in ids.tolist()])
            node_traj = node_traj[order]
            mcd.write_support_3d(dbg/'motion_coherence_support_3d.html', res, node_xyz=node_xyz, traj=node_traj, gt_node=gt_node,
                                 frame_ids=fids, title=f'{data.root.name} {anchor} - motion coherence support')
            pages += [dbg/p for p in ('finite_strain_heatmap.html', 'differential_strain_heatmap.html', 'representative_edges.html',
                                       'coherence_component_tree.html', 'static_dynamic_comparison_3d.html', 'motion_coherence_support_3d.html')]
        # ---- rigid Phase 2 and joint reduction ----------------------------------------
        if (d/'rigid/poses.npy').exists():
            Tm = torch.as_tensor(np.load(d/'rigid/poses.npy'))[order]; Km = torch.as_tensor(np.load(d/'rigid/kabsch_poses.npy'))[order]
            tw = torch.as_tensor(np.load(d/'rigid/twists.npy'))[order]; Pc = torch.as_tensor(np.load(d/'rigid/pose_covariance.npy'))[order]
            members = torch.as_tensor(np.load(d/'rigid/moving_core_ids.npy')).long()
            rigid_part = SimpleNamespace(R=Tm[:, :3, :3].clone(), p=Tm[:, :3, 3].clone(), v=tw[:, :3].clone(), omega=tw[:, 3:].clone(), num_frames=K)
            X0 = mu0[members]
            resid = [float((positions[t][members]-(X0@Km[t, :3, :3].T+Km[t, :3, 3])).norm(dim=1).mean()) for t in range(K)]
            travel = [float((positions[t][members]-X0).norm(dim=1).mean()) for t in range(K)]
            unexplained = resid[-1]/travel[-1] if travel[-1] > 1e-9 else float('nan')
            red = reduce_trajectory(rigid_part.R, rigid_part.p, mu0, members, P_tau=Pc, times=proc_times)
            gt_joint = load_joint_gt(scene) if scene else {'available': False}
            ev = evaluate(red, gt_joint, frame_ids=fids) if gt_joint.get('available') else gt_joint
            json_write(d/'rigid/html_gt_evaluation.json', ev)
            cams = [DebugCamera(c) for c in data.cameras]
            html = d/'rigid/html'
            rd.write_rigid_phase2_3d(html/'rigid_phase2_debug.html', mu0, members, positions, rigid_part, Km[:, :3, :3], Km[:, :3, 3], cams,
                                     title=f'{data.root.name} {anchor} - rigid SE(3) image IEKF', unexplained=unexplained)
            rd.write_rigid_to_joint_3d(html/'rigid_to_joint_debug.html', mu0, members, positions, rigid_part, Km[:, :3, :3], Km[:, :3, 3], red, ev,
                                       unexplained=unexplained, title=f'{data.root.name} {anchor} - rigid reconstruction and joint reduction')
            recs, rows = [], []
            for t, k in enumerate(order):
                r = json.loads((d/'rigid/frames'/f'{k:06d}.json').read_text())
                info = r.get('posterior_information_eigenvalues')
                recs.append(SimpleNamespace(energy_after=r.get('energy_final', r.get('image_energy')),
                                            observability={'posterior_eigenvalues': [1/x for x in info] if info else [float('nan')]}))
                dT = torch.linalg.inv(Km[t])@Tm[t]
                dR = float(se3_log(dT[:3, :3].reshape(1, 3, 3), dT[:3, 3].reshape(1, 3))[0, 3:].norm())
                rows.append([t, dR, float(np.degrees(dR)), float((Km[t, :3, 3]-Tm[t, :3, 3]).norm())])
            rd.write_timelines(html/'rigid_timelines.html', rigid_part, red, recs, rows, title=f'{data.root.name} {anchor} - timelines')
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)
            canon, rgb, rho_t = t(part.xyz), t(part.rgb), t(rho)
            core = torch.zeros(len(canon), dtype=torch.bool, device=dev); core[members.to(dev)] = True
            tcams = [CameraTensors(c, dev) for c in data.cameras]
            pose_of = {k: (Km[i], Tm[i]) for i, k in enumerate(order)}

            def layers(k):
                kab, rig = (m.to(dev) for m in pose_of[k])
                move = lambda T: torch.where(core[:, None], canon@T[:3, :3].T+T[:3, 3], canon)
                return [('phase1', t(xyz[k])), ('kabsch', move(kab)), ('phase2', move(rig))]
            renders = _write_renders(d/'rigid/renders', data, tcams, camera, rgb, rho_t, policy, layers, order)
            rd.write_render_html(html/'rigid_renders.html', renders, rel='../renders',
                                 title=f'{data.root.name} {anchor} - camera {renders["camera"]} (observed / Phase 1 / Kabsch / rigid IEKF)')
            pages += [html/p for p in ('rigid_phase2_debug.html', 'rigid_to_joint_debug.html', 'rigid_timelines.html', 'rigid_renders.html')]
        for p in pages: _local_three(p)
    index = run/'debug_index.html'
    links = ''.join(f'<h2>{a}</h2><ul>'+''.join(f'<li><a href="{p.relative_to(run)}">{p.relative_to(run)}</a></li>' for p in ps)+'</ul>' for a, ps in written.items())
    index.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>RigEuler-style debug</title><style>body{{background:#0e1116;color:#e6edf3;'
                     f'font:14px/1.6 ui-monospace,monospace;padding:20px}}a{{color:#58a6ff}}h2{{color:#58a6ff;font-size:14px}}</style></head><body>'
                     f'<h1>RigEuler3DGS debug pages - {data.root.name}</h1><p>Pinned RigEuler 481b2cf2 page writers fed with the rendered-image RGB-D pipeline outputs. '
                     f'Interactive viewer: <a href="viewer/index.html">viewer/index.html</a>.</p>{links}</body></html>')
    return index, written
