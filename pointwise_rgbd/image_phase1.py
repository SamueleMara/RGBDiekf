"""Phase 1 driver: free [XYZ, velocity] points, rendered-image grouped IEKF.

Forward (anchor 'first') initializes from real first-frame RGB-D of every camera and processes
chronologically; backward (anchor 'last') initializes from final-frame RGB-D and processes in reverse
with a positive elapsed duration and the reverse-time transition on the PHYSICAL velocity
(process6: F = [[I, -dt I], [0, I]]; equivalent to the reference's processing-time velocity v_p = -v).
Endpoint source pixels enter once as the canonical prior; every other frame's images are new
measurements, assimilated exactly once. No association: the measurement is the rendered RGB-D image
of the transported coloured cloud.

Catalogue: every valid endpoint pixel of every camera, back-projected (surface_supernodes.endpoint_samples).
Correction groups: the pinned build_supernodes(target_size=64) on the canonical catalogue (computational
only; no persistent pose; the association-gated surface graph is not used).
"""
from pathlib import Path
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from .surface_supernodes import Partition, endpoint_samples
from .state6 import anchor_posterior
from .process6 import predict
from .supernodes import GroupBasis, build_supernodes
from .rgbd_point_renderer import CameraTensors, footprints, render
from .image_observation import FrameObservation, levels
from .image_iekf import frame_step
from .phase1_covariance import make_covariance
from .store import json_write

MEASUREMENT_MODEL = 'rendered_rgbd_point_cloud_v2'


def endpoint_catalogue(dataset, anchor, config, iekf):
    """(endpoint index, Partition with the Phase-1 computational supernodes, SupernodePartition)."""
    k0, _, xyz, rgb, cov, source = endpoint_samples(dataset, anchor, config)
    sn = build_supernodes(torch.as_tensor(xyz, dtype=torch.float64), target_size=iekf.supernode_target_size,
                          knn=iekf.supernode_knn, seed=iekf.supernode_seed)
    member = sn.supernode_id.numpy()
    part = Partition(xyz, rgb, source, cov, member, sn.centroids.numpy(), sn.radius.numpy(), np.zeros((0, 2), np.int64),
                     sn.adjacency.numpy(), np.arange(len(xyz)))
    return k0, part, sn


def crop_box(*masks, pad=4):
    hit = np.zeros_like(masks[0])
    for m in masks: hit |= m
    if not hit.any(): return None
    ys, xs = np.nonzero(hit)
    h, w = hit.shape
    return [max(0, ys.min()-pad), min(h, ys.max()+pad+1), max(0, xs.min()-pad), min(w, xs.max()+pad+1)]


def save_images(path, frames, lin):
    """Observed / predicted RGB-D, domain and coverage per camera (cropped to the object)."""
    arrays = {}
    for ci, (f, v) in enumerate(zip(frames, lin.views)):
        h, w = f.depth.shape
        cov = v.covered.reshape(h, w).cpu().numpy(); dom = np.asarray(f.valid)
        box = crop_box(cov, dom)
        if box is None: continue
        y0, y1, x0, x1 = box; sl = (slice(y0, y1), slice(x0, x1))
        arrays.update({f'c{ci}_box': np.array(box), f'c{ci}_domain': dom[sl], f'c{ci}_covered': cov[sl],
            f'c{ci}_obs_rgb': np.round(np.asarray(f.rgb)[sl]*255).astype(np.uint8),
            f'c{ci}_obs_depth': np.asarray(f.depth, np.float32)[sl],
            f'c{ci}_pred_rgb': np.round(v.rgb.reshape(h, w, 3).cpu().numpy()[sl]*255).clip(0, 255).astype(np.uint8),
            f'c{ci}_pred_depth': v.depth.reshape(h, w).cpu().numpy().astype(np.float32)[sl],
            f'c{ci}_contributors': np.bincount(v.pixel.cpu().numpy(), minlength=h*w).reshape(h, w)[sl].astype(np.int16)})
    np.savez_compressed(path, **arrays)


def run(dataset, config, backend, root, anchor='first', iekf=None, render_policy=None,
        obs_policy=None, footprint_scale=1.5, catalog=None, save_every=1):
    from .image_iekf import IEKFConfig
    from .rgbd_point_renderer import RenderPolicy
    from .image_observation import ObservationPolicy
    iekf = iekf or IEKFConfig()
    render_policy = render_policy or RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
    obs_policy = obs_policy or ObservationPolicy()
    root = Path(root); (root/'frames').mkdir(parents=True, exist_ok=True); (root/'images').mkdir(exist_ok=True)
    k0, partition, sn = endpoint_catalogue(dataset, anchor, config, iekf) if catalog is None else catalog
    partition.save(root/'partition.npz')
    json_write(root/'supernodes.json', sn.to_dict())
    rho = footprints(partition, dataset, footprint_scale)
    np.save(root/'footprint_radius.npy', rho)
    state, p = anchor_posterior(partition.xyz, partition.source_covariance, config.velocity_covariance)
    n, K = len(state), len(dataset.times)
    arrays = {}
    for name, shape, dtype in [('state', (K, n, 6), 'float64'), ('covariance', (K, n, 6, 6), 'float64'), ('point_observed', (K, n), 'bool')]:
        arrays[name] = np.lib.format.open_memmap(root/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
    arrays['xyz'] = arrays['state'][..., :3]
    t = backend.tensor
    G = GroupBasis(t(partition.xyz), t(partition.membership, torch.long), t(partition.centroids))
    cams = [CameraTensors(c, backend.device) for c in dataset.cameras]
    rgb, rho_t = t(partition.rgb), t(rho)
    order = list(range(K)) if anchor == 'first' else list(range(K-1, -1, -1))
    arrays['state'][k0], arrays['covariance'][k0], arrays['point_observed'][k0] = state, p, True
    manifest = {'measurement_model': MEASUREMENT_MODEL, 'state': 'individual XYZ + physical-time velocity per point, no group pose',
        'anchor': anchor, 'anchor_frame': k0, 'order': order, 'timestamps_seconds': dataset.times.tolist(),
        'points': n, 'groups': partition.count, 'camera_policy': 'all_available',
        'production_camera_indices': list(range(len(dataset.cameras))), 'association_used': False,
        'footprint': {'scale_source_pixels': footprint_scale, 'rule': 'rho_i = scale * z_src / sqrt(fx fy) of the actual source pixel'},
        'renderer': render_policy.manifest(), 'observation': obs_policy.manifest(), 'iekf': iekf.manifest(),
        'posterior': iekf.posterior, 'process': {'transition': 'F = [[I, dt I], [0, I]] (reverse: -dt on the physical velocity)',
                                                 'noise': 'white acceleration qc [dt^3/3, dt^2/2, dt] + position random walk qx dt (documented densities)'},
        'correction_basis': 'raw reference grouped tangent [dt, dtheta, du, domega] per pinned build_supernodes group; no persistent supernode pose'}
    json_write(root/'run.json', manifest)
    frames = [dataset.observation(k0, ci) for ci in range(len(dataset.cameras))]
    obs = FrameObservation(frames, config, obs_policy, backend.device)
    lin = render(t(state[:, :3]), rgb, rho_t, cams, render_policy)
    terms = obs.terms(lin)
    save_images(root/'images'/f'{k0:06d}.npz', frames, lin)
    json_write(root/'frames'/f'{k0:06d}.json', {'frame': k0, 'original_frame': dataset.frame_ids[k0], 'role': 'ANCHOR_CANONICAL_GAUGE_NO_UPDATE',
        'note': 'endpoint source pixels are the canonical prior; this energy is endpoint model error, not motion', 'image_energy': terms['energy'], 'cameras': terms['cameras']})
    s = t(state); cov = make_covariance(iekf.posterior, t(p), G)
    summary, start = [], time.monotonic()
    for previous, k in tqdm(list(zip(order, order[1:])), desc=f'Image IEKF ({anchor})', unit='frame'):
        dt = abs(float(dataset.times[k]-dataset.times[previous])); reverse = k < previous
        s_minus = predict(s, cov.marginals(), dt, config, reverse)[0]; cov.predict(dt, reverse, config)
        frames = [dataset.observation(k, ci) for ci in range(len(dataset.cameras))]
        assert len(frames) == len(cams), 'Every available camera must contribute to each update'
        s, P, observed, lin, terms, report = frame_step(s_minus, cov, levels(frames, config, obs_policy, backend.device, iekf.pyramid),
                                                        rgb, rho_t, render_policy, G, iekf)
        arrays['state'][k] = s.cpu().numpy(); arrays['covariance'][k] = P.cpu().numpy(); arrays['point_observed'][k] = observed.cpu().numpy()
        if save_every and (k % save_every == 0 or k in (order[1], order[-1])): save_images(root/'images'/f'{k:06d}.npz', frames, lin)
        qs = torch.tensor([0, .05, .5, .95, 1], dtype=torch.float64)
        ev_x, ev_v = (torch.linalg.eigvalsh(P[:, sl, sl].cpu()) for sl in (slice(0, 3), slice(3, 6)))
        report['covariance_spectrum'] = {'position_eigenvalue_quantiles_m2': ev_x.flatten().quantile(qs).tolist(),
                                         'velocity_eigenvalue_quantiles_m2_s2': ev_v.flatten().quantile(qs).tolist(), 'quantiles': qs.tolist()}
        report.update(frame=k, original_frame=dataset.frame_ids[k], signed_dt_seconds=float(dataset.times[k]-dataset.times[previous]),
                      gpu=backend.memory() if backend.device.type == 'cuda' else None)
        json_write(root/'frames'/f'{k:06d}.json', report)
        summary.append({**{key: report[key] for key in ('frame', 'energy_initial', 'energy_final', 'accepted_steps', 'rejected_trials', 'stop', 'seconds', 'directly_observed_points')},
                        'posterior_psd_blocks_floored': report['posterior']['psd_repair']['blocks_floored']})
        for a in arrays.values():
            if hasattr(a, 'flush'): a.flush()
    json_write(root/'summary.json', {'measurement_model': MEASUREMENT_MODEL, 'frames': K, 'cameras': len(cams), 'points': n,
        'groups': partition.count, 'runtime_seconds': time.monotonic()-start, 'frame_reports': summary,
        'physical_rigidity_imposed': False, 'persistent_group_poses': False, 'association_used': False})
    return partition, arrays, rho
