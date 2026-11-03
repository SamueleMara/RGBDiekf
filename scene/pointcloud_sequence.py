"""Strict masked, point-only frame materialization for the SVGD moving-support Stage 1 (spec 1.1, 6, 6.1).

Q_t = {(y_j, d_j, c_j, o_j, Sigma_j)}: every mask-positive valid-depth pixel of every camera, back-projected by the
repository's own scene.endpoint.samples (same K, world_to_camera, depth-noise covariance). Masks come from the strict
binary loader (scene.dataloader.exact_binary_mask); no background sample is ever created. j is an observation sample,
never a material ID. The canonical catalogue P_0 is the masked frame-0 cloud in endpoint_samples order (camera-major,
pixel-ascending); original_id = that index, stable under any display subsampling (display never touches this module).
RGB: the dataset loader's encoded RGB in [0,1]^3, no transfer applied (same units as config.color_covariance).
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from scene.endpoint import samples


@dataclass
class PointCloudFrame:
    k: int                       # frame index in the selected sequence
    frame_id: int                # original frame id
    time_s: float                # acquisition timestamp [s]
    xyz: np.ndarray              # [M,3] world m
    rgb: np.ndarray              # [M,3] encoded RGB [0,1]
    cov: np.ndarray              # [M,3,3] first-order XYZ covariance m^2 (scene.endpoint.backproject)
    depth: np.ndarray            # [M] optical depth m
    camera: np.ndarray           # [M] int64 camera index
    source: np.ndarray           # [M,3] int64 (original frame id, camera, pixel)
    camera_centers: np.ndarray   # [C,3] world camera centres (3D ray provenance o_j = camera_centers[camera[j]])
    counts: list = field(default_factory=list)   # per camera accounting (spec 6.1)

    @property
    def M(self): return len(self.xyz)

    def digest(self):
        h = hashlib.sha256()
        for a in (self.xyz, self.rgb, self.source): h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


@dataclass
class Canonical:
    xyz0: np.ndarray
    rgb0: np.ndarray
    cov0: np.ndarray
    original_id: np.ndarray
    source: np.ndarray

    @property
    def N(self): return len(self.xyz0)

    def ids_sha256(self):
        return hashlib.sha256(np.ascontiguousarray(self.original_id).tobytes()+np.ascontiguousarray(self.source).tobytes()).hexdigest()


def camera_center(camera):
    W = np.asarray(camera.world_to_camera, float)
    return -W[:3, :3].T@W[:3, 3]


def build_masked_cloud(dataset, k, config, cameras=None):
    """PointCloudFrame of frame k from the strict masked RGB-D observations of `cameras` (default: all)."""
    cams = range(len(dataset.cameras)) if cameras is None else cameras
    parts, counts = [], []
    for c in cams:
        fr = dataset.observation(k, c)
        mask, raw = np.asarray(fr.mask, bool), np.asarray(fr.raw_valid, bool)
        s = samples(fr, config)
        if not np.array_equal(np.sort(s.pixel), np.flatnonzero((mask & raw).ravel())):
            raise ValueError(f'MASK_SAMPLE_MISMATCH frame {k} camera {c}: samples are not exactly the mask-positive valid-depth pixels')
        counts.append({'camera': int(c), 'raw_pixels': int(mask.size), 'mask_pixels': int(mask.sum()), 'valid_depth': int(raw.sum()),
                       'foreground_valid': int((mask & raw).sum()), 'exported_world_points': int(len(s.pixel))})
        parts.append((c, s))
    xyz = np.concatenate([s.xyz for _, s in parts]) if parts else np.zeros((0, 3))
    fid = int(dataset.frame_ids[k])
    cloud = PointCloudFrame(k, fid, float(dataset.times[k]), xyz,
                           np.concatenate([np.asarray(s.rgb, float) for _, s in parts]) if parts else np.zeros((0, 3)),
                           np.concatenate([s.covariance for _, s in parts]) if parts else np.zeros((0, 3, 3)),
                           np.concatenate([s.z[:, 2] for _, s in parts]) if parts else np.zeros(0),
                           np.concatenate([np.full(len(s.pixel), c) for c, s in parts]).astype(np.int64) if parts else np.zeros(0, np.int64),
                           np.concatenate([np.c_[np.full(len(s.pixel), fid), np.full(len(s.pixel), c), s.pixel] for c, s in parts]).astype(np.int64)
                           if parts else np.zeros((0, 3), np.int64),
                           np.array([camera_center(dataset.cameras[c]) for c in range(len(dataset.cameras))]), counts)
    if len(np.unique(cloud.source, axis=0)) != cloud.M: raise ValueError(f'DUPLICATE_SOURCE_ID frame {k}: one raw measurement entered twice')
    return cloud


def deduplicate_sources(cloud):
    """Bookkeeping duplicates of the SAME raw measurement (identical source triple) carry no new evidence (TF.6 item 3):
    keep the first occurrence. A different camera exposure has a different source and is kept (camera-sampling law)."""
    _, first = np.unique(cloud.source, axis=0, return_index=True); first = np.sort(first)
    return PointCloudFrame(cloud.k, cloud.frame_id, cloud.time_s, cloud.xyz[first], cloud.rgb[first], cloud.cov[first], cloud.depth[first],
                           cloud.camera[first], cloud.source[first], cloud.camera_centers, cloud.counts)


def canonical_from_frame(cloud):
    if cloud.M == 0: raise ValueError('NO_OBSERVATION: the masked frame-0 cloud is empty; no canonical catalogue can be built')
    return Canonical(cloud.xyz.copy(), cloud.rgb.copy(), cloud.cov.copy(), np.arange(cloud.M, dtype=np.int64), cloud.source.copy())


def increment_noise_study(dataset, cameras=None, triples=8, max_pixels=200_000, seed=0):
    """Input-noise study (spec 3.2; PDF (4)): pixels mask-positive valid in frames k, k+1, k+2 of one camera.
    dz_t = z_t+1 - z_t; for i.i.d. stationary errors corr(dz_t, dz_t+1) = -1/2. Reports depth and the 3x3 RGB increment
    covariance and lag-1 correlation, plus depth quantization peaks. Not a residual of the estimator."""
    K = len(dataset.times); cams = range(len(dataset.cameras)) if cameras is None else cameras
    if K < 3: return {'status': 'NOT_RUN_FEWER_THAN_3_FRAMES'}
    g = np.random.default_rng(seed)
    ks = sorted(set(np.linspace(0, K-3, min(triples, K-2)).round().astype(int).tolist()))
    dz0, dz1, dc0, dc1, zq = [], [], [], [], []
    for k in ks:
        for c in cams:
            fr = [dataset.observation(k+i, c) for i in range(3)]
            ok = np.logical_and.reduce([np.asarray(f.mask, bool) & np.asarray(f.raw_valid, bool) for f in fr]).ravel()
            idx = np.flatnonzero(ok)
            if len(idx) > max_pixels: idx = np.sort(g.choice(idx, max_pixels, replace=False))
            z = [np.asarray(f.depth, float).ravel()[idx] for f in fr]; rgb = [np.asarray(f.rgb, float).reshape(-1, 3)[idx] for f in fr]
            dz0.append(z[1]-z[0]); dz1.append(z[2]-z[1]); dc0.append(rgb[1]-rgb[0]); dc1.append(rgb[2]-rgb[1]); zq.append(np.diff(np.unique(z[0][:50_000])))
    dz0, dz1, dc0, dc1 = map(np.concatenate, (dz0, dz1, dc0, dc1))
    q = np.concatenate(zq); q = q[q > 0]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None
    return {'pixels': int(len(dz0)), 'triples_start_frames': ks,
            'depth_increment_variance_m2': float(dz0.var()), 'depth_increment_lag1_correlation': corr(dz0, dz1),
            'rgb_increment_covariance': np.cov(dc0.T).tolist(), 'rgb_increment_lag1_correlation': [corr(dc0[:, i], dc1[:, i]) for i in range(3)],
            'iid_stationary_expectation_lag1': -0.5, 'depth_quantum_min_m': float(q.min()) if len(q) else None,
            'depth_exactly_unchanged_fraction': float((dz0 == 0).mean()),
            'cross_camera_bias': 'NOT_RUN (scene/noise_calibration.cross_camera_mismatch is the projective-model study; not reused here)',
            'qualification': 'static-majority statistics include genuinely moving pixels; moving minority not removed (no labels at inference)'}
