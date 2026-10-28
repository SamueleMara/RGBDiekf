"""Endpoint catalogue inputs: back-projected valid RGB-D pixels of every camera at one endpoint frame.

Partition is the saved canonical catalogue record (points, colours, source pixels, source covariance and the Phase-1
computational groups); its membership is a computational hypothesis, not a physical part."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass
class Samples:
    frame: object
    pixel: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    covariance: np.ndarray
    R: np.ndarray
    rgb: np.ndarray


@dataclass
class Partition:
    xyz: np.ndarray
    rgb: np.ndarray
    source: np.ndarray
    source_covariance: np.ndarray
    membership: np.ndarray
    centroids: np.ndarray
    radius: np.ndarray
    edges: np.ndarray
    adjacency: np.ndarray
    duplicate_unit: np.ndarray
    version: int = 0

    @property
    def count(self): return len(self.centroids)

    def digest(self):
        h = hashlib.sha256()
        for a in (self.xyz, self.rgb, self.source, self.membership, self.edges, self.duplicate_unit): h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(path, **vars(self), partition_hash=np.array(self.digest()), membership_status=np.array('COMPUTATIONAL_HYPOTHESIS_NOT_PHYSICAL_PART'),
                            canonical_uncertainty=np.array('conditioned measured endpoint; source covariance retained for provenance'))


def backproject(z, camera, measurement_covariance):
    """Pixel (u, v, depth) -> world XYZ and its first-order covariance."""
    z = np.asarray(z, dtype=np.float64); fx, fy, cx, cy = camera.intrinsics
    ray = np.stack(((z[..., 0]-cx)/fx, (z[..., 1]-cy)/fy, np.ones(z.shape[:-1])), -1)
    xyz = (ray*z[..., 2, None]-camera.world_to_camera[:3, 3])@camera.world_to_camera[:3, :3]
    j = np.zeros((*z.shape[:-1], 3, 3)); j[..., 0, 0] = z[..., 2]/fx; j[..., 1, 1] = z[..., 2]/fy; j[..., 0, 2] = ray[..., 0]; j[..., 1, 2] = ray[..., 1]; j[..., 2, 2] = 1
    j = camera.world_to_camera[:3, :3].T@j
    return xyz, j@measurement_covariance@j.swapaxes(-1, -2)


def samples(frame, config):
    pixel = np.flatnonzero(frame.valid.ravel())
    depth = np.asarray(frame.depth).ravel()[pixel]; w = frame.camera.width
    z = np.c_[pixel % w, pixel//w, depth]; r = config.measurement_covariance(depth)
    xyz, covariance = backproject(z, frame.camera, r)
    return Samples(frame, pixel, z, xyz, covariance, r, np.asarray(frame.rgb).reshape(-1, 3)[pixel])


def endpoint_samples(dataset, anchor, config):
    if anchor not in ('first', 'last'): raise ValueError('anchor must be first or last')
    k = 0 if anchor == 'first' else len(dataset.times)-1
    raws = [samples(dataset.observation(k, ci), config) for ci in range(len(dataset.cameras))]
    if not sum(len(r.pixel) for r in raws): raise ValueError('No valid foreground endpoint observations')
    xyz = np.concatenate([r.xyz for r in raws]); rgb = np.concatenate([r.rgb for r in raws]); cov = np.concatenate([r.covariance for r in raws])
    source = np.concatenate([np.c_[np.full(len(r.pixel), dataset.frame_ids[k]), np.full(len(r.pixel), ci), r.pixel] for ci, r in enumerate(raws)]).astype(np.int64)
    return k, raws, xyz, rgb, cov, source
