"""Point-cloud-only boundary of the joint K = 1 pipeline (CODEX_RGBDiekf_SVGD_JOINT_RIGID_SUPPORT_K1_2026-11-03.md, hard constraint 1).

The camera-aware loader (scene.pointcloud_sequence.build_masked_cloud: strict masks, valid depth, calibrated back-projection) materializes
each frame; this adapter then keeps ONLY world XYZ, encoded RGB in [0,1], the per-point XYZ covariance, the RGB measurement covariance,
reference IDs and timestamps. Camera indices, pixels, depths, rays and camera centres never cross into the estimator: the frame object has
no such attributes and `PointOnlyFrame.__getattr__` raises for any of them (tests/test_svgd_k1_observation.py checks it).
"""
from dataclasses import dataclass
import hashlib
import numpy as np

FORBIDDEN = ('camera', 'pixel', 'depth', 'ray', 'source', 'camera_centers', 'mask', 'intrinsics', 'normal')


@dataclass(frozen=True)
class PointOnlyFrame:
    xyz: np.ndarray        # [M,3] world m
    rgb: np.ndarray        # [M,3] encoded RGB [0,1]
    cov_xyz: np.ndarray    # [M,3,3] m^2 measurement covariance
    cov_rgb: np.ndarray    # [3,3] RGB measurement covariance (shared, measured)
    time_s: float

    @property
    def M(self): return len(self.xyz)

    def __getattr__(self, k):
        if any(f in k for f in FORBIDDEN): raise AttributeError(f'POINT_ONLY_BOUNDARY: {k!r} is not available to the K=1 estimator')
        raise AttributeError(k)

    def digest(self):
        h = hashlib.sha256()
        for a in (self.xyz, self.rgb, self.cov_xyz): h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class PointOnlySequence:
    frames: tuple           # PointOnlyFrame per selected frame; frames[0] is the reference cloud P_0
    reference_ids: np.ndarray   # [N] int64 canonical IDs of P_0 (= endpoint_samples order of the loader)
    times: np.ndarray

    @property
    def reference(self): return self.frames[0]

    def __len__(self): return len(self.frames)


def from_dataset(dataset, config, frames=None):
    """Materialize with the camera-aware loader, then drop everything that is not a point attribute."""
    from scene.pointcloud_sequence import build_masked_cloud
    ks = range(len(dataset.times)) if frames is None else frames
    R = np.asarray(config.color_covariance, float)
    out, counts = [], []
    for k in ks:
        cl = build_masked_cloud(dataset, k, config)
        out.append(PointOnlyFrame(cl.xyz.copy(), np.asarray(cl.rgb, float).copy(), cl.cov.copy(), R.copy(), float(cl.time_s)))
        counts.append({'frame': int(k), 'M': cl.M, 'cloud_sha256': cl.digest(), 'per_camera_foreground_valid': [c['foreground_valid'] for c in cl.counts]})
    seq = PointOnlySequence(tuple(out), np.arange(out[0].M, dtype=np.int64), np.array([f.time_s for f in out]))
    return seq, counts        # counts: loader-side provenance for the manifest only, never passed to the estimator


def from_arrays(clouds, times, cov_rgb):
    """Tests / synthetic: clouds = [(xyz, rgb, cov_xyz), ...]."""
    fr = tuple(PointOnlyFrame(np.asarray(x, float), np.asarray(c, float), np.asarray(S, float), np.asarray(cov_rgb, float), float(t)) for (x, c, S), t in zip(clouds, times))
    return PointOnlySequence(fr, np.arange(fr[0].M, dtype=np.int64), np.asarray(times, float))
