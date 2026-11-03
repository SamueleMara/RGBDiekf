"""Point-only 3D visibility DIAGNOSTIC (PDF 8.2, spec 6.3/6.6/6.1; contract E15). The visibility LAW p(V | X, C) is
NOT_DEFINED_IN_THEORY, so nothing here enters a likelihood, a weight or a label: it classifies, per camera, the ray from
the calibrated camera centre o_c through a provisional particle x against the OBSERVED 3D samples of that camera that
lie in the same angular bin (the calibrated pixel footprint of K is the camera's own angular sampling; samples carry
their source pixel as ray provenance). Only ray ORDERING of 3D points is used; no RGB/depth image residual, gradient or
mask energy. n_sigma is the declared projective visibility_n_sigma; sigma^2 = depth variance of the observed sample +
that of the particle at its depth (config.measurement_covariance).
Per (point, camera): SUPPORTED |z_obs - z| <= n sigma | OCCLUDED_BY_OBSERVED_SURFACE z_obs < z - n sigma |
FREE_SPACE_CONFLICT z_obs > z + n sigma | NO_SAMPLE_IN_RAY_BIN (bin not mask-valid) | OUTSIDE_FRUSTUM.
Per point: VISIBLE (any camera supported) | OCCLUDED (none supported, some occluded) | UNSUPPORTED.
"""
import numpy as np
import torch

CAM_CODES = {0: 'SUPPORTED', 1: 'OCCLUDED_BY_OBSERVED_SURFACE', 2: 'FREE_SPACE_CONFLICT', 3: 'NO_SAMPLE_IN_RAY_BIN', 4: 'OUTSIDE_FRUSTUM'}
POINT_CODES = {0: 'VISIBLE', 1: 'OCCLUDED', 2: 'UNSUPPORTED', 3: 'NOT_EVALUATED'}


def ray_visibility(x, cloud, cameras, depth_var, n_sigma):
    """x [N,3] torch; cloud PointCloudFrame (numpy); cameras: dataset cameras; depth_var(z) -> variance [m^2]."""
    dev, dt = x.device, x.dtype; N = len(x)
    codes = torch.full((N, len(cameras)), 4, dtype=torch.int8, device=dev)
    unexplained = []
    for c, cam in enumerate(cameras):
        W = torch.as_tensor(np.asarray(cam.world_to_camera, float), dtype=dt, device=dev)
        fx, fy, cx, cy = (float(v) for v in cam.intrinsics)
        xc = x@W[:3, :3].T+W[:3, 3]; z = xc[:, 2]
        ok = z > 0
        u = torch.round(fx*xc[:, 0]/z.clamp_min(1e-12)+cx); v = torch.round(fy*xc[:, 1]/z.clamp_min(1e-12)+cy)
        ok &= (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        sel = cloud.camera == c
        zmap = torch.full((cam.width*cam.height,), float('nan'), dtype=dt, device=dev)
        zmap[torch.as_tensor(cloud.source[sel, 2], device=dev)] = torch.as_tensor(cloud.depth[sel], dtype=dt, device=dev)
        pix = (v*cam.width+u).long().clamp(0, cam.width*cam.height-1)
        zo = zmap[pix]
        sig = torch.sqrt(depth_var(zo)+depth_var(z))*n_sigma
        code = torch.full((N,), 3, dtype=torch.int8, device=dev)
        has = torch.isfinite(zo)
        code[has & (zo-z).abs().le(sig)] = 0; code[has & (zo < z-sig)] = 1; code[has & (zo > z+sig)] = 2
        code[~ok] = 4; codes[:, c] = code
        hit = torch.zeros_like(zmap, dtype=torch.bool); hit[pix[code == 0]] = True
        unexplained.append(int((torch.isfinite(zmap) & ~hit).sum()))
    point = torch.full((N,), 2, dtype=torch.int8, device=dev)
    point[(codes == 1).any(1)] = 1; point[(codes == 0).any(1)] = 0
    return codes, point, unexplained
