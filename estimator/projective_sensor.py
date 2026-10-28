"""Measured RGB-D frames on the GPU and validity-aware bilinear sampling of the ACTUAL images at projected coordinates.

Nothing here synthesizes an image: every returned value is a bilinear combination of four measured pixels of camera c
at frame k, and every gradient is the exact derivative of that bilinear interpolant (not a Sobel approximation):

    w00=(1-du)(1-dv), w10=du(1-dv), w01=(1-du)dv, w11=du dv      (u0=floor u, v0=floor v, du=u-u0, dv=v-v0)
    F(u,v)   = w00 F00 + w10 F10 + w01 F01 + w11 F11
    dF/du    = (1-dv)(F10-F00) + dv(F11-F01),    dF/dv = (1-du)(F01-F00) + du(F11-F10)

The interpolant is piecewise bilinear, so its derivative is exact inside a pixel cell and jumps across cell borders.
A sample exists only when all four neighbours are inside the image, inside the hard foreground mask and have valid
metric depth, and lie on one surface (configs/projective_rgbd.json same-surface rule); otherwise a reason code is
returned and no value, gradient or variance is used. Invalid neighbours are never interpolated as zero.

Variance (contract 2.1): independent pixel noise through the interpolation weights, sum_j w_j^2 sigma_j^2, plus the
declared model-mismatch variances (0 unless declared).
"""
from dataclasses import dataclass, field
import math
import numpy as np
import torch

REASONS = {0: 'VALID', 1: 'BEHIND_CAMERA', 2: 'OUTSIDE_IMAGE_FOOTPRINT', 3: 'MASK_EXCLUDED', 4: 'INVALID_DEPTH',
           5: 'DEPTH_DISCONTINUITY', 6: 'SELF_OCCLUDED', 7: 'DUPLICATE_PIXEL_SUPPRESSED', 8: 'MEASURED_SURFACE_IN_FRONT',
           9: 'NOT_PROCESSED'}
CODE = {v: k for k, v in REASONS.items()}


@dataclass
class SensorNoise:
    rgb_variance: torch.Tensor        # [3] sensor variance per channel (encoded RGB in [0,1])
    depth_variance: float             # m^2
    depth_squared_variance: float     # dimensionless (sigma_D^2 = a + b z^2)
    rgb_model_variance: float = 0.
    depth_model_variance: float = 0.
    interpolation_variance: str = 'sum_w2'
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config, policy, device, dtype=torch.float64):
        c = np.asarray(config.color_covariance, float)
        if not np.allclose(c, np.diag(np.diag(c))): raise ValueError('diagonal RGB sensor covariance required (correlated colour noise is not implemented)')
        return cls(torch.as_tensor(np.diag(c), dtype=dtype, device=device), float(config.depth_variance), float(config.depth_squared_variance),
                   float(policy['rgb_model_variance']), float(policy['depth_model_variance']), policy['interpolation_variance'],
                   {k: config.provenance.get(k) for k in ('color_covariance', 'depth_variance', 'depth_squared_variance')})

    def depth_sigma2(self, z):
        return self.depth_variance+self.depth_squared_variance*z*z


class MeasuredFrame:
    """All cameras of ONE acquisition k: rgb [C,H,W,3], depth [C,H,W], mask [C,H,W], valid = mask AND finite depth > 0."""
    def __init__(self, dataset, k, device, dtype=torch.float32):
        frames = [dataset.observation(k, c) for c in range(len(dataset.cameras))]
        shapes = {f.depth.shape for f in frames}
        if len(shapes) != 1: raise ValueError(f'cameras of frame {k} have different image sizes {shapes}: ragged camera tensors are not implemented')
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt, device=device)
        self.k = k
        self.rgb = t([np.asarray(f.rgb) for f in frames], dtype)
        self.depth = t([np.asarray(f.depth, np.float64) for f in frames], dtype)
        self.mask = t([np.asarray(f.mask, bool) for f in frames], torch.bool)
        self.raw_valid = t([np.asarray(f.raw_valid, bool) for f in frames], torch.bool)
        self.valid = self.mask & self.raw_valid
        self.H, self.W = frames[0].depth.shape
        self.bytes = sum(int(x.numel()*x.element_size()) for x in (self.rgb, self.depth, self.mask, self.raw_valid))
        base = getattr(dataset, 'base', dataset)
        entries, masks = getattr(base, 'entries', None), getattr(base, 'masks', None)
        self.sources = [{'camera': dataset.cameras[c].name, 'rgb': entries[k][c][0] if entries else None, 'depth': entries[k][c][1] if entries else None,
                         'mask': masks[k][c] if masks else None} for c in range(len(frames))]
        fp = getattr(base, 'observation_fingerprints', {})
        self.fingerprints = [fp.get(f'{k:06d}_{c:03d}') for c in range(len(frames))]


@dataclass
class Samples:
    rgb: torch.Tensor          # [M,3] measured, bilinear
    depth: torch.Tensor        # [M]
    grad_rgb: torch.Tensor     # [M,3,2] d rgb / d(u,v)
    grad_depth: torch.Tensor   # [M,2]
    var_rgb: torch.Tensor      # [M,3]
    var_depth: torch.Tensor    # [M]
    reason: torch.Tensor       # [M] uint8, 0 = valid
    pixel00: torch.Tensor      # [M,2] (u0, v0) of the 2x2 support (provenance)


def sample(frame, cam, uv, noise, policy, K, dtype=torch.float64):
    """Bilinear samples of frame at rows (cam [M], uv [M,2]); K [C,4] intrinsics for the same-surface rule."""
    M = len(cam)
    dev = uv.device
    u, v = uv[:, 0].to(dtype), uv[:, 1].to(dtype)
    finite = torch.isfinite(u) & torch.isfinite(v)
    u0f, v0f = torch.floor(torch.where(finite, u, torch.zeros_like(u))), torch.floor(torch.where(finite, v, torch.zeros_like(v)))
    inside = finite & (u0f >= 0) & (v0f >= 0) & (u0f+1 <= frame.W-1) & (v0f+1 <= frame.H-1)
    u0 = u0f.clamp(0, frame.W-2).long(); v0 = v0f.clamp(0, frame.H-2).long()
    du, dv = (u-u0f).clamp(0, 1), (v-v0f).clamp(0, 1)
    c = cam.long()
    idx = [(v0, u0), (v0, u0+1), (v0+1, u0), (v0+1, u0+1)]                     # 00, 10, 01, 11  (F_ab: a = +u, b = +v)
    D = torch.stack([frame.depth[c, y, x] for y, x in idx], 1).to(dtype)        # [M,4]
    I = torch.stack([frame.rgb[c, y, x] for y, x in idx], 1).to(dtype)          # [M,4,3]
    mk = torch.stack([frame.mask[c, y, x] for y, x in idx], 1).all(1)
    dok = torch.stack([frame.raw_valid[c, y, x] for y, x in idx], 1).all(1)
    reason = torch.full((M,), CODE['VALID'], dtype=torch.uint8, device=dev)
    reason[~dok] = CODE['INVALID_DEPTH']
    reason[~mk] = CODE['MASK_EXCLUDED']
    reason[~inside] = CODE['OUTSIDE_IMAGE_FOOTPRINT']
    zbar = D.mean(1)
    f = torch.minimum(K[c, 0], K[c, 1]).to(dtype)
    allow = policy['same_surface_n_sigma']*math.sqrt(2.)*noise.depth_sigma2(zbar).sqrt() + \
        zbar*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2.)/f
    broken = (D.max(1).values-D.min(1).values) > allow
    reason[(reason == 0) & broken] = CODE['DEPTH_DISCONTINUITY']
    w = torch.stack(((1-du)*(1-dv), du*(1-dv), (1-du)*dv, du*dv), 1)            # [M,4]
    depth = (w*D).sum(1); rgb = (w[:, :, None]*I).sum(1)
    gD = torch.stack(((1-dv)*(D[:, 1]-D[:, 0])+dv*(D[:, 3]-D[:, 2]), (1-du)*(D[:, 2]-D[:, 0])+du*(D[:, 3]-D[:, 1])), 1)
    gI = torch.stack(((1-dv)[:, None]*(I[:, 1]-I[:, 0])+dv[:, None]*(I[:, 3]-I[:, 2]),
                      (1-du)[:, None]*(I[:, 2]-I[:, 0])+du[:, None]*(I[:, 3]-I[:, 1])), 2)   # [M,3,2]
    if noise.interpolation_variance == 'sum_w2':
        w2 = w*w
        var_rgb = w2.sum(1)[:, None]*noise.rgb_variance.to(dtype)[None]
        var_depth = (w2*noise.depth_sigma2(D)).sum(1)
    elif noise.interpolation_variance == 'sensor':
        var_rgb = noise.rgb_variance.to(dtype)[None].expand(M, 3).clone(); var_depth = noise.depth_sigma2(depth)
    else:
        raise ValueError(noise.interpolation_variance)
    var_rgb = var_rgb+noise.rgb_model_variance; var_depth = var_depth+noise.depth_model_variance
    return Samples(rgb, depth, gI, gD, var_rgb, var_depth, reason, torch.stack((u0, v0), 1))


def sample_cpu_oracle(rgb, depth, valid, u, v):
    """Independent float64 NumPy oracle for ONE sample (tests): value and exact bilinear gradient, or None if invalid."""
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    H, W = depth.shape
    if u0 < 0 or v0 < 0 or u0+1 > W-1 or v0+1 > H-1: return None
    nb = [(v0, u0), (v0, u0+1), (v0+1, u0), (v0+1, u0+1)]
    if not all(valid[y, x] for y, x in nb): return None
    du, dv = u-u0, v-v0
    F = lambda img: np.array([np.asarray(img[y, x], float) for y, x in nb])
    out = {}
    for name, img in (('rgb', rgb), ('depth', depth)):
        a = F(img)
        out[name] = (1-du)*(1-dv)*a[0]+du*(1-dv)*a[1]+(1-du)*dv*a[2]+du*dv*a[3]
        out[name+'_du'] = (1-dv)*(a[1]-a[0])+dv*(a[3]-a[2])
        out[name+'_dv'] = (1-du)*(a[2]-a[0])+du*(a[3]-a[1])
    return out
