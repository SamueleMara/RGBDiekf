"""Point-only camera visibility and measured-pixel reuse control (contract 2.2). NOT a renderer, NOT a correspondence.

For one frame and camera, every projected point in front of the camera whose nearest pixel lies in the image enters
a sparse per-pixel table keyed by (camera, nearest pixel). Within a pixel the FRONT-MOST point (smallest camera Z,
ties: smallest point id) is the only point that may use that camera's measured sample there (reuse control, item 5).
Every other point of the pixel is

    SELF_OCCLUDED              z_i - z_front > n_sigma sqrt(sigma_zi^2 + sigma_zfront^2)   (behind the front surface)
    DUPLICATE_PIXEL_SUPPRESSED otherwise (same surface within the uncertainty; same pixel)

with sigma_z^2 = r3^T Sigma_x r3 from the PRIOR position covariance (r3 = third row of R_cw). Those points are
prediction-only for THIS camera; other cameras can still measure them. No image is formed from the cloud: the table
holds point ids and depths only. Points that are not measured (e.g. static-compatible occluders of the rigid pass)
still occupy pixels and can hide measured points.
"""
import torch
from estimator.projective_sensor import CODE


def visibility(proj, sigma_z2, H, W, near, n_sigma, measured=None):
    """proj: Projection [C,N]; sigma_z2 [C,N]; measured [N] bool or None (all).
    Returns reason [C,N] uint8 (0 = may be sampled), front [C,N] bool."""
    C, N = proj.z.shape
    dev = proj.z.device
    z, uv = proj.z, proj.uv
    finite = torch.isfinite(uv).all(-1) & torch.isfinite(z)
    infront = finite & (z > near[:, None])
    pu, pv = torch.round(torch.where(finite, uv[..., 0], torch.zeros_like(z))), torch.round(torch.where(finite, uv[..., 1], torch.zeros_like(z)))
    inimg = infront & (pu >= 0) & (pu <= W-1) & (pv >= 0) & (pv <= H-1)
    reason = torch.full((C, N), CODE['VALID'], dtype=torch.uint8, device=dev)
    reason[~infront] = CODE['BEHIND_CAMERA']
    reason[infront & ~inimg] = CODE['OUTSIDE_IMAGE_FOOTPRINT']
    front = torch.zeros((C, N), dtype=torch.bool, device=dev)
    cam = torch.arange(C, device=dev)[:, None].expand(C, N)
    flat = inimg.reshape(-1)
    ids = torch.nonzero(flat).flatten()
    if len(ids) == 0: return reason, front
    key = (cam.reshape(-1)[ids]*(H*W)+pv.reshape(-1)[ids].long()*W+pu.reshape(-1)[ids].long())
    zz = z.reshape(-1)[ids]
    pid = ids % N
    order = torch.argsort(pid, stable=True)                        # ties: smallest point id
    order = order[torch.argsort(zz[order], stable=True)]           # then depth
    order = order[torch.argsort(key[order], stable=True)]          # grouped by pixel
    k_sorted = key[order]
    first = torch.ones_like(k_sorted, dtype=torch.bool); first[1:] = k_sorted[1:] != k_sorted[:-1]
    seg = torch.cumsum(first.long(), 0)-1
    head = order[first]                                            # front-most row of each pixel
    zf = zz[head][seg]; sf = sigma_z2.reshape(-1)[ids][head][seg]
    zi = zz[order]; si = sigma_z2.reshape(-1)[ids][order]
    behind = (zi-zf) > n_sigma*torch.sqrt(si+sf)
    r = torch.where(first, torch.full_like(zi, CODE['VALID'], dtype=torch.uint8),
                    torch.where(behind, torch.full_like(zi, CODE['SELF_OCCLUDED'], dtype=torch.uint8),
                                torch.full_like(zi, CODE['DUPLICATE_PIXEL_SUPPRESSED'], dtype=torch.uint8)))
    rr = reason.reshape(-1); rr[ids[order]] = r
    front.reshape(-1)[ids[order[first]]] = True
    if measured is not None:
        reason[:, ~measured] = CODE['NOT_PROCESSED']
    return reason, front


def position_sigma_z2(R, Sigma_x):
    """sigma_z^2 [C,N] = r3^T Sigma_x r3 for every camera's optical axis row r3 (R [C,3,3], Sigma_x [N,3,3])."""
    r3 = R[:, 2, :]
    return torch.einsum('ci,nij,cj->cn', r3.to(Sigma_x.dtype), Sigma_x, r3.to(Sigma_x.dtype)).clamp_min(0)


def apply_frozen_assignment(reason, frozen):
    """Pixel-reuse assignment fixed for the whole frame (--pixel-reuse-assignment per-frame): the SELF_OCCLUDED /
    DUPLICATE_PIXEL_SUPPRESSED decisions of the first call (the prior prediction) are kept for every IEKF iteration;
    geometric reasons of the current iterate (behind camera, outside the image) still apply. A point allowed at the
    prediction stays allowed even if an iterate moves it next to another; the same pixel is still never assigned to two
    points at the prediction (contract 2.2 item 5)."""
    if frozen is None: return reason
    sup = (frozen == CODE['SELF_OCCLUDED']) | (frozen == CODE['DUPLICATE_PIXEL_SUPPRESSED'])
    now = (reason == CODE['SELF_OCCLUDED']) | (reason == CODE['DUPLICATE_PIXEL_SUPPRESSED'])
    return torch.where(sup, frozen, torch.where(now, torch.zeros_like(reason), reason))
