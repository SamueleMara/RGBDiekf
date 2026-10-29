"""Rendered white-point pseudomask as an IEKF image measurement (RGBDiekf_Codex_Rendered_Binary_Mask_IEKF_Replacement.md,
section 3). Every transported point is rendered WHITE on BLACK in every camera (visible, occluded, with or without a
direct RGB-D row: no z-buffer, no SELF_OCCLUDED / DUPLICATE exclusion); the measured binary mask of the same camera and
frame is the observation:

    y_obs = vec(M_obs) (the actual mask, 1 = white foreground),   y_hat = vec(rendered white points),
    r = y_obs - y_hat,  ASSIMILATED ONLY where M_obs = 0 (one-sided: white where the object is not), so r = -y_hat there.

Hard binary pseudomask B (audit / debug, the user's literal image): the pixel whose centre is nearest to the projection
(u = column, v = row, centres at integers, torch.round = round half to even, as projective_visibility) is white; OR
over points (two points in one pixel = white once). A point behind the camera (Z <= near) or whose pixel is outside the
image renders nothing in that camera (no black padding is fabricated); the other cameras still render it.

Differentiable pseudomask m_hat (the IEKF linearization; the same white points, anti-aliased): each projection splats
on the four centres of its floor cell (u0 = floor u, du = u - u0; the bilinear weights of projective_sensor.sample):
    a_00 = (1-du)(1-dv), a_10 = du(1-dv), a_01 = (1-du)dv, a_11 = du dv          (a = max(0,1-|u-u_p|) max(0,1-|v-v_p|))
    m_hat[p] = 1 - prod_{i contributes to p} (1 - a[p,i])                        (union occupancy, in [0,1])
    d m_hat[p] / d x_i = prod_{j != i} (1 - a[p,j]) * d a[p,i]/d(u,v) * J_u,i     (1 x 3; J_u from projective_camera)
The exclusive product is evaluated without dividing by (1 - a) when a = 1: log of the factors < 1 summed per pixel and
the factors equal to 1 counted (a pixel with two exact-centre contributors has derivative 0 for both). Derivative
convention at the measure-zero grid lines (integer u or v): the floor cell's one-sided (right) derivative, the same
convention as the RGB-D sampler; finite-difference tests use non-grid-aligned points.
The anti-aliased m_hat and the hard B differ away from pixel centres (B is 0/1, m_hat is fractional); both are
reported, never claimed identical.

All reductions over contributors are deterministic segment sums over pairs sorted by (camera, pixel, point) or by
(point, camera, pixel); no atomics, no dense pixel x point matrix. Only measured-background pixels touched by a
footprint carry a row: every other selected pixel has m_hat = 0 and zero derivative, so omitting it is exact.
"""
from dataclasses import dataclass
import torch

CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass
class Footprints:
    """Point-camera-corner pairs of the differentiable splat (every pair whose pixel is inside the image)."""
    cam: torch.Tensor        # [Q] long
    pix: torch.Tensor        # [Q] long, v * W + u
    point: torch.Tensor      # [Q] long
    a: torch.Tensor          # [Q] coverage weight in [0,1]
    g_uv: torch.Tensor       # [Q,2] d a / d(u,v)
    Ju: torch.Tensor         # [Q,2,3] d(u,v)/dx
    projected: torch.Tensor  # [C,N] bool: in front of the camera (Z > near), finite
    in_image: torch.Tensor   # [C,N] bool: nearest pixel inside the image
    nearest: torch.Tensor    # [C,N] long: v * W + u of the nearest pixel centre (valid where in_image)
    pairs_total: int = 0     # footprint corners inside the image (before any background selection)
    straddle: torch.Tensor = None   # [N] bool: some camera's footprint has a background AND a foreground corner


def footprints(x, rig, H, W, background=None):
    """Project every point into every camera and build its 2x2 bilinear footprint (exact projection Jacobian).
    background [C,H,W] bool: keep only the corners on measured-background pixels (the only ones that can carry a row;
    ~2% of the corners on a dense surface), the total and the per-point boundary straddling still counted."""
    proj = rig.project(x, jacobian=True)
    C, N = proj.z.shape
    u, v, z = proj.uv[..., 0], proj.uv[..., 1], proj.z
    finite = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(z)
    front = finite & (z > rig.near[:, None])
    uu = torch.where(front, u, torch.zeros_like(u)); vv = torch.where(front, v, torch.zeros_like(v))
    pu, pv = torch.round(uu), torch.round(vv)
    in_img = front & (pu >= 0) & (pu <= W-1) & (pv >= 0) & (pv <= H-1)
    nearest = (pv.clamp(0, H-1)*W+pu.clamp(0, W-1)).long()
    u0, v0 = torch.floor(uu), torch.floor(vv)
    du, dv = uu-u0, vv-v0
    cams, pts, pixs, a, gs, Js = [], [], [], [], [], []
    cam_idx = torch.arange(C, device=x.device)[:, None].expand(C, N); pt_idx = torch.arange(N, device=x.device)[None].expand(C, N)
    total = 0; n_img = torch.zeros((C, N), dtype=torch.int32, device=x.device); n_bg = torch.zeros_like(n_img)
    for cx, cy in CORNERS:
        wu = du if cx else 1-du; wv = dv if cy else 1-dv
        gu = (1. if cx else -1.)*wv; gv = (1. if cy else -1.)*wu
        uc, vc = u0+cx, v0+cy
        ok = front & (uc >= 0) & (uc <= W-1) & (vc >= 0) & (vc <= H-1)
        total += int(ok.sum()); n_img += ok.int()
        pix_all = (vc.clamp(0, H-1)*W+uc.clamp(0, W-1)).long()
        if background is not None:
            ok = ok & background.reshape(C, -1).gather(1, pix_all)
            n_bg += ok.int()
        cams.append(cam_idx[ok]); pts.append(pt_idx[ok]); pixs.append(pix_all[ok])
        a.append((wu*wv)[ok]); gs.append(torch.stack((gu, gv), -1)[ok]); Js.append(proj.Ju[ok])
    straddle = ((n_bg > 0) & (n_bg < n_img)).any(0) if background is not None else None
    return Footprints(torch.cat(cams), torch.cat(pixs), torch.cat(pts), torch.cat(a), torch.cat(gs), torch.cat(Js), front, in_img, nearest, total, straddle)


def hard_pseudomask(fp, C, H, W):
    """B [C,H,W] bool: nearest-pixel-centre white points, OR over points."""
    B = torch.zeros(C*H*W, dtype=torch.bool, device=fp.nearest.device)
    c = torch.arange(C, device=fp.nearest.device)[:, None].expand_as(fp.nearest)
    B[(c*H*W+fp.nearest)[fp.in_image]] = True
    return B.reshape(C, H, W)


@dataclass
class MaskRows:
    """Measured-background pixels touched by a footprint, their contributors and the sparse Jacobian."""
    row_cam: torch.Tensor     # [Rn]
    row_pix: torch.Tensor     # [Rn]
    m_hat: torch.Tensor       # [Rn] anti-aliased union occupancy
    r: torch.Tensor           # [Rn] = y_obs - m_hat = -m_hat (y_obs = 0 on these rows)
    var: torch.Tensor         # [Rn] R_mask of the row
    contributors: torch.Tensor  # [Rn] number of contributing points
    pair_row: torch.Tensor    # [Qb] row of each background pair (pairs sorted by row, then point)
    pair_point: torch.Tensor  # [Qb]
    g: torch.Tensor           # [Qb,3] d m_hat[row] / d x_point
    a: torch.Tensor           # [Qb]
    row_len: torch.Tensor     # [Rn] pairs per row (segment lengths)
    by_point: torch.Tensor    # [Qb] permutation sorting pairs by point (stable: row order kept inside a point)
    point_ids: torch.Tensor   # [Np] points with at least one background pair (sorted)
    point_len: torch.Tensor   # [Np] pairs per point in by_point order

    @property
    def n_rows(self): return len(self.r)

    def jv(self, v):
        """(J v)[row] = sum_i J[row,i] . v_i  (v [N,3] global point indexing)."""
        if len(self.r) == 0: return torch.zeros(0, dtype=v.dtype, device=v.device)
        return torch.segment_reduce((self.g*v[self.pair_point]).sum(-1), 'sum', lengths=self.row_len)

    def jtw(self, w, N):
        """(J^T w)[i] = sum_rows J[row,i]^T w[row]  -> [N,3]."""
        out = torch.zeros((N, 3), dtype=w.dtype, device=w.device)
        if len(self.r) == 0: return out
        contrib = (self.g*w[self.pair_row][:, None])[self.by_point]
        out[self.point_ids] = torch.segment_reduce(contrib, 'sum', lengths=self.point_len, axis=0)
        return out

    def block_information(self, N, row_weight):
        """sum_rows J[row,i]^T w_row J[row,i] per point [N,3,3] (the point's own diagonal block)."""
        out = torch.zeros((N, 3, 3), dtype=self.g.dtype, device=self.g.device)
        if len(self.r) == 0: return out
        gg = (self.g[:, :, None]*self.g[:, None, :]*row_weight[self.pair_row][:, None, None])[self.by_point]
        out[self.point_ids] = torch.segment_reduce(gg, 'sum', lengths=self.point_len, axis=0)
        return out

    def objective(self):
        return 0.5*(self.r*self.r/self.var).sum()

    def restrict(self, idx, N):
        """The rows and pairs of the points idx (a union of whole components: rows are never split), with point indices
        relabelled 0..len(idx)-1: matrix-free products then cost O(pairs of idx), not O(N)."""
        loc = torch.full((N,), -1, dtype=torch.long, device=idx.device); loc[idx] = torch.arange(len(idx), device=idx.device)
        keep = loc[self.pair_point] >= 0
        rows_kept = torch.unique(self.pair_row[keep])
        newrow = torch.full((self.n_rows,), -1, dtype=torch.long, device=idx.device); newrow[rows_kept] = torch.arange(len(rows_kept), device=idx.device)
        pr, pp = newrow[self.pair_row[keep]], loc[self.pair_point[keep]]
        g, a = self.g[keep], self.a[keep]
        by_point = torch.argsort(pp, stable=True)
        pids, plen = torch.unique_consecutive(pp[by_point], return_counts=True)
        return MaskRows(self.row_cam[rows_kept], self.row_pix[rows_kept], self.m_hat[rows_kept], self.r[rows_kept], self.var[rows_kept],
                        self.contributors[rows_kept], pr, pp, g, a, torch.bincount(pr, minlength=len(rows_kept)), by_point, pids, plen)


def mask_rows(fp, background, var_per_camera, H, W, N):
    """Background rows of a frame: background [C,H,W] bool (= measured mask is 0), var_per_camera [C] R_mask."""
    C = background.shape[0]
    bg = background.reshape(-1)[fp.cam*H*W+fp.pix]
    cam, pix, pt, a, g_uv, Ju = fp.cam[bg], fp.pix[bg], fp.point[bg], fp.a[bg], fp.g_uv[bg], fp.Ju[bg]
    dev, dt = a.device, a.dtype
    if len(a) == 0:
        e = torch.zeros(0, dtype=dt, device=dev); el = torch.zeros(0, dtype=torch.long, device=dev)
        return MaskRows(el, el, e, e, e, el, el, el, torch.zeros((0, 3), dtype=dt, device=dev), e, el, el, el, el)
    key = cam*(H*W)+pix
    order = torch.argsort(pt, stable=True); order = order[torch.argsort(key[order], stable=True)]     # by row, then point
    key, cam, pix, pt, a, g_uv, Ju = key[order], cam[order], pix[order], pt[order], a[order], g_uv[order], Ju[order]
    first = torch.ones_like(key, dtype=torch.bool); first[1:] = key[1:] != key[:-1]
    row = torch.cumsum(first.long(), 0)-1
    row_len = torch.bincount(row)
    one = a >= 1
    log1ma = torch.where(one, torch.zeros_like(a), torch.log1p(-torch.where(one, torch.zeros_like(a), a)))
    logP = torch.segment_reduce(log1ma, 'sum', lengths=row_len)               # sum over the factors < 1
    nz = torch.segment_reduce(one.to(dt), 'sum', lengths=row_len)            # number of factors equal to 1 (a = 1)
    m_hat = torch.where(nz > 0, torch.ones_like(logP), -torch.expm1(logP))   # 1 - prod (1 - a)
    nzr, lPr = nz[row], logP[row]
    excl = torch.where(one, torch.where(nzr == 1, torch.exp(lPr), torch.zeros_like(lPr)),
                       torch.where(nzr == 0, torch.exp(lPr-log1ma), torch.zeros_like(lPr)))
    g = excl[:, None]*(g_uv[:, :, None]*Ju).sum(1)                          # [Qb,3]
    by_point = torch.argsort(pt, stable=True)
    pids, plen = torch.unique_consecutive(pt[by_point], return_counts=True)
    row_cam, row_pix = cam[first], pix[first]
    return MaskRows(row_cam, row_pix, m_hat, -m_hat, var_per_camera[row_cam], row_len, row, pt, g, a, row_len, by_point, pids, plen)


def frame_statistics(fp, rows, background, B, H, W):
    """Counts of section 9 for one linearization (no image is stored)."""
    C = background.shape[0]
    outside = B & background
    contributors = rows.contributors
    return {'points_projected_in_front': int(fp.projected.sum()), 'points_projected_in_image': int(fp.in_image.sum()),
            'per_camera_in_image': fp.in_image.sum(1).tolist(), 'contributor_pairs_total': int(fp.pairs_total or len(fp.a)),
            'background_pixels_touched': rows.n_rows, 'background_contributor_pairs': int(len(rows.pair_point)),
            'points_with_mask_rows': int(len(rows.point_ids)), 'overlapping_background_pixels': int((contributors > 1).sum()) if rows.n_rows else 0,
            'max_contributors_per_pixel': int(contributors.max()) if rows.n_rows else 0,
            'hard_white_pixels': int(B.sum()), 'hard_white_outside_measured_mask': int(outside.sum()),
            'hard_white_outside_per_camera': outside.reshape(C, -1).sum(1).tolist(),
            'mask_residual_norm': float(torch.linalg.vector_norm(rows.r)) if rows.n_rows else 0.,
            'mask_objective': float(rows.objective()) if rows.n_rows else 0.}
