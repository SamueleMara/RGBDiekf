"""Material-following temporal RGB-D observation (opt-in `--temporal-mode material-following`).

PROPOSED CHANGE OF TEMPORAL OBSERVATION DOMAIN. It is NOT algebraically equivalent to the Eulerian fixed-current-pixel
warp of temporal_rgbd_terms.build_temporal_rows (which attaches a row to the current integer owner pixel p and samples
the REFERENCE image at p - pi(x_t) + pi(x_ref)). Here the row is attached to a material identity
(point i, camera c, reference frame tau, measured integer source pixel p_ref) and the CURRENT image is sampled where
that material has moved:

    u_t = pi_c(x_t),  u_ref = pi_c(x_ref),  q_t = p_ref + u_t - u_ref,  dZ = Z_c(x_t) - Z_c(x_ref)
    f(a) = sum_a w_a(q_t) (y_{t,a} - eta_{t,a}) - (y_ref - eta_ref) - e4 dZ ,      0 = f(a) + eps_warp

with the four current bilinear raw pixels y_{t,a} (fresh sensor errors eta_{t,a}) and the ONE measured reference raw
pixel y_ref = y_tau(p_ref) (its error eta_ref is the same local variable that the point used when it observed that pixel,
e.g. the frame-0 source raw error, so the source observation is never counted twice). Jacobians, with
G = d/dq of the clean current bilinear interpolant (4x2) and velocity columns ZERO:

    df/dx_t   =  G J_pi(x_t)   - e4 J_Z(x_t)          df/deta_{t,a} = -w_a I4
    df/dx_ref = -G J_pi(x_ref) + e4 J_Z(x_ref)        df/deta_ref   = +I4

p_ref is the fixed measured pixel (no state derivative); the anchor convention is not mixed with a subpixel anchor.

Validity is decided at every candidate state, never frozen:
  * the selected point may change its CURRENT target pixel; first-hit ownership is recomputed with all tracked points;
  * the current 2x2 support at q_t must be inside the image, inside the hard mask, finite positive depth and one surface
    (validate_reference_patch, configs/projective_rgbd.json same-surface rule);
  * HIDDEN (prediction-only for that camera) when a different projected point at the point's own nearest pixel is in front
    by more than the same documented single-surface allowance (SELF_OCCLUDED), or the measured depth there, or the
    bilinear support depth at q_t, is in front of the predicted depth (Z_t, resp. Z_t + D_ref(p_ref) - Z_ref) by more than
    that allowance (MEASURED_SURFACE_IN_FRONT; occlusion-explicable sign only); a same-surface collision is an
    association event, not occlusion;
  * the point's own nearest pixel must be foreground (never a background RGB-D update).
"""
from dataclasses import dataclass
import math
import torch
from estimator.temporal_rgbd_terms import validate_reference_patch, bilinear, TEMPORAL_REASONS
from estimator.temporal_zbuffer import owners_at_gpu
from estimator.temporal_local_covariance import STATE, CHART, KEY, usable_references

MATERIAL_REASONS = {**TEMPORAL_REASONS, 13: 'REFERENCE_NOT_SAME_SURFACE', 14: 'INTERPOLATION_SUPPORT_INCOMPLETE'}
HIDDEN_CODES = (6, 8)                  # losses allowed during a step: genuine occlusion of the material
INVALIDATING_CODES = (1, 2, 3, 4, 5, 14)   # losses that reject a step: identity/support validity destroyed
OBSERVATION_DOMAIN = ('PROPOSED_CHANGE_OF_TEMPORAL_OBSERVATION_DOMAIN: material-following (fixed reference raw pixel, '
                      'moving current bilinear sample); not equivalent to the fixed-current-pixel Eulerian warp')


def single_surface_allowance(z, cam, rig, noise, policy):
    """Documented single-surface depth allowance of configs/projective_rgbd.json (same formula as validate_reference_patch)."""
    f = rig.K[cam, :2].min(-1).values
    return (policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(z).clamp_min(0).sqrt()
            + z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/f)


@dataclass
class MaterialRows:
    point: torch.Tensor
    camera: torch.Tensor
    reference_frame: torch.Tensor
    reference_pixel: torch.Tensor      # (M,) measured integer source pixel at tau
    reference_state: torch.Tensor      # (M,6)
    reference_values: torch.Tensor     # (M,4) raw RGB-D at (tau, p_ref)
    current_pixels: torch.Tensor       # (M,4) flat raw pixels of the current 2x2 support (00,10,01,11)
    current_values: torch.Tensor       # (M,4,4)
    cell: torch.Tensor                 # (M,2) floor(q_t)
    q: torch.Tensor                    # (M,2)
    u_t: torch.Tensor
    u_ref: torch.Tensor
    z_t: torch.Tensor
    z_ref: torch.Tensor
    J_t: torch.Tensor                  # (M,2,3) d pi / d x_t
    Jz_t: torch.Tensor                 # (M,3)
    J_ref: torch.Tensor
    Jz_ref: torch.Tensor
    target_pixel: torch.Tensor         # (M,) the point's nearest current pixel (may change between iterates)
    h: torch.Tensor = None             # predicted current sample: clean reference + e4 dZ (y - h = profiled residual)
    f: torch.Tensor = None             # material residual function f(a^j) (current errors at the iterate)
    H_t: torch.Tensor = None
    H_ref: torch.Tensor = None
    weights: torch.Tensor = None
    y: torch.Tensor = None             # raw current bilinear sample sum_a w_a y_{t,a}

    def __len__(self):
        return len(self.point)

    # names used by shared reporting code
    @property
    def pixel(self):
        return self.target_pixel

    @property
    def reference_pixels(self):
        return self.reference_pixel[:, None]

    @property
    def values(self):
        return self.reference_values

    @property
    def pixel_shift(self):
        return self.q-torch.stack((self.reference_pixel % self.width, self.reference_pixel//self.width), -1).to(self.q.dtype)


def material_prediction(rows, clean_current, clean_reference):
    """f (M,4), df/dx_t (M,4,6), df/dx_ref (M,4,6), weights (M,4) at clean current/reference values."""
    h, grad, w, _ = bilinear(rows.q, clean_current, rows.cell)
    f = h-clean_reference
    f[:, 3] -= rows.z_t-rows.z_ref
    Ht = torch.zeros((len(rows), 4, 6), dtype=f.dtype, device=f.device)
    Hr = torch.zeros_like(Ht)
    Ht[:, :, :3] = grad@rows.J_t
    Hr[:, :, :3] = -grad@rows.J_ref
    Ht[:, 3, :3] -= rows.Jz_t
    Hr[:, 3, :3] += rows.Jz_ref
    return f, Ht, Hr, w


def resampling_discrepancy_variance(frame, cam, q, cell):
    """OPTIONAL data-derived bilinear resampling discrepancy of the CURRENT sample (separate from R_sensor and R_warp).
    Taylor remainder of bilinear interpolation, e = 1/2 [du(1-du) F_uu + dv(1-dv) F_vv] (the uv cross term is exact),
    with E[F_uu^2], E[F_vv^2] the mean squared 3-point second differences of the measured image on the two support rows
    (columns) that lie entirely on valid foreground raw pixels. var = 1/4 [(du(1-du))^2 E F_uu^2 + (dv(1-dv))^2 E F_vv^2]
    per channel: exactly 0 at integer q (a static sample), no tuned constant. When one axis has no complete second
    difference the other axis' estimate is used; with none at all the sample is INTERPOLATION_SUPPORT_INCOMPLETE."""
    H, W = frame.H, frame.W
    du, dv = (q-cell.to(q.dtype)).unbind(-1)
    x0, y0 = cell[:, 0], cell[:, 1]
    F = lambda y, x: torch.cat((frame.rgb[cam, y.clamp(0, H-1), x.clamp(0, W-1)], frame.depth[cam, y.clamp(0, H-1), x.clamp(0, W-1), None]), -1).to(q.dtype)
    ok = lambda y, x: (y >= 0) & (y < H) & (x >= 0) & (x < W) & frame.mask[cam, y.clamp(0, H-1), x.clamp(0, W-1)] & frame.raw_valid[cam, y.clamp(0, H-1), x.clamp(0, W-1)]

    def second(axis):
        acc = torch.zeros((len(q), 4), dtype=q.dtype, device=q.device)
        n = torch.zeros(len(q), dtype=q.dtype, device=q.device)
        for a in (0, 1):            # the two support rows (axis u) or columns (axis v)
            for b in (0, 1):        # second difference centred on each support pixel
                if axis == 0:
                    pts = [(y0+a, x0+b-1), (y0+a, x0+b), (y0+a, x0+b+1)]
                else:
                    pts = [(y0+b-1, x0+a), (y0+b, x0+a), (y0+b+1, x0+a)]
                good = ok(*pts[0]) & ok(*pts[1]) & ok(*pts[2])
                d2 = F(*pts[0])-2*F(*pts[1])+F(*pts[2])
                acc += torch.where(good[:, None], d2*d2, torch.zeros_like(d2))
                n += good.to(q.dtype)
        return acc, n
    su, nu = second(0)
    sv, nv = second(1)
    Euu = torch.where((nu > 0)[:, None], su/nu.clamp_min(1)[:, None], sv/nv.clamp_min(1)[:, None])
    Evv = torch.where((nv > 0)[:, None], sv/nv.clamp_min(1)[:, None], su/nu.clamp_min(1)[:, None])
    var = .25*((du*(1-du))**2)[:, None]*Euu+.25*((dv*(1-dv))**2)[:, None]*Evv
    return var, (nu > 0) | (nv > 0)


def build_material_rows(s, owners, frame, references, frames, rig, noise, policy, ordinal, lag, resampling='none'):
    """All usable (camera, point) material identities evaluated at the candidate state s (N,6)."""
    C, N = references.frame.shape
    dev, W = s.device, frame.W
    reason = owners.reason.clone()
    age = ordinal-references.ordinal
    has = references.frame >= 0
    ready = has & (age < lag) & (age > 0)
    reason[~has] = 11
    reason[has & (age >= lag)] = 12
    lifecycle = {'bootstrap': int((~has & (owners.reason == 0)).sum()), 'reference_outside_lag': int((has & (age >= lag)).sum())}
    c, p = torch.where(ready)
    rs = references.state[c, p]
    ut, zt, Jt, Jzt = rig.project_rows(s[p, :3], c)
    ur, zr, Jr, Jzr = rig.project_rows(rs[:, :3], c)
    own_reason = owners.reason[c, p]
    tp = owners.pixel[c, p]
    inside = (own_reason != 1) & (own_reason != 2)
    tpc = tp.clamp(0, frame.H*W-1)
    ty, tx = tpc//W, tpc % W
    fg = frame.mask[c, ty, tx] & frame.raw_valid[c, ty, tx]
    allow = single_surface_allowance(zt, c, rig, noise, policy)
    winner = owners_at_gpu(owners, c, tpc)
    zw = torch.where(winner >= 0, owners.z[c, winner.clamp_min(0)], zt)
    self_hidden = inside & (own_reason == 6) & (zt-zw > allow)
    D = frame.depth[c, ty, tx].to(s.dtype)
    measured_front = inside & fg & torch.isfinite(D) & (zt-D > allow)
    code = torch.zeros(len(p), dtype=torch.uint8, device=dev)
    code[~fg] = 3
    code[measured_front] = 8
    code[self_hidden] = 6
    code[~inside] = own_reason[~inside]
    rpix = references.pixel[c, p]
    q = torch.stack((rpix % W, rpix//W), -1).to(s.dtype)+ut-ur
    pix, vals, sup, base = validate_reference_patch(frame, c, q, rig, noise, policy)
    code[(code == 0) & (sup != 0)] = sup[(code == 0) & (sup != 0)]
    # first-depth visibility AT THE MATERIAL SAMPLE: the current support depth must not be in front of the material's
    # predicted depth Z_t + (D_ref(p_ref) - Z_ref) by more than the single-surface allowance (occlusion-explicable sign
    # only; a measured surface behind the prediction is the motion signal and is kept)
    rf_all = references.frame[c, p]
    dref = torch.zeros(len(p), dtype=s.dtype, device=dev)
    for k in rf_all.unique().tolist():
        ix = torch.where(rf_all == k)[0]
        dref[ix] = frames[int(k)].depth[c[ix], rpix[ix]//W, rpix[ix] % W].to(s.dtype)
    wq = bilinear(q, vals, base)[2]
    dq = (wq*vals[:, :, 3]).sum(1)
    sample_front = (code == 0) & (dq < zt+(dref-zr)-allow)
    code[sample_front] = 8
    rvar = torch.zeros((len(p), 4), dtype=s.dtype, device=dev)
    if resampling == 'second-difference':
        rvar, avail = resampling_discrepancy_variance(frame, c, q, base)
        code[(code == 0) & ~avail] = 14
    elif resampling != 'none':
        raise ValueError(f'unknown resampling discrepancy {resampling!r}')
    lifecycle.update({'usable_identities': len(p), 'hidden_self_occluded': int((code == 6).sum()),
                      'hidden_measured_surface_in_front': int((code == 8).sum()), 'hidden_at_material_sample': int(sample_front.sum()), 'target_background': int((code == 3).sum()),
                      'outside_or_behind': int(((code == 1) | (code == 2)).sum()),
                      'support_invalid': int(((code == 4) | (code == 5) | ((code == 3) & fg)).sum()),
                      'same_surface_collision_kept': int((inside & (own_reason == 6) & ~self_hidden & (code == 0)).sum())})
    reason[c, p] = code
    # identities whose visibility (not their current support) is valid: used by the coarse pyramid levels
    visible = inside & fg & ~self_hidden & ~measured_front
    ic, ip = c, p
    ok = code == 0
    lifecycle['rows'] = int(ok.sum())
    c, p, rs, ut, zt, Jt, Jzt, ur, zr, Jr, Jzr, tp, rpix, q, pix, vals, base, rvar = (x[ok] for x in
        (c, p, rs, ut, zt, Jt, Jzt, ur, zr, Jr, Jzr, tp, rpix, q, pix, vals, base, rvar))
    rf = references.frame[c, p]
    rv = torch.zeros((len(p), 4), dtype=s.dtype, device=dev)
    for k in rf.unique().tolist():
        ix = torch.where(rf == k)[0]
        F = frames[int(k)]
        yy, xx = rpix[ix]//W, rpix[ix] % W
        good = F.mask[c[ix], yy, xx] & F.raw_valid[c[ix], yy, xx]
        if not bool(good.all()):
            raise ValueError('MATERIAL_REFERENCE_PIXEL_NOT_VALID_FOREGROUND: a committed reference pixel must be a valid raw foreground pixel')
        rv[ix] = torch.cat((F.rgb[c[ix], yy, xx], F.depth[c[ix], yy, xx, None]), -1).to(s.dtype)
    rows = MaterialRows(p, c, rf, rpix, rs, rv, pix, vals, base, q, ut, ur, zt, zr, Jt, Jzt, Jr, Jzr, tp)
    rows.width = W
    rows.resampling_variance = rvar
    rows.identities = (ic, ip, visible)
    return rows, reason, lifecycle


def commit_material_references(references, owners, frame, state, ordinal, allowed, rig, noise, policy):
    """New same-camera reference (frame t, nearest integer pixel, state) where the point is the first-hit winner on a
    valid foreground raw pixel whose measured depth shows the point's own surface (|D - Z| <= single-surface allowance),
    and only for `allowed` points (accepted update, or no temporal row this frame). Elsewhere the previous reference is
    preserved (occlusion / rejected update) until it leaves the lag window."""
    c, p, pix = owners.selected_camera, owners.selected_point, owners.selected_pixel
    if c is None:
        cc, yy, xx = torch.where(owners.owner_id >= 0)
        c, p, pix = cc, owners.owner_id[cc, yy, xx], yy*frame.W+xx
    y, x = pix//frame.W, pix % frame.W
    ok = frame.mask[c, y, x] & frame.raw_valid[c, y, x] & torch.isfinite(frame.rgb[c, y, x]).all(1) & torch.isfinite(frame.depth[c, y, x]) & (frame.depth[c, y, x] > 0)
    z = owners.z[c, p]
    same = (frame.depth[c, y, x].to(z.dtype)-z).abs() <= single_surface_allowance(z, c, rig, noise, policy)
    ok &= same & allowed[p]
    c, p, pix = c[ok], p[ok], pix[ok]
    references.frame[c, p] = frame.k
    references.ordinal[c, p] = ordinal
    references.pixel[c, p] = pix
    references.state[c, p] = state[p]
    return {'committed': int(len(p)), 'winner_not_same_surface': int((~same).sum())}


def material_retention_mask(layout, refs, chart, current_frame, ordinal, lag):
    """Keep: current state; every state/chart a camera may still reference; for each camera ONLY the raw key of its
    usable reference pixel (frame, p_ref) -- the only raw key a material row can reuse; the source error while its
    chart is kept. Other current-support keys can never be referenced again (a reference frame has one fixed pixel)."""
    use = usable_references(refs, ordinal, lag)
    p, kind, cam, frame = layout.point, layout.kind, layout.camera, layout.frame
    referenced = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    for c in range(refs.frame.shape[0]):
        referenced |= (refs.frame[c, p] == frame) & use[c, p]
    keep = torch.zeros_like(referenced)
    keep |= (kind == STATE) & ((frame == int(current_frame)) | referenced)
    keep |= (kind == CHART) & referenced
    own = (kind == KEY) & (refs.frame[cam, p] == frame) & (refs.pixel[cam, p] == layout.pixel) & use[cam, p]
    chart_kept = torch.zeros(layout.N, dtype=torch.bool, device=p.device)
    chart_kept[p[(kind == CHART) & keep]] = True
    source = (kind == KEY) & (cam == chart.camera[p]) & (frame == chart.anchor_frame) & (layout.pixel == chart.pixel[p])
    keep |= own | (source & chart_kept[p])
    return keep


def material_record_expiry(layout, keep, refs, expired_upto, frame_ordinal):
    """Forbidden-recreation ledger per (camera, point): the highest processing ordinal of an expired key that a future
    material row could have needed, i.e. a key at or before the camera's current reference frame which is the reference
    pixel itself or belongs to an older frame. Expired support keys at other pixels of the reference frame, or of a
    frame that never became a reference, can never be needed again and do not block a fresh, never-used raw pixel."""
    gone = (~keep) & (layout.kind == KEY)
    if bool(gone.any()):
        c, p = layout.camera[gone], layout.point[gone]
        rf = refs.frame[c, p]
        fo = frame_ordinal[layout.frame[gone]]
        rfo = torch.where(rf >= 0, frame_ordinal[rf.clamp_min(0)], torch.full_like(rf, -1))
        needed = (fo < rfo) | ((fo == rfo) & (layout.pixel[gone] == refs.pixel[c, p]))
        if bool(needed.any()):
            flat = expired_upto.view(-1)
            flat.scatter_reduce_(0, c[needed]*expired_upto.shape[1]+p[needed], fo[needed], reduce='amax')
    return int(gone.sum())
