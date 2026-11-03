"""Image-coordinate RGB / depth derivatives at subpixel positions, with support validity and raw-pixel provenance.

Spatial derivatives are the EXACT derivatives of the bilinear interpolant of the measured image (projective_sensor.sample:
2x2 support, hard mask, valid finite positive depth, documented single-surface rule; nothing is computed across a depth
break or a mask edge). Rows are pixel coordinates (u = column, v = row), d/du first.

Two temporal differences are provided and kept distinct:
  * eulerian_difference      I_t(u) - I_r(u) at a FIXED image location (contains a different material after motion;
                             only meaningful where the same material is verified at both times, reported separately);
  * material_difference      I_t(q_t) - I_r(p_ref) along the material-following association (finite displacement).
Neither is a new likelihood: both reuse the raw pixels of the material-following observation.
"""
from dataclasses import dataclass
import torch
from estimator.projective_sensor import sample


@dataclass
class GradientSample:
    rgb: torch.Tensor          # (M,3)
    depth: torch.Tensor        # (M,)
    grad_rgb: torch.Tensor     # (M,3,2)  d rgb / d(u,v)
    grad_depth: torch.Tensor   # (M,2)
    var_rgb: torch.Tensor      # (M,3)    sum_a w_a^2 sigma^2 (independent raw pixels)
    var_depth: torch.Tensor    # (M,)
    reason: torch.Tensor       # (M,) uint8 projective_sensor.REASONS, 0 = valid
    raw_pixels: torch.Tensor   # (M,4) flat raw pixel ids of the support (00,10,01,11)

    @property
    def valid(self):
        return self.reason == 0


def gradients_at(frame, cam, uv, noise, policy, rig):
    s = sample(frame, cam, uv, noise, policy, rig.K, torch.float64)
    u0, v0 = s.pixel00[:, 0], s.pixel00[:, 1]
    W = frame.W
    raw = torch.stack((v0*W+u0, v0*W+u0+1, (v0+1)*W+u0, (v0+1)*W+u0+1), -1)
    return GradientSample(s.rgb, s.depth, s.grad_rgb, s.grad_depth, s.var_rgb, s.var_depth, s.reason, raw)


def eulerian_difference(frame_r, frame_t, cam, uv, noise, policy, rig):
    """I_t(u) - I_r(u), D_t(u) - D_r(u) at one fixed image location; valid only where both supports are valid."""
    a = gradients_at(frame_r, cam, uv, noise, policy, rig)
    b = gradients_at(frame_t, cam, uv, noise, policy, rig)
    ok = a.valid & b.valid
    d = torch.cat((b.rgb-a.rgb, (b.depth-a.depth)[:, None]), -1)
    return torch.where(ok[:, None], d, torch.zeros_like(d)), ok, b


def material_difference(frame_t, cam, q, ref_values, noise, policy, rig):
    """I_t(q_t) - y_ref(p_ref) (finite-displacement material association); depth difference excludes dZ."""
    b = gradients_at(frame_t, cam, q, noise, policy, rig)
    d = torch.cat((b.rgb, b.depth[:, None]), -1)-ref_values
    return torch.where(b.valid[:, None], d, torch.zeros_like(d)), b.valid, b
