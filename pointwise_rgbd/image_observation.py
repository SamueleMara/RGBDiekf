"""Image-space RGB-D measurement terms of one frame (RigEuler3DGS 481b2cf2 observation_terms.camera_terms,
adapted to the coloured-point rasterizer).

Observation domain Omega_c: observed foreground pixels with valid metric depth (mask AND raw validity).
It does not depend on the state; background RGB/depth is never read.

Per pixel p in Omega_c, with predicted rgb, depth and coverage A_p = 1 - exp(-W_p) from the point render:

  RGB, depth   reference robust channel per ELEMENT, z = r / sigma, rho(z) = (nu+1)/2 log1p(z^2/nu), nu = 5,
               TEMPERED by the predicted support:  A_p * rho(z)   (likelihood t(y; h, sigma)^A_p)
  coverage     explicit foreground-coverage observation (contract section 3): observed mask M_p = A_p(s) + n,
               n ~ N(0, 1/kappa); on M_p = 1 pixels  1/2 kappa (1 - A_p)^2,  kappa = tr(W_rgb)/12
               (the reference masked-mode kappa). Negative-mask pixels (M = 0) carry 1/2 kappa A_p^2 only
               under an explicitly declared object-only acquisition premise.

Why A_p * rho and not a hard gate. A prediction exists only where W_p > 0; the reference gates depth at
alpha > 0.05 (its RGB is alpha-composited and never gated). Applied to both channels of a point render,
a hard gate makes the FIXED objective discontinuous: a pixel crossing it adds rho(z) at once
(measured: +6.4 RGB, +7.4 depth for a step of 1e-6 m), and the trust region, whose gain ratio assumes a
model that agrees with the objective as the step shrinks, then rejects every step and collapses the
radius with the moving plate half-way (tests/test_image_iekf.py documents the case; the hard gate is kept
as an ABLATION, existence='hard_gate'). Tempering by A_p makes the RGB-D evidence of a pixel proportional
to the fraction of its support predicted by the cloud: continuous, zero exactly where no prediction exists
(nothing is synthesized there) and equal to the reference term where the support is complete. The
coverage derivative it implies is that of the explicit coverage model above.

Gauss-Newton system frozen per linearization (reference IRLS): residual adjoint = the exact negative
gradient of the objective in the predicted values (coverage channel: kappa (1 - A) - rho_rgb - rho_d);
weights A w(z)/sigma^2 (RGB, depth) and kappa (coverage). For the Joseph covariance the channel noise
variances are the Student-t variance nu/(nu-2) sigma^2 and 1/kappa.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ObservationPolicy:
    nu: float = 5.
    coverage_gate: float = .05            # reporting threshold (and the ablation's hard gate), reference min_alpha_for_depth
    negative_mask: bool = False
    existence: str = 'coverage_weight'    # 'hard_gate': ablation only

    def manifest(self):
        return {'domain': 'observed mask AND valid depth; state independent; background never read',
                'rgb_depth': (f'reference Student-t per element, nu={self.nu}, tempered by predicted coverage A_p' if self.existence == 'coverage_weight'
                              else f'reference Student-t per element, nu={self.nu}, hard gate A_p >= {self.coverage_gate} (ABLATION)'),
                'coverage': 'mask observation M = A(s) + n, n ~ N(0, 1/kappa), kappa = tr(W_rgb)/12 (reference masked-mode kappa), on M = 1 pixels'
                            + ('; M = 0 pixels too (declared object-only premise)' if self.negative_mask else ''),
                'irls': 'Student-t weights frozen per linearization (reference camera_terms); exact gradient',
                'joseph_noise': 'Student-t variance nu/(nu-2) sigma^2; coverage 1/kappa'}


def student_t(r, sigma, nu):
    """Per-element objective and IRLS precision."""
    z2 = (r/sigma)**2
    return .5*(nu+1)*torch.log1p(z2/nu), (nu+1)/(nu+z2)/sigma**2


class FrameObservation:
    def __init__(self, frames, config, policy, device):
        self.frames, self.policy = frames, policy
        t = lambda a, dtype=torch.float64: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        rgb_cov = np.asarray(config.color_covariance, float)
        if not np.allclose(rgb_cov, np.diag(np.diag(rgb_cov))): raise ValueError('Per-element Student-t needs a diagonal RGB sensor covariance')
        self.rgb_sigma = t(np.sqrt(np.diag(rgb_cov)))
        self.kappa = float(np.sum(1/np.diag(rgb_cov))/12)
        self.cameras = []
        for f in frames:
            dom = np.flatnonzero(np.asarray(f.valid).ravel())
            z = np.asarray(f.depth, float).ravel()[dom]
            cam = {'domain': t(dom, torch.long), 'rgb': t(np.asarray(f.rgb, float).reshape(-1, 3)[dom]), 'depth': t(z),
                   'depth_sigma': t(np.sqrt(config.depth_variance+config.depth_squared_variance*z*z)), 'npx': f.camera.width*f.camera.height}
            if policy.negative_mask:
                if f.object_only_no_external_occluders is not True:
                    raise ValueError('Negative-mask coverage evidence requires the declared object-only acquisition premise')
                cam['negative'] = t(np.flatnonzero(~np.asarray(f.mask).ravel()), torch.long)
            self.cameras.append(cam)

    def noise_variance(self):
        """Per camera [H*W,5] channel noise variances (RGB, depth on the domain; coverage on its pixels) for K R K^T."""
        f = self.policy.nu/(self.policy.nu-2)
        out = []
        for cam in self.cameras:
            v = torch.zeros((cam['npx'], 5), dtype=self.rgb_sigma.dtype, device=self.rgb_sigma.device)
            v[cam['domain'], :3] = f*self.rgb_sigma**2
            v[cam['domain'], 3] = f*cam['depth_sigma']**2
            v[cam['domain'], 4] = 1/self.kappa
            if 'negative' in cam: v[cam['negative'], 4] = 1/self.kappa
            out.append(v)
        return out

    def terms(self, lin):
        """Energy and frozen IRLS system at one linearization (same domain for every trial)."""
        nu, gate, kappa = self.policy.nu, self.policy.coverage_gate, self.kappa
        out = {'energy': 0., 'cameras': [], 'residual_adjoint': [], 'weights': []}
        for cam, view in zip(self.cameras, lin.views):
            dom = cam['domain']; A = view.coverage[dom]
            pred = A >= gate
            r_rgb = cam['rgb']-view.rgb[dom]; r_d = cam['depth']-view.depth[dom]
            e_rgb, w_rgb = student_t(r_rgb, self.rgb_sigma, nu)
            e_d, w_d = student_t(r_d, cam['depth_sigma'], nu)
            if self.policy.existence == 'coverage_weight':
                a = A
                cov_adj = kappa*(1-A)-e_rgb.sum(1)-e_d          # -dE/dA of A rho + 1/2 kappa (1-A)^2
            elif self.policy.existence == 'hard_gate':
                a = pred.to(A.dtype); cov_adj = kappa*(1-A)
            else:
                raise ValueError(self.policy.existence)
            e_rgb, w_rgb, e_d, w_d = a[:, None]*e_rgb, a[:, None]*w_rgb, a*e_d, a*w_d
            e_cov = .5*kappa*(1-A)**2
            energy = float(e_rgb.sum()+e_d.sum()+e_cov.sum())
            npx, dev, dt = cam['npx'], A.device, A.dtype
            rgb_p = torch.zeros((npx, 3), dtype=dt, device=dev); rgb_p[dom] = w_rgb
            dprec = torch.zeros(npx, dtype=dt, device=dev); dprec[dom] = w_d
            cprec = torch.zeros(npx, dtype=dt, device=dev); cprec[dom] = kappa
            adj = torch.zeros((npx, 5), dtype=dt, device=dev)
            adj[dom, :3] = w_rgb*r_rgb; adj[dom, 3] = w_d*r_d; adj[dom, 4] = cov_adj
            neg = 0.
            if 'negative' in cam:
                An = view.coverage[cam['negative']]
                neg = float((.5*kappa*An**2).sum()); energy += neg
                cprec[cam['negative']] = kappa; adj[cam['negative'], 4] = -kappa*An
            out['energy'] += energy
            out['residual_adjoint'].append(adj)
            out['weights'].append((torch.ones(npx, dtype=dt, device=dev), rgb_p, dprec, cprec))
            n = int(pred.sum())
            out['cameras'].append({'camera': view.camera.name, 'domain_pixels': int(len(dom)), 'predicted_domain_pixels': n,
                'unexplained_domain_pixels': int(len(dom))-n, 'predicted_outside_domain_pixels': int((view.coverage >= gate).sum())-n,
                'energy': energy, 'energy_rgb': float(e_rgb.sum()), 'energy_depth': float(e_d.sum()), 'energy_coverage': float(e_cov.sum()),
                'energy_negative_mask': neg, 'mean_domain_coverage': float(A.mean()) if len(dom) else None,
                'mean_rgb_irls_weight': float((w_rgb[pred]/a[pred, None]*self.rgb_sigma**2).mean()) if n else None,
                'rgb_rmse': float(torch.sqrt((r_rgb[pred]**2).sum(1).mean()/3)) if n else None,
                'depth_rmse_m': float(torch.sqrt((r_d[pred]**2).mean())) if n else None,
                'depth_mae_m': float(r_d[pred].abs().mean()) if n else None, **view.stats})
        return out

    def pixel_energy(self, ci, pix, rgb, depth, A):
        """Objective contribution of arbitrary (pixel, prediction) rows of camera ci, identical to terms():
        A rho_rgb + A rho_d + 1/2 kappa (1 - A)^2 on domain pixels (hard gate in the ablation), negative-mask term
        where declared, zero elsewhere."""
        cam = self.cameras[ci]
        if 'lookup' not in cam:
            cam['lookup'] = torch.full((cam['npx'],), -1, dtype=torch.long, device=cam['domain'].device)
            cam['lookup'][cam['domain']] = torch.arange(len(cam['domain']), device=cam['domain'].device)
            if 'negative' in cam:
                cam['is_negative'] = torch.zeros(cam['npx'], dtype=torch.bool, device=cam['domain'].device); cam['is_negative'][cam['negative']] = True
        d = cam['lookup'][pix]; dom = d >= 0; d = d.clamp_min(0)
        nu, kappa = self.policy.nu, self.kappa
        e_rgb = student_t(cam['rgb'][d]-rgb, self.rgb_sigma, nu)[0].sum(1)
        e_d = student_t(cam['depth'][d]-depth, cam['depth_sigma'][d], nu)[0]
        a = A if self.policy.existence == 'coverage_weight' else (A >= self.policy.coverage_gate).to(A.dtype)
        e = torch.where(dom, a*(e_rgb+e_d)+.5*kappa*(1-A)**2, torch.zeros_like(A))
        if 'is_negative' in cam: e = e+torch.where(cam['is_negative'][pix], .5*kappa*A**2, torch.zeros_like(A))
        return e

    def apply_W(self, terms, hv):
        """W_p (H v)_p per camera on the full grid (zero off the active support)."""
        out = []
        for (scale, P, dprec, cprec), h in zip(terms['weights'], hv):
            o = torch.empty_like(h)
            o[:, :3] = scale[:, None]*(h[:, :3]@P.T if P.shape == (3, 3) else h[:, :3]*P)
            o[:, 3] = dprec*h[:, 3]; o[:, 4] = cprec*h[:, 4]
            out.append(o)
        return out


def downsample(frame, s):
    """s x s block average of the observed RGB-D over valid pixels (ABLATION ONLY: coarse-to-fine)."""
    from rgbd_common.dataloader import RGBDFrame
    H, W = frame.depth.shape; h, w = H//s, W//s
    v = np.asarray(frame.valid)[:h*s, :w*s].reshape(h, s, w, s).astype(float)
    n = v.sum((1, 3)); ok = n >= s*s/2
    rgb = (np.asarray(frame.rgb)[:h*s, :w*s].reshape(h, s, w, s, 3)*v[..., None]).sum((1, 3))/np.maximum(n, 1)[..., None]
    depth = (np.asarray(frame.depth)[:h*s, :w*s].reshape(h, s, w, s)*v).sum((1, 3))/np.maximum(n, 1)
    c = frame.camera; fx, fy, cx, cy = c.intrinsics
    cam = type(c)(c.name, w, h, np.array([fx/s, fy/s, (cx+.5)/s-.5, (cy+.5)/s-.5]), c.world_to_camera, c.near, c.far)
    return RGBDFrame(rgb, np.where(ok, depth, 0.), ok, cam, frame.k, frame.ci, None, frame.object_only_no_external_occluders)


def levels(frames, config, policy, device, factors=(1,)):
    """[(FrameObservation, [CameraTensors], footprint scale)], finest last. Production: the original images only."""
    from .rgbd_point_renderer import CameraTensors
    out = []
    for s in factors:
        fs = frames if s == 1 else [downsample(f, s) for f in frames]
        out.append((FrameObservation(fs, config, policy, device), [CameraTensors(f.camera, torch.device(device)) for f in fs], float(s)))
    if out[-1][2] != 1.: raise ValueError('The finest level must be the original images')
    return out


@torch.no_grad()
def estimate_model_error(lin, frames, beta=1., blur=1, rgb_floor=1e-4, depth_floor=1e-6):
    """Pinned RigEuler estimator/model_error.estimate_model_error, used here as a DIAGNOSTIC only (contract section 3:
    one sensor model in every estimation stage). sigma^2_model,p = max(floor, beta e_ref,p^2), 3x3 box blur, from the
    canonical catalogue rendered at its own endpoint frame."""
    out, stats = [], []
    for v, f in zip(lin.views, frames):
        H, W = f.depth.shape
        ok = torch.as_tensor(np.asarray(f.valid).ravel(), device=v.rgb.device) & (v.total > 0)
        e_rgb = torch.as_tensor(np.asarray(f.rgb, float).reshape(-1, 3), device=v.rgb.device)-v.rgb
        e_d = torch.as_tensor(np.asarray(f.depth, float).ravel(), device=v.rgb.device)-v.depth
        vr = torch.where(ok[:, None], (beta*e_rgb*e_rgb).clamp_min(rgb_floor), torch.full_like(e_rgb, rgb_floor))
        vd = torch.where(ok, (beta*e_d*e_d).clamp_min(depth_floor), torch.full_like(e_d, depth_floor))
        if blur > 0:
            k = 2*blur+1; ker = torch.ones((1, 1, k, k), dtype=vr.dtype, device=vr.device)/(k*k)
            box = lambda m: torch.nn.functional.conv2d(m[None, None], ker, padding=blur)[0, 0]
            vr = torch.stack([box(vr[:, c].reshape(H, W)).reshape(-1) for c in range(3)], 1); vd = box(vd.reshape(H, W)).reshape(-1)
        out.append({'rgb': vr.cpu().numpy(), 'depth': vd.cpu().numpy()})
        stats.append({'camera': v.camera.name, 'rgb_model_sigma_median_domain': float(vr[ok].sqrt().median()) if bool(ok.any()) else None,
                      'depth_model_sigma_median_domain_m': float(vd[ok].sqrt().median()) if bool(ok.any()) else None})
    return out, {'reference': 'canonical catalogue at its endpoint frame', 'beta': beta, 'blur': blur, 'per_camera': stats}
