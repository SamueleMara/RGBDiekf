#!/usr/bin/env python3
"""GT-ONLY OFFLINE measurement-information and likelihood-falsification study (never read by any estimator).

For label-verified catalogue points (USB / Blade; original source IDs, calibrated cameras, actual camera-local reference
pixels) and the true motion tau -> t, this compares material observations of ONE independent 3D point state x_t:

  rgbd-k     k x k patch around the measured reference pixel p_ref, every patch pixel sampled at p_ref + o + du(x_t)
             (du = pi(x_t) - pi(x_ref) of the point: a RIGID-IMAGE-PATCH APPROXIMATION, validated against GT below);
             k = 1 is the existing single-pixel material-following row
  depth-k    the depth channel only
  gainnorm-k RGB divided by its patch mean (gain-invariant; exact delta-method covariance, rank-1 null space projected)
             plus metric depth

Covariances (per channel): current samples share raw pixels, R_cur = W diag(sigma^2_raw) W^T with the bilinear weights
of every sample (exact for the shared raw errors); reference pixels are distinct raw pixels, diagonal; plus optionally
  interp  data-derived bilinear remainder variance of each current sample (second differences; 0 at integer positions)
  xcam    a constant per-channel discrepancy measured at ONE instant (frame tau) WITHOUT GT: bilinear samples of other
          cameras at the measured backprojection of each catalogue point (same-surface winners only), robust variance.
          Depth: pure interpolation/rasterization discrepancy; RGB also contains view-dependent shading.
Patch membership: reference pixel valid foreground on the centre's surface (documented single-surface allowance over
the offset distance); current sample needs a complete valid one-surface 2x2 support. Background is never sampled.

Studied quantities (GT used only to place states and to score): information rank / recoverable directions and their
geometry vs the source surface normal; exact-data linear oracle error; objective curves along the true motion and the
weak / middle information directions; local MAP drift when started at GT (prior = isotropic zero-motion prediction,
sigma_p = sigma_v dt from the run configuration); rigid-patch validity from GT first-hit material of every patch pixel;
silhouette validity of true states; noise decomposition; synthetic gain / offset perturbations of the current image.

    python -B tools/study_material_measurement_information.py --output-root "/media/.../RGBDekf/measurement_information_study"
"""
import argparse
import datetime
import json
import math
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_temporal_dynamic_correspondence import ground_truth, verified_labels, q_stats, DEFAULT_DATA
from utils.io import json_write, serial

REPO = Path(__file__).resolve().parents[1]
MODELS_INFO = [('rgbd', 1), ('rgbd', 3), ('rgbd', 5), ('rgbd', 7), ('rgbd', 9), ('depth', 1), ('depth', 5), ('gainnorm', 5), ('gainnorm', 9)]
MODELS_CURVE = [('rgbd', 1), ('rgbd', 3), ('rgbd', 5), ('rgbd', 9), ('depth', 1), ('gainnorm', 5)]
MODELS_PHOTO = [('rgbd', 1), ('rgbd', 5), ('rgbd', 9), ('depth', 1), ('gainnorm', 5), ('gainnorm', 9)]
NOISE = ['declared', 'interp', 'xcam']
ALPHA = np.round(np.linspace(-.5, 1.5, 21), 3)
BETA = np.round(np.linspace(-1., 1., 21), 3)


def robust_var(x):
    """(68.27% quantile of |x|)^2: equals sigma^2 for a zero-mean Gaussian; unlike the MAD it does not collapse to 0 when
    most residuals are exactly zero after quantization while a tail exists."""
    x = x[torch.isfinite(x)]
    return float(torch.quantile(x.abs(), .6827)**2) if len(x) else float('nan')


def allowance(z, f, noise, policy, span_px):
    return (policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(z).clamp_min(0).sqrt()
            + z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*span_px/f)


class Sampler:
    """Bilinear RGB-D samples of one frame with exact interpolant gradient and the documented validity rule."""
    def __init__(self, frame, rig, noise, policy):
        self.F, self.rig, self.noise, self.policy = frame, rig, noise, policy
        self.f = rig.K[:, :2].min(1).values

    def __call__(self, cam, uv):
        F = self.F
        H, W = F.H, F.W
        base = torch.floor(torch.nan_to_num(uv))
        inside = torch.isfinite(uv).all(-1) & (base[:, 0] >= 0) & (base[:, 1] >= 0) & (base[:, 0]+1 < W) & (base[:, 1]+1 < H)
        x0 = base[:, 0].long().clamp(0, W-2)
        y0 = base[:, 1].long().clamp(0, H-2)
        idx = [(y0, x0), (y0, x0+1), (y0+1, x0), (y0+1, x0+1)]
        V = torch.stack([torch.cat((F.rgb[cam, y, x], F.depth[cam, y, x, None]), -1) for y, x in idx], 1).double()
        ok = inside & torch.stack([F.mask[cam, y, x] & F.raw_valid[cam, y, x] for y, x in idx], 1).all(1) & torch.isfinite(V).all((1, 2))
        V = torch.where(ok[:, None, None], V, torch.ones_like(V))
        z = V[:, :, 3]
        ok &= (z > 0).all(1) & ((z.amax(1)-z.amin(1)) <= allowance(z.mean(1), self.f[cam], self.noise, self.policy, math.sqrt(2)))
        u, v = (uv-base).unbind(-1)
        w = torch.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), -1)
        dw = torch.stack((torch.stack((-(1-v), 1-v, -v, v), -1), torch.stack((-(1-u), -u, 1-u, u), -1)), -1)
        val = torch.einsum('sa,sad->sd', w, V)
        grad = torch.einsum('sab,sad->sdb', dw, V)
        s2 = torch.cat((self.noise.rgb_variance.double().expand(len(uv), 4, 3), self.noise.depth_sigma2(z)[..., None]), -1)
        return val, grad, w, ok, s2, base.long()


class Study:
    def __init__(self, scene, device, n_moving, n_static, seed):
        from scene.dataset import Dataset
        from scene.config import load_config
        from scene.endpoint import endpoint_samples
        from estimator.projective_camera import CameraRig
        from estimator.projective_sensor import MeasuredFrame, SensorNoise
        from estimator.temporal_noise import temporal_sensor_only
        self.scene, self.dev = scene, device
        self.policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
        self.data = Dataset(scene, 3)
        cfg, self.pol, self.calib = temporal_sensor_only(self.data, load_config(), self.policy)
        self.noise = SensorNoise.from_config(cfg, self.pol, device, torch.float64)
        self.rig = CameraRig(self.data.cameras, device, torch.float64)
        _, _, xyz, _, _, self.source = endpoint_samples(self.data, 'first', cfg)
        self.xyz = xyz
        X, nn_moving, _, self.evidence = ground_truth(scene, self.data, xyz)
        total = len(Dataset(scene).times)
        v, _, rep = verified_labels(scene, self.rig, self.noise, self.pol, xyz, nn_moving, device, sorted({total//2, total-1}))
        self.label_report = rep
        self.X = X
        self.verified = v
        rng = np.random.default_rng(seed)
        mv, st = np.flatnonzero(v == 1), np.flatnonzero(v == 0)
        self.sel = np.sort(np.r_[rng.choice(mv, min(n_moving, len(mv)), replace=False), rng.choice(st, min(n_static, len(st)), replace=False)])
        self.moving = torch.as_tensor(v[self.sel] == 1, device=device)
        self.frames = {k: MeasuredFrame(self.data, k, device, torch.float64) for k in range(3)}
        self.dt = float(self.data.times[1]-self.data.times[0])
        self.sigma_p = math.sqrt(float(np.asarray(cfg.velocity_covariance)[0, 0]))*self.dt
        self.normals = self._normals()

    def _normals(self):
        """Source-surface normal of each selected point from its own frame-0 source depth (data only)."""
        F, W = self.frames[0], self.frames[0].W
        src = torch.as_tensor(self.source[self.sel], device=self.dev)
        c, pix = src[:, 1], src[:, 2]
        u, v = pix % W, pix//W
        out = torch.full((len(c), 3), float('nan'), dtype=torch.float64, device=self.dev)
        ok = (u >= 1) & (v >= 1) & (u < W-1) & (v < F.H-1)

        def bp(du, dv):
            uu, vv = (u+du).clamp(0, W-1), (v+dv).clamp(0, F.H-1)
            z = F.depth[c, vv, uu].double()
            good = F.mask[c, vv, uu] & F.raw_valid[c, vv, uu]
            uv = torch.stack((uu, vv), -1).double()
            return self.rig.backproject(c, uv, z), good
        (xr, gr), (xl, gl), (xd, gd), (xu, gu) = bp(1, 0), bp(-1, 0), bp(0, 1), bp(0, -1)
        n = torch.cross(xr-xl, xd-xu, dim=1)
        good = ok & gr & gl & gd & gu & (n.norm(dim=1) > 0)
        out[good] = n[good]/n[good].norm(dim=1, keepdim=True)
        return out

    # ---------------------------------------------------------------- identities (true camera-local references)
    def identities(self, a, b):
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        dev, rig = self.dev, self.rig
        Fa, Fb = self.frames[a], self.frames[b]
        xa_all = torch.as_tensor(self.X[a], device=dev)
        xb_all = torch.as_tensor(self.X[b], device=dev)
        own_a = first_hit_pixel_owner(xa_all, rig, Fa)       # full cloud, all cameras: actual first-hit references
        self.owner_a = own_a
        S = torch.as_tensor(self.sel, device=dev)
        W = Fa.W
        sb = Sampler(Fb, rig, self.noise, self.pol)
        rec = {k: [] for k in ('c', 'p', 'pref', 'xref', 'xt')}
        sil = {'eligible': 0, 'true_pixel_background': 0, 'true_pixel_background_sample_support_valid': 0,
               'true_pixel_background_any_bilinear_neighbour_foreground': 0, 'distance_px': []}
        for c in range(rig.count):
            cc = torch.full((len(S),), c, device=dev)
            pa = own_a.pixel[c, S].clamp(0, Fa.H*W-1)
            ya, xa_ = pa//W, pa % W
            ua, za, _, _ = rig.project_rows(xa_all[S], cc)
            ub, zb, _, _ = rig.project_rows(xb_all[S], cc)
            f = float(self.rig.K[c, :2].min())
            elig = (own_a.reason[c, S] == 0) & Fa.mask[c, ya, xa_] & Fa.raw_valid[c, ya, xa_]
            elig &= (Fa.depth[c, ya, xa_].double()-za).abs() <= allowance(za, f, self.noise, self.pol, math.sqrt(2))
            pb = torch.round(ub).long()
            inb = (pb[:, 0] >= 0) & (pb[:, 1] >= 0) & (pb[:, 0] < W) & (pb[:, 1] < Fb.H)
            pbc = pb.clone()
            pbc[:, 0].clamp_(0, W-1)
            pbc[:, 1].clamp_(0, Fb.H-1)
            fgb = inb & Fb.mask[c, pbc[:, 1], pbc[:, 0]] & Fb.raw_valid[c, pbc[:, 1], pbc[:, 0]]
            Db = Fb.depth[c, pbc[:, 1], pbc[:, 0]].double()
            front = fgb & (zb-Db > allowance(zb, f, self.noise, self.pol, math.sqrt(2)))
            val, _, _, okC, _, _ = sb(cc, ub)
            offsurf = okC & ((val[:, 3]-zb).abs() > allowance(zb, f, self.noise, self.pol, math.sqrt(2)))
            pref = torch.stack((xa_, ya), -1)
            q = pref.double()+ub-ua
            _, _, _, okq, _, _ = sb(cc, q)
            mvg = elig & self.moving
            sil['eligible'] += int(mvg.sum())
            bg = mvg & inb & ~fgb
            sil['true_pixel_background'] += int(bg.sum())
            sil['true_pixel_background_sample_support_valid'] += int((bg & okq).sum())
            nb = torch.floor(ub).long()
            anyfg = torch.zeros_like(bg)
            for dy in (0, 1):
                for dx in (0, 1):
                    yy, xx = (nb[:, 1]+dy).clamp(0, Fb.H-1), (nb[:, 0]+dx).clamp(0, W-1)
                    anyfg |= Fb.mask[c, yy, xx]
            sil['true_pixel_background_any_bilinear_neighbour_foreground'] += int((bg & anyfg).sum())
            if bool(bg.any()):
                edt = torch.as_tensor(distance_transform_edt(~Fb.mask[c].cpu().numpy()), device=dev)
                sil['distance_px'].append(edt[pbc[bg, 1], pbc[bg, 0]].cpu().numpy())
            keep = elig & ~front & ~offsurf & okq
            j = torch.where(keep)[0]
            rec['c'].append(cc[j])
            rec['p'].append(j)
            rec['pref'].append(pref[j])
            rec['xref'].append(xa_all[S][j])
            rec['xt'].append(xb_all[S][j])
        ident = {k: torch.cat(v) for k, v in rec.items()}
        ident['a'], ident['b'] = a, b
        d = sil.pop('distance_px')
        sil['distance_to_foreground_px'] = q_stats(np.concatenate(d)) if d else None
        sil['semantics'] = ('moving identities with a valid true reference whose TRUE current position rounds to a background pixel; '
                            'the existing nearest-pixel rule rejects these true states')
        return ident, sil

    # ---------------------------------------------------------------- one evaluation of a patch model at states X
    def evaluate(self, ident, X, model, k, noise_model, frame_b=None, S_fixed=None, extra_var=None, chunk=1500):
        kind = model
        dev, rig = self.dev, self.rig
        Fa = self.frames[ident['a']]
        Fb = frame_b if frame_b is not None else self.frames[ident['b']]
        sb = Sampler(Fb, rig, self.noise, self.pol)
        h = k//2
        oy, ox = torch.meshgrid(torch.arange(-h, h+1, device=dev), torch.arange(-h, h+1, device=dev), indexing='ij')
        O = torch.stack((ox.reshape(-1), oy.reshape(-1)), -1)            # (K,2) integer (u,v) offsets
        K = len(O)
        G = (k+1)**2
        corner = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=dev)
        gidx = ((O[:, None, :]+h+corner[None])[..., 1]*(k+1)+(O[:, None, :]+h+corner[None])[..., 0])   # (K,4)
        M = len(ident['c'])
        out = {'cost': torch.zeros(M, dtype=torch.float64, device=dev), 'info': torch.zeros((M, 3, 3), dtype=torch.float64, device=dev),
               'grad': torch.zeros((M, 3), dtype=torch.float64, device=dev), 'nvalid': torch.zeros(M, dtype=torch.long, device=dev),
               'nlost': torch.zeros(M, dtype=torch.long, device=dev), 'S': torch.zeros((M, K), dtype=torch.bool, device=dev),
               'res': torch.zeros((M, K, 4), dtype=torch.float64, device=dev)}
        chans = [3] if kind == 'depth' else [0, 1, 2, 3]
        for s0 in range(0, M, chunk):
            sl = slice(s0, min(M, s0+chunk))
            c, pref, xref = ident['c'][sl], ident['pref'][sl], ident['xref'][sl]
            m = len(c)
            ut, zt, Jt, Jzt = rig.project_rows(X[sl], c)
            ur, zr, _, _ = rig.project_rows(xref, c)
            # reference patch (fixed measured pixels)
            P = pref[:, None, :]+O[None]
            inside = (P[..., 0] >= 0) & (P[..., 1] >= 0) & (P[..., 0] < Fa.W) & (P[..., 1] < Fa.H)
            Pc = P.clone()
            Pc[..., 0].clamp_(0, Fa.W-1)
            Pc[..., 1].clamp_(0, Fa.H-1)
            ce = c[:, None].expand(-1, K)
            Vr = torch.cat((Fa.rgb[ce, Pc[..., 1], Pc[..., 0]], Fa.depth[ce, Pc[..., 1], Pc[..., 0], None]), -1).double()
            okr = inside & Fa.mask[ce, Pc[..., 1], Pc[..., 0]] & Fa.raw_valid[ce, Pc[..., 1], Pc[..., 0]] & torch.isfinite(Vr).all(-1)
            Dc = Vr[:, K//2, 3]
            span = O.double().norm(dim=1)[None].expand(m, -1)
            f = self.rig.K[c, :2].min(1).values[:, None]
            okr &= (Vr[..., 3]-Dc[:, None]).abs() <= allowance(Dc[:, None].expand(-1, K), f, self.noise, self.pol, span)
            Vr = torch.where(okr[..., None], Vr, torch.ones_like(Vr))
            s2r = torch.cat((self.noise.rgb_variance.double().expand(m, K, 3), self.noise.depth_sigma2(Vr[..., 3])[..., None]), -1)
            # current samples at p_ref + o + du(x_t)
            q = pref.double()+ut-ur
            uv = (q[:, None, :]+O[None].double()).reshape(-1, 2)
            val, grad, w, okc, s2c, base = sb(ce.reshape(-1), uv)
            val, grad, okc = val.reshape(m, K, 4), grad.reshape(m, K, 4, 2), okc.reshape(m, K)
            w = w.reshape(m, K, 4)[:, 0]
            s2c = s2c.reshape(m, K, 4, 4)
            S = okr & okc if S_fixed is None else S_fixed[sl]
            valid = S & okc
            out['S'][sl] = S
            out['nvalid'][sl] = valid.sum(1)
            out['nlost'][sl] = (S & ~okc).sum(1)
            r = val-Vr
            r[..., 3] -= (zt-zr)[:, None]
            J = grad@Jt[:, None, :, :]                                        # (m,K,4,3)
            J[..., 3, :] -= Jzt[:, None, :]
            vf = valid.double()
            r = r*vf[..., None]
            J = J*vf[..., None, None]
            out['res'][sl] = r
            # exact shared-raw-pixel covariance of the current samples: R = W diag(s2_raw) W^T
            Wm = torch.zeros((m, K, G), dtype=torch.float64, device=dev)
            Wm.scatter_add_(2, gidx[None].expand(m, -1, -1), (w[:, None, :]*vf[..., None]).expand(-1, -1, 4).contiguous())
            s2g = torch.zeros((m, G, 4), dtype=torch.float64, device=dev)
            cnt = torch.zeros((m, G, 1), dtype=torch.float64, device=dev)
            src = torch.where(valid[..., None, None], s2c, torch.zeros_like(s2c))           # (m,K,4corner,4ch)
            gi = gidx[None].expand(m, -1, -1).reshape(m, K*4)
            s2g.scatter_add_(1, gi[..., None].expand(-1, -1, 4), src.reshape(m, K*4, 4))
            cnt.scatter_add_(1, gi[..., None], valid[..., None].expand(-1, -1, 4).reshape(m, K*4, 1).double())
            s2g = s2g/cnt.clamp_min(1)
            extra = torch.zeros((m, K, 4), dtype=torch.float64, device=dev)
            if noise_model == 'interp':
                from estimator.temporal_material_terms import resampling_discrepancy_variance
                ev, _ = resampling_discrepancy_variance(Fb, ce.reshape(-1), uv, base)
                extra = ev.reshape(m, K, 4)
            elif noise_model == 'xcam':
                extra = extra+torch.as_tensor(self.xcam_var, device=dev)[None, None]
            if extra_var is not None:
                extra = extra+extra_var
            eye = torch.eye(K, dtype=torch.float64, device=dev)
            for ch in chans:
                Rc = torch.einsum('mkg,mg,mjg->mkj', Wm, s2g[..., ch], Wm)
                Rt = Rc+torch.diag_embed((s2r[..., ch]+extra[..., ch])*vf)
                rc, Jc = r[..., ch], J[..., ch, :]
                if kind == 'gainnorm' and ch < 3:
                    # n = I / mean_valid(I) on both sides; exact delta-method covariance; null space span(1_valid)
                    nv = vf.sum(1, keepdim=True).clamp_min(1)
                    Ic = torch.where(valid, val[..., ch], torch.zeros_like(rc))
                    Ir = torch.where(valid, Vr[..., ch], torch.zeros_like(rc))
                    mc, mr = Ic.sum(1, keepdim=True)/nv, Ir.sum(1, keepdim=True)/nv
                    # a channel whose patch mean is exactly 0 (black) carries no gain-invariant information: excluded
                    chan_ok = ((mc > 0) & (mr > 0)).double()
                    mc, mr = torch.where(mc > 0, mc, torch.ones_like(mc)), torch.where(mr > 0, mr, torch.ones_like(mr))
                    nc, nr = Ic/mc, Ir/mr
                    rc = (nc-nr)*vf
                    gm = (Jc*vf[..., None]).sum(1, keepdim=True)/nv[..., None]
                    Jc = (Jc-nc[..., None]*gm)/mc[..., None]*vf[..., None]
                    Ac = (torch.diag_embed(vf)-nc[..., None]*vf[:, None, :]/nv[..., None])/mc[..., None]
                    Ar = (torch.diag_embed(vf)-nr[..., None]*vf[:, None, :]/nv[..., None])/mr[..., None]
                    Rc_cur = torch.einsum('mkg,mg,mjg->mkj', Wm, s2g[..., ch], Wm)+torch.diag_embed(extra[..., ch]*vf)
                    Rt = Ac@Rc_cur@Ac.transpose(1, 2)+Ar@torch.diag_embed(s2r[..., ch]*vf)@Ar.transpose(1, 2)
                    tr = Rt.diagonal(dim1=1, dim2=2).sum(1)/nv[:, 0]
                    Rt = Rt+tr[:, None, None]*vf[:, :, None]*vf[:, None, :]/nv[..., None]
                    rc, Jc = rc*chan_ok, Jc*chan_ok[..., None]
                    dead = chan_ok[:, 0] == 0
                    Rt[dead] = torch.eye(K, dtype=torch.float64, device=dev)
                    tr = torch.where(dead, torch.ones_like(tr), tr)
                Rt = Rt+torch.diag_embed(1-vf)
                Rt = (Rt+Rt.transpose(1, 2))/2
                L, info = torch.linalg.cholesky_ex(Rt)
                bad = info != 0
                if bool(bad.any()):
                    Rt[bad] = Rt[bad]+torch.eye(K, dtype=torch.float64, device=dev)*Rt[bad].diagonal(dim1=1, dim2=2).amax(1)[:, None, None]*1e-12
                    L = torch.linalg.cholesky(Rt)
                wr = torch.cholesky_solve(rc[..., None], L)[..., 0]
                wJ = torch.cholesky_solve(Jc, L)
                logdet = 2*torch.log(L.diagonal(dim1=1, dim2=2)).sum(1)
                if kind == 'gainnorm' and ch < 3:
                    logdet = logdet-torch.log(tr.clamp_min(1e-300))*(vf.sum(1) > 0)
                out['cost'][sl] += (rc*wr).sum(1)/2+logdet/2
                out['info'][sl] += Jc.transpose(1, 2)@wJ
                out['grad'][sl] += (Jc*wr[..., None]).sum(1)
        return out

    def per_point(self, ident, o, key):
        n = len(self.sel)
        x = o[key]
        acc = torch.zeros((n,)+tuple(x.shape[1:]), dtype=x.dtype, device=self.dev)
        acc.index_add_(0, ident['p'], x)
        return acc

    # ---------------------------------------------------------------- same-instant cross-camera discrepancy (no GT)
    def cross_camera_variance(self, a=0):
        """Frame-0 measured backprojections (the catalogue itself, no GT) sampled bilinearly in every OTHER camera where
        they are the first-hit winner on one surface: depth residual = interpolation/rasterization discrepancy; RGB
        residual = interpolation + view-dependent shading. Robust variances per channel."""
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        F = self.frames[a]
        x = torch.as_tensor(self.xyz, device=self.dev)
        own = first_hit_pixel_owner(x, self.rig, F, dense=False)
        src_c = torch.as_tensor(self.source[:, 1], device=self.dev)
        src_p = torch.as_tensor(self.source[:, 2], device=self.dev)
        rgb_src = F.rgb[src_c, src_p//F.W, src_p % F.W].double()
        sm = Sampler(F, self.rig, self.noise, self.pol)
        rs, ds = [], []
        for c in range(self.rig.count):
            j = torch.where((own.reason[c] == 0) & (src_c != c))[0]
            cc = torch.full((len(j),), c, device=self.dev)
            u, z, _, _ = self.rig.project_rows(x[j], cc)
            val, _, _, ok, _, _ = sm(cc, u)
            f = float(self.rig.K[c, :2].min())
            ok &= (val[:, 3]-z).abs() <= allowance(z, f, self.noise, self.pol, math.sqrt(2))
            ds.append((val[ok, 3]-z[ok]))
            rs.append(val[ok, :3]-rgb_src[j[ok]])
        d, r = torch.cat(ds), torch.cat(rs)
        var = [robust_var(r[:, ch]) for ch in range(3)]+[robust_var(d)]
        self.xcam_var = var
        return {'rows': int(len(d)), 'robust_variance_rgb_depth_q6827': var, 'robust_sigma_rgb_depth': [math.sqrt(v) for v in var],
                'depth_abs_m': q_stats(d.abs().cpu().numpy()), 'rgb_abs_max_channel': q_stats(r.abs().amax(1).cpu().numpy()),
                'ground_truth_used': False}


# ==================================================================== analyses
def eig_summary(study, ident, info_pt, d_pt, obs):
    mv = study.moving & obs
    A = info_pt.cpu().numpy()
    sc = np.abs(A).reshape(len(A), -1).max(1)
    sc = np.where(np.isfinite(sc) & (sc > 0), sc, 1.)
    ev, evec = np.linalg.eigh(A/sc[:, None, None])          # per-matrix scaling: eigenvalues span > 20 decades
    ev, evec = torch.as_tensor(ev*sc[:, None], device=study.dev), torch.as_tensor(evec, device=study.dev)
    std = ev.clamp_min(1e-300).rsqrt()
    dn = d_pt.norm(dim=1)
    rec1 = std < 1e-3
    recd = std < (dn/3)[:, None]
    proj = ((evec.transpose(1, 2)@d_pt.unsqueeze(-1)).squeeze(-1))**2
    energy = (proj*recd.double()).sum(1)/(dn**2).clamp_min(1e-300)
    Pinv = torch.eye(3, dtype=torch.float64, device=study.dev)/study.sigma_p**2
    dhat = torch.linalg.solve(Pinv+info_pt, (info_pt@d_pt.unsqueeze(-1))).squeeze(-1)
    oerr = (dhat-d_pt).norm(dim=1)
    n = study.normals
    wn = (evec[:, :, 0]*n).sum(1).abs()
    sn = (evec[:, :, 2]*n).sum(1).abs()
    g = mv
    hist = lambda m: {str(i): int(((m[g]).sum(1) == i).sum()) for i in range(4)}
    return {'moving_points_observed': int(g.sum()),
            'data_std_mm_weak_mid_strong': [q_stats(1e3*std[g, j].cpu().numpy()) for j in range(3)],
            'recoverable_directions_std_lt_1mm_hist': hist(rec1), 'recoverable_directions_std_lt_third_of_true_motion_hist': hist(recd),
            'true_motion_energy_in_recoverable_subspace': q_stats(energy[g].cpu().numpy()),
            'oracle_linear_error_mm': q_stats(1e3*oerr[g].cpu().numpy()), 'zero_motion_error_mm': q_stats(1e3*dn[g].cpu().numpy()),
            'weakest_direction_abs_cos_to_source_normal': q_stats(wn[g & torch.isfinite(wn)].cpu().numpy()),
            'strongest_direction_abs_cos_to_source_normal': q_stats(sn[g & torch.isfinite(sn)].cpu().numpy())}, evec


def curves(study, ident, model, k, noise_model, evec, S_gt):
    dev = study.dev
    p = ident['p']
    xr, xt = ident['xref'], ident['xt']
    d = (xt-xr)
    npt = len(study.sel)
    dpt = torch.zeros((npt, 3), dtype=torch.float64, device=dev)
    dpt.index_add_(0, p, d)
    cnt = torch.bincount(p, minlength=npt).clamp_min(1)[:, None]
    dpt = dpt/cnt
    scale = torch.where(study.moving, dpt.norm(dim=1), torch.full_like(dpt[:, 0], float(dpt[study.moving].norm(dim=1).median())))
    res = {}
    for name, grid, mk in (('true_direction', ALPHA, lambda a: xr+a*d),
                           ('weak_direction', BETA, lambda b: xt+b*(scale[p][:, None]*evec[p, :, 0])),
                           ('middle_direction', BETA, lambda b: xt+b*(scale[p][:, None]*evec[p, :, 1]))):
        J, lost = [], []
        for g in grid:
            o = study.evaluate(ident, mk(float(g)), model, k, noise_model, S_fixed=S_gt)
            J.append(study.per_point(ident, o, 'cost'))
            lost.append(study.per_point(ident, o, 'nlost'))
        J, lost = torch.stack(J, 1), torch.stack(lost, 1)
        obs = torch.bincount(p, minlength=npt) > 0
        ref_i = int(np.argmin(np.abs(grid-(1. if name == 'true_direction' else 0.))))
        dJ = J-J[:, ref_i:ref_i+1]
        am = torch.as_tensor(grid, device=dev)[J.argmin(1)]
        rec = {}
        for grp, gm in (('moving', study.moving & obs), ('static', ~study.moving & obs)):
            if name == 'true_direction' and grp == 'static':
                continue
            if not bool(gm.any()):
                continue
            rec[grp] = {'points': int(gm.sum()), 'argmin_quantiles': q_stats(am[gm].cpu().numpy()),
                        'argmin_within_0.1_of_truth_fraction': float(((am[gm]-grid[ref_i]).abs() <= .1+1e-9).double().mean()),
                        'median_deltaJ_vs_truth_curve': [float(x) for x in dJ[gm].median(0).values],
                        'p10_deltaJ_vs_truth_curve': [float(x) for x in torch.quantile(dJ[gm], .1, dim=0)],
                        'points_with_lost_support_on_curve_fraction': float((lost[gm] > 0).any(1).double().mean())}
            if name == 'true_direction':
                i0 = int(np.argmin(np.abs(grid)))
                rec[grp]['truth_lower_than_zero_motion_fraction'] = float((J[gm, ref_i] < J[gm, i0]).double().mean())
        res[name] = {'grid': grid.tolist(), **rec}
    return res


def map_from(study, ident, model, k, noise_model, start, S_gt, frame_b=None, iters=8):
    """Per-point Gauss-Newton MAP (independent points, prior N(x_ref, sigma_p^2 I) = zero-motion prediction) from `start`;
    trial steps that lose any patch sample are rejected (identity validity), no point coupling."""
    dev = study.dev
    p = ident['p']
    npt = len(study.sel)
    xr = torch.zeros((npt, 3), dtype=torch.float64, device=dev)
    xr[p] = ident['xref']
    obs = torch.bincount(p, minlength=npt) > 0
    x = torch.zeros_like(xr)
    x[p] = start
    ip = 1/study.sigma_p**2
    # material identity = the patch samples valid at the START state (never compared after deleting samples)
    S_gt = study.evaluate(ident, start, model, k, noise_model, frame_b=frame_b)['S']

    def total(xx):
        o = study.evaluate(ident, xx[p], model, k, noise_model, frame_b=frame_b, S_fixed=S_gt)
        return (study.per_point(ident, o, 'cost')+.5*ip*((xx-xr)**2).sum(1), study.per_point(ident, o, 'info'),
                study.per_point(ident, o, 'grad'), study.per_point(ident, o, 'nlost'))
    J, A, g, _ = total(x)
    for it in range(iters):
        H = A+ip*torch.eye(3, dtype=torch.float64, device=dev)
        step = -torch.linalg.solve(H, (g+ip*(x-xr)).unsqueeze(-1)).squeeze(-1)
        pending = obs.clone()
        newx = x.clone()
        for ls in range(10):
            tr = x+(.5**ls)*step
            tr[~pending] = newx[~pending]
            Jt, _, _, lost = total(tr)
            good = pending & (Jt <= J+1e-12*J.abs().clamp_min(1)) & (lost == 0)
            newx[good] = tr[good]
            pending &= ~good
            if not bool(pending.any()):
                break
        if torch.equal(newx, x):
            break
        x = newx
        J, A, g, _ = total(x)
    return x


def patch_rigidity(study, ident, ks):
    """GT validation of the rigid-image-patch approximation: material of every reference patch pixel = GT first-hit
    owner at tau on the same surface; its true image displacement / depth change / part vs the centre point."""
    dev, rig = study.dev, study.rig
    a, b = ident['a'], ident['b']
    Fa = study.frames[a]
    own = study.owner_a
    xa = torch.as_tensor(study.X[a], device=dev)
    xb = torch.as_tensor(study.X[b], device=dev)
    lab = torch.as_tensor(study.verified, device=dev)
    S = torch.as_tensor(study.sel, device=dev)
    ci = S[ident['p']]
    out = {}
    for k in ks:
        h = k//2
        oy, ox = torch.meshgrid(torch.arange(-h, h+1, device=dev), torch.arange(-h, h+1, device=dev), indexing='ij')
        O = torch.stack((ox.reshape(-1), oy.reshape(-1)), -1)
        K, M = len(O), len(ci)
        P = ident['pref'][:, None, :]+O[None]
        inside = (P[..., 0] >= 0) & (P[..., 1] >= 0) & (P[..., 0] < Fa.W) & (P[..., 1] < Fa.H)
        P[..., 0].clamp_(0, Fa.W-1)
        P[..., 1].clamp_(0, Fa.H-1)
        ce = ident['c'][:, None].expand(-1, K)
        j = own.owner_id[ce, P[..., 1], P[..., 0]]
        jj = j.clamp_min(0)
        u_a, z_a, _, _ = rig.project_rows(xa[jj].reshape(-1, 3), ce.reshape(-1))
        u_b, z_b, _, _ = rig.project_rows(xb[jj].reshape(-1, 3), ce.reshape(-1))
        D = Fa.depth[ce, P[..., 1], P[..., 0]].double()
        f = rig.K[ident['c'], :2].min(1).values[:, None]
        same = inside & (j >= 0) & Fa.mask[ce, P[..., 1], P[..., 0]] & ((D-z_a.reshape(M, K)).abs() <= allowance(D, f, study.noise, study.pol, math.sqrt(2)))
        du_o = (u_b-u_a).reshape(M, K, 2)
        dz_o = (z_b-z_a).reshape(M, K)
        uc_a, zc_a, _, _ = rig.project_rows(xa[ci], ident['c'])
        uc_b, zc_b, _, _ = rig.project_rows(xb[ci], ident['c'])
        dev_px = (du_o-(uc_b-uc_a)[:, None]).norm(dim=2)
        dev_z = (dz_o-(zc_b-zc_a)[:, None]).abs()
        # 3D-TRANSLATION patch alternative: each reference patch pixel back-projected with ITS OWN measured depth,
        # translated by the centre point's 3D displacement, re-projected (perspective and depth variation exact for a
        # locally translating surface; no new geometric representation, measured depth only)
        Xo = rig.backproject(ce.reshape(-1), P.reshape(-1, 2).double(), D.reshape(-1))
        dx = (xb[ci]-xa[ci])[:, None, :].expand(-1, K, -1).reshape(-1, 3)
        u0, z0, _, _ = rig.project_rows(Xo, ce.reshape(-1))
        u1, z1, _, _ = rig.project_rows(Xo+dx, ce.reshape(-1))
        dev_px3 = ((u1-u0).reshape(M, K, 2)-du_o).norm(dim=2)
        dev_z3 = ((z1-z0).reshape(M, K)-dz_o).abs()
        mixed = same & (lab[jj] != lab[ci][:, None]) & (lab[jj] >= 0)
        rec = {}
        for grp, gm in (('moving', lab[ci] == 1), ('static', lab[ci] == 0)):
            s = same[gm]
            mx = torch.where(s, dev_px[gm], torch.zeros_like(dev_px[gm])).amax(1)
            mz = torch.where(s, dev_z[gm], torch.zeros_like(dev_z[gm])).amax(1)
            rec[grp] = {'identities': int(gm.sum()), 'unidentified_or_other_surface_pixel_fraction': float(1-s.double().mean()),
                        'patches_containing_other_part_fraction': float(mixed[gm].any(1).double().mean()),
                        'max_displacement_deviation_px': q_stats(mx.cpu().numpy()),
                        'max_depth_change_deviation_m': q_stats(mz.cpu().numpy()),
                        'patches_with_deviation_above_0.1px_fraction': float((mx > .1).double().mean()),
                        'patches_with_deviation_above_0.5px_fraction': float((mx > .5).double().mean())}
            mx3 = torch.where(s, dev_px3[gm], torch.zeros_like(dev_px3[gm])).amax(1)
            mz3 = torch.where(s, dev_z3[gm], torch.zeros_like(dev_z3[gm])).amax(1)
            single = s & ~mixed[gm]
            mx3s = torch.where(single, dev_px3[gm], torch.zeros_like(dev_px3[gm])).amax(1)
            rec[grp]['translation3d_max_displacement_deviation_px'] = q_stats(mx3.cpu().numpy())
            rec[grp]['translation3d_max_depth_change_deviation_m'] = q_stats(mz3.cpu().numpy())
            rec[grp]['translation3d_deviation_above_0.1px_fraction'] = float((mx3 > .1).double().mean())
            rec[grp]['translation3d_same_part_pixels_max_deviation_px'] = q_stats(mx3s.cpu().numpy())
            rec[grp]['translation3d_same_part_deviation_above_0.1px_fraction'] = float((mx3s > .1).double().mean())
        out[f'k{k}'] = rec
        out[f'k{k}']['_exact'] = (same, du_o, dz_o)
    return out


def exact_motion_patch_residual(study, ident, rig_rec, k):
    """Patch residual when every patch pixel uses ITS OWN true displacement (no rigid approximation)."""
    same, du_o, dz_o = rig_rec[f'k{k}']['_exact']
    dev = study.dev
    h = k//2
    oy, ox = torch.meshgrid(torch.arange(-h, h+1, device=dev), torch.arange(-h, h+1, device=dev), indexing='ij')
    O = torch.stack((ox.reshape(-1), oy.reshape(-1)), -1)
    M, K = same.shape
    Fa, Fb = study.frames[ident['a']], study.frames[ident['b']]
    sb = Sampler(Fb, study.rig, study.noise, study.pol)
    P = ident['pref'][:, None, :]+O[None]
    Pc = P.clone()
    Pc[..., 0].clamp_(0, Fa.W-1)
    Pc[..., 1].clamp_(0, Fa.H-1)
    ce = ident['c'][:, None].expand(-1, K)
    ref = torch.cat((Fa.rgb[ce, Pc[..., 1], Pc[..., 0]], Fa.depth[ce, Pc[..., 1], Pc[..., 0], None]), -1).double()
    val, _, _, ok, _, _ = sb(ce.reshape(-1), (P.double()+du_o).reshape(-1, 2))
    r = val.reshape(M, K, 4)-ref
    r[..., 3] -= dz_o
    return r, same & ok.reshape(M, K)


def stats_res(r, m):
    rr = r[m]
    if not len(rr):
        return None
    return {'samples': int(len(rr)), 'sigma_q6827_rgb_depth': [math.sqrt(robust_var(rr[:, ch])) for ch in range(4)],
            'abs_depth_m': q_stats(rr[:, 3].abs().cpu().numpy()), 'abs_rgb_max_channel': q_stats(rr[:, :3].abs().amax(1).cpu().numpy())}


def run_scene(name, a, out):
    dev = torch.device(a.device)
    st = Study(a.data_root/name, dev, a.moving_points, a.static_points, a.seed)
    rep = {'scene': name, 'ground_truth_namespace': 'POSTHOC_GT_STUDY_ONLY', 'read_by_inference': False,
           'label_verification': st.label_report, 'selected_points': {'moving': int(st.moving.sum()), 'static': int((~st.moving).sum()),
           'seed': a.seed, 'original_point_ids': st.sel.tolist()}, 'sensor_noise': st.calib['result'], 'sensor_detail': st.calib['sensor'],
           'prior_sigma_p_m': st.sigma_p, 'prior_semantics': 'isotropic zero-motion prediction sigma_v*dt (run configuration velocity covariance)'}
    rep['noise_cross_camera_same_instant'] = st.cross_camera_variance(0)
    q_rgb = 1/255.
    rep['noise_decomposition'] = {'declared_sensor_sigma_rgb': math.sqrt(st.calib['result']['rgb_variance'][0]),
                                  'quantization_sigma_rgb': q_rgb/math.sqrt(12),
                                  'declared_sensor_sigma_depth_m_at_1m': math.sqrt(st.calib['result']['depth_variance_m2']+st.calib['result']['depth_squared_variance']),
                                  'depth_quantum_m': st.calib['sensor'].get('depth_quantum_m'),
                                  'cross_camera_same_instant_sigma_rgb_depth': rep['noise_cross_camera_same_instant']['robust_sigma_rgb_depth']}
    rep['pairs'] = []
    for (ta, tb) in ((0, 1), (1, 2)):
        ident, sil = st.identities(ta, tb)
        if a.only_rigidity:
            rig_rec = patch_rigidity(st, ident, [3, 5, 7, 9])
            rep['pairs'].append({'pair': [ta, tb], 'identities': len(ident['c']), 'silhouette_true_state_validity': sil,
                                 'patch_rigidity_gt': {kk: {g: v for g, v in rv.items() if g != '_exact'} for kk, rv in rig_rec.items()}})
            continue
        pr = {'pair': [ta, tb], 'identities': len(ident['c']), 'silhouette_true_state_validity': sil}
        npt = len(st.sel)
        obs = torch.bincount(ident['p'], minlength=npt) > 0
        dpt = torch.zeros((npt, 3), dtype=torch.float64, device=dev)
        dpt.index_add_(0, ident['p'], ident['xt']-ident['xref'])
        dpt = dpt/torch.bincount(ident['p'], minlength=npt).clamp_min(1)[:, None]
        # information / recoverable directions at the true state
        pr['information'] = {}
        evecs = {}
        S_gt = {}
        gtres = {}
        for model, k in MODELS_INFO:
            for nm in NOISE:
                o = st.evaluate(ident, ident['xt'], model, k, nm)
                info = st.per_point(ident, o, 'info')
                summ, evec = eig_summary(st, ident, info, dpt, obs)
                cost = st.per_point(ident, o, 'cost')
                nv = st.per_point(ident, o, 'nvalid')
                summ['valid_samples_per_identity'] = q_stats(o['nvalid'].double().cpu().numpy())
                pr['information'][f'{model}-{k}/{nm}'] = summ
                evecs[(model, k, nm)] = evec
                S_gt[(model, k)] = o['S']
                if nm == 'declared':
                    mvi = st.moving[ident['p']]
                    gtres[(model, k)] = {'moving': stats_res(o['res'], o['S'] & mvi[:, None]), 'static': stats_res(o['res'], o['S'] & ~mvi[:, None])}
            print(f'[study] {name} {ta}->{tb} {model}-{k}: done', flush=True)
        pr['residual_at_truth_declared_patch_rigid'] = {f'{m}-{k}': v for (m, k), v in gtres.items() if m == 'rgbd'}
        # rigid-patch validity against GT
        rig_rec = patch_rigidity(st, ident, [3, 5, 7, 9])
        pr['patch_rigidity_gt'] = {kk: {g: v for g, v in rv.items() if g != '_exact'} for kk, rv in rig_rec.items()}
        mvi = st.moving[ident['p']]
        pr['residual_at_truth_exact_per_pixel_motion'] = {}
        for k in (3, 5, 9):
            r, m = exact_motion_patch_residual(st, ident, rig_rec, k)
            pr['residual_at_truth_exact_per_pixel_motion'][f'k{k}'] = {'moving': stats_res(r, m & mvi[:, None]), 'static': stats_res(r, m & ~mvi[:, None])}
        if (ta, tb) == (0, 1):
            pr['curves'], pr['map_from_truth'], pr['map_from_prediction'] = {}, {}, {}
            xpt_gt = torch.zeros((npt, 3), dtype=torch.float64, device=dev)
            xpt_gt[ident['p']] = ident['xt']
            for model, k in MODELS_CURVE:
                for nm in ('declared', 'xcam'):
                    key = f'{model}-{k}/{nm}'
                    pr['curves'][key] = curves(st, ident, model, k, nm, evecs[(model, k, nm)], S_gt[(model, k)])
                    for tag, start in (('map_from_truth', ident['xt']), ('map_from_prediction', ident['xref'])):
                        x = map_from(st, ident, model, k, nm, start, S_gt[(model, k)])
                        e = (x-xpt_gt).norm(dim=1)
                        pr[tag][key] = {g: q_stats(1e3*e[gm & obs].cpu().numpy()) for g, gm in (('moving_error_mm', st.moving), ('static_error_mm', ~st.moving))}
                    print(f'[study] {name} curves/MAP {key}', flush=True)
            # photometric perturbations of the current image (static and moving), MAP from truth
            pr['photometric'] = {}
            Fb = st.frames[tb]
            for tag, g, off in (('gain_1.02', 1.02, 0.), ('gain_1.05', 1.05, 0.), ('gain_1.08', 1.08, 0.), ('offset_+0.02', 1., .02)):
                Fp = type('F', (), {})()
                for att in ('H', 'W', 'k', 'depth', 'mask', 'raw_valid'):
                    setattr(Fp, att, getattr(Fb, att))
                Fp.rgb = Fb.rgb*g+off
                for model, k in MODELS_PHOTO:
                    x = map_from(st, ident, model, k, 'declared', ident['xt'], S_gt[(model, k)], frame_b=Fp)
                    e = (x-xpt_gt).norm(dim=1)
                    pr['photometric'][f'{tag}/{model}-{k}'] = {gname: q_stats(1e3*e[gm & obs].cpu().numpy())
                                                               for gname, gm in (('moving_false_motion_mm', st.moving), ('static_false_motion_mm', ~st.moving))}
                print(f'[study] {name} photometric {tag}', flush=True)
        rep['pairs'].append(pr)
    json_write(out/f'{name}.json', serial(rep))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default='USB_100109,blade_103706')
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-root', type=Path, required=True, help='parent; a unique study_<stamp>_<id> child is created')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--moving-points', type=int, default=2000)
    ap.add_argument('--static-points', type=int, default=800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only-rigidity', action='store_true', help='identities + GT patch-approximation validity only')
    a = ap.parse_args()
    out = a.output_root/f'study_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    summary = {'tool': 'tools/study_material_measurement_information.py', 'scenes': {}}
    for name in [s for s in a.scenes.split(',') if s]:
        summary['scenes'][name] = str(out/f'{name}.json')
        run_scene(name, a, out)
    json_write(out/'study_index.json', serial(summary))
    print(out)


if __name__ == '__main__':
    raise SystemExit(main())
