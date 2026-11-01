"""Analytic RGB-D generator for the Phase-1 temporal solver gates (RGBDiekf_Phase1 spec section 7). Generator only; ground
truth is returned separately and never handed to the estimator.

Cameras look along world +z (R_cw = I) from laterally translated centres o_c (oblique rays: different incidence per
camera). A plate is a rectangle of the plane z = z0 with colour c(x, y) = c0 + Gx (x - x_c) + Gy (y - y_c) in its own
material coordinates; it moves rigidly by a translation T(t). For R_cw = I a plane z = const projects affinely, so in every
camera and frame the observed colour and 1/Z are affine functions of the pixel coordinates (and Z itself is constant when
T_z = 0): the bilinear colour sample and the bilinear inverse-depth sample are EXACT at the true projection, which is
what makes the zero-residual static oracle (G2.1) exact to floating point. Ray casting picks the nearest plate per pixel
(occlusion between plates is physical). Noise (optional): independent Gaussian per pixel, seeded per (frame, camera).
"""
from dataclasses import dataclass, field
import numpy as np
from scene.camera import Camera
from scene.dataloader import RGBDFrame


def fronto_cameras(centres, W=96, H=72, f=120.):
    out = []
    for i, o in enumerate(centres):
        g = np.eye(4); g[:3, 3] = -np.asarray(o, float)
        out.append(Camera(f'cam_{i:03d}', W, H, np.array([f, f, (W-1)/2, (H-1)/2]), g, .01, 50.))
    return out


@dataclass
class AffinePlate:
    z0: float
    xlim: tuple
    ylim: tuple
    c0: tuple = (.5, .45, .4)
    Gx: tuple = (.8, -.3, .2)
    Gy: tuple = (.1, .6, -.5)
    motion: object = None            # t -> translation [3] (m); None = static
    texture: object = None           # optional callable (s, t) -> rgb [...,3] in material coordinates (non-affine tests)

    def T(self, t):
        return np.zeros(3) if self.motion is None else np.asarray(self.motion(t), float)

    def colour(self, x, y):
        if self.texture is not None: return self.texture(x, y)
        xc, yc = np.mean(self.xlim), np.mean(self.ylim)
        return np.asarray(self.c0)+np.multiply.outer(x-xc, self.Gx)+np.multiply.outer(y-yc, self.Gy)


class AnalyticDataset:
    def __init__(self, cameras, plates, times, noise=(0., 0.), mask_edit=None, seed=0):
        self.cameras, self.plates, self.times, self.seed = cameras, plates, np.asarray(times, float), seed
        self.frame_ids = list(range(len(times))); self.base = self; self.noise = noise; self._cache = {}
        self.mask_edit = mask_edit                                       # optional (k, ci, mask) -> mask (G6 contradictory masks)

    def render(self, k, ci):
        cam = self.cameras[ci]; t = self.times[k]
        fx, fy, cx, cy = cam.intrinsics; W, H = cam.width, cam.height
        o = -cam.world_to_camera[:3, 3]
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        dx, dy = (u-cx)/fx, (v-cy)/fy
        best = np.full((H, W), np.inf); rgb = np.zeros((H, W, 3))
        for pl in self.plates:
            T = pl.T(t)
            Z = pl.z0+T[2]-o[2]
            if Z <= cam.near: continue
            x, y = o[0]+Z*dx, o[1]+Z*dy
            xm, ym = x-T[0], y-T[1]
            hit = (xm >= pl.xlim[0]) & (xm <= pl.xlim[1]) & (ym >= pl.ylim[0]) & (ym <= pl.ylim[1]) & (Z < best)
            best[hit] = Z; rgb[hit] = pl.colour(xm[hit], ym[hit])
        mask = np.isfinite(best); depth = np.where(mask, best, 0.)
        return rgb, depth, mask

    def observation(self, k, ci):
        if (k, ci) not in self._cache:
            rgb, depth, mask = self.render(k, ci)
            if self.noise[0] or self.noise[1]:
                g = np.random.default_rng([self.seed, k, ci])
                rgb = np.where(mask[..., None], rgb+g.normal(0, self.noise[0], rgb.shape), rgb)
                depth = np.where(mask, depth+g.normal(0, self.noise[1], depth.shape), depth)
            if self.mask_edit is not None: mask = self.mask_edit(k, ci, mask.copy())
            self._cache[k, ci] = RGBDFrame(rgb, depth, mask, self.cameras[ci], k, ci, None, True)
        return self._cache[k, ci]

    def true_point(self, plate_index, x_material, t):
        """World position at time t of the material point x_material (frame-0 world coordinates of a frame-0 point)."""
        pl = self.plates[plate_index]
        return np.asarray(x_material, float)+pl.T(t)-pl.T(self.times[0])


def catalogue(ds, source_cam, pixels, pixel_var=1/12., depth_var=1e-6, k=0):
    """Frame-0 back-projection of chosen source pixels (u, v integer) of one camera: x_ref, rgb0, P_x0, source id. The
    catalogue is a measured one (depth and colour of the frame-0 image, noisy if the images are)."""
    cam = ds.cameras[source_cam]; f = ds.observation(k, source_cam)
    fx, fy, cx, cy = cam.intrinsics; o = -cam.world_to_camera[:3, 3]
    xs, cols, covs = [], [], []
    for u, v in pixels:
        D = float(f.depth[v, u]); b = np.array([(u-cx)/fx, (v-cy)/fy, 1.])
        xs.append(o+D*b); cols.append(f.rgb[v, u])
        J = np.array([[D/fx, 0, b[0]], [0, D/fy, b[1]], [0, 0, 1.]])       # d x / d(u, v, D)
        covs.append(J@np.diag([pixel_var, pixel_var, depth_var])@J.T)
    n = len(pixels)
    return {'x_ref': np.array(xs), 'rgb0': np.array(cols, float), 'P_x0': np.array(covs), 'src_cam': np.full(n, source_cam),
            'src_pixel': np.array([v*cam.width+u for u, v in pixels])}


def plate_of(ds, x0):
    """Index of the plate containing frame-0 world point x0 (evaluation only)."""
    for i, pl in enumerate(ds.plates):
        T = pl.T(ds.times[0])
        if abs(x0[2]-(pl.z0+T[2])) < 1e-9 and pl.xlim[0] <= x0[0]-T[0] <= pl.xlim[1] and pl.ylim[0] <= x0[1]-T[1] <= pl.ylim[1]: return i
    return -1


# --------------------------------------------------------------------------------------------- solver / oracle helpers
STAGE_A = {'latent_colour': False, 'depth_interpolation': 'metric'}   # anchor colour known (spec P1.1.5); metric = the dense oracle's sampler


def run_solver(ds, cat, noise_sig=(.01, 1e-3), P_v0=1., qa=.01, device=None, trace_ids=(), production=False, **tcfg_over):
    """The production Phase-1 solver on an analytic dataset (declared R from noise_sig, P_v0 [m^2/s^2], q_a [m^2/s^3]).
    Stage-A settings unless production=True (then configs/temporal_phase1.json as shipped) or overridden."""
    import json
    from pathlib import Path
    import torch
    from tests.projective_fixtures import POLICY
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import SensorNoise
    from estimator.ieks_point_trajectory import FrameStore, TemporalModel
    from estimator.temporal_phase1 import solve_trajectories
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tcfg = {**json.loads((Path(__file__).resolve().parents[1]/'configs/temporal_phase1.json').read_text()), **({} if production else STAGE_A), **tcfg_over}
    noise = SensorNoise(torch.tensor([noise_sig[0]**2]*3, dtype=torch.float64, device=dev), noise_sig[1]**2, 0., 0., 0., 'sum_w2')
    M = TemporalModel(CameraRig(ds.cameras, dev), noise, POLICY, tcfg, dev)
    K = len(ds.times)
    fr = FrameStore(ds, range(K), dev)
    return solve_trajectories(fr, M, list(range(K)), ds.times, cat['x_ref'], cat['P_x0'], cat['rgb0'], cat['src_cam'], P_v0*np.eye(3), qa*np.eye(3),
                              log=lambda s: None, trace_ids=trace_ids)


def dense_oracle_map(ds, cat, i, support, S_init, noise_sig=(.01, 1e-3), P_v0=1., qa=.01, iters=60):
    """INDEPENDENT dense NumPy Gauss-Newton of ONE point's full-trajectory posterior (tools/root_cause_oracle sampler and
    projection; nothing from the temporal solver), on the given support (list over frames of camera lists) with the row
    covariance fixed at each linearization (the production objective). Returns (S [K,6], dense posterior covariance)."""
    from tools import root_cause_oracle as O
    from tests.projective_fixtures import POLICY
    K = len(ds.times)
    cams = [O.Cam(c) for c in ds.cameras]
    imgs = [[O.Image(ds.observation(k, c)) for c in range(len(cams))] for k in range(K)]
    on = O.Noise([noise_sig[0]**2]*3, noise_sig[1]**2, 0., 0., 0., 'sum_w2')
    pol = {**POLICY, 'row_innovation_gate': 'depth_front_only', 'visibility_n_sigma': float('inf')}
    dt = np.diff(ds.times)
    F = [np.block([[np.eye(3), d*np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]) for d in dt]
    Q = [np.block([[d**3/3*qa*np.eye(3), d**2/2*qa*np.eye(3)], [d**2/2*qa*np.eye(3), d*qa*np.eye(3)]]) for d in dt]
    mu0 = np.r_[cat['x_ref'][i], 0, 0, 0]; P0 = np.zeros((6, 6)); P0[:3, :3] = cat['P_x0'][i]; P0[3:, 3:] = P_v0*np.eye(3)
    P0i = np.linalg.inv(P0); Qi = [np.linalg.inv(q) for q in Q]; a = cat['rgb0'][i]
    S = np.array(S_init, float).copy()

    def terms(S, Svar):
        rows = []
        for k in range(K):
            for c in support[k]:
                o = O.row(imgs[k][c], cams[c], S[k, :3], a, on, pol, np.eye(3), np.zeros(3))
                ov = O.row(imgs[k][c], cams[c], Svar[k, :3], a, on, pol, np.eye(3), np.zeros(3))
                if 'r' not in o or 'r' not in ov: return None
                rows.append((k, o['r'], o['H'], ov['var']))
        return rows

    def objective(S, rows):
        e0 = S[0]-mu0; J = .5*e0@P0i@e0
        for k in range(K-1):
            e = S[k+1]-F[k]@S[k]; J += .5*e@Qi[k]@e
        return J+.5*sum(float((r*r/v).sum()) for _, r, _, v in rows)

    for _ in range(iters):
        rows = terms(S, S)
        N = 6*K; A = np.zeros((N, N)); g = np.zeros(N)
        A[:6, :6] += P0i; g[:6] += P0i@(S[0]-mu0)
        for k in range(K-1):
            E = np.zeros((6, N)); E[:, 6*k:6*k+6] = -F[k]; E[:, 6*k+6:6*k+12] = np.eye(6)
            e = S[k+1]-F[k]@S[k]; A += E.T@Qi[k]@E; g += E.T@Qi[k]@e
        for k, r, H, v in rows:
            A[6*k:6*k+3, 6*k:6*k+3] += H.T@(H/v[:, None]); g[6*k:6*k+3] -= H.T@(r/v)
        d = -np.linalg.solve(A, g)
        if -g@d < 1e-14: break
        J0, al = objective(S, rows), 1.
        while al > 1e-12:
            Sc = S+al*d.reshape(K, 6); rc = terms(Sc, S)
            if rc is not None and objective(Sc, rc) < J0: S = Sc; break
            al /= 2
        else:
            break
    return S, np.linalg.inv(A)


def nees(err, cov):
    return float(err@np.linalg.solve(cov, err))


def run_original_filter(ds, cat, noise_sig=(.01, 1e-3), P_v0=1., qa=.01, device=None):
    """The ORIGINAL directional free filter (estimator/point_kalman_cuda predict + update_points, the per-frame loop of
    free_projective_run.run_free) on the same data, prior and R: frame 0 is the anchor (no update)."""
    import torch
    from tests.projective_fixtures import POLICY
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import SensorNoise, MeasuredFrame
    from estimator.point_kalman_cuda import predict, update_points
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    noise = SensorNoise(torch.tensor([noise_sig[0]**2]*3, dtype=torch.float64, device=dev), noise_sig[1]**2, 0., 0., 0., 'sum_w2')
    rig = CameraRig(ds.cameras, dev); n = len(cat['x_ref'])
    s = torch.zeros((n, 6), dtype=torch.float64, device=dev); s[:, :3] = torch.as_tensor(cat['x_ref'], device=dev)
    P = torch.zeros((n, 6, 6), dtype=torch.float64, device=dev); P[:, :3, :3] = torch.as_tensor(cat['P_x0'], device=dev); P[:, 3:, 3:] = P_v0*torch.eye(3, dtype=torch.float64, device=dev)
    rgb0 = torch.as_tensor(cat['rgb0'], dtype=torch.float64, device=dev); qc = qa*torch.eye(3, dtype=torch.float64, device=dev)
    out = [s.cpu().numpy()]
    for k in range(1, len(ds.times)):
        s_m, P_m = predict(s, P, float(ds.times[k]-ds.times[k-1]), qc)
        s, P, _, _ = update_points(s_m, P_m, MeasuredFrame(ds, k, dev), rig, rgb0, noise, POLICY, int(POLICY['max_iekf_iters']), float(POLICY['iekf_step_tolerance']))
        out.append(s.cpu().numpy())
    return np.stack(out)


def same_map(S_prod, S_oracle, C_oracle):
    """Squared Mahalanobis distance between two MAP trajectories under the oracle's posterior covariance (the production
    solver stops at a Newton decrement < 1e-8, the oracle at 1e-14)."""
    d = (np.asarray(S_prod, float)-np.asarray(S_oracle, float)).ravel()
    return float(d@np.linalg.solve(C_oracle, d))
