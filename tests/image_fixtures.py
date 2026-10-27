"""Analytic ray-cast RGB-D scenes for the image-IEKF tests.

Observed images come from exact ray/rectangle intersection with procedural
texture, NOT from the point renderer under test, and no point correspondence
is ever handed to an estimator.
"""
import numpy as np
from rgbd_common.dataloader import RGBDFrame
from rgbd_common.projection_cuda import Camera
from rgbd_common.lie import exp as se3_exp


def look_at(name, position, target, width=96, height=72, f=150., up=(0, 0, 1)):
    position, target, up = (np.asarray(a, float) for a in (position, target, up))
    z = target-position; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    g = np.eye(4); g[:3, :3] = np.stack((x, y, z)); g[:3, 3] = -g[:3, :3]@position
    return Camera(name, width, height, np.array([f, f, (width-1)/2, (height-1)/2]), g, .01, 20.)


def texture(s, t, phase=0.):
    return np.stack((.5+.35*np.sin(2*np.pi*s/.09+phase)*np.cos(2*np.pi*t/.13),
                     .5+.35*np.sin(2*np.pi*(s+t)/.11),
                     .45+.3*np.cos(2*np.pi*(s-.7*t)/.07+phase)), -1)


class Plate:
    """Textured rectangle: center c, in-plane axes a,b, half extents ea,eb."""
    def __init__(self, center, a, b, ea, eb, part, phase=0.):
        self.c, self.a, self.b = (np.asarray(v, float) for v in (center, a, b))
        self.a /= np.linalg.norm(self.a); self.b /= np.linalg.norm(self.b)
        self.n = np.cross(self.a, self.b); self.ea, self.eb, self.part, self.phase = ea, eb, part, phase


def raycast(camera, plates, poses):
    """Per-pixel nearest hit; poses[part] is the 4x4 world motion of that part."""
    W, H = camera.width, camera.height; fx, fy, cx, cy = camera.intrinsics
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    ray_c = np.stack(((px-cx)/fx, (py-cy)/fy, np.ones_like(px, dtype=float)), -1).reshape(-1, 3)
    Rcw, tcw = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    origin = -Rcw.T@tcw; ray = ray_c@Rcw
    best = np.full(W*H, np.inf); rgb = np.zeros((W*H, 3)); part = np.full(W*H, -1)
    for plate in plates:
        T = poses[plate.part]
        c = T[:3, :3]@plate.c+T[:3, 3]; a, b, n = (T[:3, :3]@v for v in (plate.a, plate.b, plate.n))
        denom = ray@n
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = ((c-origin)@n)/denom
        hit = origin+lam[:, None]*ray-c
        s, t = hit@a, hit@b
        ok = np.isfinite(lam) & (lam > 0) & (abs(s) <= plate.ea) & (abs(t) <= plate.eb) & (lam < best)
        best[ok] = lam[ok]; rgb[ok] = texture(s[ok], t[ok], plate.phase); part[ok] = plate.part
    depth = np.where(np.isfinite(best), best, 0.)   # ray_c has unit optical z: lam is metric Z
    mask = np.isfinite(best)
    return rgb.reshape(H, W, 3), depth.reshape(H, W), mask.reshape(H, W), part.reshape(H, W)


class SyntheticDataset:
    """Dataset-like object: cameras, times, frame_ids, observation(k, ci). noise: (rgb sigma, depth sigma) added
    deterministically per (frame, camera) to the observed foreground."""
    def __init__(self, cameras, plates, motion, frames, dt=1/30., noise=None):
        self.cameras, self.plates, self.motion, self.noise = cameras, plates, motion, noise
        self.times = np.arange(frames)*dt; self.frame_ids = list(range(frames))
        self.base = self; self._cache = {}

    def poses(self, k):
        return {part: f(self.times[k]) for part, f in self.motion.items()}

    def observation(self, k, ci):
        if (k, ci) not in self._cache:
            rgb, depth, mask, part = raycast(self.cameras[ci], self.plates, self.poses(k))
            if self.noise:
                g = np.random.default_rng(1000*k+ci)
                rgb = np.where(mask[..., None], np.clip(rgb+g.normal(0, self.noise[0], rgb.shape), 0, 1), rgb)
                depth = np.where(mask, depth+g.normal(0, self.noise[1], depth.shape), depth)
            self._cache[k, ci] = RGBDFrame(rgb, depth, mask, self.cameras[ci], k, ci, None, True)
        return self._cache[k, ci]


def two_plate_scene(motion="prismatic", frames=6, cameras=4, speed=.02, width=96, height=72, offset=.06, noise=None, rate=.12):
    """Static textured base plus a textured moving plate (one part of interest)."""
    plates = [Plate([0, 0, 0], [1, 0, 0], [0, 1, 0], .45, .3, 0, 0.),
              Plate([0, 0, offset], [1, 0, 0], [0, 1, 0], .2, .12, 1, 1.3)]
    static = lambda t: np.eye(4)
    if motion == 'prismatic':
        moving = lambda t: se3_exp(np.r_[speed*t*30*np.array([1., .3, 0.]), 0, 0, 0])
    elif motion == 'revolute':
        def moving(t):
            # Hinge about the moving plate's edge line x=-0.2 (axis +y), angle `rate` rad/frame.
            a = np.array([-.2, 0, .06]); R = se3_exp(np.r_[0, 0, 0, 0, -rate*t*30, 0])
            T = R.copy(); T[:3, 3] = a-R[:3, :3]@a
            return T
    elif motion == 'static':
        moving = static
    else:
        raise ValueError(motion)
    ring = [look_at(f'cam_{i:03d}', [1.6*np.cos(th), 1.6*np.sin(th), 1.1], [0, 0, 0], width, height)
            for i, th in enumerate(np.linspace(0, 2*np.pi, cameras, endpoint=False)+.4)]
    return SyntheticDataset(ring, plates, {0: static, 1: moving}, frames, noise=noise)


def catalog(dataset, k=0, scale=1.5):
    """Endpoint colored samples back-projected from every valid pixel of every camera."""
    xyz, rgb, rho, source = [], [], [], []
    for ci, camera in enumerate(dataset.cameras):
        f = dataset.observation(k, ci); fx, fy, cx, cy = camera.intrinsics
        pix = np.flatnonzero(f.valid.ravel()); z = f.depth.ravel()[pix]
        q = np.stack(((pix % camera.width-cx)/fx*z, (pix//camera.width-cy)/fy*z, z), 1)
        g = camera.world_to_camera
        xyz.append((q-g[:3, 3])@g[:3, :3]); rgb.append(f.rgb.reshape(-1, 3)[pix])
        rho.append(scale*z/np.sqrt(fx*fy)); source.append(np.c_[np.full(len(pix), k), np.full(len(pix), ci), pix])
    return np.concatenate(xyz), np.concatenate(rgb), np.concatenate(rho), np.concatenate(source)


def source_parts(dataset, source):
    """Ground-truth part of each catalog sample (TEST evaluation only)."""
    out = np.empty(len(source), np.int64)
    for k, ci in {(int(a), int(b)) for a, b in source[:, :2]}:
        part = raycast(dataset.cameras[ci], dataset.plates, dataset.poses(k))[3].ravel()
        use = (source[:, 0] == k) & (source[:, 1] == ci)
        out[use] = part[source[use, 2]]
    return out


def true_positions(dataset, xyz0, parts, k, k0=0):
    out = np.empty_like(xyz0)
    for part in np.unique(parts):
        T = dataset.poses(k)[part]@np.linalg.inv(dataset.poses(k0)[part])
        out[parts == part] = xyz0[parts == part]@T[:3, :3].T+T[:3, 3]
    return out
