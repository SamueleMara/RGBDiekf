"""Synthetic multi-camera masked RGB-D sequences for the SVGD Stage-1 tests (evaluation-only GT kept OUTSIDE the inference
inputs: `gt_moving` is never passed to the runner). Analytic ray casting of textured planar patches: a static base patch
and an optional moving plate (translation or rotation about an axis on the base). Two or more pinhole cameras."""
import json
import math
from types import SimpleNamespace
import numpy as np

WITNESS = {}


def record(key, **values):
    WITNESS.setdefault(key, []).append(values)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target-eye; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z]); W = np.eye(4); W[:3, :3] = R; W[:3, 3] = -R@eye
    return W


class Camera:
    def __init__(self, name, W, width=64, height=48, f=150.):
        self.name, self.world_to_camera, self.width, self.height = name, W, width, height
        self.intrinsics = np.array([f, f, (width-1)/2, (height-1)/2])


def texture(p):
    """Smooth deterministic colour field of world position (encoded RGB in [0.1, 0.9])."""
    return np.clip(.5+.4*np.stack([np.sin(41*p[..., 0]+3*p[..., 1]), np.sin(37*p[..., 1]-5*p[..., 2]+1), np.sin(29*(p[..., 0]+p[..., 1]+p[..., 2])+2)], -1), 0, 1)


class Patch:
    """Planar rectangle centre o, axes u, v (half extents), optional flat colour."""
    def __init__(self, o, u, v, colour=None):
        self.o, self.u, self.v = (np.asarray(a, float) for a in (o, u, v)); self.colour = None if colour is None else np.asarray(colour, float)

    def moved(self, R, b):
        p = Patch(R@self.o+b, R@self.u, R@self.v); p.colour = self.colour; p.R, p.b = R, b; return p

    def hit(self, eye, d):
        n = np.cross(self.u, self.v); den = d@n
        with np.errstate(divide='ignore', invalid='ignore'):
            s = ((self.o-eye)@n)/den
        p = eye+s[:, None]*d; r = p-self.o
        a, b = r@self.u/(self.u@self.u), r@self.v/(self.v@self.v)
        ok = (s > 0) & (np.abs(a) <= 1) & (np.abs(b) <= 1) & np.isfinite(s)
        return s, p, ok


class Frame(SimpleNamespace):
    @property
    def valid(self): return self.mask & self.raw_valid


class SyntheticDataset:
    """Dataset-compatible object: cameras, times, frame_ids, observation(k, c) -> masked RGB-D frame, manifest()."""
    def __init__(self, K=3, motion='translate', speed=.05, depth_noise=0., rgb_noise=0., n_cams=2, seed=0, dt=1/30, plate_colour=None, extra_static=False):
        self.times = np.arange(K)*dt; self.frame_ids = list(range(K)); self.root = '/synthetic'
        eyes = [(.0, -.6, .35), (.45, -.45, .4), (-.45, -.45, .4), (.0, .6, .45)][:n_cams]
        self.cameras = [Camera(f'cam_{i:03d}', look_at(e, (0, 0, .02))) for i, e in enumerate(eyes)]
        self.base = SimpleNamespace(times=self.times, timing_report={'source': 'synthetic'}, metadata_report={}, root='/synthetic')
        self.static = Patch((0, 0, 0), (.08, 0, 0), (0, .06, 0))
        self.plate = Patch((0, -.01, .03), (.03, 0, 0), (0, 0, .02), colour=plate_colour)
        self.extra = Patch((0, .03, .015), (.03, 0, 0), (0, .0, .012)) if extra_static else None
        self.motion, self.speed, self.dn, self.rn, self.seed = motion, speed, depth_noise, rgb_noise, seed
        self.gt = []                                        # evaluation only

    def pose(self, k):
        t = self.times[k]
        if self.motion == 'translate': return np.eye(3), np.array([self.speed*t, 0, 0])
        if self.motion == 'rotate':                         # about the x axis through the plate's bottom edge (a hinge)
            a = self.speed*t*20; c, s = math.cos(a), math.sin(a); R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            p = np.array([0, -.01, .01]); return R, p-R@p
        return np.eye(3), np.zeros(3)

    def observation(self, k, c):
        cam = self.cameras[c]; W = cam.world_to_camera; fx, fy, cx, cy = cam.intrinsics
        u, v = np.meshgrid(np.arange(cam.width), np.arange(cam.height)); ray_c = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u, float)], -1).reshape(-1, 3)
        eye = -W[:3, :3].T@W[:3, 3]; d = ray_c@W[:3, :3]                   # world directions, optical z = 1 in camera
        R, b = self.pose(k)
        best = np.full(len(d), np.inf); rgb = np.zeros((len(d), 3)); moving = np.zeros(len(d), bool)
        patches = [(self.static, False), (self.plate.moved(R, b), True)]+([(self.extra, False)] if self.extra is not None else [])
        for P, mov in patches:
            s, p, ok = P.hit(eye, d); ok &= s < best
            best[ok] = s[ok]; moving[ok] = mov
            src = p if not mov else (p-b)@R                                  # material coordinates of the moving plate
            col = texture(src) if P.colour is None else np.broadcast_to(P.colour, p.shape)
            rgb[ok] = col[ok]
        g = np.random.default_rng(self.seed*1000+k*10+c)
        mask = np.isfinite(best); depth = np.where(mask, best, 0.)
        if self.dn: depth = np.where(mask, depth+g.normal(0, self.dn, depth.shape), 0.)
        if self.rn: rgb = np.clip(rgb+g.normal(0, self.rn, rgb.shape), 0, 1)
        sh = (cam.height, cam.width)
        return Frame(rgb=rgb.reshape(*sh, 3), depth=depth.reshape(sh).astype(np.float32), mask=mask.reshape(sh), raw_valid=(depth > 0).reshape(sh),
                     camera=cam, k=k, ci=c, gt_moving=moving.reshape(sh))

    def manifest(self):
        return {'input': self.root, 'synthetic': True, 'motion': self.motion, 'timestamps_seconds': self.times.tolist(),
                'cameras': [{'name': c.name, 'size': [c.width, c.height], 'intrinsics': c.intrinsics.tolist(), 'world_to_camera': c.world_to_camera.tolist()} for c in self.cameras]}


def runner_args(out, config, mode='experimental', **kw):
    a = SimpleNamespace(scene='/synthetic', output=out, config=config, svgd_theory_fidelity=mode, device='cpu', frame_limit=None, fps=None, dt_seconds=None,
                        timestamps_file=None, noise_config=None, sensor_noise='config', debug='none', display_cap=60000, debug_point_ids=None,
                        svgd_workspace_mib=64., svgd_kernel_backend='torch', stop_after_step=None, dry_run=False)
    for k, v in kw.items(): setattr(a, k, v)
    return a


def small_config(path, base, **override):
    """Copy of a repository config with numerical caps reduced for unit-test speed (statistical laws unchanged)."""
    cfg = json.loads(open(base).read())
    cfg['stein_numerics']['convergence']['value'].update(override.get('convergence', {'max_outer': 2, 'pcg_max_iter': 20}))
    for k, v in override.items():
        if k != 'convergence':
            g, p = k.split('.'); cfg[g][p]['value'] = v
    open(path, 'w').write(json.dumps(cfg))
    return path
