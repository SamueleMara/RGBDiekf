"""Analytic RGB-D scenes and an independent NumPy oracle for the material-following tests.

Scene: a textured plane Z = Z0 + a X + b Y (world = camera 0) whose MATERIAL translates by d per frame. f = 128 px on a
32 px image (one pixel = Z/128 m, about 1.6 cm at Z = 2: the real rigs have about 1.2 cm), 0.1 m camera baseline. Every pixel is
rendered exactly by ray-plane intersection: depth = optical Z, RGB = smooth texture of the material coordinate
(X - k dx, Y - k dy). Camera c is translated by t_c (identity rotation). Nothing here imports estimator code.
"""
from types import SimpleNamespace
import numpy as np
import torch

S, FOC = 32, 128.
CX = CY = 15.5
POLICY = {'same_surface_n_sigma': 3., 'same_surface_max_incidence_deg': 80.}
RGB_VAR, DEPTH_VAR = 1.28e-6, 1e-8
DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def texture(X, Y, kind='textured'):
    if kind == 'textureless':
        return np.stack((np.full_like(X, .5), np.full_like(X, .4), np.full_like(X, .3)), -1)
    return np.stack((.5+.2*np.sin(9*X)+.1*np.cos(5*Y), .5+.15*np.cos(7*X+3*Y), .4+.2*np.sin(6*Y)), -1)


class Scene:
    def __init__(self, d=(0., 0., 0.), slope=(.15, -.1), Z0=2., cams=((0., 0., 0.), (-.1, 0., 0.)), kind='textured', gain=None):
        self.d, self.slope, self.Z0, self.t, self.kind = np.asarray(d, float), slope, Z0, np.asarray(cams, float), kind
        self.gain = gain or {}

    def render(self, k, device, mask=None):
        a, b = self.slope
        yy, xx = np.meshgrid(np.arange(S, dtype=float), np.arange(S, dtype=float), indexing='ij')
        xt, yt = (xx-CX)/FOC, (yy-CY)/FOC
        rgb, depth = [], []
        for t in self.t:
            # world point = lam (xt, yt, 1) - t ; plane at frame k: Z = Z0 + dz k + a (X - dx k) + b (Y - dy k)
            lam = (self.Z0+self.d[2]*k-a*(t[0]+self.d[0]*k)-b*(t[1]+self.d[1]*k)+t[2])/(1-a*xt-b*yt)
            X, Y = lam*xt-t[0], lam*yt-t[1]
            rgb.append(texture(X-self.d[0]*k, Y-self.d[1]*k, self.kind)*self.gain.get(k, 1.))
            depth.append(lam)
        f = SimpleNamespace(H=S, W=S, k=k)
        f.rgb = torch.tensor(np.stack(rgb), dtype=torch.float64, device=device)
        f.depth = torch.tensor(np.stack(depth), dtype=torch.float64, device=device)
        f.mask = torch.ones((len(self.t), S, S), dtype=torch.bool, device=device) if mask is None else torch.as_tensor(mask, device=device)
        f.raw_valid = f.mask.clone()
        f.valid = f.mask & f.raw_valid
        return f

    def point(self, cam, pixel, k=0):
        """World position of the surface seen by `cam` at integer flat `pixel` in frame k."""
        f = self.render(k, 'cpu')
        u, v = pixel % S, pixel//S
        lam = float(f.depth[cam, v, u])
        return np.array([lam*(u-CX)/FOC, lam*(v-CY)/FOC, lam])-self.t[cam]


def rig(device, n=2, cams=((0., 0., 0.), (-.1, 0., 0.))):
    from estimator.projective_camera import CameraRig
    out = []
    for c in range(n):
        T = np.eye(4)
        T[:3, 3] = cams[c]
        out.append(SimpleNamespace(name=f'c{c}', width=S, height=S, intrinsics=np.array([FOC, FOC, CX, CY]), near=.01, world_to_camera=T))
    return CameraRig(out, device)


def noise(device):
    from estimator.projective_sensor import SensorNoise
    return SensorNoise(torch.full((3,), RGB_VAR, dtype=torch.float64, device=device), DEPTH_VAR, 0.)


# ---------------- independent NumPy oracle
def project(x, t):
    xc = np.asarray(x, float)+t
    return np.array([FOC*xc[0]/xc[2]+CX, FOC*xc[1]/xc[2]+CY]), xc[2]


def np_bilinear(img, q, cell):
    u, v = q[0]-cell[0], q[1]-cell[1]
    x0, y0 = int(cell[0]), int(cell[1])
    F = [img[y0, x0], img[y0, x0+1], img[y0+1, x0], img[y0+1, x0+1]]
    w = [(1-u)*(1-v), u*(1-v), (1-u)*v, u*v]
    return sum(wi*np.asarray(Fi, float) for wi, Fi in zip(w, F)), np.array(w)


def np_material_f(x_t, x_ref, t_cam, p_ref, cur_rgbd, ref_rgbd, cell, eta_cur=None, eta_ref=None):
    """f = sum_a w_a (y_a - eta_a) - (y_ref - eta_ref) - e4 (Z_t - Z_ref), sampled at q = p_ref + pi(x_t) - pi(x_ref)."""
    ut, zt = project(x_t, t_cam)
    ur, zr = project(x_ref, t_cam)
    q = np.array([p_ref % S, p_ref//S], float)+ut-ur
    img = cur_rgbd.copy()
    x0, y0 = int(cell[0]), int(cell[1])
    if eta_cur is not None:
        for (dy, dx), e in zip(((0, 0), (0, 1), (1, 0), (1, 1)), eta_cur):
            img[y0+dy, x0+dx] = img[y0+dy, x0+dx]-e
    val, w = np_bilinear(img, q, cell)
    ref = ref_rgbd[p_ref//S, p_ref % S]-(0 if eta_ref is None else eta_ref)
    f = val-ref
    f[3] -= zt-zr
    return f, q, w


def np_chart(theta, eta_depth, base, uv0, D0, t_cam):
    """Source chart x0 = base + R^T[-eta ray0 + z shift], z = D0 - eta (identity rotation here)."""
    ray0 = np.array([(uv0[0]-CX)/FOC, (uv0[1]-CY)/FOC, 1.])
    z = D0-eta_depth
    shift = np.array([theta[0]/FOC, theta[1]/FOC, 0.])
    return base+(-eta_depth*ray0+z*shift)


def rgbd(frame, cam):
    return np.concatenate((frame.rgb[cam].cpu().numpy(), frame.depth[cam].cpu().numpy()[..., None]), -1)


def fd(fun, x, h=1e-6):
    """5-point central finite-difference Jacobian of a vector function."""
    x = np.asarray(x, float)
    f0 = fun(x)
    J = np.zeros((len(f0), len(x)))
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h
        J[:, j] = (-fun(x+2*e)+8*fun(x+e)-8*fun(x-e)+fun(x-2*e))/(12*h)
    return J


def make_filter(scene, device, pixels, cls=None, lag=3, warp=None, pyramid=0, resampling='none', frame0=None, velocity=1.):
    """A material (default) or local-correlated filter whose catalogue is the camera-0 backprojection of `pixels`."""
    from estimator.temporal_material_run import MaterialFilter
    cls = cls or MaterialFilter
    f0 = frame0 if frame0 is not None else scene.render(0, device)
    src = np.array([[0, 0, p] for p in pixels])
    base = np.array([scene.point(0, p) for p in pixels])
    filt = cls(rig(device), noise(device), POLICY, np.zeros((4, 4)) if warp is None else warp, np.eye(3)*1e-4, lag, 10**8, 8, f0, src, base)
    if hasattr(filt, 'pyramid_levels'):
        filt.pyramid_levels, filt.resampling = pyramid, resampling
    filt.initialize(f0, np.eye(2)/12, np.eye(3)*velocity**2)
    return filt, f0, base


def run_frames(filt, frames, upto, dt=.04, traced=()):
    out = []
    for k in range(1, upto+1):
        out.append(filt.step({j: frames[j] for j in range(max(0, k-2), k+1)}, k, k, dt, 10, 1e-10, traced))
    return out


RING8 = tuple((.06*np.cos(2*np.pi*j/8), .06*np.sin(2*np.pi*j/8), 0.) for j in range(8))


def ring_scene(**kw):
    """Same analytic plane seen by 8 translated cameras on a 6 cm ring (identity rotations)."""
    return Scene(cams=RING8, **kw)


def ring_filter(scene, device, pixels, **kw):
    """make_filter on the 8-camera ring (catalogue from camera 0)."""
    from estimator.temporal_material_run import MaterialFilter
    f0 = kw.pop('frame0', None) or scene.render(0, device)
    src = np.array([[0, 0, p] for p in pixels])
    base = np.array([scene.point(0, p) for p in pixels])
    filt = MaterialFilter(rig(device, 8, RING8), noise(device), POLICY, np.zeros((4, 4)), np.eye(3)*1e-4, kw.get('lag', 3), 10**8, 8, f0, src, base)
    filt.gradient_guide = kw.get('gradient_guide', 'off')
    filt.initialize(f0, np.eye(2)/12, np.eye(3))
    return filt, f0, base
