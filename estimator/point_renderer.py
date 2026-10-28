"""Colored point-cloud RGB-D rasterizer: the image measurement function h(s).

Each measured endpoint sample is a fixed-color surface sample with a compact,
non-learned footprint of world radius rho_i (from its source pixel). In camera
c it projects to (u_i, v_i, z_i) and covers pixel p with weight

    q_ip = (dx/s_x)^2 + (dy/s_y)^2,  s = f rho_i / z_i,  w_ip = (1 - q_ip)^2 [q < 1].

Front-surface visibility is geometric and continuous. Each contributor has its own same-surface band
b_i = 3 sigma_D(z_i) + rho_i (declared depth noise plus the footprint's slanted-surface extent) and an
occlusion depth zeta_i = z_i + b_i (1/w_i - 1): a sample occludes from its own depth at the centre of its
footprint (w = 1), and its power to occlude fades continuously to nothing at the footprint edge (w -> 0),
the pixel-area reading of a compact footprint. Per pixel zeta_front = min_i zeta_i and a contributor at
gap g_j = z_j - zeta_front has visibility

    v_j = 1  (g <= b_j/2),    v_j = (1 - u^2)^2,  u = (2g - b_j)/b_j   (b_j/2 < g < b_j),    v_j = 0  (g >= b_j),

a C^1 taper INSIDE the declared same-surface band, so nothing beyond the band is visible (as with the hard
band) and the render is a CONTINUOUS function of the point positions. (A hard band made the fixed IEKF
objective jump whenever a sample crossed it or an edge sample left a pixel: measured +50 coverage energy
for a 1e-6 m step, which collapses the trust region. A taper OUTSIDE the band, [b, 2b], let a surface 6 cm
behind another show through when 3 sigma_D + rho = 4.8 cm, which corrupted support evidence.) The prediction is normalized splatting with omega = v w:

    W_p = sum omega_ip,  I_p = sum omega_ip c_i / W_p,  D_p = sum omega_ip z_i / W_p,
    coverage A_p = 1 - exp(-W_p)   (rendered support; A_p > 0 iff W_p > 0).

covered_p = [W_p >= w_min] is a reporting/display threshold only.

The Jacobian is exact for the frozen front contributor (argmin of z; piecewise at ordering ties): it
includes dv/dz_i of every contributor and dv/dz_front, which is added to the front contributor's own
row. No opacity, alpha compositing or background exists here. apply_H / apply_HT are the exact
Jacobian and its adjoint (finite differences, <Hv,a> = <v,H^T a>). Reductions use sort/segment sums,
never floating atomics, so CPU and CUDA results are deterministic.
"""
from dataclasses import dataclass
import numpy as np
import torch
from estimator import small_contractions as sc

BOUNDARY = 'compact (1-q)^2 footprint, s=f*rho/z (depth derivative included)'


@dataclass
class RenderPolicy:
    band_sigma: float = 3.
    slope_factor: float = 1.
    min_weight: float = .05
    depth_variance: float = 1e-4
    depth_squared_variance: float = 0.

    def band(self, z, rho):
        """Same-surface depth band of a contributor at depth z with footprint radius rho."""
        # Depth gap still on the same surface: sensor noise plus the sample's own
        # footprint extent along a slanted surface (slope_factor = tan of the
        # steepest same-surface incidence tolerated per footprint radius).
        return self.band_sigma*torch.sqrt(self.depth_variance+self.depth_squared_variance*z*z)+self.slope_factor*rho

    def band_derivative(self, z):
        """d band / d z_front."""
        return self.band_sigma*self.depth_squared_variance*z/torch.sqrt(self.depth_variance+self.depth_squared_variance*z*z)

    def manifest(self):
        return {'visibility': f'own band b_i = {self.band_sigma} sigma_D(z_i) + {self.slope_factor} rho_i; occlusion depth zeta_i = z_i + b_i (1/w_i - 1); '
                              'front = min zeta; v = 1 for gap <= b/2, C1 taper (1-u^2)^2, u = (2 gap - b)/b, zero at b',
                'coverage': f'W_p >= {self.min_weight} (peak footprint weight 1)', 'footprint': BOUNDARY,
                'depth_variance': self.depth_variance, 'depth_squared_variance': self.depth_squared_variance}


def footprints(partition, dataset, scale=1.5):
    """World support radius from the actual source pixel: scale * z_src / f_src."""
    rho = np.empty(len(partition.xyz))
    frame = {int(fid): k for k, fid in enumerate(dataset.frame_ids)}
    for ci, camera in enumerate(dataset.cameras):
        use = partition.source[:, 1] == ci
        if not use.any(): continue
        k = frame[int(partition.source[use, 0][0])]
        depth = np.asarray(dataset.observation(k, ci).depth).ravel()[partition.source[use, 2]]
        rho[use] = scale*depth/np.sqrt(camera.intrinsics[0]*camera.intrinsics[1])
    if not (np.isfinite(rho).all() and (rho > 0).all()): raise ValueError('Every endpoint sample needs a positive source footprint')
    return rho


def _segments(keys):
    order = torch.argsort(keys, stable=True)
    unique, lengths = torch.unique_consecutive(keys[order], return_counts=True)
    return order, unique, lengths


def _segsum(values, segments, count):
    order, unique, lengths = segments
    out = values.new_zeros((count, *values.shape[1:]))
    if len(order): out[unique] = torch.segment_reduce(values[order], 'sum', lengths=lengths)
    return out


class CameraTensors:
    def __init__(self, camera, device):
        g = torch.as_tensor(camera.world_to_camera, dtype=torch.float64, device=device)
        self.R, self.t = g[:3, :3], g[:3, 3]
        self.fx, self.fy, self.cx, self.cy = map(float, camera.intrinsics)
        self.width, self.height, self.near, self.name = int(camera.width), int(camera.height), float(camera.near), camera.name

    def project(self, x):
        q = x@self.R.T+self.t
        z = q[:, 2]
        safe = torch.where(z > 0, z, torch.ones_like(z))
        uv = torch.stack((self.fx*q[:, 0]/safe+self.cx, self.fy*q[:, 1]/safe+self.cy), 1)
        j = x.new_zeros((len(x), 3, 3))
        j[:, 0, 0] = self.fx/safe; j[:, 0, 2] = -self.fx*q[:, 0]/safe**2
        j[:, 1, 1] = self.fy/safe; j[:, 1, 2] = -self.fy*q[:, 1]/safe**2
        j[:, 2, 2] = 1
        return uv, z, j@self.R


@dataclass
class View:
    """One camera's frozen render. Contributor arrays are pixel-sorted."""
    camera: CameraTensors
    point: torch.Tensor
    pixel: torch.Tensor
    weight: torch.Tensor
    jac: torch.Tensor          # [K,5,3] d(I_p, D_p, c_p)/dx_i of this contributor
    rgb: torch.Tensor          # [H*W,3] predicted, zero where uncovered
    depth: torch.Tensor        # [H*W]
    total: torch.Tensor        # [H*W] W_p
    coverage: torch.Tensor     # [H*W] c_p = 1 - exp(-W_p)
    covered: torch.Tensor      # [H*W] bool, W_p >= w_min (reporting)
    z_front: torch.Tensor
    pix_seg: tuple
    pt_seg: tuple
    stats: dict
    extra: torch.Tensor = None  # [H*W,m] normalized splat of optional per-point attributes (no derivative)


def _candidates(cam, x, rho, policy):
    uv, z, jproj = cam.project(x)
    ok = torch.isfinite(uv).all(1) & (z > max(cam.near, 0.))
    sx = torch.where(ok, cam.fx*rho/torch.where(ok, z, torch.ones_like(z)), torch.zeros_like(z))
    sy = torch.where(ok, cam.fy*rho/torch.where(ok, z, torch.ones_like(z)), torch.zeros_like(z))
    ok &= (uv[:, 0]+sx >= -.5) & (uv[:, 0]-sx <= cam.width-.5) & (uv[:, 1]+sy >= -.5) & (uv[:, 1]-sy <= cam.height-.5)
    radius = torch.ceil(torch.maximum(sx, sy)).long()
    parts = []
    for r in torch.unique(radius[ok]).tolist():
        ids = torch.nonzero(ok & (radius == r)).flatten()
        # floor(frac(u)+s) <= ceil(s): this window holds every pixel with |dx| < s.
        r = max(int(r), 1)
        off = torch.arange(-r, r+1, device=x.device)
        ox, oy = torch.meshgrid(off, off, indexing='xy')
        px = torch.round(uv[ids, 0])[:, None].long()+ox.reshape(1, -1)
        py = torch.round(uv[ids, 1])[:, None].long()+oy.reshape(1, -1)
        dx = px-uv[ids, 0, None]; dy = py-uv[ids, 1, None]
        q = (dx/sx[ids, None])**2+(dy/sy[ids, None])**2
        keep = (q < 1) & (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height)
        rows, cols = torch.nonzero(keep, as_tuple=True)
        parts.append((ids[rows], (py*cam.width+px)[rows, cols], dx[rows, cols], dy[rows, cols], q[rows, cols]))
    if not parts:
        e = x.new_zeros(0)
        return uv, z, jproj, sx, sy, torch.zeros(0, dtype=torch.long, device=x.device), torch.zeros(0, dtype=torch.long, device=x.device), e, e, e
    return (uv, z, jproj, sx, sy, *[torch.cat(v) for v in zip(*parts)])


def render_view(cam, x, rgb, rho, policy, n, extra=None):
    uv, z, jproj, sx, sy, pt, pix, dx, dy, q = _candidates(cam, x, rho, policy)
    npx = cam.width*cam.height
    zc = z[pt]
    w = (1-q)**2
    band = policy.band(zc, rho[pt])
    zeta = zc+band*(1/w-1)
    seg = _segments(pix)
    front = torch.full((npx,), float('inf'), dtype=x.dtype, device=x.device)
    if len(pt): front[seg[1]] = torch.segment_reduce(zeta[seg[0]], 'min', lengths=seg[2])
    gap = zc-front[pix]
    visible = gap < band
    occluded = int((~visible).sum())
    idx = torch.arange(len(pt), device=x.device)
    is_front = zeta == front[pix]                       # exact minima (before filtering: the front is always visible)
    pt, pix, dx, dy, q, zc, band, gap, w, is_front, idx = (a[visible] for a in (pt, pix, dx, dy, q, zc, band, gap, w, is_front, idx))
    u = ((2*gap-band)/band).clamp_min(0)
    taper = 2*gap > band
    v = torch.where(taper, (1-u*u)**2, torch.ones_like(u))
    dv_du = torch.where(taper, -4*u*(1-u*u), torch.zeros_like(u))
    om = v*w
    seg = _segments(pix)
    pos = torch.arange(len(pt), device=x.device)
    front_entry = torch.full((npx,), len(pt), dtype=torch.long, device=x.device)
    front_entry.scatter_reduce_(0, pix[is_front], pos[is_front], 'amin')   # lowest entry among ties: deterministic
    total = _segsum(om, seg, npx)
    srgb = _segsum(om[:, None]*rgb[pt], seg, npx)
    sz = _segsum(om*zc, seg, npx)
    covered = total >= policy.min_weight
    hit = total > 0
    safe = torch.where(hit, total, torch.ones_like(total))
    prgb = torch.where(hit[:, None], srgb/safe[:, None], torch.zeros_like(srgb))
    pdep = torch.where(hit, sz/safe, torch.zeros_like(sz))
    coverage = -torch.expm1(-total)
    pextra = None if extra is None else torch.where(hit[:, None], _segsum(om[:, None]*extra[pt], seg, npx)/safe[:, None], torch.zeros((npx, extra.shape[1]), dtype=x.dtype, device=x.device))
    # Frozen-front Jacobian. d w / d x_i through the projection (footprint scale s = f rho / z included):
    dwdq = -2*(1-q)
    dw = (torch.stack((dwdq*(-2*dx/sx[pt]**2), dwdq*(-2*dy/sy[pt]**2), dwdq*(2*q/zc)), 1)[:, None, :]@jproj[pt])[:, 0]
    jz = jproj[pt, 2]
    db = policy.band_derivative(zc)
    du_dz = 2/band-2*gap*db/band**2                     # own depth (band from own depth); u = 2 gap / b - 1
    domega = v[:, None]*dw+(w*dv_du*du_dz)[:, None]*jz
    inv = 1/safe[pix]
    dpix = torch.cat([(rgb[pt]-prgb[pix])*inv[:, None], ((zc-pdep[pix])*inv)[:, None], torch.exp(-total[pix])[:, None]], 1)   # d(I,D,A)/d omega
    jac = dpix[:, :, None]*domega[:, None, :]
    jac[:, 3] += (om*inv)[:, None]*jz
    # d omega_j / d zeta_front = -2 w_j v'_j / b_j, summed per pixel, chained through d zeta_front / d x_front
    if bool(taper.any()):
        vec = _segsum(dpix*(-2*w*dv_du/band)[:, None], seg, npx)
        rows = torch.nonzero(_segsum(taper.to(x.dtype), seg, npx) > 0).flatten()
        fe = front_entry[rows]
        if bool((fe >= len(pt)).any()): raise AssertionError('front contributor missing from a pixel with tapered contributors')
        dzeta = jz[fe]*(1+db[fe]*(1/w[fe]-1))[:, None]-(band[fe]/w[fe]**2)[:, None]*dw[fe]
        jac[fe] += vec[rows][:, :, None]*dzeta[:, None, :]
    use = hit[pix]
    stats = {'camera': cam.name, 'candidate_contributors': int(len(visible)), 'occluded_contributors': occluded,
             'visible_contributors': int(len(pt)), 'active_contributors': int(use.sum()),
             'covered_pixels': int(covered.sum()), 'projected_points': int(torch.unique(pt).numel()) if len(pt) else 0}
    pt, pix, om, jac = pt[use], pix[use], om[use], jac[use]
    return View(cam, pt, pix, om, jac, prgb, pdep, total, coverage, covered, front, _segments(pix), _segments(pt), stats, pextra)


class Linearization:
    """All cameras of one frame at one state; H acts on point XYZ only."""
    def __init__(self, views, n):
        self.views, self.n = views, n

    def predicted(self):
        return [(v.rgb, v.depth, v.coverage) for v in self.views]

    def apply_H(self, dx):
        out = []
        for v in self.views:
            npx = v.camera.width*v.camera.height
            out.append(_segsum(sc.matvec(v.jac, dx[v.point]), v.pix_seg, npx))
        return out

    def apply_HT(self, adjoints):
        total = None
        for v, a in zip(self.views, adjoints):
            part = _segsum(sc.rmatvec(v.jac, a[v.pixel]), v.pt_seg, self.n)
            total = part if total is None else total+part
        return total

    def block_fisher(self, weights):
        """Per-point 3x3 diagonal blocks of H^T W H.

        weights per camera: (rgb_scale [H*W], rgb_precision [3,3] or per-pixel diagonal [H*W,3], depth_precision [H*W],
        coverage_precision [H*W]), i.e. W_p = blockdiag(rgb_scale_p * rgb_precision,
        depth_precision_p, coverage_precision_p)."""
        total = None
        for v, (scale, rgb_precision, depth_precision, coverage_precision) in zip(self.views, weights):
            jr, jd, jc = v.jac[:, :3], v.jac[:, 3], v.jac[:, 4]
            if rgb_precision.shape == (3, 3): block = scale[v.pixel, None, None]*sc.congruence(jr, rgb_precision)
            else: block = scale[v.pixel, None, None]*sc.gram(jr*rgb_precision[v.pixel, :, None], jr)
            block = block+depth_precision[v.pixel, None, None]*jd[:, :, None]*jd[:, None, :]
            block = block+coverage_precision[v.pixel, None, None]*jc[:, :, None]*jc[:, None, :]
            part = _segsum(block, v.pt_seg, self.n)
            total = part if total is None else total+part
        return total

    def contributing(self):
        hit = torch.zeros(self.n, dtype=torch.bool, device=self.views[0].jac.device)
        for v in self.views: hit[v.point] = True
        return hit


def render(xyz, rgb, rho, cameras, policy, extra=None):
    n = len(xyz)
    return Linearization([render_view(c, xyz, rgb, rho, policy, n, extra) for c in cameras], n)


def reference_render(xyz, rgb, rho, camera, policy):
    """Naive CPU loop oracle for small scenes: the same model, no vectorization."""
    xyz, rgb, rho = (np.asarray(a, float) for a in (xyz, rgb, rho))
    g = np.asarray(camera.world_to_camera, float); fx, fy, cx, cy = map(float, camera.intrinsics)
    W, H = int(camera.width), int(camera.height)
    hits = {}
    for i, x in enumerate(xyz):
        q = g[:3, :3]@x+g[:3, 3]
        if not q[2] > max(float(camera.near), 0.): continue
        u, v = fx*q[0]/q[2]+cx, fy*q[1]/q[2]+cy
        sx, sy = fx*rho[i]/q[2], fy*rho[i]/q[2]
        for py in range(int(np.floor(v-sy))-1, int(np.ceil(v+sy))+2):
            for px in range(int(np.floor(u-sx))-1, int(np.ceil(u+sx))+2):
                if not (0 <= px < W and 0 <= py < H): continue
                d = ((px-u)/sx)**2+((py-v)/sy)**2
                if d < 1: hits.setdefault(py*W+px, []).append((i, (1-d)**2, q[2]))
    out_rgb = np.zeros((W*H, 3)); out_d = np.zeros(W*H); cov = np.zeros(W*H, bool); out_c = np.zeros(W*H)
    for p, h in hits.items():
        band = lambda i, z: policy.band_sigma*np.sqrt(policy.depth_variance+policy.depth_squared_variance*z*z)+policy.slope_factor*rho[i]
        front = min(z+band(i, z)*(1/w-1) for i, w, z in h)
        vis = []
        for i, w, z in h:
            b = band(i, z); g = z-front
            if g < b: vis.append((i, w*(1 if 2*g <= b else (1-((2*g-b)/b)**2)**2), z))
        h = vis
        total = sum(w for _, w, _ in h)
        cov[p] = total >= policy.min_weight
        out_c[p] = 1-np.exp(-total)
        if total > 0:
            out_rgb[p] = sum(w*rgb[i] for i, w, _ in h)/total
            out_d[p] = sum(w*z for _, w, z in h)/total
    return out_rgb, out_d, cov, out_c
