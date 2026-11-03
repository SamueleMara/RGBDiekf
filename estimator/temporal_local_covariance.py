"""Point-local joint Gaussians of the local-correlated temporal RGB-D IEKF.

Every canonical point i owns ONE Gaussian over
    a_i = [ s_{i,t} (6) | retained reference states s_{i,tau} (6) or the 5D source chart |
            retained raw sensor errors eta_{i,(camera,frame,pixel)} (4 each) ].
The first block of every point is always its current state (the source chart at the
anchor frame). Cross-point covariance is ZERO by the declared block-diagonal
assumed-density approximation: two points that use the same raw pixel carry separate
copies of its sensor error. WITHIN a point a raw key is one variable with one
calibrated prior, reused by every row that touches it.

Storage is packed and variable-length (block table + flat mean sum(n_i) + flat
covariance sum(n_i^2)); there is no all-point padded cube. Dense work happens in
bounded chunks of points with similar n_i. Padding columns carry zero covariance and
never enter a Jacobian, which is exact for covariance-form Gaussian algebra.
"""
import torch

STATE, CHART, KEY = 0, 1, 2
KIND_NAMES = {STATE: 'state', CHART: 'source_chart', KEY: 'raw_sensor_error'}
BLOCK_SIZE = (6, 5, 4)
CROSS_POINT_COVARIANCE = 'block_diagonal_assumed_density'
QUALIFIER = 'BLOCK_DIAGONAL_ACROSS_POINTS; TEMPORAL_AND_SENSOR_CROSS_CORRELATION_RETAINED_WITHIN_POINT'
REFERENCE_UNCERTAINTY = ('point-local joint Gaussian: current/reference states (or source chart), position-velocity blocks, '
                         'raw sensor errors already used by the SAME point (one prior per (camera,frame,pixel) key) and their '
                         'cross-covariances; cross-point covariance set to zero (block-diagonal assumed density)')
# dense chunk residency, measured: covariance, int64 gather indices, masks, Joseph factor, products, certificates
DENSE_TENSORS_PER_CHUNK = 24
# proposal pass only: Sigma, gather indices, H, Sigma H^T, S (no Joseph n x n products)
PROPOSAL_TENSORS_PER_CHUNK = 6


class LocalResourceLimit(RuntimeError):
    status = 'LOCAL_RESOURCE_LIMIT'


class LocalCovarianceFailure(RuntimeError):
    status = 'LOCAL_COVARIANCE_FAILURE'


class ForbiddenKeyRecreation(RuntimeError):
    status = 'FORBIDDEN_EXPIRED_KEY_RECREATION'


def exclusive_cumsum(x):
    out = torch.zeros(len(x)+1, dtype=torch.long, device=x.device)
    out[1:] = torch.cumsum(x, 0)
    return out


class Codec:
    """Unique int64 code of a block: (point, kind, camera, original frame, flat pixel)."""
    def __init__(self, C, K, HW):
        self.C, self.K, self.HW = int(C), int(K), int(HW)

    def encode(self, point, kind, camera, frame, pixel):
        return (((point*3+kind)*self.C+camera)*self.K+frame)*self.HW+pixel

    def state(self, point, frame, kind=STATE):
        z = torch.zeros_like(point)
        return self.encode(point, torch.full_like(point, kind), z, frame, z)

    def key(self, point, camera, frame, pixel):
        return self.encode(point, torch.full_like(point, KEY), camera, frame, pixel)


class Layout:
    """Block table grouped by point; order inside a point is the column order."""
    def __init__(self, N, point, kind, camera, frame, pixel):
        self.N = int(N)
        self.point, self.kind, self.camera, self.frame, self.pixel = (t.long() for t in (point, kind, camera, frame, pixel))
        if len(self.point) > 1 and bool((self.point[1:] < self.point[:-1]).any()):
            raise ValueError('local blocks must be grouped by point')
        self.size = torch.tensor(BLOCK_SIZE, device=self.point.device)[self.kind]
        counts = torch.bincount(self.point, minlength=self.N)
        self.boff = exclusive_cumsum(counts)
        self.dims = torch.zeros(self.N, dtype=torch.long, device=self.point.device).index_add_(0, self.point, self.size)
        cs = torch.cumsum(self.size, 0)-self.size
        self.start = cs-cs[self.boff[self.point]] if len(cs) else cs

    def __len__(self):
        return len(self.point)

    def codes(self, codec):
        return codec.encode(self.point, self.kind, self.camera, self.frame, self.pixel)

    def select(self, keep):
        return Layout(self.N, self.point[keep], self.kind[keep], self.camera[keep], self.frame[keep], self.pixel[keep])

    @staticmethod
    def concat(first, second):
        """Per point: blocks of `first`, then blocks of `second` (stable)."""
        cat = lambda a, b: torch.cat((a, b))
        point = cat(first.point, second.point)
        order = torch.argsort(point, stable=True)
        return Layout(first.N, point[order], cat(first.kind, second.kind)[order], cat(first.camera, second.camera)[order],
                      cat(first.frame, second.frame)[order], cat(first.pixel, second.pixel)[order])

    def describe(self, p):
        b = range(int(self.boff[p]), int(self.boff[p+1]))
        return [{'kind': KIND_NAMES[int(self.kind[j])], 'camera': int(self.camera[j]) if int(self.kind[j]) == KEY else None,
                 'frame': int(self.frame[j]), 'pixel': int(self.pixel[j]) if int(self.kind[j]) == KEY else None,
                 'columns': [int(self.start[j]), int(self.start[j]+self.size[j])]} for j in b]


class BlockIndex:
    """Sorted code -> (point-local start column, block row) lookup; -1 when absent."""
    def __init__(self, layout, codec, mask=None):
        ix = torch.arange(len(layout), device=layout.point.device)
        if mask is not None:
            ix = ix[mask]
        codes = layout.codes(codec)[ix]
        order = torch.argsort(codes)
        self.codes, self.block = codes[order], ix[order]
        self.layout = layout

    def lookup(self, codes):
        flat = codes.reshape(-1)
        out = torch.full_like(flat, -1)
        if len(self.codes) and len(flat):
            at = torch.searchsorted(self.codes, flat).clamp_max(len(self.codes)-1)
            hit = self.codes[at] == flat
            out[hit] = self.block[at[hit]]
        return out.reshape(codes.shape)


class LocalGaussian:
    """Packed variable-length point-local Gaussians."""
    def __init__(self, layout, mean, cov):
        self.layout, self.mean, self.cov = layout, mean, cov
        self.moff = exclusive_cumsum(layout.dims)
        self.coff = exclusive_cumsum(layout.dims*layout.dims)
        if len(mean) != int(self.moff[-1]) or len(cov) != int(self.coff[-1]):
            raise ValueError('packed local Gaussian storage does not match its layout')

    @classmethod
    def empty(cls, layout, dtype=torch.float64):
        d = layout.dims
        dev = layout.point.device
        return cls(layout, torch.zeros(int(d.sum()), dtype=dtype, device=dev), torch.zeros(int((d*d).sum()), dtype=dtype, device=dev))

    @property
    def N(self):
        return self.layout.N

    @property
    def nbytes(self):
        return int(self.mean.numel()+self.cov.numel())*self.mean.element_size()

    def _index(self, points, n):
        d = self.layout.dims[points]
        a = torch.arange(n, device=points.device)
        vm = a[None, :] < d[:, None]
        mi = self.moff[points][:, None]+a[None, :]
        ci = self.coff[points][:, None, None]+a[None, :, None]*d[:, None, None]+a[None, None, :]
        return vm, mi, vm[:, :, None] & vm[:, None, :], ci

    def gather(self, points, n=None):
        n = int(self.layout.dims[points].max()) if n is None else int(n)
        vm, mi, vc, ci = self._index(points, n)
        mean = torch.zeros((len(points), n), dtype=self.mean.dtype, device=self.mean.device)
        cov = torch.zeros((len(points), n, n), dtype=self.cov.dtype, device=self.cov.device)
        mean[vm] = self.mean[mi[vm]]
        cov[vc] = self.cov[ci[vc]]
        return mean, cov

    def scatter(self, points, mean, cov):
        vm, mi, vc, ci = self._index(points, mean.shape[1])
        self.mean[mi[vm]] = mean[vm]
        self.cov[ci[vc]] = cov[vc]

    def current_state_index(self):
        """Flat positions of each point's current 6D state (first block must be STATE)."""
        first = self.layout.boff[:-1]
        if bool((self.layout.kind[first] != STATE).any()):
            raise ValueError('current block is not a 6D state')
        return self.moff[:-1, None]+torch.arange(6, device=self.mean.device)


def chunk_plan(points, dims, budget_bytes, entry_bytes=8, tensors=DENSE_TENSORS_PER_CHUNK):
    """Size-sorted bounded chunks. Grouping only controls batching, never coupling."""
    if not len(points):
        return []
    order = torch.argsort(dims[points], stable=True)
    pts, d = points[order], dims[points][order].tolist()
    unit = entry_bytes*tensors
    if d[-1]**2*unit > budget_bytes:
        raise LocalResourceLimit(f'LOCAL_RESOURCE_LIMIT: one exact local block of dimension {d[-1]} needs '
                                 f'{d[-1]**2*unit} bytes > configured chunk budget {budget_bytes}; no correlation is dropped')
    out, i = [], 0
    while i < len(d):
        B = max(1, budget_bytes//(unit*d[i]**2))
        j = min(len(d), i+B)
        while j-i > 1 and (j-i)*d[j-1]**2*unit > budget_bytes:
            j = i+max(1, (j-i)//2)
        out.append(pts[i:j])
        i = j
    return out


def select_dense(mean, cov, columns, valid):
    """Per-point column subsets (B, n_new); invalid slots become zero padding."""
    B = len(mean)
    cols = torch.where(valid, columns, torch.zeros_like(columns))
    m = mean.gather(1, cols)*valid
    b = torch.arange(B, device=mean.device)[:, None, None]
    c = cov[b, cols[:, :, None], cols[:, None, :]]*(valid[:, :, None] & valid[:, None, :])
    return m, c


def kept_columns(layout, keep, points, n_new):
    """For chunk points: old point-local columns of kept blocks, in order, padded."""
    loc = torch.full((layout.N,), -1, dtype=torch.long, device=layout.point.device)
    loc[points] = torch.arange(len(points), device=points.device)
    blocks = torch.where(keep & (loc[layout.point] >= 0))[0]
    size = layout.size[blocks]
    p = layout.point[blocks]
    new_start = torch.cumsum(size, 0)-size
    first = torch.zeros(len(blocks), dtype=torch.bool, device=blocks.device)
    if len(blocks):
        first[0] = True
        first[1:] = p[1:] != p[:-1]
    base = torch.cummax(torch.where(first, new_start, torch.zeros_like(new_start)), 0).values
    new_start = new_start-base
    e = torch.arange(int(size.sum()), device=blocks.device)-torch.repeat_interleave(torch.cumsum(size, 0)-size, size)
    rep = torch.repeat_interleave(torch.arange(len(blocks), device=blocks.device), size)
    cols = torch.zeros((len(points), n_new), dtype=torch.long, device=blocks.device)
    valid = torch.zeros((len(points), n_new), dtype=torch.bool, device=blocks.device)
    bi, ni = loc[p[rep]], new_start[rep]+e
    cols[bi, ni] = layout.start[blocks][rep]+e
    valid[bi, ni] = True
    return cols, valid


def retain(g, keep, budget_bytes):
    """EXACT Gaussian marginal: keep the posterior mean/covariance submatrix of retained blocks.
    No information-form Schur complement and no global fill is formed."""
    new = LocalGaussian.empty(g.layout.select(keep), g.mean.dtype)
    for pts in chunk_plan(torch.arange(g.N, device=g.mean.device), g.layout.dims, budget_bytes):
        mean, cov = g.gather(pts)
        n_new = int(new.layout.dims[pts].max())
        cols, valid = kept_columns(g.layout, keep, pts, n_new)
        m, c = select_dense(mean, cov, cols, valid)
        new.scatter(pts, m, c)
    return new


class SourceChart:
    """Minimal source anchor chart theta=(du,dv,v) with ONE shared source raw-depth error.

    x0 = base + R^T[ (z-D0) ray0 + z (du/fx, dv/fy, 0) ],  z = D0 - eta_source.depth
    which is the same coupled backprojection as the globally correlated full mode,
    centred so that theta=0, eta=0 reproduces the recorded canonical position exactly.
    """
    def __init__(self, rig, anchor_frame, camera, pixel, width, base, depth):
        self.rig, self.anchor_frame = rig, int(anchor_frame)
        self.camera, self.pixel = camera.long(), pixel.long()
        self.uv = torch.stack((self.pixel % width, self.pixel//width), -1).double()
        self.base, self.depth = base.double(), depth.double()

    def evaluate(self, points, theta, eta_depth):
        c = self.camera[points]
        fx, fy, cx, cy = self.rig.K[c].unbind(-1)
        uv0 = self.uv[points]
        z = self.depth[points]-eta_depth
        ray0 = torch.stack(((uv0[:, 0]-cx)/fx, (uv0[:, 1]-cy)/fy, torch.ones_like(z)), -1)
        shift = torch.stack((theta[:, 0]/fx, theta[:, 1]/fy, torch.zeros_like(z)), -1)
        Rt = self.rig.R[c].transpose(1, 2)
        x = self.base[points]+torch.einsum('mij,mj->mi', Rt, -eta_depth[:, None]*ray0+z[:, None]*shift)
        ja = torch.zeros((len(points), 6, 5), dtype=x.dtype, device=x.device)
        ja[:, :3, :2] = Rt[:, :, :2]*torch.stack((z/fx, z/fy), -1)[:, None, :]
        ja[:, 3:, 2:] = torch.eye(3, dtype=x.dtype, device=x.device)
        jl = torch.zeros((len(points), 6, 4), dtype=x.dtype, device=x.device)
        jl[:, :3, 3] = -torch.einsum('mij,mj->mi', Rt, ray0+shift)
        return torch.cat((x, theta[:, 2:]), -1), ja, jl

    def source_codes(self, codec, points):
        return codec.key(points, self.camera[points], torch.full_like(points, self.anchor_frame), self.pixel[points])


def initial_local_gaussian(chart, source_sensor_variance, pixel_covariance, velocity_covariance):
    """Frame-0 local Gaussians: [chart (5) | eta_source (4)], one prior each, no independent Pxx."""
    N = len(chart.camera)
    dev = chart.camera.device
    p = torch.arange(N, device=dev)
    k0 = torch.full_like(p, chart.anchor_frame)
    point = torch.stack((p, p), 1).reshape(-1)
    kind = torch.tensor([CHART, KEY], device=dev).repeat(N)
    camera = torch.stack((torch.zeros_like(p), chart.camera), 1).reshape(-1)
    frame = torch.stack((k0, k0), 1).reshape(-1)
    pixel = torch.stack((torch.zeros_like(p), chart.pixel), 1).reshape(-1)
    g = LocalGaussian.empty(Layout(N, point, kind, camera, frame, pixel))
    block = torch.zeros((N, 9, 9), dtype=torch.float64, device=dev)
    block[:, :2, :2] = torch.as_tensor(pixel_covariance, dtype=torch.float64, device=dev)
    block[:, 2:5, 2:5] = torch.as_tensor(velocity_covariance, dtype=torch.float64, device=dev)
    block[:, 5:, 5:] = torch.diag_embed(source_sensor_variance.double())
    g.cov.copy_(block.reshape(-1))
    return g


def current_jacobian(g, chart, codec, points, mean):
    """d s_current / d a (B,6,n) and s_current at `mean` (B,6), chain rule through the chart."""
    B, n = mean.shape
    first = g.layout.boff[points]
    kind = g.layout.kind[first]
    J = torch.zeros((B, 6, n), dtype=mean.dtype, device=mean.device)
    s = torch.zeros((B, 6), dtype=mean.dtype, device=mean.device)
    st = torch.where(kind == STATE)[0]
    if len(st):
        s[st] = mean[st, :6]
        J[st, :, :6] = torch.eye(6, dtype=mean.dtype, device=mean.device)
    ch = torch.where(kind == CHART)[0]
    if len(ch):
        index = BlockIndex(g.layout, codec, g.layout.kind == KEY)
        src = index.lookup(chart.source_codes(codec, points[ch]))
        if bool((src < 0).any()):
            raise LocalCovarianceFailure('source chart without its source raw-depth error variable')
        sc = g.layout.start[src]
        eta = mean[ch][torch.arange(len(ch), device=mean.device), sc+3]
        state, ja, jl = chart.evaluate(points[ch], mean[ch, :5], eta)
        s[ch] = state
        J[ch, :, :5] = ja
        cols = sc[:, None]+torch.arange(4, device=mean.device)
        J[ch[:, None, None], torch.arange(6, device=mean.device)[None, :, None], cols[:, None, :]] += jl
    return s, J


def predict_local(g, chart, codec, frame, F, Q, budget_bytes):
    """Affine Gaussian propagation: append s_t = F s_{t-1} + w, w ~ N(0,Q) ONCE.
    Cov(s_t, a) = F J Cov(s_{t-1}-chart, a); all earlier cross blocks are unchanged."""
    N = g.N
    dev = g.mean.device
    p = torch.arange(N, device=dev)
    z = torch.zeros_like(p)
    head = Layout(N, p, z, z, torch.full_like(p, int(frame)), z)
    new = LocalGaussian.empty(Layout.concat(head, g.layout))
    F = F.to(g.mean)
    Q = Q.to(g.mean)
    for pts in chunk_plan(p, g.layout.dims+6, budget_bytes):
        mean, cov = g.gather(pts)
        s, J = current_jacobian(g, chart, codec, pts, mean)
        FJ = F@J
        cross = FJ@cov
        top = cross@FJ.transpose(1, 2)+Q
        B, n = mean.shape
        out = torch.zeros((B, n+6, n+6), dtype=cov.dtype, device=dev)
        out[:, :6, :6] = (top+top.transpose(1, 2))/2
        out[:, :6, 6:] = cross
        out[:, 6:, :6] = cross.transpose(1, 2)
        out[:, 6:, 6:] = cov
        new.scatter(pts, torch.cat((s@F.T, mean), 1), out)
    return new


def usable_references(refs, ordinal, lag):
    """References usable at the NEXT processing step: age = (ordinal+1) - r < lag."""
    return (refs.frame >= 0) & (refs.ordinal >= ordinal+2-lag)


def retention_mask(layout, refs, chart, current_frame, ordinal, lag):
    """Keep: current state; any state/chart a camera may still reference; every raw key of a
    camera's still-usable reference frame; the source error while its chart is kept.
    Anything else can never be referenced again by this point (references only move forward)."""
    use = usable_references(refs, ordinal, lag)
    p, kind, cam, frame = layout.point, layout.kind, layout.camera, layout.frame
    referenced = torch.zeros(len(p), dtype=torch.bool, device=p.device)
    for c in range(refs.frame.shape[0]):
        referenced |= (refs.frame[c, p] == frame) & use[c, p]
    keep = torch.zeros_like(referenced)
    keep |= (kind == STATE) & ((frame == int(current_frame)) | referenced)
    keep |= (kind == CHART) & referenced
    own = (kind == KEY) & (refs.frame[cam, p] == frame) & use[cam, p]
    chart_kept = torch.zeros(layout.N, dtype=torch.bool, device=p.device)
    chart_kept[p[(kind == CHART) & keep]] = True
    source = (kind == KEY) & (cam == chart.camera[p]) & (frame == chart.anchor_frame) & (layout.pixel == chart.pixel[p])
    keep |= own | (source & chart_kept[p])
    return keep


def record_expiry(layout, keep, expired_upto, frame_ordinal):
    """Highest processing ordinal of expired raw keys per (camera, point); never recreated."""
    gone = (~keep) & (layout.kind == KEY)
    if bool(gone.any()):
        c, p = layout.camera[gone], layout.point[gone]
        flat = expired_upto.view(-1)
        flat.scatter_reduce_(0, c*expired_upto.shape[1]+p, frame_ordinal[layout.frame[gone]], reduce='amax')
    return int(gone.sum())


def check_no_recreation(point, camera, frame, expired_upto, frame_ordinal):
    bad = frame_ordinal[frame] <= expired_upto[camera, point]
    if bool(bad.any()):
        i = int(torch.where(bad)[0][0])
        raise ForbiddenKeyRecreation(f'FORBIDDEN_EXPIRED_KEY_RECREATION: point {int(point[i])} camera {int(camera[i])} frame '
                                     f'{int(frame[i])} was marginalized; an independent copy would double count its sensor')
    return 0


def batched_eigvalsh(matrix, chunk_size=256):
    """Bounded cuSOLVER batch count (large batched eigvalsh workspaces dominate GPU memory); independent matrices."""
    return torch.cat([torch.linalg.eigvalsh(part) for part in matrix.split(chunk_size)]) if len(matrix) else matrix.new_empty((0, matrix.shape[-1]))


def symmetric_psd_certificate(cov, dims):
    """Correlation-normalized min eigenvalue per point (padding replaced by identity)."""
    B, n, _ = cov.shape
    d = cov.diagonal(dim1=1, dim2=2)
    valid = torch.arange(n, device=cov.device)[None, :] < dims[:, None]
    scale = torch.where(valid & (d > 0), d.clamp_min(torch.finfo(cov.dtype).tiny).rsqrt(), torch.ones_like(d))
    corr = cov*scale[:, :, None]*scale[:, None, :]
    pad = ~valid
    corr = corr+torch.diag_embed(pad.to(cov.dtype))
    asym = (cov-cov.transpose(1, 2)).abs().amax((1, 2))/d.abs().amax(1).clamp_min(torch.finfo(cov.dtype).tiny)
    eig = batched_eigvalsh((corr+corr.transpose(1, 2))/2)
    return eig[:, 0], asym, bool(((d < 0) & valid).any()) or not bool(torch.isfinite(cov).all())


def memory_profile(g):
    d = g.layout.dims.double()
    q = torch.quantile(d, torch.tensor([0., .5, .95, .99, 1.], dtype=torch.float64, device=d.device)).tolist() if len(d) else []
    keys = torch.bincount(g.layout.point[g.layout.kind == KEY], minlength=g.N).double()
    kq = torch.quantile(keys, torch.tensor([0., .5, .95, .99, 1.], dtype=torch.float64, device=d.device)).tolist() if len(d) else []
    return {'points': g.N, 'local_dimension_quantiles_min_median_p95_p99_max': q,
            'raw_keys_per_point_quantiles_min_median_p95_p99_max': kq,
            'reference_state_blocks': int(((g.layout.kind == STATE) & (g.layout.start > 0)).sum()),
            'chart_blocks': int((g.layout.kind == CHART).sum()), 'packed_bytes': g.nbytes,
            'padded_cube_bytes_avoided': int(g.N*(int(g.layout.dims.max()) if g.N else 0)**2*8)}
