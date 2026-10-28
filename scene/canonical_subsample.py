"""Optional deterministic subsampling of ONE measured endpoint cloud before its free filter (contract section 8).

N_target = ceil(N_full / f). f = 1 returns every original point id in its original order (bitwise identity).
For f > 1 a subset of EXISTING canonical ids is returned, sorted ascending, with their original measured XYZ, colour,
source pixel and covariance untouched: nothing is averaged, interpolated or invented.

spatial-stratified (default), label free, canonical endpoint data only:
  1. voxel hashing of canonical XYZ; the voxel edge h is found by bisection so that the number of occupied voxels V(h)
     is the largest value <= N_target (O(N log N) per evaluation, no N^2 distances);
  2. one representative per occupied voxel (every voxel, so no spatial component disappears);
  3. the remaining N_target - V slots are filled by priority. Priority (when boundary preservation is on) favours
     points whose SOURCE pixel in the endpoint image lies on the measured silhouette (a 4-neighbour outside the
     valid foreground) or on a measured depth break (same-surface rule of configs/projective_rgbd.json), then larger
     measured RGB gradient magnitude at the source pixel; ties and the rest follow a seeded hash;
     within-voxel rank (second pick of a voxel before third) precedes priority so fills stay spatially spread.
seeded-uniform (diagnostic baseline): a seeded permutation, first N_target ids, sorted.

Diagnostics (written to the subsample manifest): N_full, N_target, N_selected, actual reduction, voxel size, occupied
voxels, per-camera source counts full vs selected, image-grid (16 px tile) coverage per camera full vs selected,
voxel-graph component sizes full vs selected, boundary points full vs selected.
"""
import hashlib
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _voxel_keys(xyz, h, origin):
    q = np.floor((xyz-origin)/h).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0)+1
    return (q[:, 0]*dims[1]+q[:, 1])*dims[2]+q[:, 2], q, dims


def _hash_order(n, seed):
    """Seeded pseudo-random tie-break per point id (deterministic, platform independent)."""
    return np.random.default_rng(np.random.SeedSequence([seed, n])).permutation(n)


def boundary_priority(dataset, source, endpoint_frame, policy, noise):
    """(boundary flag, rgb gradient magnitude) per point from its SOURCE endpoint image only."""
    n = len(source)
    flag = np.zeros(n, bool); grad = np.zeros(n)
    for ci in np.unique(source[:, 1]):
        use = np.flatnonzero(source[:, 1] == ci)
        f = dataset.observation(endpoint_frame, int(ci))
        H, W = f.depth.shape
        valid = np.asarray(f.valid, bool); depth = np.asarray(f.depth, float); rgb = np.asarray(f.rgb, float)
        fx, fy = f.camera.intrinsics[:2]
        pix = source[use, 2]; r, c = pix//W, pix % W
        sil = np.zeros(len(use), bool); brk = np.zeros(len(use), bool)
        z = depth[r, c]
        allow = policy['same_surface_n_sigma']*np.sqrt(noise.depth_variance+noise.depth_squared_variance*z*z) + \
            z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))/min(fx, fy)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = r+dr, c+dc
            inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr, cc = rr.clip(0, H-1), cc.clip(0, W-1)
            nb_valid = inside & valid[rr, cc]
            sil |= ~nb_valid
            brk |= nb_valid & (np.abs(depth[rr, cc]-z) > allow)
        gx = np.zeros_like(z); gy = np.zeros_like(z)
        for (dr, dc), out in (((0, 1), gx), ((1, 0), gy)):
            a = (r+dr).clip(0, H-1), (c+dc).clip(0, W-1); b = (r-dr).clip(0, H-1), (c-dc).clip(0, W-1)
            ok = valid[a] & valid[b]
            out[ok] = np.linalg.norm(rgb[a][ok]-rgb[b][ok], axis=1)/2
        flag[use] = sil | brk; grad[use] = np.hypot(gx, gy)
    return flag, grad


def subsample(xyz, source, factor, method='spatial-stratified', seed=0, boundary=None, rgb_grad=None, report=True, widths=None):
    """Returns (selected ids sorted ascending [n], manifest dict)."""
    n = len(xyz)
    if int(factor) != factor or factor < 1: raise ValueError('--pcd-subsample-factor must be an integer >= 1')
    factor = int(factor)
    target = math.ceil(n/factor)
    man = {'factor': factor, 'method': method, 'seed': seed, 'N_full': n, 'N_target': target}
    if factor == 1:
        sel = np.arange(n)
        man.update(N_selected=n, actual_reduction=1., note='factor 1: identity, every original id in original order')
        return sel, man
    if method == 'seeded-uniform':
        sel = np.sort(_hash_order(n, seed)[:target])
        man.update(note='diagnostic baseline: seeded permutation, no spatial stratification')
    elif method == 'spatial-stratified':
        origin = xyz.min(0)
        ext = float(np.ptp(xyz, axis=0).max())
        lo, hi = ext*1e-6, ext*2
        count = lambda h: len(np.unique(_voxel_keys(xyz, h, origin)[0]))
        for _ in range(60):                       # largest V(h) <= target (V decreases with h)
            mid = math.sqrt(lo*hi)
            if count(mid) > target: lo = mid
            else: hi = mid
            if hi/lo < 1+1e-6: break
        h = hi
        keys, q, dims = _voxel_keys(xyz, h, origin)
        tie = np.empty(n, np.int64); tie[_hash_order(n, seed)] = np.arange(n)
        bflag = np.zeros(n, bool) if boundary is None else np.asarray(boundary, bool)
        g = np.zeros(n) if rgb_grad is None else np.asarray(rgb_grad, float)
        # rank points inside their voxel by priority: boundary first, then rgb gradient, then hash
        order = np.lexsort((tie, -g, ~bflag, keys))
        k_sorted = keys[order]
        first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
        rank = np.empty(n, np.int64); rank[order] = np.arange(n)-start
        reps = rank == 0
        V = int(reps.sum())
        rest = np.flatnonzero(~reps)
        need = target-V
        if boundary is not None:   # boundary samples first, then within-voxel rank, then gradient, then hash
            fill = rest[np.lexsort((tie[rest], -g[rest], rank[rest], ~bflag[rest]))][:need]
        else:
            fill = rest[np.lexsort((tie[rest], rank[rest]))][:need]
        sel = np.sort(np.r_[np.flatnonzero(reps), fill])
        man.update(voxel_size_m=h, occupied_voxels=V, filled_after_one_per_voxel=int(len(fill)),
                   boundary_preserve=boundary is not None,
                   boundary_points_full=int(bflag.sum()), boundary_points_selected=int(bflag[sel].sum()))
        if report: man['voxel_components'] = _components(q, dims, keys, sel)
    else:
        raise ValueError(f'unknown --pcd-subsample-method {method!r}')
    if len(sel) != target: raise AssertionError(f'subsample selected {len(sel)} != target {target}')
    man.update(N_selected=int(len(sel)), actual_reduction=n/len(sel))
    if report and source is not None: man['camera_coverage'] = _camera_coverage(source, sel, widths)
    return sel, man


def _components(q, dims, keys, sel):
    """26-connected occupied-voxel components: points per component, full vs selected."""
    uk, inv = np.unique(keys, return_inverse=True)
    lut = {int(k): i for i, k in enumerate(uk)}
    vq = np.zeros((len(uk), 3), np.int64); vq[inv] = q
    edges = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0): continue
                nb = vq+[dx, dy, dz]
                ok = (nb >= 0).all(1) & (nb < dims).all(1)
                nk = (nb[:, 0]*dims[1]+nb[:, 1])*dims[2]+nb[:, 2]
                j = np.searchsorted(uk, nk); j = j.clip(0, len(uk)-1)
                hit = ok & (uk[j] == nk)
                edges.append(np.c_[np.flatnonzero(hit), j[hit]])
    e = np.concatenate(edges) if edges else np.zeros((0, 2), int)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(uk), len(uk)))
    nc, lab = connected_components(g, directed=False)
    full = np.bincount(lab[inv], minlength=nc); s = np.bincount(lab[inv][sel], minlength=nc)
    o = np.argsort(-full)
    return {'components': int(nc), 'components_lost': int((s == 0).sum()),
            'largest_10_full_vs_selected': [[int(full[i]), int(s[i])] for i in o[:10]],
            'min_selected_fraction': float((s/np.maximum(full, 1)).min())}


def _camera_coverage(source, sel, widths=None, tile=16):
    out = []
    for ci in np.unique(source[:, 1]):
        m = source[:, 1] == ci
        W = None if widths is None else int(widths[int(ci)])
        def tiles(ids):          # 16 x 16 pixel tiles of the source image holding at least one point
            p = source[ids, 2]
            return len(np.unique(p//tile)) if W is None else len(np.unique((p//W//tile)*100000+(p % W)//tile))
        full_ids = np.flatnonzero(m); sel_ids = sel[m[sel]]
        out.append({'camera_index': int(ci), 'full': int(m.sum()), 'selected': int(len(sel_ids)),
                    'selected_fraction': float(len(sel_ids)/max(m.sum(), 1)),
                    'image_tiles_full': tiles(full_ids), 'image_tiles_selected': tiles(sel_ids) if len(sel_ids) else 0})
    return out


def ids_hash(ids):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, np.int64)).tobytes()).hexdigest()
