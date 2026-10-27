"""Computational supernodes and the grouped correction basis (RigEuler3DGS 481b2cf2 estimator/supernodes.py).

build_supernodes is the pinned function line for line (farthest-point seeding from one seeded index,
Lloyd iterations on canonical geometry, empty groups dropped, centroid kNN adjacency); only the CUDA
expand/reduce kernels are replaced by their torch equivalent.

The basis is the RAW reference tangent zeta_g = [dt, dtheta, du, domega] (12 per group):

    dx_i = dt_g + dtheta_g x (x_i^can - c_g),     dv_i = du_g + domega_g x (x_i^can - c_g)

It is a change of variables of the Phase-1 correction only (contract section 4). A supernode is
never a physical rigid body and has no persistent pose: the state stays the free per-point [x, v].
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np
import torch
from .rgbd_point_renderer import _segments, _segsum
from . import small_contractions as sc


@dataclass
class SupernodePartition:
    supernode_id: torch.Tensor      # [N] long
    centroids: torch.Tensor         # [C,3]
    member_counts: torch.Tensor     # [C]
    radius: torch.Tensor            # [C]
    adjacency: torch.Tensor         # [E,2]
    canonical_means: torch.Tensor   # [N,3]
    config: dict = field(default_factory=dict)

    @property
    def num_supernodes(self):
        return int(self.centroids.shape[0])

    def canonical_hash(self):
        h = hashlib.sha256()
        for a in (self.canonical_means, self.supernode_id, self.centroids): h.update(a.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def to_dict(self):
        return {'num_supernodes': self.num_supernodes, 'num_points': int(len(self.supernode_id)),
                'member_count_quantiles': np.quantile(self.member_counts.cpu().numpy(), [0, .5, 1]).tolist(),
                'radius_quantiles_m': np.quantile(self.radius.cpu().numpy(), [0, .5, 1]).tolist(),
                'num_edges': int(len(self.adjacency)), 'canonical_hash': self.canonical_hash(), 'config': self.config}


@torch.no_grad()
def build_supernodes(means, *, target_size=64, num_supernodes=None, knn=8, iterations=25, seed=0):
    x = torch.as_tensor(means).detach()
    n = int(x.shape[0])
    c = int(num_supernodes) if num_supernodes else max(1, n//max(target_size, 1))
    c = max(1, min(c, n))
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=g).item())
    seeds = [first]
    d = (x-x[first]).norm(dim=1)
    for _ in range(c-1):
        nxt = int(d.argmax()); seeds.append(nxt)
        d = torch.minimum(d, (x-x[nxt]).norm(dim=1))
    centroids = x[torch.tensor(seeds, device=x.device)].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(int(iterations)):
        new = torch.cat([torch.cdist(x[s:s+8192], centroids).argmin(dim=1) for s in range(0, n, 8192)])
        if torch.equal(new, assign): break
        assign = new
        for k in range(c):
            m = assign == k
            if bool(m.any()): centroids[k] = x[m].mean(dim=0)
    counts = torch.bincount(assign, minlength=c); keep = counts > 0
    if not bool(keep.all()):
        remap = -torch.ones(c, dtype=torch.long, device=x.device); remap[keep] = torch.arange(int(keep.sum()), device=x.device)
        assign, centroids, counts, c = remap[assign], centroids[keep], counts[keep], int(keep.sum())
    radius = torch.zeros(c, device=x.device, dtype=x.dtype)
    for k in range(c):
        m = assign == k
        if bool(m.any()): radius[k] = (x[m]-centroids[k]).norm(dim=1).max()
    kk = min(int(knn), max(c-1, 1)); edges = []
    if c > 1:
        dist = torch.cdist(centroids, centroids); dist.fill_diagonal_(float('inf'))
        for a, row in enumerate(dist.topk(kk, dim=1, largest=False).indices.tolist()):
            edges += [(min(a, b), max(a, b)) for b in row if a != b]
        edges = sorted(set(edges))
    adjacency = torch.tensor(edges, dtype=torch.long, device=x.device) if edges else torch.zeros((0, 2), dtype=torch.long, device=x.device)
    return SupernodePartition(assign, centroids, counts, radius, adjacency, x.clone(),
                              {'target_size': target_size, 'num_supernodes': c, 'knn': knn, 'iterations': iterations, 'seed': seed})


def skew(r):
    out = r.new_zeros((*r.shape, 3)); x, y, z = r.unbind(-1)
    out[..., 0, 1], out[..., 0, 2], out[..., 1, 0], out[..., 1, 2], out[..., 2, 0], out[..., 2, 1] = -z, y, z, -x, -y, x
    return out


class GroupBasis:
    """delta s = G zeta with the raw reference tangent; G^T is its exact adjoint."""
    def __init__(self, canonical, membership, centroids):
        self.member = membership.long()
        self.count = int(centroids.shape[0])
        self.lever = canonical.to(torch.float64)-centroids.to(canonical.device, torch.float64)[self.member]   # x_i^can - c_g
        self.seg = _segments(self.member)
        p = self.lever.new_zeros((len(self.lever), 3, 6)); p[:, :, :3] = torch.eye(3, dtype=p.dtype, device=p.device); p[:, :, 3:] = -skew(self.lever)
        self.pos = p                                              # [N,3,6]: dx_i = pos_i [dt; dtheta]

    def matrix(self):
        """[N,6,12] per-point block of G (tests and dense oracles)."""
        out = self.pos.new_zeros((len(self.pos), 6, 12)); out[:, :3, :6] = self.pos; out[:, 3:, 6:] = self.pos
        return out

    def expand(self, z):
        zi = z[self.member]
        return torch.cat([sc.matvec(self.pos, zi[:, :6]), sc.matvec(self.pos, zi[:, 6:])], 1)

    def reduce(self, f):
        return _segsum(torch.cat([sc.rmatvec(self.pos, f[:, :3]), sc.rmatvec(self.pos, f[:, 3:])], 1), self.seg, self.count)

    def blocks(self, M):
        """G^T M G per group for per-point [N,6,6] blocks M -> [C,12,12]."""
        if sc.is_on():                                            # G_i = blockdiag(pos_i, pos_i): four [N,6,6] congruences, no [N,6,12]
            p = self.pos; out = M.new_empty((len(M), 12, 12))
            for a in (0, 1):
                for b in (0, 1): out[:, 6*a:6*a+6, 6*b:6*b+6] = sc.gram(p, sc.gram(M[:, 3*a:3*a+3, 3*b:3*b+3].transpose(1, 2), p))
            return _segsum(out, self.seg, self.count)
        g = self.matrix()
        return _segsum(g.transpose(1, 2)@M@g, self.seg, self.count)

    def null_projector(self, rel=1e-10):
        """Per group orthogonal projector onto null(G_g) (single points, collinear groups). [C,12,12]"""
        gram = self.blocks(torch.eye(6, dtype=self.pos.dtype, device=self.pos.device).expand(len(self.pos), 6, 6))
        vals, vecs = torch.linalg.eigh(gram.cpu())
        null = vals <= vals[:, -1:].clamp_min(1e-300)*rel
        return ((vecs*null[:, None, :].to(vecs.dtype))@vecs.transpose(1, 2)).to(gram.device), int(null.sum())
