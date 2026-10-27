"""Coherence on rendered-image Phase-1 trajectories: the pinned RigEuler3DGS 481b2cf2 pipeline
(build_supernodes(mu0, num_supernodes=192, seed=0) -> canonical_knn_graph(k=12) -> edge_strain ->
extract_moving_component(folds=5)) with the information-supported RGB-D adaptation of supported_coherence:
node displacements from image-supported members only, node-frame support weights in every statistic,
edges without evidence removed (derivation, reduction to the pinned functions and occlusion simulations:
supported_coherence.py, tests/test_image_coherence.py).

Documented differences from the pinned CALL:
1. the canonical graph and Kabsch gauge use the direction's actual endpoint (frame 0 forward, final frame backward);
2. the image-support weighting above (replaces the earlier ad-hoc visibility x precision weights and zero-order hold).
Representative nodes are an independent fixed-count graph, never the Phase-1 correction groups and never a part
count. MOVING_RIGID, STATIC_COMPATIBLE and INCOHERENT_OR_AMBIGUOUS stay distinct; the pinned extractor resolves a
component even on static noise-only scenes (test_pinned_extractor_resolves_a_component_on_static_noise), so a static
scene must be rejected by the joint stage's predictive comparison.
"""
from pathlib import Path
import numpy as np
import torch
from .group_visualization import serial
from .store import json_write

REVISION = '481b2cf2d5602932286948cf14114f9caee4de3c'


@torch.no_grad()
def build_supernodes(means, num_supernodes=192, iterations=25, seed=0):
    """Pinned estimator/supernodes.build_supernodes: farthest-point seeding + Lloyd (assignment only)."""
    x = torch.as_tensor(means).float()
    n = int(x.shape[0]); c = max(1, min(int(num_supernodes), n))
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=g).item())
    seeds = [first]; d = (x-x[first]).norm(dim=1)
    for _ in range(c-1):
        nxt = int(d.argmax()); seeds.append(nxt); d = torch.minimum(d, (x-x[nxt]).norm(dim=1))
    centroids = x[torch.tensor(seeds)].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(int(iterations)):
        new = torch.cat([torch.cdist(x[s:s+8192], centroids).argmin(dim=1) for s in range(0, n, 8192)])
        if torch.equal(new, assign): break
        assign = new
        for k in range(c):
            m = assign == k
            if bool(m.any()): centroids[k] = x[m].mean(dim=0)
    counts = torch.bincount(assign, minlength=c); keep = counts > 0
    if not bool(keep.all()):
        remap = -torch.ones(c, dtype=torch.long); remap[keep] = torch.arange(int(keep.sum())); assign = remap[assign]
    return assign


def canonical_knn_graph(x0, k=12):
    """Pinned motion_coherence_support.canonical_knn_graph: symmetric kNN built ONCE."""
    x = torch.as_tensor(x0).double().cpu(); n = x.shape[0]; k = max(1, min(int(k), n-1))
    d = torch.cdist(x, x); d.fill_diagonal_(float('inf'))
    nb = d.topk(k, largest=False).indices
    src = torch.arange(n).repeat_interleave(k); dst = nb.reshape(-1)
    key = torch.unique(torch.minimum(src, dst)*n+torch.maximum(src, dst))
    return key//n, key % n


def run(partition, arrays, times, root, endpoint, count=192, knn=12, folds=5):
    from .supported_coherence import supported_node_trajectories, weighted_edge_strain, extract_moving_component
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    K = len(times)
    if endpoint not in (0, K-1): raise ValueError('Coherence requires the actual first/last source endpoint')
    sid = build_supernodes(partition.xyz, count)
    observed = np.asarray(arrays['point_observed'], bool)
    traj, omega, node_ids, members = supported_node_trajectories(np.asarray(arrays['xyz']), observed, sid, endpoint)
    n = int(node_ids.numel())
    flip = (lambda a: a) if endpoint == 0 else (lambda a: a.flip(0))
    directed, om = flip(traj), flip(omega)                          # processing order: index 0 is the gauge
    ei, ej = canonical_knn_graph(directed[0], knn)
    st = weighted_edge_strain(directed, om, ei, ej)
    keep = st['supported_edges']
    if n < 4:
        result = {'status': 'NO_COHERENT_MOVING_COMPONENT', 'reason': 'fewer than four representative nodes', 'labels': [], 'moving_nodes': torch.zeros(0, dtype=torch.long)}
    else:
        result = extract_moving_component(directed, om, ei[keep], ej[keep], st['C'][keep], n, folds=folds)
    moving = node_ids[result['moving_nodes']].numpy() if len(result['moving_nodes']) else np.zeros(0, int)
    support = np.isin(sid.numpy(), moving)
    labels = np.array(result.get('labels', ['INCOHERENT_OR_AMBIGUOUS']*n))
    np.savez_compressed(root/'representatives.npz', membership=sid.numpy(), node_ids=node_ids.numpy(), trajectory=traj.numpy(),
                        node_support=omega.numpy(), members=members.numpy(), edges=np.c_[ei.numpy(), ej.numpy()], edge_supported=keep.numpy(),
                        C=st['C'].numpy(), log_strain=st['z'].numpy(), edge_weights=st['weights'].numpy(), labels=labels, moving_support=support)
    core = np.isin(node_ids.numpy(), moving)
    disp = (traj-traj[endpoint]).norm(dim=2).numpy()
    result.update(point_support=support, representative_membership=sid.numpy(), reference_revision=REVISION, representative_nodes=n,
                  representative_count_requested=count, knn=knn, folds=folds, anchor_frame=endpoint, computational_groups=partition.count,
                  core_points=int(support.sum()), core_nodes=int(core.sum()), edges_total=int(len(ei)), edges_supported=int(keep.sum()),
                  node_frame_support={'core_mean': float(omega[:, core].mean()) if core.any() else None,
                                      'non_core_mean': float(omega[:, ~core].mean()) if (~core).any() else None,
                                      'nodes_never_supported_after_gauge': int((omega.sum(0) <= 1).sum())},
                  core_max_node_displacement_m=float(disp[:, core].max()) if core.any() else None,
                  adaptation='information-supported (supported_coherence): node-frame image support weights; unsupported edges removed',
                  semantics='Pinned held-out representative-node selection on rendered-image IEKF trajectories; cross-validation splits nodes, never cameras')
    json_write(root/'result.json', serial({k: v for k, v in result.items() if k not in ('point_support', 'representative_membership', 'moving_nodes')}))
    return result
