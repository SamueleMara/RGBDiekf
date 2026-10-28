# Exact reference trajectory reduction and log-distance invariant.
import torch
from torch import Tensor
EPS=1e-12

def supernode_trajectories(positions, supernode_id: Tensor,
                           weights: Tensor | None = None) -> dict:
    """`x_s(t)` from the fixed Phase-1 supernode membership.

        x_s(t) = sum_{i in s} w_i x_i(t) / sum_{i in s} w_i

    The weight convention is uniform `w_i = 1` unless the caller supplies one,
    and it is recorded in the output so the reduction is reproducible. The
    membership is Phase-1's own and is never recomputed here -- an edge must
    refer to the same two supernodes for the whole sequence (section 4).
    """
    sid = torch.as_tensor(supernode_id).long().reshape(-1).cpu()
    valid = sid >= 0
    n_s = int(sid[valid].max()) + 1 if int(valid.sum()) else 0
    w = (torch.ones(sid.numel(), dtype=torch.float64) if weights is None
         else torch.as_tensor(weights).double().reshape(-1).cpu())
    w = torch.where(valid, w, torch.zeros_like(w))
    denom = torch.zeros(n_s, dtype=torch.float64).index_add_(
        0, sid.clamp_min(0), w)
    traj = []
    for p in positions:
        x = torch.tensor(p, dtype=torch.float64).cpu()
        acc = torch.zeros((n_s, 3), dtype=torch.float64).index_add_(
            0, sid.clamp_min(0), x * w[:, None])
        traj.append(acc / denom.clamp_min(EPS)[:, None])
    keep = denom > 0
    return {"trajectories": torch.stack(traj),          # [T, n_s, 3]
            "valid": keep, "num_supernodes": n_s,
            "weight_convention": ("uniform w_i = 1" if weights is None
                                  else "caller-supplied"),
            "members_per_supernode": torch.zeros(n_s, dtype=torch.long)
            .index_add_(0, sid.clamp_min(0), valid.long())}



def edge_strain(traj: Tensor, edge_i: Tensor, edge_j: Tensor,
                times: Tensor | None = None, w_t: Tensor | None = None,
                eps_len: float | None = None) -> dict:
    """`l_ij(t)`, `z_ij(t)`, `C_ij`, `eps_ij(t)`, `J_ij`.

    `z_ij(t) = log((l + eps) / (lbar + eps))` with `lbar = median_t l` -- the
    median over the WHOLE window, not frame 0 and not the endpoint, so an edge
    that moves and returns (section 36 S6) is still measured against its own
    typical length, and an isolated bad frame cannot set the reference.

    `eps_len` is a numerical guard derived from the data scale, never a motion
    threshold: it only keeps the log finite for degenerate zero-length edges.
    """
    X = torch.as_tensor(traj).double()                    # [T, n, 3]
    T = X.shape[0]
    r = X[:, edge_j, :] - X[:, edge_i, :]                 # [T, E, 3]
    ell = r.norm(dim=2)                                   # [T, E]
    if eps_len is None:
        eps_len = float(ell[ell > 0].median()) * 1e-9 if (ell > 0).any() else EPS
    lbar = ell.median(dim=0).values                       # [E]
    z = torch.log((ell + eps_len) / (lbar + eps_len).clamp_min(EPS))

    w = (torch.ones_like(z) if w_t is None
         else torch.as_tensor(w_t).double().reshape(T, -1).expand_as(z))
    C = ((w * z * z).sum(0) / (w.sum(0) + EPS)).clamp_min(0.0).sqrt()

    t_arr = (torch.arange(T, dtype=torch.float64) if times is None
             else torch.as_tensor(times).double().reshape(-1))
    dt = (t_arr[1:] - t_arr[:-1]).clamp_min(EPS)[:, None]
    dz = z[1:] - z[:-1]
    epsilon = dz / dt                                     # [T-1, E]
    J = ((dz * dz).sum(0) / max(T - 1, 1)).sqrt()
    return {"length": ell, "z": z, "C": C, "epsilon": epsilon, "J": J,
            "reference_length": lbar, "eps_len": eps_len}


