"""Operator D: whole-sequence pairwise rigidity on a candidate reference graph (PDF (23)-(26), (36)-(38); spec D1-D4).

Graph: exact k nearest neighbours of the frame-0 canonical XYZ (computational hypothesis generator; k is a declared
search parameter, an edge is NOT a same-part label). Undirected, no self edges; exactly coincident pairs are dropped
(undefined distance Jacobian) and counted.
    d_il,t = |x_it - x_lt|,  e_il,t = d_il,t - d_il,0 (t = 1..T),  n_t = (x_it - x_lt)/d_il,t
    de_t = n_t^T (dx_it - dx_lt) - n_0^T (dx_i0 - dx_l0)                                   (E22, exact first order)
    Omega = J P_zeta J^T                                                                    (E23, exact given P_zeta)
Production covariance (DEV-0007, MODEL_APPROX): P_zeta with trajectory states independent across time and points and
independent of the frame-0 state, marginals P_t from the prior propagation (data information omitted: conservative).
Then Omega = c 1 1^T + diag(delta_t) with the SHARED reference c = n_0^T (P_i0 + P_l0) n_0 counted once and
delta_t = n_t^T (P_it + P_lt) n_t (+ Omega_model); its inverse and log det are exact by Sherman-Morrison:
    e^T Omega^-1 e = sum e_t^2/delta_t - c (sum e_t/delta_t)^2 / (1 + c sum 1/delta_t),
    log det Omega = sum log delta_t + log(1 + c sum 1/delta_t)
(no E x T x T array). rigid_edge_loglik is the conditional same-rigid likelihood N_T(e; 0, Omega) (D3): static pairs
satisfy it too, so it is never a moving-membership statistic by itself.
"""
import math
import numpy as np
import torch


def knn_graph(x0, k, chunk=512):
    N = len(x0); k = min(k, N-1)
    if k < 1: return torch.zeros(0, 2, dtype=torch.int64, device=x0.device), {'coincident_pairs_dropped': 0, 'k': k}
    nb = torch.empty(N, k, dtype=torch.int64, device=x0.device); coincident = 0
    for i in range(0, N, chunk):
        D = torch.cdist(x0[i:i+chunk], x0, compute_mode='donot_use_mm_for_euclid_dist')   # exact differences (mm path cancels)
        D[torch.arange(len(D)), torch.arange(i, i+len(D))] = float('inf')
        val, idx = torch.topk(D, k, largest=False)                 # ties broken by torch (deterministic on a fixed device)
        coincident += int((val == 0).sum()); idx[val == 0] = -1; nb[i:i+chunk] = idx
    rows = torch.arange(N, device=x0.device)[:, None].expand(N, k)
    e = torch.stack([rows.reshape(-1), nb.reshape(-1)], 1); e = e[e[:, 1] >= 0]
    e = torch.sort(e, 1).values; e = torch.unique(e, dim=0)
    return e, {'k': k, 'coincident_pairs_dropped': coincident, 'edges': int(len(e)), 'method': 'exact kNN of canonical frame-0 XYZ (torch.cdist, float64)'}


def edge_distances(X, edges):
    return torch.linalg.norm(X[:, edges[:, 0]]-X[:, edges[:, 1]], dim=-1).T          # [E, T+1]


def edge_residuals(X, edges):
    d = edge_distances(X, edges)
    return d[:, 1:]-d[:, :1], d


def edge_jacobian(X, edges):
    """Unit relative vectors n_t [E, T+1, 3]; de_t/dx_it = n_t, /dx_lt = -n_t, /dx_i0 = -n_0, /dx_l0 = n_0."""
    r = (X[:, edges[:, 0]]-X[:, edges[:, 1]]).transpose(0, 1)
    return r/torch.linalg.norm(r, dim=-1, keepdim=True)


def dense_edge_covariance(X, edge, P_zeta):
    """Exact first-order Omega [T,T] for ONE edge given the joint covariance P_zeta of zeta = (x_i0, x_l0, x_i1, x_l1, ...,
    x_iT, x_lT) [6(T+1), 6(T+1)] (oracle / tiny problems only)."""
    n = edge_jacobian(X, edge[None])[0]; T = X.shape[0]-1
    J = torch.zeros(T, 6*(T+1), dtype=X.dtype, device=X.device)
    for t in range(1, T+1):
        J[t-1, 6*t:6*t+3] = n[t]; J[t-1, 6*t+3:6*t+6] = -n[t]; J[t-1, 0:3] = -n[0]; J[t-1, 3:6] = n[0]
    return J@P_zeta@J.T


def edge_covariance_structured(X, edges, P_pos, omega_model=0.):
    """(c [E], delta [E,T]) of Omega = c 1 1^T + diag(delta); P_pos [T+1, N, 3, 3] marginal position covariances."""
    n = edge_jacobian(X, edges)
    def proj(t):
        P = P_pos[t][edges[:, 0]]+P_pos[t][edges[:, 1]]
        return torch.einsum('ei,eij,ej->e', n[:, t], P, n[:, t])
    c = proj(0)
    delta = torch.stack([proj(t) for t in range(1, X.shape[0])], 1)+omega_model
    return c, delta


def structured_quad_logdet(e, c, delta):
    s1 = (1/delta).sum(1); se = (e/delta).sum(1); den = 1+c*s1
    return (e*e/delta).sum(1)-c*se*se/den, torch.log(delta).sum(1)+torch.log(den)


def rigid_edge_loglik(e, c, delta):
    q, ld = structured_quad_logdet(e, c, delta)
    return -.5*q-.5*ld-.5*e.shape[1]*math.log(2*math.pi), q


def differential_rigidity(X, V, edges):
    """(x_i - x_l)^T (v_i - v_l) [E, T+1] (E20 diagnostic; 0 for shared instantaneous rigid motion)."""
    return ((X[:, edges[:, 0]]-X[:, edges[:, 1]])*(V[:, edges[:, 0]]-V[:, edges[:, 1]])).sum(-1).T


def prior_position_marginals(temporal):
    """P_t = F P_t-1 F^T + Q from the temporal prior's P_i0 (positions block) [T+1, N, 3, 3] (DEV-0007 input)."""
    P = torch.cholesky_inverse(torch.linalg.cholesky(temporal.P0i)); out = [P[:, :3, :3].clone()]
    for F, Qi in zip(temporal.F, temporal.Qi):
        Q = torch.cholesky_inverse(torch.linalg.cholesky(Qi))
        P = F@P@F.T+Q; P = (P+P.transpose(-1, -2))/2; out.append(P[:, :3, :3].clone())
    return torch.stack(out)
