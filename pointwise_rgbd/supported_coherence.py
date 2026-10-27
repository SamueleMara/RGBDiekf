"""Information-supported motion coherence (contract section 7 RGB-D adaptation of the pinned extractor).

Problem. A Phase-1 point that has no active rendered-image pixel at frame k carries no image evidence about its
position there: its estimate is the constant-velocity prediction (growing covariance). Such prediction-only
trajectories are smooth and mutually coherent, so the pinned extractor, which treats every node-frame as an
observation, can read them as a rigid moving body (measured on the blade before this adaptation: a drift
component of 7% of the blade was selected). A zero-order hold instead freezes them, which manufactures strain
whenever the hidden material actually moved.

Derivation. Let o_i(k) = 1 iff point i had image support at frame k (frozen-linearization contributor of some
camera); the endpoint frame is the canonical gauge (o = 1). For representative node s with members M_s:

    omega_s(k) = |{i in M_s : o_i(k)}| / |M_s|                                           (node-frame support)
    x_s(k)     = xbar_s(k_e) + sum_i o_i(k) (x_i(k) - x_i(k_e)) / sum_i o_i(k)            (omega_s(k) > 0)

i.e. the node moves by the mean displacement of its image-supported members (a subset centroid would jump when
the visible subset changes). Every statistic of the pinned extractor becomes a weighted sum over node-frames:

    edge strain   w_ab(k) = omega_a(k) omega_b(k);  lbar_ab = median over supported frames;
                  C_ab = sqrt(sum_k w z^2 / sum_k w)   (the pinned edge_strain w_t form); an edge supported at no
                  frame other than the gauge (where its strain is zero by construction) is REMOVED: no evidence is
                  not zero strain
    Kabsch        weighted centroid and cross-covariance; a frame needs >= 3 supported nodes
    held-out CV   RSS sums weighted by omega; frames without a fit are skipped; BIC with N_eff = sum omega
    complement    the reference frame at k is fitted on supported complement nodes; if fewer than 3, frame k
                  carries no relative evidence (weight 0 for every candidate node)
    refinement    E^M, E^I weighted by omega; a node with no support is never recruited (0 < 0 is false)

With omega == 1 every quantity equals the pinned one (tests/test_image_coherence.py). The candidate hierarchy,
edge distributions and the selection rule are the pinned functions unchanged.
"""
import math
import torch
from .rigeuler_ref.coherence_extraction import (candidate_hierarchy, edge_distributions, select_moving_component,
                                                MOVING_RIGID, STATIC_COMPATIBLE, INCOHERENT)

EPS = 1e-12


def supported_node_trajectories(xyz, observed, sid, endpoint):
    """xyz [K,N,3], observed [K,N] bool, sid [N] -> (traj [K,n,3], omega [K,n], node ids, members per node)."""
    X = torch.as_tensor(xyz, dtype=torch.float64); o = torch.as_tensor(observed, dtype=torch.float64).clone()
    o[endpoint] = 1.
    sid = torch.as_tensor(sid).long()
    n = int(sid.max())+1
    members = torch.bincount(sid, minlength=n).double()
    disp = X-X[endpoint]
    num = torch.zeros(X.shape[0], n, 3, dtype=torch.float64).index_add_(1, sid, disp*o[..., None])
    cnt = torch.zeros(X.shape[0], n, dtype=torch.float64).index_add_(1, sid, o)
    base = torch.zeros(n, 3, dtype=torch.float64).index_add_(0, sid, X[endpoint])/members.clamp_min(1)[:, None]
    traj = base[None]+num/cnt.clamp_min(1)[..., None]
    keep = members > 0
    return traj[:, keep], (cnt/members.clamp_min(1))[:, keep], torch.nonzero(keep).reshape(-1), members[keep]


def weighted_edge_strain(traj, omega, edge_i, edge_j, gauge=0):
    """traj, omega in processing order with the gauge frame at index `gauge` (0)."""
    X = torch.as_tensor(traj).double(); W = omega[:, edge_i]*omega[:, edge_j]                 # [T,E]
    ell = (X[:, edge_j]-X[:, edge_i]).norm(dim=2)
    eps_len = float(ell[ell > 0].median())*1e-9 if (ell > 0).any() else EPS
    lbar = torch.stack([ell[W[:, e] > 0, e].median() if bool((W[:, e] > 0).any()) else ell[:, e].median() for e in range(ell.shape[1])])
    z = torch.log((ell+eps_len)/(lbar+eps_len).clamp_min(EPS))
    support = W.sum(0)
    C = ((W*z*z).sum(0)/(support+EPS)).clamp_min(0.).sqrt()
    evidence = torch.cat([W[:gauge], W[gauge+1:]]).sum(0)
    return {'length': ell, 'z': z, 'C': C, 'weights': W, 'support': support, 'reference_length': lbar, 'eps_len': eps_len,
            'supported_edges': evidence > 0}


def _kabsch(P, Q, w=None):
    """Pinned coherence_extraction._kabsch with optional node weights (identical when w is None)."""
    if w is None:
        pc, qc = P.mean(0), Q.mean(0)
        H = (P-pc).T@(Q-qc)
    else:
        sw = w.sum()
        pc, qc = (w[:, None]*P).sum(0)/sw, (w[:, None]*Q).sum(0)/sw
        H = (w[:, None]*(P-pc)).T@(Q-qc)
    U, _S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T@U.T))
    D = torch.diag(torch.tensor([1., 1., float(d)], dtype=torch.float64))
    R = Vt.T@D@U.T
    return R, qc-R@pc


def _fit(P, Q, w):
    """Weighted Kabsch over supported rows; None when fewer than 3 are supported."""
    use = w > 0
    if int(use.sum()) < 3: return None
    if bool(use.all()) and bool((w == 1).all()): return _kabsch(P, Q)
    return _kabsch(P[use], Q[use], w[use])


def cross_validated_rigidity(traj, omega, members, folds=5):
    idx = torch.as_tensor(members).long().reshape(-1)
    X = torch.as_tensor(traj).double(); Om = omega[:, idx]
    T, N = X.shape[0], int(idx.numel())
    if N < max(4, folds):
        return {'degenerate': True, 'num_nodes': N, 'eta_cv': float('nan'), 'rss_rigid_cv': float('nan'), 'rss_identity_cv': float('nan'),
                'delta_rss_cv': float('nan'), 'delta_bic': float('nan')}
    x0 = X[0, idx]
    fold_of = torch.arange(N) % folds
    rss_r = rss_i = rss_r_in = rss_i_in = 0.
    for f in range(folds):
        te = fold_of == f; tr = ~te
        if int(tr.sum()) < 3 or int(te.sum()) < 1: continue
        for t in range(T):
            xt = X[t, idx]
            fit = _fit(x0[tr], xt[tr], Om[t, tr])
            if fit is None: continue
            R, p = fit
            pred = x0[te]@R.T+p
            rss_r += float(((xt[te]-pred).pow(2)*Om[t, te, None]).sum())
            rss_i += float(((xt[te]-x0[te]).pow(2)*Om[t, te, None]).sum())
    used, nobs = 0, 0.
    for t in range(T):
        xt = X[t, idx]
        fit = _fit(x0, xt, Om[t])
        if fit is None: continue
        R, p = fit; used += 1; nobs += float(Om[t].sum())
        rss_r_in += float(((xt-x0@R.T-p).pow(2)*Om[t, :, None]).sum())
        rss_i_in += float(((xt-x0).pow(2)*Om[t, :, None]).sum())
    nobs = max(nobs, 1.)
    k = 6*used
    bic_i = nobs*math.log(max(rss_i_in, EPS)/nobs)
    bic_r = nobs*math.log(max(rss_r_in, EPS)/nobs)+k*math.log(nobs)
    neff = max(float(Om.sum()), 1.)
    return {'degenerate': False, 'num_nodes': N, 'folds': folds, 'rss_rigid_cv': rss_r, 'rss_identity_cv': rss_i,
            'delta_rss_cv': rss_i-rss_r, 'delta_rss_cv_per_node_frame': (rss_i-rss_r)/neff, 'eta_cv': 1.-rss_r/(rss_i+EPS),
            'rss_rigid_in_sample': rss_r_in, 'rss_identity_in_sample': rss_i_in, 'bic_identity': bic_i, 'bic_rigid': bic_r,
            'delta_bic': bic_i-bic_r, 'supported_node_frames': neff, 'frames_fitted': used}


def to_reference_frame(traj, omega, complement):
    """(relative trajectory, relative weights); a frame whose complement has < 3 supported nodes carries no evidence."""
    X = torch.as_tensor(traj).double()
    k = torch.as_tensor(complement).long().reshape(-1)
    if k.numel() < 3: return None, None
    y0 = X[0, k]
    out = torch.empty_like(X); w = omega.clone()
    for t in range(X.shape[0]):
        fit = _fit(y0, X[t, k], omega[t, k])
        if fit is None:
            out[t] = X[t]; w[t] = 0.; continue
        R, p = fit
        out[t] = (X[t]-p)@R
    return out, w


def refine_from_core(traj, omega, core, candidates_pool, sigma_k=3.):
    X = torch.as_tensor(traj).double()
    core = torch.as_tensor(core).long().reshape(-1)
    T = X.shape[0]; x0c = X[0, core]
    fits = [_fit(x0c, X[t, core], omega[t, core]) for t in range(T)]

    def score(idx, moved):
        e = torch.zeros(idx.numel(), dtype=torch.float64)
        for t, fit in enumerate(fits):
            if fit is None: continue
            d = X[t, idx]-X[0, idx]@fit[0].T-fit[1] if moved else X[t, idx]-X[0, idx]   # pinned evaluation order
            e += d.pow(2).sum(1)*omega[t, idx]
        return e
    ec = score(core, True)
    scale = float(ec.median())+sigma_k*float((ec-ec.median()).abs().median())*1.4826
    pool = torch.as_tensor(candidates_pool).long().reshape(-1)
    pool = pool[~torch.isin(pool, core)]
    if not pool.numel():
        return {'recruited': torch.zeros(0, dtype=torch.long), 'core_residual_scale': scale, 'num_considered': 0, 'num_recruited': 0}
    em, ei = score(pool, True), score(pool, False)
    keep = em < ei
    return {'recruited': pool[keep], 'core_residual_scale': scale, 'num_considered': int(pool.numel()),
            'num_recruited': int(keep.sum()), 'num_rejected_static': int((~keep).sum()),
            'rule': 'pinned section 19 rule, node-frames weighted by image support'}


def extract_moving_component(traj, omega, edge_i, edge_j, C, n, *, min_nodes=3, folds=5, max_candidates=48,
                             min_merge_support=2, refine=True):
    """The pinned extract_moving_component with node-frame support weights (identical when omega == 1)."""
    cands = candidate_hierarchy(edge_i, edge_j, C, n, min_nodes=min_nodes, min_merge_support=min_merge_support)
    seen, by_bin = set(), {}
    for c in cands:
        key = (c['num_nodes'], min(c['members']), max(c['members']))
        if key in seen: continue
        seen.add(key)
        by_bin.setdefault(int(math.log2(max(c['num_nodes'], 1))), []).append(c)
    bins = sorted(by_bin)
    per_bin = max(2, max_candidates//max(len(bins), 1))
    pool = []
    for b in bins:
        pool.extend(sorted(by_bin[b], key=lambda c: -(c['persistence'] if c['persistence'] != float('inf') else 1e18))[:per_bin])
    pool = pool[:max_candidates]
    all_nodes = torch.arange(n)
    for c in pool:
        c['edges'] = edge_distributions(c, edge_i, edge_j, C, n)
        inside = torch.zeros(n, dtype=torch.bool); inside[torch.tensor(c['members'], dtype=torch.long)] = True
        comp = all_nodes[~inside]
        c['complement_nodes'] = int(comp.numel())
        c['rigidity_world'] = cross_validated_rigidity(traj, omega, c['members'], folds)
        rel, w_rel = to_reference_frame(traj, omega, comp)
        if rel is None:
            c['rigidity'] = {'degenerate': True, 'num_nodes': c['num_nodes'], 'reason': 'fewer than 3 complement nodes',
                             'eta_cv': float('nan'), 'delta_rss_cv': float('nan'), 'delta_bic': float('nan')}
        else:
            c['rigidity'] = cross_validated_rigidity(rel, w_rel, c['members'], folds)
            c['rigidity']['frame'] = 'complement_relative'
            mem = torch.tensor(c['members'], dtype=torch.long)
            ext = float((traj[0, mem]-traj[0, mem].mean(0)).norm(dim=1).median().clamp_min(EPS))
            nf = max(float(w_rel[:, mem].sum()), 1.)
            c['rigidity']['relative_motion_ratio'] = math.sqrt(max(c['rigidity']['rss_identity_cv'], 0.)/nf)/ext
    sel = select_moving_component(pool)
    labels = [INCOHERENT]*n
    core = torch.zeros(0, dtype=torch.long); recruit = None
    if sel['status'] == 'RESOLVED':
        chosen = next(c for c in pool if c['candidate_id'] == sel['selected'])
        core = torch.tensor(chosen['members'], dtype=torch.long)
        if refine:
            recruit = refine_from_core(traj, omega, core, torch.arange(n))
            core = torch.unique(torch.cat([core, recruit['recruited']]))
        for v in core.tolist(): labels[v] = MOVING_RIGID
        static_pool = [c for c in pool if not c['rigidity'].get('degenerate') and c['rigidity']['delta_bic'] <= 0.]
        if static_pool:
            for v in max(static_pool, key=lambda c: c['num_nodes'])['members']:
                if labels[v] == INCOHERENT: labels[v] = STATIC_COMPATIBLE
    return {'status': sel['status'], 'selection': sel, 'labels': labels,
            'counts': {k: labels.count(k) for k in (MOVING_RIGID, STATIC_COMPATIBLE, INCOHERENT)}, 'moving_nodes': core,
            'candidates': [{k: v for k, v in c.items() if k != 'members'} | {'num_members': c['num_nodes']} for c in pool],
            'candidate_members': {c['candidate_id']: c['members'] for c in pool},
            'refinement': ({k: v for k, v in recruit.items() if k != 'recruited'} | {'recruited_nodes': recruit['recruited'].tolist()} if recruit else None),
            'seed_core_nodes': sorted(chosen['members']) if sel['status'] == 'RESOLVED' else [],
            'num_merge_candidates': len(cands), 'method': 'coherence_single_linkage_heldout_rigid, image-support weighted'}
