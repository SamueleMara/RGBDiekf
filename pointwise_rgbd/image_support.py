"""Support completion with a global latent label vector on the complete-scene RGB-D image likelihood (contract 11).

Model. For the endpoint points outside the high-confidence moving core, z_i in {0 static, 1 moving}; the core is
moving. Under labelling z every point is rendered at
    static:  x_i(k) = p_i,     moving:  x_i(k) = T_J(k) T_J(anchor)^-1 p_i     (the selected joint, q from Phase 3)
and the observation model is the estimation objective of every stage (image_observation: tempered Student-t RGB-D
+ explicit coverage term, all cameras, every non-anchor frame):
    p(Y | z, T_J) proportional to exp(-E(z)),   E(z) = sum_k sum_c E_kc(render(scene_k(z))),
    p(z) = prod_i Bernoulli(1/2)   (no preference).
The RGB-D channels are tempered by the rendered support, so exp(-E) is a Gibbs pseudo-likelihood: its pixels are
dependent and their sum is not a calibrated log-likelihood ratio.

Inference.
  1. MAP z* by block iterated conditional modes: for one block of points the exact single-flip differences
     Delta_i = E(z with i flipped) - E(z) (others at the current labels) propose flipping every Delta_i < 0; the
     proposal is ACCEPTED only if the full re-rendered E decreases, otherwise its most negative half is tried (so E
     decreases monotonically). Converged when a cycle of blocks proposes nothing acceptable.
  2. Labels at z* by CONTRADICTION. With the other points at z*, per frame k
         C^s_ik = E_k(i static) - E_k(i absent),     C^m_ik = E_k(i moving) - E_k(i absent)
     ("absent": the point renders nothing). A hypothesis is contradicted when it is significantly WORSE than the point
     not being there: one-sided frame-level tests (one number per frame, so pixel dependence inside a frame is never
     assumed away; conservative max(Newey-West, AR(1)) long-run variance, Harvey-Leybourne-Newbold, t(n-1)),
     Benjamini-Hochberg over points at level alpha per family.
         MOVING_JOINT_COMPATIBLE   static contradicted AND moving supported (significantly better than absent)
         STATIC_COMPATIBLE         moving contradicted AND static supported
         AMBIGUOUS                 otherwise: indistinguishable, redundant samples, textureless slides, near-axis points,
                                   a hypothesis that merely hides the point; also without influenced pixels
     "Not contradicted" is not evidence: a static point that the moving part later covers is simply hidden under the
     moving hypothesis (measured: requiring only non-contradiction gave moving-label precision 0.11 on the fixture).
         MOVING_CORE               the coherence core (unchanged)
     Why not the sign of E(moving) - E(static) alone: on the textured fixtures the MAP moves redundant STATIC samples
     into pixels disoccluded by the moving part (no catalogue point exists there and each uncovered pixel costs
     kappa/2), so "moving fits better" there is evidence of redundancy, not of motion (contract: never label a point
     moving because an overlapping sample renders similar pixels). Measured: the sign rule gave moving-label
     precision 0.81 on the 16-frame prismatic fixture.

Fast path. delta_flips recomputes, for every (point, pixel) pair touched by the point's footprint under either
hypothesis, the pixel from its COMPLETE contributor list with the point's entry replaced (front zeta, taper,
normalized splat, coverage, pixel energy), i.e. exactly the renderer and objective; tests/test_image_support.py
verifies it against full re-renders at several frames and after other label changes, and the ICM MAP against exact
enumeration of all labellings on small coupled scenes. No point is inserted.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .rgbd_point_renderer import CameraTensors, _candidates, _segments, render
from .image_observation import FrameObservation
from .joint_image import joint_transform, POLICY
from .predictive_tests import conservative_long_run_variance
from scipy import stats
from .group_visualization import serial
from .store import json_write


def _entries(cam, x, rgb, rho, policy, ids=None):
    """Candidate footprint entries of points x (optionally a subset ids): point, pixel, w, z, band, zeta, colour."""
    xs = x if ids is None else x[ids]
    _, z, _, _, _, pt, pix, dx, dy, q = _candidates(cam, xs, rho if ids is None else rho[ids], policy)
    if ids is not None: pt = ids[pt]
    zc = z[pt] if ids is None else z[torch.searchsorted(ids, pt)]
    w = (1-q)**2; b = policy.band(zc, rho[pt])
    return pt, pix, w, zc, b, zc+b*(1/w-1), rgb[pt]


def _pixel_predictions(seg_id, nseg, w, z, b, zeta, col):
    """Renderer pixel model on arbitrary row groups (one group = one pixel's contributor list)."""
    dev = w.device
    front = torch.full((nseg,), float('inf'), dtype=w.dtype, device=dev).scatter_reduce(0, seg_id, zeta, 'amin')
    gap = z-front[seg_id]
    u = ((2*gap-b)/b).clamp_min(0)
    v = torch.where(2*gap <= b, torch.ones_like(u), torch.where(gap < b, (1-u*u)**2, torch.zeros_like(u)))
    om = v*w
    W = torch.zeros(nseg, dtype=w.dtype, device=dev).index_add_(0, seg_id, om)
    C = torch.zeros((nseg, 3), dtype=w.dtype, device=dev).index_add_(0, seg_id, om[:, None]*col)
    Z = torch.zeros(nseg, dtype=w.dtype, device=dev).index_add_(0, seg_id, om*z)
    safe = torch.where(W > 0, W, torch.ones_like(W))
    return torch.where((W > 0)[:, None], C/safe[:, None], torch.zeros_like(C)), torch.where(W > 0, Z/safe, torch.zeros_like(Z)), -torch.expm1(-W)


@torch.no_grad()
def delta_flips(cam, ci, obs, x_base, x_alt, active, rgb, rho, policy, chunk=2_000_000):
    """Exact E(scene with point i alone at x_alt[i]) - E(scene) for every active i, one camera and frame.
    Returns (delta [N], influenced pixel count [N])."""
    dev = x_base.device; N = len(x_base); npx = cam.width*cam.height
    pt, pix, w, z, b, zeta, col = _entries(cam, x_base, rgb, rho, policy)
    order, upix, lengths = _segments(pix)
    pt, pix, w, z, b, zeta, col = pt[order], pix[order], w[order], z[order], b[order], zeta[order], col[order]
    start = torch.full((npx,), 0, dtype=torch.long, device=dev); count = torch.zeros(npx, dtype=torch.long, device=dev)
    start[upix] = torch.cumsum(lengths, 0)-lengths; count[upix] = lengths
    ids = torch.nonzero(active).flatten()
    a_pt, a_pix, a_w, a_z, a_b, a_zeta, a_col = _entries(cam, x_alt, rgb, rho, policy, ids)
    s_sel = active[pt]
    keys = torch.unique(torch.cat([pt[s_sel]*npx+pix[s_sel], a_pt*npx+a_pix]))          # affected (point, pixel) pairs
    p_pt, p_pix = keys//npx, keys % npx
    alt_row = torch.full((len(keys),), -1, dtype=torch.long, device=dev)
    alt_row[torch.searchsorted(keys, a_pt*npx+a_pix)] = torch.arange(len(a_pt), device=dev)
    delta = torch.zeros(N, dtype=w.dtype, device=dev); infl = torch.zeros(N, dtype=torch.long, device=dev)
    L = count[p_pix]; csum = torch.cumsum(L, 0)
    lo = 0
    while lo < len(keys):
        hi = int(torch.searchsorted(csum, (csum[lo-1] if lo else 0)+chunk, right=True)); hi = max(hi, lo+1)
        sl = slice(lo, hi); n = hi-lo
        Ls = L[sl]
        grp = torch.repeat_interleave(torch.arange(n, device=dev), Ls)
        off = torch.arange(int(Ls.sum()), device=dev)-torch.repeat_interleave(torch.cumsum(Ls, 0)-Ls, Ls)
        rows = start[p_pix[sl]][grp]+off
        keep = pt[rows] != p_pt[sl][grp]                                                   # the flipped point's own entry removed
        ar = alt_row[sl]; has = ar >= 0; ag = torch.nonzero(has).flatten(); ai = ar[has]
        # base prediction (all rows) and modified (rows without the point + its alternative entry)
        pb = _pixel_predictions(grp, n, w[rows], z[rows], b[rows], zeta[rows], col[rows])
        g2 = torch.cat([grp[keep], ag]); cat = lambda base, alt: torch.cat([base[rows][keep], alt[ai]])
        pm = _pixel_predictions(g2, n, cat(w, a_w), cat(z, a_z), cat(b, a_b), cat(zeta, a_zeta), torch.cat([col[rows][keep], a_col[ai]]))
        pp = p_pix[sl]
        e_b = obs.pixel_energy(ci, pp, *pb); e_m = obs.pixel_energy(ci, pp, *pm)
        ch = e_m-e_b
        delta.index_add_(0, p_pt[sl], ch); infl.index_add_(0, p_pt[sl], (ch != 0).long())
        lo = hi
    return delta, infl


def scene_points(p, moving, J):
    moved = p@J[:3, :3].T+J[:3, 3]
    return torch.where(moving[:, None], moved, p), torch.where(moving[:, None], p, moved)


def total_energy(obs, cams, p, moving, Js, rgb, rho, policy):
    return sum(obs[k].terms(render(scene_points(p, moving, J)[0], rgb, rho, cams, policy))['energy'] for k, J in Js.items())


def per_frame_flips(obs, cams, p, moving, Js, active, rgb, rho, policy, desc='flip differences'):
    """D[k, i] = E_k(z with i flipped) - E_k(z) for active i; influenced pixel counts."""
    frames = list(Js)
    D = torch.zeros(len(frames), len(p), dtype=torch.float64, device=p.device); infl = torch.zeros(len(p), dtype=torch.long, device=p.device)
    for j, k in enumerate(tqdm(frames, desc=desc, unit='frame')):
        base, alt = scene_points(p, moving, Js[k])
        for ci, cam in enumerate(cams):
            d, n = delta_flips(cam, ci, obs[k], base, alt, active, rgb, rho, policy)
            D[j] += d; infl += n
    return D, infl


def per_frame_removals(obs, cams, p, moving, Js, active, rgb, rho, policy):
    """D[k, i] = E_k(z with i absent) - E_k(z) for active i (a non-finite position renders nothing)."""
    frames = list(Js)
    D = torch.zeros(len(frames), len(p), dtype=torch.float64, device=p.device); infl = torch.zeros(len(p), dtype=torch.long, device=p.device)
    gone = torch.full_like(p, float('nan'))
    for j, k in enumerate(tqdm(frames, desc='support removal differences at the MAP', unit='frame')):
        base, _ = scene_points(p, moving, Js[k])
        for ci, cam in enumerate(cams):
            d, n = delta_flips(cam, ci, obs[k], base, gone, active, rgb, rho, policy)
            D[j] += d; infl += n
    return D, infl


def block_icm(obs, cams, p, core, Js, rgb, rho, policy, blocks=4, cycles=3, seed=0, cand=None):
    """Monotone block ICM for the MAP of exp(-E(z)) with a uniform prior. cand: labelled points (default: all but the
    core); points neither core nor candidate stay static."""
    dev = p.device; cand = ~core if cand is None else cand
    block = torch.as_tensor(np.random.default_rng(seed).integers(0, blocks, len(p)), device=dev)
    moving = core.clone(); E = total_energy(obs, cams, p, moving, Js, rgb, rho, policy)
    history, quiet = [{'round': -1, 'energy': E, 'moving_non_core': 0}], 0
    for r in range(blocks*cycles):
        active = cand & (block == r % blocks)
        D, _ = per_frame_flips(obs, cams, p, moving, Js, active, rgb, rho, policy, f'ICM block {r % blocks}')
        delta = D.sum(0)
        prop = torch.nonzero(active & (delta < 0)).flatten()
        prop = prop[torch.argsort(delta[prop])]
        accepted, tried = 0, []
        while len(prop):
            trial = moving.clone(); trial[prop] = ~trial[prop]
            Et = total_energy(obs, cams, p, trial, Js, rgb, rho, policy)
            tried.append({'flips': int(len(prop)), 'energy': Et})
            if Et < E: moving, E, accepted = trial, Et, int(len(prop)); break
            prop = prop[:len(prop)//2]
        history.append({'round': r, 'block': r % blocks, 'proposed': int((active & (delta < 0)).sum()), 'accepted_flips': accepted,
                        'energy': E, 'trials': tried, 'moving_non_core': int((moving & cand).sum())})
        quiet = quiet+1 if accepted == 0 else 0
        if quiet >= blocks: break
    return moving, E, history, quiet >= blocks


def _one_sided(series, core, alpha):
    """Frame-level HAC t-tests of mean > 0 per point (series [n, N]), BH over non-core points; returns (passed, mean, p)."""
    n, N = series.shape; mean = series.mean(0); p = np.ones(N)
    t = stats.t(n-1)
    for i in np.flatnonzero(~core):
        s2 = conservative_long_run_variance(series[:, i])[0]
        if s2 <= 0: continue
        p[i] = t.sf(mean[i]/math.sqrt(s2/n)*math.sqrt((n-1)/n))
    tested = np.flatnonzero(~core); order = tested[np.argsort(p[tested])]; m = len(tested)
    passed = np.zeros(N, bool)
    if m:
        ok = p[order] <= alpha*np.arange(1, m+1)/m
        if ok.any(): passed[order[:np.flatnonzero(ok).max()+1]] = True
    return passed, mean, p


def label_tests(D_flip, D_remove, influenced, core, moving, alpha):
    """D_flip[k,i] = E_k(i flipped) - E_k(z*), D_remove[k,i] = E_k(i absent) - E_k(z*). Contradiction labels."""
    Df, Dr = D_flip.cpu().numpy(), D_remove.cpu().numpy(); mv = moving.cpu().numpy()[None]
    C_cur = -Dr; C_alt = Df-Dr                                   # current / alternative hypothesis minus absent
    Cs, Cm = np.where(mv, C_alt, C_cur), np.where(mv, C_cur, C_alt)
    core = core.cpu().numpy(); seen = influenced.cpu().numpy() > 0
    excl = core | ~seen
    s_bad, cs, ps = _one_sided(Cs, excl, alpha); m_bad, cm, pm = _one_sided(Cm, excl, alpha)
    s_ok, _, ps_ok = _one_sided(-Cs, excl, alpha); m_ok, _, pm_ok = _one_sided(-Cm, excl, alpha)
    labels = np.full(len(core), 'AMBIGUOUS', dtype=object)
    labels[core] = 'MOVING_CORE'
    labels[~excl & s_bad & m_ok] = 'MOVING_JOINT_COMPATIBLE'
    labels[~excl & m_bad & s_ok] = 'STATIC_COMPATIBLE'
    return labels, {'contradiction_static_mean': cs, 'contradiction_moving_mean': cm, 'p_static_contradicted': ps, 'p_moving_contradicted': pm,
                    'p_static_supported': ps_ok, 'p_moving_supported': pm_ok, 'mean_moving_minus_static': (Cm-Cs).mean(0)}


def run(dataset, config, backend, partition, rho, core, selected, endpoint, root, policy, obs_policy, blocks=4, cycles=3, frames=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    alpha = json.loads(POLICY.read_text())['tail_probability']
    dev = backend.device; t = backend.tensor
    p, rgb, rho_t = t(partition.xyz), t(partition.rgb), t(rho)
    core_t = torch.as_tensor(core, device=dev)
    cams = [CameraTensors(c, dev) for c in dataset.cameras]
    q = np.asarray(selected['q']); q = q-q[endpoint]
    frames = [k for k in range(len(dataset.times)) if k != endpoint] if frames is None else frames
    Js = {k: joint_transform(selected, float(q[k])).to(dev) for k in frames}
    obs = {k: FrameObservation([dataset.observation(k, c) for c in range(len(cams))], config, obs_policy, dev) for k in frames}
    moving, E, history, converged = block_icm(obs, cams, p, core_t, Js, rgb, rho_t, policy, blocks, cycles)
    D, infl = per_frame_flips(obs, cams, p, moving, Js, ~core_t, rgb, rho_t, policy, 'support flip differences at the MAP')
    Dr, infl_r = per_frame_removals(obs, cams, p, moving, Js, ~core_t, rgb, rho_t, policy)
    labels, ev = label_tests(D, Dr, infl+infl_r, core_t, moving, alpha)
    counts = {name: int((labels == name).sum()) for name in ('MOVING_CORE', 'MOVING_JOINT_COMPATIBLE', 'STATIC_COMPATIBLE', 'AMBIGUOUS')}
    np.savez_compressed(root/'support.npz', labels=labels.astype(str), map_moving=moving.cpu().numpy(), influenced_pixels=(infl+infl_r).cpu().numpy(),
                        per_frame_flip=D.cpu().numpy().astype(np.float32), per_frame_removal=Dr.cpu().numpy().astype(np.float32), **ev)
    out = {'counts': counts, 'alpha_fdr': alpha, 'map': {'energy': E, 'converged': converged, 'rounds': history,
           'moving_non_core': int((moving & ~core_t).sum())}, 'frames_scored': len(frames),
           'joint': {k: v for k, v in selected.items() if k not in ('q', 'qdot', 'q_pose')}, 'anchor_frame': endpoint,
           'uninfluenced_candidates': int((~core_t & (infl == 0)).sum()),
           'map_moving_but_ambiguous': int(((labels == 'AMBIGUOUS') & moving.cpu().numpy()).sum()),
           'semantics': 'global labels z, p(Y|z) ~ exp(-E(z)) (estimation objective, complete scene, all cameras), uniform prior; '
                        'monotone block-ICM MAP; labels need one hypothesis contradicted AND the other supported against the point being '
                        'absent (frame-level HAC tests, BH-FDR per family)'}
    json_write(root/'result.json', serial(out))
    return {**out, 'labels': labels, 'evidence': ev}
