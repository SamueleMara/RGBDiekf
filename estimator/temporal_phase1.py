"""Phase-1 temporal solver driver (RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md, opt-in `--free-solver temporal-ieks`).

    prior (P1.1)  ->  forward inner IEKF warm start (P1.2)  ->  RTS (P1.3)  ->  outer IEKS on the ORIGINAL posterior (P1.4)
    ->  Laplace covariance with lag-one and frame-0 / frame-t blocks  ->  b, d_t, v_t and their covariances (P1.5)
    ->  residual audit (P1.6), mask feasibility diagnostics (P1.8 stage 1), statuses (spec 0.8), histories (P1.9).

The original prior (mu_0, P_0) is used in every outer iteration; the smoothed posterior is never re-used as a prior.
Parameter identification (P1.7) is NOT run here (estimator/temporal_model_identification.py, disabled by default).
Nothing here reads ground truth, groups points, matches points, renders images, uses masks as a loss, or uses a speed
percentile; the old directional free filter (free_projective_run.run_free) is untouched.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from estimator.projective_sensor import CODE
from estimator.ieks_point_trajectory import (FrameStore, TemporalModel, TrajectoryProblem, evaluate_frame, forward_frame, solve_chunk, point_chunks,
                                            assert_single_counted, SUPPORT_LOSS, FORWARD_STATUS, FRAME_STATE, TRAJ_STATUS, TEMPORAL_REASONS)
from estimator.rts_point_smoother import rts_smooth, transitions
from estimator.temporal_observability import frame_information_eigs, geometry_identifiability
from estimator.temporal_history import History

SUPPORT_HASH_SEED = 20261029


class ResidualAudit:
    """Signed residual statistics of the accepted rows of the final linearization (spec P1.6): bias E[z], spread, tails,
    by camera / frame / depth / fractional pixel / Jacobian magnitude / source camera / source brightness, the 4x4
    cross-channel covariance and the lag-one temporal autocorrelation of each (point, camera) series (subsampled points).
    z = r / sqrt(R) (whitened, sign convention r = -h)."""
    DEPTH_EDGES = (0., .5, 1., 1.5, 2., 3., 4., 6., np.inf)
    FRAC_EDGES = (0., .05, .1, .2, .3, .5)
    JAC_EDGES = (0., 1e-2, 1e-1, 1., 10., 100., np.inf)

    def __init__(self, K, C, N, src_cam, rgb0, max_points):
        self.K, self.C, self.N = K, C, N
        self.src, self.bright = src_cam, rgb0.mean(1)
        self.stride = max(1, N//max(1, max_points))
        self.sub = np.arange(0, N, self.stride)
        self.series = np.full((K, C, len(self.sub), 4), np.nan, np.float32)
        self.acc = {}
        self.rows_sub = []
        self.i = 0

    def _add(self, key, z, mask=None):
        if mask is not None: z = z[mask]
        if not len(z): return
        a = self.acc.setdefault(key, [0, np.zeros(4), np.zeros(4), np.zeros(4)])
        a[0] += len(z); a[1] += z.sum(0); a[2] += (z*z).sum(0); a[3] += (np.abs(z) > 3).sum(0)

    def __call__(self, t, ok, p, c):
        if not bool(ok.any()): return
        z = (t.r[ok]/t.var[ok].sqrt()).double().cpu().numpy(); pp = p[ok].cpu().numpy(); cc = c[ok].cpu().numpy()
        Z = t.z[ok].double().cpu().numpy(); fr = (t.uv[ok]-torch.floor(t.uv[ok])).cpu().numpy(); fd = np.minimum(fr, 1-fr).min(1)
        jac = t.Hx[ok].norm(dim=-1).cpu().numpy()                                 # [m,4] per-channel Jacobian magnitude
        self._add(('all',), z)
        for ci in np.unique(cc): self._add(('camera', int(ci)), z, cc == ci)
        self._add(('frame', self.i), z)
        db = np.digitize(Z, self.DEPTH_EDGES)-1; fb = np.digitize(fd, self.FRAC_EDGES)-1; jb = np.digitize(jac[:, 3], self.JAC_EDGES)-1
        for b in np.unique(db): self._add(('depth_bin', int(b)), z, db == b)
        for b in np.unique(fb): self._add(('fractional_pixel_bin', int(b)), z, fb == b)
        for b in np.unique(jb): self._add(('depth_jacobian_bin', int(b)), z, jb == b)
        sc = self.src[pp]
        for ci in np.unique(sc): self._add(('source_camera', int(ci)), z, sc == ci)
        self._add(('row_is_source_camera', 1), z, sc == cc); self._add(('row_is_source_camera', 0), z, sc != cc)
        bb = np.digitize(self.bright[pp], (0., .2, .4, .6, .8, np.inf))-1
        for b in np.unique(bb): self._add(('source_brightness_bin', int(b)), z, bb == b)
        sel = (pp % self.stride) == 0
        if sel.any():
            self.series[self.i, cc[sel], pp[sel]//self.stride] = z[sel]
            self.rows_sub.append(z[sel])

    def finalize(self):
        out = {}
        for key, (n, s, s2, t3) in self.acc.items():
            m = s/n
            out[' '.join(map(str, key))] = {'rows': int(n), 'mean_z_RGBD': m.tolist(), 'std_z_RGBD': np.sqrt(np.maximum(s2/n-m*m, 0)).tolist(),
                                            'mean_square_z_RGBD': (s2/n).tolist(), 'fraction_abs_z_over_3_RGBD': (t3/n).tolist()}
        Zs = np.concatenate(self.rows_sub) if self.rows_sub else np.zeros((0, 4))
        res = {'groups': out, 'subsample_points': int(len(self.sub)), 'subsample_rows': int(len(Zs))}
        if len(Zs):
            res['subsample_median_z_RGBD'] = np.median(Zs, 0).tolist(); res['subsample_median_square_z_RGBD'] = np.median(Zs*Zs, 0).tolist()
            res['subsample_mean_square_z_RGBD'] = (Zs*Zs).mean(0).tolist()
            res['cross_channel_covariance_z'] = np.cov(Zs.T).tolist()
        a, b = self.series[:-1], self.series[1:]
        both = np.isfinite(a) & np.isfinite(b)
        ac = []
        for ch in range(4):
            m = both[..., ch]
            ac.append(float(np.corrcoef(a[..., ch][m], b[..., ch][m])[0, 1]) if m.sum() > 2 else None)
        res['lag1_temporal_autocorrelation_z_RGBD'] = ac; res['lag1_pairs'] = int(both[..., 0].sum())
        res['note'] = ('z = r / sqrt(R) of accepted rows at the final linearization. E[z] = bias, std = random spread, lag-1 '
                       'autocorrelation = persistent correlation; mean z^2 = 1 with a tiny median z^2 is heterogeneity, not calibration.')
        return res


def mask_diagnostics(proj, frame, rig):
    """Per point (all points, every camera, occluded or not): counts of in-view cameras, nearest-pixel inside, strict
    2x2-footprint outside, footprint straddling the silhouette boundary, footprint touching the image border (P1.8 st. 1)."""
    C, N = proj.z.shape
    W, H = frame.W, frame.H
    uv, z = proj.uv, proj.z
    fin = torch.isfinite(uv).all(-1) & torch.isfinite(z)
    inview = fin & (z > rig.near[:, None]) & (uv[..., 0] >= -.5) & (uv[..., 0] <= W-.5) & (uv[..., 1] >= -.5) & (uv[..., 1] <= H-.5)
    u, v = torch.where(fin, uv[..., 0], torch.zeros_like(z)), torch.where(fin, uv[..., 1], torch.zeros_like(z))
    u0 = torch.floor(u).clamp(0, W-2).long(); v0 = torch.floor(v).clamp(0, H-2).long()
    cam = torch.arange(C, device=z.device)[:, None].expand(C, N)
    m = frame.mask
    four = torch.stack([m[cam, v0, u0], m[cam, v0, u0+1], m[cam, v0+1, u0], m[cam, v0+1, u0+1]], 0)
    near = m[cam, torch.round(v).clamp(0, H-1).long(), torch.round(u).clamp(0, W-1).long()]
    border = (u < 1) | (u > W-2) | (v < 1) | (v > H-2)
    cnt = lambda b: (b & inview).sum(0).clamp_max(255).to(torch.uint8)
    return {'mask_cameras_in_view': cnt(torch.ones_like(inview)), 'mask_cameras_nearest_inside': cnt(near),
            'mask_cameras_strict_outside': cnt(~four.any(0)), 'mask_cameras_boundary_ambiguous': cnt(four.any(0) & ~four.all(0)),
            'mask_cameras_image_border': cnt(border)}


class Pass:
    """One factor pass over all frames at a trajectory: per point-frame sums on the host."""
    def __init__(self, K, N, C, latent):
        f = lambda *s: torch.zeros((K, N, *s), dtype=torch.float64)
        self.Ixx, self.gx, self.q = f(3, 3), f(3), f(); self.nrows = torch.zeros((K, N), dtype=torch.int16)
        self.border = torch.zeros((K, N), dtype=torch.bool); self.valid = torch.ones((K, N), dtype=torch.bool)
        self.support = torch.zeros((K, C, N), dtype=torch.bool); self.reason = torch.zeros((K, C, N), dtype=torch.uint8)
        if latent: self.Ixa, self.Iaa, self.ga = f(3, 3), f(3), f(3)
        self.latent = latent

    def __getitem__(self, name):
        return getattr(self, name)


def factor_pass(frames, M, order, S, a, Sig_pos, src, pts=None, frozen=None, extras=None, rows_only=False, S_lin=None):
    """S [K,N,6] host (processing order), a [N,3] host, Sig_pos [K,N,3,3] host. frozen: a Pass whose support is used;
    S_lin: the linearization trajectory whose row covariances are frozen with that support."""
    K, N = S.shape[:2]; C = M.rig.count; dev = M.device
    out = Pass(K, N, C, M.latent) if not rows_only else None
    q = torch.zeros(N, dtype=torch.float64); valid = torch.ones(N, dtype=torch.bool)
    a_d = a.to(dev); pts_d = None if pts is None else pts.to(dev); src_d = torch.as_tensor(src, device=dev)
    for i, k in enumerate(order):
        frame = frames.get(k)
        if extras is not None and 'audit' in extras: extras['audit'].i = i
        ev = evaluate_frame(M, frame, S[i, :, :3].to(dev), a_d, Sig_pos[i].to(dev), pts=pts_d,
                            support=None if frozen is None else frozen.support[i].to(dev),
                            src=src_d if (i == 0 and frozen is None) else None,
                            audit=None if extras is None else extras.get('audit'),
                            X_lin=None if S_lin is None else S_lin[i, :, :3].to(dev))
        if rows_only:
            q += ev['q'].cpu(); valid &= ev['valid'].cpu(); continue
        out.Ixx[i] = ev['Ixx'].cpu(); out.gx[i] = ev['gx'].cpu(); out.q[i] = ev['q'].cpu(); out.nrows[i] = ev['nrows'].cpu().to(torch.int16)
        out.border[i] = ev['border'].cpu(); out.valid[i] = ev['valid'].cpu()
        if frozen is None:
            out.support[i] = ev['support'].cpu(); out.reason[i] = ev['reason'].cpu()
            if i == 0: assert_single_counted(ev['support'], src_d)
        if M.latent: out.Ixa[i] = ev['Ixa'].cpu(); out.Iaa[i] = ev['Iaa'].cpu(); out.ga[i] = ev['ga'].cpu()
        if extras is not None and 'mask' in extras and frozen is None:
            for kk, v in mask_diagnostics(ev['proj'], frame, M.rig).items(): extras['mask'].setdefault(kk, []).append(v.cpu())
        del frame, ev
    return (q, valid) if rows_only else out


def _objective_prior(P, S, a, idx, dev):
    J, _ = P.prior_dyn(S[:, idx].to(dev), idx, dev)
    if P.latent: J = J+P.latent_prior(a[idx].to(dev), idx, dev)[0]
    return J.cpu()


def solve_all(P, S, a, fac, dev, covariance=False, sink=None):
    K, N = S.shape[:2]
    dT = torch.empty_like(S); da = torch.zeros((N, 3), dtype=torch.float64) if P.latent else None
    dec, J, gn = (torch.empty(N, dtype=torch.float64) for _ in range(3))
    for sl in point_chunks(N, K):
        idx = torch.arange(sl.start, sl.stop)
        r = solve_chunk(P, S[:, sl].to(dev), None if a is None else a[sl].to(dev), fac, idx, dev, covariance)
        dT[:, sl] = r['dT'].cpu(); dec[sl] = r['dec'].cpu(); J[sl] = r['J'].cpu(); gn[sl] = r['grad_norm'].cpu()
        if P.latent: da[sl] = r['da'].cpu()
        if covariance and sink is not None: sink(sl, r)
        del r
    return dT, da, dec, J, gn


def support_hash(fac, weights):
    """Per-point 61-bit hash of the support pattern over (frame, camera) (oscillation detection only)."""
    K, C, N = fac.support.shape
    h = torch.zeros(N, dtype=torch.int64)
    for i in range(K):
        h = (h*1_000_003+(fac.support[i].to(torch.int64)*weights[i][:, None]).sum(0)) % (2**61-1)
    return h


def solve_trajectories(frames, M, order, times, x_ref, P_x0, rgb0, src_cam, P_v0, qa, hist=None, trace_ids=(), log=print, audit_points=20000,
                       save_lag1=False):
    """Phase-1 posterior of every point (host float64 inputs, processing order = order). Returns a dict of summary arrays
    and writes every history to `hist` (estimator/temporal_history.History) when given."""
    t_start = time.perf_counter()
    dev = M.device
    N, K, C = len(x_ref), len(order), M.rig.count
    T = torch.as_tensor(np.asarray(times, float)[order])
    F, Q = transitions(T, torch.as_tensor(qa, dtype=torch.float64))
    x_ref = torch.as_tensor(x_ref, dtype=torch.float64); P_x0 = torch.as_tensor(P_x0, dtype=torch.float64)
    rgb0 = torch.as_tensor(rgb0, dtype=torch.float64); src = torch.as_tensor(src_cam, dtype=torch.long)
    mean0 = torch.cat([x_ref, torch.zeros_like(x_ref)], 1)
    cov0 = torch.zeros((N, 6, 6), dtype=torch.float64); cov0[:, :3, :3] = P_x0; cov0[:, 3:, 3:] = torch.as_tensor(P_v0, dtype=torch.float64)
    timing, trace = {}, {'forward': {}, 'ieks': []} if len(trace_ids) else None
    tid = torch.as_tensor(list(trace_ids), dtype=torch.long)
    # ------------------------------------------------------------------------------------------- forward warm start
    t0 = time.perf_counter()
    filt_mean = torch.empty((K, N, 6), dtype=torch.float64); filt_cov = torch.empty((K, N, 6, 6), dtype=torch.float64)
    fstatus = torch.zeros((K, N), dtype=torch.uint8); frows = torch.zeros((K, N), dtype=torch.int16)
    # visibility / row-gate covariance of every outer iteration: the warm start's PRIOR-PREDICTIVE position covariance P-_k,
    # fixed for the whole IEKS (a gate built from the iterate's own posterior covariance depends on the support it selects
    # and made the support oscillate with period two: measured, see reports/temporal_phase1_ieks.md)
    Sig_gate = torch.empty((K, N, 3, 3), dtype=torch.float64)
    fwd_stats = []
    s, P = mean0.to(dev), cov0.to(dev); a_dev = rgb0.to(dev)
    prev_rows = None; frame0 = {}
    Fd, Qd = F.to(dev), Q.to(dev)
    for i, k in enumerate(order):
        tfr = time.perf_counter()
        frame = frames.get(k)
        if i == 0: s_m, P_m = s, P
        else:
            s_m = s@Fd[i-1].T; P_m = Fd[i-1]@P@Fd[i-1].T+Qd[i-1]; P_m = (P_m+P_m.transpose(1, 2))/2
        Sig_gate[i] = P_m[:, :3, :3].cpu()
        s, P, st, ev, stats = forward_frame(M, frame, s_m, P_m, a_dev, src=src.to(dev) if i == 0 else None, prev_rows=prev_rows)
        if i == 0:
            assert_single_counted(ev['support'], src.to(dev))
            frame0 = {'independent_cameras_used': ev['support'].sum(0).cpu(), 'information': ev['Ixx'].cpu()}
        filt_mean[i] = s.cpu(); filt_cov[i] = P.cpu(); fstatus[i] = st.cpu(); frows[i] = ev['nrows'].cpu().to(torch.int16)
        prev_rows = ev['nrows']
        fwd_stats.append({'frame': int(k), **stats, 'seconds': time.perf_counter()-tfr})
        if trace is not None:
            trace['forward'][int(k)] = {int(p): {'status': FORWARD_STATUS[int(st[p])], 'rows': int(ev['nrows'][p]), 's_plus': s[p].tolist()} for p in tid.tolist()}
        log(f'[temporal] forward frame {k} ({i+1}/{K}): rows {stats["points_with_rows"]} conv {stats["converged"]} notconv {stats["not_converged"]} '
            f'failed {stats["numerically_failed"]} lost {stats["lost_after_moving"]} accepted steps {stats["accepted_steps"]} border retries {stats["border_retries"]}')
        del frame, ev
    timing['forward_iekf_s'] = time.perf_counter()-t0
    # ------------------------------------------------------------------------------------------------- RTS warm start
    t0 = time.perf_counter()
    S = torch.empty((K, N, 6), dtype=torch.float64)
    for sl in point_chunks(N, K):
        sm = rts_smooth(filt_mean[:, sl].to(dev), filt_cov[:, sl].to(dev), Fd, Qd, want_lag1=False, want_cross0=False)
        S[:, sl] = sm['smooth_mean'].cpu()
        del sm
    Sig_pos = Sig_gate
    timing['rts_warm_start_s'] = time.perf_counter()-t0
    if hist is not None:
        hist.create('filter_mean', (K, N, 6), np.float64, 'forward inner IEKF filtered means s_{k|k} (warm start)')[:] = filt_mean.numpy()
        fc = hist.create('filter_cov', (K, N, 6, 6), np.float32, 'forward filtered covariances P_{k|k} (computed float64)')
        for i in range(K): fc[i] = filt_cov[i].numpy()
        hist.create('forward_status', (K, N), np.uint8, f'forward IEKF status per point-frame {FORWARD_STATUS}')[:] = fstatus.numpy()
        hist.create('forward_valid_rows', (K, N), np.int16, 'accepted rows of the forward IEKF final linearization')[:] = frows.numpy()
        hist.create('warm_start_smooth_mean', (K, N, 6), np.float64, 'RTS-smoothed warm start = initial IEKS trajectory S^0')[:] = S.numpy()
        hist.create('transition_F', (K-1, 6, 6), np.float64, 'F_k of every processing interval')[:] = F.numpy()
        hist.create('process_Q', (K-1, 6, 6), np.float64, 'Q_k of every processing interval (exact white-acceleration discretization)')[:] = Q.numpy()
    del filt_cov
    S_warm = S.clone()
    # ----------------------------------------------------------------------------------------------- outer IEKS
    t0 = time.perf_counter()
    Pr = TrajectoryProblem(mean0, cov0, F, Q, rgb0, M.sigma_a2.cpu(), M.latent)
    a = rgb0.clone() if M.latent else rgb0
    g = torch.Generator().manual_seed(SUPPORT_HASH_SEED)
    hw = torch.randint(1, 2**40, (K, C), generator=g)
    hashes = []
    osc = torch.zeros(N, dtype=torch.bool); accepted_ever = torch.zeros(N, dtype=torch.bool); accepted_last = torch.zeros(N, dtype=torch.bool)
    first_dec = None
    history = []
    ended = 'iteration_cap'
    for it in range(M.max_outer):
        ti = time.perf_counter()
        fac = factor_pass(frames, M, order, S, a, Sig_pos, src)
        tf = time.perf_counter()-ti
        dT, da, dec, J, gn = solve_all(Pr, S, a if M.latent else None, fac, dev)
        if first_dec is None: first_dec = dec.clone()
        hashes.append(support_hash(fac, hw))
        if len(hashes) >= 3: osc |= (hashes[-1] == hashes[-3]) & (hashes[-1] != hashes[-2])
        go = (dec >= M.tol) & (fac.nrows.sum(0) > 0)
        rec = {'iteration': it, 'objective_total_on_this_support': float(J.sum()), 'points_moving': int(go.sum()), 'newton_decrement_max': float(dec.max()),
               'rows_total': int(fac.nrows.sum()), 'factor_pass_s': tf}
        if it > 0: rec['support_changes_vs_previous'] = int((hashes[-1] != hashes[-2]).sum())
        if not bool(go.any()):
            rec['ended'] = 'all points stationary under the current support'; history.append(rec); ended = 'stationary'; break
        # ---- frozen-support backtracking on the true objective
        alpha = torch.ones(N, dtype=torch.float64); pending = go.clone(); acc = torch.zeros(N, dtype=torch.bool)
        Snew, anew = S.clone(), (a.clone() if M.latent else a)
        halv = 0; tl = time.perf_counter()

        def trial(direction, dA, pend, alpha):
            Sc = S+alpha[None, :, None]*direction
            ac = (a+alpha[:, None]*dA) if M.latent else a
            qs, val = factor_pass(frames, M, order, Sc, ac, Sig_pos, src, pts=pend, frozen=fac, rows_only=True, S_lin=S)
            Jc = torch.zeros(N, dtype=torch.float64)
            idx = torch.nonzero(pend).flatten()
            for sl in point_chunks(len(idx), K):
                ii = idx[sl]; Jc[ii] = _objective_prior(Pr, Sc, ac, ii, dev)
            return Sc, ac, Jc+.5*qs, val

        def backtrack(direction, dA, pend):
            nonlocal halv
            al = torch.ones(N, dtype=torch.float64); ok_all = torch.zeros(N, dtype=torch.bool)
            Sb, ab = S.clone(), (a.clone() if M.latent else a)
            for h in range(M.halvings+1):
                if not bool(pend.any()): break
                Sc, ac, Jc, val = trial(direction, dA, pend, al)
                ok = pend & val & M.improves(Jc, J)
                Sb[:, ok] = Sc[:, ok]
                if M.latent: ab[ok] = ac[ok]
                ok_all |= ok; pend = pend & ~ok; al = torch.where(pend, al/2, al); halv += int(pend.sum())
            return ok_all, Sb, ab
        acc, Snew, anew = backtrack(dT, da, pending.clone())
        retry = go & ~acc & fac.border.any(0)
        n_retry, n_retry_ok = int(retry.sum()), 0
        if n_retry:
            nd = dT[..., :3].norm(dim=-1).amax(0).clamp_min(1e-300)
            Ss = S+M.eta*dT/nd[None, :, None]
            facs = factor_pass(frames, M, order, Ss, (a+M.eta*da/nd[:, None]) if M.latent else a, Sig_pos, src, pts=retry, frozen=fac)
            retry &= facs.valid.all(0)
            facs.q = fac.q                                                 # objective terms stay at S; only the linearization moves
            dT2, da2, _, _, _ = solve_all(Pr, S, a if M.latent else None, facs, dev)
            acc2, S2, a2 = backtrack(dT2, da2, retry.clone())
            Snew[:, acc2] = S2[:, acc2]
            if M.latent: anew[acc2] = a2[acc2]
            acc |= acc2; n_retry_ok = int(acc2.sum())
        rec.update(accepted=int(acc.sum()), line_search_halvings=halv, border_retries=n_retry, border_retry_accepted=n_retry_ok, line_search_s=time.perf_counter()-tl)
        if bool(acc.any()):
            Jafter = J.clone()
            # objective after the step ON THE SAME (frozen) support, for the accepted points
            _, _, Jacc, _ = trial((Snew-S), (anew-a) if M.latent else None, acc.clone(), torch.ones(N, dtype=torch.float64))
            Jafter[acc] = Jacc[acc]
            rec['objective_after_step_same_support'] = float(Jafter.sum())
            rec['accepted_actual_decrease_same_support'] = float((J-Jafter)[acc].sum())
            rec['accepted_gauss_newton_full_step_decrease'] = float(.5*dec[acc].sum())       # quadratic model at alpha = 1
            rec['accepted_min_actual_decrease'] = float((J-Jafter)[acc].min())
        if trace is not None:
            trace['ieks'].append({'iteration': it, **{str(p): {'J': float(J[p]), 'newton_decrement': float(dec[p]), 'accepted': bool(acc[p]),
                                                               'rows_per_frame': fac.nrows[:, p].tolist(), 'step_max_m': float(dT[:, p, :3].norm(dim=-1).max())}
                                                      for p in tid.tolist()}})
        history.append(rec)
        log(f'[temporal] IEKS iteration {it}: J {rec["objective_total_on_this_support"]:.6g} moving {rec["points_moving"]} accepted {rec["accepted"]} '
            f'retries {n_retry}/{n_retry_ok} rows {rec["rows_total"]} ({time.perf_counter()-ti:.1f} s)')
        accepted_last = acc
        if not bool(acc.any()):
            ended = 'no point found an improving step'; break
        accepted_ever |= acc
        S = torch.where(acc[None, :, None], Snew, S)
        if M.latent: a = torch.where(acc[:, None], anew, a)
    timing['ieks_s'] = time.perf_counter()-t0
    # ------------------------------------------------------------------- final linearization at the accepted trajectory
    t0 = time.perf_counter()
    audit = ResidualAudit(K, C, N, src.numpy(), rgb0.numpy(), audit_points)
    extras = {'audit': audit, 'mask': {}}
    fac = factor_pass(frames, M, order, S, a, Sig_pos, src, extras=extras)
    final = {'smooth_cov': torch.empty((K, N, 6, 6), dtype=torch.float32), 'dcov': torch.empty((K, N, 3, 3), dtype=torch.float32),
             'b_cov': torch.empty((N, 3, 3), dtype=torch.float64), 'lag1': torch.empty((max(K-1, 0), N, 6, 6), dtype=torch.float32) if save_lag1 else None,
             'cov_a': torch.empty((N, 3, 3), dtype=torch.float64) if M.latent else None, 'Sx0': torch.empty((N, 3, 3), dtype=torch.float64)}

    def sink(sl, r):
        D, Cx = r['diag'], r['cross0']
        final['smooth_cov'][:, sl] = D.float().cpu()
        Dx = D[..., :3, :3]; C0 = Cx[..., :3, :3]
        dc = Dx+Dx[:1]-C0-C0.transpose(-1, -2); final['dcov'][:, sl] = ((dc+dc.transpose(-1, -2))/2).float().cpu()
        final['b_cov'][sl] = Dx[0].cpu(); final['Sx0'][sl] = Dx[0].cpu()
        if save_lag1: final['lag1'][:, sl] = r['lag1'].float().cpu()
        if M.latent: final['cov_a'][sl] = r['cov_a'].cpu()
    _, _, dec_final, J_final, gn_final = solve_all(Pr, S, a if M.latent else None, fac, dev, covariance=True, sink=sink)
    timing['final_linearization_s'] = time.perf_counter()-t0
    # --------------------------------------------------------------------------------------------- statuses
    rows = fac.nrows.to(torch.int64)
    any_rows = rows.sum(0) > 0
    traj = torch.full((N,), 1, dtype=torch.uint8)                               # LINE_SEARCH_STALLED unless shown otherwise
    traj[dec_final < M.tol] = 0
    traj[(dec_final >= M.tol) & accepted_last & (ended == 'iteration_cap')] = 2
    traj[~accepted_ever & (dec_final >= M.tol)] = 3                          # no IEKS step ever improved J: warm start kept
    traj[osc & (dec_final >= M.tol)] = 5
    traj[~any_rows] = 4
    loss = torch.zeros((K, N), dtype=torch.bool)
    for code in SUPPORT_LOSS: loss |= (fac.reason == code).any(1)
    state = torch.where(rows > 0, 1, 2).to(torch.uint8)
    prev = torch.cat([torch.zeros((1, N), dtype=torch.bool), rows[:-1] > 0])
    state[(rows == 0) & prev & loss] = 3
    state[:, traj == 3] = 4
    b = S[0, :, :3]-x_ref
    lam, ident, nonid = geometry_identifiability(P_x0, final['Sx0'])
    info_eig = frame_information_eigs(fac.Ixx).float()
    mask = {kk: torch.stack(v) for kk, v in extras['mask'].items()}
    mask_infeasible = mask['mask_cameras_strict_outside'] > 0
    res = {'smooth_mean': S, 'b': b, 'b_cov': final['b_cov'], 'displacement': S[..., :3]-S[:1, :, :3], 'displacement_cov': final['dcov'],
           'velocity': S[..., 3:], 'smooth_cov': final['smooth_cov'], 'lag1': final['lag1'], 'latent_colour': a if M.latent else None, 'latent_colour_cov': final['cov_a'],
           'frame_state': state, 'trajectory_status': traj, 'valid_rows': rows.to(torch.int16), 'camera_reason': fac.reason, 'data_information_eig': info_eig,
           'frame0_information_eig': frame_information_eigs(frame0['information']).float(), 'frame0_independent_cameras': frame0['independent_cameras_used'],
           'b_identifiability_lambda': lam, 'b_identified_directions': ident.sum(-1).to(torch.uint8), 'b_unidentified_directions': nonid.to(torch.uint8),
           'unidentifiable': nonid > 0, 'mask_infeasible': mask_infeasible, **mask, 'forward_status': fstatus, 'filter_mean': filt_mean, 'warm_start_smooth_mean': S_warm,
           'newton_decrement_final': dec_final, 'objective_final': J_final, 'gradient_norm_final': gn_final, 'support_oscillation': osc,
           'history': history, 'forward_stats': fwd_stats, 'residual_audit': audit.finalize(), 'timing': timing, 'trace': trace, 'ended': ended}
    if hist is not None: write_results(hist, res, K, N, C, M)
    timing['total_s'] = time.perf_counter()-t_start
    return res


def write_results(hist, res, K, N, C, M):
    w = lambda name, arr, dt, doc: hist.create(name, tuple(arr.shape), dt, doc).__setitem__(slice(None), arr.numpy() if torch.is_tensor(arr) else arr)
    w('smooth_mean', res['smooth_mean'], np.float64, 'IEKS posterior mean trajectory [x, v] (MAP of the original posterior)')
    sc = hist.create('smooth_cov', (K, N, 6, 6), np.float32, 'Laplace posterior covariance blocks Cov(s_k, s_k) at the final linearization')
    for i in range(K): sc[i] = res['smooth_cov'][i].numpy()
    if res['lag1'] is not None:
        l1 = hist.create('smooth_lag1_cov', (K-1, N, 6, 6), np.float32, 'Cov(s_{k+1}, s_k) at the final linearization')
        for i in range(K-1): l1[i] = res['lag1'][i].numpy()
    w('geometry_correction_b', res['b'], np.float64, 'b = x_0|T - x_ref')
    w('geometry_correction_b_cov', res['b_cov'], np.float64, 'Cov(b) = Cov(x_0|T) (x_ref fixed)')
    w('displacement', res['displacement'], np.float64, 'd_t = x_t|T - x_0|T')
    dc = hist.create('displacement_cov', (K, N, 3, 3), np.float32, 'Cov(d_t) = P_tt + P_00 - P_t0 - P_0t (exact cross-time blocks)')
    for i in range(K): dc[i] = res['displacement_cov'][i].numpy()
    w('velocity', res['velocity'], np.float32, 'v_t|T (m/s)')
    if res['latent_colour'] is not None:
        w('latent_colour', res['latent_colour'], np.float64, 'posterior mean of the per-point canonical colour a (latent, anchored at the source pixel)')
        w('latent_colour_cov', res['latent_colour_cov'], np.float64, 'Cov(a)')
    w('frame_state', res['frame_state'], np.uint8, f'per point-frame state {FRAME_STATE}')
    w('trajectory_status', res['trajectory_status'], np.uint8, f'per point {TRAJ_STATUS}')
    w('valid_rows', res['valid_rows'], np.int16, 'accepted rows at the final linearization')
    w('camera_reason', res['camera_reason'], np.uint8, f'per frame / camera / point reason at the final linearization {TEMPORAL_REASONS}')
    w('data_information_eig', res['data_information_eig'], np.float32, 'eigenvalues of the 3x3 DATA information per frame (no prior)')
    w('frame0_information_eig', res['frame0_information_eig'], np.float32, 'frame-0 data information of the independent (non-source) cameras')
    w('frame0_independent_cameras', res['frame0_independent_cameras'], np.int16, 'non-source cameras with an admitted frame-0 row')
    w('b_identifiability_lambda', res['b_identifiability_lambda'], np.float64, 'prior / posterior variance ratios of x_0 (generalized eigenvalues)')
    w('b_unidentified_directions', res['b_unidentified_directions'], np.uint8, 'directions of b with data information < prior information')
    w('unidentifiable', res['unidentifiable'], np.bool_, 'b has at least one non-identified direction')
    w('mask_infeasible', res['mask_infeasible'], np.bool_, 'some in-view camera sees the whole 2x2 footprint outside the measured mask')
    for kk in ('mask_cameras_in_view', 'mask_cameras_nearest_inside', 'mask_cameras_strict_outside', 'mask_cameras_boundary_ambiguous', 'mask_cameras_image_border'):
        w(kk, res[kk], np.uint8, 'mask feasibility diagnostic (all points, all in-view cameras, occluded or not)')
    w('newton_decrement_final', res['newton_decrement_final'], np.float64, 'delta^T A delta at the final linearization (stationarity)')
    w('objective_final', res['objective_final'], np.float64, 'original posterior objective at the final support')
    w('support_oscillation', res['support_oscillation'], np.bool_, 'support pattern returned to an earlier state (A-B-A)')


def run_temporal_stage(dataset, config, policy, tcfg, direction, root, device, sub, debug_point_ids=None, log=print, provenance=None):
    """F1 with the Phase-1 temporal solver: writes free_forward_temporal/ (never the old free_forward/)."""
    from utils.io import json_write, serial
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import SensorNoise
    from estimator.free_projective_run import build_catalogue
    if direction != 'forward':
        raise ValueError('--free-solver temporal-ieks runs F1 only: the backward catalogue receives the same algorithm after F1 validation (spec 0.1)')
    if policy.get('rendered_mask_measurement', {}).get('mode', 'off') != 'off':
        raise ValueError('the rendered pseudomask measurement must stay off: no rendering-based energy enters Phase-1 IEKS (G6.5)')
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    rig = CameraRig(dataset.cameras, device, torch.float64)
    noise = SensorNoise.from_config(config, policy, device)
    k0, xyz, rgb, cov, source, sel, sub_man = build_catalogue(dataset, config, direction, sub, policy, noise)
    K, N, C = len(dataset.times), len(sel), rig.count
    order = list(range(K))
    if order[0] != k0: raise AssertionError('F1 starts at the canonical frame 0')
    np.savez_compressed(root/'canonical.npz', canonical_frame=k0, canonical_xyz=xyz[sel], canonical_rgb=rgb[sel], original_source_ids=source[sel], point_id=np.arange(N),
                        subsampled_to_full_index=sel, canonical_source_covariance=cov[sel])
    P_v0 = np.asarray(config.velocity_covariance, float); qa = np.asarray(config.qc, float)
    np.savez_compressed(root/'anchor_prior.npz', x_ref=xyz[sel], P_x0=cov[sel], source_frame_camera_pixel=source[sel], mu_v0=np.zeros(3), P_v0=P_v0,
                        provenance=np.array(json.dumps({'x_ref': 'frame-0 back-projection of the source pixel (scene/endpoint.py)',
                                                        'P_x0': 'J_bproj diag(pixel_covariance, sigma_D^2(z)) J_bproj^T (scene/endpoint.py); calibration uncertainty not '
                                                                'available: conditional on the declared calibration',
                                                        'P_v0': config.provenance.get('velocity_covariance'), 'q_a': config.provenance.get('qc'),
                                                        'source_measurement': 'the source depth / colour define x_ref, P_x0 and a: the source camera is NOT a frame-0 '
                                                                              'measurement (reason FRAME0_SOURCE_DUPLICATE_EXCLUDED)'}, default=str)))
    json_write(root/'subsample_manifest.json', serial(sub_man))
    M = TemporalModel(rig, noise, policy, tcfg, device)
    frames = FrameStore(dataset, order, device)
    hist = History(root, K, N, C)
    t_load = time.perf_counter()-t0
    ids = [int(i) for i in (debug_point_ids or []) if 0 <= int(i) < N]
    res = solve_trajectories(frames, M, order, dataset.times, xyz[sel], cov[sel], rgb[sel], source[sel, 1], P_v0, qa, hist=hist, trace_ids=ids, log=log,
                             audit_points=int(tcfg['residual_audit_points']), save_lag1=bool(tcfg['save_lag_one']))
    hist.flush({'frame_times_s': np.asarray(dataset.times, float).tolist(), 'original_frame_ids': list(dataset.frame_ids)})
    json_write(root/'objective_history.json', serial({'ieks': res['history'], 'forward': res['forward_stats'], 'ended': res['ended']}))
    json_write(root/'residual_audit.json', serial(res['residual_audit']))
    if res['trace'] is not None: json_write(root/'trace_points.json', serial(res['trace']))
    traj = np.bincount(res['trajectory_status'].numpy(), minlength=len(TRAJ_STATUS))
    fs = np.bincount(res['frame_state'].numpy().ravel(), minlength=len(FRAME_STATE))
    manifest = {'pass': 'F1', 'solver': 'temporal-ieks', 'spec': 'RGBDiekf_Phase1_Iterated_Temporal_IEKF_IEKS_Spec.md', 'direction': direction, 'canonical_frame': k0,
                'points': N, 'frames': K, 'cameras': C, 'camera_geometry_hash': rig.calibration_hash(), 'selected_ids_sha256': sub_man['selected_ids_sha256'],
                'frame_times_s': np.asarray(dataset.times, float).tolist(), 'original_frame_ids': list(dataset.frame_ids), 'temporal_config': tcfg,
                'measurement_policy': {k: M.policy.get(k) for k in ('row_innovation_gate', 'visibility_n_sigma', 'same_surface_n_sigma', 'same_surface_max_incidence_deg',
                                                                   'interpolation_variance', 'rgb_model_variance', 'depth_model_variance', 'depth_interpolation',
                                                                   'iekf_step_tolerance', 'sensor_noise')},
                'R': {'rgb_sensor_variance': noise.rgb_variance.cpu().tolist(), 'depth_variance_m2': noise.depth_variance, 'depth_squared_variance': noise.depth_squared_variance,
                      'rgb_model_variance': noise.rgb_model_variance, 'depth_model_variance_m2': noise.depth_model_variance,
                      'provenance': {k: config.provenance.get(k) for k in ('color_covariance', 'depth_variance', 'depth_squared_variance')},
                      'label': 'the run\'s declared / measured R (scene/noise_calibration.py when --sensor-noise measured); a BASELINE calibration, not validated as an '
                               'unbiased independent R (spec section 3)'},
                'prior': {'mu_v0_m_s': [0., 0., 0.], 'P_v0_m2_s2': P_v0.tolist(), 'q_a_m2_s3': qa.tolist(),
                          'classes': {k: (config.provenance.get(k) or {}).get('class') for k in ('velocity_covariance', 'qc')}},
                'codes': {'frame_state': FRAME_STATE, 'trajectory_status': TRAJ_STATUS, 'forward_status': FORWARD_STATUS, 'camera_reason': TEMPORAL_REASONS},
                'outputs_note': 'predicted history = F_k filter history, F_k P_{k|k} F_k^T + Q_k (transition_F / process_Q stored); covariances stored float32',
                'ground_truth_used_in_inference': False, 'groups_or_rigid_coupling': False, 'rendered_pseudomask': False, 'speed_percentile_used': False,
                'provenance': provenance}
    json_write(root/'run_manifest.json', serial(manifest))
    status = {'pass': 'F1', 'solver': 'temporal-ieks', 'status': 'COMPLETE', 'points': N, 'frames': K, 'ended': res['ended'],
              'trajectory_status_counts': {TRAJ_STATUS[i]: int(v) for i, v in enumerate(traj)}, 'frame_state_counts': {FRAME_STATE[i]: int(v) for i, v in enumerate(fs)},
              'unidentifiable_points': int(res['unidentifiable'].sum()), 'mask_infeasible_point_frames': int(res['mask_infeasible'].sum()),
              'support_oscillation_points': int(res['support_oscillation'].sum()), 'timing_s': {**res['timing'], 'catalogue_and_frame_load_s': t_load},
              'frame_cache_bytes': frames.bytes, 'gpu_peak_allocated_MB': torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
              'cpu_peak_rss_MB': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024,
              'bytes_written': int(sum(f.stat().st_size for f in root.rglob('*') if f.is_file())),
              'forward_frame_seconds_median': float(np.median([r['seconds'] for r in res['forward_stats']])),
              'ieks_iteration_seconds': [h.get('factor_pass_s', 0)+h.get('line_search_s', 0) for h in res['history']],
              'factor_pass_seconds': [h.get('factor_pass_s') for h in res['history']]}
    json_write(root/'status.json', serial(status))
    return status
