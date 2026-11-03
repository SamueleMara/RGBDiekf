"""Fixed-prior point-local IEKF of the local-correlated temporal RGB-D model.

Observation (unchanged physical warp, raw errors explicit):
    f_{i,c,t}(a_i) = sum_a w_a(q) (y_{k_a} - eta_{i,k_a}) + e4 dZ + eta_{i,k_t},   r = y_{k_t} - f(a_i) = eps_warp
with q = p - pi_c(x_t) + pi_c(x_tau), dZ = Z_c(x_t) - Z_c(x_tau). eps_warp ~ N(0, R_warp) is ONLY the additional
warp discrepancy (zero for --temporal-warp-noise ideal): current and reference sensor errors are variables of a_i,
so R_current + sum w^2 R_ref is never added (that independent marginal sum is the simplified ablation).

Per point and frame, all valid cameras are stacked into one 4m-row update. The prior (mu^-, Sigma^-) is FIXED over
the inner iterations:  a^{j+1} = mu + K_j [ r_j + H_j (a^j - mu) ],  K_j = Sigma H_j^T S_j^{-1},
S_j = H_j Sigma H_j^T + R_warp (SPD because every row owns a fresh current sensor error with positive prior).
Raw keys first touched in this frame enter with one independent N(0, R_sensor) prior; a key already in the point's
local Gaussian is reused (its columns sum). The covariance is computed ONCE, at the accepted iterate, in Joseph form.
Global discrete visibility (first-hit ownership, hard masks, reference supports) is validated exactly as the
simplified ablation does: common current pixel/owner/reference domain, mask feasibility, per-point MAP descent,
then rollback of both parties of every changed owner.
"""
from dataclasses import dataclass
import torch
from estimator.temporal_rgbd_terms import temporal_prediction, sensor_covariance
from estimator.temporal_simplified_iekf import point_domains, outside_points
from estimator.temporal_local_covariance import (STATE, CHART, KEY, Layout, BlockIndex, LocalGaussian, chunk_plan,
    kept_columns, select_dense, retention_mask, batched_eigvalsh, record_expiry, check_no_recreation, symmetric_psd_certificate,
    LocalCovarianceFailure, QUALIFIER, PROPOSAL_TENSORS_PER_CHUNK)

VECTORS = ('it', 'z', 'base', 'zbase', 'prop', 'zprop', 'trial', 'ztrial')


class Extras:
    """Raw sensor errors first touched in THIS frame (fresh, independent N(0, R_sensor) priors)."""
    def __init__(self, N, device):
        e = torch.empty(0, dtype=torch.long, device=device)
        self.N, self.device = N, device
        self.code, self.point, self.camera, self.frame, self.pixel = e, e, e, e, e
        self.variance = torch.empty((0, 4), dtype=torch.float64, device=device)
        self.v = {name: torch.empty((0, 4), dtype=torch.float64, device=device) for name in VECTORS}

    def __len__(self):
        return len(self.code)

    def lookup(self, codes):
        flat = codes.reshape(-1)
        out = torch.full_like(flat, -1)
        if len(self.code) and len(flat):
            at = torch.searchsorted(self.code, flat).clamp_max(len(self.code)-1)
            hit = self.code[at] == flat
            out[hit] = at[hit]
        return out.reshape(codes.shape)

    def add(self, code, point, camera, frame, pixel, variance):
        code, first = torch.unique(code, return_inverse=True)
        pick = torch.zeros(len(code), dtype=torch.long, device=code.device).scatter_(0, first, torch.arange(len(first), device=code.device))
        fresh = self.lookup(code) < 0
        if not bool(fresh.any()):
            return 0
        pick = pick[fresh]
        cat = lambda a, b: torch.cat((a, b))
        allcode = cat(self.code, code[fresh])
        order = torch.argsort(allcode)
        self.code = allcode[order]
        self.point, self.camera, self.frame, self.pixel = (cat(old, new[pick])[order] for old, new in
            ((self.point, point), (self.camera, camera), (self.frame, frame), (self.pixel, pixel)))
        self.variance = cat(self.variance, variance[pick])[order]
        for name in VECTORS:
            self.v[name] = cat(self.v[name], torch.zeros((int(fresh.sum()), 4), dtype=torch.float64, device=self.device))[order]
        return int(fresh.sum())

    def layout(self, rows=None):
        ix = torch.arange(len(self), device=self.device) if rows is None else rows
        return Layout(self.N, self.point[ix], torch.full_like(ix, KEY), self.camera[ix], self.frame[ix], self.pixel[ix])


@dataclass
class LocalRows:
    rows: object
    owners: object
    reason: torch.Tensor
    life: dict
    ref_code: torch.Tensor
    src_code: torch.Tensor
    ja: torch.Tensor
    jl: torch.Tensor
    key_code: torch.Tensor
    cur_code: torch.Tensor
    weights: torch.Tensor
    residual: torch.Tensor
    current_variance: torch.Tensor
    clean_reference: torch.Tensor
    reused_history_keys: int
    reused_source_keys: int

    def __len__(self):
        return len(self.rows)


def _add_blocks(H, b, slot, col, block):
    w = block.shape[-1]
    r = 4*slot[:, None, None]+torch.arange(4, device=H.device)[None, :, None]
    c = col[:, None, None]+torch.arange(w, device=H.device)[None, None, :]
    bb = b[:, None, None].expand(-1, 4, w)
    H.index_put_((bb, r.expand(-1, -1, w), c.expand(-1, 4, -1)), block, accumulate=True)


class LocalFrameUpdate:
    def __init__(self, prior, chart, codec, refs, build, frame, k, ordinal, lag, rig, noise, warp, budget_bytes,
                 expired_upto, frame_ordinal):
        self.prior, self.chart, self.codec, self.refs, self.build = prior, chart, codec, refs, build
        self.frame, self.k, self.ordinal, self.lag = frame, int(k), int(ordinal), int(lag)
        self.rig, self.noise, self.warp, self.budget = rig, noise, warp.to(prior.mean), int(budget_bytes)
        self.expired_upto, self.frame_ordinal = expired_upto, frame_ordinal
        self.N, self.C, self.dev = prior.N, refs.frame.shape[0], prior.mean.device
        L = prior.layout
        self.state_index = BlockIndex(L, codec, L.kind != KEY)
        self.key_index = BlockIndex(L, codec, L.kind == KEY)
        self.cur = prior.current_state_index()
        self.entry_point = torch.repeat_interleave(torch.arange(self.N, device=self.dev), L.dims)
        self.mu = prior.mean
        self.P = {name: torch.zeros_like(self.mu) for name in VECTORS}
        self.P['it'] = self.mu.clone()
        self.ex = Extras(self.N, self.dev)
        self.counters = {'linearizations': 0, 'fresh_raw_keys_registered': 0}
        self._work = None

    # ---------- packed vectors
    def _work_layout(self):
        if self._work is None or self._work[0] != len(self.ex):
            layout, order = _concat_with_origin(self.prior.layout, self.ex.layout())
            self._work = (len(self.ex), layout, order, BlockIndex(layout, self.codec))
        return self._work[1:]

    def _set(self, dst, src_a, src_b=None, rate=None, points=None):
        e = slice(None) if points is None else points[self.entry_point]
        x = slice(None) if points is None else points[self.ex.point]
        for P in (self.P, self.ex.v):
            m = e if P is self.P else x
            if src_b is None:
                P[dst][m] = P[src_a][m]
            else:
                P[dst][m] = P[src_a][m]+rate*(P[src_b][m]-P[src_a][m])

    def current_states(self, name='it'):
        return self.P[name][self.cur]

    def _values(self, name, blocks, width, offset=0):
        """Values of prior blocks (point-local start) for vector `name`."""
        L = self.prior.layout
        pos = self.prior.moff[L.point[blocks]]+L.start[blocks]+offset
        return self.P[name][pos[..., None]+torch.arange(width, device=self.dev)]

    def _key_values(self, name, codes):
        out = torch.zeros((*codes.shape, 4), dtype=torch.float64, device=self.dev)
        pb = self.key_index.lookup(codes)
        hit = pb >= 0
        if bool(hit.any()):
            out[hit] = self._values(name, pb[hit], 4)
        xb = self.ex.lookup(codes)
        hx = xb >= 0
        if bool(hx.any()):
            out[hx] = self.ex.v[name][xb[hx]]
        return out, pb, xb

    def _state_codes(self, p, rf):
        kind = torch.where(rf == self.chart.anchor_frame, torch.full_like(rf, CHART), torch.full_like(rf, STATE))
        z = torch.zeros_like(p)
        return self.codec.encode(p, kind, z, rf, z), kind

    def _reference_states(self, name, p, rf):
        codes, kind = self._state_codes(p, rf)
        blk = self.state_index.lookup(codes)
        if bool((blk < 0).any()):
            raise LocalCovarianceFailure('a usable camera reference has no retained local state; reference lifecycle violated')
        out = torch.zeros((len(p), 6), dtype=torch.float64, device=self.dev)
        ja = torch.zeros((len(p), 6, 5), dtype=torch.float64, device=self.dev)
        jl = torch.zeros((len(p), 6, 4), dtype=torch.float64, device=self.dev)
        st = kind == STATE
        if bool(st.any()):
            out[st] = self._values(name, blk[st], 6)
        ch = ~st
        if bool(ch.any()):
            eta, pb, _ = self._key_values(name, self.chart.source_codes(self.codec, p[ch]))
            if bool((pb < 0).any()):
                raise LocalCovarianceFailure('retained chart without its source raw-depth error')
            out[ch], ja[ch], jl[ch] = self.chart.evaluate(p[ch], self._values(name, blk[ch], 5), eta[:, 3])
        return out, codes, ja, jl

    # ---------- linearization at a packed iterate
    def linearize(self, name='it', register=True):
        """Rows at a packed iterate. Only linearizations that DEFINE a proposal (register=True) add the fresh raw keys
        they touch as local variables; a line-search trial evaluates unseen fresh keys at their prior mean (0, z=0),
        which is exact for its MAP objective and keeps rejected cells out of the local Gaussian."""
        self.counters['linearizations'] += 1
        refs, k, W = self.refs, self.k, self.frame.W
        s = self.current_states(name)
        age = self.ordinal-refs.ordinal
        c, p = torch.where((refs.frame >= 0) & (age < self.lag) & (age > 0))
        if len(p):
            refs.state[c, p] = self._reference_states(name, p, refs.frame[c, p])[0]
        rows, owners, reason, life = self.build(s)
        M = len(rows)
        p, c, rf = rows.point, rows.camera, rows.reference_frame
        # same local iterate as refs.state above; the Jacobian of a chart reference is taken at it
        _, ref_code, ja, jl = self._reference_states(name, p, rf)
        ref_state = rows.reference_state
        src_code = torch.full_like(p, -1)
        ch = rf == self.chart.anchor_frame
        src_code[ch] = self.chart.source_codes(self.codec, p[ch])
        key_code = self.codec.key(p[:, None].expand(-1, 4), c[:, None].expand(-1, 4), rf[:, None].expand(-1, 4), rows.reference_pixels)
        cur_code = self.codec.key(p, c, torch.full_like(p, k), rows.pixel)
        ref_var = sensor_covariance(rows.values, self.noise).diagonal(dim1=-2, dim2=-1)
        cur_var = sensor_covariance(rows.y, self.noise).diagonal(dim1=-2, dim2=-1)
        codes = torch.cat((key_code.reshape(-1), cur_code))
        missing = (self.key_index.lookup(codes) < 0) & (self.ex.lookup(codes) < 0)
        if bool(missing.any()):
            pp = torch.cat((p[:, None].expand(-1, 4).reshape(-1), p))[missing]
            cc = torch.cat((c[:, None].expand(-1, 4).reshape(-1), c))[missing]
            ff = torch.cat((rf[:, None].expand(-1, 4).reshape(-1), torch.full_like(p, k)))[missing]
            xx = torch.cat((rows.reference_pixels.reshape(-1), rows.pixel))[missing]
            check_no_recreation(pp, cc, ff, self.expired_upto, self.frame_ordinal)
            if register:
                self.counters['fresh_raw_keys_registered'] += self.ex.add(codes[missing], pp, cc, ff, xx, torch.cat((ref_var.reshape(-1, 4), cur_var))[missing])
        eta_ref, pb, _ = self._key_values(name, key_code)
        eta_cur, _, _ = self._key_values(name, cur_code)
        clean = rows.values-eta_ref
        uv = torch.stack((rows.pixel % W, rows.pixel//W), -1).to(s.dtype)
        h, ht, hr, q, w, _, _ = temporal_prediction(self.rig, s[p], ref_state, c, uv, clean, rows.cell)
        rows.h, rows.H_t, rows.H_ref, rows.q, rows.weights = h, ht, hr, q, w
        reused = int((pb >= 0).sum())
        src = int(((pb >= 0) & (key_code == self.chart.source_codes(self.codec, p)[:, None]) & ch[:, None]).sum()) if M else 0
        return LocalRows(rows, owners, reason, life, ref_code, src_code, ja, jl, key_code, cur_code, w,
                         rows.y-h-eta_cur, cur_var, clean, reused, src)

    # ---------- dense chunk algebra
    def _dense(self, pts, name, layout, origin, extra_rows=None):
        n = int(layout.dims[pts].max())
        mu, Sigma = self.prior.gather(pts, n)
        dp = self.prior.layout.dims[pts]
        a = torch.zeros_like(mu)
        vm = torch.arange(n, device=self.dev)[None, :] < dp[:, None]
        a[vm] = self.P[name][(self.prior.moff[pts][:, None]+torch.arange(n, device=self.dev))[vm]]
        loc = torch.full((self.N,), -1, dtype=torch.long, device=self.dev)
        loc[pts] = torch.arange(len(pts), device=self.dev)
        nprior = len(self.prior.layout)
        j = torch.where((loc[layout.point] >= 0) & (origin >= nprior))[0]
        b, col, row = loc[layout.point[j]], layout.start[j], origin[j]-nprior
        if extra_rows is not None:
            row = extra_rows[row]
        d4 = torch.arange(4, device=self.dev)
        cc = col[:, None]+d4
        Sigma[b[:, None], cc, cc] = self.ex.variance[row]
        a[b[:, None], cc] = self.ex.v[name][row]
        return mu, Sigma, a, loc, (b, cc, row)

    def _scatter(self, pts, dense, name, extras):
        n = dense.shape[1]
        dp = self.prior.layout.dims[pts]
        vm = torch.arange(n, device=self.dev)[None, :] < dp[:, None]
        self.P[name][(self.prior.moff[pts][:, None]+torch.arange(n, device=self.dev))[vm]] = dense[vm]
        b, cc, row = extras
        self.ex.v[name][row] = dense[b[:, None], cc]

    def _system(self, pts, lr, loc, index, n):
        B, C, dev = len(pts), self.C, self.dev
        rix = torch.where(loc[lr.rows.point] >= 0)[0]
        b, slot = loc[lr.rows.point[rix]], lr.rows.camera[rix]
        look = lambda codes: index.layout.start[index.lookup(codes)]
        src = lr.src_code[rix]
        for codes in (lr.ref_code[rix], lr.key_code[rix], lr.cur_code[rix], src[src >= 0]):
            if bool((index.lookup(codes) < 0).any()):
                raise LocalCovarianceFailure('row variable missing from the local Gaussian')
        H = torch.zeros((B, 4*C, n), dtype=torch.float64, device=dev)
        _add_blocks(H, b, slot, torch.zeros_like(b), lr.rows.H_t[rix])
        rc = look(lr.ref_code[rix])
        ch = lr.src_code[rix] >= 0
        st = ~ch
        if bool(st.any()):
            _add_blocks(H, b[st], slot[st], rc[st], lr.rows.H_ref[rix][st])
        if bool(ch.any()):
            hr = lr.rows.H_ref[rix][ch]
            _add_blocks(H, b[ch], slot[ch], rc[ch], hr@lr.ja[rix][ch])
            _add_blocks(H, b[ch], slot[ch], look(lr.src_code[rix][ch]), hr@lr.jl[rix][ch])
        I4 = torch.eye(4, dtype=torch.float64, device=dev)
        kc = look(lr.key_code[rix])
        for a in range(4):
            _add_blocks(H, b, slot, kc[:, a], -lr.weights[rix, a, None, None]*I4)
        _add_blocks(H, b, slot, look(lr.cur_code[rix]), I4.expand(len(rix), -1, -1))
        valid = torch.zeros((B, C), dtype=torch.bool, device=dev)
        valid[b, slot] = True
        valid = valid.repeat_interleave(4, 1)
        r = torch.zeros((B, 4*C), dtype=torch.float64, device=dev)
        r[b[:, None], 4*slot[:, None]+torch.arange(4, device=dev)] = lr.residual[rix]
        Rw = torch.block_diag(*[self.warp]*C).expand(B, -1, -1)*(valid[:, :, None] & valid[:, None, :])
        return H, r, Rw, valid, b

    def _factor(self, H, Sigma, Rw, valid):
        G = Sigma@H.transpose(1, 2)
        S = H@G+Rw
        S = (S+S.transpose(1, 2))/2+torch.diag_embed((~valid).to(S.dtype))
        if not bool(torch.isfinite(S).all()):
            raise LocalCovarianceFailure('LOCAL_INNOVATION_NOT_FINITE: non-finite local prior covariance or Jacobian')
        L, info = torch.linalg.cholesky_ex(S)
        if bool((info != 0).any()):
            bad = int(torch.where(info != 0)[0][0])
            eig = torch.linalg.eigvalsh(S[bad]).tolist()
            raise LocalCovarianceFailure(f'LOCAL_INNOVATION_NOT_SPD: rank-deficient local innovation (eigenvalues {eig}); '
                                         'no jitter is injected; diagnose the degenerate rows')
        return G, S, L

    def proposal(self, lr, observed, first=False):
        layout, origin, index = self._work_layout()
        self.P['prop'] = self.P['it'].clone()
        self.P['zprop'] = self.P['z'].clone()
        self.ex.v['prop'] = self.ex.v['it'].clone()
        self.ex.v['zprop'] = self.ex.v['z'].clone()
        nis = torch.zeros(self.N, dtype=torch.float64, device=self.dev)
        if bool(observed.any()):
            self.counters['max_working_dimension'] = max(self.counters.get('max_working_dimension', 0), int(layout.dims[observed].max()))
        for pts in chunk_plan(torch.where(observed)[0], layout.dims, self.budget, tensors=PROPOSAL_TENSORS_PER_CHUNK):
            mu, Sigma, a, loc, extras = self._dense(pts, 'it', layout, origin)
            H, r, Rw, valid, _ = self._system(pts, lr, loc, index, mu.shape[1])
            G, S, L = self._factor(H, Sigma, Rw, valid)
            nu = r+(H@(a-mu)[:, :, None]).squeeze(-1)
            x = torch.cholesky_solve(nu[:, :, None], L)
            self._scatter(pts, mu+(G@x).squeeze(-1), 'prop', extras)
            self._scatter(pts, (H.transpose(1, 2)@x).squeeze(-1), 'zprop', extras)
            if first:
                nis[pts] = (nu*x.squeeze(-1)).sum(1)
        return nis

    # ---------- per-point MAP objective (current sensor errors profiled out exactly)
    def objective(self, lr, name='it', zname='z'):
        """J(a) = 1/2 delta_o^T Sigma_oo^{-1} delta_o + 1/2 sum_rows e^T (R_current + R_warp)^{-1} e,  e = y - h(a_o).
        Iterates satisfy delta = Sigma z, so the prior term is 1/2 delta^T z without inverting Sigma; the fresh
        current errors eta_t (independent prior R_current, enters f with +I) are eliminated analytically."""
        cost = torch.zeros(self.N, dtype=torch.float64, device=self.dev)
        cost.index_add_(0, self.entry_point, (self.P[name]-self.mu)*self.P[zname])
        old = self.ex.frame != self.k
        if bool(old.any()):
            cost.index_add_(0, self.ex.point[old], (self.ex.v[name][old]*self.ex.v[zname][old]).sum(1))
        if len(lr):
            e = lr.rows.y-lr.rows.h
            R = torch.diag_embed(lr.current_variance)+self.warp
            cost.index_add_(0, lr.rows.point, (e*torch.linalg.solve(R, e.unsqueeze(-1)).squeeze(-1)).sum(1))
        return cost/2

    def _metric(self, a, za, b, zb):
        m = torch.zeros(self.N, dtype=torch.float64, device=self.dev)
        m.index_add_(0, self.entry_point, (self.P[b]-self.P[a])*(self.P[zb]-self.P[za]))
        if len(self.ex):
            m.index_add_(0, self.ex.point, ((self.ex.v[b]-self.ex.v[a])*(self.ex.v[zb]-self.ex.v[za])).sum(1))
        return m

    # ---------- nonlinear iterations with the existing common-domain acceptance rules
    def solve(self, max_iters=10, tol=1e-8):
        N, C, frame = self.N, self.C, self.frame
        lr = self.linearize('it')
        signature = point_domains(lr.rows, N, C)
        initial_outside = outside_points(lr.owners, frame, N)
        ever_accepted = torch.zeros(N, dtype=torch.bool, device=self.dev)
        point_converged = torch.zeros(N, dtype=torch.bool, device=self.dev)
        observed0 = torch.zeros(N, dtype=torch.bool, device=self.dev)
        observed0[lr.rows.point] = True
        history, converged, nis = [], not len(lr), None
        prior_cost = self.objective(lr)
        for it in range(max_iters if len(lr) else 0):
            observed = torch.zeros(N, dtype=torch.bool, device=self.dev)
            observed[lr.rows.point] = True
            step_nis = self.proposal(lr, observed, first=it == 0)
            if it == 0:
                nis = step_nis
            old_point = self.objective(lr)
            old = float(old_point.sum())
            self._set('base', 'it')
            self._set('zbase', 'z')
            base_owner = lr.owners
            proposal_metric = self._metric('it', 'z', 'prop', 'zprop')
            point_converged |= observed & (proposal_metric < tol)
            pending = observed.clone()
            accepted_points = torch.zeros_like(pending)
            trial_cost = None
            for ls in range(25):
                rate = 2.**-ls
                self._set('trial', 'it')
                self._set('ztrial', 'z')
                self._set('trial', 'base', 'prop', rate, pending)
                self._set('ztrial', 'zbase', 'zprop', rate, pending)
                tr = self.linearize('trial', register=False)
                domain = (point_domains(tr.rows, N, C) == signature).all(dim=(0, 2))
                feasible = ~outside_points(tr.owners, frame, N) | initial_outside
                costs = self.objective(tr, 'trial', 'ztrial')
                descent = costs <= old_point+1e-12*old_point.abs().clamp_min(1.)
                good = pending & domain & feasible & descent
                reasons = {'common_domain_changed': int((pending & ~domain).sum()), 'outside_mask': int((pending & domain & ~feasible).sum()),
                           'no_map_descent': int((pending & domain & feasible & ~descent).sum())}
                if ls == 0:
                    full_step_rejections = reasons
                self._set('it', 'trial', points=good)
                self._set('z', 'ztrial', points=good)
                accepted_points |= good
                pending &= ~good
                trial_cost = float(costs.sum())
                if not bool(pending.any()):
                    break
            for verification in range(25):
                vr = self.linearize('it')
                bad = ~(point_domains(vr.rows, N, C) == signature).all(dim=(0, 2)) | (outside_points(vr.owners, frame, N) & ~initial_outside)
                if not bool(bad.any()):
                    break
                before = torch.where(base_owner.reason == 0, base_owner.pixel, -1)
                after = torch.where(vr.owners.reason == 0, vr.owners.pixel, -1)
                bad |= (before != after).any(0)
                rollback = bad & accepted_points
                if not bool(rollback.any()):
                    rollback = accepted_points.clone()
                self._set('it', 'base', points=rollback)
                self._set('z', 'zbase', points=rollback)
                accepted_points[rollback] = False
            else:
                self._set('it', 'base')
                self._set('z', 'zbase')
                accepted_points[:] = False
                vr = self.linearize('it')
            lr = vr
            accepted = bool(accepted_points.any())
            ever_accepted |= accepted_points
            actual = self._metric('base', 'zbase', 'it', 'z')
            metric = float(actual.max()) if N else 0.
            history.append({'iteration': it, 'objective': old, 'proposal_objective': trial_cost, 'accepted': accepted,
                            'step_prior_metric_max': metric, 'backtracks': ls, 'accepted_points': int(accepted_points.sum()),
                            'rejected_points': int((pending & ~accepted_points).sum()), 'rollback_verifications': verification,
                            'full_step_rejections': full_step_rejections, 'smallest_step_rejections': reasons,
                            'reason': 'ACCEPTED' if accepted else 'COMMON_DOMAIN_OR_MASK_OR_DESCENT_REJECTED'})
            if not accepted:
                break
            unresolved = (~accepted_points) & observed & (proposal_metric >= tol)
            if metric < tol and not bool(unresolved.any()):
                converged = True
                break
        observed = torch.zeros(N, dtype=torch.bool, device=self.dev)
        observed[lr.rows.point] = True
        self.final = lr
        self.assimilated = observed & ever_accepted
        final_cost = self.objective(lr)
        return {'history': history, 'converged': converged, 'point_converged': point_converged | ~observed,
                'assimilated_points': int(self.assimilated.sum()), 'eligible_but_rejected_points': int((observed & ~ever_accepted).sum()),
                'observed_points': int(observed.sum()), 'observed_at_prior': int(observed0.sum()),
                'nonconverged_points': int((observed & ~point_converged).sum()),
                'predictive_nis': nis if nis is not None else torch.zeros(N, dtype=torch.float64, device=self.dev),
                'prior_objective': float(prior_cost.sum()), 'accepted_objective': float(final_cost.sum()),
                'final_point_objective': final_cost, 'counters': dict(self.counters),
                'reference_uncertainty': QUALIFIER, 'fixed_prior': True,
                'objective_scope': 'fixed current pixel/owner/reference domain; valid reference cells refreshed; '
                                   'current sensor errors profiled analytically; R_warp only additional discrepancy'}

    # ---------- the ONE posterior covariance, then exact local marginalization
    def _row_codes(self, lr, rows):
        return torch.cat((lr.key_code[rows].reshape(-1), lr.cur_code[rows]))

    def _retention(self, layout, refs):
        return retention_mask(layout, refs, self.chart, self.k, self.ordinal, self.lag)

    def _expire(self, layout, keep, refs):
        return record_expiry(layout, keep, self.expired_upto, self.frame_ordinal)

    def posterior(self, refs, traced=(), joseph_check=True):
        lr, prior, dev = self.final, self.prior, self.dev
        assim = self.assimilated
        rows = torch.where(assim[lr.rows.point])[0]
        codes = self._row_codes(lr, rows)
        xi = self.ex.lookup(codes)
        xi = torch.unique(xi[xi >= 0])
        post_layout, origin = _concat_with_origin(prior.layout, self.ex.layout(xi))
        index = BlockIndex(post_layout, self.codec)
        keep = self._retention(post_layout, refs)
        expired = self._expire(post_layout, keep, refs)
        new = LocalGaussian.empty(post_layout.select(keep))
        N = self.N
        s = torch.zeros((N, 6), dtype=torch.float64, device=dev)
        P = torch.zeros((N, 6, 6), dtype=torch.float64, device=dev)
        S_min = torch.full((N,), float('nan'), dtype=torch.float64, device=dev)
        psd_min = torch.zeros(N, dtype=torch.float64, device=dev)
        asym = torch.zeros(N, dtype=torch.float64, device=dev)
        norms = {name: torch.zeros(N, dtype=torch.float64, device=dev) for name in
                 ('position_velocity', 'current_reference_state', 'current_state_raw_error', 'retained_current_reference_state', 'retained_current_state_raw_error')}
        joseph_gap, traces, bad_psd = 0., {}, False
        traced = set(int(i) for i in traced)
        for pts in chunk_plan(torch.arange(N, device=dev), post_layout.dims, self.budget):
            mu, Sigma, a, loc, _ = self._dense(pts, 'it', post_layout, origin, xi)
            n = mu.shape[1]
            mean, cov = mu.clone(), Sigma.clone()
            hit = assim[pts]
            if bool(hit.any()):
                sub = pts[hit]
                sl = torch.full((N,), -1, dtype=torch.long, device=dev)
                sl[sub] = torch.arange(len(sub), device=dev)
                H, r, Rw, valid, _ = self._system(sub, lr, sl, index, n)
                Sg = Sigma[hit]
                G, S, L = self._factor(H, Sg, Rw, valid)
                K = torch.cholesky_solve(G.transpose(1, 2), L).transpose(1, 2)
                A = torch.eye(n, dtype=torch.float64, device=dev)-K@H
                post = A@Sg@A.transpose(1, 2)+K@Rw@K.transpose(1, 2)
                post = (post+post.transpose(1, 2))/2
                if joseph_check:
                    simple = A@Sg
                    scale = post.diagonal(dim1=1, dim2=2).clamp_min(0).sqrt()
                    den = (scale[:, :, None]*scale[:, None, :]).clamp_min(torch.finfo(post.dtype).tiny)
                    joseph_gap = max(joseph_gap, float(((post-simple).abs()/den).amax()))
                Sp = S.clone()
                dmax = torch.where(valid, Sp.diagonal(dim1=1, dim2=2), torch.zeros_like(r)).amax(1)
                Sp = Sp+torch.diag_embed((~valid).to(Sp.dtype)*(dmax-1)[:, None])
                S_min[sub] = batched_eigvalsh(Sp)[:, 0]
                mean[hit] = a[hit]
                cov[hit] = post
            s[pts] = mean[:, :6]
            P[pts] = cov[:, :6, :6]
            self._cross_norms(norms, pts, post_layout, cov, '')
            n_new = int(new.layout.dims[pts].max())
            cols, valid_cols = kept_columns(post_layout, keep, pts, n_new)
            m, c = select_dense(mean, cov, cols, valid_cols)
            ev, gap, bad = symmetric_psd_certificate(c, new.layout.dims[pts])
            psd_min[pts], asym[pts] = ev, gap
            bad_psd |= bad
            self._cross_norms(norms, pts, new.layout, c, 'retained_')
            new.scatter(pts, m, c)
            for i in traced & set(pts.tolist()):
                b = int(torch.where(pts == i)[0][0])
                traces[i] = {'posterior_layout': post_layout.describe(i), 'retained_layout': new.layout.describe(i),
                             'mean_plus': mean[b, :int(post_layout.dims[i])].tolist(),
                             'Sigma_plus': cov[b, :int(post_layout.dims[i]), :int(post_layout.dims[i])].tolist()}
        if not bool(torch.isfinite(new.cov).all()) or not bool(torch.isfinite(new.mean).all()) or bad_psd:
            raise LocalCovarianceFailure('LOCAL_POSTERIOR_NOT_FINITE_OR_NEGATIVE_VARIANCE')
        tol_psd = -1e-9
        if bool((psd_min < tol_psd).any()):
            raise LocalCovarianceFailure(f'LOCAL_POSTERIOR_NOT_PSD: min correlation eigenvalue {float(psd_min.min())}')
        L6, info = torch.linalg.cholesky_ex(P)
        if bool((info != 0).any()):
            raise LocalCovarianceFailure('CURRENT_6x6_MARGINAL_NOT_SPD')
        return new, s, P, {'expired_raw_keys': expired, 'innovation_min_eigenvalue': S_min,
                           'local_correlation_min_eigenvalue': psd_min, 'relative_asymmetry_max': float(asym.max()) if N else 0.,
                           'joseph_vs_algebraic_max_correlation_units': joseph_gap, 'cross_norms': norms,
                           'posterior_dimensions': post_layout.dims, 'traces': traces, 'retained_blocks': int(keep.sum()),
                           'posterior_blocks': len(post_layout)}

    @staticmethod
    def _cross_norms(norms, pts, layout, cov, prefix):
        n = cov.shape[1]
        kind = torch.full((len(pts), n), -1, dtype=torch.long, device=cov.device)
        loc = torch.full((layout.N,), -1, dtype=torch.long, device=cov.device)
        loc[pts] = torch.arange(len(pts), device=cov.device)
        j = torch.where(loc[layout.point] >= 0)[0]
        rep = torch.repeat_interleave(j, layout.size[j])
        e = torch.arange(len(rep), device=cov.device)-torch.repeat_interleave(torch.cumsum(layout.size[j], 0)-layout.size[j], layout.size[j])
        kind[loc[layout.point[rep]], layout.start[rep]+e] = torch.where((layout.kind[rep] == STATE) & (layout.start[rep] == 0),
                                                                       torch.full_like(rep, 9), layout.kind[rep])
        top = cov[:, :6, :]
        refcol = ((kind == STATE) | (kind == CHART))[:, None, :]
        keycol = (kind == KEY)[:, None, :]
        if not prefix:
            norms['position_velocity'][pts] = cov[:, :3, 3:6].square().sum((1, 2)).sqrt()
        norms[prefix+'current_reference_state'][pts] = (top*refcol).square().sum((1, 2)).sqrt()
        norms[prefix+'current_state_raw_error'][pts] = (top*keycol).square().sum((1, 2)).sqrt()


def _concat_with_origin(first, second):
    point = torch.cat((first.point, second.point))
    order = torch.argsort(point, stable=True)
    layout = Layout(first.N, point[order], torch.cat((first.kind, second.kind))[order], torch.cat((first.camera, second.camera))[order],
                    torch.cat((first.frame, second.frame))[order], torch.cat((first.pixel, second.pixel))[order])
    return layout, order


def current_marginals(g):
    """(N,6) current mean and (N,6,6) marginal from packed storage (first block is the current state)."""
    dev = g.mean.device
    a = torch.arange(6, device=dev)
    s = g.mean[g.moff[:-1, None]+a]
    d = g.layout.dims
    P = g.cov[g.coff[:-1, None, None]+a[None, :, None]*d[:, None, None]+a[None, None, :]]
    return s, P
