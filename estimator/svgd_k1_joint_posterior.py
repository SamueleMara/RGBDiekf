"""K = 1 joint pipeline, Rao-Blackwellized static/moving posterior (spec section 4).

    F(T, pi) = log p(nu) + log p(h) + sum_i log[(1 - pi) L_i^S + pi L_i^M(T)]        (labels summed exactly, log-sum-exp)
    gamma_i(T, pi) = sigmoid(logit pi + B_i(T)),   B_i = log L_i^M(T) - log L_i^S
    log p(h) = a_pi log pi + b_pi log(1 - pi) - log B(a_pi, b_pi),   h = logit pi   (Beta prior with the logit Jacobian)
dF/dnu = dlog p(nu)/dnu + sum_i gamma_i dlog L_i^M/dnu;  dF/dh = sum_i (gamma_i - pi) + a_pi (1 - pi) - b_pi pi.

Exact gradient with frame streaming (memory independent of T): pass 1 accumulates f, g, H of every point over all frames without a graph;
the closed-form evidence (Laplace at the prior mean / CRN-QMC fallback, estimator/svgd_k1_reference.py) and F are then differentiated
w.r.t. those accumulators (adjoints); pass 2 recomputes each frame with a graph to the pose and back-propagates only the adjoint-weighted
sum. This is the chain rule, not an approximation (tests compare it with direct autograd and finite differences).
At T_t = I for all t, L^M = L^S bitwise and gamma = pi.
"""
import math
import torch
from estimator.svgd_k1_observation import log_potential
from estimator.svgd_k1_reference import laplace_at_mean, sobol_normal
from estimator.svgd_k1_motion import compose


def log_prior_h(h, a, b):
    return a*torch.nn.functional.logsigmoid(h)+b*torch.nn.functional.logsigmoid(-h)-(math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b))


def _J(R):
    J = torch.zeros(6, 6, dtype=R.dtype, device=R.device); J[:3, :3] = R; J[3:, 3:] = torch.eye(3, dtype=R.dtype, device=R.device); return J


class JointPosterior:
    def __init__(self, frames, mu, Lchol, a_eff, prior, beta_ab, tile=4096, qmc_nodes=128, qmc_seed=0, force_qmc=False):
        """frames: [(FrameKDE, Background, rho_t)] for t = 1..T (device tensors); mu [N,6], Lchol [N,6,6], a_eff [N]."""
        self.frames, self.mu, self.L, self.a, self.prior = frames, mu, Lchol, a_eff, prior
        self.ab, self.tile, self.qn, self.qs = beta_ab, tile, qmc_nodes, qmc_seed
        self.lS, self.force_qmc = None, force_qmc     # force_qmc: validated higher-accuracy route (CRN-QMC for every point)

    # ---------------------------------------------------------------- evidence of every point for one pose sequence
    def _accumulate(self, poses):
        N, dev, dt = len(self.mu), self.mu.device, self.mu.dtype
        f = torch.zeros(N, dtype=dt, device=dev); G = torch.zeros(N, 6, dtype=dt, device=dev); H = torch.zeros(N, 6, 6, dtype=dt, device=dev)
        cert = torch.zeros(N, dtype=dt, device=dev)
        with torch.no_grad():
            for t, (kde, bg, rho) in enumerate(self.frames, start=1):
                R, tr = poses[t, :3, :3], poses[t, :3, 3]; J = _J(R)
                for i in range(0, N, self.tile):
                    sl = slice(i, i+self.tile)
                    ll, g, Hh, c = log_potential(kde, bg, rho, self.mu[sl, :3]@R.T+tr, self.mu[sl, 3:])
                    f[sl] += ll; G[sl] += g@J; H[sl] += J.T@Hh@J; cert[sl] += c
        return f, G, H, cert

    def _qmc_values(self, poses, idx):
        z = sobol_normal(self.qn, self.qs, self.mu.dtype, self.mu.device)
        U = (self.mu[idx][:, None, :]+torch.einsum('nij,sj->nsi', self.L[idx], z)).reshape(-1, 6)
        fq = torch.zeros(len(U), dtype=U.dtype, device=U.device)
        with torch.no_grad():
            for t, (kde, bg, rho) in enumerate(self.frames, start=1):
                R, tr = poses[t, :3, :3], poses[t, :3, 3]
                for i in range(0, len(U), self.tile):
                    fq[i:i+self.tile] += log_potential(kde, bg, rho, U[i:i+self.tile, :3]@R.T+tr, U[i:i+self.tile, 3:], need_hessian=False)[0]
        return U, fq.reshape(len(idx), self.qn)

    def log_evidence(self, poses):
        """(log L [N], diag) without gradients (used for the static evidence and diagnostics)."""
        f, G, H, cert = self._accumulate(poses); lM, ok = laplace_at_mean(f, G, H, self.L, self.a)
        if self.force_qmc: ok = torch.zeros_like(ok)
        bad = torch.nonzero(~ok)[:, 0]
        if len(bad):
            _, fq = self._qmc_values(poses, bad); lM = lM.clone(); lM[bad] = torch.logsumexp(self.a[bad, None]*fq, 1)-math.log(self.qn)
        return lM, {'qmc_fallback_points': int(len(bad)), 'certificate_max_sum_over_frames': float(cert.max()) if len(cert) else 0.}

    def set_static(self):
        I = torch.eye(4, dtype=self.mu.dtype, device=self.mu.device).expand(len(self.frames)+1, 4, 4)
        self.lS, self.static_diag = self.log_evidence(I)
        return self.lS

    # ---------------------------------------------------------------- F and its exact gradient
    def evaluate(self, nu, h, grad=True):
        nu = nu.detach(); h = h.detach()
        with torch.no_grad(): poses = compose(nu, self.prior.dts)
        f, G, H, cert = self._accumulate(poses)
        leaves = [x.clone().requires_grad_(grad) for x in (f, G, H)]; hl = h.clone().requires_grad_(grad)
        lM, ok = laplace_at_mean(*leaves, self.L, self.a)
        if self.force_qmc: ok = torch.zeros_like(ok)
        bad = torch.nonzero(~ok)[:, 0]
        if len(bad):
            U, fq = self._qmc_values(poses, bad); fq = fq.requires_grad_(grad)
            lM = lM.clone(); lM[bad] = torch.logsumexp(self.a[bad, None]*fq, 1)-math.log(self.qn)
        lpi, l1pi = torch.nn.functional.logsigmoid(hl), torch.nn.functional.logsigmoid(-hl)
        data = torch.logaddexp(l1pi+self.lS, lpi+lM).sum()
        out = {'logLM': lM.detach(), 'B': (lM-self.lS).detach(), 'gamma': torch.sigmoid(h+lM.detach()-self.lS),
               'diag': {'qmc_fallback_points': int(len(bad)), 'certificate_max_sum_over_frames': float(cert.max()) if len(cert) else 0.}}
        nul = nu.clone().requires_grad_(grad); lp = self.prior.log_prob(nul)+log_prior_h(hl, *self.ab)
        out['F'] = float(data)+float(lp)
        if not grad: return out
        wanted = leaves+([fq] if len(bad) else [])+[hl]
        adj = torch.autograd.grad(data, wanted, allow_unused=True)
        wf, wG, wH = adj[0], adj[1], adj[2]; wq = adj[3] if len(bad) else None; gh_data = adj[-1]
        g_nu_p, g_h_p = torch.autograd.grad(lp, (nul, hl))
        g_nu = g_nu_p.clone()
        # pass 2: frame by frame, back-propagate the adjoint-weighted per-frame quantities into the twists
        nv = nu.clone().requires_grad_(True); P = compose(nv, self.prior.dts)
        for t, (kde, bg, rho) in enumerate(self.frames, start=1):
            R = P[t, :3, :3].detach().requires_grad_(True); tr = P[t, :3, 3].detach().requires_grad_(True)   # per-frame leaves
            gR = torch.zeros_like(R); gt = torch.zeros_like(tr)
            for i in range(0, len(self.mu), self.tile):                         # each chunk back-propagated at once: bounded graph
                sl = slice(i, i+self.tile); J = _J(R)
                ll, g, Hh, _ = log_potential(kde, bg, rho, self.mu[sl, :3]@R.T+tr, self.mu[sl, 3:])
                si = (wf[sl]*ll).sum()+(wG[sl]*(g@J)).sum()+(wH[sl]*(J.T@Hh@J)).sum()
                a, b = torch.autograd.grad(si, (R, tr), allow_unused=True)
                gR += a if a is not None else 0.; gt += b if b is not None else 0.
            if len(bad):
                w = wq.reshape(-1)
                for i in range(0, len(U), self.tile):
                    Ui = U[i:i+self.tile]
                    si = (w[i:i+self.tile]*log_potential(kde, bg, rho, Ui[:, :3]@R.T+tr, Ui[:, 3:], need_hessian=False)[0]).sum()
                    a, b = torch.autograd.grad(si, (R, tr), allow_unused=True)
                    gR += a if a is not None else 0.; gt += b if b is not None else 0.
            g_nu += torch.autograd.grad((P[t, :3, :3], P[t, :3, 3]), nv, (gR, gt), retain_graph=True)[0]  # chain through the pose composition
        out['grad_nu'] = g_nu; out['grad_h'] = gh_data+g_h_p
        return out


def enumerate_exact(logLS, logLM, h, a, b):
    """Dense 2^N oracle (N <= ~12): F by explicit summation over all labelings and exact label marginals."""
    import itertools
    N = len(logLS); pi = 1/(1+math.exp(-h)); terms, z_all = [], []
    for z in itertools.product((0, 1), repeat=N):
        terms.append(sum((math.log(pi)+logLM[i]) if z[i] else (math.log(1-pi)+logLS[i]) for i in range(N))); z_all.append(z)
    m = max(terms); w = [math.exp(t-m) for t in terms]; Z = sum(w)
    gam = [sum(wk*zk[i] for wk, zk in zip(w, z_all))/Z for i in range(N)]
    lph = a*math.log(pi)+b*math.log(1-pi)-(math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b))
    return m+math.log(Z)+lph, gam
