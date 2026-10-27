"""Phase 1: grouped image-based iterated EKF over the free per-point state s_i = [x_i, v_i].

Port of RigEuler3DGS 481b2cf2 estimator/sparse_iekf.iekf_frame_step (trust-region path, grouped state)
with the 12D Gaussian state replaced by the 6D point state and the Gaussian rasterizer replaced by the
coloured-point RGB-D rasterizer. Per frame:

    s- = F s+,  P- = F P+ F^T + Q                                   ONCE (process6.predict); prior fixed
    E(s) = 1/2 (s - s-)^T Lambda- (s - s-) + sum_{c,p} rho_cp(y - h(s))        (image_observation)

Inner iteration j (frozen render, IRLS weights W_j, Jacobian H_j):

    b = G^T (H^T W r - Lambda- e),    A v = G^T (Lambda- + H^T W H + lambda I) G v,  lambda = 1e-3
    Steihaug-CG on  min -b^T z + 1/2 z^T A z,  ||z||_S <= radius   (vendored trust_region.steihaug_cg)
    S = ScalingMetric.from_physical_scales((C,12), sigma_mu=.05, sigma_phi=.05, sigma_v=.5, sigma_omega=.5, rms)
    preconditioner: block Jacobi of G^T (Lambda- + blockdiag(F,0) + lambda I) G (+1e-6 tr/12 I)  (group_precond)
    forcing eta = clip(.5 sqrt(|g|/|g0|), 1e-6, 1e-2); rho = actual / predicted (significance floor 1e-12)
    adapt_radius (eta .1/.25/.75); accept on the TRUE objective of a fresh render
    stop: accepted step with rel decrease < 1e-6 AND ||z||_S < 1e-7, radius floor, or 10 iterations

The trust-region damping lambda is a solver term only; it never enters the posterior.
Posterior ONCE per frame at the final linearization (phase1_covariance, named interface; production: restricted-Joseph
marginals under the assumed block density).
"""
from dataclasses import dataclass, asdict
import math
import time
import torch
from .rgbd_point_renderer import render
from . import small_contractions
from .rigeuler_ref.trust_region import (ScalingMetric, TrustRegionConfig, adapt_radius, forcing_eta, predicted_reduction,
                                        steihaug_cg, STOP_FORCING)
from .rigeuler_ref.pcg import block_jacobi_preconditioner
from .phase1_covariance import make_covariance


@dataclass
class IEKFConfig:
    """Pinned sparse_iekf.IEKFConfig values of the trust-region path."""
    max_iters: int = 10
    lm_initial: float = 1e-3
    pcg_max_iters: int = 30
    rel_energy_tol: float = 1e-6
    step_norm_tol: float = 1e-7
    trust_radius_init: float = 1.
    trust_radius_max: float = 100.
    trust_radius_min: float = 1e-8
    trust_eta_accept: float = .1
    trust_eta_weak: float = .25
    trust_eta_good: float = .75
    scale_sigma_mu: float = .05
    scale_sigma_phi: float = .05
    scale_sigma_v: float = .5
    scale_sigma_omega: float = .5
    forcing_c: float = .5
    forcing_min: float = 1e-6
    forcing_max: float = .01
    supernode_target_size: int = 64
    supernode_knn: int = 8
    supernode_seed: int = 0
    posterior: str = 'rigeuler_block_information'    # restricted-Joseph methods diverge at blade scale (see phase1_covariance)
    pyramid: tuple = (1,)                  # (2, 1): coarse-to-fine ABLATION only
    stale_jacobian: bool = False           # ABLATION only: renderer Jacobian of the FIRST linearization in every inner system
    small_contractions: bool = False       # STEP 2B-B1 opt-in: tiny batched GEMMs as broadcast contractions (round-off-equal)

    def manifest(self):
        return asdict(self)

    def trust_region(self):
        return TrustRegionConfig(radius_init=self.trust_radius_init, radius_max=self.trust_radius_max, radius_min=self.trust_radius_min,
                                 eta_accept=self.trust_eta_accept, eta_weak=self.trust_eta_weak, eta_good=self.trust_eta_good,
                                 forcing_c=self.forcing_c, forcing_min=self.forcing_min, forcing_max=self.forcing_max,
                                 max_pcg_iters=self.pcg_max_iters)


def embed(dx):
    out = dx.new_zeros((len(dx), 6)); out[:, :3] = dx
    return out


def group_precond(M, basis):
    """Pinned sparse_iekf.group_precond: G^T M G per group + 1e-6 tr/12 jitter, block Jacobi (preconditioner only)."""
    blocks = basis.blocks(M)
    blocks = (blocks+blocks.transpose(1, 2))/2
    trace = blocks.diagonal(dim1=1, dim2=2).abs().sum(-1).clamp_min(1e-12)
    blocks = blocks+(1e-6*trace/12)[:, None, None]*torch.eye(12, dtype=M.dtype, device=M.device)
    return block_jacobi_preconditioner(blocks)


def linear_system(s, s_minus, Lam, lin, terms, obs, basis, lm):
    """Grouped Gauss-Newton system at one frozen linearization: (b, A_solve, preconditioner)."""
    e = s-s_minus
    b = basis.reduce(embed(lin.apply_HT(terms['residual_adjoint']))-small_contractions.matvec(Lam, e))
    M = Lam.clone(); M[:, :3, :3] += lin.block_fisher(terms['weights'])
    M = M+lm*torch.eye(6, dtype=s.dtype, device=s.device)

    def A_solve(z):
        ds = basis.expand(z)
        meas = embed(lin.apply_HT(obs.apply_W(terms, lin.apply_H(ds[:, :3]))))
        return basis.reduce(small_contractions.matvec(Lam, ds)+meas+lm*ds)
    return b, A_solve, group_precond(M, basis)


def trust_region_solve(s, s_minus, Lam, objective, obs, basis, cfg):
    """The pinned trust-region loop on one objective; returns the accepted iterate and its linearization."""
    tr = cfg.trust_region()
    metric = ScalingMetric.from_physical_scales((basis.count, 12), s.device, s.dtype, sigma_mu=cfg.scale_sigma_mu,
                                                sigma_phi=cfg.scale_sigma_phi, sigma_v=cfg.scale_sigma_v, sigma_omega=cfg.scale_sigma_omega)
    lm = cfg.lm_initial
    E, prior_e, lin, terms = objective(s)
    E0, radius, g0, history, stop = E, cfg.trust_radius_init, None, [], 'iteration_budget'
    lin0 = lin
    for j in range(cfg.max_iters):
        b, A_solve, precond = linear_system(s, s_minus, Lam, lin0 if cfg.stale_jacobian else lin, terms, obs, basis, lm)   # stale: Jacobian of the first render, fresh residual and weights
        grad = float(b.norm())
        if g0 is None: g0 = grad
        eta = forcing_eta(grad, g0, c=cfg.forcing_c, eta_min=cfg.forcing_min, eta_max=cfg.forcing_max)
        sres = steihaug_cg(A_solve, b, radius, metric, M_inv=precond, max_iters=cfg.pcg_max_iters, eta=eta)
        step = basis.expand(sres.step)
        predicted = predicted_reduction(sres.step, b, A_solve)
        Et, pt, lint, termst = objective(s+step)
        actual = E-Et
        rho = actual/predicted if predicted > 1e-12*max(abs(E), 1.) else float('nan')
        radius_before = radius
        radius, accepted = adapt_radius(radius, rho, sres.boundary_hit, actual, tr)
        E_before = E
        if accepted: s, E, prior_e, lin, terms = s+step, Et, pt, lint, termst
        history.append({'iteration': j, 'accepted': accepted, 'objective_before': E_before, 'objective_trial': Et,
                        'predicted_reduction': predicted, 'actual_reduction': actual, 'rho': rho, 'trust_radius_before': radius_before,
                        'trust_radius_after': radius, 'step_norm_scaled': sres.step_norm_scaled, 'boundary_hit': sres.boundary_hit,
                        'pcg_stop_reason': sres.stop_reason, 'pcg_iterations': sres.iterations, 'pcg_rel_residual': sres.rel_residual,
                        'forcing_eta': eta, 'gradient_norm': grad, 'gradient_norm_scaled': metric.scaled_gradient_norm(b),
                        'max_point_step_m': float(step[:, :3].norm(dim=1).max()), 'preconditioner': precond.report()})
        if accepted:
            rel = (E_before-E)/max(abs(E_before), 1.)
            if rel < cfg.rel_energy_tol and sres.step_norm_scaled < cfg.step_norm_tol: stop = 'converged'; break
        elif radius <= tr.radius_min*1.0000001:
            stop = 'trust_radius_floor'; break
    return s, E, E0, prior_e, lin, terms, {'accepted_steps': sum(h['accepted'] for h in history),
                                           'rejected_trials': sum(not h['accepted'] for h in history), 'stop': stop,
                                           'final_trust_radius': radius, 'scaling_metric': metric.to_dict(), 'history': history}


def frame_step(s_minus, cov, levels, rgb, rho, policy, basis, cfg):
    with small_contractions.enabled(cfg.small_contractions):
        return _frame_step(s_minus, cov, levels, rgb, rho, policy, basis, cfg)


def _frame_step(s_minus, cov, levels, rgb, rho, policy, basis, cfg):
    """One video frame: fixed prior, iterated render-based update, ONE posterior.

    cov: the PREDICTED covariance model (phase1_covariance) or, for a single frame, its marginal blocks [N,6,6]
    (wrapped with cfg.posterior; exact for one frame since a block prior has no cross-point covariance).
    levels = [(FrameObservation, cameras, footprint scale)], finest last; production has one level.
    Returns (s+, P+ marginal blocks, observed, final linearization, terms, report); cov is updated in place."""
    clock = time.monotonic()
    if torch.is_tensor(cov): cov = make_covariance(cfg.posterior, cov, basis)
    P_minus = cov.marginals()
    Lam = torch.linalg.inv(P_minus); Lam = (Lam+Lam.transpose(1, 2))/2
    s, reports = s_minus.clone(), []
    for li, (obs, cams, scale) in enumerate(levels):
        def objective(state, obs=obs, cams=cams, scale=scale):
            lin = render(state[:, :3], rgb, rho*scale, cams, policy)
            terms = obs.terms(lin)
            e = state-s_minus
            prior = .5*float(torch.einsum('ni,nij,nj->', e, Lam, e))
            return prior+terms['energy'], prior, lin, terms
        restart = False
        if li == len(levels)-1 and li > 0 and objective(s_minus)[0] < objective(s)[0]: s, restart = s_minus.clone(), True
        s, E, E0, prior_e, lin, terms, rep = trust_region_solve(s, s_minus, Lam, objective, obs, basis, cfg)
        reports.append({'footprint_scale': scale, 'restarted_from_prediction': restart, 'energy_initial': E0, 'energy_final': E, **rep})
    final = reports[-1]
    # Adjoint identity <H v, a> = <v, H^T a> on this frame's final linearization (pinned debug_adjoint_test).
    g = torch.Generator(device='cpu').manual_seed(0)
    v = torch.randn(lin.n, 3, generator=g, dtype=s.dtype).to(s.device)
    a = [torch.randn(len(t_), 5, generator=g, dtype=s.dtype).to(s.device) for t_ in terms['residual_adjoint']]
    lhs = float(sum((h*ai).sum() for h, ai in zip(lin.apply_H(v), a))); rhs = float((v*lin.apply_HT(a)).sum())
    adjoint = {'lhs': lhs, 'rhs': rhs, 'relative_error': abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-300)}
    post_report = cov.update(lin, levels[-1][0], terms, basis)
    post = cov.marginals()
    observed = lin.contributing()
    report = {'energy_initial': reports[0]['energy_initial'], 'energy_final': final['energy_final'], 'prior_energy_final': prior_e,
              'image_energy_final': terms['energy'], 'accepted_steps': sum(r['accepted_steps'] for r in reports),
              'rejected_trials': sum(r['rejected_trials'] for r in reports), 'stop': final['stop'], 'history': final['history'],
              'levels': [{k: v for k, v in r.items() if k != 'history'} for r in reports],
              'directly_observed_points': int(observed.sum()), 'prediction_only_points': int((~observed).sum()),
              'posterior': post_report, 'cameras': terms['cameras'], 'jvp_vjp_adjoint_check': adjoint, 'seconds': time.monotonic()-clock}
    if not math.isfinite(report['energy_final']): raise FloatingPointError('non-finite objective after the update')
    return s, post, observed, lin, terms, report
