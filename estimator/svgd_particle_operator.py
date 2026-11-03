"""Operator B: restricted physical-particle Stein transport (PDF (13)-(15), spec B1-B3; contract E05-E07).

    K(l,i) = exp(-1/2 (x_l-x_i)^T HX^-1 (x_l-x_i) - 1/2 (c_l-c_i)^T HC^-1 (c_l-c_i))          (colour term off: kernel_color=False)
    attraction_i = 1/N sum_{l=1..N} K(l,i) g(x_l, c_l)
    repulsion_i  = 1/N sum_{l=1..N} grad_{x_l} K(l,i) = 1/N sum_l K(l,i) HX^-1 (x_i - x_l)
    Phi_i = attraction_i + repulsion_i,    proposal x_i <- x_i + eta Phi_i (accepted by the caller against the common objective)

Label: restricted_physical_particle_stein_transport. The physical points are NOT samples of one posterior; Liu & Wang's
KL-descent theorem is not claimed. All N^2 pairs are evaluated in tiles (exact; 1/N is the global particle count, never a
neighbour count or degree); no cutoff, no clipping, no speed gate. Colours are fixed; derivatives act on XYZ only.
Switches (spec 4.1 ablations): attraction / repulsion / kernel_color; stein=off is the caller's (MMD-only motion).
kernel_mass_i = sum_l K(l,i) is stored as a diagnostic (and for the declared initial step policy).
"""
from dataclasses import dataclass
import math
import torch
from estimator.svgd_target_score import _precision, _tiles, quad_coefficients, phi


@dataclass
class SteinResult:
    attraction: torch.Tensor     # [N,3] 1/m
    repulsion: torch.Tensor      # [N,3] 1/m
    total: torch.Tensor          # [N,3] 1/m
    kernel_mass: torch.Tensor    # [N]  sum_l K(l,i)
    pair_count: int              # N (all pairs)
    truncation_error_bound: float    # 0: exact


def stein_field(x, c, g, H_x, H_c=None, attraction=True, repulsion=True, kernel_color=True, workspace_bytes=512*2**20):
    N, dt, dev = len(x), x.dtype, x.device
    Px, _ = _precision(torch.as_tensor(H_x, dtype=dt, device=dev), 'H_X')
    if kernel_color and H_c is None: raise ValueError('kernel_color needs H_C')
    Pc = _precision(torch.as_tensor(H_c, dtype=dt, device=dev), 'H_C')[0] if kernel_color else None
    att = torch.zeros(N, 3, dtype=dt, device=dev); rep = torch.zeros_like(att); mass = torch.zeros(N, dtype=dt, device=dev)
    if attraction and not bool(torch.isfinite(g).all()): raise FloatingPointError('NUMERICAL_FAILURE: non-finite target score entering the Stein field')
    ni, nl = _tiles(N, N, workspace_bytes, per_pair=4*8)
    for i0 in range(0, N, ni):                      # matmul form (OPT-02): exponent phi(x_i') a + x_i' b + k_l, centred on the tile
        xi, ci = x[i0:i0+ni], c[i0:i0+ni]; ox, oc = xi.mean(0), ci.mean(0); xp = xi-ox; fx, fc = phi(xp), phi(ci-oc)
        a = torch.zeros(len(xi), 3, dtype=dt, device=dev); kx_sum = torch.zeros(len(xi), 3, dtype=dt, device=dev); ms = torch.zeros(len(xi), dtype=dt, device=dev)
        for l0 in range(0, N, nl):
            xl = x[l0:l0+nl]-ox; Ax, kx, _, _ = quad_coefficients(Px, xl)
            q = fx@Ax.T+kx[None]
            if kernel_color: Ac, kc, _, _ = quad_coefficients(Pc, c[l0:l0+nl]-oc); q = q+fc@Ac.T+kc[None]
            K = torch.exp(-.5*q)
            ms += K.sum(1)
            if attraction: a += K@g[l0:l0+nl]
            if repulsion: kx_sum += K@xl
        att[i0:i0+len(xi)] = a/N; mass[i0:i0+len(xi)] = ms
        if repulsion: rep[i0:i0+len(xi)] = ((xp*ms[:, None]-kx_sum)@Px)/N        # sum_l K HX^-1 (x_i - x_l), HX^-1 symmetric
    return SteinResult(att, rep, att+rep, mass, N, 0.)


def initial_step(lam_x_max_eig, N, kernel_mass):
    """Declared solver policy (stein_numerics.step_policy 'N_lambda_over_kernel_mass'): eta_0 = lambda_max(LX) N / median_i
    sum_l K(l,i), the step that moves a particle with kernel-averaged score g by ~LX g (one target-bandwidth Newton step of
    an isolated Gaussian). A numerical step size, not a velocity, a physical dt or a model parameter."""
    return float(lam_x_max_eig*N/torch.median(kernel_mass).clamp_min(1.))
