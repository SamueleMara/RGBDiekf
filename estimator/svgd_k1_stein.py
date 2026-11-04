"""K = 1 joint pipeline, SVGD over SHARED-MOTION HYPOTHESES only (spec section 5). No physical-point Stein field anywhere.

Particle m: theta_m = (nu_1..nu_T, h) in whitened coordinates v = (L^-1 nu, (h - h0)/sigma_h) (L: Cholesky of the motion prior,
h0 = logit of the Beta prior mean, sigma_h declared). grad_v F = (L^T grad_nu F, sigma_h dF/dh) -- the SAME F as the MAP control.
    phi(v_m) = 1/M sum_n [k(v_n, v_m) grad_{v_n} F(v_n) + grad_{v_n} k(v_n, v_m)],   k = exp(-|v_n - v_m|^2 / h_k)
h_k = med^2 / log M (Liu & Wang median heuristic, recomputed each iteration). Update: Liu & Wang reference AdaGrad
(hist = alpha hist + (1 - alpha) phi^2, v += eps phi / (fudge + sqrt(hist))). Repulsive updates are never rejected by an objective test.
"""
import math
import torch


def svgd_direction(V, G):
    """V [M,d] particles, G [M,d] scores -> (phi [M,d], diagnostics)."""
    M = len(V); D2 = torch.cdist(V, V, compute_mode='donot_use_mm_for_euclid_dist')**2
    med2 = float(torch.median(D2[torch.triu_indices(M, M, 1, device=V.device).unbind()])) if M > 1 else 1.
    hk = med2/math.log(max(M, 2)) if med2 > 0 else 1.
    K = torch.exp(-D2/hk)
    rep = (V*K.sum(1, keepdim=True)-K@V)*(2/hk)                           # sum_n grad_{v_n} k(v_n, v_m) = (2/h) sum_n k (v_m - v_n)
    phi = (K@G+rep)/M
    return phi, {'kernel_bandwidth': hk, 'mean_kernel_mass': float(K.sum(1).mean()), 'repulsion_norm': float(rep.norm()/M), 'attraction_norm': float((K@G).norm()/M)}


class AdaGrad:
    def __init__(self, eps, alpha=.9, fudge=1e-6): self.eps, self.alpha, self.fudge, self.hist = eps, alpha, fudge, None

    def step(self, phi):
        self.hist = phi*phi if self.hist is None else self.alpha*self.hist+(1-self.alpha)*phi*phi
        return self.eps*phi/(self.fudge+torch.sqrt(self.hist))
