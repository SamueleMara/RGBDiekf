"""Operator E: static/dynamic model evidence and point marginals (PDF (29)-(34), spec E1-E5; contract E25-E33).

Exact pieces (validated against quadrature / enumeration in tests/test_svgd_support_posterior.py):
    linear-Gaussian evidence  log L = log N(y; A mu, A Sigma A^T + R)                          (E29, closed form)
    Laplace                   log L ~ -E(theta*) + d/2 log 2 pi - 1/2 log det Hess E(theta*)    (E31, MODEL_APPROX)
    binary Bayes rule         P(H1 | y) = pi L1 / (pi L1 + (1 - pi) L0)  (log-sum-exp)          (E30)
    marginal                  gamma_i = sum_h P(h | y) m_i(h) over an enumerated coherent hypothesis set (E27)
    seed risk                 FDR(S) = sum_{i in S} (1 - gamma_i) / max(1, |S|) <= alpha        (E33)
Production (support_posterior) is BLOCKED_MODEL_UNSPECIFIED until the support prior p(m, Z), the moving-motion prior,
the visibility / unexplained-support laws and a calibrated Gamma are declared: it then returns NO probabilities, and the
runner writes no p_dynamic.npy (spec 8.4, Gate 6). Never a sigmoid of displacement, strain or Stein force.
"""
import itertools
import math
import numpy as np

PROBABILITY_STATUS = {0: 'SUPPORTED_STATIC_OR_DYNAMIC', 1: 'NO_OBSERVATION', 2: 'SCORE_OUT_OF_SUPPORT', 3: 'NON_IDENTIFIABLE',
                      4: 'NUMERICAL_FAILURE', 5: 'UNRESOLVED_MODEL'}
SEMANTICS = ('APPROXIMATE_MODEL_POSTERIOR', 'EXACT_TINY_MODEL_POSTERIOR')


def gaussian_logpdf(y, mean, cov):
    L = np.linalg.cholesky(cov); r = np.linalg.solve(L, y-mean)
    return float(-.5*r@r-np.log(np.diag(L)).sum()-.5*len(y)*math.log(2*math.pi))


def linear_gaussian_log_evidence(y, A, mu, Sigma, R):
    return gaussian_logpdf(np.asarray(y, float), A@mu, A@Sigma@A.T+R)


def laplace_log_evidence(E_opt, hessian):
    H = (np.asarray(hessian, float)+np.asarray(hessian, float).T)/2
    ev = np.linalg.eigvalsh(H)
    if not ev.min() > 0: raise FloatingPointError('NUMERICAL_FAILURE: Laplace Hessian not positive definite in the chosen gauge')
    return float(-E_opt+.5*len(H)*math.log(2*math.pi)-.5*np.log(ev).sum())


def binary_posterior(log_L1, log_L0, prior_pi):
    if not 0 < prior_pi < 1: raise ValueError('prior_pi must be in (0, 1)')
    z = math.log(prior_pi)+log_L1-math.log1p(-prior_pi)-log_L0
    p1 = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
    p0 = 1/(1+math.exp(z)) if z <= 0 else math.exp(-z)/(1+math.exp(-z))
    return p1, p0, log_L1-log_L0


def set_partitions(items):
    items = list(items)
    if not items: yield []; return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)): yield p[:i]+[[first]+p[i]]+p[i+1:]
        yield [[first]]+p


def coherent_hypotheses(n):
    """Every (partition of n points into rigid supports, static/moving label per support): coherent Z (transitive by
    construction) and m. Returns list of (blocks, labels, m [n] int)."""
    out = []
    for blocks in set_partitions(range(n)):
        for labels in itertools.product((0, 1), repeat=len(blocks)):
            m = np.zeros(n, int)
            for b, l in zip(blocks, labels): m[b] = l
            out.append((blocks, labels, m))
    return out


def enumerate_marginals(m, log_evidence, log_prior):
    """m [H, N] labels, log evidence and log prior [H]. Exact posterior over the enumerated set and gamma [N]."""
    lp = np.asarray(log_evidence, float)+np.asarray(log_prior, float)
    mx = lp.max(); w = np.exp(lp-mx); Z = w.sum(); post = w/Z
    return post@np.asarray(m, float), post, float(mx+math.log(Z))


def edge_marginals(hyps, post, n):
    """P(z_il = 1) from coherent partitions (marginals of a coherent joint; need not be individually transitive)."""
    W = np.zeros((n, n))
    for (blocks, _, _), p in zip(hyps, post):
        for b in blocks:
            for i in b:
                for l in b: W[i, l] += p
    return W


def select_seeds(gamma, original_ids, alpha, semantics):
    """Largest top-gamma prefix with posterior expected FDR <= alpha (E33). Refuses uncalibrated scores."""
    if semantics not in SEMANTICS: raise ValueError(f'PROBABILITY_NOT_YET_ESTABLISHED: seeds need {SEMANTICS}, got {semantics}')
    g = np.asarray(gamma, float); ok = np.isfinite(g); order = np.argsort(-np.where(ok, g, -np.inf), kind='stable')
    order = order[ok[order]]
    fdr = np.cumsum(1-g[order])/np.arange(1, len(order)+1)
    k = int(np.nonzero(fdr <= alpha)[0].max()+1) if (fdr <= alpha).any() else 0
    return np.asarray(original_ids)[order[:k]], float(fdr[k-1]) if k else 0.


def support_posterior(model_contract):
    """Production entry. Returns the BLOCKED record while E25/E26 parameters are not usable (no fabricated gamma)."""
    probs = model_contract['stage_parameter_problems']['MODEL_EVIDENCE']
    if probs:
        return {'status': 'BLOCKED_MODEL_UNSPECIFIED', 'probability_semantics': 'PROBABILITY_NOT_YET_ESTABLISHED', 'gamma': None,
                'missing': probs, 'note': 'no normalized support posterior: support prior p(m,Z), moving-motion prior, visibility and '
                                          'unexplained-support laws and a calibrated Gamma are not declared in the theory'}
    return {'status': 'BLOCKED_THEORY_FIDELITY', 'probability_semantics': 'PROBABILITY_NOT_YET_ESTABLISHED', 'gamma': None,
            'missing': [], 'note': 'parameters declared but the production evidence approximation is not implemented/validated (Gate 6)'}
