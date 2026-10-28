"""Prequential predictive model comparison across frames (contract section 9; replaces chi-square tests that
treated pixels as independent).

For two models A, B scored on the SAME observed frames by their one-step-ahead predictive image NLL
L_k = E_meas(x_k^-) (the frame's objective at the prediction, before its update), the loss differential
d_k = L_k^A - L_k^B is a time series whose mean is the expected predictive advantage of B. Pixels inside a frame
are never treated as independent: each frame contributes one number, and serial dependence across frames is
handled by a conservative long-run variance of the residuals around a linear time trend (stationary formulas read a
trend as serial correlation), the larger of
    Newey-West (Bartlett)   gamma_0 + 2 sum_{l=1..L} (1 - l/(L+1)) gamma_l,  L = floor(4 (n/100)^(2/9))  (Newey & West 1994)
    AR(1) parametric        gamma_0 / (1 - r)^2,  r the lag-1 autocorrelation (clipped to |r| <= 0.97)
Simulated at n = 56 (tests/test_image_rigid_joint.py), Newey-West alone rejects a true H0 13% (AR(1) 0.5) to 21%
(AR(1) 0.7) of the time at nominal 5%; the detrended, bias-corrected (r + (2+4r)/n) AR(1) form gives 4.9-6.7% for AR(1)
0 to 0.7 and keeps power 0.94 on a trending alternative; the maximum keeps validity when the dependence is not AR(1).
These choices were made on simulated series only.

The Diebold-Mariano statistic DM = dbar / sqrt(s^2/n) is used with the Harvey-Leybourne-Newbold small-sample
form for one-step forecasts, DM* = DM sqrt((n-1)/n), referred to Student-t with n-1 degrees of freedom.
"""
import math
import numpy as np
from scipy import stats


def newey_west_lag(n):
    return int(math.floor(4*(n/100.)**(2/9)))


def long_run_variance(d, lag=None):
    d = np.asarray(d, float); n = len(d)
    lag = newey_west_lag(n) if lag is None else lag
    e = d-d.mean()
    s2 = e@e/n
    for l in range(1, min(lag, n-1)+1):
        s2 += 2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return max(s2, 0.), lag


def conservative_long_run_variance(d):
    """max(Newey-West, AR(1)) long-run variance of the DEPENDENCE around a linear time trend: losses of a moving scene trend
    (a hypothesis falls further behind as the motion grows), and a trend read as serial correlation drives r -> 1 and the
    AR(1) factor 1/(1-r)^2 to ~1000, which removed all power exactly where the evidence was strongest (measured on the
    revolute fixture: mean contradiction +107/frame not significant). The test itself is still on the mean."""
    d = np.asarray(d, float); n = len(d)
    t = np.arange(n, dtype=float)-(n-1)/2
    e = d-d.mean()-(t@(d-d.mean()))/max(t@t, 1e-300)*t
    nw, lag = long_run_variance(e)
    r = float((e[1:]@e[:-1])/(e@e)) if e@e > 0 else 0.
    r = r+(2+4*r)/n                                   # first-order small-sample bias of r with intercept and trend removed
    r = min(max(r, -.97), .97)
    return max(nw, (e@e/n)/(1-r)**2), lag, r


def diebold_mariano(loss_a, loss_b, alternative='two-sided'):
    """H0: equal expected predictive loss. alternative 'greater': A worse than B (B predicts better)."""
    la, lb = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if la.shape != lb.shape or la.ndim != 1: raise ValueError('loss series must be aligned 1-D arrays on the same frames')
    d = la-lb; n = len(d)
    if n < 3: return {'n': n, 'statistic': float('nan'), 'p_value': float('nan'), 'mean_differential': float(d.mean()) if n else float('nan')}
    s2, lag, r = conservative_long_run_variance(d)
    if s2 <= 0:
        dm = math.copysign(math.inf, d.mean()) if d.mean() != 0 else 0.
    else:
        dm = d.mean()/math.sqrt(s2/n)
    hln = dm*math.sqrt((n-1)/n)
    t = stats.t(n-1)
    p = {'two-sided': 2*t.sf(abs(hln)), 'greater': t.sf(hln), 'less': t.cdf(hln)}[alternative]
    return {'n': n, 'newey_west_lag': lag, 'lag1_autocorrelation': r, 'mean_differential': float(d.mean()), 'long_run_variance': s2, 'dm': float(dm),
            'statistic': float(hln), 'p_value': float(p), 'alternative': alternative,
            'reference': 'Diebold-Mariano, max(Newey-West, AR(1)) long-run variance, Harvey-Leybourne-Newbold correction, t(n-1)'}
