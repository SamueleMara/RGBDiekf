#!/usr/bin/env python3
"""50-digit arbiter for saved real-component oracles (oracle_frame_*_component_*.npz); read-only.

The evidence auditor (tools/report_temporal_sparse_gpu.py) compares the production step/covariance with an
independent float64 SciPy QR solve and requires QR condition < 1e6. On ill-conditioned components (e.g. Blade, whose
measured depth noise is tiny) two float64 methods can only agree to ~cond*eps, so a disagreement does not say WHICH
one is wrong. This tool solves the same saved component objective
    min ||A d + r||^2 + d^T H d + 2 g^T d   (H, g: saved information incl. the eliminated Gaussian priors)
in 50-digit arithmetic (mpmath) and reports each float64 result's error against it. It never changes the
auditor's acceptance rule and is never used by inference.

    python -B tools/arbitrate_component_oracles_mp.py RUN/<scene>_full/free_forward/temporal/oracle_frame_*.npz --output arbiter.json
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import scipy.linalg as la
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.temporal_output import atomic_json


def arbitrate(path, digits=50):
    import mpmath as mp
    mp.mp.dps = digits
    d = np.load(path, allow_pickle=False); A, r, H, g = d['A'], d['r'], d['FEJ_information'], d['FEJ_gradient']
    if len(d['C']):raise ValueError(f'{path}: hard rows present; this arbiter covers unconstrained saved components only')
    Am, Hm = mp.matrix(A.tolist()), mp.matrix(H.tolist())
    N = Am.T*Am+Hm; rhs = -(Am.T*mp.matrix(r.tolist())+mp.matrix(g.tolist()))
    step = np.array([float(v) for v in mp.lu_solve(N, rhs)]); P = mp.inverse(N); selected = d['selected']
    covariance = np.array([[float(P[i, j]) for j in selected] for i in selected])
    L = la.cholesky(H, lower=True); B = np.concatenate((A, L.T)); b = np.concatenate((r, la.solve_triangular(L, g, lower=True)))
    Q, R = la.qr(B, mode='economic'); qr_step = la.solve_triangular(R, -Q.T@b)
    rel = lambda a, e: float(np.linalg.norm(a-e)/max(np.linalg.norm(e), np.finfo(float).tiny))
    return {'path': str(path), 'dimension': int(A.shape[1]), 'rows': int(len(A)), 'QR_condition': float(np.linalg.cond(R)), 'digits': digits,
            'production_step_relative_error': rel(d['step'], step), 'production_covariance_relative_error': rel(d['selected_covariance'], covariance),
            'float64_QR_step_relative_error': rel(qr_step, step)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('oracles', nargs='+', type=Path); ap.add_argument('--output', type=Path); ap.add_argument('--digits', type=int, default=50)
    a = ap.parse_args()
    if a.output is not None and a.output.exists():ap.error('Output exists; never overwrite evidence')
    report = {'components': [arbitrate(p, a.digits) for p in a.oracles]}
    if a.output is not None:atomic_json(a.output, report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':raise SystemExit(main())
