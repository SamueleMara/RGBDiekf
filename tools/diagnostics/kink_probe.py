#!/usr/bin/env python3
"""At the stalled iterate of a frame (100 accurate iterations from the saved prior), compare one-sided directional
derivatives of the TRUE objective along the solver's own next step direction with the model slope -b^T d. A kink
(non-differentiable objective) shows as left != right; a smooth point shows left = right = -b^T d."""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from tools.audit_phase1_operators import blade_setup, saved_prior, observation, energy
from estimator.sparse_iekf import IEKFConfig, frame_step, linear_system
from estimator.observation_terms import levels, ObservationPolicy
from rigeuler_ref.trust_region import ScalingMetric, steihaug_cg
run, scene, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
B = blade_setup(run, scene, 'cuda', 59); sm, Pm = saved_prior(B, k); Lam = torch.linalg.inv(Pm); Lam = (Lam+Lam.transpose(1, 2))/2
obs = observation(B, k, 'cuda'); lv = levels([B['data'].observation(k, c) for c in range(len(B['cams']))], B['config'], ObservationPolicy(), 'cuda')
out = {}
for label, cfg in (('production_end', IEKFConfig()), ('stalled_100_iterations', IEKFConfig(max_iters=100, forcing_max=1e-10, forcing_min=1e-12, pcg_max_iters=500))):
    s = frame_step(sm, Pm.clone(), lv, B['rgb'], B['rho'], B['policy'], B['G'], cfg)[0]
    E0, lin, terms = energy(B, obs, s, sm, Lam)
    b, A, M = linear_system(s, sm, Lam, lin, terms, obs, B['G'], 1e-3)
    metric = ScalingMetric.from_physical_scales((B['G'].count, 12), s.device, s.dtype)
    z = steihaug_cg(A, b, 1e-4, metric, M_inv=M, max_iters=30, eta=1e-2).step
    d = B['G'].expand(z); slope = -float((b*z).sum())
    rows = []
    for h in (1., 1e-1, 1e-2, 1e-3):
        Ep = energy(B, obs, s+h*d, sm, Lam)[0]; Em = energy(B, obs, s-h*d, sm, Lam)[0]
        rows.append({'h': h, 'right': (Ep-E0)/h, 'left': (E0-Em)/h, 'max_point_move_m': h*float(d[:, :3].norm(dim=1).max())})
    out[label] = {'model_slope_minus_bTd': slope, 'one_sided': rows}
    print(label, 'model slope %.4g' % slope, ' '.join('h=%g R=%.4g L=%.4g' % (r['h'], r['right'], r['left']) for r in rows), flush=True)
(REPO/f'reports/diagnostics/phase1_kink_probe_frame{k}.json').write_text(json.dumps(out, indent=1))
