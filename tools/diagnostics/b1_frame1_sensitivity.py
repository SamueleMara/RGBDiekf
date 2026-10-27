#!/usr/bin/env python3
"""Frame-1 reproducibility: OFF vs OFF, ON vs ON and OFF vs ON (same saved prior and images); per-iteration objective
trace, position vs velocity discrepancy, and the post-hoc moving-slide difference. Read-only; writes only the JSON."""
import sys, json, itertools
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from tools.audit_phase1_operators import blade_setup, saved_prior, gt_labels
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.image_observation import levels, ObservationPolicy
run, scene, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
B = blade_setup(run, scene, 'cuda', 59); sm, Pm = saved_prior(B, k)
lv = levels([B['data'].observation(k, c) for c in range(len(B['cams']))], B['config'], ObservationPolicy(), 'cuda')
lab = gt_labels(B, scene)[0]; mv = torch.as_tensor(lab == 1, device='cuda'); ax = torch.as_tensor(gt_labels(B, scene)[1], device='cuda')
runs = {}
for name, flag in (('off_a', False), ('off_b', False), ('on_a', True), ('on_b', True)):
    s, P, *_, rep = frame_step(sm, Pm.clone(), lv, B['rgb'], B['rho'], B['policy'], B['G'], IEKFConfig(small_contractions=flag))
    runs[name] = (s, P, [h['objective_trial'] for h in rep['history']])
out = {'frame': k, 'pairs': {}}
for a, b in itertools.combinations(runs, 2):
    (sa, Pa, ta), (sb, Pb, tb) = runs[a], runs[b]
    da, db = sa-sm, sb-sm
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if abs(x-y) > 1e-9*abs(x)), None)
    out['pairs'][f'{a} vs {b}'] = {'bitwise_equal': bool(torch.equal(sa, sb)), 'correction_rel': float((da-db).norm()/db.norm()),
        'position_max_abs_m': float((sa[:, :3]-sb[:, :3]).abs().max()), 'velocity_max_abs_mps': float((sa[:, 3:]-sb[:, 3:]).abs().max()),
        'posthoc_moving_slide_mm': [1e3*float(((sa[mv, :3]-sm[mv, :3])@ax).median()), 1e3*float(((sb[mv, :3]-sm[mv, :3])@ax).median())],
        'first_iteration_trial_objective_differs_1e-9': first, 'trial_objectives': [ta, tb], 'posterior_rel': float((Pa-Pb).norm()/Pb.norm())}
    print(a, b, {k_: v for k_, v in out['pairs'][f'{a} vs {b}'].items() if k_ != 'trial_objectives'}, flush=True)
(REPO/f'reports/diagnostics/b1_frame{k}_sensitivity.json').write_text(json.dumps(out, indent=1))
