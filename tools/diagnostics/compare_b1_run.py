#!/usr/bin/env python3
"""Compare a B1 (small_contractions) Phase-1 + coherence run with the saved v2 production run on the same anchor."""
import sys, json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
base, new = Path(sys.argv[1])/'first', Path(sys.argv[2])/'first'
out = {}
sb, sn = np.load(base/'phase1/state.npy', mmap_mode='r'), np.load(new/'phase1/state.npy', mmap_mode='r')
db, dn = sb-sb[0], sn-sn[0]
out['state_shape'] = list(sb.shape)
out['position_max_abs_diff_m_per_frame'] = [float(np.abs(sb[k, :, :3]-sn[k, :, :3]).max()) for k in range(len(sb))]
out['position_median_abs_diff_m_per_frame'] = [float(np.median(np.linalg.norm(sb[k, :, :3]-sn[k, :, :3], axis=1))) for k in range(len(sb))]
out['displacement_rel_diff_final'] = float(np.linalg.norm(db[-1, :, :3]-dn[-1, :, :3])/np.linalg.norm(db[-1, :, :3]))
cb, cn = (json.loads((d/'coherence/result.json').read_text()) for d in (base, new))
rb, rn = (np.load(d/'coherence/representatives.npz') for d in (base, new))
out['coherence'] = {k: [cb.get(k), cn.get(k)] for k in ('status', 'counts', 'core_nodes', 'core_points', 'selection')}
out['coherence']['seed_core_nodes'] = [len(cb['seed_core_nodes']), len(cn['seed_core_nodes'])]
out['coherence']['labels_equal'] = bool(np.array_equal(rb['labels'], rn['labels']))
out['coherence']['membership_equal'] = bool(np.array_equal(rb['membership'], rn['membership']))
out['coherence']['node_trajectory_max_abs_diff_m'] = float(np.abs(rb['trajectory']-rn['trajectory']).max())
out['coherence']['seed_equal'] = sorted(cb['seed_core_nodes']) == sorted(cn['seed_core_nodes'])
for name, d in (('base', base), ('new', new)):
    for f in ('report.json', 'phase1_report.json', 'result.json'):
        p = d/'phase1'/f
        if p.exists(): out.setdefault('phase1_reports', {})[name] = str(p)
print(json.dumps({k: v for k, v in out.items() if 'per_frame' not in k}, indent=1, default=str))
print('max position diff by frame (mm):', [round(1e3*x, 3) for x in out['position_max_abs_diff_m_per_frame']])
(REPO/'reports/diagnostics/b1_full_run_vs_v2.json').write_text(json.dumps(out, indent=1, default=str))
