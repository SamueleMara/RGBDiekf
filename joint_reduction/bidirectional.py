"""Forward/backward joint comparison in the calibrated world frame: DIAGNOSTICS ONLY (contract section 10).

Each direction has its own canonical endpoint catalogue and gauge (q_f(0) = 0 forward, q_b(K-1) = 0 backward)
and its own joint decision and support; the directions are never averaged, pooled, matched point by point, or
multiplied as posteriors (they reuse the same pixels). Reported for BOTH candidate types, even when unresolved:
axis angle up to sign, revolute axis-line distance (axial gauge free) and the chronological relative displacement
q_f(t) vs s (q_b(t) - q_b(0)), s = sign(e_f . e_b). No compatibility threshold decides anything.
"""
from pathlib import Path
import numpy as np
from utils.io import serial
from utils.io import json_write


def line_distance(a1, e1, a2, e2):
    n = np.cross(e1, e2)
    if np.linalg.norm(n) < 1e-9: return float(np.linalg.norm(np.cross(a2-a1, e1)))
    return float(abs((a2-a1)@n)/np.linalg.norm(n))


def compare(cf, cb, resolution, scale):
    ef, eb = np.asarray(cf['axis']), np.asarray(cb['axis'])
    s = 1. if ef@eb >= 0 else -1.
    qf, qb = np.asarray(cf['q']), np.asarray(cb['q'])
    qb_rel = s*(qb-qb[0])
    out = {'type': cf['type'], 'axis_sign': s, 'axis_angle_rad': float(np.arccos(min(1., abs(ef@eb)))),
           'q_rmse': float(np.sqrt(np.mean((qf-qb_rel)**2))), 'q_max_abs_difference': float(np.abs(qf-qb_rel).max()),
           'travel': float(max(np.abs(qf).max(), np.abs(qb_rel).max())), 'resolution_m': resolution, 'part_scale': scale}
    if cf['type'] == 'REVOLUTE':
        out['axis_line_distance_m'] = line_distance(np.asarray(cf['axis_point']), ef, np.asarray(cb['axis_point']), eb)
    return out


def run(joint_f, joint_b, dataset, root, resolution):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    scale = float(joint_f.get('one_dof', {}).get('part_scale_L', 1.))
    diagnostics = {t: compare(joint_f['candidates'][t], joint_b['candidates'][t], resolution, scale) for t in ('PRISMATIC', 'REVOLUTE')}
    tf, tb = joint_f.get('selected_type'), joint_b.get('selected_type')
    agreement = ('SAME_TYPE' if tf == tb and tf is not None else 'DIFFERENT_TYPES' if tf and tb else 'NOT_BOTH_RESOLVED')
    out = {'status': agreement, 'forward': {'status': joint_f['status'], 'selected_type': tf},
           'backward': {'status': joint_b['status'], 'selected_type': tb}, 'candidate_comparison': diagnostics,
           'world_reference': 'calibrated world frame; chronological frame ids; no point-ID correspondence',
           'semantics': 'diagnostics only: no averaging, pooling or posterior product; each direction keeps its own joint and support'}
    json_write(root/'result.json', serial(out))
    return out
