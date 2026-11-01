#!/usr/bin/env python3
"""Forensic root cause of static-point drift and moving-point loss in the STANDARD directional forward F1 IEKF
(RGBDiekf_Static_Drift_INNER_IEKF_Only_Codex_2026-10-30.md). INVESTIGATION ONLY: nothing here reaches inference; no
RTS / IEKS / temporal solver is run or read. Ground truth only SELECTS and EVALUATES points after estimation.

    select   BASELINE --scene S --out FDIR     standard-F1 output -> population events (every static / moving point),
                                               deterministic audit selection (selected_point_ids.csv), the trace plan and
                                               the exact traced train.py command (FDIR/trace_plan.json)
    analyze  TRACED --baseline BASELINE --forensics FDIR
                                               the same F1 re-executed with --debug-point-ids / --debug-frames from the
                                               plan: tracing invariance, per-point timelines, decisive inner iterations,
                                               raw 2x2 rows, float64 oracle reproduction, channel / row / interpolation
                                               counterfactuals C0-C5, H1-H7 evidence, moving-point survival
    report   FDIR [FDIR ...] --out REPORT.md   combined report of analysed scenes

Reused, not re-implemented: the independent float64 oracle (tools/root_cause_oracle.py), the trace reader
(tools/root_cause_trace.py), the static basis / events / parity of tools/audit_static_equilibrium.py.

Reporting thresholds (CLASSIFY evidence only, none reaches a filter): 0.1 mm / 1 mm / 1 cm / 10 cm transport drift,
1 mm/s / 1 cm/s speed. "Transport drift" is ||x+_k - x+_0|| (canonical deviation); for grounded-static points x_GT,k =
x0 up to the labelled-cloud spacing h (never called exact GT error); moving points use the GT joint transform of x0.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import root_cause_oracle as O                                                         # noqa: E402
from tools.root_cause_trace import Run, jsonable, write_json, git, problem_from_trace, prediction_section, frozen_codes  # noqa: E402
from tools.audit_static_equilibrium import gt_basis, trajectories, parity, contract, snapshot     # noqa: E402
from tools.f1_diagnostic_units import compare_state6     # noqa: E402

DRIFT_LEVELS = (1e-4, 1e-3, 1e-2, 1e-1)
SPEED_LEVELS = (1e-3, 1e-2)
CATEGORY_QUANTILES = {'static_over_10cm': (.05, .2, .35, .5, .65, .8, .9, .95, .99), 'static_1_to_10cm': (.1, .3, .5, .7, .9, .99)}


# ----------------------------------------------------------------------------------------------------------- helpers
def first_k(mask, order):
    """mask [S,n] over processing steps 1..S -> first original frame index, or -1."""
    any_ = mask.any(0)
    return np.where(any_, np.asarray(order)[np.where(any_, mask.argmax(0)+1, 0)], -1)


def gt_moving_positions(g, x0, fids):
    from tools.evaluate_projective_gt import gt_transform
    q = g['q'][fids] if len(g['q']) > max(fids) else g['q']
    T = [gt_transform(g['joint'], g['axis'], g['point'], q[k]-q[0]) for k in range(len(fids))]
    return np.stack([x0@Tk[:3, :3].T+Tk[:3, 3] for Tk in T]), q


def population(run, scene):
    """Events of every grounded static and grounded moving point from the saved standard-F1 arrays."""
    order = run.fman['processing_order']
    x0 = run.can['canonical_xyz']
    g = gt_basis(Path(scene), x0)
    fids = run.man['dataset']['original_frame_ids']
    n = len(x0)
    allidx = np.arange(n)
    T = trajectories(run, allidx)
    X, V, t = T['X'], T['V'], T['t']
    dt = np.diff(t)
    xm = X[:-1]+dt[:, None, None]*V[:-1]                                   # exact mean prediction of each step
    dX = X[1:]-xm; dV = V[1:]-V[:-1]
    drift = np.linalg.norm(X-x0[None], axis=-1)                             # transport drift ||x+_k - x+_0||
    speed = np.linalg.norm(V, axis=-1)
    st, rows = T['st'], T['nrows']
    obs = st[1:] == 1
    ev = {'first_measured': first_k(obs, order),
          **{f'first_accepted_dx_over_{int(1e4*l)}e-4m': first_k(obs & (np.linalg.norm(dX, axis=-1) > l), order) for l in (1e-4, 1e-3)},
          **{f'first_speed_over_{l:g}m_s': first_k(speed[1:] > l, order) for l in SPEED_LEVELS},
          **{f'first_drift_over_{l:g}m': first_k(drift[1:] > l, order) for l in DRIFT_LEVELS}}
    had = np.maximum.accumulate(rows > 0, 0)
    lost = (rows[1:] == 0) & had[:-1]
    ev['first_row_loss'] = first_k(lost, order)
    readm = (rows[1:] > 0) & np.maximum.accumulate(lost, 0)
    ev['first_readmission_after_loss'] = first_k(readm, order)
    big = np.argmax(np.linalg.norm(dV, axis=-1), 0)
    ev['largest_velocity_injection_frame'] = np.asarray(order)[big+1]
    ev['largest_velocity_injection_m_s'] = np.linalg.norm(dV, axis=-1)[big, allidx]
    # drift identity x+_T - x+_0 = sum dt v+_{k-1} + sum (x+_k - x-_k) (signed vectors)
    vel_int = (dt[:, None, None]*V[:-1]).sum(0); corr = dX.sum(0); net = X[-1]-X[0]
    closure = np.abs(net-vel_int-corr).max(-1)
    nn = np.linalg.norm(net, axis=-1); dh = net/np.where(nn > 0, nn, 1)[:, None]
    out = {'x0': x0, 'drift_final': drift[-1], 'drift_max': drift.max(0), 'speed_final': speed[-1], 'rows_frame1': rows[1] if len(rows) > 1 else rows[0],
           'measured_frames': obs.sum(0), 'prediction_only_frames': (st[1:] == 2).sum(0), 'velocity_integration_vector': vel_int, 'corrections_vector': corr,
           'net_vector': net, 'identity_closure_max_abs_m': closure, 'velocity_integration_projected_m': (vel_int*dh).sum(-1),
           'corrections_projected_m': (corr*dh).sum(-1), **ev}
    gtpos, q = gt_moving_positions(g, x0, fids)
    err_mov = np.linalg.norm(X-gtpos, axis=-1)
    out.update(gt_static=g['static'], gt_moving=g['moving_grounded'], d_static=g['d_static'], d_moving=g['d_moving'], gt_error_final_moving=err_mov[-1],
               gt_motion_final=np.linalg.norm(gtpos[-1]-gtpos[0], axis=-1), gt_error_moving_max=err_mov.max(0))
    return out, g, T


def source_features(run, idx):
    """Frame-0 image features at each point's source pixel (source camera): depth, local depth slope (m/px), RGB gradient."""
    src = run.can['original_source_ids'][idx]
    W = [c.width for c in run.data.cameras]
    k0 = run.fman['processing_order'][0]
    feats = np.zeros((len(idx), 5))
    for c in np.unique(src[:, 1]):
        obs = run.data.observation(k0, int(c))
        D = np.asarray(obs.depth, float); I = np.asarray(obs.rgb, float).mean(-1)
        gy, gx = np.gradient(D); iy, ix = np.gradient(I)
        m = src[:, 1] == c; pix = src[m, 2]; u, v = pix % W[c], pix//W[c]
        feats[m] = np.stack([np.full(m.sum(), c), D[v, u], np.hypot(gx[v, u], gy[v, u]), np.hypot(ix[v, u], iy[v, u]), u+v*0.], 1)
    return feats                                                        # camera, depth, depth slope, rgb gradient, u


def pick_quantiles(cand, key, qs, diversity):
    """Deterministic: for each target quantile of `key` among cand, the nearest-rank candidate whose diversity label is
    new (within +-5% rank), else the nearest-rank one not yet taken."""
    o = cand[np.argsort(key[cand], kind='stable')]
    taken, labels = [], set()
    for qv in qs:
        r = int(round(qv*(len(o)-1))); win = max(1, int(.05*len(o)))
        best = None
        for d in range(0, win+1):
            for rr in (r-d, r+d):
                if 0 <= rr < len(o) and o[rr] not in taken and diversity(o[rr]) not in labels: best = o[rr]; break
            if best is not None: break
        if best is None:
            rest = [p for p in o if p not in taken]
            if not rest: break
            best = min(rest, key=lambda p: abs(int(np.flatnonzero(o == p)[0])-r))
        taken.append(int(best)); labels.add(diversity(best))
    return taken


def cmd_select(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise SystemExit(f'{out} exists and is not empty: never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    run = Run(a.baseline, 'forward')
    if run.man['arguments'].get('free_solver', 'directional') != 'directional': raise SystemExit('baseline is not the standard directional F1')
    t0 = time.perf_counter()
    pop, g, T = population(run, a.scene)
    order = run.fman['processing_order']; K = len(order)
    st_, mv = pop['gt_static'], pop['gt_moving']
    df, dmax = pop['drift_final'], pop['drift_max']
    onset = pop['first_drift_over_0.001m']
    src = run.can['original_source_ids']
    diversity = lambda p: (int(src[p, 1]), int(min(2, max(0, (onset[p] if onset[p] > 0 else K)*3//K))))
    sel = {}
    severe = np.flatnonzero(st_ & (df > .10)); medium = np.flatnonzero(st_ & (df > .01) & (df <= .10))
    sel['static_over_10cm'] = pick_quantiles(severe, df, CATEGORY_QUANTILES['static_over_10cm'], diversity) if len(severe) else []
    sel['static_1_to_10cm'] = pick_quantiles(medium, df, CATEGORY_QUANTILES['static_1_to_10cm'], diversity) if len(medium) else []
    # stable controls (< 1 mm over the whole sequence, measured at frame 1) matched to the first severe cases
    stable = np.flatnonzero(st_ & (dmax < 1e-3) & (pop['rows_frame1'] > 0))
    targets = sel['static_over_10cm'][:6]
    feats_t = source_features(run, np.asarray(targets, int)) if targets else np.zeros((0, 5))
    feats_s = source_features(run, stable) if len(stable) else np.zeros((0, 5))
    rows1 = pop['rows_frame1']
    ctrl, match = [], {}
    for p, ft in zip(targets, feats_t):
        same = np.flatnonzero((feats_s[:, 0] == ft[0]) & ~np.isin(stable, ctrl))
        if not len(same): same = np.flatnonzero(~np.isin(stable, ctrl))
        if not len(same): break
        f = feats_s[same]; sc = np.array([max(abs(ft[1]), 1e-6), max(ft[2], 1e-6), max(ft[3], 1e-6)])
        d = np.abs(np.log(np.maximum(f[:, 1:4], 1e-9)/np.maximum(ft[1:4], 1e-9))).sum(1)+np.abs(rows1[stable[same]]-rows1[p])
        c = int(stable[same[np.argmin(d)]]); ctrl.append(c); match[c] = p
    sel['static_stable_control'] = ctrl
    # moving: tracked (final GT error < 1 cm) and lost (> 10 cm), deterministic by error rank
    err = pop['gt_error_final_moving']
    ok = np.flatnonzero(mv & (err < .01)); bad = np.flatnonzero(mv & (err > .10))
    sel['moving_tracked'] = [int(ok[np.argsort(-pop['gt_motion_final'][ok], kind='stable')][i]) for i in range(min(3, len(ok)))]
    if len(bad):
        bo = bad[np.argsort(err[bad], kind='stable')]
        sel['moving_lost'] = sorted({int(bo[int(round(qv*(len(bo)-1)))]) for qv in (.25, .5, .99)})
        po = bad[np.argsort(-pop['prediction_only_frames'][bad], kind='stable')]
        sel['moving_lost'] = sorted(set(sel['moving_lost']) | {int(po[0])})
    else: sel['moving_lost'] = []
    neg = np.flatnonzero(st_ & (pop['measured_frames'] == 0))
    sel['negative_control_never_measured'] = [int(p) for p in neg[:2]]
    ids = [p for v in sel.values() for p in v]
    if len(ids) != len(set(ids)): raise AssertionError('a point id was selected twice')
    # selected_point_ids.csv
    W = [c.width for c in run.data.cameras]
    full = run.can['subsampled_to_full_index']
    fields = ['point_id', 'category', 'matched_to', 'full_endpoint_index', 'source_frame', 'source_camera', 'source_u', 'source_v', 'x0_x', 'x0_y', 'x0_z',
              'gt_basis', 'd_nearest_static_label_m', 'd_nearest_moving_label_m', 'transport_drift_final_m', 'transport_drift_max_m', 'gt_error_final_m',
              'gt_motion_final_m', 'rows_frame1', 'measured_frames', 'prediction_only_frames', 'first_measured', 'first_accepted_dx_over_1e-4m',
              'first_speed_over_0.001m_s', 'first_drift_over_0.001m', 'first_drift_over_0.01m', 'first_drift_over_0.1m', 'first_row_loss',
              'first_readmission_after_loss', 'largest_velocity_injection_frame', 'largest_velocity_injection_m_s', 'velocity_integration_projected_m',
              'corrections_projected_m', 'caveat']
    with open(out/'selected_point_ids.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for cat, v in sel.items():
            for p in v:
                fr, c, pix = (int(x) for x in src[p])
                w.writerow({'point_id': p, 'category': cat, 'matched_to': match.get(p, ''), 'full_endpoint_index': int(full[p]), 'source_frame': fr,
                            'source_camera': c, 'source_u': pix % W[c], 'source_v': pix//W[c], 'x0_x': pop['x0'][p, 0], 'x0_y': pop['x0'][p, 1], 'x0_z': pop['x0'][p, 2],
                            'gt_basis': 'GROUNDED_STATIC' if st_[p] else ('GROUNDED_MOVING' if mv[p] else 'UNGROUNDED'),
                            'd_nearest_static_label_m': pop['d_static'][p], 'd_nearest_moving_label_m': pop['d_moving'][p],
                            'transport_drift_final_m': df[p], 'transport_drift_max_m': dmax[p], 'gt_error_final_m': err[p] if mv[p] else df[p],
                            'gt_motion_final_m': pop['gt_motion_final'][p] if mv[p] else 0., 'rows_frame1': int(rows1[p]),
                            'measured_frames': int(pop['measured_frames'][p]), 'prediction_only_frames': int(pop['prediction_only_frames'][p]),
                            **{k: int(pop[k][p]) for k in ('first_measured', 'first_accepted_dx_over_1e-4m', 'first_speed_over_0.001m_s', 'first_drift_over_0.001m',
                                                          'first_drift_over_0.01m', 'first_drift_over_0.1m', 'first_row_loss', 'first_readmission_after_loss',
                                                          'largest_velocity_injection_frame')},
                            'largest_velocity_injection_m_s': pop['largest_velocity_injection_m_s'][p], 'velocity_integration_projected_m': pop['velocity_integration_projected_m'][p],
                            'corrections_projected_m': pop['corrections_projected_m'][p],
                            'caveat': ('static part membership from the nearest labelled GT point (spacing h); transport drift is canonical deviation, '
                                       'not an exactly matched GT material-point error') if st_[p] else
                                      ('moving: GT position = GT joint transform of x0 (nearest-label membership)' if mv[p] else 'ungrounded')})
    # population summary (all grounded points; no per-point dumps beyond one compressed table)
    def qs(x): x = np.asarray(x, float); return {f'p{int(100*q)}': float(np.quantile(x, q)) for q in (.5, .9, .99)} if x.size else None
    def ev_hist(key, m):
        v = pop[key][m]; v = v[v > 0]
        return {'points_with_event': int(len(v)), 'frame_quantiles': qs(v)}
    summ = {'scene': str(a.scene), 'baseline': str(a.baseline), 'points': int(len(df)), 'frames': K, 'gt_basis_rule': g['meta']['rule'], 'gt_meta': g['meta'],
            'static': {'points': int(st_.sum()), 'drift_final_m': qs(df[st_]), 'over_10cm_fraction': float((df[st_] > .1).mean()),
                       'over_1cm_fraction': float((df[st_] > .01).mean()), 'stable_under_1mm_fraction': float((dmax[st_] < 1e-3).mean()),
                       'identity_closure_max_abs_m': float(pop['identity_closure_max_abs_m'][st_].max()),
                       'velocity_integration_share_of_departures_over_10cm': qs((pop['velocity_integration_projected_m']/np.maximum(np.linalg.norm(pop['net_vector'], axis=-1), 1e-300))[st_ & (df > .1)]),
                       'events': {k: ev_hist(k, st_) for k in pop if k.startswith('first_')}},
            'moving': {'points': int(mv.sum()), 'gt_motion_final_m': qs(pop['gt_motion_final'][mv]), 'gt_error_final_m': qs(err[mv]),
                       'within_1cm': float((err[mv] < .01).mean()), 'within_5cm': float((err[mv] < .05).mean()), 'over_10cm': float((err[mv] > .1).mean()),
                       'events': {k: ev_hist(k, mv) for k in pop if k.startswith('first_')}},
            'selection': sel, 'selection_rules': {'quantiles_of_final_transport_drift': CATEGORY_QUANTILES, 'diversity': '(source camera, onset-frame tercile)',
                                                  'stable_control': 'max transport drift < 1 mm, rows at frame 1, same source camera, nearest in log depth / depth slope / '
                                                                    'RGB gradient + |rows at frame 1 difference| to its matched severe case',
                                                  'moving_tracked': 'final GT error < 1 cm, largest GT motion first', 'moving_lost': 'final GT error > 10 cm at rank quantiles '
                                                  '0.25/0.5/0.99 + the one with most prediction-only frames', 'negative_control': 'grounded static, never measured after frame 0'},
            'seconds': time.perf_counter()-t0}
    np.savez_compressed(out/'population_events.npz', **{k: v for k, v in pop.items() if isinstance(v, np.ndarray)})
    write_json(out/'population_summary.json', summ)
    # trace plan: the baseline command with only the output and the trace switches added
    base = run.man['arguments']
    plan = {'debug_point_ids': ids, 'debug_frames': list(order[1:]), 'baseline_arguments': base,
            'trace_note': 'every processed frame of the selected ids (full prediction / iteration / acceptance trace); tracing is opt-in and verified bitwise '
                          'invariant against the baseline arrays (parity) before any analysis'}
    write_json(out/'trace_plan.json', plan)
    print(json.dumps({'selected': {k: len(v) for k, v in sel.items()}, 'static': summ['static']['drift_final_m'], 'moving_error': summ['moving']['gt_error_final_m']}))


def traced_command(plan, scene, output):
    """train.py argv reproducing the baseline's resolved arguments + the trace switches (checked in analyze)."""
    b = plan['baseline_arguments']
    argv = ['--scene', str(scene), '--output', str(output), '--device', b['device'], '--sensor-noise', b['sensor_noise'], '--direction', b['direction'],
            '--stop-after', 'free', '--debug', b['debug'], '--no-evaluate', '--no-qa', '--pcd-subsample-factor', str(b['pcd_subsample_factor']),
            '--save-state-interval', str(b['save_state_interval']), '--debug-point-ids', ','.join(map(str, plan['debug_point_ids'])),
            '--debug-frames', ','.join(map(str, plan['debug_frames']))]
    return argv


def cmd_command(a):
    plan = json.loads((Path(a.forensics)/'trace_plan.json').read_text())
    print(' '.join(f'"{x}"' if (' ' in x or x == '') else x for x in traced_command(plan, a.scene, a.output)))


def cmd_trace_run(a):
    """The one instrumented standard-F1 run of the plan (train.py in-process, exact argv printed first)."""
    plan = json.loads((Path(a.forensics)/'trace_plan.json').read_text())
    argv = traced_command(plan, a.scene, a.output)
    print('train.py ' + ' '.join(f'"{x}"' if (' ' in x or x == '') else x for x in argv), flush=True)
    import train
    return train.main(argv)


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('select'); p.add_argument('baseline', type=Path); p.add_argument('--scene', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sp.add_parser('command'); p.add_argument('forensics', type=Path); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sp.add_parser('trace-run'); p.add_argument('forensics', type=Path); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sp.add_parser('analyze'); p.add_argument('traced', type=Path); p.add_argument('--baseline', type=Path, required=True); p.add_argument('--forensics', type=Path, required=True)
    p.add_argument('--points', type=int, nargs='*', help='analyse only these ids (default: the whole selection)')
    p.add_argument('--max-frame', type=int, help='SMOKE TEST ONLY: stop at this frame (partial traces)')
    p = sp.add_parser('report'); p.add_argument('forensics', type=Path, nargs='+'); p.add_argument('--out', type=Path, required=True)
    p = sp.add_parser('where'); p.add_argument('baseline', type=Path); p.add_argument('--forensics', type=Path, required=True)
    a = ap.parse_args(argv)
    return {'select': cmd_select, 'command': cmd_command, 'trace-run': cmd_trace_run, 'analyze': cmd_analyze, 'report': cmd_report, 'where': cmd_where}[a.cmd](a)


# ------------------------------------------------------------------------------------------------------------ analyze
ACC_NAMES = {0: 'ANCHOR_OR_NOT_PROCESSED', 1: 'ACCEPTED_FINAL_ITERATE', 2: 'ACCEPTED_EARLIER_CONSISTENT_ITERATE', 3: 'REJECTED_INCONSISTENT', 4: 'NO_VALID_ROWS'}
DX_EVENT, DV_EVENT = 1e-4, 1e-3        # reporting only: an accepted correction (m) / velocity change (m/s) worth a forensic look
GROUP = {'R': (0,), 'G': (1,), 'B': (2,), 'RGB': (0, 1, 2), 'D': (3,)}


def load_selection(fdir):
    with open(Path(fdir)/'selected_point_ids.csv') as f: return list(csv.DictReader(f))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a-b).max()/max(np.abs(b).max(), 1e-300))


def absd(a, b):
    return float(np.abs(np.asarray(a, float)-np.asarray(b, float)).max())


class GroundTruth:
    """OFFLINE ONLY: the labelled GT part cloud of the canonical endpoint moved by the GT joint to every frame; used to
    identify which part a sampled depth pixel images and where a point physically is. Never reaches the filter."""
    def __init__(self, run, scene):
        from scipy.spatial import cKDTree
        from tools.evaluate_projective_gt import gt_joint, labelled_endpoints, gt_transform
        scene = Path(scene)
        kind, axis, point, _ = gt_joint(scene)
        vals = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True); q = np.asarray(vals[vals.dtype.names[1]], float)
        (xg, lg), _, moving, _ = labelled_endpoints(scene, kind, axis, point, q[-1]-q[0])
        fids = run.man['dataset']['original_frame_ids']
        self.q = q[fids] if len(q) > max(fids) else q
        self.T = [gt_transform(kind, axis, point, self.q[k]-self.q[0]) for k in range(len(fids))]
        self.xg, self.mov = np.asarray(xg, float), np.asarray(lg) == moving
        self.KD = cKDTree
        self.h = float(np.median(cKDTree(self.xg).query(self.xg, k=2)[0][:, 1]))
        self._tree = {}

    def moved(self, x, k):
        T = self.T[k]; return np.asarray(x, float)@T[:3, :3].T+T[:3, 3]

    def position(self, x0, moving, k):
        return self.moved(x0, k) if moving else np.asarray(x0, float)

    def identify(self, k, pts):
        if k not in self._tree:
            x = self.xg.copy(); x[self.mov] = self.moved(self.xg[self.mov], k); self._tree = {k: self.KD(x)}
        d, i = self._tree[k].query(np.atleast_2d(np.asarray(pts, float)))
        return np.where(d > 2*self.h, 'UNLABELLED', np.where(self.mov[i], 'MOVING_PART', 'STATIC_PART')), d


def make_problem(run, k, pid, tr, imgs, cams, policy=None, **kw):
    """problem_from_trace with an optional policy (counterfactual interpolation); fixed prior, frozen assignment and
    recheck flags exactly as the production frame recorded them."""
    sp = str(pid)
    pre, it0, fin = tr['prediction']['points'][sp], tr['iterations'][0]['points'][sp], tr['final'][sp]
    recheck = None
    if run.policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion':
        recheck = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']]
    return O.Problem(cams, imgs, run.can['canonical_rgb'][pid], np.array(pre['s_minus']), np.array(pre['P_minus']), frozen_codes(it0['cameras']),
                     run.noise, policy or run.policy, run.fman['max_iekf_iters'], run.fman['step_tolerance'], recheck, **kw)


def row_record(o, cam, c, gt, k, x_gt, noise, n_sigma):
    """One camera row (oracle) with raw 2x2 samples, variance components, gate numbers and OFFLINE GT surface identity."""
    s = o.get('sample', {})
    rec = {'camera': c, 'reason': O.NAMES[o['reason']], 'u': o.get('u'), 'v': o.get('v'), 'Z': o.get('Z')}
    if 'neighbour_depth' in s:
        u0, v0 = s['pixel00']
        D = np.asarray(s['neighbour_depth'], float)
        pix = [(u0, v0), (u0+1, v0), (u0, v0+1), (u0+1, v0+1)]
        P3 = np.array([cam.centre()+D[i]*cam.ray_world(pu, pv) for i, (pu, pv) in enumerate(pix)])
        fin_ = np.isfinite(P3).all(1)
        lab = np.full(4, 'INVALID_DEPTH_PIXEL', dtype=object); dist = np.full(4, np.nan)
        if fin_.any(): l_, d_ = gt.identify(k, P3[fin_]); lab[fin_] = l_; dist[fin_] = d_
        rec.update(pixel00_u_v=[int(u0), int(v0)], du_dv=[s['du'], s['dv']], neighbour_depth_m=D.tolist(), neighbour_rgb=np.asarray(s['neighbour_rgb']).tolist(),
                   neighbour_mask=np.asarray(s['neighbour_mask']).tolist(), neighbour_raw_valid=np.asarray(s['neighbour_raw_valid']).tolist(),
                   neighbour_gt_part=lab.tolist(), neighbour_gt_label_distance_m=dist.tolist(), local_depth_min_max_m=[float(D.min()), float(D.max())])
    if 'depth' in s:
        w = s['w']; D = np.asarray(s['neighbour_depth'], float)
        surf = cam.centre()+s['depth_metric']*cam.ray_world(o['u'], o['v'])
        lab, dist = gt.identify(k, surf[None])
        rec.update(same_surface_spread_m=s['same_surface_spread_m'], same_surface_allowance_m=s['same_surface_allowance_m'],
                   depth_metric_m=s['depth_metric'], depth_inverse_m=s['depth_inverse'], depth_used_m=s['depth'],
                   grad_depth_uv=np.asarray(s['grad_depth']).tolist(), grad_rgb_uv=np.asarray(s['grad_rgb']).tolist(),
                   var_depth_sensor_interp_m2=float((w*w*noise.sigma2(D)).sum()), var_depth_model_m2=noise.depth_model,
                   var_rgb_sensor_interp=float((w*w).sum())*noise.rgb_var.tolist()[0], var_rgb_model=noise.rgb_model,
                   sampled_surface_gt_part=str(lab[0]), sampled_surface_gt_label_distance_m=float(dist[0]),
                   sampled_surface_to_point_gt_m=float(np.linalg.norm(surf-x_gt)))
    if 'H' in o:
        S = o['S']
        rec.update(J_u=np.asarray(o['J_u']).tolist(), J_Z=np.asarray(o['J_Z']).tolist(), H_x=o['H'].tolist(), r=o['r'].tolist(), R_diag=o['var'].tolist(),
                   r_D_m=float(o['r'][3]), r_D_inverse_m=float(o['Z']-s['depth_inverse']), r_D_over_sqrt_R=float(o['r'][3]/math.sqrt(o['var'][3])),
                   sqrt_R_D_m=float(math.sqrt(o['var'][3])), sqrt_S_DD_m=float(math.sqrt(max(S[3, 3], 0))), S=S.tolist(), nu=o['nu'].tolist(), nis=o['nis'],
                   nis_threshold_chi2_4=O.row_gate_threshold(n_sigma))
    return rec


def channel_split(prob, s_lin, rows):
    """Exact additive split of the accepted correction e = s_lin - s- at the linearization s_lin (same A):
    e = sum_{c,ch} A^-1 [H_c,ch^T (r_c,ch + H_c,ch e_x) / R_c,ch ; 0] - A^-1 b(s_lin)   (b = -Lam e + H^T W r).
    At s_lin = s- (iteration 0) the terms are the first Gauss-Newton push split by camera and channel."""
    used = prob.used(rows)
    Ixx, gx, q = prob.info(rows)
    A = prob.Lam.copy(); A[:3, :3] += Ixx; Ainv = np.linalg.inv(A)
    e = np.asarray(s_lin, float)-prob.s_minus
    b = -prob.Lam@e; b[:3] += gx
    resid = Ainv@b
    terms = {}
    for o in used:
        for ch in range(4):
            if not prob.ch[ch]: continue
            g = o['H'][ch]*(o['r'][ch]+o['H'][ch]@e[:3])/o['var'][ch]
            terms[(o['camera'], ch)] = Ainv[:, :3]@g
    tot = sum(terms.values()) if terms else np.zeros(6)
    closure = float(np.abs(tot-resid-e).max())
    groups = {gname: sum((v for (c, ch), v in terms.items() if ch in chs), np.zeros(6)) for gname, chs in GROUP.items()}
    cams = {}
    for (c, ch), v in terms.items(): cams[c] = cams.get(c, np.zeros(6))+v
    return {'e': e, 'groups': groups, 'cameras': cams, 'iterate_residual': resid, 'closure_max_abs': closure, 'A': A, 'b': b}


def counterfactual(run, k, pid, tr, imgs, cams, name, **kw):
    pol = kw.pop('policy', None)
    prob = make_problem(run, k, pid, tr, imgs, cams, policy=pol, **kw)
    res = prob.run(run.policy.get('update_acceptance', 'final_iterate'))
    d = res['s_plus']-prob.s_minus
    return {'variant': name, 'observed': res['observed'], 'acceptance': ACC_NAMES[res['acceptance_code']], 'dx': d[:3], 'dv': d[3:],
            'rows': [o['camera'] for o in prob.used(res['rows_accepted'])] if res['observed'] else []}


def timeline_pass(run, sel, gt, x0, order):
    """Pass 1 (trace files only): every frame of every selected point."""
    t = np.asarray(run.fman['timestamps_seconds'], float)
    pids = [int(r['point_id']) for r in sel]; cat = {int(r['point_id']): r['category'] for r in sel}
    moving = {p: cat[p].startswith('moving') for p in pids}
    K = len(order); n = len(pids)
    Pm = np.full((n, K, 6, 6), np.nan); Pp = np.full((n, K, 6, 6), np.nan); Sm = np.full((n, K, 6), np.nan); Sp = np.full((n, K, 6), np.nan)
    rows_tl, rows_adm = [], []
    pred_err = np.zeros(n); P0 = None
    for i, p in enumerate(pids):
        Sp[i, 0] = np.r_[x0[p], 0, 0, 0]; Sm[i, 0] = Sp[i, 0]
    used_sets = {p: [] for p in pids}
    for kk, k in enumerate(order[1:], 1):
        tr = run.trace(k)
        if tr is None: raise SystemExit(f'no trace for frame {k}: the traced run must cover every frame of the plan')
        pre = tr['prediction']
        for i, p in enumerate(pids):
            sp = str(p)
            pp, fin = pre['points'][sp], tr['final'][sp]
            if kk == 1: Pp[i, 0] = np.array(pp['P_previous'])
            sm_, Pm_ = np.array(pp['s_minus']), np.array(pp['P_minus'])
            so, Po, _, _ = O.predict(np.array(pp['s_previous']), np.array(pp['P_previous']), pre['dt_s'], np.array(pre['q_c']))
            pred_err[i] = max(pred_err[i], absd(so, sm_), rel(Po, Pm_))
            spl, Ppl = np.array(fin['s_plus']), np.array(fin['P_plus'])
            Sm[i, kk], Pm[i, kk], Sp[i, kk], Pp[i, kk] = sm_, Pm_, spl, Ppl
            act = [j for j, it in enumerate(tr['iterations']) if it['points'][sp]['active']]
            cams_f = fin.get('cameras_fallback_state') or fin['cameras_final_iterate']
            used = sorted({r['camera'] for r in fin.get('rows', [])}) if fin['acceptance'].startswith('ACCEPTED') else []
            used_sets[p].append(used)
            xgt = gt.position(x0[p], moving[p], k)
            dx, dv = spl[:3]-sm_[:3], spl[3:]-sm_[3:]
            rows_tl.append({'point_id': p, 'category': cat[p], 'frame': k, 't_s': t[k], 'dt_s': pre['dt_s'],
                            **{f'{nm}_{a}': v for nm, vec in (('x_minus', sm_[:3]), ('v_minus', sm_[3:]), ('x_plus', spl[:3]), ('v_plus', spl[3:]), ('dx', dx), ('dv', dv))
                               for a, v in zip('xyz', vec)},
                            'abs_dx_m': float(np.linalg.norm(dx)), 'abs_dv_m_s': float(np.linalg.norm(dv)), 'speed_plus_m_s': float(np.linalg.norm(spl[3:])),
                            'transport_drift_m': float(np.linalg.norm(spl[:3]-x0[p])), 'gt_error_m': float(np.linalg.norm(spl[:3]-xgt)),
                            **{f'sqrt_Pminus_{nm}_{a}': float(math.sqrt(max(Pm_[i0+j, i0+j], 0))) for nm, i0 in (('xx', 0), ('vv', 3)) for j, a in enumerate('xyz')},
                            **{f'Pminus_xv_{a}{a}': float(Pm_[j, 3+j]) for j, a in enumerate('xyz')},
                            **{f'sqrt_Pplus_{nm}_{a}': float(math.sqrt(max(Ppl[i0+j, i0+j], 0))) for nm, i0 in (('xx', 0), ('vv', 3)) for j, a in enumerate('xyz')},
                            'acceptance': fin['acceptance'], 'valid_rows': fin['valid_rows'], 'cameras_used': ';'.join(map(str, used)),
                            'camera_reasons': '|'.join(f"c{c['camera']}:{c['reason_used']}" for c in cams_f),
                            'posterior_cost': fin['posterior_cost'], 'cost_threshold': fin['cost_threshold'], 'converged': fin['converged'],
                            'inner_iterations': len(act), 'last_consistent_iteration': fin['last_consistent_iteration'],
                            'fallback_state_used': not fin['final_iterate_state_sampled']})
            it0 = tr['iterations'][0]['points'][sp]['cameras']
            for c, (c0, cf) in enumerate(zip(it0, cams_f)):
                sampled = cf['reason_if_sampled'] in ('VALID', 'MEASURED_SURFACE_IN_FRONT', 'INNOVATION_INCONSISTENT')   # else r / R are placeholders
                rows_adm.append({'point_id': p, 'category': cat[p], 'frame': k, 'camera': c, 'visibility_fresh_prior': c0['visibility_fresh'],
                                 'visibility_frozen': c0['visibility_frozen'], 'reason_at_prior': c0['reason_used'], 'reason_at_accepted_state': cf['reason_used'],
                                 'reason_if_sampled_at_accepted': cf['reason_if_sampled'], 'admitted': c in used,
                                 'pixel00_prior': '%d,%d' % tuple(c0['pixel00_u_v']), 'pixel00_accepted': '%d,%d' % tuple(cf['pixel00_u_v']),
                                 'pixel_support_changed': c0['pixel00_u_v'] != cf['pixel00_u_v'],
                                 'r_D_m_accepted': cf['r'][3] if sampled else None, 'sqrt_R_D_m': math.sqrt(cf['R_diag'][3]) if sampled else None,
                                 'sqrt_S_DD_m': math.sqrt(max(cf['S_zz'], 0)) if sampled and cf['S_zz'] is not None and math.isfinite(cf['S_zz']) else None,
                                 'row_nis_prior_linearized': cf['row_nis_prior_linearized'], 'pixel_occupants': cf['pixel_occupants_total'],
                                 'front_owner': cf['pixel_occupants_front_to_back'][0]['point'] if cf['pixel_occupants_front_to_back'] else None})
    return pids, cat, moving, rows_tl, rows_adm, (Sm, Pm, Sp, Pp), pred_err, used_sets


def point_events(rows):
    """Events of one point from its timeline rows (original frame indices; -1 = never)."""
    f = lambda cond: next((r['frame'] for r in rows if cond(r)), -1)
    obs = lambda r: r['acceptance'].startswith('ACCEPTED')
    ev = {'first_measured': f(obs), 'first_accepted_dx_nonzero': f(lambda r: obs(r) and r['abs_dx_m'] > 0),
          'first_accepted_dx_over_1e-4m': f(lambda r: obs(r) and r['abs_dx_m'] > DX_EVENT), 'first_dv_over_1e-3m_s': f(lambda r: r['abs_dv_m_s'] > DV_EVENT),
          'first_speed_over_1e-3m_s': f(lambda r: r['speed_plus_m_s'] > 1e-3), 'first_speed_over_1e-2m_s': f(lambda r: r['speed_plus_m_s'] > 1e-2),
          **{f'first_drift_over_{l:g}m': f(lambda r, l=l: r['transport_drift_m'] > l) for l in DRIFT_LEVELS},
          'first_prediction_only': f(lambda r: not obs(r))}
    had, lost, readm, prev, cam_lost, exit_ = False, -1, -1, None, -1, -1
    for r in rows:
        u = set(r['cameras_used'].split(';')) - {''} if obs(r) else set()
        if had and not u and lost < 0: lost = r['frame']
        if lost >= 0 and u and readm < 0 and r['frame'] > lost: readm = r['frame']
        if prev is not None and obs(r) and prev - u and cam_lost < 0: cam_lost = r['frame']
        if exit_ < 0 and any(x.split(':')[1] in ('OUTSIDE_IMAGE_FOOTPRINT', 'BEHIND_CAMERA') for x in r['camera_reasons'].split('|')) and had: exit_ = r['frame']
        had |= bool(u)
        if obs(r): prev = u
    ev.update(first_row_loss=lost, first_readmission_after_loss=readm, first_camera_lost=cam_lost, first_camera_exit_after_measured=exit_)
    dvs = [r['abs_dv_m_s'] for r in rows]
    ev['largest_dv_frame'] = rows[int(np.argmax(dvs))]['frame'] if rows else -1
    ev['largest_dv_m_s'] = float(max(dvs)) if rows else 0.
    grow = [r['frame'] for i, r in enumerate(rows) if not obs(r) and i > 0 and r['transport_drift_m'] > rows[i-1]['transport_drift_m']+1e-12]
    ev['first_drift_growth_without_accepted_row'] = grow[0] if grow else -1
    return ev


def decisive_frames(ev, K):
    k1 = ev['first_accepted_dx_over_1e-4m'] if ev['first_accepted_dx_over_1e-4m'] > 0 else ev['first_dv_over_1e-3m_s']
    ks = {k1, k1-1, k1+1, ev['first_drift_over_0.001m'], ev['largest_dv_frame'], ev['first_row_loss'], ev['first_row_loss']-1, ev['first_readmission_after_loss'],
          ev['first_camera_lost'], ev['first_measured']}
    if k1 <= 0: ks |= {1, 2}
    return sorted(k for k in ks if 1 <= k < K)


def cmd_analyze(a):
    t_start = time.perf_counter()
    fdir = Path(a.forensics)
    run = Run(a.traced, 'forward'); base = Run(a.baseline, 'forward')
    sel = load_selection(fdir)
    if a.points: sel = [r for r in sel if int(r['point_id']) in set(a.points)]
    plan = json.loads((fdir/'trace_plan.json').read_text())
    scene = Path(run.man['arguments']['scene'])
    # ---- provenance, contract equality and tracing invariance
    ta, ba = dict(run.man['arguments']), dict(base.man['arguments'])
    diff = {k: (ba.get(k), ta.get(k)) for k in set(ta) | set(ba) if ta.get(k) != ba.get(k) and k not in ('output', 'debug_point_ids', 'debug_frames')}
    # CLI keys added after the baseline commit, at their defaults (standard solver): not a configuration difference
    later = {k: diff.pop(k) for k in ('free_solver', 'temporal_config') if k in diff and k not in ba and ta.get('free_solver', 'directional') == 'directional'}
    par = parity(a.traced, a.baseline)
    prov = {'snapshot': snapshot([a.baseline, a.traced]), 'contract': contract(base), 'argument_differences_other_than_output_and_trace': diff,
            'cli_keys_added_after_baseline_commit_at_default': later,
            'tracing_invariance': par, 'tracing_invariant': all(v == 'IDENTICAL' for v in par.values()), 'trace_plan': plan,
            'traced_manifest_solver': run.fman.get('pass'), 'free_solver': ta.get('free_solver', 'directional (default: argument absent at this commit)')}
    write_json(fdir/'provenance.json', prov)
    if diff or not prov['tracing_invariant']:
        print(json.dumps({'STOP': 'traced run differs from the baseline', 'arguments': diff, 'parity': par}, default=str)); return 2
    gt = GroundTruth(run, scene)
    x0 = run.can['canonical_xyz']; order = run.fman['processing_order']
    if a.max_frame is not None: order = [k for k in order if k <= a.max_frame]     # smoke tests on a partial trace only
    K = len(order)
    t = np.asarray(run.fman['timestamps_seconds'], float)[:K]
    pids, cat, moving, rows_tl, rows_adm, (Sm, Pm, Sp, Pp), pred_err, used_sets = timeline_pass(run, sel, gt, x0, order)
    by = {p: [r for r in rows_tl if r['point_id'] == p] for p in pids}
    events = {p: point_events(by[p]) for p in pids}
    dec = {p: decisive_frames(events[p], K) for p in pids}
    # drift identity per point: x+_T - x+_0 = sum dt v+_{k-1} + sum (x+_k - x-_k)
    ident = {}
    for i, p in enumerate(pids):
        dtv = np.diff(t)[:, None]*Sp[i, :-1, 3:]; corr = Sp[i, 1:, :3]-Sm[i, 1:, :3]; net = Sp[i, -1, :3]-Sp[i, 0, :3]
        nn = np.linalg.norm(net); dh = net/nn if nn > 0 else np.zeros(3)
        ident[p] = {'net_vector_m': net, 'velocity_integration_vector_m': dtv.sum(0), 'corrections_vector_m': corr.sum(0),
                    'closure_max_abs_m': float(np.abs(net-dtv.sum(0)-corr.sum(0)).max()), 'velocity_integration_projected_m': float(dtv.sum(0)@dh),
                    'corrections_projected_m': float(corr.sum(0)@dh), 'net_m': float(nn)}
    # ---- pass 2: images of every frame (H1 at the stationary reference) and the decisive frames' full forensics
    cams = run.cams(); n_sigma = run.policy['visibility_n_sigma']
    inv_policy = {**run.policy, 'depth_interpolation': 'inverse'}
    depth_samples, inner, oracle_rows, cf_rows, h1_rows = [], [], [], [], []
    hyp_ev = {h: [] for h in ('H1', 'H2', 'H3', 'H4', 'H5', 'H7')}
    for kk, k in enumerate(order[1:], 1):
        tr = run.trace(k)
        need = [p for p in pids if k in dec[p]]
        imgs = run.images(k)
        # H1: every frame, every camera, at the physically correct position (static: x0; moving: GT transform of x0)
        for p in pids:
            xg = gt.position(x0[p], moving[p], k)
            fz = frozen_codes(tr['iterations'][0]['points'][str(p)]['cameras'])
            vis = O.candidacy(cams, xg, fz)
            for c, cam in enumerate(cams):
                if vis[c] != O.VALID: continue
                om = O.row(imgs[c], cam, xg, run.can['canonical_rgb'][p], run.noise, run.policy, np.array(tr['prediction']['points'][str(p)]['P_minus'])[:3, :3], np.zeros(3))
                if 'H' not in om: continue
                oi = O.row(imgs[c], cam, xg, run.can['canonical_rgb'][p], run.noise, inv_policy, np.array(tr['prediction']['points'][str(p)]['P_minus'])[:3, :3], np.zeros(3))
                if 'r' not in oi: continue
                sm = om['sample']
                h1_rows.append({'point_id': p, 'category': cat[p], 'frame': k, 'camera': c, 'is_source_camera': int(run.can['original_source_ids'][p, 1]) == c,
                                'reason_metric': O.NAMES[om['reason']], 'du': sm['du'], 'dv': sm['dv'], 'spread_m': sm['same_surface_spread_m'],
                                'allowance_m': sm['same_surface_allowance_m'], 'r_D_metric_m': float(om['r'][3]), 'r_D_inverse_m': float(oi['r'][3]),
                                'sqrt_R_D_metric_m': float(math.sqrt(om['var'][3])), 'sqrt_R_D_inverse_m': float(math.sqrt(oi['var'][3])),
                                'z_metric': float(om['r'][3]/math.sqrt(om['var'][3])), 'z_inverse': float(oi['r'][3]/math.sqrt(oi['var'][3])),
                                'depth_slope_m_per_px': float(np.linalg.norm(sm['grad_depth'])), 'rgb_r_over_sigma_max': float(np.abs(om['r'][:3]/np.sqrt(om['var'][:3])).max()),
                                'range_minus_Z_m': float(np.linalg.norm(cam.R@xg+cam.t)-om['Z']), 'nis_metric_prior_cov': om['nis']})
        for p in need:
            sp = str(p); fin = tr['final'][sp]; pre = tr['prediction']['points'][sp]
            prob = make_problem(run, k, p, tr, imgs, cams)
            res = prob.run(run.policy.get('update_acceptance', 'final_iterate'))
            s_minus = prob.s_minus; s_lin = np.array(fin['linearization_state']); s_plus = np.array(fin['s_plus'])
            xg = gt.position(x0[p], moving[p], k)
            # C0: frame reproduction and every production iterate's algebra
            its = []
            for j, it in enumerate(tr['iterations']):
                q = it['points'][sp]
                if not q['active']: continue
                sj = np.array(q['state']); rows_o = prob.rows(sj[:3])
                A, b, d, step, Ixx, gx, rq = prob.solve(sj, rows_o)
                its.append({'scene': scene.name, 'point_id': p, 'frame': k, 'iteration': j, 'state': sj, 'delta_production': q['delta'], 'delta_oracle': d,
                            'A_rel': rel(A, q['A']), 'b_rel': rel(b, q['b']) if np.abs(np.asarray(q['b'])).max() > 0 else absd(b, q['b']), 'delta_rel': rel(d, q['delta']) if np.abs(np.asarray(q['delta'])).max() > 0 else absd(d, q['delta']), 'delta_position_linf_m': absd(d[:3], np.asarray(q['delta'])[:3]), 'delta_velocity_linf_m_s': absd(d[3:], np.asarray(q['delta'])[3:]),
                            'posterior_cost_production': q['posterior_cost'], 'cost_threshold': q['cost_threshold'], 'nrows': q['nrows'], 'step_delta_A_delta': q['step_delta_A_delta'],
                            'cameras_used': [o['camera'] for o in prob.used(rows_o)], 'pixel00_per_camera': {c['camera']: c['pixel00_u_v'] for c in q['cameras']},
                            'rows': [{'camera': o['camera'], 'u': o['u'], 'v': o['v'], 'r': o['r'], 'H_x': o['H'], 'R_diag': o['var'], 'nis': o['nis']} for o in prob.used(rows_o)],
                            'cholesky_ok': bool(np.all(np.linalg.eigvalsh(A) > 0))})
            for i_ in range(1, len(its)):
                its[i_]['pixel_cells_changed_vs_previous_iterate'] = [c for c in its[i_]['pixel00_per_camera'] if its[i_]['pixel00_per_camera'][c] != its[i_-1]['pixel00_per_camera'][c]]
            inner += its
            # the accepted linearization's rows (recheck of self-occlusion as production) and the first push at s-
            rows_acc = prob.rows(s_lin[:3], final=True); rows0 = prob.rows(s_minus[:3])
            sa, s0 = channel_split(prob, s_lin, rows_acc), channel_split(prob, s_minus, rows0)
            j1 = prob.joseph(rows_acc)
            Kgain = j1['K'] if j1 else None
            oracle_rows.append({'scene': scene.name, 'point_id': p, 'category': cat[p], 'frame': k, 'acceptance_production': fin['acceptance'],
                                'acceptance_oracle': ACC_NAMES[res['acceptance_code']], **{f's_plus_{kk}': vv for kk, vv in compare_state6(res['s_plus'], s_plus).items()}, 'P_plus_rel': rel(res['P_plus'], fin['P_plus']),
                                'iterations_A_rel_max': max((i_['A_rel'] for i_ in its), default=0.), 'iterations_delta_rel_max': max((i_['delta_rel'] for i_ in its), default=0.),
                                'joseph_vs_information_max_abs': j1['max_abs_difference'] if j1 else None,
                                'dx_m': (s_plus-s_minus)[:3].tolist(), 'dv_m_s': (s_plus-s_minus)[3:].tolist(), 'abs_dx_m': float(np.linalg.norm((s_plus-s_minus)[:3])),
                                'abs_dv_m_s': float(np.linalg.norm((s_plus-s_minus)[3:])), 'P_minus_xv': np.array(pre['P_minus'])[:3, 3:].tolist(),
                                'velocity_gain_P_vx_Pxx_inv': (np.array(pre['P_minus'])[3:, :3]@np.linalg.inv(np.array(pre['P_minus'])[:3, :3])).tolist(),
                                'K_velocity_rows_norm': float(np.linalg.norm(Kgain[3:])) if Kgain is not None else None,
                                'accepted_split_closure_max_abs': sa['closure_max_abs'], 'accepted_state_iterate_residual_norm': float(np.linalg.norm(sa['iterate_residual'])),
                                'cond_A': float(np.linalg.cond(sa['A']))})
            for nm, spl_ in (('accepted_linearization', sa), ('first_push_at_prior', s0)):
                for gname, v in spl_['groups'].items():
                    cf_rows.append({'scene': scene.name, 'point_id': p, 'category': cat[p], 'frame': k, 'kind': f'same_A_{nm}', 'variant': gname,
                                    'dx': v[:3].tolist(), 'dv': v[3:].tolist(), 'abs_dx_m': float(np.linalg.norm(v[:3])), 'abs_dv_m_s': float(np.linalg.norm(v[3:]))})
                for c, v in spl_['cameras'].items():
                    cf_rows.append({'scene': scene.name, 'point_id': p, 'category': cat[p], 'frame': k, 'kind': f'same_A_{nm}', 'variant': f'camera_{c}',
                                    'dx': v[:3].tolist(), 'dv': v[3:].tolist(), 'abs_dx_m': float(np.linalg.norm(v[:3])), 'abs_dv_m_s': float(np.linalg.norm(v[3:]))})
            # whole-Hessian counterfactuals: channel omitted (C2), inverse depth (C1), wrong-surface / suspected row removed (C3)
            used_c = [o['camera'] for o in prob.used(rows_acc)]
            recs = {o['camera']: row_record(o, cams[o['camera']], o['camera'], gt, k, xg, run.noise, n_sigma) for o in rows_acc if 'sample' in o}
            variants = [counterfactual(run, k, p, tr, imgs, cams, 'C0_production_rule_oracle'),
                        counterfactual(run, k, p, tr, imgs, cams, 'C2_rgb_only_resolved', channels=(1, 1, 1, 0)),
                        counterfactual(run, k, p, tr, imgs, cams, 'C2_depth_only_resolved', channels=(0, 0, 0, 1)),
                        counterfactual(run, k, p, tr, imgs, cams, 'C1_inverse_depth_resolved', policy=inv_policy),
                        counterfactual(run, k, p, tr, imgs, cams, 'C1_inverse_depth_frozen_rows', policy=inv_policy, fixed_rows=set(used_c))]
            own = 'MOVING_PART' if moving[p] else 'STATIC_PART'
            other = 'STATIC_PART' if moving[p] else 'MOVING_PART'        # a row imaging the OTHER part is proven wrong; UNLABELLED is unproven
            wrong = [c for c in used_c if c in recs and (recs[c].get('sampled_surface_gt_part') == other or other in recs[c].get('neighbour_gt_part', []))]
            if wrong:
                variants.append(counterfactual(run, k, p, tr, imgs, cams, 'C3_gt_wrong_surface_rows_removed_ORACLE_AUDIT',
                                               cam_mask=[c not in wrong for c in range(len(cams))]))
            if used_c:
                worst = max(used_c, key=lambda c: abs(recs[c]['r_D_over_sqrt_R']) if c in recs and 'r_D_over_sqrt_R' in recs[c] else -1)
                variants.append(counterfactual(run, k, p, tr, imgs, cams, f'C3_largest_depth_z_row_removed_SUSPECTED_cam{worst}',
                                               cam_mask=[c != worst for c in range(len(cams))]))
            base_d = s_plus-s_minus
            for v in variants:
                cf_rows.append({'scene': scene.name, 'point_id': p, 'category': cat[p], 'frame': k, 'kind': 'whole_hessian_resolve', 'variant': v['variant'],
                                'observed': v['observed'], 'acceptance': v['acceptance'], 'rows': ';'.join(map(str, v['rows'])), 'dx': v['dx'].tolist(), 'dv': v['dv'].tolist(),
                                'abs_dx_m': float(np.linalg.norm(v['dx'])), 'abs_dv_m_s': float(np.linalg.norm(v['dv'])),
                                'dx_ratio_to_production': float(np.linalg.norm(v['dx'])/max(np.linalg.norm(base_d[:3]), 1e-300)),
                                'dv_ratio_to_production': float(np.linalg.norm(v['dv'])/max(np.linalg.norm(base_d[3:]), 1e-300))})
            # raw rows of every camera at prediction, accepted state and the physical reference (C5 support change)
            for where, x in (('prediction_s_minus', s_minus[:3]), ('accepted_state', s_lin[:3]), ('physical_reference', xg)):
                fz = frozen_codes(tr['iterations'][0]['points'][sp]['cameras'])
                vis = O.candidacy(cams, x, fz, prob.recheck_self if where == 'accepted_state' else None)
                for c, cam in enumerate(cams):
                    o = O.row(imgs[c], cam, x, run.can['canonical_rgb'][p], run.noise, run.policy, np.array(pre['P_minus'])[:3, :3], x-s_minus[:3])
                    r_ = row_record(o, cam, c, gt, k, xg, run.noise, n_sigma)
                    r_.update(scene=scene.name, point_id=p, category=cat[p], frame=k, evaluated_at=where, visibility=O.NAMES[vis[c]],
                              admitted_in_accepted_solve=(where == 'accepted_state' and c in used_c),
                              ownership=tr['final'][sp]['cameras_final_iterate'][c]['pixel_occupants_front_to_back'][:3])
                    depth_samples.append(r_)
            # hypothesis evidence of this event
            dz = [recs[c] for c in used_c if c in recs and 'r_D_m' in recs[c]]
            hyp_ev['H2'].append({'point_id': p, 'frame': k, 'category': cat[p], 'admitted_rows': len(used_c), 'gt_wrong_surface_rows': wrong,
                                 'max_sampled_surface_to_point_gt_m': max((r_['sampled_surface_to_point_gt_m'] for r_ in dz), default=None)})
            hyp_ev['H3'].append({'point_id': p, 'frame': k, 'rows': [{'camera': r_['camera'], 'r_D_m': r_['r_D_m'], 'sqrt_R_D_m': r_['sqrt_R_D_m'], 'sqrt_S_DD_m': r_['sqrt_S_DD_m'],
                                                                     'nis': r_['nis'], 'threshold': r_['nis_threshold_chi2_4'],
                                                                     'position_sigma_m': float(np.sqrt(np.trace(np.array(pre['P_minus'])[:3, :3])/3))} for r_ in dz]})
            hyp_ev['H5'].append({'point_id': p, 'frame': k, 'acceptance': fin['acceptance'], 'posterior_cost': fin['posterior_cost'], 'threshold': fin['cost_threshold'],
                                 'converged': fin['converged'], 'iterations': len(its), 'fallback': not fin['final_iterate_state_sampled'],
                                 'cells_changed_during_iterations': sum(len(i_.get('pixel_cells_changed_vs_previous_iterate', [])) for i_ in its),
                                 'accepted_rows_invalid_at_accepted_state': [c for c in used_c if recs.get(c, {}).get('reason') != 'VALID'],
                                 **fixed_support_objective(prob, s_minus, s_lin, used_c)})
        print(f'[forensics] frame {k} ({kk}/{K-1}) decisive points {len(need)} ({time.perf_counter()-t_start:.0f} s)', flush=True)
    # ---- C4: pure prediction of the largest / first velocity injection (no later measurement)
    c4 = []
    for i, p in enumerate(pids):
        for nm in ('largest_dv_frame', 'first_dv_over_1e-3m_s'):
            k = events[p][nm]
            if k is None or k < 1: continue
            s = Sp[i, k]; xT_pred = s[:3]+(t[-1]-t[k])*s[3:]
            dep = Sp[i, -1, :3]-x0[p]; nn = np.linalg.norm(dep)
            c4.append({'point_id': p, 'category': cat[p], 'from_event': nm, 'frame': k, 'v_plus_m_s': s[3:].tolist(),
                       'pure_prediction_displacement_to_T_m': ((t[-1]-t[k])*s[3:]).tolist(),
                       'pure_prediction_projected_on_actual_departure_m': float(((t[-1]-t[k])*s[3:])@(dep/nn)) if nn > 0 else 0.,
                       'actual_departure_m': float(nn), 'x_T_pure_prediction_minus_actual_m': float(np.linalg.norm(xT_pred-Sp[i, -1, :3]))})
    # ---- H6: moving-point survival from the (identical) baseline arrays, all grounded moving points
    pop = dict(np.load(fdir/'population_events.npz'))
    surv = moving_survival(base, gt, pop)
    # ---- write compact outputs
    def wcsv(name, rows):
        if not rows: return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(fdir/name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k: json.dumps(jsonable(v)) if isinstance(v, (list, dict, np.ndarray)) else v for k, v in r.items()})
    wcsv('f1_state_timeline.csv', rows_tl); wcsv('f1_row_admission.csv', rows_adm); wcsv('f1_update_oracle.csv', oracle_rows)
    wcsv('f1_channel_counterfactuals.csv', cf_rows); wcsv('h1_depth_at_physical_reference.csv', h1_rows); wcsv('f1_moving_point_survival.csv', surv)
    wcsv('c4_pure_prediction.csv', c4)
    with open(fdir/'f1_inner_iterations.jsonl', 'w') as f:
        for r in inner: f.write(json.dumps(jsonable(r))+'\n')
    with open(fdir/'depth_samples_first_events.jsonl', 'w') as f:
        for r in depth_samples: f.write(json.dumps(jsonable(r))+'\n')
    np.savez_compressed(fdir/'f1_state_full_covariance.npz', point_id=np.array(pids), frames=np.array(order), s_minus=Sm, P_minus=Pm, s_plus=Sp, P_plus=Pp)
    h7 = numeric_checks(run, h1_rows, pred_err, oracle_rows, inner, ident, pids, t)
    hyp = hypotheses(pids, cat, events, ident, h1_rows, oracle_rows, cf_rows, hyp_ev, c4, surv, h7, gt)
    write_json(fdir/'hypotheses.json', hyp)
    write_json(fdir/'point_events.json', {'events': {str(p): events[p] for p in pids}, 'decisive_frames': {str(p): dec[p] for p in pids},
                                          'drift_identity': {str(p): ident[p] for p in pids}, 'gt_label_spacing_h_m': gt.h})
    readme(fdir, run, base, prov, scene, plan)
    scene_report(fdir, scene.name, prov, pids, cat, events, dec, ident, oracle_rows, cf_rows, hyp, c4, surv, h1_rows, depth_samples)
    print(json.dumps({'seconds': time.perf_counter()-t_start, 'hypotheses': {h: v['status'] for h, v in hyp.items() if isinstance(v, dict) and 'status' in v}}))
    return 0


def fixed_support_objective(prob, s_minus, s_lin, used_c):
    """H5: does the accepted state decrease the fixed-support nonlinear posterior J = prior + data on the ACCEPTED camera
    set (resampled at each state)? The production rule is a chi-square adequacy test, not a decrease test."""
    if not used_c: return {'fixed_support_J_prior_state': None, 'fixed_support_J_accepted_state': None}
    a, b = prob.cost_on(s_minus, used_c), prob.cost_on(s_lin, used_c)
    Ja = prob.prior_cost(s_minus)+a['rgb']+a['depth']; Jb = prob.prior_cost(s_lin)+b['rgb']+b['depth']
    return {'fixed_support_J_prior_state': Ja, 'fixed_support_J_accepted_state': Jb, 'fixed_support_decreased': bool(Jb < Ja),
            'rows_lost_at_prior_state': a['lost_rows'], 'rows_lost_at_accepted_state': b['lost_rows'],
            'depth_chi2_prior_state': a['depth'], 'depth_chi2_accepted_state': b['depth'], 'rgb_chi2_prior_state': a['rgb'], 'rgb_chi2_accepted_state': b['rgb']}


def moving_survival(base, gt, pop):
    """Per frame, every grounded moving point: rows / prediction-only / reason histogram / offscreen / measured-but-far
    (attached to another surface: rows while > 5 cm from its GT position) / still present in the output arrays."""
    from estimator.projective_sensor import REASONS
    mv = np.flatnonzero(pop['gt_moving'])
    order = base.fman['processing_order']
    X = np.asarray(base.mm('posterior_xyz')[:, mv], float); rows = np.asarray(base.mm('valid_cameras')[:, mv]); st = np.asarray(base.mm('point_status')[:, mv])
    reason = base.mm('camera_reason')
    x0 = base.can['canonical_xyz'][mv]
    out = []
    for k in order[1:]:
        gtp = gt.moved(x0, k); e = np.linalg.norm(X[k]-gtp, axis=-1)
        rk = np.asarray(reason[k][:, mv])
        off = np.isin(rk, [1, 2]).all(0)
        hist = {REASONS[c]: int((rk == c).any(0).sum()) for c in range(len(REASONS))}
        out.append({'frame': k, 'moving_points': len(mv), 'present_in_arrays': int(np.isfinite(X[k]).all(-1).sum()), 'with_rows': int((rows[k] > 0).sum()),
                    'prediction_only': int((st[k] == 2).sum()), 'offscreen_all_cameras': int(off.sum()),
                    'with_rows_but_over_5cm_from_gt': int(((rows[k] > 0) & (e > .05)).sum()), 'within_1cm': int((e < .01).sum()), 'within_5cm': int((e < .05).sum()),
                    'gt_error_p50_m': float(np.median(e)), 'gt_error_p90_m': float(np.quantile(e, .9)),
                    **{f'points_with_any_camera_{n}': v for n, v in hist.items()}})
    return out


def numeric_checks(run, h1_rows, pred_err, oracle_rows, inner, ident, pids, t):
    """H7: data / provenance / algebra checks with their float precision."""
    m = run.man; fm = run.fman
    dts = np.diff(t)
    zr = [(r['range_minus_Z_m'], r['r_D_metric_m']) for r in h1_rows if not r['is_source_camera']]
    corr = float(np.corrcoef(np.array(zr).T)[0, 1]) if len(zr) > 3 else None
    src = [r['r_D_metric_m'] for r in h1_rows if r['is_source_camera']]
    return {'prediction_identity_max_abs_or_rel': float(max(pred_err)) if len(pred_err) else None,
            'oracle_s_plus_position_linf_m_max': max((r['s_plus_position_linf_m'] for r in oracle_rows), default=None),
            'oracle_s_plus_velocity_linf_m_s_max': max((r['s_plus_velocity_linf_m_s'] for r in oracle_rows), default=None),
            'oracle_P_plus_rel_max': max((r['P_plus_rel'] for r in oracle_rows), default=None),
            'oracle_iteration_A_rel_max': max((r['A_rel'] for r in inner), default=None), 'oracle_iteration_delta_rel_max': max((r['delta_rel'] for r in inner), default=None),
            'oracle_iteration_delta_position_linf_m_max': max((r['delta_position_linf_m'] for r in inner), default=None),
            'oracle_iteration_delta_velocity_linf_m_s_max': max((r['delta_velocity_linf_m_s'] for r in inner), default=None),
            'note_delta_rel': 'relative step differences grow only for converging steps |delta| ~ 1e-8 m (b is a cancellation of large terms); the absolute difference is the test',
            'acceptance_disagreements': [(r['point_id'], r['frame'], r['acceptance_production'], r['acceptance_oracle']) for r in oracle_rows if r['acceptance_production'] != r['acceptance_oracle']],
            'joseph_vs_information_max_abs': max((r['joseph_vs_information_max_abs'] or 0 for r in oracle_rows), default=None),
            'drift_identity_closure_max_abs_m': max(v['closure_max_abs_m'] for v in ident.values()),
            'dt_s_min_max': [float(dts.min()), float(dts.max())], 'timestamps_source': (m['dataset'].get('timing') or {}).get('time_source'),
            'depth_is_Z_check': {'corr_r_D_vs_range_minus_Z_other_cameras': corr, 'rows': len(zr),
                                 'note': 'if the depth images stored Euclidean range, r_D at the physical reference would track range - Z'},
            'source_camera_r_D_at_reference_m': {'rows': len(src), 'max_abs': float(np.abs(src).max()) if src else None,
                                                 'note': 'x0 is back-projected from the frame-0 source depth; a static source pixel re-measured later gives r_D ~ 0 '
                                                         'unless the render changed: the frame-0 source row itself is never a measurement (F1 starts at k0 without update)'},
            'free_solver': m['arguments'].get('free_solver', 'directional (argument absent: standard F1)'), 'precision': fm['precision'],
            'pixel_reuse_assignment': fm['pixel_reuse_assignment'], 'update_acceptance': fm['update_acceptance'], 'row_innovation_gate': fm['row_innovation_gate'],
            'final_visibility': fm['final_visibility']}


def hypotheses(pids, cat, events, ident, h1_rows, oracle_rows, cf_rows, hyp_ev, c4, surv, h7, gt):
    """Evidence tables of H1-H7. Status rules are stated next to each status; the narrative verdict is in the report."""
    static_bad = [p for p in pids if cat[p] in ('static_over_10cm', 'static_1_to_10cm')]
    stable = [p for p in pids if cat[p] == 'static_stable_control']
    def onset(p):
        e = events[p]; k = e['first_accepted_dx_over_1e-4m'] if e['first_accepted_dx_over_1e-4m'] > 0 else e['first_dv_over_1e-3m_s']
        return k
    def cf(p, k, variant_prefix, kind='whole_hessian_resolve'):
        return [r for r in cf_rows if r['point_id'] == p and r['frame'] == k and r['kind'] == kind and r['variant'].startswith(variant_prefix)]
    rows = []
    for p in static_bad+stable:
        k = onset(p)
        if k <= 0: rows.append({'point_id': p, 'category': cat[p], 'onset_frame': None}); continue
        prod = next((r for r in oracle_rows if r['point_id'] == p and r['frame'] == k), None)
        g = {r['variant']: r for r in cf_rows if r['point_id'] == p and r['frame'] == k and r['kind'] == 'same_A_accepted_linearization' and r['variant'] in GROUP}
        g0 = {r['variant']: r for r in cf_rows if r['point_id'] == p and r['frame'] == k and r['kind'] == 'same_A_first_push_at_prior' and r['variant'] in GROUP}
        rgbonly, donly, inv, invf = (cf(p, k, s) for s in ('C2_rgb_only', 'C2_depth_only', 'C1_inverse_depth_resolved', 'C1_inverse_depth_frozen'))
        h1k = [r for r in h1_rows if r['point_id'] == p and r['frame'] == k]
        rows.append({'point_id': p, 'category': cat[p], 'onset_frame': k, 'production_abs_dx_m': prod['abs_dx_m'] if prod else None,
                     'production_abs_dv_m_s': prod['abs_dv_m_s'] if prod else None,
                     'same_A_accepted_dx_D_m': g.get('D', {}).get('abs_dx_m'), 'same_A_accepted_dx_RGB_m': g.get('RGB', {}).get('abs_dx_m'),
                     'same_A_accepted_dv_D_m_s': g.get('D', {}).get('abs_dv_m_s'), 'same_A_accepted_dv_RGB_m_s': g.get('RGB', {}).get('abs_dv_m_s'),
                     'first_push_dx_D_m': g0.get('D', {}).get('abs_dx_m'), 'first_push_dx_RGB_m': g0.get('RGB', {}).get('abs_dx_m'),
                     'rgb_only_resolved_abs_dx_m': rgbonly[0]['abs_dx_m'] if rgbonly else None, 'rgb_only_resolved_abs_dv_m_s': rgbonly[0]['abs_dv_m_s'] if rgbonly else None,
                     'depth_only_resolved_abs_dx_m': donly[0]['abs_dx_m'] if donly else None,
                     'inverse_depth_resolved_abs_dx_m': inv[0]['abs_dx_m'] if inv else None, 'inverse_depth_resolved_abs_dv_m_s': inv[0]['abs_dv_m_s'] if inv else None,
                     'inverse_depth_frozen_rows_abs_dx_m': invf[0]['abs_dx_m'] if invf else None,
                     'at_reference_max_abs_z_metric': max((abs(r['z_metric']) for r in h1k), default=None),
                     'at_reference_max_abs_z_inverse': max((abs(r['z_inverse']) for r in h1k), default=None)})
    def med(key, cats):
        v = [r[key] for r in rows if r.get(key) is not None and r['category'] in cats]
        return float(np.median(v)) if v else None
    bad = ('static_over_10cm', 'static_1_to_10cm')
    h1_st = [r for r in h1_rows if r['category'].startswith('static')]
    zm = np.array([r['z_metric'] for r in h1_st]); zi = np.array([r['z_inverse'] for r in h1_st])
    out = {'onset_events': rows}
    out['H1'] = {'question': 'interpolation / model discrepancy at the physically correct stationary position initiates the update',
                 'at_reference_all_static_rows': {'rows': int(len(zm)), 'mean_z_metric': float(zm.mean()) if len(zm) else None,
                                                  'median_abs_z_metric': float(np.median(np.abs(zm))) if len(zm) else None,
                                                  'fraction_abs_z_metric_over_3': float((np.abs(zm) > 3).mean()) if len(zm) else None,
                                                  'median_abs_z_inverse': float(np.median(np.abs(zi))) if len(zi) else None,
                                                  'fraction_abs_z_inverse_over_3': float((np.abs(zi) > 3).mean()) if len(zi) else None},
                 'onset_median_abs_dx_production_m': med('production_abs_dx_m', bad), 'onset_median_abs_dx_inverse_depth_resolved_m': med('inverse_depth_resolved_abs_dx_m', bad),
                 'onset_median_abs_dx_inverse_depth_frozen_rows_m': med('inverse_depth_frozen_rows_abs_dx_m', bad),
                 'status_rule': 'CONFIRMED if at the onset of most drifting points the metric-depth rows at the physical reference exceed 3 sigma AND the same-row '
                                'inverse-depth re-solve removes most of the onset correction; PARTIAL if only one holds'}
    n_bad = len([r for r in rows if r['category'] in bad and r['onset_frame']])
    big_ref = len([r for r in rows if r['category'] in bad and (r.get('at_reference_max_abs_z_metric') or 0) > 3])
    inv_fix = len([r for r in rows if r['category'] in bad and r.get('inverse_depth_frozen_rows_abs_dx_m') is not None and r['production_abs_dx_m']
                   and r['inverse_depth_frozen_rows_abs_dx_m'] < .5*r['production_abs_dx_m']])
    out['H1'].update(drifting_onsets=n_bad, onsets_with_reference_metric_rows_over_3sigma=big_ref, onsets_where_inverse_depth_halves_dx=inv_fix,
                     status='CONFIRMED' if n_bad and big_ref > n_bad/2 and inv_fix > n_bad/2 else ('PARTIAL' if n_bad and (big_ref > n_bad/2 or inv_fix > n_bad/2) else
                                                                                                ('DISPROVED' if n_bad else 'UNPROVEN')))
    h2 = hyp_ev['H2']
    out['H2'] = {'question': 'wrong-surface or tangentially ambiguous depth association', 'events': h2,
                 'events_with_gt_wrong_surface_rows': sum(1 for e in h2 if e['gt_wrong_surface_rows']), 'events': h2,
                 'identity_note': f'GT identity = nearest labelled GT point (spacing h = {gt.h:.4f} m) of each back-projected depth pixel at that frame; '
                                  'within-part tangential identity is NOT provable from depth (identity_unproven)',
                 'status_rule': 'CONFIRMED if wrong-part rows are admitted at the onset of most drifting points and their removal (C3) removes most of the correction'}
    c3 = [r for r in cf_rows if r['variant'].startswith('C3_gt_wrong')]
    out['H2']['c3_wrong_surface_removed'] = c3
    on_wrong = sum(1 for r in rows if r['category'] in bad and r['onset_frame'] and any(e['point_id'] == r['point_id'] and e['frame'] == r['onset_frame'] and e['gt_wrong_surface_rows'] for e in h2))
    out['H2'].update(drifting_onsets_with_wrong_part_rows=on_wrong,
                     status='CONFIRMED' if n_bad and on_wrong > n_bad/2 else ('PARTIAL' if on_wrong else ('DISPROVED' if n_bad else 'UNPROVEN')))
    out['H3'] = {'question': 'prediction covariance widens admission; a precise depth row then drives the update', 'events': hyp_ev['H3'],
                 'status': 'UNPROVEN', 'status_note': 'classified in the report from sqrt(S_DD) / sqrt(R_D) / position sigma of the H2 rows'}
    h4 = [r for r in rows if r['category'] in bad and r['onset_frame']]
    out['H4'] = {'question': 'position-velocity coupling turns a position correction into persistent motion',
                 'drift_identity': {str(p): ident[p] for p in static_bad}, 'pure_prediction_c4': c4,
                 'velocity_integration_share_median': float(np.median([ident[p]['velocity_integration_projected_m']/max(ident[p]['net_m'], 1e-300) for p in static_bad])) if static_bad else None,
                 'status': 'CONFIRMED' if static_bad and np.median([ident[p]['velocity_integration_projected_m']/max(ident[p]['net_m'], 1e-300) for p in static_bad]) > .5 else 'UNPROVEN',
                 'status_rule': 'CONFIRMED (as AMPLIFIER) if the integrated velocity explains > 50% of the median drifting point\'s net departure'}
    h5 = hyp_ev['H5']
    out['H5'] = {'question': 'standard-F1 acceptance / iteration mistake', 'events': h5,
                 'accepted_without_fixed_support_decrease': [(e['point_id'], e['frame']) for e in h5 if e.get('fixed_support_decreased') is False],
                 'accepted_rows_invalid_at_accepted_state': [(e['point_id'], e['frame'], e['accepted_rows_invalid_at_accepted_state']) for e in h5 if e['accepted_rows_invalid_at_accepted_state']],
                 'status': 'UNPROVEN', 'status_note': 'the production rule is a chi-square adequacy test (documented in point_kalman_cuda.py); a non-decrease is a behaviour, '
                                                      'a bug only if the code promises a decrease'}
    out['H6'] = {'question': 'moving-part point loss', 'per_frame': surv[-1] if surv else None}
    out['H7'] = {'question': 'numerical bug or data / provenance error', **h7,
                 'status': 'DISPROVED' if (h7['oracle_s_plus_position_linf_m_max'] or 0) < 1e-9 and (h7['oracle_s_plus_velocity_linf_m_s_max'] or 0) < 1e-7 and not h7['acceptance_disagreements'] and (h7['drift_identity_closure_max_abs_m'] < 1e-9) else 'PARTIAL',
                 'status_rule': 'DISPROVED if the independent float64 oracle reproduces every decisive accepted state (position < 1e-9 m, velocity < 1e-7 m/s, compared separately) and acceptance decision and the drift identity closes'}
    return out


def readme(fdir, run, base, prov, scene, plan):
    txt = f"""# F1 inner-IEKF static-drift forensics: {scene.name}

Spec: RGBDiekf_Static_Drift_INNER_IEKF_Only_Codex_2026-10-30.md (standard directional forward F1 only; no RTS / IEKS / temporal solver).

- Baseline (untraced standard F1): `{base.root}`
- Traced re-run (same resolved arguments + `--debug-point-ids` / `--debug-frames`): `{run.root}`
- Tracing invariance: {'every published array IDENTICAL' if prov['tracing_invariant'] else 'DIFFERENT (analysis stopped)'}; argument differences other than output / trace switches: {prov['argument_differences_other_than_output_and_trace'] or 'none'}
- Git HEAD {prov['snapshot']['git_head']} (dirty working tree: see provenance.json snapshot), Python {prov['snapshot']['python']}, torch {prov['snapshot']['torch']}, CUDA {prov['snapshot']['cuda']}, GPU {prov['snapshot']['gpu']}

Commands (repository root, `PY=~/miniconda3/envs/rigeuler3dgs/bin/python`):

```bash
$PY tools/forensic_f1_drift.py select "{base.root}" --scene "{scene}" --out "{fdir}"
$PY tools/forensic_f1_drift.py trace-run "{fdir}" --scene "{scene}" --output "{run.root}"
$PY tools/forensic_f1_drift.py analyze "{run.root}" --baseline "{base.root}" --forensics "{fdir}"
```

Traced train.py argv: `{' '.join(traced_command(plan, scene, run.root))}`

Files: selected_point_ids.csv, population_summary.json / population_events.npz (every grounded point, saved arrays), f1_state_timeline.csv (every
frame of every selected id), f1_state_full_covariance.npz (full 6x6 P-/P+), f1_inner_iterations.jsonl (decisive frames' production iterates with
the float64 oracle algebra), depth_samples_first_events.jsonl (every camera's raw 2x2 row at prediction / accepted state / physical reference
with GT surface identity), f1_row_admission.csv (every frame / camera of every selected id), f1_update_oracle.csv, f1_channel_counterfactuals.csv
(same-A splits and whole-Hessian re-solves C0-C3), c4_pure_prediction.csv, h1_depth_at_physical_reference.csv, f1_moving_point_survival.csv,
hypotheses.json, point_events.json, provenance.json, root_cause_report.md.
"""
    (fdir/'README.md').write_text(txt)


def fmt(v, p=3):
    if v is None: return '-'
    if isinstance(v, float):
        if v == 0: return '0'
        return f'{v:.{p}g}'
    return str(v)


def scene_report(fdir, scene, prov, pids, cat, events, dec, ident, oracle_rows, cf_rows, hyp, c4, surv, h1_rows, depth_samples):
    pop = json.loads((fdir/'population_summary.json').read_text())
    L = [f'# {scene}: standard F1 static-drift forensics (generated tables)', '',
         'Generated by tools/forensic_f1_drift.py analyze; the causal argument across scenes is in reports/f1_static_drift_forensics_20261030.md.', '',
         f"Tracing invariance: {'IDENTICAL' if prov['tracing_invariant'] else 'DIFFERENT'}.", '',
         '## Population (saved standard-F1 arrays, every grounded point)', '',
         f"Static points {pop['static']['points']}: final transport drift p50/p90/p99 = {fmt(pop['static']['drift_final_m']['p50'])} / {fmt(pop['static']['drift_final_m']['p90'])} / "
         f"{fmt(pop['static']['drift_final_m']['p99'])} m; > 10 cm {fmt(pop['static']['over_10cm_fraction'])}; > 1 cm {fmt(pop['static']['over_1cm_fraction'])}; "
         f"< 1 mm for the whole sequence {fmt(pop['static']['stable_under_1mm_fraction'])}.",
         f"Moving points {pop['moving']['points']}: final GT error p50/p90 = {fmt(pop['moving']['gt_error_final_m']['p50'])} / {fmt(pop['moving']['gt_error_final_m']['p90'])} m; "
         f"within 1 cm {fmt(pop['moving']['within_1cm'])}, within 5 cm {fmt(pop['moving']['within_5cm'])}, over 10 cm {fmt(pop['moving']['over_10cm'])}.", '',
         '## Selected points: events (original frame index; -1 never)', '',
         '| id | category | measured | 1st dx>0.1mm | 1st dv>1mm/s | drift>1mm | >1cm | >10cm | row loss | readmit | largest dv (frame, m/s) | net (m) | velocity-integration share |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in pids:
        e = events[p]; d = ident[p]
        L.append(f"| {p} | {cat[p]} | {e['first_measured']} | {e['first_accepted_dx_over_1e-4m']} | {e['first_dv_over_1e-3m_s']} | {e['first_drift_over_0.001m']} | "
                 f"{e['first_drift_over_0.01m']} | {e['first_drift_over_0.1m']} | {e['first_row_loss']} | {e['first_readmission_after_loss']} | "
                 f"{e['largest_dv_frame']}, {fmt(e['largest_dv_m_s'])} | {fmt(d['net_m'])} | {fmt(d['velocity_integration_projected_m']/max(d['net_m'], 1e-300))} |")
    L += ['', '## Onset event of every static point (first accepted |dx| > 0.1 mm or |dv| > 1 mm/s)', '',
          '| id | category | frame | prod |dx| | prod |dv| | same-A D |dx| | same-A RGB |dx| | RGB-only re-solve |dx| | inverse-depth re-solve |dx| | inverse, same rows |dx| | max |z_D| at reference (metric / inverse) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in hyp['onset_events']:
        if not r.get('onset_frame'): L.append(f"| {r['point_id']} | {r['category']} | none | | | | | | | | |"); continue
        L.append(f"| {r['point_id']} | {r['category']} | {r['onset_frame']} | {fmt(r['production_abs_dx_m'])} | {fmt(r['production_abs_dv_m_s'])} | {fmt(r['same_A_accepted_dx_D_m'])} | "
                 f"{fmt(r['same_A_accepted_dx_RGB_m'])} | {fmt(r['rgb_only_resolved_abs_dx_m'])} | {fmt(r['inverse_depth_resolved_abs_dx_m'])} | {fmt(r['inverse_depth_frozen_rows_abs_dx_m'])} | "
                 f"{fmt(r['at_reference_max_abs_z_metric'])} / {fmt(r['at_reference_max_abs_z_inverse'])} |")
    L += ['', '## Hypotheses (automatic status by the stated rule; see the cross-scene report for the verdict)', '']
    for h in ('H1', 'H2', 'H3', 'H4', 'H5', 'H7'):
        v = hyp[h]; L.append(f"- **{h}** {v['question']}: {v.get('status')} ({v.get('status_rule', v.get('status_note', ''))})")
    L += ['', f"H7 numeric checks: oracle s+ vs production: position max {fmt(hyp['H7']['oracle_s_plus_position_linf_m_max'])} m, velocity max {fmt(hyp['H7']['oracle_s_plus_velocity_linf_m_s_max'])} m/s, P+ rel {fmt(hyp['H7']['oracle_P_plus_rel_max'])}, "
          f"iteration A rel {fmt(hyp['H7']['oracle_iteration_A_rel_max'])}, delta rel {fmt(hyp['H7']['oracle_iteration_delta_rel_max'])}, acceptance disagreements "
          f"{len(hyp['H7']['acceptance_disagreements'])}, drift identity closure {fmt(hyp['H7']['drift_identity_closure_max_abs_m'])} m, prediction identity "
          f"{fmt(hyp['H7']['prediction_identity_max_abs_or_rel'])}, depth-is-Z correlation {fmt(hyp['H7']['depth_is_Z_check']['corr_r_D_vs_range_minus_Z_other_cameras'])}.", '']
    if surv:
        L += ['## Moving-point survival (every grounded moving point)', '', '| frame | with rows | prediction-only | offscreen | rows but > 5 cm from GT | within 1 cm | within 5 cm | GT err p50 (m) |',
              '|---|---|---|---|---|---|---|---|']
        for s in surv[::max(1, len(surv)//12)]+[surv[-1]]:
            L.append(f"| {s['frame']} | {s['with_rows']} | {s['prediction_only']} | {s['offscreen_all_cameras']} | {s['with_rows_but_over_5cm_from_gt']} | {s['within_1cm']} | {s['within_5cm']} | {fmt(s['gt_error_p50_m'])} |")
    (fdir/'root_cause_report.md').write_text('\n'.join(L)+'\n')


def cmd_report(a):
    raise NotImplementedError('the cross-scene report is written from the per-scene outputs')


# ------------------------------------------------------------------------------------------------- where / when (all)
def velocity_kick(run, si, df, pop):
    """Population C4: how much of each static point's final departure x+_T - x0 is the PURE PREDICTION of its velocity after
    its first measured frame k1 ((t_T - t_k1) v+_k1, no later correction), and of the frame-1 accepted correction / velocity
    (sizes, and the velocity-to-correction ratio vs 1/dt: the prior gain P_vx P_xx^-1 ~ 1/dt when P_vv0 dominates)."""
    T = trajectories(run, si)
    X, V, t, st = T['X'], T['V'], T['t'], T['st']
    meas = st[1:] == 1
    k1 = np.where(meas.any(0), meas.argmax(0)+1, -1)
    net = X[-1]-X[0]; nn = np.linalg.norm(net, axis=-1); dh = net/np.where(nn > 0, nn, 1)[:, None]
    ok = k1 > 0; i = np.flatnonzero(ok); kk = k1[ok]
    pure = ((t[-1]-t[kk])[:, None]*V[kk, i])
    share = np.full(len(si), np.nan); share[i] = (pure*dh[i]).sum(-1)/np.maximum(nn[i], 1e-300)
    dx1 = np.full(len(si), np.nan); dv1 = np.full(len(si), np.nan)
    xm = X[kk-1, i]+(t[kk]-t[kk-1])[:, None]*V[kk-1, i]
    dx1[i] = np.linalg.norm(X[kk, i]-xm, axis=-1); dv1[i] = np.linalg.norm(V[kk, i]-V[kk-1, i], axis=-1)
    dt = float(np.median(np.diff(t)))
    q = lambda v: {f'p{int(100*x)}': float(np.quantile(v, x)) for x in (.1, .5, .9)} if len(v) else None
    out = {'dt_s': dt, 'one_over_dt': 1/dt}
    for nm, m in (('drifting_over_10cm', ok & (df > .1)), ('drifting_1_to_10cm', ok & (df > .01) & (df <= .1)), ('stable_under_1mm', ok & (pop['drift_max'][si] < 1e-3))):
        out[nm] = {'points': int(m.sum()), 'first_measured_frame': q(k1[m]), 'first_accepted_correction_m': q(dx1[m]), 'first_velocity_change_m_s': q(dv1[m]),
                   'dv_over_dx_per_s': q(dv1[m]/np.maximum(dx1[m], 1e-300)),
                   'pure_prediction_share_of_final_departure_from_first_measured_velocity': q(share[m])}
    return out


def consistency_at_reference(run, pop, si, df, k):
    """Population initiator test (no trace needed): the rows the standard F1 ACTUALLY used at frame k (camera_reason == VALID
    of an accepted update) re-evaluated at the static reference x0 with the frame-k images: signed r_D = Z - D (metric and
    inverse bilinear), its standardized value with the run's R (sum_w2 + model), RGB |z| max. A physically static point at
    x0 should give |z| ~ 1 in every used row; large values are rows that push it (initiator candidates)."""
    reason = np.asarray(run.mm('camera_reason')[k][:, si]); acc = np.asarray(run.mm('update_acceptance')[k][si])
    used = (reason == 0) & np.isin(acc, (1, 2))[None]
    x = pop['x0'][si]; nz = run.noise; pol = run.policy
    cams = run.cams()
    zD = np.full(reason.shape, np.nan); zDi = np.full(reason.shape, np.nan); rD = np.full(reason.shape, np.nan); zRGB = np.full(reason.shape, np.nan)
    spread_ok = np.zeros(reason.shape, bool)
    for c, cam in enumerate(cams):
        m = used[c]
        if not m.any(): continue
        obs = run.data.observation(k, c)
        D = np.asarray(obs.depth, np.float32).astype(float); I = np.asarray(obs.rgb, np.float32).astype(float)
        Xc = x[m]@cam.R.T+cam.t; Z = Xc[:, 2]
        u, v = cam.fx*Xc[:, 0]/Z+cam.cx, cam.fy*Xc[:, 1]/Z+cam.cy
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        ins = (u0 >= 0) & (v0 >= 0) & (u0+1 <= cam.W-1) & (v0+1 <= cam.H-1)
        u0c, v0c = np.clip(u0, 0, cam.W-2), np.clip(v0, 0, cam.H-2)
        du, dv = u-u0, v-v0
        Dn = np.stack([D[v0c, u0c], D[v0c, u0c+1], D[v0c+1, u0c], D[v0c+1, u0c+1]], 1)
        In = np.stack([I[v0c, u0c], I[v0c, u0c+1], I[v0c+1, u0c], I[v0c+1, u0c+1]], 1)
        w = np.stack([(1-du)*(1-dv), du*(1-dv), (1-du)*dv, du*dv], 1)
        zbar = Dn.mean(1); f = min(cam.fx, cam.fy)
        allow = pol['same_surface_n_sigma']*math.sqrt(2.)*np.sqrt(nz.sigma2(zbar))+zbar*math.tan(math.radians(pol['same_surface_max_incidence_deg']))*math.sqrt(2.)/f
        ok = ins & (Dn.max(1)-Dn.min(1) <= allow) & (Dn > 0).all(1)
        dm = (w*Dn).sum(1); di = 1/(w/np.where(Dn > 0, Dn, np.inf)).sum(1)
        var = (w*w*nz.sigma2(Dn)).sum(1)+nz.depth_model
        vari = di**4*(w*w*nz.sigma2(Dn)/np.where(Dn > 0, Dn, np.inf)**4).sum(1)+nz.depth_model
        rgb = (w[..., None]*In).sum(1); a = run.can['canonical_rgb'][si][m]
        vrgb = (w*w).sum(1)[:, None]*nz.rgb_var[None]+nz.rgb_model
        idx = np.flatnonzero(m)
        rD[c, idx] = Z-dm; zD[c, idx] = (Z-dm)/np.sqrt(var); zDi[c, idx] = (Z-di)/np.sqrt(vari)
        zRGB[c, idx] = np.abs((a-rgb)/np.sqrt(vrgb)).max(1); spread_ok[c, idx] = ok
    has = used.any(0)
    mx = np.nanmax(np.where(used, np.abs(zD), np.nan), 0, initial=0.)
    mxi = np.nanmax(np.where(used, np.abs(zDi), np.nan), 0, initial=0.)
    mxr = np.nanmax(np.where(used, zRGB, np.nan), 0, initial=0.)
    out = {'frame': int(k), 'static_points_with_used_rows': int(has.sum()),
           'used_rows': int(used.sum()), 'used_rows_same_surface_rule_holds_at_x0': int((spread_ok & used).sum()),
           'note': 'rows used by the production accepted update of frame k, re-sampled at x0 (static reference = canonical sample); z = r / sqrt(R)'}
    tab = []
    for lo, hi in ((0, 3), (3, 10), (10, 100), (100, 1e3), (1e3, 1e12)):
        mm = has & (mx >= lo) & (mx < hi)
        if mm.any(): tab.append({'max_abs_z_depth_metric_at_x0': [lo, hi], 'points': int(mm.sum()), 'over_10cm': float((df[mm] > .1).mean()),
                                 'over_1cm': float((df[mm] > .01).mean()), 'drift_p50_m': float(np.median(df[mm]))})
    out['drift_by_max_abs_z_depth_at_x0'] = tab
    big = has & (df > .1); stab = has & (pop['drift_max'][si] < 1e-3)
    q = lambda v: {f'p{int(100*x)}': float(np.quantile(v, x)) for x in (.1, .5, .9)} if len(v) else None
    out['drifting_over_10cm'] = {'points': int(big.sum()), 'max_abs_z_depth_metric': q(mx[big]), 'max_abs_z_depth_inverse': q(mxi[big]), 'max_abs_z_rgb': q(mxr[big]),
                                 'fraction_max_abs_z_depth_over_3': float((mx[big] > 3).mean()) if big.any() else None,
                                 'signed_r_D_mm_of_worst_row': q(1e3*np.array([rD[np.nanargmax(np.where(used[:, i], np.abs(zD[:, i]), -1)), i] for i in np.flatnonzero(big)]))}
    out['stable_under_1mm'] = {'points': int(stab.sum()), 'max_abs_z_depth_metric': q(mx[stab]), 'max_abs_z_depth_inverse': q(mxi[stab]), 'max_abs_z_rgb': q(mxr[stab]),
                               'fraction_max_abs_z_depth_over_3': float((mx[stab] > 3).mean()) if stab.any() else None}
    out['sqrt_R_D_m_typical'] = float(np.sqrt(nz.sigma2(1.)+nz.depth_model))
    return out


def cmd_where(a):
    """Population WHERE / WHEN of static drift and moving loss from the saved standard-F1 arrays (every grounded point):
    onset frames against the GT joint motion, drift by distance to the moving part / source camera / source-pixel depth
    edge and slope / visible cameras, the camera reasons present at the onset frame vs stable points at the same frame,
    and the order of events (row loss before / after onset). GT only classifies (offline)."""
    from estimator.projective_sensor import REASONS
    fdir = Path(a.forensics); run = Run(a.baseline, 'forward')
    pop = dict(np.load(fdir/'population_events.npz'))
    order = run.fman['processing_order']; K = len(order)
    gtb = gt_basis(Path(run.man['arguments']['scene']), pop['x0'])
    fids = run.man['dataset']['original_frame_ids']
    q = gtb['q'][fids] if len(gtb['q']) > max(fids) else gtb['q']
    st_, mv = pop['gt_static'], pop['gt_moving']
    si = np.flatnonzero(st_)
    df = pop['drift_final'][si]; on = pop['first_drift_over_0.001m'][si]
    # source-pixel features (frame-0 source image of each point): depth edge = max |D_nb - D| over the 3x3 neighbourhood, slope
    src = run.can['original_source_ids'][si]; W = [c.width for c in run.data.cameras]; k0 = order[0]
    edge = np.zeros(len(si)); slope = np.zeros(len(si)); zsrc = np.zeros(len(si))
    for c in np.unique(src[:, 1]):
        D = np.asarray(run.data.observation(k0, int(c)).depth, float)
        Dp = np.pad(D, 1, mode='edge')
        nb = np.stack([Dp[1+dy:1+dy+D.shape[0], 1+dx:1+dx+D.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        e = np.abs(nb-D[None]).max(0); gy, gx = np.gradient(D)
        m = src[:, 1] == c; pix = src[m, 2]; u, v = pix % W[c], pix//W[c]
        edge[m], slope[m], zsrc[m] = e[v, u], np.hypot(gx, gy)[v, u], D[v, u]
    rows1 = pop['rows_frame1'][si]; dmov = pop['d_moving'][si]
    def table(x, edges, name):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum(): out.append({name: [float(lo), float(hi)], 'points': int(m.sum()), 'over_10cm': float((df[m] > .1).mean()), 'over_1cm': float((df[m] > .01).mean()),
                                    'drift_p50_m': float(np.median(df[m])), 'onset_frame_p50': float(np.median(on[m][on[m] > 0])) if (on[m] > 0).any() else None})
        return out
    h = gtb['meta']['gt_cloud_spacing_h_m']
    rep = {'scene': run.man['arguments']['scene'], 'static_points': int(len(si)), 'gt_label_spacing_h_m': h,
           'by_distance_to_moving_part_m': table(dmov, np.array([0, 2, 4, 8, 16, 32, 1e9])*h, 'd_moving_m'),
           'by_source_camera': table(src[:, 1].astype(float), np.arange(-.5, len(W)+.5, 1.), 'source_camera'),
           'by_rows_frame1': table(rows1.astype(float), np.arange(-.5, len(W)+1.5, 1.), 'rows_frame1'),
           'by_source_depth_edge_m': table(edge, np.array([0, 1e-4, 1e-3, 1e-2, 5e-2, 1e9]), 'edge_m'),
           'by_source_depth_slope_m_per_px': table(slope, np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e9]), 'slope_m_px'),
           'joint_q': q.tolist(), 'joint_q_note': 'GT joint value per processed frame (offline): when does the moving part move?'}
    # WHEN: onset histogram vs joint motion
    dq = np.abs(np.diff(q))
    first_motion = int(np.argmax(dq > 1e-9*max(dq.max(), 1e-300))+1) if (dq > 0).any() else -1
    hist = np.bincount(on[on > 0], minlength=K)
    rep['onset_drift_over_1mm_histogram'] = hist.tolist(); rep['first_frame_joint_moves'] = first_motion
    rep['onset_before_joint_moves'] = int(((on > 0) & (on < first_motion)).sum()) if first_motion > 0 else None
    rep['onset_frame_quantiles_drifting_over_10cm'] = {f'p{int(100*x)}': float(np.quantile(on[(df > .1) & (on > 0)], x)) for x in (.1, .5, .9)} if ((df > .1) & (on > 0)).any() else None
    # event order among drifting points
    loss = pop['first_row_loss'][si]; big = df > .1
    rep['event_order_drift_over_10cm'] = {'points': int(big.sum()), 'onset_before_first_row_loss': int((big & (on > 0) & ((loss < 0) | (on < loss))).sum()),
                                          'onset_after_first_row_loss': int((big & (on > 0) & (loss > 0) & (on > loss)).sum()),
                                          'onset_same_frame_as_row_loss': int((big & (on > 0) & (on == loss)).sum()),
                                          'velocity_integration_share_p50': float(np.median((pop['velocity_integration_projected_m']/np.maximum(np.linalg.norm(pop['net_vector'], axis=-1), 1e-300))[si][big])) if big.any() else None}
    # camera reasons at the onset frame: drifting points vs stable points at the same frame
    reason = run.mm('camera_reason'); stable = pop['drift_max'][si] < 1e-3
    rs = {}
    for k in sorted(set(on[big & (on > 0)].tolist()))[:K]:
        mb = big & (on == k)
        rk = np.asarray(reason[k][:, si])
        for nm, m in (('drifting_at_their_onset', mb), ('stable_same_frame', stable)):
            if not m.any(): continue
            d = rs.setdefault(nm, np.zeros(len(REASONS)))
            d += np.array([(rk[:, m] == c).sum() for c in range(len(REASONS))])
    rep['camera_reasons_at_onset_frames'] = {nm: {REASONS[c]: float(v[c]/max(v.sum(), 1)) for c in range(len(REASONS)) if v[c]} for nm, v in rs.items()}
    # moving: when do moving points lose rows / leave the GT surface
    mvi = np.flatnonzero(mv)
    rep['moving'] = {'points': int(len(mvi)), 'first_row_loss_quantiles': {f'p{int(100*x)}': float(np.quantile(pop['first_row_loss'][mvi][pop['first_row_loss'][mvi] > 0], x)) for x in (.1, .5, .9)} if (pop['first_row_loss'][mvi] > 0).any() else None,
                     'never_lose_rows_fraction': float((pop['first_row_loss'][mvi] < 0).mean()), 'final_error_quantiles_m': {f'p{int(100*x)}': float(np.quantile(pop['gt_error_final_moving'][mvi], x)) for x in (.5, .9)}}
    rep['rows_used_at_frame1_evaluated_at_x0'] = consistency_at_reference(run, pop, si, df, k=order[1])
    rep['velocity_kick_population_c4'] = velocity_kick(run, si, df, pop)
    write_json(fdir/'population_where_when.json', rep)
    print(json.dumps({k: rep[k] for k in ('by_distance_to_moving_part_m', 'first_frame_joint_moves', 'onset_before_joint_moves', 'event_order_drift_over_10cm')}, default=float)[:3000])


if __name__ == '__main__':
    main()
