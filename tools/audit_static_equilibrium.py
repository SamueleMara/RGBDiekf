"""Static-point equilibrium audit (RGBDiekf_Codex_Static_Point_Equilibrium_Root_Cause_Audit.md): why does a point that is
physically static, starts with zero velocity and keeps its identity under forward transport get an accepted non-zero
correction, a velocity, and escape?  DIAGNOSTIC ONLY: nothing here reaches inference. Every alternative is an offline
evaluation of the measurement function or of one linearized update (tools/root_cause_oracle.py, independent float64
NumPy), never a new policy and never a published state.

    fixture   OUT                       the completely static ray-cast fixture (tests/image_fixtures two_plate_scene
                                        'static') written as a dataset folder OUT/scene, for train.py
    select    BASELINE --scene S --kind synthetic|fixture|real
                                        untraced forward run -> static basis, baseline numbers, events of every static
                                        point, the audit samples and the trace plan (BASELINE/static_audit/selection.json)
    analyze   TRACED --baseline BASELINE
                                        the same run re-executed with --debug-point-ids / --debug-frames from the plan:
                                        parity with the baseline, then per audit point and traced frame the rows, algebra,
                                        linearization, comparable-row objectives, velocity coupling, invariants and the
                                        one-step counterfactuals (TRACED/static_audit/)

Static basis (offline only, never fed to the filter):
  synthetic  GT part labels (pointcloud/points3D_parts.ply, the labelled sparse cloud of the canonical endpoint) and the
             GT joint (tools/evaluate_projective_gt.py). The labelled cloud is sparse, so a point is GROUNDED STATIC only
             if its nearest labelled point is static and within one GT-cloud spacing h (median nearest-neighbour distance
             of the labelled cloud) and no moving-labelled point lies within 2h; anything else is UNGROUNDED and excluded.
             The GT position of a static point at every frame is its canonical surface sample x0 (identity transform).
  fixture    every point static by construction; x_GT = x0 (ray-cast surface sample).
  real       NO ground truth exists (README: "Masks and joint GT are absent"). A measurement-only PROXY is reported and
             labelled as such: DEPTH-CONSTANT points, whose nearest pixel in >= 2 cameras keeps a valid foreground sample
             with depth equal to frame 0 within n_sigma sqrt(2) sigma_D (the declared visibility_n_sigma and the run's
             measured temporal sensor sigma) in EVERY frame of a declared early window (--window frames; the audit is then
             restricted to that window). Colour is not required: with the measured temporal RGB sigma no pixel of the real
             scenes keeps its colour over the sequence (lighting / exposure; the count is reported). It cannot exclude a
             surface sliding along itself; it is not GT.

Thresholds below only CLASSIFY evidence for the report (each is named where used); none reaches a filter.
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import root_cause_oracle as O                                                         # noqa: E402
from tools.root_cause_trace import Run, jsonable, write_json, git, rel, absd, prediction_section, problem_from_trace, frozen_codes, endpoint_cross_camera  # noqa: E402

REPORTING_THRESHOLD_M = 0.10       # the spec's ">0.10 m" departure: reporting only
STABLE_FRACTION = 0.10             # a control is "stable" if its error never exceeds 10% of the reporting threshold (1 cm)
ALPHAS = (1., .5, .25, .125)
EPS64, EPS32 = float(np.finfo(np.float64).eps), float(np.finfo(np.float32).eps)
SAMPLEABLE = (O.VALID, O.FRONT, O.INCONS)      # a sample exists (only the innovation gate decided)
FUNCTIONS = {'scene/endpoint.py': ['endpoint_samples'], 'estimator/free_projective_run.py': ['run_free'],
             'estimator/point_kalman_cuda.py': ['predict', 'PointMeasurement', 'update_points'], 'estimator/projective_terms.py': ['build_terms'],
             'estimator/projective_sensor.py': ['sample'], 'estimator/projective_visibility.py': ['visibility', 'apply_frozen_assignment']}
TOUCHED = ['tools/audit_static_equilibrium.py', 'tests/test_static_equilibrium_audit.py']


# ----------------------------------------------------------------------------------------------------------- snapshot
def snapshot(runs):
    import importlib, torch
    funcs = {}
    for f, names in FUNCTIONS.items():
        mod = importlib.import_module(f[:-3].replace('/', '.'))
        funcs[f] = {n: hasattr(mod, n) for n in names}
    src = sorted(set(json.loads((Path(runs[0])/'run_manifest.json').read_text())['source_hashes']) | set(TOUCHED))
    return {'time': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'git_head': git('rev-parse', 'HEAD').strip(), 'git_status_short': git('status', '--short'),
            'modified_vs_head': git('diff', '--stat', 'HEAD'),
            'file_sha256': {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in src if (REPO/f).exists()},
            'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'platform': platform.platform(),
            'functions_present': funcs,
            'run_commands': {str(r): json.loads((Path(r)/'run_manifest.json').read_text())['arguments'] for r in runs}}


def contract(run):
    """The experimental contract of one run: CLI, policy, noise (with provenance class), timing and cameras."""
    m, f = run.man, run.fman
    t = np.asarray(f['timestamps_seconds'])
    noise = m['noise']
    pv = noise['provenance']
    calib = None
    if (run.root/'noise_calibration.json').exists():
        calib = json.loads((run.root/'noise_calibration.json').read_text())['result']
    return {'arguments': m['arguments'], 'git_commit': m.get('git_commit'), 'source_hashes': m.get('source_hashes'),
            'policy': {k: m['policy'].get(k) for k in ('update_acceptance', 'row_innovation_gate', 'final_visibility', 'visibility_n_sigma', 'same_surface_n_sigma',
                                                       'same_surface_max_incidence_deg', 'interpolation_variance', 'rgb_model_variance', 'depth_model_variance',
                                                       'max_iekf_iters', 'iekf_step_tolerance', 'sensor_noise')},
            'rendered_mask_measurement_mode': m['policy']['rendered_mask_measurement']['mode'],
            'pixel_reuse_assignment': f['pixel_reuse_assignment'], 'precision': f['precision'], 'max_iekf_iters': f['max_iekf_iters'], 'step_tolerance': f['step_tolerance'],
            'timing': {'frames': len(t), 'dt_s_unique': sorted({round(float(d), 12) for d in np.diff(t)})[:10], 'duration_s': float(t[-1]-t[0]),
                       'source': {k: (m['dataset'].get('timing') or {}).get(k) for k in ('time_source', 'source_kind', 'fps', 'interval_count', 'interval_min_max_seconds',
                                                                                          'metadata_disagreements')}},
            'cameras': m['cameras'],
            'prior': {'P_xx0': 'per-point back-projection covariance of the canonical source pixel (scene/endpoint.py backproject, pixel covariance '
                               f"{noise['pixel_covariance']} px^2, depth variance {noise['depth_variance']} + {noise['depth_squared_variance']} z^2 m^2)",
                      'P_vv0_m2_s2': noise['velocity_covariance'], 'P_vv0_class': pv.get('velocity_covariance', {}).get('class'),
                      'P_xv0': 'zero', 'q_c_m2_s3': noise['qc'], 'q_c_class': pv.get('qc', {}).get('class'),
                      'uncalibrated_note': 'P_vv0 and q_c are the documented initial guess (configs/example_uncalibrated.json, UNCALIBRATED) unless the class says otherwise'},
            'R': {'rgb_variance': np.diag(np.asarray(noise['color_covariance'])).tolist(), 'depth_variance_m2': noise['depth_variance'],
                  'depth_squared_variance': noise['depth_squared_variance'], 'rgb_model_variance': m['policy']['rgb_model_variance'],
                  'depth_model_variance_m2': m['policy']['depth_model_variance'], 'class': pv.get('color_covariance', {}).get('class'), 'calibration_result': calib}}


# ------------------------------------------------------------------------------------------------------- trajectories
def trajectories(run, idx, window=None):
    """Processing-ordered float64 arrays of the points idx: X (float64 on disk), V (float32 on disk), status, rows,
    acceptance, predicted (float32 on disk), times; window = last processing step kept (real-scene proxy window)."""
    order = run.fman['processing_order']
    if window is not None: order = order[:window+1]
    t = np.asarray(run.fman['timestamps_seconds'], float)[order]
    g = lambda n, dt=float: np.asarray(run.mm(n)[:, idx], dt)[order]
    return {'order': order, 't': t, 'X': g('posterior_xyz'), 'V': g('velocity'), 'V32': np.asarray(run.mm('velocity')[:, idx])[order],
            'st': g('point_status', np.int64), 'nrows': g('valid_cameras', np.int64), 'acc': g('update_acceptance', np.int64), 'Xpred': g('predicted_xyz')}


def first(mask):
    """mask [S,n] over processing steps 1..K-1 -> first step index (1-based processing step) or -1."""
    any_ = mask.any(0)
    return np.where(any_, mask.argmax(0)+1, -1)


def events(T, xgt):
    """Events of the spec (section 1) for every point, in processing steps (0 = canonical). xgt [n,3] or None (real).
    Exact mean dynamics x-_k = x+_{k-1} + dt_k v+_{k-1}; v is stored float32, so x-_k recomputed from disk carries a
    rounding of |dt| |v| eps32: the change thresholds are that bound plus 4 ulp of the position (not a tuning)."""
    X, V, V32, t = T['X'], T['V'], T['V32'], T['t']
    dt = np.diff(t)
    xm = X[:-1]+dt[:, None, None]*V[:-1]
    dX, dV = X[1:]-xm, V[1:]-V[:-1]
    eps = 4*EPS64*np.maximum(1., np.abs(X[1:]).max(-1))+np.abs(dt)[:, None]*np.linalg.norm(V[:-1], axis=-1)*EPS32
    st, nrows, acc = T['st'][1:], T['nrows'][1:], T['acc'][1:]
    e = {'first_accepted_correction': first((st == 1) & (np.linalg.norm(dX, axis=-1) > eps)),
         'first_velocity_change': first((V32[1:] != V32[:-1]).any(-1))}
    ref = X[0] if xgt is None else xgt
    errp, errm = np.linalg.norm(X[1:]-ref, axis=-1), np.linalg.norm(xm-ref, axis=-1)
    e['first_error_increase'] = first(errp > errm+eps)
    e['first_departure_over_10cm'] = first(errp > REPORTING_THRESHOLD_M)
    had = np.maximum.accumulate(nrows > 0, axis=0)
    lost = (nrows == 0) & np.vstack([np.zeros((1, nrows.shape[1]), bool), had[:-1]])
    e['first_observation_loss'] = first(lost)
    S = len(t)-1
    rec = np.full(X.shape[1], -1)
    fl = e['first_observation_loss']
    for i in np.flatnonzero(fl > 0):
        after = np.flatnonzero(nrows[fl[i]:, i] > 0)              # nrows[s-1] is processing step s: index fl[i] is step fl[i]+1
        if len(after): rec[i] = fl[i]+1+after[0]
    e['first_recovery_after_loss'] = rec
    tT = t[-1]
    pos = dX.sum(0); vel = ((tT-t[1:])[:, None, None]*dV).sum(0)
    net = X[-1]-X[0]
    closure = net-pos-vel
    nn = np.linalg.norm(net, axis=-1)
    dh = net/np.where(nn > 0, nn, 1.)[:, None]
    per = ((dX+(tT-t[1:])[:, None, None]*dV)*dh[None]).sum(-1)          # [S,n] projected contribution of each step
    j1 = e['first_velocity_change']
    c1 = np.where(j1 > 0, ((tT-t[np.maximum(j1, 1)])[:, None]*dV[np.maximum(j1, 1)-1, np.arange(X.shape[1])]*dh).sum(-1), 0.)
    att = {'net_m': nn, 'position_corrections_projected_m': (pos*dh).sum(-1), 'velocity_injections_projected_m': (vel*dh).sum(-1),
           'first_velocity_event_projected_m': c1, 'closure_max_abs_m': np.abs(closure).max(-1), 'dominant_step': per.argmax(0)+1,
           'position_corrections_vector': pos, 'velocity_injections_vector': vel, 'net_vector': net}
    return e, att, {'errp': errp, 'errm': errm, 'dX': dX, 'dV': dV, 'xm': xm, 'eps': eps, 'S': S}


# ------------------------------------------------------------------------------------------------------- static basis
def gt_basis(scene, x0):
    from scipy.spatial import cKDTree
    from tools.evaluate_projective_gt import gt_joint, labelled_endpoints
    kind, axis, point, declared = gt_joint(scene)
    vals = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True); q = np.asarray(vals[vals.dtype.names[1]], float)
    (xg, lg), _, moving, note = labelled_endpoints(scene, kind, axis, point, q[-1]-q[0])
    h = float(np.median(cKDTree(xg).query(xg, k=2)[0][:, 1]))
    ds = cKDTree(xg[lg != moving]).query(x0)[0]; dm = cKDTree(xg[lg == moving]).query(x0)[0]
    static = (ds <= h) & (dm > 2*h)
    moving_g = (dm <= h) & (ds > 2*h)
    return {'kind': 'GT_PART_LABELS', 'static': static, 'moving_grounded': moving_g, 'nearest_label_moving': dm < ds, 'd_static': ds, 'd_moving': dm,
            'meta': {'joint': kind, 'declared_joint': declared, 'axis': axis.tolist(), 'q_range': float(np.ptp(q)), 'moving_label': int(moving), 'label_note': note,
                     'gt_cloud_points': int(len(xg)), 'gt_cloud_spacing_h_m': h,
                     'rule': 'GROUNDED STATIC: nearest labelled GT point static within h and no moving-labelled GT point within 2h (h = median NN spacing of '
                             'the labelled cloud); x_GT,k = x0 (identity transform of the static part)',
                     'counts': {'grounded_static': int(static.sum()), 'grounded_moving': int(moving_g.sum()), 'ungrounded': int((~static & ~moving_g).sum())}},
            'q': q, 'axis': axis, 'point': point, 'joint': kind}


def proxy_basis(run, x0, window):
    """Real scenes: DEPTH-CONSTANT proxy over processing steps 0..window (module docstring). Nearest pixel of x0."""
    data, cams = run.data, run.cams()
    order = run.fman['processing_order'][:window+1]
    C, n = len(cams), len(x0)
    nsig = float(run.policy['visibility_n_sigma'])
    rgb_sd = np.sqrt(run.noise.rgb_var)
    pol = run.policy
    pix, ok, D0, I0 = [], np.zeros((C, n), bool), [], []
    for c, cam in enumerate(cams):
        Xc = x0@cam.R.T+cam.t
        Z = Xc[:, 2]; u = cam.fx*Xc[:, 0]/Z+cam.cx; v = cam.fy*Xc[:, 1]/Z+cam.cy
        pu, pv = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        ins = (Z > cam.near) & (pu >= 0) & (pu < cam.W) & (pv >= 0) & (pv < cam.H)
        pu, pv = np.clip(pu, 0, cam.W-1), np.clip(pv, 0, cam.H-1)
        o = data.observation(order[0], c)
        d0 = np.asarray(o.depth, float)[pv, pu]; i0 = np.asarray(o.rgb, float)[pv, pu]
        val = np.asarray(o.mask, bool)[pv, pu] & np.asarray(o.raw_valid, bool)[pv, pu]
        allow = pol['same_surface_n_sigma']*math.sqrt(2.)*np.sqrt(run.noise.sigma2(d0)) + d0*math.tan(math.radians(pol['same_surface_max_incidence_deg']))*math.sqrt(2.)/min(cam.fx, cam.fy)
        ok[c] = ins & val & (np.abs(Z-d0) <= allow)
        pix.append((pu, pv)); D0.append(d0); I0.append(i0)
    okc = ok.copy()
    for k in order[1:]:
        for c in range(C):
            o = data.observation(k, c); pu, pv = pix[c]
            d = np.asarray(o.depth, float)[pv, pu]; im = np.asarray(o.rgb, float)[pv, pu]
            val = np.asarray(o.mask, bool)[pv, pu] & np.asarray(o.raw_valid, bool)[pv, pu]
            ok[c] &= val & (np.abs(d-D0[c]) <= nsig*math.sqrt(2.)*np.sqrt(run.noise.sigma2(D0[c])))
            okc[c] &= ok[c] & (np.abs(im-I0[c]) <= nsig*math.sqrt(2.)*rgb_sd).all(-1)
    cnt = ok.sum(0)
    static = cnt >= 2
    return {'kind': 'DEPTH_CONSTANT_PROXY_NOT_GT', 'static': static, 'moving_grounded': np.zeros(n, bool), 'cameras_constant': cnt,
            'meta': {'rule': f'nearest pixel of x0 valid foreground in >= 2 cameras with |D_k - D_0| <= n sqrt(2) sigma_D in EVERY frame of processing steps '
                             f'0..{window} (frames {order[0]}..{order[-1]}, {float(run.fman["timestamps_seconds"][order[-1]]-run.fman["timestamps_seconds"][order[0]]):.2f} s; '
                             'n = visibility_n_sigma, sigma_D = the run\'s measured temporal sensor sigma); frame 0 requires |Z - D_0| within the same-surface '
                             'allowance. PROXY, NOT GROUND TRUTH: a surface sliding along itself, or a point whose pixels are re-covered by an equally deep '
                             'surface, passes. The audit of real scenes ends at the window.', 'n_sigma': nsig, 'window_steps': window,
                     'counts': {'proxy_static': int(static.sum()), 'one_camera_constant': int((cnt == 1).sum()), 'none': int((cnt == 0).sum()),
                                'also_colour_constant_in_2_cameras_(not_required)': int((okc.sum(0) >= 2).sum())}}}


def fixture_basis(x0):
    n = len(x0)
    return {'kind': 'STATIC_BY_CONSTRUCTION', 'static': np.ones(n, bool), 'moving_grounded': np.zeros(n, bool),
            'meta': {'rule': "two_plate_scene('static'): no part moves; x_GT = x0 (ray-cast surface sample)", 'counts': {'grounded_static': n}}}


# -------------------------------------------------------------------------------------------------------------- select
def q_(x, qs=(.5, .9, .99)):
    x = np.asarray(x, float)
    return {f'p{int(q*100)}': float(np.quantile(x, q)) for q in qs} if x.size else None


def baseline_numbers(run, T, ev, att, aux, basis, idx, gt):
    st_all = T['st'][1:]
    errT = aux['errp'][-1]
    speed = np.linalg.norm(T['V'][-1], axis=-1)
    acc = T['acc'][1:]
    codes = {O_ACC[c]: int((acc == c).sum()) for c in range(5) if (acc == c).any()}
    drifting = errT > REPORTING_THRESHOLD_M
    out = {'static_points': int(len(idx)), 'static_basis': basis['kind'],
           'final_error_m' if gt else 'final_drift_from_canonical_m': {**q_(errT), 'max': float(errT.max()), 'over_10cm_fraction': float(drifting.mean())},
           'final_speed_m_s (static truth: 0)': {**q_(speed), 'max': float(speed.max())},
           'drift_from_own_canonical_m_final': {**q_(np.linalg.norm(T['X'][-1]-T['X'][0], axis=-1)), 'note': 'identical to the GT error when x_GT = x0' if gt else 'no GT'},
           'no_valid_rows_point_frame_fraction': float((T['nrows'][1:] == 0).mean()),
           'points_never_accepted_fraction': float((~(st_all == 1).any(0)).mean()),
           'acceptance_counts_point_frames': codes,
           'status_json': json.loads((run.free/'status.json').read_text())}
    S = aux['S']
    hist = lambda a: {str(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    out['event_step_histograms'] = {k: hist(v) for k, v in ev.items()}
    if drifting.any():
        n_ = att['net_m'][drifting]
        out['drifting_points_attribution'] = {
            'count': int(drifting.sum()),
            'velocity_injection_share_of_net_projected': q_(att['velocity_injections_projected_m'][drifting]/n_, (.1, .5, .9)),
            'position_correction_share_of_net_projected': q_(att['position_corrections_projected_m'][drifting]/n_, (.1, .5, .9)),
            'first_velocity_event_share_of_net_projected': q_(att['first_velocity_event_projected_m'][drifting]/n_, (.1, .5, .9)),
            'closure_max_abs_m': float(att['closure_max_abs_m'].max()),
            'note': 'projections on the net drift direction; the two shares sum to 1 exactly, each can exceed 1 or be negative (cancellation)'}
        # after the >10 cm departure: why is the point not pulled back?  acceptance mix of the later frames
        e3 = ev['first_departure_over_10cm']
        mix = np.zeros(5, np.int64)
        for i in np.flatnonzero(drifting & (e3 > 0)):
            mix += np.bincount(acc[e3[i]:, i], minlength=5)
        out['after_departure_acceptance_mix_point_frames'] = {O_ACC[c]: int(mix[c]) for c in range(5)}
    # recovery: first frame with rows again after an observation gap
    rec = ev['first_recovery_after_loss']; okr = rec > 0
    if okr.any():
        i = np.flatnonzero(okr); s = rec[i]
        code = acc[s-1, i]; better = aux['errp'][s-1, i] < aux['errm'][s-1, i]
        out['first_recovery_after_loss'] = {'points': int(len(i)), 'acceptance': {O_ACC[c]: int((code == c).sum()) for c in range(5) if (code == c).any()},
                                            'accepted_and_error_decreased': int(((code == 1) | (code == 2))[better].sum()),
                                            'accepted_and_error_increased': int(((code == 1) | (code == 2))[~better].sum())}
    return out


O_ACC = {0: 'ANCHOR_OR_NOT_PROCESSED', 1: 'ACCEPTED_FINAL_ITERATE', 2: 'ACCEPTED_EARLIER_CONSISTENT_ITERATE', 3: 'REJECTED_INCONSISTENT', 4: 'NO_VALID_ROWS'}


def cmd_select(a):
    run = Run(a.baseline, 'forward')
    out = run.root/'static_audit'; out.mkdir(exist_ok=True)
    if (out/'selection.json').exists(): raise SystemExit(f'{out}/selection.json exists: never overwritten')
    x0 = run.can['canonical_xyz']; n = len(x0)
    t0 = time.time()
    window = a.window if a.kind == 'real' else None
    if a.kind == 'real' and window is None: raise SystemExit('--kind real needs --window (the proxy holds only over an early window)')
    basis = gt_basis(Path(a.scene), x0) if a.kind == 'synthetic' else fixture_basis(x0) if a.kind == 'fixture' else proxy_basis(run, x0, window)
    gt = a.kind != 'real'
    idx = np.flatnonzero(basis['static'])
    if len(idx) == 0: raise SystemExit('no static point with a grounded basis: nothing to audit')
    T = trajectories(run, idx, window)
    ev, att, aux = events(T, x0[idx] if gt else None)
    base = baseline_numbers(run, T, ev, att, aux, basis, idx, gt)
    errT, emax = aux['errp'][-1], aux['errp'].max(0)
    src = run.can['original_source_ids'][idx]
    src_z = np.array([np.linalg.norm(x0[i]-O.Cam(run.data.cameras[int(s[1])]).centre()) for i, s in zip(idx, src)])
    meanrows = T['nrows'][1:6].mean(0)
    nis1, dof1 = np.asarray(run.mm('nis')[T['order'][1], idx], float), np.asarray(run.mm('nis_dof')[T['order'][1], idx], float)
    chosen, why = [], {}
    def add(i, reason):
        if i not in why: chosen.append(i); why[i] = reason
    for i in np.argsort(-errT, kind='stable')[:a.worst]: add(int(i), 'worst_final_error' if gt else 'worst_final_drift_proxy_static')
    stable = np.flatnonzero(emax <= STABLE_FRACTION*REPORTING_THRESHOLD_M)
    feat = np.c_[meanrows, src_z]; sc = feat.std(0)+1e-12
    for i in chosen[:a.controls]:
        if not len(stable): break
        cand = [j for j in stable[np.argsort(np.linalg.norm((feat[stable]-feat[i])/sc, axis=1), kind='stable')] if j not in why][:1]
        if cand: add(int(cand[0]), f'stable_control_matched_to_{int(idx[i])}')
    never = np.flatnonzero(~(T['st'][1:] == 1).any(0))
    for i in never[:2]: add(int(i), 'control_never_accepted_any_rgbd_update')
    nisd = np.where(dof1 > 0, nis1/np.maximum(dof1, 1), -np.inf)
    for i in [j for j in np.argsort(-nisd, kind='stable') if j not in why and nisd[j] > -np.inf][:a.innovation]: add(int(i), 'high_initial_innovation_frame1_nis_per_dof')
    order = T['order']; S = aux['S']
    plan = set(order[1:min(6, S+1)])
    pts = []
    for i in chosen:
        e = {k: int(v[i]) for k, v in ev.items()}
        dom = int(att['dominant_step'][i])
        for s in [e['first_accepted_correction'], e['first_error_increase'], e['first_departure_over_10cm'], e['first_velocity_change'], dom,
                  e['first_observation_loss'], e['first_recovery_after_loss']]:
            if s > 0: plan |= {order[j] for j in range(max(1, s-2), min(S, s+3)+1)}
        f, c, pix = (int(v) for v in run.can['original_source_ids'][idx[i]])
        W = run.data.cameras[c].width
        pts.append({'point_id': int(idx[i]), 'selection_reason': why[i], 'source_frame_camera_pixel_uv': [f, c, pix % W, pix//W],
                    'original_source_id_row': run.can['original_source_ids'][idx[i]].tolist(), 'canonical_xyz': x0[idx[i]].tolist(),
                    'static_evidence': {k: (float(basis[k][idx[i]]) if k in basis else None) for k in ('d_static', 'd_moving', 'cameras_constant')},
                    'final_error_m' if gt else 'final_drift_m': float(errT[i]), 'max_error_m': float(emax[i]), 'events_processing_step': e,
                    'events_frame': {k: (order[v] if v > 0 else None) for k, v in e.items()}, 'dominant_step': dom, 'dominant_frame': order[dom],
                    'mean_valid_rows_frames_1_5': float(meanrows[i]), 'source_camera_distance_m': float(src_z[i]), 'frame1_nis_per_dof': float(nisd[i]) if np.isfinite(nisd[i]) else None,
                    'attribution': {k: (v[i].tolist() if np.ndim(v) > 1 else float(v[i])) for k, v in att.items()}})
    sel = {'baseline_run': str(run.root), 'scene': a.scene, 'kind': a.kind, 'window': window, 'static_basis': {kk: v for kk, v in basis['meta'].items()},
           'selected_ids_sha256': run.fman['selected_ids_sha256'], 'points_total': n, 'contract': contract(run), 'baseline': base,
           'audit_points': pts, 'trace_frames': sorted(int(k) for k in plan), 'trace_point_ids': [p['point_id'] for p in pts], 'seconds': time.time()-t0,
           'note': 'audit samples, NOT a tuning subset; ground truth / proxy used here only, never by the filter'}
    np.savez_compressed(out/'static_basis.npz', static=basis['static'], moving_grounded=basis['moving_grounded'],
                        **{k: basis[k] for k in ('d_static', 'd_moving', 'cameras_constant') if k in basis})
    write_json(out/'selection.json', sel)
    print(json.dumps({'baseline': base, 'trace_point_ids': sel['trace_point_ids'], 'trace_frames': sel['trace_frames']}, default=str)[:4000])
    print('--debug-point-ids', ','.join(map(str, sel['trace_point_ids'])))
    print('--debug-frames', ','.join(map(str, sel['trace_frames'])))


# ------------------------------------------------------------------------------------------------------------- analyze
def parity(traced, baseline):
    out = {}
    for f in sorted(Path(baseline, 'free_forward').glob('*.npy')):
        g = Path(traced, 'free_forward', f.name)
        if not g.exists(): out[f.name] = 'MISSING_IN_TRACED'; continue
        A, B = np.load(f, mmap_mode='r'), np.load(g, mmap_mode='r')
        same = A.shape == B.shape and A.dtype == B.dtype and all(np.array_equal(A[k], B[k], equal_nan=A.dtype.kind == 'f') for k in range(len(A)))
        out[f.name] = 'IDENTICAL' if same else 'DIFFERENT'
    return out


def camera_rows(prob, x, rows, probe):
    """Spec section 2 table, per camera at state x: projection, 2x2 support, reasons (fresh / frozen / used), frontmost
    owner, r, H, R, S, prior-linearized NIS and gate, and the per-channel information split g, I (RGB vs depth)."""
    out = []
    n = prob.policy['visibility_n_sigma']; thr = O.row_gate_threshold(n)
    for o, p in zip(rows, probe):
        s = o.get('sample', {})
        d = {'camera': o['camera'], 'u': o.get('u'), 'v': o.get('v'), 'Z': o.get('Z'), 'reason_oracle': O.NAMES[o['reason']],
             'visibility_fresh': p['visibility_fresh'], 'visibility_frozen': p['visibility_frozen'], 'reason_used_production': p['reason_used'],
             'used': p['reason_used'] == 'VALID', 'nearest_pixel_u_v': p['nearest_pixel_u_v'],
             'frontmost': p['pixel_occupants_front_to_back'][:2], 'pixel_occupants_total': p['pixel_occupants_total'],
             'pixel00_u_v': p['pixel00_u_v'], 'du_dv': p['du_dv'], 'neighbour_rgb': p['neighbour_rgb'], 'neighbour_depth_m': p['neighbour_depth_m'],
             'neighbour_mask': p['neighbour_mask'], 'neighbour_raw_valid': p['neighbour_raw_valid']}
        if 'r' in o:
            H, r, var = o['H'], o['r'], o['var']
            d.update(sampled_rgb=s['rgb'], sampled_depth_m=s['depth'], r=r, r_over_sigma=r/np.sqrt(var), H_x=H, R_diag=var, S=o['S'],
                     nu_prior_linearized=o['nu'], nis_prior_linearized=o['nis'], gate_threshold_chi2_4=thr, depth_front_test=float(o['nu'][3]/math.sqrt(o['S_zz'])),
                     g_rgb=H[:3].T@(r[:3]/var[:3]), g_depth=H[3]*r[3]/var[3], I_rgb=H[:3].T@(H[:3]/var[:3, None]), I_depth=np.outer(H[3], H[3])/var[3],
                     same_surface_spread_m=s.get('same_surface_spread_m'), same_surface_allowance_m=s.get('same_surface_allowance_m'))
        out.append(d)
    return out


def info_sum(rows, used_cams, channels=(1, 1, 1, 1)):
    ch = np.asarray(channels, float)
    I, g, q = np.zeros((3, 3)), np.zeros(3), 0.
    for o in rows:
        if o['camera'] not in used_cams or 'r' not in o: continue
        w = ch/o['var']; I += o['H'].T@(w[:, None]*o['H']); g += o['H'].T@(w*o['r']); q += float((w*o['r']**2).sum())
    return I, g, q


def chi2_stat(g, I):
    """g^T I^+ g on the range of I (dof = numerical rank): chi2_rank if the rows are unbiased with covariance R at the
    evaluation point (exact for a linear model; the reference distribution, not a gate)."""
    w, U = np.linalg.eigh(I)
    keep = w > w.max()*1e-12 if w.max() > 0 else np.zeros(3, bool)
    s = float(((U[:, keep].T@g)**2/w[keep]).sum())
    k = int(keep.sum())
    return {'statistic': s, 'dof': k, 'p_value': float(chi2.sf(s, k)) if k else None}


def static_gradient(prob, x, frozen, label):
    """E1: the image-only information gradient at a FIXED location x with v = 0 (no update applied). Two row sets:
    'admitted' = the production rules at this frame (frozen pixel assignment, prior-linearized gate with this frame's
    P- and dx = x - x-), 'sampled' = every camera whose 2x2 support is valid (gate ignored)."""
    dx = x-prob.s_minus[:3]
    res = {'location': label, 'x': x}
    rows = []
    for c, cam in enumerate(prob.cams):
        vis = frozen[c] if frozen[c] in (O.SELF_OCC, O.DUP) else O.geometric_visibility(cam, x)
        o = O.row(prob.imgs[c], cam, x, prob.a, prob.noise, prob.policy, prob.P_minus[:3, :3], dx) if vis == O.VALID else {'reason': vis}
        o['camera'] = c; rows.append(o)
    for name, cams in (('admitted', [o['camera'] for o in rows if o['reason'] == O.VALID]), ('sampled', [o['camera'] for o in rows if o['reason'] in SAMPLEABLE and 'r' in o])):
        I, g, q = info_sum(rows, cams)
        A = prob.Lam.copy(); A[:3, :3] += I
        d = np.linalg.solve(A, np.r_[g, 0, 0, 0])
        res[name] = {'cameras': cams, 'g': g, 'g_norm': float(np.linalg.norm(g)), 'I': I, 'chi2_data': q, 'g_statistic': chi2_stat(g, I),
                     'per_camera': {o['camera']: {'reason': O.NAMES[o['reason']], 'r_over_sigma': o['r']/np.sqrt(o['var']), 'g_rgb': o['H'][:3].T@(o['r'][:3]/o['var'][:3]),
                                                  'g_depth': o['H'][3]*o['r'][3]/o['var'][3]} for o in rows if o['camera'] in cams},
                     'implied_step_from_x_with_this_prior': {'delta_x_m': d[:3], 'delta_x_norm_m': float(np.linalg.norm(d[:3])), 'delta_v_m_s': d[3:],
                                                              'delta_v_norm_m_s': float(np.linalg.norm(d[3:]))}}
    res['note'] = ('implied step = (Lambda- + I)^-1 [g, 0]: the update a prior CENTRED at x with this frame\'s P- would publish from x (first Gauss-Newton '
                   'step); non-zero means the measurement alone moves a correctly predicted point')
    return res


def step_linearization(prob, s, delta, used):
    """E2 for one IEKF step s -> s + alpha delta on the SAME cameras: projected-pixel displacement (linear vs actual),
    cell / support changes, the R-whitened linearization error of h, and actual vs Gauss-Newton objective change for
    alpha in ALPHAS (diagnostic fractions; nothing is published). Gain ratio rho = actual / predicted decrease."""
    x, dx = s[:3], delta[:3]
    rows0 = {o['camera']: o for o in prob.rows(x) if o['camera'] in used and 'r' in o}
    out = {'cameras': {}, 'alphas': []}
    for al in ALPHAS:
        xa = x+al*dx
        dact = dgn = 0.; lost = []
        lin_err_max = 0.
        for c, o in rows0.items():
            o1 = O.row(prob.imgs[c], prob.cams[c], xa, prob.a, prob.noise, prob.policy, prob.P_minus[:3, :3], xa-prob.s_minus[:3])
            if 'r' not in o1: lost.append((c, O.NAMES[o1['reason']])); continue
            rl = o['r']-al*(o['H']@dx)
            dact += float((o1['r']**2/o1['var']).sum()-(o['r']**2/o['var']).sum()); dgn += float((rl**2/o['var']).sum()-(o['r']**2/o['var']).sum())
            e = ((o['r']-o1['r'])-al*(o['H']@dx))/np.sqrt(o['var'])
            lin_err_max = max(lin_err_max, float(np.abs(e).max()))
            if al == 1.:
                du_lin = o['J_u']@dx
                out['cameras'][c] = {'du_dv_linear_px': du_lin, 'du_dv_actual_px': [o1['u']-o['u'], o1['v']-o['v']],
                                     'pixel_displacement_px': float(math.hypot(o1['u']-o['u'], o1['v']-o['v'])),
                                     'projection_linearization_error_px': float(math.hypot(o1['u']-o['u']-du_lin[0], o1['v']-o['v']-du_lin[1])),
                                     'cell_changed': tuple(o1['sample']['pixel00']) != tuple(o['sample']['pixel00']),
                                     'reason_at_new_state': O.NAMES[o1['reason']], 'support_mask_all_white_at_new_state': bool(np.all(o1['sample']['neighbour_mask'])),
                                     'h_linearization_error_whitened': e, 'h_linearization_error_whitened_rgb_norm': float(np.linalg.norm(e[:3])),
                                     'h_linearization_error_whitened_depth': float(e[3]), 'r_before': o['r'], 'r_after': o1['r'], 'r_gauss_newton': rl}
        pa, p0 = prob.prior_cost(s+al*delta), prob.prior_cost(s)
        tot_act, tot_gn = pa-p0+dact, pa-p0+dgn
        out['alphas'].append({'alpha': al, 'data_change_actual': dact, 'data_change_gauss_newton': dgn, 'prior_change': pa-p0,
                              'objective_change_actual': tot_act, 'objective_change_gauss_newton': tot_gn,
                              'gain_ratio_rho': tot_act/tot_gn if tot_gn != 0 else None, 'rows_lost': lost, 'max_abs_whitened_linearization_error': lin_err_max})
    a1 = out['alphas'][0]
    out['classification'] = {'any_cell_crossing': any(v['cell_changed'] for v in out['cameras'].values()),
                             'any_support_lost': bool(a1['rows_lost']) or any(v['reason_at_new_state'] not in ('VALID', 'MEASURED_SURFACE_IN_FRONT', 'INNOVATION_INCONSISTENT') for v in out['cameras'].values()),
                             'full_step_increases_same_row_objective': a1['objective_change_actual'] > 0,
                             'full_step_rho': a1['gain_ratio_rho'], 'max_pixel_displacement_px': max([v['pixel_displacement_px'] for v in out['cameras'].values()] or [0.])}
    return out


def fd_check(prob, x, used):
    out = {}
    for c in used:
        r = O.finite_difference_H(prob, x, c, eps=1e-7)
        if r is not None: out[c] = {'rel_error': r['rel_error'], 'max_abs_error': r['max_abs_error'], 'crosses_pixel_cell': r['crosses_pixel_cell']}
    return out


def comparable_objective(prob, s_a, s_b, cams_a, cams_b):
    """E3: objective at two states on the COMMON admitted cameras (each re-sampled at both states), plus the prior."""
    common = sorted(set(cams_a) & set(cams_b))
    ca, cb = prob.cost_on(s_a, common), prob.cost_on(s_b, common)
    Ja = prob.prior_cost(s_a)+ca['rgb']+ca['depth']; Jb = prob.prior_cost(s_b)+cb['rgb']+cb['depth']
    return {'common_cameras': common, 'lost_at_b': sorted(set(cams_a)-set(cams_b)), 'gained_at_b': sorted(set(cams_b)-set(cams_a)),
            'a': {'prior': prob.prior_cost(s_a), 'rgb': ca['rgb'], 'depth': ca['depth'], 'total': Ja, 'per_camera': ca['per_camera'], 'unsampleable': ca['lost_rows']},
            'b': {'prior': prob.prior_cost(s_b), 'rgb': cb['rgb'], 'depth': cb['depth'], 'total': Jb, 'per_camera': cb['per_camera'], 'unsampleable': cb['lost_rows']},
            'b_worse_than_a': bool(Jb > Ja), 'data_only_b_worse_than_a': bool(cb['rgb']+cb['depth'] > ca['rgb']+ca['depth'])}


def coupling(pre, s_minus, s_plus, dt_next, xgt):
    """E4: L = P_vx P_xx^-1 of the fixed prior, Delta_v_expected = L Delta_x, decomposition of P_vx- and P_xx-."""
    Pm, Pp, Q, dt = np.asarray(pre['P_minus']), np.asarray(pre['P_previous']), np.asarray(pre['Q']), pre['dt_s']
    Pxx, Pvx = Pm[:3, :3], Pm[3:, :3]
    L = Pvx@np.linalg.inv(Pxx)
    dx, dv = s_plus[:3]-s_minus[:3], s_plus[3:]-s_minus[3:]
    dve = L@dx
    sv = np.linalg.svd(L, compute_uv=False)
    tr = lambda M: float(np.trace(M))
    out = {'L_per_s': L, 'L_singular_values_per_s': sv, 'dtL_singular_values': np.abs(dt)*sv, 'dt_s': dt,
           'delta_x_published_m': dx, 'delta_x_norm_mm': 1e3*float(np.linalg.norm(dx)), 'delta_v_published_m_s': dv, 'delta_v_norm_m_s': float(np.linalg.norm(dv)),
           'delta_v_expected_L_dx': dve, 'identity_abs_error_m_s': float(np.abs(dv-dve).max()),
           'identity_rel_error': float(np.linalg.norm(dv-dve)/max(np.linalg.norm(dve), 1e-300)) if np.linalg.norm(dx) > 0 else None,
           'cos_dx_dv': float(dx@dv/(np.linalg.norm(dx)*np.linalg.norm(dv))) if np.linalg.norm(dx) > 0 and np.linalg.norm(dv) > 0 else None,
           'v_minus': s_minus[3:], 'v_plus': s_plus[3:],
           'P_vx_minus_terms_trace': {'P_vx_previous': tr(Pp[3:, :3]), 'dt_P_vv_previous': tr(dt*Pp[3:, 3:]), 'Q_vx': tr(Q[3:, :3]), 'total': tr(Pvx)},
           'P_xx_minus_terms_trace_m2': {'P_xx_previous': tr(Pp[:3, :3]), 'dt_(P_xv+P_vx)_previous': tr(dt*(Pp[:3, 3:]+Pp[3:, :3])),
                                         'dt2_P_vv_previous': tr(dt*dt*Pp[3:, 3:]), 'Q_xx': tr(Q[:3, :3]), 'total': tr(Pxx)}}
    if dt_next is not None:
        nxt_act, nxt_cf = s_plus[:3]+dt_next*s_plus[3:], s_plus[:3]+dt_next*s_minus[3:]
        out['next_prediction'] = {'dt_next_s': dt_next, 'displacement_dt_v_plus_m': dt_next*s_plus[3:], 'displacement_norm_mm': 1e3*float(np.linalg.norm(dt_next*s_plus[3:])),
                                  'x_next_actual_prediction': nxt_act, 'x_next_if_v_not_injected': nxt_cf}
        if xgt is not None:
            out['next_prediction'].update(gt_error_actual_mm=1e3*float(np.linalg.norm(nxt_act-xgt)), gt_error_if_v_not_injected_mm=1e3*float(np.linalg.norm(nxt_cf-xgt)),
                                          gt_error_at_x_plus_mm=1e3*float(np.linalg.norm(s_plus[:3]-xgt)))
    return out


def invariants(pid, fin, cams_used, n_sigma):
    """E6 at the accepted linearization: (a) a used row whose pixel's front-most transported point is another point
    statistically in front (SELF_OCCLUDED by the projective_visibility rule) = violation; (b) a used row where another
    point shares the pixel within the uncertainty (DUPLICATE, allowed by the per-frame frozen assignment, reported);
    (c) a used row whose own residual says the measured surface is in front by > n sigma (the gate tests nu, not r)."""
    probe = fin.get('cameras_fallback_state') or fin['cameras_final_iterate']
    out = {'violations_hidden_point_used': [], 'duplicates_used': [], 'measured_surface_in_front_by_own_residual': []}
    pid_rows = {r['camera']: r for r in fin.get('rows', [])}
    for p in probe:
        c = p['camera']
        if c not in cams_used: continue
        occ = p['pixel_occupants_front_to_back']
        if occ and occ[0]['point'] != pid:
            f = occ[0]
            own = [o for o in occ if o['point'] == pid]
            z_own = own[0]['z'] if own else p['Z']
            sep = z_own-f['z']; allow = n_sigma*math.sqrt(p['sigma_z2_prior']+f['sigma_z2_prior'])
            rec = {'camera': c, 'front_point': f['point'], 'front_z': f['z'], 'own_z': z_own, 'separation_m': sep, 'allowance_m': allow}
            (out['violations_hidden_point_used'] if sep > allow else out['duplicates_used']).append(rec)
        r = pid_rows.get(c)
        if r is not None and r['r'][3] > n_sigma*math.sqrt(r['S_zz']):
            out['measured_surface_in_front_by_own_residual'].append({'camera': c, 'r_D_m': r['r'][3], 'sqrt_S_zz_m': math.sqrt(r['S_zz'])})
    return out


def counterfactuals(prob, xgt, src_cam):
    """Section 6: ONE linearized update at s- (production frozen assignment, prior-linearized gate at dx = 0) under row
    subsets. Not rollouts, not policies; correlated rows / nonlinear trajectories are not represented."""
    s = prob.s_minus
    rows = prob.rows(s[:3])
    used = [o['camera'] for o in prob.used(rows)]
    def one(cams, ch=(1, 1, 1, 1)):
        I, g, q = info_sum(rows, cams, ch)
        A = prob.Lam.copy(); A[:3, :3] += I
        d = np.linalg.solve(A, np.r_[g, 0, 0, 0])
        r = {'cameras': cams, 'channels': list(ch), 'delta_x_m': d[:3], 'delta_x_norm_mm': 1e3*float(np.linalg.norm(d[:3])), 'delta_v_m_s': d[3:],
             'delta_v_norm_m_s': float(np.linalg.norm(d[3:])), 'data_chi2_at_s_minus': q}
        if xgt is not None:
            r['gt_error_change_mm'] = 1e3*float(np.linalg.norm(s[:3]+d[:3]-xgt)-np.linalg.norm(s[:3]-xgt))
        return r, d
    out = {'cameras_admitted_at_s_minus': used}
    base, d_all = one(used)
    out['all_rows'] = base; out['rgb_only'] = one(used, (1, 1, 1, 0))[0]; out['depth_only'] = one(used, (0, 0, 0, 1))[0]
    for c in used: out[f'leave_out_camera_{c}'] = one([u for u in used if u != c])[0]
    if src_cam in used:
        out['source_camera_only'] = one([src_cam])[0]; out['other_cameras_only'] = one([u for u in used if u != src_cam])[0]
    out['prediction_only'] = one([])[0]
    if used:
        A = prob.Lam.copy(); A[:3, :3] += info_sum(rows, used)[0]
        att = prob.attribution(s, rows, A)
        n2 = float(d_all[:3]@d_all[:3])
        sh = {k: float(np.asarray(v['delta'])[:3]@d_all[:3]/n2) for k, v in att.items()} if n2 > 0 else {}
        out['row_shares_of_delta_x'] = dict(sorted(sh.items(), key=lambda kv: -abs(kv[1])))
        out['dominant_row'] = max(sh, key=lambda k: abs(sh[k])) if sh else None
    out['limitation'] = 'single linearized step at s-; correlated observations and the nonlinear iterate trajectory are not represented'
    return out


def frame_analysis(run, k, pid, tr, imgs, cams, xgt, dt_next, src_cam):
    pre = prediction_section(run, tr, pid)
    fin = tr['final'][str(pid)]
    prob = problem_from_trace(run, k, pid, tr, imgs, cams)
    frozen = frozen_codes(tr['iterations'][0]['points'][str(pid)]['cameras'])
    s_minus, s_plus = np.array(fin['s_minus']), np.array(fin['s_plus'])
    n_sigma = float(run.policy['visibility_n_sigma'])
    its = []
    trace_its = [t['points'][str(pid)] for t in tr['iterations']]
    for j, t in enumerate(trace_its):
        if j > 0 and not t['active']: break
        s = np.array(t['state']); delta = np.array(t['delta'])
        used = [r['camera'] for r in t['rows'] if r['reason'] == 'VALID']
        Ixx, gx = np.zeros((3, 3)), np.zeros(3)
        for r in t['rows']:
            if r['reason'] != 'VALID': continue
            H, rr, w = np.array(r['H_x']), np.array(r['r']), 1/np.array(r['R_diag'])
            Ixx += H.T@(w[:, None]*H); gx += H.T@(w*rr)
        A = prob.Lam.copy(); A[:3, :3] += Ixx; b = -prob.Lam@(s-prob.s_minus); b[:3] += gx
        d = np.linalg.solve(A, b)
        rows = prob.rows(s[:3])
        it = {'iteration': j, 'state': s, 'state_minus_prior': s-s_minus, 'cameras_used': used, 'nrows': t['nrows'],
              'posterior_cost_J': t['posterior_cost'], 'cost_threshold': t['cost_threshold'],
              'passes_consistency': bool(j >= 1 and t['nrows'] > 0 and t['posterior_cost'] <= t['cost_threshold']),
              'delta': delta, 'delta_x_norm_mm': 1e3*float(np.linalg.norm(delta[:3])), 'step_delta_A_delta': t['step_delta_A_delta'],
              'A': t['A'], 'b': t['b'], 'Ixx': t['Ixx'], 'gx': t['gx'],
              'oracle_algebra': {'A_rel': rel(A, t['A']), 'b_rel': rel(b, t['b']), 'delta_rel': rel(d, delta)},
              'oracle_resample_reason_agreement': all(O.NAMES[o['reason']] == p['reason_used'] or p['reason_used'] in ('SELF_OCCLUDED', 'DUPLICATE_PIXEL_SUPPRESSED', 'NOT_PROCESSED')
                                                      for o, p in zip(rows, t['cameras'])),
              'camera_rows': camera_rows(prob, s[:3], rows, t['cameras']) if j == 0 else None,
              'linearization': step_linearization(prob, s, delta, used) if used else None,
              'jacobian_fd_in_cell': fd_check(prob, s[:3], used) if j == 0 else None}
        its.append(it)
    n_active = len(its)
    acc = fin['acceptance']
    acc_iter = n_active if acc == 'ACCEPTED_FINAL_ITERATE' else fin['last_consistent_iteration'] if acc == 'ACCEPTED_EARLIER_CONSISTENT_ITERATE' else None
    res = prob.run(run.policy.get('update_acceptance', 'final_iterate'))
    used_final = [r['camera'] for r in fin.get('rows', [])]
    used0 = its[0]['cameras_used'] if its else []
    out = {'frame': k, 'prediction': {kk: pre[kk] for kk in ('t_previous_s', 't_s', 'dt_s', 's_previous', 'P_previous', 'F', 'Q', 's_minus', 'P_minus', 'P_xx', 'P_xv',
                                                             'P_vv', 'eigenvalues', 'condition_number', 'oracle_check')},
           'acceptance': acc, 'accepted_iterate': acc_iter, 'iterates_run': n_active, 'converged': fin['converged'], 'valid_rows_accepted': fin['valid_rows'],
           'posterior_cost': fin['posterior_cost'], 'posterior_cost_threshold': fin['cost_threshold'], 'nis_dof': 4*fin['valid_rows'],
           's_plus': s_plus, 'P_plus': fin['P_plus'], 'cameras_used_at_s_minus': used0, 'cameras_used_accepted': used_final,
           'oracle_replay': {'legacy_mixed_state_max_UNITLESS_INVALID': absd(res['s_plus'], s_plus), 's_plus_position_linf_m': absd(res['s_plus'][:3], np.asarray(s_plus)[:3]),
                             's_plus_velocity_linf_m_s': absd(res['s_plus'][3:], np.asarray(s_plus)[3:]), 'P_plus_rel': rel(res['P_plus'], fin['P_plus']),
                             'acceptance_code_oracle': O_ACC[res['acceptance_code']], 'acceptance_agrees': O_ACC[res['acceptance_code']] == acc},
           'iterates': its,
           'coupling': coupling(pre, s_minus, s_plus, dt_next, xgt)}
    if acc in ('ACCEPTED_FINAL_ITERATE', 'ACCEPTED_EARLIER_CONSISTENT_ITERATE'):
        out['objective_accepted_vs_prior_common_rows'] = comparable_objective(prob, s_minus, s_plus, used0, used_final)
        if acc_iter is not None and acc_iter >= 1 and acc_iter-1 < len(its):
            prev = its[acc_iter-1]
            out['objective_accepted_vs_previous_iterate_common_rows'] = comparable_objective(prob, prev['state'], s_plus, prev['cameras_used'], used_final)
        out['invariants'] = invariants(pid, fin, used_final, n_sigma)
        out['accepted_camera_rows'] = camera_rows(prob, s_plus[:3], prob.rows(s_plus[:3], final=True), fin.get('cameras_fallback_state') or fin['cameras_final_iterate'])
    if xgt is not None:
        out['gt'] = {'x_gt': xgt, 'error_prior_mm': 1e3*float(np.linalg.norm(s_minus[:3]-xgt)), 'error_posterior_mm': 1e3*float(np.linalg.norm(s_plus[:3]-xgt))}
    loc = xgt if xgt is not None else run.can['canonical_xyz'][pid]
    out['E1_static_location'] = static_gradient(prob, loc, frozen, 'x_GT' if xgt is not None else 'x0_canonical_proxy')
    out['E1_prior_location'] = {'g_production_iteration0': trace_its[0]['gx'], 'g_norm': float(np.linalg.norm(trace_its[0]['gx'])),
                                'g_statistic': chi2_stat(np.array(trace_its[0]['gx']), np.array(trace_its[0]['Ixx']))}
    mem = {}
    for c, cam in enumerate(cams):
        u, v, Z, _, _ = cam.project(s_plus[:3])
        pu, pv = round(u), round(v)
        mem[c] = bool(imgs[c].mask[pv, pu]) if (Z > cam.near and 0 <= pu < cam.W and 0 <= pv < cam.H) else None
    out['mask_membership_nearest_pixel_s_plus'] = mem
    if acc not in ('ACCEPTED_FINAL_ITERATE', 'ACCEPTED_EARLIER_CONSISTENT_ITERATE'):
        out['prediction_only_check'] = {'s_plus_equals_s_minus': bool(np.array_equal(fin['s_plus'], fin['s_minus'])),
                                        'P_plus_equals_P_minus': bool(np.array_equal(fin['P_plus'], fin['P_minus']))}
    out['_counterfactual_problem'] = prob
    out['_src_cam'] = src_cam
    return out


def timeline(run, T, i, pid, steps, xgt, traced):
    order = T['order']; t = T['t']
    rows = []
    cr = run.mm('camera_reason')
    for s in steps:
        if s < 1 or s >= len(order): continue
        k = order[s]
        xm = T['X'][s-1, i]+(t[s]-t[s-1])*T['V'][s-1, i]
        d = {'frame': k, 'time_s': float(t[s]), 'x_gt': None if xgt is None else xgt, 'x_minus': xm, 'x_plus': T['X'][s, i], 'v_minus': T['V'][s-1, i], 'v_plus': T['V'][s, i],
             'status': {1: 'MEASURED', 2: 'PREDICTION_ONLY'}.get(int(T['st'][s, i]), int(T['st'][s, i])), 'acceptance': O_ACC[int(T['acc'][s, i])],
             'accepted_rows': int(T['nrows'][s, i]), 'reason_by_camera': [O.NAMES[int(r)] for r in cr[k, :, pid]],
             'error_mm' if xgt is not None else 'drift_from_x0_mm': 1e3*float(np.linalg.norm(T['X'][s, i]-(xgt if xgt is not None else T['X'][0, i])))}
        if k in traced: d.update(traced[k])
        rows.append(d)
    return rows


def cmd_analyze(a):
    run, base = Run(a.traced, 'forward'), Run(a.baseline, 'forward')
    sel = json.loads((base.root/'static_audit'/'selection.json').read_text())
    out = run.root/'static_audit'; out.mkdir(exist_ok=True)
    if (out/'analysis.json').exists(): raise SystemExit(f'{out}/analysis.json exists: never overwritten')
    if run.fman['selected_ids_sha256'] != sel['selected_ids_sha256']: raise SystemExit('point ids of the traced run differ from the baseline selection')
    t0 = time.time()
    par = parity(run.root, base.root)
    kind = sel['kind']; gt = kind != 'real'
    ids = [p['point_id'] for p in sel['audit_points']]
    T = trajectories(run, np.asarray(ids), sel.get('window'))
    x0 = run.can['canonical_xyz']
    ev, att, aux = events(T, x0[ids] if gt else None)
    cams = run.cams()
    order = T['order']; times = T['t']
    traced_frames = sorted(int(p.stem.split('_')[-1]) for p in (run.free/'numeric').glob('trace_frame_*.json'))
    results = []
    lines = []
    for ii, p in enumerate(sel['audit_points']):
        pid = p['point_id']; xgt = x0[pid] if gt else None
        src_cam = p['source_frame_camera_pixel_uv'][1]
        # canonical colour provenance: rgb0 is the source pixel's measured colour
        f0, c0, us, vs = p['source_frame_camera_pixel_uv']
        o0 = run.data.observation(order[0], c0)
        prov = {'rgb0': run.can['canonical_rgb'][pid], 'source_pixel_rgb': np.asarray(o0.rgb, float)[vs, us], 'source_pixel_depth_m': float(np.asarray(o0.depth, float)[vs, us]),
                'rgb0_equals_source_pixel': bool(np.allclose(run.can['canonical_rgb'][pid], np.asarray(o0.rgb, float)[vs, us], rtol=0, atol=1e-6)),
                'canonical_xyz': x0[pid], 'canonical_source_covariance_m2': run.can['canonical_source_covariance'][pid]}
        frame0 = endpoint_cross_camera(run, pid, cams)
        per_frame = {}
        for s in range(1, len(order)):
            k = order[s]
            if k not in traced_frames: continue
            tr = run.trace(k)
            if tr is None or str(pid) not in tr['final']: continue
            imgs = run.images(k)
            dt_next = float(times[s+1]-times[s]) if s+1 < len(order) else None
            fa = frame_analysis(run, k, pid, tr, imgs, cams, xgt, dt_next, src_cam)
            per_frame[k] = fa
        # counterfactuals at the first offending frame (first GT-error increase for GT scenes, else first velocity change)
        e = {kk: int(v[ii]) for kk, v in ev.items()}
        off_step = e['first_error_increase'] if gt and e['first_error_increase'] > 0 else e['first_velocity_change']
        dom_step = int(att['dominant_step'][ii])
        cfs = {}
        for name, s in (('first_offending', off_step), ('dominant_contribution', dom_step)):
            if s > 0 and order[s] in per_frame:
                fa = per_frame[order[s]]
                cfs[name] = {'frame': order[s], **counterfactuals(fa['_counterfactual_problem'], xgt, fa['_src_cam'])}
            elif s > 0:
                cfs[name] = {'frame': order[s], 'unavailable': 'frame not traced'}
        for fa in per_frame.values():
            fa.pop('_counterfactual_problem', None); fa.pop('_src_cam', None)
        short = {k: {'traced': True, 'acceptance_trace': fa['acceptance'], 'accepted_iterate': fa['accepted_iterate'],
                     'g_at_prior': fa['E1_prior_location']['g_production_iteration0']} for k, fa in per_frame.items()}
        onset = e['first_error_increase'] if gt and e['first_error_increase'] > 0 else e['first_accepted_correction']
        tl = {'around_onset': timeline(run, T, ii, pid, range(onset-2, onset+4), xgt, short) if onset > 0 else [],
              'around_first_departure_over_10cm': timeline(run, T, ii, pid, range(e['first_departure_over_10cm']-2, e['first_departure_over_10cm']+4), xgt, short)
              if e['first_departure_over_10cm'] > 0 else []}
        # prediction-only frames over the whole trajectory (memmaps): v unchanged exactly, x = x + dt v within the float32 bound
        po = np.flatnonzero(T['st'][1:, ii] == 2)+1
        pchk = {'frames': int(len(po))}
        if len(po):
            dv = np.abs(T['V32'][po, ii]-T['V32'][po-1, ii]).max()
            dx = np.linalg.norm(aux['dX'][po-1, ii], axis=-1); epsb = aux['eps'][po-1, ii]
            pchk.update(velocity_unchanged_exactly=bool(dv == 0), position_equals_ballistic_within_bound=bool((dx <= epsb).all()), max_position_residual_m=float(dx.max()))
        rec = {'scene': Path(sel['scene']).name if sel.get('scene') else run.root.name, 'point_id': pid, 'selection_reason': p['selection_reason'], 'static_basis': sel['static_basis']['rule'],
               'source_frame_camera_pixel_uv': p['source_frame_camera_pixel_uv'], 'canonical_provenance': prov, 'frame0_same_instant_check': frame0,
               'events_processing_step': e, 'events_frame': {kk: (order[v] if v > 0 else None) for kk, v in e.items()},
               'trajectory_attribution': {kk: (v[ii].tolist() if np.ndim(v) > 1 else float(v[ii])) for kk, v in att.items()},
               'trajectory_attribution_dominant_frame': order[dom_step], 'predicted_xyz_check_max_abs_m': float(np.abs(aux['xm'][:, ii]-T['Xpred'][1:, ii]).max()),
               'prediction_only_frames_check': pchk, 'timeline': tl, 'counterfactuals': cfs, 'traced_frames': per_frame}
        results.append(rec)
        lines.append(compact(rec, gt))
        print(f'[analyze] point {pid} ({p["selection_reason"]}): {len(per_frame)} traced frames, {time.time()-t0:.0f} s', flush=True)
    snap = snapshot([run.root, base.root])
    write_json(out/'analysis.json', {'snapshot': snap, 'parity_traced_vs_baseline': par, 'selection': sel, 'points': results, 'seconds': time.time()-t0})
    with open(out/'static_onset_trace.jsonl', 'w') as fh:
        for l in lines: fh.write(json.dumps(jsonable(l), allow_nan=False)+'\n')
    write_json(out/'summary.json', {'snapshot': snap, 'parity_traced_vs_baseline': par, 'baseline': sel['baseline'], 'static_basis': sel['static_basis'],
                                    'contract': sel['contract'], 'points': lines})
    print(json.dumps({'parity': par, 'points': len(results), 'seconds': time.time()-t0}))


def compact(rec, gt):
    """One JSONL line per audit point: the onset table of the report (section 8.3) with its key numbers."""
    e = rec['events_frame']
    tf = rec['traced_frames']
    def at(k):
        return tf.get(k) if k is not None else None
    onset = e['first_error_increase'] if gt and e['first_error_increase'] is not None else e['first_accepted_correction']
    fa = at(onset) or at(e['first_velocity_change'])
    row = {'scene': rec['scene'], 'point_id': rec['point_id'], 'selection_reason': rec['selection_reason'], 'source': rec['source_frame_camera_pixel_uv'],
           'events_frame': e, 'attribution': {k: rec['trajectory_attribution'][k] for k in ('net_m', 'position_corrections_projected_m', 'velocity_injections_projected_m',
                                                                                              'first_velocity_event_projected_m', 'closure_max_abs_m')},
           'dominant_frame': rec['trajectory_attribution_dominant_frame']}
    if fa is not None:
        it0 = fa['iterates'][0] if fa['iterates'] else None
        lin = [it['linearization']['classification'] for it in fa['iterates'] if it.get('linearization')]
        e1 = fa['E1_static_location']
        row['onset_frame'] = fa['frame']
        row['onset'] = {'acceptance': fa['acceptance'], 'accepted_iterate': fa['accepted_iterate'], 'iterates_run': fa['iterates_run'], 'converged': fa['converged'],
                        'cameras_used_at_s_minus': fa['cameras_used_at_s_minus'], 'cameras_used_accepted': fa['cameras_used_accepted'],
                        'delta_x_mm': fa['coupling']['delta_x_norm_mm'], 'delta_v_m_s': fa['coupling']['delta_v_norm_m_s'],
                        'identity_rel_error': fa['coupling']['identity_rel_error'], 'max_dtL_singular_value': float(np.max(fa['coupling']['dtL_singular_values'])),
                        'max_pixel_displacement_px_any_iterate': max([c['max_pixel_displacement_px'] for c in lin] or [0.]),
                        'any_cell_crossing': any(c['any_cell_crossing'] for c in lin), 'any_full_step_increasing_same_rows': any(c['full_step_increases_same_row_objective'] for c in lin),
                        'accepted_worse_than_prior_common_rows': (fa.get('objective_accepted_vs_prior_common_rows') or {}).get('b_worse_than_a'),
                        'E1_static_admitted_g_statistic': e1['admitted']['g_statistic'], 'E1_static_implied_dx_mm': 1e3*e1['admitted']['implied_step_from_x_with_this_prior']['delta_x_norm_m'],
                        'E1_static_implied_dv_m_s': e1['admitted']['implied_step_from_x_with_this_prior']['delta_v_norm_m_s'],
                        'invariant_violations': len((fa.get('invariants') or {}).get('violations_hidden_point_used', [])),
                        'gt': fa.get('gt')}
        if it0 and it0.get('camera_rows'):
            gs = {}
            for r in it0['camera_rows']:
                if r['used'] and 'g_rgb' in r: gs[f"cam{r['camera']}"] = {'g_rgb_norm': float(np.linalg.norm(r['g_rgb'])), 'g_depth_norm': float(np.linalg.norm(r['g_depth']))}
            row['onset']['per_camera_gradient_at_s_minus'] = gs
    cf = rec['counterfactuals'].get('first_offending')
    if cf and 'all_rows' in cf:
        row['counterfactual_first_offending'] = {'frame': cf['frame'], 'dominant_row': cf.get('dominant_row'),
                                                 **{k: {'dx_mm': v['delta_x_norm_mm'], 'dv_m_s': v['delta_v_norm_m_s'], 'gt_err_change_mm': v.get('gt_error_change_mm')}
                                                    for k, v in cf.items() if isinstance(v, dict) and 'delta_x_norm_mm' in v}}
    return row


# ---------------------------------------------------------------------------------------------- population E1 (GPU)
def cmd_population(a):
    """E1 over ALL grounded static points: rows evaluated AT the static location (x_GT, v = 0; every other point at its
    GT location for visibility) with the production measurement code, no update. For unbiased rows with covariance R
    the score g^T I^+ g is chi2_rank; its tail fraction above the 99% quantile is 1% under the model. Real scenes: the
    canonical x0 of IMAGE-CONSTANT proxy points (others at x0), labelled proxy."""
    import torch
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise, CODE
    from estimator.projective_terms import build_terms
    from estimator.projective_visibility import visibility, position_sigma_z2
    from estimator.point_kalman_cuda import predict
    run = Run(a.baseline, 'forward')
    src = Path(a.selection_from) if a.selection_from else run.root          # an arm of the covariance experiment reuses the baseline's basis
    sel = json.loads((src/'static_audit'/'selection.json').read_text())
    (run.root/'static_audit').mkdir(exist_ok=True)
    out = run.root/'static_audit'/'population_E1.json'
    if out.exists(): raise SystemExit(f'{out} exists: never overwritten')
    if run.fman['selected_ids_sha256'] != sel['selected_ids_sha256']: raise SystemExit('point ids differ from the selection run')
    basis = np.load(src/'static_audit'/'static_basis.npz')
    static = basis['static']
    x0 = run.can['canonical_xyz']; n = len(x0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rig = CameraRig(run.data.cameras, dev, torch.float64)
    nz = run.noise
    noise = SensorNoise(torch.as_tensor(nz.rgb_var, dtype=torch.float64, device=dev), nz.a, nz.b, nz.rgb_model, nz.depth_model, nz.interp)
    order = run.fman['processing_order']; t = np.asarray(run.fman['timestamps_seconds'])[order]
    m = run.man['noise']
    P0 = np.zeros((n, 6, 6)); P0[:, :3, :3] = run.can['canonical_source_covariance']; P0[:, 3:, 3:] = np.asarray(m['velocity_covariance'])
    _, P1 = predict(torch.zeros((1, 6), dtype=torch.float64), torch.as_tensor(P0[:1]), float(t[1]-t[0]), torch.as_tensor(np.asarray(m['qc'])))
    dP = (P1[0]-torch.as_tensor(P0[0])).numpy()            # P-_1 - P_0 is the same for every point (F, Q do not depend on the point)
    P1 = torch.as_tensor(P0+dP, device=dev)
    Sig = P1[:, :3, :3]
    Lam = torch.linalg.inv(P1)
    rgb0 = torch.as_tensor(run.can['canonical_rgb'], dtype=torch.float64, device=dev)
    gt = sel['kind'] == 'synthetic'
    res = {'kind': sel['kind'], 'static_points': int(static.sum()), 'gate_covariance': 'the exact frame-1 prior P-_1 = F P_0 F^T + Q of every point (used for the gate '
           'at every evaluated frame; at later frames the production prior differs)', 'frames': {}}
    frames = sorted({order[s] for s in a.steps if 0 < s < len(order)} | {order[-1] if gt else order[min(5, len(order)-1)]})
    for k in frames:
        X = x0.copy()
        if gt:
            from tools.evaluate_projective_gt import gt_transform
            g_ = gt_basis(Path(sel['scene']), x0)
            q = g_['q'][run.man['dataset']['original_frame_ids']] if len(g_['q']) > max(run.man['dataset']['original_frame_ids']) else g_['q']
            T = gt_transform(g_['joint'], g_['axis'], g_['point'], q[k]-q[order[0]])
            mv = g_['nearest_label_moving']
            X[mv] = x0[mv]@T[:3, :3].T+T[:3, 3]
        xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        frame = MeasuredFrame(run.data, k, dev)
        proj = rig.project(xt, jacobian=False)
        vr, _ = visibility(proj, position_sigma_z2(rig.R, Sig), frame.H, frame.W, rig.near, run.policy['visibility_n_sigma'])
        sid = torch.as_tensor(np.flatnonzero(static), device=dev)
        cand = (vr[:, sid] == CODE['VALID']).T
        rows = torch.nonzero(cand)
        p, c = sid[rows[:, 0]], rows[:, 1]
        N = len(sid)
        acc = {nm: {'I': torch.zeros((N, 3, 3), dtype=torch.float64, device=dev), 'g': torch.zeros((N, 3), dtype=torch.float64, device=dev)} for nm in ('admitted', 'sampled')}
        zsum = torch.zeros((rig.count, 4), dtype=torch.float64, device=dev); zsq = torch.zeros_like(zsum); zcnt = torch.zeros(rig.count, dtype=torch.float64, device=dev)
        for a0 in range(0, len(p), 400_000):
            pp, cc, rr = p[a0:a0+400_000], c[a0:a0+400_000], rows[a0:a0+400_000, 0]
            tt = build_terms(rig, frame, cc, pp, xt[pp], rgb0[pp], Sig[pp], noise, run.policy, torch.float64)
            okm = {'admitted': tt.reason == CODE['VALID'], 'sampled': (tt.reason == CODE['VALID']) | (tt.reason == CODE['MEASURED_SURFACE_IN_FRONT']) | (tt.reason == CODE['INNOVATION_INCONSISTENT'])}
            for nm, ok in okm.items():
                w = torch.where(ok[:, None], 1/tt.var, torch.zeros_like(tt.var)); r = torch.where(ok[:, None], tt.r, torch.zeros_like(tt.r))
                Hx = torch.where(ok[:, None, None], tt.Hx, torch.zeros_like(tt.Hx))
                acc[nm]['I'].index_add_(0, rr, (Hx[:, :, :, None]*w[:, :, None, None]*Hx[:, :, None, :]).sum(1))
                acc[nm]['g'].index_add_(0, rr, (Hx*(w*r)[:, :, None]).sum(1))
            ok = okm['admitted']
            z = torch.where(ok[:, None], tt.r/tt.var.sqrt(), torch.zeros_like(tt.r))
            zsum.index_add_(0, cc, z); zsq.index_add_(0, cc, z*z); zcnt.index_add_(0, cc, ok.double())
        fr = {}
        for nm, v in acc.items():
            I, g = v['I'], v['g']
            ev, U = np.linalg.eigh(I.cpu().numpy())                       # host: cuSOLVER batched syev fails on 1e5 matrices
            ev, U = torch.as_tensor(ev, device=dev), torch.as_tensor(U, device=dev)
            keep = ev > ev[:, -1:]*1e-12
            proj_ = (U.transpose(1, 2)@g[..., None])[..., 0]
            s = torch.where(keep & (ev > 0), proj_**2/ev.clamp_min(1e-300), torch.zeros_like(ev)).sum(1)
            dof = keep.sum(1)
            has = dof > 0
            q99 = torch.as_tensor(chi2.ppf(.99, dof.clamp_min(1).cpu().numpy()), device=dev)
            A = Lam[sid].clone(); A[:, :3, :3] += I
            d = torch.linalg.solve(A, torch.cat([g, torch.zeros_like(g)], 1)[..., None])[..., 0]
            dxn, dvn = d[:, :3].norm(dim=1)[has].cpu().numpy(), d[:, 3:].norm(dim=1)[has].cpu().numpy()
            sd = (s/dof.clamp_min(1))[has].cpu().numpy()
            fr[nm] = {'points_with_rows': int(has.sum()), 'score_per_dof_quantiles': q_(sd, (.5, .9, .99)), 'mean_score_per_dof': float(sd.mean()) if len(sd) else None,
                      'fraction_above_chi2_99': float((s > q99)[has].double().mean()) if bool(has.any()) else None,
                      'implied_static_step_dx_mm': q_(1e3*dxn, (.5, .9, .99)), 'implied_static_velocity_m_s': q_(dvn, (.5, .9, .99)),
                      'implied_next_frame_displacement_mm': q_(1e3*dvn*abs(float(t[1]-t[0])), (.5, .9, .99))}
        cz = zcnt.clamp_min(1)[:, None]
        fr['admitted_rows_whitened_residual_by_camera'] = {f'cam{ci}': {'rows': int(zcnt[ci]), 'mean_r_over_sigma_RGBD': (zsum[ci]/cz[ci]).tolist(),
                                                                       'rms_r_over_sigma_RGBD': (zsq[ci]/cz[ci]).sqrt().tolist()} for ci in range(rig.count)}
        res['frames'][int(k)] = fr
        print(f'[population] frame {k}: {json.dumps(jsonable(fr["admitted"]))[:400]}', flush=True)
    res['note'] = ('score = g^T I^+ g at the static location, per point over its admitted rows; under an unbiased model with covariance R it is chi2_rank '
                   '(mean per dof 1, 1% above the 99% quantile). Implied step: (Lambda-_1 + I)^-1 [g, 0], the first update a prior CENTRED at the truth would '
                   'publish. The per-camera mean r/sigma is the signed bias of each channel; rms is its spread (1 if R is right).')
    write_json(out, res)


# ------------------------------------------------------------------------------------------------- population onset
def cmd_onset(a):
    """Every static point (not only the audit samples), from the untraced run: the published update at its FIRST velocity
    injection (rows, convergence, IEKF iterations, acceptance, |dx|, |dv|, |dv| dt / |dx|) and the share of accepted
    one-row updates, for points that later depart (> reporting threshold) vs points that stay stable (< 1 cm)."""
    run = Run(a.baseline, 'forward')
    sel = json.loads((run.root/'static_audit'/'selection.json').read_text())
    out = run.root/'static_audit'/'onset_population.json'
    if out.exists(): raise SystemExit(f'{out} exists: never overwritten')
    idx = np.flatnonzero(np.load(run.root/'static_audit'/'static_basis.npz')['static'])
    T = trajectories(run, idx, sel.get('window'))
    gt = sel['kind'] != 'real'
    ev, att, aux = events(T, run.can['canonical_xyz'][idx] if gt else None)
    order = T['order']
    g = lambda n, dt=float: np.asarray(run.mm(n)[:, idx], dt)[order]
    conv, iters = g('converged', bool), g('iekf_iterations', np.int64)
    j1 = ev['first_velocity_change']; has = j1 > 0
    n = len(idx); ar = np.arange(n)
    s = np.maximum(j1, 1)
    dx = np.linalg.norm(aux['dX'][s-1, ar], axis=-1); dv = np.linalg.norm(aux['dV'][s-1, ar], axis=-1)
    dt = np.abs(T['t'][s]-T['t'][s-1])
    emax = aux['errp'].max(0); final = aux['errp'][-1]
    groups = {'departing_over_10cm': final > REPORTING_THRESHOLD_M, 'stable_never_over_1cm': emax <= STABLE_FRACTION*REPORTING_THRESHOLD_M,
              'all_static': np.ones(n, bool)}
    res = {'kind': sel['kind'], 'window': sel.get('window'), 'static_points': n, 'groups': {}}
    acc_all, rows_all = T['acc'][1:], T['nrows'][1:]
    for name, m in groups.items():
        mm = m & has
        if not mm.any(): res['groups'][name] = {'points': int(m.sum()), 'with_velocity_event': 0}; continue
        rows = T['nrows'][s, ar][mm]; acc = T['acc'][s, ar][mm]
        accepted = (acc_all == 1) | (acc_all == 2)
        res['groups'][name] = {
            'points': int(m.sum()), 'with_velocity_event': int(mm.sum()),
            'first_velocity_event_step': q_(j1[mm], (.1, .5, .9)),
            'rows_at_first_velocity_event': {str(r): int((rows == r).sum()) for r in np.unique(rows)},
            'converged_at_first_velocity_event_fraction': float(conv[s, ar][mm].mean()),
            'iekf_iterations_at_first_velocity_event': {str(r): int((iters[s, ar][mm] == r).sum()) for r in np.unique(iters[s, ar][mm])},
            'acceptance_at_first_velocity_event': {O_ACC[c]: int((acc == c).sum()) for c in range(5) if (acc == c).any()},
            'dx_mm': q_(1e3*dx[mm], (.1, .5, .9)), 'dv_m_s': q_(dv[mm], (.1, .5, .9)),
            'dv_dt_over_dx': q_((dv*dt/np.maximum(dx, 1e-300))[mm & (dx > 0)], (.1, .5, .9)),
            'accepted_point_frames': int(accepted[:, m].sum()), 'accepted_one_row_fraction': float((accepted & (rows_all == 1))[:, m].sum()/max(accepted[:, m].sum(), 1)),
            'accepted_not_converged_fraction': float((accepted & ~conv[1:])[:, m].sum()/max(accepted[:, m].sum(), 1))}
        # the NEXT frame after the first injection: can the measurements take the injected velocity back?
        nx = mm & (j1+1 < len(order))
        if nx.any():
            s2 = s[nx]+1; pts = idx[nx]
            cr = np.stack([np.asarray(run.mm('camera_reason')[order[k]][:, p]) for k, p in zip(s2, pts)])       # [m,C]
            acc2 = T['acc'][s2, ar[nx]]
            better = aux['errp'][s2-1, ar[nx]] < aux['errm'][s2-1, ar[nx]]
            ok2 = (acc2 == 1) | (acc2 == 2)
            res['groups'][name]['next_frame_after_first_injection'] = {
                'points': int(nx.sum()), 'acceptance': {O_ACC[c]: int((acc2 == c).sum()) for c in range(5) if (acc2 == c).any()},
                'accepted_and_error_decreased': int((ok2 & better).sum()), 'accepted_and_error_increased': int((ok2 & ~better).sum()),
                'camera_reason_counts': {O.NAMES[c]: int((cr == c).sum()) for c in np.unique(cr)},
                'prediction_displacement_mm': q_(1e3*np.linalg.norm(T['V'][s2-1, ar[nx]], axis=-1)*np.abs(T['t'][s2]-T['t'][s2-1]), (.1, .5, .9))}
    res['note'] = 'dv dt / dx ~ 1 means the whole position correction became velocity (dt L ~ I); converged = the step criterion delta^T A delta < tol was met'
    write_json(out, res)
    print(json.dumps(jsonable(res))[:3000])


# ----------------------------------------------------------------------------- 2x2 covariance experiment (offline)
def _mean_match(r2, v):
    """m >= 0 with mean(r^2 / (v + m)) = 1: the additive variance that makes the SECOND MOMENT of the standardized
    residual one (0 when the sensor variance already explains it)."""
    f = lambda m: float(np.mean(r2/(v+m)))-1.
    if len(r2) == 0 or f(0.) <= 0: return 0.
    lo, hi = 0., float(np.mean(r2))+1e-30
    while f(hi) > 0: hi *= 2
    for _ in range(200):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def gt_speed99(scene, run, x0):
    """GT v99: 99th percentile of the true speed of EVERY canonical point (static ones at 0) over every frame interval;
    prismatic |dq/dt|, revolute |dq/dt| times the distance to the axis; nearest-label part identity."""
    g = gt_basis(scene, x0)
    fids = run.man['dataset']['original_frame_ids']; q = g['q'][fids] if len(g['q']) > max(fids) else g['q']
    t = np.asarray(run.fman['timestamps_seconds'], float)
    w = np.abs(np.diff(q)/np.diff(t))                                       # [K-1] joint rate
    mv = g['nearest_label_moving']
    if g['joint'] == 'PRISMATIC': rad = np.ones(len(x0))
    else:
        d = x0-g['point']; rad = np.linalg.norm(d-(d@g['axis'])[:, None]*g['axis'][None], axis=1)
    sp = np.where(mv[None], w[:, None]*rad[None], 0.)                         # [K-1, N]
    return {'v99_m_s': float(np.quantile(sp, .99)), 'moving_fraction_nearest_label': float(mv.mean()), 'max_true_speed_m_s': float(sp.max()),
            'mean_square_speed_m2_s2': float((sp**2).mean()), 'joint': g['joint'], 'rule': 'v99 = 99th percentile of true point speeds over all '
            'points (static at 0) and all frame intervals; P_vv = (v99 / sqrt(chi2_3(0.99)))^2 I via --velocity-prior-speed99 (scene/noise_calibration.velocity_prior)'}


def cmd_covariance_calibration(a):
    """Offline calibration for the 2x2 experiment (GROUND TRUTH / proxy used here only):
    R  : rows of every grounded static point evaluated AT ITS STATIC LOCATION (every other point at its GT location for
         visibility) on several frames, with the run's own sensor variance through the interpolation weights; the
         additive model variance per block (RGB pooled over channels, depth) is the second-moment match
         mean(r^2 / (v_sensor + m)) = 1. Rows: visibility VALID, valid 2x2 sample, measured surface not in front
         (the 'depth_front_only' rule with the source covariance, as scene/noise_calibration.cross_camera_mismatch).
    P_vv: GT v99 per scene (synthetic); real scenes have no GT speed -> unavailable.
    Writes the two noise configs (current R / calibrated R, both with the run's measured SENSOR variances, P_vv and q_c
    of configs/example_uncalibrated.json) and the policy overrides; nothing else changes."""
    import torch, copy
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise, CODE
    from estimator.projective_terms import build_terms
    from estimator.projective_visibility import visibility, position_sigma_z2
    from tools.evaluate_projective_gt import gt_transform
    run = Run(a.baseline, 'forward')
    sel = json.loads((run.root/'static_audit'/'selection.json').read_text())
    out = Path(a.out)
    if (out/'covariance_calibration.json').exists(): raise SystemExit(f'{out} has a calibration: never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    static = np.load(run.root/'static_audit'/'static_basis.npz')['static']
    x0 = run.can['canonical_xyz']; n = len(x0)
    calib = json.loads((run.root/'noise_calibration.json').read_text())['result']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rig = CameraRig(run.data.cameras, dev, torch.float64)
    noise = SensorNoise(torch.as_tensor(calib['rgb_variance'], dtype=torch.float64, device=dev), calib['depth_variance_m2'], calib['depth_squared_variance'], 0., 0.,
                        run.policy['interpolation_variance'])
    pol = {**run.policy, 'row_innovation_gate': 'depth_front_only', 'rgb_model_variance': 0., 'depth_model_variance': 0.}
    order = run.fman['processing_order'][:(sel.get('window') or 10**9)+1]
    steps = sorted({s for s in (1, 2, 5, 10, 20, 30, 40, 50, 59) if s < len(order)} | {len(order)-1})
    gt = sel['kind'] == 'synthetic'
    g_ = gt_basis(Path(sel['scene']), x0) if gt else None
    S0 = torch.as_tensor(run.can['canonical_source_covariance'], dtype=torch.float64, device=dev)
    rgb0 = torch.as_tensor(run.can['canonical_rgb'], dtype=torch.float64, device=dev)
    sid = torch.as_tensor(np.flatnonzero(static), device=dev)
    R2r, Vr, R2d, Vd, per = [], [], [], [], {}
    for s in steps:
        k = order[s]
        X = x0.copy()
        if gt:
            fids = run.man['dataset']['original_frame_ids']; q = g_['q'][fids] if len(g_['q']) > max(fids) else g_['q']
            T = gt_transform(g_['joint'], g_['axis'], g_['point'], q[k]-q[order[0]]); mv = g_['nearest_label_moving']
            X[mv] = x0[mv]@T[:3, :3].T+T[:3, 3]
        xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
        frame = MeasuredFrame(run.data, k, dev)
        vr, _ = visibility(rig.project(xt, jacobian=False), position_sigma_z2(rig.R, S0), frame.H, frame.W, rig.near, pol['visibility_n_sigma'])
        rows = torch.nonzero((vr[:, sid] == CODE['VALID']).T)
        p, c = sid[rows[:, 0]], rows[:, 1]
        nk = 0
        for a0 in range(0, len(p), 400_000):
            pp, cc = p[a0:a0+400_000], c[a0:a0+400_000]
            tt = build_terms(rig, frame, cc, pp, xt[pp], rgb0[pp], S0[pp], noise, pol, torch.float64)
            ok = tt.valid()
            R2r.append((tt.r[ok, :3]**2).cpu().numpy().ravel()); Vr.append(tt.var[ok, :3].cpu().numpy().ravel())
            R2d.append((tt.r[ok, 3]**2).cpu().numpy()); Vd.append(tt.var[ok, 3].cpu().numpy()); nk += int(ok.sum())
        per[int(k)] = nk
    R2r, Vr, R2d, Vd = (np.concatenate(v) for v in (R2r, Vr, R2d, Vd))
    m_rgb, m_d = _mean_match(R2r, Vr), _mean_match(R2d, Vd)
    q99 = float(chi2.ppf(.99, 1))
    stats = lambda r2, v, m: {'rows': int(len(r2)), 'mean_standardized': float(np.mean(r2/(v+m))), 'median_standardized': float(np.median(r2/(v+m))),
                              'fraction_above_chi2_1_99': float(np.mean(r2/(v+m) > q99))}
    cur = {'rgb_model_variance': calib['rgb_model_variance'], 'depth_model_variance': calib['depth_model_variance_m2']}
    res = {'kind': sel['kind'], 'scene': sel['scene'], 'steps': steps, 'rows_per_frame': per,
           'R': {'sensor_unchanged': {'rgb_variance': calib['rgb_variance'], 'depth_variance_m2': calib['depth_variance_m2'], 'depth_squared_variance': calib['depth_squared_variance']},
                 'current_model_variance': cur, 'calibrated_model_variance': {'rgb_model_variance': m_rgb, 'depth_model_variance': m_d},
                 'standardized_current': {'rgb': stats(R2r, Vr, cur['rgb_model_variance']), 'depth': stats(R2d, Vd, cur['depth_model_variance'])},
                 'standardized_calibrated': {'rgb': stats(R2r, Vr, m_rgb), 'depth': stats(R2d, Vd, m_d)},
                 'sigma_total_typical': {'rgb': float(np.sqrt(np.median(Vr)+m_rgb)), 'depth_m': float(np.sqrt(np.median(Vd)+m_d))},
                 'rule': 'second moment at the static location: mean(r^2/(v_sensor + m)) = 1 per block; '
                         + ('GROUND TRUTH static identity' if gt else 'fixture static by construction' if sel['kind'] == 'fixture' else 'DEPTH-CONSTANT PROXY (not GT)')},
           'P_vv': gt_speed99(Path(sel['scene']), run, x0) if gt else {'v99_m_s': None, 'note': 'no ground-truth speeds' + (' (static fixture: GT v99 = 0, a degenerate prior)' if sel['kind'] == 'fixture' else
                                                                                                                             ' for real scenes: arms C / D unavailable (no declared value given)')}}
    base = json.loads((REPO/'configs/example_uncalibrated.json').read_text())
    for tag, mm in (('R_current', cur), ('R_calibrated', {'rgb_model_variance': m_rgb, 'depth_model_variance': m_d})):
        cfg = copy.deepcopy(base)
        cfg['sensor']['covariance'] = np.diag(list(calib['rgb_variance'])+[calib['depth_variance_m2']]).tolist()
        cfg['sensor']['covariance_depth_squared'] = np.diag([0., 0., 0., calib['depth_squared_variance']]).tolist()
        cfg['calibration_status'] = f'2x2 covariance experiment: measured SENSOR variances of {run.root} ({tag}); model variances via --policy-override'
        (out/f'noise_config_{tag}.json').write_text(json.dumps(cfg, indent=1))
        res[f'train_args_{tag}'] = ['--sensor-noise', 'config', '--noise-config', str(out/f'noise_config_{tag}.json'), '--policy-override',
                                    f"rgb_model_variance={mm['rgb_model_variance']!r}", f"depth_model_variance={mm['depth_model_variance']!r}"]
    if res['P_vv'].get('v99_m_s'): res['train_args_Pvv_calibrated'] = ['--velocity-prior-speed99', repr(res['P_vv']['v99_m_s'])]
    write_json(out/'covariance_calibration.json', res)
    print(json.dumps(jsonable({'R': res['R'], 'P_vv': res['P_vv']}), indent=1))


def cmd_arms(a):
    """Metrics of the arms of one scene (static basis and ids from --baseline): first-frame static displacement and
    velocity, injected velocity at the first event, later drift, inconsistency at the truth (population_E1.json of the
    arm), moving-point accuracy (evaluation_gt of the arm, synthetic)."""
    base = Run(a.baseline, 'forward')
    sel = json.loads((base.root/'static_audit'/'selection.json').read_text())
    idx = np.flatnonzero(np.load(base.root/'static_audit'/'static_basis.npz')['static'])
    gt = sel['kind'] != 'real'
    res = {'scene': sel['scene'], 'kind': sel['kind'], 'window': sel.get('window'), 'static_points': int(len(idx)), 'arms': {}}
    for spec in a.arms:
        name, path = str(spec).split('=', 1)
        run = Run(path, 'forward')
        if run.fman['selected_ids_sha256'] != sel['selected_ids_sha256']: raise SystemExit(f'{name}: point ids differ')
        x0 = run.can['canonical_xyz'][idx]
        T = trajectories(run, idx, sel.get('window'))
        ev, att, aux = events(T, x0 if gt else None)
        d1 = np.linalg.norm(T['X'][1]-x0, axis=-1); v1 = np.linalg.norm(T['V'][1], axis=-1)
        j1 = ev['first_velocity_change']; has = j1 > 0; ar = np.arange(len(idx)); s = np.maximum(j1, 1)
        dvi = np.linalg.norm(aux['dV'][s-1, ar], axis=-1)[has]
        err = aux['errp']
        m = run.man
        r = {'run': str(run.root), 'P_vv_m2_s2': m['noise']['velocity_covariance'][0][0], 'v99_declared': m['arguments'].get('velocity_prior_speed99'),
             'rgb_model_variance': m['policy']['rgb_model_variance'], 'depth_model_variance': m['policy']['depth_model_variance'],
             'rgb_sensor_variance': np.diag(np.asarray(m['noise']['color_covariance'])).tolist(), 'depth_sensor_variance': m['noise']['depth_variance'],
             'seconds': json.loads((run.free/'status.json').read_text())['seconds'],
             'first_frame_static_displacement_mm': {**q_(1e3*d1, (.5, .9, .99)), 'over_1mm_fraction': float((d1 > 1e-3).mean())},
             'first_frame_speed_m_s': q_(v1, (.5, .9, .99)),
             'first_injection_dv_m_s': {**(q_(dvi, (.5, .9, .99)) or {}), 'points_with_injection_fraction': float(has.mean())},
             'drift': {f'step_{st}': {**q_(1e3*err[st-1], (.5, .9, .99)), 'over_10cm_fraction': float((err[st-1] > REPORTING_THRESHOLD_M).mean())}
                       for st in sorted({min(10, aux['S']), min(30, aux['S']), aux['S']})},
             'final_speed_m_s': q_(np.linalg.norm(T['V'][-1], axis=-1), (.5, .9, .99)),
             'drifting_velocity_share_median': (lambda dm: float(np.median(att['velocity_injections_projected_m'][dm]/att['net_m'][dm])) if dm.any() else None)(err[-1] > REPORTING_THRESHOLD_M)}
        pe = run.root/'static_audit'/'population_E1.json'
        if pe.exists():
            fr = list(json.loads(pe.read_text())['frames'].values())[0]
            r['at_truth_frame1'] = {nm: {'fraction_above_chi2_99': fr[nm]['fraction_above_chi2_99'], 'implied_step_mm': fr[nm]['implied_static_step_dx_mm'],
                                         'implied_velocity_m_s': fr[nm]['implied_static_velocity_m_s'], 'points_with_rows': fr[nm]['points_with_rows']} for nm in ('admitted', 'sampled')}
        ev_ = run.root/'evaluation_gt'/'evaluation.json'
        if ev_.exists():
            f = json.loads(ev_.read_text())['directions']['forward']['free_final']
            r['moving_final'] = {k: f[k] for k in ('true_motion_m_median_moving', 'error_m_median_moving', 'error_m_median_moving_measured', 'within_5mm_fraction_moving',
                                                   'error_m_p99_moving', 'fraction_over_10cm_moving', 'error_m_median_static', 'fraction_over_10cm_static')}
        res['arms'][name] = r
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists: never overwritten')
    write_json(out, res)
    print(json.dumps(jsonable(res))[:6000])


# ------------------------------------------------------------------------------------------------------------- fixture
def cmd_fixture(a):
    from tests.image_fixtures import two_plate_scene
    from tests.projective_fixtures import write_scene, hires
    root = Path(a.out)
    if root.exists() and any(root.iterdir()): raise SystemExit(f'{root} exists and is not empty')
    ds = hires(two_plate_scene('static', frames=a.frames, cameras=4))
    write_scene(ds, root/'scene')
    write_json(root/'fixture.json', {'generator': "tests.image_fixtures.two_plate_scene('static', frames=%d, cameras=4) + tests.projective_fixtures.hires" % a.frames,
                                     'truth': 'every surface static; x_GT = canonical ray-cast surface sample', 'train_py_extra': ['--fps', '30']})
    print(root/'scene')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('fixture'); p.add_argument('out', type=Path); p.add_argument('--frames', type=int, default=6)
    p = sp.add_parser('select'); p.add_argument('baseline', type=Path); p.add_argument('--scene', required=True); p.add_argument('--kind', choices=['synthetic', 'fixture', 'real'], required=True)
    p.add_argument('--window', type=int, help='real scenes: last processing step of the proxy / audit window')
    p.add_argument('--worst', type=int, default=10); p.add_argument('--controls', type=int, default=5); p.add_argument('--innovation', type=int, default=5)
    p = sp.add_parser('analyze'); p.add_argument('traced', type=Path); p.add_argument('--baseline', type=Path, required=True)
    p = sp.add_parser('population'); p.add_argument('baseline', type=Path); p.add_argument('--steps', type=int, nargs='*', default=[1, 2, 5])
    p.add_argument('--selection-from', type=Path, help='take selection.json / static_basis.npz from this run (same point ids)')
    p = sp.add_parser('covariance-calibration'); p.add_argument('baseline', type=Path); p.add_argument('--out', type=Path, required=True)
    p = sp.add_parser('arms'); p.add_argument('arms', type=Path, nargs='+', help='NAME=RUN pairs of one scene'); p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sp.add_parser('onset'); p.add_argument('baseline', type=Path)
    a = ap.parse_args(argv)
    {'fixture': cmd_fixture, 'select': cmd_select, 'analyze': cmd_analyze, 'population': cmd_population, 'onset': cmd_onset,
     'covariance-calibration': cmd_covariance_calibration, 'arms': cmd_arms}[a.cmd](a)


if __name__ == '__main__':
    main()
