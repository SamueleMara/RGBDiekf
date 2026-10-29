"""Deterministic root-cause investigation of free-point run-aways (RGBDiekf_deterministic_root_cause_debug_spec.md).

Investigation only: no gate, threshold, acceptance rule, limit, damping or filter change. Every alternative below is an
OFFLINE replay in tools/root_cause_oracle.py (independent float64 NumPy), never substituted into an update.

    find        RUN [--direction D] [--top N]
                rank the run-away points of a completed free pass and their first divergent update (largest accepted
                velocity change on an observed frame, then prediction-only)
    analyze     RUN --point ID [--frame K] [--direction D] [--alt-noise measured|PATH] [--alt-speed99 V ..] [--alt-qc Q ..]
                full float64 analysis of one point in a run traced with --debug-point-ids ID --debug-frames <all frames>
    population  RUN [--direction D] [--points N]
                same-instant cross-camera residuals of the endpoint cloud by camera pair, image gradient, mask border,
                viewing angle and depth discrepancy (is one Gaussian R a description of them?)

RUN is a train.py output folder (one scene). Results go to RUN/root_cause/<name>/, never overwritten. A thresholds
used below only CLASSIFY the evidence for the report (stated where used); none of them reaches a filter.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import root_cause_oracle as O          # noqa: E402

SOURCES = ['estimator/point_kalman_cuda.py', 'estimator/free_projective_run.py', 'estimator/projective_sensor.py', 'estimator/projective_terms.py',
           'estimator/projective_visibility.py', 'estimator/projective_camera.py', 'scene/endpoint.py', 'scene/config.py', 'scene/noise_calibration.py',
           'configs/projective_rgbd.json', 'configs/example_uncalibrated.json', 'train.py', 'tools/root_cause_oracle.py', 'tools/root_cause_trace.py']
TOL_ALGEBRA = 1e-9          # report classification: relative mismatch above this = implementation disagreement (float64)


# ----------------------------------------------------------------------------------------------------------- utilities
def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, (np.floating, float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def write_json(path, v):
    Path(path).write_text(json.dumps(jsonable(v), indent=1, allow_nan=False)+'\n')


def git(*a):
    try: return subprocess.check_output(['git', *a], cwd=REPO, text=True)
    except Exception as e: return f'unavailable: {e}'


def provenance_tool():
    return {'git_head': git('rev-parse', 'HEAD').strip(), 'git_status_short': git('status', '--short'), 'git_diff_stat': git('diff', '--stat'),
            'dirty_patch_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
            'source_sha256': {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in SOURCES if (REPO/f).exists()},
            'argv': sys.argv, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def out_dir(run, name):
    d = Path(run)/'root_cause'/name
    if d.exists() and any(d.iterdir()): raise SystemExit(f'{d} exists and is not empty: results are never overwritten')
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a-b).max()/max(np.abs(b).max(), 1e-300)) if a.size else 0.


def absd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a-b).max()) if a.size else 0.


class Run:
    def __init__(self, root, direction):
        self.root = Path(root); self.direction = direction
        self.man = json.loads((self.root/'run_manifest.json').read_text())
        self.free = self.root/f'free_{direction}'
        self.fman = json.loads((self.free/'run_manifest.json').read_text())
        self.can = dict(np.load(self.free/'canonical.npz'))
        self.policy = self.man['policy']
        self.noise = O.Noise.from_manifest(self.man['noise'], self.policy)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            from scene.dataset import Dataset
            a = self.man['arguments']
            p = lambda k: None if a.get(k) is None else a[k]
            self._data = Dataset(Path(a['scene']), a.get('frame_limit'), None, fps=p('fps'), dt_seconds=p('dt_seconds'),
                                 timestamps_file=None if a.get('timestamps_file') is None else Path(a['timestamps_file']))
        return self._data

    def cams(self):
        return [O.Cam(c) for c in self.data.cameras]

    def images(self, k):
        return [O.Image(self.data.observation(k, c)) for c in range(len(self.data.cameras))]

    def trace(self, k):
        f = self.free/'numeric'/f'trace_frame_{k:04d}.json'
        return json.loads(f.read_text())['points'] if f.exists() else None

    def mm(self, name):
        return np.load(self.free/f'{name}.npy', mmap_mode='r')


# ---------------------------------------------------------------------------------------------------------------- find
def cmd_find(a):
    run = Run(a.run, a.direction)
    x, v, st = run.mm('posterior_xyz'), run.mm('velocity'), run.mm('point_status')
    order = run.fman['processing_order']; k0 = order[0]
    x0 = np.asarray(x[k0])
    disp = np.zeros(len(x0))
    for k in order: disp = np.maximum(disp, np.linalg.norm(np.asarray(x[k])-x0, axis=1))
    top = np.argsort(-disp)[:a.top]
    src = run.can['original_source_ids']; W = [c.width for c in run.data.cameras]
    rows = []
    for p in top:
        dv = []
        for prev, k in zip(order, order[1:]):
            if st[k, p] == 1: dv.append((float(np.linalg.norm(np.asarray(v[k, p], float)-np.asarray(v[prev, p], float))), k))
        dv.sort(reverse=True)
        k_div = dv[0][1] if dv else None
        after = [int(st[k, p]) for k in order[order.index(k_div)+1:]] if k_div is not None else []
        f, c, pix = (int(t) for t in src[p])
        rows.append({'point': int(p), 'max_displacement_m': float(disp[p]), 'first_divergent_frame': k_div, 'velocity_change_m_s': dv[0][0] if dv else None,
                     'speed_after_m_s': float(np.linalg.norm(v[k_div, p])) if k_div is not None else None,
                     'top3_velocity_changes': dv[:3], 'prediction_only_frames_after': int(sum(s == 2 for s in after)), 'observed_frames_after': int(sum(s == 1 for s in after)),
                     'source_frame_camera_pixel_uv': [f, c, pix % W[c], pix//W[c]]})
    for r in rows: print(json.dumps(r))
    return rows


# ------------------------------------------------------------------------------------------------------------- analyze
def frozen_codes(cam_list):
    return [O.CODE[c['visibility_frozen']] if c['visibility_frozen'] is not None else O.VALID for c in cam_list]


def problem_from_trace(run, k, pid, tr, imgs, cams, noise=None, **kw):
    pre = tr['prediction']['points'][str(pid)]
    it0 = tr['iterations'][0]['points'][str(pid)]
    fin = tr['final'][str(pid)]
    recheck = None
    if run.policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion':
        recheck = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']]
    return O.Problem(cams, imgs, run.can['canonical_rgb'][pid], np.array(pre['s_minus']), np.array(pre['P_minus']), frozen_codes(it0['cameras']),
                     noise or run.noise, run.policy, run.fman['max_iekf_iters'], run.fman['step_tolerance'], recheck, **kw)


def summarize_rows(rows, Sigma=None):
    """Sigma: prior position covariance -> the prior's major 1-sigma extent in pixels (how many bilinear cells the
    Gauss-Newton model is extrapolated over)."""
    out = []
    for o in rows:
        d = {'camera': o['camera'], 'reason': O.NAMES[o['reason']]}
        if 'u' in o: d.update(u=o['u'], v=o['v'], Z=o['Z'])
        s = o.get('sample', {})
        if 'pixel00' in s:
            d.update(pixel00_u_v=list(s['pixel00']), du_dv=[s['du'], s['dv']], neighbour_rgb=s['neighbour_rgb'], neighbour_depth_m=s['neighbour_depth'],
                     neighbour_mask=s['neighbour_mask'], neighbour_raw_valid=s['neighbour_raw_valid'])
        if 'w' in s:
            d.update(bilinear_weights=s['w'], sampled_rgb=s['rgb'], sampled_depth_m=s['depth'], grad_rgb_uv=s['grad_rgb'], grad_depth_uv=s['grad_depth'],
                     same_surface_spread_m=s['same_surface_spread_m'], same_surface_allowance_m=s['same_surface_allowance_m'])
        if 'du' in s: d['cell_border_distance_px'] = float(min(s['du'], 1-s['du'], s['dv'], 1-s['dv']))
        if 'H' in o: d.update(r=o['r'], H_x=o['H'], R_diag=o['var'], S=o['S'], S_zz=o['S_zz'], nu_prior_linearized=o['nu'], row_nis=o['nis'],
                              r_over_sigma=o['r']/np.sqrt(o['var']), J_u=o['J_u'], J_Z=o['J_Z'],
                              depth_gradient_m_per_px=float(np.linalg.norm(s['grad_depth'])), H_depth_norm=float(np.linalg.norm(o['H'][3])))
        if 'J_u' in o and Sigma is not None:
            d['prior_sigma_px_major'] = float(math.sqrt(max(np.linalg.eigvalsh(o['J_u']@Sigma@o['J_u'].T))))
        out.append(d)
    return out


def compare_rows(oracle_rows, prod_cams):
    """Oracle rows (raw re-sample) vs the production probe of the same state, camera by camera."""
    out, worst = [], 0.
    for o, p in zip(oracle_rows, prod_cams):
        d = {'camera': o['camera'], 'oracle_reason': O.NAMES[o['reason']], 'production_reason': p['reason_used']}
        if 'H' in o and p['reason_used'] == 'VALID':
            d.update(r_max_abs=absd(o['r'], p['r']), H_rel=rel(o['H'], p['H_x']), R_rel=rel(o['var'], p['R_diag']), uv_max_abs=absd([o['u'], o['v']], [p['u'], p['v']]))
            worst = max(worst, d['H_rel'], d['R_rel'], d['r_max_abs'])
        d['reason_agrees'] = d['oracle_reason'] == d['production_reason']
        out.append(d)
    return out, worst


def prediction_section(run, tr, pid):
    pre = tr['prediction']; p = pre['points'][str(pid)]
    s0, P0 = np.array(p['s_previous']), np.array(p['P_previous'])
    sm, Pm, F, Q = O.predict(s0, P0, pre['dt_s'], np.array(pre['q_c']))
    Pt = np.array(p['P_minus'])
    ev = np.linalg.eigvalsh(Pt)
    return {'t_previous_s': pre['t_previous_s'], 't_s': pre['t_s'], 'dt_s': pre['dt_s'], 'previous_frame': pre['previous_frame'], 'q_c': pre['q_c'],
            's_previous': s0, 'P_previous': P0, 'F': F, 'Q': Q, 's_minus': np.array(p['s_minus']), 'P_minus': Pt,
            'P_xx': Pt[:3, :3], 'P_xv': Pt[:3, 3:], 'P_vx': Pt[3:, :3], 'P_vv': Pt[3:, 3:],
            'eigenvalues': ev, 'condition_number': float(ev[-1]/ev[0]), 'position_sigma_m': np.sqrt(np.diag(Pt[:3, :3])),
            'velocity_sigma_m_s': np.sqrt(np.diag(Pt[3:, 3:])),
            'oracle_check': {'F_max_abs': absd(F, pre['F']), 'Q_max_abs': absd(Q, pre['Q']), 's_minus_max_abs': absd(sm, p['s_minus']), 'P_minus_rel': rel(Pm, Pt)},
            'velocity_from_position_gain_P_vx_Pxx_inv': Pt[3:, :3]@np.linalg.inv(Pt[:3, :3])}


def iterate_section(prob, tr, pid, run):
    """Every production iterate re-derived: (a) algebra from the dumped rows, (b) raw re-sample at the production state,
    (c) per-camera / channel attribution, (d) nonlinear replay of the step on the same rows."""
    its = []
    trace_its = [t['points'][str(pid)] for t in tr['iterations']]
    for j, t in enumerate(trace_its):
        if not t.get('active', True) and j > 0: continue
        s = np.array(t['state'])
        # (a) algebra from the dumped production rows
        Ixx, gx = np.zeros((3, 3)), np.zeros(3)
        for r in t['rows']:
            if r['reason'] != 'VALID': continue
            H, rr, w = np.array(r['H_x']), np.array(r['r']), 1/np.array(r['R_diag'])
            Ixx += H.T@(w[:, None]*H); gx += H.T@(w*rr)
        A = prob.Lam.copy(); A[:3, :3] += Ixx
        b = -prob.Lam@(s-prob.s_minus); b[:3] += gx
        d = np.linalg.solve(A, b)
        alg = {'A_rel': rel(A, t['A']), 'b_rel': rel(b, t['b']), 'delta_rel': rel(d, t['delta']), 'Ixx_rel': rel(Ixx, t['Ixx']), 'gx_rel': rel(gx, t['gx']),
               'step_delta_A_delta_oracle': float(d@b), 'step_production': t['step_delta_A_delta']}
        # (b) raw re-sample at the production state
        rows = prob.rows(s[:3])
        cmp, worst = compare_rows(rows, t['cameras'])
        A2, b2, d2, step2, Ixx2, gx2, q2 = prob.solve(s, rows)
        raw = {'per_camera': cmp, 'worst_row_disagreement': worst, 'delta_from_raw_rel': rel(d2, t['delta']), 'rTWr_rel': rel(q2, t['rTWr'])}
        # (c) attribution (exact additive split of delta)
        att = prob.attribution(s, rows, A2)
        # (d) nonlinear replay of the step s -> s + delta on the SAME cameras
        used = [o['camera'] for o in prob.used(rows)]
        s1 = s+d2
        c0, c1 = prob.cost_on(s, used), prob.cost_on(s1, used)
        lin_rgb = lin_d = 0.
        for o in prob.used(rows):
            z = (o['r']-o['H']@d2[:3])**2/o['var']; lin_rgb += float(z[:3].sum()); lin_d += float(z[3])
        rows1 = prob.rows(s1[:3])
        used1 = [o['camera'] for o in prob.used(rows1)]
        # per-row linearization discrepancy (actual residual at s+delta vs Gauss-Newton prediction r - H delta), in sigmas
        disc = {}
        for o in prob.used(rows):
            o1 = O.row(prob.imgs[o['camera']], prob.cams[o['camera']], s1[:3], prob.a, prob.noise, prob.policy, prob.P_minus[:3, :3], s1[:3]-prob.s_minus[:3])
            if 'r' not in o1: disc[o['camera']] = {'lost_at_new_state': O.NAMES[o1['reason']]}; continue
            rl = o['r']-o['H']@d2[:3]
            disc[o['camera']] = {'r_actual': o1['r'], 'r_gauss_newton': rl, 'discrepancy_sigma': (o1['r']-rl)/np.sqrt(o['var']),
                                 'pixel_moved_px': float(math.hypot(o1['u']-o['u'], o1['v']-o['v'])), 'cell_changed': o1['sample']['pixel00'] != o['sample']['pixel00'],
                                 'sampled_depth_change_m': o1['sample']['depth']-o['sample']['depth'], 'Z_change_m': o1['Z']-o['Z']}
        prior0, prior1 = prob.prior_cost(s), prob.prior_cost(s1)
        alpha = np.linspace(0, 1, 11)
        profile = []
        for al in alpha:
            sa = s+al*d2; ca = prob.cost_on(sa, used)
            gn = 0.
            for o in prob.used(rows):
                rl = o['r']-al*(o['H']@d2[:3]); gn += float((rl**2/o['var']).sum())
            profile.append({'alpha': float(al), 'prior': prob.prior_cost(sa), 'data_actual_same_rows': ca['rgb']+ca['depth'], 'data_gauss_newton': gn,
                            'rows_lost': len(ca['lost_rows'])})
        imap = iteration_map_jacobian(prob, s)
        its.append({'iteration': j, 'state': s, 'state_minus_prior': s-prob.s_minus, 'delta': d2, 'iteration_map': imap, 'delta_norm_x_m': float(np.linalg.norm(d2[:3])),
                    'delta_norm_v_m_s': float(np.linalg.norm(d2[3:])), 'step_delta_A_delta': step2, 'converged_after_this_step': bool(step2 < prob.tol),
                    'production': {'posterior_cost': t['posterior_cost'], 'cost_threshold': t['cost_threshold'], 'nrows': t['nrows']},
                    'algebra_from_dump': alg, 'raw_resample': raw, 'rows': summarize_rows(rows, prob.P_minus[:3, :3]), 'A': A2, 'b': b2,
                    'attribution': att, 'cameras_used': used, 'cameras_used_next_state': used1,
                    'row_set_change': {'missing_at_next': sorted(set(used)-set(used1)), 'added_at_next': sorted(set(used1)-set(used))},
                    'nonlinear_replay_same_rows': {'prior_cost_before': prior0, 'prior_cost_after': prior1,
                                                   'data_before': {'rgb': c0['rgb'], 'depth': c0['depth']}, 'data_after_actual': {'rgb': c1['rgb'], 'depth': c1['depth']},
                                                   'data_after_gauss_newton': {'rgb': lin_rgb, 'depth': lin_d}, 'rows_lost_at_next_state': c1['lost_rows'],
                                                   'total_before': prior0+c0['rgb']+c0['depth'], 'total_after_actual': prior1+c1['rgb']+c1['depth'],
                                                   'total_after_gauss_newton': prior1+lin_rgb+lin_d,
                                                   'objective_decreased': bool(prior1+c1['rgb']+c1['depth'] < prior0+c0['rgb']+c0['depth']),
                                                   'measurement_objective_decreased': bool(c1['rgb']+c1['depth'] < c0['rgb']+c0['depth']),
                                                   'per_row_linearization': disc},
                    'all_valid_rows_objective': {'at_state': {'rows': len(used), 'prior': prior0, 'data': sum(prob.data_cost(rows))},
                                                 'at_next_state': {'rows': len(used1), 'prior': prior1, 'data': sum(prob.data_cost(rows1))},
                                                 'note': 'different row sets: absolute sums are NOT comparable; see nonlinear_replay_same_rows'},
                    'line_profile_same_rows': profile})
    return its


def iteration_map_jacobian(prob, s, eps=(1e-8, 1e-6)):
    """dG/ds of the IEKF iteration map G(s) = s + delta(s) by central differences on the oracle (rows re-sampled at the
    perturbed states, frozen assignment). For an affine measurement G is constant (one-step convergence); its spectral
    radius measures the residual-weighted curvature A^-1 sum_a w_a r_a d2h_a: Gauss-Newton converges locally only if
    rho < 1. Differences whose row set or pixel cell changes are flagged (the bilinear gradient jumps at cell borders)."""
    G = lambda x: (lambda rows: (x+prob.solve(x, rows)[2], [(o['camera'], o['reason'], o.get('sample', {}).get('pixel00')) for o in rows]))(prob.rows(x[:3]))
    J = np.zeros((6, 6)); flags = []
    _, key0 = G(s)
    for k in range(6):
        e = np.zeros(6); e[k] = eps[0] if k < 3 else eps[1]
        gp, kp = G(s+e); gm, km = G(s-e)
        J[:, k] = (gp-gm)/(2*e[k])
        if kp != key0 or km != key0: flags.append(k)
    ev = np.linalg.eigvals(J)
    return {'jacobian': J, 'eigenvalues_abs': np.sort(np.abs(ev))[::-1], 'spectral_radius': float(np.abs(ev).max()),
            'position_block_spectral_radius': float(np.abs(np.linalg.eigvals(J[:3, :3])).max()), 'columns_crossing_a_cell_or_row_change': flags}


def total_attribution(its):
    """Sum of the per-iterate exact splits = split of s_final_iterate - s-, per camera / channel / prior."""
    tot = {}
    for it in its:
        for k, v in it['attribution'].items():
            tot[k] = tot.get(k, np.zeros(6))+np.asarray(v['delta'])
    return tot


def shares(tot, dx):
    """Projection of every term's position correction on the total correction direction (sums to 1)."""
    n2 = float(dx@dx)
    if n2 == 0: return {}
    return {k: float(v[:3]@dx/n2) for k, v in tot.items()}


def endpoint_cross_camera(run, pid, cams):
    """M1: the canonical point at its OWN instant seen by every camera (nothing moved): colour / depth residuals, image
    gradient, mask border distance, viewing angle and the source pixel's neighbourhood."""
    from scipy.ndimage import distance_transform_edt
    k0 = run.fman['canonical_frame']
    imgs = run.images(k0)
    x0, a = run.can['canonical_xyz'][pid], run.can['canonical_rgb'][pid]
    f, cs, pix = (int(t) for t in run.can['original_source_ids'][pid])
    W = cams[cs].W; us, vs = pix % W, pix//W
    src = imgs[cs]
    nbh = {'rgb_3x3': src.rgb[max(vs-1, 0):vs+2, max(us-1, 0):us+2], 'depth_3x3_m': src.depth[max(vs-1, 0):vs+2, max(us-1, 0):us+2],
           'mask_3x3': src.mask[max(vs-1, 0):vs+2, max(us-1, 0):us+2]}
    gsrc = None
    if 0 < us < W-1 and 0 < vs < cams[cs].H-1:
        gsrc = float(np.linalg.norm(np.stack([(src.rgb[vs, us+1]-src.rgb[vs, us-1])/2, (src.rgb[vs+1, us]-src.rgb[vs-1, us])/2])))
    # surface normal from the source depth map (back-projected neighbours)
    def bp(u, v):
        D = src.depth[v, u]
        return cams[cs].R.T@(D*np.array([(u-cams[cs].cx)/cams[cs].fx, (v-cams[cs].cy)/cams[cs].fy, 1.])-cams[cs].t)
    normal = None
    try:
        if all(src.mask[v, u] and src.raw_valid[v, u] for u, v in ((us+1, vs), (us-1, vs), (us, vs+1), (us, vs-1))):
            n = np.cross(bp(us+1, vs)-bp(us-1, vs), bp(us, vs+1)-bp(us, vs-1)); normal = n/np.linalg.norm(n)
    except IndexError:
        pass
    prob = O.Problem(cams, imgs, a, np.r_[x0, 0, 0, 0], np.eye(6), [O.VALID]*len(cams), run.noise, run.policy, 1, 0.)
    rows = []
    for c, cam in enumerate(cams):
        o = O.row(imgs[c], cam, x0, a, run.noise, {**run.policy, 'row_innovation_gate': 'depth_front_only', 'visibility_n_sigma': float('inf')},
                  run.can['canonical_source_covariance'][pid], np.zeros(3))
        d = {'camera': c, 'is_source_camera': c == cs, 'reason': O.NAMES[o['reason']], 'u': o['u'], 'v': o['v'], 'Z': o['Z']}
        if 'r' in o:
            dist = distance_transform_edt(imgs[c].mask)
            pu, pv = int(round(o['u'])), int(round(o['v']))
            view = cam.centre()-x0; view /= np.linalg.norm(view)
            d.update(sampled_rgb=o['sample']['rgb'], sampled_depth_m=o['sample']['depth'], r=o['r'], r_over_sigma=o['r']/np.sqrt(o['var']),
                     rgb_chi2=float((o['r'][:3]**2/o['var'][:3]).sum()), depth_chi2=float(o['r'][3]**2/o['var'][3]),
                     image_gradient_rgb_per_px=float(np.linalg.norm(o['sample']['grad_rgb'])), depth_gradient_m_per_px=float(np.linalg.norm(o['sample']['grad_depth'])),
                     mask_border_distance_px=float(dist[pv, pu]) if 0 <= pv < cam.H and 0 <= pu < cam.W else None,
                     viewing_angle_deg=None if normal is None else float(np.degrees(np.arccos(min(1., abs(normal@view))))),
                     depth_discrepancy_Z_minus_D_m=float(o['r'][3]),
                     ray_position_discrepancy_m=float(abs(o['r'][3])*np.linalg.norm(cam.ray_world(o['u'], o['v']))),
                     source_covariance_sigma_along_ray_m=float(math.sqrt(cam.R[2]@run.can['canonical_source_covariance'][pid]@cam.R[2])))
        rows.append(d)
    return {'canonical_frame': k0, 'source_camera': cs, 'source_pixel_u_v': [us, vs], 'canonical_xyz': x0, 'canonical_rgb': a,
            'source_covariance': run.can['canonical_source_covariance'][pid], 'source_neighbourhood': nbh, 'source_rgb_gradient_per_px': gsrc,
            'surface_normal_from_source_depth': normal, 'cameras': rows,
            'note': 'same instant, same XYZ: no temporal motion can explain these residuals (occlusion, border mixing, view-dependent '
                    'reflectance, calibration or pixel correspondence error remain).'}


def tail_section(run, pid, k_div):
    """K3: every later prediction-only step is x' = x + dt v, v' = v, P' = F P F^T + Q exactly (signed dt)."""
    order = run.fman['processing_order']; times = np.asarray(run.fman['timestamps_seconds'])
    st, X = run.mm('point_status'), run.mm('posterior_xyz')
    i0 = order.index(k_div)
    tr = run.trace(k_div)
    s_last = np.array(tr['final'][str(pid)]['s_plus']); t_last = times[k_div]
    first_unobserved, worst_pos, steps, per_frame = None, 0., [], []
    ballistic = True
    for k in order[i0+1:]:
        if st[k, pid] != 2: ballistic = False; per_frame.append({'frame': k, 'status': int(st[k, pid])}); break
        if first_unobserved is None: first_unobserved = k
        pred = s_last[:3]+(times[k]-t_last)*s_last[3:]
        e = float(np.abs(np.asarray(X[k, pid])-pred).max()); worst_pos = max(worst_pos, e)
        d = {'frame': k, 'status': 'PREDICTION_ONLY', 'x_minus_ballistic_max_abs_m': e}
        t = run.trace(k)
        if t is not None and str(pid) in t['prediction']['points']:
            p = t['prediction']['points'][str(pid)]
            sm, Pm, _, _ = O.predict(np.array(p['s_previous']), np.array(p['P_previous']), t['prediction']['dt_s'], np.array(t['prediction']['q_c']))
            fin = t['final'][str(pid)]
            d.update(s_minus_vs_F_s_max_abs=absd(sm, p['s_minus']), P_minus_vs_FPFt_Q_rel=rel(Pm, p['P_minus']),
                     s_plus_equals_s_minus=bool(np.array_equal(fin['s_plus'], p['s_minus'])), P_plus_equals_P_minus=bool(np.array_equal(fin['P_plus'], p['P_minus'])),
                     velocity_unchanged_max_abs=absd(np.array(fin['s_plus'])[3:], s_last[3:]), dt_s=t['prediction']['dt_s'],
                     reasons=[c['reason_used'] for c in t['iterations'][0]['points'][str(pid)]['cameras']])
        per_frame.append(d)
    cr = run.mm('camera_reason')
    return {'last_observed_frame': k_div, 'state_at_last_update': s_last, 'first_prediction_only_frame': first_unobserved,
            'first_prediction_only_reasons_from_memmap': None if first_unobserved is None else [O.NAMES[int(r)] for r in cr[first_unobserved, :, pid]],
            'remaining_steps_prediction_only': ballistic, 'max_abs_position_vs_x_last_plus_dt_v_m': worst_pos, 'frames': per_frame,
            'note': 'posterior_xyz is float64 on disk; s_plus from the float64 trace.'}


def chain_replay(run, pid, k_div, cams, override=None, noise=None):
    """Replay this point alone from the canonical endpoint to k_div with every frame's production frozen assignment
    (other points held at their production states) and an optional one-variable change: override = ('pvv', P) or
    ('qc', Q). Needs a trace of every frame up to k_div."""
    order = run.fman['processing_order']
    seq = order[1:order.index(k_div)+1]
    trs = {k: run.trace(k) for k in seq}
    if any(t is None or str(pid) not in t['prediction']['points'] for t in trs.values()): return None
    first = trs[seq[0]]['prediction']['points'][str(pid)]
    s, P = np.array(first['s_previous']), np.array(first['P_previous'])
    if override and override[0] == 'pvv': P = P.copy(); P[3:, 3:] = override[1]
    frames = []
    for k in seq:
        t = trs[k]; qc = np.array(t['prediction']['q_c']) if not (override and override[0] == 'qc') else override[1]
        sm, Pm, _, _ = O.predict(s, P, t['prediction']['dt_s'], qc)
        it0 = t['iterations'][0]['points'][str(pid)]
        recheck = None
        if run.policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion':
            recheck = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in t['final'][str(pid)]['cameras_final_iterate']]
        prob = O.Problem(cams, run.images(k), run.can['canonical_rgb'][pid], sm, Pm, frozen_codes(it0['cameras']), noise or run.noise, run.policy,
                         run.fman['max_iekf_iters'], run.fman['step_tolerance'], recheck)
        res = prob.run(run.policy.get('update_acceptance', 'final_iterate'))
        prod = t['final'][str(pid)]
        frames.append({'frame': k, 's_minus': sm, 'P_xv_minus': Pm[:3, 3:], 's_plus': res['s_plus'], 'observed': res['observed'],
                       'production_s_plus_max_abs': absd(res['s_plus'], prod['s_plus']), 'production_P_plus_rel': rel(res['P_plus'], prod['P_plus'])})
        s, P = res['s_plus'], res['P_plus']
        last = (prob, res)
    return frames, last


def variant_summary(prob, res, base_dx=None):
    s_m = prob.s_minus; d = res['s_plus']-s_m
    Pxx = prob.P_minus[:3, :3]
    final_rows = res['rows_accepted']
    rgb, dep = prob.data_cost(final_rows)
    jos = prob.joseph(final_rows) if res['observed'] else None
    return {'observed': res['observed'], 'acceptance_code': res['acceptance_code'], 'iterations': len(res['history']), 'converged': res['converged'],
            'delta_x': d[:3], 'delta_x_norm_m': float(np.linalg.norm(d[:3])), 'delta_v': d[3:], 'delta_v_norm_m_s': float(np.linalg.norm(d[3:])),
            'speed_after_m_s': float(np.linalg.norm(res['s_plus'][3:])),
            'position_mahalanobis2_prior': float(d[:3]@np.linalg.solve(Pxx, d[:3])), 'state_mahalanobis2_prior': float(d@prob.Lam@d),
            'final_rows': len(prob.used(final_rows)), 'final_data_cost_rgb': rgb, 'final_data_cost_depth': dep,
            'prior_P_xv': prob.P_minus[:3, 3:], 'velocity_gain_P_vx_Pxx_inv': prob.P_minus[3:, :3]@np.linalg.inv(Pxx),
            'gain_K_frobenius': None if jos is None else float(np.linalg.norm(jos['K'])), 'gain_K_velocity_rows_frobenius': None if jos is None else float(np.linalg.norm(jos['K'][3:])),
            'cameras_used_first_iterate': [o['camera'] for o in prob.used(res['history'][0]['rows'])],
            'iterate_offset_from_prior_m': [float(np.linalg.norm(h['s'][:3]-s_m[:3])) for h in res['history']]+[float(np.linalg.norm(res['s_final_iterate'][:3]-s_m[:3]))],
            'iterate_rows': [h['nrows'] for h in res['history']],
            'information_eigenvalues_final': np.linalg.eigvalsh(prob.info(final_rows)[0])}


def factorial(run, pid, k_div, tr, cams, imgs, a):
    base = problem_from_trace(run, k_div, pid, tr, imgs, cams)
    rule = run.policy.get('update_acceptance', 'final_iterate')
    out = {}
    r0 = base.run(rule); out['original_full_rgb_depth'] = variant_summary(base, r0)
    first_cams = [o['camera'] for o in base.used(r0['history'][0]['rows'])]
    all_cams = sorted({o['camera'] for h in r0['history'] for o in base.used(h['rows'])} | {o['camera'] for o in base.used(r0['rows_final'])})
    for name, ch in (('rgb_only', (1, 1, 1, 0)), ('depth_only', (0, 0, 0, 1))):
        p = problem_from_trace(run, k_div, pid, tr, imgs, cams, channels=ch); out[name] = variant_summary(p, p.run(rule))
    for c in all_cams:
        m = np.ones(len(cams), bool); m[c] = False
        p = problem_from_trace(run, k_div, pid, tr, imgs, cams, cam_mask=m); out[f'leave_out_camera_{c}'] = variant_summary(p, p.run(rule))
    p = problem_from_trace(run, k_div, pid, tr, imgs, cams, fixed_rows=set(first_cams)); out['fixed_first_iterate_row_set'] = variant_summary(p, p.run(rule))
    out['_notes'] = {'rule': rule, 'cameras_admitted_first_iterate': first_cams, 'cameras_admitted_any_iterate': all_cams,
                     'channel_variants': 'rows admitted exactly as in production (the row gate sees the full 4-vector); only the listed channels enter A, b and the cost',
                     'leave_out': 'diagnostic only, not a gate', 'fixed_row_set': 'cameras admitted at the first iterate only; a camera lost later stays lost'}
    if a.alt_noise:
        noise_alt, desc = alt_noise(run, a.alt_noise)
        p = problem_from_trace(run, k_div, pid, tr, imgs, cams, noise=noise_alt)
        out['noise_alternative'] = {**variant_summary(p, p.run(rule)), 'noise': desc, 'noise_original': run.noise.describe(),
                                    'note': 'same s-, P- (the endpoint covariance is NOT recomputed with this R: one variable at a time)'}
    chains = {}
    base_chain = chain_replay(run, pid, k_div, cams)
    if base_chain is None:
        chains['_unavailable'] = 'needs --debug-frames covering every frame from the canonical endpoint to the divergent frame'
    else:
        chains['original'] = {'frames': base_chain[0], 'final': variant_summary(*base_chain[1])}
        from scene.noise_calibration import velocity_prior
        for v99 in a.alt_speed99 or []:
            Pvv, sv = velocity_prior(v99)
            c = chain_replay(run, pid, k_div, cams, ('pvv', Pvv))
            chains[f'P_vv_from_speed99_{v99}'] = {'sigma_v_m_s': sv, 'frames': c[0], 'final': variant_summary(*c[1])}
        for q in a.alt_qc or []:
            c = chain_replay(run, pid, k_div, cams, ('qc', q*np.eye(3)))
            chains[f'q_c_{q}'] = {'frames': c[0], 'final': variant_summary(*c[1])}
        chains['_note'] = ('one point replayed from the canonical endpoint with every frame\'s production frozen pixel assignment (the other points '
                           'held at their production states); the alternative P_vv / q_c values are the ones given on the command line')
    out['covariance_chain_replays'] = chains
    return out


def alt_noise(run, spec):
    if spec == 'measured':
        cache = run.root/'root_cause'/'noise_calibration_measured.json'           # computed once per run, reused by every point
        if cache.exists(): rep = json.loads(cache.read_text())
        else:
            import torch, copy
            from scene.noise_calibration import calibrate
            from scene.config import load_config
            cfg = load_config(None if run.man['arguments'].get('noise_config') is None else Path(run.man['arguments']['noise_config']))
            _, pol, rep = calibrate(run.data, cfg, copy.deepcopy(run.policy), 'cuda' if torch.cuda.is_available() else 'cpu')
            cache.parent.mkdir(parents=True, exist_ok=True); write_json(cache, rep); rep = json.loads(cache.read_text())
        r = rep['result']
    else:
        rep = json.loads(Path(spec).read_text()); r = rep['result']
    n = O.Noise(r['rgb_variance'], r['depth_variance_m2'], r['depth_squared_variance'], r['rgb_model_variance'], r['depth_model_variance_m2'],
                run.policy['interpolation_variance'])
    return n, {**n.describe(), 'source': spec, 'calibration_report': rep}


def verdicts(A):
    """The spec's interpretation table, evaluated on the numbers (classification only)."""
    its, fin = A['iterates'], A['final']
    v = {}
    worst_alg = max(max(i['algebra_from_dump'][k] for k in ('A_rel', 'b_rel', 'delta_rel')) for i in its)
    worst_raw = max(i['raw_resample']['worst_row_disagreement'] for i in its)
    rep = A['oracle_replay']
    v['algebra'] = {'max_relative_mismatch_dump': worst_alg, 'max_row_mismatch_raw_resample': worst_raw, 'replay_s_plus_max_abs_m': rep['s_plus_max_abs'],
                    'replay_P_plus_rel': rep['P_plus_rel'], 'joseph_vs_information_max_abs': fin['joseph']['max_abs_difference'] if fin.get('joseph') else None,
                    'free_running_replay_divergence_per_iterate_m': rep['free_running_state_divergence_per_iterate_m'],
                    'holds': bool(worst_alg < TOL_ALGEBRA and worst_raw < 1e-9 and (fin['joseph'] is None or fin['joseph']['max_abs_difference'] < 1e-10)),
                    'meaning': 'at every production state the 6x6 solve, the raw-pixel rows and the Joseph covariance match an independent float64 '
                               'oracle (no numerical / sign / accumulator bug); a free-running replay can still drift if the iteration map amplifies round-off'}
    off = [float(np.linalg.norm(np.asarray(i['state_minus_prior'])[:3])) for i in its]
    returns = [i['iteration'] for i, o in zip(its, off) if o < 1e-12 and i['iteration'] > 0]
    period = returns[0] if returns else None
    v['limit_cycle'] = {'iterate_offsets_from_prior_m': off, 'iterates_back_at_prior': returns, 'period': period,
                        'accepted_phase': None if period is None else len(its) % period,
                        'holds': bool(returns),
                        'meaning': 'an iterate with no valid row steps exactly back to s- (b = -Lam (s - s-)); the iteration then repeats, and the '
                                   'accepted final iterate is the phase reached when max_iekf_iters stops it'}
    border = [(i['iteration'], r['camera'], r['cell_border_distance_px']) for i in its for r in i['rows'] if r.get('cell_border_distance_px', 1) < 1e-9 and r['reason'] == 'VALID']
    v['cell_corner_knife_edge'] = {'admitted_rows_within_1e-9_px_of_a_cell_border': border, 'holds': bool(border),
                                   'meaning': 'the projection sits on a bilinear cell border (a static point at its prior projects onto its own source pixel '
                                              'centre): floor(u), hence the 2x2 cell, gradient and every later iterate, is decided by round-off'}
    r0 = [r for r in its[0]['rows'] if r['reason'] == 'VALID']
    v['gauss_newton_extrapolation'] = {'prior_sigma_px_major_per_admitted_row': {r['camera']: r.get('prior_sigma_px_major') for r in r0},
                                       'depth_gradient_m_per_px': {r['camera']: r.get('depth_gradient_m_per_px') for r in r0},
                                       'meaning': 'the bilinear gradient is valid inside one pixel cell; a prior sigma of several pixels means the '
                                                  'Gauss-Newton quadratic is extrapolated across many cells (edges, surfaces)'}
    rho = [i['iteration_map']['spectral_radius'] for i in its]
    v['iteration_map'] = {'spectral_radius_per_iterate': rho, 'expansive_iterates': [i['iteration'] for i in its if i['iteration_map']['spectral_radius'] > 1],
                          'meaning': 'rho(dG/ds) > 1: the Gauss-Newton iteration is locally expansive (residual-weighted curvature dominates the '
                                     'information): it cannot converge there, and the accepted iterate depends on where the iteration count stops'}
    cid = fin['coupling_identity']
    v['velocity_from_position_coupling'] = {**cid, 'holds': bool(cid['max_abs_m_s'] < 1e-6),
                                            'meaning': 'delta_v = P_vx Pxx^-1 delta_x exactly: the velocity jump is the Kalman covariance coupling of the position correction, not a separate velocity measurement'}
    inc = [i['iteration'] for i in its if not i['nonlinear_replay_same_rows']['objective_decreased']]
    minc = [i['iteration'] for i in its if not i['nonlinear_replay_same_rows']['measurement_objective_decreased']]
    changes = [i['iteration'] for i in its if i['row_set_change']['missing_at_next'] or i['row_set_change']['added_at_next']]
    big = []
    for i in its:
        for c, d in i['nonlinear_replay_same_rows']['per_row_linearization'].items():
            if 'discrepancy_sigma' in d and np.max(np.abs(d['discrepancy_sigma'])) > 3: big.append((i['iteration'], int(c), float(np.max(np.abs(d['discrepancy_sigma'])))))
    v['linearization'] = {'iterates_where_total_objective_increased': inc, 'iterates_where_measurement_objective_increased': minc,
                          'iterates_with_row_set_change': changes, 'rows_with_linearization_error_over_3_sigma': big,
                          'holds': bool(inc or changes or big),
                          'meaning': 'the Gauss-Newton model of the step is wrong on the same rows (cell / surface / edge crossing) or the row set changes'}
    sh = A['attribution']['position_correction_shares']
    rgb = sum(x for k, x in sh.items() if k.endswith(('_R', '_G', '_B'))); dep = sum(x for k, x in sh.items() if k.endswith('_D'))
    v['dominant_channel'] = {'rgb_share': rgb, 'depth_share': dep, 'prior_share': sh.get('prior', 0.),
                             'per_camera': {c: sum(x for k, x in sh.items() if k.startswith(f'cam{c}_')) for c in range(8) if any(k.startswith(f'cam{c}_') for k in sh)}}
    ep = A['endpoint_cross_camera']['cameras']
    other = [c for c in ep if not c['is_source_camera'] and 'rgb_chi2' in c]
    v['invariant_appearance'] = {'same_instant_other_camera_rgb_chi2_per_channel': [c['rgb_chi2']/3 for c in other],
                                 'same_instant_other_camera_rgb_r': [c['r'][:3] for c in other],
                                 'violated': bool(other and np.median([c['rgb_chi2']/3 for c in other]) > 9),
                                 'meaning': 'one source pixel colour is not what the other cameras measure at the same XYZ and instant (median per-channel chi2 > 9 = beyond 3 sigma)'}
    v['surface_identity_same_instant'] = {'same_instant_other_camera_depth_r_over_sigma': [c['r_over_sigma'][3] for c in other],
                                          'violated': bool(other and np.median([abs(c['r_over_sigma'][3]) for c in other]) > 3)}
    nis = its[0]['production']
    v['covariance_consistency'] = {'prior_iterate_cost_per_dof': nis['posterior_cost']/max(nis['nrows']*4, 1),
                                   'note': 'J at s- = r^T R^-1 r over the prior rows; ~1 per dof if R describes the residuals'}
    t = A['tail']
    v['ballistic_tail'] = {'max_abs_m': t['max_abs_position_vs_x_last_plus_dt_v_m'], 'exact': bool(t['max_abs_position_vs_x_last_plus_dt_v_m'] < 1e-9),
                           'meaning': 'expected constant-velocity propagation after loss of observation (not itself a bug)'}
    return v


def cmd_analyze(a):
    run = Run(a.run, a.direction)
    pid = a.point
    traced = sorted(int(f.stem.split('_')[-1]) for f in (run.free/'numeric').glob('trace_frame_*.json'))
    if not traced: raise SystemExit('no trace_frame_*.json: rerun train.py with --debug-point-ids and --debug-frames')
    if str(pid) not in run.trace(traced[0])['final']: raise SystemExit(f'point {pid} is not traced in {run.free} (debug ids {run.fman["debug_point_ids"]})')
    k_div = a.frame
    if k_div is None:
        v, st = run.mm('velocity'), run.mm('point_status'); order = run.fman['processing_order']
        dv = [(float(np.linalg.norm(np.asarray(v[k, pid], float)-np.asarray(v[p, pid], float))), k) for p, k in zip(order, order[1:]) if st[k, pid] == 1]
        k_div = max(dv)[1]
    out = out_dir(a.run, a.name or f'{a.direction}_point{pid}_frame{k_div}')
    print(f'[root-cause] point {pid}, frame {k_div} -> {out}', flush=True)
    tr = run.trace(k_div)
    if tr is None: raise SystemExit(f'frame {k_div} not traced (traced: {traced})')
    cams, imgs = run.cams(), run.images(k_div)
    prob = problem_from_trace(run, k_div, pid, tr, imgs, cams)
    src = run.can['original_source_ids'][pid]
    A = {'provenance': {'tool': provenance_tool(), 'run_git_commit': run.man.get('git_commit'), 'run_source_hashes': run.man.get('source_hashes'),
                        'run_arguments': run.man['arguments'], 'policy': run.policy, 'noise_manifest': run.man['noise'],
                        'noise_calibration_file': run.man.get('noise_calibration'), 'free_manifest': run.fman,
                        'source_files_changed_since_run': [f for f, h in (run.man.get('source_hashes') or {}).items()
                                                          if (REPO/f).exists() and hashlib.sha256((REPO/f).read_bytes()).hexdigest() != h],
                        'frame_input_fingerprints': json.loads((run.free/'numeric'/f'frame_{k_div:04d}.json').read_text()).get('source_fingerprints'),
                        'frame_input_sha256': [{'camera': c, 'rgb_float32': hashlib.sha256(np.ascontiguousarray(imgs[c].rgb.astype(np.float32)).tobytes()).hexdigest(),
                                                'depth_float32': hashlib.sha256(np.ascontiguousarray(imgs[c].depth.astype(np.float32)).tobytes()).hexdigest(),
                                                'mask': hashlib.sha256(np.ascontiguousarray(imgs[c].mask).tobytes()).hexdigest(),
                                                'raw_valid': hashlib.sha256(np.ascontiguousarray(imgs[c].raw_valid).tobytes()).hexdigest()} for c in range(len(cams))],
                        'cameras': [{'name': c.name, 'fx_fy_cx_cy': [c.fx, c.fy, c.cx, c.cy], 'R_cw': c.R, 't_cw': c.t, 'near': c.near, 'W': c.W, 'H': c.H} for c in cams]},
         'identity': {'point': pid, 'direction': a.direction, 'pass': run.fman['pass'], 'canonical_frame': run.fman['canonical_frame'],
                      'subsampled_to_full_index': int(run.can['subsampled_to_full_index'][pid]),
                      'source_frame_camera_pixel': [int(t) for t in src], 'source_pixel_u_v': [int(src[2]) % cams[int(src[1])].W, int(src[2])//cams[int(src[1])].W],
                      'canonical_xyz': run.can['canonical_xyz'][pid], 'canonical_rgb': run.can['canonical_rgb'][pid],
                      'canonical_source_covariance': run.can['canonical_source_covariance'][pid], 'timestamps_s': run.fman['timestamps_seconds'],
                      'divergent_frame': k_div, 'noise_used': run.noise.describe()}}
    A['prediction'] = prediction_section(run, tr, pid)
    print('[root-cause] iterates', flush=True)
    A['iterates'] = iterate_section(prob, tr, pid, run)
    res = prob.run(run.policy.get('update_acceptance', 'final_iterate'))
    fin = tr['final'][str(pid)]
    replay_div = [absd(h['s'], tr['iterations'][j]['points'][str(pid)]['state']) for j, h in enumerate(res['history']) if j < len(tr['iterations'])]
    A['oracle_replay'] = {'free_running_state_divergence_per_iterate_m': replay_div,
                          'note': 'the oracle iterated from s- on its own; at every PRODUCTION state it agrees to round-off (iterates[*].raw_resample). '
                                  'Growth of this divergence = amplification of round-off by the iteration map (see iterates[*].iteration_map).',
'accepted_linearization_state_max_abs_m': absd(res['s_plus'] if res['observed'] else res['s_final_iterate'], fin['linearization_state']),
                          's_plus_max_abs': absd(res['s_plus'], fin['s_plus']), 'P_plus_rel': rel(res['P_plus'], fin['P_plus']),
                          'acceptance_oracle': res['acceptance_code'], 'acceptance_production': fin['acceptance'], 'observed_oracle': res['observed'],
                          'iterations_oracle': len(res['history']), 'iterations_production': len(tr['iterations']), 'converged_oracle': res['converged'],
                          'converged_production': fin['converged']}
    s_plus, s_minus = np.array(fin['s_plus']), np.array(fin['s_minus'])
    d = s_plus-s_minus; Pm = np.array(fin['P_minus'])
    jos = prob.joseph(res['rows_accepted']) if res['observed'] else None
    pred_dv = Pm[3:, :3]@np.linalg.solve(Pm[:3, :3], d[:3])
    rows0 = prob.rows(s_minus[:3]); rowsf = res['rows_accepted']
    common = sorted(set(o['camera'] for o in prob.used(rows0)) & set(o['camera'] for o in prob.used(rowsf)))
    fd = {}
    for c in [o['camera'] for o in prob.used(rowsf)]:
        fd[c] = O.finite_difference_H(prob, s_plus[:3], c)
    A['final'] = {'s_minus': s_minus, 's_final_iterate': res['s_final_iterate'], 's_plus': s_plus, 'P_minus': Pm, 'P_plus': np.array(fin['P_plus']),
                  'delta': d, 'delta_x_norm_m': float(np.linalg.norm(d[:3])), 'delta_v_norm_m_s': float(np.linalg.norm(d[3:])), 'speed_after_m_s': float(np.linalg.norm(s_plus[3:])),
                  'converged': fin['converged'], 'acceptance': fin['acceptance'], 'valid_rows': fin['valid_rows'], 'production_posterior_cost': fin['posterior_cost'],
                  'production_cost_threshold': fin['cost_threshold'],
                  'C_prior_state_mahalanobis2': float(d@prob.Lam@d), 'position_mahalanobis2': float(d[:3]@np.linalg.solve(Pm[:3, :3], d[:3])),
                  'coupling_identity': {'delta_v': d[3:], 'P_vx_Pxx_inv_delta_x': pred_dv, 'max_abs_m_s': absd(d[3:], pred_dv)},
                  'posterior_information_eigenvalues': np.linalg.eigvalsh(prob.Lam+np.pad(prob.info(rowsf)[0], ((0, 3), (0, 3)))),
                  'measurement_information_eigenvalues_xx': np.linalg.eigvalsh(prob.info(rowsf)[0]),
                  'joseph': None if jos is None else {k: jos[k] for k in ('K', 'P_joseph', 'max_abs_difference')},
                  'joseph_vs_production_P_plus_rel': None if jos is None else rel(jos['P_joseph'], fin['P_plus']),
                  'data_cost': {'at_prior_all_rows': {'rows': len(prob.used(rows0)), 'rgb': prob.data_cost(rows0)[0], 'depth': prob.data_cost(rows0)[1]},
                                'at_accepted_all_rows': {'rows': len(prob.used(rowsf)), 'rgb': prob.data_cost(rowsf)[0], 'depth': prob.data_cost(rowsf)[1]},
                                'common_cameras': common, 'at_prior_common': prob.cost_on(s_minus, common), 'at_accepted_common': prob.cost_on(s_plus, common)},
                  'rows_accepted': summarize_rows(rowsf, Pm[:3, :3]), 'H_finite_difference_at_accepted': fd}
    tot = total_attribution(A['iterates'])
    A['attribution'] = {'total_delta_by_term': tot, 'position_correction_shares': shares(tot, res['s_final_iterate'][:3]-s_minus[:3]),
                        'note': 'exact: delta_j = A_j^-1[-Lam e_j] + sum_{c,a} A_j^-1[H_ca^T r_ca / R_ca, 0]; summed over iterates = s_final_iterate - s-'}
    print('[root-cause] endpoint cross-camera / tail / factorial', flush=True)
    A['endpoint_cross_camera'] = endpoint_cross_camera(run, pid, cams)
    A['tail'] = tail_section(run, pid, k_div)
    A['factorial'] = factorial(run, pid, k_div, tr, cams, imgs, a)
    A['verdicts'] = verdicts(A)
    write_json(out/'analysis.json', A)
    np.savez_compressed(out/'arrays.npz', s_minus=s_minus, P_minus=Pm, s_plus=s_plus, P_plus=np.array(fin['P_plus']), Lam=prob.Lam,
                        states=np.array([i['state'] for i in A['iterates']]), deltas=np.array([i['delta'] for i in A['iterates']]),
                        A=np.array([i['A'] for i in A['iterates']]), b=np.array([i['b'] for i in A['iterates']]))
    (out/'report.md').write_text(report(A))
    print(f'[root-cause] wrote {out}/report.md, analysis.json, arrays.npz', flush=True)


# ---------------------------------------------------------------------------------------------------------------- report
def f(x, p=4):
    if x is None: return '—'
    if isinstance(x, (list, tuple, np.ndarray)): return '[' + ', '.join(f(v, p) for v in np.asarray(x, float).ravel()) + ']'
    x = float(x)
    return f'{x:.{p}g}'


def report(A):
    I, P, F, V = A['identity'], A['prediction'], A['final'], A['verdicts']
    L = [f"# Root cause: {I['pass']} point {I['point']}, frame {I['divergent_frame']}", '',
         f"Run commit `{A['provenance']['run_git_commit']}`; analysis tool at `{A['provenance']['tool']['git_head'][:10]}`, dirty patch sha256 "
         f"`{A['provenance']['tool']['dirty_patch_sha256'][:16]}`. Policy: update_acceptance `{A['provenance']['policy'].get('update_acceptance')}`, "
         f"row gate `{A['provenance']['policy'].get('row_innovation_gate')}`, final visibility `{A['provenance']['policy'].get('final_visibility')}`. "
         f"Source files changed since the run: {A['provenance']['source_files_changed_since_run'] or 'none'}.", '',
         f"Source: frame {I['source_frame_camera_pixel'][0]}, camera {I['source_frame_camera_pixel'][1]}, pixel (u, v) = {tuple(I['source_pixel_u_v'])}; "
         f"canonical XYZ {f(I['canonical_xyz'])}, RGB {f(I['canonical_rgb'], 3)}.", '',
         '## Verdicts (spec interpretation table)', '', '| Hypothesis | Holds | Evidence |', '|---|---|---|']
    al = V['algebra']
    L.append(f"| Algebra / implementation bug | {'no bug' if al['holds'] else '**MISMATCH**'} | dump-vs-trace rel {f(al['max_relative_mismatch_dump'], 2)}, raw re-sample rows {f(al['max_row_mismatch_raw_resample'], 2)}, "
             f"replay s+ {f(al['replay_s_plus_max_abs_m'], 2)} m, P+ rel {f(al['replay_P_plus_rel'], 2)}, Joseph {f(al['joseph_vs_information_max_abs'], 2)} |")
    im = V['iteration_map']
    L.append(f"| Gauss–Newton iteration map expansive (ρ(∂G/∂s) > 1) | {'yes at '+str(im['expansive_iterates']) if im['expansive_iterates'] else 'no'} | ρ per iterate {f(im['spectral_radius_per_iterate'], 3)}; "
             f"free-running replay divergence {f(al['free_running_replay_divergence_per_iterate_m'], 2)} m |")
    lc, ke, gn = V['limit_cycle'], V['cell_corner_knife_edge'], V['gauss_newton_extrapolation']
    L.append(f"| Limit cycle through s⁻ (zero-row iterate) | {'yes, period '+str(lc['period'])+', accepted phase '+str(lc['accepted_phase']) if lc['holds'] else 'no'} | ‖s_j − s⁻‖ {f(lc['iterate_offsets_from_prior_m'], 3)} m |")
    L.append(f"| Admitted row on a bilinear cell border (round-off decides the cell) | {'yes' if ke['holds'] else 'no'} | (iterate, camera, px) {ke['admitted_rows_within_1e-9_px_of_a_cell_border'][:6]} |")
    L.append(f"| Gauss–Newton extrapolated over many pixel cells | — | prior 1σ in px per admitted camera {', '.join(f'{k}: {f(x, 3)}' for k, x in gn['prior_sigma_px_major_per_admitted_row'].items())}; ‖∇D‖ m/px {', '.join(f'{k}: {f(x, 3)}' for k, x in gn['depth_gradient_m_per_px'].items())} |")
    c = V['velocity_from_position_coupling']
    L.append(f"| Velocity = covariance coupling of the position correction | {'yes' if c['holds'] else '**no**'} | δv {f(c['delta_v'])} vs P_vx P_xx⁻¹ δx {f(c['P_vx_Pxx_inv_delta_x'])} (max diff {f(c['max_abs_m_s'], 2)} m/s) |")
    li = V['linearization']
    L.append(f"| Nonlinear linearization / changing rows | {'yes' if li['holds'] else 'no'} | objective up at iterates {li['iterates_where_total_objective_increased']}, measurement objective up at {li['iterates_where_measurement_objective_increased']}, "
             f"row-set changes at {li['iterates_with_row_set_change']}, rows with GN error > 3σ: {li['rows_with_linearization_error_over_3_sigma'][:6]} |")
    dc = V['dominant_channel']
    L.append(f"| Dominant contribution to the position correction | — | RGB {f(dc['rgb_share'], 3)}, depth {f(dc['depth_share'], 3)}, prior {f(dc['prior_share'], 3)}; per camera {', '.join(f'{k}: {f(v, 3)}' for k, v in dc['per_camera'].items())} |")
    ia = V['invariant_appearance']
    L.append(f"| Invariant single-pixel appearance violated (same instant) | {'yes' if ia['violated'] else 'no'} | other cameras' per-channel χ² {f(ia['same_instant_other_camera_rgb_chi2_per_channel'], 3)} |")
    si = V['surface_identity_same_instant']
    L.append(f"| Depth / surface identity violated (same instant) | {'yes' if si['violated'] else 'no'} | depth r/σ {f(si['same_instant_other_camera_depth_r_over_sigma'], 3)} |")
    L.append(f"| R consistent at the prior | — | J(s⁻)/dof = {f(V['covariance_consistency']['prior_iterate_cost_per_dof'], 3)} |")
    bt = V['ballistic_tail']
    L.append(f"| Ballistic tail exact (expected) | {'yes' if bt['exact'] else '**no**'} | max |x − (x_last + Δt v_last)| = {f(bt['max_abs_m'], 2)} m |")
    L += ['', '## Prior (prediction)', '',
          f"dt = {f(P['dt_s'], 6)} s (frame {P['previous_frame']} → {I['divergent_frame']}); position σ {f(P['position_sigma_m'])} m; velocity σ {f(P['velocity_sigma_m_s'])} m/s; "
          f"P⁻ eigenvalues {f(P['eigenvalues'], 3)}, condition {f(P['condition_number'], 3)}.",
          f"Velocity gain of a position correction P_vx P_xx⁻¹ = {f(P['velocity_from_position_gain_P_vx_Pxx_inv'], 4)} (1/s); 1/dt = {f(1/P['dt_s'], 4)}.",
          f"Oracle prediction check: F {f(P['oracle_check']['F_max_abs'], 2)}, Q {f(P['oracle_check']['Q_max_abs'], 2)}, s⁻ {f(P['oracle_check']['s_minus_max_abs'], 2)}, P⁻ rel {f(P['oracle_check']['P_minus_rel'], 2)}.",
          '', '## Iterates', '', '`*` = a finite difference of ∂G/∂s crossed a pixel cell or changed the row set (the bilinear gradient jumps there).', '', '| j | rows | ‖δx‖ m | ‖δv‖ m/s | δᵀAδ | ρ(∂G/∂s) | prior→ | data (same rows) before → actual / GN | objective ↓ | row change | algebra rel |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for it in A['iterates']:
        nl = it['nonlinear_replay_same_rows']
        L.append(f"| {it['iteration']} | {it['cameras_used']} | {f(it['delta_norm_x_m'])} | {f(it['delta_norm_v_m_s'])} | {f(it['step_delta_A_delta'], 3)} | {f(it['iteration_map']['spectral_radius'], 3)}{'*' if it['iteration_map']['columns_crossing_a_cell_or_row_change'] else ''} | "
                 f"{f(nl['prior_cost_before'], 4)}→{f(nl['prior_cost_after'], 4)} | {f(nl['data_before']['rgb']+nl['data_before']['depth'], 4)} → {f(nl['data_after_actual']['rgb']+nl['data_after_actual']['depth'], 4)} / "
                 f"{f(nl['data_after_gauss_newton']['rgb']+nl['data_after_gauss_newton']['depth'], 4)} | {nl['objective_decreased']} | -{it['row_set_change']['missing_at_next']} +{it['row_set_change']['added_at_next']} | "
                 f"{f(max(it['algebra_from_dump'][k] for k in ('A_rel', 'b_rel', 'delta_rel')), 2)} |")
    L += ['', '### Rows at the prior iterate (raw 2×2 support, oracle)', '', '| cam | reason | (u, v) | Z | sampled RGB | D | r/σ (R, G, B, D) | ‖∇I‖ | ‖∇D‖ m/px | spread / allowance m | prior 1σ px | cell-border px |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in A['iterates'][0]['rows']:
        if 'r' in r:
            L.append(f"| {r['camera']} | {r['reason']} | ({f(r['u'], 6)}, {f(r['v'], 6)}) | {f(r['Z'], 5)} | {f(r['sampled_rgb'], 3)} | {f(r['sampled_depth_m'], 5)} | {f(r['r_over_sigma'], 3)} | "
                     f"{f(np.linalg.norm(r['grad_rgb_uv']), 3)} | {f(r['depth_gradient_m_per_px'], 3)} | {f(r['same_surface_spread_m'], 3)} / {f(r['same_surface_allowance_m'], 3)} | {f(r.get('prior_sigma_px_major'), 3)} | {f(r.get('cell_border_distance_px'), 3)} |")
        else:
            uv = f"({f(r['u'], 6)}, {f(r['v'], 6)})" if 'u' in r else ''
            L.append(f"| {r['camera']} | {r['reason']} | {uv} | {f(r['Z'], 5) if 'Z' in r else ''} | | | | | | | | |")
    L += ['', '## Accepted update', '',
          f"Acceptance `{F['acceptance']}`, converged {F['converged']}, rows {F['valid_rows']}; ‖δx‖ = {f(F['delta_x_norm_m'])} m, ‖δv‖ = {f(F['delta_v_norm_m_s'])} m/s, speed after {f(F['speed_after_m_s'])} m/s.",
          f"Prior penalty of the accepted state: (s⁺−s⁻)ᵀΛ(s⁺−s⁻) = {f(F['C_prior_state_mahalanobis2'])}; position-only Mahalanobis² {f(F['position_mahalanobis2'])}.",
          f"Data cost, all valid rows: at s⁻ {F['data_cost']['at_prior_all_rows']} → at s⁺ {F['data_cost']['at_accepted_all_rows']} (different row sets).",
          f"Common cameras {F['data_cost']['common_cameras']}: at s⁻ RGB {f(F['data_cost']['at_prior_common']['rgb'])} + D {f(F['data_cost']['at_prior_common']['depth'])} → at s⁺ RGB {f(F['data_cost']['at_accepted_common']['rgb'])} + D {f(F['data_cost']['at_accepted_common']['depth'])}.",
          f"Measurement information eigenvalues (xx) {f(F['measurement_information_eigenvalues_xx'], 3)}; Joseph vs production P⁺ rel {f(F['joseph_vs_production_P_plus_rel'], 2)}.",
          'Finite-difference H at s⁺ (same interpolant): ' + ', '.join(f"cam {c}: rel {f(v['rel_error'], 2)}{' (crosses a pixel cell)' if v['crosses_pixel_cell'] else ''}" for c, v in F['H_finite_difference_at_accepted'].items() if v) + '.',
          '', '## Attribution of the position correction (exact split, summed over iterates)', '', '| term | share of δx | δx m | δv m/s |', '|---|---|---|---|']
    sh = A['attribution']['position_correction_shares']
    for k in sorted(sh, key=lambda k: -abs(sh[k]))[:14]:
        dd = np.asarray(A['attribution']['total_delta_by_term'][k])
        L.append(f"| {k} | {f(sh[k], 3)} | {f(dd[:3])} | {f(dd[3:])} |")
    E = A['endpoint_cross_camera']
    L += ['', f"## Same-instant cross-camera check at the canonical frame {E['canonical_frame']} (M1)", '',
          f"Source camera {E['source_camera']} pixel {E['source_pixel_u_v']}, source RGB gradient {f(E['source_rgb_gradient_per_px'], 3)} per px, normal {f(E['surface_normal_from_source_depth'], 3)}.", '',
          '| cam | reason | sampled RGB | r/σ (R, G, B, D) | ‖∇I‖ /px | mask border px | view angle ° | Z − D m |', '|---|---|---|---|---|---|---|---|']
    for c in E['cameras']:
        if 'r' in c:
            L.append(f"| {c['camera']}{' (src)' if c['is_source_camera'] else ''} | {c['reason']} | {f(c['sampled_rgb'], 3)} | {f(c['r_over_sigma'], 3)} | {f(c['image_gradient_rgb_per_px'], 3)} | "
                     f"{f(c['mask_border_distance_px'], 3)} | {f(c['viewing_angle_deg'], 3)} | {f(c['depth_discrepancy_Z_minus_D_m'], 3)} |")
        else:
            L.append(f"| {c['camera']}{' (src)' if c['is_source_camera'] else ''} | {c['reason']} | | | | | | |")
    T = A['tail']
    L += ['', '## After the update (K3)', '', f"First prediction-only frame {T['first_prediction_only_frame']}, reasons {T['first_prediction_only_reasons_from_memmap']}; "
          f"all remaining steps prediction-only: {T['remaining_steps_prediction_only']}; max |x − (x_last + Δt v_last)| = {f(T['max_abs_position_vs_x_last_plus_dt_v_m'], 3)} m.",
          '', '## Offline factorial replay (never substituted into the filter)', '', '| variant | observed | rows | ‖δx‖ m | ‖δv‖ m/s | speed m/s | Mahal² (x) | iters | data RGB / D at s⁺ | ‖s_j − s⁻‖ per iterate (m) |', '|---|---|---|---|---|---|---|---|---|---|']
    fac = A['factorial']
    for k, v in fac.items():
        if k.startswith('_') or k == 'covariance_chain_replays': continue
        L.append(f"| {k} | {v['observed']} | {v['final_rows']} | {f(v['delta_x_norm_m'])} | {f(v['delta_v_norm_m_s'])} | {f(v['speed_after_m_s'])} | {f(v['position_mahalanobis2_prior'])} | "
                 f"{v['iterations']} | {f(v['final_data_cost_rgb'])} / {f(v['final_data_cost_depth'])} | {f(v['iterate_offset_from_prior_m'], 3)} |")
    ch = fac['covariance_chain_replays']
    if '_unavailable' in ch: L.append(f"\nCovariance chain replays: {ch['_unavailable']}.")
    else:
        L += ['', 'Covariance chain replays (this point alone from the endpoint; other points at production states):', '',
              '| variant | ‖δx‖ m | ‖δv‖ m/s | speed m/s | P_vx P_xx⁻¹ diag (1/s) | baseline reproduces production |', '|---|---|---|---|---|---|']
        for k, v in ch.items():
            if k.startswith('_'): continue
            g = np.diag(np.asarray(v['final']['velocity_gain_P_vx_Pxx_inv']))
            repro = max(fr['production_s_plus_max_abs'] for fr in v['frames']) if k == 'original' else None
            L.append(f"| {k} | {f(v['final']['delta_x_norm_m'])} | {f(v['final']['delta_v_norm_m_s'])} | {f(v['final']['speed_after_m_s'])} | {f(g, 4)} | {'' if repro is None else f(repro, 2)+' m'} |")
    L += ['', 'All numbers in `analysis.json` (float64); matrices in `arrays.npz`.', '']
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------------------- population
def cmd_population(a):
    """C2: residuals of the endpoint cloud in every other camera at the SAME instant, by camera pair, image gradient,
    mask border distance, viewing angle and depth discrepancy; per-point common offset (shared canonical-colour error)."""
    from scipy.ndimage import distance_transform_edt
    run = Run(a.run, a.direction)
    out = out_dir(a.run, a.name or f'population_{a.direction}')
    k0 = run.fman['canonical_frame']
    cams, imgs = run.cams(), run.images(k0)
    xyz, rgb, cov, src = run.can['canonical_xyz'], run.can['canonical_rgb'], run.can['canonical_source_covariance'], run.can['original_source_ids']
    n = len(xyz); pick = np.unique(np.linspace(0, n-1, min(a.points, n)).round().astype(int))
    dist = [distance_transform_edt(im.mask) for im in imgs]
    # front-most surface per pixel from the WHOLE endpoint cloud (the production self-occlusion rule at the endpoint)
    zbuf = []
    for c, cam in enumerate(cams):
        Xc = xyz@cam.R.T+cam.t; Z = Xc[:, 2]
        u = np.round(cam.fx*Xc[:, 0]/Z+cam.cx).astype(np.int64); v = np.round(cam.fy*Xc[:, 1]/Z+cam.cy).astype(np.int64)
        ok = (Z > cam.near) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        zb = np.full(cam.H*cam.W, np.inf); np.minimum.at(zb, v[ok]*cam.W+u[ok], Z[ok]); zbuf.append(zb)
    pol = {**run.policy, 'row_innovation_gate': 'depth_front_only', 'visibility_n_sigma': float('inf')}
    rec = []
    for p in pick:
        cs = int(src[p, 1]); x = xyz[p]
        for c, cam in enumerate(cams):
            if c == cs: continue
            o = O.row(imgs[c], cam, x, rgb[p], run.noise, pol, cov[p], np.zeros(3))
            if 'r' not in o: continue
            pu, pv = int(round(o['u'])), int(round(o['v']))
            sz = math.sqrt(cam.R[2]@cov[p]@cam.R[2])
            if o['Z']-zbuf[c][pv*cam.W+pu] > run.policy['visibility_n_sigma']*math.sqrt(2)*sz: continue        # behind the endpoint surface: occluded
            view = cam.centre()-x; view /= np.linalg.norm(view)
            src_view = cams[cs].centre()-x; src_view /= np.linalg.norm(src_view)
            rec.append([p, cs, c, *o['r'], *o['var'], float(np.linalg.norm(o['sample']['grad_rgb'])), float(dist[c][pv, pu]),
                        float(np.degrees(np.arccos(np.clip(view@src_view, -1, 1)))), o['Z']])
    R = np.array(rec)
    cols = ['point', 'src_cam', 'cam', 'r_R', 'r_G', 'r_B', 'r_D', 'v_R', 'v_G', 'v_B', 'v_D', 'grad_rgb', 'mask_dist_px', 'angle_between_views_deg', 'Z']
    np.save(out/'rows.npy', R); (out/'rows_columns.json').write_text(json.dumps(cols))
    z2 = R[:, 3:7]**2/R[:, 7:11]
    tail = float(2*O.norm.sf(run.policy['visibility_n_sigma']))
    thr1 = float(O.chi2.isf(tail, 1))

    def stats(m):
        if m.sum() < 20: return {'rows': int(m.sum())}
        zz = z2[m]
        return {'rows': int(m.sum()), 'mean_r2_over_R_rgb': float(zz[:, :3].mean()), 'mean_r2_over_R_depth': float(zz[:, 3].mean()),
                'median_r2_over_R_rgb': float(np.median(zz[:, :3])), 'median_r2_over_R_depth': float(np.median(zz[:, 3])),
                'fraction_beyond_n_sigma_rgb': float((zz[:, :3] > thr1).mean()), 'fraction_beyond_n_sigma_depth': float((zz[:, 3] > thr1).mean()),
                'expected_fraction_if_R_correct': tail,
                'median_abs_r_rgb': float(np.median(np.abs(R[m, 3:6]))), 'median_abs_r_depth_m': float(np.median(np.abs(R[m, 6]))),
                'excess_kurtosis_rgb_over_sigma': float(_kurt((R[m, 3:6]/np.sqrt(R[m, 7:10])).ravel())),
                'excess_kurtosis_depth_over_sigma': float(_kurt(R[m, 6]/np.sqrt(R[m, 10])))}
    tab = {'all': stats(np.ones(len(R), bool))}
    tab['by_camera_pair'] = {f'{s}->{c}': stats((R[:, 1] == s) & (R[:, 2] == c)) for s in range(len(cams)) for c in range(len(cams)) if s != c}
    g = R[:, 11]; qs = np.quantile(g, [0, .25, .5, .75, .9, 1])
    tab['by_image_gradient'] = {f'{qs[i]:.3g}-{qs[i+1]:.3g}': stats((g >= qs[i]) & (g <= qs[i+1])) for i in range(len(qs)-1)}
    md = R[:, 12]
    tab['by_mask_border_distance_px'] = {f'{lo}-{hi}': stats((md >= lo) & (md < hi)) for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9))}
    ang = R[:, 13]
    tab['by_angle_between_source_and_target_view_deg'] = {f'{lo}-{hi}': stats((ang >= lo) & (ang < hi)) for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 180))}
    dd = np.abs(R[:, 6]); dq = np.quantile(dd, [0, .5, .9, .99, 1])
    tab['by_depth_discrepancy_m'] = {f'{dq[i]:.3g}-{dq[i+1]:.3g}': stats((dd >= dq[i]) & (dd <= dq[i+1])) for i in range(len(dq)-1)}
    # shared per-point error (the fixed canonical colour / position enters every row of the point identically)
    pts = R[:, 0].astype(int); up, inv, cnt = np.unique(pts, return_inverse=True, return_counts=True)
    multi = cnt[inv] >= 2
    shared = {}
    for k, name in enumerate(('R', 'G', 'B', 'D')):
        r = R[multi, 3+k]; ii = inv[multi]
        mean = np.bincount(ii, r, len(up))/np.maximum(np.bincount(ii, None, len(up)), 1)
        between, within = float(np.var(mean[np.bincount(ii, None, len(up)) > 0])), float(np.mean((r-mean[ii])**2))
        shared[name] = {'between_point_variance_of_mean_residual': between, 'within_point_variance': within,
                        'shared_fraction': between/max(between+within, 1e-300),
                        'intra_point_correlation_two_cameras': _icc(r, ii)}
    tab['per_point_shared_error'] = {**shared, 'note': 'a large shared fraction / intra-point correlation means one error common to all rows of a point '
                                     '(the canonical colour or position), which a diagonal per-row R treats as independent evidence'}
    summary = {'provenance': provenance_tool(), 'run': str(run.root), 'direction': a.direction, 'canonical_frame': k0, 'points_sampled': int(len(pick)),
               'rows': int(len(R)), 'noise': run.noise.describe(), 'n_sigma_tail': tail, 'tables': tab,
               'row_rule': 'every camera except the source, valid 2x2 sample, not behind the endpoint z-buffer by more than n_sigma sqrt(2) sigma_z'}
    write_json(out/'population.json', summary)
    L = [f"# Same-instant cross-camera residuals, {run.fman['pass']} canonical frame {k0}", '', f"{len(pick)} points, {len(R)} rows; R = {run.noise.describe()}.", '',
         f"If R were right: mean r²/R = 1, fraction beyond {run.policy['visibility_n_sigma']}σ = {tail:.4f}, excess kurtosis 0.", '']
    for key in ('all',):
        L.append(f"All rows: {tab['all']}")
    for title in ('by_image_gradient', 'by_mask_border_distance_px', 'by_angle_between_source_and_target_view_deg', 'by_depth_discrepancy_m', 'by_camera_pair'):
        L += ['', f'## {title}', '', '| bin | rows | mean r²/R RGB | mean r²/R D | > nσ RGB | > nσ D | median |r| RGB | median |r_D| m | kurtosis RGB |', '|---|---|---|---|---|---|---|---|---|']
        for k, s in tab[title].items():
            if 'mean_r2_over_R_rgb' not in s: continue
            L.append(f"| {k} | {s['rows']} | {f(s['mean_r2_over_R_rgb'], 3)} | {f(s['mean_r2_over_R_depth'], 3)} | {f(s['fraction_beyond_n_sigma_rgb'], 3)} | {f(s['fraction_beyond_n_sigma_depth'], 3)} | "
                     f"{f(s['median_abs_r_rgb'], 3)} | {f(s['median_abs_r_depth_m'], 3)} | {f(s['excess_kurtosis_rgb_over_sigma'], 3)} |")
    L += ['', '## Per-point shared error', '', '| channel | between-point var | within-point var | shared fraction | intra-point correlation |', '|---|---|---|---|---|']
    for k in ('R', 'G', 'B', 'D'):
        s = shared[k]; L.append(f"| {k} | {f(s['between_point_variance_of_mean_residual'], 3)} | {f(s['within_point_variance'], 3)} | {f(s['shared_fraction'], 3)} | {f(s['intra_point_correlation_two_cameras'], 3)} |")
    (out/'population.md').write_text('\n'.join(L)+'\n')
    print(f'[population] {len(R)} rows -> {out}/population.md', flush=True)


def _kurt(x):
    x = np.asarray(x, float); x = x-x.mean(); v = (x*x).mean()
    return (x**4).mean()/v**2-3 if v > 0 else float('nan')


def _icc(r, ii):
    """Correlation of the residuals of two different cameras of the same point (first two rows of every point)."""
    order = np.argsort(ii, kind='stable'); ii, r = ii[order], r[order]
    first = np.r_[True, ii[1:] != ii[:-1]]
    a_idx = np.flatnonzero(first); b_idx = a_idx+1
    ok = (b_idx < len(ii)) & (ii[np.minimum(b_idx, len(ii)-1)] == ii[a_idx])
    if ok.sum() < 3: return None
    return float(np.corrcoef(r[a_idx[ok]], r[b_idx[ok]])[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('find', 'analyze', 'population'):
        p = sp.add_parser(name); p.add_argument('run', type=Path); p.add_argument('--direction', choices=['forward', 'backward'], default='forward')
        p.add_argument('--name')
        if name == 'find': p.add_argument('--top', type=int, default=10)
        if name == 'analyze':
            p.add_argument('--point', type=int, required=True); p.add_argument('--frame', type=int)
            p.add_argument('--alt-noise', help="'measured' (scene/noise_calibration.py on this scene) or a noise_calibration.json")
            p.add_argument('--alt-speed99', type=float, nargs='*', default=[], help='alternative declared v99 (m/s) for the P_vv chain replay')
            p.add_argument('--alt-qc', type=float, nargs='*', default=[], help='alternative isotropic q_c (m^2/s^3) for the chain replay')
        if name == 'population': p.add_argument('--points', type=int, default=20000)
    a = ap.parse_args(argv)
    return {'find': cmd_find, 'analyze': cmd_analyze, 'population': cmd_population}[a.cmd](a)


if __name__ == '__main__':
    main()
