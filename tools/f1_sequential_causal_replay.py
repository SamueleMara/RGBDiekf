#!/usr/bin/env python3
"""FOURTH PASS: sequential single-point CPU replay of the standard F1 IEKF with frame-0 camera-reference counterfactuals
(RGBDiekf_F1_Fourth_Pass_Sequential_Replay_and_Diagnostic_Bugfix_Codex.md). DIAGNOSTIC ONLY: nothing is reachable from inference.

For every selected point and arm the filter runs CONTINUOUSLY from the canonical frame 0 (s0+ = [x_ref, 0], archived P0) over
frames 1..N: production prediction (F_k, Q_k of estimator/point_kalman_cuda.process_noise, verified against the trace every frame),
the unchanged oracle IEKF inner loop / row gate / acceptance (tools/root_cause_oracle.Problem, verified bit-for-bit-class against
production), and a per-arm residual model (tools/f1_temporal_reference_diagnostics.RefProblem): A0 original, A1 RGB reference,
A2 depth reference, A3 RGBD reference (ALGEBRA_ONLY), A4 RGBD reference with R_t + R_0 (independent-samples approximation).
Pixel ownership: Mode P (PRODUCTION_OWNERSHIP_REPLAY) borrows production's per-frame frozen assignment and final recheck flags of
this point; Mode C (CONDITIONAL_RECOMPUTED_OWNERSHIP) recomputes the point's front-most status every frame from the saved full
cloud (competitors at s- = X+_{k-1} + dt V+_{k-1} from the saved arrays; SELF vs DUPLICATE labels use the saved diagonal variances
and are flagged approximate; near ties are flagged OWNER_PARITY_UNRESOLVED). Neither equals a production full-scene run.
Ground truth is read ONLY by evaluate() after a replay (GT_ORACLE_ONLY columns); replay() has no access to it.

    python tools/f1_sequential_causal_replay.py --forensics-root FOR --third-pass-root THIRD --out OUT [--frames 0:10] [--points ...]
"""
import argparse
import csv
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
from tools import root_cause_oracle as O                                                       # noqa: E402
from tools.root_cause_trace import jsonable, write_json, frozen_codes                         # noqa: E402
from tools.f1_second_pass import Scene, all_reasons, ShiftCam, wcsv, fl, NAMES, ACC          # noqa: E402
from tools.f1_temporal_reference_diagnostics import RefProblem, references, model_cost       # noqa: E402
from tools.f1_diagnostic_units import compare_state6, compare_iterates, compare_cov6, dominance  # noqa: E402

ARMS = {'A0_ORIGINAL_F1': None, 'A1_REFERENCE_RGB': ('RGB', 'ALGEBRA_ONLY'), 'A2_REFERENCE_DEPTH': ('D', 'ALGEBRA_ONLY'),
        'A3_REFERENCE_RGBD': ('RGBD', 'ALGEBRA_ONLY'), 'A4_REFERENCE_RGBD_RPLUSR0': ('RGBD', 'INDEPENDENT_SAMPLES_APPROX')}
POINTS = {'storage_8444': ('storage_45135', 8444, 10, 'static'), 'usb_67827': ('USB_100109', 67827, 10, 'static'),
          'usb_87290': ('USB_100109', 87290, 10, 'moving'), 'usb_85696': ('USB_100109', 85696, 10, 'moving'),
          'storage_8416': ('storage_45135', 8416, 5, 'knife_edge_moving'),
          'usb_51974': ('USB_100109', 51974, 5, 'static_control'), 'usb_51975': ('USB_100109', 51975, 5, 'static'),
          'storage_60019': ('storage_45135', 60019, 5, 'static_control')}
MODES = ('P', 'C')
EPS_PX = (1e-12, -1e-12, 1e-10, -1e-10, 1e-8, -1e-8)
EVAL_STEP_M = 1e-3            # EVALUATION-ONLY: an accepted |dx| above 1 mm on a GT-static point is reported as a false update
PARITY_POS_M, PARITY_VEL_M_S = 1e-9, 1e-7      # spec 3.3 parity resolution for the stable cases (diagnostic resolution)
SOURCES = ['tools/f1_sequential_causal_replay.py', 'tools/f1_diagnostic_units.py', 'tools/f1_temporal_reference_diagnostics.py', 'tools/f1_second_pass.py',
           'tools/root_cause_oracle.py', 'tools/root_cause_trace.py', 'tools/forensic_f1_drift.py', 'estimator/point_kalman_cuda.py', 'estimator/projective_sensor.py',
           'estimator/projective_terms.py', 'estimator/projective_visibility.py', 'estimator/free_projective_run.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else 'MISSING'


# ----------------------------------------------------------------------------------------------- ownership (Mode C)
class CloudOwnership:
    """Front-most pixel ownership of the selected point recomputed from the SAVED full-cloud standard-F1 state (no GT)."""
    def __init__(self, sc):
        self.sc = sc
        self.X = sc.run.mm('posterior_xyz'); self.V = sc.run.mm('velocity')
        self.vx = sc.run.mm('position_variance_diag'); self.vv = sc.run.mm('velocity_variance_diag')
        self._cache = {}

    def competitors(self, k, dt, final):
        key = (k, final)
        if key not in self._cache:
            if final: X = np.asarray(self.X[k], float); var = np.asarray(self.vx[k], float)
            else:
                X = np.asarray(self.X[k-1], float)+dt*np.asarray(self.V[k-1], float)
                var = np.asarray(self.vx[k-1], float)+dt*dt*np.asarray(self.vv[k-1], float)    # diagonal approximation of P-_xx
            self._cache = {key: (X, var)}
        return self._cache[key]

    def codes(self, pid, x, Pxx, k, dt, final=False):
        """Per camera: VALID if front-most at its nearest pixel (ties: smallest id), else SELF_OCCLUDED / DUPLICATE (approx sigma),
        or the point's own geometric reason. Returns codes, flags (OWNER_PARITY_UNRESOLVED on near ties)."""
        X, var = self.competitors(k, dt, final)
        n_sig = self.sc.policy['visibility_n_sigma']; out, flags = [], []
        for c, cam in enumerate(self.sc.cams):
            g = O.geometric_visibility(cam, x)
            if g != O.VALID: out.append(g); flags.append(''); continue
            u, v, Z, _, _ = cam.project(x); pu, pv = round(u), round(v)
            Xc = X@cam.R.T+cam.t; z = Xc[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                uu, vv = cam.fx*Xc[:, 0]/z+cam.cx, cam.fy*Xc[:, 1]/z+cam.cy
            same = (z > cam.near) & (np.round(uu) == pu) & (np.round(vv) == pv); same[pid] = False
            if not same.any(): out.append(O.VALID); flags.append(''); continue
            ids = np.flatnonzero(same); zc = z[ids]
            front_z = zc.min(); front_id = int(ids[zc == front_z].min())
            if Z < front_z or (Z == front_z and pid < front_id): out.append(O.VALID)
            else:
                r3 = cam.R[2]; si = float(r3@Pxx@r3); sf = float((r3**2)@var[front_id])
                out.append(O.SELF_OCC if (Z-front_z) > n_sig*math.sqrt(si+sf) else O.DUP)
            flags.append('OWNER_PARITY_UNRESOLVED' if abs(Z-front_z) < 1e-6 else '')
        return out, flags


class SeqProblem(RefProblem):
    """RefProblem whose final self-occlusion recheck can be computed at the actual final state (Mode C)."""
    def __init__(self, *a, recheck_fn=None, **kw):
        super().__init__(*a, **kw); self.recheck_fn = recheck_fn

    def rows(self, x, final=False):
        if final and self.recheck_fn is not None: self.recheck_self = self.recheck_fn(x)
        return super().rows(x, final)


# ---------------------------------------------------------------------------------------------------------- replay
def replay(sc, pid, arm, mode, N, refs, own=None, shift=None):
    """Continuous single-point F1 replay frames 0..N (no GT anywhere). shift = (frame, camera, eps_px): one-sided projection
    perturbation of one camera at one frame (knife-edge diagnostic)."""
    tr1 = sc.trace(sc.order[1]); p1 = tr1['prediction']['points'][str(pid)]
    s, P = np.array(p1['s_previous'], float), np.array(p1['P_previous'], float)
    cfg = ARMS[arm]; rule = sc.policy.get('update_acceptance', 'final_iterate')
    frames = []
    for k in sc.order[1:N+1]:
        tr = sc.trace(k); pre = tr['prediction']; sp = str(pid)
        sm, Pm, F, Q = O.predict(s, P, pre['dt_s'], np.array(pre['q_c']))
        it0 = tr['iterations'][0]['points'][sp]['cameras']; fin = tr['final'][sp]
        flags = []
        if mode == 'P':
            frozen = frozen_codes(it0)
            rc = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']]
            recheck_fn = None
            prod_sm = np.array(pre['points'][sp]['s_minus'])
            if np.abs(sm[:3]-prod_sm[:3]).max() > 1e-9: flags.append('OWNERSHIP_NOT_PHYSICALLY_RECOMPUTED')
        else:
            frozen, fl_ = own.codes(pid, sm[:3], Pm[:3, :3], k, pre['dt_s'])
            flags += [f'cam{c}:{f}' for c, f in enumerate(fl_) if f]
            rc = None
            recheck_fn = lambda x, k=k, Pm=Pm, dt=pre['dt_s']: [cc == O.SELF_OCC for cc in own.codes(pid, x, Pm[:3, :3], k, dt, final=True)[0]]
        cams = sc.cams
        if shift is not None and shift[0] == k: cams = [ShiftCam(c, shift[2], shift[2]) if i == shift[1] else c for i, c in enumerate(sc.cams)]
        kw = {} if cfg is None else {'refs': refs, 'channels': cfg[0], 'mode': cfg[1]}
        prob = SeqProblem(cams, sc.images(k), sc.run.can['canonical_rgb'][pid], sm, Pm, frozen, sc.noise, sc.policy, sc.run.fman['max_iekf_iters'],
                          sc.run.fman['step_tolerance'], rc, recheck_fn=recheck_fn, **kw)
        rows0 = prob.rows(sm[:3]); S_prior = [o['camera'] for o in prob.used(rows0)]
        res = prob.run(rule)
        sp_, Pp = res['s_plus'], res['P_plus']
        acc_rows = prob.used(res['rows_accepted']) if res['observed'] else []
        Jb, lb = model_cost(prob, sm, S_prior) if S_prior else (None, [])
        Ja, la = model_cost(prob, sp_, S_prior) if S_prior else (None, [])
        cams_rec = []
        for c, cam in enumerate(sc.cams):
            r_ = all_reasons(sc.images(k)[c], cam, sm[:3], prob.a, sc.noise, sc.policy, Pm[:3, :3], np.zeros(3), frozen[c])
            o0 = next((o for o in rows0 if o['camera'] == c), {})
            cams_rec.append({'camera': c, 'frozen_or_recomputed_code': NAMES[frozen[c]], 'admitted_at_s_minus': c in S_prior, 'in_accepted_solve': c in [o['camera'] for o in acc_rows],
                             'reasons_all_at_s_minus': r_.get('reasons_true'), 'cell': r_.get('cell'), 'u': r_.get('u'), 'v': r_.get('v'),
                             'r_model': fl(o0['r']) if 'r' in o0 else None, 'R_model': fl(o0['var']) if 'var' in o0 else None,
                             'nis_model': o0.get('nis'), 'sqrt_S_DD_m': math.sqrt(o0['S'][3, 3]) if 'S' in o0 else None,
                             'sqrt_R_D_m': math.sqrt(o0['var'][3]) if 'var' in o0 else None, 'reference_applied': o0.get('reference_applied'),
                             'reference_status': (refs[c]['status'] if cfg is not None else 'ORIGINAL_MODEL') + ('' if cfg is None or refs[c]['valid'] else ' -> NO_VALID_REFERENCE_ORIGINAL_USED')})
        frames.append({'frame': k, 'dt_s': pre['dt_s'], 'F_matches_trace': bool(np.array_equal(F, np.array(pre['F']))), 'Q_max_abs_vs_trace': float(np.abs(Q-np.array(pre['Q'])).max()),
                       's_minus': sm, 'P_minus': Pm, 's_plus': sp_, 'P_plus': Pp, 'observed': bool(res['observed']), 'acceptance': ACC[res['acceptance_code']],
                       'converged': bool(res['converged']), 'iterations': len(res['history']), 'rows_initial': S_prior, 'rows_accepted': [o['camera'] for o in acc_rows],
                       'cells_accepted': {o['camera']: list(o['sample']['pixel00']) for o in acc_rows}, 'iterates': [h['s'] for h in res['history']],
                       'J_fixed_set_before': Jb, 'J_fixed_set_after': Ja,
                       'J_fixed_set_status': 'UNDEFINED_SUPPORT_CHANGED' if (lb or la) else ('NO_ROWS' if not S_prior else 'DEFINED'),
                       'J_own_rows_accepted': float(prob.prior_cost(sp_)+prob.info(res['rows_accepted'])[2]) if res['observed'] else None,
                       'own_rows_threshold_dof': 4*len(acc_rows), 'flags': flags, 'cameras': cams_rec})
        s, P = sp_, Pp
    return frames


def evaluate(sc, pid, role, frames, gt):
    """GT_ORACLE_ONLY evaluation of a finished replay (static: x_ref; moving: GT joint transform of x_ref)."""
    x0 = sc.x0[pid]; out = []
    for f in frames:
        xg = gt.moved(x0, f['frame']) if role.endswith('moving') or role == 'moving' else x0
        out.append({'frame': f['frame'], 'gt_error_plus_m_GT_ORACLE_ONLY': float(np.linalg.norm(f['s_plus'][:3]-xg)),
                    'gt_error_minus_m_GT_ORACLE_ONLY': float(np.linalg.norm(f['s_minus'][:3]-xg)), 'x_gt_GT_ORACLE_ONLY': fl(xg)})
    return out


def gt_camera_oracle(sc, pid, f, cams_sel, gt):
    """GT_ORACLE_ONLY: rows of cameras cams_sel at the GT position with the arm's own prior (prior-centred and recentred)."""
    k = f['frame']; xg = gt.moved(sc.x0[pid], k); out = []
    frz = [O.CODE.get(c['frozen_or_recomputed_code'], O.VALID) for c in f['cameras']]
    for c in cams_sel:
        for nm, dx in (('prior_centred', xg-f['s_minus'][:3]), ('recentred', np.zeros(3))):
            r_ = all_reasons(sc.images(k)[c], sc.cams[c], xg, sc.run.can['canonical_rgb'][pid], sc.noise, sc.policy, f['P_minus'][:3, :3], dx, frz[c])
            out.append({'frame': k, 'camera': c, 'variant': f'GT_ORACLE_ONLY_{nm}', 'reasons': r_.get('reasons_true'), 'nis': r_.get('nis'),
                        'r': r_.get('r'), 'valid_production_rules': r_.get('valid_production_rules')})
    return out


# ---------------------------------------------------------------------------------------- covariance oracle (4.1)
def covariance_oracle(T=4, d=2, seed=0, correlated=False):
    """Linear Gaussian fixture: y_t = H x + e_t - e_0 (t = 1..T), x ~ N(0, Px). Dense GLS with the full shared-reference covariance
    vs the latent formulation [x, e_0] with independent e_t; and the naive independent R_t + R_0 treatment."""
    g = np.random.default_rng(seed)
    m = 3; Px = np.diag(g.uniform(.5, 2, m)); H = [g.normal(size=(d, m)) for _ in range(T)]
    R = [np.diag(g.uniform(.1, 1, d)) for _ in range(T+1)]                    # R[0] = R_0 (reference)
    Cfull = np.zeros(((T+1)*d, (T+1)*d))
    for t in range(T+1): Cfull[t*d:(t+1)*d, t*d:(t+1)*d] = R[t]
    if correlated:
        for t in range(1, T+1):
            Ct0 = .3*np.sqrt(np.diag(R[t]))[:, None]*np.sqrt(np.diag(R[0]))[None]*np.eye(d)
            Cfull[t*d:(t+1)*d, :d] = Ct0; Cfull[:d, t*d:(t+1)*d] = Ct0.T
    L = np.zeros((T*d, (T+1)*d))
    for t in range(1, T+1): L[(t-1)*d:t*d, :d] = -np.eye(d); L[(t-1)*d:t*d, t*d:(t+1)*d] = np.eye(d)
    Sig = L@Cfull@L.T; Hs = np.vstack(H)
    x = g.multivariate_normal(np.zeros(m), Px); e = g.multivariate_normal(np.zeros((T+1)*d), Cfull); y = Hs@x+L@e
    def post(Sig_):
        A = np.linalg.inv(Px)+Hs.T@np.linalg.solve(Sig_, Hs); C = np.linalg.inv(A); mu = C@Hs.T@np.linalg.solve(Sig_, y)
        Cy = Hs@Px@Hs.T+Sig_; ll = float(-.5*(y@np.linalg.solve(Cy, y)+np.linalg.slogdet(2*np.pi*Cy)[1]))
        return mu, C, ll
    mu_f, C_f, ll_f = post(Sig)
    out = {'T': T, 'd': d, 'correlated_ref_noise': correlated, 'Sigma_eigs_min_max': [float(np.linalg.eigvalsh(Sig).min()), float(np.linalg.eigvalsh(Sig).max())]}
    if not correlated:      # latent [x, e0] with independent e_t: exact marginalization must agree
        mz = m+d; Pz = np.zeros((mz, mz)); Pz[:m, :m] = Px; Pz[m:, m:] = R[0]
        Hz = np.hstack([Hs, np.tile(-np.eye(d), (T, 1))]); Re = np.zeros((T*d, T*d))
        for t in range(1, T+1): Re[(t-1)*d:t*d, (t-1)*d:t*d] = R[t]
        A = np.linalg.inv(Pz)+Hz.T@np.linalg.solve(Re, Hz); Cz = np.linalg.inv(A); muz = Cz@Hz.T@np.linalg.solve(Re, y)
        Cy = Hz@Pz@Hz.T+Re; ll_z = float(-.5*(y@np.linalg.solve(Cy, y)+np.linalg.slogdet(2*np.pi*Cy)[1]))
        out.update(latent_vs_dense_mean_rel=float(np.abs(muz[:m]-mu_f).max()/max(np.abs(mu_f).max(), 1e-300)),
                   latent_vs_dense_cov_rel=float(np.abs(Cz[:m, :m]-C_f).max()/np.abs(C_f).max()), latent_vs_dense_loglik_abs=abs(ll_z-ll_f))
    Sn = np.zeros_like(Sig)
    for t in range(1, T+1): Sn[(t-1)*d:t*d, (t-1)*d:t*d] = R[t]+R[0]
    mu_n, C_n, ll_n = post(Sn)
    out.update(naive_vs_exact_mean_rel=float(np.abs(mu_n-mu_f).max()/max(np.abs(mu_f).max(), 1e-300)),
               naive_vs_exact_cov_trace_ratio=float(np.trace(C_n)/np.trace(C_f)), naive_vs_exact_loglik_diff=ll_n-ll_f,
               assumptions='independent sensor noise across times except the shared reference e_0' if not correlated else 'C_t0 = 0.3 sqrt(R_t R_0) (diagonal)')
    return out


# -------------------------------------------------------------------------------------------- units corrections
def units_corrections(root, second_root, third_root):
    """Recompute, from STORED six-vectors, the historical mixed-state numbers as position (m) and velocity (m/s) separately."""
    rows = []
    lc = list(csv.DictReader(open(Path(second_root)/'local_counterfactuals.csv')))
    base = {(r['scene'], r['point_id'], r['frame']): r for r in lc if r['arm'] == 'A_baseline_replay'}
    for r in lc:
        if not r['arm'].startswith('F_shift'): continue
        b = base[(r['scene'], r['point_id'], r['frame'])]
        a6 = np.r_[json.loads(r['dx_m']), json.loads(r['dv_m_s'])]; b6 = np.r_[json.loads(b['dx_m']), json.loads(b['dv_m_s'])]   # same s-: s+ diff = d diff
        c = compare_state6(a6, b6)
        rows.append({'source': 'second_pass/local_counterfactuals.csv (stored dx, dv vectors; same s-)', 'scene': r['scene'], 'point_id': r['point_id'], 'frame': r['frame'],
                     'item': r['arm'], 'legacy_field': 'accepted_state_diff_vs_A_m', 'legacy_value_UNITLESS_INVALID': float(r.get('accepted_state_diff_vs_A_m') or 'nan'),
                     'recomputed_legacy_mixed_max': float(np.abs(a6-b6).max()), **c, 'dominated_by': dominance(c)})
    for r in lc:
        if r['arm'] != 'A_baseline_replay' or r['point_id'] not in ('8416', '8417', '8418'): continue
        sc = Scene(root, r['scene']); tr = sc.trace(int(r['frame'])); fin = tr['final'][r['point_id']]; pre = tr['prediction']['points'][r['point_id']]
        oracle6 = np.array(pre['s_minus'])+np.r_[json.loads(r['dx_m']), json.loads(r['dv_m_s'])]
        c = compare_state6(oracle6, fin['s_plus'])
        rows.append({'source': 'second_pass A_baseline_replay dx/dv + production trace s+', 'scene': r['scene'], 'point_id': r['point_id'], 'frame': r['frame'],
                     'item': 'oracle_vs_production_parity', 'legacy_field': 'production_s_plus_max_abs_m', 'legacy_value_UNITLESS_INVALID': float(r.get('production_s_plus_max_abs_m') or 'nan'),
                     'recomputed_legacy_mixed_max': float(np.abs(oracle6-np.array(fin['s_plus'])).max()), **c, 'dominated_by': dominance(c)})
    tp = Path(third_root)/'csv'/'pixel_shift_sensitivity.csv'
    if tp.exists():
        for r in csv.DictReader(open(tp)):
            rows.append({'source': 'third_pass pixel_shift_sensitivity.csv (stored norms only; vectors not stored: recomputed below by the fourth pass)',
                         'scene': r['case'], 'point_id': '', 'frame': '', 'item': f"{r['model']} eps {r['eps_px']}", 'legacy_field': 'accepted_state_diff_vs_unshifted_m',
                         'legacy_value_UNITLESS_INVALID': float(r['accepted_state_diff_vs_unshifted_m']), 'abs_dx_m_stored': float(r['abs_dx_m']), 'abs_dv_m_s_stored': float(r['abs_dv_m_s'])})
    return rows


# ---------------------------------------------------------------------------------------------------------- driver
def static_metrics(sc, pid, frames):
    x_ref = sc.x0[pid]
    acc = np.zeros(3); prev_v = np.zeros(3); rows = []
    for f in frames:
        acc += f['dt_s']*prev_v+(f['s_plus'][:3]-f['s_minus'][:3]); prev_v = f['s_plus'][3:]
        rows.append({'frame': f['frame'], 'drift_from_xref_m': float(np.linalg.norm(f['s_plus'][:3]-x_ref)), 'speed_m_s': float(np.linalg.norm(f['s_plus'][3:])),
                     'decomposition_closure_m': float(np.abs(acc-(f['s_plus'][:3]-x_ref)).max())})
    return rows


def run_point(name, sc, N, out, R, gt, own):
    scene, pid, _, role = POINTS[name]
    refs = references(sc, pid, sc.frame0_ownership([pid])[pid])
    res = {}
    for mode in MODES:
        for arm in ARMS:
            fr = replay(sc, pid, arm, mode, N, refs, own=own)
            res[(arm, mode)] = fr
            ev = evaluate(sc, pid, role, fr, gt); sm_ = static_metrics(sc, pid, fr)
            for f, e, m in zip(fr, ev, sm_):
                d = f['s_plus']-f['s_minus']
                R['state'].append({'case': name, 'scene': scene, 'point_id': pid, 'role': role, 'arm': arm, 'mode': mode, 'frame': f['frame'], 'dt_s': f['dt_s'],
                                   'x_minus_m': fl(f['s_minus'][:3]), 'v_minus_m_s': fl(f['s_minus'][3:]), 'x_plus_m': fl(f['s_plus'][:3]), 'v_plus_m_s': fl(f['s_plus'][3:]),
                                   'sqrt_diag_Pplus_xx_m': fl(np.sqrt(np.diag(f['P_plus'])[:3])), 'sqrt_diag_Pplus_vv_m_s': fl(np.sqrt(np.diag(f['P_plus'])[3:])),
                                   'observed': f['observed'], 'acceptance': f['acceptance'], 'converged': f['converged'], 'iterations': f['iterations'],
                                   'rows_initial': f['rows_initial'], 'rows_accepted': f['rows_accepted'], 'flags': f['flags'], 'F_matches_trace': f['F_matches_trace'],
                                   'Q_max_abs_vs_trace': f['Q_max_abs_vs_trace'], **m, **e})
                R['steps'].append({'case': name, 'arm': arm, 'mode': mode, 'frame': f['frame'], 'dx_m': fl(d[:3]), 'dv_m_s': fl(d[3:]), 'abs_dx_m': float(np.linalg.norm(d[:3])),
                                   'abs_dv_m_s': float(np.linalg.norm(d[3:])), 'observed': f['observed'], 'converged': f['converged'],
                                   'gain_check_dv_minus_Pvx_Pxx_inv_dx_linf_m_s': float(np.abs(d[3:]-f['P_minus'][3:, :3]@np.linalg.solve(f['P_minus'][:3, :3], d[:3])).max())})
                R['objective'].append({'case': name, 'arm': arm, 'mode': mode, 'frame': f['frame'], 'fixed_camera_set': f['rows_initial'], 'J_fixed_set_before': f['J_fixed_set_before'],
                                       'J_fixed_set_after': f['J_fixed_set_after'], 'J_fixed_set_status': f['J_fixed_set_status'], 'J_own_rows_accepted': f['J_own_rows_accepted'],
                                       'own_rows_dof': f['own_rows_threshold_dof'], 'model': arm, 'note': 'never compare J across arms: residual models differ'})
                for c in f['cameras']:
                    R['cams'].append({'case': name, 'arm': arm, 'mode': mode, 'frame': f['frame'], **c})
            R['npz'][f'{name}__{arm}__{mode}__s_plus'] = np.array([f['s_plus'] for f in fr]); R['npz'][f'{name}__{arm}__{mode}__P_plus'] = np.array([f['P_plus'] for f in fr])
            R['npz'][f'{name}__{arm}__{mode}__s_minus'] = np.array([f['s_minus'] for f in fr])
            R['events'].append(events(name, arm, mode, role, fr, ev))
            if role == 'moving' and pid == 87290:
                for f in fr:
                    if f['frame'] in (1, 2, 3, 4, 5, 10): R['gtcam'] += [{'case': name, 'arm': arm, 'mode': mode, **x} for x in gt_camera_oracle(sc, pid, f, (0, 1, 7), gt)]
    # baseline parity: A0 / Mode P vs production, frame by frame
    for f in res[('A0_ORIGINAL_F1', 'P')]:
        tr = sc.trace(f['frame']); pre = tr['prediction']['points'][str(pid)]; fin = tr['final'][str(pid)]
        prod_rows = sorted({r['camera'] for r in fin.get('rows', [])}) if fin['acceptance'].startswith('ACCEPTED') else []
        cm, cp = compare_state6(f['s_minus'], pre['s_minus']), compare_state6(f['s_plus'], fin['s_plus'])
        R['parity'].append({'case': name, 'frame': f['frame'], **{f's_minus_{k}': v for k, v in cm.items()}, **{f's_plus_{k}': v for k, v in cp.items()},
                            **{f'P_minus_{k}': v for k, v in compare_cov6(f['P_minus'], pre['P_minus']).items()}, **{f'P_plus_{k}': v for k, v in compare_cov6(f['P_plus'], fin['P_plus']).items()},
                            'rows_replay': f['rows_accepted'], 'rows_production': prod_rows, 'rows_equal': f['rows_accepted'] == prod_rows,
                            'acceptance_replay': f['acceptance'], 'acceptance_production': fin['acceptance'], 'acceptance_equal': f['acceptance'] == fin['acceptance'],
                            'converged_replay': f['converged'], 'converged_production': fin['converged'],
                            'within_resolution': bool(cp['position_linf_m'] <= PARITY_POS_M and cp['velocity_linf_m_s'] <= PARITY_VEL_M_S and f['rows_accepted'] == prod_rows
                                                      and f['acceptance'] == fin['acceptance'])})
    # Mode C ownership parity on the baseline arm (recomputed codes vs production frozen codes of this point)
    for f in res[('A0_ORIGINAL_F1', 'C')]:
        tr = sc.trace(f['frame']); prod = frozen_codes(tr['iterations'][0]['points'][str(pid)]['cameras'])
        rec = [O.CODE[c['frozen_or_recomputed_code']] for c in f['cameras']]
        sup = lambda c: c in (O.SELF_OCC, O.DUP)
        R['owner'].append({'case': name, 'frame': f['frame'], 'production_frozen': [NAMES[c] for c in prod], 'mode_C_recomputed': [NAMES[c] for c in rec],
                           'eligibility_mismatch_cameras': [c for c in range(len(prod)) if sup(prod[c]) != sup(rec[c])],
                           'label_mismatch_cameras': [c for c in range(len(prod)) if prod[c] != rec[c]], 'flags': f['flags']})
    if role == 'knife_edge_moving':
        src = int(sc.src[pid, 1])
        for arm in ('A0_ORIGINAL_F1', 'A3_REFERENCE_RGBD'):
            b = res[(arm, 'P')]
            for e in EPS_PX:
                fr = replay(sc, pid, arm, 'P', N, refs, shift=(1, src, e))
                it = compare_iterates(fr[0]['iterates'], b[0]['iterates'])
                for f, fb in zip(fr, b):
                    R['knife'].append({'case': name, 'arm': arm, 'eps_px': e, 'shifted_camera': src, 'shifted_frame': 1, 'frame': f['frame'],
                                       **{f'state_vs_unperturbed_{k}': v for k, v in compare_state6(f['s_plus'], fb['s_plus']).items()},
                                       'rows': f['rows_accepted'], 'cells': f['cells_accepted'], 'acceptance': f['acceptance'], 'converged': f['converged'],
                                       **({'frame1_iterate_position_linf_m': it['position_linf_m_per_iterate'], 'frame1_iterate_velocity_linf_m_s': it['velocity_linf_m_s_per_iterate'],
                                           'first_iteration_position_diverging_over_1e-9_m': it['first_iteration_position_diverging_over_1e-9_m'],
                                           'first_iteration_velocity_diverging_over_1e-9_m_s': it['first_iteration_velocity_diverging_over_1e-9_m_s']} if f['frame'] == 1 else {})})
    figures(name, role, res, R, out, sc, pid)
    return res


def events(name, arm, mode, role, fr, ev):
    obs = [f['frame'] for f in fr if f['observed']]
    step = lambda f: float(np.linalg.norm((f['s_plus']-f['s_minus'])[:3]))
    big = [f['frame'] for f in fr if f['observed'] and step(f) > EVAL_STEP_M]
    nonconv = [f['frame'] for f in fr if f['observed'] and not f['converged']]
    had, loss, readm = False, None, None
    for f in fr:
        if had and not f['rows_accepted'] and loss is None: loss = f['frame']
        if loss is not None and f['rows_accepted'] and readm is None and f['frame'] > loss: readm = f['frame']
        had |= bool(f['rows_accepted'])
    first_zero = next((f['frame'] for f in fr if not f['rows_accepted'] and f['frame'] > 0 and any(g['rows_accepted'] for g in fr if g['frame'] < f['frame'])), None)
    return {'case': name, 'arm': arm, 'mode': mode, 'role': role, 'observed_frames': obs, 'n_observed': len(obs),
            'frames_with_accepted_step_over_1mm_EVAL_ONLY': big, 'first_nonconverged_accepted': nonconv[0] if nonconv else None, 'first_row_loss': loss,
            'first_readmission': readm, 'first_zero_row_after_support': first_zero, 'final_drift_or_gt_error_m_GT_ORACLE_ONLY': ev[-1]['gt_error_plus_m_GT_ORACLE_ONLY'],
            'final_speed_m_s': float(np.linalg.norm(fr[-1]['s_plus'][3:]))}


def figures(name, role, res, R, out, sc, pid):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for (arm, mode), fr in res.items():
        ls = '-' if mode == 'P' else '--'; k = [f['frame'] for f in fr]
        ev = [r for r in R['state'] if r['case'] == name and r['arm'] == arm and r['mode'] == mode]
        ax[0, 0].plot(k, [1e3*r['gt_error_plus_m_GT_ORACLE_ONLY'] for r in ev], ls, label=f'{arm} {mode}')
        ax[0, 1].plot(k, [np.linalg.norm(f['s_plus'][3:]) for f in fr], ls)
        ax[1, 0].plot(k, [1e3*np.linalg.norm((f['s_plus']-f['s_minus'])[:3]) for f in fr], ls)
        ax[1, 1].plot(k, [len(f['rows_accepted']) for f in fr], ls)
    ax[0, 0].set_ylabel(('transport drift |x+ - x_ref|' if role != 'moving' and not role.endswith('moving') else 'GT error (EVALUATION ONLY)')+' (mm)'); ax[0, 0].legend(fontsize=6)
    ax[0, 1].set_ylabel('speed |v+| (m/s)'); ax[1, 0].set_ylabel('accepted position step |x+ - x-| (mm)'); ax[1, 1].set_ylabel('accepted camera rows (count)')
    for a in ax.ravel(): a.set_xlabel('frame')
    fig.suptitle(f'{name}: sequential single-point replay (solid: Mode P production ownership; dashed: Mode C recomputed ownership)', fontsize=9)
    fig.tight_layout(); fig.savefig(out/'figures'/f'{name}_trajectory.png', dpi=100); plt.close(fig)
    if pid == 87290:
        g = [r for r in R['cams'] if r['case'] == name and r['camera'] in (0, 1, 7) and r['mode'] == 'P']
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for i, c in enumerate((0, 1, 7)):
            for arm in ARMS:
                rr = [r for r in g if r['camera'] == c and r['arm'] == arm]
                ax[i].semilogy([r['frame'] for r in rr], [r['nis_model'] if r['nis_model'] is not None else np.nan for r in rr], 'o-', label=arm)
            ax[i].axhline(O.row_gate_threshold(sc.policy['visibility_n_sigma']), color='r', ls='--'); ax[i].set_title(f'camera {c}: row NIS at each arm\'s own s- (Mode P)', fontsize=8)
            ax[i].set_xlabel('frame'); ax[i].set_ylabel('NIS (dimensionless)')
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(out/'figures'/'usb_87290_per_camera_nis.png', dpi=100); plt.close(fig)


def verdicts(R):
    out = {}
    par = {}
    for r in R['parity']: par.setdefault(r['case'], []).append(r['within_resolution'])
    for e in R['events']:
        c = e['case']; role = e['role']; ok = all(par.get(c, [False]))
        if not ok and c != 'storage_8416': v = 'BASELINE_PARITY_FAILED -> causal claims INDETERMINATE'
        elif role.startswith('static'):
            if e['n_observed'] == 0: v = 'UNOBSERVED (no accepted measurement: not evidence of success)'
            elif not e['frames_with_accepted_step_over_1mm_EVAL_ONLY'] and e['final_drift_or_gt_error_m_GT_ORACLE_ONLY'] < EVAL_STEP_M: v = 'STATIC_FALSE_DRIFT_SUPPRESSED_THROUGH_FRAME' + str(max(e['observed_frames']+[0]))
            elif e['frames_with_accepted_step_over_1mm_EVAL_ONLY'] and e['frames_with_accepted_step_over_1mm_EVAL_ONLY'][0] > 1: v = f"STATIC_FALSE_DRIFT_DELAYED (first accepted step > 1 mm at frame {e['frames_with_accepted_step_over_1mm_EVAL_ONLY'][0]})"
            else: v = f"STATIC_FALSE_DRIFT_PERSISTS (first accepted step > 1 mm at frame {(e['frames_with_accepted_step_over_1mm_EVAL_ONLY'] or ['-'])[0]})"
        else:
            v = f"see report (first zero-row frame {e['first_zero_row_after_support']}, final GT error {e['final_drift_or_gt_error_m_GT_ORACLE_ONLY']:.3f} m EVALUATION ONLY)"
        out[f"{c}|{e['arm']}|{e['mode']}"] = v
    return out


def write_report(out, R, args, status):
    V = verdicts(R); L = ['# Fourth pass: sequential single-point replay and diagnostic units fix (standard F1, offline CPU)', '',
                          f'## 1. Executive status: {status}', '', f'Command: `{args}`', '',
                          '## 2. Diagnostic units bug', '', 'Historical fields that took one max over the mixed [x (m), v (m/s)] difference and labelled it metres were replaced by '
                          'separate position (m) / velocity (m/s) metrics (tools/f1_diagnostic_units.py). Recomputed values (`diagnostic_units_corrections.csv`):', '',
                          '| source | point | item | legacy mixed value | position linf (m) | velocity linf (m/s) | dominated by |', '|---|---|---|---|---|---|---|']
    for r in R['units']:
        if 'position_linf_m' in r:
            L.append(f"| {r['source'][:40]} | {r['scene']} {r['point_id']} f{r['frame']} | {r['item']} | {r['legacy_value_UNITLESS_INVALID']:.3g} | {r['position_linf_m']:.3g} | {r['velocity_linf_m_s']:.3g} | {r['dominated_by']} |")
    L += ['', '## 3. Baseline parity (A0, Mode P, frame by frame vs production trace)', '', '| case | frame | s+ position linf (m) | s+ velocity linf (m/s) | P+_xx rel | rows equal | acceptance equal | within resolution |', '|---|---|---|---|---|---|---|---|']
    for r in R['parity']:
        L.append(f"| {r['case']} | {r['frame']} | {r['s_plus_position_linf_m']:.2g} | {r['s_plus_velocity_linf_m_s']:.2g} | {r['P_plus_P_xx_rel_max']:.2g} | {r['rows_equal']} | {r['acceptance_equal']} | {r['within_resolution']} |")
    L += ['', '## 4-8. Per-case trajectories (drift / GT error evaluation-only, speed, rows)', '']
    for name in dict.fromkeys(r['case'] for r in R['state']):
        L += [f'### {name}', '', '| arm | mode | frame: transport drift or GT error (mm) / speed (m/s) / rows |', '|---|---|---|']
        for arm in ARMS:
            for mode in MODES:
                rr = [r for r in R['state'] if r['case'] == name and r['arm'] == arm and r['mode'] == mode]
                L.append(f"| {arm} | {mode} | " + ' ; '.join(f"{r['frame']}: {1e3*r['gt_error_plus_m_GT_ORACLE_ONLY']:.2f}/{r['speed_m_s']:.3f}/{r['rows_accepted']}" for r in rr) + ' |')
        L += ['', 'Verdicts: ' + '; '.join(f"{k.split('|')[1]} {k.split('|')[2]}: {v}" for k, v in V.items() if k.startswith(name+'|')), '']
    L += ['## Ownership Mode C parity on A0', '', '| case | frame | eligibility mismatch cams | label mismatch cams | flags |', '|---|---|---|---|---|']
    for r in R['owner']: L.append(f"| {r['case']} | {r['frame']} | {r['eligibility_mismatch_cameras']} | {r['label_mismatch_cameras']} | {r['flags']} |")
    L += ['', '## 9. Covariance status', '', 'A1-A3 keep R_t (algebraic probes); A4 uses R_t + R_0 per frame and ignores the shared-reference cross-time covariance. '
          'Neither is a calibrated posterior. Oracle (`reference_covariance_tests.csv`):', '']
    for r in R['cov']: L.append(f"- {json.dumps(r, default=float)}")
    L += ['', '## 10-11. Failure analysis and next research decision', '', 'Written after review of the numbers above (see the terminal summary); this file is regenerated from data only.']
    (out/'FOURTH_PASS_CAUSAL_REPORT.md').write_text('\n'.join(L)+'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forensics-root', type=Path, required=True); ap.add_argument('--third-pass-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--frames', default='0:10', help='0:N (per-point maxima: controls / 8416 use min(N, 5))')
    ap.add_argument('--points', help=f'comma list of {list(POINTS)} (default all)'); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    N = int(a.frames.split(':')[1]); pts = a.points.split(',') if a.points else list(POINTS)
    bad = [p for p in pts if p not in POINTS]
    if bad: raise SystemExit(f'unknown points {bad}')
    second = a.forensics_root/'second_pass_20261030'
    need = [second/'local_counterfactuals.csv', a.third_pass_root/'THIRD_PASS_CAUSAL_RESULT.md', a.third_pass_root/'csv'/'pixel_shift_sensitivity.csv']
    missing = [str(p) for p in need if not p.exists()]
    if missing: raise SystemExit(f'missing inputs (refusing to fabricate): {missing}')
    git = lambda *c: subprocess.run(['git', *c], cwd=REPO, capture_output=True, text=True).stdout.strip()
    inputs = {'command': ' '.join(sys.argv), 'options': vars(a) | {'forensics_root': str(a.forensics_root), 'third_pass_root': str(a.third_pass_root), 'out': str(a.out)},
              'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')), 'sources_sha256': {s: sha(REPO/s) for s in SOURCES},
              'third_pass_sha256': {str(p.relative_to(a.third_pass_root)): sha(p) for p in sorted(a.third_pass_root.rglob('*')) if p.is_file()},
              'points': {p: POINTS[p] for p in pts}, 'frames_per_point': {p: min(N, POINTS[p][2]) for p in pts}}
    if a.dry_run: print(json.dumps(inputs, indent=1, default=str)); return
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit(f'{a.out} exists and is not empty: never overwritten')
    (a.out/'figures').mkdir(parents=True, exist_ok=True)
    write_json(a.out/'input_hashes.json', inputs)
    t0 = time.perf_counter()
    from tools.forensic_f1_drift import GroundTruth
    R = {k: [] for k in ('state', 'steps', 'objective', 'cams', 'parity', 'owner', 'events', 'knife', 'gtcam')}; R['npz'] = {}
    scenes, gts, owns = {}, {}, {}
    for p in pts:
        scene = POINTS[p][0]
        sc = scenes.setdefault(scene, Scene(a.forensics_root, scene))
        gt = gts.setdefault(scene, GroundTruth(sc.run, Path(sc.run.man['arguments']['scene'])))
        own = owns.setdefault(scene, CloudOwnership(sc))
        print(f'[fourth-pass] {p} ({time.perf_counter()-t0:.0f} s)', flush=True)
        run_point(p, sc, min(N, POINTS[p][2]), a.out, R, gt, own)
    R['units'] = units_corrections(a.forensics_root, second, a.third_pass_root)
    R['cov'] = [covariance_oracle(T, 2, s, corr) for T in (3, 5) for s in (0, 1) for corr in (False, True)]
    for nm, key in (('baseline_parity.csv', 'parity'), ('sequence_state.csv', 'state'), ('sequence_camera_rows.csv', 'cams'), ('sequence_accepted_steps.csv', 'steps'),
                    ('sequence_failure_events.csv', 'events'), ('sequence_objective.csv', 'objective'), ('reference_covariance_tests.csv', 'cov'),
                    ('diagnostic_units_corrections.csv', 'units'), ('pixel_shift_sequence_8416.csv', 'knife'), ('ownership_mode_C_parity.csv', 'owner'),
                    ('gt_oracle_cameras_87290.csv', 'gtcam')):
        wcsv(a.out/nm, R[key])
    np.savez_compressed(a.out/'sequence_arrays.npz', **R['npz'])
    status = 'PARTIAL' if any(not r['within_resolution'] for r in R['parity'] if r['case'] != 'storage_8416') else 'COMPLETE (pending human verdicts)'
    write_report(a.out, R, ' '.join(sys.argv), status)
    (a.out/'README.md').write_text(f"# Fourth pass (offline, CPU, sequential single-point replays)\n\nExecuted command:\n\n```bash\n{' '.join(sys.argv)}\n```\n\n"
                                   "Tests: `CUDA_VISIBLE_DEVICES= python -m unittest tests.test_f1_diagnostic_units tests.test_f1_sequential_causal_replay -v > tests.log 2>&1`.\n"
                                   "No production file was modified; GT is used only in evaluate() / *_GT_ORACLE_ONLY columns.\n")
    write_json(a.out/'verdicts_automatic.json', verdicts(R))
    print(json.dumps({'seconds': time.perf_counter()-t0, 'status': status, 'parity_failures': [(r['case'], r['frame']) for r in R['parity'] if not r['within_resolution']],
                      'verdicts': verdicts(R)}, default=str)[:4000])


if __name__ == '__main__':
    main()
