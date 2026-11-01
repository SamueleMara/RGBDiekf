#!/usr/bin/env python3
"""THIRD PASS: camera-specific frame-0 references, depth creases and moving-point reacquisition in the standard F1 IEKF
(RGBDiekf_F1_Temporal_Reference_Depth_Surface_Reacquisition_Debug_Codex.md). DEBUG AND OFFLINE COUNTERFACTUALS ONLY.

Every experiment is a CPU single-point / single-frame replay of the first-pass traced standard-F1 runs (bit-identical to the
baselines) with the independent float64 oracle (tools/root_cause_oracle.py) through the second-pass machinery
(tools/f1_second_pass.py). Nothing here is reachable from inference; no production file, config or default is touched.

Reference-differenced diagnostic residual (section 3), b_c0 computed ONCE at the immutable x_ref from frame-0 images:
    r^Delta_{c,t}(x_t) = r_{c,t}(x_t) - b_{c,0},    b_{c,0} = r_{c,0}(x_ref)
applied per channel group (RGB / depth / both) to the cameras with a VALID frame-0 reference; cameras without one keep the
original residual and are flagged NO_VALID_REFERENCE (never a zero bias, never another pixel). The Jacobian is unchanged.
Variance modes: ALGEBRA_ONLY (R_t unchanged), INDEPENDENT_SAMPLES_APPROX (R_t + R_0: an assumption, sensitivity only),
MEASURED_CORRELATION_DIAGNOSTIC (NOT_ESTIMABLE_FROM_AVAILABLE_DATA unless empirical covariance exists: it does not).
The production row gate (rgbd_chi2 on nu = r + H dx, S = H P-_xx H^T + R) is re-applied to the modified residual exactly.

    python tools/f1_temporal_reference_diagnostics.py --forensics-root FOR --second-pass-root FOR/second_pass_20261030 \
        --out FOR/third_pass_temporal_reference_20261030 [--case storage_8444,usb_67827,...] [--dry-run]
GT (labelled part cloud + GT joint, rendered depth) is used only for post-hoc oracle columns, labelled GT_ORACLE_ONLY.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import root_cause_oracle as O                                                    # noqa: E402
from tools.root_cause_trace import jsonable, write_json, frozen_codes                      # noqa: E402
from tools.f1_diagnostic_units import compare_state6, compare_iterates      # noqa: E402
from tools.f1_second_pass import (Scene, all_reasons, run_prob, ShiftCam, wcsv, fl, NAMES, ACC, REASON_KEYS, xref_residuals, cf0)   # noqa: E402

CASES = {   # case: (scene, point, decisive frame, role)
    'storage_8444': ('storage_45135', 8444, 1, 'primary'), 'usb_67827': ('USB_100109', 67827, 1, 'primary'),
    'usb_87290': ('USB_100109', 87290, None, 'moving'), 'usb_85696': ('USB_100109', 85696, None, 'moving'),
    'storage_8416': ('storage_45135', 8416, 1, 'knife_edge'),
    'usb_51975': ('USB_100109', 51975, 2, 'control'), 'usb_51974': ('USB_100109', 51974, 1, 'control'),
    'storage_60019': ('storage_45135', 60019, 1, 'control'), 'storage_100564': ('storage_45135', 100564, 1, 'control'),
    'usb_30502': ('USB_100109', 30502, 3, 'control'),
}
PHYSICAL_REF_INVALID = ('behind', 'outside_footprint', 'cell_outside', 'mask_excluded', 'invalid_depth', 'discontinuity')
EPS_PX = (1e-12, -1e-12, 1e-10, -1e-10, 1e-8, -1e-8)
CH = {'RGB': np.array([1., 1., 1., 0.]), 'D': np.array([0., 0., 0., 1.]), 'RGBD': np.ones(4)}
SOURCES = ['tools/f1_temporal_reference_diagnostics.py', 'tools/f1_second_pass.py', 'tools/root_cause_oracle.py', 'tools/root_cause_trace.py',
           'tools/forensic_f1_drift.py', 'estimator/free_projective_run.py', 'estimator/point_kalman_cuda.py', 'estimator/projective_terms.py',
           'estimator/projective_sensor.py', 'estimator/projective_visibility.py', 'scene/endpoint.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else 'MISSING'


# ------------------------------------------------------------------------------------------------ frame-0 references
def references(sc, pid, own0):
    """b_{c,0} = r_{c,0}(x_ref) for every camera, with its validity. Physical validity: a sample exists and none of
    behind / outside / cell outside / mask / invalid depth / discontinuity, and the camera is not SELF_OCCLUDED by the
    frame-0 canonical cloud. Ownership DUPLICATE (same surface within uncertainty) is recorded, not invalidating. The
    source camera's reference is the prior definition itself (b = 0 by construction, correlated with x_ref / P_x0 / a)."""
    x = sc.x0[pid]; a = sc.run.can['canonical_rgb'][pid]; srcc = int(sc.src[pid, 1]); P0 = sc.cov0(pid)
    out = {}
    for c, cam in enumerate(sc.cams):
        r_ = all_reasons(sc.images(0)[c], cam, x, a, sc.noise, sc.policy, P0[:3, :3], np.zeros(3), O.VALID)
        inval = [k for k in PHYSICAL_REF_INVALID if r_.get(k)]+(['frame0_self_occluded'] if int(own0[c]) == O.SELF_OCC else [])
        ok = 'r' in r_ and not inval
        out[c] = {'camera': c, 'valid': ok, 'status': 'VALID_REFERENCE' if ok else 'NO_VALID_REFERENCE', 'invalid_reasons': inval or ([] if 'r' in r_ else ['no_sample']),
                  'b': np.array(r_['r']) if ok else None, 'var0': np.array(r_['R']) if ok else None, 'H0': r_.get('H'), 'cell0': r_.get('cell'),
                  'du_dv0': [r_.get('du'), r_.get('dv')], 'depth4_0': r_.get('depth4'), 'rgb4_0': r_.get('rgb4'), 'mask4_0': r_.get('mask4'),
                  'frame0_owner': NAMES[int(own0[c])], 'is_source_camera': c == srcc,
                  'source_note': 'SOURCE_REFERENCE_IS_PRIOR_DEFINITION (b = 0 by construction; correlated with x_ref, P_x0, a)' if c == srcc else ''}
    return out


# ------------------------------------------------------------------------------------------- referenced row and replay
def gate(policy, r, H, var, Sigma, dx):
    S = H@Sigma@H.T+np.diag(var); nu = r+H@dx; n = policy['visibility_n_sigma']
    nis = float(nu@np.linalg.solve(S, nu)); g = policy.get('row_innovation_gate', 'depth_front_only')
    front = r[3] if g == 'depth_front_only' else nu[3]
    reason = O.VALID
    if front > n*math.sqrt(max(S[3, 3], 0)): reason = O.FRONT
    elif g == 'depth_two_sided' and nu[3] < -n*math.sqrt(max(S[3, 3], 0)): reason = O.INCONS
    elif g == 'rgbd_chi2' and not nis <= O.row_gate_threshold(n): reason = O.INCONS
    return S, nu, nis, reason


def ref_row(img, cam, x, a, noise, policy, Sigma, dx, ref=None, channels='RGBD', mode='ALGEBRA_ONLY', depth_model=None, gated=True):
    """O.row with (i) an optional replacement depth model (B5 oracle), (ii) an optional frame-0 reference offset on a channel
    group, (iii) a variance mode; then the production gate exactly on the modified residual (gated=False: no gate)."""
    o = O.row(img, cam, x, a, noise, {**policy, 'visibility_n_sigma': float('inf'), 'row_innovation_gate': 'depth_front_only'}, Sigma, dx)
    if 'r' not in o: return o
    r, var, H = o['r'].copy(), o['var'].copy(), o['H'].copy()
    o['r_original'], o['var_original'] = o['r'].copy(), o['var'].copy()
    if depth_model is not None:
        D, gD = depth_model(o['u'], o['v'])
        if D is None: o['reason'] = O.DISCONT; o['depth_model'] = 'NOT_APPLICABLE'; return o
        r[3] = o['Z']-D; H[3] = np.asarray(gD)@o['J_u']-o['J_Z']; o['depth_model'] = 'APPLIED'
    o['reference_applied'] = False
    if ref is not None:
        if ref['valid']:
            m = CH[ref.get('channels', channels)]; r = r-m*ref['b']; o['reference_applied'] = True
            if mode == 'INDEPENDENT_SAMPLES_APPROX': var = var+m*ref['var0']
        else: o['reference_status'] = 'NO_VALID_REFERENCE (original residual kept)'
    o['r'], o['var'], o['H'] = r, var, H
    if gated:
        S, nu, nis, reason = gate(policy, r, H, var, Sigma, dx)
        o.update(S=S, nu=nu, nis=nis, S_zz=float(S[3, 3]), reason=reason)
    else:
        o['reason'] = O.VALID
    return o


class RefProblem(O.Problem):
    """O.Problem whose rows carry the per-camera reference / depth-model / variance-mode changes; with fixed_membership the
    camera rows of a frozen set are used without candidacy or gate and invalid samples are SIGNALLED, not silently dropped."""
    def __init__(self, *a, refs=None, channels='RGBD', mode='ALGEBRA_ONLY', depth_models=None, fixed_membership=None, **kw):
        super().__init__(*a, **kw)
        self.refs, self.ch_group, self.mode = refs or {}, channels, mode
        self.depth_models, self.fixed_membership = depth_models or {}, fixed_membership
        self.invalid_events = []

    def rows(self, x, final=False):
        dx = x-self.s_minus[:3]; out = []
        if self.fixed_membership is not None:
            for c, cam in enumerate(self.cams):
                if c not in self.fixed_membership: out.append({'camera': c, 'reason': O.NOT_PROC, 'visibility': O.NOT_PROC}); continue
                o = ref_row(self.imgs[c], cam, x, self.a, self.noise, self.policy, self.P_minus[:3, :3], dx, self.refs.get(c), self.ch_group, self.mode,
                            self.depth_models.get(c), gated=False)
                o.update(camera=c, visibility=O.VALID)
                if o['reason'] != O.VALID: self.invalid_events.append((c, NAMES[o['reason']], fl(x)))
                out.append(o)
            return out
        vis = O.candidacy(self.cams, x, self.frozen, self.recheck_self if final else None)
        for c, cam in enumerate(self.cams):
            if vis[c] != O.VALID: out.append({'camera': c, 'reason': vis[c], 'visibility': vis[c]}); continue
            o = ref_row(self.imgs[c], cam, x, self.a, self.noise, self.policy, self.P_minus[:3, :3], dx, self.refs.get(c), self.ch_group, self.mode, self.depth_models.get(c))
            o.update(camera=c, visibility=O.VALID)
            if o['reason'] == O.VALID and not self.cam_mask[c]: o['excluded_by_variant'] = True
            if o['reason'] == O.VALID and self.fixed_rows is not None and c not in self.fixed_rows: o['excluded_by_variant'] = True
            out.append(o)
        return out


def make(sc, k, pid, cams=None, **kw):
    tr = sc.trace(k); sp = str(pid)
    pre, it0, fin = tr['prediction']['points'][sp], tr['iterations'][0]['points'][sp], tr['final'][sp]
    rc = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']] if sc.policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion' else None
    P = kw.pop('P', None)
    return RefProblem(cams or sc.cams, sc.images(k), sc.run.can['canonical_rgb'][pid], np.array(pre['s_minus']), np.array(pre['P_minus']) if P is None else P,
                      frozen_codes(it0['cameras']), sc.noise, sc.policy, sc.run.fman['max_iekf_iters'], sc.run.fman['step_tolerance'], rc, **kw)


def model_cost(prob, s, cams):
    """Within-model objective on a stated camera set at state s (rows resampled at s, model applied, no gate): prior + sum
    r^T R^-1 r; rows not evaluable at s are listed and the cost is None (never zero-filled)."""
    dx = s[:3]-prob.s_minus[:3]; q = 0.; lost = []
    for c in cams:
        o = ref_row(prob.imgs[c], prob.cams[c], s[:3], prob.a, prob.noise, prob.policy, prob.P_minus[:3, :3], dx, prob.refs.get(c), prob.ch_group,
                    prob.mode, prob.depth_models.get(c), gated=False)
        if 'r' not in o or o['reason'] != O.VALID: lost.append((c, NAMES[o['reason']])); continue
        q += float((o['r']**2/o['var']).sum())
    return (None if lost else prob.prior_cost(s)+q), lost


def arm(sc, pid, k, name, prob, S_prior, rows_out, label_model):
    res = run_prob(prob, sc)
    d = res['s_plus']-prob.s_minus
    acc = prob.used(res['rows_accepted']) if res['observed'] else []
    Jb, lb = model_cost(prob, prob.s_minus, S_prior); Ja, la = model_cost(prob, res['s_plus'], S_prior)
    for h in res['history']:
        used = prob.used(h['rows'])
        rows_out.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'arm': name, 'model': label_model, 'iteration': h['iteration'], 's': fl(h['s']),
                         'delta': fl(h['delta']), 'dx_from_prior_m': fl(h['s'][:3]-prob.s_minus[:3]), 'J_model_own_rows': h['J'], 'threshold': h['threshold'],
                         'nrows': h['nrows'], 'rows': [o['camera'] for o in used], 'cells': {o['camera']: list(o['sample']['pixel00']) for o in used},
                         'residuals': {o['camera']: fl(o['r']) for o in used}, 'reference_applied': {o['camera']: o.get('reference_applied', False) for o in used},
                         'step': h['step']})
    return {'scene': sc.name, 'point_id': pid, 'frame': k, 'arm': name, 'model': label_model, 'acceptance': ACC[res['acceptance_code']], 'observed': res['observed'],
            'dx_m': fl(d[:3]), 'dv_m_s': fl(d[3:]), 'abs_dx_m': float(np.linalg.norm(d[:3])), 'abs_dv_m_s': float(np.linalg.norm(d[3:])),
            'iterations': len(res['history']), 'converged': res['converged'], 'rows_initial': [o['camera'] for o in prob.used(prob.rows(prob.s_minus[:3]))],
            'rows_accepted': [o['camera'] for o in acc], 'cells_accepted': {o['camera']: list(o['sample']['pixel00']) for o in acc},
            'J_model_prior_state_on_S_prior': Jb, 'J_model_accepted_state_on_S_prior': Ja, 'not_evaluable_on_S_prior': lb+la, 'S_prior': S_prior,
            'J_model_accepted_own_rows': float(prob.prior_cost(res['s_plus'])+prob.info(res['rows_accepted'])[2]) if res['observed'] else None,
            'invalid_support_events': prob.invalid_events[:10], 's_plus': fl(res['s_plus'])}, res


def hierarchy(sc, pid, k, refs, rows_out, extra=()):
    """The section 9.1 hierarchy for one point: BASELINE, REFERENCE_RGB_ONLY / DEPTH_ONLY / RGBD (ALGEBRA_ONLY and
    INDEPENDENT_SAMPLES_APPROX), CAMERA_DROP (each initially admitted camera), ORIGINAL / REFERENCE_FIXED_ELIGIBILITY."""
    base = make(sc, k, pid)
    S_prior = [o['camera'] for o in base.used(base.rows(base.s_minus[:3]))]
    out = []; res = {}
    o, r = arm(sc, pid, k, 'BASELINE', base, S_prior, rows_out, 'original'); out.append(o); res['BASELINE'] = r
    fin = sc.trace(k)['final'][str(pid)]
    out[-1].update({f'production_parity_{kk}': vv for kk, vv in compare_state6(r['s_plus'], fin['s_plus']).items()})   # units bug fixed (2026-10-30)
    out[-1]['production_rows'] = sorted({rr['camera'] for rr in fin.get('rows', [])}) if fin['acceptance'].startswith('ACCEPTED') else []
    for mode in ('ALGEBRA_ONLY', 'INDEPENDENT_SAMPLES_APPROX'):
        for g in ('RGB', 'D', 'RGBD'):
            nm = {'RGB': 'REFERENCE_RGB_ONLY', 'D': 'REFERENCE_DEPTH_ONLY', 'RGBD': 'REFERENCE_RGBD'}[g]+('' if mode == 'ALGEBRA_ONLY' else '_INDEPENDENT_SAMPLES_APPROX')
            p = make(sc, k, pid, refs=refs, channels=g, mode=mode); o, r = arm(sc, pid, k, nm, p, S_prior, rows_out, f'reference[{g}]/{mode}'); out.append(o); res[nm] = r
    for c in S_prior:
        p = make(sc, k, pid, cam_mask=[cc != c for cc in range(len(sc.cams))]); o, _ = arm(sc, pid, k, f'CAMERA_DROP_cam{c}', p, S_prior, rows_out, 'original'); out.append(o)
    if S_prior:
        p = make(sc, k, pid, fixed_membership=S_prior); o, _ = arm(sc, pid, k, 'ORIGINAL_FIXED_ELIGIBILITY', p, S_prior, rows_out, 'original/fixed rows'); out.append(o)
        p = make(sc, k, pid, fixed_membership=S_prior, refs=refs, channels='RGBD'); o, _ = arm(sc, pid, k, 'REFERENCE_FIXED_ELIGIBILITY', p, S_prior, rows_out, 'reference[RGBD]/fixed rows'); out.append(o)
    for nm, kw in extra:
        kw = dict(kw); label = kw.pop('_label', nm)
        p = make(sc, k, pid, **kw); o, _ = arm(sc, pid, k, nm, p, S_prior, rows_out, label); out.append(o)
    return out, S_prior, res


def reference_table(sc, pid, refs, frames=range(0, 6)):
    """Fixed x_ref, frames 0-5, every camera: original r_t, b_c0, r^Delta_t = r_t - b_c0, bitwise pixel identity with frame 0."""
    x = sc.x0[pid]; a = sc.run.can['canonical_rgb'][pid]; out = []
    for t in frames:
        for c, cam in enumerate(sc.cams):
            r_ = all_reasons(sc.images(t)[c], cam, x, a, sc.noise, sc.policy, sc.cov0(pid)[:3, :3], np.zeros(3), O.VALID)
            ref = refs[c]
            row = {'scene': sc.name, 'point_id': pid, 'frame': t, 'camera': c, 'is_source_camera': ref['is_source_camera'], 'reference_status': ref['status'],
                   'reference_invalid_reasons': ref['invalid_reasons'], 'frame0_owner': ref['frame0_owner'], 'u': r_.get('u'), 'v': r_.get('v'), 'Z': r_.get('Z'),
                   'cell': r_.get('cell'), 'du_dv': [r_.get('du'), r_.get('dv')], 'depth4': r_.get('depth4'), 'rgb4': r_.get('rgb4'), 'mask4': r_.get('mask4'),
                   'raw_valid4': r_.get('raw_valid4'), 'sample_reasons_at_x_ref': r_.get('reasons_true'), 'r_t': r_.get('r'), 'R_t': r_.get('R'), 'H_t': r_.get('H'),
                   'b_c0': fl(ref['b']) if ref['valid'] else 'NO_VALID_REFERENCE', 'R_0': fl(ref['var0']) if ref['valid'] else None, 'source_note': ref['source_note']}
            if ref['valid'] and 'r' in r_:
                rd = np.array(r_['r'])-ref['b']; row['r_delta_t'] = fl(rd); row['r_delta_z_ALGEBRA_ONLY'] = fl(rd/np.sqrt(np.array(r_['R'])))
                row['pixels_bitwise_identical_to_frame0'] = bool(np.array_equal(np.array(r_['rgb4']), np.array(ref['rgb4_0'])) and np.array_equal(np.array(r_['depth4']), np.array(ref['depth4_0'])))
                row['r_t_equals_r_0_exactly'] = bool(np.array_equal(np.array(r_['r']), ref['b']))
                row['H_t_equals_H_0'] = bool(np.allclose(np.array(r_['H']), np.array(ref['H0']), rtol=0, atol=0))
            out.append(row)
    return out


def uncertainty_rows(sc, pid, k, refs):
    x = sc.x0[pid]; a = sc.run.can['canonical_rgb'][pid]; out = []
    for c, ref in refs.items():
        if not ref['valid']: out.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'camera': c, 'status': 'NO_VALID_REFERENCE'}); continue
        r_ = all_reasons(sc.images(k)[c], sc.cams[c], x, a, sc.noise, sc.policy, sc.cov0(pid)[:3, :3], np.zeros(3), O.VALID)
        if 'r' not in r_: continue
        Rt, R0 = np.array(r_['R']), ref['var0']
        same = bool(np.array_equal(np.array(r_['rgb4']), np.array(ref['rgb4_0'])) and np.array_equal(np.array(r_['depth4']), np.array(ref['depth4_0'])))
        out.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'camera': c, 'status': 'VALID_REFERENCE', 'R_t': fl(Rt), 'R_0': fl(R0),
                    'R_delta_ALGEBRA_ONLY': fl(Rt), 'R_delta_INDEPENDENT_SAMPLES_APPROX': fl(Rt+R0),
                    'R_delta_MEASURED_CORRELATION_DIAGNOSTIC': 'NOT_ESTIMABLE_FROM_AVAILABLE_DATA',
                    'empirical_note': ('the 2x2 pixels at x_ref are BITWISE identical at frame 0 and frame t: for these deterministic synthetic renders the '
                                       'sample "noise" is perfectly correlated (C_t0 = R_0 = R_t would give R_delta = 0, a degenerate likelihood); no '
                                       'empirical pixel covariance exists to estimate C_t0') if same else 'pixels changed between frame 0 and t',
                    'source_camera_warning': ref['source_note']})
    return out


# ---------------------------------------------------------------------------------------------- B: crease geometry
def crease_audit(sc, pid, cam_id, frames=(0, 1), half=3):
    """Camera cam_id around pi(x_ref): 7x7 depth/mask at frames 0 and 1, the 2x2 corners, metric and inverse interpolation,
    first and second differences, single-plane consistency of the 2x2 (planarity of the four corners in inverse depth: an
    affine function of (u, v) has zero mixed difference), a side partition by the largest second difference and per-side
    inverse-depth plane fits (held-out corner check). Returns rows and the per-side depth models (B5 oracle candidates)."""
    cam = sc.cams[cam_id]; x = sc.x0[pid]; u, v, Z, Ju, Jz = cam.project(x)
    u0, v0 = int(math.floor(u)), int(math.floor(v)); rows = []; models = {}
    for t in frames:
        img = sc.images(t)[cam_id]
        P = img.depth[v0-half+1:v0+half+2, u0-half+1:u0+half+2].astype(float); M = img.mask[v0-half+1:v0+half+2, u0-half+1:u0+half+2]
        uu, vv = np.meshgrid(np.arange(u0-half+1, u0+half+2), np.arange(v0-half+1, v0+half+2))
        D4 = np.array([img.depth[v0, u0], img.depth[v0, u0+1], img.depth[v0+1, u0], img.depth[v0+1, u0+1]], float)
        du, dv = u-u0, v-v0; w = np.array([(1-du)*(1-dv), du*(1-dv), (1-du)*dv, du*dv])
        mixed_inv = float(1/D4[0]-1/D4[1]-1/D4[2]+1/D4[3])                     # zero for one plane (1/Z affine in u, v)
        col, row = half-1, half-1                                               # the 2x2 cell's first column / row in patch coordinates
        d2v = np.diff(P, 2, axis=0); d2u = np.diff(P, 2, axis=1)
        k_v = int(np.argmax(np.abs(d2v[:, col]))); k_u = int(np.argmax(np.abs(d2u[row, :])))
        axis = 'v' if abs(d2v[k_v, col]) >= abs(d2u[row, k_u]) else 'u'
        kink = (k_v if axis == 'v' else k_u)+1                                  # patch row / column of the largest second difference
        sides = {}
        if t == 0:
            sig = math.sqrt(sc.noise.sigma2(float(np.nanmean(P))))
            tol = max(3*sig, 3*np.finfo(np.float32).eps*float(np.nanmean(P))*4)   # classification only: fit residual vs sensor sigma
            idx = np.arange(P.shape[0])[:, None]*np.ones((1, P.shape[1])) if axis == 'v' else np.ones((P.shape[0], 1))*np.arange(P.shape[1])[None, :]
            hold = np.zeros(P.shape, bool); hold[row:row+2, col:col+2] = True        # the 2x2 cell corners are held out of the fits
            for side, sel in (('A', idx < kink), ('B', idx > kink)):          # the kink line itself is in neither side
                fit = sel & M & ~hold & np.isfinite(P) & (P > 0)
                if fit.sum() < 6: sides[side] = {'status': 'UNDETERMINED (too few pixels on this side)'}; continue
                A = np.stack([np.ones(fit.sum()), uu[fit], vv[fit]], 1); coef, *_ = np.linalg.lstsq(A, 1/P[fit], rcond=None)
                rms = float(np.sqrt(np.mean((1/(A@coef)-P[fit])**2)))
                held = [(int(uu[i, j]), int(vv[i, j]), float(P[i, j]), float(1/(coef@[1, uu[i, j], vv[i, j]]))) for i, j in zip(*np.nonzero(sel & hold & M))]
                sides[side] = {'status': ('FIT_PLANAR' if rms <= tol else 'UNDETERMINED (side not planar: no clean row/column partition)'),
                               'inverse_depth_plane_coef': fl(coef), 'fit_rms_m': rms, 'tolerance_m': tol, 'pixels': int(fit.sum()), 'held_out_2x2_corners': held,
                               'depth_at_x_ref_projection_m': float(1/(coef@[1, u, v])), 'Z_minus_plane_depth_at_x_ref_m': float(Z-1/(coef@[1, u, v]))}
                # a side is a candidate same-surface model of THIS point only if its plane passes through x_ref within the
                # row's depth noise (3 sqrt(R_D) at x_ref; x_ref is the non-GT source back-projection): otherwise it is another face
                sdD = math.sqrt(sc.noise.sigma2(Z)+sc.noise.depth_model)
                sides[side]['passes_through_x_ref'] = bool(abs(sides[side]['Z_minus_plane_depth_at_x_ref_m']) <= 3*sdD)
                sides[side]['x_ref_consistency_tolerance_m'] = 3*sdD
                if rms <= tol and sides[side]['passes_through_x_ref']:
                    c_ = coef.copy()
                    def model(uq, vq, c_=c_):
                        wi = c_@[1, uq, vq]
                        if wi <= 0: return None, None
                        Dq = 1/wi; return Dq, -Dq*Dq*np.array([c_[1], c_[2]])
                    models[side] = model
        k_v, k_u = (kink-1, k_u) if axis == 'v' else (k_v, kink-1)
        rows.append({'scene': sc.name, 'point_id': pid, 'camera': cam_id, 'frame': t, 'u': u, 'v': v, 'Z_x_ref': Z, 'cell': [u0, v0], 'du_dv': [du, dv],
                     'patch_origin_uv': [u0-half+1, v0-half+1], 'depth_patch_m': P.tolist(), 'mask_patch': M.tolist(), 'corners_2x2_m': fl(D4),
                     'metric_interp_m': float(w@D4), 'inverse_interp_m': float(1/(w@(1/D4))), 'r_D_metric_m': float(Z-w@D4),
                     'mixed_difference_inverse_depth_2x2': mixed_inv, 'first_diff_v_m': np.diff(P, axis=0)[:, col].tolist(), 'first_diff_u_m': np.diff(P, axis=1)[row].tolist(),
                     'second_diff_v_m': d2v[:, col].tolist(), 'second_diff_u_m': d2u[row].tolist(), 'crease_axis': axis, 'kink_patch_index': kink,
                     'crease_second_difference_m': float(d2v[k_v, col] if axis == 'v' else d2u[row, k_u]),
                     'temporally_unchanged_vs_frame0': None if t == 0 else bool(np.array_equal(P, rows[0]['_P'])), 'side_fits': sides if t == 0 else None, '_P': P})
    # single-plane consistency of the 2x2 in inverse depth relative to the declared sensor noise (classification only)
    r0 = rows[0]; D4 = np.array(r0['corners_2x2_m']); sig = math.sqrt(sc.noise.sigma2(float(D4.mean())))
    sig_mixed = 2*sig/float(D4.mean())**2                                       # std of the mixed difference of 1/D for 4 independent corners
    r0['single_plane_2x2'] = ('INCONSISTENT' if abs(r0['mixed_difference_inverse_depth_2x2']) > 3*sig_mixed else 'CONSISTENT') + \
        f" (|mixed 1/D difference| {abs(r0['mixed_difference_inverse_depth_2x2']):.3g} vs 3 sigma {3*sig_mixed:.3g} 1/m, sigma_D = {sig:.3g} m; a 2x2 can be planar across a crease line that does not cut it)"
    for r in rows: r.pop('_P')
    return rows, models


# ------------------------------------------------------------------------------- C: moving-point reacquisition
def moving_case(sc, pid, refs):
    from tools.forensic_f1_drift import GroundTruth
    gt = GroundTruth(sc.run, Path(sc.run.man['arguments']['scene']))
    tl = [r for r in csv.DictReader(open(sc.fdir/'f1_state_timeline.csv')) if int(r['point_id']) == pid]
    acc = [int(r['frame']) for r in tl if r['acceptance'].startswith('ACCEPTED')]
    loss = next((int(r['frame']) for r in tl if acc and int(r['frame']) > acc[0] and not r['acceptance'].startswith('ACCEPTED')), None)
    last_ok = max((f for f in acc if loss is None or f < loss), default=None)
    rows, sens, reacq = [], [], []
    a = sc.run.can['canonical_rgb'][pid]
    for k in sorted({1, 2} | ({last_ok} if last_ok else set()) | ({loss} if loss else set())):
        tr = sc.trace(k); sp = str(pid); pre = tr['prediction']['points'][sp]; fin = tr['final'][sp]
        sm, Pm = np.array(pre['s_minus']), np.array(pre['P_minus']); xg = gt.moved(sc.x0[pid], k); xr = sc.x0[pid]
        it0 = tr['iterations'][0]['points'][sp]['cameras']; frz = frozen_codes(it0)
        used = sorted({r['camera'] for r in fin.get('rows', [])}) if fin['acceptance'].startswith('ACCEPTED') else []
        for c, cam in enumerate(sc.cams):
            img = sc.images(k)[c]; rec = {'scene': sc.name, 'point_id': pid, 'frame': k, 'camera': c,
                                          'role': 'last_successful' if k == last_ok else ('first_zero_row' if k == loss else 'early'),
                                          'acceptance': fin['acceptance'], 'USED_IN_F1': c in used, 'frozen_assignment': NAMES[frz[c]],
                                          'owner_front_to_back': it0[c]['pixel_occupants_front_to_back'][:3]}
            for nm, x, dx, P in (('x_pred', sm[:3], np.zeros(3), Pm), ('x_GTprop_prior_centered', xg, xg-sm[:3], Pm),
                                 ('x_GTprop_recentered_ORACLE', xg, np.zeros(3), Pm)):
                r_ = all_reasons(img, cam, x, a, sc.noise, sc.policy, P[:3, :3], dx, frz[c])
                rec.update({f'{nm}_{kk}': r_.get(kk) for kk in ('u', 'v', 'Z', 'cell', 'mask4', 'depth4', 'r', 'R', 'S_DD', 'nis', 'nu_D', 'reasons_true', 'valid_production_rules')})
            ug, vg, Zg, _, _ = cam.project(xg)
            if 0 <= round(ug) < cam.W and 0 <= round(vg) < cam.H and Zg > cam.near:
                Dm = img.depth[int(round(vg)), int(round(ug))]
                rec['GT_ORACLE_ONLY_rendered_depth_visibility'] = ('PART_SURFACE_VISIBLE_MATERIAL_POINT_UNKNOWN (rendered D = GT Z within 5 mm)'
                                                                   if img.mask[int(round(vg)), int(round(ug))] and abs(Dm-Zg) < 5e-3 else f'OTHER_SURFACE_OR_OCCLUDED (D - Z = {Dm-Zg:+.4f} m)')
            else: rec['GT_ORACLE_ONLY_rendered_depth_visibility'] = 'OUTSIDE_IMAGE'
            # reference-differenced residual at x_pred (diagnostic only) and the motion-evidence sensitivity
            ref = refs[c]
            if ref['valid']:
                def rd(x, img=img, cam=cam):
                    r__ = all_reasons(img, cam, x, a, sc.noise, sc.policy, Pm[:3, :3], np.zeros(3), O.VALID)
                    if 'r' not in r__: return None, None
                    rr = np.array(r__['r'])-ref['b']; return rr, float(((rr**2)/np.array(r__['R'])).sum())
                rp, chp = rd(sm[:3]); rgt, chg = rd(xg); rre, chr_ = rd(xr)
                orig = lambda x: (lambda q: float(((np.array(q['r'])**2)/np.array(q['R'])).sum()) if 'r' in q else None)(all_reasons(img, cam, x, a, sc.noise, sc.policy, Pm[:3, :3], np.zeros(3), O.VALID))
                rec.update(reference_status='VALID_REFERENCE', r_delta_x_pred=fl(rp) if rp is not None else None, chi2_delta_x_pred=chp)
                sens.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'camera': c, 'reference_status': 'VALID_REFERENCE', 'USED_IN_F1': c in used,
                             'chi2_original_at_x_ref': orig(xr), 'chi2_original_at_x_pred': orig(sm[:3]), 'chi2_original_at_x_GTprop_ORACLE': orig(xg),
                             'chi2_delta_at_x_ref': chr_, 'chi2_delta_at_x_pred': chp, 'chi2_delta_at_x_GTprop_ORACLE': chg,
                             'gt_motion_from_x_ref_m': float(np.linalg.norm(xg-xr)), 'GT_vis': rec['GT_ORACLE_ONLY_rendered_depth_visibility']})
            else:
                rec['reference_status'] = 'NO_VALID_REFERENCE'
                sens.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'camera': c, 'reference_status': 'NO_VALID_REFERENCE', 'USED_IN_F1': c in used})
            rec.update(velocity_m_s=fl(sm[3:]), pred_to_GT_m=float(np.linalg.norm(sm[:3]-xg)), gt_motion_m=float(np.linalg.norm(xg-xr)))
            rows.append(rec)
    # verdict on motion evidence (classification thresholds named here, reporting only): the differenced residual retains
    # motion evidence at a frame if, on valid-reference cameras with samples at both locations, the GT-propagated position
    # explains the differenced data clearly better than the static hypothesis (sum chi2_delta(x_GT) < 0.25 sum chi2_delta(x_ref))
    verdict = {}
    for k in sorted({s['frame'] for s in sens}):
        v = [s for s in sens if s['frame'] == k and s.get('chi2_delta_at_x_ref') is not None and s.get('chi2_delta_at_x_GTprop_ORACLE') is not None]
        if not v: verdict[k] = 'UNDETERMINED (no valid-reference camera samples both locations)'; continue
        a_, b_ = sum(s['chi2_delta_at_x_GTprop_ORACLE'] for s in v), sum(s['chi2_delta_at_x_ref'] for s in v)
        verdict[k] = ('MOTION_EVIDENCE_RETAINED' if a_ < .25*b_ else 'MOTION_EVIDENCE_LOST' if a_ >= .8*b_ else 'UNDETERMINED') + f' (sum chi2_delta x_GT {a_:.3g} vs x_ref {b_:.3g}, {len(v)} cameras)'
    return rows, sens, {'last_successful_frame': last_ok, 'first_zero_row_frame': loss, 'motion_evidence': verdict}


# ------------------------------------------------------------------------------------------------------- figures
def fig_arms(path, title, arms_, iters):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    names = [a['arm'] for a in arms_]; y = np.arange(len(names))
    ax[0].barh(y, [1e3*a['abs_dx_m'] for a in arms_]); ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=7); ax[0].set_xlabel('|accepted dx| (mm)')
    ax[1].barh(y, [a['abs_dv_m_s'] for a in arms_]); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel('|accepted dv| (m/s)')
    for nm in ('BASELINE', 'REFERENCE_RGB_ONLY', 'REFERENCE_DEPTH_ONLY', 'REFERENCE_RGBD'):
        it = [r for r in iters if r['arm'] == nm]
        if it: ax[2].plot([r['iteration'] for r in it], [1e3*np.linalg.norm(r['dx_from_prior_m']) for r in it], 'o-', label=nm)
    ax[2].set_xlabel('inner iteration j'); ax[2].set_ylabel('|x_j - x-| (mm)'); ax[2].legend(fontsize=7)
    fig.suptitle(title+' (each arm scored with its own residual model; costs are not comparable across models)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_crease(path, crows, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(crows), figsize=(6*len(crows), 5))
    ax = np.atleast_1d(ax); lo = min(np.nanmin(np.array(r['depth_patch_m'])) for r in crows); hi = max(np.nanmax(np.array(r['depth_patch_m'])) for r in crows)
    for a_, r in zip(ax, crows):
        P = np.array(r['depth_patch_m']); u0, v0 = r['patch_origin_uv']
        im = a_.imshow(P, extent=(u0-.5, u0+P.shape[1]-.5, v0+P.shape[0]-.5, v0-.5), vmin=lo, vmax=hi, interpolation='nearest')
        plt.colorbar(im, ax=a_, label='depth (m)')
        for i in range(P.shape[0]):
            for j in range(P.shape[1]): a_.text(u0+j, v0+i, f'{P[i, j]:.4f}', ha='center', va='center', fontsize=6, color='w')
        a_.plot(r['u'], r['v'], 'r+', ms=14); a_.add_patch(plt.Rectangle((r['cell'][0], r['cell'][1]), 1, 1, fill=False, ec='r'))
        a_.set_title(f"cam {r['camera']} frame {r['frame']}: crease axis {r['crease_axis']}, 2nd diff {1e3*r['crease_second_difference_m']:.1f} mm; r_D(x_ref) {1e3*r['r_D_metric_m']:.2f} mm", fontsize=8)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_objective_models(path, sc, pid, k, refs, base_res, title):
    """Original model and reference model objectives along the original accepted path (separate panels: not comparable)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    po = make(sc, k, pid); pr = make(sc, k, pid, refs=refs, channels='RGBD')
    S_prior = [o['camera'] for o in po.used(po.rows(po.s_minus[:3]))]
    sm, sa = po.s_minus, base_res['s_plus']; al = np.linspace(-.25, 1.25, 91); jo, jr = [], []
    for a_ in al:
        s = sm+a_*(sa-sm); jo.append(model_cost(po, s, S_prior)[0]); jr.append(model_cost(pr, s, S_prior)[0])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(al, [np.nan if v is None else v for v in jo]); ax[0].set_title('ORIGINAL model J on S_prior'); ax[0].set_yscale('symlog')
    ax[1].plot(al, [np.nan if v is None else v for v in jr], 'g'); ax[1].set_title('REFERENCE[RGBD] model J on S_prior'); ax[1].set_yscale('symlog')
    for a_ in ax: a_.axvline(0, color='g'); a_.axvline(1, color='r'); a_.set_xlabel('alpha along the ORIGINAL accepted step (0 = s-, 1 = accepted)')
    fig.suptitle(title+f' rows {S_prior}; gaps = not evaluable', fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_gate(path, rows, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fr = sorted({r['frame'] for r in rows}); fig, ax = plt.subplots(1, len(fr), figsize=(6*len(fr), 4), squeeze=False)
    thr = O.row_gate_threshold(3.0)
    for a_, k in zip(ax[0], fr):
        rr = [r for r in rows if r['frame'] == k]; c = np.array([r['camera'] for r in rr])
        for off, key, lab in ((-.25, 'x_pred_nis', 'x_pred (prior centred)'), (0, 'x_GTprop_prior_centered_nis', 'x_GT, prior centred'), (.25, 'x_GTprop_recentered_ORACLE_nis', 'x_GT recentred (ORACLE)')):
            y = [r[key] if r[key] is not None else np.nan for r in rr]; a_.bar(c+off, y, width=.25, label=lab)
        a_.axhline(thr, color='r', ls='--', label='chi2_4 gate'); a_.set_yscale('log'); a_.set_xlabel('camera'); a_.set_ylabel('row NIS')
        a_.set_title(f"frame {k} ({rr[0]['role']}); used {[r['camera'] for r in rr if r['USED_IN_F1']]}", fontsize=8); a_.legend(fontsize=6)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_knife(path, recs, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for model in sorted({r['model'] for r in recs}):
        rr = [r for r in recs if r['model'] == model]
        ax[0].plot([r['eps_px'] for r in rr], [1e3*r['abs_dx_m'] for r in rr], 'o', label=model)
        for r in rr: ax[1].semilogy(range(len(r['iterate_position_linf_m_vs_unshifted'])), np.maximum(r['iterate_position_linf_m_vs_unshifted'], 1e-20), alpha=.6)
    ax[0].set_xscale('symlog', linthresh=1e-13); ax[0].set_xlabel('pixel shift (px)'); ax[0].set_ylabel('|accepted dx| (mm)'); ax[0].legend(fontsize=7)
    ax[1].set_xlabel('inner iteration'); ax[1].set_ylabel('position |x_j(shift) - x_j(0)|_inf (m)')
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- driver
def run_case(case, sc, out, R):
    scene, pid, k, role = CASES[case]
    own0 = sc.frame0_ownership([pid])[pid]; refs = references(sc, pid, own0)
    R['references'] += [{'case': case, 'scene': scene, 'point_id': pid, **{kk: (fl(vv) if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items() if kk not in ('H0',)}}
                        for r in refs.values()]
    md = [f'# {case}: {scene} point {pid} ({role})', '']
    if role == 'moving':
        rows, sens, info = moving_case(sc, pid, refs)
        R['reacq'] += rows; R['moving'] += sens
        md += [f"- last successful update frame {info['last_successful_frame']}, first zero-row frame {info['first_zero_row_frame']}",
               f"- motion evidence under the reference-differenced residual (GT used ONLY as the sensitivity oracle): {json.dumps(info['motion_evidence'])}", '',
               '| frame | cam | used | frozen | reasons at x_pred | NIS x_pred | reasons x_GT (prior-centred) | NIS x_GT prior-centred | NIS x_GT recentred (ORACLE) | GT visibility (ORACLE) | reference |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            f_ = lambda v: '-' if v is None else f'{v:.3g}'
            md.append(f"| {r['frame']} | {r['camera']} | {r['USED_IN_F1']} | {r['frozen_assignment']} | {r['x_pred_reasons_true']} | {f_(r['x_pred_nis'])} | "
                      f"{r['x_GTprop_prior_centered_reasons_true']} | {f_(r['x_GTprop_prior_centered_nis'])} | {f_(r['x_GTprop_recentered_ORACLE_nis'])} | "
                      f"{r['GT_ORACLE_ONLY_rendered_depth_visibility']} | {r['reference_status']} |")
        if pid == 87290: fig_gate(out/'figures'/'usb_87290_last_camera_gate.png', [r for r in rows if r['role'] != 'early'], f'USB 87290 last-camera gate (frames {info})')
        R['case_info'][case] = info
        return md
    R['reftab'] += reference_table(sc, pid, refs)
    R['unc'] += uncertainty_rows(sc, pid, k, refs)
    extra = []
    if case == 'usb_67827':
        crows, models = crease_audit(sc, pid, 6); R['crease'] += crows
        crows4, _ = crease_audit(sc, pid, 4); R['crease'] += [{**r, 'note': 'camera 4 was FROZEN OUT (DUPLICATE) in production: diagnostic probe only'} for r in crows4]
        fig_crease(out/'figures'/'usb_67827_depth_crease_frame0_frame1.png', crows, 'USB 67827 camera 6 depth 7x7 around pi(x_ref), frames 0 and 1')
        for side, m in models.items():
            extra.append((f'B5_camera6_depth_from_side_{side}_plane_ORACLE', {'depth_models': {6: m}, '_label': f'original + cam6 depth from side-{side} inverse-depth plane (post-hoc oracle)'}))
        if not models:
            R['arms'].append({'case': case, 'scene': scene, 'point_id': pid, 'frame': k, 'arm': 'B5_same_surface_depth_oracle', 'model': '-',
                              'acceptance': 'NOT_EXECUTED: UNPROVEN_SURFACE (no side of the camera-6 crease is a planar patch passing through x_ref)',
                              'abs_dx_m': float('nan'), 'abs_dv_m_s': float('nan'), 'dx_m': [], 'dv_m_s': [], 'rows_initial': [], 'rows_accepted': [],
                              'iterations': 0, 'converged': None, 'J_model_prior_state_on_S_prior': None, 'J_model_accepted_state_on_S_prior': None, 's_plus': []})
        R['case_info'][case] = {'camera6_side_fits': crows[0]['side_fits'], 'camera6_single_plane_2x2': crows[0]['single_plane_2x2'],
                                'camera6_temporally_unchanged': crows[1]['temporally_unchanged_vs_frame0'], 'face_identity': 'FACE_IDENTITY_UNPROVEN (no mesh / ray-cast GT available)'}
        extra.append(('B1_REFERENCE_DEPTH_camera6_only', {'refs': {6: refs[6]}, 'channels': 'D', '_label': 'reference[D] camera 6 only'}))
        extra.append(('B2_REFERENCE_RGB_all_cameras_DEPTH_camera6_only', {'refs': {c: {**refs[c], 'channels': 'RGBD' if c == 6 else 'RGB'} for c in refs}, '_label': 'reference: RGB on every valid camera, depth on camera 6 only'}))
    arms_, S_prior, res = hierarchy(sc, pid, k, refs, R['iters'], extra)
    R['arms'] += [{'case': case, **a} for a in arms_]
    if case == 'usb_67827':
        # B7: suppressed camera 4 contributes nothing to the production information sum
        p = make(sc, k, pid); rows = p.rows(p.s_minus[:3])
        R['case_info'][case]['B7_camera4_in_baseline_rows'] = any(o['camera'] == 4 for o in p.used(rows))
        R['case_info'][case]['B7_camera4_candidacy_at_s_minus'] = NAMES[[o for o in rows if o['camera'] == 4][0]['reason']]
        fig_objective_models(out/'figures'/'usb_67827_original_vs_reference_objective.png', sc, pid, k, refs, res['BASELINE'], 'USB 67827 frame 1')
    if case == 'storage_8444':
        fig_arms(out/'figures'/'storage_8444_reference_vs_original.png', 'Storage 8444 frame 1', arms_, [r for r in R['iters'] if r['point_id'] == pid])
        c0 = cf0(sc, pid)
        R['case_info'][case] = {'A6_frame0_fusion_then_frame1': {kk: c0[kk] for kk in ('B_frame0', 'A_frame1', 'B_frame1')}}
    if role in ('knife_edge', 'primary'):
        recs = []
        for model, kw in (('original', {}), ('reference[RGBD]/ALGEBRA_ONLY', {'refs': refs, 'channels': 'RGBD'})):
            p0 = make(sc, k, pid, **kw); r0 = run_prob(p0, sc)
            for e in EPS_PX:
                p = make(sc, k, pid, cams=[ShiftCam(c, e, e) for c in sc.cams], **kw); r = run_prob(p, sc)
                it = compare_iterates([h['s'] for h in r['history']], [h['s'] for h in r0['history']])
                recs.append({'case': case, 'model': model, 'eps_px': e, 'abs_dx_m': float(np.linalg.norm((r['s_plus']-p.s_minus)[:3])),
                             'abs_dv_m_s': float(np.linalg.norm((r['s_plus']-p.s_minus)[3:])),
                             **{f'accepted_vs_unshifted_{kk}': vv for kk, vv in compare_state6(r['s_plus'], r0['s_plus']).items()},
                             'iterate_position_linf_m_vs_unshifted': it['position_linf_m_per_iterate'], 'iterate_velocity_linf_m_s_vs_unshifted': it['velocity_linf_m_s_per_iterate'],
                             'first_iteration_position_diverging_over_1e-9_m': it['first_iteration_position_diverging_over_1e-9_m'],
                             'first_iteration_velocity_diverging_over_1e-9_m_s': it['first_iteration_velocity_diverging_over_1e-9_m_s'], 'acceptance': ACC[r['acceptance_code']],
                             'converged': r['converged'], 'unshifted_abs_dx_m': float(np.linalg.norm((r0['s_plus']-p0.s_minus)[:3]))})
        R['knife'] += recs
        if case == 'storage_8416': fig_knife(out/'figures'/'storage_8416_one_sided_sensitivity.png', recs, 'Storage 8416 frame 1: pixel-shift sensitivity, original vs reference model')
    md += case_markdown(case, sc, pid, k, refs, arms_, S_prior, R)
    return md


def case_markdown(case, sc, pid, k, refs, arms_, S_prior, R):
    f_ = lambda v, s=1., p=3: '-' if v is None else f'{v*s:.{p}g}'
    L = [f'Decisive frame {k}; rows admitted at the prediction (production frozen ownership): {S_prior}.', '',
         '## Frame-0 references (fixed x_ref)', '', '| cam | status | invalid reasons | frame-0 owner | b_c0 (RGB, D) | source |', '|---|---|---|---|---|---|']
    for c, r in refs.items():
        L.append(f"| {c} | {r['status']} | {r['invalid_reasons']} | {r['frame0_owner']} | {np.round(r['b'], 6).tolist() if r['valid'] else '-'} | {r['source_note'][:40]} |")
    L += ['', '## Fixed x_ref, frames 0-5, admitted cameras', '', '| frame | cam | r_t | r_delta_t | pixels identical to frame 0 | r_t == r_0 exactly |', '|---|---|---|---|---|---|']
    for r in R['reftab']:
        if r['point_id'] == pid and r['camera'] in S_prior and r['scene'] == sc.name:
            L.append(f"| {r['frame']} | {r['camera']} | {np.round(r['r_t'], 6).tolist() if r['r_t'] else '-'} | {np.round(r.get('r_delta_t'), 6).tolist() if r.get('r_delta_t') else r['b_c0'] if isinstance(r['b_c0'], str) else '-'} | "
                     f"{r.get('pixels_bitwise_identical_to_frame0', '-')} | {r.get('r_t_equals_r_0_exactly', '-')} |")
    L += ['', '## Arms (each scored within its own model; J columns are NOT comparable across models)', '',
          '| arm | model | acceptance | iters / conv | rows initial -> accepted | dx (mm) | dv (m/s) | J_model before -> after on S_prior |', '|---|---|---|---|---|---|---|---|']
    for a in arms_:
        L.append(f"| {a['arm']} | {a['model']} | {a['acceptance']} | {a['iterations']} / {a['converged']} | {a['rows_initial']} -> {a['rows_accepted']} | "
                 f"{np.round(np.array(a['dx_m'])*1e3, 3).tolist()} | {np.round(a['dv_m_s'], 4).tolist()} | {f_(a['J_model_prior_state_on_S_prior'])} -> {f_(a['J_model_accepted_state_on_S_prior'])} |")
    b = arms_[0]
    L += ['', f"Baseline parity vs production: position |dx|_inf = {b.get('production_parity_position_linf_m'):.2g} m, velocity |dv|_inf = "
              f"{b.get('production_parity_velocity_linf_m_s'):.2g} m/s; production rows {b.get('production_rows')}."]
    return L


def write_result(out, R, root, second):
    A = {(a['case'], a['arm']): a for a in R['arms']}
    def g(case, arm, key, s=1.):
        a = A.get((case, arm)); return None if a is None else (a[key]*s if isinstance(a[key], float) else a[key])
    L = ['# Third pass: temporal camera references, depth creases, moving-point reacquisition (standard F1, offline)', '',
         'Generated by `tools/f1_temporal_reference_diagnostics.py`; every number comes from single-point / single-frame replays (`csv/`).',
         'Reference model = r - b_c0 with b_c0 = r_c,0(x_ref) (frame 0, fixed x_ref). Costs of different residual models are never ranked against each other.', '']
    for case in ('storage_8444', 'usb_67827'):
        if (case, 'BASELINE') not in A: continue
        L += [f'## {case}', '', '| arm | dx (mm) | dv (m/s) | acceptance | rows |', '|---|---|---|---|---|']
        for (c, nm), a in A.items():
            if c == case: L.append(f"| {nm} | {np.round(np.array(a['dx_m'])*1e3, 3).tolist()} | {np.round(a['dv_m_s'], 4).tolist()} | {a['acceptance']} | {a['rows_initial']} -> {a['rows_accepted']} |")
        L.append('')
    if 'usb_67827' in R['case_info']:
        ci = R['case_info']['usb_67827']
        L += ['## usb_67827 camera-6 geometry', '', f"- 2x2 single plane: {ci['camera6_single_plane_2x2']}", f"- temporally unchanged (frame 1 vs 0): {ci['camera6_temporally_unchanged']}",
              f"- per-side inverse-depth plane fits (2x2 corners held out): {json.dumps(ci['camera6_side_fits'], default=float)[:1500]}",
              f"- {ci['face_identity']}", f"- B7: camera 4 in baseline rows: {ci.get('B7_camera4_in_baseline_rows')} (candidacy at s-: {ci.get('B7_camera4_candidacy_at_s_minus')})", '']
    for case in ('usb_87290', 'usb_85696'):
        if case in R['case_info']: L += [f'## {case}', '', f"- {json.dumps(R['case_info'][case])}", '']
    if R['knife']:
        L += ['## Pixel-shift sensitivity', '', '| case | model | shift (px) | accepted dx (mm) | accepted position change vs unshifted (m, linf) | velocity change (m/s, linf) | first diverging iteration position / velocity |', '|---|---|---|---|---|---|---|']
        for r in R['knife']:
            L.append(f"| {r['case']} | {r['model']} | {r['eps_px']:+.0e} | {r['abs_dx_m']*1e3:.3f} | {r['accepted_vs_unshifted_position_linf_m']:.2g} | {r['accepted_vs_unshifted_velocity_linf_m_s']:.2g} | "
                     f"{r['first_iteration_position_diverging_over_1e-9_m']} / {r['first_iteration_velocity_diverging_over_1e-9_m_s']} |")
    L += ['', '## Verdicts V1-V5', '', 'Filled from the measurements after review: see the terminal summary and `cases/*.md`. Automatic facts:', '']
    for case in ('storage_8444', 'usb_67827'):
        b, rgb, d, rgbd, fe = (A.get((case, n)) for n in ('BASELINE', 'REFERENCE_RGB_ONLY', 'REFERENCE_DEPTH_ONLY', 'REFERENCE_RGBD', 'REFERENCE_FIXED_ELIGIBILITY'))
        if b: L.append(f"- {case}: baseline |dx| {b['abs_dx_m']*1e3:.3f} mm / |dv| {b['abs_dv_m_s']:.3f}; reference RGB-only {rgb['abs_dx_m']*1e3:.3f} mm / {rgb['abs_dv_m_s']:.3f}; "
                       f"depth-only {d['abs_dx_m']*1e3:.3f} / {d['abs_dv_m_s']:.3f}; RGBD {rgbd['abs_dx_m']*1e3:.3f} / {rgbd['abs_dv_m_s']:.3f}; fixed-eligibility RGBD "
                       f"{fe['abs_dx_m']*1e3 if fe else float('nan'):.3f} mm")
    (out/'THIRD_PASS_CAUSAL_RESULT.md').write_text('\n'.join(L)+'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forensics-root', type=Path, required=True); ap.add_argument('--second-pass-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--case', help=f'comma list of {list(CASES)} (default: all)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    cases = a.case.split(',') if a.case else list(CASES)
    bad = [c for c in cases if c not in CASES]
    if bad: raise SystemExit(f'unknown cases {bad}')
    inputs = {'second_pass': {f.name: sha(f) for f in sorted(a.second_pass_root.glob('*')) if f.is_file()} if a.second_pass_root.exists() else 'MISSING',
              'sources': {s: sha(REPO/s) for s in SOURCES}}
    frames = {}
    for c in cases:
        scene, pid, k, role = CASES[c]
        fr = sorted(set(range(0, 6)) | ({k} if k else set()) | ({1, 2, 3} if role == 'moving' else set()))
        frames[c] = {'scene': scene, 'point_id': pid, 'decisive_frame': k, 'role': role, 'frames': fr,
                     'trace_files': {f: sha(a.forensics_root/scene/'traced'/'free_forward'/'numeric'/f'trace_frame_{f:04d}.json') for f in fr if f > 0}}
    inputs['cases'] = frames
    if a.dry_run:
        print(json.dumps(inputs, indent=1)); return
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit(f'{a.out} exists and is not empty: never overwritten')
    for d in ('cases', 'csv', 'traces', 'figures'): (a.out/d).mkdir(parents=True, exist_ok=True)
    write_json(a.out/'traces'/'input_hashes.json', inputs)
    t0 = time.perf_counter(); scenes = {}
    R = {k: [] for k in ('references', 'reftab', 'unc', 'arms', 'iters', 'crease', 'reacq', 'moving', 'knife')}; R['case_info'] = {}
    for c in cases:
        scene = CASES[c][0]; sc = scenes.setdefault(scene, Scene(a.forensics_root, scene))
        print(f'[third-pass] {c} ({time.perf_counter()-t0:.0f} s)', flush=True)
        md = run_case(c, sc, a.out, R)
        (a.out/'cases'/f'{c}.md').write_text('\n'.join(md)+'\n')
    wcsv(a.out/'csv'/'reference_residuals_frames0_to5.csv', R['reftab']); wcsv(a.out/'csv'/'frame0_references.csv', R['references'])
    wcsv(a.out/'csv'/'differential_uncertainty.csv', R['unc']); wcsv(a.out/'csv'/'arm_outcomes.csv', R['arms'])
    wcsv(a.out/'csv'/'per_iteration_objective_and_rows.csv', R['iters']); wcsv(a.out/'csv'/'per_camera_depth_crease.csv', R['crease'])
    wcsv(a.out/'csv'/'last_camera_reacquisition.csv', R['reacq']); wcsv(a.out/'csv'/'moving_sensitivity.csv', R['moving'])
    wcsv(a.out/'csv'/'pixel_shift_sensitivity.csv', R['knife'])
    np.savez_compressed(a.out/'traces'/'exact_replays.npz', **{f"{r['case']}__{r['arm']}": np.array(r['s_plus']) for r in R['arms']})
    write_json(a.out/'traces'/'case_info.json', R['case_info'])
    (a.out/'README.md').write_text(f"""# Third pass (offline, CPU, single-point replays)

Spec: RGBDiekf_F1_Temporal_Reference_Depth_Surface_Reacquisition_Debug_Codex.md. Inputs: first-pass traced standard-F1 runs under `{a.forensics_root}`,
second pass `{a.second_pass_root}` (hashes in traces/input_hashes.json). Command:

```bash
CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/rigeuler3dgs/bin/python tools/f1_temporal_reference_diagnostics.py --forensics-root "{a.forensics_root}" \\
    --second-pass-root "{a.second_pass_root}" --out "{a.out}"{' --case '+a.case if a.case else ''}
```
No production file, config or default was modified; nothing here reaches inference. GT columns are labelled GT_ORACLE_ONLY.
""")
    write_result(a.out, R, a.forensics_root, a.second_pass_root)
    print(json.dumps({'seconds': time.perf_counter()-t0, 'cases': cases,
                      'arms': [(r['case'], r['arm'], round(r['abs_dx_m']*1e3, 3), round(r['abs_dv_m_s'], 4)) for r in R['arms'] if r['case'] in ('storage_8444', 'usb_67827')],
                      'moving': {c: R['case_info'].get(c) for c in ('usb_87290', 'usb_85696')}}, default=str)[:4000])


if __name__ == '__main__':
    main()
