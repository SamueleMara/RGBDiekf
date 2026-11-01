#!/usr/bin/env python3
"""F1 static drift, SECOND PASS: prove the initiating cause (RGBDiekf_Inner_F1_Second_Pass_Causal_Debug_Codex.md).

INVESTIGATION ONLY. Reads the first-pass traced standard-F1 runs (forensics_F1_inner_drift_20261030/<scene>/traced, bit-identical
to the baselines), their raw input frames and the saved whole-cloud arrays; every experiment is a SINGLE-POINT / SINGLE-FRAME
replay with the independent float64 oracle (tools/root_cause_oracle.py) or a read-only call of the production visibility
(pixel ownership from the saved cloud of one frame). No production module is modified, no scene is re-filtered.
Ground truth (labelled part cloud + GT joint, rendered depth as a visibility oracle) only AUDITS, labelled as such.

    python tools/f1_second_pass.py --forensics-root FOR --out FOR/second_pass_20261030 [--only scene:id,scene:id]

Sections (spec): 3 frame-0 vs frame-1 residuals at the fixed x_ref (+ CF0), 4 first-update decomposition (every inner j,
eigen / normal-tangent split, 4b equality), 5 support / objective bookkeeping, 6 arms A-H, 7 1-D landscapes, 8 last usable
camera of moving points, 9 provenance. Thresholds below only CLASSIFY (named where used).
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
from tools import root_cause_oracle as O                                                     # noqa: E402
from tools.root_cause_trace import Run, jsonable, write_json, frozen_codes                  # noqa: E402
from tools.f1_diagnostic_units import compare_state6, compare_iterates      # noqa: E402

CASES = {   # (scene, point id): frames to analyse (first accepted false update + spec-named frames); role
    ('storage_45135', 8444): ([1], 'static'), ('storage_45135', 49199): ([1], 'static'), ('storage_45135', 262894): ([1], 'static'),
    ('storage_45135', 67801): ([2], 'static'), ('storage_45135', 102354): ([1, 3], 'static'),
    ('storage_45135', 8416): ([1], 'knife_edge_moving'), ('storage_45135', 8417): ([1], 'knife_edge_moving'), ('storage_45135', 8418): ([1], 'knife_edge_moving'),
    ('storage_45135', 60019): ([1], 'control'), ('storage_45135', 100564): ([1], 'control'),
    ('USB_100109', 67827): ([1], 'static'), ('USB_100109', 30502): ([3], 'static'), ('USB_100109', 51975): ([2], 'static'),
    ('USB_100109', 51974): ([1, 2], 'control'), ('USB_100109', 30503): ([1, 3], 'control'),
    ('blade_103706', 10251): ([2], 'static'), ('blade_103706', 14104): ([1], 'static'),
}
MOVING = {'USB_100109': [87290, 85696, 87588]}   # 87588: best available control (no USB moving point stays tracked)
LANDSCAPE = [('storage_45135', 8444, 1), ('USB_100109', 67827, 1), ('storage_45135', 8416, 1)]
EPS_PX = (1e-12, -1e-12, 1e-10, -1e-10, 1e-8, -1e-8)
FRAMES_H = range(0, 6)
NAMES = O.NAMES


# ------------------------------------------------------------------------------------------------------------ utilities
def fl(x):
    return [float(v) for v in np.ravel(x)]


class ShiftCam:
    """A camera whose projection is shifted by (du, dv) pixels (arm F only: rounding / boundary perturbation)."""
    def __init__(self, cam, du, dv):
        self.__dict__.update(cam.__dict__); self._cam, self._du, self._dv = cam, du, dv

    def project(self, x):
        u, v, Z, Ju, Jz = self._cam.project(x)
        return u+self._du, v+self._dv, Z, Ju, Jz

    def ray_world(self, u, v): return self._cam.ray_world(u, v)

    def centre(self): return self._cam.centre()


class Scene:
    def __init__(self, root, name):
        self.name = name
        self.run = Run(Path(root)/name/'traced', 'forward')
        self.fdir = Path(root)/name/'forensics'
        self.cams = self.run.cams(); self._img = {}; self._tr = {}
        self.policy, self.noise = self.run.policy, self.run.noise
        self.x0 = self.run.can['canonical_xyz']; self.src = self.run.can['original_source_ids']
        self.P0 = self.run.can['canonical_source_covariance']
        self.order = self.run.fman['processing_order']; self.t = np.asarray(self.run.fman['timestamps_seconds'], float)
        self.sel = {int(r['point_id']): r for r in csv.DictReader(open(self.fdir/'selected_point_ids.csv'))}
        self._own0 = None

    def images(self, k):
        if k not in self._img:
            if len(self._img) > 8: self._img.pop(next(iter(self._img)))
            self._img[k] = self.run.images(k)
        return self._img[k]

    def trace(self, k):
        if k not in self._tr: self._tr[k] = self.run.trace(k)
        return self._tr[k]

    def frame0_ownership(self, pids):
        """Production pixel ownership (projective_visibility.visibility, read-only) of the CANONICAL cloud at frame 0 with
        the frame-0 prior position covariance P0: what a frame-0 update of the non-source cameras would have used."""
        import torch
        from estimator.projective_camera import CameraRig
        from estimator.projective_visibility import visibility, position_sigma_z2
        if self._own0 is None:
            rig = CameraRig(self.run.data.cameras, 'cpu', torch.float64)
            x = torch.as_tensor(self.x0, dtype=torch.float64); P = torch.as_tensor(self.P0, dtype=torch.float64)
            proj = rig.project(x, jacobian=False)
            H, W = self.cams[0].H, self.cams[0].W
            reason, front = visibility(proj, position_sigma_z2(rig.R, P), H, W, rig.near, self.policy['visibility_n_sigma'])
            self._own0 = reason.numpy()
        return {p: self._own0[:, p] for p in pids}

    def mean0(self, pid):
        return np.r_[self.x0[pid], 0, 0, 0]

    def cov0(self, pid):
        tr = self.trace(self.order[1]); return np.array(tr['prediction']['points'][str(pid)]['P_previous'])


REASON_KEYS = ('behind', 'outside_footprint', 'cell_outside', 'mask_excluded', 'invalid_depth', 'discontinuity', 'frozen_self_occluded', 'frozen_duplicate',
               'recheck_self_occluded', 'front', 'chi2_inconsistent')


def all_reasons(img, cam, x, a, noise, policy, Sigma, dx, frozen_code=O.VALID, recheck_self=False):
    """Every rejection condition of one camera row evaluated SEPARATELY (no priority collapse), plus the raw sample."""
    out = _all_reasons(img, cam, x, a, noise, policy, Sigma, dx, frozen_code, recheck_self)
    out['reasons_true'] = [k for k in REASON_KEYS if out.get(k)] + ([] if 'r' in out or any(out.get(k) for k in REASON_KEYS) else ['no_sample'])
    out['valid_production_rules'] = bool('r' in out and not any(out.get(k) for k in REASON_KEYS))
    return out


def _all_reasons(img, cam, x, a, noise, policy, Sigma, dx, frozen_code, recheck_self):
    u, v, Z, Ju, Jz = cam.project(x)
    out = {'u': u, 'v': v, 'Z': Z, 'behind': bool(not math.isfinite(Z) or Z <= cam.near)}
    gv = O.geometric_visibility(cam, x)
    out.update(outside_footprint=gv == O.OUTSIDE, frozen_self_occluded=frozen_code == O.SELF_OCC, frozen_duplicate=frozen_code == O.DUP,
               recheck_self_occluded=bool(recheck_self))
    if out['behind'] or not (math.isfinite(u) and math.isfinite(v)):
        out.update(cell_outside=True); return out
    u0f, v0f = math.floor(u), math.floor(v)
    out['cell_outside'] = not (u0f >= 0 and v0f >= 0 and u0f+1 <= cam.W-1 and v0f+1 <= cam.H-1)
    if out['cell_outside']: return out
    nb = [(v0f, u0f), (v0f, u0f+1), (v0f+1, u0f), (v0f+1, u0f+1)]
    D = np.array([img.depth[y, xx] for y, xx in nb]); mk = np.array([img.mask[y, xx] for y, xx in nb]); rv = np.array([img.raw_valid[y, xx] for y, xx in nb])
    out.update(cell=[u0f, v0f], du=u-u0f, dv=v-v0f, depth4=fl(D), rgb4=[fl(img.rgb[y, xx]) for y, xx in nb], mask4=[bool(b) for b in mk], raw_valid4=[bool(b) for b in rv],
               mask_excluded=not mk.all(), invalid_depth=not (rv.all() and np.isfinite(D).all()))
    zbar = D.mean(); f = min(cam.fx, cam.fy)
    allow = policy['same_surface_n_sigma']*math.sqrt(2.)*math.sqrt(noise.sigma2(zbar))+zbar*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2.)/f
    out.update(spread=float(D.max()-D.min()), allowance=float(allow), discontinuity=bool(D.max()-D.min() > allow))
    if out['mask_excluded'] or out['invalid_depth']: return out
    o = O.row(img, cam, x, a, noise, {**policy, 'visibility_n_sigma': policy['visibility_n_sigma']}, Sigma, dx)
    if 'r' not in o:
        o2 = O.row(img, cam, x, a, noise, {**policy, 'same_surface_n_sigma': 1e12}, Sigma, dx)       # audit sample beyond the spread rule
        if 'r' in o2: o = o2; out['audit_only_sample_beyond_spread_rule'] = True
        else: return out
    n = policy['visibility_n_sigma']; S = o['S']
    out.update(r=fl(o['r']), R=fl(o['var']), H=o['H'].tolist(), z=fl(o['r']/np.sqrt(o['var'])), S_DD=float(S[3, 3]), nis=float(o['nis']),
               nu_D=float(o['nu'][3]), front=bool(o['nu'][3] > n*math.sqrt(max(S[3, 3], 0))),
               chi2_inconsistent=bool(o['nis'] > O.row_gate_threshold(n)), depth_metric=o['sample']['depth_metric'], depth_inverse=o['sample']['depth_inverse'],
               rgb_sample=fl(o['sample']['rgb']), grad_rgb=o['sample']['grad_rgb'].tolist(), grad_depth=fl(o['sample']['grad_depth']))
    return out


def problem(sc, k, pid, imgs=None, P=None, s=None, frozen=None, recheck=True, policy=None, cams=None, **kw):
    """Single-point replay of frame k: production prior / frozen assignment / recheck flags from the trace unless overridden."""
    tr = sc.trace(k); sp = str(pid)
    pre, it0, fin = tr['prediction']['points'][sp], tr['iterations'][0]['points'][sp], tr['final'][sp]
    rc = None
    if recheck and sc.policy.get('final_visibility', 'frozen') == 'recheck_self_occlusion':
        rc = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']]
    return O.Problem(cams or sc.cams, imgs or sc.images(k), sc.run.can['canonical_rgb'][pid], np.array(pre['s_minus']) if s is None else s,
                     np.array(pre['P_minus']) if P is None else P, frozen_codes(it0['cameras']) if frozen is None else frozen, sc.noise,
                     policy or sc.policy, sc.run.fman['max_iekf_iters'], sc.run.fman['step_tolerance'], rc, **kw)


def run_prob(prob, sc):
    return prob.run(sc.policy.get('update_acceptance', 'final_iterate'))


ACC = {0: 'ANCHOR_OR_NOT_PROCESSED', 1: 'ACCEPTED_FINAL_ITERATE', 2: 'ACCEPTED_EARLIER_CONSISTENT_ITERATE', 3: 'REJECTED_INCONSISTENT', 4: 'NO_VALID_ROWS'}


def summarize_result(prob, res):
    d = res['s_plus']-prob.s_minus
    acc_rows = prob.used(res['rows_accepted']) if res['observed'] else []
    return {'observed': bool(res['observed']), 'acceptance': ACC[res['acceptance_code']], 'dx_m': fl(d[:3]), 'dv_m_s': fl(d[3:]), 'abs_dx_m': float(np.linalg.norm(d[:3])),
            'abs_dv_m_s': float(np.linalg.norm(d[3:])), 'iterations': len(res['history']), 'converged': bool(res['converged']),
            'rows_accepted': [o['camera'] for o in acc_rows], 'cells_accepted': {o['camera']: list(o['sample']['pixel00']) for o in acc_rows},
            'J_accepted_own_rows': float(prob.prior_cost(res['s_plus'])+prob.info(res['rows_accepted'])[2]) if res['observed'] else None}


# --------------------------------------------------------------------------------------------- section 3: x_ref residuals
def xref_residuals(sc, pid, own0, frames=FRAMES_H):
    """Per camera and frame t, the raw row at the FIXED x_ref (no update), all reasons, with frame-0 ownership at t=0 and the
    production frozen assignment at t>=1. Delta r(t) = r(t) - r(0) (source colour cancels)."""
    x = sc.x0[pid]; a = sc.run.can['canonical_rgb'][pid]; srcc = int(sc.src[pid, 1]); P0 = sc.cov0(pid)
    rows = []
    for t in frames:
        imgs = sc.images(t)
        if t == 0: frz = [int(c) if int(c) in (O.SELF_OCC, O.DUP) else O.VALID for c in own0]; own_note = 'frame-0 canonical-cloud ownership (read-only production visibility, P0)'
        else:
            tr = sc.trace(t); frz = frozen_codes(tr['iterations'][0]['points'][str(pid)]['cameras']); own_note = 'production frozen assignment of frame t (at s-, not x_ref)'
        Sig = P0[:3, :3] if t == 0 else np.array(sc.trace(t)['prediction']['points'][str(pid)]['P_minus'])[:3, :3]
        for c, cam in enumerate(sc.cams):
            r_ = all_reasons(imgs[c], cam, x, a, sc.noise, sc.policy, Sig, np.zeros(3), frz[c])
            r_.update(scene=sc.name, point_id=pid, frame=t, camera=c, is_source_camera=c == srcc, ownership_basis=own_note,
                      owner_code=NAMES[frz[c]])
            rows.append(r_)
    by = {(r['frame'], r['camera']): r for r in rows}
    for r in rows:
        b = by.get((0, r['camera']))
        if b is not None and 'r' in r and 'r' in b:
            r['delta_r_vs_t0'] = fl(np.array(r['r'])-np.array(b['r'])); r['delta_z_vs_t0'] = fl((np.array(r['r'])-np.array(b['r']))/np.sqrt(np.array(r['R'])))
            r['delta_rgb_pixels_vs_t0'] = float(np.abs(np.array(r['rgb4'])-np.array(b['rgb4'])).max()); r['delta_depth_pixels_vs_t0'] = float(np.abs(np.array(r['depth4'])-np.array(b['depth4'])).max())
    return rows


def ignored_frame0_rows(rows, pid):
    r0 = [r for r in rows if r['frame'] == 0 and not r['is_source_camera']]
    return [r['camera'] for r in r0 if r.get('valid_production_rules')]


# ---------------------------------------------------------------------------------------------------- CF0 / arm C
def cf0(sc, pid, k1=1):
    """Schedule A (production): source anchor at t0, non-source cameras first at t1. Schedule B (diagnostic): the same anchor
    PLUS the independent validated non-source frame-0 rows (source camera excluded: its depth/colour already define x_ref, P0
    and a), then predict and update with the frame-1 rows. Nonlinear IEKF (oracle, production acceptance) and the exactly
    linearized one-step Gaussian update, both schedules, same P0 / R / Q / images."""
    own0 = sc.frame0_ownership([pid])[pid]
    s0, P0 = sc.mean0(pid), sc.cov0(pid)
    srcc = int(sc.src[pid, 1]); C = len(sc.cams)
    frz0 = [int(c) if int(c) in (O.SELF_OCC, O.DUP) else O.VALID for c in own0]
    tr1 = sc.trace(k1); pre1 = tr1['prediction']; dt = pre1['dt_s']; qc = np.array(pre1['q_c'])
    out = {}
    # schedule B, frame 0
    pb0 = O.Problem(sc.cams, sc.images(0), sc.run.can['canonical_rgb'][pid], s0, P0, frz0, sc.noise, sc.policy, sc.run.fman['max_iekf_iters'],
                    sc.run.fman['step_tolerance'], None, cam_mask=[c != srcc for c in range(C)])
    rb0 = run_prob(pb0, sc)
    out['B_frame0'] = summarize_result(pb0, rb0)
    sB, PB = rb0['s_plus'], rb0['P_plus']
    for name, (s_, P_) in (('A', (s0, P0)), ('B', (sB, PB))):
        sm, Pm, _, _ = O.predict(s_, P_, dt, qc)
        p1 = problem(sc, k1, pid, s=sm, P=Pm)
        r1 = run_prob(p1, sc)
        out[f'{name}_frame1'] = summarize_result(p1, r1)
        out[f'{name}_frame1']['x_plus'] = fl(r1['s_plus'][:3]); out[f'{name}_frame1']['v_plus'] = fl(r1['s_plus'][3:])
        out[f'{name}_frame1']['x_plus_minus_x_ref_m'] = fl(r1['s_plus'][:3]-sc.x0[pid])
        # exact linearized one-step Gaussian update at the predicted state (rows evaluated there)
        rows = p1.used(p1.rows(sm[:3]))
        if rows:
            H = np.zeros((4*len(rows), 6)); Rd = np.zeros(4*len(rows)); y = np.zeros(4*len(rows))
            for i, o in enumerate(rows): H[4*i:4*i+4, :3] = o['H']; Rd[4*i:4*i+4] = o['var']; y[4*i:4*i+4] = o['r']
            K = Pm@H.T@np.linalg.inv(H@Pm@H.T+np.diag(Rd)); d = K@y
            out[f'{name}_frame1']['linearized_one_step'] = {'rows': [o['camera'] for o in rows], 'dx_m': fl(d[:3]), 'dv_m_s': fl(d[3:]),
                                                           'abs_dx_m': float(np.linalg.norm(d[:3])), 'abs_dv_m_s': float(np.linalg.norm(d[3:]))}
        out[f'{name}_P_minus_frame1'] = Pm
    out['frame0_rows_used_in_B'] = out['B_frame0']['rows_accepted']
    out['note'] = ('frame-1 frozen assignment and recheck flags are the production ones (computed with every OTHER point at its production prediction); '
                   'in a real t0-fused filter the other points would also move, so association could differ')
    return out


# ------------------------------------------------------------------------------------ section 4: first-update decomposition
def surface_normal(sc, pid):
    """Independent local normal: plane fit to the back-projected 3x3 source-camera frame-0 depth patch around the source pixel,
    only if single-surface (fit RMS <= 3 sigma_D(z) or float32 resolution); else 'undefined/multiple'."""
    f, c, pix = (int(v) for v in sc.src[pid]); cam = sc.cams[c]; img = sc.images(0)[c]
    u0, v0 = pix % cam.W, pix//cam.W
    P = []
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            D = img.depth[v0+dv, u0+du]
            if img.mask[v0+dv, u0+du] and img.raw_valid[v0+dv, u0+du] and np.isfinite(D) and D > 0: P.append(cam.centre()+D*cam.ray_world(u0+du, v0+dv))
    if len(P) < 9: return {'normal': None, 'status': 'undefined (masked / invalid pixels in the 3x3 patch)'}
    P = np.array(P); c0 = P.mean(0); U, S, Vt = np.linalg.svd(P-c0)
    n = Vt[-1]; rms = float(S[-1]/3.)
    z = float(np.linalg.norm(sc.x0[pid]-cam.centre()))
    tol = max(3*math.sqrt(sc.noise.sigma2(z)), 3*np.finfo(np.float32).eps*z*4)
    if n@(cam.centre()-c0) < 0: n = -n
    return {'normal': fl(n), 'fit_rms_m': rms, 'tolerance_m': tol, 'status': 'single-surface' if rms <= tol else 'undefined/multiple (crease or edge in the 3x3 patch)'}


def first_update(sc, pid, k):
    """Every inner iterate of production frame k from the trace (state, A, b, delta, rows), with per-camera/channel gradient
    g_r = H^T R^-1 r and Fisher H^T R^-1 H recomputed by the oracle at that iterate; eigen / normal-tangent decomposition of
    the accepted step; the exact amplifier check; 4b equality at the accepted linearization."""
    tr = sc.trace(k); sp = str(pid)
    pre, fin = tr['prediction']['points'][sp], tr['final'][sp]
    Pm = np.array(pre['P_minus']); sm = np.array(pre['s_minus']); sp_ = np.array(fin['s_plus'])
    prob = problem(sc, k, pid)
    its = []
    for j, it in enumerate(tr['iterations']):
        q = it['points'][sp]
        if not q['active']: continue
        s = np.array(q['state']); rows = prob.rows(s[:3])
        per = []
        for o in rows:
            if o.get('reason') != O.VALID and 'H' not in o:
                per.append({'camera': o['camera'], 'reason': NAMES[o['reason']]}); continue
            W = 1/o['var']; g = o['H'].T@(W*o['r']); Fm = o['H'].T@(W[:, None]*o['H'])
            per.append({'camera': o['camera'], 'reason': NAMES[o['reason']], 'cell': list(o['sample']['pixel00']), 'du_dv': [o['sample']['du'], o['sample']['dv']],
                        'depth4': fl(o['sample']['neighbour_depth']), 'r': fl(o['r']), 'z': fl(o['r']/np.sqrt(o['var'])), 'R': fl(o['var']), 'H_x': o['H'].tolist(),
                        'S_DD': float(o['S'][3, 3]), 'nis': o['nis'], 'g_rgb': fl(o['H'][:3].T@(W[:3]*o['r'][:3])), 'g_depth': fl(o['H'][3]*W[3]*o['r'][3]),
                        'fisher_rgb': (o['H'][:3].T@(W[:3, None]*o['H'][:3])).tolist(), 'fisher_depth': (np.outer(o['H'][3], o['H'][3])*W[3]).tolist(),
                        'chi2_rgb': float((W[:3]*o['r'][:3]**2).sum()), 'chi2_depth': float(W[3]*o['r'][3]**2)})
        used = [p for p in per if p['reason'] == 'VALID']
        its.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'iteration': j, 's_iter': q['state'], 's_minus': fl(sm), 'P_minus': Pm.tolist(), 'A': q['A'], 'b': q['b'],
                    'delta': q['delta'], 'step_delta_A_delta': q['step_delta_A_delta'], 'posterior_cost_own_rows': q['posterior_cost'], 'cost_threshold': q['cost_threshold'],
                    'nrows': q['nrows'], 'support_bitset': sum(1 << p['camera'] for p in used), 'cells': {p['camera']: p['cell'] for p in used},
                    'rgb_chi2_own_rows': sum(p['chi2_rgb'] for p in used), 'depth_chi2_own_rows': sum(p['chi2_depth'] for p in used),
                    'prior_cost': float((s-sm)@np.linalg.solve(Pm, s-sm)), 'per_camera': per})
    # accepted step: exact amplifier and decomposition
    dx, dv = sp_[:3]-sm[:3], sp_[3:]-sm[3:]
    gain = Pm[3:, :3]@np.linalg.inv(Pm[:3, :3])
    lin = np.array(fin['linearization_state'])
    rows_acc = prob.rows(lin[:3], final=True); used = prob.used(rows_acc)
    Fd = np.zeros((3, 3)); Fr = np.zeros((3, 3))
    for o in used:
        W = 1/o['var']; Fr += o['H'][:3].T@(W[:3, None]*o['H'][:3]); Fd += np.outer(o['H'][3], o['H'][3])*W[3]
    Eff = np.linalg.inv(Pm[:3, :3])+Fd+Fr                      # Schur complement of A on x (measurements on x only)
    w, V = np.linalg.eigh(Eff); wd = np.linalg.eigvalsh(Fd) if used else np.zeros(3)
    comp = V.T@dx
    nrm = surface_normal(sc, pid)
    proj = {}
    if nrm['normal'] is not None and nrm['status'] == 'single-surface':
        n = np.array(nrm['normal']); proj = {'dx_normal_m': float(dx@n), 'dx_tangent_m': float(np.linalg.norm(dx-(dx@n)*n))}
    # 4b: dx = sum_c Eff^-1 g_eff,c - residual, g_eff,c = H^T W (r + H dx) at the accepted linearization (exact identity)
    e = lin[:3]-sm[:3]; contrib = {}
    for o in used:
        W = 1/o['var']
        for ch, nm in ((slice(0, 3), 'RGB'), (slice(3, 4), 'D')):
            g = o['H'][ch].T@(W[ch]*(o['r'][ch]+o['H'][ch]@e))
            contrib[f"cam{o['camera']}_{nm}"] = np.linalg.solve(Eff, g)
    tot = sum(contrib.values()) if contrib else np.zeros(3)
    A6 = np.linalg.inv(Pm); A6[:3, :3] += Fd+Fr                # the 6D information at the accepted linearization
    b6 = -np.linalg.solve(Pm, lin-sm); b6[:3] += sum((o['H'].T@((1/o['var'])*o['r']) for o in used), np.zeros(3))
    resid = np.linalg.solve(A6, b6)[:3]                         # e = A^-1 [g_eff; 0] - delta  (position block of A^-1 = Eff^-1)
    u = dx/max(np.linalg.norm(dx), 1e-300)
    dec = {'equality_dx_m': fl(e), 'sum_contributions_m': fl(tot), 'iterate_residual_m': fl(resid), 'closure_max_abs_m': float(np.abs(tot-resid-e).max()),
           'contributions_along_accepted_direction_m': {k_: float(v_@u) for k_, v_ in contrib.items()},
           'contributions_m': {k_: fl(v_) for k_, v_ in contrib.items()}}
    # gradient at x_ref with frame-0 vs frame-1 images (same prior covariance): is the frame-1 'evidence' new?
    gx = {}
    for t in (0, k):
        p_ = problem(sc, k, pid, imgs=sc.images(t), s=np.r_[sc.x0[pid], 0, 0, 0], frozen=None)
        rr = p_.used(p_.rows(sc.x0[pid]))
        gx[t] = {o['camera']: {'g_rgb': fl(o['H'][:3].T@((1/o['var'][:3])*o['r'][:3])), 'g_depth': fl(o['H'][3]*o['r'][3]/o['var'][3])} for o in rr}
    out = {'scene': sc.name, 'point_id': pid, 'frame': k, 'acceptance': fin['acceptance'], 'converged': fin['converged'], 'iterations': len(its),
           'dx_m': fl(dx), 'dv_m_s': fl(dv), 'gain_P_vx_Pxx_inv': gain.tolist(), 'amplifier_check_max_abs': float(np.abs(gain@dx-dv).max()),
           'effective_information_eigenvalues': fl(w), 'effective_information_condition': float(w[-1]/max(w[0], 1e-300)),
           'depth_fisher_eigenvalues': fl(wd), 'rgb_fisher_eigenvalues': fl(np.linalg.eigvalsh(Fr)),
           'dx_along_eigenvectors_m': fl(comp), 'eigenvectors': V.tolist(), 'surface_normal': nrm, **proj, 'equality_4b': dec,
           'gradient_at_x_ref': {str(t): v_ for t, v_ in gx.items()}, 'rows_accepted': [o['camera'] for o in used],
           'P_minus': Pm.tolist(), 'prior_sigma_along_dx_m': float(np.linalg.norm(dx)/math.sqrt(max(dx@np.linalg.solve(Pm[:3, :3], dx), 1e-300)))}
    return out, its


# -------------------------------------------------------------------------------------- section 5: support / objectives
def support_objectives(sc, pid, k, its):
    tr = sc.trace(k); sp = str(pid); fin = tr['final'][sp]
    prob = problem(sc, k, pid)
    sm = prob.s_minus
    S_prior = [p['camera'] for p in its[0]['per_camera'] if p['reason'] == 'VALID']
    lin = np.array(fin['linearization_state'])
    S_final = [o['camera'] for o in prob.used(prob.rows(lin[:3], final=True))]
    def common_cost(s):
        if not S_prior: return None, []
        c = prob.cost_on(s, S_prior)
        return (prob.prior_cost(s)+c['rgb']+c['depth'] if not c['lost_rows'] else None), c
    rows = []
    Jp, cp = common_cost(sm)
    prev = None
    for it in its:
        s = np.array(it['s_iter'])
        Jc, cc = common_cost(s)
        cur = {c: tuple(v) for c, v in it['cells'].items()}
        sup = sorted(cur)
        ra = all_reasons_map(sc, prob, s, k, pid)
        rows.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'iteration': it['iteration'], 'support': ';'.join(map(str, sup)),
                     'support_bitset': it['support_bitset'], 'cells': json.dumps({str(c): v for c, v in cur.items()}),
                     'dropped_vs_prior': ';'.join(map(str, sorted(set(S_prior)-set(sup)))), 'added_vs_prior': ';'.join(map(str, sorted(set(sup)-set(S_prior)))),
                     'cells_changed_vs_previous': ';'.join(str(c) for c in sup if prev and c in prev and prev[c] != cur[c]),
                     'reasons_all_cameras': json.dumps(ra), 'J_own_rows': it['posterior_cost_own_rows'], 'dof_own': 4*it['nrows'], 'threshold_own': it['cost_threshold'],
                     'J_common_prior_rows': Jc, 'common_rows_not_evaluable': json.dumps(cc.get('lost_rows', [])) if cc else '',
                     'rgb_chi2_common': cc['rgb'] if cc else None, 'depth_chi2_common': cc['depth'] if cc else None, 'prior_cost': it['prior_cost'],
                     'rgb_chi2_own': it['rgb_chi2_own_rows'], 'depth_chi2_own': it['depth_chi2_own_rows']})
        prev = cur
    Jf, cf = common_cost(lin)
    verdict = []
    cells_changed = any(r['cells_changed_vs_previous'] for r in rows[1:])
    if set(S_prior)-set(S_final) and (Jf is None or Jp is None or Jf >= Jp): verdict.append('row-drop advantage')
    if Jf is not None and Jp is not None and Jf < Jp:
        verdict.append('pixel-cell change advantage' if cells_changed else 'model preference on unchanged rows')
    if Jf is not None and Jp is not None and Jf >= Jp and not (set(S_prior)-set(S_final)): verdict.append('no decrease on the original rows (accepted by chi2 adequacy only)')
    if not fin['converged']: verdict.append('nonconverged iteration artifact (accepted at the iteration cap / fallback)')
    summ = {'scene': sc.name, 'point_id': pid, 'frame': k, 'S_prior': S_prior, 'S_final': S_final, 'J_common_prior_state': Jp, 'J_common_accepted_state': Jf,
            'common_rows_lost_at_accepted': cf.get('lost_rows', []) if cf else None, 'J_own_accepted': fin['posterior_cost'], 'threshold_own_accepted': fin['cost_threshold'],
            'acceptance': fin['acceptance'], 'converged': fin['converged'], 'cells_changed_during_iterations': cells_changed, 'verdict': verdict or ['decrease on unchanged rows']}
    return rows, summ


def all_reasons_map(sc, prob, s, k, pid):
    out = {}
    for c, cam in enumerate(sc.cams):
        r_ = all_reasons(prob.imgs[c], cam, s[:3], prob.a, sc.noise, sc.policy, prob.P_minus[:3, :3], s[:3]-prob.s_minus[:3], prob.frozen[c])
        out[c] = r_.get('reasons_true', ['no_sample'] if 'r' not in r_ else []) or ['VALID']
    return out


# ------------------------------------------------------------------------------------------------- section 6: arms
class FrozenProblem(O.Problem):
    """Arm B: the camera-row membership of the original prediction is FROZEN; every trial samples the actual nonlinear row of
    those cameras (no gate); a frozen row whose sample is invalid at the trial is SIGNALLED (not silently dropped)."""
    def __init__(self, *a, frozen_rows=(), **kw):
        super().__init__(*a, **kw); self.frozen_rows = list(frozen_rows); self.invalid_events = []

    def rows(self, x, final=False):
        out = []; dx = x-self.s_minus[:3]
        pol = {**self.policy, 'row_innovation_gate': 'depth_front_only', 'visibility_n_sigma': float('inf')}
        for c, cam in enumerate(self.cams):
            if c not in self.frozen_rows: out.append({'camera': c, 'reason': O.NOT_PROC, 'visibility': O.NOT_PROC}); continue
            o = O.row(self.imgs[c], cam, x, self.a, self.noise, pol, self.P_minus[:3, :3], dx); o.update(camera=c, visibility=O.VALID)
            if o['reason'] != O.VALID: self.invalid_events.append((c, NAMES[o['reason']], fl(x)))
            out.append(o)
        return out


def arms(sc, pid, k, s_prior_rows, knife=False):
    rule = sc.policy.get('update_acceptance', 'final_iterate')
    res = {}
    pa = problem(sc, k, pid); ra = run_prob(pa, sc); res['A_baseline_replay'] = summarize_result(pa, ra)
    fin = sc.trace(k)['final'][str(pid)]
    res['A_baseline_replay'].update({f'production_parity_{kk}': vv for kk, vv in compare_state6(ra['s_plus'], fin['s_plus']).items()})   # units bug fixed
    if s_prior_rows:
        tr = sc.trace(k); pre = tr['prediction']['points'][str(pid)]
        pb = FrozenProblem(sc.cams, sc.images(k), sc.run.can['canonical_rgb'][pid], np.array(pre['s_minus']), np.array(pre['P_minus']),
                           [O.VALID]*len(sc.cams), sc.noise, sc.policy, sc.run.fman['max_iekf_iters'], sc.run.fman['step_tolerance'], None, frozen_rows=s_prior_rows)
        rb = pb.run(rule); res['B_frozen_original_rows'] = summarize_result(pb, rb)
        res['B_frozen_original_rows']['invalid_support_events'] = pb.invalid_events[:20]
    for c in sorted(set(s_prior_rows) | set(res['A_baseline_replay']['rows_accepted'])):
        p = problem(sc, k, pid, cam_mask=[cc != c for cc in range(len(sc.cams))]); res[f'D_drop_cam{c}'] = summarize_result(p, run_prob(p, sc))
    if s_prior_rows:
        for nm, ch in (('E_rgb_only_original_support', (1, 1, 1, 0)), ('E_depth_only_original_support', (0, 0, 0, 1))):
            p = problem(sc, k, pid, channels=ch, fixed_rows=set(s_prior_rows)); res[nm] = summarize_result(p, run_prob(p, sc))
    if knife:
        base = ra
        for e in EPS_PX:
            cams = [ShiftCam(c, e, e) for c in sc.cams]
            p = problem(sc, k, pid, cams=cams); r = run_prob(p, sc)
            it = compare_iterates([h['s'] for h in r['history']], [h['s'] for h in base['history']])
            d = it['position_linf_m_per_iterate']
            amp = [d[i+1]/d[i] for i in range(len(d)-1) if d[i] > 0]          # POSITION divergence amplification per iteration
            res[f'F_shift_{e:+.0e}px'] = {**summarize_result(p, r), **{f'accepted_vs_A_{kk}': vv for kk, vv in compare_state6(r['s_plus'], base['s_plus']).items()},
                                          'iterate_position_linf_m_vs_A': d, 'iterate_velocity_linf_m_s_vs_A': it['velocity_linf_m_s_per_iterate'],
                                          'first_iteration_position_diverging_over_1e-9_m': it['first_iteration_position_diverging_over_1e-9_m'],
                                          'first_iteration_velocity_diverging_over_1e-9_m_s': it['first_iteration_velocity_diverging_over_1e-9_m_s'],
                                          'per_iteration_amplification': amp,
                                          'median_amplification': float(np.median(amp)) if amp else None}
    tr = sc.trace(k); Pm = np.array(tr['prediction']['points'][str(pid)]['P_minus']); Pg = Pm.copy(); Pg[:3, 3:] = 0; Pg[3:, :3] = 0
    pg = problem(sc, k, pid, P=Pg); res['G_zero_Pxv_one_step'] = summarize_result(pg, run_prob(pg, sc))
    return res


# -------------------------------------------------------------------------------------- section 7: 1-D landscape + plots
def landscape(sc, pid, k, out_png, its):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tr = sc.trace(k); fin = tr['final'][str(pid)]
    prob = problem(sc, k, pid); sm = prob.s_minus; sa = np.array(fin['s_plus'])
    S_prior = [p['camera'] for p in its[0]['per_camera'] if p['reason'] == 'VALID']
    rows0 = prob.used(prob.rows(sm[:3]))
    al = np.r_[np.linspace(-.25, 1.25, 121)]
    rec = []
    for a in al:
        s = sm+a*(sa-sm)
        rows = prob.rows(s[:3]); used = prob.used(rows)
        _, _, q = prob.info(rows); rgb, dep = prob.data_cost(rows)
        cc = prob.cost_on(s, S_prior) if S_prior else {'rgb': 0, 'depth': 0, 'lost_rows': []}
        lin = sum(float(((o['r']+a*(o['H']@(sa[:3]-sm[:3])))**2/o['var']).sum()) for o in rows0)
        rec.append({'alpha': float(a), 'prior': prob.prior_cost(s), 'rgb_own': rgb, 'depth_own': dep, 'support': [o['camera'] for o in used],
                    'cells': {o['camera']: list(o['sample']['pixel00']) for o in used}, 'rgb_common': None if cc['lost_rows'] else cc['rgb'],
                    'depth_common': None if cc['lost_rows'] else cc['depth'], 'linear_model_data': lin,
                    'reasons': {o['camera']: NAMES[o['reason']] for o in rows}})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    A = np.array([r['alpha'] for r in rec]); g = lambda k_: np.array([np.nan if r[k_] is None else r[k_] for r in rec], float)
    pr = g('prior'); tot_own = pr+g('rgb_own')+g('depth_own'); tot_c = pr+g('rgb_common')+g('depth_common')
    ax[0, 0].plot(A, pr, label='prior'); ax[0, 0].plot(A, g('rgb_own'), label='RGB (own rows)'); ax[0, 0].plot(A, g('depth_own'), label='depth (own rows)')
    ax[0, 0].plot(A, tot_own, 'k', label='total (own rows)'); ax[0, 0].set_title(f'{sc.name} {pid} frame {k}: costs on each candidate\'s OWN rows'); ax[0, 0].set_yscale('symlog')
    ax[0, 1].plot(A, g('rgb_common'), label='RGB'); ax[0, 1].plot(A, g('depth_common'), label='depth'); ax[0, 1].plot(A, tot_c, 'k', label='total incl. prior')
    ax[0, 1].set_title(f'common rows = prediction support {S_prior} (gaps: not evaluable)'); ax[0, 1].set_yscale('symlog')
    ax[1, 0].plot(A, pr+g('linear_model_data'), label='Gauss-Newton model at s-'); ax[1, 0].plot(A, tot_own, 'k', label='actual (own rows)')
    ax[1, 0].set_title('linearized prediction vs actual'); ax[1, 0].set_yscale('symlog')
    nrows = np.array([len(r['support']) for r in rec]); ax[1, 1].step(A, nrows, where='mid'); ax[1, 1].set_title('admitted rows along the path')
    ch = [rec[i]['alpha'] for i in range(1, len(rec)) if rec[i]['cells'] != rec[i-1]['cells'] or rec[i]['support'] != rec[i-1]['support']]
    for a_ in ax.ravel():
        for x_ in ch: a_.axvline(x_, color='0.85', lw=.6, zorder=0)
        a_.axvline(0, color='g', lw=1); a_.axvline(1, color='r', lw=1); a_.legend(fontsize=7)
    for a_ in ax[1]: a_.set_xlabel('alpha (0 = prediction s-, 1 = accepted state); grey: cell / support change')
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return rec, ch


def patch_png(sc, pid, k, out_png):
    """15x15 RGB / depth crops around pi_c(x_ref) at t0 and t1 for every camera with a sample, marked subpixel location and 2x2."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = sc.x0[pid]; cams = []
    for c, cam in enumerate(sc.cams):
        u, v, Z, _, _ = cam.project(x)
        if Z > cam.near and 7 <= u < cam.W-8 and 7 <= v < cam.H-8: cams.append((c, u, v))
    if not cams: return
    fig, ax = plt.subplots(len(cams), 4, figsize=(12, 3*len(cams)), squeeze=False)
    for i, (c, u, v) in enumerate(cams):
        ui, vi = int(round(u)), int(round(v))
        for j, t in enumerate((0, k)):
            img = sc.images(t)[c]
            rgb = img.rgb[vi-7:vi+8, ui-7:ui+8]; dep = img.depth[vi-7:vi+8, ui-7:ui+8]
            ax[i, 2*j].imshow(np.clip(rgb, 0, 1), extent=(ui-7.5, ui+7.5, vi+7.5, vi-7.5), interpolation='nearest')
            d = np.where(img.mask[vi-7:vi+8, ui-7:ui+8], dep, np.nan); lo, hi = np.nanmin(d), np.nanmax(d)
            im = ax[i, 2*j+1].imshow(d, extent=(ui-7.5, ui+7.5, vi+7.5, vi-7.5), interpolation='nearest', vmin=lo, vmax=hi, cmap='viridis')
            plt.colorbar(im, ax=ax[i, 2*j+1], fraction=.046)
            for a_ in (ax[i, 2*j], ax[i, 2*j+1]):
                a_.plot(u, v, 'r+', ms=12); a_.add_patch(plt.Rectangle((math.floor(u), math.floor(v)), 1, 1, fill=False, ec='r', lw=1))
                a_.set_title(f'cam {c}{" (source)" if c == int(sc.src[pid, 1]) else ""} t={t} {"RGB" if a_ is ax[i, 2*j] else "depth m"}', fontsize=8)
    fig.suptitle(f'{sc.name} point {pid}: x_ref projected (red +), 2x2 support (red box), 15x15 crops, t=0 and t={k}')
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)


# ------------------------------------------------------------------------------------- section 8: last usable camera
def last_camera(sc, pid):
    """Moving point: the last successful frame and the first all-camera loss; every camera at both frames with EVERY gate
    reason, measured depth difference, S_DD / NIS, the pixel winner / runner-up (estimated and GT-audit positions),
    a read-only per-iteration-style ownership recomputed from the saved cloud, and the rendered-depth visibility oracle."""
    import torch
    from estimator.projective_camera import CameraRig
    from estimator.projective_visibility import visibility, position_sigma_z2
    from tools.forensic_f1_drift import GroundTruth
    gt = GroundTruth(sc.run, Path(sc.run.man['arguments']['scene']))
    tl = [r for r in csv.DictReader(open(sc.fdir/'f1_state_timeline.csv')) if int(r['point_id']) == pid]
    acc = [int(r['frame']) for r in tl if r['acceptance'].startswith('ACCEPTED')]
    loss = next((int(r['frame']) for r in tl if acc and int(r['frame']) > acc[0] and not r['acceptance'].startswith('ACCEPTED')), None)
    last_ok = max((f for f in acc if loss is None or f < loss), default=None)
    out = []
    X = sc.run.mm('posterior_xyz'); Xp = sc.run.mm('predicted_xyz')
    pop = np.load(sc.fdir/'population_events.npz'); mov = pop['gt_moving']
    rig = CameraRig(sc.run.data.cameras, 'cpu', torch.float64)
    for k in [f for f in (last_ok, loss) if f is not None]:
        tr = sc.trace(k); sp = str(pid); pre = tr['prediction']['points'][sp]; fin = tr['final'][sp]
        sm = np.array(pre['s_minus']); Pm = np.array(pre['P_minus']); xgt = gt.moved(sc.x0[pid], k)
        it0 = tr['iterations'][0]['points'][sp]['cameras']; frz = frozen_codes(it0)
        # per-iteration-style ownership (approximation): every OTHER point at its saved posterior of frame k, this point at s-
        xs = np.asarray(X[k], float).copy(); xs[pid] = sm[:3]
        proj = rig.project(torch.as_tensor(xs), jacobian=False)
        sz = position_sigma_z2(rig.R, torch.as_tensor(np.broadcast_to(Pm[:3, :3], (len(xs), 3, 3)).copy()))
        reason_alt, _ = visibility(proj, sz, sc.cams[0].H, sc.cams[0].W, rig.near, sc.policy['visibility_n_sigma'])
        for c, cam in enumerate(sc.cams):
            r_ = all_reasons(sc.images(k)[c], cam, sm[:3], sc.run.can['canonical_rgb'][pid], sc.noise, sc.policy, Pm[:3, :3], np.zeros(3), frz[c])
            ug, vg, Zg, _, _ = cam.project(xgt)
            img = sc.images(k)[c]; vis_oracle = 'cannot_establish (outside image)'
            if 0 <= round(ug) < cam.W and 0 <= round(vg) < cam.H and Zg > cam.near:
                Dm = img.depth[int(round(vg)), int(round(ug))]
                vis_oracle = ('visible (rendered depth = GT Z within 5 mm)' if img.mask[int(round(vg)), int(round(ug))] and abs(Dm-Zg) < 5e-3 else
                              f'not visible / other surface (rendered D - GT Z = {Dm-Zg:+.4f} m)')
            occ = it0[c]['pixel_occupants_front_to_back']
            win = [{'point': o['point'], 'z': o['z'], 'estimated_xyz': fl(np.asarray(Xp[k][o['point']], float)), 'gt_audit_xyz': fl(gt.moved(sc.x0[o['point']], k) if mov[o['point']] else sc.x0[o['point']]),
                    'is_moving_part': bool(mov[o['point']])} for o in occ[:2]]
            out.append({'scene': sc.name, 'point_id': pid, 'frame': k, 'role': 'last_successful' if k == last_ok else 'first_all_camera_loss', 'camera': c,
                        'acceptance': fin['acceptance'], 'u_pred': r_['u'], 'v_pred': r_['v'], 'Z_pred': r_['Z'], 'u_gt': ug, 'v_gt': vg, 'Z_gt': Zg,
                        'reasons_all': ';'.join(r_.get('reasons_true', [])) or ('VALID' if r_.get('valid_production_rules') else 'no_sample'),
                        'sample_exists': 'r' in r_, 'depth_diff_Z_minus_D_m': r_['r'][3] if 'r' in r_ else None, 'S_DD': r_.get('S_DD'), 'nis': r_.get('nis'),
                        'mask4': json.dumps(r_.get('mask4')), 'depth4': json.dumps(r_.get('depth4')), 'frozen_assignment': NAMES[frz[c]],
                        'per_iteration_style_ownership_approx': NAMES[int(reason_alt[c, pid])], 'winner_runner_up': json.dumps(win),
                        'gt_visibility_oracle_rendered_depth': vis_oracle,
                        'velocity_m_s': fl(sm[3:]), 'predicted_displacement_m': float(np.linalg.norm(sm[:3]-sc.x0[pid])),
                        'true_displacement_m': float(np.linalg.norm(xgt-sc.x0[pid])), 'prediction_to_gt_m': float(np.linalg.norm(sm[:3]-xgt))})
    return out, {'last_successful_frame': last_ok, 'first_all_camera_loss_frame': loss}


# ---------------------------------------------------------------------------------------------- section 9: provenance
def provenance_checks(sc, pids):
    out = {'scene': sc.name, 'camera_geometry_hash': sc.run.fman['camera_geometry_hash'], 'cameras_fixed_over_time': 'one CameraRig for all frames (run_free)'}
    res = []
    for p in pids:
        f, c, pix = (int(v) for v in sc.src[p]); cam = sc.cams[c]; u, v, Z, _, _ = cam.project(sc.x0[p])
        D = sc.images(0)[c].depth[pix//cam.W, pix % cam.W]
        res.append({'point_id': p, 'source_camera': c, 'source_pixel': [pix % cam.W, pix//cam.W], 'projection': [u, v], 'pixel_offset_px': [u-pix % cam.W, v-pix//cam.W],
                    'Z_minus_source_depth_m': float(Z-D), 'positive_Z': bool(Z > 0)})
    out['x_ref_source_projection'] = res
    stat = []
    for t in (1, 5):
        for c in range(len(sc.cams)):
            a, b = sc.images(0)[c], sc.images(t)[c]
            both = a.mask & b.mask & a.raw_valid & b.raw_valid
            same_d = both & (a.depth == b.depth)
            drgb = np.abs(a.rgb-b.rgb).max(-1)
            stat.append({'frame': t, 'camera': c, 'valid_both': int(both.sum()), 'depth_identical': int(same_d.sum()),
                         'depth_identical_rgb_identical': int((same_d & (drgb == 0)).sum()), 'depth_identical_rgb_max_abs_change': float(drgb[same_d].max()) if same_d.any() else None,
                         'depth_changed': int((both & (a.depth != b.depth)).sum())})
    out['static_pixel_constancy_vs_frame0'] = stat
    return out


# ------------------------------------------------------------------------------------------------------------------ main
def wcsv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(jsonable(v)) if isinstance(v, (list, dict, tuple, np.ndarray)) else v for k, v in r.items()})


# ----------------------------------------------------------------------------------------------------------- report
def _f(v, p=3, scale=1.):
    if v is None or (isinstance(v, float) and not math.isfinite(v)): return '-'
    return f'{v*scale:.{p}g}'


def write_report(out, root):
    """SECOND_PASS_ROOT_CAUSE.md from the files of this pass (numbers only; the closing statement of every point is filled
    from the measured quantities with the spec's template; 'already existed at frame 0' requires |Delta z| < 1e-6 on every
    contributing row at the fixed x_ref)."""
    S = json.loads((out/'summary.json').read_text())
    rd = lambda f: list(csv.DictReader(open(out/f))) if (out/f).exists() else []
    R3, AR = rd('frame0_frame1_raw_residuals.csv'), rd('local_counterfactuals.csv')
    LC = list(csv.DictReader(open(out/'last_camera_loss.csv'))) if (out/'last_camera_loss.csv').exists() else []
    J = lambda x: json.loads(x) if x not in (None, '') else None
    tls, sels = {}, {}
    def timeline(scene, pid):
        if scene not in tls:
            tls[scene] = list(csv.DictReader(open(Path(root)/scene/'forensics'/'f1_state_timeline.csv')))
            sels[scene] = {int(r['point_id']): r for r in csv.DictReader(open(Path(root)/scene/'forensics'/'selected_point_ids.csv'))}
        return [r for r in tls[scene] if int(r['point_id']) == pid], sels[scene].get(pid, {})
    sup = {(v['scene'], v['point_id'], v['frame']): v for v in S['support_verdicts']}
    cf0 = {(c['scene'], c['point_id']): c for c in S['cf0']}
    L = ['# F1 static drift, second pass: the initiating cause, point by point', '',
         'Spec: `RGBDiekf_Inner_F1_Second_Pass_Causal_Debug_Codex.md`. Investigation only: single-point / single-frame replays of the first-pass '
         'traced standard-F1 runs (bit-identical to the baselines) with the independent float64 oracle and read-only production visibility; '
         'no inference code or configuration was modified. GT (labelled part cloud, GT joint, rendered depth) only audits.', '',
         f"Inputs: {json.dumps(S['inputs'], default=str)[:600]}...", '',
         '## G0 parity (baseline micro-replay vs production)', '', '| scene | id | frame | accepted POSITION vs production (m, linf) | accepted VELOCITY vs production (m/s, linf) | acceptance |', '|---|---|---|---|---|---|']
    for a in AR:
        if a['arm'] == 'A_baseline_replay':
            L.append(f"| {a['scene']} | {a['point_id']} | {a['frame']} | {_f(float(a['production_parity_position_linf_m']), 2)} | {_f(float(a['production_parity_velocity_linf_m_s']), 2)} | {a['acceptance']} |")
    L += ['', '## Point by point', '']
    for fu in S['first_updates']:
        sc, pid, k = fu['scene'], fu['point_id'], fu['frame']
        tl, sel = timeline(sc, pid)
        sv = sup.get((sc, pid, k), {})
        rows_used = fu['rows_accepted']
        eq = fu['equality_4b']['contributions_along_accepted_direction_m']
        main_c = sorted(eq.items(), key=lambda kv: -abs(kv[1]))[:3]
        cams_main = sorted({int(n.split('_')[0][3:]) for n, v in main_c if abs(v) > .1*max(abs(x) for x in eq.values())}) if eq else []
        r3 = {(int(r['camera']), int(r['frame'])): r for r in R3 if r['scene'] == sc and int(r['point_id']) == pid}
        disc, existed = [], []
        for c in cams_main:
            r0, r1 = r3.get((c, 0)), r3.get((c, k))
            if r1 and r1.get('z'):
                dz = J(r1.get('delta_z_vs_t0')) or [float('nan')]*4
                disc.append(f"camera {c}: z = {[round(x, 2) for x in J(r1['z'])]} (RGB, D) at x_ref, r_D = {_f(J(r1['r'])[3], 3, 1e3)} mm, Delta z vs t0 = {[round(x, 3) for x in dz]}"
                            f"{', SOURCE camera' if r1['is_source_camera'] == 'True' else ''}; frame-0 status {r0.get('reasons_true') if r0 else '-'} / valid {r0.get('valid_production_rules') if r0 else '-'}")
                existed.append(all(abs(x) < 1e-6 for x in dz))
            else:
                disc.append(f"camera {c}: no valid sample at x_ref at frame {k} (row exists only at the moved iterate)"); existed.append(None)
        when = ('already existed at frame 0 (identical sample at the fixed x_ref)' if existed and all(e is True for e in existed) else
                'arose (at least partly) only at the analysed frame' if existed and any(e is False for e in existed) else 'not decidable at x_ref (no sample there)')
        dx = np.array(fu['dx_m']); dv = np.array(fu['dv_m_s'])
        later = [(r['frame'], r['acceptance'][:10], r['cameras_used'] or '-') for r in tl if int(r['frame']) > k][:8]
        arms = {a['arm']: a for a in AR if a['scene'] == sc and int(a['point_id']) == pid and int(a['frame']) == k}
        armtxt = '; '.join(f"{n}: |dx| {_f(float(v['abs_dx_m']), 3, 1e3)} mm, |dv| {_f(float(v['abs_dv_m_s']), 3)} m/s ({v['acceptance'][:12]})"
                           for n, v in arms.items() if n != 'A_baseline_replay' and not n.startswith('F_'))
        ftxt = '; '.join(f"{n[2:]}: accepted position change {_f(float(v['accepted_vs_A_position_linf_m']), 2)} m, velocity change {_f(float(v['accepted_vs_A_velocity_linf_m_s']), 2)} m/s"
                         for n, v in arms.items() if n.startswith('F_'))
        removed = [n for n, v in arms.items() if n.startswith(('D_', 'E_', 'C_')) and float(v['abs_dx_m']) < .1*max(float(fu and np.linalg.norm(dx)), 1e-12)]
        L += [f"### {sc} {pid} (frame {k}, {fu['role']})", '',
              f"- production: {fu['acceptance']}, {fu['iterations']} inner iterations (converged {fu['converged']}); dx = {np.round(dx*1e3, 3).tolist()} mm, dv = {np.round(dv, 4).tolist()} m/s; "
              f"rows in the accepted solve {rows_used}; prior sigma along dx {_f(fu['prior_sigma_along_dx_m'], 3, 1e3)} mm",
              f"- effective position information (Schur complement, rows of the accepted linearization) eigenvalues {[_f(x, 3) for x in fu['effective_information_eigenvalues']]} 1/m^2 "
              f"(sigma {[_f(1/math.sqrt(max(x, 1e-300)), 3, 1e3) for x in fu['effective_information_eigenvalues']]} mm); dx along those eigenvectors {np.round(np.array(fu['dx_along_eigenvectors_m'])*1e3, 3).tolist()} mm; "
              f"surface normal: {fu['surface_normal']['status']}" + (f", dx normal {_f(fu.get('dx_normal_m'), 3, 1e3)} mm, tangent {_f(fu.get('dx_tangent_m'), 3, 1e3)} mm" if fu.get('dx_normal_m') is not None else ''),
              f"- 4b equality: dx = sum_c Eff^-1 g_eff,c - delta (closure {_f(fu['equality_4b']['closure_max_abs_m'], 2)} m); contributions along dx: "
              + ', '.join(f"{n} {_f(v, 4, 1e3)} mm" for n, v in sorted(eq.items(), key=lambda kv: -abs(kv[1]))),
              f"- amplifier: dv = P_vx P_xx^-1 dx holds to {_f(fu['amplifier_check_max_abs'], 2)} m/s (gain diag {[round(fu['gain_P_vx_Pxx_inv'][i][i], 2) for i in range(3)]} 1/s)",
              '- discrepancy at the fixed x_ref: ' + ' | '.join(disc),
              f"- support / objective: S_prior {sv.get('S_prior')} -> S_final {sv.get('S_final')}; J on the original rows {_f(sv.get('J_common_prior_state'), 4)} -> {_f(sv.get('J_common_accepted_state'), 4)} "
              f"(own rows {_f(sv.get('J_own_accepted'), 4)} <= {_f(sv.get('threshold_own_accepted'), 4)}); verdict: {', '.join(sv.get('verdict', []))}",
              f"- local counterfactuals: {armtxt}"]
        if ftxt: L.append(f"- arm F (pixel shifts): {ftxt}")
        c0 = cf0.get((sc, pid))
        if c0 and k == 1:
            L.append(f"- CF0 frame-0 non-source rows ignored by F1: {c0['ignored_frame0_nonsource_valid_rows']}; schedule B frame-0 update |dx| {_f(c0['B_frame0']['abs_dx_m'], 3, 1e3)} mm "
                     f"(v unchanged: P_xv0 = 0); frame-1 |dx| A {_f(c0['A_frame1']['abs_dx_m'], 3, 1e3)} mm / B {_f(c0['B_frame1']['abs_dx_m'], 3, 1e3)} mm, v A {_f(c0['A_frame1']['abs_dv_m_s'], 3)} / B {_f(c0['B_frame1']['abs_dv_m_s'], 3)} m/s")
        L += [f"- later frames (frame, acceptance, cameras): {later}; final transport drift {_f(float(sel.get('transport_drift_final_m', 'nan')), 3)} m", '',
              f"> This point first moves at frame {k}, because observation(s) from camera(s) {cams_main} have a discrepancy at the fixed x_ref ({'; '.join(disc)}) that {when}. "
              f"The exact contributions {', '.join(f'{n} {_f(v, 3, 1e3)} mm' for n, v in main_c)} combined with an effective information whose weakest direction has sigma "
              f"{_f(1/math.sqrt(max(fu['effective_information_eigenvalues'][0], 1e-300)), 3, 1e3)} mm produce a position step of {_f(float(np.linalg.norm(dx)), 3, 1e3)} mm. "
              f"The inner loop {'retains' if set(sv.get('S_prior', [])) <= set(sv.get('S_final', [])) else 'loses'} rows ({sv.get('S_prior')} -> {sv.get('S_final')}) and "
              f"{'decreases' if (sv.get('J_common_accepted_state') is not None and sv.get('J_common_prior_state') is not None and sv['J_common_accepted_state'] < sv['J_common_prior_state']) else 'does not decrease (or cannot evaluate)'} "
              f"the objective on comparable rows. The Kalman gain converts this step to a velocity of {_f(float(np.linalg.norm(dv)), 3)} m/s; afterwards {later[:4]} ..., final drift "
              f"{_f(float(sel.get('transport_drift_final_m', 'nan')), 3)} m. The controlled replay {', '.join(removed) if removed else 'none of D / E / C'} removes the initiating correction. "
              f"Remaining uncertainty: {', '.join(sv.get('verdict', []))}; material identity UNPROVEN (sparse part labels).", '']
    if LC:
        L += ['## G5 moving points: last usable camera', '', '| id | frame | role | camera | all reasons | Z-D (m) | frozen | per-iteration-style ownership (approx.) | rendered-depth visibility oracle | pred-to-GT (m) |', '|---|---|---|---|---|---|---|---|---|---|']
        for r in LC:
            L.append(f"| {r['point_id']} | {r['frame']} | {r['role']} | {r['camera']} | {r['reasons_all']} | {_f(float(r['depth_diff_Z_minus_D_m']), 3) if r['depth_diff_Z_minus_D_m'] not in ('', 'None') else '-'} | "
                     f"{r['frozen_assignment']} | {r['per_iteration_style_ownership_approx']} | {r['gt_visibility_oracle_rendered_depth']} | {_f(float(r['prediction_to_gt_m']), 3)} |")
    L += ['', '## Provenance (section 9)', '']
    for s, p in S['provenance'].items():
        off = [x['pixel_offset_px'] for x in p['x_ref_source_projection']]
        L.append(f"- {s}: x_ref source projection offset max |{_f(float(np.abs(off).max()) if off else float('nan'), 2)}| px, Z - source depth max |{_f(max((abs(x['Z_minus_source_depth_m']) for x in p['x_ref_source_projection']), default=float('nan')), 2)}| m; "
                 f"static-pixel constancy vs frame 0: " + '; '.join(f"t{x['frame']} cam{x['camera']}: {x['depth_identical_rgb_identical']}/{x['depth_identical']} depth-identical pixels have identical RGB (max |dRGB| {_f(x['depth_identical_rgb_max_abs_change'], 2)}), {x['depth_changed']} depth-changed"
                                                                 for x in p['static_pixel_constancy_vs_frame0'][:16]))
    (out/'SECOND_PASS_ROOT_CAUSE.md').write_text('\n'.join(L)+'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--forensics-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', help='comma list scene:id (smoke tests); default: every mandatory case')
    ap.add_argument('--report-only', action='store_true', help='regenerate SECOND_PASS_ROOT_CAUSE.md from an existing --out')
    a = ap.parse_args(argv)
    if a.report_only: write_report(a.out, a.forensics_root); return
    if a.out.exists() and any(a.out.iterdir()): raise SystemExit(f'{a.out} exists and is not empty: never overwritten')
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cases = dict(CASES)
    moving = {s: list(v) for s, v in MOVING.items()}
    if a.only:
        want = {(s, int(i)) for s, i in (x.split(':') for x in a.only.split(','))}
        cases = {k: v for k, v in cases.items() if k in want}
        moving = {s: [p for p in v if (s, p) in want] for s, v in moving.items()}
    scenes = {}
    get = lambda s: scenes.setdefault(s, Scene(a.forensics_root, s))
    R3, R4, ITS, SUP, SUPS, ARMS, CF, LAND, NPZ = [], [], [], [], [], [], [], {}, {}
    for (s, pid), (frames, role) in cases.items():
        sc = get(s); print(f'[second-pass] {s} {pid} {role} frames {frames} ({time.perf_counter()-t0:.0f} s)', flush=True)
        own0 = sc.frame0_ownership([pid])[pid]
        xr = xref_residuals(sc, pid, own0); R3 += xr
        cf = cf0(sc, pid) if role in ('static', 'control') else None
        if cf is not None:
            CF.append({'scene': s, 'point_id': pid, 'role': role, 'ignored_frame0_nonsource_valid_rows': ignored_frame0_rows(xr, pid),
                       **{k: v for k, v in cf.items() if not k.endswith('P_minus_frame1')}})
        for k in frames:
            fu, its = first_update(sc, pid, k); R4.append({**fu, 'role': role}); ITS += its
            rows, summ = support_objectives(sc, pid, k, its); SUP += rows; SUPS.append(summ)
            prior_rows = summ['S_prior']
            knife = role == 'knife_edge_moving' or any(abs(r['u']-round(r['u'])) < 1e-9 and abs(r['v']-round(r['v'])) < 1e-9 for r in xr if r['frame'] == k and 'u' in r and r.get('valid_production_rules'))
            ar = arms(sc, pid, k, prior_rows, knife=knife)
            for nm, v in ar.items(): ARMS.append({'scene': s, 'point_id': pid, 'frame': k, 'arm': nm, **v})
            if cf is not None and k == 1:
                for nm in ('A_frame1', 'B_frame1'): ARMS.append({'scene': s, 'point_id': pid, 'frame': 1, 'arm': 'C_frame0_nonsource_first_' + nm, **{kk: vv for kk, vv in cf[nm].items() if kk != 'linearized_one_step'}})
            NPZ[f'{s}_{pid}_{k}_P_minus'] = np.array(fu['P_minus'])
        if (s, pid, frames[0]) in LANDSCAPE or (s, pid) in [(x[0], x[1]) for x in LANDSCAPE]:
            k = frames[0]
            rec, ch = landscape(sc, pid, k, a.out/f'objective_1D_{s}_{pid}.png', [i for i in ITS if i['point_id'] == pid and i['frame'] == k and i['scene'] == s])
            LAND[f'{s}_{pid}_{k}'] = {'path': rec, 'cell_or_support_changes_at_alpha': ch}
            patch_png(sc, pid, k, a.out/f'rgbd_patch_{s}_{pid}.png')
    LC, LCS = [], {}
    for s, pids in moving.items():
        sc = get(s)
        for p in pids:
            print(f'[second-pass] moving {s} {p}', flush=True)
            r_, info = last_camera(sc, p); LC += r_; LCS[f'{s}_{p}'] = info
    prov = {s: provenance_checks(sc, [p for (ss, p) in cases if ss == s]) for s, sc in scenes.items()}
    wcsv(a.out/'frame0_frame1_raw_residuals.csv', R3)
    wcsv(a.out/'first_update_decomposition.csv', [{k: v for k, v in r.items() if k not in ('eigenvectors',)} for r in R4])
    wcsv(a.out/'iteration_support_objectives.csv', SUP)
    wcsv(a.out/'local_counterfactuals.csv', ARMS)
    wcsv(a.out/'last_camera_loss.csv', LC)
    with open(a.out/'first_update_iterations.jsonl', 'w') as f:
        for r in ITS: f.write(json.dumps(jsonable(r))+'\n')
    np.savez_compressed(a.out/'first_update_arrays.npz', **NPZ)
    write_json(a.out/'summary.json', {'support_verdicts': SUPS, 'cf0': CF, 'landscape': LAND, 'moving_loss_frames': LCS, 'provenance': prov,
                                      'first_updates': R4, 'seconds': time.perf_counter()-t0,
                                      'inputs': {s: {'traced_run': str(sc.run.root), 'git_commit_of_traced_run': sc.run.man.get('git_commit'),
                                                     'source_hashes': sc.run.man.get('source_hashes')} for s, sc in scenes.items()}})
    write_report(a.out, a.forensics_root)
    print(json.dumps({'seconds': time.perf_counter()-t0, 'points': len(cases), 'moving': sum(len(v) for v in moving.values()),
                      'verdicts': [(v['scene'][:4], v['point_id'], v['frame'], v['verdict']) for v in SUPS]}, default=str)[:3000])


if __name__ == '__main__':
    main()
