"""Independent float64 NumPy oracle of the free projective IEKF for ONE point (RGBDiekf_deterministic_root_cause_debug_spec.md).

Debug only, never called by inference. Nothing is imported from estimator/: projection, the bilinear interpolant, the
same-surface rule, the row terms, the row gate, the fixed-prior information solve, the acceptance rule and the Joseph
covariance are re-derived here from the formulas, so agreement with the production trace tests the algebra and the
sampling, not a shared code path. Images are the original dataset arrays cast to float32 exactly as
estimator/projective_sensor.MeasuredFrame stores them on the GPU (that cast is part of the production measurement).

Per-row model (camera c, canonical colour a, world point x):
    X_c = R x + t,  u = fx X/Z + cx,  v = fy Y/Z + cy,  J_u = J_pi R,  J_Z = R[2]
    r = [a - I(u,v) ; Z - D(u,v)],  H_x = [dI/duv J_u ; dD/duv J_u - J_Z],  R = diag(var_rgb, var_D)
Fixed-prior IEKF:  A = Lam + sum H^T W H,  b = -Lam (s_j - s-) + [sum H^T W r, 0],  delta = A^-1 b.
Exact additive attribution of one step: delta = A^-1 [-Lam e] + sum_{c,a} A^-1 [H_ca^T r_ca / R_ca, 0].
"""
import math
import numpy as np
from scipy.stats import chi2, norm

VALID, BEHIND, OUTSIDE, MASK, INVALID_DEPTH, DISCONT, SELF_OCC, DUP, FRONT, NOT_PROC, INCONS = range(11)
NAMES = ['VALID', 'BEHIND_CAMERA', 'OUTSIDE_IMAGE_FOOTPRINT', 'MASK_EXCLUDED', 'INVALID_DEPTH', 'DEPTH_DISCONTINUITY', 'SELF_OCCLUDED',
         'DUPLICATE_PIXEL_SUPPRESSED', 'MEASURED_SURFACE_IN_FRONT', 'NOT_PROCESSED', 'INNOVATION_INCONSISTENT']
CODE = {n: i for i, n in enumerate(NAMES)}
CHANNELS = ('R', 'G', 'B', 'D')


class Cam:
    def __init__(self, camera):
        T = np.asarray(camera.world_to_camera, float)
        self.R, self.t = T[:3, :3], T[:3, 3]
        self.fx, self.fy, self.cx, self.cy = (float(v) for v in camera.intrinsics)
        self.near, self.W, self.H, self.name = float(camera.near), int(camera.width), int(camera.height), camera.name

    def project(self, x):
        Xc = self.R@x+self.t
        X, Y, Z = Xc
        s = Z if Z != 0 else 1.
        u, v = self.fx*X/s+self.cx, self.fy*Y/s+self.cy
        Jpi = np.array([[self.fx/s, 0, -self.fx*X/s**2], [0, self.fy/s, -self.fy*Y/s**2]])
        return u, v, Z, Jpi@self.R, self.R[2].copy()

    def ray_world(self, u, v):
        b = np.array([(u-self.cx)/self.fx, (v-self.cy)/self.fy, 1.])
        return self.R.T@b

    def centre(self):
        return -self.R.T@self.t


class Image:
    """One camera at one frame, as the GPU holds it (MeasuredFrame default: rgb, depth float32; mask, raw_valid bool)."""
    def __init__(self, obs, dtype=np.float32):
        self.rgb = np.asarray(obs.rgb, np.float64).astype(dtype).astype(np.float64)
        self.depth = np.asarray(obs.depth, np.float64).astype(dtype).astype(np.float64)
        self.mask = np.asarray(obs.mask, bool); self.raw_valid = np.asarray(obs.raw_valid, bool)


class Noise:
    def __init__(self, rgb_var, depth_var, depth_sq, rgb_model=0., depth_model=0., interp='sum_w2'):
        self.rgb_var, self.a, self.b = np.asarray(rgb_var, float), float(depth_var), float(depth_sq)
        self.rgb_model, self.depth_model, self.interp = float(rgb_model), float(depth_model), interp

    @classmethod
    def from_manifest(cls, noise, policy):
        c = np.asarray(noise['color_covariance'], float)
        return cls(np.diag(c), noise['depth_variance'], noise['depth_squared_variance'], policy['rgb_model_variance'],
                   policy['depth_model_variance'], policy['interpolation_variance'])

    def sigma2(self, z):
        return self.a+self.b*np.asarray(z, float)**2

    def describe(self):
        return {'rgb_sensor_variance': self.rgb_var.tolist(), 'rgb_model_variance': self.rgb_model, 'depth_sensor_a_m2': self.a,
                'depth_sensor_b_per_z2': self.b, 'depth_model_variance_m2': self.depth_model, 'interpolation_variance': self.interp}


def sample(img, cam, u, v, noise, policy):
    """Bilinear sample of the raw image: reason, 2x2 neighbours, weights, value, exact interpolant gradient, variance."""
    out = {'reason': VALID}
    if not (math.isfinite(u) and math.isfinite(v)):
        out['reason'] = OUTSIDE; return out
    u0f, v0f = math.floor(u), math.floor(v)
    inside = u0f >= 0 and v0f >= 0 and u0f+1 <= cam.W-1 and v0f+1 <= cam.H-1
    u0, v0 = min(max(u0f, 0), cam.W-2), min(max(v0f, 0), cam.H-2)
    du, dv = min(max(u-u0f, 0.), 1.), min(max(v-v0f, 0.), 1.)
    nb = [(v0, u0), (v0, u0+1), (v0+1, u0), (v0+1, u0+1)]
    D = np.array([img.depth[y, x] for y, x in nb]); I = np.array([img.rgb[y, x] for y, x in nb])
    mk = np.array([img.mask[y, x] for y, x in nb]); rv = np.array([img.raw_valid[y, x] for y, x in nb])
    out.update(pixel00=(u0, v0), du=du, dv=dv, neighbour_rgb=I, neighbour_depth=D, neighbour_mask=mk, neighbour_raw_valid=rv)
    if not (rv.all() and np.isfinite(D).all() and np.isfinite(I).all()): out['reason'] = INVALID_DEPTH
    if not mk.all(): out['reason'] = MASK
    if not inside: out['reason'] = OUTSIDE
    if out['reason'] != VALID: return out
    zbar = D.mean(); f = min(cam.fx, cam.fy)
    allow = policy['same_surface_n_sigma']*math.sqrt(2.)*math.sqrt(noise.sigma2(zbar)) + zbar*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2.)/f
    out['same_surface_spread_m'], out['same_surface_allowance_m'] = float(D.max()-D.min()), float(allow)
    w = np.array([(1-du)*(1-dv), du*(1-dv), (1-du)*dv, du*dv])
    out['w'] = w
    out['depth'], out['rgb'] = float(w@D), w@I
    out['grad_depth'] = np.array([(1-dv)*(D[1]-D[0])+dv*(D[3]-D[2]), (1-du)*(D[2]-D[0])+du*(D[3]-D[1])])
    out['grad_rgb'] = np.stack([(1-dv)*(I[1]-I[0])+dv*(I[3]-I[2]), (1-du)*(I[2]-I[0])+du*(I[3]-I[1])], 1)      # [3,2]
    if noise.interp == 'sum_w2':
        out['var_rgb'] = (w*w).sum()*noise.rgb_var+noise.rgb_model
        out['var_depth'] = float((w*w*noise.sigma2(D)).sum())+noise.depth_model
    else:
        out['var_rgb'] = noise.rgb_var+noise.rgb_model; out['var_depth'] = float(noise.sigma2(out['depth']))+noise.depth_model
    if D.max()-D.min() > allow: out['reason'] = DISCONT
    return out


def row_gate_threshold(n_sigma, dof=4):
    return float(chi2.isf(2*norm.sf(n_sigma), dof))


def row(img, cam, x, a, noise, policy, Sigma_x, dx):
    """One camera row at world x: projection, sample, r, H_x, R, prior-linearized innovation and the row gate."""
    u, v, Z, Ju, Jz = cam.project(x)
    s = sample(img, cam, u, v, noise, policy)
    o = {'u': u, 'v': v, 'Z': Z, 'J_u': Ju, 'J_Z': Jz, 'sample': s, 'reason': s['reason']}
    if Z <= cam.near or not math.isfinite(Z): o['reason'] = BEHIND
    if o['reason'] != VALID: return o
    H = np.zeros((4, 3)); H[:3] = s['grad_rgb']@Ju; H[3] = s['grad_depth']@Ju-Jz
    r = np.r_[np.asarray(a, float)-s['rgb'], Z-s['depth']]
    var = np.r_[s['var_rgb'], s['var_depth']]
    S = H@Sigma_x@H.T+np.diag(var); nu = r+H@dx
    gate = policy.get('row_innovation_gate', 'depth_front_only')
    o.update(H=H, r=r, var=var, S=S, nu=nu, S_zz=float(S[3, 3]))
    n = policy['visibility_n_sigma']
    o['nis'] = float(nu@np.linalg.solve(S, nu))
    front = r[3] if gate == 'depth_front_only' else nu[3]          # the audited rule tests the iterate's own residual
    if front > n*math.sqrt(max(S[3, 3], 0)): o['reason'] = FRONT
    elif gate == 'depth_two_sided' and nu[3] < -n*math.sqrt(max(S[3, 3], 0)): o['reason'] = INCONS
    elif gate == 'rgbd_chi2' and not o['nis'] <= row_gate_threshold(n): o['reason'] = INCONS
    return o


def geometric_visibility(cam, x):
    """The point's own visibility reasons (projective_visibility without the other points): behind / outside image."""
    u, v, Z, _, _ = cam.project(x)
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(Z)) or Z <= cam.near: return BEHIND
    pu, pv = round_half_even(u), round_half_even(v)
    return VALID if (0 <= pu <= cam.W-1 and 0 <= pv <= cam.H-1) else OUTSIDE


def round_half_even(x):
    return float(np.round(x))       # torch.round rounds half to even, like np.round


def candidacy(cams, x, frozen, recheck_self=None):
    """Per-frame frozen pixel assignment (apply_frozen_assignment): a camera frozen SELF_OCCLUDED / DUPLICATE at the prior
    stays excluded; otherwise the point's own geometric reason at x. recheck_self: cameras that are freshly SELF_OCCLUDED
    at the acceptance linearization (needs the other points' states: taken from the production trace)."""
    out = []
    for c, cam in enumerate(cams):
        if frozen[c] in (SELF_OCC, DUP): out.append(frozen[c]); continue
        if recheck_self is not None and recheck_self[c]: out.append(SELF_OCC); continue
        out.append(geometric_visibility(cam, x))
    return out


class Problem:
    """One point at one frame: the fixed prior, the frame's images, cameras, frozen assignment and the policy."""
    def __init__(self, cams, imgs, a, s_minus, P_minus, frozen, noise, policy, max_iters, tol, recheck_self=None,
                 channels=(1, 1, 1, 1), cam_mask=None, fixed_rows=None):
        self.cams, self.imgs, self.a, self.noise, self.policy = cams, imgs, np.asarray(a, float), noise, policy
        self.s_minus, self.P_minus = np.asarray(s_minus, float), np.asarray(P_minus, float)
        self.Lam = np.linalg.inv(self.P_minus); self.Lam = (self.Lam+self.Lam.T)/2
        self.frozen, self.recheck_self = frozen, recheck_self
        self.max_iters, self.tol = max_iters, tol
        self.ch = np.asarray(channels, float)
        self.cam_mask = np.ones(len(cams), bool) if cam_mask is None else np.asarray(cam_mask, bool)
        self.fixed_rows = fixed_rows                         # None, or cameras allowed (first-iterate row set)

    def rows(self, x, final=False):
        dx = x-self.s_minus[:3]
        vis = candidacy(self.cams, x, self.frozen, self.recheck_self if final else None)
        out = []
        for c, cam in enumerate(self.cams):
            if vis[c] != VALID:
                out.append({'camera': c, 'reason': vis[c], 'visibility': vis[c]}); continue
            o = row(self.imgs[c], cam, x, self.a, self.noise, self.policy, self.P_minus[:3, :3], dx)
            o.update(camera=c, visibility=VALID)
            if o['reason'] == VALID and not self.cam_mask[c]: o['excluded_by_variant'] = True
            if o['reason'] == VALID and self.fixed_rows is not None and c not in self.fixed_rows: o['excluded_by_variant'] = True
            out.append(o)
        return out

    def used(self, rows):
        return [o for o in rows if o['reason'] == VALID and not o.get('excluded_by_variant')]

    def info(self, rows):
        Ixx, gx, q = np.zeros((3, 3)), np.zeros(3), 0.
        for o in self.used(rows):
            w = self.ch/o['var']
            Ixx += o['H'].T@(w[:, None]*o['H']); gx += o['H'].T@(w*o['r']); q += float((w*o['r']**2).sum())
        return Ixx, gx, q

    def prior_cost(self, s):
        e = s-self.s_minus
        return float(e@self.Lam@e)

    def solve(self, s, rows):
        Ixx, gx, q = self.info(rows)
        A = self.Lam.copy(); A[:3, :3] += Ixx
        b = -self.Lam@(s-self.s_minus); b[:3] += gx
        d = np.linalg.solve(A, b)
        return A, b, d, float(d@b), Ixx, gx, q

    def attribution(self, s, rows, A):
        """Exact additive split of delta: prior pull and every (camera, channel) push, plus that term's information."""
        Ainv = np.linalg.inv(A)
        e = s-self.s_minus
        terms = {'prior': {'delta': (Ainv@(-self.Lam@e)).tolist()}}
        for o in self.used(rows):
            for k in range(4):
                if not self.ch[k]: continue
                g = o['H'][k]*o['r'][k]/o['var'][k]
                d = Ainv[:, :3]@g
                terms[f"cam{o['camera']}_{CHANNELS[k]}"] = {'g': g.tolist(), 'delta': d.tolist(), 'chi2': float(o['r'][k]**2/o['var'][k]),
                                                           'info_trace': float((o['H'][k]**2).sum()/o['var'][k]),
                                                           'I': (np.outer(o['H'][k], o['H'][k])/o['var'][k]).tolist()}
        return terms

    def data_cost(self, rows):
        rgb = dep = 0.
        for o in self.used(rows):
            z = o['r']**2/o['var']
            rgb += float((self.ch[:3]*z[:3]).sum()); dep += float(self.ch[3]*z[3])
        return rgb, dep

    def cost_on(self, s, cameras):
        """Actual nonlinear data cost of a FIXED camera set at state s (resampled at s; rows invalid at s are listed)."""
        dx = s[:3]-self.s_minus[:3]
        rgb = dep = 0.; lost = []; per = {}
        for c in cameras:
            o = row(self.imgs[c], self.cams[c], s[:3], self.a, self.noise, self.policy, self.P_minus[:3, :3], dx)
            if 'r' not in o: lost.append((c, NAMES[o['reason']])); continue
            z = o['r']**2/o['var']
            rgb += float((self.ch[:3]*z[:3]).sum()); dep += float(self.ch[3]*z[3])
            per[c] = {'rgb_chi2': float(z[:3].sum()), 'depth_chi2': float(z[3]), 'reason_at_state': NAMES[o['reason']], 'u': o['u'], 'v': o['v'],
                      'sampled_depth_m': o['sample']['depth'], 'Z': o['Z'], 'sampled_rgb': o['sample']['rgb'].tolist()}
        return {'rgb': rgb, 'depth': dep, 'lost_rows': lost, 'per_camera': per}

    def run(self, rule='final_iterate', record=True):
        thr = [float('inf')]+[float(chi2.isf(2*norm.sf(self.policy['visibility_n_sigma']), int(self.ch.sum())*m)) for m in range(1, len(self.cams)+1)]
        sj = self.s_minus.copy(); hist = []; acc_state, acc_iter, had = self.s_minus.copy(), -1, False
        converged = False
        for j in range(self.max_iters):
            rows = self.rows(sj[:3])
            A, b, d, step, Ixx, gx, q = self.solve(sj, rows)
            m = len(self.used(rows)); had |= m > 0
            J = self.prior_cost(sj)+q
            if j >= 1 and m > 0 and J <= thr[min(m, len(self.cams))]: acc_state, acc_iter = sj.copy(), j
            h = {'iteration': j, 's': sj.copy(), 'rows': rows, 'A': A, 'b': b, 'delta': d, 'step': step, 'Ixx': Ixx, 'gx': gx, 'rTWr': q,
                 'nrows': m, 'J': J, 'threshold': thr[min(m, len(self.cams))]}
            hist.append(h)
            sj = sj+d
            if step < self.tol: converged = True; break
        rows_f = self.rows(sj[:3], final=True)
        Ixx_f, _, q_f = self.info(rows_f); m_f = len(self.used(rows_f))
        J_f = self.prior_cost(sj)+q_f
        state, Ixx_acc, rows_acc, code = sj, Ixx_f, rows_f, 4
        if rule == 'final_iterate':
            ok = m_f > 0; code = 1 if ok else 4
        else:
            ok = m_f > 0 and J_f <= thr[min(m_f, len(self.cams))]
            code = 1
            if not ok:
                if acc_iter >= 1:
                    rows_e = self.rows(acc_state[:3], final=True); Ixx_e, _, q_e = self.info(rows_e); m_e = len(self.used(rows_e))
                    J_e = self.prior_cost(acc_state)+q_e
                    ok = m_e > 0 and J_e <= thr[min(m_e, len(self.cams))]
                    state, Ixx_acc, rows_acc, code = acc_state, Ixx_e, rows_e, 2 if ok else 3
                else:
                    code = 3 if (had or m_f > 0) else 4
        A = self.Lam.copy(); A[:3, :3] += Ixx_acc
        P_info = np.linalg.inv(A); P_info = (P_info+P_info.T)/2
        s_plus, P_plus = (state, P_info) if ok else (self.s_minus.copy(), self.P_minus.copy())
        return {'history': hist, 's_final_iterate': sj, 'converged': converged, 'rows_final': rows_f, 'rows_accepted': rows_acc, 'acceptance_code': code,
                'observed': bool(ok), 's_plus': s_plus, 'P_plus': P_plus, 'P_information': P_info, 'J_final': J_f}

    def joseph(self, rows):
        """Dense textbook Kalman on exactly these admitted rows: K, (I-KH)P(I-KH)^T + K R K^T and inv(Lam + H^T R^-1 H)."""
        u = self.used(rows)
        if not u: return None
        keep = [(o, k) for o in u for k in range(4) if self.ch[k]]
        H = np.zeros((len(keep), 6)); H[:, :3] = [o['H'][k] for o, k in keep]
        R = np.diag([o['var'][k] for o, k in keep])
        S = H@self.P_minus@H.T+R; K = self.P_minus@H.T@np.linalg.inv(S)
        IKH = np.eye(6)-K@H
        Pj = IKH@self.P_minus@IKH.T+K@R@K.T
        Pi = np.linalg.inv(self.Lam+H.T@np.linalg.inv(R)@H)
        return {'K': K, 'P_joseph': Pj, 'P_information': Pi, 'max_abs_difference': float(np.abs(Pj-Pi).max()), 'H': H, 'R_diag': np.diag(R)}


def finite_difference_H(prob, x, cam, eps=1e-7):
    """Central differences of the SAME bilinear interpolant (h = [I - a ; D - Z] = -r) at x, camera cam. A difference that
    crosses a pixel-cell border is flagged: the interpolant's gradient jumps there (piecewise bilinear)."""
    o = row(prob.imgs[cam], prob.cams[cam], x, prob.a, prob.noise, {**prob.policy, 'row_innovation_gate': 'depth_front_only', 'visibility_n_sigma': float('inf')}, prob.P_minus[:3, :3], np.zeros(3))
    if 'H' not in o: return None
    fd = np.zeros((4, 3)); cell_cross = False
    for k in range(3):
        e = np.zeros(3); e[k] = eps
        hp = row(prob.imgs[cam], prob.cams[cam], x+e, prob.a, prob.noise, {**prob.policy, 'visibility_n_sigma': float('inf'), 'row_innovation_gate': 'depth_front_only'}, prob.P_minus[:3, :3], np.zeros(3))
        hm = row(prob.imgs[cam], prob.cams[cam], x-e, prob.a, prob.noise, {**prob.policy, 'visibility_n_sigma': float('inf'), 'row_innovation_gate': 'depth_front_only'}, prob.P_minus[:3, :3], np.zeros(3))
        if 'r' not in hp or 'r' not in hm: return None
        cell_cross |= hp['sample']['pixel00'] != hm['sample']['pixel00']
        fd[:, k] = (-hp['r']+hm['r'])/(2*eps)
    return {'H_analytic': o['H'], 'H_fd': fd, 'max_abs_error': float(np.abs(fd-o['H']).max()),
            'rel_error': float(np.linalg.norm(fd-o['H'])/max(np.linalg.norm(o['H']), 1e-300)), 'crosses_pixel_cell': bool(cell_cross)}


def predict(s, P, dt, qc):
    tau, sg = abs(dt), math.copysign(1., dt)
    F = np.eye(6); F[:3, 3:] = dt*np.eye(3)
    qc = np.asarray(qc, float)
    Q = np.block([[tau**3/3*qc, sg*tau**2/2*qc], [sg*tau**2/2*qc, tau*qc]])
    Pm = F@P@F.T+Q
    return F@s, (Pm+Pm.T)/2, F, Q
