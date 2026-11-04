#!/usr/bin/env python3
"""POST-HOC falsification and root-cause audit of the SVGD Stage-1 objective (CODEX_RGBDiekf_SVGD_Falsification_And_Root_Cause_2026-11-03.md).

Separate process, AFTER the run's inference artifacts are committed. GT (membership + joint transform) is built here only and never
reaches estimator/svgd_* (tests/test_svgd_falsification.py has a leak sentinel). Nothing in the run directory is modified: every
output goes to a NEW folder RUN/posthoc/svgd_falsification[_N]/.

    python -B tools/audit_svgd_falsification.py --run RUN --scene SCENE                # Sapien GT (tools/audit_temporal_dynamic_correspondence)
    python -B tools/audit_svgd_falsification.py --run RUN --synthetic                 # run made by tools/svgd_synthetic_run.py (fixture pose = GT)

Model reconstruction (Gate F0): the production objective E = sum_t E_F,t + E_time + normalizers is rebuilt by calling the SAME functions with
the SAME inputs as estimator/svgd_support_run._stages (config file verified by SHA-256, noise re-estimated by the same routine and compared to
noise_calibration.json, every cloud compared to completed_frames.json, Lambda / H / RFF hash compared to status.json, E at the stationary
state and every logged backtrack compared to objective_timeline.csv). Any mismatch -> AUDIT_INVALID.

Gates (spec sections 3-7): F0 provenance + GT verification; F1 E_obs of S / GT / I(alpha) / logged Stein candidates (+ diagnostic feature
models, labelled DIFFERENT OBSERVATION MODEL); F2 E_time* = min_V E_time (exact velocity_lift, KKT residual) with anchor / per-interval /
cross-term decomposition; F3 Stein-field audit (norms, cosines, FD closure, attraction / repulsion / colour, post-hoc source split, KDE basin
and responsibility transfer); F4 offline proposal arms A-D on the identical objective. F5 is run only if F1 and F2 prefer GT.
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
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ALPHAS = [0, .01, .05, .1, .25, .5, .75, 1, 1.25]


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def sha_array(a): return sha_bytes(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())


def q(v):
    v = np.asarray(v, float).ravel(); v = v[np.isfinite(v)]
    return {'n': int(len(v)), 'p10': float(np.quantile(v, .1)), 'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)),
            'mean': float(v.mean())} if len(v) else {'n': 0}


# ------------------------------------------------------------------ Gate F0: exact read-only rebuild of the production model
class Model:
    """Production objective of one run, rebuilt with the runner's own functions (read-only wrapper; no inference code is changed)."""
    def __init__(self, run, data, device):
        from scene.config import load_config
        from estimator.temporal_noise import temporal_sensor_only
        from scene.pointcloud_sequence import build_masked_cloud, canonical_from_frame
        from estimator.svgd_support_run import representation_scales, source_order, median_heuristic, SequenceObjective, value
        from estimator.svgd_feature_measurement import RFFFeatures, FeatureGaussian, iid_sampling_gamma, FrameFeatureTerm
        from estimator.svgd_temporal import TemporalPrior
        self.run, self.dev, self.dt = Path(run), torch.device(device), torch.float64
        self.FrameFeatureTerm, self.FeatureGaussian, self.iid = FrameFeatureTerm, FeatureGaussian, iid_sampling_gamma
        sv = self.run/'svgd_support'
        self.man = json.loads((self.run/'run_manifest.json').read_text()); self.status = json.loads((sv/'status.json').read_text())
        A = self.man['arguments']; self.checks = []
        cfg_bytes = Path(A['config']).read_bytes(); self.check('config_sha256', sha_bytes(cfg_bytes) == self.man['config_sha256'], A['config'])
        self.cfg = cfg = json.loads(cfg_bytes)
        cur = {s: hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in self.man['source_sha256'] if (REPO/s).is_file()}
        changed = [s for s, h in self.man['source_sha256'].items() if cur.get(s) != h]
        self.source_changes_since_run = changed
        self.check('inference_sources_unchanged_since_run', not [s for s in changed if s.startswith(('estimator/', 'scene/'))], changed)
        config = load_config(A.get('noise_config'))
        if A['sensor_noise'] == 'measured': config, _, calib = temporal_sensor_only(data, config, {})
        saved = json.loads((self.run/'noise_calibration.json').read_text())
        if A['sensor_noise'] == 'measured':
            self.check('noise_reestimated_equal', saved['temporal_sensor']['result'] == json.loads(json.dumps(calib['result'])), saved['temporal_sensor']['result'])
        self.config, self.data = config, data
        tt = self.tt
        self.clouds = [build_masked_cloud(data, k, config) for k in range(len(data.times))]
        frames = json.loads((sv/'completed_frames.json').read_text())
        self.check('clouds_equal_completed_frames', all(c.digest() == f['cloud_sha256'] for c, f in zip(self.clouds, frames)) and len(frames) == len(self.clouds), len(frames))
        canon = canonical_from_frame(self.clouds[0]); can = np.load(sv/'canonical.npz'); self.canon = canon
        self.check('canonical_equal', np.array_equal(canon.xyz0, can['xyz0']) and np.array_equal(canon.source, can['source']) and np.array_equal(canon.cov0, can['cov0']), canon.N)
        self.N, self.K = canon.N, len(self.clouds)
        self.x0, self.c0, self.cov0 = tt(canon.xyz0), tt(canon.rgb0), tt(canon.cov0)
        names = [c.name for c in data.cameras]; R_c = np.asarray(config.color_covariance, float); self.R_c = R_c
        self.scales = []
        for cl in self.clouds:
            s2, lc = representation_scales(tt(cl.xyz), tt(cl.rgb), R_c, value(cfg, 'target_density.Lambda_X'), source_order(cl.source, names))
            self.scales.append((s2, lc))
        sp = self.status['gates']['TARGET']['representation_scales']['Lambda_X_spacing_m']
        self.check('Lambda_X_spacing_equal', np.allclose([math.sqrt(s) for s, _ in self.scales], sp, rtol=0, atol=0), sp[:3])
        o0 = source_order(canon.source, names); hx, _ = median_heuristic(self.x0, o0); hc, _ = median_heuristic(self.c0, o0)
        F = self.status['gates']['FEATURES']
        self.check('H_equal', hx == F['H_X_m2'] and hc == F['H_C'], (hx, F['H_X_m2']))
        self.H_x, self.H_c = hx*torch.eye(3, dtype=self.dt, device=self.dev), hc*torch.eye(3, dtype=self.dt, device=self.dev)
        self.feature_args = dict(D=int(value(cfg, 'features.D')), seed=int(value(cfg, 'features.fixed_seed')))
        self.feats = RFFFeatures(self.H_x.cpu(), self.H_c.cpu(), self.feature_args['D'], self.feature_args['seed'], self.dt, self.dev)
        self.check('rff_sha256_equal', self.feats.digest() == F['rff_sha256'], F['rff_sha256'][:16])
        self.terms = self.feature_terms(self.feats)
        self.check('gamma_logdet_equal', all(abs(t.G.logdet-g['logdet']) <= 1e-9*abs(g['logdet']) for t, g in zip(self.terms, F['gamma'])), [g['logdet'] for g in F['gamma'][:2]])
        self.temporal = TemporalPrior(data.times, config.qc, self.x0, self.cov0, np.asarray(config.velocity_covariance, float))
        self.obj = SequenceObjective(self.terms, self.temporal)
        self.S0 = torch.zeros(self.N, self.K, 6, dtype=self.dt, device=self.dev); self.S0[:, :, :3] = self.x0[:, None]
        self.E0 = self.obj.value(self.S0)
        self.check('E_stationary_equal_timeline_init', self.E0['total'] == F['E_initial']['total'], (self.E0['total'], F['E_initial']['total']))
        self.timeline = list(csv.DictReader(open(sv/'objective_timeline.csv')))

    def tt(self, v): return torch.as_tensor(np.asarray(v), dtype=self.dt, device=self.dev)

    def check(self, name, ok, detail):
        self.checks.append({'check': name, 'ok': bool(ok), 'detail': json.loads(json.dumps(detail, default=str))})

    @property
    def valid(self): return all(c['ok'] for c in self.checks)

    def feature_terms(self, feats):
        """FrameFeatureTerm per frame with the production Gamma rule applied to `feats` (the production model when feats = self.feats)."""
        out = []
        for cl in self.clouds:
            y, d = self.tt(cl.xyz), self.tt(cl.rgb)
            out.append(self.FrameFeatureTerm(feats, y, d, self.FeatureGaussian(self.iid(feats, y, d, self.x0, self.c0)), self.c0))
        return out


def time_parts(temporal, S):
    """Exact decomposition of E_time(S): frame-0 anchor (position / velocity / cross blocks of P0^-1) and, per interval, the process
    quadratic w^T Q^-1 w split into xx, 2 xv and vv blocks. Parts add up to temporal.energy(S) (checked by the caller)."""
    r0 = S[:, 0]-temporal.s0bar; P = temporal.P0i
    a_xx = .5*float(torch.einsum('ni,nij,nj->', r0[:, :3], P[:, :3, :3], r0[:, :3])); a_vv = .5*float(torch.einsum('ni,nij,nj->', r0[:, 3:], P[:, 3:, 3:], r0[:, 3:]))
    a_xv = float(torch.einsum('ni,nij,nj->', r0[:, :3], P[:, :3, 3:], r0[:, 3:]))
    Pn = P[:, :3, :3].cpu().numpy(); ev, V = np.linalg.eigh(np.linalg.inv(Pn))       # source covariance eigen-directions (CPU: cuSOLVER batch limit)
    proj = np.einsum('nij,ni->nj', V, r0[:, :3].cpu().numpy()); per_axis = (.5*proj**2/ev).sum(0)
    out = {'anchor_xx': a_xx, 'anchor_vv': a_vv, 'anchor_xv': a_xv, 'anchor_by_source_eigendirection_small_to_large': per_axis.tolist(), 'intervals': []}
    for t in range(1, S.shape[1]):
        w = S[:, t]-S[:, t-1]@temporal.F[t-1].T; Qi = temporal.Qi[t-1]
        out['intervals'].append({'t': t, 'xx': .5*float(((w[:, :3]@Qi[:3, :3])*w[:, :3]).sum()), 'xv': float(((w[:, :3]@Qi[:3, 3:])*w[:, 3:]).sum()),
                                 'vv': .5*float(((w[:, 3:]@Qi[3:, 3:])*w[:, 3:]).sum())})
    out['anchor'] = a_xx+a_vv+a_xv; out['process'] = sum(i['xx']+i['xv']+i['vv'] for i in out['intervals']); out['total'] = out['anchor']+out['process']
    return out


def lift_positions(m, X):
    """[X, V*] with V* = argmin_V E_time([X, V]) by the production velocity_lift (exact), and its KKT residual."""
    S = torch.zeros(m.N, m.K, 6, dtype=m.dt, device=m.dev); S[:, :, :3] = X
    S[:, :, 3:] = m.temporal.velocity_lift(S, torch.zeros_like(X))
    g = m.temporal.grad(S)[:, :, 3:]; Sv = S.clone(); Sv[:, :, 3:] = 0
    g0 = m.temporal.grad(Sv)[:, :, 3:]
    return S, float(g.abs().max()/max(float(g0.abs().max()), 1e-300))


# ------------------------------------------------------------------ GT oracle (post hoc only)
def gt_oracle(m, scene=None, synthetic=None):
    if synthetic is not None:
        mv = np.array([synthetic.observation(0, c).gt_moving.ravel()[p] for _, c, p in m.canon.source])
        X = np.stack([m.canon.xyz0@synthetic.pose(k)[0].T+synthetic.pose(k)[1] for k in range(m.K)]); X[:, ~mv] = m.canon.xyz0[~mv]
        ev = {'source': 'synthetic fixture pose (tests/svgd_fixtures.SyntheticDataset)', 'motion': synthetic.motion}
    else:
        from tools.audit_temporal_dynamic_correspondence import ground_truth
        X, mv, dist, ev = ground_truth(Path(scene), m.data, m.canon.xyz0)
    # Verification against the MEASURED clouds, using only the evidence the stationary hypothesis cannot explain. A tangentially sliding,
    # uniformly coloured part re-samples identical rays, so plain XYZ/RGB NN distances favour "stationary" without saying anything about GT.
    # U_t = observed frame-t samples farther than the KDE spacing s_t from every canonical X0 point. Along the post-hoc path
    # X(lam) = X0 + lam (X_GT[t] - X0), lam in [-1.5, 1.5] step 0.05, the scale explaining most of U_t (NN < s_t) must be lam* within 0.1
    # of 1 ... replaced by a binomial criterion: at every frame whose true displacement exceeds s_t, the GT scale explains U_t as well as the
    # best scale within 2 standard errors and better than the stationary scale by > 2 SE (declared criterion; GT used post hoc only).
    from scipy.spatial import cKDTree
    from scipy.stats import chi2
    lams = np.round(np.arange(-1.5, 1.5001, .05), 2); ver = []; r6 = math.sqrt(chi2.ppf(.99, 6))
    for k in range(m.K):
        y = m.clouds[k].xyz; s_t = math.sqrt(m.scales[k][0]); lc = np.sqrt(np.diag(m.scales[k][1].cpu().numpy())); row = {'frame': k, 'kde_spacing_m': s_t}
        w6 = lambda P, C: np.c_[P/s_t, C/lc]                                   # whitened XYZ-RGB (KDE spacing, Lambda_C): joint geometry + colour
        Y6 = w6(y, m.clouds[k].rgb); d0, _ = cKDTree(w6(m.canon.xyz0, m.canon.rgb0)).query(Y6)
        U = Y6[d0 > r6]; row['unexplained_by_stationary_fraction'] = float((d0 > r6).mean())
        if mv.any():
            row['true_displacement_moving_m'] = q(np.linalg.norm(X[k, mv]-X[0, mv], axis=1))
            for nm, P in (('gt', X[k, mv]), ('stationary', m.canon.xyz0[mv])):
                dist, j = cKDTree(y).query(P); row[f'moving_{nm}_nn_m'] = q(dist); row[f'moving_{nm}_colour_residual'] = q(np.linalg.norm(m.clouds[k].rgb[j]-m.canon.rgb0[mv], axis=1))
            if len(U):
                expl = [float((cKDTree(w6(m.canon.xyz0+l*(X[k]-X[0]), m.canon.rgb0)).query(U)[0] <= r6).mean()) for l in lams]
                row['U_explained_by_lambda'] = dict(zip(lams.tolist(), expl)); row['lambda_star'] = float(lams[int(np.argmax(expl))])
                row['U_explained_at_gt'] = expl[int(np.argmin(np.abs(lams-1)))]; row['U_explained_at_stationary'] = expl[int(np.argmin(np.abs(lams)))]
                row['U_count'] = int(len(U)); row['U_explained_best'] = max(expl)
                row['binomial_2se'] = 2*math.sqrt(max(row['U_explained_best']*(1-row['U_explained_best']), 1e-12)/len(U))
        ver.append(row)
    frame0 = bool(np.array_equal(X[0], m.canon.xyz0))
    moving_frames = [v for v in ver[1:] if mv.any() and v['true_displacement_moving_m']['median'] > v['kde_spacing_m']]
    ok = frame0 and mv.any() and bool(moving_frames) and all('lambda_star' in v and v['U_explained_at_gt'] >= v['U_explained_best']-v['binomial_2se']
                                                             and v['U_explained_at_gt'] > v['U_explained_at_stationary']+v['binomial_2se'] for v in moving_frames)
    return X, mv, {'status': 'GT_TRAJECTORY_VERIFIED' if ok else 'GT_TRAJECTORY_UNVERIFIED', 'frame0_identical_to_X0': frame0, 'moving_members': int(mv.sum()),
                   'static_members': int((~mv).sum()), 'evidence': ev, 'observation_support_check': ver,
                   'criterion': 'X_GT[0] == X0 bitwise; joint XYZ-RGB whitened distance d^2 = |dx|^2/s_t^2 + dc^T Lambda_C^-1 dc, explained iff d^2 <= chi2_6(0.99); '
                                'U_t = observed samples unexplained by X0; at every frame with true displacement > s_t the fraction of U_t explained at lam = 1 '
                                '>= best lam in [-1.5, 1.5] minus 2 binomial SE and > the fraction at lam = 0 plus 2 SE'}


# ------------------------------------------------------------------ Gates F1 + F2
def candidate_energies(m, X, name, kind, alpha, terms=None, gt_disp=None):
    terms = terms or m.terms
    Eo = [t.nll(X[:, k]) for k, t in enumerate(terms)]; const = [.5*t.G.logdet+.5*t.G.D*math.log(2*math.pi) for t in terms]
    S, kkt = lift_positions(m, X); tp = time_parts(m.temporal, S); et = m.temporal.energy(S)
    return {'candidate': name, 'kind': kind, 'alpha': alpha, 'E_obs_t': Eo, 'E_obs_quadratic_t': [a-b for a, b in zip(Eo, const)], 'E_obs': float(sum(Eo)),
            'E_time_star': et, 'time_parts': tp, 'time_parts_closure_abs': abs(tp['total']-et), 'kkt_relative': kkt,
            'E_total_star': float(sum(Eo))+et+m.temporal.log_normalizer, 'gt_disp_median_moving_m_t': gt_disp}


def gate_f1_f2(m, Xgt, mv):
    X0 = m.x0[:, None].expand(m.N, m.K, 3); Xg = m.tt(Xgt).transpose(0, 1)
    disp = [float(np.median(np.linalg.norm(Xgt[k, mv]-Xgt[0, mv], axis=1))) if mv.any() else 0. for k in range(m.K)]
    rows = [candidate_energies(m, X0.clone(), 'S', 'stationary', 0., gt_disp=disp), candidate_energies(m, Xg.clone(), 'GT', 'gt_posthoc', 1., gt_disp=disp)]
    for a in ALPHAS: rows.append(candidate_energies(m, X0+a*(Xg-X0), f'I({a})', 'interpolation_posthoc', a, gt_disp=disp))
    S, G = rows[0], rows[1]
    dEo = [g-s for g, s in zip(G['E_obs_t'], S['E_obs_t'])]
    # attribution: feature residual e_t = mu_Q - mu_P, mu_P split into static / moving canonical contributions (exact, alpha = 1/N)
    attr = []
    for k, t in enumerate(m.terms):
        w_mov = torch.as_tensor(mv, device=m.dev).to(m.dt)/m.N; w_st = (1-torch.as_tensor(mv, device=m.dev).to(m.dt))/m.N
        for nm, X in (('S', X0[:, k]), ('GT', Xg[:, k])):
            e = t.mu_q-m.feats.mean(X, m.c0, w_st)-m.feats.mean(X, m.c0, w_mov)
            attr.append({'frame': k, 'candidate': nm, 'whitened_residual_sq': t.G.quad(e),
                         'moving_contribution_sq_norm': float((m.feats.mean(X, m.c0, w_mov)**2).sum())})
    return rows, {'dE_obs_GT_minus_S_t': dEo, 'dE_obs_GT_minus_S': float(sum(dEo)),
                  'dE_time_star_GT_minus_S': G['E_time_star']-S['E_time_star'], 'dE_total_star_GT_minus_S': G['E_total_star']-S['E_total_star'],
                  'frame0_dE_obs_exact_zero': dEo[0] == 0., 'every_frame_prefers_GT': all(x < 0 for x in dEo[1:]),
                  'alpha_path_E_obs': [(r['alpha'], r['E_obs']) for r in rows[2:]], 'alpha_path_E_total_star': [(r['alpha'], r['E_total_star']) for r in rows[2:]],
                  'feature_residual_attribution': attr}


def feature_sensitivity(m, Xgt, mv):
    """F1 supplemental: SAME positions under DIFFERENT, independently declared diagnostic observation models (no GT tuning; not production).
    B_X^1/2 ladder from GT-free quantities: KDE spacing s (frame 0), geometric mean sqrt(s h), production h; plus finite-D / seed checks."""
    from estimator.svgd_feature_measurement import RFFFeatures
    s = math.sqrt(m.scales[0][0]); h = math.sqrt(float(m.H_x[0, 0])); Hc = m.H_c.cpu()
    X0 = m.x0[:, None].expand(m.N, m.K, 3); Xg = m.tt(Xgt).transpose(0, 1); out = []
    models = [('production', h, m.feature_args['D'], m.feature_args['seed']), ('production_D4096', h, 4096, m.feature_args['seed']),
              ('production_seed+1', h, m.feature_args['D'], m.feature_args['seed']+1), ('B_geometric_mean', math.sqrt(s*h), m.feature_args['D'], m.feature_args['seed']),
              ('B_kde_spacing', s, m.feature_args['D'], m.feature_args['seed']), ('B_kde_spacing_D4096', s, 4096, m.feature_args['seed'])]
    for name, b, D, seed in models:
        f = RFFFeatures(torch.eye(3, dtype=torch.float64)*b*b, Hc, D, seed, m.dt, m.dev)
        try:
            terms = m.feature_terms(f)
            eS = [t.nll(X0[:, k]) for k, t in enumerate(terms)]; eG = [t.nll(Xg[:, k]) for k, t in enumerate(terms)]
            out.append({'model': name, 'label': 'PRODUCTION' if name == 'production' else 'DIFFERENT_OBSERVATION_MODEL (diagnostic only)', 'B_X_sqrt_m': b, 'D': D, 'seed': seed,
                        'rff_sha256': f.digest(), 'dE_obs_GT_minus_S_t': [g-s_ for g, s_ in zip(eG, eS)], 'dE_obs_GT_minus_S': float(sum(eG)-sum(eS))})
        except FloatingPointError as e:
            out.append({'model': name, 'B_X_sqrt_m': b, 'D': D, 'status': f'NUMERICAL_FAILURE {e}'})
    # exact-kernel MMD on a deterministic small subsample (diagnostic; separates finite-D error from the choice of B)
    from estimator.svgd_feature_measurement import exact_mmd2
    idx = np.unique(np.linspace(0, m.N-1, min(m.N, 1500)).round().astype(int)); mmd = []
    for k in range(1, m.K):
        cl = m.clouds[k]; j = np.unique(np.linspace(0, cl.M-1, min(cl.M, 1500)).round().astype(int))
        for b in (s, math.sqrt(s*h), h):
            Bx, Bc = np.eye(3)*b*b, Hc.numpy()
            mmd.append({'frame': k, 'B_X_sqrt_m': b, 'mmd2_S': exact_mmd2(m.canon.xyz0[idx], m.canon.rgb0[idx], cl.xyz[j], cl.rgb[j], Bx, Bc),
                        'mmd2_GT': exact_mmd2(Xgt[k, idx], m.canon.rgb0[idx], cl.xyz[j], cl.rgb[j], Bx, Bc)})
    return {'models': out, 'exact_kernel_mmd_subsample_1500': mmd, 'note': 'subsample MMD is an independent diagnostic, not the production likelihood'}


# ------------------------------------------------------------------ Gate F3: Stein field audit
def cosine(a, b, eps_rel=1e-9):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    tiny = (na <= eps_rel*max(na.max(), 1e-300)) | (nb <= eps_rel*max(nb.max(), 1e-300))
    c = (a*b).sum(-1)/np.maximum(na*nb, 1e-300); c[tiny] = np.nan
    return c, int(tiny.sum())


def fd_closure(f, analytic, scale):
    """Central differences of f along a direction for h = scale * 10^-k, k = 0..12 (truncation/roundoff plateau visible)."""
    rows = []
    for k in range(0, 13):
        h = scale*10.**-k; fd = (f(h)-f(-h))/(2*h)
        rows.append({'h': h, 'fd': fd, 'analytic': analytic, 'rel_error': abs(fd-analytic)/max(abs(analytic), 1e-300)})
    return {'rows': rows, 'best_rel_error': min(r['rel_error'] for r in rows)}


def gate_f3(m, Xgt, mv, saved):
    from estimator.svgd_particle_operator import stein_field
    from estimator.svgd_target_score import target_score, responsibilities
    Phi, score = saved['Phi'], saved['score']; kw = dict(attraction=bool(saved['attraction_enabled']), repulsion=bool(saved['repulsion_enabled']))
    G_obs = (m.obj.grad(m.S0)-m.temporal.grad(m.S0))[:, :, :3].cpu().numpy(); G_tot = m.obj.grad(m.S0).cpu().numpy()
    frames, csv_rows = [], []
    lab_obs = []
    for k in range(m.K):
        X = m.x0; g = m.tt(score[:, k])
        full = stein_field(X, m.c0, g, m.H_x, m.H_c, kernel_color=bool(saved['kernel_color']), **kw)
        parity = float(np.abs(full.total.cpu().numpy()-Phi[:, k]).max()/max(np.abs(Phi[:, k]).max(), 1e-300))
        nocol = stein_field(X, m.c0, g, m.H_x, m.H_c, kernel_color=False, **kw)
        parts = {nm: stein_field(X, m.c0, g, m.H_x, m.H_c, kernel_color=bool(saved['kernel_color']), source_mask=m.tt(msk.astype(float)), **kw)
                 for nm, msk in (('from_moving', mv), ('from_static', ~mv))}
        ph = Phi[:, k]; att, rep = full.attraction.cpu().numpy(), full.repulsion.cpu().numpy()
        inc = Xgt[k]-Xgt[k-1] if k else np.zeros_like(Xgt[0]); cum = Xgt[k]-Xgt[0]
        c_obs, u_obs = cosine(ph, -G_obs[:, k]); c_cum, u_cum = cosine(ph[mv], cum[mv]); c_inc, u_inc = cosine(ph[mv], inc[mv])
        # KDE basin: density and responsibility transfer at the stationary vs true positions (post hoc labels of observed samples)
        cl = m.clouds[k]; y, d = m.tt(cl.xyz), m.tt(cl.rgb); lam_x = m.tt(cl.cov)+m.scales[k][0]*torch.eye(3, dtype=m.dt, device=m.dev); lam_c = m.scales[k][1]
        from scipy.spatial import cKDTree
        nn = cKDTree(Xgt[k]).query(cl.xyz)[1]                         # exact NN, CPU (a dense M x N cdist does not fit 8 GB)
        lab = torch.as_tensor(mv[nn].astype(np.int64), device=m.dev); lab_obs.append(int(mv[nn].sum()))
        r_st = target_score(m.x0[torch.as_tensor(mv, device=m.dev)], m.c0[torch.as_tensor(mv, device=m.dev)], y, d, lam_x, lam_c, obs_camera=lab, cameras=2)
        r_gt = target_score(m.tt(Xgt[k, mv]), m.c0[torch.as_tensor(mv, device=m.dev)], y, d, lam_x, lam_c, obs_camera=lab, cameras=2)
        sub = np.flatnonzero(mv)[np.unique(np.linspace(0, mv.sum()-1, min(mv.sum(), 256)).round().astype(int))]
        R = responsibilities(m.x0[torch.as_tensor(sub, device=m.dev)], m.c0[torch.as_tensor(sub, device=m.dev)], y, d, lam_x, lam_c).cpu().numpy()
        peak = R.argmax(1); peak_to_true = np.linalg.norm(cl.xyz[peak]-Xgt[k, sub], axis=1)
        row = {'frame': k, 'phi_parity_rel': parity, 'gt_moving_observed_samples': lab_obs[-1],
               'norm_phi_moving': q(np.linalg.norm(ph[mv], axis=1)), 'norm_phi_static': q(np.linalg.norm(ph[~mv], axis=1)),
               'norm_grad_Eobs_moving': q(np.linalg.norm(G_obs[mv, k], axis=1)), 'norm_grad_Eobs_static': q(np.linalg.norm(G_obs[~mv, k], axis=1)),
               'norm_attraction': q(np.linalg.norm(att, axis=1)), 'norm_repulsion': q(np.linalg.norm(rep, axis=1)),
               'cos_phi_neg_grad_Eobs_moving': q(c_obs[mv]), 'cos_phi_neg_grad_Eobs_static': q(c_obs[~mv]), 'undefined_cosines_grad': u_obs,
               'cos_phi_cumulative_gt_moving': q(c_cum), 'cos_phi_incremental_gt_moving': q(c_inc), 'undefined_cosines_gt': u_cum,
               'kernel_mass_colour_on_over_off_median': float(np.median(full.kernel_mass.cpu().numpy()/nocol.kernel_mass.cpu().numpy())),
               'moving_mass_from_static': q((parts['from_static'].kernel_mass/full.kernel_mass).cpu().numpy()[mv]),
               'static_mass_from_moving': q((parts['from_moving'].kernel_mass/full.kernel_mass).cpu().numpy()[~mv]),
               'static_dominated_by_moving_fraction': float((parts['from_moving'].total.norm(dim=1) > parts['from_static'].total.norm(dim=1)).cpu().numpy()[~mv].mean()),
               'kde_logp_moving_at_stationary': q(r_st.log_density.cpu().numpy()), 'kde_logp_moving_at_gt': q(r_gt.log_density.cpu().numpy()),
               'kde_resp_on_moving_samples_at_stationary': q(r_st.camera_mass[:, 1].cpu().numpy()), 'kde_resp_on_moving_samples_at_gt': q(r_gt.camera_mass[:, 1].cpu().numpy()),
               'kde_effective_components_moving_stationary': q(r_st.effective_components.cpu().numpy()),
               'peak_responsibility_sample_distance_to_true_position_m': q(peak_to_true), 'peak_sample_is_moving_fraction': float(mv[nn][peak].mean())}
        frames.append(row)
        csv_rows.append({'frame': k, **{f'{a}_{b}': v[b] if isinstance(v, dict) and b in v else None for a, v in row.items() if isinstance(v, dict) for b in ('median', 'p10', 'p90')},
                         **{a: v for a, v in row.items() if not isinstance(v, dict)}})
    P = np.zeros((m.N, m.K, 6)); P[:, :, :3] = Phi; Pt = m.tt(P)
    an_obs = float((G_obs*Phi).sum()); an_tot = float((G_tot*P).sum())
    scale = float(saved['eta0'])
    fd_o = fd_closure(lambda h: m.obj.value(m.S0+h*Pt)['E_obs'], an_obs, scale); fd_t = fd_closure(lambda h: m.obj.value(m.S0+h*Pt)['total'], an_tot, scale)
    return frames, csv_rows, {'dot_grad_Eobs_Phi': an_obs, 'dot_grad_Etotal_P': an_tot, 'fd_E_obs': fd_o, 'fd_E_total': fd_t}


# ------------------------------------------------------------------ Gate F4: offline proposal arms on the identical objective
def gate_f4(m, Xgt, mv, saved):
    Phi = m.tt(saved['Phi']); E0 = m.E0; rows = []
    logged = [r for r in m.timeline if r['step'] == 'stein_accept']
    etas = [float(r['eta_m2']) for r in logged]; ext = [etas[0]*2.**-k for k in range(0, 61)]
    parity = []
    for arm in 'ABCD':
        for search, seq in (('original', etas), ('extended', ext if arm in 'CD' else [])):
            for b, eta in enumerate(seq):
                dX = eta*Phi
                if arm in 'CD': dX = dX.clone(); dX[:, 0] = 0
                S = m.S0.clone(); S[:, :, :3] += dX
                if arm in 'BD': S[:, :, 3:] += m.temporal.velocity_lift(m.S0, dX)
                E = m.obj.value(S); tp = time_parts(m.temporal, S)
                disp = torch.linalg.norm(dX[:, 1:], dim=-1).cpu().numpy() if m.K > 1 else np.zeros((m.N, 1))
                ok = E['total'] < E0['total']
                significant = E0['total']-E['total'] > 1e-12*abs(E0['total'])      # beyond float64 roundoff of the total (|E| ~ 1e5-1e6)
                row = {'arm': arm, 'search': search, 'backtrack': b, 'eta_m2': eta, 'dE_total': E['total']-E0['total'], 'dE_obs': E['E_obs']-E0['E_obs'],
                       'dE_anchor': tp['anchor'], 'dE_process': tp['process'], 'accepted': ok, 'accepted_significant': ok and significant, 'disp_median_m': float(np.median(disp)), 'disp_p90_m': float(np.quantile(disp, .9)),
                       'disp_moving_median_m': float(np.median(disp[mv])) if mv.any() else None, 'disp_static_median_m': float(np.median(disp[~mv]))}
                if ok and mv.any() and m.K > 1:
                    c, _ = cosine(dX[:, 1:].cpu().numpy()[mv].reshape(-1, 3), (Xgt[1:]-Xgt[:1]).transpose(1, 0, 2)[mv].reshape(-1, 3))
                    row['cos_step_gt_cumulative_moving_median'] = float(np.nanmedian(c))
                rows.append(row)
                if search == 'original' and arm in 'AB':
                    want = float(logged[b]['positions_only_dE_total' if arm == 'A' else 'temporal_lift_dE_total'])
                    parity.append(abs(row['dE_total']-want)/max(abs(want), 1e-300))
    # E: directional derivatives along each arm (linear part), FD verified
    deriv = {}
    for arm in 'ABCD':
        P = torch.zeros_like(m.S0); P[:, :, :3] = Phi
        if arm in 'CD': P[:, 0, :3] = 0
        if arm in 'BD': P[:, :, 3:] = m.temporal.velocity_lift(m.S0, P[:, :, :3])-m.temporal.velocity_lift(m.S0, torch.zeros_like(Phi))
        g = m.obj.grad(m.S0); an = float((g*P).sum())
        deriv[arm] = {'analytic_dE_total': an, 'analytic_dE_obs': float(((g-m.temporal.grad(m.S0))*P).sum()),
                      'fd_E_total': fd_closure(lambda h: m.obj.value(m.S0+h*P)['total'], an, float(saved['eta0']))['best_rel_error'],
                      'quadratic_E_time_coefficient': .5*float((P*m.temporal.hess_apply(P)).sum())}
    first = {arm: next((r for r in rows if r['arm'] == arm and r['accepted']), None) for arm in 'ABCD'}
    meaningful = {arm: next((r for r in rows if r['arm'] == arm and r['accepted_significant'] and deriv[arm]['analytic_dE_total'] < 0
                             and (r['disp_moving_median_m'] or 0) > 0), None) for arm in 'ABCD'}
    return rows, {'arm_definitions': {'A': 'dX_t = eta Phi_t all t, V unchanged (original)', 'B': 'A + exact conditional velocity lift (DEV-0011)',
                                      'C': 'dX_0 = 0, dX_t = eta Phi_t (t >= 1), V unchanged (offline counterfactual)', 'D': 'C + exact conditional velocity lift'},
                  'A_B_reconstruction_max_rel_error': max(parity) if parity else None, 'audit_valid': bool(parity) and max(parity) < 1e-9,
                  'directional_derivatives': deriv, 'first_accepted': first, 'first_meaningful_acceptance': meaningful, 'E0_total': E0['total'],
                  'meaningful_definition': 'decrease of E_total > 1e-12 |E_total| (beyond roundoff), along a descent direction (<grad E, P> < 0), with nonzero moving displacement'}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)); w = csv.DictWriter(open(path, 'w', newline=''), keys); w.writeheader()
    for r in rows: w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


PHYSICAL_STEP_FRACTION = 1e-3      # declared reporting criterion: an accepted step moves GT-moving points by >= 1e-3 of their final true displacement


def classify(f12, f4, f3, final_disp=None):
    """Spec section 12 decision table. An arm 'works' only with a meaningful acceptance (beyond roundoff, descent direction, nonzero
    motion) whose moving-point displacement is >= PHYSICAL_STEP_FRACTION of the final true displacement (physically meaningful)."""
    if f12['dE_obs_GT_minus_S'] >= 0: return 'OBSERVATION_DISCRIMINATION_FAILURE'
    if f12['dE_total_star_GT_minus_S'] >= 0: return 'MOTION_PRIOR_OR_RELATIVE_CALIBRATION_FAILURE'
    fd = final_disp if final_disp else 0.
    ok = {k: v is not None and (v['disp_moving_median_m'] or 0) >= PHYSICAL_STEP_FRACTION*fd for k, v in f4['first_meaningful_acceptance'].items()}
    if not ok['A'] and (ok['C'] or ok['D']): return 'FRAME0_PROPOSAL_INCONSISTENCY'
    if not (ok['C'] or ok['D']): return 'KDE_OR_FEATURE_DIRECTION_MISMATCH'
    return 'CONTINUATION_OR_SUPPORT_COUPLING_FAILURE_CANDIDATE (requires F5)'


def reclassify(audit_dir):
    """Recompute the classification from a saved audit (no re-evaluation); writes classification_final.json (new file)."""
    a = Path(audit_dir); s = json.loads((a/'falsification_summary.json').read_text())
    rows = list(csv.DictReader(open(a/'energy_oracle.csv'))); fd = max(float(r['gt_disp_median_moving_m']) for r in rows if r['candidate'] == 'GT' and r['frame'] != 'ALL')
    c = classify(s['gate_F1_F2'], s['gate_F4'], None, fd)
    out = {'classification': c, 'final_true_displacement_median_m': fd, 'physical_step_fraction': PHYSICAL_STEP_FRACTION, 'previous_classification': s['classification']}
    (a/'classification_final.json').write_text(json.dumps(out, indent=1)); return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--skip-sensitivity', action='store_true')
    ap.add_argument('--reclassify', type=Path, help='recompute the classification of a saved audit folder and exit')
    a = ap.parse_args()
    if a.reclassify: print(json.dumps(reclassify(a.reclassify), indent=1)); return 0
    if a.scene is None and not a.synthetic: ap.error('give --scene (Sapien GT) or --synthetic')
    torch.backends.cuda.matmul.allow_tf32 = False; t0 = time.time()
    out = a.run/'posthoc'/'svgd_falsification'; n = 1
    while out.exists(): out = a.run/'posthoc'/f'svgd_falsification_{n}'; n += 1
    out.mkdir(parents=True)
    man = json.loads((a.run/'run_manifest.json').read_text()); A = man['arguments']
    if a.synthetic:
        from tests.svgd_fixtures import SyntheticDataset
        spec = json.loads((a.run/'synthetic_fixture.json').read_text()); data = SyntheticDataset(**spec)
    else:
        from scene.dataset import Dataset
        data = Dataset(a.scene, A['frame_limit'], None, fps=A.get('fps'), dt_seconds=A.get('dt_seconds'), timestamps_file=A.get('timestamps_file'))
    m = Model(a.run, data, a.device)
    saved = dict(np.load(a.run/'svgd_support/stein_diagnostics.npz')); saved['eta0'] = m.status['gates']['STEIN_DIAGNOSTICS'][0]['eta0_m2']
    Xgt, mv, gtv = gt_oracle(m, a.scene, data if a.synthetic else None)
    cov = {'source_covariance_eigen_sigma_m': q(np.sqrt(np.linalg.eigvalsh(m.canon.cov0))[:, 0]) | {'axes': 'smallest'},
           'source_sigma_eigen_median_m': np.median(np.sqrt(np.linalg.eigvalsh(m.canon.cov0)), 0).tolist(),
           'depth_variance_m2': m.config.depth_variance, 'depth_squared_variance': m.config.depth_squared_variance, 'pixel_covariance_px2': m.config.pixel_covariance.tolist(),
           'rgb_covariance': np.asarray(m.config.color_covariance).tolist(), 'qc_m2_s3': np.asarray(m.config.qc).tolist(), 'velocity_covariance_m2_s2': np.asarray(m.config.velocity_covariance).tolist(),
           'Lambda_X_spacing_m': [math.sqrt(s) for s, _ in m.scales], 'Lambda_C': [lc.cpu().numpy().tolist() for _, lc in m.scales],
           'H_X_m2': float(m.H_x[0, 0]), 'H_C': float(m.H_c[0, 0]), 'rff': {**m.feature_args, 'B': 'H (DEV-0005)', 'sha256': m.feats.digest()},
           'gamma_logdet_condition': m.status['gates']['FEATURES']['gamma'], 'provenance': m.config.provenance,
           'statuses': {'depth / rgb noise': 'measured (scene/noise_calibration via temporal_sensor_only)', 'qc, velocity covariance': 'repository noise file (documented-initial-guess)',
                        'Lambda, H, B, D, Gamma': 'PROPOSED_NOT_AUTHORIZED (DEV-0002..0005)'},
           'quantization_check': None}
    nm = json.loads((a.run/'noise_calibration.json').read_text())
    qd = nm.get('temporal_sensor', {}).get('sensor', {}).get('depth_quantum_m')
    if qd: cov['quantization_check'] = {'depth_quantum_m': qd, 'q2_over_12': qd*qd/12, 'depth_variance_m2': m.config.depth_variance,
                                        'equal': abs(qd*qd/12-m.config.depth_variance) <= 1e-12*m.config.depth_variance}
    (out/'covariance_provenance.json').write_text(json.dumps(cov, indent=1, default=str))
    pre = {'run': str(a.run), 'scene': str(a.scene) if a.scene else 'synthetic', 'git': man['git'], 'audit_commit': __import__('subprocess').run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           'N': m.N, 'K': m.K, 'cameras': len(data.cameras), 'M_t': [c.M for c in m.clouds], 'per_camera_counts_frame0': m.clouds[0].counts,
           'timestamps_s': [float(t) for t in data.times], 'reconstruction_checks': m.checks, 'reconstruction_valid': m.valid,
           'source_changes_since_run': m.source_changes_since_run, 'gt_verification': gtv,
           'in_sample_note': 'frame-0 canonical = frame-0 observed samples: E_obs,0 at X0 is in-sample (conditioned) evidence; frame-0 comparisons carry no motion information'}
    (out/'preflight.json').write_text(json.dumps(pre, indent=1, default=str))
    print(f"[F0] reconstruction {'VALID' if m.valid else 'INVALID'}; GT {gtv['status']} ({gtv['moving_members']} moving)", flush=True)
    if not m.valid or gtv['status'] != 'GT_TRAJECTORY_VERIFIED':
        (out/'diagnosis.md').write_text(f"# AUDIT_INVALID or GT_TRAJECTORY_UNVERIFIED\n\nSee preflight.json: {json.dumps([c for c in m.checks if not c['ok']], default=str)} / {gtv['status']}\n")
        return 2
    rows, f12 = gate_f1_f2(m, Xgt, mv)
    eo = []
    for r in rows:
        for k in range(m.K):
            iv = r['time_parts']['intervals'][k-1] if k else None
            eo.append({'candidate': r['candidate'], 'kind': r['kind'], 'alpha': r['alpha'], 'frame': k, 'E_obs_t': r['E_obs_t'][k], 'E_obs_quadratic_t': r['E_obs_quadratic_t'][k],
                       'gt_disp_median_moving_m': r['gt_disp_median_moving_m_t'][k], 'anchor_xx': r['time_parts']['anchor_xx'] if k == 0 else None,
                       'anchor_vv': r['time_parts']['anchor_vv'] if k == 0 else None, 'process_xx': iv['xx'] if iv else None, 'process_xv': iv['xv'] if iv else None,
                       'process_vv': iv['vv'] if iv else None, 'config_sha256': man['config_sha256'], 'rff_sha256': m.feats.digest()})
        eo.append({'candidate': r['candidate'], 'kind': r['kind'], 'alpha': r['alpha'], 'frame': 'ALL', 'E_obs': r['E_obs'], 'E_time_star': r['E_time_star'],
                   'anchor': r['time_parts']['anchor'], 'process': r['time_parts']['process'], 'E_total_star': r['E_total_star'], 'kkt_relative': r['kkt_relative'],
                   'time_parts_closure_abs': r['time_parts_closure_abs'], 'config_sha256': man['config_sha256'], 'rff_sha256': m.feats.digest()})
    write_csv(out/'energy_oracle.csv', eo)
    print(f"[F1] dE_obs(GT - S) = {f12['dE_obs_GT_minus_S']:+.6e}  per frame {[f'{x:+.3e}' for x in f12['dE_obs_GT_minus_S_t']]}", flush=True)
    print(f"[F2] dE_time*(GT - S) = {f12['dE_time_star_GT_minus_S']:+.6e}  dE_total*(GT - S) = {f12['dE_total_star_GT_minus_S']:+.6e}", flush=True)
    sens = None if a.skip_sensitivity else feature_sensitivity(m, Xgt, mv)
    f3rows, f3csv, f3fd = gate_f3(m, Xgt, mv, saved)
    write_csv(out/'stein_field_audit.csv', f3csv)
    (out/'stein_field_audit.json').write_text(json.dumps({'per_frame': f3rows, 'fd_closure': f3fd}, indent=1, default=float))
    f4rows, f4 = gate_f4(m, Xgt, mv, saved)
    write_csv(out/'line_search_counterfactual.csv', f4rows)
    arms = {}
    for arm in 'ACD':                                           # display-only counterfactual positions at each arm's first meaningful step
        r = f4['first_meaningful_acceptance'][arm]
        if r is None: continue
        dX = r['eta_m2']*torch.as_tensor(saved['Phi'])
        if arm in 'CD': dX[:, 0] = 0
        arms[f'X_arm_{arm}'] = (m.canon.xyz0[:, None]+dX.numpy()).transpose(1, 0, 2); arms[f'eta_arm_{arm}'] = np.array(r['eta_m2'])
    np.savez_compressed(out/'oracle_trajectories.npz', X0=m.canon.xyz0, X_gt_POSTHOC=Xgt, moving_POSTHOC=mv, Phi=saved['Phi'], **arms,
                        semantics=np.array('POST-HOC ORACLE / OFFLINE COUNTERFACTUAL: display and audit only, never inference'))
    cls = classify(f12, f4, f3rows, max(rows[1]['gt_disp_median_moving_m_t']))
    summary = {'classification': cls, 'gate_F1_F2': {k: v for k, v in f12.items() if k != 'feature_residual_attribution'}, 'feature_residual_attribution': f12['feature_residual_attribution'],
               'gate_F1_sensitivity': sens, 'gate_F3_fd': f3fd, 'gate_F4': f4, 'gate_F5': 'SKIPPED (runs only if F1 and F2 prefer GT and F4 accepts physically meaningful steps)',
               'seconds': time.time()-t0}
    (out/'falsification_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    (out/'diagnosis.md').write_text(diagnosis_md(pre, cov, f12, sens, f3rows, f3fd, f4, cls, rows))
    (out/'theory_deviation_audit.md').write_text(deviation_md())
    print(f"[F3] FD closure E_obs {f3fd['fd_E_obs']['best_rel_error']:.2e}  E_total {f3fd['fd_E_total']['best_rel_error']:.2e}", flush=True)
    print(f"[F4] A/B reconstruction {f4['A_B_reconstruction_max_rel_error']:.2e}; first accepted: " + ', '.join(f"{k}: {'none' if v is None else 'eta %.2e' % v['eta_m2']}" for k, v in f4['first_accepted'].items()), flush=True)
    print(f'[result] {cls}  -> {out}', flush=True)
    return 0


def deviation_md():
    from estimator.svgd_contract import DEVIATIONS
    L = ['# Theory-deviation audit of the falsification study', '', 'Existing register entries (none authorized):', '']
    L += [f"- {d['id']} {d['classification']} — {d['status']} — {d['conclusion']}" for d in DEVIATIONS]
    L += ['', 'Counterfactuals evaluated OFFLINE in this audit (never in inference, never authorized):', '',
          '- Arm C/D: frame-0 position change removed from the proposal (would be DEV-0012 if ever proposed; NOT implemented).',
          '- I(alpha) and GT trajectories: post-hoc oracle paths through position space (GT only here).',
          '- Diagnostic feature models (B ladder, D, seed): DIFFERENT OBSERVATION MODELS, labelled as such; not candidates for production without a theory amendment.', '']
    return '\n'.join(L)


def fmt(x, p=3): return 'n/a' if x is None else f'{x:+.{p}e}' if isinstance(x, float) else str(x)


def diagnosis_md(pre, cov, f12, sens, f3, f3fd, f4, cls, rows):
    L = [f'# SVGD Stage-1 falsification diagnosis — {cls}', '', f"Run `{pre['run']}` (N = {pre['N']}, K = {pre['K']}, {pre['cameras']} cameras, M_t {min(pre['M_t'])}..{max(pre['M_t'])}).",
         f"Reconstruction: {'VALID' if pre['reconstruction_valid'] else 'INVALID'} ({len(pre['reconstruction_checks'])} checks). GT: {pre['gt_verification']['status']}, "
         f"{pre['gt_verification']['moving_members']} moving members (post hoc only).", '',
         '## F1 — does the unchanged E_obs prefer the true motion?', '', '| frame | GT disp (mm) | ΔE_obs,t (GT − S) |', '|---|---|---|']
    disp = rows[1]['gt_disp_median_moving_m_t']
    L += [f'| {k} | {disp[k]*1e3:.2f} | {fmt(v)} |' for k, v in enumerate(f12['dE_obs_GT_minus_S_t'])]
    L += ['', f"Sum ΔE_obs = {fmt(f12['dE_obs_GT_minus_S'])}; frame-0 difference exactly zero: {f12['frame0_dE_obs_exact_zero']}; every frame t ≥ 1 prefers GT: {f12['every_frame_prefers_GT']}.", '',
          '| α | E_obs − E_obs(S) | E_total* − E_total*(S) |', '|---|---|---|']
    s_o, s_t = rows[0]['E_obs'], rows[0]['E_total_star']
    L += [f"| {r['alpha']} | {fmt(r['E_obs']-s_o)} | {fmt(r['E_total_star']-s_t)} |" for r in rows[2:]]
    if sens:
        L += ['', 'Diagnostic observation models (same positions; NOT production):', '', '| model | B_X^½ (mm) | D | ΔE_obs (GT − S) |', '|---|---|---|---|']
        L += [f"| {s['model']} | {s['B_X_sqrt_m']*1e3:.2f} | {s['D']} | {fmt(s.get('dE_obs_GT_minus_S'))} |" for s in sens['models']]
    G, S = rows[1]['time_parts'], rows[0]['time_parts']
    L += ['', '## F2 — full objective with velocities minimized exactly', '',
          f"ΔE_time* (GT − S) = {fmt(f12['dE_time_star_GT_minus_S'])}: anchor {fmt(G['anchor']-S['anchor'])} (xx {fmt(G['anchor_xx'])}, vv {fmt(G['anchor_vv'])}), process {fmt(G['process']-S['process'])}.",
          f"ΔE_total* (GT − S) = {fmt(f12['dE_total_star_GT_minus_S'])}. KKT relative residual of the lift: GT {rows[1]['kkt_relative']:.1e}.", '',
          '## F3 — Stein field vs likelihood gradient vs motion', '', f"FD closure (best relative error over h ladder): ⟨∇E_obs, Φ⟩ {f3fd['fd_E_obs']['best_rel_error']:.1e}, ⟨∇E_total, P⟩ {f3fd['fd_E_total']['best_rel_error']:.1e}.", '',
          '| frame | cos(Φ,−∇E_obs) moving | static | cos(Φ, GT cum.) moving | cos(Φ, GT incr.) | KDE log p moving @S → @GT | resp. on moving samples @S | peak sample → true pos (mm) | static dominated by moving |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in f3:
        g = lambda key: r[key].get('median', float('nan')) if r[key].get('n') else float('nan')
        L.append(f"| {r['frame']} | {g('cos_phi_neg_grad_Eobs_moving'):.3f} | {g('cos_phi_neg_grad_Eobs_static'):.3f} | {g('cos_phi_cumulative_gt_moving'):.3f} | {g('cos_phi_incremental_gt_moving'):.3f} | "
                 f"{g('kde_logp_moving_at_stationary'):.1f} → {g('kde_logp_moving_at_gt'):.1f} | {g('kde_resp_on_moving_samples_at_stationary'):.3f} | "
                 f"{g('peak_responsibility_sample_distance_to_true_position_m')*1e3:.2f} | {r['static_dominated_by_moving_fraction']:.3f} |")
    L += ['', '## F4 — offline proposal arms on the identical objective', '', f"A/B reconstruction of the logged line search: max relative error {fmt(f4['A_B_reconstruction_max_rel_error'])} (valid: {f4['audit_valid']}).", '',
          '| arm | ⟨∇E,P⟩ | ⟨∇E_obs,P⟩ | ½PᵀL P (E_time curvature) | FD rel. err. | first accepted η (m²) | ΔE_obs / ΔE_anchor / ΔE_process there | moving disp. median (mm) | cos(step, GT) |', '|---|---|---|---|---|---|---|---|---|']
    for k, d in f4['directional_derivatives'].items():
        r = f4['first_accepted'][k]
        L.append(f"| {k} | {fmt(d['analytic_dE_total'])} | {fmt(d['analytic_dE_obs'])} | {fmt(d['quadratic_E_time_coefficient'])} | {d['fd_E_total']:.1e} | "
                 + (f"{r['eta_m2']:.2e} | {fmt(r['dE_obs'])} / {fmt(r['dE_anchor'])} / {fmt(r['dE_process'])} | {(r['disp_moving_median_m'] or 0)*1e3:.4f} | {fmt(r.get('cos_step_gt_cumulative_moving_median'))} |"
                    if r else 'none | — | — | — |'))
    L += ['', f'## Classification: `{cls}`', '', 'See falsification_summary.json, energy_oracle.csv, line_search_counterfactual.csv, stein_field_audit.{csv,json}, covariance_provenance.json.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    sys.exit(main())
