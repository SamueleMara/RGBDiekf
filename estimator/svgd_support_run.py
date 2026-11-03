"""Opt-in runner of the SVGD-assisted moving-support Stage 1 (spec 8.4, 10, 14). Never called by train.py's default path.

Stages INIT -> TARGET -> STEIN_PROPOSAL -> TEMPORAL_REFINEMENT -> RIGID_GRAPH -> MODEL_EVIDENCE -> JOINT_REFINEMENT ->
SEED_SELECTION. Before allocating anything a stage needs, its parameters are validated against estimator/svgd_contract.py
(strict: every statistical law declared and no unauthorized deviation; experimental: PROPOSED deviations accepted and
the results quarantined as EXPERIMENTAL_DEVIATION, never published as probabilities). The first unusable stage stops
the run with BLOCKED_MODEL_UNSPECIFIED / BLOCKED_THEORY_CONFLICT and the reasons; nothing downstream is fabricated.

Accepted-step objective (contract E32, DEV-0008 conditional on frozen support V_t = all):
    E(S) = sum_t E_F,t(X_t) + E_time(S) + log_normalizer
Stein proposal X <- X + eta Phi (all frames at once, Phi from (14) with the frame's own target score) is ACCEPTED only
if E decreases (eta halved otherwise; rejected proposals are logged). Temporal refinement: one Gauss-Newton step on E,
(J^T Gamma^-1 J + L) delta = -grad E by matrix-free PCG preconditioned with the exact L^-1, backtracked on the same E.
The Stein field is never assimilated as a measurement; the score never enters E.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from estimator import svgd_contract as SC
from estimator.svgd_target_score import target_score, SCORE_STATUS
from estimator.svgd_particle_operator import stein_field, initial_step
from estimator.svgd_feature_measurement import RFFFeatures, FeatureGaussian, iid_sampling_gamma, FrameFeatureTerm
from estimator.svgd_temporal import TemporalPrior, pcg
from estimator.point_kalman_cuda import jacobi_eigvalsh      # cuSOLVER batched syev needs ~0.5 MB workspace per matrix
from estimator.svgd_rigidity import knn_graph, edge_residuals, edge_covariance_structured, rigid_edge_loglik, differential_rigidity, prior_position_marginals
from estimator.svgd_visibility import ray_visibility, POINT_CODES, CAM_CODES
from estimator.svgd_support_posterior import support_posterior, PROBABILITY_STATUS
from scene.pointcloud_sequence import build_masked_cloud, canonical_from_frame, increment_noise_study
from utils.svgd_support_output import Ledger, sha256_file

REPO = Path(__file__).resolve().parents[1]
SOURCES = ['estimator/svgd_contract.py', 'estimator/svgd_target_score.py', 'estimator/svgd_particle_operator.py', 'estimator/svgd_feature_measurement.py',
           'estimator/svgd_temporal.py', 'estimator/svgd_rigidity.py', 'estimator/svgd_visibility.py', 'estimator/svgd_support_posterior.py',
           'estimator/svgd_support_run.py', 'scene/pointcloud_sequence.py', 'scene/endpoint.py', 'scene/dataset.py', 'scene/dataloader.py',
           'scene/timing.py', 'scene/config.py', 'scene/noise_calibration.py', 'estimator/temporal_noise.py', 'estimator/point_kalman_cuda.py',
           'utils/svgd_support_output.py', 'debug/svgd_support_debug.py', 'tools/run_svgd_support.py']
EXIT = {'OK': 0, 'BLOCKED': 3, 'CUDA_OOM': 4, 'FAILED': 5}


class Blocked(Exception):
    def __init__(self, status, stage, reasons): super().__init__(f'{status} at {stage}: {reasons}'); self.status, self.stage, self.reasons = status, stage, reasons


def value(cfg, key): return SC.lookup(cfg, key)['value']


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'status_short': run('status', '--short').splitlines()}


def source_hashes():
    return {s: sha256_file(REPO/s) for s in SOURCES if (REPO/s).is_file()}


def config_problems(cfg, mode, stage):
    probs = SC.parameter_problems(cfg, SC.STAGE_PARAMETERS[stage], mode)
    if stage == 'TARGET' and SC.open_conflicts(cfg, mode): probs.append(('reference.color_reference_uncertainty', 'BLOCKED_THEORY_CONFLICT TC-0001 unresolved'))
    return probs


# ---------------------------------------------------------------- declared representation scales (resolved from data)
def _nn(points, queries_idx, chunk=512):
    """Exact nearest OTHER sample of the query subset among all points: (distance [q], neighbour index [q])."""
    d_out, i_out = [], []
    for i in range(0, len(queries_idx), chunk):
        q = queries_idx[i:i+chunk]; D = torch.cdist(points[q], points, compute_mode='donot_use_mm_for_euclid_dist')
        D[torch.arange(len(q)), q] = float('inf'); v, j = D.min(1); d_out.append(v); i_out.append(j)
    return torch.cat(d_out), torch.cat(i_out)


def even_subset(n, k):
    return torch.unique(torch.linspace(0, n-1, min(n, k)).round().long())


def source_order(source, camera_names):
    """Order of samples by (camera NAME, pixel): independent of the camera loading order and of point order (TF.6 item 5)."""
    rank = np.argsort(np.argsort(np.asarray(camera_names)))
    return torch.as_tensor(np.lexsort((source[:, 2], rank[source[:, 1]])))


def representation_scales(y, d, R_c, spec, order, k=4096):
    """Lambda_X_j = Sigma_j + s^2 I, s = median exact nearest-sample distance; Lambda_C = R_C + diag(1/2 median (d_j - d_nn)^2)
    (method 'measured_noise_plus_nn_sampling_scale', DEV-0003). Query subset: evenly spaced sample indices (statistic
    estimation only, DEV-0009); every component still enters the density."""
    if spec.get('method') != 'measured_noise_plus_nn_sampling_scale': raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'TARGET', [('Lambda method', spec)])
    q = order[even_subset(len(y), k)].to(y.device); dist, j = _nn(y, q)
    s2 = float(torch.median(dist)**2); dc2 = torch.median((d[q]-d[j])**2, 0).values
    return s2, torch.as_tensor(R_c, dtype=y.dtype, device=y.device)+torch.diag(.5*dc2)


def median_heuristic(x, order, k=4096):
    """Liu & Wang reference: h = med^2 / log N for exp(-|d|^2/h)  ->  H = med^2/(2 log N) I (per block; DEV-0004)."""
    q = order[even_subset(len(x), k)].to(x.device); D = torch.cdist(x[q], x[q], compute_mode='donot_use_mm_for_euclid_dist'); iu = torch.triu_indices(len(q), len(q), 1, device=x.device)
    med = float(torch.median(D[iu[0], iu[1]]))
    return med*med/(2*math.log(max(len(x), 2))), med


# ---------------------------------------------------------------- the common accepted-step objective
class SequenceObjective:
    def __init__(self, terms, temporal):
        self.terms, self.temporal = terms, temporal

    def value(self, S):
        obs = [t.nll(S[:, k, :3]) for k, t in enumerate(self.terms)]; et = self.temporal.energy(S)
        return {'E_obs': float(sum(obs)), 'E_time': et, 'E_obs_per_frame': obs, 'total': float(sum(obs))+et+self.temporal.log_normalizer}

    def grad(self, S):
        g = self.temporal.grad(S)
        for k, t in enumerate(self.terms): g[:, k, :3] += t.grad(S[:, k, :3])
        return g

    def gn_apply(self, S, U):
        out = self.temporal.hess_apply(U)
        for k, t in enumerate(self.terms): out[:, k, :3] += t.gn_apply(S[:, k, :3], U[:, k, :3])
        return out


def quantiles(v):
    v = v[torch.isfinite(v)]
    if not len(v): return None
    q = torch.quantile(v.double(), torch.tensor([0, .5, .9, .99, 1.], dtype=torch.float64, device=v.device)) if len(v) < 2**24 else None
    return [float(a) for a in q] if q is not None else [float(v.min()), float(v.median()), None, None, float(v.max())]


# ---------------------------------------------------------------- run
def run(a, data=None):
    """a: argparse namespace from tools/run_svgd_support.py; data: an already constructed dataset (tests). Returns the exit code."""
    from scene.dataset import Dataset
    from scene.config import load_config
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(Path(a.config).read_text()); mode = a.svgd_theory_fidelity
    contract = SC.model_contract(cfg, mode)
    if a.svgd_kernel_backend != 'torch': raise SystemExit('NOT_IMPLEMENTED: --svgd-kernel-backend triton has no verified kernels; only torch is implemented')
    dev, dt = torch.device(a.device), torch.float64
    if dev.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable (no silent CPU fallback)')
    if data is None: data = Dataset(a.scene, a.frame_limit, None, fps=a.fps, dt_seconds=a.dt_seconds, timestamps_file=a.timestamps_file)
    K, C = len(data.times), len(data.cameras)
    prefix = a.frame_limit is not None and a.frame_limit < len(data.base.times)
    first_block = next((st for st in SC.STAGES if config_problems(cfg, mode, st)), None)
    if a.dry_run:
        counts = [c['foreground_valid'] for c in build_masked_cloud(data, 0, load_config()).counts]
        print(json.dumps({'scene': str(a.scene), 'frames': K, 'cameras': C, 'frame0_foreground_valid_per_camera': counts, 'mode': mode,
                          'times_s': [float(t) for t in data.times[:5]]+(['...'] if K > 5 else []), 'first_blocked_stage': first_block,
                          'blocked_reasons': config_problems(cfg, mode, first_block) if first_block else [], 'prefix_diagnostic': prefix,
                          'open_theory_conflicts': contract['open_conflicts']}, indent=1))
        return EXIT['OK']
    out = Path(a.output); sv = out/'svgd_support'
    if out.exists() and any(out.iterdir()): raise SystemExit(f'output {out} exists and is not empty (immutable run directories)')
    if REPO in out.resolve().parents: raise SystemExit('output must be outside the repository tree')
    sv.mkdir(parents=True); L = Ledger(out)
    t_start = time.time(); perf = {'device': str(dev), 'device_name': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu', 'operators': {}}
    if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
    def timed(name, fn, *args, **kw):
        if dev.type == 'cuda': torch.cuda.synchronize(dev)
        t0 = time.perf_counter(); r = fn(*args, **kw)
        if dev.type == 'cuda': torch.cuda.synchronize(dev)
        o = perf['operators'].setdefault(name, {'calls': 0, 'seconds': 0.}); o['calls'] += 1; o['seconds'] += time.perf_counter()-t0
        return r
    progress = lambda **kw: L.json(out/'progress.json', {**kw, 'elapsed_s': time.time()-t_start})
    manifest = {'run_kind': 'svgd_support_stage1', 'transport_label': SC.TRANSPORT_LABEL, 'git': git_state(), 'source_sha256': source_hashes(),
                'scene': str(Path(a.scene).resolve()), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                'config_path': str(a.config), 'config_sha256': hashlib.sha256(Path(a.config).read_bytes()).hexdigest(), 'mode': mode,
                'dataset': data.manifest(), 'hardware': perf['device_name'], 'torch': torch.__version__, 'prefix_diagnostic': prefix,
                'ground_truth_used': False}
    parity = REPO/'reports/svgd/theory_parity_summary.json'
    manifest['theory_parity_summary'] = {'path': str(parity), 'sha256': sha256_file(parity), 'content': json.loads(parity.read_text())} if parity.is_file() else 'NOT_GENERATED'
    L.json(out/'run_manifest.json', manifest); L.json(sv/'model_contract.json', contract)
    status = {'stage_reached': None, 'gates': {}, 'blocked': None, 'probability_semantics': 'PROBABILITY_NOT_YET_ESTABLISHED',
              'theory_fidelity': 'EXPERIMENTAL_DEVIATION_NOT_AUTHORIZED' if mode == 'experimental' else 'STRICT',
              'sequence_scope': 'PREFIX_DIAGNOSTIC' if prefix else 'WHOLE_SEQUENCE', 'not_suitable_for_joint_recovery': True}
    timeline, traces = [], {}
    try:
        code = _stages(a, cfg, mode, contract, data, dev, dt, out, sv, L, timed, progress, status, timeline, traces, perf)
    except Blocked as b:
        status['blocked'] = {'status': b.status, 'stage': b.stage, 'reasons': [{'parameter': p, 'reason': r} for p, r in b.reasons]}
        print(f'[svgd] {b.status} at {b.stage}:', *[f'\n   {p}: {r}' for p, r in b.reasons], flush=True); code = EXIT['BLOCKED']
    except torch.cuda.OutOfMemoryError as e:
        status['blocked'] = {'status': 'CUDA_OOM', 'stage': status['stage_reached'], 'reasons': [{'parameter': 'compute', 'reason': str(e)[:400]}],
                             'memory': {'allocated': torch.cuda.memory_allocated(dev), 'reserved': torch.cuda.memory_reserved(dev), 'peak': torch.cuda.max_memory_allocated(dev)}}
        print('[svgd] CUDA_OOM: change --svgd-workspace-mib / chunk sizes explicitly; no subsampling is applied', flush=True); code = EXIT['CUDA_OOM']
    except (FloatingPointError, ValueError) as e:          # named failure journal; never a model swap or added jitter
        kind = 'NUMERICAL_FAILURE' if isinstance(e, FloatingPointError) else 'FAILED'
        status['blocked'] = {'status': kind, 'stage': status['stage_reached'], 'reasons': [{'parameter': type(e).__name__, 'reason': str(e)[:600]}]}
        print(f'[svgd] {kind} at {status["stage_reached"]}: {e}', flush=True); code = EXIT['FAILED']
    perf['wall_seconds'] = time.time()-t_start
    if dev.type == 'cuda': perf.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(dev), peak_reserved_bytes=torch.cuda.max_memory_reserved(dev))
    L.json(sv/'performance.json', perf); L.csv(sv/'objective_timeline.csv', timeline)
    (sv/'trace_points').mkdir(exist_ok=True)
    for pid, tr in traces.items(): L.json(sv/'trace_points'/f'{pid}.json', tr)
    L.json(sv/'status.json', status)
    res = 'FAILED' if code in (EXIT['CUDA_OOM'], EXIT['FAILED']) else 'UNRESOLVED'
    L.json(out/'result.json', {'status': res, 'stage_reached': status['stage_reached'], 'blocked': status['blocked'], 'probability_semantics': status['probability_semantics'],
                               'sequence_scope': status['sequence_scope'], 'note': 'UNRESOLVED: no normalized support posterior is published by this stage'})
    if a.debug != 'none' and (sv/'canonical.npz').is_file():
        try:
            from debug.svgd_support_debug import write_debug
            write_debug(out, a.debug, data, display_cap=a.display_cap)
        except Exception as e:                      # the numerical run stays valid; the viewer failure is recorded, not hidden
            status['debug'] = f'DEBUG_NOT_WRITTEN: {type(e).__name__}: {e}'; L.json(sv/'status.json', status); print('[svgd]', status['debug'], flush=True)
    L.close(sv/'artifact_ledger.json'); progress(stage='DONE', exit_code=code)
    return code


def _stages(a, cfg, mode, contract, data, dev, dt, out, sv, L, timed, progress, status, timeline, traces, perf):
    from scene.config import load_config
    from estimator.temporal_noise import temporal_sensor_only
    K, C = len(data.times), len(data.cameras); stop = a.stop_after_step
    # ------------------------------------------------ INIT
    status['stage_reached'] = 'INIT'; config = load_config(a.noise_config)
    noise = {'sensor_noise': a.sensor_noise}
    if a.sensor_noise == 'measured':
        config, _, calib = temporal_sensor_only(data, config, {}); noise['temporal_sensor'] = calib
    noise['increment_study'] = increment_noise_study(data); noise['config_provenance'] = config.provenance
    L.json(out/'noise_calibration.json', noise)
    clouds, counts_rows = [], []
    for k in range(K):
        progress(stage='INIT', frame=k+1, frames=K)
        cl = build_masked_cloud(data, k, config); clouds.append(cl)
        for c in cl.counts: counts_rows.append({'frame': k, 'frame_id': cl.frame_id, 'time_s': cl.time_s, **c})
    canon = canonical_from_frame(clouds[0]); N = canon.N
    L.npz(sv/'canonical.npz', xyz0=canon.xyz0, rgb0=canon.rgb0, cov0=canon.cov0, original_id=canon.original_id, source=canon.source,
          selected_ids_sha256=np.array(canon.ids_sha256()), rgb_units=np.array('encoded RGB [0,1] as loaded'), xyz_units=np.array('world m'))
    L.json(sv/'completed_frames.json', [{'frame': c.k, 'frame_id': c.frame_id, 'time_s': c.time_s, 'M': c.M, 'cloud_sha256': c.digest(),
                                         'status': 'LOADED' if c.M else 'NO_OBSERVATION'} for c in clouds])
    status['gates']['INIT'] = {'N': N, 'frames': K, 'cameras': C, 'M_t': [c.M for c in clouds]}
    print(f'[svgd] INIT N={N} canonical points, {K} frames x {C} cameras, M_t {min(c.M for c in clouds)}..{max(c.M for c in clouds)}', flush=True)
    tt = lambda v: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)
    x0, c0, cov0 = tt(canon.xyz0), tt(canon.rgb0), tt(canon.cov0)
    S = torch.zeros(N, K, 6, dtype=dt, device=dev); S[:, :, :3] = x0[:, None]               # stationary initialization, v = 0 prior mean
    trace_ids = sorted(set(a.debug_point_ids)) if a.debug_point_ids else even_subset(N, 8).tolist()
    trace_ids = [i for i in trace_ids if 0 <= i < N]
    for i in trace_ids: traces[i] = {'original_id': i, 'source': canon.source[i].tolist(), 'x0': canon.xyz0[i].tolist(), 'rgb0': canon.rgb0[i].tolist(),
                                     'cov0': canon.cov0[i].tolist(), 'semantics': 'INTERNAL_NOT_FINAL_TRACKS', 'iterations': []}
    def finish_counts(extra=None):
        rows = [dict(r, **(extra.get((r['frame'], r['camera']), {}) if extra else {})) for r in counts_rows]
        L.csv(sv/'observed_counts.csv', rows)
    finish_counts()
    if stop == 'INIT': status['stopped_after'] = 'INIT'; return EXIT['OK']
    # ------------------------------------------------ TARGET
    probs = config_problems(cfg, mode, 'TARGET')
    if probs: raise Blocked('BLOCKED_THEORY_CONFLICT' if any('CONFLICT' in r for _, r in probs) else 'BLOCKED_MODEL_UNSPECIFIED', 'TARGET', probs)
    status['stage_reached'] = 'TARGET'
    R_c = np.asarray(config.color_covariance, float)
    ref = value(cfg, 'reference.color_reference_uncertainty')
    Sigma_a = None if ref == 'fixed_reference_pdf' else tt(R_c) if ref == 'marginalize_measured_rgb_noise' else None
    if ref not in ('fixed_reference_pdf', 'marginalize_measured_rgb_noise'): raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'TARGET', [('reference.color_reference_uncertainty', ref)])
    if value(cfg, 'target_density.source_weights') != 'uniform_1_over_M': raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'TARGET', [('source_weights', 'only uniform_1_over_M (PDF (8)) is implemented')])
    obs = []; names = [c.name for c in data.cameras]
    for cl in clouds:
        y, d = tt(cl.xyz), tt(cl.rgb)
        if cl.M == 0: obs.append(None); continue
        s2, lam_c = representation_scales(y, d, R_c, value(cfg, 'target_density.Lambda_X'), source_order(cl.source, names))
        obs.append({'s2': s2, 'lam_c': lam_c, 'M': cl.M, 'lam_x_max': float(jacobi_eigvalsh(tt(cl.cov))[:, -1].max())+s2})
    scale = {'Lambda_X_spacing_m': [math.sqrt(o['s2']) if o else None for o in obs], 'Lambda_C': [o['lam_c'].tolist() if o else None for o in obs]}
    def frame_tensors(k):
        cl = clouds[k]; y, d = tt(cl.xyz), tt(cl.rgb)
        lam_x = tt(cl.cov)+obs[k]['s2']*torch.eye(3, dtype=dt, device=dev)
        return y, d, lam_x, obs[k]['lam_c'], torch.as_tensor(cl.camera, device=dev)
    ws = int(a.svgd_workspace_mib*2**20)
    def score(k, X):
        if obs[k] is None: return None
        y, d, lx, lc, cam = frame_tensors(k)
        return timed('target_score', target_score, X, c0, y, d, lx, lc, query_color_cov=Sigma_a, workspace_bytes=ws)
    score_rows = []
    for k in range(K):
        progress(stage='TARGET', frame=k+1, frames=K)
        r = score(k, S[:, k, :3])
        score_rows.append({'frame': k, 'status': 'NO_OBSERVATION' if r is None else 'VALID',
                           'score_norm_q': quantiles(torch.linalg.norm(r.grad_xyz, dim=1)) if r is not None else None,
                           'log_density_q': quantiles(r.log_density) if r is not None else None,
                           'effective_components_q': quantiles(r.effective_components) if r is not None else None})
    status['gates']['TARGET'] = {'representation_scales': scale, 'initial_scores': score_rows, 'exact_all_components': True, 'tail_bound': 0.}
    if stop == 'TARGET': status['stopped_after'] = 'TARGET'; return EXIT['OK']
    # ------------------------------------------------ STEIN_PROPOSAL + TEMPORAL_REFINEMENT (one common objective)
    probs = config_problems(cfg, mode, 'STEIN_PROPOSAL')
    if probs: raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'STEIN_PROPOSAL', probs)
    st_cfg = cfg['stein']; num = value(cfg, 'stein_numerics.step_policy'); conv = cfg['stein_numerics']['convergence']['value']
    for key, want in (('features.feature_method', 'rff_declared_finite_map'), ('features.Gamma_model', None), ('observations.visibility_model', 'all_canonical_points_comparable'),
                      ('observations.unexplained_support_model', 'none'), ('stein.H_X', None), ('motion_prior.process_model', None)):
        v = value(cfg, key)
        if want is not None and v != want: raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'STEIN_PROPOSAL', [(key, f'{v!r}: only {want!r} is implemented')])
    o0 = source_order(canon.source, names); hx, med_x = median_heuristic(x0, o0); hc, med_c = median_heuristic(c0, o0)
    H_x, H_c = hx*torch.eye(3, dtype=dt, device=dev), hc*torch.eye(3, dtype=dt, device=dev)
    B = value(cfg, 'features.xyz_rgb_kernel_B')
    if B.get('method') != 'equal_stein_bandwidth': raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'STEIN_PROPOSAL', [('features.xyz_rgb_kernel_B', B)])
    feats = RFFFeatures(H_x.cpu(), H_c.cpu(), int(value(cfg, 'features.D')), int(value(cfg, 'features.fixed_seed')), dt, dev)
    if value(cfg, 'features.Gamma_model').get('method') != 'iid_sampling_covariance_frozen_at_initialization':
        raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'STEIN_PROPOSAL', [('features.Gamma_model', value(cfg, 'features.Gamma_model'))])
    terms, gamma_diag = [], []
    for k in range(K):
        progress(stage='FEATURES', frame=k+1, frames=K)
        if obs[k] is None: raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'STEIN_PROPOSAL', [(f'frame {k}', 'NO_OBSERVATION: the feature likelihood of an empty frame is not defined')])
        y, d = tt(clouds[k].xyz), tt(clouds[k].rgb)
        G = FeatureGaussian(timed('gamma_estimate', iid_sampling_gamma, feats, y, d, x0, c0))
        terms.append(FrameFeatureTerm(feats, y, d, G, c0)); gamma_diag.append({'frame': k, 'logdet': G.logdet, 'condition': G.condition})
    mp = value(cfg, 'motion_prior.initial_P')
    Pvv = np.asarray(config.velocity_covariance, float)
    temporal = TemporalPrior(data.times, config.qc, x0, cov0, Pvv)
    obj = SequenceObjective(terms, temporal)
    E = timed('objective', obj.value, S)
    timeline.append({'outer': 0, 'step': 'init', 'accepted': True, 'E_total': E['total'], 'E_obs': E['E_obs'], 'E_time': E['E_time']})
    status['gates']['FEATURES'] = {'D': feats.D, 'rff_sha256': feats.digest(), 'seed': feats.seed, 'H_X_m2': hx, 'H_C': hc, 'median_pair_distance_m': med_x,
                                   'median_pair_colour_distance': med_c, 'gamma': gamma_diag, 'E_initial': E}
    print(f'[svgd] objective at stationary init: E_obs {E["E_obs"]:.6g}  E_time {E["E_time"]:.6g}  (H_X^1/2 {math.sqrt(hx)*1e3:.2f} mm, D {feats.D})', flush=True)
    stein_on = bool(st_cfg['stein_enabled']['value']); kw = dict(attraction=bool(st_cfg['attraction_enabled']['value']), repulsion=bool(st_cfg['repulsion_enabled']['value']),
                                                                 kernel_color=bool(st_cfg['kernel_color_enabled']['value']))
    tid = torch.as_tensor(trace_ids, device=dev, dtype=torch.long)
    convergence = 'ITERATION_CAP_NOT_CONVERGED'
    max_outer = 1 if stop == 'STEIN_PROPOSAL' else int(conv['max_outer'])
    for outer in range(1, max_outer+1):
        status['stage_reached'] = 'STEIN_PROPOSAL'; E_begin = E['total']; S_begin = S.clone(); it_trace = {i: {'outer': outer, 'frames': []} for i in trace_ids}
        stein_rel = 0.
        if stein_on:
            Phi = torch.zeros(N, K, 3, dtype=dt, device=dev); mass_med, lam_max = [], 0.
            for k in range(K):
                progress(stage='STEIN_PROPOSAL', outer=outer, frame=k+1, frames=K)
                X = S[:, k, :3]; r = score(k, X)
                sf = timed('stein_field', stein_field, X, c0, r.grad_xyz, H_x, H_c, workspace_bytes=ws, **kw)
                Phi[:, k] = sf.total; mass_med.append(sf.kernel_mass)
                lam_max = max(lam_max, obs[k]['lam_x_max'])
                y, d, lx, lc, cam = frame_tensors(k)
                rt = target_score(X[tid], c0[tid], y, d, lx, lc, query_color_cov=Sigma_a, obs_camera=cam, cameras=C, workspace_bytes=ws)
                for n, i in enumerate(trace_ids):
                    it_trace[i]['frames'].append({'frame': k, 'x': X[i].tolist(), 'log_density': float(rt.log_density[n]), 'score': rt.grad_xyz[n].tolist(),
                                                  'effective_components': float(rt.effective_components[n]), 'camera_score_mass': rt.camera_mass[n].tolist(),
                                                  'attraction': sf.attraction[i].tolist(), 'repulsion': sf.repulsion[i].tolist(), 'stein_total': sf.total[i].tolist(),
                                                  'kernel_mass': float(sf.kernel_mass[i])})
                timeline.append({'outer': outer, 'step': 'stein_field', 'frame': k, 'score_norm_median': float(torch.linalg.norm(r.grad_xyz, dim=1).median()),
                                 'attraction_norm_median': float(torch.linalg.norm(sf.attraction, dim=1).median()),
                                 'repulsion_norm_median': float(torch.linalg.norm(sf.repulsion, dim=1).median()), 'kernel_mass_median': float(sf.kernel_mass.median())})
            eta = initial_step(lam_max, N, torch.cat(mass_med))
            for b in range(int(num['max_backtracks'])+1):
                Sc = S.clone(); Sc[:, :, :3] += eta*Phi; Ec = timed('objective', obj.value, Sc)
                ok = Ec['total'] < E['total']
                timeline.append({'outer': outer, 'step': 'stein_accept', 'eta_m2': eta, 'backtrack': b, 'accepted': ok, 'E_total': Ec['total'], 'E_obs': Ec['E_obs'], 'E_time': Ec['E_time'],
                                 'step_norm_median_m': float(torch.linalg.norm(eta*Phi, dim=-1).median())})
                if ok: stein_rel = (E['total']-Ec['total'])/max(1., abs(E['total'])); S, E = Sc, Ec; break
                eta *= float(num['backtracking_factor'])
            print(f'[svgd] outer {outer} Stein proposal {"ACCEPTED" if stein_rel > 0 else "REJECTED (no descent of the common objective)"}  E {E["total"]:.8g}', flush=True)
        if stop == 'STEIN_PROPOSAL': _record_traces(traces, it_trace, S_begin, S, trace_ids); convergence = 'STOPPED_AFTER_STEIN_PROPOSAL'; break
        status['stage_reached'] = 'TEMPORAL_REFINEMENT'; progress(stage='TEMPORAL_REFINEMENT', outer=outer)
        G = obj.grad(S)
        delta, cert = timed('pcg', pcg, lambda U: obj.gn_apply(S, U), -G, temporal.solve, float(conv['pcg_rtol']), int(conv['pcg_max_iter']))
        gcert = math.sqrt(max(0., float((G*temporal.solve(G)).sum()))/G.numel())
        alpha, gn_rel = 1., 0.
        for b in range(int(num['max_backtracks'])+1):
            Sc = S+alpha*delta; Ec = timed('objective', obj.value, Sc); ok = Ec['total'] < E['total']
            timeline.append({'outer': outer, 'step': 'gauss_newton', 'alpha': alpha, 'backtrack': b, 'accepted': ok, 'E_total': Ec['total'], 'E_obs': Ec['E_obs'],
                             'E_time': Ec['E_time'], 'pcg_iterations': cert['iterations'], 'pcg_relative_residual': cert['relative_residual'], 'pcg_status': cert['status'],
                             'grad_certificate': gcert})
            if ok: gn_rel = (E['total']-Ec['total'])/max(1., abs(E['total'])); S, E = Sc, Ec; break
            alpha *= .5
        _record_traces(traces, it_trace, S_begin, S, trace_ids)
        print(f'[svgd] outer {outer}: E {E["total"]:.8g} (obs {E["E_obs"]:.6g} time {E["E_time"]:.6g}) Stein rel {stein_rel:.2e} GN rel {gn_rel:.2e} '
              f'PCG {cert["iterations"]} {cert["status"]} grad-cert {gcert:.3e}', flush=True)
        if max(stein_rel, gn_rel) < float(conv['rel_tol']):
            convergence = 'CONVERGED_FIRST_ORDER' if gcert < float(conv['grad_certificate_tol']) else 'STALLED_NOT_CONVERGED'; break
    status['gates']['TEMPORAL_REFINEMENT'] = {'convergence': convergence, 'E_final': E, 'outer_iterations': outer,
                                              'objective_semantics': 'E_obs + E_time conditional on frozen V_t = all (DEV-0001, DEV-0008)'}
    X = S[:, :, :3].transpose(0, 1).contiguous(); V = S[:, :, 3:].transpose(0, 1).contiguous()
    L.npy(sv/'transport_provisional.npy', X.cpu().numpy()); L.npy(sv/'velocity_provisional.npy', V.cpu().numpy())
    disp = torch.linalg.norm(X-X[:1], dim=-1).max(0).values
    L.npy(sv/'motion_score_uncalibrated.npy', disp.cpu().numpy())
    # visibility diagnostic at the final provisional states (never in E)
    dv = lambda z: torch.as_tensor(config.depth_variance, dtype=dt, device=dev)+config.depth_squared_variance*z*z
    n_sigma = float(value(cfg, 'observations.visibility_n_sigma')) if SC.lookup(cfg, 'observations.visibility_n_sigma') else 3.
    tstat = np.zeros((K, N), np.int8); extra = {}
    for k in range(K):
        codes, point, unexpl = timed('visibility', ray_visibility, X[k], clouds[k], data.cameras, dv, n_sigma)
        tstat[k] = point.cpu().numpy()
        for c in range(C):
            extra[(k, c)] = {'particles_supported_in_camera': int((codes[:, c] == 0).sum()), 'particles_occluded_in_camera': int((codes[:, c] == 1).sum()),
                             'particles_free_space_conflict': int((codes[:, c] == 2).sum()), 'observed_samples_without_supported_particle': unexpl[c],
                             'visible_particles_frame': int((point == 0).sum()), 'occluded_particles_frame': int((point == 1).sum()),
                             'unsupported_particles_frame': int((point == 2).sum()), 'E_F_frame': E['E_obs_per_frame'][k]}
    L.npy(sv/'transport_status.npy', tstat); finish_counts(extra)
    L.json(sv/'transport_status_codes.json', {'point': POINT_CODES, 'camera': CAM_CODES, 'semantics': 'DIAGNOSTIC: visibility law NOT_DEFINED_IN_THEORY', 'n_sigma': n_sigma})
    if stop in ('STEIN_PROPOSAL', 'TEMPORAL_REFINEMENT'): status['stopped_after'] = stop; return EXIT['OK']
    # ------------------------------------------------ RIGID_GRAPH
    probs = config_problems(cfg, mode, 'RIGID_GRAPH')
    if probs: raise Blocked('BLOCKED_MODEL_UNSPECIFIED', 'RIGID_GRAPH', probs)
    status['stage_reached'] = 'RIGID_GRAPH'; progress(stage='RIGID_GRAPH')
    gm = value(cfg, 'rigid_graph.edge_proposal_method')
    edges, ginfo = timed('knn_graph', knn_graph, x0, int(gm['k']))
    if K > 1 and len(edges):
        e, dist = edge_residuals(X, edges)
        P_pos = prior_position_marginals(temporal)
        cc, delta = edge_covariance_structured(X, edges, P_pos, float(value(cfg, 'rigid_graph.Omega_model')))
        ll, chi2 = rigid_edge_loglik(e, cc, delta)
        drig = differential_rigidity(X, V, edges).abs().max(1).values
        L.npz(sv/'support_edges.npz', edges=edges.cpu().numpy(), original_ids=edges.cpu().numpy(), reference_distance_m=dist[:, 0].cpu().numpy(),
              max_abs_strain_m=e.abs().max(1).values.cpu().numpy(), rigid_chi2=chi2.cpu().numpy(), rigid_dof=np.full(len(edges), K-1),
              rigid_loglik=ll.cpu().numpy(), shared_reference_variance=cc.cpu().numpy(), max_abs_differential_rigidity=drig.cpu().numpy(),
              rho=np.full(len(edges), np.nan), rho_status=np.array('BLOCKED_MODEL_UNSPECIFIED: support prior not declared'),
              semantics=np.array('candidate hypotheses with conditional same-rigid likelihoods; NOT part labels'))
        status['gates']['RIGID_GRAPH'] = {**ginfo, 'chi2_over_dof_q': quantiles(chi2/(K-1)), 'max_abs_strain_q_m': quantiles(e.abs().max(1).values),
                                          'covariance': 'DEV-0007 shared-reference exact + prior-propagated independent marginals'}
    else: status['gates']['RIGID_GRAPH'] = {**ginfo, 'status': 'NOT_EVALUATED_SINGLE_FRAME'}
    if stop == 'RIGID_GRAPH': status['stopped_after'] = 'RIGID_GRAPH'; return EXIT['OK']
    # ------------------------------------------------ MODEL_EVIDENCE (no fabricated probabilities)
    status['stage_reached'] = 'MODEL_EVIDENCE'
    post = support_posterior(contract)
    L.json(sv/'posterior_evidence.json', post); L.npy(sv/'probability_status.npy', np.full(N, 5, np.int8))
    L.json(sv/'probability_status_codes.json', PROBABILITY_STATUS)
    raise Blocked(post['status'], 'MODEL_EVIDENCE', [(m['parameter'] if isinstance(m, dict) else m[0], m['reason'] if isinstance(m, dict) else m[1]) for m in post['missing']]
                  or [('model_evidence', post['note'])])


def _record_traces(traces, it_trace, S_begin, S, ids):
    for i in ids:
        it_trace[i]['accepted_delta_xyz'] = (S[i, :, :3]-S_begin[i, :, :3]).tolist(); it_trace[i]['x_after'] = S[i, :, :3].tolist()
        it_trace[i]['v_after'] = S[i, :, 3:].tolist(); traces[i]['iterations'].append(it_trace[i])
