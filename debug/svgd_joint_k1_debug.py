#!/usr/bin/env python3
"""Visual debug of a joint K = 1 static/moving-support + shared-rigid-motion SVGD run (estimator/svgd_k1_run.py). DISPLAY ONLY:
reads RUN/run_manifest.json, RUN/svgd_joint_k1/* (and, if present, RUN/synthetic_fixture.json and RUN/posthoc/k1_oracle[_N]/oracle.json)
and writes ONLY inside RUN/debug_k1/:

    index.html, k1_3d.html, k1_numerical.html                                   (templates debug/assets/svgd_joint_k1_*.html)
    data/k1_meta.js, data/k1_numeric.js, data/k1_obs_fNNNN.js (observed masked cloud of frame NNNN, lazy)
    three.min.js + THREE_LICENSE.txt, projective_common.js, projective_theme.css   (copied, never fetched)

    python -B debug/svgd_joint_k1_debug.py --run RUN [--level light|full] [--display-cap 60000]

Semantics shown on every page: gamma_provisional is the UNCALIBRATED posterior-mean P(moving) (probability_semantics of status.json);
identifiability_status == 0 points carry no information beyond the certified numerical error and get a distinct 'no information' colour,
never colour(0). Observed masked clouds are not saved by the runner: they are rebuilt here with the same camera-aware loader
(scene.svgd_k1_cloud_adapter.from_dataset) and their sha256 is compared with RUN/svgd_joint_k1/loader_counts.json. Ground truth appears
only when a post-hoc oracle folder exists, recomputed for display with tools/audit_svgd_joint_k1_oracles.gt_for and labelled POST-HOC.
Display discipline: ONE deterministic display-ID subset of the reference catalogue (evenly spaced IDs, cap display_cap, plus an evenly
spaced subset of the SVGD / MAP seeds) for every frame; observed clouds use ONE fixed world voxel lattice (edge chosen once from frame 0,
first point in loader order per occupied voxel), never per-frame resampling. Arrays are base64 little-endian buffers with a stated dtype
(debug.projective_debug.b64 / write_js), window.PD_DATA; file:// works (script tags, no fetch).
level 'light': observed clouds for <= LIGHT_FRAMES evenly spaced frames (0 and last included); 'full': every frame. Poses: every frame.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debug.projective_debug import ASSETS, b64, display_ids, write_js

PAGES = {'k1_3d.html': 'svgd_joint_k1_3d.html', 'k1_numerical.html': 'svgd_joint_k1_numerical.html'}
NOINFO_HEX = '#ff7bd5'          # identifiability == 0: outside viridis and the diverging map, never mapped to 0
LIGHT_FRAMES = 12
GT_LABEL = 'GT — POST-HOC ORACLE, not inference'
GAMMA_LABEL = 'UNCALIBRATED P(moving)'
HIST_BINS = 40


def _none(v): return None if v in (None, 'None', '') else v


def read_json(p, default=None):
    p = Path(p); return json.loads(p.read_text()) if p.is_file() else default


def even(a, cap):
    a = np.asarray(a, int)
    return a[np.unique(np.linspace(0, len(a)-1, min(cap, len(a))).round().astype(int))] if len(a) and cap > 0 else np.zeros(0, int)


def rot_angle_deg(T):
    """Rotation angle (deg) of [..., 4, 4] transforms."""
    c = (np.trace(np.asarray(T)[..., :3, :3], axis1=-2, axis2=-1)-1)/2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pose_summary(P):
    P = np.asarray(P, float); return {'translation_norm_m': np.linalg.norm(P[..., :3, 3], axis=-1).tolist(), 'rotation_deg': rot_angle_deg(P).tolist()}


def hist(v, edges):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return np.histogram(v, edges)[0].tolist()


def oracle_dirs(run):
    ph = Path(run)/'posthoc'
    if not ph.is_dir(): return []
    ds = [d for d in ph.iterdir() if d.is_dir() and (d.name == 'k1_oracle' or d.name.startswith('k1_oracle_')) and (d/'oracle.json').is_file()]
    num = lambda d: 0 if d.name == 'k1_oracle' else int(d.name.rsplit('_', 1)[1]) if d.name.rsplit('_', 1)[1].isdigit() else -1
    return sorted(ds, key=num)


def open_dataset(run, man):
    """The same dataset object the runner used (synthetic fixture parameters or the scene + frame limit of the manifest)."""
    A = man.get('arguments', {})
    if str(man.get('scene', A.get('scene'))) == '/synthetic' or (Path(run)/'synthetic_fixture.json').is_file():
        from tests.svgd_fixtures import SyntheticDataset
        return SyntheticDataset(**json.loads((Path(run)/'synthetic_fixture.json').read_text())), 'synthetic'
    from scene.dataset import Dataset
    fl = _none(A.get('frame_limit')); timing = {}
    for k in ('fps', 'dt_seconds'):
        if _none(A.get(k)) is not None: timing[k] = float(A[k])
    if _none(A.get('timestamps_file')) is not None: timing['timestamps_file'] = Path(A['timestamps_file'])
    return Dataset(Path(man.get('scene', A.get('scene'))), None if fl is None else int(fl), None, **timing), 'scene'


def voxel_keep(xyz, h, origin):
    """First point (loader order) of every occupied voxel of the fixed world lattice (edge h, origin fixed per run)."""
    if h <= 0 or not len(xyz): return np.arange(len(xyz))
    q = np.floor((np.asarray(xyz, float)-origin)/h).astype(np.int64)
    _, first = np.unique(q, axis=0, return_index=True); return np.sort(first)


def voxel_edge(xyz, cap):
    """Smallest lattice edge (log bisection to 1 %) whose frame-0 kept count is <= cap; 0 = keep everything."""
    if len(xyz) <= cap: return 0., xyz.min(0) if len(xyz) else np.zeros(3)
    o = xyz.min(0); ext = max(float(np.max(xyz.max(0)-o)), 1e-9); lo, hi = ext/max(cap, 1), 2*ext
    while hi/lo > 1.01:
        mid = math.sqrt(lo*hi); lo, hi = (lo, mid) if len(voxel_keep(xyz, mid, o)) <= cap else (mid, hi)
    return hi, o


def write_debug(run_dir, level='light', display_cap=60000):
    if level not in ('light', 'full'): raise ValueError(f"debug level must be 'light' or 'full' (got {level!r}); 'none' writes no debug payload")
    run = Path(run_dir).resolve(); sv = run/'svgd_joint_k1'
    if not (sv/'gamma_provisional.npy').is_file(): raise FileNotFoundError(f'{sv/"gamma_provisional.npy"} missing: nothing to display (run did not reach OUTPUTS)')
    cap = int(display_cap)
    if cap <= 0: raise ValueError(f'display_cap must be positive (got {display_cap})')
    man = read_json(run/'run_manifest.json', {}) or {}; status = read_json(sv/'status.json', {}) or {}
    if man.get('run_kind') not in (None, 'svgd_joint_k1'): raise ValueError(f"run_kind {man.get('run_kind')!r} is not svgd_joint_k1")
    L = lambda n: np.load(sv/n) if (sv/n).is_file() else None
    gam, ES, EM, B, Bs, ident = L('gamma_provisional.npy'), L('evidence_static.npy'), L('evidence_moving_conditional.npy'), L('log_evidence_ratio.npy'), \
        L('log_evidence_ratio_particle_std.npy'), L('identifiability_status.npy')
    PP, PF, seeds = L('pose_particles.npy'), L('particle_F.npy'), L('moving_seed_indices.npy')
    MP, MG, MB, mseeds = L('map_pose.npy'), L('map_gamma.npy'), L('map_log_evidence_ratio.npy'), L('map_moving_seed_indices.npy')
    cat = np.load(sv/'reference_catalogue.npz'); xyz, rgb = np.asarray(cat['xyz'], float), np.asarray(cat['rgb'], float)
    a_eff = np.asarray(cat['a_eff'], float) if 'a_eff' in cat.files else None
    ref_ids = L('reference_ids.npy'); N = len(xyz)
    for nm, a in (('gamma_provisional', gam), ('evidence_static', ES), ('evidence_moving_conditional', EM), ('log_evidence_ratio', B),
                  ('log_evidence_ratio_particle_std', Bs), ('identifiability_status', ident), ('map_gamma', MG), ('map_log_evidence_ratio', MB)):
        if a is not None and a.shape != (N,): raise ValueError(f'{nm} shape {a.shape} != ({N},)')
    if PP is None or PP.ndim != 4 or PP.shape[2:] != (4, 4): raise ValueError(f'pose_particles shape {None if PP is None else PP.shape} is not [M,T+1,4,4]')
    Mp, K = PP.shape[:2]
    if MP is not None and MP.shape != (K, 4, 4): raise ValueError(f'map_pose shape {MP.shape} != ({K},4,4)')
    seeds = np.zeros(0, int) if seeds is None else np.asarray(seeds, int); mseeds = None if mseeds is None else np.asarray(mseeds, int)
    for nm, s in (('moving_seed_indices', seeds), ('map_moving_seed_indices', mseeds)):
        if s is not None and len(s) and (s.min() < 0 or s.max() >= N): raise ValueError(f'{nm} outside the reference catalogue')
    if ident is None: ident = np.ones(N, np.int8)
    num_err = status.get('numerical_error_bound_nats')
    ident_map = None if MB is None or num_err is None else (np.abs(MB) > float(num_err)).astype(np.int8)
    # ---- one deterministic display subset (all frames): evenly spaced IDs + evenly spaced seeds (bounded)
    keep = np.union1d(even(seeds, cap//4), even(mseeds, cap//4) if mseeds is not None else np.zeros(0, int))
    ids = display_ids(N, cap, keep); n = len(ids)
    # ---- dataset (same loader as the runner) -> observed clouds + true timestamps
    data, dkind = open_dataset(run, man)
    from scene.config import load_config
    from scene.svgd_k1_cloud_adapter import from_dataset
    nc = _none((man.get('arguments') or {}).get('noise_config')); cfg = load_config(nc)
    obs_frames = list(range(K)) if level == 'full' or K <= LIGHT_FRAMES else sorted(set(np.linspace(0, K-1, LIGHT_FRAMES).round().astype(int).tolist()))
    seq, counts = from_dataset(data, cfg, frames=obs_frames)
    times_all = [float(t) for t in np.asarray(data.times, float)[:K]]
    if len(times_all) != K: raise ValueError(f'dataset has {len(times_all)} frames, poses have {K}')
    run_counts = {int(r['frame']): r for r in (read_json(sv/'loader_counts.json', []) or [])}
    if obs_frames and 0 in obs_frames:
        f0 = seq.frames[obs_frames.index(0)]
        if f0.M != N or not np.allclose(f0.xyz, xyz, atol=1e-9): raise ValueError('rebuilt frame-0 cloud differs from reference_catalogue.npz: wrong dataset for this run')
    h, origin = voxel_edge(seq.frames[0].xyz if obs_frames else np.zeros((0, 3)), cap)
    dbg = run/'debug_k1'; dd = dbg/'data'; dd.mkdir(parents=True, exist_ok=True)
    obs_meta = []
    for j, k in enumerate(obs_frames):
        fr = seq.frames[j]; kp = voxel_keep(fr.xyz, h, origin); rc = run_counts.get(k, {}); c = counts[j]
        match = None if 'cloud_sha256' not in rc else rc['cloud_sha256'] == c['cloud_sha256']
        F = {'k': k, 'time_s': times_all[k], 'M_t': int(fr.M), 'M_display': int(len(kp)), 'xyz': b64(fr.xyz[kp], np.float32),
             'rgb': b64(np.clip(np.round(255*fr.rgb[kp]), 0, 255), np.uint8), 'cloud_sha256': c['cloud_sha256'], 'matches_run_cloud': match,
             'M_run': rc.get('M'), 'complete': True}
        write_js(dd/f'k1_obs_f{k:04d}.js', f'k1_obs_f{k:04d}', F)
        obs_meta.append({'k': k, 'M_t': int(fr.M), 'M_display': int(len(kp)), 'matches_run_cloud': match, 'cloud_sha256': c['cloud_sha256']})
    # ---- post-hoc GT (display only, never written to the run)
    ors = oracle_dirs(run); oracle = read_json(ors[-1]/'oracle.json') if ors else None
    gt = {'available': False, 'status': 'NO_ORACLE_FOLDER' if not ors else None, 'label': GT_LABEL}
    gt_mv = gt_poses = None
    if ors:
        try:
            from tools.audit_svgd_joint_k1_oracles import gt_for
            mv, poses, _ = gt_for(run, man.get('scene'), True if dkind == 'synthetic' else None, data, xyz)
            mv = np.asarray(mv, bool); gp = np.asarray(poses, float)
            if mv.shape != (N,) or gp.shape[1:] != (4, 4): raise ValueError(f'GT shapes moving {mv.shape} poses {gp.shape}')
            gt_mv, gt_poses = mv, gp[:K]
            gt.update(available=True, status='POSTHOC_GT', n_moving_full=int(mv.sum()), n_moving_display=int(mv[ids].sum()), poses=gt_poses.tolist(),
                      pose_summary=pose_summary(gt_poses))
        except Exception as e:   # display only: GT failure never blocks the viewer
            gt.update(status=f'GT_UNAVAILABLE: {type(e).__name__}: {e}'[:400])
    # ---- layers / legends (fixed here; the page only renders them)
    fin = lambda a: a[np.isfinite(a)]
    Bhi = float(np.quantile(np.abs(fin(B)), .99)) if B is not None and len(fin(B)) and np.quantile(np.abs(fin(B)), .99) > 0 else 1.
    Bshi = float(np.nanmax(Bs)) if Bs is not None and np.isfinite(Bs).any() and np.nanmax(Bs) > 0 else 1.
    psem = status.get('probability_semantics', 'UNCALIBRATED_APPROXIMATION')
    layers = {'rgb': {'name': 'measured RGB of the reference catalogue (frame 0)', 'kind': 'rgb'},
              'gamma': {'name': f'gamma_provisional — {GAMMA_LABEL} (SVGD posterior mean, {psem})', 'kind': 'viridis', 'lo': 0., 'hi': 1., 'lo_label': '0', 'hi_label': '1',
                        'title': GAMMA_LABEL, 'noinfo': 'identifiability_status', 'noinfo_label': 'no information (identifiability_status = 0: |B| within the certified numerical error)'}}
    if MG is not None:
        layers['map_gamma'] = {'name': f'map_gamma — {GAMMA_LABEL} at the MAP pose (CONTROL)', 'kind': 'viridis', 'lo': 0., 'hi': 1., 'lo_label': '0', 'hi_label': '1', 'title': GAMMA_LABEL,
                               'noinfo': 'map_identifiability', 'noinfo_label': 'no information (|B_MAP| <= numerical_error_bound_nats of status.json)' if ident_map is not None else None}
    layers['B'] = {'name': 'log evidence ratio B = log L^M − log L^S (mean over SVGD particles, nats)', 'kind': 'diverging', 'lo': -Bhi, 'hi': Bhi,
                   'lo_label': f'−{Bhi:.4g} (static)', 'hi_label': f'+{Bhi:.4g} (moving); clipped at q99 |B| over all N', 'noinfo': 'identifiability_status',
                   'noinfo_label': 'no information (identifiability_status = 0)'}
    if Bs is not None:
        layers['B_std'] = {'name': 'particle std of B (nats)', 'kind': 'viridis', 'lo': 0., 'hi': Bshi, 'lo_label': '0', 'hi_label': f'{Bshi:.4g} (max over all N)'}
    if gt_mv is not None: layers['gt'] = {'name': f'membership — {GT_LABEL}', 'kind': 'categorical'}
    banners = [{'code': psem, 'level': 'warn', 'text': f'gamma / P(moving) are {psem}: not calibrated probabilities'}]
    if status.get('sequence_scope') == 'PREFIX_DIAGNOSTIC': banners.append({'code': 'PREFIX_DIAGNOSTIC', 'level': 'warn', 'text': 'frame prefix only (--frame-limit), not the whole sequence'})
    if status.get('stage') != 'DONE': banners.append({'code': f"STAGE_{status.get('stage')}", 'level': 'bad', 'text': 'run did not finish (status.json stage != DONE)'})
    if int(np.asarray(ident).sum()) == 0: banners.append({'code': 'NO_INFORMATION', 'level': 'bad', 'text': f'identifiability_status = 0 for all {N} points (numerical error bound {num_err} nats)'})
    if not len(seeds): banners.append({'code': 'NO_SEEDS', 'level': 'warn', 'text': f"0 SVGD moving seeds at FDR alpha {status.get('seed_alpha')}"})
    if status.get('F_svgd_best') is not None and status.get('F_identity') is not None and status['F_svgd_best'] < status['F_identity']:
        banners.append({'code': 'SVGD_BELOW_IDENTITY', 'level': 'bad', 'text': f"best SVGD F {status['F_svgd_best']:.6g} < F at identity {status['F_identity']:.6g}"})
    if any(o['matches_run_cloud'] is False for o in obs_meta): banners.append({'code': 'OBSERVED_MISMATCH', 'level': 'bad', 'text': 'a rebuilt observed cloud differs from the run (loader_counts sha256)'})
    if gt['available']: banners.append({'code': 'POST-HOC', 'level': 'warn', 'text': f'{GT_LABEL}: GT layers come from {ors[-1].name}; never used by inference'})
    rs = lambda a, dt: None if a is None else b64(np.asarray(a)[ids], dt)
    meta = {'schema': 'svgd_joint_k1_debug/1', 'level': level, 'display_cap': cap, 'run': str(run), 'scene': man.get('scene'), 'dataset_kind': dkind,
            'N': N, 'N_display': n, 'K': K, 'T': K-1, 'M_particles': int(Mp), 'times_s': times_all, 'frame_ids': [int(f) for f in list(data.frame_ids)[:K]],
            'status': status, 'probability_semantics': psem, 'sequence_scope': status.get('sequence_scope'), 'banners': banners, 'layers': layers,
            'noinfo_colour': NOINFO_HEX, 'gamma_label': GAMMA_LABEL,
            'display_ids': b64(ids, np.int32), 'display_ids_sha256': hashlib.sha256(np.ascontiguousarray(ids, np.int64).tobytes()).hexdigest(),
            'reference_ids': rs(ref_ids, np.int64) if ref_ids is not None else None,
            'xyz': b64(xyz[ids], np.float32), 'rgb': b64(np.clip(np.round(255*rgb[ids]), 0, 255), np.uint8),
            'gamma': rs(gam, np.float32), 'map_gamma': rs(MG, np.float32), 'B': rs(B, np.float32), 'B_std': rs(Bs, np.float32),
            'evidence_static': rs(ES, np.float32), 'evidence_moving': rs(EM, np.float32), 'map_B': rs(MB, np.float32), 'a_eff': rs(a_eff, np.float32),
            'identifiability': rs(ident, np.uint8), 'map_identifiability': rs(ident_map, np.uint8),
            'seeds': {'n_full': int(len(seeds)), 'display': np.flatnonzero(np.isin(ids, seeds)).tolist()},
            'map_seeds': None if mseeds is None else {'n_full': int(len(mseeds)), 'display': np.flatnonzero(np.isin(ids, mseeds)).tolist()},
            'pose_particles': PP.tolist(), 'particle_F': None if PF is None else PF.tolist(), 'map_pose': None if MP is None else MP.tolist(),
            'gt': gt, 'gt_moving': None if gt_mv is None else b64(gt_mv[ids], np.uint8),
            'observed': {'frames': obs_meta, 'voxel_edge_m': h, 'voxel_origin_m': np.asarray(origin, float).tolist(), 'cap': cap,
                         'rule': (f'fixed world voxel lattice (edge {h*1e3:.3f} mm chosen once from frame 0 so that <= {cap} points; first point in loader order per voxel), same lattice every frame'
                                  if h > 0 else 'all observed points shown (M_t <= display cap)'),
                         'source': 'rebuilt with scene.svgd_k1_cloud_adapter.from_dataset (same loader as the runner); sha256 compared with loader_counts.json'},
            'counts': {'informative_points_full': int(np.asarray(ident).sum()), 'informative_points_display': int(np.asarray(ident)[ids].sum())},
            'display_rule': f'one deterministic subset of reference IDs for every frame: evenly spaced, cap {cap}, plus <= {cap//4} evenly spaced SVGD and MAP seeds; never re-sampled per frame'}
    write_js(dd/'k1_meta.js', 'k1_meta', meta)
    # ---- numerical page (full N; nothing depends on the display subset)
    opt = read_json(sv/'optimizer_diagnostics.json', {}) or {}; cal = read_json(sv/'calibration_provenance.json', {}) or {}
    ge = np.linspace(0, 1, HIST_BINS+1).tolist()
    Bf = fin(B) if B is not None else np.zeros(0); bl = float(np.quantile(np.abs(Bf), .995)) if len(Bf) and np.quantile(np.abs(Bf), .995) > 0 else 1.
    be = np.linspace(-bl, bl, HIST_BINS+1).tolist(); idb = np.asarray(ident).astype(bool)
    hists = {'gamma': {'edges': ge, 'all': hist(gam, ge), 'informative': hist(gam[idb], ge), 'no_information': hist(gam[~idb], ge)},
             'map_gamma': None if MG is None else {'edges': ge, 'all': hist(MG, ge)},
             'B': {'edges': be, 'informative': hist(np.clip(B[idb], -bl, bl), be), 'no_information': hist(np.clip(B[~idb], -bl, bl), be),
                   'clip': bl, 'note': f'B clipped to ±q99.5 |B| = ±{bl:.4g} nats (end bins include the tails)'} if B is not None else None}
    if gt_mv is not None:
        hists['gamma']['gt_moving_POSTHOC'] = hist(gam[gt_mv], ge); hists['gamma']['gt_static_POSTHOC'] = hist(gam[~gt_mv], ge)
        if MG is not None: hists['map_gamma'].update(gt_moving_POSTHOC=hist(MG[gt_mv], ge), gt_static_POSTHOC=hist(MG[~gt_mv], ge))
        if B is not None: hists['B'].update(gt_moving_POSTHOC=hist(np.clip(B[gt_mv], -bl, bl), be), gt_static_POSTHOC=hist(np.clip(B[~gt_mv], -bl, bl), be))
    q = lambda a: None if a is None or not np.isfinite(a).any() else [float(v) for v in np.quantile(fin(np.asarray(a, float)), [0, .01, .5, .99, 1])]
    rho = [{'frame': t+1, 'time_s': times_all[t+1] if t+1 < K else None, 'rho': r.get('rho'), 'dispersion_iqr': r.get('dispersion_iqr'),
            'folds': [f.get('rho') for f in r.get('folds', [])], 'folds_at_lower_bound': sum(bool(f.get('at_lower_bound')) for f in r.get('folds', [])),
            'block_edge_m': r.get('block_edge_m')} for t, r in enumerate(cal.get('rho', []))]
    write_js(dd/'k1_numeric.js', 'k1_numeric', {
        'svgd': opt.get('svgd', []), 'map': opt.get('map', []), 'F_identity': opt.get('F_identity', status.get('F_identity')), 'semantics': opt.get('semantics'),
        'status': status, 'hist': hists, 'quantiles': {'gamma': q(gam), 'map_gamma': q(MG), 'B': q(B), 'B_std': q(Bs), 'evidence_static': q(ES), 'evidence_moving': q(EM), 'a_eff': q(a_eff)},
        'rho': rho, 'kappa': cal.get('kappa'), 'grid_radius_m': cal.get('grid_radius_m'), 'representation_spacing_m': cal.get('representation_spacing_m'),
        'calibration': {k: cal.get(k) for k in ('a_eff', 'certificate_tolerance_nats', 'domain_mahalanobis2', 'motion_prior', 'support_prior_beta', 'sensor')},
        'times_s': times_all, 'pose': {'particles': [pose_summary(P) for P in PP], 'map': None if MP is None else pose_summary(MP), 'gt_POSTHOC': gt.get('pose_summary')},
        'particle_F': None if PF is None else PF.tolist(), 'oracle': oracle, 'oracle_folder': str(ors[-1]) if ors else None, 'oracle_folders': [d.name for d in ors],
        'gt_status': gt['status'], 'performance': read_json(sv/'performance.json'), 'loader_counts': read_json(sv/'loader_counts.json', []), 'observed': obs_meta,
        'counts': {'N': N, 'seeds': int(len(seeds)), 'map_seeds': None if mseeds is None else int(len(mseeds)), 'informative': int(idb.sum()),
                   'gt_moving_POSTHOC': None if gt_mv is None else int(gt_mv.sum())},
        'banners': banners, 'probability_semantics': psem, 'files_present': sorted(p.name for p in sv.iterdir() if p.is_file())})
    for s_, d_ in (('three-0.160.0.min.js', 'three.min.js'), ('THREE_LICENSE.txt', 'THREE_LICENSE.txt'), ('projective_common.js', 'projective_common.js'), ('projective_theme.css', 'projective_theme.css')):
        shutil.copyfile(ASSETS/s_, dbg/d_)
    for d_, s_ in PAGES.items(): shutil.copyfile(ASSETS/s_, dbg/d_)
    (dbg/'index.html').write_text(_index(meta))
    return {'index': str(dbg/'index.html'), 'pages': ['index.html', *PAGES], 'K': K, 'observed_frames': obs_frames, 'N_display': n, 'gt': gt['status']}


def _index(meta):
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    ban = ''.join(f'<div class="b {b["level"]}"><b>{esc(b["code"])}</b> {esc(b["text"])}</div>' for b in meta['banners'])
    st = meta['status']
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Joint K1 debug</title>
<link rel="stylesheet" href="projective_theme.css">
<style>body{{padding:16px 24px;max-width:1100px}}li{{margin:3px 0}}#banners .b{{padding:3px 6px;margin:3px 0;border-radius:4px;background:#21262d}}#banners .b.bad{{border-left:3px solid var(--bad)}}#banners .b.warn{{border-left:3px solid var(--warn)}}</style></head>
<body><h1>Joint K = 1 support + shared rigid motion (SVGD) — debug index ({esc(meta['level'])})</h1><div id="errors"></div><div id="banners">{ban}</div>
<p>stage <b>{esc(st.get('stage'))}</b> · probability semantics <b>{esc(meta['probability_semantics'])}</b> · scope <b>{esc(meta['sequence_scope'])}</b> ·
N {meta['N']} (displayed {meta['N_display']}) · {meta['K']} frames · {meta['M_particles']} SVGD particles · seeds {esc(st.get('seeds'))} (MAP control {esc(st.get('map_seeds'))})</p>
<p>F identity {esc(st.get('F_identity'))} · F SVGD best {esc(st.get('F_svgd_best'))} · F MAP {esc(st.get('F_map'))} ({esc(st.get('map_best_start'))})</p>
<ul><li><a href="k1_3d.html">k1_3d.html</a> — reference cloud (RGB / {esc(meta['gamma_label'])} / B), observed masked cloud per frame, inferred moving subset transformed by the MAP pose and every SVGD particle, seeds, post-hoc GT</li>
<li><a href="k1_numerical.html">k1_numerical.html</a> — SVGD / MAP traces, π, diversity, bandwidth, score norm, γ and B histograms, ρ per frame, pose trajectories, oracle summary</li></ul>
<p class="note">Display only: nothing is written outside this folder. {esc(meta['display_rule'])}. Observed: {esc(meta['observed']['rule'])}.</p>
<script>window.__state = {{ready: true, page: 'index'}};</script></body></html>
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--level', choices=['light', 'full'], default='light'); ap.add_argument('--display-cap', type=int, default=60000)
    a = ap.parse_args(argv); out = write_debug(a.run, a.level, display_cap=a.display_cap)
    print(f"[k1 viewer] {out['index']}  (N_display {out['N_display']}, {out['K']} frames, observed frames {out['observed_frames']}, GT {out['gt']})", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
