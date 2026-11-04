#!/usr/bin/env python3
"""Visual debug of the POST-HOC SVGD falsification audit (tools/audit_svgd_falsification.py). DISPLAY ONLY: reads the audit folder
A = RUN/posthoc/svgd_falsification[_N] (and, read-only, RUN/run_manifest.json + RUN/svgd_support/canonical.npz); writes ONLY

    A/viewer/index.html, falsification_3d.html, falsification_numerical.html                 (templates in debug/assets)
    A/viewer/data/falsification_{meta,numeric}.js, falsification_fNNNN.js (one per frame, lazy)
    A/viewer/three.min.js + THREE_LICENSE.txt, projective_common.js, projective_theme.css   (copied, never fetched)

    python -B debug/svgd_falsification_debug.py --audit A [--run RUN] [--display-cap 20000]

Semantics: X_gt_POSTHOC / moving_POSTHOC are the post-hoc ORACLE (never inference); X_arm_* are OFFLINE COUNTERFACTUAL positions at that
arm's first meaningful accepted step; Phi is the saved Stein field at the stationary state S0 (X = X0 at every frame), so Stein vectors start at X0.
Display discipline: ONE deterministic display-ID subset (evenly spaced original IDs, cap display_cap, plus the vector subset) for every frame (no
per-frame resampling); the Stein-vector subset is a fixed evenly spaced subset of the displayed moving and static IDs. Arrays are base64 of
little-endian buffers with a stated dtype (debug.projective_debug.write_js / b64), window.PD_DATA; file:// works (script tags, no fetch).
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debug.projective_debug import ASSETS, b64, display_ids, write_js
from debug.svgd_support_debug import read_csv, read_json

PAGES = {'falsification_3d.html': 'svgd_falsification_3d.html', 'falsification_numerical.html': 'svgd_falsification_numerical.html'}
VEC_CAP = 300                                   # Stein vectors drawn per support class (moving / static), fixed subset
GT_LABEL = 'GT — POST-HOC ORACLE, not inference'
ARM_LABEL = 'OFFLINE COUNTERFACTUAL'
NAN_HEX = '#d2a8ff'                            # undefined / NaN: outside viridis and the blue-grey-red diverging map, never mapped to 0


def even(a, cap):
    a = np.asarray(a, int)
    return a[np.unique(np.linspace(0, len(a)-1, min(cap, len(a))).round().astype(int))] if len(a) and cap > 0 else np.zeros(0, int)


def cosine_rows(a, b, eps_rel=1e-9):
    """Per-row cosine, NaN where either vector is tiny (same rule as tools/audit_svgd_falsification.cosine, recomputed for display)."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    tiny = (na <= eps_rel*max(na.max(initial=0), 1e-300)) | (nb <= eps_rel*max(nb.max(initial=0), 1e-300))
    c = (a*b).sum(-1)/np.maximum(na*nb, 1e-300); c[tiny] = np.nan
    return c


def cameras_from_manifest(man):
    out = []
    for c in (man.get('dataset') or {}).get('cameras') or []:
        try:
            W = np.asarray(c['world_to_camera'], float); fx, fy, cx, cy = c['intrinsics']
            out.append({'name': str(c['name']), 'K': [fx, fy, cx, cy], 'width': int(c['size'][0]), 'height': int(c['size'][1]), 'T_wc': np.linalg.inv(W).tolist()})
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError): pass
    return out


def write_falsification_debug(audit_dir, run_dir=None, display_cap=20000):
    A = Path(audit_dir).resolve(); run = Path(run_dir).resolve() if run_dir else A.parent.parent
    if not (A/'oracle_trajectories.npz').is_file(): raise FileNotFoundError(f'{A/"oracle_trajectories.npz"} missing: not a completed falsification audit folder')
    cap = int(display_cap)
    if cap <= 0: raise ValueError(f'display_cap must be positive (got {display_cap})')
    Z = np.load(A/'oracle_trajectories.npz'); X0, Xg, mv, Phi = Z['X0'], Z['X_gt_POSTHOC'], np.asarray(Z['moving_POSTHOC'], bool), Z['Phi']
    N, K = len(X0), len(Xg)
    if Xg.shape != (K, N, 3) or mv.shape != (N,) or Phi.shape != (N, K, 3): raise ValueError(f'oracle shapes X0 {X0.shape} X_gt {Xg.shape} moving {mv.shape} Phi {Phi.shape} inconsistent')
    arms = {a: (Z[f'X_arm_{a}'], float(Z[f'eta_arm_{a}'])) for a in 'ABCD' if f'X_arm_{a}' in Z.files}
    man = read_json(run/'run_manifest.json', {}) or {}; pre = read_json(A/'preflight.json', {}) or {}
    times = (man.get('dataset') or {}).get('timestamps_seconds') or pre.get('timestamps_s'); tsrc = 'run_manifest.json dataset.timestamps_seconds'
    if not times or len(times) != K: times, tsrc = (pre.get('timestamps_s') if len(pre.get('timestamps_s') or []) == K else list(range(K))), 'preflight.json timestamps_s' if len(pre.get('timestamps_s') or []) == K else 'FRAME_INDEX (no timestamps)'
    times = [float(t) for t in times]
    can = np.load(run/'svgd_support'/'canonical.npz') if (run/'svgd_support'/'canonical.npz').is_file() else None
    rgb = can['rgb0'] if can is not None and len(can['rgb0']) == N and np.array_equal(can['xyz0'], X0) else None
    src = can['source'] if rgb is not None else None
    # ---- one deterministic display subset (all frames) + fixed Stein-vector subset among displayed moving / static IDs
    base = display_ids(N, cap)
    vec = np.union1d(even(base[mv[base]], VEC_CAP), even(base[~mv[base]], VEC_CAP))
    ids = display_ids(N, cap, vec); n = len(ids); vj = np.searchsorted(ids, vec)
    disp = np.linalg.norm(Xg-X0[None], axis=2)                                         # [K, N] true displacement from X0 (m)
    pn = np.linalg.norm(Phi, axis=2)                                                    # [N, K]
    hi = lambda a: float(np.nanmax(a)) if np.isfinite(a).any() and np.nanmax(a) > 0 else 1.
    layers = {'support': {'name': 'post-hoc support (moving / static)', 'kind': 'categorical', 'semantics': 'moving_POSTHOC from the GT oracle; never inference'},
              'gt_disp': {'name': '|X_GT,t − X0| (m) — POST-HOC', 'kind': 'viridis', 'lo': 0., 'hi': hi(disp), 'lo_label': '0 m', 'hi_label': f'{hi(disp):.4g} m (max over all N, K)'},
              'phi_norm': {'name': '|Φ_t| (1/m; Δx = η Φ, η in m²)', 'kind': 'viridis', 'lo': 0., 'hi': hi(pn), 'lo_label': '0', 'hi_label': f'{hi(pn):.4g} 1/m (max over all N, K)'},
              'cos_gt': {'name': 'cos(Φ_t, X_GT,t − X0) — POST-HOC', 'kind': 'diverging', 'lo': -1., 'hi': 1., 'lo_label': '−1 (opposite to true motion)', 'hi_label': '+1 (along true motion)',
                         'nan_label': 'undefined (zero true displacement or |Φ| tiny, e.g. static / frame 0)'}}
    if rgb is not None: layers['rgb'] = {'name': 'measured RGB of the canonical sample (frame 0)', 'kind': 'rgb'}
    V = A/'viewer'; dd = V/'data'; dd.mkdir(parents=True, exist_ok=True)
    for old in dd.glob('falsification_*.js'): old.unlink()                              # only inside A/viewer (stale payload of an earlier write)
    for k in range(K):
        c = cosine_rows(Phi[ids, k], Xg[k, ids]-X0[ids]).astype(np.float32)
        F = {'k': k, 'time_s': times[k], 'gt': b64(Xg[k, ids], np.float32), 'phi_vec': b64(Phi[vec, k], np.float32), 'phi_norm': b64(pn[ids, k], np.float32),
             'gt_disp': b64(disp[k, ids], np.float32), 'cos_gt': b64(c, np.float32), 'arms': {a: b64(X[k, ids], np.float32) for a, (X, _) in arms.items()},
             'stats_full_N': {'gt_disp_moving_median_m': float(np.median(disp[k, mv])) if mv.any() else None, 'gt_disp_static_max_m': float(disp[k, ~mv].max()) if (~mv).any() else None,
                              'phi_norm_moving_median': float(np.median(pn[mv, k])) if mv.any() else None, 'phi_norm_static_median': float(np.median(pn[~mv, k])) if (~mv).any() else None,
                              **{f'arm_{a}_disp_moving_median_m': float(np.median(np.linalg.norm(X[k, mv]-X0[mv], axis=1))) if mv.any() else None for a, (X, _) in arms.items()}},
             'N_display': n, 'complete': True}
        write_js(dd/f'falsification_f{k:04d}.js', f'falsification_f{k:04d}', F)
    S = read_json(A/'falsification_summary.json', {}) or {}
    meta = {'schema': 'svgd_falsification_debug/1', 'audit': str(A), 'run': str(run), 'N': N, 'K': K, 'N_display': n, 'display_cap': cap, 'times_s': times, 'times_source': tsrc,
            'n_moving_full': int(mv.sum()), 'n_moving_display': int(mv[ids].sum()), 'display_ids': b64(ids, np.int32),
            'display_ids_sha256': hashlib.sha256(np.ascontiguousarray(ids, np.int64).tobytes()).hexdigest(),
            'vector_ids': vec.tolist(), 'vector_index': b64(vj, np.int32), 'X0': b64(X0[ids], np.float32), 'moving': b64(mv[ids], np.uint8),
            'rgb0': None if rgb is None else b64(np.clip(np.round(255*rgb[ids]), 0, 255), np.uint8), 'source': None if src is None else b64(src[ids], np.int32),
            'arms': [{'arm': a, 'eta_m2': e, 'label': f'{ARM_LABEL} arm {a} (η = {e:.3e} m², first meaningful accepted step)',
                      'definition': ((S.get('gate_F4') or {}).get('arm_definitions') or {}).get(a)} for a, (_, e) in arms.items()],
            'arms_missing': [a for a in 'ACD' if a not in arms], 'layers': layers, 'nan_colour': NAN_HEX, 'gt_label': GT_LABEL, 'arm_label': ARM_LABEL,
            'classification': S.get('classification'), 'cameras': cameras_from_manifest(man), 'semantics': str(Z['semantics']) if 'semantics' in Z.files else '',
            'gt_status': (pre.get('gt_verification') or {}).get('status'), 'reconstruction_valid': pre.get('reconstruction_valid'),
            'display_rule': f'one deterministic subset of original IDs for all frames: evenly spaced, cap {cap}, plus the Stein-vector subset; never re-sampled per frame',
            'vector_rule': f'Stein vectors: fixed evenly spaced subset of the displayed IDs, <= {VEC_CAP} moving + <= {VEC_CAP} static ({len(vec)} total); origin X0 (field at the stationary state S0)'}
    write_js(dd/'falsification_meta.js', 'falsification_meta', meta)
    # ---- numerical page: the audit's own tables, unmodified (the page derives differences and states how)
    F3 = read_json(A/'stein_field_audit.json', {}) or {}
    incr = [float(np.median(np.linalg.norm(Xg[k, mv]-Xg[k-1, mv], axis=1))) if k and mv.any() else 0. for k in range(K)]
    write_js(dd/'falsification_numeric.js', 'falsification_numeric', {
        'energy_oracle': read_csv(A/'energy_oracle.csv'), 'line_search': read_csv(A/'line_search_counterfactual.csv'),
        'stein_per_frame': F3.get('per_frame', []), 'fd_closure': F3.get('fd_closure', {}), 'summary': {k: v for k, v in S.items() if k != 'feature_residual_attribution'},
        'preflight': {k: pre.get(k) for k in ('run', 'scene', 'N', 'K', 'cameras', 'M_t', 'timestamps_s', 'reconstruction_valid', 'source_changes_since_run', 'in_sample_note', 'git', 'audit_commit')},
        'reconstruction_checks': pre.get('reconstruction_checks', []), 'gt_verification': {k: v for k, v in (pre.get('gt_verification') or {}).items() if k != 'observation_support_check'},
        'covariance_provenance': read_json(A/'covariance_provenance.json'), 'diagnosis_md': (A/'diagnosis.md').read_text() if (A/'diagnosis.md').is_file() else None,
        'times_s': times, 'gt_incremental_disp_median_moving_m': incr, 'files_present': sorted(p.name for p in A.iterdir() if p.is_file())})
    for s_, d_ in (('three-0.160.0.min.js', 'three.min.js'), ('THREE_LICENSE.txt', 'THREE_LICENSE.txt'), ('projective_common.js', 'projective_common.js'), ('projective_theme.css', 'projective_theme.css')):
        shutil.copyfile(ASSETS/s_, V/d_)
    for d_, s_ in PAGES.items(): shutil.copyfile(ASSETS/s_, V/d_)
    (V/'index.html').write_text(_index(meta))
    return {'index': str(V/'index.html'), 'pages': ['index.html', *PAGES], 'frames': K, 'N_display': n, 'vectors': len(vec)}


def _index(meta):
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>SVGD falsification audit</title><link rel="stylesheet" href="projective_theme.css">
<style>body{{padding:16px 24px;max-width:1100px}}li{{margin:3px 0}}.cls{{font-size:16px;padding:6px 10px;border-left:4px solid var(--warn);background:#21262d}}</style></head>
<body><h1>SVGD Stage-1 falsification audit — post-hoc viewer</h1><div id="errors"></div>
<div class="cls">classification <b id="cls">{esc(meta['classification'])}</b></div>
<p>GT {esc(meta['gt_status'])} · reconstruction valid {esc(meta['reconstruction_valid'])} · N {meta['N']} (displayed {meta['N_display']}) · {meta['K']} frames · moving (post hoc) {meta['n_moving_full']}</p>
<ul><li><a href="falsification_3d.html">falsification_3d.html</a> — X0, {esc(meta['gt_label'])}, {esc(meta['arm_label'])} arm positions, Stein vectors Φ, post-hoc support colouring</li>
<li><a href="falsification_numerical.html">falsification_numerical.html</a> — diagnosis, E_obs / E_total* along the α path, per-frame ΔE_obs, Stein cosines, line-search decomposition, FD closure, support distributions</li></ul>
<p class="note">Display only: nothing here is written outside this viewer folder; the audit files and RUN/svgd_support are never modified. {esc(meta['display_rule'])}.</p>
<script>window.__state = {{ready: true, page: 'index'}};</script></body></html>
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', type=Path, required=True); ap.add_argument('--run', type=Path); ap.add_argument('--display-cap', type=int, default=20000)
    a = ap.parse_args(argv); out = write_falsification_debug(a.audit, a.run, a.display_cap)
    print(f"[viewer] {out['index']}  (N_display {out['N_display']}, {out['vectors']} Stein vectors, {out['frames']} frames)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
