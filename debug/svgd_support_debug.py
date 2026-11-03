"""Visual debug of the SVGD moving-support Stage-1 run (spec section 11). DISPLAY ONLY: reads RUN/svgd_support (never
writes, renames or touches anything there) and the dataset's strict masked RGB-D frames; writes

    RUN/debug/index.html, svgd_support_3d.html, svgd_support_numerical.html          (templates in debug/assets)
    RUN/debug/data/svgd_support_{meta,numeric,traj,traces}.js, svgd_support_fNNNN.js (one per displayed frame, lazy)
    RUN/debug/three.min.js + THREE_LICENSE.txt, projective_common.js, projective_theme.css   (copied, never fetched)

Display discipline (nothing here feeds back into inference; inference outputs never depend on display_cap):
  * canonical / transported particles: ONE deterministic display-ID subset for the whole run (evenly spaced original IDs
    capped at display_cap, traced IDs and moving seeds always added) -> the same IDs in every frame (no subsampling flicker);
  * observed masked clouds: ONE fixed image-pixel lattice per run (keep u % s == 0 and v % s == 0, s chosen once from
    frame 0 so its total is <= display_cap), never a list-order stride; positions float32;
  * N_full and M_t are exact counts of the run; N_display / M_display state the displayed subset;
  * arrays: base64 of little-endian buffers with a stated dtype (write_js / b64 of debug.projective_debug), window.PD_DATA.
transport_provisional is INTERNAL_NOT_FINAL_TRACKS. Without p_dynamic.npy the colour layer is the UNCALIBRATED motion
evidence score (max provisional displacement, m), never called a probability. level 'light' = keyframes {0, mid, last}.
"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from debug.projective_debug import ASSETS, CAPS, b64, display_ids, write_js

PAGES = {'svgd_support_3d.html': 'projective_svgd_support.html', 'svgd_support_numerical.html': 'projective_svgd_numerical.html'}
NAN_HEX = '#ff7b72'                     # unresolved / no evidence / NaN: outside the viridis range, never mapped to 0
TRAJ_BUDGET = 1_500_000                 # float32 values of the clicked-ID trajectory payload (display only)
TRANSPORT_LABEL = 'INTERNAL_NOT_FINAL_TRACKS'


def _num(v):
    if v in ('', None): return None
    if v in ('True', 'False'): return v == 'True'
    try: return float(v)
    except ValueError: return v


def read_csv(p):
    if not p.is_file() or not p.stat().st_size: return []
    with open(p, newline='') as f: return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def read_json(p, default=None):
    return json.loads(p.read_text()) if p.is_file() else default


def cameras_spec(data):
    return [{'name': str(c.name), 'K': np.asarray(c.intrinsics, float).tolist(), 'width': int(c.width), 'height': int(c.height),
             'T_wc': np.linalg.inv(np.asarray(c.world_to_camera, float)).tolist()} for c in data.cameras]


def lattice_step(cl, cams, cap):
    """Smallest pixel-lattice step whose frame-0 masked cloud has <= cap points (one step for the whole run)."""
    if cl.M <= cap: return 1
    s = max(1, int(math.ceil(math.sqrt(cl.M/cap))))
    while lattice_keep(cl, cams, s).sum() > cap: s += 1
    return s


def lattice_keep(cl, cams, s):
    w = np.asarray([c['width'] for c in cams])[cl.camera]; pix = cl.source[:, 2]
    return (pix % w % s == 0) & (pix // w % s == 0)


def quant(a, qs=(0, .5, .9, .99, 1)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(v) for v in np.quantile(a, qs)] if len(a) else None


def write_debug(out, level, data, display_cap=None):
    from scene.config import load_config
    from scene.pointcloud_sequence import build_masked_cloud
    if level not in ('light', 'full'): raise ValueError(f"debug level must be 'light' or 'full' (got {level!r}); 'none' writes no debug payload")
    out = Path(out); sv = out/'svgd_support'
    if not (sv/'canonical.npz').is_file(): raise FileNotFoundError(f'{sv/"canonical.npz"} missing: nothing to display (run stopped before INIT committed)')
    cap = int(display_cap) if display_cap else CAPS[level]['points3d']
    if cap <= 0: raise ValueError(f'display_cap must be positive (got {display_cap})')
    can = np.load(sv/'canonical.npz'); N = len(can['xyz0'])
    if not np.array_equal(can['original_id'], np.arange(N)): raise ValueError('canonical original_id is not 0..N-1; display IDs would be ambiguous')
    opt = lambda n: np.load(sv/n, mmap_mode='r') if (sv/n).is_file() else None
    X, V, TS = opt('transport_provisional.npy'), opt('velocity_provisional.npy'), opt('transport_status.npy')
    score, pdyn, psta, pst = opt('motion_score_uncalibrated.npy'), opt('p_dynamic.npy'), opt('p_static.npy'), opt('probability_status.npy')
    seeds = opt('reliable_moving_seed_ids.npy')
    K = len(data.times); Ktr = 0 if X is None else int(X.shape[0])
    if X is not None and (X.shape[1:] != (N, 3) or Ktr > K): raise ValueError(f'transport_provisional shape {X.shape} incompatible with N={N}, K={K}')
    for nm, a in (('motion_score', score), ('p_dynamic', pdyn), ('p_static', psta), ('probability_status', pst)):
        if a is not None and a.shape != (N,): raise ValueError(f'{nm} shape {a.shape} != ({N},)')
    if seeds is not None:
        seeds = np.unique(np.asarray(seeds, int))
        if len(seeds) and (seeds.min() < 0 or seeds.max() >= N): raise ValueError('reliable_moving_seed_ids outside the published canonical IDs')
    status = read_json(sv/'status.json', {}); result = read_json(out/'result.json', {}); man = read_json(out/'run_manifest.json', {})
    tcodes = read_json(sv/'transport_status_codes.json', {}); pcodes = read_json(sv/'probability_status_codes.json', {})
    traces = {}
    for p in sorted((sv/'trace_points').glob('*.json')) if (sv/'trace_points').is_dir() else []:
        t = json.loads(p.read_text()); traces[int(t['original_id'])] = t
    traced = sorted(i for i in traces if 0 <= i < N)
    keep = np.union1d(np.asarray(traced, int), seeds if seeds is not None else np.zeros(0, int))
    ids = display_ids(N, cap, keep); n = len(ids); I = {int(i): j for j, i in enumerate(ids)}
    frames = list(range(K)) if level == 'full' else sorted({0, (K-1)//2, K-1})
    cams = cameras_spec(data); times = [float(t) for t in data.times]
    dbg = out/'debug'; dd = dbg/'data'; dd.mkdir(parents=True, exist_ok=True)
    for old in dd.glob('svgd_support_*.js'): old.unlink()                      # stale frames of an earlier level
    committed = [bool(X is not None and k < Ktr and (TS is None or not np.all(np.asarray(TS[k]) == 3))) for k in range(K)]
    # ---- layers: names and legend endpoints are fixed here (python), the page only renders them
    psem = status.get('probability_semantics', 'PROBABILITY_NOT_YET_ESTABLISHED')
    layers = {}
    if score is not None:
        hi = float(np.nanmax(score)) if np.isfinite(score).any() else 1.
        layers['score'] = {'name': 'motion evidence score (UNCALIBRATED)', 'units': 'm', 'lo': 0., 'hi': hi, 'lo_label': '0 m (no provisional displacement)',
                           'hi_label': f'{hi:.6g} m (max over all N)', 'nan_label': 'NaN / no evidence',
                           'semantics': 'max provisional displacement over frames; NOT a probability'}
    if pdyn is not None:
        layers['prob'] = {'name': f'p_dynamic ({psem})', 'units': '', 'lo': 0., 'hi': 1., 'lo_label': '0 (static explanation)', 'hi_label': '1 (moving explanation)',
                          'nan_label': 'unresolved (NaN or probability_status != SUPPORTED)', 'semantics': psem}
    banners = []
    blk = status.get('blocked')
    if blk: banners.append({'code': blk.get('status', 'BLOCKED'), 'level': 'bad', 'text': f"stage {blk.get('stage')}: " + '; '.join(f"{r.get('parameter')}: {r.get('reason')}" for r in blk.get('reasons', []))})
    if X is None: banners.append({'code': 'NO_ESTIMATION', 'level': 'bad', 'text': f"no transported cloud (stage reached {status.get('stage_reached')}); only canonical and observed clouds are shown"})
    unc = [k for k in range(K) if not committed[k]]
    if unc and X is not None: banners.append({'code': 'FRAME_UNCOMMITTED', 'level': 'warn', 'text': f'frames without committed provisional transport: {unc[:20]}' + (' ...' if len(unc) > 20 else '')})
    if pdyn is None or psem == 'PROBABILITY_NOT_YET_ESTABLISHED':
        banners.append({'code': 'PROBABILITY_UNCALIBRATED', 'level': 'warn', 'text': f'{psem}: ' + ('no p_dynamic published; colour layer = motion evidence score (UNCALIBRATED), not a probability' if pdyn is None else 'p_dynamic present but its semantics are not established')})
    if status.get('sequence_scope') == 'PREFIX_DIAGNOSTIC': banners.append({'code': 'PREFIX_DIAGNOSTIC', 'level': 'warn', 'text': 'frame prefix only, not the whole sequence'})
    if status.get('theory_fidelity', 'STRICT') != 'STRICT': banners.append({'code': status['theory_fidelity'], 'level': 'warn', 'text': 'experimental deviations from the theory are active'})
    banners.append({'code': TRANSPORT_LABEL, 'level': 'warn', 'text': 'transported cloud = provisional Stein transport, never final tracks or a rigid-motion fit'})
    # ---- edges among displayed points (bounded), coloured by strain / chi2 / boundary disagreement / rho
    edges = None; ez = dict(np.load(sv/'support_edges.npz')) if (sv/'support_edges.npz').is_file() else None
    if ez is not None:
        E = np.asarray(ez['edges'], int); m = np.isin(E[:, 0], ids) & np.isin(E[:, 1], ids); sel = np.flatnonzero(m)
        if len(sel) > 4*cap: sel = sel[np.unique(np.linspace(0, len(sel)-1, 4*cap).round().astype(int))]
        g = lambda k_, dt: ez[k_][sel].astype(dt) if k_ in ez else np.full(len(sel), np.nan, dt)
        dof = np.maximum(ez['rigid_dof'][sel], 1) if 'rigid_dof' in ez else np.ones(len(sel))
        st = ez['max_abs_strain_m'] if 'max_abs_strain_m' in ez else np.zeros(0)
        edges = {'E_full': int(len(E)), 'E_display': int(len(sel)), 'pairs': b64(E[sel], np.int32), 'reference_distance_m': b64(g('reference_distance_m', np.float32), np.float32),
                 'max_abs_strain_m': b64(g('max_abs_strain_m', np.float64), np.float64), 'chi2_per_dof': b64(g('rigid_chi2', np.float64)/dof, np.float64),
                 'rho': b64(g('rho', np.float64), np.float64), 'strain_hi_m': float(np.nanquantile(st, .98)) if np.isfinite(st).any() else 1.,
                 'semantics': str(ez['semantics']) if 'semantics' in ez else '', 'rho_status': str(ez['rho_status']) if 'rho_status' in ez else '',
                 'label': 'rigidity candidate graph: candidate hypotheses, not part labels'}
    # ---- observed masked clouds on one fixed pixel lattice
    cfg = load_config(); done = read_json(sv/'completed_frames.json', []); done = {int(r['frame']): r for r in done}
    counts = read_csv(sv/'observed_counts.csv'); step = None; Mt, Md = [None]*K, [None]*K
    for k in frames:
        cl = build_masked_cloud(data, k, cfg)
        if step is None: step = lattice_step(build_masked_cloud(data, 0, cfg) if k else cl, cams, cap)
        kp = np.flatnonzero(lattice_keep(cl, cams, step)); Mt[k], Md[k] = int(cl.M), int(len(kp))
        ref = done.get(k, {}).get('cloud_sha256')
        obs = {'M_t': cl.M, 'M_display': len(kp), 'xyz': b64(cl.xyz[kp], np.float32), 'rgb': b64(np.clip(np.round(255*cl.rgb[kp]), 0, 255), np.uint8),
               'camera': b64(cl.camera[kp], np.uint8), 'pixel': b64(cl.source[kp, 2], np.int32), 'lattice_step': step,
               'per_camera': [{'camera': c, 'M': int((cl.camera == c).sum()), 'M_display': int((cl.camera[kp] == c).sum())} for c in range(len(cams))],
               'cloud_sha256': cl.digest(), 'matches_run_cloud': None if ref is None else ref == cl.digest()}
        F = {'k': k, 'frame_id': int(data.frame_ids[k]), 'time_s': times[k], 'committed': committed[k], 'frame_status': 'COMMITTED' if committed[k] else 'FRAME_UNCOMMITTED',
             'transported': None, 'velocity': None, 'transport_status': None, 'status_counts_full': None, 'observed': obs,
             'counts': [r for r in counts if r.get('frame') == k], 'N_display': n}
        if X is not None and k < Ktr:
            F['transported'] = b64(np.asarray(X[k])[ids], np.float32)
            if V is not None: F['velocity'] = b64(np.asarray(V[k])[ids], np.float32)
            if TS is not None:
                ts = np.asarray(TS[k]); F['transport_status'] = b64(ts[ids], np.uint8)
                F['status_counts_full'] = {tcodes.get('point', {}).get(str(c), str(c)): int((ts == c).sum()) for c in np.unique(ts)}
        F['complete'] = True                                                   # written last: the page draws complete frames only
        write_js(dd/f'svgd_support_f{k:04d}.js', f'svgd_support_f{k:04d}', F)
    # ---- clicked-ID trajectories (all frames) for traced, seeds and an even display subset within a fixed budget
    budget = max(0, TRAJ_BUDGET//(3*max(K, 1)))
    tid = np.union1d(keep, ids[display_ids(n, budget)] if budget and n else np.zeros(0, int)).astype(int)
    trajX = np.full((len(tid), K, 3), np.nan, np.float32); trajS = np.full((len(tid), K), 255, np.uint8)
    if X is not None: trajX[:, :Ktr] = np.asarray(X[:, tid]).transpose(1, 0, 2)
    if TS is not None: trajS[:, :Ktr] = np.asarray(TS[:, tid]).T
    inc = {}
    if ez is not None:
        E = np.asarray(ez['edges'], int); sel = np.flatnonzero(np.isin(E[:, 0], keep) | np.isin(E[:, 1], keep))
        for e in sel:
            for a_, b_ in ((E[e, 0], E[e, 1]), (E[e, 1], E[e, 0])):
                if a_ in traces or (seeds is not None and a_ in seeds):
                    inc.setdefault(int(a_), []).append([int(b_)]+[float(ez[f][e]) if f in ez and np.isfinite(ez[f][e]) else None
                                                                for f in ('reference_distance_m', 'max_abs_strain_m', 'rigid_chi2', 'rigid_dof', 'rho')])
    write_js(dd/'svgd_support_traj.js', 'svgd_support_traj', {'ids': tid.tolist(), 'xyz': b64(trajX, np.float32), 'status': b64(trajS, np.uint8),
             'K': K, 'K_transport': Ktr, 'incident_edges': inc, 'incident_edge_fields': ['neighbor', 'reference_distance_m', 'max_abs_strain_m', 'rigid_chi2', 'rigid_dof', 'rho'],
             'note': f'{TRANSPORT_LABEL}: provisional XYZ per frame (NaN = not committed); 255 = no transport status'})
    write_js(dd/'svgd_support_traces.js', 'svgd_support_traces', {'traces': {str(i): traces[i] for i in traced},
             'vectors': {'attraction': 'target attraction: kernel-weighted target score (grad log density of the observed masked RGB-D mixture) pulling the particle toward observed material',
                         'repulsion': 'Stein repulsion: kernel-gradient term pushing nearby particles apart (keeps the particle set spread over the support)',
                         'stein_total': 'Stein total = attraction + repulsion: the transport direction proposed at this outer iteration',
                         'accepted_step': 'accepted step: accepted_delta_xyz of this outer iteration at this frame (metres, after the line search / Gauss-Newton), drawn at true scale x step factor'}})
    # ---- meta
    rs = lambda a, dt: None if a is None else b64(np.asarray(a)[ids], dt)
    sd = np.sqrt(np.maximum(np.diagonal(can['cov0'][ids], axis1=1, axis2=2), 0))*1e3
    meta = {'schema': 'svgd_support_debug/1', 'level': level, 'display_cap': cap, 'run': str(out), 'N': N, 'N_display': n, 'K': K, 'K_transport': Ktr,
            'frames': frames, 'times_s': times, 'frame_ids': [int(f) for f in data.frame_ids], 'committed': committed, 'cameras': cams,
            'stage_reached': status.get('stage_reached'), 'blocked': blk, 'probability_semantics': psem, 'theory_fidelity': status.get('theory_fidelity'),
            'sequence_scope': status.get('sequence_scope'), 'debug_status': status.get('debug'), 'result': result, 'banners': banners,
            'transport_label': TRANSPORT_LABEL, 'display_ids': b64(ids, np.int32), 'display_ids_sha256': hashlib.sha256(np.ascontiguousarray(ids, np.int64).tobytes()).hexdigest(),
            'canonical_xyz': b64(can['xyz0'][ids], np.float32), 'canonical_rgb': b64(np.clip(np.round(255*can['rgb0'][ids]), 0, 255), np.uint8),
            'source': b64(can['source'][ids], np.int32), 'sigma0_mm': b64(sd, np.float32),
            'motion_score': rs(score, np.float64), 'p_dynamic': rs(pdyn, np.float64), 'p_static': rs(psta, np.float64), 'probability_status': rs(pst, np.uint8),
            'probability_status_codes': pcodes, 'transport_status_codes': tcodes, 'layers': layers, 'nan_colour': NAN_HEX,
            'seeds': None if seeds is None else {'ids': seeds.tolist(), 'trajectory_in': 'svgd_support_traj'}, 'traced_ids': traced,
            'edges': edges, 'observed': {'lattice_step': step, 'M_t': Mt, 'M_display': Md, 'cap': cap,
                                         'rule': f'fixed pixel lattice u % {step} == 0 and v % {step} == 0 per camera (chosen from frame 0), same pixels every frame'},
            'files_present': sorted(p.name for p in sv.iterdir() if p.is_file()),
            'display_rule': f'one deterministic subset of original IDs for the whole run: evenly spaced, cap {cap}, plus traced and seed IDs; never re-sampled per frame'}
    write_js(dd/'svgd_support_meta.js', 'svgd_support_meta', meta)
    # ---- numerical page data
    perf = read_json(sv/'performance.json', {}); E_ = None
    if ez is not None:
        dof = np.maximum(ez['rigid_dof'], 1) if 'rigid_dof' in ez else 1
        E_ = {'max_abs_strain_m_q': quant(ez['max_abs_strain_m']) if 'max_abs_strain_m' in ez else None,
              'chi2_per_dof_q': quant(ez['rigid_chi2']/dof) if 'rigid_chi2' in ez else None,
              'rho_finite': int(np.isfinite(ez['rho']).sum()) if 'rho' in ez else 0, 'E': int(len(ez['edges']))}
    hist = None
    if score is not None and np.isfinite(score).any():
        h, be = np.histogram(np.asarray(score)[np.isfinite(score)], 40); hist = {'counts': h.tolist(), 'edges': be.tolist()}
    tsc = None
    if TS is not None:
        tsc = {nm: [int((np.asarray(TS[k]) == int(c)).sum()) for k in range(Ktr)] for c, nm in tcodes.get('point', {}).items()}
    write_js(dd/'svgd_support_numeric.js', 'svgd_support_numeric', {
        'timeline': read_csv(sv/'objective_timeline.csv'), 'counts': counts, 'performance': perf, 'status': status, 'completed_frames': list(done.values()),
        'theory_parity_summary': man.get('theory_parity_summary', 'NOT_GENERATED'), 'edge_quantiles': E_, 'score_hist': hist,
        'score_q': quant(score) if score is not None else None, 'p_dynamic_q': quant(pdyn) if pdyn is not None else None,
        'p_dynamic_unresolved': int((~np.isfinite(pdyn)).sum()) if pdyn is not None else None,
        'probability_status_counts': None if pst is None else {pcodes.get(str(c), str(c)): int((np.asarray(pst) == c).sum()) for c in np.unique(pst)},
        'transport_status_counts': tsc, 'seed_count': None if seeds is None else int(len(seeds)), 'posterior_evidence': read_json(sv/'posterior_evidence.json')})
    # ---- pages and assets
    for src, dst in (('three-0.160.0.min.js', 'three.min.js'), ('THREE_LICENSE.txt', 'THREE_LICENSE.txt'), ('projective_common.js', 'projective_common.js'),
                     ('projective_theme.css', 'projective_theme.css')):
        shutil.copyfile(ASSETS/src, dbg/dst)
    for dst, tpl in PAGES.items(): shutil.copyfile(ASSETS/tpl, dbg/dst)
    (dbg/'index.html').write_text(_index(meta))
    return {'index': str(dbg/'index.html'), 'pages': ['index.html', *PAGES], 'frames': frames, 'N_display': n, 'lattice_step': step}


def _index(meta):
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    ban = ''.join(f'<div class="{b["level"]}"><b>{esc(b["code"])}</b> {esc(b["text"])}</div>' for b in meta['banners'])
    return f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>SVGD support debug</title><link rel="stylesheet" href="projective_theme.css">
<style>body{{padding:16px 24px;max-width:1100px}}li{{margin:3px 0}}#banners div{{margin:3px 0}}</style></head>
<body><h1>SVGD moving-support Stage 1 — debug index ({esc(meta['level'])})</h1><div id="errors"></div><div id="banners">{ban}</div>
<p>stage reached <b>{esc(meta['stage_reached'])}</b> · probability semantics <b>{esc(meta['probability_semantics'])}</b> · N {meta['N']} (displayed {meta['N_display']}) · {meta['K']} frames ({len(meta['frames'])} displayed) · {len(meta['cameras'])} cameras</p>
<ul><li><a href="svgd_support_3d.html">svgd_support_3d.html</a> — canonical, observed masked, provisional transported clouds, score/probability layer, seeds, rigidity candidate edges, traced vectors</li>
<li><a href="svgd_support_numerical.html">svgd_support_numerical.html</a> — objective, line search, Stein norms, PCG, counts, timings, clicked-ID trace</li></ul>
<p class="note">Display subsets are display-only ({esc(meta['display_rule'])}; observed: {esc(meta['observed']['rule'])}). Inference outputs in ../svgd_support are never modified.</p>
<script>window.__state = {{ready: true, page: 'index'}};</script></body></html>
'''
