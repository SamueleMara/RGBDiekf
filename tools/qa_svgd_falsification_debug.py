#!/usr/bin/env python3
"""Headless-Chrome QA of the post-hoc SVGD falsification viewer (debug/svgd_falsification_debug.py), checked against the AUDIT files.

    python -B tools/qa_svgd_falsification_debug.py --audit RUN/posthoc/svgd_falsification[_N]

Every page loads without JavaScript errors and reaches window.__state.ready === true. 3D page: the display-ID subset equals the meta subset
and never changes across frames; at the checked frames the drawn X0 / GT / arm positions equal oracle_trajectories.npz (float32, exact) at the
displayed IDs, the drawn Stein vectors Phi equal the npz at the vector subset, the frame slider changes the data, the HUD shows the true
timestamp and the GT / counterfactual labels, N_full / N_display are on screen; support colours and the cos(Phi, GT) layer equal colours
recomputed here; the arm exaggeration is applied exactly and labelled. Numerical page: every chart series equals the value recomputed here from
energy_oracle.csv / line_search_counterfactual.csv / stein_field_audit.json / falsification_summary.json; classification and diagnosis.md
text shown verbatim. Screenshots go to A/viewer/qa/*.png with sha256 in A/viewer/qa/qa_report.json.
status: PASS | FAIL | BROWSER_UNAVAILABLE | NO_WEBGL | NO_VIEWER (exit 0 | 1 | 2 | 2 | 1); never silent success.
"""
import argparse, base64, csv, hashlib, io, json, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qa_svgd_support_debug import close, read_js, dec

COL = {'moving': '#f0883e', 'static': '#58a6ff'}


class Series(list): pass


def hex3(h): v = int(h.lstrip('#'), 16); return np.array([(v >> 16 & 255)/255, (v >> 8 & 255)/255, (v & 255)/255])


def diverging(t):                                      # == PD.diverging (debug/assets/projective_common.js)
    t = np.clip(t, -1, 1)[:, None]; a, m, b = np.array([.35, .65, 1]), np.array([.55, .58, .62]), np.array([.97, .32, .29])
    return np.where(t < 0, m+(a-m)*(-t), m+(b-m)*t)


def num(v):
    if v in ('', None): return None
    if v in ('True', 'False'): return v == 'True'
    try: return float(v)
    except ValueError: return v


def rows(p): return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(p, newline=''))]


def same(got, want, rtol=1e-12):
    g = np.asarray([np.nan if v is None else v for v in got], float); w = np.asarray([np.nan if v is None else v for v in want], float)
    return g.shape == w.shape and bool(np.all((np.isnan(g) & np.isnan(w)) | (np.abs(g-w) <= rtol*np.maximum(np.abs(w), 1e-300))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audit', type=Path, required=True); ap.add_argument('--timeout', type=float, default=120)
    a = ap.parse_args(); A = a.audit.resolve(); V = A/'viewer'; qa = V/'qa'
    report = {'audit': str(A), 'status': None, 'pages': {}, 'checks': [], 'screenshots': [], 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    def finish(status, code):
        report['status'] = status; report['n_checks'] = len(report['checks']); report['n_failed'] = sum(not c['pass'] for c in report['checks'])
        qa.mkdir(parents=True, exist_ok=True); (qa/'qa_report.json').write_text(json.dumps(report, indent=1, default=str)); print(status, qa/'qa_report.json', flush=True); return code
    if not all((V/p).is_file() for p in ('index.html', 'falsification_3d.html', 'falsification_numerical.html', 'data/falsification_meta.js')):
        report['reason'] = f'{V} has no falsification viewer (python -B debug/svgd_falsification_debug.py --audit {A})'; return finish('NO_VIEWER', 1)
    qa.mkdir(parents=True, exist_ok=True); ok_all = True
    def check(name, ok, detail=''):
        nonlocal ok_all; ok_all &= bool(ok); report['checks'].append({'check': name, 'pass': bool(ok), 'detail': detail}); print(('PASS ' if ok else 'FAIL ')+name, str(detail)[:200], flush=True)
    from tools import qa_projective_debug as Q
    if not Path(Q.CHROME).exists(): report['reason'] = f'{Q.CHROME} not found'; return finish('BROWSER_UNAVAILABLE', 2)
    try:
        B = Q.Browser(V); B.pipe.call('Browser.getVersion', timeout=30)
    except Exception as e:
        hint = ' (TMPDIR path too long for Chrome\'s singleton socket; use a short TMPDIR)' if len(tempfile.gettempdir()) > 70 else ''
        report['reason'] = f'Chrome did not start: {type(e).__name__}: {e}{hint}'; return finish('BROWSER_UNAVAILABLE', 2)
    M = read_js(V/'data'/'falsification_meta.js'); ids = dec(M['display_ids'], np.int32).astype(int); vec = np.asarray(M['vector_ids'], int)
    Z = np.load(A/'oracle_trajectories.npz'); X0, Xg, mv, Phi = Z['X0'], Z['X_gt_POSTHOC'], np.asarray(Z['moving_POSTHOC'], bool), Z['Phi']; K = len(Xg)
    arms = {r['arm']: Z[f"X_arm_{r['arm']}"] for r in M['arms']}
    EO, LS = rows(A/'energy_oracle.csv'), rows(A/'line_search_counterfactual.csv')
    F3 = json.loads((A/'stein_field_audit.json').read_text()); S = json.loads((A/'falsification_summary.json').read_text())
    def shot(s, name, full=False):
        p = qa/f'{name}.png'
        if full:
            from PIL import Image
            Image.open(io.BytesIO(base64.b64decode(B.pipe.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': True}, s)['data']))).save(p)
        else: B.shot(s, p)
        report['screenshots'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    def wait(s, cond, timeout=a.timeout):
        t0 = time.time()
        while time.time()-t0 < timeout:
            if B.js(s, f'!!({cond})'): return True
            if B.js(s, '(window.__errors||[]).length || !!(window.__state && window.__state.error)'): return False
            time.sleep(.2)
        return False
    J = lambda s, e: B.js(s, e)
    f32 = lambda s, e: np.asarray(J(s, f'Array.from({e})'), np.float32)
    try:
        s = B.open('index.html'); wait(s, 'window.__state && window.__state.ready === true')
        gl = J(s, "(()=>{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl'))})()"); report['webgl'] = bool(gl)
        if not gl: report['reason'] = 'WebGL context unavailable in headless Chrome (swiftshader)'; return finish('NO_WEBGL', 2)
        for page in ('index.html', 'falsification_3d.html', 'falsification_numerical.html'):
            s = B.open(page); ready = wait(s, 'window.__state && window.__state.ready === true')
            errs = J(s, 'window.__errors||[]'); st_err = J(s, 'window.__state && window.__state.error || null')
            report['pages'][page] = {'ready': ready, 'errors': errs, 'state_error': st_err}
            if st_err == 'NO_WEBGL': report['reason'] = f'{page}: NO_WEBGL'; return finish('NO_WEBGL', 2)
            check(f'{page} loads without JS errors and reaches __state.ready', ready and not errs and not st_err, errs or st_err)
            if page == 'index.html':
                check('index links both pages and shows the classification', {'falsification_3d.html', 'falsification_numerical.html'} <= set(J(s, "[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))"))
                      and J(s, "document.getElementById('cls').textContent") == str(S['classification']))
            shot(s, page[:-5], full=page != 'falsification_3d.html')
        # ------------------------------------------------ 3D page
        s = B.open('falsification_3d.html'); wait(s, 'window.__state && window.__state.ready === true')
        check('3D: display IDs are the meta subset', J(s, 'Array.from(__state.display_ids)') == ids.tolist(), f'{len(ids)} of N {len(X0)}')
        check('3D: vector IDs are a subset of the display IDs', set(vec.tolist()) <= set(ids.tolist()) and J(s, '__state.vector_ids') == vec.tolist(), len(vec))
        x0 = f32(s, '__state.X0').reshape(-1, 3); check('3D: X0 layer == oracle X0[display IDs] (float32)', np.array_equal(x0, X0[ids].astype(np.float32)))
        check('3D: support flags == moving_POSTHOC[display IDs]', np.array_equal(np.asarray(J(s, 'Array.from(__state.moving)'), bool), mv[ids]))
        prev = None
        for k in sorted({0, K//2, K-1}):
            J(s, f"(()=>{{const e=document.getElementById('frame');e.value={k};e.dispatchEvent(new Event('input'))}})()")
            check(f'3D frame {k}: slider draws the complete frame', wait(s, f'__state.drawn_frame === {k} && __state.frame_complete === true', 60), J(s, '__state.drawn_frame'))
            gt = f32(s, '__state.gt').reshape(-1, 3); ref = Xg[k, ids]
            check(f'3D frame {k}: displayed GT == X_gt_POSTHOC[k, IDs] (float32)', np.array_equal(gt, ref.astype(np.float32)), f'max |displayed - saved f64| {np.abs(gt-ref).max():.3e} m')
            for arm, X in arms.items():
                d = f32(s, f'__state.arms[{json.dumps(arm)}]').reshape(-1, 3)
                check(f'3D frame {k}: displayed arm {arm} == X_arm_{arm}[k, IDs] (float32, exaggeration 1)', np.array_equal(d, X[k, ids].astype(np.float32)), f'max err {np.abs(d-X[k, ids]).max():.3e} m')
            pv = f32(s, '__state.phi_vec').reshape(-1, 3)
            check(f'3D frame {k}: Stein vectors == Phi[vector IDs, k] (float32)', np.array_equal(pv, Phi[vec, k].astype(np.float32)))
            check(f'3D frame {k}: every vector of the fixed subset drawn', J(s, '__state.vectors_drawn') == len(vec), [J(s, '__state.vectors_drawn'), len(vec)])
            hud = J(s, "document.getElementById('hud').textContent")
            check(f'3D frame {k}: HUD shows the true timestamp and the GT / counterfactual labels', f"{M['times_s'][k]:.4f}" in hud and M['gt_label'] in hud and (not arms or M['arm_label'] in hud), hud)
            cnt = J(s, "document.getElementById('counts').textContent")
            check(f'3D frame {k}: N_full and N_display on screen', 'N_full' in cnt and 'N_display' in cnt and str(len(X0)) in cnt and str(len(ids)) in cnt)
            check(f'3D frame {k}: display IDs unchanged (no per-frame resampling)', J(s, 'Array.from(__state.display_ids)') == ids.tolist())
            shot(s, f'falsification_3d_f{k:04d}')
            sig = (k, hud, float(gt.sum()))
            if prev is not None: check(f'3D: frame change {prev[0]} -> {k} changes the drawn data', sig[1] != prev[1] and (sig[2] != prev[2] or not mv.any()), [prev[0], k])
            prev = sig
        # colours: support (default) and cos(Phi, GT) recomputed here
        k = J(s, '__state.drawn_frame')
        J(s, "(()=>{const e=document.getElementById('mode');e.value='support';e.onchange()})()"); time.sleep(.3)
        cols = np.asarray(J(s, 'Array.from(__state.layer_colors)'), float).reshape(-1, 3); want = np.where(mv[ids][:, None], hex3(COL['moving']), hex3(COL['static']))
        check('3D colours [support]: moving / static colours == moving_POSTHOC', cols.shape == want.shape and np.abs(cols-want).max() < 1e-6)
        J(s, "(()=>{const e=document.getElementById('mode');e.value='cos_gt';e.onchange()})()"); time.sleep(.3)
        a_, b_ = Phi[ids, k], Xg[k, ids]-X0[ids]; na, nb = np.linalg.norm(a_, axis=1), np.linalg.norm(b_, axis=1)
        tiny = (na <= 1e-9*max(na.max(), 1e-300)) | (nb <= 1e-9*max(nb.max(), 1e-300)); c = np.where(tiny, np.nan, (a_*b_).sum(1)/np.maximum(na*nb, 1e-300)).astype(np.float32)
        v = np.asarray([np.nan if x is None else x for x in J(s, 'Array.from(__state.layer_values)')], np.float32)
        check(f'3D layer [cos_gt] frame {k}: values == cos(Phi, X_GT - X0) recomputed (float32)', np.allclose(v, c, rtol=0, atol=2e-6, equal_nan=True), f'{int(np.isnan(c).sum())} undefined')
        cols = np.asarray(J(s, 'Array.from(__state.layer_colors)'), float).reshape(-1, 3)
        want = np.where(np.isfinite(c)[:, None], diverging(np.nan_to_num(c.astype(float))), hex3(M['nan_colour']))
        check('3D colours [cos_gt]: diverging(cos), undefined = NaN colour', np.abs(cols-want).max() < 1e-5, f'max err {np.abs(cols-want).max():.2e}')
        shot(s, 'falsification_3d_cos_gt')
        J(s, "(()=>{const e=document.getElementById('mode');e.value='support';e.onchange()})()")
        if arms:
            J(s, "(()=>{const e=document.getElementById('aex');e.value='100';e.onchange()})()"); time.sleep(.3); arm = next(iter(arms))
            d = f32(s, f'__state.arms[{json.dumps(arm)}]').reshape(-1, 3); X0f, Xa = X0[ids].astype(np.float32), arms[arm][k, ids].astype(np.float32)
            check(f'3D: arm exaggeration x100 = X0 + 100 (X_arm - X0) and labelled', np.allclose(d, X0f+np.float32(100)*(Xa-X0f), rtol=0, atol=1e-6)
                  and 'display only' in J(s, "document.getElementById('hud').textContent"))
            shot(s, 'falsification_3d_arm_x100')
            J(s, "(()=>{const e=document.getElementById('aex');e.value='1';e.onchange()})()")
        i = int(ids[np.flatnonzero(mv[ids])[0]]) if mv[ids].any() else int(ids[0])
        J(s, f"(()=>{{const e=document.getElementById('pid');e.value='{i}';e.dispatchEvent(new Event('change'))}})()"); wait(s, f'__state.selected_info && __state.selected_info.id === {i}', 20)
        I = J(s, '__state.selected_info') or {}
        check(f'3D: inspect ID {i} shows X0 / GT / support of the npz', I.get('moving') == bool(mv[i]) and np.array_equal(np.asarray(I.get('gt'), np.float32), Xg[k, i].astype(np.float32))
              and np.array_equal(np.asarray(I.get('x0'), np.float32), X0[i].astype(np.float32)), I)
        errs = J(s, 'window.__errors||[]'); check('3D: no JS errors after all interactions', not errs, errs)
        # ------------------------------------------------ numerical page
        s = B.open('falsification_numerical.html'); wait(s, 'window.__state && window.__state.ready === true')
        def SR(cid):                                               # chart series as drawn: list of {label, x, y, x_raw, y_raw} + .markers / .frames
            r = J(s, f'__state.series[{json.dumps(cid)}] || null')
            if r is None: return None
            g = Series(r['s']); g.markers, g.frames = r['markers'], r['frames']; return g
        check('numerical: classification shown verbatim', J(s, "document.getElementById('clsv').textContent") == str(S['classification']), S['classification'])
        dm = (A/'diagnosis.md').read_text() if (A/'diagnosis.md').is_file() else 'diagnosis.md not written'
        check('numerical: diagnosis.md shown verbatim', J(s, "document.getElementById('diag').textContent") == dm)
        row = lambda cand, k: next(r for r in EO if r['candidate'] == cand and (r['frame'] == 'ALL' if k == 'ALL' else r['frame'] == k))
        sA = row('S', 'ALL'); I_ = sorted([r for r in EO if r['kind'] == 'interpolation_posthoc' and r['frame'] == 'ALL'], key=lambda r: r['alpha'])
        g = SR('alpha_sum')
        check('numerical alpha_sum: x = alpha and E_obs / E_total* / E_time* differences == energy_oracle.csv', g is not None and same(g[0]['x'], [r['alpha'] for r in I_])
              and same(g[0]['y'], [r['E_obs']-sA['E_obs'] for r in I_]) and same(g[1]['y'], [r['E_total_star']-sA['E_total_star'] for r in I_])
              and same(g[2]['y'], [r['E_time_star']-sA['E_time_star'] for r in I_]))
        kk = K-1
        J(s, f"(()=>{{const e=document.getElementById('aframe');e.value='{kk}';e.onchange()}})()"); time.sleep(.2); g = SR('alpha_frame')
        Ik = sorted([r for r in EO if r['kind'] == 'interpolation_posthoc' and r['frame'] == kk], key=lambda r: r['alpha'])
        check(f'numerical alpha_frame (frame {kk} selected): E_obs,t(I(alpha)) - E_obs,t(S) == CSV', g is not None and len(g) == 1 and same(g[0]['y'], [r['E_obs_t']-row('S', kk)['E_obs_t'] for r in Ik]))
        g = SR('dEobs_vs_disp'); o = g.frames if g else []
        d = [row('GT', k)['E_obs_t']-row('S', k)['E_obs_t'] for k in range(K)]
        check('numerical dEobs_vs_disp: per-frame dE_obs,t(GT - S) and GT displacement (mm) == CSV', g is not None and same(g[0]['y'], [d[k] for k in o])
              and same(g[0]['x'], [1e3*row('GT', k)['gt_disp_median_moving_m'] for k in o]) and sorted(o) == list(range(K)))
        check('numerical: per-frame dE_obs,t(GT - S) from the CSV == falsification_summary.json', same(d, S['gate_F1_F2']['dE_obs_GT_minus_S_t'], 1e-9))
        P = F3['per_frame']; disp = [row('GT', r['frame'])['gt_disp_median_moving_m'] for r in P]; oo = sorted(range(len(P)), key=lambda i: (disp[i], i))
        g = SR('cos_cumulative'); q = lambda key, st, order: [P[i][key].get(st) for i in order]
        check('numerical cos_cumulative: median / p10 / p90 == stein_field_audit.json', g is not None and all(same(g[j]['y'], q('cos_phi_cumulative_gt_moving', st, oo)) for j, st in enumerate(('median', 'p10', 'p90')))
              and same(g[0]['x'], [1e3*disp[i] for i in oo]))
        inc = [float(np.median(np.linalg.norm(Xg[r['frame'], mv]-Xg[r['frame']-1, mv], axis=1))) if r['frame'] and mv.any() else 0. for r in P]
        oi = sorted(range(len(P)), key=lambda i: (inc[i], i)); g = SR('cos_incremental')
        check('numerical cos_incremental: values == json, x = incremental GT displacement recomputed from the npz (mm)', g is not None
              and all(same(g[j]['y'], q('cos_phi_incremental_gt_moving', st, oi)) for j, st in enumerate(('median', 'p10', 'p90'))) and same(g[0]['x'], [1e3*inc[i] for i in oi], 1e-9))
        E0 = S['gate_F4']['E0_total']; c = 1e-12*abs(E0); sym = lambda v: np.sign(v)*np.log10(1+np.abs(v)/c)
        nls = 0
        for arm, search in dict.fromkeys((r['arm'], r['search']) for r in LS):
            rr = sorted([r for r in LS if r['arm'] == arm and r['search'] == search], key=lambda r: r['eta_m2']); g = SR(f'ls_{arm}_{search}'); nls += 1
            okk = g is not None and same(g[0]['x'], np.log10([r['eta_m2'] for r in rr]))
            for j, f in enumerate(('dE_total', 'dE_obs', 'dE_anchor', 'dE_process')):
                raw = [r[f] for r in rr]; okk = okk and same(g[j]['y_raw'], raw) and same(g[j]['y'], sym(np.asarray(raw, float)), 1e-9)
            nacc = sum(r['accepted'] is True for r in rr)
            check(f'numerical ls_{arm}_{search}: dE_total/obs/anchor/process vs log10 eta == CSV (symlog c = 1e-12|E0|), {nacc} accepted marked', okk and g.markers == nacc, [len(rr), g and g.markers])
        check('numerical: one line-search chart per (arm, search) in the CSV', nls == len({(r['arm'], r['search']) for r in LS}) and nls > 0, nls)
        for key in ('fd_E_obs', 'fd_E_total'):
            rr = F3['fd_closure'][key]['rows']; g = SR(key)
            check(f'numerical {key}: log10 rel_error vs log10 h == stein_field_audit.json', g is not None and same(g[0]['x_raw'], [r['h'] for r in rr]) and same(g[0]['y_raw'], [r['rel_error'] for r in rr])
                  and same(g[0]['x'], np.log10([r['h'] for r in rr]), 1e-12))
        T = [M['times_s'][r['frame']] for r in P]
        for cid, km, ks in (('norm_phi', 'norm_phi_moving', 'norm_phi_static'), ('norm_grad', 'norm_grad_Eobs_moving', 'norm_grad_Eobs_static'), ('cross_mass', 'moving_mass_from_static', 'static_mass_from_moving')):
            g = SR(cid); ix = list(range(len(P)))
            check(f'numerical {cid}: moving / static median and p90 == stein_field_audit.json', g is not None and same(g[0]['x'], T) and same(g[0]['y'], q(km, 'median', ix))
                  and same(g[1]['y'], q(km, 'p90', ix)) and same(g[2]['y'], q(ks, 'median', ix)) and same(g[3]['y'], q(ks, 'p90', ix)))
        g = SR('static_dominated'); check('numerical static_dominated == json', g is not None and same(g[0]['y'], [r['static_dominated_by_moving_fraction'] for r in P]))
        nz = J(s, '__state.normalizers'); cst = [row('S', k)['E_obs_t']-row('S', k)['E_obs_quadratic_t'] for k in range(K)]
        check('numerical: constant normalizers shown == E_obs_t - E_obs_quadratic_t (CSV)', same(nz['E_obs_t_const'], cst))
        J(s, "(()=>{const e=document.getElementById('ys');e.value='linear';e.onchange()})()"); time.sleep(.2)
        r0 = next(iter(dict.fromkeys((r['arm'], r['search']) for r in LS))); g = SR(f'ls_{r0[0]}_{r0[1]}')
        check('numerical: y-scale toggle switches the line-search charts to linear dE', g is not None and same(g[0]['y'], g[0]['y_raw']))
        J(s, "(()=>{const e=document.getElementById('ys');e.value='symlog';e.onchange()})()"); time.sleep(.2)
        check('numerical: charts drawn', J(s, '__state.charts') >= 12, J(s, '__state.charts'))
        shot(s, 'falsification_numerical_full', full=True)
        errs = J(s, 'window.__errors||[]'); check('numerical: no JS errors after interactions', not errs, errs)
    finally:
        close(B)
    return finish('PASS' if ok_all else 'FAIL', 0 if ok_all else 1)


if __name__ == '__main__':
    raise SystemExit(main())
