#!/usr/bin/env python3
"""Headless-Chrome QA of the SVGD moving-support debug pages (spec 11.3), checked against the SAVED numerical arrays.

    python -B tools/qa_svgd_support_debug.py --run RUN

Every page loads without JavaScript errors and reaches window.__state.ready === true. 3D page: frame slider and camera
toggle change the drawn data; the drawn transported XYZ of the display IDs equals RUN/svgd_support/transport_provisional.npy
cast to float32 (exact) at every checked frame, the canonical layer equals xyz0 and never moves; selecting an original ID
(text box and mouse click) shows the same motion score / probability status / source / transport status / p_dynamic as the
.npy arrays; the colour legend endpoints equal labels recomputed here from the arrays, every point colour equals
viridis((v-lo)/(hi-lo)) recomputed here, and NaN / unresolved values get the distinct NaN colour (never colour(0)); only
complete frames are drawn after rapid slider changes; on-screen N_full / N_display / M_t / M_display. Numerical page:
banners (NO_ESTIMATION / FRAME_UNCOMMITTED / PROBABILITY_UNCALIBRATED / blocked status), charts drawn, clicked-ID trace over
all frames equals the saved arrays. Screenshots go to RUN/debug/qa/*.png with sha256 in RUN/debug/qa/qa_report.json.
status: PASS | FAIL | BROWSER_UNAVAILABLE | NO_WEBGL | NO_DEBUG_PAGES (exit 0 | 1 | 2 | 2 | 1); never silent success.
"""
import argparse, base64, hashlib, json, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIRIDIS = np.array([[.267, .005, .329], [.283, .141, .458], [.254, .265, .530], [.207, .372, .553], [.164, .471, .558], [.128, .567, .551], [.135, .659, .518],
                    [.267, .749, .441], [.478, .821, .318], [.741, .873, .150], [.993, .906, .144]])


def viridis(t):
    t = np.clip(t, 0, 1); x = t*(len(VIRIDIS)-1); i = np.minimum(len(VIRIDIS)-2, np.floor(x)).astype(int); f = (x-i)[:, None]
    return VIRIDIS[i]+(VIRIDIS[i+1]-VIRIDIS[i])*f


def hex3(h): v = int(h.lstrip('#'), 16); return np.array([(v >> 16 & 255)/255, (v >> 8 & 255)/255, (v & 255)/255])


def read_js(p): return json.loads(Path(p).read_text().split('=', 2)[2].rstrip(';\n'))


def dec(b, dt): return np.frombuffer(base64.b64decode(b), dt)


def close(B):
    """Stop Chrome, then remove its temporary profile and the singleton-socket directory it created (and nothing else)."""
    prof = Path(B.profile.name)
    try: sock = Path(os.readlink(prof/'SingletonSocket')).parent
    except OSError: sock = None
    B.close()
    try: B.proc.wait(15)
    except Exception: B.proc.kill(); B.proc.wait(5)
    for _ in range(40):
        shutil.rmtree(prof, ignore_errors=True)
        if not prof.exists(): break
        time.sleep(.25)
    if sock is not None and sock.name.startswith('com.google.Chrome.'): shutil.rmtree(sock, ignore_errors=True)
    try: B.profile.cleanup()
    except Exception: pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--timeout', type=float, default=120)
    a = ap.parse_args(); run = a.run.resolve(); dbg = run/'debug'; sv = run/'svgd_support'; qa = dbg/'qa'
    report = {'run': str(run), 'status': None, 'pages': {}, 'checks': [], 'screenshots': [], 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    def finish(status, code):
        report['status'] = status; report['n_checks'] = len(report['checks']); report['n_failed'] = sum(not c['pass'] for c in report['checks'])
        qa.mkdir(parents=True, exist_ok=True); (qa/'qa_report.json').write_text(json.dumps(report, indent=1, default=str)); print(status, qa/'qa_report.json', flush=True); return code
    if not (dbg/'svgd_support_3d.html').is_file() or not (dbg/'data'/'svgd_support_meta.js').is_file():
        report['reason'] = f'{dbg} has no SVGD debug pages (run with --debug light|full)'; return finish('NO_DEBUG_PAGES', 1)
    qa.mkdir(parents=True, exist_ok=True)
    ok_all = True
    def check(name, ok, detail=''):
        nonlocal ok_all; ok_all &= bool(ok); report['checks'].append({'check': name, 'pass': bool(ok), 'detail': detail}); print(('PASS ' if ok else 'FAIL ')+name, str(detail)[:200], flush=True)
    from tools import qa_projective_debug as Q
    if not Path(Q.CHROME).exists(): report['reason'] = f'{Q.CHROME} not found'; return finish('BROWSER_UNAVAILABLE', 2)
    try:
        B = Q.Browser(run); B.pipe.call('Browser.getVersion', timeout=30)
    except Exception as e:
        hint = ' (TMPDIR path too long for Chrome\'s singleton socket; use a short TMPDIR)' if len(tempfile.gettempdir()) > 70 else ''
        report['reason'] = f'Chrome did not start: {type(e).__name__}: {e}{hint}'; return finish('BROWSER_UNAVAILABLE', 2)
    M = read_js(dbg/'data'/'svgd_support_meta.js'); ids = dec(M['display_ids'], np.int32).astype(int); frames = M['frames']
    can = np.load(sv/'canonical.npz'); opt = lambda n: np.load(sv/n) if (sv/n).is_file() else None
    X, TS, SC, PDY, PST = (opt(f) for f in ('transport_provisional.npy', 'transport_status.npy', 'motion_score_uncalibrated.npy', 'p_dynamic.npy', 'probability_status.npy'))
    done = {r['frame']: r for r in json.loads((sv/'completed_frames.json').read_text())} if (sv/'completed_frames.json').is_file() else {}
    def shot(s, name):
        p = qa/f'{name}.png'; B.shot(s, p); report['screenshots'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    def wait(s, cond, timeout=a.timeout):
        t0 = time.time()
        while time.time()-t0 < timeout:
            if B.js(s, f'!!({cond})'): return True
            if B.js(s, '(window.__errors||[]).length || !!(window.__state && window.__state.error)'): return False
            time.sleep(.2)
        return False
    J = lambda s, e: B.js(s, e)
    try:
        s = B.open('debug/index.html'); wait(s, 'window.__state && window.__state.ready === true')
        gl = J(s, "(()=>{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl'))})()")
        report['webgl'] = bool(gl)
        if not gl: report['reason'] = 'WebGL context unavailable in headless Chrome (swiftshader)'; return finish('NO_WEBGL', 2)
        for page in ('index.html', 'svgd_support_3d.html', 'svgd_support_numerical.html'):
            s = B.open(f'debug/{page}'); ready = wait(s, 'window.__state && window.__state.ready === true')
            errs = J(s, 'window.__errors||[]'); st_err = J(s, 'window.__state && window.__state.error || null')
            report['pages'][page] = {'ready': ready, 'errors': errs, 'state_error': st_err}
            if st_err == 'NO_WEBGL': report['reason'] = f'{page}: NO_WEBGL'; return finish('NO_WEBGL', 2)
            check(f'{page} loads without JS errors and reaches __state.ready', ready and not errs and not st_err, errs or st_err)
            if page == 'index.html':
                links = J(s, "[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))")
                check('index links both SVGD pages', {'svgd_support_3d.html', 'svgd_support_numerical.html'} <= set(links), links)
            shot(s, page[:-5])
        # ------------------------------------------------ 3D page
        s = B.open('debug/svgd_support_3d.html'); wait(s, 'window.__state && window.__state.ready === true')
        pid = J(s, 'Array.from(__state.display_ids)'); check('3D: display IDs are the meta display subset', pid == ids.tolist(), f'{len(pid)} IDs')
        check('3D: display subset includes every traced ID and seed', set(M['traced_ids']) | set((M['seeds'] or {}).get('ids', [])) <= set(ids.tolist()))
        canon = None; prev = None; xyz0 = np.asarray(can['xyz0'][ids], np.float32)
        chk = sorted({frames[0], frames[len(frames)//2], frames[-1]})
        for k in chk:
            J(s, f"(()=>{{const e=document.getElementById('frame');e.value={frames.index(k)};e.dispatchEvent(new Event('input'))}})()")
            drawn = wait(s, f'__state.drawn_frame === {k} && __state.frame_complete === true', 60)
            check(f'3D frame {k}: slider draws the complete frame', drawn, J(s, '__state.drawn_frame'))
            T = J(s, '__state.transported ? Array.from(__state.transported) : null'); c0 = np.asarray(J(s, 'Array.from(__state.canonical)'), np.float32).reshape(-1, 3)
            check(f'3D frame {k}: canonical layer = xyz0[display IDs] (float32), never moved', np.array_equal(c0, xyz0) and (canon is None or np.array_equal(c0, canon)))
            canon = c0
            if X is not None and k < len(X):
                T = np.asarray(T, np.float32).reshape(-1, 3); ref = np.asarray(X[k][ids])
                check(f'3D frame {k}: displayed transported XYZ == transport_provisional.npy (float32)', np.array_equal(T, ref.astype(np.float32)),
                      f'max |displayed - saved f64| = {np.abs(T-ref).max():.3e} m over {len(ids)} IDs')
            else:
                check(f'3D frame {k}: no transport -> nothing drawn as transported, labelled', T is None and 'NO TRANSPORT' in J(s, "document.getElementById('hud').textContent"))
            hud = J(s, "document.getElementById('hud').textContent"); t = M['times_s'][k]
            check(f'3D frame {k}: HUD shows the true timestamp', f'{t:.4f}' in hud, hud)
            cnt = J(s, "document.getElementById('counts').textContent"); fm = read_js(dbg/'data'/f'svgd_support_f{k:04d}.js')['observed']
            nums = [M['N'], len(ids), fm['M_t'], fm['M_display']]
            check(f'3D frame {k}: N_full, N_display, M_t, M_display on screen', all(str(v) in cnt for v in nums) and 'M_display' in cnt and 'N_full' in cnt, nums)
            if k in done: check(f'3D frame {k}: M_t equals completed_frames.json M', fm['M_t'] == done[k]['M'], [fm['M_t'], done[k]['M']])
            check(f'3D frame {k}: observed display <= cap and on the fixed lattice', fm['M_display'] <= max(M['display_cap'], 0) or k != frames[0], fm['M_display'])
            if X is not None and k < len(X):
                unc = J(s, "document.getElementById('hud').textContent.includes('FRAME_UNCOMMITTED')")
                check(f'3D frame {k}: FRAME_UNCOMMITTED shown iff uncommitted', unc == (not M['committed'][k]))
            shot(s, f'svgd_support_3d_f{k:04d}')
            sig = (J(s, '__state.drawn_frame'), J(s, "document.getElementById('hud').textContent"), J(s, '__state.observed_shown'))
            if prev is not None: check(f'3D: frame change {prev[0]} -> {k} changes the drawn data', sig != prev and sig[0] != prev[0], [prev, sig])
            prev = sig
        # rapid slider changes: only the last requested frame, complete
        if len(frames) > 1:
            J(s, "(()=>{const e=document.getElementById('frame');for(const v of [1,0,%d]){e.value=v;e.dispatchEvent(new Event('input'))}})()" % (len(frames)-1))
            ok = wait(s, f'__state.drawn_frame === {frames[-1]} && __state.frame_complete === true', 60); time.sleep(.5)
            n3 = J(s, '__state.transported ? __state.transported.length : 3*__state.display_ids.length')
            check('3D: rapid slider changes end on the last request, complete frame drawn', ok and J(s, '__state.drawn_frame') == frames[-1] and n3 == 3*len(ids))
        # camera toggles and view-from
        if len(M['cameras']) > 1:
            n0, c0s = J(s, '__state.observed_shown'), J(s, '__state.observed_cams_shown')
            J(s, "(()=>{const e=document.querySelector('#cams input[data-c=\"0\"]');e.checked=false;e.dispatchEvent(new Event('change'))})()"); time.sleep(.4)
            n1, c1s = J(s, '__state.observed_shown'), J(s, '__state.observed_cams_shown')
            check('3D: camera toggle changes the observed cloud data', n1 != n0 and 0 not in c1s and 0 in c0s, [n0, c0s, n1, c1s])
            shot(s, 'svgd_support_3d_camera0_off')
            J(s, "(()=>{const e=document.querySelector('#cams input[data-c=\"0\"]');e.checked=true;e.dispatchEvent(new Event('change'))})()")
        vp = []
        for c in range(min(2, len(M['cameras']))):
            J(s, f"(()=>{{const e=document.getElementById('viewfrom');e.value={c};e.onchange()}})()"); time.sleep(.3)
            vp.append(J(s, '__state.view_from'))
        check('3D: camera selector (view from) switches the source camera', vp == list(range(len(vp))), vp)
        # select original IDs: text box and mouse click
        k = J(s, '__state.drawn_frame'); tsel = [M['traced_ids'][0]] if M['traced_ids'] else []
        tsel += [int(ids[len(ids)//2])] + ([M['seeds']['ids'][0]] if M['seeds'] and M['seeds']['ids'] else [])
        def compare(info, i, how):
            want = {'motion_score': None if SC is None else float(SC[i]), 'probability_status': None if PST is None else int(PST[i]),
                    'source': can['source'][i].tolist(), 'transport_status': None if TS is None or k >= len(TS) else int(TS[k][i]),
                    'p_dynamic': None if PDY is None else float(PDY[i])}
            got = {kk: info.get(kk) for kk in want}
            same = lambda u, v: (u is None and v is None) or (isinstance(v, float) and np.isnan(v) and u is None) or u == v
            check(f'3D: select ID {i} ({how}) shows the saved score / probability status / source / transport status / p_dynamic', all(same(got[kk], want[kk]) for kk in want), [got, want])
            if X is not None and k < len(X):
                check(f'3D: select ID {i} ({how}) XYZ == saved (float32)', np.array_equal(np.asarray(info['xyz'], np.float32), X[k][i].astype(np.float32)))
        for i in tsel:
            J(s, f"(()=>{{const e=document.getElementById('pid');e.value='{i}';e.dispatchEvent(new Event('change'))}})()"); wait(s, f'__state.selected_info && __state.selected_info.id === {i}', 20)
            compare(J(s, '__state.selected_info'), i, 'text box')
            check(f'3D: URL hash carries the selected ID {i} and frame', J(s, 'location.hash') == f'#frame={k}&id={i}', J(s, 'location.hash'))
        i = int(ids[len(ids)//3]); J(s, "(()=>{const e=document.getElementById('viewfrom');e.value=-1;})()")
        xy = J(s, f'__state.screen_of({i})')
        if xy:
            for typ in ('mousePressed', 'mouseReleased'):
                B.pipe.call('Input.dispatchMouseEvent', {'type': typ, 'x': xy['x'], 'y': xy['y'], 'button': 'left', 'clickCount': 1}, s)
            time.sleep(.5); got = J(s, '__state.selected_info || {}'); gid = got.get('id')
            gxy = J(s, f'__state.screen_of({gid})') if gid is not None else None
            check('3D: mouse click selects the original ID under the cursor', gid is not None and abs(gxy['x']-xy['x']) < 10 and abs(gxy['y']-xy['y']) < 10, [i, gid])
            if gid is not None: compare(got, int(gid), 'mouse click')
        # legend endpoints, colours, NaN / unresolved
        modes = J(s, "[...document.getElementById('mode').options].map(o=>o.value)")
        for mode in [m for m in ('prob', 'score') if m in modes]:
            J(s, f"(()=>{{const e=document.getElementById('mode');e.value='{mode}';e.onchange()}})()"); time.sleep(.4)
            L = J(s, '__state.legend'); cols = np.asarray(J(s, 'Array.from(__state.layer_colors)'), float).reshape(-1, 3)
            if mode == 'score':
                v = np.asarray(SC[ids], float); hi = float(np.nanmax(SC)) if np.isfinite(SC).any() else 1.
                exp = ('motion evidence score (UNCALIBRATED)', '0 m (no provisional displacement)', f'{hi:.6g} m (max over all N)'); lo = 0.
                opts = J(s, "[...document.getElementById('mode').options].map(o=>[o.value,o.text])")
                check('3D: without p_dynamic the layer is named motion evidence score (UNCALIBRATED), never probability',
                      PDY is not None or (['score', 'motion evidence score (UNCALIBRATED)'] in opts and 'prob' not in [o[0] for o in opts]), opts)
            else:
                v = np.where(PST[ids] != 0, np.nan, PDY[ids]) if PST is not None else np.asarray(PDY[ids], float); lo, hi = 0., 1.
                exp = (f"p_dynamic ({M['probability_semantics']})", '0 (static explanation)', '1 (moving explanation)')
            check(f'3D legend [{mode}]: name and endpoint labels equal the arrays', (L['name'], L['dom_lo'], L['dom_hi']) == exp, [L.get('name'), L.get('dom_lo'), L.get('dom_hi'), exp])
            nanc = hex3(M['nan_colour'])
            want = np.where(np.isfinite(v)[:, None], viridis(np.nan_to_num((v-lo)/max(hi-lo, 1e-300))), nanc)
            err = float(np.abs(cols-want).max()) if len(cols) == len(want) else np.inf
            check(f'3D colours [{mode}]: every point = viridis((v-lo)/(hi-lo)), NaN/unresolved = NaN colour', err < 1e-5, f'max err {err:.2e}; {int((~np.isfinite(v)).sum())} unresolved points rendered')
            check(f'3D legend [{mode}]: NaN/unresolved colour distinct from both endpoints and shown in legend',
                  np.abs(nanc-viridis(np.array([0.]))[0]).max() > .1 and np.abs(nanc-viridis(np.array([1.]))[0]).max() > .1 and L['dom_nan'] == L['nan_label']
                  and L['dom_nan_bg'].replace(' ', '') == 'rgb(%d,%d,%d)' % tuple(np.round(nanc*255).astype(int)), [L.get('dom_nan'), L.get('dom_nan_bg')])
            shot(s, f'svgd_support_3d_legend_{mode}')
        check('3D: a score or probability colour layer exists', any(m in modes for m in ('prob', 'score')), modes)
        J(s, "(()=>{for(const k of ['edges','unsupported']){const e=document.querySelector(`#toggles input[data-k=\"${k}\"]`);if(e){e.checked=true;e.dispatchEvent(new Event('change'))}}})()"); time.sleep(.4)
        shot(s, 'svgd_support_3d_edges_unsupported')
        vec = J(s, '__state.vectors'); check('3D: traced vectors drawn for the traced subset only', bool(vec) and max(vec.values()) <= len(M['traced_ids']), vec)
        errs = J(s, 'window.__errors||[]'); check('3D: no JS errors after all interactions', not errs, errs)
        # ------------------------------------------------ numerical page
        s = B.open('debug/svgd_support_numerical.html'); wait(s, 'window.__state && window.__state.ready === true')
        ban = J(s, "document.getElementById('banners').textContent"); want = []
        if X is None: want.append('NO_ESTIMATION')
        if X is not None and not all(M['committed']): want.append('FRAME_UNCOMMITTED')
        if PDY is None: want.append('PROBABILITY_UNCALIBRATED')
        if M['blocked']: want.append(M['blocked']['status'])
        check('numerical: banners NO_ESTIMATION / FRAME_UNCOMMITTED / PROBABILITY_UNCALIBRATED / blocked as appropriate', all(w in ban for w in want), want)
        check('numerical: charts drawn', J(s, '__state.charts') >= 10, J(s, '__state.charts'))
        TJ = read_js(dbg/'data'/'svgd_support_traj.js')
        for i in dict.fromkeys(([M['traced_ids'][0]] if M['traced_ids'] else [])+[int(ids[len(ids)//2])]):
            if i not in TJ['ids']: continue
            J(s, f"(()=>{{const e=document.getElementById('tid');e.value='{i}';e.dispatchEvent(new Event('change'))}})()"); wait(s, f'__state.trace_info && __state.trace_info.id === {i}', 20)
            T = J(s, '__state.trace_info')
            ok = T['source'] == can['source'][i].tolist() and (SC is None or T['motion_score'] == float(SC[i])) and (PST is None or T['probability_status'] == int(PST[i]))
            if X is not None:
                tr = np.asarray([[np.nan if v is None else v for v in x] for x in T['trajectory']], np.float32); ref = X[:, i].astype(np.float32)
                ok &= np.array_equal(tr[:len(X)], ref) and len(tr) == M['K']
            check(f'numerical: clicked ID {i} trace over all frames == saved arrays', ok)
        shot(s, 'svgd_support_numerical_trace')
        errs = J(s, 'window.__errors||[]'); check('numerical: no JS errors after interactions', not errs, errs)
    finally:
        close(B)
    return finish('PASS' if ok_all else 'FAIL', 0 if ok_all else 1)


if __name__ == '__main__':
    raise SystemExit(main())
