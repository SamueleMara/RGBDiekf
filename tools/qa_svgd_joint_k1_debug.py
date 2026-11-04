#!/usr/bin/env python3
"""Headless-Chrome QA of the joint K = 1 debug pages (debug/svgd_joint_k1_debug.py), checked against the SAVED arrays of RUN/svgd_joint_k1.

    TMPDIR=/tmp/claude-1000 python -B tools/qa_svgd_joint_k1_debug.py --run RUN

Every page loads without JavaScript errors and reaches window.__state.ready === true. 3D page: the display IDs equal the meta subset;
reference xyz / gamma / map_gamma / B / identifiability shown equal the .npy / .npz values at the displayed IDs (float32); for frames
{0, mid, last}: the slider draws that frame, the HUD shows the true timestamp, N_full / N_display / M_t / M_display are on screen, the
observed cloud drawn equals the data payload and its rebuilt sha256 matches loader_counts.json, and the moving subset transformed by the
MAP pose and by every SVGD particle pose equals pose @ x_ref computed here with NumPy (float32 tolerance); a frame change changes the drawn
data; legend: gamma mode titled "UNCALIBRATED P(moving)" with 0..1 endpoints, every point colour = viridis(gamma) where identifiability == 1
and = the distinct no-information colour where identifiability == 0 (swatch rendered in the legend), the same for B (diverging).
Numerical page: banners (probability semantics, PREFIX_DIAGNOSTIC when applicable), charts drawn, F traces equal optimizer_diagnostics.json,
histograms equal numpy histograms of the arrays, pose traces equal numpy pose summaries, rho equals calibration_provenance.json, oracle
table present iff an oracle folder exists. Screenshots + sha256 into RUN/debug_k1/qa/, report RUN/debug_k1/qa/qa_report.json.
status: PASS | FAIL | BROWSER_UNAVAILABLE | NO_WEBGL | NO_DEBUG_PAGES (exit 0 | 1 | 2 | 2 | 1); never silent success.
"""
import argparse, hashlib, json, tempfile, time
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qa_svgd_support_debug import read_js, dec, viridis, hex3, close

PAGES = ('index.html', 'k1_3d.html', 'k1_numerical.html')


def diverging(t):
    t = np.clip(t, -1, 1)[:, None]; a, m, b = np.array([.35, .65, 1]), np.array([.55, .58, .62]), np.array([.97, .32, .29])
    return np.where(t < 0, m+(a-m)*(-t), m+(b-m)*t)


def rot_deg(T): return np.degrees(np.arccos(np.clip((np.trace(T[..., :3, :3], axis1=-2, axis2=-1)-1)/2, -1, 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--timeout', type=float, default=120)
    a = ap.parse_args(argv); run = a.run.resolve(); dbg = run/'debug_k1'; sv = run/'svgd_joint_k1'; qa = dbg/'qa'
    report = {'run': str(run), 'status': None, 'pages': {}, 'checks': [], 'screenshots': [], 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    def finish(status, code):
        report['status'] = status; report['n_checks'] = len(report['checks']); report['n_failed'] = sum(not c['pass'] for c in report['checks'])
        qa.mkdir(parents=True, exist_ok=True); (qa/'qa_report.json').write_text(json.dumps(report, indent=1, default=str)); print(status, qa/'qa_report.json', flush=True); return code
    if not all((dbg/p).is_file() for p in PAGES) or not (dbg/'data'/'k1_meta.js').is_file():
        report['reason'] = f'{dbg} has no joint K1 debug pages (debug/svgd_joint_k1_debug.py --run RUN)'; return finish('NO_DEBUG_PAGES', 1)
    qa.mkdir(parents=True, exist_ok=True); ok_all = True
    def check(name, ok, detail=''):
        nonlocal ok_all; ok_all &= bool(ok); report['checks'].append({'check': name, 'pass': bool(ok), 'detail': detail}); print(('PASS ' if ok else 'FAIL ')+name, str(detail)[:200], flush=True)
    from tools import qa_projective_debug as Q
    if not Path(Q.CHROME).exists(): report['reason'] = f'{Q.CHROME} not found'; return finish('BROWSER_UNAVAILABLE', 2)
    try:
        B = Q.Browser(run); B.pipe.call('Browser.getVersion', timeout=30)
    except Exception as e:
        hint = ' (TMPDIR path too long for Chrome\'s singleton socket; use TMPDIR=/tmp/claude-1000)' if len(tempfile.gettempdir()) > 70 else ''
        report['reason'] = f'Chrome did not start: {type(e).__name__}: {e}{hint}'; return finish('BROWSER_UNAVAILABLE', 2)
    M = read_js(dbg/'data'/'k1_meta.js'); NU = read_js(dbg/'data'/'k1_numeric.js'); ids = dec(M['display_ids'], np.int32).astype(int); K = M['K']
    L = lambda n: np.load(sv/n) if (sv/n).is_file() else None
    cat = np.load(sv/'reference_catalogue.npz'); xyz = np.asarray(cat['xyz'], float)
    gam, MG, Bv, ident = L('gamma_provisional.npy'), L('map_gamma.npy'), L('log_evidence_ratio.npy'), L('identifiability_status.npy')
    PP, MP = L('pose_particles.npy'), L('map_pose.npy'); lc = {int(r['frame']): r for r in (json.loads((sv/'loader_counts.json').read_text()) if (sv/'loader_counts.json').is_file() else [])}
    st = json.loads((sv/'status.json').read_text())
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
    arr = lambda s, e, dt=np.float64: np.asarray(J(s, f'(()=>{{const v={e};return v?Array.from(v):null}})()') or [], dt)
    try:
        s = B.open('debug_k1/index.html'); wait(s, 'window.__state && window.__state.ready === true')
        gl = J(s, "(()=>{const c=document.createElement('canvas');return !!(c.getContext('webgl2')||c.getContext('webgl'))})()"); report['webgl'] = bool(gl)
        if not gl: report['reason'] = 'WebGL context unavailable in headless Chrome (swiftshader)'; return finish('NO_WEBGL', 2)
        for page in PAGES:
            s = B.open(f'debug_k1/{page}'); ready = wait(s, 'window.__state && window.__state.ready === true')
            errs = J(s, 'window.__errors||[]'); st_err = J(s, 'window.__state && window.__state.error || null')
            report['pages'][page] = {'ready': ready, 'errors': errs, 'state_error': st_err}
            if st_err == 'NO_WEBGL': report['reason'] = f'{page}: NO_WEBGL'; return finish('NO_WEBGL', 2)
            check(f'{page} loads without JS errors and reaches __state.ready', ready and not errs and not st_err, errs or st_err)
            if page == 'index.html':
                links = J(s, "[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))"); check('index links both K1 pages', {'k1_3d.html', 'k1_numerical.html'} <= set(links), links)
            shot(s, page[:-5])
        # ------------------------------------------------ 3D page
        s = B.open('debug_k1/k1_3d.html'); wait(s, 'window.__state && window.__state.ready === true')
        pid = arr(s, '__state.display_ids', int); check('3D: display IDs are the meta display subset', np.array_equal(pid, ids), f'{len(pid)} IDs of N {len(xyz)}')
        X32 = arr(s, '__state.ref_xyz', np.float32).reshape(-1, 3)
        check('3D: reference xyz == reference_catalogue.npz xyz[display IDs] (float32)', np.array_equal(X32, xyz[ids].astype(np.float32)), f'max err {np.abs(X32-xyz[ids]).max():.2e} m')
        for nm, ref, key in (('gamma_provisional', gam, 'gamma'), ('map_gamma', MG, 'map_gamma'), ('log_evidence_ratio', Bv, 'B')):
            if ref is None: continue
            v = arr(s, f'__state.{key}', np.float32); check(f'3D: {key} == {nm}.npy[display IDs] (float32)', np.array_equal(v, ref[ids].astype(np.float32)), f'{len(v)} values')
        I = arr(s, '__state.identifiability', int); check('3D: identifiability == identifiability_status.npy[display IDs]', np.array_equal(I, ident[ids].astype(int)), f'{int(I.sum())} informative displayed')
        fr = sorted({0, (K-1)//2, K-1}); prev = None
        J(s, "(()=>{const e=document.getElementById('subset');e.value='all';e.onchange()})()"); time.sleep(.3)
        for k in fr:
            J(s, f"(()=>{{const e=document.getElementById('frame');e.value={k};e.dispatchEvent(new Event('input'))}})()")
            drawn = wait(s, f'__state.drawn_frame === {k} && __state.frame_complete === true', 60); check(f'3D frame {k}: slider draws the complete frame', drawn, J(s, '__state.drawn_frame'))
            hud = J(s, "document.getElementById('hud').textContent"); t = M['times_s'][k]
            check(f'3D frame {k}: HUD shows the true timestamp {t:.4f} s', f'{t:.4f}' in hud, hud)
            obs = [o for o in M['observed']['frames'] if o['k'] == k]; cnt = J(s, "document.getElementById('counts').textContent")
            nums = [M['N'], len(ids)] + ([obs[0]['M_t'], obs[0]['M_display']] if obs else [])
            check(f'3D frame {k}: N_full / N_display / M_t / M_display on screen', all(str(v) in cnt for v in nums) and 'N_full' in cnt and 'N_display' in cnt and 'M_display' in cnt, nums)
            if obs:
                P = read_js(dbg/'data'/f'k1_obs_f{k:04d}.js'); ox = arr(s, '__state.observed', np.float32)
                check(f'3D frame {k}: observed cloud drawn == payload ({P["M_display"]} of M_t {P["M_t"]})', np.array_equal(ox, dec(P['xyz'], np.float32)) and len(ox) == 3*P['M_display'])
                if k in lc: check(f'3D frame {k}: rebuilt observed cloud sha256 == loader_counts.json', P['cloud_sha256'] == lc[k]['cloud_sha256'] and P['M_t'] == lc[k]['M'], [P['M_t'], lc[k]['M']])
            tr = J(s, '(()=>{const t=__state.transformed;return {ids:t.ids, map:t.map?Array.from(t.map):null, particles:t.particles.map(a=>Array.from(a)), gt:t.gt?Array.from(t.gt):null, gt_ids:t.gt_ids||null}})()')
            sid = np.asarray(tr['ids'], int); check(f'3D frame {k}: moving subset "all displayed" = every display ID', np.array_equal(sid, ids))
            x32 = xyz[sid].astype(np.float32).astype(float)
            if MP is not None:
                want = x32@MP[k, :3, :3].T+MP[k, :3, 3]; got = np.asarray(tr['map'], float).reshape(-1, 3); err = float(np.abs(got-want).max()) if got.shape == want.shape else np.inf
                check(f'3D frame {k}: MAP-transformed positions == map_pose[k] @ x_ref (float32 tol)', err < 1e-5*max(1, np.abs(want).max()), f'max err {err:.2e} m')
            errs_p = []
            for m in range(len(PP)):
                want = x32@PP[m, k, :3, :3].T+PP[m, k, :3, 3]; got = np.asarray(tr['particles'][m], float).reshape(-1, 3)
                errs_p.append(float(np.abs(got-want).max()) if got.shape == want.shape else np.inf)
            check(f'3D frame {k}: every SVGD particle transform == pose_particles[m,k] @ x_ref (float32 tol)', len(errs_p) == len(PP) and max(errs_p) < 1e-5*max(1, np.abs(x32).max()+1), f'max err {max(errs_p):.2e} m over {len(PP)} particles')
            if M['gt']['available'] and tr['gt'] is not None:
                G = np.asarray(M['gt']['poses'])[k]; gi = np.asarray(tr['gt_ids'], int); want = xyz[gi].astype(np.float32).astype(float)@G[:3, :3].T+G[:3, 3]
                got = np.asarray(tr['gt'], float).reshape(-1, 3); check(f'3D frame {k}: GT (POST-HOC) transform == GT pose @ x_ref', got.shape == want.shape and (not len(want) or np.abs(got-want).max() < 1e-5))
            shot(s, f'k1_3d_f{k:04d}')
            sig = (J(s, '__state.drawn_frame'), hud, None if not tr['particles'] else tr['particles'][0][:30], J(s, '__state.observed ? __state.observed.length : null'), J(s, '__state.observed ? Array.from(__state.observed.slice(0,30)) : null'))
            if prev is not None: check(f'3D: frame change {prev[0]} -> {k} changes the drawn data', sig[0] != prev[0] and sig[1] != prev[1] and (sig[2] != prev[2] or sig[4] != prev[4]), [prev[0], sig[0]])
            prev = sig
        if K > 1:
            J(s, "(()=>{const e=document.getElementById('frame');for(const v of [1,0,%d]){e.value=v;e.dispatchEvent(new Event('input'))}})()" % (K-1))
            ok = wait(s, f'__state.drawn_frame === {K-1} && __state.frame_complete === true', 60); time.sleep(.5)
            check('3D: rapid slider changes end on the last request, complete frame drawn', ok and J(s, '__state.drawn_frame') == K-1)
        # legends: gamma (viridis + no-information colour), B (diverging + no-information colour)
        nanc = hex3(M['noinfo_colour'])
        for mode in [m for m in ('gamma', 'B', 'map_gamma') if m in M['layers']]:
            J(s, f"(()=>{{const e=document.getElementById('mode');e.value='{mode}';e.onchange()}})()"); time.sleep(.4)
            Lg = J(s, '__state.legend'); cols = arr(s, '__state.layer_colors').reshape(-1, 3); Ld = M['layers'][mode]
            if mode == 'gamma': v, info = gam[ids].astype(np.float32).astype(float), ident[ids].astype(bool)
            elif mode == 'map_gamma': v = MG[ids].astype(np.float32).astype(float); mi = M.get('map_identifiability'); info = dec(mi, np.uint8).astype(bool) if mi else np.ones(len(ids), bool)
            else: v, info = Bv[ids].astype(np.float32).astype(float), ident[ids].astype(bool)
            base = viridis(v) if Ld['kind'] == 'viridis' else diverging(v/max(Ld['hi'], 1e-300))
            want = np.where((info & np.isfinite(v))[:, None], base, nanc); err = float(np.abs(cols-want).max()) if cols.shape == want.shape else np.inf
            check(f'3D colours [{mode}]: informative = colour map(value), identifiability 0 = no-information colour', err < 1e-5, f'max err {err:.2e}; {int((~info).sum())} no-information points drawn')
            if mode in ('gamma', 'map_gamma'):
                check(f'3D legend [{mode}]: titled "UNCALIBRATED P(moving)" with 0..1 endpoints', Lg['title'] == 'UNCALIBRATED P(moving)' and Lg['lo_label'] == '0' and Lg['hi_label'] == '1'
                      and 'UNCALIBRATED P(moving)' in Lg['text'], [Lg.get('title'), Lg.get('lo_label'), Lg.get('hi_label')])
            if Lg.get('noinfo_label') is not None or mode != 'map_gamma':
                check(f'3D legend [{mode}]: no-information swatch rendered, distinct from both colour-map endpoints',
                      Lg.get('noinfo_bg') is not None and Lg['noinfo_bg'].replace(' ', '') == 'rgb(%d,%d,%d)' % tuple(np.round(nanc*255).astype(int))
                      and 'no information' in (Lg.get('noinfo_label') or '') and min(np.abs(nanc-viridis(np.array([0.]))[0]).max(), np.abs(nanc-viridis(np.array([1.]))[0]).max()) > .1,
                      [Lg.get('noinfo_label'), Lg.get('noinfo_bg')])
            shot(s, f'k1_3d_legend_{mode}')
        # selection of a displayed ID shows the saved values
        i = int(ids[len(ids)//2]); J(s, f"(()=>{{const e=document.getElementById('pid');e.value='{i}';e.dispatchEvent(new Event('change'))}})()")
        wait(s, f'__state.selected_info && __state.selected_info.id === {i}', 20); info_ = J(s, '__state.selected_info')
        check(f'3D: selecting reference ID {i} shows its saved gamma / identifiability', info_ and info_.get('displayed') and np.float32(info_['gamma']) == np.float32(gam[i]) and info_['identifiability'] == int(ident[i]), info_)
        # seed subset (or gamma threshold) still consistent with saved seed indices
        seeds = L('moving_seed_indices.npy'); J(s, "(()=>{const e=document.getElementById('subset');e.value='seeds';e.onchange()})()"); time.sleep(.3)
        sid = np.asarray(J(s, '__state.transformed.ids'), int)
        check('3D: seed subset = moving_seed_indices.npy ∩ display IDs', np.array_equal(np.sort(sid), np.intersect1d(seeds, ids)), f'{len(sid)} displayed of {len(seeds)} seeds')
        errs = J(s, 'window.__errors||[]'); check('3D: no JS errors after all interactions', not errs, errs)
        # ------------------------------------------------ numerical page
        s = B.open('debug_k1/k1_numerical.html'); wait(s, 'window.__state && window.__state.ready === true')
        ban = J(s, "document.getElementById('banners').textContent"); want = [st.get('probability_semantics', 'UNCALIBRATED_APPROXIMATION')]
        if st.get('sequence_scope') == 'PREFIX_DIAGNOSTIC': want.append('PREFIX_DIAGNOSTIC')
        check('numerical: banners show probability semantics (and PREFIX_DIAGNOSTIC when applicable)', all(w in ban for w in want), want)
        check('numerical: charts drawn', J(s, '__state.charts') >= 15, J(s, '__state.charts'))
        S = J(s, '__state.series'); od = json.loads((sv/'optimizer_diagnostics.json').read_text())
        fm = np.asarray([r['F_mean'] for r in od['svgd']]); fx = np.asarray([r['F_max'] for r in od['svgd']])
        check('numerical: SVGD F mean / max traces == optimizer_diagnostics.json', np.allclose(S['svgd_F'][0]['y'], fm, rtol=0, atol=1e-9) and np.allclose(S['svgd_F'][1]['y'], fx, rtol=0, atol=1e-9))
        check('numerical: F identity line == optimizer_diagnostics F_identity', any(x.get('label') == 'F identity' and np.allclose(x['y'], od['F_identity']) for x in S['svgd_F']))
        if od.get('map'):
            starts = list(dict.fromkeys(r['start'] for r in od['map'])); ok = True
            for j, sn in enumerate(starts): ok &= np.allclose(S['map_F'][j]['y'], [r['F'] for r in od['map'] if r['start'] == sn])
            check('numerical: MAP traces per start == optimizer_diagnostics.json', ok, starts)
        pi = np.asarray([r['pi_mean'] for r in od['svgd']]); check('numerical: pi_mean trace == optimizer_diagnostics.json', np.allclose(S['pi'][0]['y'], pi))
        ed = np.linspace(0, 1, 41); h = S['gamma_hist']; ib = ident.astype(bool)
        check('numerical: gamma histogram (informative / no information) == numpy histogram of gamma_provisional.npy',
              h['sets'][0]['counts'] == np.histogram(gam[ib], ed)[0].tolist() and h['sets'][1]['counts'] == np.histogram(gam[~ib], ed)[0].tolist())
        bh = NU['hist']['B']; be = np.asarray(bh['edges']); bl = bh['clip']
        check('numerical: B histogram == numpy histogram of log_evidence_ratio.npy (clipped tails)', S['B_hist']['sets'][0]['counts'] == np.histogram(np.clip(Bv[ib], -bl, bl), be)[0].tolist()
              and S['B_hist']['sets'][1]['counts'] == np.histogram(np.clip(Bv[~ib], -bl, bl), be)[0].tolist() and sum(map(sum, (x['counts'] for x in S['B_hist']['sets']))) == len(Bv))
        tn = np.linalg.norm(PP[..., :3, 3], axis=-1)*1e3; ra = rot_deg(PP)
        ok = all(np.allclose(S['pose_t'][m]['y'], tn[m], atol=1e-9) and np.allclose(S['pose_r'][m]['y'], ra[m], atol=1e-7) for m in range(len(PP)))
        if MP is not None: ok &= np.allclose(S['pose_t'][len(PP)]['y'], np.linalg.norm(MP[:, :3, 3], axis=-1)*1e3, atol=1e-9) and np.allclose(S['pose_r'][len(PP)]['y'], rot_deg(MP), atol=1e-7)
        check('numerical: pose traces (|t_k|, angle) of every particle and MAP == numpy from the saved poses', ok)
        check('numerical: pose traces use the true timestamps', np.allclose(S['pose_t'][0]['x'], M['times_s']))
        cal = json.loads((sv/'calibration_provenance.json').read_text())
        check('numerical: rho per frame == calibration_provenance.json', np.allclose(S['rho'][0]['y'], [r['rho'] for r in cal['rho']]), [r['rho'] for r in cal['rho']])
        orc = J(s, '__state.oracle'); has = bool(NU.get('oracle'))
        if has and M['gt']['available'] and 'gt_moving' in NU['oracle']:
            check('GT (POST-HOC) membership recomputed for display == oracle.json gt_moving count', M['gt']['n_moving_full'] == NU['oracle']['gt_moving'], [M['gt']['n_moving_full'], NU['oracle']['gt_moving']])
        check('numerical: oracle table present iff an oracle folder exists', (orc is not None) == has and (not has or orc['status'] == NU['oracle']['status']), orc)
        shot(s, 'k1_numerical_full')
        J(s, "(()=>{const e=document.getElementById('ys');e.value='linear';e.onchange()})()"); time.sleep(.3); shot(s, 'k1_numerical_linear_hist')
        errs = J(s, 'window.__errors||[]'); check('numerical: no JS errors after interactions', not errs, errs)
    except Exception as e:                       # a QA crash is a failure, never a silent success
        import traceback; check(f'QA raised {type(e).__name__}', False, traceback.format_exc()[-1500:])
    finally:
        close(B)
    return finish('PASS' if ok_all else 'FAIL', 0 if ok_all else 1)


if __name__ == '__main__':
    raise SystemExit(main())
