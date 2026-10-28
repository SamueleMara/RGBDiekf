#!/usr/bin/env python3
"""Gate H: headless-Chrome QA of the four-pass debug pages, with numerical checks against independent data.

For each page: loads, no JavaScript error, state.ready. Screenshots (debug/qa/*.png) of the measured RGB-D pages at
the debug frames (0, 1, 5, 10, last when present) for both directions, and of every other page. Numerical checks:
  * the RGB underlay drawn by the page equals the source RGB file pixel values (sampled pixels, read back in the page)
  * displayed posterior marker coordinates equal pi(T_cw x) computed here with NumPy from the saved posterior states and
    the calibration (independent of the torch code that produced the page data)
  * changing frame / camera / 3D slider changes the displayed data
  * a backward page labels the frame-T endpoint as canonical (and frame 0 not)
Writes debug/qa/qa_report.json; exit 1 when a check fails.
"""
import argparse, base64, functools, http.server, io, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qa_free_viewer import Pipe

CHROME = '/usr/bin/google-chrome'


class Browser:
    def __init__(self, root):
        class H(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a): pass
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(H, directory=str(root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.profile = tempfile.TemporaryDirectory()
        ri, wi = os.pipe(); ro, wo = os.pipe()
        cmd = [CHROME, '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run', '--enable-unsafe-swiftshader', '--use-gl=angle',
               '--use-angle=swiftshader', f'--user-data-dir={self.profile.name}', '--window-size=1600,1000', '--remote-debugging-pipe', 'about:blank']
        wrapper = 'import os,sys\nos.dup2(int(sys.argv[1]),3)\nos.dup2(int(sys.argv[2]),4)\nos.execv(sys.argv[3],sys.argv[3:])'
        self.proc = subprocess.Popen([sys.executable, '-c', wrapper, str(ri), str(wo), *cmd], pass_fds=(ri, wo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(ri); os.close(wo); self.pipe = Pipe(ro, wi)

    def open(self, rel):
        t = self.pipe.call('Target.createTarget', {'url': 'about:blank'})['targetId']
        s = self.pipe.call('Target.attachToTarget', {'targetId': t, 'flatten': True})['sessionId']
        self.pipe.call('Page.enable', session=s); self.pipe.call('Runtime.enable', session=s)
        self.pipe.call('Page.addScriptToEvaluateOnNewDocument', {'source': "window.__errors=[];addEventListener('error',e=>__errors.push(String(e.message||e)));"
                       "addEventListener('unhandledrejection',e=>__errors.push(String(e.reason)));"}, s)
        self.pipe.call('Page.navigate', {'url': f'http://127.0.0.1:{self.server.server_port}/{rel}'}, s)
        return s

    def js(self, s, expr, timeout=60):
        r = self.pipe.call('Runtime.evaluate', {'expression': expr, 'awaitPromise': True, 'returnByValue': True}, s, timeout=timeout)
        if 'exceptionDetails' in r: raise RuntimeError(str(r['exceptionDetails'])[:500])
        return r['result'].get('value')

    def wait_ready(self, s, timeout=120):
        t0 = time.time()
        while time.time()-t0 < timeout:
            if self.js(s, 'window.__state && window.__state.ready === true'): return True
            if self.js(s, '(window.__errors||[]).length'): return False
            time.sleep(.3)
        return False

    def shot(self, s, path):
        Image.open(io.BytesIO(base64.b64decode(self.pipe.call('Page.captureScreenshot', {'format': 'png'}, s)['data']))).save(path)

    def close(self):
        self.proc.terminate(); self.server.shutdown()


def project(T_cw, K, x):
    X = x@T_cw[:3, :3].T+T_cw[:3, 3]
    return np.c_[K[0]*X[:, 0]/X[:, 2]+K[2], K[1]*X[:, 1]/X[:, 2]+K[3]]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True); ap.add_argument('--scene', type=Path)
    a = ap.parse_args(); run = a.run.resolve(); dbg = run/'debug'; qa = dbg/'qa'; qa.mkdir(exist_ok=True)
    man = json.loads((run/'run_manifest.json').read_text())
    cams = man['cameras']['cameras']
    dirs = ['forward', 'backward'] if man['arguments']['direction'] == 'both' else [man['arguments']['direction']]
    K = len(man['dataset']['original_frame_ids'])
    report = {'pages': {}, 'checks': [], 'screenshots': []}
    ok_all = True
    def check(name, ok, detail):
        nonlocal ok_all; ok_all &= bool(ok); report['checks'].append({'check': name, 'pass': bool(ok), 'detail': detail}); print(('PASS ' if ok else 'FAIL ')+name, detail, flush=True)
    B = Browser(run)
    try:
        for page in sorted(p.name for p in dbg.glob('*.html')):
            s = B.open(f'debug/{page}'); ready = B.wait_ready(s)
            errs = B.js(s, 'window.__errors||[]')
            report['pages'][page] = {'ready': ready, 'errors': errs}
            check(f'{page} loads without errors', ready and not errs, errs)
            png = qa/f'{page[:-5]}.png'; B.shot(s, png); report['screenshots'].append(str(png))
        for d in dirs:
            meta = json.loads((dbg/'data'/f'free_{d}_meta.js').read_text().split('=', 2)[2].rstrip(';\n'))
            X = np.load(run/f'free_{d}'/'posterior_xyz.npy', mmap_mode='r')
            want = [k for k in (0, 1, 5, 10, K-1) if k in meta['debug_frames']]
            for k in want:
                for c in (0, len(cams)//2):
                    s = B.open(f'debug/free_{d}_rgbd.html?frame={k}&camera={c}'); B.wait_ready(s)
                    st = B.js(s, 'JSON.stringify({f: __state.frame, c: __state.camera, m: __state.markers.slice(0, 400), img: __state.image})')
                    st = json.loads(st)
                    check(f'free {d} rgbd frame {k} camera {c}: page shows the requested frame/camera', st['f'] == k and st['c'] == c, [st['f'], st['c']])
                    if st['m']:
                        ids = np.array([m['id'] for m in st['m']]); uv = np.array([[m['u'], m['v']] for m in st['m']])
                        ref = project(np.asarray(cams[c]['T_cw']), np.asarray(cams[c]['intrinsics_fx_fy_cx_cy']), np.asarray(X[k][ids], float))
                        err = float(np.abs(uv-ref).max())
                        check(f'free {d} frame {k} cam {c}: markers = pi(T_cw x) (NumPy)', err < 1e-2, f'max |du,dv| = {err:.2e} px over {len(ids)} markers')
                    # RGB underlay pixel values equal the source file
                    rec = json.loads((dbg/'data'/f'free_{d}_rgbd_f{k:04d}.js').read_text().split('=', 2)[2].rstrip(';\n'))['cameras'][c]
                    pix = B.js(s, """(async()=>{const im=document.getElementById('img_rgb');await im.decode();const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;
                        const x=c.getContext('2d');x.drawImage(im,0,0);const out=[];for(let i=0;i<50;i++){const u=(i*97)%c.width,v=(i*61)%c.height;const d=x.getImageData(u,v,1,1).data;out.push([u,v,d[0],d[1],d[2]]);}return out;})()""")
                    src = rec.get('rgb_source')
                    if src and Path(src).exists():
                        arr = np.asarray(Image.open(src).convert('RGB'))
                        bad = [p for p in pix if tuple(arr[p[1], p[0]]) != tuple(p[2:])]
                        check(f'free {d} frame {k} cam {c}: RGB underlay equals source file', not bad, f'{len(pix)} pixels compared, {len(bad)} differ; {src}')
                    png = qa/f'free_{d}_rgbd_f{k:04d}_c{c}.png'; B.shot(s, png); report['screenshots'].append(str(png))
            # camera / frame changes really change the markers
            s = B.open(f'debug/free_{d}_rgbd.html'); B.wait_ready(s)
            m0 = B.js(s, 'JSON.stringify(__state.markers.slice(0,50))')
            B.js(s, "document.getElementById('cam').value=1;document.getElementById('cam').onchange();new Promise(r=>setTimeout(r,1500))")
            m1 = B.js(s, 'JSON.stringify(__state.markers.slice(0,50))')
            check(f'free {d} rgbd: camera selector changes the markers', m0 != m1, '')
            # 3D slider changes the drawn posterior; backward canonical label
            s = B.open(f'debug/free_{d}_3d.html'); B.wait_ready(s)
            k0 = meta['canonical_frame']; other = 0 if k0 != 0 else K-1
            lab0 = B.js(s, "document.getElementById('flabel').textContent")
            B.js(s, f"document.getElementById('frame').value={other};document.getElementById('frame').oninput();new Promise(r=>setTimeout(r,2500))")
            lab1 = B.js(s, "document.getElementById('flabel').textContent"); drawn = B.js(s, '__state.drawn_frame')
            check(f'free {d} 3d: canonical endpoint labelled at frame {k0} only', ('CANONICAL' in lab0) and ('CANONICAL' not in lab1), [lab0, lab1])
            check(f'free {d} 3d: slider changes the drawn frame', drawn == other, drawn)
            png = qa/f'free_{d}_3d_f{other:04d}.png'; B.shot(s, png); report['screenshots'].append(str(png))
            note = B.js(s, "document.getElementById('note').textContent")
            check(f'free {d} 3d: page states which endpoint is canonical', f'frame-{k0}' in note, note[:160])
    finally:
        B.close()
    report['status'] = 'PASS' if ok_all else 'FAIL'
    (qa/'qa_report.json').write_text(json.dumps(report, indent=1, default=str))
    print(report['status'])
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
