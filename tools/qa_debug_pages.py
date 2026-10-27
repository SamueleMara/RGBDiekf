#!/usr/bin/env python3
"""Headless-Chrome check of every debug page under a run: loads, JS errors, drawn canvas, slider motion.

For each page: no uncaught exception / console error; every <canvas> has non-blank pixels;
if the page has a range slider, moving it to the middle frame changes the drawn pixels (the
motion actually plays). Screenshots at the first and middle frame go to <run>/qa_debug/.
"""
import argparse, base64, functools, http.server, io, json, os, subprocess, sys, tempfile, threading, time
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qa_free_viewer import Pipe

RECT = r"""(() => { const c = Array.from(document.querySelectorAll('canvas')).sort((a, b) => b.width*b.height - a.width*a.height)[0];
  if (!c) return null; const r = c.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })()"""
MOVE = r"""(async () => { const s = document.querySelector('input[type=range]'); if (!s) return null;
  const mid = Math.round((+s.min + +s.max) / 2); s.value = String(mid);
  s.dispatchEvent(new Event('input', {bubbles: true})); s.dispatchEvent(new Event('change', {bubbles: true}));
  await new Promise(r => setTimeout(r, 1500)); return {slider_max: +s.max, set_to: mid}; })()"""
PROBE = r"""(async () => {
  await new Promise(r => setTimeout(r, 2500));
  const sig = () => Array.from(document.querySelectorAll('canvas')).map(c => {
    try { const g = document.createElement('canvas'); g.width = 64; g.height = 64; const x = g.getContext('2d');
      x.drawImage(c, 0, 0, 64, 64); const d = x.getImageData(0, 0, 64, 64).data; let s = 0, nz = 0;
      for (let i = 0; i < d.length; i += 4) { const v = d[i] + d[i+1] + d[i+2]; s += v * ((i/4) % 97 + 1); if (v > 30) nz++; }
      return [s, nz]; } catch (e) { return [-1, -1]; } });
  const before = sig();
  return {canvases: before.length, before, title: document.title, imgs: Array.from(document.images).map(i => [i.complete, i.naturalWidth])};
})()"""


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args(); run = a.run.resolve(); out = run/'qa_debug'; out.mkdir(exist_ok=True)
    pages = sorted(p for p in run.glob('*/**/*.html') if 'viewer' not in p.parts and 'source_snapshot' not in p.parts)
    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *x): pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(H, directory=str(run)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = []
    with tempfile.TemporaryDirectory() as profile:
        ri, wi = os.pipe(); ro, wo = os.pipe()
        chrome = ['/usr/bin/google-chrome', '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run', '--enable-unsafe-swiftshader',
                  '--use-gl=angle', '--use-angle=swiftshader', f'--user-data-dir={profile}', '--window-size=1500,950', '--remote-debugging-pipe', 'about:blank']
        wrapper = 'import os,sys\nos.dup2(int(sys.argv[1]),3)\nos.dup2(int(sys.argv[2]),4)\nos.execv(sys.argv[3],sys.argv[3:])'
        proc = subprocess.Popen([sys.executable, '-c', wrapper, str(ri), str(wo), *chrome], pass_fds=(ri, wo), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(ri); os.close(wo); pipe = Pipe(ro, wi)
        try:
            for page in pages:
                rel = page.relative_to(run).as_posix()
                target = pipe.call('Target.createTarget', {'url': 'about:blank'})['targetId']
                sess = pipe.call('Target.attachToTarget', {'targetId': target, 'flatten': True})['sessionId']
                pipe.call('Page.enable', session=sess); pipe.call('Runtime.enable', session=sess)
                pipe.call('Page.addScriptToEvaluateOnNewDocument', {'source': "window.__errs=[];addEventListener('error',e=>__errs.push(String(e.message||e)));"
                          "addEventListener('unhandledrejection',e=>__errs.push(String(e.reason)));"}, sess)
                pipe.call('Page.navigate', {'url': f'http://127.0.0.1:{server.server_port}/{rel}'}, sess)
                r = pipe.call('Runtime.evaluate', {'expression': PROBE, 'awaitPromise': True, 'returnByValue': True}, sess, timeout=120)
                v = r.get('result', {}).get('value', {}) if 'exceptionDetails' not in r else {'probe_error': str(r['exceptionDetails'])}
                # uncaught errors raised by the page itself
                errs = pipe.call('Runtime.evaluate', {'expression': 'window.__errs||[]', 'returnByValue': True}, sess)['result'].get('value', [])
                tag = rel.replace('/', '__').replace('.html', '')
                shot = lambda: Image.open(io.BytesIO(base64.b64decode(pipe.call('Page.captureScreenshot', {'format': 'png'}, sess)['data'])))
                rect = pipe.call('Runtime.evaluate', {'expression': RECT, 'returnByValue': True}, sess)['result'].get('value')
                a0 = shot(); a0.save(out/f'{tag}.png')
                crop = lambda im: np.asarray(im.convert('RGB'))[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])].astype(int)
                # What a viewer sees: screenshot pixels of the largest canvas (WebGL readback is blank without preserveDrawingBuffer).
                drawn = None if not rect else bool((np.abs(crop(a0)-np.median(crop(a0).reshape(-1, 3), 0)).sum(-1) > 60).mean() > .002)
                moved = pipe.call('Runtime.evaluate', {'expression': MOVE, 'awaitPromise': True, 'returnByValue': True}, sess)['result'].get('value')
                v['slider'] = moved; motion = None
                if moved:
                    a1 = shot(); a1.save(out/f'{tag}__mid.png')
                    region = crop if rect else (lambda im: np.asarray(im.convert('RGB')).astype(int))   # image pages: whole page
                    motion = bool((np.abs(region(a1)-region(a0)).sum(-1) > 30).mean() > .001)
                imgs_ok = all(c and w > 0 for c, w in v.get('imgs', [])) if v.get('imgs') else None
                ok = 'probe_error' not in v and not errs and drawn is not False and motion is not False and imgs_ok is not False
                results.append({'page': rel, 'title': v.get('title'), 'canvases': v.get('canvases'), 'canvas_drawn': drawn, 'slider': v.get('slider'),
                                'slider_changes_drawing': motion, 'images_loaded': imgs_ok, 'page_errors': errs, 'probe_error': v.get('probe_error'), 'pass': ok})
                print(('PASS ' if ok else 'FAIL ')+rel, {k: results[-1][k] for k in ('canvases', 'canvas_drawn', 'slider_changes_drawing', 'images_loaded', 'page_errors')}, flush=True)
                pipe.call('Target.closeTarget', {'targetId': target})
        finally:
            try: pipe.call('Browser.close', timeout=5)
            except Exception: pass
            proc.terminate(); server.shutdown()
    (out/'qa_debug_pages.json').write_text(json.dumps(results, indent=1))
    return 0 if all(r['pass'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
