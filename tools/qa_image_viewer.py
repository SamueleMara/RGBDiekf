#!/usr/bin/env python3
"""Headless-Chrome QA of the rendered-image viewer: displayed pixels vs saved numbers.

Checks, inside the real browser after the page reports ready:
  * the loaded mosaic has exactly the tile geometry recorded for (direction, stage, frame, camera);
  * coverage-map pixel colours reproduce the recorded domain/covered/unexplained/outside counts;
  * RGB RMSE recomputed from the displayed observed/rendered tiles equals the recorded value;
  * the frame's 3D points projected through that camera's calibration land on the panel's observed domain;
  * both joint-candidate axes are drawn whenever joint candidates exist.
Writes qa_<tag>.png (screenshot) and qa_<tag>.json next to the viewer.
"""
import argparse, base64, functools, http.server, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
from urllib.parse import urlencode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qa_free_viewer import Pipe

CHECK = r"""(async () => {
  const deadline = Date.now() + 60000;
  while (document.body.dataset.ready !== '1') { if (Date.now() > deadline) throw Error('viewer not ready'); await new Promise(r => setTimeout(r, 100)); }
  const img = document.getElementById('mosaic');
  while (!img.complete) await new Promise(r => setTimeout(r, 50));
  const dir = document.getElementById('dir').value, stage = document.getElementById('stage').value;
  const ci = +document.getElementById('cam').value, k = +document.getElementById('frame').value;
  const M = await (await fetch('manifest.json')).json(); const S = M.directions[dir].panels[stage][k][ci];
  const [h, w] = S.tile_hw; const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
  const g = c.getContext('2d'); g.drawImage(img, 0, 0); const px = g.getImageData(0, 0, c.width, c.height).data;
  const at = (x, y) => { const i = 4*(y*c.width + x); return [px[i], px[i+1], px[i+2]]; };
  let green = 0, red = 0, orange = 0, se = 0, n = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const m = at(3*w + x, y);
    if (m[0] === 40 && m[1] === 200 && m[2] === 70) { green++; const o = at(x, y), p = at(w + x, y); for (let j = 0; j < 3; j++) { const d = (o[j] - p[j])/255; se += d*d; } n++; }
    else if (m[0] === 220 && m[1] === 40 && m[2] === 40) red++; else if (m[0] === 240 && m[1] === 160 && m[2] === 30) orange++; }
  const rmse = n ? Math.sqrt(se/n/3) : null;
  const cam = M.cameras[ci], R = cam.world_to_camera, K = cam.intrinsics, box = S.box_y0_y1_x0_x1;
  const xyz = new Float32Array(await (await fetch(`points/${dir}/xyz_${String(k).padStart(6,'0')}.bin`)).arrayBuffer());
  let inBox = 0, onDomain = 0, front = 0;
  for (let i = 0; i < xyz.length/3; i++) {
    const X = [xyz[3*i], xyz[3*i+1], xyz[3*i+2]]; const q = [0,1,2].map(r => R[r][0]*X[0] + R[r][1]*X[1] + R[r][2]*X[2] + R[r][3]);
    if (q[2] <= 0) continue; front++;
    const u = Math.round(K[0]*q[0]/q[2] + K[2]), v = Math.round(K[1]*q[1]/q[2] + K[3]);
    if (v >= box[0] && v < box[1] && u >= box[2] && u < box[3]) { inBox++; const m = at(3*w + (u - box[2]), v - box[0]); if (m[1] === 200 || m[0] === 220) onDomain++; } }
  const J = M.directions[dir].joint; const axes = J && J.candidates ? Object.keys(J.candidates).length : 0;
  return {direction: dir, stage, frame: k, camera: cam.name, natural: [img.naturalWidth, img.naturalHeight], expected: [4*w, 2*h],
          counts_displayed: {covered_domain: green, unexplained_domain: red, predicted_outside_domain: orange},
          counts_recorded: {covered_domain: S.covered_domain_pixels, unexplained_domain: S.unexplained_domain_pixels, predicted_outside_domain: S.predicted_outside_domain_pixels},
          rgb_rmse_displayed: rmse, rgb_rmse_recorded: S.rgb_rmse, points_in_front: front, projected_into_panel: inBox, projected_onto_observed_domain: onDomain,
          joint_candidates_drawn: axes, joint_status: J && J.status};
})()"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--view', type=Path, required=True); ap.add_argument('--direction', default='first'); ap.add_argument('--stage', default='phase1')
    ap.add_argument('--camera', type=int, default=0); ap.add_argument('--frame', type=int, default=30); ap.add_argument('--color', default='core')
    a = ap.parse_args(); view = a.view.resolve()
    tag = f'{a.direction}_{a.stage}_f{a.frame:03d}_c{a.camera}'
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(view)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    process = None; fds = []
    try:
        with tempfile.TemporaryDirectory(prefix='rgbd_image_chrome_') as profile:
            read_in, write_in = os.pipe(); read_out, write_out = os.pipe(); fds = [read_out, write_in]
            chrome = ['/usr/bin/google-chrome', '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run', '--enable-unsafe-swiftshader',
                      '--use-gl=angle', '--use-angle=swiftshader', f'--user-data-dir={profile}', '--window-size=1700,1150', '--remote-debugging-pipe', 'about:blank']
            wrapper = 'import os,sys\nos.dup2(int(sys.argv[1]),3)\nos.dup2(int(sys.argv[2]),4)\nos.execv(sys.argv[3],sys.argv[3:])'
            process = subprocess.Popen([sys.executable, '-c', wrapper, str(read_in), str(write_out), *chrome], pass_fds=(read_in, write_out),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.close(read_in); os.close(write_out)
            pipe = Pipe(read_out, write_in)
            target = pipe.call('Target.createTarget', {'url': 'about:blank'})['targetId']
            session = pipe.call('Target.attachToTarget', {'targetId': target, 'flatten': True})['sessionId']
            pipe.call('Page.enable', session=session)
            query = urlencode({'dir': a.direction, 'stage': a.stage, 'cam': a.camera, 'frame': a.frame, 'color': a.color})
            pipe.call('Page.navigate', {'url': f'http://127.0.0.1:{server.server_port}/index.html?{query}'}, session)
            ans = pipe.call('Runtime.evaluate', {'expression': CHECK, 'awaitPromise': True, 'returnByValue': True}, session, timeout=120)
            if 'exceptionDetails' in ans: raise RuntimeError(ans['exceptionDetails'])
            qa = ans['result']['value']
            qa['checks'] = {'panel_geometry': qa['natural'] == qa['expected'],
                            'coverage_counts': qa['counts_displayed'] == qa['counts_recorded'],
                            'rgb_rmse': qa['rgb_rmse_recorded'] is None or abs(qa['rgb_rmse_displayed']-qa['rgb_rmse_recorded']) < 1e-6,
                            'projection_onto_observed_domain': qa['projected_into_panel'] > 0 and qa['projected_onto_observed_domain']/qa['projected_into_panel'] > .5,
                            'both_candidates_drawn': qa['joint_status'] is None or qa['joint_candidates_drawn'] == 2}
            qa['pass'] = all(qa['checks'].values())
            time.sleep(.5)
            shot = pipe.call('Page.captureScreenshot', {'format': 'png'}, session)['data']
            (view/f'qa_{tag}.png').write_bytes(base64.b64decode(shot)); (view/f'qa_{tag}.json').write_text(json.dumps(qa, indent=2)+'\n')
            print(json.dumps(qa, indent=2))
            try: pipe.call('Browser.close', timeout=5)
            except (RuntimeError, TimeoutError): pass
            return 0 if qa['pass'] else 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()
        for fd in fds: os.close(fd)
        server.shutdown(); server.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
