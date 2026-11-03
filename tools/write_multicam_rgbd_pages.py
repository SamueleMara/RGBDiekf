#!/usr/bin/env python3
"""Debug pages: the MASKED multi-camera RGB-D point cloud of every frame, back-projected with the calibrated cameras,
with camera frusta. One standalone page per dataset, all in ONE output folder:

    OUT/index.html                     every dataset, status, frames, cameras, links
    OUT/<dataset>.html                 3D viewer: frame slider / play, RGB or per-camera colour, camera toggles, frusta
    OUT/<dataset>_data/f_NNNNNN.js     one lazily loaded chunk per frame (works from file://, no server)
    OUT/three.min.js, projective_common.js, projective_theme.css, THREE_LICENSE.txt   shared viewer assets (copied once)

Only pixels inside the hard foreground mask with valid finite positive depth are shown: X_c = D K^-1 [u, v, 1],
X_w = R_cw^T (X_c - t_cw) (u = column, v = row, depth = optical camera Z). Nothing is rendered or interpolated. A dataset
without complete aligned masks is SKIPPED (reason recorded), never shown unmasked. By default EVERY masked valid pixel is
shown, positions as exact float32. Optional --max-points-per-frame N thins with ONE fixed image-pixel grid per dataset
(keep pixels with u % s == 0 and v % s == 0, s chosen once from frame 0), so a static pixel is either always or never
shown: thinning never makes static material flicker (counts of all masked pixels are still reported).

    python -B tools/write_multicam_rgbd_pages.py --output-root "/media/.../results_11_10_26/RGBDekf"
"""
import argparse
import base64
import datetime
import html
import json
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output')
DEFAULT_ROOTS = [ROOT/'preparation_8cams/partnet_subset_fixed_render_linear', ROOT/'real']
DEFAULT_OUT = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf')
ASSETS = [('three-0.160.0.min.js', 'three.min.js'), ('projective_common.js', 'projective_common.js'),
          ('projective_theme.css', 'projective_theme.css'), ('THREE_LICENSE.txt', 'THREE_LICENSE.txt')]


def discover(roots, names):
    out = []
    for r in roots:
        for p in sorted(Path(r).iterdir()) if Path(r).is_dir() else []:
            if p.is_dir() and (p/'rgb').is_dir() and (p/'metadata').is_dir() and (not names or p.name in names):
                out.append((p, 'synthetic' if 'preparation' in str(r) else 'real'))
    return out


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def frame_cloud(data, k, grid=1):
    xyz, rgb, cam, stats = [], [], [], []
    for c, camera in enumerate(data.cameras):
        ob = data.observation(k, c)
        depth = np.asarray(ob.depth, np.float64)
        m = np.asarray(ob.mask, bool) & np.asarray(ob.raw_valid, bool)
        masked_valid = int(m.sum())
        if grid > 1:                                   # fixed pixel lattice: the same image pixels in every frame
            lattice = np.zeros_like(m)
            lattice[::grid, ::grid] = True
            m &= lattice
        v, u = np.nonzero(m)
        z = depth[v, u]
        fx, fy, cx, cy = np.asarray(camera.intrinsics, float)
        T = np.asarray(camera.world_to_camera, float)
        Xc = np.stack(((u-cx)/fx*z, (v-cy)/fy*z, z), 1)
        Xw = (Xc-T[:3, 3])@T[:3, :3]
        col = np.asarray(ob.rgb)[v, u]
        stats.append({'camera': camera.name, 'mask_pixels': int(np.asarray(ob.mask, bool).sum()), 'masked_valid_depth': masked_valid, 'displayed_points': int(len(z)),
                      'depth_m_min_median_max': [float(z.min()), float(np.median(z)), float(z.max())] if len(z) else None})
        xyz.append(Xw)
        rgb.append(col)
        cam.append(np.full(len(z), c, np.uint8))
    total = sum(len(x) for x in xyz)
    X = np.concatenate(xyz).astype(np.float32) if total else np.zeros((0, 3), np.float32)
    C = np.clip(np.round(255*np.concatenate(rgb)), 0, 255).astype(np.uint8) if total else np.zeros((0, 3), np.uint8)
    return {'n': int(len(X)), 'xyz': b64(X), 'rgb': b64(C), 'cam': b64(np.concatenate(cam) if total else np.zeros(0, np.uint8)), 'stats': stats}


def cameras_spec(data):
    out = []
    for c in data.cameras:
        T = np.asarray(c.world_to_camera, float)
        out.append({'name': c.name, 'K': np.asarray(c.intrinsics, float).tolist(), 'width': int(c.width), 'height': int(c.height),
                    'T_wc': np.linalg.inv(T).tolist()})
    return out


def write_dataset(path, kind, out, max_points, frame_stride):
    from scene.dataset import Dataset
    name = path.name
    t0 = time.time()
    rec = {'dataset': name, 'kind': kind, 'source': str(path), 'html': f'{name}.html'}
    try:
        data = Dataset(path)
    except Exception as e:
        rec.update(status='SKIPPED_NO_MASK' if 'MISSING_FOREGROUND_MASK' in str(e) or 'mask' in str(e).lower() else 'FAILED_TO_LOAD', reason=str(e)[:400])
        return rec
    ddir = out/f'{name}_data'
    ddir.mkdir(exist_ok=True)
    frames = list(range(0, len(data.times), frame_stride))
    written, totals = [], []
    grid = 1
    if max_points > 0:                                  # one lattice step for the whole dataset (frame 0 sets it)
        n0 = sum(int((np.asarray(data.observation(0, c).mask, bool) & np.asarray(data.observation(0, c).raw_valid, bool)).sum()) for c in range(len(data.cameras)))
        grid = max(1, int(np.ceil(np.sqrt(n0/max_points))))
    try:
        for k in frames:
            cl = frame_cloud(data, k, grid)
            (ddir/f'f_{k:06d}.js').write_text(f'window.MC=window.MC||{{}};window.MC[{k}]={json.dumps(cl)};')
            written.append(k)
            totals.append(sum(s['masked_valid_depth'] for s in cl['stats']))
        status = 'COMPLETE'
        err = None
    except Exception as e:
        status, err = 'PARTIAL_FAILED', f'{type(e).__name__}: {e}'[:400]
    meta = {'dataset': name, 'kind': kind, 'source': str(path), 'cameras': cameras_spec(data), 'frames': written,
            'original_frame_ids': [int(data.frame_ids[k]) for k in written], 'times_s': [float(data.times[k]) for k in written],
            'masked_valid_points_per_frame': totals, 'max_points_per_frame': max_points, 'pixel_grid_step': grid, 'frame_stride': frame_stride, 'status': status, 'error': err,
            'generated': datetime.datetime.now().isoformat(timespec='seconds')}
    (out/f'{name}.html').write_text(PAGE.replace('__TITLE__', html.escape(name)).replace('__META__', json.dumps(meta)))
    rec.update(status=status, error=err, frames=len(written), cameras=len(data.cameras), seconds=round(time.time()-t0, 1),
               median_masked_points=int(np.median(totals)) if totals else 0, bytes=sum(p.stat().st_size for p in ddir.iterdir()))
    return rec


def write_index(out, records):
    rows = []
    for r in records:
        link = f'<a href="{html.escape(r["html"])}">{html.escape(r["dataset"])}</a>' if r['status'] in ('COMPLETE', 'PARTIAL_FAILED') else html.escape(r['dataset'])
        cls = 'good' if r['status'] == 'COMPLETE' else 'bad'
        rows.append(f'<tr><td>{link}</td><td>{r["kind"]}</td><td class="{cls}">{r["status"]}</td><td>{r.get("frames", "")}</td><td>{r.get("cameras", "")}</td>'
                    f'<td>{r.get("median_masked_points", "")}</td><td>{round(r.get("bytes", 0)/1e6, 1) if r.get("bytes") else ""}</td>'
                    f'<td>{html.escape(r.get("reason") or r.get("error") or "")}</td></tr>')
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Masked multicam RGB-D</title><link rel="stylesheet" href="projective_theme.css">
<style>body{{padding:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid var(--line);padding:3px 8px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}</style>
<h1>Masked multi-camera RGB-D back-projection — {len(records)} datasets</h1>
<p>Each page shows, frame by frame, only mask-positive pixels with valid depth of every calibrated camera, back-projected to world, with camera frusta.</p>
<table><tr><th>dataset</th><th>kind</th><th>status</th><th>frames</th><th>cameras</th><th>median masked points / frame</th><th>data MB</th><th>reason</th></tr>{"".join(rows)}</table>''')


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__ — masked multicam RGB-D</title>
<link rel="stylesheet" href="projective_theme.css"><script src="three.min.js"></script><script src="projective_common.js"></script>
<style>#side label{display:block;margin:4px 0}#side input[type=range]{width:100%}#cams span{display:inline-block;margin-right:8px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:2px 4px;text-align:right}td:first-child,th:first-child{text-align:left}
.sw{display:inline-block;width:10px;height:10px;margin-right:4px}#hud{position:absolute;left:10px;top:8px;color:var(--dim);pointer-events:none}</style></head>
<body class="app"><div id="app"><div id="view"><div id="hud"></div></div><div id="side">
<h3 id="title"></h3><div id="src" style="color:var(--dim);word-break:break-all"></div><p><a href="index.html">all datasets</a></p>
<label>Frame <span id="flabel"></span><input id="frame" type="range" min="0" value="0"></label>
<button id="prev">&lt;</button> <button id="play">play</button> <button id="next">&gt;</button> <label style="display:inline">fps <input id="fps" type="number" value="10" min="1" max="60" style="width:50px"></label>
<label>Colour <select id="color"><option value="rgb">measured RGB</option><option value="cam">camera</option></select></label>
<label>Point size <input id="psize" type="range" min="1" max="40" value="6"></label>
<label>Frustum size <input id="fsize" type="range" min="1" max="100" value="15"></label>
<label><input id="keepview" type="checkbox" checked> keep view across frames</label>
<div id="cams"></div><h4>Frame statistics</h4><div id="stats"></div><p id="note" style="color:var(--dim)"></p><div id="errors" class="bad"></div></div></div>
<script>
const M=__META__,$=id=>document.getElementById(id);
const PAL=['#58a6ff','#3fb950','#f0883e','#f85149','#a371f7','#e3b341','#39c5cf','#db61a2','#8b949e','#56d364','#ff7b72','#d2a8ff'];
$('title').textContent=M.dataset+' ('+M.kind+')';$('src').textContent=M.source;
$('note').textContent='Only mask-positive pixels with valid depth, back-projected with the calibrated cameras (no rendering). Status '+M.status+(M.error?': '+M.error:'')+'. '+(M.pixel_grid_step>1?'Thinned with ONE fixed pixel grid (every '+M.pixel_grid_step+'th row and column, same pixels every frame)':'Every masked valid pixel shown')+'; exact float32 positions; frame stride '+M.frame_stride+'.';
const view=PD.viewer($('view'));const shown=M.cameras.map(()=>true);
$('cams').innerHTML='<h4>Cameras</h4>'+M.cameras.map((c,i)=>`<span><label style="display:inline"><input type="checkbox" data-c="${i}" checked><span class="sw" style="background:${PAL[i%PAL.length]}"></span>${c.name}</label> <button data-look="${i}">view</button></span>`).join('<br>');
$('cams').querySelectorAll('input[data-c]').forEach(e=>e.onchange=()=>{shown[+e.dataset.c]=e.checked;draw()});
$('cams').querySelectorAll('button[data-look]').forEach(e=>e.onclick=()=>view.lookFrom(M.cameras[+e.dataset.look]));
const slider=$('frame');slider.max=M.frames.length-1;let idx=0,group=null,fitted=false,timer=null;window.MC=window.MC||{};
function load(k){if(window.MC[k])return Promise.resolve(window.MC[k]);return new Promise((ok,bad)=>{const s=document.createElement('script');s.src=M.dataset+'_data/f_'+String(k).padStart(6,'0')+'.js';s.onload=()=>{s.remove();ok(window.MC[k])};s.onerror=()=>bad(new Error('missing frame '+k));document.head.appendChild(s)})}
function decode(f){if(f._xyz)return f;f._xyz=PD.f32(f.xyz);f._rgb=PD.u8(f.rgb);f._cam=PD.u8(f.cam);return f}
let current=null;
function draw(){if(!current)return;const f=decode(current);if(group){view.scene.remove(group);group.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose()})}group=new THREE.Group();
const pos=[],col=[],byc=$('color').value==='cam';for(let i=0;i<f.n;i++){const c=f._cam[i];if(!shown[c])continue;pos.push(f._xyz[3*i],f._xyz[3*i+1],f._xyz[3*i+2]);
if(byc){const h=PD.hex3(PAL[c%PAL.length]);col.push(...h)}else col.push(f._rgb[3*i]/255,f._rgb[3*i+1]/255,f._rgb[3*i+2]/255)}
if(!fitted||!$('keepview').checked){view.fit(pos.length?pos:Array.from(f._xyz));fitted=true;
// initial view encloses every camera centre so all frusta are visible (lookFrom caps the distance at 3 radii)
const st=view.state,rc=st.radius,far=Math.max(...M.cameras.map(c=>Math.hypot(c.T_wc[0][3]-st.target.x,c.T_wc[1][3]-st.target.y,c.T_wc[2][3]-st.target.z)));
if(1.4*far>st.r){const d=1.4*far;st.radius=d;view.lookFrom({T_wc:[[0,0,0,st.target.x+d*Math.sin(st.ph)*Math.cos(st.th)],[0,0,0,st.target.y+d*Math.sin(st.ph)*Math.sin(st.th)],[0,0,0,st.target.z+d*Math.cos(st.ph)]]});st.radius=rc}}
const r=view.state.radius;group.add(PD.points(pos,col,r*.0015*(+$('psize').value)));M.cameras.forEach((c,i)=>{if(shown[i])group.add(PD.frustum(c,r*.02*(+$('fsize').value)))});
view.scene.add(group);
$('hud').textContent=M.dataset+'  frame '+M.frames[idx]+' (original '+M.original_frame_ids[idx]+', t='+M.times_s[idx].toFixed(4)+' s)  points shown '+pos.length/3;
$('stats').innerHTML='<table><tr><th>camera</th><th>mask px</th><th>mask+depth</th><th>shown</th><th>depth min/med/max m</th></tr>'+f.stats.map(s=>`<tr><td>${s.camera}</td><td>${s.mask_pixels}</td><td>${s.masked_valid_depth}</td><td>${s.displayed_points}</td><td>${s.depth_m_min_median_max?s.depth_m_min_median_max.map(x=>x.toFixed(3)).join(' / '):'-'}</td></tr>`).join('')+'</table>'}
function go(i){idx=Math.max(0,Math.min(M.frames.length-1,i));slider.value=idx;$('flabel').textContent=M.frames[idx]+' / '+M.frames[M.frames.length-1];
load(M.frames[idx]).then(f=>{current=f;draw();if(idx+1<M.frames.length)load(M.frames[idx+1])}).catch(e=>$('errors').textContent=e.message)}
slider.oninput=()=>go(+slider.value);$('prev').onclick=()=>go(idx-1);$('next').onclick=()=>go(idx+1);['color','psize','fsize'].forEach(k=>$(k).oninput=draw);
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='play';return}$('play').textContent='pause';timer=setInterval(()=>go(idx+1<M.frames.length?idx+1:0),1000/Math.max(1,+$('fps').value))};
window.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(idx+1);if(e.key==='ArrowLeft')go(idx-1)});
if(M.frames.length)go(0);else $('errors').textContent='no frames written';
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-roots', nargs='+', type=Path, default=DEFAULT_ROOTS, help='each scene folder with rgb/ and metadata/ inside a root is one dataset')
    ap.add_argument('--scenes', nargs='*', help='optional subset of dataset folder names')
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT, help='parent; a new multicam_rgbd_<stamp>_<id> folder is created')
    ap.add_argument('--max-points-per-frame', type=int, default=0, help='0 (default) = every masked valid pixel; N > 0 = thin with one fixed pixel grid per dataset')
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4, help='datasets processed in parallel (CPU only, no GPU)')
    a = ap.parse_args()
    scenes = discover(a.data_roots, set(a.scenes or []))
    names = [p.name for p, _ in scenes]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        ap.error(f'dataset names repeat across roots: {sorted(dup)}')
    if not scenes:
        ap.error('no datasets found')
    out = a.output_root/f'multicam_rgbd_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    for src, dst in ASSETS:
        shutil.copy(REPO/'debug/assets'/src, out/dst)
    print(f'Output: {out}\n{len(scenes)} datasets', flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(write_dataset, p, kind, out, a.max_points_per_frame, a.frame_stride): p for p, kind in scenes}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = {'dataset': futs[f].name, 'kind': '?', 'status': 'FAILED', 'html': f'{futs[f].name}.html', 'reason': traceback.format_exc()[-400:]}
            records.append(r)
            write_index(out, sorted(records, key=lambda x: x['dataset']))
            (out/'summary.json').write_text(json.dumps(sorted(records, key=lambda x: x['dataset']), indent=1))
            print(f'[{len(records)}/{len(scenes)}] {r["dataset"]}: {r["status"]} frames={r.get("frames", 0)} {r.get("seconds", "")} s', flush=True)
    print(out/'index.html')


if __name__ == '__main__':
    raise SystemExit(main())
