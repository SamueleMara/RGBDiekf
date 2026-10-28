// Shared helpers for the point-only measured RGB-D debug pages (no rendering of synthetic images anywhere).
const PD = (() => {
  const D = () => window.PD_DATA || {};
  const loaded = {};
  function load(key) {             // data/<key>.js defines window.PD_DATA[key]; works from file:// (script tags, no fetch)
    if (D()[key]) return Promise.resolve(D()[key]);
    if (loaded[key]) return loaded[key];
    loaded[key] = new Promise((ok, bad) => { const s = document.createElement('script'); s.src = `data/${key}.js`;
      s.onload = () => ok(D()[key]); s.onerror = () => bad(new Error('missing data/' + key + '.js')); document.head.appendChild(s); });
    return loaded[key];
  }
  function bytes(b) { const s = atob(b); const u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
  const f32 = b => new Float32Array(bytes(b).buffer), u8 = b => bytes(b), i32 = b => new Int32Array(bytes(b).buffer);
  function viridis(t) { t = Math.min(1, Math.max(0, t));
    const c = [[.267,.005,.329],[.283,.141,.458],[.254,.265,.530],[.207,.372,.553],[.164,.471,.558],[.128,.567,.551],[.135,.659,.518],[.267,.749,.441],[.478,.821,.318],[.741,.873,.150],[.993,.906,.144]];
    const x = t*(c.length-1), i = Math.min(c.length-2, Math.floor(x)), f = x-i; return c[i].map((v, j) => v+(c[i+1][j]-v)*f); }
  function diverging(t) { t = Math.min(1, Math.max(-1, t)); return t < 0 ? [1+t*.8, 1+t*.6, 1] : [1, 1-t*.6, 1-t*.8]; }
  const REASON_COLORS = {0: '#1a9850', 1: '#555555', 2: '#999999', 3: '#ff7f0e', 4: '#8c564b', 5: '#e377c2', 6: '#9467bd', 7: '#17becf', 8: '#d62728', 9: '#cccccc'};
  // --- minimal orbit viewer on three.js core (OrbitControls is not part of the vendored build)
  function viewer(el) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
    const cam = new THREE.PerspectiveCamera(45, el.clientWidth/el.clientHeight, .001, 1000);
    const ren = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true}); ren.setSize(el.clientWidth, el.clientHeight); el.appendChild(ren.domElement);
    const st = {target: new THREE.Vector3(), r: 2, th: .8, ph: 1.0};
    const place = () => { cam.position.set(st.target.x+st.r*Math.sin(st.ph)*Math.cos(st.th), st.target.y+st.r*Math.sin(st.ph)*Math.sin(st.th), st.target.z+st.r*Math.cos(st.ph));
      cam.up.set(0, 0, 1); cam.lookAt(st.target); };
    let drag = null;
    ren.domElement.addEventListener('mousedown', e => drag = {x: e.clientX, y: e.clientY, b: e.button, t: st.target.clone()});
    window.addEventListener('mouseup', () => drag = null);
    ren.domElement.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('mousemove', e => { if (!drag) return; const dx = e.clientX-drag.x, dy = e.clientY-drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.b === 0) { st.th -= dx*.01; st.ph = Math.min(3.1, Math.max(.05, st.ph-dy*.01)); }
      else { const f = st.r*.002; const right = new THREE.Vector3().subVectors(cam.position, st.target).cross(cam.up).normalize();
        st.target.addScaledVector(right, dx*f); st.target.z += dy*f; }
      place(); });
    ren.domElement.addEventListener('wheel', e => { e.preventDefault(); st.r *= Math.exp(e.deltaY*.001); place(); }, {passive: false});
    const fit = (pos) => { const b = new THREE.Box3(); for (let i = 0; i < pos.length; i += 3) b.expandByPoint(new THREE.Vector3(pos[i], pos[i+1], pos[i+2]));
      b.getCenter(st.target); st.r = Math.max(.05, b.getSize(new THREE.Vector3()).length()*1.2); place(); };
    place();
    const loop = () => { ren.render(scene, cam); requestAnimationFrame(loop); }; loop();
    window.addEventListener('resize', () => { cam.aspect = el.clientWidth/el.clientHeight; cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight); });
    return {scene, camera: cam, renderer: ren, fit, state: st};
  }
  function points(pos, colors, size) {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    return new THREE.Points(g, new THREE.PointsMaterial({size: size || .004, vertexColors: true}));
  }
  function lines(pairs, color) {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs), 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: color || 0x333333}));
  }
  function frustum(c, scale) {       // camera frustum from T_wc and K (world space line set)
    const T = c.T_wc, K = c.K, W = c.width, H = c.height, s = scale || .15;
    const pt = (u, v, z) => { const x = (u-K[2])/K[0]*z, y = (v-K[3])/K[1]*z; return [0, 1, 2].map(i => T[i][0]*x+T[i][1]*y+T[i][2]*z+T[i][3]); };
    const o = [T[0][3], T[1][3], T[2][3]], cs = [pt(0, 0, s), pt(W, 0, s), pt(W, H, s), pt(0, H, s)], seg = [];
    for (let i = 0; i < 4; i++) { seg.push(...o, ...cs[i], ...cs[i], ...cs[(i+1) % 4]); }
    return lines(seg, 0x1f77b4);
  }
  function axes(T, len) {             // coordinate triad of a 4x4 transform (world)
    const o = [T[0][3], T[1][3], T[2][3]], out = new THREE.Group();
    [[0xd62728, 0], [0x2ca02c, 1], [0x1f77b4, 2]].forEach(([col, j]) => out.add(lines([...o, o[0]+len*T[0][j], o[1]+len*T[1][j], o[2]+len*T[2][j]], col)));
    return out;
  }
  function chart(canvas, series, opt) { // series: [{x:[], y:[], color, label, band:[lo,hi]?}]
    const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height, m = 42; ctx.clearRect(0, 0, W, H); ctx.font = '11px sans-serif';
    const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y.concat(s.band ? s.band.flat() : [])).filter(Number.isFinite);
    if (!ys.length) { ctx.fillText('no data', m, H/2); return; }
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); if (y1 === y0) { y1 += 1e-9; y0 -= 1e-9; } if (x1 === x0) x1 += 1;
    const X = x => m+(x-x0)/(x1-x0)*(W-m-10), Y = y => H-24-(y-y0)/(y1-y0)*(H-34);
    ctx.strokeStyle = '#999'; ctx.strokeRect(m, 10, W-m-10, H-34); ctx.fillStyle = '#333';
    ctx.fillText(y1.toPrecision(3), 2, 16); ctx.fillText(y0.toPrecision(3), 2, H-26); ctx.fillText(String(x0), m, H-8); ctx.fillText(String(x1), W-40, H-8);
    if (opt && opt.title) ctx.fillText(opt.title, m+4, 22);
    series.forEach((s, si) => { ctx.strokeStyle = s.color || '#000'; ctx.beginPath(); let pen = false;
      s.x.forEach((x, i) => { const y = s.y[i]; if (!Number.isFinite(y)) { pen = false; return; } pen ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); pen = true; }); ctx.stroke();
      ctx.fillStyle = s.color || '#000'; ctx.fillText(s.label || '', W-160, 24+12*si); });
    if (opt && opt.marker !== undefined) { ctx.strokeStyle = '#d62728'; ctx.beginPath(); ctx.moveTo(X(opt.marker), 10); ctx.lineTo(X(opt.marker), H-24); ctx.stroke(); }
  }
  function legend(el, items) { el.innerHTML = items.map(([c, t]) => `<span style="display:inline-block;margin-right:10px"><span style="display:inline-block;width:10px;height:10px;background:${c};margin-right:4px"></span>${t}</span>`).join(''); }
  function param(name, def) { const v = new URLSearchParams(location.search).get(name) ?? new URLSearchParams(location.hash.slice(1)).get(name); return v === null ? def : v; }
  window.__errors = window.__errors || []; window.addEventListener('error', e => { window.__errors.push(String(e.message)); const b = document.getElementById('errors'); if (b) b.textContent = 'ERROR: '+window.__errors.join(' | '); });
  return {load, f32, u8, i32, viridis, diverging, REASON_COLORS, viewer, points, lines, frustum, axes, chart, legend, param};
})();
