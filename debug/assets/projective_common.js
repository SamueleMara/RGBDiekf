// Shared helpers for the point-only measured RGB-D debug pages (no rendering of synthetic images anywhere).
// Visual style follows the RigEuler3DGS debug pages: dark scene, monospace side panel, round size-attenuated points.
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
  function diverging(t) { t = Math.min(1, Math.max(-1, t)); const a = [.35, .65, 1], m = [.55, .58, .62], b = [.97, .32, .29];   // blue - grey - red (dark background)
    return t < 0 ? m.map((v, j) => v+(a[j]-v)*(-t)) : m.map((v, j) => v+(b[j]-v)*t); }
  // RigEuler palette
  const C = {bg: 0x0e1116, start: '#4a515c', est: '#58a6ff', track: '#f06595', spread: '#e3b341', outlier: '#f0883e', pred: '#a371f7',
             good: '#3fb950', bad: '#f85149', dim: '#8b949e', line: '#2a313a', fg: '#e6edf3', cam: '#58a6ff'};
  const REASON_COLORS = {0: '#3fb950', 1: '#6e7681', 2: '#484f58', 3: '#f0883e', 4: '#a5714e', 5: '#db61a2', 6: '#a371f7', 7: '#39c5cf', 8: '#f85149', 9: '#30363d', 10: '#e3b341'};
  const hex3 = h => { const v = parseInt(String(h).replace('#', ''), 16); return [(v>>16 & 255)/255, (v>>8 & 255)/255, (v & 255)/255]; };
  let disc = null;                 // round point sprite
  function sprite() { if (disc) return disc; const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d');
    x.beginPath(); x.arc(32, 32, 30, 0, 2*Math.PI); x.fillStyle = '#fff'; x.fill(); disc = new THREE.CanvasTexture(c); return disc; }
  // --- minimal orbit viewer on three.js core (OrbitControls is not part of the vendored build); z up
  function viewer(el) {
    const scene = new THREE.Scene(); scene.background = new THREE.Color(C.bg);
    const cam = new THREE.PerspectiveCamera(45, el.clientWidth/Math.max(el.clientHeight, 1), .001, 1000);
    const ren = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true}); ren.setPixelRatio(Math.min(devicePixelRatio, 2));
    ren.setSize(el.clientWidth, el.clientHeight); el.appendChild(ren.domElement);
    const st = {target: new THREE.Vector3(), r: 2, th: .8, ph: 1.0, radius: .5};
    const place = () => { cam.position.set(st.target.x+st.r*Math.sin(st.ph)*Math.cos(st.th), st.target.y+st.r*Math.sin(st.ph)*Math.sin(st.th), st.target.z+st.r*Math.cos(st.ph));
      cam.up.set(0, 0, 1); cam.lookAt(st.target); };
    let drag = null;
    ren.domElement.addEventListener('mousedown', e => drag = {x: e.clientX, y: e.clientY, b: e.button});
    window.addEventListener('mouseup', () => drag = null);
    ren.domElement.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('mousemove', e => { if (!drag) return; const dx = e.clientX-drag.x, dy = e.clientY-drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.b === 0) { st.th -= dx*.006; st.ph = Math.min(3.1, Math.max(.05, st.ph-dy*.006)); }
      else { const f = st.r*.0016, fwd = new THREE.Vector3().subVectors(st.target, cam.position).normalize();
        const right = new THREE.Vector3().crossVectors(fwd, cam.up).normalize(), up = new THREE.Vector3().crossVectors(right, fwd).normalize();
        st.target.addScaledVector(right, -dx*f); st.target.addScaledVector(up, dy*f); }
      place(); });
    ren.domElement.addEventListener('wheel', e => { e.preventDefault(); st.r *= Math.exp(e.deltaY*.0012); place(); }, {passive: false});
    const fit = (pos) => { const n = pos.length/3; if (!n) return; const c = [0, 0, 0];
      for (let i = 0; i < n; i++) { c[0] += pos[3*i]; c[1] += pos[3*i+1]; c[2] += pos[3*i+2]; } c.forEach((v, j) => c[j] = v/n);
      const d = []; for (let i = 0; i < n; i += Math.max(1, Math.floor(n/20000))) d.push(Math.hypot(pos[3*i]-c[0], pos[3*i+1]-c[1], pos[3*i+2]-c[2]));
      d.sort((a, b) => a-b); st.radius = Math.max(.02, d[Math.floor(.98*(d.length-1))]);     // robust: a few far outliers do not shrink the view
      st.target.set(...c); st.r = st.radius*3.2; place(); };
    const lookFrom = (c) => { const o = [c.T_wc[0][3], c.T_wc[1][3], c.T_wc[2][3]], d = o.map((v, j) => v-st.target.getComponent(j)), r = Math.hypot(...d);
      st.r = Math.min(r, st.radius*3); st.th = Math.atan2(d[1], d[0]); st.ph = Math.acos(Math.max(-1, Math.min(1, d[2]/r))); place(); };
    place();
    const resize = () => { cam.aspect = el.clientWidth/Math.max(el.clientHeight, 1); cam.updateProjectionMatrix(); ren.setSize(el.clientWidth, el.clientHeight); };
    const loop = () => { ren.render(scene, cam); requestAnimationFrame(loop); }; loop();
    window.addEventListener('resize', resize);
    return {scene, camera: cam, renderer: ren, fit, lookFrom, state: st, resize};
  }
  // points: colors is a Float32Array (per vertex) or a '#rrggbb' string (uniform); size in world units (scene-relative at the call site)
  function points(pos, colors, size, opacity) {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    const uni = typeof colors === 'string';
    if (!uni) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    const m = new THREE.PointsMaterial({size: size || .004, sizeAttenuation: true, map: sprite(), alphaTest: .5, vertexColors: !uni,
      transparent: opacity !== undefined && opacity < 1, opacity: opacity === undefined ? 1 : opacity, depthWrite: !(opacity < 1)});
    if (uni) m.color = new THREE.Color(colors);
    return new THREE.Points(g, m);
  }
  function lines(pairs, color, opacity) {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs), 3));
    return new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: color === undefined ? 0x8b949e : color, transparent: opacity !== undefined && opacity < 1, opacity: opacity === undefined ? 1 : opacity}));
  }
  function frustum(c, scale) {       // camera frustum from T_wc and K (world space line set)
    const T = c.T_wc, K = c.K, W = c.width, H = c.height, s = scale || .15;
    const pt = (u, v, z) => { const x = (u-K[2])/K[0]*z, y = (v-K[3])/K[1]*z; return [0, 1, 2].map(i => T[i][0]*x+T[i][1]*y+T[i][2]*z+T[i][3]); };
    const o = [T[0][3], T[1][3], T[2][3]], cs = [pt(0, 0, s), pt(W, 0, s), pt(W, H, s), pt(0, H, s)], seg = [];
    for (let i = 0; i < 4; i++) { seg.push(...o, ...cs[i], ...cs[i], ...cs[(i+1) % 4]); }
    return lines(seg, 0x58a6ff, .6);
  }
  function axes(T, len) {             // coordinate triad of a 4x4 transform (world)
    const o = [T[0][3], T[1][3], T[2][3]], out = new THREE.Group();
    [[0xff5555, 0], [0x55ff55, 1], [0x5599ff, 2]].forEach(([col, j]) => out.add(lines([...o, o[0]+len*T[0][j], o[1]+len*T[1][j], o[2]+len*T[2][j]], col)));
    return out;
  }
  function chart(canvas, series, opt) { // series: [{x:[], y:[], color, label, dash?}]
    const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height, m = 46; ctx.clearRect(0, 0, W, H); ctx.font = '11px ui-monospace,monospace';
    ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);
    const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(Number.isFinite);
    ctx.fillStyle = '#8b949e'; if (opt && opt.title) ctx.fillText(opt.title, m, 12);
    if (!ys.length) { ctx.fillText('no data', m, H/2); return; }
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); if (y1 === y0) { y1 += 1e-9; y0 -= 1e-9; } if (x1 === x0) x1 += 1;
    const X = x => m+(x-x0)/(x1-x0)*(W-m-8), Y = y => H-20-(y-y0)/(y1-y0)*(H-40);
    ctx.strokeStyle = '#2a313a'; ctx.strokeRect(m, 20, W-m-8, H-40);
    if (y0 < 0 && y1 > 0) { ctx.beginPath(); ctx.moveTo(m, Y(0)); ctx.lineTo(W-8, Y(0)); ctx.stroke(); }
    ctx.fillStyle = '#8b949e'; ctx.fillText(y1.toPrecision(3), 2, 28); ctx.fillText(y0.toPrecision(3), 2, H-22); ctx.fillText(String(x0), m, H-6); ctx.fillText(String(x1), W-30, H-6);
    series.forEach((s, si) => { ctx.strokeStyle = s.color || '#e6edf3'; ctx.setLineDash(s.dash || []); ctx.lineWidth = 1.4; ctx.beginPath(); let pen = false;
      s.x.forEach((x, i) => { const y = s.y[i]; if (!Number.isFinite(y)) { pen = false; return; } pen ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); pen = true; }); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = s.color || '#e6edf3'; if (s.label) ctx.fillText(s.label, m+6+Math.floor(si/2)*((W-m)/2), 32+12*(si % 2)); });
    if (opt && opt.marker !== undefined) { ctx.strokeStyle = '#f0883e'; ctx.beginPath(); ctx.moveTo(X(opt.marker), 20); ctx.lineTo(X(opt.marker), H-20); ctx.stroke(); }
  }
  function legend(el, items) { el.className = 'legend'; el.innerHTML = items.map(([c, t]) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`).join(''); }
  function colorbar(el, f, lo, hi) { const st = [...Array(11).keys()].map(i => { const c = f(i/10).map(v => Math.round(255*v)); return `rgb(${c})`; }).join(',');
    el.innerHTML = `<div class="cbar" style="background:linear-gradient(90deg,${st})"></div><div class="cbar-l"><span>${lo}</span><span>${hi}</span></div>`; }
  function rows(el, pairs) { el.innerHTML = pairs.map(([a, b]) => `<div class="row"><span>${a}</span><span>${b}</span></div>`).join(''); }
  function param(name, def) { const v = new URLSearchParams(location.search).get(name) ?? new URLSearchParams(location.hash.slice(1)).get(name); return v === null ? def : v; }
  window.__errors = window.__errors || []; window.addEventListener('error', e => { window.__errors.push(String(e.message)); const b = document.getElementById('errors'); if (b) b.textContent = 'ERROR: '+window.__errors.join(' | '); });
  return {load, f32, u8, i32, viridis, diverging, C, REASON_COLORS, hex3, viewer, points, lines, frustum, axes, chart, legend, colorbar, rows, param};
})();
