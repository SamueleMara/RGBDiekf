"""Live file:// compatible diagnostics, actual measured panels and point motion."""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image,ImageDraw
from utils.io import serial


def _color_depth(z,valid):
    out=np.zeros((*z.shape,3),np.uint8)
    if valid.any():
        lo,hi=np.quantile(z[valid],[.01,.99]);t=np.clip((z-lo)/max(hi-lo,1e-15),0,1)
        out[...,0]=(255*t).astype(np.uint8);out[...,1]=(255*(1-np.abs(2*t-1))).astype(np.uint8);out[...,2]=(255*(1-t)).astype(np.uint8)
    out[~valid]=0;return out


def temporal_frame_debug(root,frame,owners,state,prior,rig,rows,k):
    root=Path(root);dest=root/'temporal'/'debug_frames';dest.mkdir(parents=True,exist_ok=True)
    rgb=np.nan_to_num(frame.rgb.cpu().numpy());depth=np.nan_to_num(frame.depth.cpu().numpy())
    mask=frame.mask.cpu().numpy();valid=(frame.mask&frame.raw_valid).cpu().numpy()
    own=owners.owner_id.cpu().numpy();count=owners.counts.cpu().numpy()
    uvp=rig.project(prior[:,:3],jacobian=False).uv.cpu().numpy();uv=owners.uv.cpu().numpy()
    man=json.loads((root/'run_manifest.json').read_text())
    ids=np.union1d(np.linspace(0,len(state)-1,min(len(state),12000)).astype(int),np.asarray(man['debug_point_ids'],dtype=int))  # empty when no traced ID survives subsampling
    observed=np.zeros(len(state),bool)
    if rows is not None:observed[rows.point.cpu().numpy()]=True
    can=np.load(root/'canonical.npz')
    cloud={'ids':can['point_id'][ids].tolist() if 'point_id' in can else ids.tolist(),'xyz':np.round(state[ids,:3].cpu().numpy(),7).ravel().tolist(),
           'prior':np.round(prior[ids,:3].cpu().numpy(),7).ravel().tolist(),'rgb':can['canonical_rgb'][ids].ravel().tolist(),
           'canonical':can['canonical_xyz'][ids].ravel().tolist(),'observed':observed[ids].tolist(),
           'cameras':[{'K':rig.K[c].tolist(),'T_wc':rig.T_wc[c].tolist(),'width':rig.width[c],'height':rig.height[c]} for c in range(rig.count)]}
    (dest/f'cloud_{k:04d}.js').write_text('window.TEMPORAL_CLOUD='+json.dumps(cloud)+';')
    for c in range(rig.count):
        H,W=mask[c].shape;occupied=own[c]>=0;ids=np.maximum(own[c],0).astype(np.uint64)
        colors=np.stack(((ids*73+31)%256,(ids*151+97)%256,(ids*199+113)%256),-1).astype(np.uint8);colors[~occupied]=0
        warp=np.zeros((H,W,3),np.uint8);wd=np.zeros((H,W));rd=np.zeros((H,W));rr=np.zeros((H,W,3));measured=np.zeros((H,W),bool)
        if rows is not None:
            ix=rows.camera.cpu().numpy()==c
            pix=rows.pixel.cpu().numpy()[ix];y,x=pix//W,pix%W
            h=rows.h.cpu().numpy()[ix];obs=rows.y.cpu().numpy()[ix]
            warp[y,x]=np.clip(h[:,:3]*255,0,255).astype(np.uint8);wd[y,x]=h[:,3];rd[y,x]=obs[:,3]-h[:,3];rr[y,x]=np.abs(obs[:,:3]-h[:,:3]);measured[y,x]=True
        uncovered=valid[c]&~occupied;outside=occupied&~mask[c]
        covimg=(rgb[c]*255).clip(0,255).astype(np.uint8);covimg[uncovered]=[0,255,255];covimg[outside]=[255,0,0]
        boot=np.zeros((H,W,3),np.uint8);boot[occupied & valid[c] & ~measured]=[255,210,0];boot[measured]=[0,220,80]
        resid=np.zeros_like(warp)
        if measured.any():
            scale=max(float(np.quantile(rr[measured],.95)),1e-15);resid=np.clip(rr/scale*255,0,255).astype(np.uint8)
        depthres=np.zeros_like(warp)
        if measured.any():
            scale=max(float(np.quantile(np.abs(rd[measured]),.95)),1e-15);t=np.clip(rd/scale,-1,1)
            depthres[...,0]=(255*np.maximum(t,0)).astype(np.uint8);depthres[...,2]=(255*np.maximum(-t,0)).astype(np.uint8);depthres[~measured]=0
        overlay=Image.fromarray((rgb[c]*255).clip(0,255).astype(np.uint8));draw=ImageDraw.Draw(overlay)
        for a,b in zip(uvp[c,::max(1,len(state)//3000)],uv[c,::max(1,len(state)//3000)]):
            if np.isfinite(a).all() and np.isfinite(b).all() and np.any(a!=b):draw.line((float(a[0]),float(a[1]),float(b[0]),float(b[1])),fill=(255,0,255),width=1)
        conflict=np.zeros_like(warp);conflict[count[c]>1]=[255,128,0];conflict[count[c]==1]=[40,90,150]
        panels=[('Measured RGB',(rgb[c]*255).clip(0,255).astype(np.uint8)),('Metric depth: display 1-99%',_color_depth(depth[c],valid[c])),
                ('Hard foreground mask',np.repeat(mask[c,...,None],3,-1).astype(np.uint8)*255),('First-hit point ID',colors),
                ('Prior to posterior (magenta)',np.array(overlay)),('Reference warped RGB',warp),('Predicted temporal depth',_color_depth(wd,measured)),
                ('Absolute RGB residual: display p95',resid),('Signed depth residual: red+/blue-',depthres),
                ('Uncovered cyan / outside mask red',covimg),('Temporal rows green / no row yellow',boot),('Pixel conflicts orange',conflict)]
        sheet=Image.new('RGB',(W*4,(H+24)*3),(20,24,32));painter=ImageDraw.Draw(sheet)
        for j,(label,img) in enumerate(panels):
            xx=(j%4)*W;yy=(j//4)*(H+24);painter.text((xx+5,yy+5),label,fill='white');sheet.paste(Image.fromarray(img),(xx,yy+24))
        sheet.save(dest/f'frame_{k:04d}_cam{c}.png')


def write_temporal_dashboard(root,manifest,history,failure=None):
    root=Path(root);td=root/'temporal';td.mkdir(exist_ok=True)
    data={'manifest':manifest,'history':history,'failure':failure,'debug_frames':[int(p.name.split('_')[1]) for p in sorted((td/'debug_frames').glob('*_cam0.png'))]}
    if (root/'completed_frames.json').exists():
        from utils.temporal_output import require_valid_frame
        completed=json.loads((root/'completed_frames.json').read_text())['completed_frames']
        for k in completed:require_valid_frame(root,k)
        data['debug_frames']=[k for k in data['debug_frames'] if k in completed]
        data['history']=[h for h in history if h['frame'] in completed]
    data['point_traces']={}
    for p in (td/'traced_points').glob('*.json'):
        trace=json.loads(p.read_text());records=[]
        for f in trace['frames']:
            if (root/'completed_frames.json').exists() and f['frame'] not in completed:continue
            P=np.asarray(f['P_plus']);records.append({'frame':f['frame'],'state':f['s_plus'],'source':f['source'],
              'position_std_m':[float(np.sqrt(v)) if v>=0 else None for v in np.diag(P)[:3]],'covariance_min_eigenvalue_mixed_units':float(np.linalg.eigvalsh(P).min()),'rows':len(f['rows']),
              'measurement_update_accepted':f.get('measurement_update_accepted'),'converged':f['solver']['converged'],
              'camera_reason':f['camera_reason'],'reference_frames':f['reference_frames'],
              'measurements':[{'camera':r['camera'],'current_pixel':r['current_pixel'],'q':r['q'],'residual':r['residual']} for r in f['rows']]})
            lp=f.get('local_posterior')
            if lp:
                kinds=[b['kind'] for b in lp['retained_layout']]
                records[-1].update({'status':f.get('status'),'local_retained':{k:kinds.count(k) for k in ('state','source_chart','raw_sensor_error')},
                  'local_cross_norm':{k:float(np.sqrt(sum(np.square(np.asarray(b['Cov_current_block'])).sum() for b in lp['cross_with_current_state'] if b['kind']==k)))
                                      for k in ('state','source_chart','raw_sensor_error')}})
        data['point_traces'][str(trace['point_id'])]=records
    assets=Path(__file__).parent/'assets'
    for src,dest in [('three-0.160.0.min.js','three.min.js'),('projective_common.js','projective_common.js'),('THREE_LICENSE.txt','THREE_LICENSE.txt')]:
        if not (td/dest).exists():shutil.copyfile(assets/src,td/dest)
    (td/'dashboard.js').write_text('window.TEMPORAL='+json.dumps(serial(data))+';')
    (td/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>Temporal RGB-D diagnostics</title>
<style>body{background:#101723;color:#e0e9f5;font:16px system-ui;margin:24px}a{color:#69cfff}select,input,button{padding:8px;background:#23344b;color:white}img{width:100%;border:1px solid #456}pre{white-space:pre-wrap;background:#182437;padding:16px}canvas{width:100%;height:200px;background:#182437}.bad{color:#ff8b79}nav{display:flex;gap:16px}</style></head>
<body><h1>Temporal RGB-D: measured evidence and uncertainty</h1><nav><a href="../../debug/index.html">Complete 3D / RGB-D / numerical viewer</a><a href="../status.json">Run status</a><a href="config.json">Noise provenance</a></nav>
<p id="mode"></p><p id="failure" class="bad"></p><label>Debug frame <select id="frame"></select></label> <label>Camera <select id="cam"></select></label> <button onclick="location.reload()">Refresh live run</button>
<p>Cyan is renderer-uncovered foreground; red is a first hit outside the mask. Yellow has no temporal row (bootstrap or invalid reference). Display scales do not affect inference. Click the panels to inspect the lossless image at native resolution.</p><a id="panelLink" target="_blank" rel="noopener"><img id="panel" alt="Measured RGB-D, hard mask, first-hit ownership, temporal warp, residual and support panels"></a>
<h2>Live 3D point motion</h2><p>Drag to orbit, right-drag to pan, wheel to zoom. Click a point for its canonical ID. Display samples at most 12,000 points plus requested IDs; inference uses the complete cloud.</p>
<label><input id="prior" type="checkbox"> Predicted positions (purple)</label> <label><input id="canonical" type="checkbox"> Canonical positions (gray)</label> <label><input id="support" type="checkbox"> Color by temporal support</label><button id="play">Play debug frames</button>
<div id="viewport" style="height:550px"></div><style>#viewport canvas{height:100%;width:100%}</style>
<h2>Observation support and residual evolution</h2><canvas id="plot" width="1400" height="240"></canvas>
<h2>Selected frame: nonlinear convergence</h2><canvas id="objective" width="1400" height="240"></canvas>
<h2>Motion and uncertainty across the pass</h2><canvas id="motion" width="1400" height="240"></canvas><canvas id="covariance" width="1400" height="240"></canvas><pre id="stats"></pre>
<label>Canonical point ID <input id="point" type="number" value="0"></label> <button onclick="window.open('traced_points/'+document.getElementById('point').value+'.json')">Open complete numerical trace</button>
<h2>Selected original point: estimated trajectory and uncertainty</h2><canvas id="pointposition" width="1400" height="240"></canvas><canvas id="pointspeed" width="1400" height="240"></canvas><canvas id="pointuncertainty" width="1400" height="240"></canvas><pre id="pointstats"></pre>
<div id="localpanel" style="display:none"><h2>Point-local covariance retention (local-correlated)</h2><p>Retained local blocks after exact local marginalization and the norm of their covariance with the current state. Cross-point covariance is zero by the declared block-diagonal approximation.</p><canvas id="pointlocal" width="1400" height="240"></canvas><canvas id="pointlocalcross" width="1400" height="240"></canvas><pre id="pointlocalstatus"></pre><canvas id="localdims" width="1400" height="240"></canvas></div>
<p>Point motion, orbit/pan, camera frusta, covariance glyphs, and image-space point selection are available in the complete viewer after the pass finishes. This page is written during filtering and survives failures.</p>
<script src="three.min.js"></script><script src="projective_common.js"></script><script src="dashboard.js"></script><script>
const D=window.TEMPORAL,$=id=>document.getElementById(id);$('mode').textContent=D.manifest.temporal_mode+' — '+D.manifest.reference_uncertainty;$('failure').textContent=D.failure||'';
if(D.manifest.execution_kind)$('mode').textContent=D.manifest.execution_kind+' — '+D.manifest.temporal_mode.toUpperCase()+' — '+D.manifest.noise_assumption+' — '+D.manifest.reference_uncertainty+(D.manifest.diagnostic_focal_control?' — FOCAL CONTROL; reduced occlusion catalogue':'');
if(D.manifest.execution_kind&&!D.history.length)$('mode').textContent='INITIALIZATION_ONLY_NO_FILTERED_FRAME — '+$('mode').textContent;
if(D.history.length&&!D.history.some(h=>(h.assimilated_points??h.solver?.assimilated_points??0)>0))$('mode').textContent+=' — NO_ACCEPTED_UPDATES';
if(D.manifest.temporal_mode==='GEOMETRY_PREFLIGHT_ONLY_NO_ESTIMATION')document.querySelector('nav a').style.display='none';
for(const k of D.debug_frames){let o=new Option(k,k);$('frame').add(o)}for(let c=0;c<(D.manifest.camera_count||8);c++)$('cam').add(new Option(c,c));
$('frame').selectedIndex=Math.max(0,$('frame').options.length-1);$('point').value=(D.manifest.debug_original_point_ids||D.manifest.debug_point_ids||Object.keys(D.point_traces||{}))[0]??0;
let view,cloud,group,pointsObj,hasFit=false;
try{view=PD.viewer($('viewport'))}catch(e){$('viewport').textContent='WebGL unavailable: '+e.message}
function drawCloud(){if(!view||!cloud)return;if(group){view.scene.remove(group);group.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose()})}group=new THREE.Group();
let colors=cloud.rgb;if($('support').checked)colors=cloud.observed.flatMap(x=>x?[.1,.9,.4]:[.95,.6,.1]);pointsObj=PD.points(cloud.xyz,colors,.003);group.add(pointsObj);
if($('prior').checked)group.add(PD.points(cloud.prior,'#a371f7',.002,.5));if($('canonical').checked)group.add(PD.points(cloud.canonical,'#8592a2',.002,.3));
cloud.cameras.forEach(c=>group.add(PD.frustum(c,.08)));view.scene.add(group);if(!hasFit){view.fit(cloud.xyz);hasFit=true}}
function show(){let k=$('frame').value,c=$('cam').value;let panel='debug_frames/frame_'+String(k).padStart(4,'0')+'_cam'+c+'.png';$('panel').src=panel;$('panelLink').href=panel;let h=D.history.find(h=>h.frame==k);$('stats').textContent=JSON.stringify(h||{anchor:true,source_identity:'canonical.npz'},null,2);
showPoint();
let it=h?.solver?.history||[];PD.chart($('objective'),[{x:it.map(i=>i.iteration),y:it.map(i=>i.objective),label:'Linearization MAP cost',color:'#53dfad'},
{x:it.map(i=>i.iteration),y:it.map(i=>i.proposal_objective),label:'Trial MAP cost (rejections recorded below)',color:'#ffbd69'}],{title:'Convergence is separate from statistical consistency'});
let tag=document.createElement('script');tag.src='debug_frames/cloud_'+String(k).padStart(4,'0')+'.js';tag.onload=()=>{cloud=window.TEMPORAL_CLOUD;drawCloud();tag.remove()};document.head.appendChild(tag)}
$('frame').onchange=show;$('cam').onchange=()=>{show();if(view&&cloud)view.lookFrom(cloud.cameras[+$('cam').value])};['prior','canonical','support'].forEach(k=>$(k).onchange=drawCloud);
if(view)view.renderer.domElement.addEventListener('click',e=>{if(!cloud)return;let rect=view.renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2(2*(e.clientX-rect.left)/rect.width-1,1-2*(e.clientY-rect.top)/rect.height);let ray=new THREE.Raycaster();ray.params.Points.threshold=.006;ray.setFromCamera(mouse,view.camera);let hits=ray.intersectObject(pointsObj);if(hits.length){$('point').value=cloud.ids[hits[0].index];showPoint()}});
function showPoint(){let caption={title:'Original ID '+$('point').value+' / '+D.manifest.temporal_mode+' / '+(D.manifest.noise_assumption||'NO_ESTIMATION')};let records=D.point_traces?.[$('point').value]||[],selected=records.find(r=>r.frame==$('frame').value);$('pointstats').textContent=JSON.stringify(selected||{message:'No saved trace for this original ID/frame',available_original_ids:Object.keys(D.point_traces||{})},null,2);
PD.chart($('pointposition'),['X','Y','Z'].map((name,i)=>({x:records.map(r=>r.frame),y:records.map(r=>r.state[i]),label:name+' world [m]',color:['#f4797d','#53dfad','#69a7ff'][i]})),caption);
PD.chart($('pointspeed'),[{x:records.map(r=>r.frame),y:records.map(r=>Math.hypot(...r.state.slice(3))),label:'Estimated speed [m/s]; rejected updates explicit below',color:'#ffbd69'}],caption);
PD.chart($('pointuncertainty'),['X','Y','Z'].map((name,i)=>({x:records.map(r=>r.frame),y:records.map(r=>r.position_std_m[i]),label:name+' conditional position std [m]',color:['#f4797d','#53dfad','#69a7ff'][i]})),caption);
let local=records.filter(r=>r.local_retained);$('localpanel').style.display=local.length||D.manifest.temporal_mode==='local-correlated'?'block':'none';
if(local.length){const kinds=[['state','#53dfad'],['source_chart','#ffbd69'],['raw_sensor_error','#69a7ff']];
PD.chart($('pointlocal'),kinds.map(([k,c])=>({x:local.map(r=>r.frame),y:local.map(r=>r.local_retained[k]),label:'retained '+k+' blocks',color:c})),caption);
PD.chart($('pointlocalcross'),kinds.map(([k,c])=>({x:local.map(r=>r.frame),y:local.map(r=>r.local_cross_norm[k]>0?Math.log10(r.local_cross_norm[k]):NaN),label:'log10 ||Cov(s_t, '+k+')|| (posterior, before expiry)',color:c})),caption);
$('pointlocalstatus').textContent=local.map(r=>'frame '+r.frame+': '+r.status).join('\n')}}$('point').onchange=showPoint;
let play=null;$('play').onclick=()=>{if(play){clearInterval(play);play=null;return}play=setInterval(()=>{$('frame').selectedIndex=($('frame').selectedIndex+1)%Math.max(1,$('frame').options.length);show()},800)};show();
const ctx=$('plot').getContext('2d'),H=D.history;const series=[['observed_points','#53dfad'],['assimilated_points','#69a7ff'],['prediction_only','#ffbd69'],['unexplained_foreground','#5ae1ff']];
series.forEach(([key,col],j)=>{ctx.fillStyle=col;ctx.fillText(key==='prediction_only'&&D.manifest.temporal_mode==='full'?'no direct row (joint marginal)':key,20+j*350,20);let max=Math.max(1,...H.map(h=>h[key]||0));ctx.beginPath();ctx.strokeStyle=col;H.forEach((h,i)=>{let x=20+i*1360/Math.max(1,H.length-1),y=220-(h[key]||0)*180/max;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});
PD.chart($('motion'),[{x:H.map(h=>h.ordinal),y:H.map(h=>h.speed_median_m_s),label:'Median speed m/s',color:'#53dfad'},{x:H.map(h=>h.ordinal),y:H.map(h=>h.speed_p95_m_s),label:'p95 speed m/s',color:'#ffbd69'}]);
PD.chart($('covariance'),[{x:H.map(h=>h.ordinal),y:H.map(h=>h.posterior_position_variance_median_m2>0?Math.log10(h.posterior_position_variance_median_m2):NaN),label:'log10 median position covariance eigenvalue [m²]',color:'#69cfff'}]);
if(H.some(h=>h.local_memory))PD.chart($('localdims'),[{x:H.map(h=>h.ordinal),y:H.map(h=>h.local_memory?.local_dimension_quantiles_min_median_p95_p99_max?.[1]),label:'median retained local dimension',color:'#53dfad'},
{x:H.map(h=>h.ordinal),y:H.map(h=>h.local_memory?.local_dimension_quantiles_min_median_p95_p99_max?.[2]),label:'p95 retained local dimension',color:'#ffbd69'},
{x:H.map(h=>h.ordinal),y:H.map(h=>h.local_memory?.local_dimension_quantiles_min_median_p95_p99_max?.[4]),label:'max retained local dimension',color:'#f4797d'}]);
</script></body></html>''')
