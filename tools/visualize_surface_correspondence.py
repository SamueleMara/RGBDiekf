#!/usr/bin/env python3
"""Interactive before/after views from audited masks and cached coordinates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.frozen import FrozenCache
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.observations import samples
from pointwise_rgbd.store import json_write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--frames',type=int,nargs='+',default=[0,1,38,39,80,160,240,327]);args=parser.parse_args()
    root=args.audit;manifest=json.loads((root/'manifest.json').read_text());cache=FrozenCache(manifest['cache']);before=cache.frozen_signature();data=Dataset(cache.manifest['dataset']['input'],**cache.manifest['dataset']['test_limits']);config=cache.config()
    frames=[k for k in args.frames if k in manifest['frames']];payload=[]
    for k in frames:
        state,p,color,ids,observed=cache.frame(k);cache.verify_frame(k,state,p)
        display=np.linspace(0,len(ids)-1,min(4096,len(ids)),dtype=int)
        views=[]
        for ci in range(len(data.cameras)):
            frame=data.observation(k,ci);raw=samples(frame,config)
            with np.load(root/'raw_classification'/f'{k:06d}_{ci:03d}.npz') as values:
                masks=np.c_[values['baseline_owner']>=0,values['geometry_explained'],values['appearance_explained']]
                owned=np.isin(values['baseline_kind'],[1,3])&(values['baseline_owner']>=0)
                select=np.linspace(0,len(raw.pixel)-1,min(6000,len(raw.pixel)),dtype=int) if len(raw.pixel) else np.array([],int)
                lower=raw.xyz.min(0) if len(raw.xyz) else np.zeros(3);upper=raw.xyz.max(0) if len(raw.xyz) else np.ones(3)
                views.append({'camera':frame.camera.name,'xyz':raw.xyz[select].tolist(),'mask':masks[select].tolist(),'owned':owned[select].tolist(),'raw_count':len(raw.pixel),'counts':masks.sum(0).tolist(),'frozen_conflicts':int((owned&~masks[:,2]).sum()),'bounds':[lower.tolist(),upper.tolist()]})
                # Three actual world-coordinate views with identical bounds;
                # coordinate sampling below is DISPLAY ONLY, never inference.
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                directory=root/'visuals'/f'{k:06d}_{ci:03d}';directory.mkdir(parents=True,exist_ok=True)
                fig=plt.figure(figsize=(16,5))
                for policy,title in enumerate(('Old exclusive ownership','Geometric kNN coverage','Geometric + RGB coverage')):
                    ax=fig.add_subplot(1,3,policy+1,projection='3d')
                    ax.scatter(*state[display,:3].T,s=.5,c='#668eaa',alpha=.25,label='Transported SOLVED')
                    for explained,tint,label in [(True,'#00aa55','Explained observations'),(False,'#ff3355','Unexplained under policy')]:
                        points=raw.xyz[select[masks[select,policy]==explained]]
                        if len(points):ax.scatter(*points.T,s=1,c=tint,label=label)
                    ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(lower[2],upper[2]),xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title=title)
                    ax.legend(fontsize=6,loc='upper right')
                fig.suptitle(f'Actual frame {data.frame_ids[k]}, {frame.camera.name}: same world bounds; display samples only; red is policy-incompatible, not verified new geometry')
                fig.tight_layout();fig.savefig(directory/'world_comparison.png',dpi=160);plt.close(fig)
                # This is a diagnostic overlay, not input selection or an update.
                directory=root/'visuals'/f'{k:06d}_{ci:03d}';directory.mkdir(parents=True,exist_ok=True)
                rgb=np.uint8(np.clip(frame.rgb,0,1)*255);flat=rgb.reshape(-1,3)
                conflict=owned&~masks[:,2];flat[raw.pixel[conflict]]=np.uint8(.4*flat[raw.pixel[conflict]]+.6*np.array([255,220,0]))
                image=Image.fromarray(rgb);draw=ImageDraw.Draw(image);draw.text((5,5),'YELLOW: frozen measurement owner, but current surface/appearance incompatible',fill='yellow');image.save(directory/'frozen_measurement_conflicts.png')
        payload.append({'frame':k,'original_frame':data.frame_ids[k],'time':float(data.times[k]),'solved':state[display,:3].tolist(),'solved_observed':observed[display].tolist(),'solved_count':len(ids),'views':views})
    from plotly.offline import get_plotlyjs
    asset=root/'visuals/plotly.min.js';asset.parent.mkdir(exist_ok=True)
    if not asset.exists():asset.write_text(get_plotlyjs())
    page='''<!doctype html><html><head><meta charset="utf-8"><script src="visuals/plotly.min.js"></script></head><body style="font-family:sans-serif">
<h2>Frozen RGB-D surface correspondence: before / geometry / geometry + appearance</h2>
<p>All numerical classifications use every valid foreground observation. This browser samples coordinates FOR DISPLAY ONLY. No filtering, trajectory fitting or interpolation was performed. The initial viewport uses measured-surface bounds; use “Show all frozen estimates” to inspect outlying predictions. Red means incompatible under the declared tests, not proof of missing physical geometry.</p>
<label>Representative frame <input id="frame" type="range" min="0" value="0"></label>
<label>Camera <select id="camera"></select></label>
<label>Policy <select id="mode"><option value="0">Original exclusive-owner coverage</option><option value="1">Observed → SOLVED geometric kNN</option><option value="2" selected>Geometric + RGB cosine / brightness</option></select></label>
<button onclick="full=false;draw()">Measured surface view</button><button onclick="full=true;draw()">Show all frozen estimates</button>
<p id="status"></p><div id="plot" style="height:78vh"></div><script>
const data=DATA;let full=false;const f=document.getElementById('frame'),c=document.getElementById('camera'),mode=document.getElementById('mode');f.max=data.length-1;f.value=data.length-1;
for(let i=0;i<data[0].views.length;i++){const o=document.createElement('option');o.value=i;o.textContent=data[0].views[i].camera;c.appendChild(o);}
function trace(name,xyz,color,size){return {type:'scatter3d',mode:'markers',name,x:xyz.map(p=>p[0]),y:xyz.map(p=>p[1]),z:xyz.map(p=>p[2]),marker:{size,color}};}
function draw(){const t=data[+f.value],v=t.views[+c.value],m=+mode.value;const yes=[],no=[],conflict=[];
for(let i=0;i<v.xyz.length;i++){(v.mask[i][m]?yes:no).push(v.xyz[i]);if(v.owned[i]&&!v.mask[i][m])conflict.push(v.xyz[i]);}
document.getElementById('status').textContent=`Original frame ${t.original_frame}, ${t.time.toFixed(3)} s, ${v.camera}: ${v.counts[m]}/${v.raw_count} explained. Cached SOLVED: ${t.solved_count}. Appearance-incompatible frozen-owned observations: ${v.frozen_conflicts}. Display samples: ${t.solved.length} SOLVED, ${v.xyz.length} observed.`;
const scene={aspectmode:'data',xaxis:{title:'world X (m)'},yaxis:{title:'world Y (m)'},zaxis:{title:'world Z (m)'}};
if(!full){for(const [a,i] of [['xaxis',0],['yaxis',1],['zaxis',2]])scene[a].range=[v.bounds[0][i],v.bounds[1][i]];}
Plotly.react('plot',[trace('Transported SOLVED estimates (includes predictions)',t.solved,'#668eaa',2),trace('Explained observations',yes,'#00bb55',3),trace('Unexplained under selected policy',no,'#ff3355',3),trace('Frozen-measurement consistency conflict',conflict,'#edbd00',4)],{scene,uirevision:full?'full':'measured'},{responsive:true});}
f.oninput=draw;c.onchange=draw;mode.onchange=draw;draw();
</script></body></html>'''
    (root/'coverage_comparison.html').write_text(page.replace('DATA',json.dumps(payload,allow_nan=False,separators=(',',':'))))
    cache.assert_old_frozen(before)
    json_write(root/'visualization_provenance.json',{'frames':frames,'camera_count':len(data.cameras),'display_only_limits':{'SOLVED':4096,'observed':6000},'inference_subsampling':False,'source':'actual saved classifications and frozen world coordinates','cache_unchanged':True})
    print(str(root/'coverage_comparison.html'))


if __name__=='__main__':main()
