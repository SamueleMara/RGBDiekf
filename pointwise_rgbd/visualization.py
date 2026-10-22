"""Actual stored point tracks and raw RGB-D diagnostics; display sampling only."""
from pathlib import Path
import json,os
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .projection import backproject


def plotly_asset(root):
    from plotly.offline import get_plotlyjs
    path=Path(root)/'assets/plotly.min.js';path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_text(get_plotlyjs())
    return path


def trajectory_viewer(path,store,dataset,root,selected_point=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ids=np.linspace(0,store.n0-1,min(512,store.n0),dtype=int)
    old=np.asarray(store.arrays['state'][:,:,:3])[:,ids,:].tolist()
    old_observed=np.asarray(store.arrays['observed'])[:,ids].tolist()
    chosen=store.added if selected_point is None else [r for r in store.added if r['point_id']==selected_point]
    # Limit only the display payload; no observation or estimator subsampling.
    if len(chosen)>64:chosen=chosen[-64:]
    tracks=[]
    for record in chosen:
        with np.load(record['path']) as data:
            measured=[[] for _ in dataset.times]
            for k,ci,pixel in data['measurement_refs']:
                k,ci,pixel=int(k),int(ci),int(pixel);frame=dataset.observation(k,ci)
                y,x=divmod(pixel,frame.camera.width)
                if not frame.valid[y,x]:raise ValueError('Recorded display measurement became invalid')
                xyz,_=backproject(np.array([x,y,float(frame.depth[y,x])]),frame.camera,np.eye(3))
                measured[k].append(xyz.tolist())
            tracks.append({'id':record['point_id'],'xyz':data['state'][:,:3].tolist(),'observed':data['observed'].tolist(),'source':record['source'],'measured_world':measured})
    payload={'old':old,'old_observed':old_observed,'new':tracks,'times':dataset.times.tolist(),'original_frames':dataset.frame_ids,'n0':store.n0,'display_old_ids':ids.tolist()}
    script=os.path.relpath(plotly_asset(root),path.parent)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Measured pointwise RGB-D tracks</title><script src="SCRIPT"></script></head><body style="font-family:sans-serif">
<h2>Frozen old tracks and independently estimated new points</h2><p>Actual stored 6D estimates. Old geometry is sampled FOR DISPLAY ONLY. Green new markers: directly observed; orange X: prediction only. No rigid transforms, reconstructed animation or fabricated measurements.</p>
<label>Frame <input id="frame" type="range" min="0" value="0"><span id="time"></span></label>
<label>Discovery event <input id="event" type="range" min="0"><span id="eventText"></span></label><button id="play">Play / pause</button>
<div id="plot" style="height:78vh"></div><script>
const data=DATA;let running=false;
const slider=document.getElementById('frame'),eventSlider=document.getElementById('event');slider.max=data.times.length-1;eventSlider.max=data.new.length;eventSlider.value=data.new.length;
function draw(){const k=+slider.value,e=+eventSlider.value,p=data.old[k],added=data.new.slice(0,e),fresh=added.map(t=>t.xyz[k]);
document.getElementById('time').textContent=` original frame ${data.original_frames[k]}, ${data.times[k].toFixed(6)} seconds; ${p.length}/${data.n0} old points displayed`;
document.getElementById('eventText').textContent=` ${e} selected new tracks`;
let traces=[{x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),type:'scatter3d',mode:'markers',name:'Frozen old trajectories (display sample)',marker:{size:2,color:data.old_observed[k].map(v=>v?'#547b9c':'#aab6bf')}}];
if(fresh.length)traces.push({x:fresh.map(v=>v[0]),y:fresh.map(v=>v[1]),z:fresh.map(v=>v[2]),type:'scatter3d',mode:'markers',name:'New independent point estimates',text:added.map(t=>`point ${t.id}; ${t.observed[k]?'REAL_MEASUREMENT':'PREDICTION_ONLY'}`),marker:{size:7,color:added.map(t=>t.observed[k]?'#00aa55':'#ee8800'),symbol:added.map(t=>t.observed[k]?'circle':'x')}});
for(const t of added)traces.push({x:t.xyz.map(v=>v[0]),y:t.xyz.map(v=>v[1]),z:t.xyz.map(v=>v[2]),type:'scatter3d',mode:'lines',line:{width:2},name:`point ${t.id} full estimated path`});
for(const t of added){const m=t.measured_world[k];traces.push({x:m.map(v=>v[0]),y:m.map(v=>v[1]),z:m.map(v=>v[2]),type:'scatter3d',mode:'markers',marker:{size:4,symbol:'diamond',color:'#00cccc'},name:`point ${t.id} actual RGB-D backprojections`});}
Plotly.react('plot',traces,{scene:{aspectmode:'data',xaxis:{title:'world X (m)'},yaxis:{title:'world Y (m)'},zaxis:{title:'world Z (m)'}},uirevision:'fixed'},{responsive:true});}
slider.oninput=draw;eventSlider.oninput=draw;document.getElementById('play').onclick=()=>running=!running;setInterval(()=>{if(running){slider.value=(+slider.value+1)%data.times.length;draw();}},180);draw();
</script></body></html>'''.replace('SCRIPT',script).replace('DATA',json.dumps(payload,allow_nan=False,separators=(',',':')))
    path.write_text(page)
    # Static preview uses precisely the same stored data as this HTML.
    k=len(dataset.times)-1;figure=plt.figure(figsize=(8,6));ax=figure.add_subplot(111,projection='3d');xyz=np.asarray(old[k]);ax.scatter(*xyz.T,s=2,c='#547b9c',label='Frozen old (display sample)')
    for track in tracks:
        xyz=np.asarray(track['xyz']);ax.plot(*xyz.T,label=f"point {track['id']}");ax.scatter(*xyz[k],s=60,c='#00aa55' if track['observed'][k] else '#ee8800')
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f'Actual HTML data at frame {dataset.frame_ids[k]}');ax.legend();figure.tight_layout();figure.savefig(path.with_suffix('.preview.png'));plt.close(figure)


def event_plots(directory,track,dataset):
    directory=Path(directory);times=dataset.times;s=track['state'];p=track['covariance'];obs=track['observed']
    fig,axes=plt.subplots(2,1,figsize=(9,6))
    for i,name in enumerate(('X','Y','Z')):
        axes[0].plot(times,s[:,i],label=name);axes[0].scatter(times[obs],s[obs,i],s=12)
    axes[0].set(ylabel='World position (m)',title='Stored point trajectory; dots indicate direct RGB-D updates');axes[0].legend()
    axes[1].plot(times,np.linalg.norm(s[:,3:],axis=1));axes[1].set(xlabel='Seconds',ylabel='Speed (m/s)');fig.tight_layout();fig.savefig(directory/'trajectory.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6));axes[0].plot(times,np.trace(p,axis1=1,axis2=2),label='Full P trace (mixed units)');axes[0].plot(times,np.trace(p[:,:3,:3],axis1=1,axis2=2),label='Position trace (m²)');axes[0].legend()
    axes[1].plot(times,track['nis'],'o');axes[1].set(xlabel='Seconds',ylabel='Observed-update NIS',title='No fabricated NIS for missing observations or anchor initialization');fig.tight_layout();fig.savefig(directory/'covariance_nis.png');plt.close(fig)


def image_overlays(directory,track,source,store,dataset,config):
    directory=Path(directory)
    for k in sorted(set([0,source.frame,len(dataset.times)-1])):
        old=store.arrays['state'][k,::max(1,store.n0//1000),:3]
        for ci,camera in enumerate(dataset.cameras):
            frame=dataset.observation(k,ci);pixels=np.uint8(np.clip(np.asarray(frame.rgb),0,1)*255)
            tint=np.zeros_like(pixels);tint[...,1]=255
            pixels[frame.mask]=np.uint8(.85*pixels[frame.mask]+.15*tint[frame.mask])
            image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
            world=np.r_[old,track['state'][k,None,:3]];q=world@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
            uv=q[:,:2]/q[:,2,None]*camera.intrinsics[:2]+camera.intrinsics[2:]
            valid=(q[:,2]>0)&np.isfinite(uv).all(1)
            for xy in uv[:-1][valid[:-1]]:
                x,y=xy
                if 0<=x<camera.width and 0<=y<camera.height:draw.ellipse((x-1,y-1,x+1,y+1),fill='#4477cc')
            if valid[-1]:
                x,y=uv[-1];draw.ellipse((x-5,y-5,x+5,y+5),outline='#00ffff',width=2)
            residual=np.full(frame.depth.shape,np.nan)
            for row in track['associations']:
                if row['frame']!=k or row['camera']!=ci or row.get('pixel') is None:continue
                y,x=divmod(row['pixel'],camera.width)
                accepted=row['reason'] in ('REAL_MEASUREMENT','ANCHOR_INITIALIZED_ONCE')
                draw.rectangle((x-4,y-4,x+4,y+4),outline='#00ff55' if accepted else '#ff4433',width=2)
                if accepted and 'residual' in row:residual[y,x]=row['residual'][2]
            if k==source.frame and ci==source.camera:
                y,x=divmod(source.pixel,camera.width);draw.ellipse((x-8,y-8,x+8,y+8),outline='#ff00ff',width=3)
            draw.text((6,6),f'Original frame {dataset.frame_ids[k]}, camera {camera.name}; old blue / prediction cyan / raw match green / anchor magenta',fill='white')
            image.save(directory/f'image_overlay_{k:06d}_{ci:03d}.png')
            fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(np.ma.masked_invalid(residual),cmap='coolwarm');fig.colorbar(im,ax=ax,label='Observed Z minus predicted Z (m)')
            ax.set_title('Only recorded real RGB-D target residuals; missing pixels are masked');fig.tight_layout();fig.savefig(directory/f'depth_residual_{k:06d}_{ci:03d}.png');plt.close(fig)
