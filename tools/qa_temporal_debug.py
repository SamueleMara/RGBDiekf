#!/usr/bin/env python3
"""Browser-check ACTUAL temporal estimates, charts, original IDs and 3D data.
Uses the repository's existing local headless-Chrome CDP helper. No inference.
"""
import argparse,json,sys,time,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.qa_projective_debug import Browser

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--point-id',type=int);a=ap.parse_args()
 if a.output.exists() and any(a.output.iterdir()):ap.error('output must be new/empty')
 a.output.mkdir(parents=True,exist_ok=True);B=Browser(a.run.resolve());checks={}
 try:
  session=B.open('free_forward/temporal/index.html')
  ready=False
  for _ in range(100):
   ready=B.js(session,"typeof D!=='undefined'&&D.history.length>0&&typeof cloud!=='undefined'&&cloud&&document.getElementById('panel').complete")
   if ready:break
   time.sleep(.1)
  checks['actual_estimated_frames_loaded']=bool(ready)
  if a.point_id is not None:B.js(session,f"document.getElementById('point').value={a.point_id};showPoint()")
  payload=B.js(session,"({mode:document.getElementById('mode').textContent,frame:+document.getElementById('frame').value,point:+document.getElementById('point').value,pointstats:JSON.parse(document.getElementById('pointstats').textContent),cloud_points:cloud?.ids.length,errors:window.__errors||[]})")
  checks['no_javascript_errors']=not payload['errors'];checks['actual_point_state_populated']='state' in payload['pointstats'];checks['original_ID_exists']=str(payload['point']) in B.js(session,'Object.keys(D.point_traces)')
  pixels=B.js(session,"['pointposition','pointspeed','pointuncertainty'].map(id=>{let c=document.getElementById(id),v=c.getContext('2d').getImageData(0,0,c.width,c.height).data,n=0;for(let i=0;i<v.length;i+=4)if(v[i]!=22||v[i+1]!=27||v[i+2]!=34)n++;return {id,non_background_pixels:n}})")
  checks['trajectory_speed_covariance_charts_drawn']=all(v['non_background_pixels']>100 for v in pixels)
  old=B.js(session,'JSON.stringify(cloud.xyz)');B.js(session,"document.getElementById('frame').selectedIndex=0;show();new Promise(r=>setTimeout(r,300))")
  checks['frame_selector_changes_actual_geometry']=old!=B.js(session,'JSON.stringify(cloud.xyz)')
  B.js(session,"document.getElementById('frame').selectedIndex=document.getElementById('frame').options.length-1;show();new Promise(r=>setTimeout(r,300))")
  original=B.js(session,"document.getElementById('panel').src");B.js(session,"document.getElementById('cam').value=1;document.getElementById('cam').onchange();new Promise(r=>setTimeout(r,300))")
  checks['camera_selector_changes_measured_panel']=original!=B.js(session,"document.getElementById('panel').src")
  for target,name in [('viewport','actual_3d.png'),('pointposition','actual_point_trajectory.png')]:
   box=B.js(session,f"(()=>{{let r=document.getElementById('{target}').getBoundingClientRect();return {{x:0,y:r.top+scrollY,width:Math.min(1500,document.documentElement.clientWidth),height:1000,scale:1}}}})()")
   shot=B.pipe.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':box},session)
   (a.output/name).write_bytes(base64.b64decode(shot['data']))
  report={'checks':checks,'page':payload,'chart_pixels':pixels,'pass':all(checks.values()),'ground_truth_used':False,'scope':'visual/interaction integrity only, not scientific motion acceptance'}
  (a.output/'qa.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return 0 if report['pass'] else 1
 finally:B.close()
if __name__=='__main__':raise SystemExit(main())
