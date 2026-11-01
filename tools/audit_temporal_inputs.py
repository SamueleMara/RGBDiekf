#!/usr/bin/env python3
"""Geometry/resource preflight ONLY. No temporal estimation or invented warp R.
Images in this output show the unchanged measured endpoint cloud projected into
selected frames. They must not be described as optimized trajectories.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene.dataset import Dataset
from scene.config import load_config
from estimator.free_projective_run import build_catalogue
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame,SensorNoise
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_rgbd_terms import validate_reference_patch,bilinear
from estimator.temporal_run import full_component_preflight
from estimator.temporal_full_graph import FullResourceLimit
from debug.temporal_debug import temporal_frame_debug,write_temporal_dashboard
from utils.io import json_write,serial


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',default='0,1,2');ap.add_argument('--point-ids',default='');ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error('output must be new or empty')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'temporal').mkdir(exist_ok=True)
    start=time.perf_counter();data=Dataset(a.scene);cfg=load_config();pol=json.loads(Path('configs/projective_rgbd.json').read_text())
    rig=CameraRig(data.cameras,a.device);noise=SensorNoise.from_config(cfg,pol,a.device)
    k,xyz,rgb,cov,source,sel,sm=build_catalogue(data,cfg,'forward',{'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':False,'report':False},pol,noise)
    state=torch.zeros((len(xyz),6),dtype=torch.float64,device=a.device);state[:,:3]=torch.tensor(xyz,device=a.device)
    ids=[int(i) for i in a.point_ids.split(',') if i]
    man={'temporal_mode':'GEOMETRY_PREFLIGHT_ONLY_NO_ESTIMATION','reference_uncertainty':'No warp covariance assumed. Cloud is unchanged measured frame-0 geometry.',
         'camera_count':rig.count,'debug_point_ids':ids,'point_count':len(xyz),'timing':data.base.timing_report,'ground_truth_used':False}
    json_write(a.output/'run_manifest.json',serial(man));np.savez_compressed(a.output/'canonical.npz',canonical_xyz=xyz,canonical_rgb=rgb)
    json_write(a.output/'status.json',{'status':'GEOMETRY_PREFLIGHT_ONLY_NO_ESTIMATION'})
    json_write(a.output/'temporal/config.json',{'warp_covariance':None,'reason':'No warp covariance supplied or invented; geometry only'})
    result={'scene':str(a.scene),'points':len(xyz),'cameras':rig.count,'times':data.times.tolist(),'timing':data.base.timing_report,
            'uncertainty_policy':'Existing documented source/sensor guesses only used for support validation; no filtering, no warp R invented','frames':[]}
    sourceframe=MeasuredFrame(data,0,a.device,torch.float64)
    for k in [int(v) for v in a.frames.split(',')]:
        f=sourceframe if k==0 else MeasuredFrame(data,k,a.device,torch.float64)
        own=first_hit_pixel_owner(state[:,:3],rig,f)
        temporal_frame_debug(a.output,f,own,state,state,rig,None,k)
        c,y,x=torch.where(own.owner_id>=0);p=own.owner_id[c,y,x];pixel=y*f.W+x
        good=f.mask[c,y,x]&f.raw_valid[c,y,x]
        c,p,pixel=c[good],p[good],pixel[good]
        q=torch.stack((pixel%f.W,pixel//f.W),-1).double()
        keys,values,reasons,_=validate_reference_patch(sourceframe,c,q,rig,noise,pol)
        ok=reasons==0
        weights=bilinear(q,values)[2]
        rows=SimpleNamespace(point=p[ok],camera=c[ok],reference_frame=torch.zeros(int(ok.sum()),dtype=torch.long,device=a.device),reference_pixels=keys[ok],weights=weights[ok])
        resource_result='fits lower bound only; actual graph may be larger'
        try:full_component_preflight(rows,source,0,128*1024**2,a.output/'temporal')
        except FullResourceLimit as err:resource_result=str(err)
        rec={'frame':k,'occupied':int((own.owner_id>=0).sum()),'conflicts':int((own.counts>1).sum()),
             'unexplained_foreground':int((f.mask&f.raw_valid&(own.owner_id<0)).sum()),
             'outside_foreground':int(((own.owner_id>=0)&~f.mask).sum()),'valid_reference_patches_at_zero_motion':int(ok.sum()),
             'component_preflight':json.loads((a.output/'temporal/component_preflight.json').read_text()),'resource_result':resource_result,'points':{}}
        if a.device=='cuda':
            cpuown=first_hit_pixel_owner(state[:,:3].cpu(),CameraRig(data.cameras,'cpu'),f)
            rec['CPU_CUDA_owner_disagreement_pixels']=int((cpuown.owner_id!=own.owner_id.cpu()).sum())
            rec['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        for i in ids:
            if i>=len(state):continue
            rec['points'][i]={'source':source[i].tolist(),'camera_projections':[]}
            for c in range(rig.count):
                u,v=torch.round(own.uv[c,i]).long().tolist()
                if 0<=u<f.W and 0<=v<f.H:
                    win=int(own.owner_id[c,v,u]);rec['points'][i]['camera_projections'].append({'camera':c,'pixel':[u,v],
                       'winner':win,'winner_source':source[win].tolist() if win>=0 else None,'projected_z':float(own.z[c,i]),
                       'winner_z':float(own.owner_depth[c,v,u]),'count':int(own.counts[c,v,u])})
        result['frames'].append(rec);json_write(a.output/'audit.json',serial(result));write_temporal_dashboard(a.output,man,[])
        print(f'preflight frame {k}: {len(xyz)} source points, {int(ok.sum())} valid supports, largest component {rec["component_preflight"]["largest_point_components"][:1]}',flush=True)
    result['seconds']=time.perf_counter()-start;result['max_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result['device']=a.device
    json_write(a.output/'audit.json',serial(result));print(a.output/'temporal/index.html')

if __name__=='__main__':main()
