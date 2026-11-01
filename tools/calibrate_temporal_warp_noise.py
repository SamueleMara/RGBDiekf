#!/usr/bin/env python3
"""GT-free identifiability study, not automatic residual-to-R_warp conversion.
RGB optical flow is candidate support, NOT independent 3D motion truth. In the
absence of independent state/cross-time uncertainty a calibrated warp covariance
cannot be separated from selection bias and motion; no usable noise file is emitted.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import chi2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene.dataset import Dataset
from scene.config import load_config
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame,SensorNoise
from estimator.temporal_rgbd_terms import validate_reference_patch,bilinear,sensor_covariance,raw_values
from estimator.temporal_noise import temporal_sensor_only
from utils.io import json_write,serial

def robust_covariance(values,tail=.01):
 values=np.asarray(values,dtype=float)
 if len(values)<5:return None
 center=np.median(values,axis=0);scale=1.4826*np.median(np.abs(values-center),axis=0)
 active=scale>0;score=np.sum(((values[:,active]-center[active])/scale[active])**2,axis=1)
 cutoff=chi2.isf(tail,max(1,int(active.sum())))
 weights=np.sqrt(cutoff/np.maximum(score,cutoff))
 delta=values-center
 cov=(delta*weights[:,None]).T@(delta*weights[:,None])/np.square(weights).sum()
 return {'center':center.tolist(),'covariance':cov.tolist(),'effective_samples':float(weights.sum()**2/(weights@weights)),
         'method':'coordinate MAD standardized radial Huber weights; chi-square .01 tail; weighted second moment, no claim of unbiased Gaussian calibration'}

def disjoint_temporal_pairs(pairs):
 pairs=sorted(set(pairs));train=pairs[:len(pairs)//2];boundary=max(train)+1 if train else -1
 heldout=[k for k in pairs[len(pairs)//2:] if k>boundary]
 excluded=[k for k in pairs[len(pairs)//2:] if k<=boundary]
 return train,heldout,excluded

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scene',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 ap.add_argument('--pairs',default='0,1,2,5,8');ap.add_argument('--max-points-per-camera',type=int,default=600)
 ap.add_argument('--sensor-noise',choices=['measured','config'],default='measured',help='measured: per-camera conditional MAD from calibration-block frames ONLY; config: documented initial guesses')
 a=ap.parse_args()
 if a.max_points_per_camera<1:ap.error('max points must be positive')
 if a.output.exists() and any(a.output.iterdir()):ap.error('output must be new/empty')
 a.output.mkdir(parents=True,exist_ok=True)
 data=Dataset(a.scene);cfg=load_config();rig=CameraRig(data.cameras,'cpu');noise=SensorNoise.from_config(cfg,json.loads(Path('configs/projective_rgbd.json').read_text()),'cpu')
 policy=json.loads(Path('configs/projective_rgbd.json').read_text());pairs=sorted(set(int(k) for k in a.pairs.split(',') if 0<=int(k)<len(data.times)-1))
 allrows=[];breakdown=[];digest=hashlib.sha256();rejections={'LK_or_foreground_support':0,'reference_patch_invalid':0}
 train_pairs,heldout_pairs,excluded_shared_boundary=disjoint_temporal_pairs(pairs)
 noises=[];sensor_reports=[]
 for c,camera in enumerate(data.cameras):
  effective=cfg;doc={'camera':c,'policy':'documented-initial-guess','source':cfg.manifest()}
  if a.sensor_noise=='measured' and train_pairs:
   end=max(train_pairs)+2
   view=SimpleNamespace(times=data.times[:end],cameras=[camera],root=data.root,observation=lambda k,unused,c=c:data.observation(k,c))
   effective,_,estimate=temporal_sensor_only(view,cfg,policy)
   doc={'camera':c,'policy':'estimated-from-input-under-assumptions','training_frames':list(range(end)),
        'held_out_frames_used':False,'estimate':estimate,'effective_sensor':effective.manifest()}
   if effective.depth_variance<=0 and effective.depth_squared_variance<=0:
    effective=cfg;doc.update(policy='documented-initial-guess',reason='conditional depth MAD/quantization estimate is zero; not a positive sensor covariance; retain documented guess explicitly')
  noises.append(SensorNoise.from_config(effective,policy,'cpu'));sensor_reports.append(doc)
 for k in train_pairs+heldout_pairs:
  f0=MeasuredFrame(data,k,'cpu',torch.float64);f1=MeasuredFrame(data,k+1,'cpu',torch.float64)
  split='calibration' if k in train_pairs else 'held_out_temporal_block'
  for c in range(rig.count):
   noise=noises[c]
   rgb0=f0.rgb[c].numpy();rgb1=f1.rgb[c].numpy();mask0=(f0.mask[c]&f0.raw_valid[c]&torch.isfinite(f0.rgb[c]).all(-1)).numpy();mask1=(f1.mask[c]&f1.raw_valid[c]&torch.isfinite(f1.rgb[c]).all(-1)).numpy()
   for image in (rgb0,rgb1,f0.depth[c].numpy(),f1.depth[c].numpy(),mask0,mask1):digest.update(np.ascontiguousarray(image).tobytes())
   # Full LK patch plus interpolation neighbors must be foreground in BOTH images.
   safe=cv2.erode(mask1.astype(np.uint8),np.ones((23,23),np.uint8)).astype(bool)
   y,x=np.where(safe);stride=max(1,int(np.ceil(len(x)/a.max_points_per_camera)));x,y=x[::stride],y[::stride]
   if not len(x):continue
   p=np.stack((x,y),-1).astype(np.float32).reshape(-1,1,2)
   g0=cv2.cvtColor(np.clip(rgb0*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY);g1=cv2.cvtColor(np.clip(rgb1*255,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
   g0[~mask0]=0;g1[~mask1]=0
   q,status,_=cv2.calcOpticalFlowPyrLK(g1,g0,p,None,winSize=(21,21),maxLevel=0)
   back,bs,_=cv2.calcOpticalFlowPyrLK(g0,g1,q,None,winSize=(21,21),maxLevel=0)
   qp=q[:,0];ix=np.rint(qp).astype(int);inimage=(ix[:,0]>=11)&(ix[:,0]<f0.W-11)&(ix[:,1]>=11)&(ix[:,1]<f0.H-11)
   safe0=cv2.erode(mask0.astype(np.uint8),np.ones((23,23),np.uint8)).astype(bool)
   keep=status[:,0].astype(bool)&bs[:,0].astype(bool)&inimage
   keep&=safe0[np.clip(ix[:,1],0,f0.H-1),np.clip(ix[:,0],0,f0.W-1)]
   rejections['LK_or_foreground_support']+=int((~keep).sum())
   pxy=torch.tensor(p[:,0][keep],dtype=torch.float64);qxy=torch.tensor(qp[keep],dtype=torch.float64);cams=torch.full((len(pxy),),c,dtype=torch.long)
   keys,vals,reason,_=validate_reference_patch(f0,cams,qxy,rig,noise,policy)
   rejections['reference_patch_invalid']+=int((reason!=0).sum());ok=reason==0
   if not bool(ok.any()):continue
   pp=pxy[ok].long();cams=cams[ok];qxy=qxy[ok];vals=vals[ok]
   warped,grad,weights,*_=bilinear(qxy,vals)
   current=raw_values(f1,cams,pp[:,1]*f1.W+pp[:,0])
   R=sensor_covariance(current,noise)+(weights.square()[...,None,None]*sensor_covariance(vals,noise)).sum(1)
   difference=current-warped
   fb=np.linalg.norm(back[:,0][keep][ok.numpy()]-p[:,0][keep][ok.numpy()],axis=1)
   record={'frame':k+1,'reference_frame':k,'camera':c,'split':split,'count':len(pp),
      'forward_backward_pixels_quantiles':np.quantile(fb,[0,.5,.95,1]).tolist(),
      'depth_median_m':float(current[:,3].median()),'gradient_norm_median':float(grad.norm(dim=(-1,-2)).median())}
   digest.update(current.numpy().tobytes());digest.update(vals.numpy().tobytes());digest.update(qxy.numpy().tobytes())
   breakdown.append(record)
   edge=cv2.distanceTransform(mask1.astype(np.uint8),cv2.DIST_L2,5)
   for n in range(len(pp)):allrows.append({'split':split,'difference':difference[n].tolist(),'sensor_covariance':R[n].tolist(),'frame':k+1,'camera':c,
      'depth_m':float(current[n,3]),'gradient_norm':float(grad[n].norm()),'warp_pixels':float((qxy[n]-pp[n]).norm()),
      'forward_backward_pixels':float(fb[n]),'mask_edge_distance_pixels':float(edge[int(pp[n,1]),int(pp[n,0])])})
   print(k,c,len(pp),split,flush=True)
 report={'model_version':2,'status':'MODEL_DISCREPANCY_NOT_IDENTIFIABLE','covariance_semantics':'additional_conditional_warp_model_discrepancy',
  'assumption_class':'UNCALIBRATED_CANDIDATE_CORRESPONDENCE_STUDY','units':['encoded_rgb_0_1']*3+['m'],
  'method':'same-camera single-scale Lucas-Kanade candidates; full 21px patch inside valid foreground; strict existing 2x2 depth-surface validation; temporal-block holdout',
  'LK_window_pixels':21,'LK_pyramid_levels':0,'source_scene':str(a.scene),'times':data.times.tolist(),'cameras':rig.count,
  'sensor_model':{'policy':a.sensor_noise,'initial_documented_guess':cfg.manifest(),'per_camera':sensor_reports},
  'data_sha256':digest.hexdigest(),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'breakdown':breakdown,'rejections':rejections,'candidate_counts':{},'empirical_temporal_difference':{},'conditional_breakdowns':{},
  'data_hash_scope':'all RGB, metric depth and foreground/valid arrays used by the candidate flow and support selection; plus sampled support arrays',
  'calibration_hash':rig.calibration_hash(),
  'excluded_shared_boundary_pairs':excluded_shared_boundary,'held_out_protocol':'no frame shared between calibration and held-out blocks',
  'effective_count_qualification':'weighted robust count, NOT independent sample degrees of freedom; spatial correlation and selection remain',
  'not_identifiable_reasons':['RGB-derived correspondence is not independent of RGB model residual; flow estimator selection conditions the residual',
   'World motion and current/reference/cross-time state covariance are not independently known from this support',
   'Measured depth temporal difference includes real Z motion; using backprojected matching depths to subtract Z would cancel the tested measurement',
   'Sensor initial guesses are conditional, not independently calibrated on this scene; a residual excess cannot uniquely identify model discrepancy'],
  'covariance_rgbd':None,'calibration_file_usable_for_inference':False,'ground_truth_used':False}
 for split in ('calibration','held_out_temporal_block'):
  rows=[r for r in allrows if r['split']==split];report['candidate_counts'][split]=len(rows)
  if not rows:continue
  values=np.asarray([r['difference'] for r in rows]);base=np.mean([r['sensor_covariance'] for r in rows],axis=0)
  cov=robust_covariance(values);cov['sensor_contribution_mean']=base.tolist()
  cc=np.asarray(cov['covariance']);sd=np.sqrt(np.diag(cc));den=sd[:,None]*sd[None,:]
  cov['correlation_matrix']=np.divide(cc,den,out=np.zeros_like(cc),where=den>0).tolist()
  cov['empirical_minus_sensor_eigenvalues_diagnostic_only']=np.linalg.eigvalsh(np.asarray(cov['covariance'])-base).tolist()
  cov['PSD_projection_applied']=False
  z=np.einsum('ni,nij,nj->n',values,np.linalg.inv(np.asarray([r['sensor_covariance'] for r in rows])),values)
  cov['sensor_only_temporal_difference_99pct_fraction']=float(np.mean(z<=chi2.isf(.01,4)))
  cov['coverage_qualification']='NOT calibrated NIS: contains actual motion and selection bias; state uncertainty unseparated'
  report['empirical_temporal_difference'][split]=cov
  details={}
  for camera in range(rig.count):
   group=[r for r in rows if r['camera']==camera]
   if group:details[f'camera_{camera}']={'count':len(group),'robust':robust_covariance([r['difference'] for r in group])}
  for name in ('depth_m','gradient_norm','warp_pixels','mask_edge_distance_pixels','forward_backward_pixels'):
   feature=np.asarray([r[name] for r in rows]);edges=np.quantile(feature,[0,1/3,2/3,1]);bins=[]
   for i in range(3):
    keep=(feature>=edges[i])&((feature<=edges[i+1]) if i==2 else (feature<edges[i+1]))
    bins.append({'data_quantile_range':[float(edges[i]),float(edges[i+1])],'count':int(keep.sum()),'robust':robust_covariance(values[keep])})
   details[name]=bins
  report['conditional_breakdowns'][split]=details
 json_write(a.output/'calibration_analysis.json',serial(report));json_write(a.output/'candidate_rows.json',allrows)
 (a.output/'NOISE_SEMANTICS_AND_CALIBRATION.md').write_text('# Data-only temporal warp calibration\n\n**MODEL_DISCREPANCY_NOT_IDENTIFIABLE**. No usable covariance file emitted.\n\n'+ '\n'.join('- '+x for x in report['not_identifiable_reasons'])+'\n\nSee calibration_analysis.json for actual robust 4x4 candidate differences, sensor contributions, temporal held-out checks and provenance. Negative unexplained eigenvalues are reported without clipping.\n')
 print(a.output/'calibration_analysis.json')
if __name__=='__main__':main()
