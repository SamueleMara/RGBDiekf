#!/usr/bin/env python3
"""Independent SciPy dense QR oracle on ACTUAL saved, bounded components.
Also compares CPU/CUDA ownership and two-time warp operators at identical cached
real trajectories. No inference, calibration, GT, or point selection is changed.
"""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene.dataset import Dataset
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame
from estimator.temporal_zbuffer import first_hit_pixel_owner,selected_owners
from estimator.temporal_rgbd_terms import temporal_prediction
from estimator.temporal_full_graph import TemporalWindowGraph
from utils.io import json_write,serial

def dense_oracle(J,r,hard):
 C,c=J[hard],r[hard];A,b=J[~hard],r[~hard]
 if len(C):
  Z=la.null_space(C,rcond=np.finfo(float).eps*max(C.shape));d0=la.lstsq(C,-c,lapack_driver='gelsd')[0]
 else:Z=np.eye(J.shape[1]);d0=np.zeros(J.shape[1])
 if Z.shape[1]:
  Q,R=la.qr(A@Z,mode='economic');dz=la.solve_triangular(R,-Q.T@(b+A@d0));W=Z@la.solve_triangular(R,np.eye(len(R)))
  return d0+Z@dz,W@W.T,float(np.linalg.cond(R))
 return d0,np.zeros((len(d0),len(d0))),1.

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--direction',choices=['forward','backward'],default='forward');a=ap.parse_args()
 if a.output.exists():ap.error('output exists; never overwrite an oracle report')
 root=a.run/f'free_{a.direction}';globalman=json.loads((a.run/'run_manifest.json').read_text());man=json.loads((root/'run_manifest.json').read_text())
 report={'run':str(a.run),'ground_truth_used':False,'device':a.device,'components':[],'geometry':[],'warp_operator':[],
         'scope':'same real saved linearization and raw support, independent SciPy oracle; not proof of physical correspondence or global MAP convergence'}
 for path in sorted((root/'temporal/traced_points').glob('frame_*_component_*.npz')):
  data=np.load(path)
  if 'linearized_J' not in data:continue
  J,r,hard=data['linearized_J'],data['linearized_residual'],data['constraint_rows']
  step,cov,condition=dense_oracle(J,r,hard);graph=TemporalWindowGraph(a.device)
  delta,P,stats=graph.constrained_system(graph.tensor(J),graph.tensor(r),torch.tensor(hard,device=a.device))
  cached=data['marginal_covariance'];error=float(np.max(np.abs(cov-cached)));scale=max(float(np.abs(cov).max()),np.finfo(float).tiny)
  step_error=float(np.max(np.abs(step-delta.cpu().numpy())));step_scale=max(float(np.abs(step).max()),np.finfo(float).tiny)
  state_indices=[];velocity_indices=[]
  for key,lo,hi in ast.literal_eval(str(data['layout'])):
   if key[0]=='state':state_indices.extend(range(lo,lo+3));velocity_indices.extend(range(lo+3,hi))
  difference=step-delta.cpu().numpy();soft=J[~hard];v=soft@step+r[~hard]
  Z=la.null_space(J[hard],rcond=np.finfo(float).eps*max(J[hard].shape)) if hard.any() else np.eye(J.shape[1])
  B=soft@Z;den=np.linalg.norm(B)*(np.linalg.norm(B)*np.linalg.norm(step)+np.linalg.norm(r[~hard]))
  report['components'].append({'file':str(path),'rows':len(J),'scalars':J.shape[1],'free_QR_condition':condition,
    'dense_SciPy_vs_saved_Torch_covariance_max_abs':error,'covariance_relative_max':error/scale,
    'dense_SciPy_vs_current_Torch_covariance_max_abs':float(np.max(np.abs(cov-P.cpu().numpy()))),
    'step_max_abs':step_error,'dense_step_max_abs':step_scale,'step_relative_max':step_error/step_scale,
    'step_position_max_abs_m':float(np.abs(difference[state_indices]).max()) if state_indices else None,
    'step_velocity_max_abs_m_s':float(np.abs(difference[velocity_indices]).max()) if velocity_indices else None,
    'dense_reduced_normalized_stationarity':float(np.linalg.norm(B.T@v)/max(den,np.finfo(float).tiny)),
    'equality_residual_norm':float(np.linalg.norm(J[hard]@step+r[hard])),'constraint_rows':int(hard.sum())})
 data=Dataset(Path(globalman['arguments']['scene']));rigcpu=CameraRig(data.cameras,'cpu');rigdev=CameraRig(data.cameras,a.device)
 X=np.load(root/'posterior_xyz.npy',mmap_mode='r');can=np.load(root/'canonical.npz');keys=man['debug_point_ids']
 for k in man['debug_frames']:
  frame=MeasuredFrame(data,k,'cpu',torch.float64);points=torch.tensor(np.asarray(X[k]).copy(),dtype=torch.float64)
  cpu=first_hit_pixel_owner(points,rigcpu,frame,dense=False);gpu=first_hit_pixel_owner(points.to(a.device),rigdev,frame,dense=False)
  mismatch=sum(int((x!=y.cpu()).sum()) if x.shape==y.shape else max(len(x),len(y)) for x,y in zip(selected_owners(cpu),selected_owners(gpu)))
  report['geometry'].append({'frame':k,'points':len(points),'selected_record_disagreements':mismatch,
    'projection_max_abs_pixels':float((cpu.uv-gpu.uv.cpu()).abs().max()),'depth_max_abs_m':float((cpu.z-gpu.z.cpu()).abs().max())})
 for path in sorted((root/'temporal/traced_points').glob('*.json')):
  trace=json.loads(path.read_text())
  for record in trace['frames']:
   for row in record['rows']:
    c=torch.tensor([row['camera']]);st=torch.tensor([record['s_plus']],dtype=torch.float64);sr=torch.tensor([row['reference_state']],dtype=torch.float64)
    pixel=row['current_pixel'];uv=torch.tensor([[pixel%data.cameras[0].width,pixel//data.cameras[0].width]],dtype=torch.float64)
    values=torch.tensor([row.get('reference_clean_rgbd',row['reference_raw_rgbd'])],dtype=torch.float64);cell=torch.tensor([np.floor(row['q'])],dtype=torch.float64)
    cpu=temporal_prediction(rigcpu,st,sr,c,uv,values,cell)
    gpu=temporal_prediction(rigdev,st.to(a.device),sr.to(a.device),c.to(a.device),uv.to(a.device),values.to(a.device),cell.to(a.device))
    report['warp_operator'].append({'original_point_id':trace['point_id'],'frame':record['frame'],'camera':row['camera'],
      'h_max_abs':float((cpu[0]-gpu[0].cpu()).abs().max()),'H_current_max_abs':float((cpu[1]-gpu[1].cpu()).abs().max()),
      'H_reference_max_abs':float((cpu[2]-gpu[2].cpu()).abs().max()),'support_status':row['linearization_status'],
      'semantics':'fixed saved 2x2 contributor cell; nonsmooth boundary decisions reported separately'})
 report['real_Schur_parity']=[json.loads(p.read_text()) for p in sorted((root/'temporal').glob('real_schur_parity_*.json'))]
 report['status']='NUMERICAL_WITNESSES_COMPUTED_NOT_SCIENTIFIC_ACCEPTANCE'
 json_write(a.output,serial(report));print(a.output)
if __name__=='__main__':main()
