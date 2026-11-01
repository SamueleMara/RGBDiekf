#!/usr/bin/env python3
"""POST-INFERENCE ONLY: critical-point motion errors and marginal NEES.
Not imported by train or calibration. Canonical GT labels come from the existing
post-hoc evaluator; outputs are never consumed by inference.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluate_projective_gt import gt_joint,labelled_endpoints,gt_transform
from utils.io import json_write,serial
from scene.dataset import Dataset
from utils.temporal_output import require_valid_frame

def evaluate(run,scene):
 cameras=Dataset(scene).cameras
 kind,axis,origin,declared=gt_joint(scene)
 values=np.genfromtxt(scene/'metadata/frame_values.csv',delimiter=',',names=True)
 allq=np.asarray(values[values.dtype.names[1]],float)
 report={'run':str(run),'scene':str(scene),'ground_truth_namespace':'POSTHOC_ONLY','inference_readback':False,'points':[],'cloud_summaries':{}}
 for direction in ('forward','backward'):
  root=run/f'free_{direction}'
  if not (root/'canonical.npz').exists():continue
  status=json.loads((root/'status.json').read_text())
  if status['status']!='COMPLETE':continue
  man=json.loads((root/'run_manifest.json').read_text());can=np.load(root/'canonical.npz');k0=int(can['canonical_frame'])
  frames=man['original_frame_indices'];q=allq[frames];times=np.asarray(man['timestamps_seconds'])
  clouds=labelled_endpoints(scene,kind,axis,origin,allq[-1]-allq[0]);cloud=(clouds[0][0].copy(),clouds[0][1])
  if k0!=0:
   T0=gt_transform(kind,axis,origin,q[k0]-allq[0]);mv=cloud[1]==clouds[2];cloud[0][mv]=cloud[0][mv]@T0[:3,:3].T+T0[:3,3]
  distance,nn=cKDTree(cloud[0]).query(can['canonical_xyz']);moving=cloud[1][nn]==clouds[2]
  dqdt=np.gradient(q,times)
  X=np.load(root/'posterior_xyz.npy',mmap_mode='r');acceptance=np.load(root/'update_acceptance.npy',mmap_mode='r');valid=np.load(root/'valid_cameras.npy',mmap_mode='r')
  summaries=[]
  for k in range(len(times)):
   require_valid_frame(root,k)
   T=gt_transform(kind,axis,origin,q[k]-q[k0]);truth=can['canonical_xyz'].copy();truth[moving]=truth[moving]@T[:3,:3].T+T[:3,3]
   error=np.linalg.norm(np.asarray(X[k])-truth,axis=1)
   record={'frame':k,'point_count':len(moving),'GT_moving_points':int(moving.sum()),'eligible_measurement_points':int((valid[k]>0).sum()),'accepted_update_points':int((acceptance[k]>0).sum())}
   for name,group in [('static',~moving),('moving',moving)]:
    record[name+'_error_median_m']=float(np.median(error[group])) if group.any() else None
    record[name+'_error_p95_m']=float(np.quantile(error[group],.95)) if group.any() else None
   summaries.append(record)
  report['cloud_summaries'][direction]={'frames':summaries,'nearest_GT_label_distance_quantiles_m':np.quantile(distance,[0,.5,.95,1]).tolist(),
    'qualification':'nearest supplied GT cloud labels, posthoc only; label distances are reported, no GT point selection in inference'}
  for tracefile in sorted((root/'temporal/traced_points').glob('*.json')):
   trace=json.loads(tracefile.read_text());original=trace['point_id'];local=int(trace.get('local_array_index',original));x0=can['canonical_xyz'][local]
   for record in trace['frames']:
    k=record['frame'];truth=x0.copy();velocity=np.zeros(3)
    if moving[local]:
     T=gt_transform(kind,axis,origin,q[k]-q[k0]);truth=x0@T[:3,:3].T+T[:3,3]
     velocity=axis*dqdt[k] if kind=='PRISMATIC' else np.cross(axis,truth-origin)*dqdt[k]
    state=np.asarray(record['s_plus']);P=np.asarray(record['P_plus']);e=state[:3]-truth
    true_pixel_motion=[]
    for row in record['rows']:
     c=cameras[row['camera']];rf=row['reference_frame'];reference_truth=x0.copy()
     if moving[local]:
      Tr=gt_transform(kind,axis,origin,q[rf]-q[k0]);reference_truth=x0@Tr[:3,:3].T+Tr[:3,3]
     a=truth@c.world_to_camera[:3,:3].T+c.world_to_camera[:3,3];b=reference_truth@c.world_to_camera[:3,:3].T+c.world_to_camera[:3,3]
     if a[2]>0 and b[2]>0:
      fx,fy,cx,cy=c.intrinsics;shift=np.array([fx,fy])*(a[:2]/a[2]-b[:2]/b[2]);true_pixel_motion.append(float(np.linalg.norm(shift)))
    try:np.linalg.cholesky(P[:3,:3]);nees=float(e@np.linalg.solve(P[:3,:3],e))
    except np.linalg.LinAlgError:nees=None
    report['points'].append({'direction':direction,'original_point_id':original,'local_array_index':local,'frame':k,
     'posthoc_GT_moving':bool(moving[local]),'GT_label_nearest_distance_m':float(distance[local]),
     'position_error_m':float(np.linalg.norm(e)),'velocity_error_m_s':float(np.linalg.norm(state[3:]-velocity)),
     'true_displacement_from_anchor_m':float(np.linalg.norm(truth-x0)),'estimated_displacement_from_anchor_m':float(np.linalg.norm(state[:3]-x0)),
     'true_current_reference_pixel_motion_max':max(true_pixel_motion,default=None),
     'true_position':truth.tolist(),'true_velocity':velocity.tolist(),'measured_rows':len(record['rows']),
     'measurement_update_accepted':record.get('measurement_update_accepted'),
     'position_NEES':nees,'inside_99pct_position_ellipsoid':None if nees is None else bool(nees<=chi2.isf(.01,3)),
     'NEES_qualification':'conditional uncalibrated marginal; temporal samples correlated, no pooled chi-square confidence claim',
     'velocity_truth_qualification':'np.gradient of supplied q on actual times; posthoc differentiation assumption'})
 return report

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():ap.error('output exists; never overwrite evaluations')
 json_write(a.output,serial(evaluate(a.run,a.scene)));print(a.output)
if __name__=='__main__':main()
