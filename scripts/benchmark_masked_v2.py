"""Actual-map CUDA kernel timing; no reconstruction or covariance calibration claim."""
import argparse,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'tests')]
    from rgbd_common.backends.torch_cuda import Backend,numpy
    from rgbd_common.dataloader import load_dataset
    from rgbd_common.config import load_config
    from rgbd_common.geometry import PointMap,ObservationIDs
    from rgbd_common.lie import State
    from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
    from rgbd_common.matching_cuda import match_rgbd
    from rgbd_common.measurement_problem import MeasurementProblem
    from iekf.filter import IEKF
    from iekf.measurement import reduce_information
    from ukf.filter import UKF,GroupChart
    from ukf.measurement import measurement_moments,woodbury
    if args.output.exists():raise FileExistsError('Preserve prior benchmark; use a fresh output file')
    backend=Backend('cuda');data=load_dataset(args.input);config=load_config();model=PointMap.load(args.map)
    groups=int(model.group.max())+1;initial=State(np.tile(np.eye(4),(groups,1,1)),np.tile(config.twist_mean,(groups,1)))
    frame=data.observation(1,0);predicted,pminus,_=IEKF(config,backend).predict(initial,config.p0(groups),data.times[1]-data.times[0])
    rows=[]
    def timed(name,function):
        backend.synchronize();start=time.perf_counter();value=function();backend.synchronize()
        rows.append({'operation':name,'seconds':time.perf_counter()-start,'gpu':backend.memory()});return value
    xyz=backend.tensor(model.xyz,backend.input_dtype);group=backend.tensor(model.group,torch.long);poses=backend.tensor(predicted.poses,backend.input_dtype)
    projection=timed('transport_project',lambda:transport_and_project(xyz,group,poses,frame.camera))
    timed('zbuffer_depth_id_sort',lambda:build_pixel_bins_and_zbuffer(projection,frame.camera,backend.tensor(model.point_id,torch.long)))
    matches=timed('foreground_matching',lambda:match_rgbd(model,predicted,frame,config,backend)[0])
    problem=MeasurementProblem(model,[matches],backend,ObservationIDs(data))
    timed('analytical_iekf_information_12_per_group',lambda:reduce_information(problem,predicted))
    state,ukf_pminus,_=UKF(config,backend).predict(initial,config.p0(groups),data.times[1]-data.times[0])
    moments=timed('ukf_full_map_sigma_support_and_moments_25',lambda:measurement_moments(problem.for_group(0),GroupChart(state,0),ukf_pminus[:12,:12]))
    from oracle.ukf_measurement_full_map import measurement_moments as full_map
    reference=timed('ukf_full_reprojection_reference',lambda:full_map(problem.for_group(0),GroupChart(state,0),ukf_pminus[:12,:12]))
    differences={}
    for name,old,new in zip(('A','M','b'),reference[:3],moments[:3]):
        old=numpy(old) if isinstance(old,torch.Tensor) else old;new=numpy(new) if isinstance(new,torch.Tensor) else new
        np.testing.assert_allclose(old,new,rtol=1e-12,atol=1e-10);differences[name]=float(np.max(np.abs(old-new)))
    assert reference[3]['active_factors']==moments[3]['active_factors']
    timed('ukf_woodbury_25_solve',lambda:woodbury(moments[0],ukf_pminus[:12,:12],numpy(moments[1]),numpy(moments[2])))
    report={'purpose':'Kernel benchmark on actual frame-0 map and real frame 1; not successful reconstruction','scene':str(data.root),'map_source':str(args.map),'map_points':model.count,'groups':groups,'frame':1,'camera':0,'raw_matched_factors':len(matches),'noise_source':str(config.source),'sigma_prior':'actual authorized P0 propagated with Q(dt), no artificial tiny diagnostic covariance','ukf_support':moments[3],'ukf_reference_max_absolute_difference':differences,'operations':rows,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu':backend.memory()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
