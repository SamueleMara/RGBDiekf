"""Real-map GPU operator parity/timing, not a reconstruction experiment."""
import argparse,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'tests')]
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.frame_cache import CachedDataset
from rgbd_common.geometry import PointMap
from rgbd_common.lie import State
from rgbd_common.matching_cuda import match_rgbd
from rgbd_common.measurement_problem import MeasurementProblem
from ukf.filter import GroupChart
from ukf.measurement import measurement_moments
from oracle.ukf_measurement_scalar import measurement_moments as scalar


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    b=Backend('cuda');config=load_config();meta=json.loads((a.run/'checkpoint.json').read_text())
    model=PointMap.load(a.run/meta['point_file']);data=CachedDataset(load_dataset(a.input),a.run/'raw_cache')
    with np.load(a.run/meta['trajectory_file']) as trace:
        k=len(data.times)//2;state=State(trace['poses'][k],trace['twists'][k])
    matches=[match_rgbd(model,state,data.observation(k,ci),config,b)[0] for ci in range(len(data.cameras))]
    problem=MeasurementProblem(model,matches,b).for_group(0);base=GroupChart(state,0)
    # Explicit diagnostic covariance, solely to compare identical operators.
    covariance=np.eye(12)*1e-7;results=[];timings={}
    for name,function in [('scalar',scalar),('batched',measurement_moments)]:
        b.synchronize();start=time.perf_counter();value=function(problem,base,covariance);b.synchronize()
        timings[name+'_seconds']=time.perf_counter()-start;results.append(value)
    differences={}
    for label,old,new in zip(('A','M','b'),results[0][:3],results[1][:3]):
        old=numpy(old) if hasattr(old,'detach') else old;new=numpy(new) if hasattr(new,'detach') else new
        differences[label]=float(np.max(abs(old-new)));assert_allclose(old,new,atol=1e-6,rtol=1e-10)
    for key in ('active_factors','deferred_visibility_factors'):assert results[0][3][key]==results[1][3][key]
    report={'purpose':'Exact GPU sigma operator equivalence on real accepted map, not reconstruction success','frame':k,'map_points':model.count,'groups':len(state.poses),'factor_count':sum(map(len,problem.matches)),**timings,'max_absolute_difference':differences,'sigma_factors':results[1][3],'gpu':b.memory(),'diagnostic_covariance':{'value':1e-7,'class':'explicit test fixture, not production or calibration'}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
