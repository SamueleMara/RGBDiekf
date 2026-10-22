"""Exact zero-column elimination: real-map CUDA operator reference/timing."""
import argparse,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'tests')]
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.frame_cache import CachedDataset
from rgbd_common.geometry import PointMap,ObservationIDs
from rgbd_common.lie import State
from rgbd_common.matching_cuda import match_rgbd
from rgbd_common.measurement_problem import MeasurementProblem
from iekf.measurement import reduce_information
from oracle.iekf_measurement_full import reduce_information as full


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    backend=Backend('cuda');config=load_config();meta=json.loads((a.run/'checkpoint.json').read_text())
    model=PointMap.load(a.run/meta['point_file']);data=CachedDataset(load_dataset(a.input),a.run/'raw_cache')
    with np.load(a.run/meta['trajectory_file']) as trace:
        k=len(data.times)//2;state=State(trace['poses'][k],trace['twists'][k])
    matches=[match_rgbd(model,state,data.observation(k,ci),config,backend)[0] for ci in range(len(data.cameras))]
    problem=MeasurementProblem(model,matches,backend,ObservationIDs(data));values=[];timings={}
    for name,function in [('full',full),('compact',reduce_information)]:
        function(problem,state);backend.synchronize();start=time.perf_counter()
        for _ in range(3):value=function(problem,state)
        backend.synchronize();timings[name+'_seconds']=(time.perf_counter()-start)/3;values.append(tuple(numpy(x) for x in value))
    differences={}
    for name,old,new in zip(('information','rhs'),values[0],values[1]):
        assert_allclose(old,new,rtol=1e-12,atol=1e-5)
        differences[name]={'maximum_absolute':float(np.max(abs(old-new))),'relative_norm':float(np.linalg.norm(old-new)/max(1.,np.linalg.norm(old)))}
    result={'purpose':'Real-map operator equivalence and concurrent CUDA timing, not full reconstruction','frame':k,'map_points':model.count,'groups':len(state.poses),'factors':sum(map(len,problem.matches)),**timings,'differences':differences,'gpu':backend.memory(),
            'math':'Exact elimination of six zero twist columns; returned information and RHS remain 12D per group, with full six-dimensional R including correlations.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
