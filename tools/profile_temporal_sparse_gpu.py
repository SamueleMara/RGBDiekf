#!/usr/bin/env python3
"""Actual input, naturally nested point prefixes: GPU ASSEMBLY profiling only.

No GT is read, no trajectory accuracy is claimed and no inference output is
published. Full nonlinear/factorization/covariance timings come from gated runs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene.dataset import Dataset
from scene.config import load_config
from estimator.temporal_noise import temporal_sensor_only
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame,SensorNoise
from estimator.free_projective_run import build_catalogue
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_rgbd_terms import raw_values,sensor_covariance,build_temporal_rows
from estimator.temporal_sparse_gpu import TemporalSoAGraph
from estimator.temporal_sparse_solve import eliminate_current_pixel_constraints_gpu,connected_columns_gpu,SparseJointFactorization
from estimator.temporal_sparse_run import save_real_sparse_oracle,batched_eigen
from estimator.point_kalman_cuda import transition,process_noise
from utils.io import json_write,serial
from tools.temporal_batch_runtime import sparse_source_hashes


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():ap.error('Profile output exists; never overwrite experiments')
    a.output.mkdir(parents=True)
    dataset=Dataset(a.scene,frame_limit=3)
    config=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
    config,policy,calibration=temporal_sensor_only(dataset,config,policy)
    rig=CameraRig(dataset.cameras,'cuda');noise=SensorNoise.from_config(config,policy,'cuda',torch.float64)
    sub={'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':True,'report':False}
    k0,xyz,rgb,cov,source,sel,sm=build_catalogue(dataset,config,'forward',sub,policy,noise)
    frames={k:MeasuredFrame(dataset,k,'cuda',torch.float64) for k in [0,1]}
    dt=float(dataset.times[1]-dataset.times[0]);F=transition(dt,device='cuda');Q=process_noise(dt,torch.as_tensor(config.qc,device='cuda',dtype=torch.float64))
    warp=torch.zeros((4,4),device='cuda',dtype=torch.float64)
    report={'scope':'REAL_INPUT_NESTED_PREFIX_ASSEMBLY_BENCHMARK_ONLY; no filtering or reconstruction acceptance',
        'scene':str(a.scene),'full_point_count':len(xyz),'cameras':len(dataset.cameras),'calibration':calibration,'cases':[],
        'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'ground_truth_read':False,
        'production_source_sha256':sparse_source_hashes(Path(__file__).resolve().parents[1]),
        'profile_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for N in sorted(set([min(len(xyz),n) for n in [1000,10000,50000,len(xyz)]])):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        ss=torch.as_tensor(source[:N],device='cuda',dtype=torch.long)
        values=raw_values(frames[0],ss[:,1],ss[:,2]);R=sensor_covariance(values,noise)
        events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        wall=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            events[0].record()
            graph=TemporalSoAGraph(rig,3,N,ss[:,1],ss[:,2],torch.as_tensor(xyz[:N],device='cuda'),values,R,
                torch.as_tensor(config.pixel_covariance,device='cuda'),torch.as_tensor(config.velocity_covariance,device='cuda'),0)
            graph.layout();state=graph.source_state_and_blocks_gpu()[0];events[1].record()
            refs=ReferenceTable(rig.count,N,'cuda',torch.float64)
            owner=first_hit_pixel_owner(state[:,:3],rig,frames[0]);refs.commit_accepted_references(owner,frames[0],state,0)
            refs.initialize_from_source_catalogue(source[:N],0,state)
            now=first_hit_pixel_owner(state[:,:3],rig,frames[1],dense=False)
            rows,reason,life=build_temporal_rows(state,now,frames[1],refs,frames,rig,noise,policy,warp,1,3)
            graph.pack_source_process_warp_factors_gpu(0,1,F,Q,state,rows,frames,noise,warp)
            events[2].record();problem=graph.assemble_sparse_linearized_problem_gpu();events[3].record()
            events[3].synchronize()
        record={'points':N,'original_ID_range':[0,N-1],'source_setup_GPU_seconds':events[0].elapsed_time(events[1])/1000,
            'ownership_registration_GPU_seconds':events[1].elapsed_time(events[2])/1000,
            'residual_Jacobian_sparse_assembly_GPU_seconds':events[2].elapsed_time(events[3])/1000,'wall_seconds':time.perf_counter()-wall,
            'factor_counts':problem.factor_counts,'scalar_dimension':problem.scalar_dimension,'nnz_J':problem.A._nnz(),'nnz_C':problem.C._nnz(),
            'unique_raw_keys':len(graph.pixels.keys),'GPU_peak_torch_bytes':torch.cuda.max_memory_allocated(),
            'CPU_RSS_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'host_scalar_extractions':sum(item.count for item in profiler.key_averages() if item.key=='aten::item'),
            'solve_covariance':'NOT_RUN_FOR_THIS_SIZE','Schur':'NOT_RUN_IN_ASSEMBLY_PROFILE'}
        profiler.export_chrome_trace(str(a.output/f'assembly_{N}.json'))
        record['kernel_names']=sorted(set(event.name for event in profiler.events() if event.device_type==torch.autograd.DeviceType.CUDA))
        ts=time.perf_counter();AA,rr,CC,cc,T,offset=eliminate_current_pixel_constraints_gpu(problem)
        labels,iterations=connected_columns_gpu(AA,CC);_,counts=torch.unique(labels,return_counts=True)
        record.update(topology_seconds=time.perf_counter()-ts,true_components=len(counts),
            true_component_scalars_median_p95_max=torch.quantile(counts.double(),torch.tensor([.5,.95,1.],device='cuda',dtype=torch.float64)).tolist(),
            connectivity_iterations=iterations,reduced_dimension=AA.shape[1])
        del AA,rr,CC,cc,T,offset,labels,counts
        if N:
            try:
                factor=SparseJointFactorization(problem)
                try:
                    ts=time.perf_counter();step,certificate=factor.solve_constrained_sparse_joint_gpu();record['sparse_MAP_seconds']=time.perf_counter()-ts
                    P=factor.extract_joint_selected_marginals_gpu(graph.state_query_gpu(1))
                    # Diagnostic cross queries reuse the factor and replace its
                    # latest-query profile; retain the all-point query separately.
                    record['sparse_factorization_and_covariance_profile']=dict(factor.profile)
                    record['covariance_min_eigenvalue_all_points']=float(batched_eigen(P).min())
                    record['solve_covariance']='ACTUAL_FROZEN_LINEARIZATION_GPU; not nonlinear reconstruction'
                    if N==len(xyz):record['actual_complete_component_exports']=save_real_sparse_oracle(a.output,1,factor,graph)
                    del P,step
                finally:factor.close()
                del factor
            except Exception as error:
                record['sparse_numerical_failure']={'type':type(error).__name__,'reason':str(error),'profile':getattr(error,'profile',None)}
        report['cases'].append(record);json_write(a.output/'assembly_profiles.json',serial(report))
        print(json.dumps({k:v for k,v in record.items() if k!='kernel_names'}),flush=True)
        del graph,problem,rows,state,owner,now,refs,profiler
    return 0


if __name__=='__main__':raise SystemExit(main())
