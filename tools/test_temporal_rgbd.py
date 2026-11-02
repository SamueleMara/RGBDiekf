#!/usr/bin/env python3
"""Reproducible tiny numerical/integration gate with machine-readable results."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io import json_write,serial


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.checks=[]
    def startTest(self,test):self.started=time.perf_counter();super().startTest(test)
    def record(self,test,status,detail=None):self.checks.append({'test':test.id(),'status':status,'seconds':time.perf_counter()-self.started,'detail':detail})
    def addSuccess(self,test):super().addSuccess(test);self.record(test,'PASS')
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,'SKIP',reason)
    def addFailure(self,test,err):super().addFailure(test,err);self.record(test,'FAIL',self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.record(test,'ERROR',self._exc_info_to_string(err,test))
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        self.record(subtest,'PASS' if err is None else 'FAIL',None if err is None else self._exc_info_to_string(err,test))


def witnesses():
    from tests.test_temporal_full_graph import shared_graph
    from tests.test_temporal_warp import rig
    from estimator.temporal_rgbd_terms import temporal_prediction
    g=shared_graph();g.solve_sparse_map();J,r,layout,blocks=g.linearize();A=J.to_dense();P=torch.linalg.inv(A.T@A)
    c=g.posterior[0];cov_error=float((c['cov']-P[c['cols'][:,None],c['cols']]).abs().max())
    gen=torch.Generator().manual_seed(37);x=torch.randn(J.shape[1],generator=gen,dtype=torch.float64);y=torch.randn(J.shape[0],generator=gen,dtype=torch.float64)
    adjoint=float(abs(g.apply_J(J,x)@y-x@g.apply_JT(J,y)))
    before=g.covariance({('state',1,0):g.tensor([[1.]])},{('state',1,0):g.tensor([[1.]])}).clone()
    g.marginalize_expired_exactly([('state',0,0)]);g.solve_sparse_map()
    after=g.covariance({('state',1,0):g.tensor([[1.]])},{('state',1,0):g.tensor([[1.]])})
    sr=g.tensor([[.03,.08,2.,0.,0.,0.]]);st=g.tensor([[.11,.14,2.1,0.,0.,0.]])
    values=g.tensor([[[.1,.2,.3,2.],[.2,.3,.4,2.01],[.15,.25,.35,2.02],[.3,.4,.5,2.03]]]);pixel=g.tensor([[3.,3.]]);cam=torch.tensor([0]);rr=rig()
    h,ht,hr,*_=temporal_prediction(rr,st,sr,cam,pixel,values);worst=0.
    for arr,H in ((st,ht),(sr,hr)):
        for col in range(6):
            a=arr.clone();b=arr.clone();a[0,col]+=1e-6;b[0,col]-=1e-6
            args1=(a,sr) if arr is st else (st,a);args2=(b,sr) if arr is st else (st,b)
            fd=(temporal_prediction(rr,*args1,cam,pixel,values)[0]-temporal_prediction(rr,*args2,cam,pixel,values)[0])/(2e-6)
            worst=max(worst,float((fd-H[:,:,col]).abs().max()))
    gpu_metrics=None
    if torch.cuda.is_available():
        gc=shared_graph();gg=shared_graph('cuda');t0=time.perf_counter();gg.solve_sparse_map();torch.cuda.synchronize();seconds=time.perf_counter()-t0
        gc.solve_sparse_map()
        gpu_metrics={'mean_max_abs':max(float((gc.values[k]-gg.values[k].cpu()).abs().max()) for k in gc.values),
                     'covariance_max_abs':float((gc.posterior[0]['cov']-gg.posterior[0]['cov'].cpu()).abs().max()),
                     'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                     'device_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    return {'CPU_dense_vs_sparse_component_covariance_max_abs':cov_error,'JVP_VJP_dot_product_abs':adjoint,
            'Schur_marginal_covariance_max_abs':float((before-after).abs().max()),'two_time_FD_max_abs':worst,
            'tolerances':{'covariance_abs':2e-13,'adjoint_abs':1e-12,'two_time_FD_abs':3e-9,'Schur_abs':1e-12},
            'CUDA_mean_covariance_discrepancy':gpu_metrics,
            'raw_sensor_factors_per_shared_key':1}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=Path('tests/temporal_test_results.json'));a=ap.parse_args()
    modules=['tests.test_temporal_warp','tests.test_temporal_full_graph','tests.test_temporal_rgbd_pipeline','tests.test_temporal_motion',
             'tests.test_temporal_sparse_gpu','tests.test_temporal_sparse_marginalization','tests.test_temporal_sparse_runtime_safety',
             'tests.test_temporal_selected_inverse_gpu','tests.test_temporal_surgical_completion']
    import train
    source_paths=sorted(set(train.STAGE_SOURCES['free']+train.STAGE_SOURCES['free_temporal_rgbd']+[m.replace('.','/')+'.py' for m in modules]+
        ['tools/test_temporal_rgbd.py','tools/temporal_batch_runtime.py','tools/run_temporal_rgbd_critical.py','tools/report_temporal_sparse_gpu.py']))
    before_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    started=time.perf_counter();result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dirty_status':subprocess.check_output(['git','status','--short'],text=True).splitlines(),'python':sys.version,'platform':platform.platform(),
            'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'seconds':time.perf_counter()-started,'checks':result.checks,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
            'all_executed_checks_pass':result.wasSuccessful(),'complete_spec_acceptance':False,
            'not_validated':([] if torch.cuda.is_available() else ['CUDA unavailable in this execution environment'])+[
                             'Unit tests alone do not establish real-scene motion accuracy; see reports/temporal_rgbd_next_validation.md for actual pilot evidence',
                             'Full-cloud full-mode production feasibility and complete 60-frame validation',
                             'Independent calibration of additional temporal warp-model discrepancy',
                             'All mandatory adversarial/camera-handover tests are not yet exhaustive'],
            'numerical_witnesses':witnesses() if result.wasSuccessful() else None}
    from tests.test_temporal_rgbd_pipeline import CPU_CUDA_PIPELINE_COMPARISONS
    report['CPU_CUDA_pipeline_comparisons']=CPU_CUDA_PIPELINE_COMPARISONS
    from tests.test_temporal_motion import MOTION_WITNESSES
    report['motion_witnesses']=MOTION_WITNESSES
    from tests.test_temporal_full_graph import IDEAL_ORACLE_WITNESSES
    report['ideal_warp_oracle_witnesses']=IDEAL_ORACLE_WITNESSES
    from tests.test_temporal_sparse_gpu import SPARSE_WITNESSES
    from tests.test_temporal_sparse_marginalization import MARGINAL_WITNESSES
    report['sparse_gpu_witnesses']=SPARSE_WITNESSES
    report['sparse_marginal_witnesses']=MARGINAL_WITNESSES
    report['source_sha256']=before_hashes
    report['sources_changed_during_tests']=[p for p,h in before_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    json_write(a.output,serial(report));return 0 if result.wasSuccessful() and not report['sources_changed_during_tests'] else 1

if __name__=='__main__':raise SystemExit(main())
