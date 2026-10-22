"""Numerical/performance reference for exact repeated-covariance factor reuse."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rgbd_common.config import load_config
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.observation_model import covariance,gaussian_log_probability


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--points',type=int,default=160943);a=p.parse_args()
    b=Backend('cuda');rng=np.random.default_rng(13);r=covariance(load_config(),b.tensor(np.ones(a.points)),b)
    materialized=r.clone();error=b.tensor(rng.normal(size=(a.points,6))*.01)
    def reference():
        solved=torch.linalg.solve(materialized,error[...,None])[...,0]
        return -.5*((error*solved).sum(1)+torch.linalg.slogdet(materialized).logabsdet+6*np.log(2*np.pi))
    functions={'explicit_per_point_factorization':reference,'exact_shared_factorization':lambda:gaussian_log_probability(error,r)};report={}
    values={}
    for name,function in functions.items():
        function();b.synchronize();start=time.perf_counter()
        for _ in range(5):values[name]=function()
        b.synchronize();report[name+'_seconds']=(time.perf_counter()-start)/5
    report.update(points=a.points,max_absolute_log_density_difference=float(abs(values['explicit_per_point_factorization']-values['exact_shared_factorization']).max()),gpu=b.memory(),model='Same approved constant 6D covariance; no diagonal or independence approximation')
    if report['max_absolute_log_density_difference']>1e-10:raise ArithmeticError('Covariance factor reuse disagrees with independent dense small-block reference')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
