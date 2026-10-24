#!/usr/bin/env python3
"""Fit both group models to identical training raw IDs; score identical holdouts."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from tqdm.auto import tqdm
from pointwise_rgbd.group_evidence_audit import read_factors,load_cached,quantiles
from pointwise_rgbd.group_predictive import heldout_uvz_score,matches_from_factors
from pointwise_rgbd.group_iekf import correct
from pointwise_rgbd.group_pose import transport
from pointwise_rgbd.group_process import load_group_noise,initialize,predict
from pointwise_rgbd.group_pose_only import load_pose_noise,initialize as initialize6,predict as predict6
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.array_store import ArrayStore


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    source=args.run.resolve();out=args.output.resolve()
    if source==out or source in out.parents or (out.exists() and any(out.iterdir())):parser.error('Use a fresh output outside the source run')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic();b=Backend(args.device);torch.set_num_threads(1);config=load_config()
    model=json.loads((source/'manifest.json').read_text())['dataset'];timing=model['timing'];kwargs={'fps':timing['fps']} if timing['source_kind']=='explicit-user-override' else {}
    data=Dataset(model['input'],len(model['original_frame_ids']),len(model['cameras']),**kwargs)
    if not np.array_equal(data.times,model['timestamps_seconds']):raise ValueError('Timing changed')
    p,_=load_cached(source/'phase1/partition_v000');canonical=b.tensor(p.xyz);membership=b.tensor(p.membership,torch.long)
    results={};refs=ArrayStore(out/'independent_folds.sqlite')
    for name,noise,init,step in [('pose_twist_12d',load_group_noise(),initialize,predict),('pose_only_6d',load_pose_noise(),initialize6,predict6)]:
        poses,twist,cov=init(p.count,noise,b);frames=[];directory=out/name;directory.mkdir();t0=time.monotonic()
        for k in tqdm(range(1,len(data.times)),desc='Same raw holdout '+name,unit='frame'):
            factors=read_factors(source/'phase1/partition_v000',k)
            train=matches_from_factors(factors,p,data.cameras,b,False);test=matches_from_factors(factors,p,data.cameras,b,True)
            train_ids={(ci,int(pixel)) for ci,m in enumerate(train) for pixel in m['pixel']};test_ids={(ci,int(pixel)) for ci,m in enumerate(test) for pixel in m['pixel']}
            if train_ids&test_ids:raise AssertionError('Held-out raw IDs appear in training')
            poses,twist,cov=step(poses,twist,cov,float(data.times[k]-data.times[k-1]),noise)
            poses,twist,cov,diagnostics=correct(poses,twist,cov,canonical,membership,train,config,b)
            score=heldout_uvz_score(poses,cov,canonical,membership,test,b);e=score['raw_residual_uvz']
            np.savez_compressed(directory/f'{k:06d}.npz',poses=poses.cpu().numpy(),covariance=cov.cpu().numpy(),**{key:v for key,v in score.items() if isinstance(v,np.ndarray)})
            frames.append({'original_frame':data.frame_ids[k],'time_seconds':float(data.times[k]),'training_factors':len(train_ids),'heldout_factors':len(test_ids),'marginal_predictive_nll_sum':float(score['nll_per_group'].sum()),'heldout_depth_error_m':quantiles(abs(e[:,2])),'heldout_pixel_error_px':quantiles(np.linalg.norm(e[:,:2],axis=1)),'prediction_only_groups':int((diagnostics['counts']==0).sum()),'score_semantics':score['score_semantics']})
            if name=='pose_twist_12d':
                refs.write(str(k),train_refs=np.asarray(sorted(train_ids),np.int64),test_refs=np.asarray(sorted(test_ids),np.int64),canonical_provenance_unit=p.duplicate_unit)
        results[name]={'frames':frames,'noise':noise.provenance,'runtime_seconds':time.monotonic()-t0}
    refs.close();json_write(out/'comparison.json',{'models':results,'same_evaluation_factor_ids':True,'fit_evaluation_raw_overlap':0,'conditional_correspondence_caveat':'Archived associations were selected by the earlier run. These new fits and group proposals never use held-out UVZ; this is conditional prediction, NOT unbiased correspondence recovery or a joint temporal likelihood. Cross-camera canonical duplicates have linked folds. Calibration/canonical XYZ conditioned.','source_run':str(source),'original_frame_ids':data.frame_ids,'cameras':[c.name for c in data.cameras],'runtime_seconds':time.monotonic()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu':b.memory()})
    print(json.dumps({name:sum(f['marginal_predictive_nll_sum'] for f in value['frames']) for name,value in results.items()}))


if __name__=='__main__':main()
