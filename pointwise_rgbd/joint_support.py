"""Classify existing endpoint points; never recover or insert new geometry."""
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from rgbd_common.array_store import ArrayStore
from .association import associate,exclusive_assignment
from .free_phase1 import ForegroundConstraint
from .joint_rgbd import predict,prediction_covariance,poses
from .observations import samples
from .group_evidence_audit import replay_candidates,REASONS
from .projection import project
from .store import json_write

LABELS=('AMBIGUOUS','STATIC_COMPATIBLE','MOVING_JOINT_COMPATIBLE')


def decide(nll_static,nll_moving,count,separable,static_bad,moving_bad):
    labels=np.zeros(len(count),np.uint8)
    usable=(count>0)&separable
    labels[usable&~static_bad&(moving_bad|(nll_static<nll_moving))]=1
    labels[usable&~moving_bad&(static_bad|(nll_moving<nll_static))]=2
    return labels


def run(partition,dataset,config,backend,selected,anchor,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);n=len(partition.xyz)
    nll=np.zeros((n,2));count=np.zeros(n,np.int64);separable=np.zeros(n,bool)
    bad=np.zeros((n,2),bool);observability=np.zeros((n,2),np.int64);frames_report=[]
    factors=ArrayStore(root/'classification_raw_factors.sqlite');member=backend.tensor(np.arange(n),torch.long);joint_poses=poses(selected['model'])
    for k in tqdm(range(len(dataset.times)),desc=f'Joint support completion (endpoint {anchor})',unit='frame'):
        if k==anchor:continue
        static=partition.xyz;moving=predict(selected['model'],static,k,np.full(n,anchor))
        states=[np.c_[x,np.zeros_like(x)] for x in (static,moving)]
        p0=np.zeros((n,6,6));p0[:,:3,:3]=partition.source_covariance;p0[:,3:,3:]=config.velocity_covariance
        rotation=joint_poses[k,:3,:3]@joint_poses[anchor,:3,:3].T;p1=p0.copy();p1[:,:3,:3]=rotation@partition.source_covariance@rotation.T;p=[p0,p1]
        frames=[dataset.observation(k,ci) for ci in range(len(dataset.cameras))]
        # Feasibility and occlusion use the same hard-mask policy as Phase1.
        for h in (0,1):
            check=ForegroundConstraint(frames,backend.tensor(p[h]),member,n,config,backend)
            bad[:,h]|=check.violations(backend.tensor(states[h])).cpu().numpy()
        row_count=0;conflicts=0
        for ci,original in enumerate(frames):
            frame=original;raw=samples(frame,config);matches=[]
            for h in (0,1):
                _,m=replay_candidates(states[h][:,:3],p[h][:,:3,:3],partition.rgb,raw,config,exclusive=False)
                use=np.isfinite(m['cost'])&(m['reason']!=REASONS.index('NUMERIC_TIE'))
                matches.append({key:m[key][use] for key in ('point','pixel')})
            point=np.concatenate([m['point'] for m in matches]);pixel=np.concatenate([m['pixel'] for m in matches])
            if not len(point):continue
            pairs=np.unique(np.c_[point,pixel],axis=0);point=pairs[:,0];pixel=pairs[:,1]
            z=np.c_[pixel%frame.camera.width,pixel//frame.camera.width,frame.depth.ravel()[pixel]];R=config.measurement_covariance(z[:,2]);ri=np.linalg.inv(R)
            predictions=[project(backend.tensor(x[point]),frame.camera)[0].cpu().numpy() for x in (static,moving)]
            costs=np.stack([.5*np.einsum('ni,nij,nj->n',z-h,ri,z-h) for h in predictions],1)
            rgb=frame.rgb.reshape(-1,3)[pixel]-partition.rgb[point]
            affinity=np.einsum('ni,ij,nj->n',rgb,np.linalg.inv(2*config.color_covariance),rgb)/2
            # Classification is geometric support, not a Kalman assimilation.
            # Endpoint samples may share one raw surface return. Their labels
            # then share evidence; no joint independence is asserted.
            order=np.lexsort((pixel,costs.min(1)+affinity,point))
            _,first=np.unique(point[order],return_index=True);chosen=order[first]
            conflicts+=len(point)-len(chosen);point=point[chosen];pixel=pixel[chosen];z=z[chosen];R=R[chosen];costs=costs[chosen]
            if not original.valid.ravel()[pixel].all():raise AssertionError('Background contributed to support classification')
            # Both hypotheses are scored on the SAME sensor return. A model
            # that predicts an occlusion is not assigned an artificial residual.
            valid=[];jacobians=[]
            for h in (0,1):
                hp,j=project(backend.tensor(states[h][point]),frame.camera);hp=hp.cpu().numpy();j=j.cpu().numpy();jacobians.append(j[:,:,:3]);uv=np.floor(np.nan_to_num(hp[:,:2])+.5).astype(int)
                inside=np.isfinite(hp).all(1)&(hp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<frame.camera.width)&(uv[:,1]>=0)&(uv[:,1]<frame.camera.height)
                pp=np.clip(uv[:,1],0,frame.camera.height-1)*frame.camera.width+np.clip(uv[:,0],0,frame.camera.width-1);depth=original.depth.ravel()[pp]
                var=np.einsum('ni,nij,nj->n',j[:,2],p[h][point],j[:,2])+config.measurement_covariance(depth)[:,2,2]
                hidden=original.valid.ravel()[pp]&(depth<hp[:,2])&((depth-hp[:,2])**2>config.gate(1)*var)
                visible=inside&~hidden;valid.append(visible);observability[point,h]+=visible
            common=valid[0]&valid[1];use=point[common]
            nll[use]+=costs[common];count[use]+=1
            row={'p':static[point],'k':k,'anchor':np.full(len(point),anchor),'camera':frame.camera}
            hm,model_cov=prediction_covariance(selected,row)
            hs=predictions[0][chosen];delta=hm-hs
            source_jacobian=jacobians[1]@rotation-jacobians[0]
            source_cov=source_jacobian@partition.source_covariance[point]@source_jacobian.transpose(0,2,1)
            sigma=R+2*(model_cov+source_cov)
            d2=np.einsum('ni,ni->n',delta,np.linalg.solve(sigma,delta[:,:,None])[:,:,0])
            separable[point[common&(d2>config.gate(3))]]=True
            factors.write(f'{k}:{ci}',point_id=point,pixel=pixel,z=z,R=R,common_visible=common,nll_static=costs[:,0],nll_moving=costs[:,1],model_separation_d2=d2)
            row_count+=len(point)
        frames_report.append({'frame':k,'raw_pairs':row_count,'candidate_conflicts':conflicts,'processed_camera_indices':list(range(len(frames)))})
    labels=decide(nll[:,0],nll[:,1],count,separable,bad[:,0],bad[:,1]);factors.close()
    np.savez_compressed(root/'endpoint_support.npz',xyz=partition.xyz,rgb=partition.rgb,source=partition.source,label=labels,nll=nll,common_visible_count=count,hypothesis_observable_count=observability,model_separable=separable,mask_infeasible=bad)
    report={'points':n,'labels':{name:int((labels==i).sum()) for i,name in enumerate(LABELS)},'frames':frames_report,'new_points_inserted':0,'camera_policy':'all_available','production_camera_indices':list(range(len(dataset.cameras))),
        'evidence':'conditional plug-in Gaussian sensor likelihood on all available cameras; fixed fitted joint and canonical endpoints. This is not a marginalized posterior. Different endpoint samples may share returns, so their labels are NOT independent. Within one point each raw return is counted once. Shared fitted-joint covariance is retained in the separability check, not counted independently in the likelihood sum.',
        'policy':'Ambiguous unless both predictions have common visible evidence and a prediction difference outside the existing 1% compatibility set; mask-infeasible hypotheses cannot win. Unknown/occluded samples add no artificial residual. No new numeric noise or geometric threshold.'}
    json_write(root/'result.json',report);return {**report,'label':labels}
