"""Validate reference hierarchy proposals on independent raw RGB-D cameras.

This is candidate validation, not a new segmentation pattern. Point states
are frozen. No normals, physical patch splitting, joint or displacement gate.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import chi2
from tqdm.auto import tqdm
from rgbd_common.array_store import ArrayStore
from rgbd_common.backends.torch_cuda import stable_segment_sum
from .group_pose import exp,point_jacobian
from .group_partition_validation import weighted_kabsch
from .projection import project
from .filter_iekf import inverse_spd
from .store import json_write
from .group_visualization import serial


def load_rows(phase,role,K):
    store=ArrayStore(Path(phase)/f'factors_role{role}.sqlite');rows=[]
    for k in range(K):
        if store.contains(str(k)):rows.append((k,store.read(str(k))))
    store.close();return rows


def prepare(rows,mask,cameras,backend):
    data=[]
    for ci,camera in enumerate(cameras):
        pieces=[]
        for k,row in rows:
            use=(row['camera']==ci)&mask[row['point_id']]
            if use.any():pieces.append((k,row,use))
        if not pieces:continue
        point=np.concatenate([row['point_id'][use] for _,row,use in pieces]);time=np.concatenate([np.full(int(use.sum()),k) for k,row,use in pieces])
        z=np.concatenate([row['z'][use] for _,row,use in pieces]);R=np.concatenate([row['R'][use] for _,row,use in pieces])
        data.append((backend.tensor(point,torch.long),backend.tensor(time,torch.long),backend.tensor(z),inverse_spd(backend.tensor(R),'raw coherence R'),camera,float(np.linalg.slogdet(R)[1].sum())))
    return data


def terms(poses,canonical,data,K,info=False):
    value=poses.new_zeros(K);A=poses.new_zeros((K,6,6));b=poses.new_zeros((K,6));count=poses.new_zeros(K)
    for ids,k,z,ri,camera,_ in data:
        x=(poses[k,:3,:3]@canonical[ids,:,None])[...,0]+poses[k,:3,3];h,j=project(x,camera);e=z-h
        energy=.5*torch.einsum('ni,nij,nj->n',e,ri,e)
        value+=stable_segment_sum(energy,k,K);count+=torch.bincount(k,minlength=K)
        if info:
            H=j[:,:,:3]@point_jacobian(x);W=H.transpose(1,2)@ri
            A+=stable_segment_sum(W@H,k,K);b+=stable_segment_sum((W@e[:,:,None])[...,0],k,K)
    return value,A,b,count


def fit_raw(partition,arrays,mask,rows,dataset,config,backend,anchor,mask_index):
    K=len(dataset.times);ids=np.flatnonzero(mask)
    initial=np.stack([np.eye(4) if k==anchor else weighted_kabsch(partition.xyz[ids],np.asarray(arrays['xyz'][k])[ids],1/np.trace(np.asarray(arrays['covariance'][k])[ids,:3,:3],axis1=1,axis2=2)) for k in range(K)])
    pose=backend.tensor(initial);canonical=backend.tensor(partition.xyz);data=prepare(rows,mask,dataset.cameras,backend)
    for _ in range(config.iterations):
        energy,A,b,count=terms(pose,canonical,data,K,True);spectrum=torch.linalg.eigvalsh(A)
        valid=(count>=3)&(spectrum[:,0]>spectrum[:,-1]*6*torch.finfo(A.dtype).eps);valid[anchor]=False
        step=torch.zeros_like(b);step[valid]=torch.linalg.solve(A[valid],b[valid,:,None])[...,0]
        pending=valid&(step.abs().amax(1)>config.tolerance)
        if not bool(pending.any()):break
        alpha=1.
        for _ in range(config.line_search_steps):
            trial=exp(alpha*step)@pose;trial_energy=terms(trial,canonical,data,K)[0]
            use=pending&torch.isfinite(trial_energy)&(trial_energy<energy)
            for k in torch.nonzero(use).flatten().tolist():
                T=trial[k].cpu().numpy();x=partition.xyz[ids]@T[:3,:3].T+T[:3,3]
                if not mask_index.feasible(k,x,np.asarray(arrays['covariance'][k])[ids,:3,:3]):use[k]=False
            pose[use]=trial[use];pending&=~use
            if not bool(pending.any()):break
            alpha*=.5
    energy,A,b,count=terms(pose,canonical,data,K,True);spectrum=torch.linalg.eigvalsh(A)
    valid=(count>=3)&(spectrum[:,0]>spectrum[:,-1]*6*torch.finfo(A.dtype).eps);valid[anchor]=False
    return pose,A,valid


def evaluate(pose,precision,valid,canonical,data,K):
    # Training poses use disjoint raw cameras. Pose uncertainty is shared
    # by all held-out points/cameras at a frame, never copied as fresh noise.
    energy,A,b,count=terms(pose,canonical,data,K,True)
    use=valid&(count>0);total=precision[use]+A[use]
    q=2*energy[use]-torch.einsum('ni,nij,nj->n',b[use],inverse_spd(total,'raw coherence predictive information'),b[use])
    logdet=sum(d[-1] for d in data)+float((torch.linalg.slogdet(total).logabsdet-torch.linalg.slogdet(precision[use]).logabsdet).sum())
    scalars=3*int(count[use].sum());chi=float(q.sum())
    identity=torch.eye(4,device=pose.device,dtype=pose.dtype).repeat(K,1,1)
    static=terms(identity,canonical,data,K)[0][use]
    # logdet(R) cancels for the gain. Scores over omitted unobservable times
    # are NOT fabricated; a candidate must cover the full measured window.
    gain=float((static-energy[use]).sum())
    nll=.5*(chi+logdet+scalars*np.log(2*np.pi))
    return {'chi2':chi,'scalar_count':scalars,'p_value':float(chi2.sf(chi,scalars)) if scalars else 0.,'nll':float(nll),
        'identity_energy_gain_per_scalar':gain/max(scalars,1),'observable_frames':int(use.sum()),'raw_count':scalars//3}


def run(partition,arrays,coherence,dataset,config,backend,phase,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);K=len(dataset.times);anchor=coherence['anchor_frame']
    rows=[load_rows(phase,r,K) for r in (0,1)];member=coherence['representative_membership']
    qualified=set(coherence.get('selection',{}).get('qualified_ids',[]))
    candidates=[c for c in coherence.get('candidates',[]) if c['candidate_id'] in qualified]
    candidates.sort(key=lambda c:-c['rigidity_world']['delta_rss_cv_per_node_frame'])
    canonical=backend.tensor(partition.xyz);reports=[];chosen=None
    from .foreground_index import ForegroundIndex
    mask_index=ForegroundIndex(dataset,config)
    for c in tqdm(candidates,desc='Independent RGB-D coherence validation',unit='candidate'):
        nodes=coherence['candidate_members'][c['candidate_id']];mask=np.isin(member,nodes)
        pose,precision,valid=fit_raw(partition,arrays,mask,rows[0],dataset,config,backend,anchor,mask_index)
        test=prepare(rows[1],mask,dataset.cameras,backend);score=evaluate(pose,precision,valid,canonical,test,K)
        full=score['observable_frames']==K-1
        accepted=full and score['p_value']>=config.association_tail_probability and score['identity_energy_gain_per_scalar']>0
        report={'candidate_id':c['candidate_id'],'nodes':len(nodes),'points':int(mask.sum()),'status':'RAW_SUPPORTED_CORE' if accepted else 'RAW_UNRESOLVED_OR_INCONSISTENT',**score};reports.append(report)
        json_write(root/f'candidate_{c["candidate_id"]:04d}.json',serial(report))
        if accepted:chosen=(mask,nodes,c['candidate_id']);break
    json_write(root/'result.json',{'status':'RAW_SUPPORTED_CORE' if chosen else 'NO_RAW_VALIDATED_MOVING_CORE','candidates':reports,'criterion':'Reference-qualified candidates in original world held-out gain order; disjoint-camera raw UVZ consistency and nonidentity gain. Full time-window observability required for complete rigid trajectory; no new geometry/noise threshold. Canonical geometry, calibration and selected associations conditioned; 1% compatibility policy uncalibrated.'})
    result=coherence.copy()
    if chosen:
        result.update(point_support=chosen[0],moving_nodes=torch.tensor(chosen[1]),raw_validated_candidate=chosen[2],status='RESOLVED')
    else:result.update(status='NO_RAW_VALIDATED_MOVING_CORE',point_support=np.zeros(len(partition.xyz),bool))
    return result
