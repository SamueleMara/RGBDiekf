# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""Independent block-sparse 12D group IEKF; fixed raw associations per update.

Prior-coordinate optimization: T=Exp(d_pose) Tminus, xi=ximinus+d_twist.
J_coordinate = J_left_current * Jl(d_pose). Final covariance is reset to the
accepted left tangent by diag(Jl(d_pose),I). Canonical points are conditioned.
"""
import numpy as np
import torch
from rgbd_common.backends.torch_cuda import stable_segment_sum
from .group_pose import transport,point_jacobian,exp,chart_reset,left_jacobian,marginal_position_covariance
from .projection import project
from .filter_iekf import inverse_spd
from .association import associate


def associate_groups(partition,poses,covariance,dataset,k,config,backend,audit_path=None):
    canonical=backend.tensor(partition.xyz);member=backend.tensor(partition.membership,torch.long)
    xyz=transport(canonical,member,poses)
    # Temporary derived position marginals adapt the existing point matcher.
    # These arrays are NOT per-point dynamic states or independent priors.
    derived=np.zeros((len(xyz),6));derived[:,:3]=xyz.cpu().numpy();p=np.zeros((len(xyz),6,6))
    p[:,:3,:3]=marginal_position_covariance(xyz,member,covariance).cpu().numpy()
    matches=[];stats=[]
    for ci in range(len(dataset.cameras)):
        frame=dataset.observation(k,ci)
        match,report=associate(derived,p,partition.rgb,np.arange(len(xyz)),frame,config,backend,np.full(frame.depth.size,-1,np.int64))
        if len(np.unique(match['pixel']))!=len(match['pixel']):raise AssertionError('Duplicate raw filter factor')
        if not frame.valid.ravel()[match['pixel']].all():raise AssertionError('Non-foreground filter factor')
        if audit_path is not None:
            from pathlib import Path
            from .group_evidence_audit import replay_candidates
            from .observations import samples
            reason,candidates=replay_candidates(derived[:,:3],p[:,:3,:3],partition.rgb,samples(frame,config),config)
            width=frame.depth.size
            selected=match['point'].cpu().numpy()*width+match['pixel']
            candidates['actual_filter_selected']=np.isin(candidates['point']*width+candidates['pixel'],selected)
            path=Path(audit_path);path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/f'{k:06d}_{ci:03d}.npz',**candidates,point_reason=reason,
                group_id=partition.membership[candidates['point']],source=partition.source[candidates['point']],
                gate3=np.array(config.gate(3)),gate2=np.array(config.gate(2)),
                provenance=np.array('PRE_UPDATE_PRIOR: NumPy candidate scores and actual Torch filter selections stored separately; no inferred physical labels'))
        matches.append(match);stats.append(report)
    return matches,stats


def correct(prior_poses,prior_twists,pminus,canonical,membership,matches,config,backend):
    groups=len(prior_poses);dimension=pminus.shape[-1]
    if dimension not in (6,12) or (prior_twists is None)!=(dimension==6):raise ValueError('Expected a genuine 6D pose-only or 12D pose/twist state')
    reset_chart=chart_reset if dimension==12 else left_jacobian
    precision=inverse_spd(pminus,'group predicted covariance');delta=prior_poses.new_zeros((groups,dimension))
    counts=torch.zeros(groups,device=backend.device,dtype=torch.long);prepared=[]
    used_raw=set()
    for match in matches:
        if 'pixel' not in match:raise ValueError('Every group factor requires immutable raw pixel identities')
        camera_id=match.get('camera_index',match['camera'].name)
        refs=[(camera_id,int(pixel)) for pixel in match['pixel']]
        if len(set(refs))!=len(refs) or used_raw.intersection(refs):raise ValueError('Duplicate raw observation factor in grouped IEKF')
        used_raw.update(refs)
        ids=match['point']
        if not len(ids):continue
        keys=membership[ids];counts+=torch.bincount(keys,minlength=groups)
        prepared.append((ids,keys,match['z'],inverse_spd(match['R'],'raw group measurement covariance'),match['camera']))
    chunk=backend.chunk_size(12*12*8*10)
    def terms(d,information=False):
        poses=exp(d[:,:6])@prior_poses;energy=.5*torch.einsum('ni,nij,nj->n',d,precision,d)
        a=precision.clone() if information else None;b=-(precision@d[...,None])[...,0] if information else None
        measurement_info=torch.zeros_like(precision) if information else None
        reset=reset_chart(d) if information else None
        for ids,keys,z,rinv,camera in prepared:
            for lo in range(0,len(ids),chunk):
                idx=ids[lo:lo+chunk];g=keys[lo:lo+chunk];target=z[lo:lo+chunk];weight=rinv[lo:lo+chunk]
                xyz=(poses[g,:3,:3]@canonical[idx,:,None])[...,0]+poses[g,:3,3]
                h,j=project(xyz,camera);error=target-h
                valid=torch.isfinite(h).all(-1)&(h[:,2]>0)
                local=.5*torch.einsum('ni,nij,nj->n',error,weight,error)
                energy+=stable_segment_sum(torch.where(valid,local,torch.inf),g,groups)
                if information:
                    hgroup=j[:,:,:3]@point_jacobian(xyz)
                    local_j=xyz.new_zeros((len(idx),3,dimension));local_j[:,:,:6]=hgroup
                    jac=local_j@reset[g];weighted=jac.transpose(1,2)@weight
                    reduced=stable_segment_sum(weighted@jac,g,groups)
                    a+=reduced;measurement_info+=stable_segment_sum(local_j.transpose(1,2)@weight@local_j,g,groups)
                    b+=stable_segment_sum((weighted@error[:,:,None])[...,0],g,groups)
        return energy,a,b,measurement_info
    initial=terms(delta)[0];history=[initial.cpu().numpy()];iterations=torch.zeros_like(counts);stops=torch.zeros_like(counts);done=counts==0
    for _ in range(config.iterations):
        energy,a,b,_=terms(delta,True);step=torch.linalg.solve(a,b[:,:,None])[...,0]
        done|=step.abs().amax(1)<=config.tolerance
        pending=~done
        if not bool(pending.any()):break
        iterations[pending]+=1;alpha=1.
        for _ in range(config.line_search_steps):
            trial=delta+alpha*step;value=terms(trial)[0]
            accepted=pending&torch.isfinite(value)&(value<energy)
            delta[accepted]=trial[accepted];pending&=~accepted
            if not bool(pending.any()):break
            alpha*=.5
        stops[pending]+=1;done|=pending;history.append(terms(delta)[0].cpu().numpy())
    final,a,_,measurement_info=terms(delta,True)
    coordinate_covariance=inverse_spd(a,'group final information');reset=reset_chart(delta)
    posterior=reset@coordinate_covariance@reset.transpose(1,2)
    poses=exp(delta[:,:6])@prior_poses;twists=prior_twists+delta[:,6:] if dimension==12 else None
    unused=counts==0;poses[unused]=prior_poses[unused];posterior[unused]=pminus[unused]
    if dimension==12:twists[unused]=prior_twists[unused]
    posterior=(posterior+posterior.transpose(-1,-2))/2
    inverse_spd(posterior,'group posterior covariance after chart reset')
    # Diagnostic spectra do not need cuSOLVER's multi-GB batched workspace.
    spectra=np.linalg.eigvalsh(measurement_info[:,:6,:6].cpu().numpy())
    tolerance=np.finfo(float).eps*6*np.maximum(spectra[:,-1],0)
    rank=(spectra>tolerance[:,None]).sum(1)
    return poses,twists,posterior,{'counts':counts.cpu().numpy(),'measurement_pose_information_eigenvalues':spectra,'pose_rank':rank,'energy_before':initial.cpu().numpy(),'energy_after':final.cpu().numpy(),'energy_history':np.asarray(history),'iterations':iterations.cpu().numpy(),'line_search_stops':stops.cpu().numpy(),'iteration_limit':((~done)&(counts>0)).cpu().numpy(),'psd_repairs':0,'prior_count':1,'chart':'spatial left, translation first; covariance reset from prior additive coordinates'}
