"""Euclidean 6D fixed-prior iterated EKF; no Lie chart or smoother."""
import torch
from .projection import project


def inverse_spd(value,label):
    root,info=torch.linalg.cholesky_ex(value)
    if bool((info!=0).any()) or not bool(torch.isfinite(root).all()):
        raise ArithmeticError(f'{label}: non-SPD/nonfinite blocks; no hidden jitter or eigenvalue clipping')
    result=torch.cholesky_inverse(root)
    if not bool(torch.isfinite(result).all()):raise ArithmeticError(f'{label}: inverse overflow; no silent precision repair')
    return result


def correct(prior,pminus,matches,config):
    """matches: fixed camera/point-index/z/R arrays, unique raw IDs."""
    n=len(prior);precision=inverse_spd(pminus,'predicted covariance');current=prior.clone()
    prepared=[];counts=torch.zeros(n,dtype=torch.long,device=prior.device)
    for match in matches:
        ids=match['point'];z=match['z'];r=match['R']
        if not len(ids):continue
        if len(torch.unique(ids))!=len(ids):raise ValueError('At most one raw observation per point/camera')
        prepared.append((ids,z,inverse_spd(r,'measurement R'),match['camera'],r))
        counts[ids]+=1
    def terms(state,information=False):
        d=state-prior;energy=.5*torch.einsum('ni,nij,nj->n',d,precision,d)
        a=precision.clone() if information else None;b=-(precision@d[...,None])[...,0] if information else None
        for ids,z,rinv,camera,_ in prepared:
            h,j=project(state[ids],camera);error=z-h
            local=.5*torch.einsum('ni,nij,nj->n',error,rinv,error)
            valid=torch.isfinite(h).all(-1)&(h[:,2]>0)
            energy[ids]+=torch.where(valid,local,torch.inf)
            if information:
                weighted=j.transpose(1,2)@rinv
                a[ids]+=weighted@j;b[ids]+=(weighted@error[...,None])[...,0]
        return energy,a,b
    before,prior_information,prior_rhs=terms(current,True)
    linear_delta=torch.linalg.solve(prior_information,prior_rhs[...,None])[...,0]
    # Joint stacked multiview NIS via the 6D information identity. Cameras
    # share prediction error even when their raw sensor errors are independent.
    nis=torch.einsum('ni,nij,nj->n',linear_delta,precision,linear_delta)
    for ids,z,rinv,camera,_ in prepared:
        h,j=project(prior[ids],camera);error=z-h-(j@linear_delta[ids,:,None])[...,0]
        nis[ids]+=torch.einsum('ni,nij,nj->n',error,rinv,error)
    energies=[before.detach().clone()];converged=counts==0
    iterations=torch.zeros(n,dtype=torch.long,device=prior.device);rejections=torch.zeros_like(iterations)
    for _ in range(config.iterations):
        energy,a,b=terms(current,True)
        delta=torch.linalg.solve(a,b[...,None])[...,0]
        small=delta.abs().amax(-1)<=config.tolerance;converged|=small
        pending=~converged
        if not bool(pending.any()):break
        iterations[pending]+=1;accepted=torch.zeros_like(pending);alpha=1.
        for _ in range(config.line_search_steps):
            trial=current+alpha*delta;value,_,_=terms(trial)
            use=pending&torch.isfinite(value)&(value<energy)
            current[use]=trial[use];accepted|=use;pending&=~use
            if not bool(pending.any()):break
            alpha*=.5
        rejections[pending]+=1
        energies.append(terms(current)[0].detach().clone())
        converged|=pending
    after,a,_=terms(current,True);posterior=inverse_spd(a,'posterior information')
    # Unmatched tracks are predictions, bitwise, not re-inverted priors.
    posterior[counts==0]=pminus[counts==0]
    return current,(posterior+posterior.transpose(1,2))/2,{'before':before,'after':after,'iterations':iterations,'line_search_stops':rejections,'iteration_limit_reached':(~converged)&(counts>0),'measurement_count':counts,'nis':nis,'energies':energies,'psd_repairs':0,'covariance_semantics':'Fixed-association final-linearization conditional 6D Gaussian; shared calibration/correspondence uncertainty not marginalized'}
