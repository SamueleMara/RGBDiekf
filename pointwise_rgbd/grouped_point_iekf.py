"""Grouped *increments* of free Euclidean point states, never group poses.

The update gain is restricted to range(G). Its Joseph covariance is NOT the
inverse reduced Hessian expanded into points. Cross blocks are represented by
[G,V] [[C,-M],[-M,0]] [G,V]^T, plus blockdiag(Pminus). Production prediction retains the complete recursive covariance operator.
The single-update helper below is also retained as a dense-reference test target.
"""
import numpy as np
import torch
from rgbd_common.backends.torch_cuda import stable_segment_sum
from .projection import project
from .filter_iekf import inverse_spd


def basis(xyz, membership, centers,normalize=False):
    r=xyz-centers[membership]
    out=xyz.new_zeros((len(xyz),6,12));eye=torch.eye(3,device=xyz.device,dtype=xyz.dtype)
    cross=xyz.new_zeros((len(xyz),3,3));x,y,z=r.unbind(1)
    cross[:,0,1]=-z;cross[:,0,2]=y;cross[:,1,0]=z;cross[:,1,2]=-x;cross[:,2,0]=-y;cross[:,2,1]=x
    out[:,:3,:3]=eye;out[:,:3,3:6]=-cross
    out[:,3:,6:9]=eye;out[:,3:,9:12]=-cross
    if normalize:
        gp=out[:,:3,:6];groups=len(centers)
        gram=stable_segment_sum(gp.transpose(1,2)@gp,membership,groups)
        eigen,vectors=torch.linalg.eigh(gram);keep=eigen>eigen[:,-1:]*torch.finfo(xyz.dtype).eps*6
        scale=torch.where(keep,1/torch.sqrt(torch.where(keep,eigen,torch.ones_like(eigen))),0.)
        transform=vectors*scale[:,None,:];gp=gp@transform[membership]
        out.zero_();out[:,:3,:6]=gp;out[:,3:,6:]=gp
    return out


def reduced_inverse(a, gram):
    """Inverse on the structural range of G, including singleton/line groups."""
    vals,vec=torch.linalg.eigh(gram)
    keep=vals>vals[:,-1:]*torch.finfo(a.dtype).eps*12
    result=torch.zeros_like(a);ranks=keep.sum(1)
    for rank in torch.unique(ranks).tolist():
        if not rank:continue
        ids=torch.nonzero(ranks==rank).flatten()
        u=vec[ids,:,-rank:]/torch.sqrt(vals[ids,-rank:])[:,None,:]
        little=u.transpose(1,2)@a[ids]@u
        result[ids]=u@inverse_spd(little,'grouped point reduced information')@u.transpose(1,2)
    return result,ranks


def joseph_blocks(p, measurement_info, g, membership, m, groups):
    v=p@measurement_info@g
    root=torch.linalg.cholesky(p)
    measurement_root=torch.zeros((len(p),6,3),device=p.device,dtype=p.dtype)
    observed=measurement_info[:,:3,:3].abs().sum((1,2))>0
    measurement_root[observed,:3]=torch.linalg.cholesky(measurement_info[observed,:3,:3])
    fp=g.transpose(1,2)@measurement_info@root
    fr=g.transpose(1,2)@measurement_root
    mp=m[membership]@fp;mr=m[membership]@fr
    c=stable_segment_sum(mp@mp.transpose(1,2)+mr@mr.transpose(1,2),membership,groups)
    gm=g@m[membership]
    local=torch.eye(6,device=p.device,dtype=p.dtype)-gm@g.transpose(1,2)@measurement_info
    lr=local@root;posterior=lr@lr.transpose(1,2)
    # Square-root Joseph contributions are multiplied BEFORE summing. A sum
    # of 12D outer products followed by ill-conditioned congruence lost PSD on
    # actual data. Bounded group-local gain factors avoid both that problem
    # and any global point-pair matrix. At most 32 MiB per temporary factor.
    keys=membership.detach().cpu().numpy();order=np.argsort(keys,kind='stable')
    bounds=np.r_[0,np.cumsum(np.bincount(keys,minlength=groups))]
    for lo in range(0,groups,8):
        hi=min(groups,lo+8);lengths=np.diff(bounds[lo:hi+1]);width=int(lengths.max())
        index=np.zeros((hi-lo,width),np.int64);valid=np.arange(width)[None,:]<lengths[:,None]
        for row,group in enumerate(range(lo,hi)):index[row,:lengths[row]]=order[bounds[group]:bounds[group+1]]
        ix=torch.as_tensor(index,device=p.device);use=torch.as_tensor(valid,device=p.device)
        row_chunk=max(1,min(width,32*1024**2//((hi-lo)*width*36*p.element_size())))
        for begin in range(0,width,row_chunk):
            end=min(width,begin+row_chunk);ii=ix[:,begin:end];mask=use[:,begin:end]
            left=gm[ii][:,:,None]
            exclude=torch.arange(begin,end,device=p.device)[:,None]!=torch.arange(width,device=p.device)[None,:]
            qp=left@fp[ix][:,None];qr=left@fr[ix][:,None]
            qp=torch.where((use[:,None,:,None,None]&exclude[None,:,:,None,None]),qp,0.)
            qr=torch.where(use[:,None,:,None,None],qr,0.)
            addition=(qp@qp.transpose(-1,-2)).sum(2)+(qr@qr.transpose(-1,-2)).sum(2)
            posterior[ii[mask]]+=addition[mask]
    posterior=(posterior+posterior.transpose(1,2))/2
    return posterior,{'V':v,'M':m,'C':c}


def correct(prior,pminus,matches,g,membership,config,feasible=None,covariance_operator=None):
    groups=int(membership.max())+1
    precision=inverse_spd(pminus,'point predicted covariance') if covariance_operator is None else None
    group_prior=covariance_operator.information(pminus) if covariance_operator is not None else None
    gram=stable_segment_sum(g.transpose(1,2)@g,membership,groups)
    prepared=[];counts=torch.zeros(len(prior),dtype=torch.long,device=prior.device);refs=set()
    for match in matches:
        ids=match['point']
        if not len(ids):continue
        if len(torch.unique(ids))!=len(ids):raise ValueError('Duplicate point measurement in one camera')
        identities={(match['camera_index'],int(p)) for p in match['pixel']}
        if len(identities)!=len(ids) or refs&identities:raise ValueError('Duplicate assimilation of a raw pixel')
        refs|=identities;counts[ids]+=1
        prepared.append((ids,match['z'],inverse_spd(match['R'],'raw R'),match['camera']))
    d=prior.new_zeros((groups,12))
    def terms(delta,info=False):
        displacement=(g@delta[membership,:,None])[...,0];state=prior+displacement
        energies=.5*torch.einsum('ni,nij,nj->n',displacement,precision,displacement) if precision is not None else prior.new_zeros(len(prior))
        a=(precision.clone() if precision is not None else torch.zeros_like(pminus)) if info else None
        b=(-(precision@displacement[:,:,None])[...,0] if precision is not None else torch.zeros_like(prior)) if info else None
        measurement=torch.zeros_like(pminus) if info else None
        for ids,z,ri,camera in prepared:
            h,j=project(state[ids],camera);error=z-h
            value=.5*torch.einsum('ni,nij,nj->n',error,ri,error)
            valid=torch.isfinite(h).all(1)&(h[:,2]>0)
            energies[ids]+=torch.where(valid,value,torch.inf)
            if info:
                hj=j.transpose(1,2)@ri;local=hj@j
                a[ids]+=local;measurement[ids]+=local;b[ids]+=(hj@error[:,:,None])[...,0]
        energy=stable_segment_sum(energies,membership,groups)
        if group_prior is not None:energy+=.5*torch.einsum('si,sij,sj->s',delta,group_prior,delta)
        if not info:return energy
        reduced=stable_segment_sum(g.transpose(1,2)@a@g,membership,groups)
        rhs=stable_segment_sum((g.transpose(1,2)@b[:,:,None])[...,0],membership,groups)
        if group_prior is not None:
            reduced+=group_prior;rhs-=(group_prior@delta[:,:,None])[...,0]
        return energy,reduced,rhs,measurement
    energy,a,b,_=terms(d,True);history=[energy.detach().cpu().numpy()]
    group_counts=stable_segment_sum(counts.to(prior.dtype),membership,groups)
    done=group_counts==0;iterations=torch.zeros(groups,dtype=torch.long,device=prior.device)
    rejected=torch.zeros_like(iterations);mask_rejected=0
    for _ in range(config.iterations):
        energy,a,b,_=terms(d,True);m,ranks=reduced_inverse(a,gram)
        step=(m@b[:,:,None])[...,0]
        ds=(g@step[membership,:,None])[...,0]
        largest=torch.zeros(groups,device=prior.device,dtype=prior.dtype)
        largest.scatter_reduce_(0,membership,ds.abs().amax(1),reduce='amax',include_self=True)
        done|=largest<=config.tolerance;pending=~done
        if not bool(pending.any()):break
        iterations[pending]+=1;alpha=1.
        for _ in range(config.line_search_steps):
            trial=d+alpha*step;value=terms(trial)
            use=pending&torch.isfinite(value)&(value<energy)
            if feasible is not None and bool(use.any()):
                state=prior+(g@trial[membership,:,None])[...,0]
                valid=feasible(state)
                mask_rejected+=int((use&~valid).sum());use&=valid
            d[use]=trial[use];pending&=~use
            if not bool(pending.any()):break
            alpha*=.5
        rejected[pending]+=1;done|=pending;history.append(terms(d).detach().cpu().numpy())
    # Independently accepted group steps can change the global first-hit
    # ordering. Validate the assembled cloud, and discard an infeasible
    # group's measurement update (including its covariance information).
    blocked=torch.zeros(groups,dtype=torch.bool,device=prior.device)
    if feasible is not None:
        for _ in range(groups+1):
            valid=feasible(prior+(g@d[membership,:,None])[...,0])
            new=~valid&~blocked
            if not bool(new.any()):break
            blocked|=new;d[blocked]=0
        if not bool(feasible(prior+(g@d[membership,:,None])[...,0]).all()):
            raise ArithmeticError('Predicted cloud remains hard-mask infeasible after rejecting measurement updates')
    final,a,_,measurement=terms(d,True);m,ranks=reduced_inverse(a,gram)
    if bool(blocked.any()):
        counts[blocked[membership]]=0;group_counts[blocked]=0
        measurement[blocked[membership]]=0;m[blocked]=0
    if covariance_operator is None:posterior,coupling=joseph_blocks(pminus,measurement,g,membership,m,groups)
    else:posterior,coupling=covariance_operator.update(pminus,measurement,m)
    # A group with no measurement has the original prediction, bit for bit.
    untouched=group_counts[membership]==0;posterior[untouched]=pminus[untouched]
    state=prior+(g@d[membership,:,None])[...,0]
    return state,posterior,coupling,{'measurement_count':counts,'group_counts':group_counts,
        'before':history[0],'after':final.detach().cpu().numpy(),'energy_history':np.asarray(history),
        'iterations':iterations,'line_search_stops':rejected,'iteration_limit_reached':(~done)&(group_counts>0),
        'structural_rank':ranks,'mask_rejected_trials':mask_rejected,'mask_blocked_groups':blocked,'prior_count':1,
        'covariance_linear_solve':None if covariance_operator is None else covariance_operator.last_solve,
        'covariance_semantics':'Joseph covariance of restricted linearized estimator; all cross-point correlations retained through recursive matrix-free process/update operators' if covariance_operator is not None else 'Single-update Joseph reference with block prior'}
