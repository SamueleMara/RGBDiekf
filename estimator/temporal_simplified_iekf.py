"""Fixed-prior 6D IEKF ablation, with explicitly conditional covariance."""
import torch
from estimator.temporal_zbuffer import selected_owners

REFERENCE_UNCERTAINTY = ('fixed_reference_state + marginal_sensor_noise_only; '
                         'ignores_temporal_and_cross_point_correlations')


def row_signature(rows):
    return torch.cat((rows.camera[:,None], rows.point[:,None], rows.pixel[:,None],
                      rows.reference_frame[:,None]),1)


def same_domain(a,b):
    return a.shape==b.shape and torch.equal(a,b)


def foreground_violations(owners, frame):
    c,p,pixel=selected_owners(owners);y,x=pixel//frame.W,pixel%frame.W
    bg = ~frame.mask[c,y,x]
    return set(zip(c[bg].tolist(),p[bg].tolist()))


def information(rows, N, dtype, device):
    A = torch.zeros((N,6,6),dtype=dtype,device=device)
    b = torch.zeros((N,6),dtype=dtype,device=device)
    if len(rows):
        wh = torch.linalg.solve(rows.R,rows.H_t)
        wr = torch.linalg.solve(rows.R,(rows.y-rows.h).unsqueeze(-1)).squeeze(-1)
        A.index_add_(0,rows.point,rows.H_t.transpose(-1,-2)@wh)
        b.index_add_(0,rows.point,torch.einsum('mji,mj->mi',rows.H_t,wr))
    return A,b


def objective(s,mean,precision,rows):
    return float(point_objective(s,mean,precision,rows).sum())


def point_objective(s,mean,precision,rows):
    ds = s-mean
    cost = torch.einsum('ni,nij,nj->n',ds,precision,ds)
    if len(rows):
        r = rows.y-rows.h
        cost.index_add_(0,rows.point,torch.einsum('mi,mi->m',r,torch.linalg.solve(rows.R,r.unsqueeze(-1)).squeeze(-1)))
    return cost/2


def point_domains(rows,N,C):
    out=torch.full((C,N,2),-1,dtype=torch.long,device=rows.point.device)
    out[rows.camera,rows.point,0]=rows.pixel;out[rows.camera,rows.point,1]=rows.reference_frame
    return out


def outside_points(owners,frame,N):
    result=torch.zeros(N,dtype=torch.bool,device=owners.reason.device)
    c,p,pix=selected_owners(owners);result[p[~frame.mask[c,pix//frame.W,pix%frame.W]]]=True
    return result


def update_points_temporal_simplified(mean, prior, build, frame, max_iters=10, tol=1e-8):
    N = len(mean); s = mean.clone(); precision = torch.linalg.inv(prior)
    rows,owners,reason,life = build(s)
    C=owners.reason.shape[0];signature=point_domains(rows,N,C);initial_outside=outside_points(owners,frame,N)
    ever_accepted=torch.zeros(N,dtype=torch.bool,device=s.device)
    history = []; converged = not len(rows)
    for it in range(max_iters if len(rows) else 0):
        A,b = information(rows,N,s.dtype,s.device)
        A += precision; b -= torch.einsum('nij,nj->ni',precision,s-mean)
        step = torch.linalg.solve(A,b.unsqueeze(-1)).squeeze(-1)
        old_point=point_objective(s,mean,precision,rows);old=float(old_point.sum())
        base=s.clone();base_owner=owners;pending=torch.zeros(N,dtype=torch.bool,device=s.device);pending[rows.point]=True
        accepted_points=torch.zeros_like(pending);rates=torch.zeros(N,dtype=s.dtype,device=s.device)
        for ls in range(25):
            rate = 2.**-ls
            trial=s.clone();trial[pending]=base[pending]+rate*step[pending]
            rr,oo,re,ll = build(trial)
            domain=(point_domains(rr,N,C)==signature).all(dim=(0,2))
            feasible=~outside_points(oo,frame,N)|initial_outside
            costs=point_objective(trial,mean,precision,rr)
            good=pending&domain&feasible&(costs<=old_point+1e-12*old_point.abs().clamp_min(1.))
            s[good]=trial[good];rates[good]=rate;accepted_points|=good;pending&=~good
            if not bool(pending.any()):break
        # Independently accepted steps can interact through depth ordering.
        # Roll back both parties of every changed owner, then rerender. No
        # missing row can improve the frozen observed-pixel objective.
        for verification in range(25):
            rr,oo,re,ll=build(s)
            bad=~(point_domains(rr,N,C)==signature).all(dim=(0,2))|(outside_points(oo,frame,N)&~initial_outside)
            if not bool(bad.any()):break
            before=torch.where(base_owner.reason==0,base_owner.pixel,-1);after=torch.where(oo.reason==0,oo.pixel,-1)
            bad|=(before!=after).any(0);rollback=bad&accepted_points
            if not bool(rollback.any()):
                rollback=accepted_points.clone()
            s[rollback]=base[rollback];rates[rollback]=0;accepted_points[rollback]=False
        else:
            s=base;accepted_points[:]=False;rates[:]=0;rr,oo,re,ll=build(s)
        rows,owners,reason,life=rr,oo,re,ll
        accepted=bool(accepted_points.any());ever_accepted|=accepted_points
        actual=s-base;metric=float(torch.einsum('ni,nij,nj->n',actual,A,actual).max())
        cost=objective(s,mean,precision,rows)
        history.append({'iteration':it,'objective':old,'proposal_objective':cost,'accepted':accepted,
                        'step_information_norm2':metric,'gradient_norm':float(b.norm()),'backtracks':ls,
                        'accepted_points':int(accepted_points.sum()),'rejected_points':int((pending&~accepted_points).sum()),
                        'reason':'ACCEPTED' if accepted else 'COMMON_DOMAIN_OR_MASK_OR_DESCENT_REJECTED'})
        if not accepted: break
        unresolved=(~accepted_points)&(torch.einsum('ni,nij,nj->n',step,A,step)>=tol)
        if metric < tol and not bool(unresolved.any()): converged = True; break
    A,_ = information(rows,N,s.dtype,s.device)
    observed = torch.zeros(N,dtype=torch.bool,device=s.device)
    observed[rows.point] = True
    P = prior.clone()
    assimilated=observed&ever_accepted
    if bool(assimilated.any()): P[assimilated] = torch.linalg.inv((precision+A)[assimilated])
    s[~observed] = mean[~observed]  # exact no-row contract, including nonzero velocity
    return s,P,rows,owners,reason,life,{'history':history,'converged':converged,
                                      'assimilated_points':int(assimilated.sum()),'eligible_but_rejected_points':int((observed&~ever_accepted).sum()),
                                      '_assimilated_mask':assimilated,
                                      'reference_uncertainty':REFERENCE_UNCERTAINTY,
                                      'objective_scope':'fixed current pixel/owner/reference domain; re-evaluate valid reference interpolation cells'}
