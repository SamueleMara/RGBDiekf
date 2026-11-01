"""Batched nonlinear temporal objective and exact common-domain trial checks."""
import torch
from estimator.temporal_rgbd_terms import temporal_prediction, validate_reference_patch, raw_values, sensor_covariance
from estimator.temporal_zbuffer import first_hit_pixel_owner, incremental_first_hit_pixel_owner, owners_at_gpu, selected_owners
from estimator.temporal_sparse_solve import SparseJointFactorization,connected_columns_gpu
from estimator.temporal_sparse_gpu import sparse_coo
from estimator.temporal_cudss import SparseNumericalFailure


def snapshot(graph):
    graph.layout()
    return {'anchor':graph.anchor.clone(),'states':{k:s.clone() for k,s in graph.states.items()},
            'pixel_keys':graph.pixels.keys.clone(),'pixel_clean':graph.pixels.clean.clone(),
            'cells':{k:(w['cell'].clone(),w['reference_keys'].clone()) for k,w in graph.warps.items()},
            'pixel_offset':graph.pixel_offset,'state_offsets':dict(graph.state_offsets)}


def restore(graph,saved,step=None,rate=0.):
    old=0
    graph.anchor.copy_(saved['anchor'])
    if step is not None and graph.anchor_active:
        graph.anchor+=rate*step[:5*graph.N].reshape(graph.N,5)
    for k,state in saved['states'].items():
        graph.states[k].copy_(state)
        if step is not None:
            start=saved['state_offsets'][k]
            graph.states[k]+=rate*step[start:start+6*graph.N].reshape(graph.N,6)
    graph.pixels.clean=graph.pixels.mean.clone()
    index=graph.pixels.lookup(saved['pixel_keys'])
    graph.pixels.clean[index]=saved['pixel_clean']
    if step is not None:
        graph.pixels.clean[index]+=rate*step[saved['pixel_offset']:].reshape(-1,4)
    for k,(cell,keys) in saved['cells'].items():
        graph.warps[k]['cell']=cell.clone();graph.warps[k]['reference_keys']=keys.clone()
    graph.layout()


def refresh_reference_support_gpu(graph,frames,noise,policy):
    rejected={}
    for current,w in graph.warps.items():
        for ref in frames:
            ix=torch.where(w['reference_frame']==ref)[0]
            if not len(ix):continue
            point=w['point'][ix];cam=w['camera'][ix]
            st=graph.state_and_blocks_gpu(current,point)[0]
            sr=graph.state_and_blocks_gpu(ref,point)[0]
            q=w['uv'][ix]-graph.rig.project_rows(st[:,:3],cam)[0]+graph.rig.project_rows(sr[:,:3],cam)[0]
            pixels,values,reason,cell=validate_reference_patch(frames[ref],cam,q,graph.rig,noise,policy)
            if bool((reason!=0).any()):
                rejected.setdefault(current,[]).append(point[reason!=0])
                continue
            keys=graph.pixels.encode(cam[:,None],torch.full_like(pixels,ref),pixels)
            raw=values.reshape(-1,4)
            graph.pixels.build_or_update_unique_pixel_table_gpu(keys.reshape(-1),raw,sensor_covariance(raw,noise))
            w['cell'][ix]=cell;w['reference_keys'][ix]=keys
    return rejected


def project_ideal_pixels_gpu(graph):
    for current,w in graph.warps.items():
        if bool(torch.count_nonzero(w['covariance'])):continue
        for ref in [graph.anchor_frame,*graph.states]:
            ix=torch.where(w['reference_frame']==ref)[0]
            if not len(ix):continue
            point=w['point'][ix]
            st=graph.state_and_blocks_gpu(current,point)[0]
            sr=graph.state_and_blocks_gpu(ref,point)[0]
            clean=graph.pixels.clean[graph.pixels.lookup(w['reference_keys'][ix])]
            h=temporal_prediction(graph.rig,st,sr,w['camera'][ix],w['uv'][ix],clean,w['cell'][ix])[0]
            graph.pixels.clean[graph.pixels.lookup(w['current_keys'][ix])]=h


def nonlinear_objective_gpu(graph):
    total=torch.zeros((),device=graph.device,dtype=torch.float64)
    if graph.anchor_active:
        whitened=torch.linalg.solve_triangular(torch.linalg.cholesky(graph.anchor_covariance),graph.anchor.T,upper=False)
        total+=whitened.square().sum()/2
    active=graph.pixels.active_likelihood
    whitened=torch.linalg.solve_triangular(torch.linalg.cholesky(graph.pixels.covariance[active]),
        (graph.pixels.clean[active]-graph.pixels.mean[active])[...,None],upper=False)
    total+=whitened.square().sum()/2
    for current,(prev,F,Q) in graph.process.items():
        st=graph.state_and_blocks_gpu(current)[0];sr=graph.state_and_blocks_gpu(prev)[0]
        whitened=torch.linalg.solve_triangular(torch.linalg.cholesky(Q),(st-sr@F.T).T,upper=False)
        total+=whitened.square().sum()/2
    equality=torch.zeros((),device=graph.device,dtype=torch.float64)
    for current,w in graph.warps.items():
        for ref in [graph.anchor_frame,*graph.states]:
            ix=torch.where(w['reference_frame']==ref)[0]
            if not len(ix):continue
            point=w['point'][ix]
            st=graph.state_and_blocks_gpu(current,point)[0];sr=graph.state_and_blocks_gpu(ref,point)[0]
            values=graph.pixels.clean[graph.pixels.lookup(w['reference_keys'][ix])]
            h=temporal_prediction(graph.rig,st,sr,w['camera'][ix],w['uv'][ix],values,w['cell'][ix])[0]
            r=graph.pixels.clean[graph.pixels.lookup(w['current_keys'][ix])]-h
            if not bool(torch.count_nonzero(w['covariance'])):
                if r.numel():equality=torch.maximum(equality,r.abs().max())
            else:
                white=torch.linalg.solve_triangular(torch.linalg.cholesky(w['covariance']),r.T,upper=False)
                total+=white.square().sum()/2
    if graph.fej is not None:
        ids=graph.scalar_ids();order=torch.argsort(ids);index=order[torch.searchsorted(ids[order],graph.fej_scalar_ids)]
        delta=graph.flat_values()[index]-graph.fej.linearization
        total+=graph.fej.constant+delta@(graph.fej.gradient+torch.sparse.mm(graph.fej.information,delta[:,None]).flatten()/2)
        c=graph.fej.constraint_residual+torch.sparse.mm(graph.fej.constraints,delta[:,None]).flatten()
        if len(c):equality=torch.maximum(equality,c.abs().max())
    return total,equality


def outside_mask(owners,frame,points):
    c,p,pixel=selected_owners(owners)
    outside=torch.zeros((frame.mask.shape[0],points),device=p.device,dtype=torch.bool)
    outside[c,p]=~frame.mask[c,pixel//frame.W,pixel%frame.W]
    return outside


def evaluate_nonlinear_objective_and_visibility_gpu(graph,frames,noise,policy,baseline_states,baseline_owners,baseline_outside,incremental=True,domain_validator=None):
    graph.last_rejection=None
    owners={};rejected={};reasons=[]
    def reject(k,points,reason):
        rejected.setdefault(k,[]).append(points);reasons.append(reason)
    for k,frame in frames.items():
        state=graph.state_and_blocks_gpu(k)[0]
        owner=incremental_first_hit_pixel_owner(state[:,:3],baseline_states[k][:,:3],baseline_owners[k],graph.rig,frame) if incremental else first_hit_pixel_owner(state[:,:3],graph.rig,frame,dense=False)
        invalid=outside_mask(owner,frame,graph.N)&~baseline_outside[k]
        if bool(invalid.any()):
            c,p,pixel=selected_owners(owner);bad=invalid[c,p]
            previous=owners_at_gpu(baseline_owners[k],c[bad],pixel[bad])
            # A hidden stationary point can be exposed by a moving competitor.
            reject(k,torch.cat((p[bad],previous[previous>=0])),'NEW_FOREGROUND_VIOLATION')
        owners[k]=owner
    for current,w in graph.warps.items():
        pixels=w['uv'][:,1].long()*frames[current].W+w['uv'][:,0].long()
        winner=owners_at_gpu(owners[current],w['camera'],pixels)
        changed=winner!=w['point']
        if bool(changed.any()):
            competitors=winner[changed];competitors=competitors[competitors>=0]
            reject(current,torch.cat((w['point'][changed],competitors)),'ACTIVE_OWNER_CHANGED')
    for current,points in refresh_reference_support_gpu(graph,frames,noise,policy).items():
        reject(current,torch.cat(points),'INVALID_REFERENCE_SUPPORT')
    if domain_validator is not None and not domain_validator():
        reject(next(reversed(graph.states)),getattr(graph,'domain_bad_points',torch.arange(graph.N,device=graph.device)),
            'COMMON_MEASUREMENT_DOMAIN_CHANGED')
    if rejected:
        groups={k:torch.unique(torch.cat(points)) for k,points in rejected.items()}
        first=next(iter(groups));graph.last_rejection={'by_frame':groups,'frame':first,'points':groups[first]}
        return False,'+'.join(dict.fromkeys(reasons)),owners
    project_ideal_pixels_gpu(graph)
    _,equality=nonlinear_objective_gpu(graph)
    if float(equality)>2e-11:return False,'HARD_EQUALITY_INFEASIBLE',owners
    return True,'VALID_COMMON_DOMAIN',owners


def true_variable_block_components_gpu(graph,problem):
    """Connectivity only: join coordinates of the actual typed variable blocks.

    The extra symbolic chains are NEVER added to the information matrix. They
    prevent a line-search rate from splitting a point, source chart or raw pixel.
    """
    n=problem.scalar_dimension;edges=[]
    groups=[]
    if graph.anchor_active:groups.append(torch.arange(5*graph.N,device=graph.device).reshape(-1,5))
    for k,offset in graph.state_offsets.items():groups.append(offset+torch.arange(6*graph.N,device=graph.device).reshape(-1,6))
    groups.append(graph.pixel_offset+torch.arange(4*len(graph.pixels.keys),device=graph.device).reshape(-1,4))
    for group in groups:
        a,b=group[:,:-1].reshape(-1),group[:,1:].reshape(-1)
        edges.append(torch.stack((a,b)));edges.append(torch.stack((b,a)))
    if problem.information is not None:edges.append(problem.information.indices())
    index=torch.cat(edges,1)
    symbolic=sparse_coo(index,torch.ones(index.shape[1],device=graph.device,dtype=torch.float64),(n,n))
    labels,_=connected_columns_gpu(problem.A,problem.C,symbolic)
    unique,inverse=torch.unique(labels,return_inverse=True)
    return inverse,len(unique)


def point_component(graph,labels,frame,points):
    column=points*5 if frame==graph.anchor_frame else graph.state_offsets[frame]+points*6
    return labels[column]


def solve_nonlinear_sparse_iekf_gpu(graph,frames,noise,policy,max_iters,tolerance,domain_validator=None):
    graph.layout();project_ideal_pixels_gpu(graph)
    history=[];graph.nonlinear_history=history;converged=False;accepted=0
    accepted_points=torch.zeros(graph.N,device=graph.device,dtype=torch.bool)
    for iteration in range(max_iters):
        problem=graph.assemble_sparse_linearized_problem_gpu()
        with SparseJointFactorization(problem) as factor:
            step,certificate=factor.solve_constrained_sparse_joint_gpu()
            profile=dict(factor.profile)
        # The closed native handle does not release Torch COO/CSR/transform
        # tensors while Python still holds the factor across the next setup.
        del factor
        labels,count=true_variable_block_components_gpu(graph,problem)
        rates=torch.ones(count,device=graph.device,dtype=torch.float64)
        attempts=torch.zeros(count,device=graph.device,dtype=torch.long)
        saved=snapshot(graph)
        base_states={k:graph.state_and_blocks_gpu(k)[0].clone() for k in frames}
        base_owners={k:first_hit_pixel_owner(state[:,:3],graph.rig,frames[k],dense=False) for k,state in base_states.items()}
        base_outside={k:outside_mask(owner,frames[k],graph.N) for k,owner in base_owners.items()}
        old=float(nonlinear_objective_gpu(graph)[0]);success=False;reason='NO_FEASIBLE_DESCENT'
        trials=[]
        # Discrete ownership can reveal a second competitor after another
        # component stops. Allow bounded cascades; each component still gets
        # at most 25 bisections and the runner enforces a hard OS deadline.
        for backtracks in range(128):
            proposal=step*rates[labels]
            restore(graph,saved,proposal,1.)
            feasible,reason,owners=evaluate_nonlinear_objective_and_visibility_gpu(graph,frames,noise,policy,base_states,base_owners,base_outside,domain_validator=domain_validator)
            if not feasible:
                rejected=graph.last_rejection
                if rejected is None:bad=torch.arange(count,device=graph.device)
                else:
                    groups=rejected.get('by_frame',{rejected['frame']:rejected['points']})
                    bad=torch.unique(torch.cat([point_component(graph,labels,k,points) for k,points in groups.items()]))
                trials.append({'trial':backtracks,'reason':reason,'affected_true_components':len(bad),
                    'frozen_components':int((rates==0).sum())})
                attempts[bad]+=1;rates[bad]*=.5;rates[bad[attempts[bad]>=25]]=0.
                continue
            proposed=float(nonlinear_objective_gpu(graph)[0]);candidate=snapshot(graph)
            restore(graph,saved)
            # Newly registered raw sensor pixels are now in BOTH objectives.
            baseline=float(nonlinear_objective_gpu(graph)[0])
            if proposed<=baseline+32*torch.finfo(torch.float64).eps*max(1.,abs(baseline)):
                restore(graph,candidate);success=True;accepted+=1
                current=next(reversed(graph.states))
                pc=point_component(graph,labels,current,torch.arange(graph.N,device=graph.device))
                accepted_points|=rates[pc]>0
                break
            rates*=.5
            trials.append({'trial':backtracks,'reason':'GLOBAL_OBJECTIVE_INCREASE','affected_true_components':count})
        if not success:restore(graph,saved)
        info_step=float(problem.apply_J(proposal).square().sum())
        if problem.information is not None:
            info_step+=float(proposal@torch.sparse.mm(problem.information,proposal[:,None]).flatten())
        history.append({'iteration':iteration,'objective':old,'proposal_objective':proposed if success else None,
            'accepted':success,'backtracks':backtracks,'rate_min_max':[float(rates.min()),float(rates.max())],'step_information_norm2':info_step,
            'reason':reason,'registered_pixels':len(graph.pixels.keys),'solver_certificate':certificate,'sparse_profile':profile,
            'true_components':count,'frozen_components':int((rates==0).sum()),'trials':trials})
        print(f'[FULL sparse IEKF] iteration={iteration} accepted={success} objective={proposed if success else old:.9g} backtracks={backtracks} frozen_components={int((rates==0).sum())}',flush=True)
        del problem
        del saved,base_states,base_owners,base_outside,labels,step,proposal
        if success:del candidate,pc
        if success and info_step<tolerance and bool((rates>0).all()):converged=True;break
        if not success:break
    if not accepted or not bool(accepted_points.any()):
        error=SparseNumericalFailure('FULL_NONLINEAR_NO_ACCEPTED_STEP: no posterior frame will be published')
        error.profile={'nonlinear_history':history,'point_count':graph.N}
        raise error
    final=graph.assemble_sparse_linearized_problem_gpu()
    factor=SparseJointFactorization(final)
    return factor,{'history':history,'converged':converged,'accepted_iterations':accepted,'objective':float(nonlinear_objective_gpu(graph)[0]),'_accepted_point_mask':accepted_points}
