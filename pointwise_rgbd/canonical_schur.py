"""Motion / canonical-point MAP with the actual nonzero endpoint priors.

Information is [A C; C^T blockdiag(S_i)]. Only the small MOTION Schur
matrix is dense. Cov(point_i,point_j) is never materialized. Its shared
motion contribution is represented by U_i=S_i^-1 C_i^T and P_motion.
"""
import numpy as np
import torch
from tqdm.auto import tqdm
from .filter_iekf import inverse_spd
from .projection import project
from .group_pose import exp,point_jacobian


def make_prior(partition,support,anchor):
    ids=np.flatnonzero(support)
    return {'keys':np.c_[np.full(len(ids),anchor),ids],
        'xyz':partition.xyz[ids].copy(),'covariance':partition.source_covariance[ids].copy(),
        'source_refs':np.c_[np.full(len(ids),anchor),partition.source[ids,1:]]}


def combine_priors(priors):
    records={}
    for prior in priors:
        for index,(key,p,cov) in enumerate(zip(prior['keys'],prior['xyz'],prior['covariance'])):
            tag=tuple(map(int,key))
            if tag in records and (not np.array_equal(records[tag][0],p) or not np.array_equal(records[tag][1],cov)):raise ValueError('Conflicting endpoint prior')
            records[tag]=(p,cov,None if 'source_refs' not in prior else prior['source_refs'][index])
    keys=sorted(records)
    result={'keys':np.asarray(keys),'xyz':np.stack([records[k][0] for k in keys]),'covariance':np.stack([records[k][1] for k in keys])}
    if all(records[k][2] is not None for k in keys):result['source_refs']=np.stack([records[k][2] for k in keys])
    return result


def prepare(rows,prior,backend):
    lookup={tuple(map(int,key)):i for i,key in enumerate(prior['keys'])};result=[]
    for row in rows:
        keys=row['map_key'];index=np.array([lookup[tuple(map(int,key))] for key in keys],dtype=np.int64)
        if len(np.unique(index))!=len(index):raise ValueError('One point/camera/time has duplicate factors')
        result.append({**row,'mi':backend.tensor(index,torch.long),'zz':backend.tensor(row['z']),
            'ri':inverse_spd(backend.tensor(row['R']),'canonical joint sensor R')})
    return result


def dimension(model):
    if model['type']=='STATIC':return 0
    K=len(model['poses']) if model['type']=='RIGID' else len(model['q'])
    return 6*(K-1) if model['type']=='RIGID' else (2 if model['type']=='PRISMATIC' else 4)+K-1


def world(model,row,mu,backend):
    k=row['k'];p=mu[row['mi']]
    if model['type']=='STATIC':return p,p.new_zeros((len(p),3,0)),torch.eye(3,device=p.device,dtype=p.dtype),np.empty(0,np.int64)
    if model['type']=='RIGID':
        a=model['anchor'];T=backend.tensor(model['poses'][k]);x=p@T[:3,:3].T+T[:3,3]
        if k==a:return p,p.new_zeros((len(p),3,0)),torch.eye(3,device=p.device,dtype=p.dtype),np.empty(0,np.int64)
        indices=np.arange(6*(k-int(k>a)),6*(k-int(k>a))+6)
        return x,point_jacobian(x),T[:3,:3],indices
    from .joint_rgbd import predict,poses
    # Joint geometry has only <=4 columns; its analytic derivative is tested
    # against finite differences independently of this elimination machinery.
    pp=p.detach().cpu().numpy();x,g,v=predict(model,pp,k,row['anchor'],True)
    a=int(row['anchor'][0]);d=g.shape[2];columns=[backend.tensor(g)];indices=list(range(d))
    if k==a:return p,p.new_zeros((len(p),3,0)),torch.eye(3,device=p.device,dtype=p.dtype),np.empty(0,np.int64)
    if k:columns.append(backend.tensor(v[:,:,None]));indices.append(d+k-1)
    if a:columns.append(backend.tensor(-v[:,:,None]));indices.append(d+a-1)
    rotation=poses(model)[k,:3,:3]@poses(model)[a,:3,:3].T
    return backend.tensor(x),torch.cat(columns,2),backend.tensor(rotation),np.array(indices)


def normal(model,mu,data,prior,backend,include_prior=True,information=True):
    N=len(mu);D=dimension(model);A=mu.new_zeros((D,D)) if information else None;C=mu.new_zeros((N,D,3)) if information else None;S=mu.new_zeros((N,3,3)) if information else None;b=mu.new_zeros(D) if information else None;bp=mu.new_zeros((N,3)) if information else None;energy=mu.new_zeros(());count=0;logdet=0.
    if include_prior:
        precision=inverse_spd(backend.tensor(prior['covariance']),'measured endpoint covariance');delta=mu-backend.tensor(prior['xyz'])
        if information:S+=precision;bp-=torch.einsum('nij,nj->ni',precision,delta)
        energy+=.5*torch.einsum('ni,nij,nj->',delta,precision,delta)
    for row in data:
        mi=row['mi'];x,B,R,columns=world(model,row,mu,backend);h,j=project(x,row['camera']);j=j[:,:,:3];e=row['zz']-h
        if not bool((torch.isfinite(h).all(1)&(h[:,2]>0)).all()):return None
        ri=row['ri']
        if information:
            H=j@B;Hp=j@R;wh=H.transpose(1,2)@ri;wp=Hp.transpose(1,2)@ri
            index=backend.tensor(columns,torch.long)
            A[index[:,None],index[None,:]]+=(wh@H).sum(0);b[index]+=(wh@e[:,:,None])[...,0].sum(0)
            C[mi[:,None],index[None,:],:]+=wh@Hp;S[mi]+=wp@Hp;bp[mi]+=(wp@e[:,:,None])[...,0]
        energy+=.5*torch.einsum('ni,nij,nj->',e,ri,e);count+=len(mi)
        logdet+=float(np.linalg.slogdet(row['R'])[1].sum())
    return {'A':A,'C':C,'S':S,'b':b,'bp':bp,'energy':energy,'count':count,'logdet_R':logdet}


def eliminate(terms):
    sinv=inverse_spd(terms['S'],'conditional canonical information')
    cs=terms['C']@sinv
    D=terms['A'].shape[0];N=len(sinv)
    # O(N*D*3) storage, not O(N^2). Point conditional blocks are only 3x3.
    J=terms['A']-cs.permute(1,0,2).reshape(D,N*3)@terms['C'].permute(0,2,1).reshape(N*3,D)
    J=(J+J.T)/2
    rhs=terms['b']-torch.einsum('ndj,nj->d',cs,terms['bp'])
    if not D:return {'J':J,'Sinv':sinv,'P':J.clone(),'rhs':rhs,'U':cs.transpose(1,2)}
    scale=torch.sqrt(torch.diagonal(J));valid=torch.isfinite(scale)&(scale>0)
    if not bool(valid.all()):return None
    normalized=J/scale[:,None]/scale[None,:];root,info=torch.linalg.cholesky_ex(normalized)
    if int(info):return None
    pm=torch.cholesky_inverse(root)/scale[:,None]/scale[None,:]
    return {'J':J,'Sinv':sinv,'P':pm,'rhs':rhs,'U':cs.transpose(1,2)}


def retract(model,step):
    if model['type']=='STATIC':return model
    if model['type']!='RIGID':
        from .joint_rgbd import retract as joint_retract
        return joint_retract(model,step)
    result={**model,'poses':model['poses'].copy()};a=model['anchor'];offset=0
    for k in range(len(result['poses'])):
        if k==a:continue
        d=torch.as_tensor(step[offset:offset+6],dtype=torch.float64);result['poses'][k]=(exp(d).numpy()@result['poses'][k]);offset+=6
    return result


def fit(model,rows,prior,config,backend,initial_map=None,feasible=None):
    data=prepare(rows,prior,backend);mu=backend.tensor(prior['xyz'] if initial_map is None else initial_map).clone();history=[];converged=False
    progress=tqdm(range(config.iterations),desc=f'{model["type"]} motion/map MAP',unit='iteration')
    for iteration in progress:
        t=normal(model,mu,data,prior,backend)
        if t is None:return {'status':'JOINT_UNIDENTIFIABLE','model':model,'reason':'Invalid sensor projection'}
        reduced=eliminate(t)
        if reduced is None:return {'status':'JOINT_UNIDENTIFIABLE','model':model,'reason':'Motion/map Schur information rank deficient'}
        progress.set_postfix(energy=float(t['energy']),refresh=False)
        dm=reduced['P']@reduced['rhs']
        dp=(reduced['Sinv']@(t['bp']-torch.einsum('ndj,d->nj',t['C'],dm))[:,:,None])[...,0]
        history.append(float(t['energy']))
        if max(float(dm.abs().max()) if dm.numel() else 0.,float(dp.abs().max()))<=config.tolerance:converged=True;break
        accepted=False;alpha=1.
        for _ in range(config.line_search_steps):
            trial=retract(model,(alpha*dm).cpu().numpy());trial_mu=mu+alpha*dp
            if feasible is not None and not feasible(trial,trial_mu.cpu().numpy()):alpha*=.5;continue
            tt=normal(trial,trial_mu,data,prior,backend,information=False)
            if tt is not None and bool(tt['energy']<t['energy']):model=trial;mu=trial_mu;accepted=True;break
            alpha*=.5
        if not accepted:break
    t=normal(model,mu,data,prior,backend);r=eliminate(t)
    if r is None:return {'status':'JOINT_UNIDENTIFIABLE','reason':'Final Schur information rank deficient','model':model}
    output={key:value.detach().cpu().numpy() for key,value in r.items() if isinstance(value,torch.Tensor)}
    output.update({key:t[key].detach().cpu().numpy() for key in ('A','C','S')});output.update(prior=prior,mu=mu.cpu().numpy())
    return {'status':'FIT','model':model,'covariance':output['P'],'information':output['J'],'map_state':output,
        'training_count':t['count'],'chi2_training_with_map_prior':float(2*t['energy']),'energy_history':history+[float(t['energy'])],'converged':converged,'iterations':iteration+1,
        'feasible':True if feasible is None else bool(feasible(model,mu.cpu().numpy())),
        'semantics':'Joint motion/canonical-point Gaussian MAP from original source priors and unique raw factors. Shared map, pose-map and cross-time coupling retained by Schur factors; no repeated independent map-noise addition.'}


def score(result,rows,backend):
    state=result['map_state'];prior=state['prior'];mu=backend.tensor(state['mu']);data=prepare(rows,prior,backend)
    terms=normal(result['model'],mu,data,prior,backend,False)
    if terms is None:return {'nll':float('inf'),'chi2':float('inf'),'raw_count':0,'p_value':0.}
    combined={key:terms[key]+backend.tensor(state[key]) for key in ('A','C','S')}
    combined.update(b=terms['b'],bp=terms['bp']);reduced=eliminate(combined)
    if reduced is None:raise ArithmeticError('Predictive motion/map Schur lost rank')
    q=2*terms['energy']-torch.einsum('ni,nij,nj->',terms['bp'],reduced['Sinv'],terms['bp'])-reduced['rhs']@reduced['P']@reduced['rhs']
    logdet=terms['logdet_R']+float(torch.linalg.slogdet(combined['S']).logabsdet.sum()+torch.linalg.slogdet(reduced['J']).logabsdet)-float(np.linalg.slogdet(state['S'])[1].sum()+np.linalg.slogdet(state['J'])[1])
    count=terms['count'];chi=float(q)
    from scipy.stats import chi2
    return {'nll':float(.5*(q+logdet+3*count*np.log(2*np.pi))),'chi2':chi,'raw_count':count,'p_value':float(chi2.sf(chi,3*count)) if count else 0.,'semantics':'Joint held-out predictive Gaussian, integrating shared canonical-point and motion uncertainty across all times/cameras via sparse Schur; fixed associations/calibration.'}


def prediction(result,row,backend):
    state=result['map_state'];prepared=prepare([row],state['prior'],backend)[0];mu=backend.tensor(state['mu'])
    x,B,R,columns=world(result['model'],prepared,mu,backend);h,j=project(x,row['camera']);Hp=j[:,:,:3]@R;H=j[:,:,:3]@B
    mi=prepared['mi'];U=backend.tensor(state['U'])[mi];Sinv=backend.tensor(state['Sinv'])[mi];P=backend.tensor(state['P'])
    W=-Hp@U;index=backend.tensor(columns,torch.long);W[:,:,index]+=H
    covariance=Hp@Sinv@Hp.transpose(1,2)+W@P@W.transpose(1,2)
    return h.cpu().numpy(),covariance.cpu().numpy()


def save_state(path,result):
    state=result['map_state'];arrays={key:value for key,value in state.items() if isinstance(value,np.ndarray)}
    arrays.update(prior_keys=state['prior']['keys'],prior_xyz=state['prior']['xyz'],prior_covariance=state['prior']['covariance'])
    np.savez_compressed(path,**arrays)
