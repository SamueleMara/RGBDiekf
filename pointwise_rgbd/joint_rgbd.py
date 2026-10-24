"""Fixed world joint geometry, endpoint-relative motion, raw RGB-D evidence.

Only a small (K+4)-dimensional motion matrix is dense. No matrix indexed by
pairs of points or pixels is formed. All scores condition on the selected raw
correspondences and endpoint geometry; initial noise guesses are uncalibrated.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from rgbd_common.array_store import ArrayStore
from rgbd_common.lie import skew
from .group_evidence_audit import projection_numpy
from .group_visualization import serial
from .store import json_write


def plane(u):
    v=np.eye(3)[np.argmin(abs(u))];v=v-u*(u@v);v/=np.linalg.norm(v)
    return np.c_[v,np.cross(u,v)]


def so3_jac(phi):
    t=np.linalg.norm(phi);s=skew(phi)
    if t<np.sqrt(np.finfo(float).eps):return np.eye(3)+s/2+s@s/6
    return np.eye(3)+(1-np.cos(t))/t**2*s+(t-np.sin(t))/t**3*s@s


def predict(model,p,k,anchor,derivatives=False):
    """p is a measured WORLD point anchored at its own actual endpoint."""
    u=model['axis'];a=model.get('axis_point',np.zeros(3));q=model['q'];E=plane(u)
    theta=q[k]-q[np.asarray(anchor)];theta=np.broadcast_to(theta,(len(p),))
    d=2 if model['type']=='PRISMATIC' else 4
    x=np.empty_like(p);geo=np.zeros((len(p),3,d));dq=np.empty_like(p)
    for t in np.unique(theta):
        use=theta==t
        if model['type']=='PRISMATIC':
            x[use]=p[use]+t*u;geo[use]=t*E;dq[use]=u
        else:
            R=Rotation.from_rotvec(t*u).as_matrix();y=(p[use]-a)@R.T
            x[use]=y+a
            if derivatives:
                geo[use,:,:2]=np.stack([-skew(v) for v in y])@so3_jac(t*u)@(t*E)
                geo[use,:,2:]=(np.eye(3)-R)@E
            dq[use]=np.cross(np.broadcast_to(u,y.shape),y)
    return (x,geo,dq) if derivatives else x


def poses(model):
    K=len(model['q']);T=np.tile(np.eye(4),(K,1,1));u=model['axis']
    if model['type']=='PRISMATIC':T[:,:3,3]=model['q'][:,None]*u
    else:
        T[:,:3,:3]=Rotation.from_rotvec(model['q'][:,None]*u).as_matrix()
        T[:,:3,3]=model['axis_point']-np.einsum('kij,j->ki',T[:,:3,:3],model['axis_point'])
    return T


def initialize(rigid,canonical):
    # RigEuler 481b2cf2 rigid_trajectory_reduction Sections 28-31:
    # uncentred translation PCA; finite rotation-log PCA; gauged (I-R)a=t;
    # accumulated spatial relative rotations. Never instantaneous omega x v.
    T=rigid['poses'];K=len(T);translation=T[:,:3,3]
    u=np.linalg.svd(translation,full_matrices=False)[2][0];q=translation@u;q-=q[0]
    P={'type':'PRISMATIC','axis':u,'q':q}
    phi=Rotation.from_matrix(T[:,:3,:3]).as_rotvec();u=np.linalg.svd(phi,full_matrices=False)[2][0]
    q=np.zeros(K)
    for k in range(1,K):q[k]=q[k-1]+Rotation.from_matrix(T[k,:3,:3]@T[k-1,:3,:3].T).as_rotvec()@u
    E=plane(u);c=canonical.mean(0);A=(np.eye(3)-T[:,:3,:3])@E
    b=translation-np.einsum('kij,j->ki',np.eye(3)-T[:,:3,:3],c)
    offset,_,rank,singular=np.linalg.lstsq(A.reshape(-1,2),b.ravel(),rcond=None)
    axis_point=c+E@offset;axis_point-=u*(u@axis_point)
    R={'type':'REVOLUTE','axis':u,'axis_point':axis_point,'q':q,'finite_axis_initial_rank':int(rank),'finite_axis_initial_singular_values':singular}
    return P,R


def load_factors(partition,support,dataset,phase_root,anchor):
    store=ArrayStore(Path(phase_root)/'factors.sqlite');rows=[]
    for k in range(len(dataset.times)):
        if not store.contains(str(k)):continue
        row=store.read(str(k));use=support[row['point_id']]
        for ci,camera in enumerate(dataset.cameras):
            chosen=use&(row['camera']==ci)
            if not chosen.any():continue
            ids=row['point_id'][chosen]
            rows.append({'k':k,'camera_index':ci,'camera':camera,'pixel':row['pixel'][chosen],
                'p':partition.xyz[ids],'anchor':np.full(len(ids),anchor,dtype=int),'z':row['z'][chosen],
                'R':row['R'][chosen],'phase1_residual':row['residual'][chosen],'point_id':ids,
                'map_key':np.c_[np.full(len(ids),anchor),ids],'source_covariance':partition.source_covariance[ids]})
    store.close();return rows


def normal(model,rows):
    K=len(model['q']);d=2 if model['type']=='PRISMATIC' else 4;D=d+K-1
    A=np.zeros((D,D));b=np.zeros(D);energy=0.;logdet=0.;count=0;records=[]
    for row in rows:
        k=row['k'];p=row['p'];x,g,v=predict(model,p,k,row['anchor'],True)
        h,j=projection_numpy(x,row['camera']);j=j[:,:,:3]
        if not np.isfinite(h).all() or (h[:,2]<=0).any():return np.inf,A,b,count,logdet,[]
        e=row['z']-h;ri=np.linalg.inv(row['R']);H=j@g;hv=np.einsum('nij,nj->ni',j,v)
        wr=H.transpose(0,2,1)@ri
        A[:d,:d]+=(wr@H).sum(0);b[:d]+=np.einsum('nij,nj->i',wr,e)
        wv=np.einsum('ni,nij->nj',hv,ri)
        a=np.einsum('ni,ni->n',wv,hv);bb=np.einsum('ni,ni->n',wv,e);c=np.einsum('nij,nj->ni',wr,hv)
        # Each factor depends on q[k]-q[anchor], so both endpoints contribute.
        for anchor in np.unique(row['anchor']):
            use=row['anchor']==anchor
            pairs=[(t,sign) for t,sign in ((k,1),(int(anchor),-1)) if t!=0]
            for t,sign in pairs:
                index=d+t-1;A[:d,index]+=sign*c[use].sum(0);A[index,:d]+=sign*c[use].sum(0);b[index]+=sign*bb[use].sum()
                for s,sign2 in pairs:A[index,d+s-1]+=sign*sign2*a[use].sum()
        q=float(np.einsum('ni,nij,nj->',e,ri,e));energy+=q/2;logdet+=float(np.linalg.slogdet(row['R'])[1].sum());count+=len(p)
        records.append((row,h,e,H,hv))
    return energy,A,b,count,logdet,records


def retract(model,step):
    d=2 if model['type']=='PRISMATIC' else 4;E=plane(model['axis']);u=model['axis']+E@step[:2];u/=np.linalg.norm(u)
    trial={**model,'axis':u,'q':model['q']+np.r_[0.,step[d:]]}
    if d==4:
        a=model['axis_point']+E@step[2:4]
        # Gauge nearest world origin. An along-axis shift changes no prediction.
        trial['axis_point']=a-u*(u@a)
    return trial


def fit(initial,rows,config,feasible=None,map_prior=None,initial_map=None,backend=None):
    if map_prior is not None:
        from .canonical_schur import fit as fit_map
        if backend is None:
            from rgbd_common.backends.torch_cuda import Backend
            backend=Backend('cpu')
        check=None if feasible is None else lambda model,mu:feasible({**model,'_map_xyz':mu,'_map_keys':map_prior['keys']})
        result=fit_map(initial,rows,map_prior,config,backend,initial_map,check)
        if result['status']=='FIT' and initial['type'] in ('PRISMATIC','REVOLUTE'):
            d=2 if initial['type']=='PRISMATIC' else 4;q=result['model']['q'][1:]
            statistic=float(q@np.linalg.solve(result['covariance'][d:,d:],q));result['motion_statistic']=statistic;result['motion_gate']=config.gate(len(q))
            if statistic<=result['motion_gate']:result.update(status='JOINT_UNIDENTIFIABLE',reason='Motion is not distinguishable from the endpoint/sensor uncertainty')
        return result
    model={key:(value.copy() if isinstance(value,np.ndarray) else value) for key,value in initial.items()};history=[];converged=False
    for iteration in range(max(config.iterations,1)):
        energy,A,b,count,_,_=normal(model,rows);history.append(float(energy))
        if not count:return {'status':'JOINT_UNIDENTIFIABLE','reason':'No training RGB-D correspondences','model':model}
        scale=np.sqrt(np.diag(A));active=scale>0
        if not active.all():return {'status':'JOINT_UNIDENTIFIABLE','reason':'Unobserved geometry or frame coordinate','model':model,'active':active}
        normalized=A/scale[:,None]/scale[None,:];eigen=np.linalg.eigvalsh(normalized)
        rank=int((eigen>eigen[-1]*len(A)*np.finfo(float).eps).sum())
        if rank<len(A):return {'status':'JOINT_UNIDENTIFIABLE','reason':'Joint geometry rank deficient','rank':rank,'dimension':len(A),'model':model}
        step=np.linalg.solve(normalized,b/scale)/scale
        if np.max(abs(step))<=config.tolerance:converged=True;break
        accepted=False;alpha=1.
        for _ in range(config.line_search_steps):
            trial=retract(model,alpha*step);value=normal(trial,rows)[0]
            if np.isfinite(value) and value<energy and (feasible is None or feasible(trial)):
                model=trial;accepted=True;break
            alpha*=.5
        if not accepted:break
    energy,A,_,count,_,_=normal(model,rows)
    scale=np.sqrt(np.diag(A));normalized=A/scale[:,None]/scale[None,:]
    try:cov=np.linalg.inv(normalized)/scale[:,None]/scale[None,:];np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:return {'status':'JOINT_UNIDENTIFIABLE','reason':'Final joint information not SPD','model':model}
    d=2 if model['type']=='PRISMATIC' else 4;q=model['q'][1:]
    motion_stat=float(q@np.linalg.solve(cov[d:,d:],q))
    observable=motion_stat>config.gate(len(q))
    return {'status':'FIT' if observable else 'JOINT_UNIDENTIFIABLE','model':model,'covariance':cov,'information':A,'chi2_training':2*energy,'training_count':count,'iterations':iteration+1,'energy_history':history+[float(energy)],'converged':converged,
        'motion_statistic':motion_stat,'motion_gate':config.gate(len(q)),'feasible':True if feasible is None else bool(feasible(model)),
        'observability_policy':'full normalized information rank and nonzero displacement outside the existing 1% chi-square noise-compatibility set; reuse is an uncalibrated policy assumption, not physical calibration'}


def score(fit_result,rows):
    if 'map_state' in fit_result:
        from .canonical_schur import score as score_map
        from rgbd_common.backends.torch_cuda import Backend
        return score_map(fit_result,rows,Backend('cpu'))
    energy,A,b,count,logdet,_=normal(fit_result['model'],rows)
    if not count:return {'nll':float('inf'),'chi2':float('inf'),'raw_count':0,'p_value':0.}
    precision=fit_result['information'];post=precision+A
    q=2*energy-float(b@np.linalg.solve(post,b))
    det=logdet+np.linalg.slogdet(post)[1]-np.linalg.slogdet(precision)[1]
    return {'nll':float(.5*(q+det+3*count*np.log(2*np.pi))),'chi2':q,'raw_count':count,'p_value':float(chi2.sf(q,3*count)),
        'semantics':'joint held-out Gaussian predictive score with shared geometry/q uncertainty across all times, fixed canonical geometry/calibration/associations'}


def static_score(rows):
    chi=0.;count=0;logdet=0.
    for row in rows:
        h,_=projection_numpy(row['p'],row['camera']);e=row['z']-h
        chi+=float(np.einsum('ni,ni->',e,np.linalg.solve(row['R'],e[:,:,None])[:,:,0]));count+=len(e);logdet+=float(np.linalg.slogdet(row['R'])[1].sum())
    return {'chi2':chi,'raw_count':count,'nll':float(.5*(chi+logdet+3*count*np.log(2*np.pi))),'p_value':float(chi2.sf(chi,3*count)) if count else 0.}


def distinct_predictions(a,b,rows,config):
    # Conservative comparison: both models were fitted to the same training
    # data, so unknown cross-correlation is bounded, never set to zero.
    largest=0.
    for row in rows:
        ha=prediction_covariance(a,row);hb=prediction_covariance(b,row)
        e=ha[0]-hb[0];sigma=row['R']+2*(ha[1]+hb[1])
        d2=np.einsum('ni,ni->n',e,np.linalg.solve(sigma,e[:,:,None])[:,:,0])
        if len(d2):largest=max(largest,float(d2.max()))
    return {'distinguishable':largest>config.gate(3),'max_prediction_difference_d2':largest,'gate':config.gate(3),
        'policy':'conservative per-observation existing association compatibility set; no calibrated global model-selection probability is claimed'}


def prediction_covariance(fit_result,row):
    if 'map_state' in fit_result and 'map_key' in row:
        from .canonical_schur import prediction
        from rgbd_common.backends.torch_cuda import Backend
        return prediction(fit_result,row,Backend('cpu'))
    model=fit_result['model'];d=2 if model['type']=='PRISMATIC' else 4;K=len(model['q'])
    x,g,v=predict(model,row['p'],row['k'],row['anchor'],True);h,j=projection_numpy(x,row['camera']);j=j[:,:,:3]
    H=j@g;hv=np.einsum('nij,nj->ni',j,v);P=fit_result['covariance'];cov=H@P[:d,:d]@H.transpose(0,2,1)
    for anchor in np.unique(row['anchor']):
        use=row['anchor']==anchor;z=np.zeros(d+K-1)
        if row['k']:z[d+row['k']-1]+=1
        if anchor:z[d+int(anchor)-1]-=1
        cross=H[use]@(P[:d]@z);variance=z@P@z
        cov[use]+=variance*hv[use,:,None]*hv[use,None,:]+cross[:,:,None]*hv[use,None,:]+hv[use,:,None]*cross[:,None,:]
    return h,cov


def measurement_score(result,rows):
    """Actual fitted raw UVZ likelihood on the complete common measurement set.

    These are in-sample residuals, not independent posterior predictions.
    The proper canonical prior remains in the MAP objective and covariance.
    """
    model=result['model'];total=0.;logdet=0.;count=0;per_camera={}
    state=result.get('map_state');lookup=None
    if state is not None:lookup={tuple(map(int,key)):i for i,key in enumerate(state['prior']['keys'])}
    for row in rows:
        p=row['p'] if lookup is None else state['mu'][[lookup[tuple(map(int,key))] for key in row['map_key']]]
        if model['type']=='STATIC':x=p
        elif model['type']=='RIGID':
            T=model['poses'][row['k']];x=p@T[:3,:3].T+T[:3,3]
        else:x=predict(model,p,row['k'],row['anchor'])
        h,_=projection_numpy(x,row['camera']);e=row['z']-h
        q=float(np.einsum('ni,ni->',e,np.linalg.solve(row['R'],e[:,:,None])[:,:,0]))
        total+=q;logdet+=float(np.linalg.slogdet(row['R'])[1].sum());count+=len(e)
        ci=str(row['camera_index']);v=per_camera.setdefault(ci,{'raw_count':0,'chi2':0.,'sum_squared_uvz':np.zeros(3)})
        v['raw_count']+=len(e);v['chi2']+=q;v['sum_squared_uvz']+=np.sum(e*e,axis=0)
    for v in per_camera.values():v['uvz_rmse_pixels_pixels_metres']=np.sqrt(v.pop('sum_squared_uvz')/v['raw_count']).tolist()
    prior_chi=0.
    if state is not None:
        delta=state['mu']-state['prior']['xyz']
        prior_chi=float(np.einsum('ni,ni->',delta,np.linalg.solve(state['prior']['covariance'],delta[:,:,None])[:,:,0]))
    dimension=len(result['information']);dof=max(3*count-dimension,1)
    return {'nll':float(.5*(total+prior_chi+logdet+3*count*np.log(2*np.pi))), 'chi2':total,'map_prior_chi2':prior_chi,'raw_count':count,
        'p_value':float(chi2.sf(total,dof)) if count else 0.,'nominal_residual_dof':dof,'per_camera':per_camera,
        'semantics':'All-camera fitted sensor residuals and proper endpoint-prior MAP objective, conditioned on correspondences/calibration. In-sample local Gaussian compatibility diagnostic under uncalibrated assumptions, not independent validation.'}


def select(candidates,rows,config):
    valid=[];reports={}
    for kind,result in candidates.items():
        if result['status']!='FIT':reports[kind]={'status':result['status'],'reason':result.get('reason','motion not observable')};continue
        evidence=measurement_score(result,rows);result['complete_multiview_score']=evidence
        consistent=evidence['raw_count']>0 and evidence['p_value']>=config.association_tail_probability and result.get('feasible',False) and result.get('converged',False)
        reports[kind]={'status':'SUPPORTED' if consistent else 'INCONSISTENT_OR_UNCONVERGED','complete_multiview':evidence}
        if consistent:valid.append(kind)
    decision='JOINT_TYPE_UNRESOLVED';comparison=None
    if len(valid)==1:decision=valid[0]
    elif len(valid)==2:
        comparison=distinct_predictions(candidates[valid[0]],candidates[valid[1]],rows,config)
        if comparison['distinguishable']:
            decision=min(valid,key=lambda kind:candidates[kind]['complete_multiview_score']['nll'])
    return {'status':decision,'selected_hypothesis':decision,'candidates':reports,'comparison':comparison,
        'camera_policy':'all_available','measurement_camera_indices':sorted({int(row['camera_index']) for row in rows}),
        'assumptions':'Both hypotheses fit and score the SAME complete multiview measurements, never camera partitions. Existing 1% uncalibrated noise-compatibility and motion-observability policy retained. No BIC classifier or separate static fit. Candidate-specific degeneracy never vetoes the other supported model. Scores are conditional in-sample diagnostics, not held-out performance.'}


def run(partition,rigid,dataset,phase_root,config,root,feasible=None,backend=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if rigid['status']!='FIT':return {'status':'JOINT_TYPE_UNRESOLVED','reason':'Shared rigid trajectory unresolved'}
    rows=load_factors(partition,rigid['support'],dataset,phase_root,rigid['anchor_frame'])
    from .canonical_schur import make_prior,save_state
    map_prior=make_prior(partition,rigid['support'],rigid['anchor_frame'])
    initial=initialize(rigid,partition.xyz[rigid['support']]);candidates={}
    for value in initial:candidates[value['type']]=fit(value,rows,config,feasible,map_prior,rigid.get('map_estimate'),backend)
    decision=select(candidates,rows,config)
    decision['production_camera_indices']=list(range(len(dataset.cameras)))
    json_write(root/'result.json',serial(decision))
    for kind,result in candidates.items():
        json_write(root/f'{kind.lower()}.json',serial({key:value for key,value in result.items() if key not in ('covariance','information','map_state')}))
        if result['status']=='FIT':np.savez_compressed(root/f'{kind.lower()}_uncertainty.npz',covariance=result['covariance'],information=result['information'])
        if 'map_state' in result:save_state(root/f'{kind.lower()}_map_uncertainty.npz',result)
    return {**decision,'fits':candidates,'rows':rows,'map_prior':map_prior,'selected':candidates.get(decision['status'])}
