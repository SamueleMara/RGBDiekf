"""Time-coupled raw-UVZ conditional split validation with block Gaussian algebra.

Canonical endpoint gauge is eliminated. Forward IEKF supplies a linearization;
one batch Gaussian smoothing step retains temporal covariance. Scores are joint
over held-out frames and pixels within that chosen first-order approximation.
No instantaneous-rank veto, dense global matrix, or test-factor fitting.
"""
import numpy as np
import torch
from rgbd_common.lie import exp,log,inverse,left_jacobian
from rgbd_common.backends.torch_cuda import Backend
from .group_process import initialize,predict,matrices,load_group_noise
from .group_iekf import correct
from .group_pose import point_jacobian
from .projection import project
from .config import load_config


def solve_blocks(diagonal,lower,rhs):
    """SPD block-tridiagonal LDL solve and selected inverse in O(T d^3)."""
    from scipy.linalg import cho_factor,cho_solve
    schur=[];inverses=[];reduced=[];logdet=0.
    for k,d in enumerate(diagonal):
        s=d.copy();r=rhs[k].copy()
        if k:s-=lower[k-1]@inverses[-1]@lower[k-1].T;r-=lower[k-1]@inverses[-1]@reduced[-1]
        s=(s+s.T)/2;root=cho_factor(s,lower=True,check_finite=True);pi=cho_solve(root,np.eye(len(s)))
        schur.append(s);inverses.append(pi);reduced.append(r);logdet+=2*np.log(np.diag(root[0])).sum()
    mean=[None]*len(diagonal);cov=[None]*len(diagonal);cross=[None]*(len(diagonal)-1)
    for k in range(len(diagonal)-1,-1,-1):
        if k==len(diagonal)-1:mean[k]=inverses[k]@reduced[k];cov[k]=inverses[k]
        else:
            gain=-inverses[k]@lower[k].T;mean[k]=inverses[k]@reduced[k]+gain@mean[k+1]
            cov[k]=inverses[k]+gain@cov[k+1]@gain.T;cross[k]=gain@cov[k+1]
    return {'mean':mean,'covariance':cov,'cross_next':cross,'logdet':float(logdet),'precision_schur':schur}


def _measurement_terms(pose,points,z,R,camera):
    b=Backend('cpu');xyz=points@pose[:3,:3].T+pose[:3,3]
    h,j=project(b.tensor(xyz),camera);H=(j[:,:,:3]@point_jacobian(b.tensor(xyz))).numpy()
    return z-h.numpy(),H,np.linalg.inv(R)


def fit_temporal_raw(partition,rows,assignment,groups,train_masks):
    """Filtering/smoothing uses only train_masks; test values are never read."""
    b=Backend('cpu');config=load_config();noise=load_group_noise();first=rows[0]
    times=np.asarray(first['_sequence_times']);anchor=int(first['_anchor_index']);cameras=first['_cameras']
    if anchor not in (0,len(times)-1):raise ValueError('Endpoint gauge required')
    order=list(range(len(times))) if anchor==0 else list(range(len(times)-1,-1,-1))
    member=np.maximum(assignment,0).astype(np.int64);canonical=b.tensor(partition.xyz);membership=b.tensor(member,torch.long)
    row_at=None if hasattr(rows,'by_frame') else {row['frame']:row for row in rows}
    train_at={row['frame']:mask for row,mask in zip(rows,train_masks)}
    def read_row(k):return rows.by_frame(k) if row_at is None else row_at.get(k)
    poses,twist,P=initialize(groups,noise,b);T={anchor:poses.numpy().copy()};V={anchor:twist.numpy().copy()};filtered={anchor:P.numpy().copy()};iteration_limits=0
    for index in range(1,len(order)):
        k=order[index];h=float(times[k]-times[order[index-1]]);poses,twist,P=predict(poses,twist,P,h,noise);matches=[]
        row=read_row(k)
        if row is not None:
            train=train_at[k]
            for ci,camera in enumerate(cameras):
                use=train&(row['camera']==ci);matches.append({'camera':camera,'camera_index':ci,'point':b.tensor(row['point_id'][use],torch.long),'pixel':row['pixel'][use],'z':b.tensor(row['z'][use]),'R':b.tensor(row['R'][use])})
        poses,twist,P,report=correct(poses,twist,P,canonical,membership,matches,config,b);iteration_limits+=int(report['iteration_limit'].sum())
        T[k]=poses.numpy().copy();V[k]=twist.numpy().copy();filtered[k]=P.numpy().copy()
    models=[]
    for group in range(groups):
        d=[np.zeros((6,6))]+[np.zeros((12,12)) for _ in order[1:]];off=[np.zeros((12,6))]+[np.zeros((12,12)) for _ in order[2:]];rhs=[np.zeros(len(x)) for x in d]
        d[0]=np.linalg.inv(noise.initial_twist_covariance);rhs[0]=-d[0]@V[anchor][group]
        # Whole-window information about the initial physical twist, separate
        # from process-noise regularization and posterior precision.
        obs=np.zeros((6,6));sensitivity=np.r_[np.zeros((6,6)),np.eye(6)]
        for index in range(1,len(order)):
            k=order[index];previous=order[index-1];h=float(times[k]-times[previous]);F,Q=matrices(b.tensor(V[previous][group:group+1]),h,noise.acceleration_spectral_density);F=F.numpy()[0];Q=Q.numpy()[0]
            prediction=exp(h*V[previous][group])@T[previous][group]
            e=np.r_[log(T[k][group]@inverse(prediction)),V[k][group]-V[previous][group]]
            B=np.eye(12);B[:6,:6]=np.linalg.inv(left_jacobian(e[:6]))
            A=-F.copy();A[:6]=-np.linalg.inv(left_jacobian(-e[:6]))@F[:6]
            if index==1:A=A[:,6:]
            weight=np.linalg.inv(Q)
            d[index-1]+=A.T@weight@A;d[index]+=B.T@weight@B;off[index-1]+=B.T@weight@A
            rhs[index-1]-=A.T@weight@e;rhs[index]-=B.T@weight@e
            sensitivity=F@sensitivity
            row=read_row(k)
            if row is None:continue
            use=train_at[k]&(member[row['point_id']]==group)
            for ci,camera in enumerate(cameras):
                selected=use&(row['camera']==ci);ids=row['point_id'][selected]
                if not len(ids):continue
                residual,H,w=_measurement_terms(T[k][group],partition.xyz[ids],row['z'][selected],row['R'][selected],camera)
                H12=np.pad(H,((0,0),(0,0),(0,6)));htw=H12.transpose(0,2,1)@w
                d[index]+=np.einsum('nij,njk->ik',htw,H12);rhs[index]+=np.einsum('nij,nj->i',htw,residual)
                initial=H12@sensitivity;obs+=np.einsum('nji,njk,nkl->il',initial,w,initial)
        solved=solve_blocks(d,off,rhs)
        scale=np.linalg.cholesky(noise.initial_twist_covariance);dimensionless=scale.T@obs@scale
        singular=np.linalg.svd(dimensionless,compute_uv=False);tol=6*np.finfo(float).eps*(singular[0] if len(singular) else 0);rank=int((singular>tol).sum())
        smoothed_T=[];smoothed_V=[];smoothed_P=[]
        for index,k in enumerate(order):
            delta=solved['mean'][index]
            if index==0:
                smoothed_T.append(T[k][group]);smoothed_V.append(V[k][group]+delta)
                posterior=np.zeros((12,12));posterior[6:,6:]=solved['covariance'][index];smoothed_P.append(posterior)
            else:
                reset=np.eye(12);reset[:6,:6]=left_jacobian(delta[:6])
                smoothed_T.append(exp(delta[:6])@T[k][group]);smoothed_V.append(V[k][group]+delta[6:])
                smoothed_P.append(reset@solved['covariance'][index]@reset.T)
        models.append({'diagonal':d,'lower':off,'rhs':rhs,'gaussian':solved,'initial_twist_data_rank':rank,'initial_twist_data_information':singular.tolist(),'information_scale':'initial twist prior square root; resulting information is dimensionless','smoothed_poses':np.asarray(smoothed_T),'smoothed_twists':np.asarray(smoothed_V),'smoothed_covariance':np.asarray(smoothed_P)})
    return {'poses':T,'twists':V,'order':order,'models':models,'membership':member,'cameras':cameras,'time_units':'seconds','inner_iteration_limit_count':iteration_limits,'linearization':'forward training-only IEKF plus exact block-tridiagonal Gaussian smoothing step; not exact multimodal or converged nonlinear inference'}


def score_temporal_raw(partition,rows,model,test_masks,selection_frames=None):
    total=0.;chi=0.;count=0;error2=0.;refs=[]
    for group,fit in enumerate(model['models']):
        diag=[x.copy() for x in fit['diagonal']];rhs=[np.zeros(len(x)) for x in diag];square=0.;logR=0.;scalars=0
        position={k:i for i,k in enumerate(model['order'])}
        for row,mask in zip(rows,test_masks):
            k=row['frame'];index=position[k]
            if index==0 or (selection_frames is not None and k not in selection_frames):continue
            use=mask&(model['membership'][row['point_id']]==group)
            for ci,camera in enumerate(model['cameras']):
                selected=use&(row['camera']==ci);ids=row['point_id'][selected]
                if not len(ids):continue
                e,H,w=_measurement_terms(model['poses'][k][group],partition.xyz[ids],row['z'][selected],row['R'][selected],camera)
                H=np.pad(H,((0,0),(0,0),(0,6)));e=e-np.einsum('nij,j->ni',H,fit['gaussian']['mean'][index])
                htw=H.transpose(0,2,1)@w;diag[index]+=np.einsum('nij,njk->ik',htw,H);rhs[index]+=np.einsum('nij,nj->i',htw,e)
                square+=np.einsum('ni,nij,nj->',e,w,e);logR+=np.linalg.slogdet(row['R'][selected])[1].sum();scalars+=3*len(ids)
                refs.extend((k,int(ci),int(p)) for p in row['pixel'][selected]);error2+=float(np.sum(e[:,2]**2))
        posterior=solve_blocks(diag,fit['lower'],rhs)
        q=square-sum(float(r@x) for r,x in zip(rhs,posterior['mean']))
        total+=.5*(q+logR+posterior['logdet']-fit['gaussian']['logdet']+scalars*np.log(2*np.pi));chi+=q;count+=scalars//3
    return {'nll':float(total),'chi2':float(chi),'raw_count':count,'depth_squared_error_m2':error2,'evaluation_refs':refs}
