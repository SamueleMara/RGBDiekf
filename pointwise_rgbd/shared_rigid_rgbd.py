"""One shared SE(3) per frame; Kabsch initialization then actual raw UVZ MLE.

No per-supernode poses or extra motion prior. The per-frame initialization
conditions canonical means; the subsequent motion/map Schur fit retains their
actual measured uncertainty. Calibration and correspondences are conditioned.
"""
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from rgbd_common.array_store import ArrayStore
from .group_partition_validation import weighted_kabsch
from .group_pose import exp,point_jacobian
from .projection import project
from .filter_iekf import inverse_spd
from .free_phase1 import ForegroundConstraint
from .free_phase1 import pack_factors
from .association import associate
from .store import json_write


def prepare(row,point_mask,dataset,backend):
    result=[]
    for ci,camera in enumerate(dataset.cameras):
        use=(row['camera']==ci)&point_mask[row['point_id']]
        if not use.any():continue
        result.append((backend.tensor(row['point_id'][use],torch.long),backend.tensor(row['z'][use]),backend.tensor(row['R'][use]),camera))
    return result


def refine_pose(initial,canonical,factors,config,backend,feasible=None):
    pose=backend.tensor(initial);data=[(ids,z,inverse_spd(r,'rigid raw R'),camera) for ids,z,r,camera in factors]
    def terms(T,info=False):
        value=T.new_zeros(());a=T.new_zeros((6,6));b=T.new_zeros(6);count=0
        for ids,z,ri,camera in data:
            x=canonical[ids]@T[:3,:3].T+T[:3,3];h,j=project(x,camera);e=z-h
            if not bool((torch.isfinite(h).all(1)&(h[:,2]>0)).all()):return T.new_tensor(float('inf')),a,b,count
            value+=.5*torch.einsum('ni,nij,nj->',e,ri,e);count+=len(ids)
            if info:
                H=j[:,:,:3]@point_jacobian(x);W=H.transpose(1,2)@ri
                a+=(W@H).sum(0);b+=(W@e[:,:,None])[...,0].sum(0)
        return value,a,b,count
    history=[];converged=False;mask_rejections=0
    for iteration in range(config.iterations):
        value,a,b,count=terms(pose,True);history.append(float(value))
        if count<3:return {'status':'RIGID_UNIDENTIFIABLE','reason':'Fewer than three raw correspondences','raw_count':count}
        eig=torch.linalg.eigvalsh(a);rank=int((eig>eig[-1]*6*torch.finfo(a.dtype).eps).sum())
        if rank<6:return {'status':'RIGID_UNIDENTIFIABLE','rank':rank,'raw_count':count}
        step=torch.linalg.solve(a,b)
        if float(step.abs().max())<=config.tolerance:converged=True;break
        accepted=False;alpha=1.
        for _ in range(config.line_search_steps):
            trial=exp(alpha*step)@pose;trial_value=terms(trial)[0]
            valid=feasible is None or feasible(trial)
            if not valid:mask_rejections+=1
            if valid and bool(trial_value<value):pose=trial;accepted=True;break
            alpha*=.5
        if not accepted:break
    value,a,b,count=terms(pose,True)
    return {'status':'FIT','pose':pose.cpu().numpy(),'covariance':inverse_spd(a,'rigid sensor information').cpu().numpy(),
        'chi2':float(2*value),'raw_count':count,'energy_history':history+[float(value)],'converged':converged,
        'mask_rejections':mask_rejections,'feasible':bool(feasible(pose)) if feasible else True,'rank':6}


def run(partition,arrays,support,dataset,config,backend,phase_root,root,anchor_frame=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);K=len(dataset.times)
    poses=np.full((K,4,4),np.nan);cov=np.full((K,6,6),np.nan);reports=[]
    canonical=backend.tensor(partition.xyz);ids=np.flatnonzero(support)
    if len(ids)<3:return {'status':'RIGID_UNIDENTIFIABLE','reason':'Fewer than three moving core points'}
    anchor=anchor_frame if anchor_frame is not None else (0 if np.allclose(arrays['xyz'][0],partition.xyz,atol=1e-12,rtol=0) else K-1)
    if anchor not in (0,K-1):raise ValueError('Rigid fitting requires the actual first/last source endpoint')
    refined=ArrayStore(root/'factors.sqlite')
    for k in tqdm(range(K),desc='Shared rigid RGB-D refinement',unit='frame'):
        if k==anchor:poses[k]=np.eye(4);cov[k]=0;reports.append({'frame':k,'status':'CONDITIONED_ENDPOINT','processed_camera_indices':list(range(len(dataset.cameras)))});continue
        weight=1/np.trace(np.asarray(arrays['covariance'][k])[ids,:3,:3],axis1=1,axis2=2)
        initial=weighted_kabsch(partition.xyz[ids],np.asarray(arrays['xyz'][k])[ids],weight)
        frames=[dataset.observation(k,ci) for ci in range(len(dataset.cameras))]
        p=backend.tensor(np.asarray(arrays['covariance'][k])[ids]);member=backend.tensor(np.zeros(len(ids)),torch.long)
        constraint=ForegroundConstraint(frames,p,member,1,config,backend)
        def feasible(T):
            x=canonical[backend.tensor(ids,torch.long)]@T[:3,:3].T+T[:3,3]
            return bool(constraint(x)[0])
        # Free-point associations are not rigid correspondences: after the
        # common pose changes, recompute actual raw matches under that pose.
        # This retains the existing sensor model/finite association family.
        T=initial;pcov=np.asarray(arrays['covariance'][k])[ids].copy();previous=None;outer=[]
        local_ids=backend.tensor(ids,torch.long)
        for iteration in range(config.iterations):
            xyz=partition.xyz[ids]@T[:3,:3].T+T[:3,3];state=np.c_[xyz,np.zeros_like(xyz)];matches=[]
            for ci,frame in enumerate(frames):
                m,_=associate(state,pcov,partition.rgb[ids],ids,frame,config,backend,np.full(frame.depth.size,-1,np.int64))
                m['point']=local_ids[m['point']];matches.append(m)
            factors=[(m['point'],m['z'],m['R'],m['camera']) for m in matches if len(m['point'])]
            fit=refine_pose(T,canonical,factors,config,backend,feasible)
            if fit['status']!='FIT':break
            identity=[(m['camera_index'],tuple(m['point'].cpu().numpy()),tuple(m['pixel'])) for m in matches]
            outer.append({'iteration':iteration,'raw_count':fit['raw_count'],'chi2':fit['chi2'],'fixed_association_converged':fit['converged']})
            T=fit['pose'];xyz=partition.xyz[ids]@T[:3,:3].T+T[:3,3];state=np.c_[xyz,np.zeros_like(xyz)]
            J=point_jacobian(backend.tensor(xyz)).cpu().numpy();pcov[:,:3,:3]=J@fit['covariance']@J.transpose(0,2,1)
            if previous==identity and fit['converged']:break
            previous=identity
        fit['association_outer_history']=outer
        fit['processed_camera_indices']=[frame.ci for frame in frames]
        assert fit['processed_camera_indices']==list(range(len(dataset.cameras)))
        fit['raw_factors_by_camera']={str(m['camera_index']):len(m['point']) for m in matches}
        if fit['status']=='FIT':
            rigid_state=canonical.new_zeros((len(canonical),6));rigid_state[:,:3]=canonical@backend.tensor(T[:3,:3]).T+backend.tensor(T[:3,3])
            refined.write(str(k),**pack_factors(matches,frames,rigid_state))
        if fit['status']=='FIT':poses[k]=fit['pose'];cov[k]=fit['covariance']
        reports.append({'frame':k,**{key:value for key,value in fit.items() if key not in ('pose','covariance')}})
    refined.close()
    np.save(root/'poses.npy',poses);np.save(root/'covariance.npy',cov)
    result={'status':'FIT' if np.isfinite(poses).all() and all(r.get('feasible',True) for r in reports) else 'RIGID_TRAJECTORY_UNRESOLVED','frames':reports,'anchor_frame':anchor,'point_count':len(ids),'camera_policy':'all_available','production_camera_indices':list(range(len(dataset.cameras))),
        'covariance_semantics':'Raw-UVZ local Gaussian pose MLE, shared across all core points, fixed canonical geometry/calibration and correspondences; no additional temporal prior, no claim of independent Phase1 validation'}
    json_write(root/'result.json',result)
    initial={**result,'poses':poses,'covariance':cov,'support':support}
    return refine_map(partition,arrays,initial,dataset,config,backend,root,root)


def refine_map(partition,arrays,initial,dataset,config,backend,factor_root,root):
    """Refine shared poses AND measured canonical geometry with proper priors."""
    from .canonical_schur import make_prior,fit,save_state
    from .joint_rgbd import load_factors
    from .foreground_index import ForegroundIndex
    import shutil
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if initial['status']!='FIT':return initial
    json_write(root/'initial_rigid_fit.json',{key:value for key,value in initial.items() if key not in ('poses','covariance','support','map_estimate','canonical_estimate','map_fit')})
    support=initial['support'];anchor=initial['anchor_frame'];prior=make_prior(partition,support,anchor);ids=prior['keys'][:,1]
    # Exclude only the actual source-prior pixels, never the whole endpoint.
    # Cached non-endpoint associations remain valid initialization inputs.
    source_path=Path(factor_root)/'factors.sqlite';target_path=root/'factors.sqlite'
    if source_path.resolve()!=target_path.resolve():shutil.copyfile(source_path,target_path)
    store=ArrayStore(target_path);frames=[dataset.observation(anchor,ci) for ci in range(len(dataset.cameras))];matches=[]
    state=np.c_[partition.xyz[ids],np.zeros((len(ids),3))];local_ids=backend.tensor(ids,torch.long)
    for ci,frame in enumerate(frames):
        owners=np.full(frame.depth.size,-1,np.int64);source_pixels=partition.source[ids][partition.source[ids,1]==ci,2]
        owners[source_pixels]=np.iinfo(np.int64).max
        m,_=associate(state,np.asarray(arrays['covariance'][anchor])[ids],partition.rgb[ids],ids,frame,config,backend,owners)
        m['point']=local_ids[m['point']];matches.append(m)
    packed=pack_factors(matches,frames,backend.tensor(np.c_[partition.xyz,np.zeros_like(partition.xyz)]))
    refs={tuple(map(int,r)) for r in prior['source_refs']}
    if any((anchor,int(ci),int(pixel)) in refs for ci,pixel in zip(packed['camera'],packed['pixel'])):raise AssertionError('Source prior assimilated twice')
    store.write(str(anchor),**packed);store.close()
    rows=load_factors(partition,support,dataset,root,anchor)
    visibility=ForegroundIndex(dataset,config)
    def feasible(model,mu):
        for k,T in enumerate(model['poses']):
            if not visibility.needs_check(k):continue
            x=mu@T[:3,:3].T+T[:3,3]
            if not visibility.feasible(k,x,np.asarray(arrays['covariance'][k])[ids,:3,:3]):return False
        return True
    result=fit({'type':'RIGID','poses':initial['poses'].copy(),'anchor':anchor},rows,prior,config,backend,feasible=feasible)
    if result['status']!='FIT':
        report={**initial,'status':'RIGID_TRAJECTORY_UNRESOLVED','reason':result.get('reason')};json_write(root/'map_failure.json',{key:value for key,value in report.items() if key not in ('poses','covariance','support')});return report
    poses=result['model']['poses'];pose_cov=np.zeros_like(initial['covariance']);offset=0
    for k in range(len(poses)):
        if k==anchor:continue
        pose_cov[k]=result['covariance'][offset:offset+6,offset:offset+6];offset+=6
    from .joint_rgbd import measurement_score
    complete_score=measurement_score(result,rows)
    save_state(root/'canonical_map_and_motion_uncertainty.npz',result)
    np.save(root/'poses.npy',poses);np.save(root/'covariance.npy',pose_cov)
    canonical=partition.xyz.copy();canonical[ids]=result['map_state']['mu'];np.save(root/'canonical_estimate.npy',canonical)
    # Preserve actual raw references; update residuals to the refined model.
    source=ArrayStore(root/'factors.sqlite');target=ArrayStore(root/'factors.sqlite')
    for k in range(len(poses)):
        row=source.read(str(k));xyz=canonical[row['point_id']]@poses[k,:3,:3].T+poses[k,:3,3];residual=np.empty_like(row['z'])
        for ci,camera in enumerate(dataset.cameras):
            use=row['camera']==ci;pred,_=project(backend.tensor(xyz[use]),camera);residual[use]=row['z'][use]-pred.cpu().numpy()
        target.write(str(k),**{**row,'initial_residual':row.get('initial_residual',row['residual']),'residual':residual})
    source.close();target.close()
    report={key:value for key,value in result.items() if key not in ('model','map_state','covariance','information')}
    report.update(anchor_frame=anchor,point_count=len(ids),complete_multiview=complete_score,camera_policy='all_available',production_camera_indices=list(range(len(dataset.cameras))),canonical_prior='Actual measured endpoint source covariance, used once; source geometry is not treated as exact',
        endpoint_evidence={'prior_raw_count_by_camera':np.bincount(partition.source[ids,1],minlength=len(dataset.cameras)).tolist(),'additional_raw_count_by_camera':np.bincount(packed['camera'],minlength=len(dataset.cameras)).tolist(),'source_prior_raw_pixels_reassimilated':0},
        covariance_semantics='Joint motion/map Schur posterior with shared cross-time pose and pose-point coupling; point-pair covariance never allocated',map_prior_point_count=len(ids))
    json_write(root/'result.json',report)
    return {**report,'poses':poses,'covariance':pose_cov,'support':support,'map_estimate':result['map_state']['mu'],'canonical_estimate':canonical,'map_fit':result}
