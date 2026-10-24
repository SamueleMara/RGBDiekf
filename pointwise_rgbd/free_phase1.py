"""Endpoint-anchored free point trajectories with canonical grouped increments."""
from pathlib import Path
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from rgbd_common.array_store import ArrayStore
from .surface_supernodes import build_partition
from .state6 import anchor_posterior
from .process6 import predict,transition,process_covariance
from .association import associate
from .point_visibility import project_bins
from .grouped_point_iekf import basis,correct
from .projection import project
from .store import json_write


class ForegroundConstraint:
    """V2 M3 hard nominal first-hit support under the declared acquisition premise.

    Background RGB/depth is not even read. Negative support is undetermined
    when the object-only/no-external-occluder premise is not explicitly true.
    """
    def __init__(self,frames,covariance,membership,groups,config,backend):
        self.frames=frames;self.p=covariance;self.member=membership;self.groups=groups;self.config=config;self.backend=backend
        self.images=[backend.tensor(f.mask,torch.bool).flatten() for f in frames]

    def violations(self,state):
        bad=torch.zeros(len(state),dtype=torch.bool,device=state.device)
        for frame,mask in zip(self.frames,self.images):
            if frame.object_only_no_external_occluders is not True:continue
            projection,bins=project_bins(state,frame.camera,self.backend)
            ids=bins['visible_ids'];pixel=projection['pixel'][ids];linear=pixel[:,1]*frame.camera.width+pixel[:,0]
            bad[ids]|=~mask[linear]
        return bad

    def __call__(self,state):
        bad=self.violations(state);count=torch.zeros(self.groups,dtype=torch.long,device=state.device)
        count.scatter_add_(0,self.member,bad.long())
        return count==0


def pack_factors(matches,frames,state):
    fields={name:[] for name in ('point_id','camera','pixel','z','R','rgb','residual')}
    for m in matches:
        ids=m['point'].cpu().numpy();ci=m['camera_index'];pixels=m['pixel']
        if not frames[ci].valid.ravel()[pixels].all():raise AssertionError('Background entered a factor')
        h,_=project(state[m['point']],m['camera'])
        for name,value in dict(point_id=ids,camera=np.full(len(ids),ci,np.int32),pixel=pixels,
            z=m['z'].cpu().numpy(),R=m['R'].cpu().numpy(),rgb=frames[ci].rgb.reshape(-1,3)[pixels],residual=m['z'].cpu().numpy()-h.cpu().numpy()).items():fields[name].append(value)
    return {name:np.concatenate(parts) for name,parts in fields.items()}


def run(dataset,config,backend,root,anchor='first',occupancy=64,prefix=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    k0,partition=build_partition(dataset,anchor,config,occupancy);partition.save(root/'partition.npz')
    state,p=anchor_posterior(partition.xyz,partition.source_covariance,config.velocity_covariance)
    n=len(state);K=len(dataset.times);S=partition.count
    arrays={}
    for name,shape,dtype in [('state',(K,n,6),'float64'),('covariance',(K,n,6,6),'float64'),('point_observed',(K,n),'bool')]:
        arrays[name]=np.lib.format.open_memmap(root/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
    arrays['xyz']=arrays['state'][...,:3]
    member=backend.tensor(partition.membership,torch.long);g=basis(backend.tensor(partition.xyz),member,backend.tensor(partition.centroids),normalize=True)
    np.save(root/'correction_basis.npy',g.cpu().numpy())
    from .grouped_covariance import CovarianceOperator
    covariance_operator=CovarianceOperator(p,g,partition.membership,backend,config.tolerance)
    store=ArrayStore(root/'factors.sqlite')
    coupling_store=ArrayStore(root/'cross_point_coupling.sqlite')
    order=list(range(K)) if anchor=='first' else list(range(K-1,-1,-1))
    arrays['state'][k0]=state;arrays['covariance'][k0]=p;arrays['point_observed'][k0]=True
    timing=[];start=time.monotonic()
    json_write(root/'run.json',{'state':'individual XYZ + physical-time velocity, no group pose state','anchor':anchor,'anchor_frame':k0,'order':order,'timestamps_seconds':dataset.times.tolist(),'groups':S,'points':n,'occupancy':occupancy,'visibility_model':'existing pointwise measured-foreground-depth association; background never read',
        'covariance':'Joseph marginals plus closed rank-24 sparse/low-rank covariance operator retaining all cross-point correlations under the chosen linearized restricted estimator. No block assumed-density projection. Canonical G fixed. Endpoint sensor uncertainty retained; QR/Cholesky solve accuracy explicitly checked; normalized canonical G saved.',
        'camera_policy':'all_available', 'production_camera_indices':list(range(len(dataset.cameras))),
        'mask_premise':[dataset.observation(k0,ci).object_only_no_external_occluders for ci in range(len(dataset.cameras))],
        'negative_mask_support':'Hard only under an explicit object-only/no-external-occluder premise; otherwise undetermined. No background depth or RGB is used.',
        'point_status':'Direct measurement support distinguished from indirect group correction and prediction; unobserved does not imply unchanged.',
        'association_tail':config.association_tail_probability,'independent_noise_assumption':'sensor errors independent between distinct raw pixels/cameras/times conditional fixed calibration; correspondence selection remains conditioned; not calibrated'})
    imported=0
    if prefix is not None and (Path(prefix)/'run.json').exists():
        import json,shutil
        prefix=Path(prefix);old=json.loads((prefix/'run.json').read_text())
        if old.get('camera_policy')!='all_available' or old.get('production_camera_indices')!=list(range(len(dataset.cameras))):raise ValueError('Prefix used a camera partition; all-camera filtering must be rerun')
        if old.get('visibility_model')!='existing pointwise measured-foreground-depth association; background never read':raise ValueError('Prefix used superseded visibility/association semantics')
        if old['state']!='individual XYZ + physical-time velocity, no group pose state' or old['anchor']!=anchor or old['occupancy']!=occupancy or old['timestamps_seconds']!=dataset.times.tolist():raise ValueError('Incompatible phase1 prefix')
        with np.load(prefix/'partition.npz') as saved:
            if str(saved['partition_hash'])!=partition.digest():raise ValueError('Prefix canonical point/group catalogue changed')
        saved_g=np.load(prefix/'correction_basis.npy')
        if not np.array_equal(saved_g,g.cpu().numpy()):raise ValueError('Prefix correction basis changed')
        histories={name:np.load(prefix/f'{name}.npy',mmap_mode='r') for name in ('state','covariance','point_observed')}
        old_store=ArrayStore(prefix/'factors.sqlite');old_cross=ArrayStore(prefix/'cross_point_coupling.sqlite')
        for k in order[1:]:
            report_path=prefix/f'filter/{k:06d}.json'
            if not report_path.exists():break
            values=old_cross.read(str(k))
            if 'cross_factor' not in values:raise ValueError('Prefix did not retain the closed covariance operator')
            for name in histories:arrays[name][k]=histories[name][k]
            store.write(str(k),**old_store.read(str(k)))
            coupling_store.write(str(k),**values)
            covariance_operator.prediction(values['prediction_F'],values['prediction_Q'])
            covariance_operator.history.append({'F':values['prediction_F'],'Q':values['prediction_Q']})
            (root/'filter').mkdir(exist_ok=True);shutil.copyfile(report_path,root/f'filter/{k:06d}.json')
            record=json.loads(report_path.read_text());timing.append({key:record[key] for key in ('frame','seconds','directly_observed','prediction_only','actual_training_factors','remaining_mask_violations')});imported+=1
        if imported:
            k=order[imported];state=np.array(arrays['state'][k]);p=np.array(arrays['covariance'][k])
            covariance_operator.V=backend.tensor(values['cross_factor']);covariance_operator.C=backend.tensor(values['group_core'])
            json_write(root/'reused_prefix.json',{'source':str(prefix.resolve()),'imported_steps':imported,'last_frame':k,'previous_states_reoptimized':False,'preserved_original_files':True,'current_solve_relative_tolerance':config.tolerance})
        for source in (old_store,old_cross):source.close()
    for previous,k in tqdm(list(zip(order,order[1:]))[imported:],desc=f'Free grouped point IEKF ({anchor})',unit='frame',initial=imported,total=K-1):
        clock=time.monotonic();frames=[dataset.observation(k,ci) for ci in range(len(dataset.cameras))]
        dt=abs(float(dataset.times[k]-dataset.times[previous]));reverse=k<previous
        prior,pp=predict(backend.tensor(state),backend.tensor(p),dt,config,reverse)
        covariance_operator.prediction(transition(dt,reverse),process_covariance(dt,config.qc,config.qx,reverse))
        prediction=prior.cpu().numpy();predcov=pp.cpu().numpy();matches=[];statistics=[]
        for ci,frame in enumerate(frames):
            match,stats=associate(prediction,predcov,partition.rgb,np.arange(n),frame,config,backend,np.full(frame.depth.size,-1,np.int64))
            matches.append(match);statistics.append({**{key:value for key,value in stats.items() if not isinstance(value,list)},'camera_index':ci,'mask_positive_valid_observations':int(frame.valid.sum())})
        assert [r['camera_index'] for r in statistics]==list(range(len(dataset.cameras))), 'Every available camera must participate in each joint update'
        constraint=ForegroundConstraint(frames,pp,member,S,config,backend)
        updated,posterior,coupling,report=correct(prior,pp,matches,g,member,config,constraint,covariance_operator)
        blocked=report['mask_blocked_groups'][member]
        if bool(blocked.any()):
            for match in matches:
                keep=~blocked[match['point']]
                for key in ('point','z','R'):match[key]=match[key][keep]
                match['pixel']=match['pixel'][keep.cpu().numpy()]
        inverse_check=torch.linalg.cholesky_ex(posterior)[1]
        if bool((inverse_check!=0).any()):raise ArithmeticError('Non-SPD grouped Joseph point marginal; no jitter repair')
        state=updated.cpu().numpy();p=posterior.cpu().numpy();observed=report['measurement_count'].cpu().numpy()>0
        arrays['state'][k]=state;arrays['covariance'][k]=p;arrays['point_observed'][k]=observed
        store.write(str(k),**pack_factors(matches,frames,updated))
        coupling_store.write(str(k),**{key:value.cpu().numpy() for key,value in coupling.items()})
        violations=int(constraint.violations(updated).sum())
        record={'frame':k,'original_frame':dataset.frame_ids[k],'signed_dt_seconds':float(dataset.times[k]-dataset.times[previous]),'directly_observed':int(observed.sum()),'prediction_only':int((~observed).sum()),'actual_training_factors':sum(len(m['point']) for m in matches),'seconds':time.monotonic()-clock,'remaining_mask_violations':violations,'association':statistics}
        record.update({key:(value.cpu().numpy().tolist() if isinstance(value,torch.Tensor) else value.tolist() if isinstance(value,np.ndarray) else value) for key,value in report.items()})
        json_write(root/f'filter/{k:06d}.json',record);timing.append({key:record[key] for key in ('frame','seconds','directly_observed','prediction_only','actual_training_factors','remaining_mask_violations')})
        for value in arrays.values():
            if hasattr(value,'flush'):value.flush()
    for saved in (store,coupling_store):saved.close()
    json_write(root/'summary.json',{'frames':K,'cameras':len(dataset.cameras),'points':n,'groups':S,'runtime_seconds':time.monotonic()-start,'frame_diagnostics':timing,'camera_policy':'all_available','production_camera_indices':list(range(len(dataset.cameras))),'physical_rigidity_imposed':False,'persistent_group_poses':False,'cross_point_projection':'NONE: closed rank-24 chosen linearized covariance operator'})
    return partition,arrays
