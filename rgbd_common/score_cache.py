"""Exact incremental score for a declared sparse point/pixel-cell assignment.

Only sparse projected factor records are cached ON DISK, never per-observation
XYZ states or all-frame GPU renders. Edits re-evaluate all changed point
projections, depth competitions and the global 1/N association normalization.
"""
from pathlib import Path
import json,shutil,csv
import numpy as np
import torch
from .geometry import PointMap
from .projection_cuda import transport_and_project
from .observation_model import output,covariance,gaussian_log_probability,unassigned_log_probability,Matches
from .score import priors
from .backends.torch_cuda import numpy
from .progress import progress
from .foreground import check_hard_mask_feasibility


def sorted_records(pixel,point,depth,logf,bg,point_ids=None):
    tie=point if point_ids is None else np.asarray(point_ids)[np.asarray(point,dtype=np.int64)]
    order=np.lexsort((tie,depth,pixel))
    return {key:np.asarray(value)[order] for key,value in [('pixel',pixel),('point',point),('depth',depth),('logf',logf),('bg',bg)]}


def first_rows(records):
    p=records['pixel'];return np.flatnonzero(np.r_[True,p[1:]!=p[:-1]]) if len(p) else np.empty(0,dtype=int)


def table_energy(records,count):
    first=first_rows(records);gain=records['logf'][first]+np.log(records['match_prior']/count)-records['bg'][first]
    return float(-records['background_total']-np.maximum(gain,0).sum())


def records_for_points(model,state,frame,config,backend,indices):
    indices=np.asarray(indices,dtype=np.int64)
    xyz=backend.tensor(model.xyz[indices],backend.input_dtype);rgb=backend.tensor(model.rgb[indices]);group=backend.tensor(model.group[indices],torch.long)
    projection=transport_and_project(xyz,group,backend.tensor(state.poses,backend.input_dtype),frame.camera,backend.tensor(frame.mask,torch.bool))
    local=torch.nonzero(projection['valid'],as_tuple=True)[0]
    pix=projection['pixel'][local,1]*frame.camera.width+projection['pixel'][local,0]
    sampled_pixel=numpy(pix)
    yes=backend.tensor(frame.valid.ravel()[sampled_pixel],torch.bool)
    # Invalid depth creates no factor, including no unassigned/no-return term.
    local=local[yes];pix=pix[yes];sampled_pixel=sampled_pixel[numpy(yes)]
    depth=backend.tensor(np.asarray(frame.depth).ravel()[sampled_pixel]);yes=torch.ones(len(local),dtype=torch.bool,device=backend.device)
    ll=torch.full((len(local),),-torch.inf,dtype=torch.float64,device=backend.device)
    if yes.any():
        selected=local[yes];pixels=pix[yes]
        obs=torch.cat([torch.stack([pixels%frame.camera.width,pixels//frame.camera.width],dim=1),depth[yes,None],backend.tensor(frame.rgb.reshape(-1,3)[numpy(pixels)])],dim=1).double()
        prediction,_,_=output(state,xyz[selected],rgb[selected],group[selected],frame.camera,backend)
        ll[yes]=gaussian_log_probability(obs-prediction,covariance(config,depth[yes],backend))+np.log1p(-config.sensor['no_return'])
    bg=unassigned_log_probability(frame,config,backend,pix)+np.log1p(-config.match_prior)
    if backend.device.type=='cuda':
        point=backend.tensor(indices,torch.long)[local];depth=projection['depth'][local]
        order=torch.arange(len(local),device=backend.device)
        stable_ids=model.point_id[indices]
        if len(indices)>1 and np.any(stable_ids[1:]<stable_ids[:-1]):order=order[torch.argsort(backend.tensor(stable_ids,torch.long)[local],stable=True)]
        order=order[torch.argsort(depth[order],stable=True)];order=order[torch.argsort(pix[order],stable=True)]
        return {key:numpy(value[order]) for key,value in [('pixel',pix),('point',point),('depth',depth),('logf',ll),('bg',bg)]}
    return sorted_records(numpy(pix),indices[numpy(local)],numpy(projection['depth'][local]),numpy(ll),numpy(bg),model.point_id)


class ScoreCache:
    def __init__(self,directory,dataset,config,backend):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.dataset=dataset;self.config=config;self.backend=backend
        self.parent=None;self.proposal=None
        self.point_ids=None
        from .array_store import ArrayStore
        self.store=ArrayStore(self.directory/'score.sqlite')
    def path(self,k,c):return self.directory/f'factor_{k:06d}_{c:03d}.npz'
    def record_mask_violations(self,k,c,model,feasibility):
        ids=numpy(feasibility['violation_indices'])
        if not len(ids):return
        path=self.directory/'mask_constraint_violations.csv';exists=path.exists()
        with path.open('a') as stream:
            writer=csv.writer(stream)
            if not exists:writer.writerow(['frame','camera','point_id','visibility'])
            writer.writerows((k,c,int(model.point_id[i]),'visible') for i in ids)
    def read(self,k,c):
        key=f'factor:{k}:{c}'
        if self.store.contains(key):return self.store.read(key)
        if not self.path(k,c).exists() and self.parent is not None:
            old=self.parent.read(k,c)
            if self.proposal is not None:
                model,states,changed=self.proposal
                new=records_for_points(model,states[k],self.dataset.observation(k,c),self.config,self.backend,changed)
            else:
                with np.load(self.directory/f'delta_{k:06d}_{c:03d}.npz') as delta:
                    changed=delta['changed'];new={key:delta[key] for key in ('pixel','point','depth','logf','bg')}
            keep=~np.isin(old['point'],changed)
            table=sorted_records(*[np.r_[old[key][keep],new[key]] for key in ('pixel','point','depth','logf','bg')],point_ids=self.point_ids)
            table.update(background_total=old['background_total'],match_prior=old['match_prior'])
            return table
        with np.load(self.path(k,c)) as data:return {key:data[key].copy() for key in data.files}
    def build(self,model,states,description='Cache full common point score'):
        self.point_ids=model.point_id.copy()
        total=0.;rows=[];violations=0
        for k in progress(range(len(states)),desc=description,unit='frame'):
            for c in range(len(self.dataset.cameras)):
                frame=self.dataset.observation(k,c);table=records_for_points(model,states[k],frame,self.config,self.backend,np.arange(model.count))
                feasibility=check_hard_mask_feasibility(model,states[k],frame,self.config,self.backend,classify_foreground=False)
                violation_count=len(feasibility['violation_indices']);violations+=violation_count
                self.record_mask_violations(k,c,model,feasibility)
                if hasattr(self.dataset,'background_total'):background=self.dataset.background_total(k,c,self.config,self.backend)
                else:
                    bg=unassigned_log_probability(frame,self.config,self.backend)+np.log1p(-self.config.match_prior)
                    background=float(bg[self.backend.tensor(frame.valid.ravel(),torch.bool)].sum())
                table.update(background_total=background,match_prior=self.config.match_prior)
                self.store.write(f'factor:{k}:{c}',**table);value=table_energy(table,model.count) if not violation_count else float('inf');total+=value;rows.append({'frame':k,'camera':c,'energy':value,'foreground':int(frame.mask.sum()),'foreground_valid':int(frame.valid.sum()),'mask_constraint_violations':violation_count})
        prior=priors(model,states,self.dataset.times,self.config)
        self.score={'energy':float(total+sum(prior.values())),'observation_energy':total,**prior,'frames':rows,'solved_points':model.count,'groups':len(states[0].poses),'mask_constraint_violations':violations}
        return self.score
    def edit(self,new_model,states,changed_indices,directory,description='Exact changed-point score'):
        result=ScoreCache(directory,self.dataset,self.config,self.backend);result.parent=self;total=0.;rows=[];violations=0
        result.point_ids=new_model.point_id.copy()
        changed_indices=np.asarray(changed_indices,dtype=np.int64)
        result.proposal=(new_model,states,changed_indices.copy())
        for k in progress(range(len(states)),desc=description,unit='frame',leave=False):
            for c in range(len(self.dataset.cameras)):
                old=self.read(k,c);removed=np.isin(old['point'],changed_indices)
                frame=self.dataset.observation(k,c)
                feasibility=check_hard_mask_feasibility(new_model,states[k],frame,self.config,self.backend,classify_foreground=False)
                violation_count=len(feasibility['violation_indices']);violations+=violation_count
                result.record_mask_violations(k,c,new_model,feasibility)
                new=records_for_points(new_model,states[k],frame,self.config,self.backend,changed_indices)
                affected=np.unique(np.r_[old['pixel'][removed],new['pixel']])
                lower=np.searchsorted(old['pixel'],affected,'left');upper=np.searchsorted(old['pixel'],affected,'right')
                lengths=upper-lower
                index=np.repeat(lower,lengths)+np.arange(lengths.sum())-np.repeat(np.cumsum(lengths)-lengths,lengths)
                before={key:old[key][index] for key in ('pixel','point','depth','logf','bg')}
                keep=~removed[index]
                after=sorted_records(*[np.r_[before[key][keep],new[key]] for key in ('pixel','point','depth','logf','bg')],point_ids=new_model.point_id)
                before.update(background_total=0.,match_prior=old['match_prior']);after.update(background_total=0.,match_prior=old['match_prior'])
                value=table_energy(old,new_model.count)+table_energy(after,new_model.count)-table_energy(before,new_model.count)
                if violation_count:value=float('inf')
                total+=value;rows.append({'frame':k,'camera':c,'energy':value,'foreground':int(frame.mask.sum()),'foreground_valid':int(frame.valid.sum()),'mask_constraint_violations':violation_count})
        prior=priors(new_model,states,self.dataset.times,self.config)
        result.score={'energy':float(total+sum(prior.values())),'observation_energy':total,**prior,'frames':rows,'solved_points':new_model.count,'groups':len(states[0].poses),'mask_constraint_violations':violations,'incremental_exactness':'All changed point factors, nearest-depth competitions and every old factor normalization re-evaluated; complete new-map hard-mask feasibility checked, unchanged trajectory required'}
        return result
    def refresh_ownership(self,model,ownership):
        for k in range(len(self.dataset.times)):
            for c in range(len(self.dataset.cameras)):
                table=self.read(k,c);first=first_rows(table)
                if self.parent is not None:self.store.write(f'factor:{k}:{c}',**table)
                use=table['logf'][first]+np.log(self.config.match_prior/model.count)>table['bg'][first]
                selected=first[use];ownership.write(k,c,table['pixel'][selected],model.point_id[table['point'][selected]])
                # Disk ownership grid is permitted indexed observation state,
                # not XYZ or a new dynamic variable. Reverse queries map only
                # the requested pixels instead of reading every factor table.
                grid=np.full(self.dataset.cameras[c].width*self.dataset.cameras[c].height,-1,dtype=np.int64)
                grid[table['pixel'][selected]]=table['point'][selected]
                depth=np.full(len(grid),np.inf,dtype=np.float32)
                depth[table['pixel'][first]]=table['depth'][first]
                self.store.write(f'index:{k}:{c}',owner=grid,depth=depth)
        self.parent=None;self.proposal=None  # all accepted tables materialized before old cache retires
    def front_depth_at(self,k,c,pixels):
        """Exact known-map front depth at valid FG pixels, same cache epoch.

        Tables contain every projected candidate on Omega^D, not only matches.
        A distinct ScoreCache is built for every trajectory/map transaction;
        rejected trials cannot mutate these accepted-depth arrays.
        """
        index=self.directory/f'front_depth_{k:06d}_{c:03d}.npy'
        pixels=np.asarray(pixels,dtype=np.int64)
        if self.store.contains(f'index:{k}:{c}'):
            return self.store.read(f'index:{k}:{c}')['depth'][pixels]
        if index.exists():return np.asarray(np.load(index,mmap_mode='r')[pixels])
        table=self.read(k,c);first=first_rows(table);keys=table['pixel'][first]
        positions=np.searchsorted(keys,pixels);ok=positions<len(keys)
        ok[ok]&=keys[positions[ok]]==pixels[ok]
        result=np.full(len(pixels),np.inf,dtype=np.float32);result[ok]=table['depth'][first[positions[ok]]]
        return result
    def owners_at(self,k,c,pixels,model):
        if self.store.contains(f'index:{k}:{c}'):
            return self.store.read(f'index:{k}:{c}')['owner'][np.asarray(pixels,dtype=np.int64)]
        index=self.directory/f'owner_{k:06d}_{c:03d}.npy'
        if index.exists():return np.asarray(np.load(index,mmap_mode='r')[np.asarray(pixels,dtype=np.int64)])
        table=self.read(k,c);first=first_rows(table)
        use=table['logf'][first]+np.log(self.config.match_prior/model.count)>table['bg'][first];selected=first[use]
        keys=table['pixel'][selected];positions=np.searchsorted(keys,pixels);ok=positions<len(keys)
        ok[ok]&=keys[positions[ok]]==np.asarray(pixels)[ok]
        result=np.full(len(pixels),-1,dtype=np.int64);result[ok]=table['point'][selected[positions[ok]]]
        return result
    def propose_group_membership(self,model,states,group):
        """Exact single-point conditional score deltas with all others fixed.

        Parallel edits are proposals, not independent Bayesian marginals. Their
        combined occlusion effects must still pass the complete shared score.
        """
        from .geometry import ObservationIDs
        candidate=model.copy();ids=np.flatnonzero(model.group!=group)
        if not len(ids):return candidate,ids,{'proposed_reassignments':0,'conditional_delta_sum':0.,'family':'no other group assignments'}
        raw=ObservationIDs(self.dataset);source,_,_=raw.decode(model.source_id[ids])
        posebank=np.array([s.poses for s in states]);a=posebank[source,model.group[ids]];b=posebank[source,group]
        world=np.einsum('nij,nj->ni',a[:,:3,:3],model.xyz[ids])+a[:,:3,3]
        candidate.xyz[ids]=np.einsum('nji,nj->ni',b[:,:3,:3],world-b[:,:3,3])
        rotation=b[:,:3,:3].transpose(0,2,1)@a[:,:3,:3]
        jac=np.broadcast_to(np.eye(6),(len(ids),6,6)).copy();jac[:,:3,:3]=rotation
        candidate.covariance[ids]=jac@model.covariance[ids]@jac.transpose(0,2,1);candidate.group[ids]=group
        delta=np.zeros(model.count);logprior=np.log(self.config.match_prior/model.count)
        for k in progress(range(len(states)),desc=f'Conditional rigid membership g={group}',unit='frame',leave=False):
            for c in range(len(self.dataset.cameras)):
                old=self.read(k,c);first=first_rows(old);keys=old['pixel'][first]
                second=first+1;has_second=second<len(old['pixel']);has_second[has_second]&=old['pixel'][second[has_second]]==keys[has_second]
                front_point=old['point'][first];front_depth=old['depth'][first]
                front_cost=-np.maximum(old['logf'][first]+logprior,old['bg'][first])
                next_point=np.full(len(first),np.iinfo(np.int64).max,dtype=np.int64);next_depth=np.full(len(first),np.inf);next_cost=-old['bg'][first]
                if has_second.any():
                    where=second[has_second];next_point[has_second]=old['point'][where];next_depth[has_second]=old['depth'][where]
                    next_cost[has_second]=-np.maximum(old['logf'][where]+logprior,old['bg'][where])
                # Removing a nonfront point changes no raw-pixel factor.
                eligible=np.isin(front_point,ids)
                np.add.at(delta,front_point[eligible],next_cost[eligible]-front_cost[eligible])
                frame=self.dataset.observation(k,c);new=records_for_points(candidate,states[k],frame,self.config,self.backend,ids)
                p=new['point'];pixel=new['pixel'];loc=np.searchsorted(keys,pixel);present=loc<len(keys)
                present[present]&=keys[loc[present]]==pixel[present]
                nearest=np.full(len(p),np.inf);owner=np.full(len(p),np.iinfo(np.int64).max,dtype=np.int64);without=-new['bg'].copy()
                rows=np.flatnonzero(present);oldloc=loc[rows]
                nearest[rows]=front_depth[oldloc];owner[rows]=front_point[oldloc];without[rows]=front_cost[oldloc]
                self_front=owner[rows]==p[rows];affected=rows[self_front];positions=oldloc[self_front]
                nearest[affected]=next_depth[positions];owner[affected]=next_point[positions];without[affected]=next_cost[positions]
                owner_id=np.full(len(owner),np.iinfo(np.int64).max,dtype=np.int64);owned=owner!=np.iinfo(np.int64).max
                owner_id[owned]=model.point_id[owner[owned]]
                wins=(new['depth']<nearest)|((new['depth']==nearest)&(candidate.point_id[p]<owner_id))
                with_new=-np.maximum(new['logf']+logprior,new['bg'])
                np.add.at(delta,p[wins],with_new[wins]-without[wins])
        old_error=np.c_[model.xyz,model.rgb]-self.config.map_mean;new_error=np.c_[candidate.xyz,candidate.rgb]-self.config.map_mean
        precision=np.linalg.inv(self.config.map_covariance)
        delta+=.5*(np.einsum('ni,ij,nj->n',new_error,precision,new_error)-np.einsum('ni,ij,nj->n',old_error,precision,old_error))
        chosen=ids[delta[ids]<0];keep=np.ones(model.count,bool);keep[chosen]=False
        candidate.xyz[keep]=model.xyz[keep];candidate.group[keep]=model.group[keep];candidate.covariance[keep]=model.covariance[keep]
        return candidate,chosen,{'proposed_reassignments':len(chosen),'conditional_delta_sum':float(delta[chosen].sum()),'family':'single-point conditional group/anchor hypotheses; full coupled score acceptance still required'}
    def discard(self):
        self.store.close();shutil.rmtree(self.directory)
