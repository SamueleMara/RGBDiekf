"""Shared explicit forward / own smoother / reverse / chronological frontier loop."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
from .lie import State
from .geometry import RawObservationOwnership,ObservationIDs
from .fusion import fuse_frame_zero,refuse_original_factors
from .group_proposals import bootstrap_groups
from .discovery import reverse_match_unsolved
from .new_group import new_group_proposals
from .score import frame_score
from .score_cache import ScoreCache
from .measurement_problem import MeasurementProblem
from .estimator import Step
from .progress import progress
from .projection_cuda import PROJECTION_OPERATOR_VERSION
from .foreground import HARD_MASK_MODEL_VERSION


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda value:value.tolist() if isinstance(value,np.ndarray) else str(value))+'\n')


def model_digest(model):
    digest=hashlib.sha256()
    for value in (model.xyz,model.rgb,model.group,model.source_id):digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def same_estimate(first,second,first_states,second_states):
    if first is not second and any(not np.array_equal(getattr(first,key),getattr(second,key)) for key in ('xyz','rgb','group','point_id')):return False
    if first_states is second_states:return True
    return len(first_states)==len(second_states) and all(np.array_equal(a.poses,b.poses) and np.array_equal(a.twists,b.twists) for a,b in zip(first_states,second_states))


class StageLog:
    def __init__(self,directory,estimator,backend,resume=False):
        self.directory=Path(directory);self.path=self.directory/'stage_log.jsonl';self.start=time.monotonic();self.estimator=estimator;self.backend=backend;self.records=[]
        self.debug=False
        if resume and self.path.exists():
            self.records=[json.loads(line) for line in self.path.read_text().splitlines()]
            if self.records:self.start-=self.records[-1]['elapsed_seconds']
    def log(self,stage,**values):
        record={'stage':stage,'gpu':self.backend.memory(),'score_epoch':PROJECTION_OPERATOR_VERSION,'pipeline':self.estimator,'elapsed_seconds':time.monotonic()-self.start,'rss_peak_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,**values}
        with self.path.open('a') as stream:stream.write(json.dumps(record,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');stream.flush()
        self.records.append(record);write_json(self.directory/'live_progress.json',record)
        return record


def save_steps(directory,steps,smoothed=None):
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    arrays={}
    for name in ('predicted','filtered'):
        arrays[name+'_poses']=np.array([getattr(s,name).poses for s in steps]);arrays[name+'_twists']=np.array([getattr(s,name).twists for s in steps])
        arrays[name+'_covariance']=np.array([getattr(s,name+'_covariance') for s in steps])
    arrays['transitions']=np.array([s.transition for s in steps[1:]]) if steps[1].transition is not None else np.empty(0)
    arrays['sigma_cross_covariances']=np.array([s.cross_covariance for s in steps[1:]]) if steps[1].cross_covariance is not None else np.empty(0)
    if smoothed is not None:
        states,cov,report=smoothed;arrays.update(smoothed_poses=np.array([s.poses for s in states]),smoothed_twists=np.array([s.twists for s in states]),smoothed_covariance=np.array(cov),gains=np.array(report['gains']))
    np.savez(directory,**arrays)


def forward_filter(estimator,model,nominal,dataset,config,backend,directory,log,label,live=False):
    directory=Path(directory)
    if log.debug:directory.mkdir(parents=True,exist_ok=True)
    if log.debug:model.save(directory/'conditional_filter_points.npz')
    groups=len(nominal[0].poses);initial=State(np.tile(np.eye(4),(groups,1,1)),np.tile(config.twist_mean,(groups,1)))
    steps=[]
    for k in progress(range(len(dataset.times)),desc=estimator.name+' '+label+' forward',unit='frame'):
        if k==0:predicted=initial;pminus=config.p0(groups);extra={}
        else:predicted,pminus,extra=estimator.predict(steps[-1].filtered,steps[-1].filtered_covariance,dataset.times[k]-dataset.times[k-1])
        base_value,matches,_=frame_score(model,predicted,dataset,k,config,backend)
        selected='prediction'
        if k and nominal is not None:
            alternative,other,_=frame_score(model,nominal[k],dataset,k,config,backend)
            d=nominal[k].minus(predicted)
            if alternative+.5*d@np.linalg.solve(pminus,d)<base_value:matches=other;selected='external measured/previous nominal'
        problem=MeasurementProblem(model,matches,backend,ObservationIDs(dataset))
        filtered,pplus,diagnostics=estimator.correct(predicted,pminus,problem,reference=k==0,nominal=nominal[k])
        diagnostics.update(frame=k,matched_factors=sum(map(len,problem.matches)),excluded_initialization_self_factors=problem.excluded_source_factors,association_initializer=selected,**{key:value for key,value in extra.items() if key not in ('transition','cross_covariance','injection')})
        if 'injection' in extra:diagnostics['process_injection']=extra['injection'].tolist()
        steps.append(Step(predicted,pminus,filtered,pplus,extra.get('transition'),extra.get('cross_covariance'),diagnostics))
        # Compact immutable factor IDs permit exact replay; they are not point
        # states, copied XYZ or a second posterior/prior.
        from .backends.torch_cuda import numpy
        if log.debug:
            for match in problem.matches:
                np.savez(directory/f'factor_{k:06d}_{match.camera_index:03d}.npz',point_id=model.point_id[numpy(match.point)],pixel=numpy(match.pixel))
            write_json(directory/f'step_{k:06d}.json',{'predicted':{'poses':predicted.poses,'twists':predicted.twists},'filtered':{'poses':filtered.poses,'twists':filtered.twists},'predicted_covariance':pminus,'filtered_covariance':pplus,'diagnostics':diagnostics})
        log.log('forward-frame',label=label,frame=k,frames=len(dataset.times),points=model.count,groups=groups,matched=diagnostics['matched_factors'])
        if live and (k%20==0 or k==len(dataset.times)-1):
            from .outputs import live_view
            live_view(log.directory,model,[step.filtered for step in steps],label,k,log.records)
    log.log('forward-complete',label=label,frames=len(steps),cameras=len(dataset.cameras),points=model.count,groups=groups)
    if log.debug:save_steps(directory/'forward.npz',steps)
    return steps


@dataclass
class Result:
    model:object
    states:list
    covariance:object
    steps:list
    smooth:object
    cache:object
    ownership:object
    history:list
    log:object
    status:str
    initial_points:int
    initial_digest:str


def run_outer(estimator,dataset,config,backend,directory,max_outer=None,time_budget=None,visual=False,resume=False,rebase_legacy_projection=False):
    from .model_contract import require_model_selection
    require_model_selection(config)
    directory=Path(directory);reports=directory/'reports';reports.mkdir(parents=True,exist_ok=True)
    log=StageLog(reports,estimator.name,backend,resume);ownership=RawObservationOwnership(dataset,directory/'ownership')
    log.debug=visual
    offset=0;history=[]
    if resume:
        from .geometry import PointMap
        checkpoint=json.loads((directory/'checkpoint.json').read_text())
        current_payload=directory/'checkpoint_state.npz'
        if current_payload.exists():
            with np.load(current_payload) as saved:
                embedded=json.loads(str(saved['checkpoint_metadata']))
            if embedded.get('generation')!=checkpoint.get('generation'):checkpoint=embedded
        model=PointMap.load(directory/checkpoint.get('point_file','checkpoint_points.npz'))
        if checkpoint.get('hard_mask_model_version')!=HARD_MASK_MODEL_VERSION:raise ValueError('CHECKPOINT_MASK_CONTRACT_CHANGED: v2 excludes all background depth and invalid-depth factors; preserve earlier results and start a fresh experiment')
        with np.load(directory/checkpoint.get('trajectory_file','checkpoint_trajectory.npz')) as trajectory:states=[State(p,v) for p,v in zip(trajectory['poses'],trajectory['twists'])]
        if len(states)!=len(dataset.times):raise ValueError('CHECKPOINT_DATASET_CHANGED: number of times differs')
        initial=json.loads((reports/'initial_fusion.json').read_text())['points']
        digest=next(r['initialization_sha256'] for r in log.records if r['stage']=='initialization-complete')
        offset=1+max((r.get('outer',-1) for r in log.records),default=-1)
        if (reports/'objective_history.json').exists():history=json.loads((reports/'objective_history.json').read_text())
        log.log('resume-accepted-checkpoint',outer=offset,energy=checkpoint['energy'],points=model.count,original_priors_unchanged=True,rule='New complete forward from original P0; no old posterior reused as a prior; incomplete discovery is revisited')
    else:
        model,records=fuse_frame_zero(dataset,config,backend);initial=model.count
        write_json(reports/'initial_fusion.json',{'points':initial,'bytes':model.nbytes,'cameras':records})
        states,bootstrap=bootstrap_groups(model,dataset,config,backend)
        write_json(reports/'group_bootstrap.json',bootstrap);digest=model_digest(model)
        if log.debug:model.save(directory/'initial_points.npz')
        log.log('initialization-complete',points=model.count,map_bytes=model.nbytes,groups=len(states[0].poses),initialization_sha256=digest)
    cache=ScoreCache(directory/(f'score_cache_resume_{offset:03d}' if resume else 'score_cache_current'),dataset,config,backend);current=cache.build(model,states)
    if not np.isfinite(current['energy']):
        log.log('initial-feasibility-restoration',violations=current['mask_constraint_violations'],method='Data-driven bootstrap correspondences followed by the selected filter and own smoother; no geometry pruning or mask bypass')
        repair=forward_filter(estimator,model,states,dataset,config,backend,directory/'initial_feasibility/filter',log,'initial-feasibility-restoration',visual)
        smoothed_repair=estimator.smooth(repair);save_steps(directory/'initial_feasibility/forward_smooth.npz',repair,smoothed_repair)
        restored=False
        for label,proposed in [('filtered',[step.filtered for step in repair]),('smoothed',smoothed_repair[0])]:
            trial=ScoreCache(directory/f'initial_feasibility/score_{label}',dataset,config,backend);score=trial.build(model,proposed,'Full feasibility-restoration score')
            log.log('initial-feasibility-proposal',kind=label,accepted=bool(np.isfinite(score['energy'])),violations=score['mask_constraint_violations'])
            if np.isfinite(score['energy']):cache.discard();cache=trial;current=score;states=proposed;restored=True;break
            trial.discard()
        if not restored:
            write_json(reports/'initial_mask_feasibility.json',{'status':'FEASIBILITY_RESTORATION_EXHAUSTED','violations':current['mask_constraint_violations'],'attempted':'measured bootstrap, full selected-filter sweep and own smoother; neither yielded a feasible full-sequence trajectory','by_frame_camera':[{key:row[key] for key in ('frame','camera','mask_constraint_violations')} for row in current['frames']]})
            raise ValueError('FEASIBILITY_RESTORATION_EXHAUSTED: measured motion hypotheses remain inconsistent with the declared object-only masks; see reports/initial_mask_feasibility.json. No accepted reconstruction or geometry pruning.')
    if resume:
        legacy=checkpoint.get('projection_operator_version',1)!=PROJECTION_OPERATOR_VERSION
        if legacy:
            if not rebase_legacy_projection:raise ValueError('CHECKPOINT_NUMERICAL_OPERATOR_CHANGED: legacy batch-dependent float32 projection; explicit --rebase-legacy-projection-score is required. Legacy scores are not comparable to the corrected operator.')
            write_json(reports/f'legacy_checkpoint_before_projection_v{PROJECTION_OPERATOR_VERSION}.json',checkpoint)
            write_json(reports/f'legacy_history_before_projection_v{PROJECTION_OPERATOR_VERSION}.json',history);history=[]
            log.log('projection-score-rebase',legacy_energy=checkpoint['energy'],corrected_energy=current['energy'],difference=current['energy']-checkpoint['energy'],legacy_scores_validated=False,reason='Fixed per-coordinate projection contraction order eliminates batch-size-dependent discrete pixel changes; start a new explicitly identified numerical score epoch')
            checkpoint.update(energy=current['energy'],projection_operator_version=PROJECTION_OPERATOR_VERSION,score_cache=str(cache.directory))
            write_json(directory/'checkpoint.pending.json',checkpoint);(directory/'checkpoint.pending.json').replace(directory/'checkpoint.json')
        elif not np.isclose(current['energy'],checkpoint['energy'],rtol=1e-12,atol=1e-5):raise ValueError(f"CHECKPOINT_SCORE_CHANGED: {checkpoint['energy']} vs {current['energy']}")
    cache.refresh_ownership(model,ownership)
    counters=[int(p.name.rsplit('_',1)[-1]) for p in directory.glob('score_cache_proposal_*')]
    counters += [int(p.stem.rsplit('_',1)[-1]) for p in directory.glob('checkpoint_points_[0-9]*.npz')]
    counter=max(counters+[0]);status='NON_CONVERGED';last_steps=None;smoothed=None;covariance=None
    outer_limit=config.numerical['outer_iterations'] if max_outer is None else max_outer
    def candidate_cache(new_model,new_states,changed=None):
        nonlocal counter
        counter+=1;path=directory/f'score_cache_proposal_{counter:06d}'
        if changed is None:
            result=ScoreCache(path,dataset,config,backend);result.build(new_model,new_states,'Full trajectory/group proposal score');return result
        return cache.edit(new_model,new_states,changed,path)
    def commit(new_model,new_states,candidate):
        nonlocal model,states,cache,current
        previous=cache;model=new_model;states=new_states;cache=candidate;current=cache.score
        cache.refresh_ownership(model,ownership)
        point_file=trajectory_file='checkpoint_state.npz'
        metadata={'point_file':point_file,'trajectory_file':trajectory_file,'score_cache':str(cache.directory),'generation':time.time_ns(),'energy':current['energy'],'points':model.count,'groups':len(states[0].poses),'original_priors_unchanged':True,'projection_operator_version':PROJECTION_OPERATOR_VERSION,'hard_mask_model_version':HARD_MASK_MODEL_VERSION}
        model.save(directory/'checkpoint_state.pending.npz',poses=np.array([s.poses for s in states]),twists=np.array([s.twists for s in states]),checkpoint_metadata=np.array(json.dumps(metadata)))
        (directory/'checkpoint_state.pending.npz').replace(directory/point_file)
        write_json(directory/'checkpoint.pending.json',metadata)
        (directory/'checkpoint.pending.json').replace(directory/'checkpoint.json')
        previous.discard()
        if visual:
            from .outputs import live_view
            live_view(reports,model,states,"Accepted map and motion",len(states)-1,log.records)
    def accept(new_model,new_states,candidate,stage,**details):
        changed=not same_estimate(model,new_model,states,new_states)
        accepted=bool(changed and np.isfinite(candidate.score['energy']) and candidate.score['energy']<current['energy'])
        violation_file=candidate.directory/'mask_constraint_violations.csv'
        if violation_file.exists():
            import csv
            saved=directory/'mask_constraint_violations.csv';exists=saved.exists()
            with violation_file.open() as source,saved.open('a') as target:
                reader=csv.DictReader(source);writer=csv.DictWriter(target,fieldnames=['frame','camera','point_id','pose_stage','visibility'])
                if not exists:writer.writeheader()
                for row in reader:writer.writerow({**row,'pose_stage':stage})
        log.log(stage,**{'accepted':accepted,'candidate_estimate_changed':changed,'old_energy':current['energy'],'proposal_energy':candidate.score['energy'] if np.isfinite(candidate.score['energy']) else None,'mask_constraint_violations':candidate.score.get('mask_constraint_violations',0),'points':new_model.count,'groups':len(new_states[0].poses),**details})
        if accepted:commit(new_model,new_states,candidate)
        else:candidate.discard()
        return accepted
    def local_frontier_refine(s,steps):
        # This is an explicitly LOCAL conditional relinearization. The original
        # saved prediction is reused, never the old posterior as a new prior.
        # Other times retain their prior linearizations until the mandatory
        # subsequent full sweep on the revised map.
        if len(steps)<=s or len(steps[s].predicted.poses)!=len(states[0].poses):return steps,False
        _,matches,_=frame_score(model,states[s],dataset,s,config,backend)
        problem=MeasurementProblem(model,matches,backend,ObservationIDs(dataset))
        old=steps[s];filtered,pplus,diagnostics=estimator.correct(old.predicted,old.predicted_covariance,problem,reference=s==0,nominal=states[s])
        revised=steps[:s]+[Step(old.predicted,old.predicted_covariance,filtered,pplus,old.transition,old.cross_covariance,diagnostics)]
        prefix=estimator.smooth(revised);proposed=prefix[0]+states[s+1:]
        candidate=candidate_cache(model,proposed)
        accepted=accept(model,proposed,candidate,'frontier-local-motion',frame=s,approximation='original-prior prefix correction and own smoother; full new-map sweep required for final conditional covariance')
        return (revised+steps[s+1:] if accepted else steps),accepted
    for outer in range(offset,offset+outer_limit):
        before=current['energy'];edits=0
        log.log('outer-start',outer=outer,energy=before,points=model.count,groups=len(states[0].poses))
        phase=directory/f'outer_{outer:03d}'
        if log.debug:model.save(directory/f'outer_{outer:03d}_filter_points.npz')
        steps=forward_filter(estimator,model,states,dataset,config,backend,phase/'filter',log,f'outer-{outer}',visual)
        smoothed=estimator.smooth(steps)
        if log.debug:save_steps(phase/'filter/forward_smooth.npz',steps,smoothed)
        changed=sum(np.linalg.norm(a.minus(b.filtered))>1e-10 for a,b in zip(smoothed[0],steps))
        log.log('smoother-complete',outer=outer,frames=len(states),changed_frames=changed,method=smoothed[2]['method'])
        candidate=candidate_cache(model,smoothed[0])
        if accept(model,smoothed[0],candidate,'motion-stage',outer=outer):edits+=1;covariance=smoothed[1]
        last_steps=steps
        # Revisit provisional memberships among EXISTING learned trajectories,
        # not only when a new group is born. Parallel conditional changes are
        # still coupled proposals and must pass the identical complete score.
        membership_edits=False
        for target_group in range(len(states[0].poses)):
            if target_group>=len(states[0].poses):break
            revised,changed_indices,details=cache.propose_group_membership(model,states,target_group)
            if not len(changed_indices):continue
            active=np.unique(revised.group);revised.group=np.searchsorted(active,revised.group).astype(np.int32)
            nominal=[State(s.poses[active],s.twists[active]) for s in states]
            candidate=candidate_cache(revised,nominal)
            if accept(revised,nominal,candidate,'existing-group-membership',outer=outer,target_group=target_group,**details):
                edits+=1;membership_edits=True
        # One rigid-motion M-step follows the categorical proposal sweep.
        # Every label edit already passed the same complete score; no group
        # posterior is treated as independent or reused as the next prior.
        if membership_edits:
            replayed=forward_filter(estimator,model,states,dataset,config,backend,phase/'membership_replay',log,'existing-group-refinement',False)
            refined=estimator.smooth(replayed)
            if log.debug:save_steps(phase/'membership_replay/forward_smooth.npz',replayed,refined)
            log.log('smoother-complete',outer=outer,label='existing-group-refinement',frames=len(states),method=refined[2]['method'])
            candidate=candidate_cache(model,refined[0])
            if accept(model,refined[0],candidate,'existing-group-motion',outer=outer):covariance=refined[1];edits+=1;steps=replayed
        for stage,frontiers in [('reverse-discovery',range(len(states)-1,0,-1)),('chronological-frontier',range(1,len(states)))]:
            for s in progress(frontiers,desc=estimator.name+' '+stage,unit='frontier'):
                frame_edits=0
                for ci in range(len(dataset.cameras)):
                    frame=dataset.observation(s,ci);unresolved=ownership.unsolved(s,ci,frame)
                    proposal,details=reverse_match_unsolved(model,states,dataset,s,ci,unresolved,config,backend,cache,target_frames=range(s-1,-1,-1))
                    log.log('backward-match',outer=outer,pass_kind=stage,**details)
                    if proposal is not None:
                        new_model,changed_indices=proposal;candidate=candidate_cache(new_model,states,changed_indices)
                        if accept(new_model,states,candidate,'geometry-proposal',outer=outer,pass_kind=stage,source_frame=s,camera=ci):edits+=1;frame_edits+=1
                    # No fixed group count: unexplained coherent multiframe
                    # tracks may propose a new group or split/merge affiliations.
                    remaining=ownership.unsolved(s,ci,frame)
                    for group_proposal,details in new_group_proposals(model,states,dataset,s,ci,remaining,config,backend):
                        log.log('group-discovery-attempt',**{'outer':outer,'pass_kind':stage,'frame':s,'camera':ci,**details})
                        if group_proposal is not None:
                            new_model,nominal,changed_indices=group_proposal
                            membership_cache=candidate_cache(new_model,nominal)
                            revised,_,membership_report=membership_cache.propose_group_membership(new_model,nominal,len(nominal[0].poses)-1)
                            new_model=revised;membership_cache.discard()
                            active=np.unique(new_model.group);new_model.group=np.searchsorted(active,new_model.group).astype(np.int32)
                            nominal=[State(state.poses[active],state.twists[active]) for state in nominal]
                            log.log('conditional-group-membership',outer=outer,source_frame=s,**membership_report)
                            replayed=[]
                            def replay():
                                replayed[:]=forward_filter(estimator,new_model,nominal,dataset,config,backend,phase/f"group_replay_{s:04d}_{ci:03d}_{details['feature_horizon']}",log,'reverse-group-refinement',False)
                                return replayed
                            refinement=estimator.refine_reverse_hypothesis(replay)
                            if log.debug:save_steps(phase/f"group_replay_{s:04d}_{ci:03d}_{details['feature_horizon']}"/'forward_smooth.npz',replayed,refinement)
                            log.log('smoother-complete',outer=outer,label='reverse-group-refinement',frames=len(states),method=refinement[2]['method'])
                            candidate=candidate_cache(new_model,refinement[0])
                            if accept(new_model,refinement[0],candidate,'group-proposal',outer=outer,pass_kind=stage,source_frame=s,camera=ci,feature_horizon=details['feature_horizon']):
                                edits+=1;frame_edits+=1;covariance=refinement[1];steps=replayed
                                break
                if stage=='chronological-frontier' and frame_edits:
                    steps,accepted=local_frontier_refine(s,steps)
                    edits+=int(accepted)
                log.log('frontier-complete',outer=outer,pass_kind=stage,frame=s,points=model.count,energy=current['energy'],accepted_edits=frame_edits)
            log.log(stage+'-complete',outer=outer,frontiers=len(states)-1,points=model.count,energy=current['energy'])
        refined,changed_indices,details=refuse_original_factors(model,states,dataset,config,backend)
        candidate=candidate_cache(refined,states,changed_indices)
        if accept(refined,states,candidate,'original-factor-geometry-refusion',outer=outer,**details):edits+=1
        history.append({'outer':outer,'before_energy':before,'accepted_energy':current['energy'],'points':model.count,'map_bytes':model.nbytes,'groups':len(states[0].poses),'accepted_edits':edits})
        write_json(reports/'objective_history.json',history)
        if log.debug:model.save(directory/f'outer_{outer:03d}_points.npz')
        log.log('outer-complete',**history[-1])
        # The estimator certifies its OWN inner equations. A fresh UKF
        # measurement correction is not an outer-iteration fixed-point error.
        metrics=[step.diagnostics.get('inner_solve_metric') for step in steps]
        inner=max((float(value) if value is not None else np.inf for value in metrics),default=np.inf)
        log.log('outer-convergence-check',outer=outer,accepted_edits=edits,inner_solve_metric=inner if np.isfinite(inner) else None,tolerance=config.numerical['gradient_tolerance'],meaning='No improving candidate in declared finite local families plus verified own inner equations; no global or UKF MAP-stationarity claim')
        if outer>=1 and edits==0 and inner<=config.numerical['gradient_tolerance']:
            status='CONVERGED_LOCAL_SEARCH';break
        if time_budget is not None and time.monotonic()-log.start>time_budget:status='TIME_BUDGET_NOT_CONVERGED';break
    # The final map/group edit MUST influence another actual forward run,
    # including when a numerical outer safety ceiling has been reached.
    final=directory/'final_forward'
    if final.exists():final.rename(directory/f'previous_final_forward_{offset:03d}_{counter:06d}')
    if log.debug:model.save(directory/'final_filter_points.npz')
    last_steps=forward_filter(estimator,model,states,dataset,config,backend,final,log,'final-new-map',visual)
    smoothed=estimator.smooth(last_steps)
    if log.debug:save_steps(final/'forward_smooth.npz',last_steps,smoothed)
    log.log('smoother-complete',label='final-new-map',frames=len(states),method=smoothed[2]['method'])
    candidate=candidate_cache(model,smoothed[0]);accepted=accept(model,smoothed[0],candidate,'final-motion-stage')
    covariance=smoothed[1] if accepted or same_estimate(model,model,states,smoothed[0]) else None
    if accepted and status=='CONVERGED_LOCAL_SEARCH':status='NON_CONVERGED_FINAL_UPDATE'
    return Result(model,states,covariance,last_steps,smoothed,cache,ownership,history,log,status,initial,digest)
