#!/usr/bin/env python3
"""Read-only CPU source-camera factorial and winner audit, never production SLAM.

Use --experiment tests for the full logged regression suite. All experiments
write only to a fresh fifth_pass_* directory. --experiment report reads results.
"""
import argparse
import contextlib
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys
import time
import unittest
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''  # CPU-only CLI; importing diagnostics must not disable the caller's GPU
from tools import f1_source_responsibility as S
from tools import root_cause_oracle as O
from tools.f1_second_pass import Scene
from tools.f1_definitive_evidence import (DEFAULT_ROOT,make_inventory,write_json,read_json,csv_write,digest,file_record,verify_inputs,assert_protected)
from tools.f1_definitive_math import exact_information
from tools.f1_diagnostic_units import compare_state6

TABLES=('responsibility_frame0','first_update_responsibility','source_owner_conflicts','association_counterfactuals',
        'sequence_state','sequence_rows','sequential_failure_events','population_source_vs_cross','baseline_parity','responsibility_matrix','responsibility_temporal_xref')


def run_tests(out,root):
    """Capture complete output/exit status; a skipped test is not acceptance."""
    if (out/'tests.log').exists():raise FileExistsError('Preserving existing tests.log')
    os.environ['F1_SOURCE_FORENSICS_ROOT']=str(root);records=[];console=sys.stdout
    class Result(unittest.TextTestResult):
        def startTest(self,test):super().startTest(test);print('CPU test:',test.id(),file=console,flush=True)
        def addSuccess(self,test):super().addSuccess(test);records.append({'test':test.id(),'status':'PASS'})
        def addFailure(self,test,err):super().addFailure(test,err);records.append({'test':test.id(),'status':'FAIL','detail':self._exc_info_to_string(err,test)})
        def addError(self,test,err):super().addError(test,err);records.append({'test':test.id(),'status':'ERROR','detail':self._exc_info_to_string(err,test)})
        def addSkip(self,test,reason):super().addSkip(test,reason);records.append({'test':test.id(),'status':'SKIP','reason':reason})
    modules=['tests.test_f1_source_responsibility','tests.test_f1_diagnostic_units','tests.test_f1_sequential_causal_replay']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules);start=time.monotonic()
    with (out/'tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        log.write('CPU only. Full unittest output. Command: '+repr(sys.argv)+'\n')
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
        success=result.wasSuccessful() and not result.skipped;log.write(f'\nexit_code={0 if success else 1}\n')
    value={'successful':success,'exit_code':0 if success else 1,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
           'records':records,'seconds':time.monotonic()-start,'modules':modules,'log_sha256':digest(out/'tests.log')}
    write_json(out/'tests.json',value);return value


def preselect(sc,core,own,out):
    """Use archived candidates, before running any counterfactual response."""
    traced=sorted(map(int,sc.trace(1)['final']));features=[];cache={}
    seed=int(digest(sc.run.root/'run_manifest.json')[:16],16);rng=np.random.default_rng(seed)
    for pid in traced:
        audit,refs=S.frame0_audit(sc,pid,own);cache[pid]=(audit,refs)
        cross=[r for r in audit if not r['is_source'] and r['reference']['valid']];score=sum(float(np.sum(np.asarray(r['r_equals_minus_h'])**2/r['R'])) for r in cross)
        source=int(sc.src[pid,1]);drift=float(np.linalg.norm(sc.run.mm('posterior_xyz')[-1,pid]-sc.x0[pid]));slope=max((float(np.linalg.norm(r['sample'].get('grad_depth',0))) for r in audit),default=0.)
        features.append({'point_id':pid,'source_camera':source,'cross_candidates':len(cross),'frame0_residual_chi2':score,'depth_slope_m_px':slope,'source_duplicate':int(own.reason[source,pid])==O.DUP,'baseline_final_drift_m':drift})
    # GT is used only to select a diagnostic static cohort AFTER image features.
    with np.load(sc.fdir/'population_events.npz',allow_pickle=False) as z:
        if not np.array_equal(z['x0'],sc.x0):raise ValueError('Selection GT point mapping differs')
        static=z['gt_static'].astype(bool)
    pool=[r for r in features if static[r['point_id']] and r['point_id'] not in core];selected=[]
    if pool:
        edges=np.quantile([r['baseline_final_drift_m'] for r in pool],[1/3,2/3]);groups={}
        for r in pool:
            band=int(np.searchsorted(edges,r['baseline_final_drift_m']));key=(band,r['source_camera'],r['cross_candidates'],r['source_duplicate'],float(r['frame0_residual_chi2']),float(r['depth_slope_m_px']))
            groups.setdefault(key,[]).append(r['point_id'])
        bands={b:[k for k in sorted(groups) if k[0]==b] for b in (0,1,2)}
        for b in bands:rng.shuffle(bands[b])
        for j in range(max(map(len,bands.values()),default=0)):
            for b in bands:
                if j<len(bands[b]):selected.append(int(rng.choice(groups[bands[b][j]])))
                if len(selected)>=6:break
            if len(selected)>=6:break
    chosen=set(core)|set(selected)
    csv_write(out/f'{sc.name}_selection.csv',[{**r,'selected':r['point_id'] in chosen,'reason':'CORE' if r['point_id'] in core else ('PREREGISTERED_STRATIFIED' if r['point_id'] in selected else 'BUDGET_EXCLUDED'),'selection_seed':seed,'GT_use':'selection only, after image features'} for r in features])
    np.savez_compressed(out/f'{sc.name}_untraced_excluded_ids.npz',point_ids=np.setdiff1d(np.arange(len(sc.x0)),traced),reason=np.array('NO_ARCHIVED_FULL_POINT_TRACE'))
    write_json(out/f'{sc.name}_preregistration.json',{'seed':seed,'selected_extra_ids':selected,'before_counterfactual_responses':True,'maximum_extra_budget':6})
    return selected,cache


def audit_archived_update(sc,pid,k,rows,arrays,out):
    trace=sc.trace(k);sp=str(pid);p=S.problem(sc,k,pid)
    phases=[('PRIOR',-1,np.asarray(trace['prediction']['points'][sp]['s_minus']),trace['iterations'][0]['points'][sp]['cameras'],trace['iterations'][0]['points'][sp]['rows'])]
    for j,t in enumerate(trace['iterations']):
        q=t['points'][sp]
        if q['active']:phases.append(('INNER',j,np.asarray(q['state']),q['cameras'],q['rows']))
    fin=trace['final'][sp];accepted=fin['acceptance'].startswith('ACCEPTED');probes=fin.get('cameras_fallback_state',fin['cameras_final_iterate'])
    phases.append(('ACCEPTED' if accepted else 'REJECTED',-1,np.asarray(fin['linearization_state']),probes,fin.get('rows',[]) if accepted else []))
    for phase,j,state,probes,admitted in phases:
        usable=[{'camera':r['camera'],'H':np.asarray(r['H_x']),'r':np.asarray(r['r']),'var':np.asarray(r['R_diag']),'reason':0} for r in admitted if r.get('reason','VALID')=='VALID']
        raw=p.used(p.rows(state[:3],final=phase in ('ACCEPTED','REJECTED')))
        info=exact_information(state,p.s_minus,p.P_minus,usable);prefix=f'{sc.name}_{pid}_{k}_{phase}_{j}'
        for key,value in info.items():
            if isinstance(value,np.ndarray):arrays[prefix+'__'+key]=value
        if phase=='INNER':
            actual=trace['iterations'][j]['points'][sp]
            for key in ('A','b','delta'):arrays[prefix+'__production_'+key]=np.asarray(actual[key])
        for c,probe in enumerate(probes):
            owner=S.owner_from_probe(sc,probe,pid);r=O.row(p.imgs[c],p.cams[c],state[:3],p.a,p.noise,p.policy,p.P_minus[:3,:3],state[:3]-p.s_minus[:3]);admit=c in {u['camera'] for u in usable}
            without=exact_information(state,p.s_minus,p.P_minus,[u for u in usable if u['camera']!=c])
            record={'scene':sc.name,'point_id':pid,'frame':k,'phase':phase,'iteration':j,'camera':c,'is_source':c==int(sc.src[pid,1]),**owner,
                'candidate_at_projection':probe['visibility_fresh']=='VALID','production_frozen_owner':probe['visibility_frozen'],'sample_valid':r['sample']['reason']==O.VALID,'row_gate_passed':r['reason']==O.VALID,
                'admitted_at_prior':admit and phase=='PRIOR','used_in_information':admit and phase=='INNER','accepted_state_row':admit and phase=='ACCEPTED','production_reason':probe['reason_used'],'raw_probe':probe,
                'r':r.get('r'),'H':r.get('H'),'R':r.get('var'),'S':r.get('S'),'nis':r.get('nis'),'array_prefix':prefix,
                'oracle_admitted_cameras':[r['camera'] for r in raw],'production_admitted_cameras':[r['camera'] for r in usable],
                'force_component_labels':info['labels'],'force_attribution_closure':info['attribution_closure'],'normal_equation_residual':info['normal_equation_residual'],
                'production_A_error':float(abs(info['A']-np.asarray(trace['iterations'][j]['points'][sp]['A'])).max()) if phase=='INNER' else None,
                'production_b_error':float(abs(info['b']-np.asarray(trace['iterations'][j]['points'][sp]['b'])).max()) if phase=='INNER' else None,
                'full_delta_step':info['delta'],'camera_removed_delta_step':without['delta'],'camera_removal_effect_position_m':info['delta'][:3]-without['delta'][:3],
                'camera_removal_effect_velocity_m_s':info['delta'][3:]-without['delta'][3:],'interpretation':'Fixed linearization complete 6D solve; not accumulated accepted displacement or physical material proof'}
            rows['first_update_responsibility'].append(record)
            if record['is_source'] and probe['reason_used']!='VALID':
                rows['source_owner_conflicts'].append({**record,'source':S.source_record(sc,pid),'source_row_if_evaluated':S.source_probe(p,c,state[:3]),
                    'other_point_row_lost_if_priority':owner.get('owner_point_id') if probe['visibility_frozen']=='DUPLICATE_PIXEL_SUPPRESSED' else None,
                    'double_count_flag':'NOT_ASSIMILATED_IN_PROBE; PRIORITY_IS_ONE_SLOT_TRANSFER_NOT_GLOBAL_POSTERIOR'})
                if phase=='PRIOR' and ((sc.name=='storage_45135' and pid==8444 and k==1) or (sc.name=='USB_100109' and pid==85696 and k in (2,3,4,5))):
                    from tools.report_f1_source_responsibility import owner_figure
                    owner_figure(sc,pid,k,probe,owner,out)


def code_for_row(row,source,ref,owner):
    if row['camera']==source:
        return 'SRC_VALID' if row['reason']==0 else f'SRC_SUPPRESSED_BY_POINT:{owner}'
    if row['reason']==O.DISCONT:return 'CROSS_DEPTH_CREASE'
    return {O.MASK:'CROSS_MASK_INVALID',O.SELF_OCC:'CROSS_OCCLUDED',O.DUP:'CROSS_DUPLICATE',O.INCONS:'CROSS_CHI2_REJECT',O.FRONT:'CROSS_OCCLUDED',O.VALID:'CROSS_VALID_REF' if ref['valid'] else 'CROSS_VALID_NO_REF'}.get(row['reason'],'UNDECIDABLE')


def record_frames(sc,pid,v,frames,baseline,rows,qualified,scope):
    source=int(sc.src[pid,1]);initial=sc.x0[pid];summary=[];first_source_loss=None;first_all_loss=None;first_correction=None
    if scope=='CONTINUOUS':
        P=sc.cov0(pid);rows['sequence_state'].append({'scene':sc.name,'point_id':pid,'frame':0,'variant':S.variant_id(v),'arm':v.name,'variant_parameters':asdict(v),'comparison_scope':scope,
            'x_minus_m':initial,'x_plus_m':initial,'v_minus_m_s':np.zeros(3),'v_plus_m_s':np.zeros(3),'P_minus':P,'P_plus':P,'Pxx_eigenvalues_m2':np.linalg.eigvalsh(P[:3,:3]),'Pvv_eigenvalues_m2_s2':np.linalg.eigvalsh(P[3:,3:]),
            'canonical_drift_m':0.,'observed':False,'rows_accepted':[],'acceptance':'ANCHOR_CANONICAL_NO_MULTIVIEW_UPDATE','baseline_parity_qualified':qualified,'material_identity':'SOURCE_PROVENANCE_ONLY'})
    for f,old in zip(frames,baseline):
        comparison=compare_state6(f['s_plus'],old['s_plus']);matrix={};accepted=f['rows_accepted']
        fixed_same=np.array_equal(f['s_minus'],old['s_minus']) and np.array_equal(f['P_minus'],old['P_minus']);same_support=accepted==old['rows_accepted']
        base={'scene':sc.name,'point_id':pid,'frame':f['frame'],'variant':f['variant'],'arm':v.name,'variant_parameters':asdict(v),'comparison_scope':scope,
              'baseline_parity_qualified':qualified,'interpretation':'CONDITIONAL_FROZEN_COMPETITORS','material_identity':'MATERIAL_ID_UNDECIDABLE'}
        record={**base,'x_minus_m':f['s_minus'][:3],'v_minus_m_s':f['s_minus'][3:],'x_plus_m':f['s_plus'][:3],'v_plus_m_s':f['s_plus'][3:],
                'P_minus':f['P_minus'],'P_plus':f['P_plus'],'Pxx_eigenvalues_m2':f['Pxx_eigenvalues_m2'],'Pvv_eigenvalues_m2_s2':f['Pvv_eigenvalues_m2_s2'],
                'F':f['F'],'Q':f['Q'],'dt_seconds':f['dt_seconds'],'observed':f['observed'],'rows_accepted':accepted,'acceptance':f['acceptance'],'converged':f['converged'],
                'dx_m':f['dx_m'],'dv_m_s':f['dv_m_s'],'canonical_drift_m':float(np.linalg.norm(f['s_plus'][:3]-initial)),'difference_vs_A':comparison,
                'reference_uncertainty':f['variance_interpretation'],'source_priority_transfer':v.source_priority,'multi_point_posterior_quality_claim':False,'ownership_flags':f['ownership_flags']}
        rows['sequence_state'].append(record)
        rows['association_counterfactuals'].append({**record,'original_fixed_support_before':f['original_fixed_support_before'],'original_fixed_support_after':f['original_fixed_support_after'],
            'model_cost_after':f['model_cost_after'],'inner_history':f['inner_history'],'same_prior_as_A':fixed_same,'same_support_as_A':same_support,
            'support_status':'SUPPORT_COMPARABLE' if same_support else 'SUPPORT_NOT_COMPARABLE','evidence_level':'COUNTERFACTUAL_CAUSAL_WITH_FIXED_CONTEXT' if qualified and fixed_same and same_support and not v.source_priority else 'INDETERMINATE'})
        for phase,row in [('PRIOR',r) for r in f['prior_rows']]+[('ACCEPTANCE',r) for r in f['actual_rows']]:
            c=row['camera'];ref_status='NO_VALID_REFERENCE' if row.get('reference_status','').startswith('NO_VALID_REFERENCE') else ('APPLIED' if row.get('reference_applied') else 'NOT_APPLIED')
            reference=f['references'].get(c,{'valid':None,'status':'RAW_MODEL_REFERENCE_NOT_SUBTRACTED'})
            probes=sc.trace(f['frame'])['iterations'][0]['points'][str(pid)]['cameras'];owner=S.owner_from_probe(sc,probes[c],pid)
            rows['sequence_rows'].append({**base,'phase':phase,'camera':c,'is_source':c==source,'in_accepted_solve':phase=='ACCEPTANCE' and c in accepted,'reason':O.NAMES[row['reason']],
                'frozen_reason':O.NAMES[f['frozen_codes'][c]],'excluded_by_eligibility':row.get('excluded_by_variant',False),'reference_status':ref_status,'reference_definition':reference,'raw_row':row,'source_probe':f['source_probe'] if c==source else None,
                'original_production_prior_owner':owner,'priority_changes':f['source_priority_changes'] if c==source else None,'displaced_original_prior_owner':owner.get('owner_point_id') if c==source and f['source_priority_changes'] else None})
            matrix[f'camera_{c}']=code_for_row(row,source,{'valid':ref_status!='NO_VALID_REFERENCE'},None)
        rows['responsibility_matrix'].append({**base,**matrix})
        if source not in accepted and first_source_loss is None:first_source_loss=f['frame']
        if not accepted and first_all_loss is None:first_all_loss=f['frame']
        if (np.any(f['dx_m']!=0) or np.any(f['dv_m_s']!=0)) and f['observed'] and first_correction is None:first_correction=f['frame']
        summary.append(record)
    event={'scene':sc.name,'point_id':pid,'variant':S.variant_id(v),'arm':v.name,'scope':scope,'first_source_loss':first_source_loss,'first_all_row_loss':first_all_loss,'first_nonzero_correction':first_correction,
           'final_drift_m':summary[-1]['canonical_drift_m'] if summary else None,'moving_assessment':S.tracking_assessment(frames,S.CASES[sc.name].get(pid,'static_control'))}
    rows['sequential_failure_events'].append(event);return event


def analyze(args,out,inventory,tests):
    rows={name:[] for name in TABLES};arrays={};points=[];limitations=[];cache4=dict(np.load(args.fourth_pass_root/'sequence_arrays.npz',allow_pickle=False))
    for scene,core in S.CASES.items():
        sc=Scene(args.forensics_root,scene);own0=S.Frame0Ownership(sc);extra,cache=preselect(sc,core,own0,out);own=S.CloudOwnership(sc)
        for pid in [*core,*extra]:
            print(f'{scene}/{pid}: provenance and production parity',flush=True);a0,refs=cache[pid];rows['responsibility_frame0']+=a0
            rows['responsibility_temporal_xref']+=S.temporal_audit(sc,pid,refs,args.end)
            baseline=S.replay(sc,pid,S.Variant('A_ALL_RAW'),a0,refs,args.end);parity=S.baseline_parity(sc,pid,baseline,cache4);rows['baseline_parity']+=parity
            qualified=len(baseline)==args.end and all(r['status']=='PARITY_PASS' for r in parity);onset=next((f['frame'] for f in baseline if f['observed'] and (np.any(f['dx_m']!=0) or np.any(f['dv_m_s']!=0))),1)
            point={'scene':scene,'point_id':pid,'source':S.source_reprojection(sc,pid),'parity':'PARITY_PASS' if qualified else 'PARITY_FAIL','decision':'INDETERMINATE','material_identity':'MATERIAL_ID_UNDECIDABLE','arm_outcomes':[]}
            for k in sorted({1,onset}|({2,3,4,5} if scene=='USB_100109' and pid==85696 else ({4,7} if scene=='USB_100109' and pid==67827 else set()))):
                if k<=args.end:audit_archived_update(sc,pid,k,rows,arrays,out)
            if not qualified:
                point['reason']='Baseline mismatch: no quantitative intervention claim or arm execution. See baseline_parity.csv.';points.append(point)
                record_frames(sc,pid,S.Variant('A_ALL_RAW'),baseline,baseline,rows,False,'CONTINUOUS');continue
            planned=S.variants(include_recomputed=pid in (8444,67827,87290,85696,8416))
            original_cameras=baseline[onset-1]['rows_accepted']
            planned += [S.Variant(f'T_ASSOC_REMOVE_CAMERA_{c}',remove_camera=c) for c in original_cameras]
            planned += [S.Variant(f'T_BIAS_REFERENCE_CAMERA_{c}',reference_channels='RGBD',reference_camera=c) for c in original_cameras]
            c_baseline=None;c_qualified=False
            for v in planned:
                print(' ',S.variant_id(v),flush=True)
                if v.ownership=='C' and v.name!='A_ALL_RAW' and not c_qualified:
                    point['arm_outcomes'].append({'arm':S.variant_id(v),'status':'BLOCKED_RECOMPUTED_BASELINE_PARITY'});continue
                frames=baseline if v==S.Variant('A_ALL_RAW') else S.replay(sc,pid,v,a0,refs,args.end,own)
                if v.ownership=='C' and v.name=='A_ALL_RAW':
                    cp=S.baseline_parity(sc,pid,frames,cache4);point['recomputed_parity']=cp;c_baseline=frames
                    c_qualified=all(r['status']=='PARITY_PASS' for r in cp) and not any(flag for f in frames for flag in f['ownership_flags'])
                    if not c_qualified:limitations.append(f'{scene}/{pid}: recomputed baseline differs or has near-tie flags; C-mode interventions blocked')
                mode_qualified=qualified and (v.ownership=='P' or c_qualified)
                point['arm_outcomes'].append(record_frames(sc,pid,v,frames,c_baseline if v.ownership=='C' else baseline,rows,mode_qualified,'CONTINUOUS'))
                for one_k in sorted({1,onset}):
                    one=S.replay(sc,pid,v,a0,refs,args.end,own,single_frame=one_k);one_base=S.replay(sc,pid,S.Variant('A_ALL_RAW',ownership=v.ownership),a0,refs,args.end,own,single_frame=one_k)
                    event=record_frames(sc,pid,v,one,one_base,rows,mode_qualified,'SAME_SAVED_PRIOR_ONE_UPDATE')
                    point.setdefault('same_prior_updates',[]).append({'frame':one_k,'variant':S.variant_id(v),'position_correction_m':float(np.linalg.norm(one[0]['dx_m'])),
                        'velocity_correction_m_s':float(np.linalg.norm(one[0]['dv_m_s'])),'difference_vs_A':compare_state6(one[0]['s_plus'],one_base[0]['s_plus']),
                        'accepted_cameras':one[0]['rows_accepted'],'converged':one[0]['converged'],'original_objective_before':one[0]['original_fixed_support_before'],'original_objective_after':one[0]['original_fixed_support_after']})
            conflicts=[r for r in rows['source_owner_conflicts'] if r['scene']==scene and r['point_id']==pid and r['frame']==1 and r['phase']=='PRIOR']
            point['source_conflict_at_frame1']=[{'owner_point_id':r['owner_point_id'],'owner_source':r.get('owner_source'),'reason':r['production_reason'],
                'source_probe_reason':O.NAMES[r['source_row_if_evaluated']['reason']],'depth_gap_m':r.get('depth_gap_m'),'occlusion_allowance_m':r.get('production_occlusion_allowance_m')} for r in conflicts]
            point['orthogonal_tests']={'T_assoc':[r for r in point['same_prior_updates'] if 'T_ASSOC_' in r['variant']],
                'T_bias':[r for r in point['same_prior_updates'] if 'T_BIAS_' in r['variant']],
                'T_owner':[r for r in point['same_prior_updates'] if 'SRC_PRIORITY' in r['variant']],
                'T_temporal':'Frame-0 references vs actual image residuals are reported without asserting material identity.',
                'identifiability':'Association vs bias cannot be identified solely by rescue; material ID remains undecidable'}
            from tools.report_f1_source_responsibility import trajectory_figure
            trajectory_figure(scene,pid,rows['sequence_state'],out);points.append(point)
            checkpoint(out,rows,arrays,points,limitations,tests)
        # Image features contain no GT; scoring happens afterward, never in replay.
        if args.population and all(p['parity']=='PARITY_PASS' for p in points if p['scene']==scene and p['point_id'] in (8444,67827)):
            from tools.f1_source_responsibility_population import population_features,score_population
            features=population_features(sc,out,args.population_chunk_size);rows['population_source_vs_cross']+=score_population(sc,features,out)
        else:limitations.append(f'{scene}: population audit explicitly disabled or primary point parity failed')
    from tools.f1_source_responsibility_evaluation import moving_evaluation
    moving_evaluation(args.forensics_root,rows['sequence_state'],out)
    checkpoint(out,rows,arrays,points,limitations+['Mode C reproduces the fourth-pass diagonal-covariance approximation; missing full competitor covariance and float32 stored velocity preclude universal exact ownership claims.',
        'Source-priority is a measurement-slot transfer ablation, not a validated coupled posterior; material identity and persistent cross-time noise remain uncalibrated.'],tests)
    assert_protected(inventory);verify_inputs(inventory)
    return read_json(out/'causal_evidence.json')


def checkpoint(out,rows,arrays,points,limitations,tests):
    for name,records in rows.items():csv_write(out/(name+'.csv'),records,('scene','point_id'))
    np.savez_compressed(out/'information_steps.npz',**arrays)
    write_json(out/'causal_evidence.json',{'status':'CONDITIONAL_DIAGNOSTICS_WITH_QUALIFICATIONS','tests_successful':tests['successful'],'points':points,'limitations':limitations,'production_changed':False,'full_scene_refiltered':False})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--forensics-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--fourth-pass-root',type=Path)
    p.add_argument('--out',type=Path,default=DEFAULT_ROOT/'fifth_pass_source_responsibility_20261030');p.add_argument('--frames',default='0:10');p.add_argument('--experiment',choices=('tests','all','audit','report'),default='all')
    p.add_argument('--no-population',action='store_false',dest='population');p.add_argument('--population-chunk-size',type=int,default=4096);args=p.parse_args(argv)
    if args.frames!='0:10':p.error('This acceptance run requires --frames 0:10; benchmark subsets are not production evidence.')
    args.end=10;args.fourth_pass_root=args.fourth_pass_root or args.forensics_root/'fourth_pass_sequential_replay_20261030';out=args.out
    if args.experiment=='report':
        from tools.report_f1_source_responsibility import generate
        print(generate(out))
        if not read_json(out/'tests.json')['successful']:return 1
        return 3 if any(q['parity']!='PARITY_PASS' and q['point_id']!=8416 for q in read_json(out/'causal_evidence.json')['points']) else 0
    inventory=make_inventory(args.forensics_root,out)
    for name in ('RGBDiekf_Codex_Fifth_Pass_Source_Responsibility_Causal_Experiments.md','tools/f1_source_responsibility_audit.py','tools/f1_source_responsibility.py','tools/f1_source_responsibility_population.py','tools/f1_source_responsibility_evaluation.py','tools/report_f1_source_responsibility.py'):
        inventory['source_files'][name]=file_record(REPO/name)
    inventory['fourth_pass_root']=str(args.fourth_pass_root)
    for f in args.fourth_pass_root.glob('*'):
        if f.is_file():inventory['required']['explicit_fourth/'+f.name]=file_record(f)
    write_json(out/'evidence_inventory.json',inventory)
    from tools.report_f1_source_responsibility import code_map,generate
    references=code_map(out)
    for label,record in references.items():
        if 'sha256' in record:inventory['source_files']['reference/'+label]=file_record(record['path'])
    write_json(out/'evidence_inventory.json',inventory)
    (out/'figures').mkdir();(out/'README.md').write_text('# CPU source responsibility audit\n\nCommand: '+repr(sys.argv)+'\n\nActual paths, dirty source state, all baseline/source/input hashes: evidence_inventory.json.\nRun tests with --experiment tests in a fresh fifth_pass_* directory. No inference is run.\n')
    tests=run_tests(out,args.forensics_root);assert_protected(inventory)
    if not tests['successful']:
        write_json(out/'causal_evidence.json',{'status':'TEST_GATE_FAILED_NO_EXPERIMENTS_STARTED','points':[],'limitations':['Tests failed or skipped; inspect tests.log and tests.json.'],'production_changed':False})
        print(generate(out));return 1
    if args.experiment=='tests':print(f"Tests passed: {tests['tests_run']}; full log: {out/'tests.log'}");return 0
    result=analyze(args,out,inventory,tests);print(generate(out))
    write_json(out/'output_hashes.json',{str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='output_hashes.json'})
    return 3 if any(p['parity']!='PARITY_PASS' and p['point_id']!=8416 for p in result['points']) else 0


if __name__=='__main__':raise SystemExit(main())
