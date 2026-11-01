#!/usr/bin/env python3
"""Fifth-pass investigation only: preflight -> CPU tests -> point replays -> report.

Never invokes train.py, an estimator run, RTS/IEKS/EM, calibration, or a full-scene
filter. Existing files are read-only. Use `all` for the complete user-run command.
"""
import argparse,contextlib,copy,csv,hashlib,json,math,os,subprocess,sys,time,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
from tools.f1_definitive_evidence import (DEFAULT_ROOT,MANDATORY,make_inventory,read_json,write_json,csv_write,csv_read,digest,verify_inputs,assert_protected,select_population,paths_for,json_value)
from tools.f1_definitive_math import (LEVELS,PARITY_X_M,PARITY_V_M_S,exact_information,observability,plane_patch,timeline,branch_coordinates,fixed_cost)
from tools.f1_definitive_point import (replay,baseline_parity,raw_evidence,information_trace,intervention_runs,branch_runs,dense_objective_oracle)
from tools import root_cause_oracle as O

OUTPUTS=('baseline_parity_detailed.csv','point_catalogue_and_selection.csv','camera_row_lifecycle.csv','frame0_vs_onset_raw_images.csv','inner_information_index.csv','causal_interventions.csv','depth_surface_evidence.csv','rgb_appearance_evidence.csv','observability_identifiability.csv','numerical_branch_stability.csv','static_full_timeline.csv','negative_controls_moving.csv','iteration_objectives.csv','solver_oracles.csv','temporal_null_results.csv','population_summary.csv')
TEST_MODULES=('tests.test_f1_definitive_drift_forensics','tests.test_f1_diagnostic_units','tests.test_f1_sequential_causal_replay','tests.test_f1_temporal_reference_diagnostics','tests.test_f1_second_pass_oracles','tests.test_forensic_f1_oracles','tests.test_root_cause_oracle')
INTERVENTION_MATRIX=[
 ('REMOVE_ADMITTED_CAMERA','camera admission','Movement persists with same other rows'),('T1_ALL_IMAGES_FRAME0','temporal image change','Difference is explained only by changed support'),('T2_CAMERA_RGB_FRAME0','camera RGB change','Same update persists with same support'),('T2_CAMERA_DEPTH_FRAME0','camera depth change','Same update persists or physical validity changes'),
 ('A1_REFERENCE_RGB','constant RGB subtraction','Cancellation is algebraic, not identified sensor physics'),('A2_REFERENCE_DEPTH','constant depth subtraction','Late unreferenced camera still drives drift'),('A3_REFERENCE_RGBD','constant RGB-D subtraction','Zero at xref is a tautology'),('A4_REFERENCE_RGBD_RPLUSR0','naive per-frame covariance','Shared-reference GLS differs'),
 ('FIX_PHYSICAL_SUPPORT','support switching','Movement persists on the same valid support'),('RGB_FORCE_ONLY_GATE_AND_ACCEPTANCE_FIXED','RGB information force','Depth still independently initiates motion'),('DEPTH_FORCE_ONLY_GATE_AND_ACCEPTANCE_FIXED','depth information force','RGB still independently initiates motion'),('ZERO_VELOCITY_CONSTRAINT_OFFLINE','velocity/position separation','Suppression is imposed, not explanation'),('PRIOR_SENSITIVITY_ONLY','gain sensitivity','Original residual remains nonzero'),('GT_ORACLE_ONLY','moving evidence visibility','GT view is physically unavailable')]


def test_command(out):
    return [sys.executable,'-B',str(REPO/'tools/f1_definitive_drift_forensics.py'),'test','--out',str(out)]


def run_tests(out):
    out=Path(out);inventory=read_json(out/'evidence_inventory.json');verify_inputs(inventory)
    if (out/'tests.log').exists():raise FileExistsError('Preserving existing tests.log; use a fresh fifth_pass_* output')
    os.environ['F1_DEFINITIVE_ROOT']=inventory['forensics_root']
    if __name__=='__main__':os.environ['CUDA_VISIBLE_DEVICES']=''
    records=[];console=sys.stdout
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            super().startTest(test);print('CPU test:',test.id(),file=console,flush=True)
        def addSuccess(self,test):super().addSuccess(test);records.append({'test':test.id(),'status':'PASS'})
        def addFailure(self,test,err):super().addFailure(test,err);records.append({'test':test.id(),'status':'FAIL','detail':self._exc_info_to_string(err,test)})
        def addError(self,test,err):super().addError(test,err);records.append({'test':test.id(),'status':'ERROR','detail':self._exc_info_to_string(err,test)})
        def addSkip(self,test,reason):super().addSkip(test,reason);records.append({'test':test.id(),'status':'SKIP','reason':reason})
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in TEST_MODULES)
    started=time.monotonic()
    with (out/'tests.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        f.write('Command: '+repr(test_command(out))+'\nCPU only; inference/config protection verified.\nInventory SHA256: '+digest(out/'evidence_inventory.json')+'\n')
        result=unittest.TextTestRunner(stream=f,verbosity=2,resultclass=Result).run(suite)
    assert_protected(inventory)
    value={'successful':result.wasSuccessful() and not result.skipped,'unittest_success':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'seconds':time.monotonic()-started,'records':records,'modules':TEST_MODULES,'inventory_sha256':digest(out/'evidence_inventory.json'),'no_skips_required_for_accepted_verdict':True}
    write_json(out/'tests.json',value);return value


def full_timeline(sc,pid,role):
    X,V,status,counts,reason=[sc.run.mm(n) for n in ('posterior_xyz','velocity','point_status','valid_cameras','camera_reason')]
    order=sc.order;x=np.array(X[order,pid]);v=np.array(V[order,pid]);t=sc.t[order];decomp=timeline(x,v,t);rows=[];seen=False;first_loss=None;first_return=None
    for j,k in enumerate(order):
        observed=int(status[k,pid])==1;accepted=[int(c) for c in np.flatnonzero(reason[k,:,pid]==0)] if observed else []
        if j and seen and not accepted and first_loss is None:first_loss=k
        if first_loss is not None and accepted and first_return is None:first_return=k
        seen|=bool(accepted)
        rows.append({'scene':sc.name,'point_id':pid,'role':role,'frame':k,'t_seconds':float(t[j]),'source_frame_camera_pixel':sc.src[pid],'x_plus_m':x[j],'v_plus_m_s':v[j],'accepted':observed,'accepted_cameras':accepted,'accepted_count':int(counts[k,pid]),
                     'drift_vector_m':x[j]-x[0],'drift_norm_m':float(np.linalg.norm(x[j]-x[0])),'prediction_m':decomp['prediction'][j-1] if j else x[0],
                     'accepted_position_correction_m':decomp['correction'][j-1] if j else np.zeros(3),'velocity_increment_m_s':decomp['velocity_increment'][j-1] if j else np.zeros(3),
                     'integrated_velocity_m':decomp['integrated_velocity'][j-1] if j else np.zeros(3),'cumulative_correction_m':decomp['cumulative_correction'][j-1] if j else np.zeros(3),'closure_vector_m':decomp['closure_m'][j-1] if j else np.zeros(3),'first_all_camera_nonmeasurement':first_loss,'first_readmission':first_return,
                     'GT_use':'none in trajectory decomposition','evidence_level':'EXACT_CODE_AND_ARITHMETIC'})
    return rows


def analyze_controls(sc,pid,base,refs,R):
    arms=intervention_runs(sc,pid,base,refs,10);original={f['frame']:f for f in base}
    for name,frames in arms.items():
        for f in frames:
            old=original.get(f['frame'])
            if old is None:continue
            before=f.get('objective_before',{});after=f.get('objective_after',{});same=f['rows']==old['rows']
            prior_same=np.array_equal(f['sm'],old['sm']) and np.array_equal(f['Pm'],old['Pm'])
            status='COUNTERFACTUAL_CAUSAL_WITH_FIXED_CONTEXT' if same and prior_same and after.get('status')=='DEFINED' else 'INDETERMINATE'
            if name.startswith(('A3','ZERO_VELOCITY','PRIOR_SENSITIVITY','A4')):status='OBSERVATIONALLY_SUPPORTED'
            row={'scene':sc.name,'point_id':pid,'frame':f['frame'],'arm':name,'changed_variable':name,'held_fixed':'raw files except the named image field; F/Q, noise, original acceptance rule/4m threshold, production frozen owner schedule; each arm propagates its own posterior',
                 'camera_gate_evidence':f.get('cameras'),'row_nis_cutoff':O.row_gate_threshold(sc.policy['visibility_n_sigma']),'s_minus':f['sm'],'P_minus':f['Pm'],'s_plus':f['s'],'P_plus':f['P'],'dx_from_own_prior_m':f['s'][:3]-f['sm'][:3],'dv_from_own_prior_m_s':f['s'][3:]-f['sm'][3:],'rows':f['rows'],'baseline_rows':old['rows'],'rows_equal':same,
                 'position_delta_vs_A0_m':float(np.linalg.norm(f['s'][:3]-old['s'][:3])),'velocity_delta_vs_A0_m_s':float(np.linalg.norm(f['s'][3:]-old['s'][3:])),
                 'objective_before':before,'objective_after':after,'converged':f['res']['converged'],'observed':f['res']['observed'],'prior_identical_to_A0':prior_same,'confounded':not same or not prior_same,'support_status':'SUPPORT_COMPARABLE' if same else 'SUPPORT_NOT_COMPARABLE',
                 'ownership_scope':'CONDITIONAL_FROZEN_PRODUCTION_CLOUD_NOT_FULL_SCENE_COUNTERFACTUAL','evidence_level':status}
            row['inner_history']=[{'s':h['s'],'A':h['A'],'b':h['b'],'delta':h['delta'],'J':h['J'],'threshold':h['threshold'],
                                  'row_reasons':[(o['camera'],O.NAMES[o['reason']]) for o in h['rows']],'cells':[(o['camera'],o.get('sample',{}).get('pixel00')) for o in h['rows']]} if isinstance(h,dict) and 'A' in h else h for h in f['res'].get('history',[])]
            if name.startswith(('A1','A2','A3')):row['qualification']='ALGEBRA_ONLY; zero residual at unchanged xref is not independent proof'
            if name.startswith('A4'):row['qualification']='INDEPENDENT_SAMPLES_APPROX; omits across-frame R0 correlations, not exact nuisance marginalization'
            if name=='ZERO_VELOCITY_CONSTRAINT_OFFLINE':row['qualification']='Exact v=0 constraint, eliminated covariance DOFs; suppression by construction, not authorized production model'
            R['causal_interventions.csv'].append(row)
            if name.startswith(('T1','T2')):R['temporal_null_results.csv'].append(row)
    return None


def bias_models(sc,pid,raw,R):
    """Conditional linear models on physically sampled xref rows; no GT input."""
    valid=[]
    for r in raw:
        o=r['onset'];c=r['camera'];k=r['frame']
        if o['sample'].get('reason')!=0 or o['r'] is None:continue
        # Frozen-out rows are never evidence for the production motion cause.
        tr=sc.trace(k);allowed={q['camera'] for q in tr['final'][str(pid)].get('rows',[])} if tr else set()
        if c not in allowed:continue
        F=np.eye(6);F[:3,3:]=(sc.t[k]-sc.t[0])*np.eye(3)
        H=np.c_[np.asarray(o['H']),np.zeros((4,3))];valid.append((k,c,H@F,np.asarray(o['r']),np.asarray(o['R_diag'])))
    if not valid:
        R['observability_identifiability.csv'].append({'scene':sc.name,'point_id':pid,'model':'M0-M6','status':'NO_VALID_PRODUCTION_ADMITTED_SUPPORT','evidence_level':'INDETERMINATE'});return
    H=np.vstack([x[2] for x in valid]);y=np.concatenate([x[3] for x in valid]);var=np.concatenate([x[4] for x in valid]);cams=sorted({x[1] for x in valid});B=np.zeros((len(y),4*len(cams)))
    for j,(_,c,*_) in enumerate(valid):B[j*4:j*4+4,4*cams.index(c):4*cams.index(c)+4]=np.eye(4)
    for model,design in [('M0_M6_LOCAL_CONSTANT_VELOCITY',H),('M5_LOCAL_STATIONARY',H[:,:3]),('M2_POINT_CAMERA_OFFSET',np.c_[H,B]),('M4_TIME_VARYING_NUISANCE',np.c_[H,np.eye(len(y))])]:
        weighted=design/np.sqrt(var)[:,None];coef=np.linalg.lstsq(weighted,y/np.sqrt(var),rcond=None)[0];_,s,V=np.linalg.svd(weighted,full_matrices=True);rank=np.linalg.matrix_rank(weighted)
        # Same-camera train at <=5, held-out later times. No fabricated reference.
        train=np.repeat([k<=5 for k,_,*rest in valid],4);test=~train;fit=None;held=None
        if train.any() and test.any():fit=np.linalg.lstsq(weighted[train],y[train]/np.sqrt(var[train]),rcond=None)[0];held=float(np.sum((y[test]-design[test]@fit)**2/var[test]))
        R['observability_identifiability.csv'].append({'scene':sc.name,'point_id':pid,'model':model,'rows':len(y),'rank':rank,'columns':design.shape[1],'singular_values':s,'nullspace':V[rank:].T,'fitted_chi2':float(np.sum((y-design@coef)**2/var)),
            'heldout_chi2':held,'heldout_status':'TEMPORALLY_HELD_OUT_BUT_INDEPENDENCE_UNPROVEN' if held is not None else 'NO_INDEPENDENT_VALIDATION',
            'status':'NOT_IDENTIFIABLE' if rank<design.shape[1] else 'LOCAL_LIKELIHOOD_FULL_RANK','process_scope':'constant-velocity local observability linearization, not a replacement posterior or global trajectory model',
            'bias_prior':'No numerical physical bias prior supplied; no normalized Bayes evidence or proper-prior posterior is invented','source_correlation':'unknown source geometry/color cross-correlation; independent noise conditional calculation only','evidence_level':'NOT_IDENTIFIABLE' if rank<design.shape[1] else 'OBSERVATIONALLY_SUPPORTED'})
    for c in cams:
        rr=[r for r in raw if r['camera']==c and r['onset']['r'] is not None and r['onset']['sample'].get('reason')==0]
        train=[r for r in rr if 1<=r['frame']<=5];held=[r for r in rr if r['frame']>5]
        if train:
            vals=np.stack([r['onset']['r'] for r in train]);variances=np.stack([r['onset']['R_diag'] for r in train]);beta=np.sum(vals/variances,0)/np.sum(1/variances,0)
            R['rgb_appearance_evidence.csv'].append({'scene':sc.name,'point_id':pid,'camera':c,'model':'M2_CONDITIONAL_STATIONARY_POINT_CAMERA_OFFSET','estimated_offset':beta,'training_frames':[r['frame'] for r in train],
                'heldout_frames':[r['frame'] for r in held],'heldout_chi2':float(sum(np.sum((np.asarray(r['onset']['r'])-beta)**2/r['onset']['R_diag']) for r in held)) if held else None,
                'status':'NO_INDEPENDENT_VALIDATION','qualification':'Temporal split shares canonical appearance/geometry and possibly deterministic sensor errors; physical BRDF vs wrong surface is NOT_IDENTIFIABLE','evidence_level':'NOT_IDENTIFIABLE'})


def point_figures(sc,pid,k,dossier,raw,steps,times,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    admitted=dossier.get('accepted_rows',[]);c=6 if pid==67827 and 6 in admitted else (admitted[0] if admitted else int(sc.src[pid,1]));cam=sc.cams[c];u,v,*_=cam.project(sc.x0[pid]);x,y=math.floor(u),math.floor(v)
    fig,axes=plt.subplots(2,3,figsize=(13,7));records=[r for r in steps if r['scene']==sc.name and r['point_id']==pid and r['frame']==k]
    for j,t in enumerate((0,k)):
        im=sc.images(t)[c];x0,y0=max(0,x-7),max(0,y-7);xx=min(cam.W,x+9);yy=min(cam.H,y+9)
        axes[j,0].imshow(im.rgb[y0:yy,x0:xx]);axes[j,0].plot([x-x0,x-x0+1,x-x0+1,x-x0,x-x0],[y-y0,y-y0,y-y0+1,y-y0+1,y-y0],color='lime');axes[j,0].set_title(f'actual RGB camera {c}, frame {t}; 2×2 cell')
        axes[j,1].imshow(np.where(im.mask[y0:yy,x0:xx],im.depth[y0:yy,x0:xx],np.nan));axes[j,1].set_title('metric depth; masked pixels omitted')
    if records:
        axes[0,2].plot([r['iteration'] for r in records],[r['J_actual_own_support'] for r in records],'o-',label='actual J; row sets recorded');axes[0,2].legend();axes[0,2].set_xlabel('inner iteration')
    axes[1,2].plot([r['t_seconds'] for r in times],[r['drift_norm_m'] for r in times],color='tab:blue');axes[1,2].set_ylabel('drift (m)',color='tab:blue');right=axes[1,2].twinx();right.plot([r['t_seconds'] for r in times],[np.linalg.norm(r['v_plus_m_s']) for r in times],color='tab:orange');right.set_ylabel('speed (m/s)',color='tab:orange')
    fig.suptitle(f'{sc.name}/{pid}: {dossier.get("parity_status")}; face identity unproven');fig.tight_layout();fig.savefig(out/'figures'/f'{sc.name}_{pid}_dossier.png',dpi=130);plt.close(fig)
    if sc.name=='USB_100109' and pid==67827:
        cam4=sc.cams[4];uu,vv,*_=cam4.project(sc.x0[pid]);xx,yy=math.floor(uu),math.floor(vv);x0,y0=max(0,xx-7),max(0,yy-7)
        fig,axes=plt.subplots(2,2,figsize=(9,8))
        for j,t in enumerate((0,7)):
            im=sc.images(t)[4];patch=im.rgb[y0:yy+9,x0:xx+9];axes[j,0].imshow(patch);axes[j,0].set_title(f'camera 4 frame {t}; raw RGB at xref')
            axes[j,1].imshow(np.where(im.mask[y0:yy+9,x0:xx+9],im.depth[y0:yy+9,x0:xx+9],np.nan));axes[j,1].set_title('masked metric depth; reference validity separate')
        fig.suptitle('USB 67827 late camera 4: pre-existing pixels do not establish a valid frame-0 reference');fig.tight_layout();fig.savefig(out/'figures/USB_100109_67827_camera4_late.png',dpi=130);plt.close(fig)
    if pid in (8444,67827,51975,8416,10251):
        from tools.f1_temporal_reference_diagnostics import references,RefProblem,model_cost
        p=problem_at(sc,pid,k);sm=p.s_minus;accepted=np.asarray(dossier['linearization_state']);support=dossier['accepted_rows'];alpha=np.linspace(-.5,1.5,81)
        refs=references(sc,pid,sc.frame0_ownership([pid])[pid])
        delta=RefProblem(p.cams,p.imgs,p.a,sm,p.P_minus,p.frozen,p.noise,p.policy,p.max_iters,p.tol,p.recheck_self,refs=refs,channels='RGBD',mode='ALGEBRA_ONLY')
        costs=[fixed_cost(p,sm+a*(accepted-sm),support)['cost'] for a in alpha];differenced=[model_cost(delta,sm+a*(accepted-sm),support)[0] for a in alpha]
        fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(221)
        ax.plot(alpha,[np.nan if v is None else v for v in costs],label='original');ax.plot(alpha,[np.nan if v is None else v for v in differenced],label='reference difference: algebra only');ax.legend();ax.set(xlabel='fraction of accepted six-state correction',ylabel='same-camera objective; invalid support = gaps')
        rows=p.used(p.rows(accepted[:3],final=True));info=exact_information(accepted,sm,p.P_minus,rows);vectors=info['position_posterior_eigenvectors'];extent=max(float(np.linalg.norm(accepted[:3]-sm[:3])),float(np.sqrt(np.linalg.eigvalsh(p.P_minus[:3,:3]).max())))
        grid=np.linspace(-extent,extent,21);Z=np.full((21,21),np.nan);ZD=Z.copy()
        gain,_=__import__('tools.f1_definitive_math',fromlist=['velocity_transfer']).velocity_transfer(p.P_minus,np.zeros(3),np.zeros(3))
        for i,a in enumerate(grid):
            for j,b in enumerate(grid):
                dx=a*vectors[:,0]+b*vectors[:,1];state=np.r_[sm[:3]+dx,sm[3:]+gain@dx];q=fixed_cost(p,state,support)['cost'];Z[j,i]=np.nan if q is None else q
                q=model_cost(delta,state,support)[0];ZD[j,i]=np.nan if q is None else q
        ax=fig.add_subplot(222);image=ax.imshow(Z,origin='lower',extent=[grid[0],grid[-1],grid[0],grid[-1]],aspect='auto');fig.colorbar(image,ax=ax);ax.set(xlabel='weak position-information direction 1 (m)',ylabel='direction 2 (m)',title='original fixed-support objective')
        if np.isfinite(ZD).any():ax.contour(grid,grid,np.ma.masked_invalid(ZD),colors='white',linewidths=.6);ax.set_title('original J; white contours = differenced J')
        np.savez_compressed(out/'figures'/f'{sc.name}_{pid}_objective_values.npz',alpha=alpha,original_1d=np.array(costs,float),differenced_1d=np.array(differenced,float),position_grid_m=grid,original_2d=Z,differenced_2d=ZD,position_directions=vectors,prior=sm)
        ax=fig.add_subplot(223);reason=np.array(sc.run.mm('camera_reason')[:,:,pid]).T;accepted_flags=np.asarray(sc.run.mm('point_status'))[:,pid]==1;reason[:,~accepted_flags]=O.NOT_PROC;ax.imshow(reason,aspect='auto',interpolation='nearest');ax.set(xlabel='frame',ylabel='camera',title='actual accepted camera reasons (NOT_PROCESSED otherwise)')
        ax=fig.add_subplot(224,projection='3d');scale=extent
        for j in range(3):ax.quiver(*sm[:3],*(vectors[:,j]*scale),color=['r','g','b'][j])
        ax.quiver(*sm[:3],*(accepted[:3]-sm[:3]),color='k');ax.set_title('position posterior eigenvectors; black = accepted Δx')
        fig.suptitle(f'{sc.name}/{pid}; all surface identity claims remain conditional');fig.tight_layout();fig.savefig(out/'figures'/f'{sc.name}_{pid}_objective.png',dpi=130);plt.close(fig)


def problem_at(sc,pid,k):
    from tools.f1_second_pass import problem
    return problem(sc,k,pid)


def actual_observability(sc,pid,R):
    """Use archived accepted H/R, not xref gradients or hypothetical rows.

    F is the deterministic process derivative. Q still belongs to the stochastic
    replay; this rank calculation is explicitly NOT a smoothed posterior.
    """
    segments=[];current=[];previous=None
    for k in sc.order[1:11]:
        trace=sc.trace(k)
        if trace is None or str(pid) not in trace['final']:continue
        fin=trace['final'][str(pid)]
        rows=fin.get('rows',[]) if fin['acceptance'].startswith('ACCEPTED') else []
        probes=fin.get('cameras_fallback_state',fin['cameras_final_iterate'])
        signature=tuple((r['camera'],tuple(probes[r['camera']].get('pixel00_u_v') or ())) for r in rows)
        if previous is not None and signature!=previous and current:segments.append(current);current=[]
        previous=signature
        if rows:
            H=np.vstack([np.c_[np.asarray(r['H_x']),np.zeros((4,3))]/np.sqrt(np.asarray(r['R_diag']))[:,None] for r in rows])
            F=np.eye(6);F[:3,3:]=(sc.t[k]-sc.t[0])*np.eye(3)
            current.append((k,H,F,signature))
    if current:segments.append(current)
    for j,segment in enumerate(segments):
        result=observability([r[1] for r in segment],[r[2] for r in segment]);dt=sc.t[segment[0][0]]-sc.t[0]
        L=np.linalg.cholesky(sc.cov0(pid));whitened=observability([r[1] for r in segment],[r[2]@L for r in segment])
        N=np.vstack([-dt*np.eye(3),np.eye(3)])
        R['observability_identifiability.csv'].append({'scene':sc.name,'point_id':pid,'model':'ACTUAL_ACCEPTED_TEMPORAL_LINEARIZATION','segment':j,'frames':[r[0] for r in segment],
            **result,'prior_whitened_dimensionless_singular_values':whitened['singular_values'],'prior_whitened_nullspace':whitened['nullspace'],'first_time_position_velocity_nullspace':N,'first_time_null_residual':float(np.linalg.norm(segment[0][1]@segment[0][2]@N)),
            'support_signature':segment[0][3],'heldout_status':'NO_PHYSICAL_NUISANCE_PRIOR_SUPPLIED','process_scope':'Archived accepted H and R, deterministic F; segmented at camera/cell switches. Q is retained in point replay, not treated as zero posterior noise.',
            'evidence_level':'EXACT_CODE_AND_ARITHMETIC'})


def one_update_sensor_controls(sc,pid,k,R):
    """Same-prior tests; no trajectory, default, sensor file or owner is changed."""
    p=problem_at(sc,pid,k);base=p.run(sc.policy['update_acceptance']);base_rows=[r['camera'] for r in p.used(base['rows_accepted'])] if base['observed'] else []
    variants=[('INVERSE_DEPTH_INTERPOLATION',problem_at(sc,pid,k))]
    variants[0][1].policy={**p.policy,'depth_interpolation':'inverse'}
    # Conditional re-estimation from physically sampled NON-SOURCE cameras.
    own=sc.frame0_ownership([pid])[pid];colors=[];variances=[];cameras=[]
    for c,cam in enumerate(sc.cams):
        if c==int(sc.src[pid,1]) or own[c]==O.SELF_OCC:continue
        u,v,*_=cam.project(sc.x0[pid]);sample=O.sample(sc.images(0)[c],cam,u,v,sc.noise,sc.policy)
        if sample['reason']==O.VALID:colors.append(sample['rgb']);variances.append(sample['var_rgb']);cameras.append(c)
    if colors:
        W=1/np.asarray(variances);a=np.sum(W*np.asarray(colors),axis=0)/W.sum(0);color=problem_at(sc,pid,k);color.a=a
        variants.append(('CANONICAL_COLOR_FROM_OTHER_FRAME0_CAMERAS',color))
    else:R['rgb_appearance_evidence.csv'].append({'scene':sc.name,'point_id':pid,'frame':k,'model':'CANONICAL_COLOR_FROM_OTHER_FRAME0_CAMERAS','status':'NO_INDEPENDENT_VALIDATION','evidence_level':'INDETERMINATE'})
    for name,q in variants:
        result=q.run(sc.policy['update_acceptance']);used=q.used(result['rows_accepted']) if result['observed'] else [];rows=[r['camera'] for r in used]
        target='depth_surface_evidence.csv' if name.startswith('INVERSE') else 'rgb_appearance_evidence.csv'
        R[target].append({'scene':sc.name,'point_id':pid,'frame':k,'model':name,'s_minus':p.s_minus,'P_minus':p.P_minus,'s_plus':result['s_plus'],'P_plus':result['P_plus'],'rows':rows,'baseline_rows':base_rows,
            'source_color':p.a,'conditional_reestimated_color':q.a,'other_frame0_cameras':cameras,'position_delta_vs_A0_m':float(np.linalg.norm(result['s_plus'][:3]-base['s_plus'][:3])),'velocity_delta_vs_A0_m_s':float(np.linalg.norm(result['s_plus'][3:]-base['s_plus'][3:])),
            'same_original_objective_before':fixed_cost(p,p.s_minus,base_rows),'same_original_objective_after':fixed_cost(p,result['s_plus'],base_rows),'same_variant_objective_before':fixed_cost(q,q.s_minus,base_rows),'same_variant_objective_after':fixed_cost(q,result['s_plus'],base_rows),
            'same_prior':True,'support_status':'SUPPORT_COMPARABLE' if rows==base_rows else 'SUPPORT_NOT_COMPARABLE','converged':result['converged'],
            'used_rows':used,'scope':'ONE_UPDATE_CONDITIONAL_SENSITIVITY; NOT a production correction or trajectory benefit','face_status':'FACE_IDENTITY_UNPROVEN','source_covariance':'Cross-camera/color geometry correlation unknown; reestimated color is conditioned, not independently calibrated.',
            'evidence_level':'OBSERVATIONALLY_SUPPORTED' if rows==base_rows else 'INDETERMINATE'})


def holdout_samples(sc,catalogue,R):
    """Raw temporal evidence for ALL sampled controls, without inventing priors."""
    for item in catalogue:
        if item.get('fixed'):continue
        pid=item['point_id'];x=sc.x0[pid]
        for c,cam in enumerate(sc.cams):
            u,v,*_=cam.project(x);first=O.sample(sc.images(0)[c],cam,u,v,sc.noise,sc.policy)
            for k in sc.order[1:6]:
                sample=O.sample(sc.images(k)[c],cam,u,v,sc.noise,sc.policy)
                valid=first['reason']==sample['reason']==O.VALID
                R['rgb_appearance_evidence.csv'].append({'scene':sc.name,'point_id':pid,'frame':k,'camera':c,'model':'PREREGISTERED_HOLDOUT_RAW_TEMPORAL_SAMPLE',
                    'selection':item,'frame0_sample':first,'sample':sample,'delta_rgb':sample['rgb']-first['rgb'] if valid else None,'delta_depth_m':sample['depth']-first['depth'] if valid else None,
                    'baseline_tracking_reason_at_optimized_state':int(sc.run.mm('camera_reason')[k,c,pid]),'baseline_point_status':int(sc.run.mm('point_status')[k,pid]),
                    'status':'VALID_SAMPLES_FACE_IDENTITY_UNPROVEN' if valid else 'NO_VALID_REFERENCE_OR_ONSET_SAMPLE','prior_and_nis':'NOT_RECONSTRUCTED_WITHOUT_ARCHIVED_FULL_COVARIANCE_AND_OWNER_HISTORY',
                    'evidence_level':'OBSERVATIONALLY_SUPPORTED' if valid else 'INDETERMINATE'})


def branch_figure(rows,out):
    chosen=[r for r in rows if r['scene']=='storage_45135' and r['point_id']==8416]
    if not chosen:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for ax,key,unit in zip(axes,('position_l2_m','velocity_l2_m_s'),('position difference (m)','velocity difference (m/s)')):
        for axis in (0,1):
            selected=[r for r in chosen if r['axis']==axis]
            ax.plot([r['epsilon_px'] for r in selected],[r[key] for r in selected],'o',label=('u','v')[axis])
        ax.set_ylabel(unit);ax.legend();ax.set_xscale('symlog',linthresh=1e-14)
    axes[-1].set_xlabel('signed pixel perturbation (px)');fig.suptitle('Storage 8416: one-sided sensitivity; production parity is a separate test')
    fig.tight_layout();fig.savefig(out/'figures/storage_45135_8416_branches.png',dpi=130);plt.close(fig)


def analyze(out):
    from tools.f1_second_pass import Scene
    from tools.f1_temporal_reference_diagnostics import references
    out=Path(out);inventory=read_json(out/'evidence_inventory.json');verify_inputs(inventory)
    if (out/'analysis.json').exists():raise FileExistsError('Preserving existing analysis.json')
    tests=read_json(out/'tests.json') if (out/'tests.json').exists() else {'successful':False}
    R={name:[] for name in OUTPUTS};arrays={};dossiers=[];limitations=[name+': UNAVAILABLE' for name,record in inventory['required'].items() if record['status']!='AVAILABLE'];(out/'figures').mkdir(exist_ok=True)
    csv_write(out/'intervention_preregistration.csv',[{'arm':a,'isolates':b,'disconfirmation':c,'created_before_replays':True} for a,b,c in INTERVENTION_MATRIX])
    for scene,mandatory in MANDATORY.items():
        print('Read-only F1 forensics:',scene,flush=True)
        if inventory['scenes'].get(scene,{}).get('status')!='AVAILABLE':limitations.append(f'{scene}: input inventory unavailable; no numbers synthesized');continue
        sc=Scene(Path(inventory['forensics_root']),scene)
        try:
            catalogue,pop=select_population(inventory['forensics_root'],scene,inventory,out);R['population_summary.csv'].append(pop)
        except (OSError,ValueError,KeyError) as e:
            limitations.append(f'{scene}: population selection INDETERMINATE: {e}');catalogue=[{'scene':scene,'point_id':pid,'role':role,'fixed':True,'random_holdout':False} for pid,role in mandatory.items()]
        R['point_catalogue_and_selection.csv']+=catalogue
        traced_ids=set(sc.trace(sc.order[1])['final']);extra=[r['point_id'] for r in catalogue if str(r['point_id']) in traced_ids and not r.get('fixed',False)]
        # Budget is fixed by selection, not residuals. Points lacking full traces stay explicit.
        for item in catalogue:
            if str(item['point_id']) not in traced_ids:item['replay_status']='INDETERMINATE_NO_SAVED_FULL_COVARIANCE_OR_OWNER_TRACE'
        for pid in list(mandatory)+extra:
            role=mandatory.get(pid,'preregistered_population_control');print('  point',pid,role,flush=True)
            try:
                if str(pid) not in traced_ids:raise KeyError('Point missing from saved production trace')
                baseline=replay(sc,pid,10);par,failed=baseline_parity(sc,pid,baseline);R['baseline_parity_detailed.csv']+=par
                parity_ok=failed is None and len(baseline)==10;qualified=parity_ok and bool(tests.get('successful'));status='PARITY_PASS' if parity_ok else 'PARITY_FAIL'
                if len(baseline)!=10:limitations.append(f'{scene}/{pid}: fewer than 10 consecutive saved point frames')
                times=full_timeline(sc,pid,role);R['static_full_timeline.csv']+=times
                first=next((r for r in times[1:] if r['accepted'] and (np.any(np.asarray(r['accepted_position_correction_m'])!=0) or np.any(np.asarray(r['velocity_increment_m_s'])!=0))),None)
                k=int(first['frame']) if first else int(sc.order[1]);decisive=sorted({k}|({1,2,3} if pid in (67801,102354) else ({1,7} if pid==8444 else ({1,2} if pid==10251 else set())))|({1,4,7} if pid==67827 else set()))
                if k not in {f['frame'] for f in baseline}:
                    qualified=False;limitations.append(f'{scene}/{pid}: onset outside the sequential parity window; no causal intervention benefit claimed')
                dossier=information_trace(sc,pid,k,arrays,R['inner_information_index.csv'],R['camera_row_lifecycle.csv'],R['iteration_objectives.csv'],status)
                dossier.update(role=role,first_divergent_frame=failed,first_update_frame=k,tests_successful=bool(tests.get('successful')),
                               selection_and_GT_evaluation_only=next((r for r in catalogue if r['point_id']==pid),{}),
                               verdict_status='INDETERMINATE' if not qualified else 'EXACT_CODE_AND_ARITHMETIC',physical_mechanism='NOT_IDENTIFIABLE: finite RGB-D samples alone do not uniquely separate view-dependent appearance, footprint/face mismatch and calibration discrepancy')
                # Pixel source definitions are audited without feeding evaluation labels to replay.
                pmeta=inventory['scenes'][scene]['point_mapping'][str(pid)];dossier['catalogue_mapping']=pmeta
                raw=[]
                for frame in sorted(set([*range(1,11),*decisive])):
                    if sc.trace(frame) is None or str(pid) not in sc.trace(frame)['final']:continue
                    if frame in decisive and frame!=k:information_trace(sc,pid,frame,arrays,R['inner_information_index.csv'],R['camera_row_lifecycle.csv'],R['iteration_objectives.csv'],status)
                    for c,cam in enumerate(sc.cams):
                        evidence=raw_evidence(sc,pid,frame,c);raw.append(evidence);R['frame0_vs_onset_raw_images.csv'].append(evidence)
                        if frame in decisive:
                            for t in (0,frame):
                                u,v,*_=cam.project(sc.x0[pid]);im=sc.images(t)[c]
                                for size in (2,3,5):
                                    # 2x2 uses the exact floor cell; 3/5 are centered contextual fits.
                                    xx,yy=math.floor(u),math.floor(v)
                                    if size==2:
                                        sample=O.sample(im,cam,u,v,sc.noise,sc.policy);fit={'size':2,'sample':sample,'status':O.NAMES[sample['reason']],'face_status':'FACE_IDENTITY_UNPROVEN'}
                                    else:fit=plane_patch(im.depth,im.mask,im.raw_valid,xx,yy,size,cam,sc.noise,2*__import__('scipy').stats.norm.sf(sc.policy['visibility_n_sigma']))
                                    if fit.get('normal_world') is not None:
                                        n=np.asarray(fit['normal_world']);axis=np.eye(3)[np.argmin(abs(n))];t1=np.cross(n,axis);t1/=np.linalg.norm(t1);t2=np.cross(n,t1);B=np.stack([n,t1,t2],axis=1)
                                        final=sc.trace(frame)['final'][str(pid)];sm=np.asarray(sc.trace(frame)['prediction']['points'][str(pid)]['s_minus']);dx=np.asarray(final['s_plus'])[:3]-sm[:3]
                                        fit.update(basis_world_columns_n_t1_t2=B,accepted_displacement_components_m=B.T@dx,normal_scope='Conditional local xref plane; not certified face identity or a production normal')
                                    R['depth_surface_evidence.csv'].append({'scene':scene,'point_id':pid,'frame':t,'camera':c,**fit,'evidence_level':'OBSERVATIONALLY_SUPPORTED'})
                refs=references(sc,pid,sc.frame0_ownership([pid])[pid]);write_json(out/f'{scene}_{pid}_references.json',refs)
                if qualified and role!='boundary':
                    analyze_controls(sc,pid,baseline,refs,R)
                    one_update_sensor_controls(sc,pid,k,R)
                    if pid in (8444,67827,51975):
                        from tools.f1_sequential_causal_replay import CloudOwnership,replay as old_replay
                        conditional=old_replay(sc,pid,'A0_ORIGINAL_F1','C',10,refs,own=CloudOwnership(sc))
                        for f,old in zip(conditional,baseline):R['causal_interventions.csv'].append({'scene':scene,'point_id':pid,'frame':f['frame'],'arm':'MODE_C_FROZEN_COMPETING_CLOUD','rows':f['rows_accepted'],'baseline_rows':old['rows'],'rows_equal':f['rows_accepted']==old['rows'],'position_delta_vs_A0_m':float(np.linalg.norm(f['s_plus'][:3]-old['s'][:3])),'velocity_delta_vs_A0_m_s':float(np.linalg.norm(f['s_plus'][3:]-old['s'][3:])),'flags':f['flags'],'qualification':'Competing prior Pxx uses diagonal approximation; ownership near-tie flags preclude exact claims','evidence_level':'INDETERMINATE' if f['flags'] else 'OBSERVATIONALLY_SUPPORTED'})
                else:
                    R['causal_interventions.csv'].append({'scene':scene,'point_id':pid,'arm':'ALL_BENEFIT_CLAIMS_BLOCKED','status':'NUMERICAL_CAUSAL_INTERVENTION_INDETERMINATE','first_divergent_frame':failed,'evidence_level':'INDETERMINATE'})
                if role=='boundary' or pid in (49199,102354,10251,14104,8444):
                    c=dossier['accepted_rows'][0] if dossier['accepted_rows'] else int(sc.src[pid,1]);branches=branch_runs(sc,pid,k,c)
                    for b in branches:b['baseline_parity']=status
                    R['numerical_branch_stability.csv']+=branches
                oracle=dense_objective_oracle(problem_at(sc,pid,k),dossier['accepted_rows']);R['solver_oracles.csv'].append({'scene':scene,'point_id':pid,'frame':k,**oracle,'parity':status,'evidence_level':'OBSERVATIONALLY_SUPPORTED' if qualified else 'INDETERMINATE'})
                actual_observability(sc,pid,R)
                bias_models(sc,pid,raw,R)
                if role.startswith('moving'):
                    # GT enters only AFTER every simulation above; this object is not passed to replay.
                    from tools.forensic_f1_drift import GroundTruth
                    gt=GroundTruth(sc.run,Path(sc.run.man['arguments']['scene']))
                    for f in baseline:
                        true=gt.moved(sc.x0[pid],f['frame']);p=f['prob'];pred=[O.row(p.imgs[c],cam,true,p.a,p.noise,p.policy,p.P_minus[:3,:3],true-p.s_minus[:3]) for c,cam in enumerate(p.cams)]
                        R['negative_controls_moving.csv'].append({'scene':scene,'point_id':pid,'frame':f['frame'],'accepted_rows':f['rows'],'position_GT_ORACLE_ONLY':true,'error_m_GT_ORACLE_ONLY':float(np.linalg.norm(f['s'][:3]-true)),'oracle_valid_camera_samples_GT_ORACLE_ONLY':[c for c,r in enumerate(pred) if r['reason']==0], 'oracle_reason_codes_GT_ORACLE_ONLY':[O.NAMES[r['reason']] for r in pred],'inference_uses_GT':False,'identity_caveat':'closest labelled cloud, not exact material identity','evidence_level':'OBSERVATIONALLY_SUPPORTED'})
                if pid in mandatory:point_figures(sc,pid,k,dossier,raw,R['iteration_objectives.csv'],times,out)
                dossiers.append(dossier)
            except (OSError,KeyError,ValueError,np.linalg.LinAlgError) as e:
                limitations.append(f'{scene}/{pid}: {type(e).__name__}: {e}');dossiers.append({'scene':scene,'point_id':pid,'role':role,'parity_status':'UNAVAILABLE','verdict_status':'INDETERMINATE','reason':str(e)})
        # Cross-point camera-wide RGB gain/offset, trained only on preregistered controls.
        holdout_samples(sc,catalogue,R)
        camera_models(scene,sc,catalogue,R)
    # Reproduce shared-reference covariance oracles, including correlated-source negative control.
    from tools.f1_sequential_causal_replay import covariance_oracle
    covariance=[covariance_oracle(T,2,seed=T,correlated=correlated) for T in (3,4,5) for correlated in (False,True)]
    write_json(out/'shared_reference_covariance_oracles.json',covariance)
    branch_figure(R['numerical_branch_stability.csv'],out)
    for name,rows in R.items():csv_write(out/name,rows,('scene','point_id','status','evidence_level'))
    np.savez_compressed(out/'inner_information_exact.npz',**arrays)
    assert_protected(inventory);verify_inputs(inventory)
    result={'status':'ANALYZED_WITH_EXPLICIT_QUALIFICATIONS','tests_successful':bool(tests.get('successful')),'dossiers':dossiers,'limitations':limitations,'protected_inference_unchanged':True,'production_modified':False,'full_scene_refiltered':False,'gt_in_replay':False,'evidence_levels':LEVELS,'output_counts':{name:len(rows) for name,rows in R.items()}}
    write_json(out/'analysis.json',result);(out/'limitations.md').write_text('# Limitations and unavailable evidence\n\n'+'\n'.join('- '+s for s in limitations+['No trusted dense face/material correspondence was supplied.','Unknown source geometry/color and cross-frame error correlations are not independently estimated.','No proper numerical physical nuisance prior was supplied: latent-bias evidence cannot be normalized or identified by inventing one.','Untraced random holdouts lack full saved covariance/ownership histories: no production-parity-qualified causal interventions are claimed for them.'])+'\n')
    return result


def camera_models(scene,sc,catalogue,R):
    ids=sorted(r['point_id'] for r in catalogue if r.get('random_holdout',False));train=set(ids[::2]);held=set(ids[1::2]);samples={}
    for pid in ids:
        for c,cam in enumerate(sc.cams):
            if int(sc.src[pid,1])==c:continue
            r=O.row(sc.images(0)[c],cam,sc.x0[pid],sc.run.can['canonical_rgb'][pid],sc.noise,sc.policy,sc.P0[pid],np.zeros(3))
            if r['sample']['reason']!=0 or 'r' not in r:continue
            samples.setdefault(c,[]).append((pid,np.asarray(sc.run.can['canonical_rgb'][pid]),r['sample']['rgb'],r['var'][:3]))
    for c,items in samples.items():
        fit=[r for r in items if r[0] in train];test=[r for r in items if r[0] in held]
        for ch in range(3):
            if len(fit)<2 or not test:
                R['rgb_appearance_evidence.csv'].append({'scene':scene,'camera':c,'channel':ch,'model':'M3_CAMERA_WIDE_GAIN_OFFSET','status':'NO_INDEPENDENT_VALIDATION','training_ids':sorted(train),'heldout_ids':sorted(held),'evidence_level':'INDETERMINATE'});continue
            X=np.array([[r[2][ch],1.] for r in fit]);y=np.array([r[1][ch] for r in fit]);w=1/np.array([r[3][ch] for r in fit]);rank=np.linalg.matrix_rank(X*np.sqrt(w)[:,None])
            if rank<2:coef=None;new=None
            else:
                coef=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];new=float(sum((r[1][ch]-np.array([r[2][ch],1.])@coef)**2/r[3][ch] for r in test))
            old=float(sum((r[1][ch]-r[2][ch])**2/r[3][ch] for r in test))
            R['rgb_appearance_evidence.csv'].append({'scene':scene,'camera':c,'channel':ch,'model':'M3_CAMERA_WIDE_GAIN_OFFSET','gain_offset':coef,'rank':rank,'training_ids':[r[0] for r in fit],'heldout_ids':[r[0] for r in test],'M0_heldout_chi2':old,'M3_heldout_chi2':new,'status':'HELD_OUT_POINTS_CONDITIONAL_FACE_IDENTITY_UNPROVEN' if rank==2 else 'NOT_IDENTIFIABLE','qualification':'Selection uses coarse GT static grounding only; no GT coordinates or labels reach an estimator. Clustered/shared source errors prevent independent calibration claims.','evidence_level':'OBSERVATIONALLY_SUPPORTED' if rank==2 else 'NOT_IDENTIFIABLE'})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('preflight','test','analyze','report','all'));p.add_argument('--forensics-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--out',type=Path,default=DEFAULT_ROOT/'fifth_pass_definitive_static_drift_20261030');a=p.parse_args(argv)
    if __name__=='__main__':os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.command in ('preflight','all'):
        inventory=make_inventory(a.forensics_root,a.out);print(json.dumps({'inventory':str(a.out/'evidence_inventory.json'),'scenes':{k:v['status'] for k,v in inventory['scenes'].items()},'git_commit':inventory['git_commit']}),flush=True)
    if a.command in ('test','all'):
        result=run_tests(a.out);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
        if not result['successful']:
            from tools.report_f1_definitive_drift import generate_test_gate_failure
            print(generate_test_gate_failure(a.out,result),flush=True)
            print(f'Tests failed or skipped; analysis was not started. Details: {a.out / "tests.log"}',flush=True)
            return 1
    if a.command in ('analyze','all'):analyze(a.out)
    if a.command in ('report','all'):
        from tools.report_f1_definitive_drift import generate
        path=generate(a.out);print(path,flush=True)
    if a.command=='test':return 0 if result['successful'] else 1
    if a.command in ('all','report'):
        summary=read_json(a.out/'report_validation.json');return 0 if summary['accepted'] else 3
    return 0


if __name__=='__main__':raise SystemExit(main())
