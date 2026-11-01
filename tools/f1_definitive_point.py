"""Point/frame CPU diagnostics. No GT variable enters replay or row admission."""
import copy,math,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from tools import root_cause_oracle as O
from tools.f1_second_pass import problem,ACC,ShiftCam
from tools.root_cause_trace import frozen_codes
from tools.f1_diagnostic_units import compare_state6
from tools.f1_definitive_math import parity,exact_information,velocity_transfer,fixed_cost,observability,plane_patch,branch_coordinates,high_precision_step,subset_attribution,covariance_agreement


def replay(sc,pid,N=10,transform=None,channels=(1,1,1,1),prior_scale=1.,fixed_support=False,zero_velocity=False,shift=None):
    """One copied point; frozen production ownership. Not a full-cloud rerun."""
    first=sc.trace(sc.order[1])['prediction']['points'][str(pid)]
    state=np.array(first['s_previous'],float);P=np.array(first['P_previous'],float);P[3:,3:]*=prior_scale
    frames=[]
    for k in sc.order[1:N+1]:
        trace=sc.trace(k)
        if trace is None or str(pid) not in trace['final']:break
        sm,Pm,F,Q=O.predict(state,P,trace['prediction']['dt_s'],np.asarray(trace['prediction']['q_c']))
        imgs=sc.images(k)
        if transform is not None:imgs=transform(k,imgs)
        cams=sc.cams
        if shift is not None and shift[0]==k:
            ci,axis,value,mode=shift[1:];base=cams[ci]
            u,v,*_=base.project(sm[:3]);a=(u if axis==0 else v)
            delta=(np.nextafter(a,float(value))-a) if mode=='nextafter' else value
            cams=[ShiftCam(c,delta if axis==0 else 0.,delta if axis==1 else 0.) if j==ci else c for j,c in enumerate(cams)]
        prob=problem(sc,k,pid,imgs=imgs,s=sm.copy(),P=Pm.copy(),cams=cams,channels=channels)
        initial=prob.used(prob.rows(sm[:3]));support=[r['camera'] for r in initial]
        if fixed_support:prob=FrozenSupport.from_problem(prob,support)
        res=run_control(prob,sc.policy.get('update_acceptance','final_iterate'),zero_velocity) if tuple(channels)!=(1,1,1,1) or zero_velocity else prob.run(sc.policy.get('update_acceptance','final_iterate'))
        rows=[o['camera'] for o in prob.used(res['rows_accepted'])] if res['observed'] else []
        if zero_velocity and res['observed']:
            A=prob.Lam.copy();A[:3,:3]+=prob.info(res['rows_accepted'])[0]
            res['P_plus']=np.zeros((6,6));res['P_plus'][:3,:3]=np.linalg.inv(A[:3,:3])
        frames.append({'frame':k,'sm':sm,'Pm':Pm,'s':res['s_plus'].copy(),'P':res['P_plus'].copy(),'F':F,'Q':Q,'prob':prob,'res':res,'rows':rows,
                       'support_initial':support,'objective_before':fixed_cost(prob,sm,support),'objective_after':fixed_cost(prob,res['s_plus'],support)})
        state,P=res['s_plus'].copy(),res['P_plus'].copy()
    return frames


class FrozenSupport(O.Problem):
    @classmethod
    def from_problem(cls,p,cameras):
        q=cls(p.cams,p.imgs,p.a,p.s_minus.copy(),p.P_minus.copy(),p.frozen,p.noise,p.policy,p.max_iters,p.tol,p.recheck_self,channels=p.ch)
        q.members=set(cameras);return q
    def rows(self,x,final=False):
        rows=[]
        for c,cam in enumerate(self.cams):
            if c not in self.members:rows.append({'camera':c,'reason':O.NOT_PROC});continue
            p={**self.policy,'row_innovation_gate':'depth_front_only','visibility_n_sigma':float('inf')}
            r=O.row(self.imgs[c],cam,x,self.a,self.noise,p,self.P_minus[:3,:3],x-self.s_minus[:3]);r['camera']=c;rows.append(r)
        return rows


def run_control(prob,rule,zero_velocity=False):
    """Offline force controls keep ORIGINAL 4m acceptance DOF, not a retuned gate."""
    from scipy.stats import chi2,norm
    threshold=lambda m:float(chi2.isf(2*norm.sf(prob.policy['visibility_n_sigma']),4*m)) if m else float('inf')
    sj=prob.s_minus.copy();history=[];accepted=None;converged=False;had=False
    for j in range(prob.max_iters):
        rows=prob.rows(sj[:3]);A,b,d,step,I,g,q=prob.solve(sj,rows);m=len(prob.used(rows));had|=bool(m);J=prob.prior_cost(sj)+q
        if zero_velocity:
            d[3:]=-sj[3:];d[:3]=np.linalg.solve(A[:3,:3],b[:3]-A[:3,3:]@d[3:]);step=float(d@A@d)
        history.append({'iteration':j,'s':sj.copy(),'rows':rows,'A':A,'b':b,'delta':d,'step':step,'J':J,'threshold':threshold(m),'nrows':m})
        if j>=1 and m and J<=threshold(m):accepted=sj.copy()
        sj=sj+d
        if step<prob.tol:converged=True;break
    rf=prob.rows(sj[:3],final=True);m=len(prob.used(rf));cost=prob.prior_cost(sj)+prob.info(rf)[2]
    ok=m>0 and (rule=='final_iterate' or cost<=threshold(m));state=sj;used=rf;code=1
    if not ok and accepted is not None and rule!='final_iterate':
        used=prob.rows(accepted[:3],final=True);m=len(prob.used(used));q=prob.prior_cost(accepted)+prob.info(used)[2];ok=m>0 and q<=threshold(m);state=accepted;code=2 if ok else 3
    if not ok:code=3 if had or m else 4
    A=prob.Lam.copy();A[:3,:3]+=prob.info(used)[0];post=np.linalg.inv(A);post=(post+post.T)/2
    return {'history':history,'converged':converged,'rows_accepted':used,'rows_final':rf,'observed':ok,'s_plus':state if ok else prob.s_minus.copy(),'P_plus':post if ok else prob.P_minus.copy(),'acceptance_code':code,'s_final_iterate':sj,'J_final':cost}


def baseline_parity(sc,pid,frames):
    rows=[];first_failure=None
    archive=Path(sc.run.root).parent.parent/'fourth_pass_sequential_replay_20261030/sequence_arrays.npz'
    key=f'{sc.name.split("_")[0].lower()}_{pid}__A0_ORIGINAL_F1__P__'
    archived_state=archived_cov=None
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:
            if key+'s_plus' in z:
                archived_state=z[key+'s_plus'];archived_cov=z[key+'P_plus']
                if archived_state.shape!=(len(archived_state),6) or archived_cov.shape!=(len(archived_state),6,6):
                    raise ValueError(f'Inconsistent fourth-pass state/covariance arrays for {key}')
    for f in frames:
        trace=sc.trace(f['frame']);fin=trace['final'][str(pid)];expected=[r['camera'] for r in fin.get('rows',[])] if fin['acceptance'].startswith('ACCEPTED') else []
        p=parity(f['s'],fin['s_plus'],f['P'],fin['P_plus'],f['rows'],expected,ACC[f['res']['acceptance_code']],fin['acceptance'])
        pred=trace['prediction']['points'][str(pid)];p.update(scene=sc.name,point_id=pid,frame=f['frame'],rows_replay=f['rows'],rows_production=expected,decision_replay=ACC[f['res']['acceptance_code']],decision_production=fin['acceptance'],
            prediction_position_error_m=float(np.linalg.norm(f['sm'][:3]-np.asarray(pred['s_minus'])[:3])),prediction_velocity_error_m_s=float(np.linalg.norm(f['sm'][3:]-np.asarray(pred['s_minus'])[3:])),
            F_exact=bool(np.array_equal(f['F'],trace['prediction']['F'])),Q_error=float(abs(f['Q']-trace['prediction']['Q']).max()),converged_equal=f['res']['converged']==fin['converged'])
        p['prediction_covariance_error']=float(abs(f['Pm']-np.asarray(pred['P_minus'])).max())
        p['prediction_covariance_comparison']=covariance_agreement(f['Pm'],pred['P_minus'])
        if (not p['F_exact'] or p['Q_error']>1e-15 or not p['converged_equal'] or p['prediction_position_error_m']>2e-12 or p['prediction_velocity_error_m_s']>1e-10 or not p['prediction_covariance_comparison']['covariance_equal']):p['status']='PARITY_FAIL'
        j=sc.order.index(f['frame'])-1
        if archived_state is not None and 0<=j<len(archived_state):
            old=parity(f['s'],archived_state[j],f['P'],archived_cov[j],[],[],'','')
            p['fourth_pass_comparison']=old
            if old['status']!='PARITY_PASS':p['status']='PARITY_FAIL'
        else:
            p['fourth_pass_comparison']={'status':'NO_ARCHIVED_FRAME' if archived_state is not None else 'NO_ARCHIVED_CASE',
                'available_frames':len(archived_state) if archived_state is not None else 0,'requested_step':j+1,
                'production_trace_comparison':'RETAINED_WITH_ORIGINAL_TOLERANCES'}
        if p['status']=='PARITY_FAIL' and first_failure is None:first_failure=f['frame']
        p['first_divergent_frame']=first_failure;rows.append(p)
    return rows,first_failure


def raw_evidence(sc,pid,k,cam_id):
    p=problem(sc,k,pid);cam=sc.cams[cam_id];x=sc.x0[pid];a=sc.run.can['canonical_rgb'][pid];u,v,Z,Ju,Jz=cam.project(x)
    base={'scene':sc.name,'point_id':pid,'frame':k,'camera':cam_id,'x_ref':x,'X_camera':cam.R@x+cam.t,'T_cw':np.c_[cam.R,cam.t],'K':[cam.fx,cam.fy,cam.cx,cam.cy],'u':u,'v':v,'float_branch':branch_coordinates(u,v),
          'source_color':a,'source_frame_camera_pixel':sc.src[pid],'physical_face_status':'FACE_IDENTITY_UNPROVEN','camera_time_model':'fixed rig as declared by archived calibration; no independently measured time-varying poses'}
    values=[]
    for t in (0,k):
        im=sc.images(t)[cam_id];o=O.row(im,cam,x,a,sc.noise,sc.policy,p.P_minus[:3,:3],np.zeros(3));sample=o['sample']
        original=sc.run.data.observation(t,cam_id)
        cell=sample.get('pixel00');raw4=None
        if cell:
            xx,yy=cell;ij=[(yy,xx),(yy,xx+1),(yy+1,xx),(yy+1,xx+1)]
            raw4={'rgb_original':[original.rgb[y,z] for y,z in ij],'depth_original_m':[original.depth[y,z] for y,z in ij]}
        values.append({'time':t,'sample':sample,'r':o.get('r'),'H':o.get('H'),'R_diag':o.get('var'),'S':o.get('S'),'nu':o.get('nu'),'nis':o.get('nis'),'reason':O.NAMES[o['reason']],'original_samples':raw4,
                       'whole_rgb_equal_frame0':bool(np.array_equal(im.rgb,sc.images(0)[cam_id].rgb)),'whole_depth_equal_frame0':bool(np.array_equal(im.depth,sc.images(0)[cam_id].depth)),'whole_mask_equal_frame0':bool(np.array_equal(im.mask,sc.images(0)[cam_id].mask))})
    r0,rt=values;physical=lambda r:r['sample'].get('reason')==O.VALID and r['r'] is not None
    owner0=int(sc.frame0_ownership([pid])[pid][cam_id]);base['frame0_owner']=O.NAMES[owner0];base['onset_frozen_owner']=O.NAMES[p.frozen[cam_id]]
    ok=physical(r0) and physical(rt) and owner0!=O.SELF_OCC
    base.update(reference_status='VALID_SAMPLED_REFERENCE_FACE_UNPROVEN' if ok else 'NO_VALID_REFERENCE',frame0=r0,onset=rt,delta_r=np.asarray(rt['r'])-r0['r'] if ok else None,
                delta_r_over_sqrt_R=(np.asarray(rt['r'])-r0['r'])/np.sqrt(rt['R_diag']) if ok else None,temporal_difference_covariance='NOT_ESTIMABLE_FROM_AVAILABLE_INDEPENDENT_SAMPLES',source_noise_correlation='CORRELATED_WITH_XREF_AND_CANONICAL_COLOR' if int(sc.src[pid,1])==cam_id else 'SOURCE_APPEARANCE_AND_GEOMETRY_CROSS_CORRELATION_UNKNOWN')
    if 'neighbour_depth' in r0['sample'] and 'neighbour_depth' in rt['sample']:
        base['raw_2x2_rgb_equal']=bool(np.array_equal(r0['sample']['neighbour_rgb'],rt['sample']['neighbour_rgb']));base['raw_2x2_depth_equal']=bool(np.array_equal(r0['sample']['neighbour_depth'],rt['sample']['neighbour_depth']))
    return base


def information_trace(sc,pid,k,arrays,index,lifecycle,steps,unit_status):
    trace=sc.trace(k);sp=str(pid);pre=trace['prediction']['points'][sp];fin=trace['final'][sp];prob=problem(sc,k,pid)
    for j,iteration in enumerate(trace['iterations']):
        q=iteration['points'][sp]
        if not q['active']:continue
        state=np.asarray(q['state']);rows=prob.rows(state[:3]);admitted={r['camera'] for r in q['rows'] if r['reason']=='VALID'}
        oracle=prob.used(rows);oracle_cams={r['camera'] for r in oracle};exact=exact_information(state,prob.s_minus,prob.P_minus,[r for r in oracle if r['camera'] in admitted])
        tag=f'{sc.name}_{pid}_{k}_{j}'
        for name,value in exact.items():
            if isinstance(value,np.ndarray):arrays[tag+'__'+name]=value
        A,b,delta=map(np.asarray,(q['A'],q['b'],q['delta']))
        high=high_precision_step(A,b)
        row_equal=admitted==oracle_cams
        index.append({'scene':sc.name,'point_id':pid,'frame':k,'iteration':j,'array_prefix':tag,'labels':exact['labels'],'oracle_rows':sorted(oracle_cams),'production_rows':sorted(admitted),'rows_equal':row_equal,
                      'A_error':float(abs(exact['A']-A).max()),'b_error':float(abs(exact['b']-b).max()),'delta_position_error_m':float(abs(exact['delta'][:3]-delta[:3]).max()),'delta_velocity_error_m_s':float(abs(exact['delta'][3:]-delta[3:]).max()),
                      'normal_equation_residual':exact['normal_equation_residual'],'cholesky_reconstruction_residual':exact['cholesky_reconstruction_residual'],'condition':exact['condition'],'decimal70_delta':high,'decimal_position_error_m':float(abs(high[:3]-delta[:3]).max()),'decimal_velocity_error_m_s':float(abs(high[3:]-delta[3:]).max()),'attribution_closure':exact['attribution_closure'],'evidence_level':'EXACT_CODE_AND_ARITHMETIC' if row_equal and np.allclose(exact['A'],A,rtol=1e-10,atol=1e-10) and np.allclose(exact['b'],b,rtol=1e-10,atol=1e-10) else 'INDETERMINATE'})
        support=sorted(admitted);cost0=fixed_cost(prob,prob.s_minus,support);cost=fixed_cost(prob,state,support)
        steps.append({'scene':sc.name,'point_id':pid,'frame':k,'iteration':j,'state':state,'delta':delta,'prior_cost':exact['prior_cost'],'J_actual_own_support':q['posterior_cost'],'J_common_support_before':cost0,'J_common_support_iterate':cost,
                      'row_set':support,'channel_cost':exact['channel_cost'],'step_metric':q['step_delta_A_delta'],'cost_threshold':q['cost_threshold'],'parity_status':unit_status,'cells':{str(r['camera']):r['sample']['pixel00'] for r in oracle}})
        for c,probe in enumerate(q['cameras']):
            lifecycle.append({'scene':sc.name,'point_id':pid,'frame':k,'phase':'INNER','iteration':j,'camera':c,'candidate':probe['visibility_fresh']=='VALID','frozen':probe['visibility_frozen'],'reason_used':probe['reason_used'],'reason_if_sampled':probe['reason_if_sampled'],
                              'candidate_after_frozen':probe['visibility_frozen'] not in ('DUPLICATE_PIXEL_SUPPRESSED','SELF_OCCLUDED') and probe['visibility_fresh']=='VALID',
                              'admitted_production':c in admitted,'used_in_solver':c in admitted,'accepted_state':False,'nearest_owner_candidates':probe.get('pixel_occupants_front_to_back'),'owner_occupants_total':probe.get('pixel_occupants_total'),'owner_listing_scope':'archived front-to-back listing; consult total for truncation',
                              'u':probe['u'],'v':probe['v'],'cell':probe.get('pixel00_u_v'),'raw_valid':probe.get('neighbour_raw_valid'),'mask':probe.get('neighbour_mask'),'raw_rgb4':probe.get('neighbour_rgb'),'raw_depth4_m':probe.get('neighbour_depth_m'),
                              'r':probe.get('r'),'H':probe.get('H_x'),'R_diag':probe.get('R_diag'),'S_DD':probe.get('S_zz'),'nis':probe.get('row_nis_prior_linearized'),'nis_cutoff':O.row_gate_threshold(sc.policy['visibility_n_sigma']),'evidence_level':'EXACT_CODE_AND_ARITHMETIC'})
    accepted={r['camera'] for r in fin.get('rows',[])} if fin['acceptance'].startswith('ACCEPTED') else set()
    final_probes=fin.get('cameras_fallback_state',fin['cameras_final_iterate'])
    for c,probe in enumerate(final_probes):lifecycle.append({'scene':sc.name,'point_id':pid,'frame':k,'phase':'ACCEPTANCE_LINEARIZATION','iteration':fin['last_consistent_iteration'] if fin['acceptance']=='ACCEPTED_EARLIER_CONSISTENT_ITERATE' else 'final','camera':c,'frozen':probe['visibility_frozen'],'reason_used':probe['reason_used'],'admitted_production':c in accepted,'used_in_solver':False,'accepted_state':c in accepted,'acceptance':fin['acceptance'],'u':probe['u'],'v':probe['v'],'cell':probe.get('pixel00_u_v'),'nearest_owner_candidates':probe.get('pixel_occupants_front_to_back'),'evidence_level':'EXACT_CODE_AND_ARITHMETIC'})
    lin=np.asarray(fin['linearization_state'] if accepted else fin['s_plus']);accepted_rows=[]
    for r in (fin.get('rows',[]) if accepted else []):accepted_rows.append({'camera':r['camera'],'H':np.asarray(r['H_x']),'r':np.asarray(r['r']),'var':np.asarray(r['R_diag']),'reason':0})
    accepted_info=exact_information(lin,prob.s_minus,prob.P_minus,accepted_rows);tag=f'{sc.name}_{pid}_{k}_accepted'
    subsets=subset_attribution(lin,prob.s_minus,prob.P_minus,accepted_rows)
    for name,value in accepted_info.items():
        if isinstance(value,np.ndarray):arrays[tag+'__'+name]=value
    index.append({'scene':sc.name,'point_id':pid,'frame':k,'iteration':'accepted','array_prefix':tag,'labels':accepted_info['labels'],'production_rows':sorted(accepted),'normal_equation_residual':accepted_info['normal_equation_residual'],'attribution_closure':accepted_info['attribution_closure'],'evidence_level':'EXACT_CODE_AND_ARITHMETIC','interpretation':'Accepted state may be nonstationary; A^-1 b is the NEXT local step, not total accepted displacement.'})
    dx=np.asarray(fin['s_plus'])[:3]-np.asarray(pre['s_minus'])[:3]
    dv=np.asarray(fin['s_plus'])[3:]-np.asarray(pre['s_minus'])[3:];gain,e=velocity_transfer(pre['P_minus'],dx,dv)
    return {'scene':sc.name,'point_id':pid,'frame':k,'accepted_rows':sorted(accepted),'acceptance':fin['acceptance'],'converged':fin['converged'],'dx_m':dx,'dv_m_s':dv,'dx_norm_m':float(np.linalg.norm(dx)),'dv_norm_m_s':float(np.linalg.norm(dv)),'velocity_transfer_matrix_per_s':gain,'velocity_transfer_error_m_s':e,'P_minus':pre['P_minus'],'P_plus':fin['P_plus'],'xref':sc.x0[pid],'source':sc.src[pid],'canonical_color':sc.run.can['canonical_rgb'][pid], 'parity_status':unit_status,'linearization_state':fin['linearization_state'],'accepted_original_common_cost_before':fixed_cost(prob,prob.s_minus,sorted(accepted)),'accepted_original_common_cost_after':fixed_cost(prob,lin,sorted(accepted)),'accepted_next_stationarity_step':accepted_info['delta'],'camera_subset_attribution':subsets,'accepted_cost':fin['posterior_cost'],'accepted_cost_threshold':fin['cost_threshold']}


def intervention_runs(sc,pid,base,refs,N):
    from tools.f1_sequential_causal_replay import replay as legacy_replay
    onset=next((f for f in base if f['res']['observed'] and np.any(f['s']!=f['sm'])),None)
    offending=onset['rows'] if onset is not None else [];arms={}
    arms['T1_ALL_IMAGES_FRAME0']=replay(sc,pid,N,transform=lambda k,imgs:sc.images(0))
    for c in offending:
        for field in ('rgb','depth'):
            def replace(k,imgs,c=c,field=field):
                out=list(imgs);im=copy.copy(imgs[c]);setattr(im,field,getattr(sc.images(0)[c],field).copy());out[c]=im;return out
            arms[f'T2_CAMERA_{c}_{field.upper()}_FRAME0']=replay(sc,pid,N,transform=replace)
        # A zeroed image is never used: removing the camera changes only its mask in the solver.
        def remove(k,imgs,c=c):
            out=list(imgs);im=copy.copy(imgs[c]);im.mask=np.zeros_like(im.mask);out[c]=im;return out
        arms[f'REMOVE_ADMITTED_CAMERA_{c}']=replay(sc,pid,N,transform=remove)
    for name in ('A1_REFERENCE_RGB','A2_REFERENCE_DEPTH','A3_REFERENCE_RGBD','A4_REFERENCE_RGBD_RPLUSR0'):
        f=legacy_replay(sc,pid,name,'P',N,refs)
        arms[name]=[{'frame':r['frame'],'sm':r['s_minus'],'Pm':r['P_minus'],'s':r['s_plus'],'P':r['P_plus'],'rows':r['rows_accepted'],'res':{'observed':r['observed'],'converged':r['converged'],'history':r['iterates']},'objective_after':{'status':r['J_fixed_set_status'],'cost':r['J_fixed_set_after']},'cameras':r['cameras']} for r in f]
    arms['FIX_PHYSICAL_SUPPORT']=replay(sc,pid,N,fixed_support=True)
    arms['RGB_FORCE_ONLY_GATE_AND_ACCEPTANCE_FIXED']=replay(sc,pid,N,channels=(1,1,1,0))
    arms['DEPTH_FORCE_ONLY_GATE_AND_ACCEPTANCE_FIXED']=replay(sc,pid,N,channels=(0,0,0,1))
    arms['ZERO_VELOCITY_CONSTRAINT_OFFLINE']=replay(sc,pid,N,zero_velocity=True)
    for scale in (.25,4.):arms[f'PRIOR_SENSITIVITY_ONLY_PVV_X{scale}']=replay(sc,pid,N,prior_scale=scale)
    return arms


def branch_runs(sc,pid,k,c):
    values=[('nextafter',-np.inf),('nextafter',np.inf)]+[('offset',x) for e in (1e-12,1e-10,1e-8) for x in (-e,e)]
    out=[];baseline=problem(sc,k,pid);reference=baseline.run(sc.policy.get('update_acceptance','final_iterate'))
    for axis in (0,1):
        u,v,*_=sc.cams[c].project(baseline.s_minus[:3]);a=u if axis==0 else v
        for mode,value in values:
            delta=np.nextafter(a,value)-a if mode=='nextafter' else value
            cams=[ShiftCam(cam,delta if axis==0 else 0.,delta if axis==1 else 0.) if j==c else cam for j,cam in enumerate(sc.cams)]
            p=problem(sc,k,pid,cams=cams);r=p.run(sc.policy.get('update_acceptance','final_iterate'))
            out.append({'scene':sc.name,'point_id':pid,'frame':k,'camera':c,'axis':axis,'kind':mode,'epsilon_px':float(delta),'epsilon_hex':float(delta).hex(),'projected_value_hex':float(a+delta).hex(),**compare_state6(r['s_plus'],reference['s_plus']),
                        'rows_before':[o['camera'] for o in baseline.used(reference['rows_accepted'])],'rows_after':[o['camera'] for o in p.used(r['rows_accepted'])],'acceptance_before':ACC[reference['acceptance_code']],'acceptance_after':ACC[r['acceptance_code']],
                        'cell_before':[(o['camera'],o.get('sample',{}).get('pixel00')) for o in reference['rows_accepted']],
                        'cell_after':[(o['camera'],o.get('sample',{}).get('pixel00')) for o in r['rows_accepted']],
                        'inner_history':[{'s':h['s'],'A':h['A'],'b':h['b'],'delta':h['delta'],'J':h['J'],'rows':[o['camera'] for o in p.used(h['rows'])]} for h in r['history']],
                        'objective_before':fixed_cost(p,p.s_minus,[o['camera'] for o in baseline.used(reference['rows_accepted'])]),'objective_after':fixed_cost(p,r['s_plus'],[o['camera'] for o in baseline.used(reference['rows_accepted'])]),
                        'status':'NUMERICAL_CAUSAL_INTERVENTION_INDETERMINATE','held_fixed':'images, prior, noise, frozen ownership; projection perturbation changes cell/Jacobian/gate as recorded'})
    return out


def dense_objective_oracle(prob,support):
    if not support:return {'status':'NO_ROWS'}
    gain,_=velocity_transfer(prob.P_minus,np.zeros(3),np.zeros(3));sm=prob.s_minus
    def state(x):return np.r_[x,sm[3:]+gain@(x-sm[:3])]
    invalid_trials=0
    def fun(x):
        nonlocal invalid_trials
        if not np.isfinite(x).all():invalid_trials+=1;return float('inf')
        q=fixed_cost(prob,state(x),support)
        if q['cost'] is None:invalid_trials+=1;return float('inf')
        return q['cost']
    start=fun(sm[:3])
    if not np.isfinite(start):return {'status':'UNDEFINED_SUPPORT'}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always',RuntimeWarning)
        result=minimize(fun,sm[:3],method='Powell',options={'maxiter':200})
    final=fun(result.x);finite=bool(np.isfinite(final));success=bool(result.success) and finite
    return {'status':'LOCAL_NUMERICAL_ORACLE_ONLY' if success else 'NUMERICAL_ORACLE_FAILED','solver':'scipy.optimize.minimize Powell; 200-iteration numerical budget','success':success,'message':str(result.message),'state':state(result.x) if np.isfinite(result.x).all() else None,'cost_before':start,'cost_after':float(final) if finite else None,'invalid_support_trials':invalid_trials,'warnings':[str(w.message) for w in recorded],'fixed_cameras':support,'global_uniqueness_claim':False,'production_replacement':False}
