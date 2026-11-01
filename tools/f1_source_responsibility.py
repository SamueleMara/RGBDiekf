"""CPU-only source responsibility experiments; never imported by inference.

Camera eligibility, residual differencing and pixel ownership are independent
factors. Material identity is not inferred from foreground/depth agreement.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from tools import root_cause_oracle as O
from tools.f1_second_pass import problem, ACC
from tools.f1_temporal_reference_diagnostics import RefProblem, references
from tools.f1_sequential_causal_replay import CloudOwnership
from tools.f1_diagnostic_units import compare_state6, compare_cov6
from tools.f1_definitive_math import exact_information, fixed_cost
from tools.root_cause_trace import frozen_codes

CASES = {
    'storage_45135': {8444:'static', 8416:'knife_edge_moving',49199:'static',262894:'static',60019:'static_control',100564:'static_control'},
    'USB_100109': {67827:'static',87290:'moving',85696:'moving',30502:'static',51975:'static',51974:'static_control',30503:'static_control'},
    'blade_103706': {10251:'static',14104:'static'},
}
POSITION_TOL = 1e-10  # This specification section 8, not a motion/noise threshold.
VELOCITY_TOL = 1e-9


@dataclass(frozen=True)
class Variant:
    name: str
    eligibility: str = 'all'
    reference_channels: str | None = None
    variance_mode: str = 'ALGEBRA_ONLY'
    missing_reference: str = 'retain'
    ambiguous: str = 'admit'
    ownership: str = 'P'
    source_priority: bool = False
    remove_camera: int | None = None
    reference_camera: int | None = None


def variants(include_recomputed=True):
    base=[Variant('A_ALL_RAW'),Variant('B_SOURCE_RAW','source')]
    for ambiguity in ('admit','exclude'):
        base.append(Variant('C_FRAME0_PLAUSIBLE_RAW','frame0',ambiguous=ambiguity))
    for name,elig in [('D_ALL_REFERENCED','all'),('E_FRAME0_PLAUSIBLE_REFERENCED','frame0'),('F_SOURCE_REFERENCED','source')]:
        for mode in ('ALGEBRA_ONLY','INDEPENDENT_SAMPLES_APPROX'):
            for missing in ('retain','omit'):
                for ambiguity in (('admit','exclude') if elig=='frame0' else ('admit',)):
                    base.append(Variant(name,elig,'RGBD',mode,missing,ambiguity))
    base += [Variant('SRC_PRIORITY_COUNTERFACTUAL',source_priority=True),
             Variant('REFERENCE_RGB_ONLY',reference_channels='RGB'),Variant('REFERENCE_DEPTH_ONLY',reference_channels='D')]
    if include_recomputed:
        base += [replace(v,ownership='C') for v in base if v.variance_mode=='ALGEBRA_ONLY' and v.missing_reference=='retain' and v.ambiguous=='admit' and not v.source_priority]
    return base


def variant_id(v):
    return '__'.join(map(str,(v.name,v.ownership,v.variance_mode,v.missing_reference,v.ambiguous,v.remove_camera,v.reference_camera)))


def source_record(sc,pid):
    frame,camera,pixel=map(int,sc.src[pid]);cam=sc.cams[camera]
    return {'point_id':int(pid),'full_endpoint_index':int(sc.run.can['subsampled_to_full_index'][pid]),
            'source_frame':frame,'source_camera':camera,'source_u':pixel%cam.W,'source_v':pixel//cam.W,
            'canonical_xyz_m':sc.x0[pid],'canonical_rgb':sc.run.can['canonical_rgb'][pid]}


def source_reprojection(sc,pid):
    record=source_record(sc,pid);c=record['source_camera'];cam=sc.cams[c]
    u,v,z,*_=cam.project(sc.x0[pid]);D=sc.run.data.observation(0,c).depth[record['source_v'],record['source_u']]
    world=cam.R.T@(float(D)*np.array([(record['source_u']-cam.cx)/cam.fx,(record['source_v']-cam.cy)/cam.fy,1.])-cam.t)
    return {**record,'projected_uv':[u,v],'projection_error_px':[u-record['source_u'],v-record['source_v']],
            'backprojection_error_m':float(np.linalg.norm(world-sc.x0[pid])),'Z_minus_original_source_depth_m':float(z-D),
            'source_correlation':'Same source measurement defines x0, a and their priors; not independent new data.'}


def source_probe(p,c,x):
    """Bypass only duplicate ownership, never sensor/physical visibility checks."""
    row=O.row(p.imgs[c],p.cams[c],x,p.a,p.noise,p.policy,p.P_minus[:3,:3],x-p.s_minus[:3])
    if p.frozen[c]==O.SELF_OCC:row['reason']=O.SELF_OCC
    row['camera']=c;row['probe_only']=True;row['assimilated']=False
    return row


def classify_frame0(row,owner,is_source):
    if row['sample']['reason']!=O.VALID or owner==O.SELF_OCC or row['reason'] in (O.BEHIND,O.OUTSIDE,O.FRONT,O.INCONS):
        return 'FRAME0_INCONSISTENT'
    if is_source:return 'SOURCE_ID_KNOWN_AT_FRAME0'
    return 'FRAME0_GEOMETRICALLY_PLAUSIBLE'


def eligibility_mask(frame0,source,v):
    if v.eligibility not in ('all','source','frame0'):raise ValueError('Unknown eligibility policy')
    mask=np.ones(len(frame0),bool)
    if v.eligibility=='source':mask[:]=False;mask[source]=True
    if v.eligibility=='frame0':
        mask=np.array([r['frame0_class']!='FRAME0_INCONSISTENT' and (r['material_identity']!='MATERIAL_ID_UNDECIDABLE' or v.ambiguous=='admit') for r in frame0])
        mask[source]=True  # provenance permits candidacy, not visibility/admission
    if v.remove_camera is not None:mask[v.remove_camera]=False
    return mask


def owner_from_probe(sc,probe,pid):
    occupants=probe.get('pixel_occupants_front_to_back',[])
    if not occupants:return {'owner_status':'UNAVAILABLE','owner_point_id':None}
    front=occupants[0];owner=int(front['point']);source=source_record(sc,owner)
    gap=float(probe['Z']-front['z']);si=probe.get('sigma_z2_prior');sj=front.get('sigma_z2_prior')
    allowance=sc.policy['visibility_n_sigma']*math.sqrt(si+sj) if si is not None and sj is not None else None
    return {'owner_status':'EXACT_ARCHIVED_FRONT_ENTRY','owner_point_id':owner,'owner_full_endpoint_index':source['full_endpoint_index'],
            'owner_source':source,'owner_depth_m':front['z'],'point_depth_m':probe['Z'],'depth_gap_m':gap,
            'point_sigma_depth_m':math.sqrt(si) if si is not None else None,'owner_sigma_depth_m':math.sqrt(sj) if sj is not None else None,
            'production_occlusion_allowance_m':allowance,'exact_depth_tie':gap==0,
            'geometric_relation':'SELF_OCCLUDED_BY_PRODUCTION_UNCERTAINTY_RULE' if allowance is not None and gap>allowance else 'SAME_PIXEL_WITHIN_PRODUCTION_UNCERTAINTY_NOT_MATERIAL_PROOF',
            'owner_listing_total':probe.get('pixel_occupants_total'),'owner_listing_count':len(occupants),'point_is_owner':owner==pid,
            'canonical_world_separation_m':float(np.linalg.norm(sc.x0[pid]-sc.x0[owner])),
            'current_world_separation_m':None,'current_world_scope':'Owner optical depth and projected ownership cell are archived; full current owner XYZ may not be. Canonical separation is not current separation.'}


class Frame0Ownership:
    """Exact production frame-0 visibility, CPU only, one bounded cloud table."""
    def __init__(self,sc):
        import torch
        from estimator.projective_camera import CameraRig
        from estimator.projective_visibility import visibility, position_sigma_z2
        self.sc=sc;self.rig=CameraRig(sc.run.data.cameras,'cpu',torch.float64)
        self.proj=self.rig.project(torch.as_tensor(sc.x0,dtype=torch.float64),jacobian=False)
        self.sig=position_sigma_z2(self.rig.R,torch.as_tensor(sc.P0,dtype=torch.float64))
        self.reason,self.front=visibility(self.proj,self.sig,sc.cams[0].H,sc.cams[0].W,self.rig.near,sc.policy['visibility_n_sigma'])

    def probe(self,pid,c):
        uv=self.proj.uv[c].numpy();z=self.proj.z[c].numpy();pu,pv=np.rint(uv[pid]);cam=self.sc.cams[c]
        same=np.isfinite(uv).all(1)&np.isfinite(z)&(z>cam.near)&(np.rint(uv[:,0])==pu)&(np.rint(uv[:,1])==pv)
        ids=np.flatnonzero(same);ids=ids[np.lexsort((ids,z[ids]))]
        return {'Z':z[pid],'sigma_z2_prior':float(self.sig[c,pid]),'pixel_occupants_total':len(ids),
                'pixel_occupants_front_to_back':[{'point':int(i),'z':float(z[i]),'sigma_z2_prior':float(self.sig[c,i])} for i in ids],
                'nearest_pixel_u_v':[int(pu),int(pv)]}


def frame0_audit(sc,pid,ownership):
    refs=references(sc,pid,ownership.reason[:,pid].numpy());out=[];x=sc.x0[pid];a=sc.run.can['canonical_rgb'][pid]
    for c,cam in enumerate(sc.cams):
        row=O.row(sc.images(0)[c],cam,x,a,sc.noise,sc.policy,sc.P0[pid],np.zeros(3));own=int(ownership.reason[c,pid]);sample=row['sample']
        original=sc.run.data.observation(0,c);u,v,*_=cam.project(x);cell=sample.get('pixel00');raw=None
        if cell is not None:
            xx,yy=cell;indices=[(yy,xx),(yy,xx+1),(yy+1,xx),(yy+1,xx+1)]
            raw={'rgb':[original.rgb[y,z] for y,z in indices],'depth_m':[original.depth[y,z] for y,z in indices]}
        D=sample.get('neighbour_depth');second=float(D[3]-D[2]-D[1]+D[0]) if D is not None else None
        out.append({'scene':sc.name,**source_record(sc,pid),'camera':c,'is_source':c==int(sc.src[pid,1]),'u':u,'v':v,
                    'ownership_reason':O.NAMES[own],'row_reason':O.NAMES[row['reason']],'frame0_class':classify_frame0(row,own,c==int(sc.src[pid,1])),
                    'material_identity':'SOURCE_ID_KNOWN_AT_FRAME0' if c==int(sc.src[pid,1]) else 'MATERIAL_ID_UNDECIDABLE',
                    'sample':sample,'raw_original':raw,'r_equals_minus_h':row.get('r'),'b_h':-row['r'] if 'r' in row else None,
                    'H':row.get('H'),'R':row.get('var'),'S':row.get('S'),'nis':row.get('nis'),'mixed_depth_second_difference_m':second,
                    'reference':refs[c],**owner_from_probe(sc,ownership.probe(pid,c),pid)})
    return out,refs


def temporal_audit(sc,pid,refs,end=10):
    """Time differences at immutable xref, not the estimated moving trajectory."""
    x=sc.x0[pid];a=sc.run.can['canonical_rgb'][pid];out=[]
    for k in range(end+1):
        p=problem(sc,k,pid) if k else None;P=sc.P0[pid] if p is None else p.P_minus[:3,:3];dx=np.zeros(3) if p is None else x-p.s_minus[:3]
        for c,cam in enumerate(sc.cams):
            row=O.row(sc.images(k)[c],cam,x,a,sc.noise,sc.policy,P,dx);ref=refs[c]
            valid=ref['valid'] and row['sample']['reason']==O.VALID and 'r' in row
            out.append({'scene':sc.name,'point_id':pid,'frame':k,'camera':c,'x_ref_m':x,'row':row,
                'reference_status':ref['status'],'r0_equals_minus_h':ref['b'],'rt_equals_minus_h':row.get('r'),
                'delta_r':row['r']-ref['b'] if valid else None,'exact_residual_equality':bool(np.array_equal(row['r'],ref['b'])) if valid else None,
                'comparison_status':'VALID_SAMPLED_TEMPORAL_COMPARISON_MATERIAL_UNDECIDABLE' if valid else 'NO_VALID_REFERENCE_OR_CURRENT_SAMPLE',
                'geometry_scope':'Archived fixed rig; calibration accuracy/physical surface identity not independently established.'})
    return out


class ResponsibilityProblem(RefProblem):
    def __init__(self,*args,variant,source,frame0,owner=None,**kwargs):
        self.variant,self.source,self.owner=variant,source,owner
        super().__init__(*args,channels=variant.reference_channels or 'RGBD',mode=variant.variance_mode,**kwargs)
        self.cam_mask=eligibility_mask(frame0,source,variant);self.transfers=[]

    def rows(self,x,final=False):
        saved=self.frozen;self.frozen=list(saved)
        if self.variant.source_priority and self.frozen[self.source]==O.DUP:
            fresh=O.geometric_visibility(self.cams[self.source],x)
            probe=source_probe(self,self.source,x)
            if fresh==O.VALID and probe['reason']==O.VALID and not (final and self.recheck_self is not None and self.recheck_self[self.source]):
                self.frozen[self.source]=O.VALID
                self.transfers.append({'camera':self.source,'at_x_m':np.asarray(x).copy(),'final_recheck':final,
                    'interpretation':'Transfer original prior pixel slot from named archived winner; not a valid global double assimilation.'})
        try:
            rows=super().rows(x,final)
        finally:self.frozen=saved
        for r in rows:
            c=r['camera']
            if self.variant.reference_channels and self.variant.missing_reference=='omit' and not self.refs[c]['valid']:r['excluded_by_variant']=True
        return rows


def make_problem(sc,pid,k,sm,Pm,v,frame0,refs,owner=None):
    p=problem(sc,k,pid,s=sm,P=Pm);frozen=p.frozen;rc=p.recheck_self;flags=[]
    if v.ownership=='C':
        if owner is None:raise ValueError('Recomputed ownership needs the full frozen cloud')
        frozen,flags=owner.codes(pid,sm[:3],Pm[:3,:3],k,float(sc.t[k]-sc.t[k-1]));rc=None
    chosen_refs={c:dict(r) for c,r in refs.items()}
    if not v.reference_channels:chosen_refs={}
    elif v.reference_camera is not None:
        chosen_refs={c:({**r,'valid':False,'b':None,'status':'NOT_REFERENCED_BY_THIS_ONE_CAMERA_CONTROL'} if c!=v.reference_camera else r) for c,r in chosen_refs.items()}
    q=ResponsibilityProblem(p.cams,p.imgs,p.a,sm,Pm,frozen,p.noise,p.policy,p.max_iters,p.tol,rc,refs=chosen_refs,
                            variant=v,source=int(sc.src[pid,1]),frame0=frame0,owner=owner)
    if v.ownership=='C':
        original_rows=q.rows
        def rows(x,final=False):
            if final:q.recheck_self=[z==O.SELF_OCC for z in owner.codes(pid,x,Pm[:3,:3],k,float(sc.t[k]-sc.t[k-1]),final=True)[0]]
            return original_rows(x,final)
        q.rows=rows
    return q,flags


def summarize_update(p,result,source):
    used=p.used(result['rows_accepted']) if result['observed'] else [];support=[r['camera'] for r in used]
    initial=[r['camera'] for r in p.used(p.rows(p.s_minus[:3]))]
    return {'s_minus':p.s_minus,'P_minus':p.P_minus,'s_plus':result['s_plus'],'P_plus':result['P_plus'],
            'rows_initial':initial,'rows_accepted':support,'source_rows':int(source in support),'cross_rows':sum(c!=source for c in support),
            'observed':result['observed'],'acceptance':ACC[result['acceptance_code']],'converged':result['converged'],
            'iterations':len(result['history']),'dx_m':result['s_plus'][:3]-p.s_minus[:3],'dv_m_s':result['s_plus'][3:]-p.s_minus[3:],
            'original_fixed_support_before':fixed_cost(p,p.s_minus,initial),'original_fixed_support_after':fixed_cost(p,result['s_plus'],initial),
            'model_cost_after':p.prior_cost(result['s_plus'])+p.info(result['rows_accepted'])[2] if result['observed'] else None,
            'actual_rows':result['rows_accepted'],'prior_rows':p.rows(p.s_minus[:3]),'frozen_codes':list(p.frozen),'source_priority_changes':p.transfers,
            'references':{c:{'valid':r['valid'],'status':r['status']} for c,r in p.refs.items()},
            'inner_history':[{'s':h['s'],'A':h['A'],'b':h['b'],'delta':h['delta'],'J':h['J'],'threshold':h['threshold'],
                'cameras':[r['camera'] for r in p.used(h['rows'])],'cells':[r.get('sample',{}).get('pixel00') for r in p.used(h['rows'])]} for h in result['history']],
            'Pxx_eigenvalues_m2':np.linalg.eigvalsh(result['P_plus'][:3,:3]),'Pvv_eigenvalues_m2_s2':np.linalg.eigvalsh(result['P_plus'][3:,3:]),
            'source_probe':source_probe(p,source,p.s_minus[:3])}


def replay(sc,pid,v,frame0,refs,end=10,owner=None,single_frame=None):
    pre=sc.trace(sc.order[1])['prediction']['points'][str(pid)];s=np.array(pre['s_previous']);P=np.array(pre['P_previous']);output=[]
    frames=[single_frame] if single_frame is not None else sc.order[1:end+1]
    for k in frames:
        trace=sc.trace(k);prediction=trace['prediction'];dt=prediction['dt_s']
        if single_frame is None:sm,Pm,F,Q=O.predict(s,P,dt,np.asarray(prediction['q_c']))
        else:
            sm=np.array(prediction['points'][str(pid)]['s_minus']);Pm=np.array(prediction['points'][str(pid)]['P_minus']);F=np.array(prediction['F']);Q=np.array(prediction['Q'])
        p,flags=make_problem(sc,pid,k,sm,Pm,v,frame0,refs,owner)
        result=p.run(sc.policy['update_acceptance']);r=summarize_update(p,result,int(sc.src[pid,1]));r.update(frame=k,dt_seconds=dt,F=F,Q=Q,ownership_flags=flags,variant=variant_id(v),arm=v.name,
            variance_interpretation='ORIGINAL_F1' if not v.reference_channels else ('ALGEBRAIC_MEAN_RESPONSE_NOT_CALIBRATED_POSTERIOR' if v.variance_mode=='ALGEBRA_ONLY' else 'R_PLUS_R0_SENSITIVITY_OMITS_CROSS_TIME_REFERENCE_CORRELATION'),
            ownership_scope='CONDITIONAL_FROZEN_COMPETITORS',source_priority_transfer=v.source_priority)
        output.append(r);s,P=r['s_plus'].copy(),r['P_plus'].copy()
    return output


def baseline_parity(sc,pid,frames,fourth_arrays=None):
    rows=[]
    for f in frames:
        fin=sc.trace(f['frame'])['final'][str(pid)];c=compare_state6(f['s_plus'],fin['s_plus']);cov=compare_cov6(f['P_plus'],fin['P_plus'])
        expected=[r['camera'] for r in fin.get('rows',[])] if fin['acceptance'].startswith('ACCEPTED') else []
        cov_ok=np.allclose(f['P_plus'],fin['P_plus'],rtol=1e-9,atol=1e-12)
        trace=sc.trace(f['frame']);pre=trace['prediction']['points'][str(pid)];predict_error=compare_state6(f['s_minus'],pre['s_minus'])
        prior_cov_ok=np.allclose(f['P_minus'],pre['P_minus'],rtol=1e-9,atol=1e-12)
        frozen_equal=f['frozen_codes']==frozen_codes(trace['iterations'][0]['points'][str(pid)]['cameras'])
        probes=fin.get('cameras_fallback_state',fin['cameras_final_iterate']);reasons_equal=[O.NAMES[r['reason']] for r in f['actual_rows']]==[p['reason_used'] for p in probes]
        F_equal=np.array_equal(f['F'],trace['prediction']['F']);Q_equal=np.array_equal(f['Q'],trace['prediction']['Q'])
        ok=(c['position_linf_m']<=POSITION_TOL and c['velocity_linf_m_s']<=VELOCITY_TOL and cov_ok and prior_cov_ok and frozen_equal and reasons_equal and F_equal and Q_equal
            and predict_error['position_linf_m']<=POSITION_TOL and predict_error['velocity_linf_m_s']<=VELOCITY_TOL and f['rows_accepted']==expected and f['acceptance']==fin['acceptance'])
        fourth={'status':'NO_ARCHIVED_CASE'};key=f'{sc.name.split("_")[0].lower()}_{pid}__A0_ORIGINAL_F1__P__';j=sc.order.index(f['frame'])-1
        if fourth_arrays is not None and key+'s_plus' in fourth_arrays:
            states=fourth_arrays[key+'s_plus'];covs=fourth_arrays[key+'P_plus']
            if j<len(states) and j<len(covs):
                fourth={**compare_state6(f['s_plus'],states[j]),**compare_cov6(f['P_plus'],covs[j])}
                fourth['status']='PASS' if fourth['position_linf_m']<=POSITION_TOL and fourth['velocity_linf_m_s']<=VELOCITY_TOL and np.allclose(f['P_plus'],covs[j],rtol=1e-9,atol=1e-12) else 'FAIL';ok &= fourth['status']=='PASS'
            else:fourth={'status':'NO_ARCHIVED_FRAME','available_frames':len(states)}
        rows.append({'scene':sc.name,'point_id':pid,'frame':f['frame'],**c,**cov,'covariance_equal':bool(cov_ok),'rows_equal':f['rows_accepted']==expected,
                     'prediction_errors':predict_error,'prior_covariance_equal':bool(prior_cov_ok),'frozen_codes_equal':frozen_equal,'final_reason_codes_equal':reasons_equal,
                     'F_equal':F_equal,'Q_equal':Q_equal,'acceptance_equal':f['acceptance']==fin['acceptance'],'fourth_pass':fourth,'status':'PARITY_PASS' if ok else 'PARITY_FAIL'})
    return rows


def tracking_assessment(frames,role):
    if 'moving' in role and not any(f['observed'] for f in frames):return 'MOVING_TRACKING_FAILED_NO_OBSERVATIONS'
    return 'POSTHOC_EVALUATION_REQUIRED' if 'moving' in role else 'CANONICAL_DRIFT_IS_NOT_MATERIAL_ID_PROOF'
