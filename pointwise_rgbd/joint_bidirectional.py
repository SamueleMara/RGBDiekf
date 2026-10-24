"""Compare endpoint gauges, then refit UNIQUE pooled raw observations."""
from pathlib import Path
import numpy as np
from .joint_rgbd import plane,fit,select
from .group_visualization import serial
from .store import json_write


def compare(a,b,config):
    if a is None or b is None:return {'compatible':False,'reason':'At least one direction has no identifiable selected joint'}
    x=a['model'];y=b['model']
    if x['type']!=y['type']:return {'compatible':False,'reason':'Directions select different joint types','types':[x['type'],y['type']]}
    sign=1 if x['axis']@y['axis']>=0 else -1;uy=sign*y['axis'];delta=x['axis']-uy
    covariance=2*(plane(x['axis'])@a['covariance'][:2,:2]@plane(x['axis']).T+plane(y['axis'])@b['covariance'][:2,:2]@plane(y['axis']).T)
    axis_stat=float(delta@np.linalg.pinv(covariance,hermitian=True)@delta)
    qdelta=x['q'][1:]-sign*y['q'][1:];d=2 if x['type']=='PRISMATIC' else 4
    qcov=2*(a['covariance'][d:,d:]+b['covariance'][d:,d:]);qstat=float(qdelta@np.linalg.solve(qcov,qdelta))
    report={'compatible':axis_stat<=config.gate(2) and qstat<=config.gate(len(qdelta)),
        'type':x['type'],'axis_sign':sign,'axis_angle_radians':float(np.arccos(np.clip(x['axis']@uy,-1,1))),
        'axis_difference_d2':axis_stat,'chronological_displacement_difference_d2':qstat,'displacement_rmse':float(np.sqrt(np.mean(qdelta*qdelta))),
        'world_reference':'fixed calibrated world; q[0]=0 in both directions; no point-ID correspondence assumed',
        'uncertainty':'2*(P_forward+P_backward) upper bound under unknown correlation; never independent-product confidence'}
    if x['type']=='REVOLUTE':
        ax=x['axis_point']-x['axis']*(x['axis']@x['axis_point']);ay=y['axis_point']-y['axis']*(y['axis']@y['axis_point'])
        distance=np.linalg.norm(np.cross(ay-ax,x['axis']))
        C=2*(plane(x['axis'])@a['covariance'][2:4,2:4]@plane(x['axis']).T+plane(y['axis'])@b['covariance'][2:4,2:4]@plane(y['axis']).T)
        line_delta=ay-ax;line_stat=float(line_delta@np.linalg.pinv(C,hermitian=True)@line_delta)
        report.update(revolute_line_distance_m=float(distance),line_difference_d2=line_stat)
        report['compatible']&=line_stat<=config.gate(2)
    return report


def unique_pool(groups,endpoints,excluded_refs=None):
    """One raw return, one canonical anchor; overlapping directions are NOT new data."""
    chosen={};duplicates=0
    for rows in groups:
        for row in rows:
            if excluded_refs is None and row['k'] in endpoints:continue
            e=row['phase1_residual'];quality=np.einsum('ni,ni->n',e,np.linalg.solve(row['R'],e[:,:,None])[:,:,0])
            for j,pixel in enumerate(row['pixel']):
                ref=(row['k'],row['camera_index'],int(pixel));item=(float(quality[j]),row,j)
                if excluded_refs is not None and ref in excluded_refs:continue
                if ref in chosen:duplicates+=1
                tie=ref in chosen and item[0]==chosen[ref][0] and int(row['anchor'][j])==(0 if sum(ref)%2==0 else max(endpoints))
                if ref not in chosen or item[0]<chosen[ref][0] or tie:chosen[ref]=item
    buckets={}
    for ref,(_,row,j) in sorted(chosen.items()):
        key=(ref[0],ref[1],int(row['anchor'][j]));buckets.setdefault(key,[]).append((row,j))
    result=[]
    for _,items in buckets.items():
        first=items[0][0];r={key:first[key] for key in ('k','camera_index','camera')}
        for name in ('p','anchor','z','R','pixel','phase1_residual','map_key','source_covariance','point_id'):
            if name in first:r[name]=np.stack([row[name][j] for row,j in items])
        result.append(r)
    return result,{'unique_raw_returns':len(chosen),'duplicate_direction_factors_removed':duplicates,
        'canonical_choice':'lower stored association-fit standardized raw residual (shared-rigid MAP in production), deterministic tie order; selection-conditioned, not marginalized',
        'excluded_endpoint_frames':sorted(endpoints) if excluded_refs is None else [],'excluded_prior_raw_returns':None if excluded_refs is None else len(excluded_refs)}


def run(forward,backward,config,root,feasible=None,backend=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    comparison=compare(forward.get('selected'),backward.get('selected'),config)
    comparison['candidate_geometry_diagnostics']={kind:compare(forward['fits'].get(kind),backward['fits'].get(kind),config)
        for kind in ('PRISMATIC','REVOLUTE') if 'covariance' in forward['fits'].get(kind,{}) and 'covariance' in backward['fits'].get(kind,{})}
    json_write(root/'direction_comparison.json',serial(comparison))
    if not comparison['compatible']:
        report={'status':'BIDIRECTIONAL_JOINT_UNRESOLVED','comparison':comparison,'camera_policy':'all_available'}
        json_write(root/'result.json',serial(report));return report
    from .canonical_schur import combine_priors,save_state
    prior=combine_priors([forward['map_prior'],backward['map_prior']])
    excluded={tuple(map(int,r)) for r in prior['source_refs']}
    K=len(forward['selected']['model']['q'])
    rows,pool_report=unique_pool([forward['rows'],backward['rows']],{0,K-1},excluded)
    candidates={}
    for kind in ('PRISMATIC','REVOLUTE'):
        source=forward['fits'][kind]
        if 'model' not in source:continue
        candidates[kind]=fit(source['model'],rows,config,feasible,prior,backend=backend)
    decision=select(candidates,rows,config)
    decision.update(comparison=comparison,pool=pool_report,combination='Refit shared geometry/q on unique raw sensor returns. Forward/backward posteriors are not multiplied; no cross-direction point IDs matched.')
    json_write(root/'result.json',serial(decision))
    selected=candidates.get(decision['status'])
    if selected:
        json_write(root/'selected_joint.json',serial({key:value for key,value in selected.items() if key not in ('information','covariance','map_state')}))
        np.savez_compressed(root/'uncertainty.npz',covariance=selected['covariance'],information=selected['information'])
        save_state(root/'canonical_map_and_joint_uncertainty.npz',selected)
    return {**decision,'selected':selected}
