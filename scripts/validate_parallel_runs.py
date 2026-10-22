"""Fail closed on actual independent full-scene experiment acceptance gates."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def inspect(root,pipeline):
    root=Path(root);report_path=root/'validation_report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'status':'MISSING_REPORT'}
    log=root/'reports/stage_log.jsonl';rows=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    stale=bool(report_path.exists() and log.exists() and log.stat().st_mtime>report_path.stat().st_mtime)
    if stale:
        previous=report;report={'status':'INCOMPLETE_LATEST_ATTEMPT','complete_real_cycle':False,'previous_attempt_report':previous}
        audit=root/'reports/dataset_audit.json'
        if audit.exists():
            data=json.loads(audit.read_text());report.update(frames=data['frames'],cameras=data['cameras'])
    if rows:
        report.setdefault('pipeline',pipeline)
        report.setdefault('projection_operator_version',max(r.get('score_epoch',1) for r in rows))
        initial=next((r for r in rows if r['stage']=='initialization-complete'),{})
        report.setdefault('initialization_sha256',initial.get('initialization_sha256'))
        report.setdefault('initial_solved_points',initial.get('points'))
        report['latest_recorded_stage']=rows[-1]
    audit=root/'reports/dataset_audit.json'
    if audit.exists():
        data=json.loads(audit.read_text());report.setdefault('frames',data['frames']);report.setdefault('cameras',data['cameras'])
    if rows and not report_path.exists():report['status']='INCOMPLETE_NO_FINAL_REPORT'
    k=report.get('frames',0);c=report.get('cameras',0);gates={}
    epoch=report.get('projection_operator_version',1)
    current_rows=[r for r in rows if r.get('score_epoch',1)==epoch]
    completed_outer=[r['outer'] for r in current_rows if r['stage']=='outer-complete'];outer=max(completed_outer,default=0)
    gates['completed_real_cycle']=report.get('complete_real_cycle') is True
    # Resume preserves earlier final passes and interrupted prefixes. Validate
    # the latest final attempt, never concatenate or borrow their frames.
    final_start=max((i for i,r in enumerate(current_rows) if r['stage']=='forward-frame' and r.get('label')=='final-new-map' and r['frame']==0),default=len(current_rows))
    latest_final=current_rows[final_start:]
    gates['all_forward_frames']=k>0 and [r['frame'] for r in current_rows if r['stage']=='forward-frame' and r.get('label')==f'outer-{outer}']==list(range(k)) and [r['frame'] for r in latest_final if r['stage']=='forward-frame' and r.get('label')=='final-new-map']==list(range(k))
    for name,order in [('reverse-discovery',list(range(k-1,0,-1))),('chronological-frontier',list(range(1,k)))]:
        actual=[r['frame'] for r in current_rows if r['stage']=='frontier-complete' and r['outer']==outer and r['pass_kind']==name]
        gates[name+'_all_frontiers']=k>0 and actual==order
    edits=[r for r in current_rows if r['stage'] in ('geometry-proposal','group-proposal') and r.get('accepted')]
    gates['accepted_geometry_or_group_discovery']=bool(edits)
    gates['second_forward_after_discovery']=bool(edits) and any(r['stage']=='forward-frame' and r.get('label')=='final-new-map' and r['elapsed_seconds']>edits[-1]['elapsed_seconds'] for r in rows)
    epoch=report.get('projection_operator_version',1)
    accepted=[r for r in rows if 'proposal_energy' in r and r.get('accepted') and r.get('score_epoch',1)==epoch]
    gates['monotone_common_score']=bool(accepted) and all(np.isfinite(r['proposal_energy']) and r['proposal_energy']<=r['old_energy'] for r in accepted)
    compact=report.get('output_mode')=='compact'
    products=['solved_points.ply','solved_points.csv','unsolved_counts.csv' if compact else 'unsolved_observations.csv','group_poses.npz','group_twists.npz','group_covariance.npz','source_provenance.npz','objective_history.csv','stage_log.csv','run_config_resolved.yaml','validation_report.json']
    gates['all_required_products']=all((root/p).is_file() for p in products)
    gates['hard_foreground_constraint']=report.get('hard_mask_model_version')==2 and report.get('accepted_nominal_mask_feasible') is True
    gates['foreground_source_validation']=report.get('source_foreground_violations')==0 and k>0 and c>0
    gates['own_smoother']=False;gates['finite_pose_twist_covariance']=False;gates['unique_raw_sources']=False
    means=root/'group_poses.npz';cov_path=root/'group_covariance.npz'
    if means.exists() and cov_path.exists():
        with np.load(means) as data:poses=data['poses'];gates['all_pose_times']=len(poses)==k
        with np.load(root/'group_twists.npz') as data:twists=data['twists']
        with np.load(cov_path) as data:
            cov=data['smoothed'];gates['finite_pose_twist_covariance']=bool(np.isfinite(poses).all() and np.isfinite(twists).all() and np.isfinite(cov).all() and np.linalg.eigvalsh(cov).min()>-1e-9)
            gates['covariance_semantics_explicit']='Conditional' in str(data['semantics'])
            gates['accepted_trajectory_covariance_available']=bool(data['accepted_mean_matches_smoothed_proposal'])
            if 'transitions' in data and 'sigma_cross_covariances' in data:
                gates['own_smoother']=bool(data['transitions'].size>0 and data['sigma_cross_covariances'].size==0) if pipeline=='IEKF' else bool(data['sigma_cross_covariances'].size>0 and data['transitions'].size==0)
        trace=root/'final_forward/forward_smooth.npz'
        if trace.exists():
            with np.load(trace) as data:gates['own_smoother']=bool(data['transitions'].size>0 and data['sigma_cross_covariances'].size==0) if pipeline=='IEKF' else bool(data['sigma_cross_covariances'].size>0 and data['transitions'].size==0)
        report['max_estimated_translation_m']=float(np.linalg.norm(poses[:,:,:3,3],axis=-1).max())
    provenance=root/'source_provenance.npz'
    if provenance.exists():
        with np.load(provenance) as data:raw=data['raw_observation_id'];gates['unique_raw_sources']=len(np.unique(raw))==len(raw)
    gpu=report.get('gpu',{})
    gates['cuda_peak_within_card_capacity']=gpu.get('torch_peak_within_device_capacity') is True and gpu.get('device','').startswith('cuda') and 0<gpu.get('peak_allocated_bytes',0)<=gpu.get('usable_device_bytes',0)
    gates['convergence']=report.get('status')=='CONVERGED_LOCAL_SEARCH'
    report.update(acceptance_gates=gates,all_acceptance_gates_pass=all(gates.values()),output=str(root));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--iekf',required=True,type=Path);p.add_argument('--ukf',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    reports=[inspect(a.iekf,'IEKF'),inspect(a.ukf,'UKF')]
    hashes=[r.get('initialization_sha256') for r in reports];fair=hashes[0] is not None and hashes[0]==hashes[1]
    output={'independent_results':reports,'identical_initialization':fair,'all_acceptance_gates_pass':fair and all(r['all_acceptance_gates_pass'] for r in reports)}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    keys=['status','frames','cameras','initial_solved_points','final_solved_points','unresolved_observations','groups','forward_passes','reverse_passes','frontier_passes','final_energy','runtime_seconds','rss_peak_bytes','max_estimated_translation_m']
    lines=['# Independent full-scene validation','',f'Identical initial model: {fair}','', '| Metric | IEKF | UKF |','|---|---|---|']
    lines += [f"| {key} | {reports[0].get(key,'MISSING')} | {reports[1].get(key,'MISSING')} |" for key in keys]
    for r in reports:
        lines += ['',f"## {r.get('pipeline','Incomplete experiment')}",'']+[f"- {name}: {'PASS' if ok else 'FAIL'}" for name,ok in r['acceptance_gates'].items()]
    (a.output/'comparison.md').write_text('\n'.join(lines)+'\n');print(json.dumps(output,indent=2));return 0 if output['all_acceptance_gates_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
