"""Current measured v2 acceptance, without promoting partial stages to success."""
import argparse,json,time,shutil,hashlib,os
from pathlib import Path
import numpy as np
from validate_parallel_runs import inspect


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    test_path=max(root.glob('tests_*/test_results.json'),key=lambda p:p.stat().st_mtime)
    tests=json.loads(test_path.read_text());branches=[]
    for name,experiment in [('IEKF','full_scene_02'),('UKF','full_scene_04')]:
        scene=root/experiment/name.lower();report=inspect(scene,name)
        path=scene/'reports/stage_log.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        accepted=[r for r in rows if r.get('accepted') and 'proposal_energy' in r]
        report['latest_accepted_energy']=accepted[-1]['proposal_energy'] if accepted else None
        report['completed_forward_passes']=sum(r['stage']=='forward-complete' for r in rows)
        report['completed_smoother_passes']=sum(r['stage']=='smoother-complete' for r in rows)
        report['completed_reverse_passes']=sum(r['stage']=='reverse-discovery-complete' for r in rows)
        report['completed_frontier_passes']=sum(r['stage']=='chronological-frontier-complete' for r in rows)
        pidfile=root/experiment/(name.lower()+'.pid');active=False
        if pidfile.exists():
            proc=Path('/proc')/pidfile.read_text().strip()/'cmdline'
            if proc.exists():active=str(scene) in proc.read_bytes().decode().split('\0')
        report['worker_active_at_report']=active
        checkpoint=scene/'checkpoint.json'
        if checkpoint.exists():
            accepted_checkpoint=json.loads(checkpoint.read_text())
            report.update(accepted_groups_at_report=accepted_checkpoint['groups'],accepted_points_at_report=accepted_checkpoint['points'],accepted_checkpoint_energy=accepted_checkpoint['energy'])
        steps=sorted((scene/'outer_000/filter').glob('step_*.json'))
        if steps:
            data=json.loads(steps[-1].read_text());poses=np.array(data['filtered']['poses']);cov=np.array(data['filtered_covariance']);diagnostics=data['diagnostics']
            report['latest_first_forward_estimate']={'frame':diagnostics['frame'],'matched_raw_factors':diagnostics['matched_factors'],'max_group_translation_m':float(np.linalg.norm(poses[:,:3,3],axis=1).max()),'covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min()),'conditional_covariance_finite':bool(np.isfinite(cov).all()),'ukf_group_active_factors':[g.get('active_factors') for g in diagnostics.get('group_diagnostics',[])]}
        audit=scene/'reports/dataset_audit.json'
        if audit.exists():
            data=json.loads(audit.read_text());report['resolved_fps']=data['timing']['fps'];report['calibration_status']=data['calibration'].get('calibration_status');report['foreground_valid_observations']=data['foreground_valid_observations'];report['mask_visibility_premise']=data['calibration'].get('mask_visibility_premise')
        branches.append(report)
    report={'generated_unix':time.time(),'specification':'Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md','tests':tests,'test_report':str(test_path),'independent_experiments':branches,'all_acceptance_criteria_passed':tests['success'] and all(r['all_acceptance_gates_pass'] for r in branches),'qualification':'Partial real filtering, synthetic temporal cycles, component tests and operator benchmarks are not completed real reconstruction. Concurrent process timings are not a controlled speed comparison.'}
    directory=root/'reports';directory.mkdir(exist_ok=True)
    (directory/'v2_current_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['status','frames','cameras','initial_solved_points','accepted_points_at_report','accepted_groups_at_report','completed_forward_passes','completed_smoother_passes','completed_reverse_passes','completed_frontier_passes','worker_active_at_report','resolved_fps']
    lines=['# Masked v2 measured acceptance','',f"All acceptance criteria passed: **{report['all_acceptance_criteria_passed']}**",'',f"Actual GPU suite: {tests['passed']}/{tests['tests']} passed; {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skips.",'','| Metric | IEKF | UKF |','|---|---|---|']
    for key in fields:lines.append('| '+key+' | '+' | '.join(str(b.get(key,'pending')) for b in branches)+' |')
    lines+=['','Failing tests:',*['- '+s for s in tests['failed_tests']],'',report['qualification']]
    (directory/'v2_current_acceptance.md').write_text('\n'.join(lines)+'\n')
    repo=Path(__file__).resolve().parents[1]
    for name in ('AUDIT.md','VALIDATION.md','COVARIANCE_AUDIT.md','RUNNING.md'):
        target=directory/name
        if target.exists() and target.read_bytes()==(repo/name).read_bytes():continue
        if target.exists():target.rename(directory/(name+'.previous_'+str(time.time_ns())))
        shutil.copyfile(repo/name,target)
    specifications=['Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md','Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.pdf','CODEX_IMPLEMENT_PARALLEL_MASKED_RGBD_IEKF_UKF_v2.md','reference_3dgs_masked.pdf']
    (directory/'authoritative_specification_hashes.json').write_text(json.dumps({name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in specifications},indent=2)+'\n')
    print(json.dumps({'report':str(directory/'v2_current_acceptance.json'),'all_acceptance_criteria_passed':report['all_acceptance_criteria_passed'],'branches':[{'pipeline':b.get('pipeline'),'status':b['status'],'latest_first_forward_estimate':b.get('latest_first_forward_estimate'),'worker_active_at_report':b['worker_active_at_report']} for b in branches]},indent=2))


if __name__=='__main__':main()
