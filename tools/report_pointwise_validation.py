#!/usr/bin/env python3
"""Summarize actual experiment files without rerunning or modifying estimation."""
import argparse,csv,hashlib,json,re,shutil
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())
def quantile(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return dict(zip(('min','median','p90','p95','p99','max'),np.quantile(x,[0,.5,.9,.95,.99,1]).tolist())) if len(x) else None


def experiment(root):
    manifest=read(root/'run_manifest.json');summary=read(root/'final/summary.json');initial=read(root/'initial_forward/summary.json');records=initial['frame_records']
    events=[json.loads(row) for row in (root/'discovery/events.jsonl').read_text().splitlines()]
    commits=[row for row in events if row['event']=='COMMIT'];new=[]
    for path in summary['new_tracks']:
        with np.load(path) as track:
            observed=track['observed'];s=track['state'];p=track['covariance'];source=track['source']
            jumps=np.linalg.norm(np.diff(s[:,:3],axis=0),axis=1);worst=int(np.argmax(jumps))
            new.append({'path':path,'anchor':source.tolist(),'directly_observed_frames':int(observed.sum()),'prediction_only_frames':int((~observed).sum()),'raw_factors':len(track['measurement_refs']),'position_displacement_first_to_last_m':float(np.linalg.norm(s[-1,:3]-s[0,:3])),'maximum_adjacent_frame_position_jump_m':float(jumps[worst]),'maximum_jump_frame_pair':[worst,worst+1],'speed_m_per_s':quantile(np.linalg.norm(s[:,3:],axis=1)),'NIS_observed':quantile(track['nis'][observed]),'position_covariance_trace_m2':quantile(np.trace(p[:,:3,:3],axis1=1,axis2=2))})
    match_rows=list(csv.DictReader((root/'discovery/matching_by_frame.csv').open()))
    coverage={}
    for row in match_rows:
        version=int(row['model_version']);key=(int(row['frame']),int(row['camera']))
        coverage.setdefault(version,{})[key]=int(row['owned'])
    coverage={version:sum(rows.values()) for version,rows in coverage.items()}
    gap_rows=list(csv.DictReader((root/'discovery/nearest_residuals.csv').open()));gaps=[float(r['unmatched_min_minus_matched_p99']) for r in gap_rows if r['unmatched_min_minus_matched_p99']]
    log=(root/'run.log').read_text();stages=[]
    pattern=r'(Pointwise initial forward \(all points\)|Match frozen trajectories \(reverse\)|Point \d+ (?:backward|forward)):\s+100%[^\r\n]*?\[(\d+(?::\d+){1,2})<'
    for name,elapsed in re.findall(pattern,log):
        seconds=0
        for item in elapsed.split(':'):seconds=60*seconds+int(item)
        row={'stage':name,'seconds_from_progress_clock':seconds}
        if not stages or row!=stages[-1]:stages.append(row)
    changed=[name for name,digest in manifest['source_hashes'].items() if not (Path(manifest['source_tree'])/name).is_file() or hashlib.sha256((Path(manifest['source_tree'])/name).read_bytes()).hexdigest()!=digest]
    stats={'actual_summary':summary,'progress_clock_stages':stages,'initial_min_covariance_eigenvalue':min(r['filter']['min_eigenvalue'] for r in records),'initial_max_covariance_condition':max(r['filter']['max_condition'] for r in records),'initial_line_search_stops':sum(r['filter']['line_search_stops'] for r in records),'initial_iteration_limit_reached':sum(r['filter']['iteration_limit_reached'] for r in records),'initial_accepted_energy_decreases':sum(r['filter']['accepted_energy_decreases'] for r in records),'initial_psd_repairs':sum(r['filter']['psd_repairs'] for r in records),'initial_observed_point_frame_count':sum(r['real_observed_points'] for r in records)+summary['initial_points'],'initial_prediction_only_point_frame_count':sum(r['prediction_only_points'] for r in records),'per_frame_NIS_median':quantile([r['nis']['quantiles']['median'] for r in records if r['nis']['quantiles']]),'matching_ambiguity_proposals_across_logged_scans':sum(int(r['ambiguous']) for r in match_rows),'matching_conflict_proposals_across_logged_scans':sum(int(r['conflicts']) for r in match_rows),'empirical_d2_gaps':quantile(gaps),'empirical_nonpositive_gap_rows':sum(x<=0 for x in gaps),'gap_rows_with_both_distributions':len(gaps),'commits':commits,'new_tracks':new,'timing':manifest['dataset']['timing'],'calibration':manifest['dataset']['metadata'],'complete_scene_frame_camera_selection':all(v is None for v in manifest['dataset']['test_limits'].values()),'source_hashes_match_current_files':not changed,'changed_source_files_since_experiment':changed}
    stats['geometrically_explained_raw_by_model_version']=coverage
    stats['explained_raw_growth_per_committed_point']={str(version):count-coverage[version-1] for version,count in coverage.items() if version-1 in coverage}
    return stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    report={'smoke':experiment(args.results_root/'pointwise_20261022_smoke_04/pointwise_rgbd'),'all_frame_one_event_benchmark':experiment(args.results_root/'pointwise_20261022_full_01/pointwise_rgbd'),'tests':read(args.results_root/'pointwise_20261022_validation/tests_final/tests.json'),'smoke_audit':read(args.results_root/'pointwise_20261022_validation/smoke04_raw_and_covariance_audit.json'),'full_audit':read(args.results_root/'pointwise_20261022_validation/full_raw_and_covariance_audit.json'),'full_new_point_replay':read(args.results_root/'pointwise_20261022_validation/full_replay/replay_summary.json'),'qualification':'The all-frame experiment is explicitly limited to one discovery event, not an unlimited completed reconstruction. Optional UKF is not implemented. Association policy is the user-authorized 1% chi-square tail, uncalibrated.'}
    (output/'measured_results.json').write_text(json.dumps(report,indent=2)+'\n')
    authorization=Path(__file__).resolve().parents[1]/'configs/pointwise_association_authorization_2026-10-22.md'
    shutil.copy2(authorization,output/authorization.name)
    for name,folder in [('smoke','pointwise_20261022_smoke_04'),('full','pointwise_20261022_full_01')]:
        target=args.results_root/folder/'pointwise_rgbd'
        (target/'association_policy_user_authorization.md').write_text(authorization.read_text())
    screenshot=Path('/tmp/rgbdekf_pointwise_20261022_viewer.png')
    if screenshot.exists():shutil.copy2(screenshot,output/'smoke_html_browser.png')
    print(json.dumps({k:v for k,v in report['all_frame_one_event_benchmark'].items() if k not in ('actual_summary','calibration','timing','commits')},indent=2))


if __name__=='__main__':main()
